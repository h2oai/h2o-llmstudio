import logging
import math
import os
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union, no_type_check

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler

from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src.datasets.conversation_chain_handler import ConversationChainHandler
from llm_studio.src.utils.exceptions import LLMDataException
from llm_studio.src.utils.gpu_utils import sync_across_processes
from llm_studio.src.utils.utils import PatchedAttribute, set_seed

logger = logging.getLogger(__name__)


def read_dataframe(
    path: str,
    n_rows: int = -1,
    meta_only: bool = False,
    non_missing_columns: Optional[List[str]] = None,
    verbose: bool = False,
    handling: str = "warn",
    fill_columns: Optional[List[str]] = None,
    fill_value: Any = "",
    mode: str = "",
) -> pd.DataFrame:
    """Reading a dataframe from different file types

    Args:
        path: path of the dataframe
        n_rows: number of rows to limit to
        meta_only: return only meta information
        non_missing_columns: list of columns that cannot contain missing values
        verbose: if warning about dropped rows should be logged
        handling: how to handle missing values
        fill_columns: columns where empty value should be filled (used for empty text)
        fill_value: value to fill empty columns with (used for empty text)
        mode: dataset type, used only for better exception/log information
    Returns:
        DataFrame

    """

    non_missing_columns = [] if non_missing_columns is None else non_missing_columns
    fill_columns = [] if fill_columns is None else fill_columns

    meta_info_path = os.path.split(path)
    meta_info_path = os.path.join(
        *meta_info_path[:-1],
        "__meta_info__" + meta_info_path[-1] + ".csv",
    )
    if meta_only and os.path.exists(meta_info_path):
        path = meta_info_path

    if path.endswith(".csv"):
        df = pd.read_csv(path, lineterminator="\n").reset_index(drop=True)
    elif path.endswith(".pq") or path.endswith(".parquet"):
        try:
            df = pd.read_parquet(path, engine="pyarrow").reset_index(drop=True)
        except Exception:
            df = pd.read_parquet(path, engine="fastparquet").reset_index(drop=True)
    elif path.endswith(".json") or path == "":
        return pd.DataFrame()
    else:
        raise ValueError(
            f"Could not determine type of file {path}: "
            f"CSV (`.csv`) and Parquet (`.pq` and `.parquet`) are supported."
        )

    if fill_columns:
        df[fill_columns] = df[fill_columns].fillna(fill_value)

    if meta_only and os.path.exists(meta_info_path):
        return df

    non_missing_columns = [x for x in non_missing_columns if x in df]
    if len(non_missing_columns):
        orig_size = df.shape[0]
        non_missing_index = df[non_missing_columns].dropna().index
        dropped_index = [idx for idx in df.index if idx not in non_missing_index]
        df = df.loc[non_missing_index].reset_index(drop=True)
        new_size = df.shape[0]
        if new_size < orig_size and verbose:
            logger.warning(
                f"Dropped {orig_size - new_size} rows when reading dataframe '{path}' "
                f"due to missing values encountered in one of the following columns:"
                f" {non_missing_columns} in the following rows: {dropped_index}"
            )

            if handling == "error":
                dropped_str = dropped_index

                if len(dropped_str) > 10:
                    dropped_str = dropped_str[:5] + ["..."] + dropped_str[-5:]

                dropped_str = ", ".join([str(x) for x in dropped_str])
                prefix = f"{mode} " if mode else ""
                error = (
                    f"{prefix}dataset contains {len(dropped_index)} rows with missing "
                    f"values in one of the following columns: {non_missing_columns} in "
                    f"the following rows: {dropped_str}"
                )

                raise ValueError(error.capitalize())

    if n_rows > -1:
        df = df.iloc[sample_indices(len(df), n_indices=n_rows)]

    # create meta information dataframe if it does not exist
    if not os.path.exists(meta_info_path):
        df_meta = pd.DataFrame(columns=df.columns)
        df_meta.to_csv(meta_info_path, index=False)

    return df


def get_fill_columns(cfg: DefaultConfigProblemBase) -> List[str]:
    if hasattr(cfg.dataset, "prompt_column"):
        if isinstance(cfg.dataset.prompt_column, (list, tuple)):
            return list(cfg.dataset.prompt_column)
        return [cfg.dataset.prompt_column]

    return []


def read_dataframe_drop_missing_labels(
    path: str, cfg: DefaultConfigProblemBase
) -> pd.DataFrame:
    if isinstance(cfg.dataset.prompt_column, tuple):
        input_cols = list(cfg.dataset.prompt_column)
    else:
        input_cols = [cfg.dataset.prompt_column]
    fill_columns = get_fill_columns(cfg)
    df = read_dataframe(
        path,
        non_missing_columns=input_cols,
        verbose=True,
        fill_columns=fill_columns,
    )
    df[input_cols] = df[input_cols].fillna("").astype(str)
    if (
        hasattr(cfg.dataset, "answer_column")
        and not isinstance(cfg.dataset.answer_column, (list, tuple))
        and cfg.dataset.answer_column in df.columns
    ):
        df[cfg.dataset.answer_column] = (
            df[cfg.dataset.answer_column].fillna("").astype(str)
        )
    return df


def is_valid_data_frame(path: str, csv_rows: int = 100) -> bool:
    """Checking data frame format

    Args:
        path: path of the dataframe
        csv_rows: number of rows to limit to when checking csv files

    Returns:
        bool

    """
    try:
        if path.endswith(".csv"):
            pd.read_csv(path, nrows=csv_rows, lineterminator="\n")
        elif path.endswith(".pq") or path.endswith(".parquet"):
            pq.ParquetFile(path)
        else:
            raise ValueError(
                f"Could not determine type of file {path}: "
                f"CSV (`.csv`) and Parquet (`.pq` and `.parquet`) are supported."
            )
    except Exception as e:
        logger.error(str(e))
        return False
    return True


def sample_data(cfg: DefaultConfigProblemBase, df: pd.DataFrame) -> pd.DataFrame:
    """Sample data from the dataframe"""

    if cfg.dataset.parent_id_column != "None" and "id" in df.columns:
        parent_mapping = df.set_index("id")["parent_id"].to_dict()

        # A recursive function to get the root id for each node
        def get_root(node):
            parent = parent_mapping.get(node)
            if parent is None or pd.isna(parent):
                return node
            return get_root(parent)

        # Apply the function to assign each row the root id
        df["root_id"] = df["id"].apply(get_root)

        # Sample root_ids without replacement
        root_ids = df["root_id"].unique()
        n_sampled_root_ids = int(len(root_ids) * cfg.dataset.data_sample)

        np.random.seed(7331)
        sampled_root_ids = np.random.choice(
            root_ids, size=n_sampled_root_ids, replace=False
        )

        # Filter the dataframe to only include rows with sampled root_ids
        df = df[df["root_id"].isin(sampled_root_ids)].reset_index(drop=True)
        del df["root_id"]
    else:
        # at least 10 observations
        n = max(10, int(len(df) * cfg.dataset.data_sample))
        df = df.sample(n=min(n, len(df)), random_state=7331, replace=False)

    return df


def load_mt_bench_data(cfg: DefaultConfigProblemBase) -> pd.DataFrame:
    """Loads MT-BENCH data.

    Args:
        cfg: input config

    Returns:
        MT-BENCH DataFrame
    """

    prompt_column = cfg.dataset.prompt_column[0]
    answer_column = cfg.dataset.answer_column

    df = df = pd.read_json("prompts/mt-bench/question.jsonl", lines=True)
    df = df.rename(columns={"turns": prompt_column, "reference": answer_column})
    df[prompt_column] = df[prompt_column].apply(lambda x: x[0])
    df[answer_column] = (
        df[answer_column].fillna("").apply(lambda x: x[0] if x != "" else x)
    )

    if (
        hasattr(cfg.dataset, "rejected_prompt_column")
        and cfg.dataset.rejected_prompt_column != "None"
    ):
        df[cfg.dataset.rejected_prompt_column] = df[prompt_column]

    return df


def get_data(cfg: DefaultConfigProblemBase) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepares train and validation DataFrames.

    Args:
        cfg: DefaultConfigProblemBase input config

    Returns:
        Train and validation DataFrames
    """

    train_df, val_df = load_train_valid_data(cfg)

    if (
        hasattr(cfg.prediction, "metric_gpt_template")
        and cfg.prediction.metric_gpt_template == "mt-bench"
    ):
        logger.info(
            "Overwriting validation data with MT-BENCH data. Please note that "
            "respective metric is an approximation and might not fully match "
            "the original implementation."
        )
        val_df = load_mt_bench_data(cfg)

    if cfg.dataset.data_sample < 1.0:
        if "Train" in cfg.dataset.data_sample_choice:
            train_df = sample_data(cfg, train_df)
        if "Validation" in cfg.dataset.data_sample_choice:
            val_df = sample_data(cfg, val_df)

    if cfg.training.train_validation_data:
        train_df = pd.concat([train_df, val_df], axis=0)

    train_df = cfg.dataset.dataset_class.preprocess_dataframe(train_df, cfg)
    val_df = cfg.dataset.dataset_class.preprocess_dataframe(val_df, cfg)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def merge_on_common_items(lst):
    G = nx.Graph()
    for sublst in lst:
        for item in sublst:
            G.add_edge(sublst[0], item)
    return [list(c) for c in nx.connected_components(G)]


def load_train_valid_data(cfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.dataset.validation_strategy == "custom":
        if cfg.dataset.validation_dataframe == "None":
            raise LLMDataException(
                "No validation dataframe provided. "
                "Please provide a validation dataframe or "
                "choose a different validation strategy."
            )
        train_df = read_dataframe_drop_missing_labels(cfg.dataset.train_dataframe, cfg)
        val_df = read_dataframe_drop_missing_labels(
            cfg.dataset.validation_dataframe, cfg
        )
    elif cfg.dataset.validation_strategy == "automatic":
        logger.info("Setting up automatic validation split...")
        df = read_dataframe_drop_missing_labels(cfg.dataset.train_dataframe, cfg)
        if cfg.dataset.parent_id_column != "None":
            # split based on conversation_chain_ids
            # this ensures that all samples from the
            # same conversation are in the same fold
            with PatchedAttribute(cfg.dataset, "limit_chained_samples", True):
                conversation_chain_ids = ConversationChainHandler(
                    df=df, cfg=cfg
                ).conversation_chain_ids
            # Some conversations may have the same parent id, e.g. for OASST
            # 6aa548c6-65ad-4531-9411-76173ae060a3 and
            # 2a164c2a-4f0e-45aa-8990-e7dd3b51c06b
            # have the same parent a8df94e3-cfc7-4736-9587-0ec943d0fec3
            # We need to merge those into a single group
            conversation_chain_ids = merge_on_common_items(conversation_chain_ids)
            conversation_chain_labels = [
                i
                for i, conversation_chain_id in enumerate(conversation_chain_ids)
                for _ in conversation_chain_id
            ]
            group_shuffle_split = GroupShuffleSplit(
                test_size=cfg.dataset.validation_size, n_splits=1, random_state=1337
            )
            train_idx, val_idx = next(
                group_shuffle_split.split(df, groups=conversation_chain_labels)
            )
            # flatten conversation_chain_ids
            flattened_conversation_chain_ids = np.array(
                [
                    idx
                    for conversation_chain_id in conversation_chain_ids
                    for idx in conversation_chain_id
                ]
            )
            train_df = df.iloc[flattened_conversation_chain_ids[train_idx]].reset_index(
                drop=True
            )
            val_df = df.iloc[flattened_conversation_chain_ids[val_idx]].reset_index(
                drop=True
            )
        else:
            train_df, val_df = train_test_split(
                df, test_size=cfg.dataset.validation_size, random_state=1337
            )
    else:
        raise LLMDataException("No valid validation strategy provided.")
    return train_df, val_df


def worker_init_fn(worker_id: int) -> None:
    """Sets the random seed for each worker.

    Args:
        worker_id: ID of the corresponding worker
    """

    if "PYTHONHASHSEED" in os.environ:
        seed = int(os.environ["PYTHONHASHSEED"]) + worker_id
    else:
        seed = np.random.get_state()[1][0] + worker_id  # type: ignore
    set_seed(seed)


def get_train_dataset(train_df: pd.DataFrame, cfg: DefaultConfigProblemBase) -> Dataset:
    """Prepares train Dataset.

    Args:
        train_df: train DataFrame
        cfg: DefaultConfigProblemBase input config

    Returns:
        Train Dataset
    """

    logger.info("Loading train dataset...")

    train_dataset: Dataset = cfg.dataset.dataset_class(
        df=train_df, cfg=cfg, mode="train"
    )
    return train_dataset


def get_train_dataloader(train_ds: Any, cfg: DefaultConfigProblemBase) -> DataLoader:
    """Prepares train DataLoader.

    Args:
        train_ds: train Dataset
        cfg: input config

    Returns:
        Train Dataloader
    """

    sampler: Sampler
    if cfg.environment._distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds,
            num_replicas=cfg.environment._world_size,
            rank=cfg.environment._local_rank,
            shuffle=True,
            seed=cfg.environment._seed,
            drop_last=True,
        )
        sampler_length = len(sampler)
    else:
        sampler = None
        sampler_length = len(train_ds)

    if sampler_length < cfg.training.batch_size and cfg.training.drop_last_batch:
        logger.warning(
            "Training data too small when dropping last batch. Number of rows "
            "should be at least batch size multiplied by number of gpus. "
            "Forcing to keep last batch."
        )
        cfg.training.drop_last_batch = False
    if sampler_length <= 1:
        raise LLMDataException("Data too small to train model.")

    train_dataloader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.environment.number_of_workers,
        pin_memory=True,
        collate_fn=train_ds.get_train_collate_fn(),
        drop_last=cfg.training.drop_last_batch,
        worker_init_fn=worker_init_fn,
    )

    logger.info(f"Number of observations in train dataset: {len(train_ds)}")

    return train_dataloader


def get_val_dataset(val_df: pd.DataFrame, cfg: DefaultConfigProblemBase):
    """Prepares validation Dataset.

    Args:
        val_df: validation DataFrame
        cfg: DefaultConfigProblemBase input config

    Returns:
        Validation Dataset
    """

    logger.info("Loading validation dataset...")
    val_dataset = cfg.dataset.dataset_class(df=val_df, cfg=cfg, mode="validation")

    return val_dataset


def get_val_dataloader(val_ds: Any, cfg: DefaultConfigProblemBase):
    """Prepares validation DataLoader.

    Args:
        val_ds: validation Dataset
        cfg: DefaultConfigProblemBase input config

    Returns:
        Validation Dataloader
    """

    sampler: Sampler
    if cfg.environment._distributed and cfg.environment._distributed_inference:
        sampler = OrderedDistributedSampler(
            val_ds,
            num_replicas=cfg.environment._world_size,
            rank=cfg.environment._local_rank,
        )
    else:
        sampler = SequentialSampler(val_ds)

    batch_size = get_inference_batch_size(cfg)

    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.environment.number_of_workers,
        pin_memory=True,
        collate_fn=val_ds.get_validation_collate_fn(),
        worker_init_fn=worker_init_fn,
    )

    logger.info(f"Number of observations in validation dataset: {len(val_ds)}")

    return val_dataloader


@no_type_check
def cat_batches(
    data: DefaultDict[str, Union[torch.Tensor, np.ndarray]]
) -> DefaultDict[str, Union[torch.Tensor, np.ndarray]]:
    """Concatenates output data from several batches

    Args:
        data: dict with keys and list of batch outputs

    Returns:
        Concatenated dict

    """

    for key, value in data.items():
        if len(value[0].shape) == 0:
            if isinstance(value[0], torch.Tensor):
                data[key] = torch.stack(value)
            else:
                data[key] = np.stack(value)
        else:
            if isinstance(value[0], torch.Tensor):
                data[key] = torch.cat(value, dim=0)
            else:
                data[key] = np.concatenate(value, axis=0)

    return data


class OrderedDistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    Source:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/distributed_sampler.py
    """

    def __init__(
        self,
        dataset: Any,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """
        Args:
            dataset: Dataset used for sampling
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process within num_replicas
        """

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += [0] * (self.total_size - len(indices))
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self.rank * self.num_samples : self.rank * self.num_samples
            + self.num_samples
        ]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def sample_indices(length: int, n_indices: int = 10, seed: int = 1337) -> np.ndarray:
    """Samples random indices

    Args:
        length: length to sample from
        n_indices: number of indices to sample
        seed: seed for sampling

    Returns:
        sampled indices
    """
    state = np.random.get_state()
    np.random.seed(seed)
    idx = np.random.choice(
        np.arange(length), size=min(length, n_indices), replace=False
    )
    np.random.set_state(state)

    return idx


def get_inference_batch_size(cfg: DefaultConfigProblemBase) -> int:
    """Calculates inference batch size

    Args:
        cfg: DefaultConfigProblemBase config with all the hyperparameters
    Returns:
        Inference batch size
    """

    if cfg.prediction.batch_size_inference != 0:
        return cfg.prediction.batch_size_inference
    else:
        return cfg.training.batch_size


def sanity_check(cfg):
    """
    Perform sanity check on the data
    """

    df = read_dataframe_drop_missing_labels(cfg.dataset.train_dataframe, cfg)
    cfg.dataset.dataset_class.sanity_check(df=df, cfg=cfg, mode="train")
    valid_filename = cfg.dataset.validation_dataframe
    if isinstance(valid_filename, str) and os.path.exists(valid_filename):
        df = read_dataframe_drop_missing_labels(valid_filename, cfg)
        cfg.dataset.dataset_class.sanity_check(df=df, cfg=cfg, mode="validation")


def batch_padding(
    cfg: DefaultConfigProblemBase,
    batch: Dict,
    training: bool = True,
    mask_key: str = "attention_mask",
    pad_keys: List[str] = ["input_ids", "attention_mask", "special_tokens_mask"],
    padding_side: str = "left",
) -> Dict:
    """Pads a batch according to set quantile, or cuts it at maximum length"""
    if cfg.environment.compile_model:
        # logger.warning("Batch padding not functional with torch compile.")
        return batch
    elif batch[mask_key].sum() == 0:
        # continued pretraining
        return batch
    elif cfg.tokenizer.padding_quantile == 0:
        return batch
    elif training and cfg.tokenizer.padding_quantile < 1.0:
        if padding_side == "left":
            lengths = torch.stack(
                [
                    torch.where(batch[mask_key][i] == 1)[0].min()
                    for i in range(batch[mask_key].size(0))
                ]
            ).float()
            quantile = 1 - cfg.tokenizer.padding_quantile
        else:
            lengths = torch.stack(
                [
                    torch.where(batch[mask_key][i] == 1)[0].max()
                    for i in range(batch[mask_key].size(0))
                ]
            ).float()
            quantile = cfg.tokenizer.padding_quantile
        if cfg.environment._distributed:
            lengths = sync_across_processes(
                lengths, cfg.environment._world_size
            )  # type: ignore
        idx = int(torch.floor(torch.quantile(lengths, quantile)))
    else:
        if padding_side == "left":
            idx = int(torch.where(batch[mask_key] == 1)[1].min())
        else:
            idx = int(torch.where(batch[mask_key] == 1)[1].max())

    if padding_side == "left":
        for key in pad_keys:
            if key in batch:
                batch[key] = batch[key][:, idx:].contiguous()
    else:
        idx += 1
        for key in pad_keys:
            if key in batch:
                batch[key] = batch[key][:, :idx].contiguous()

    return batch
