import logging
import os
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple

import coolname
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    MixedPrecision,
)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig

from llm_studio.src.optimizers import Optimizers
from llm_studio.src.schedulers import Schedulers
from llm_studio.src.utils.data_utils import cat_batches, get_inference_batch_size
from llm_studio.src.utils.exceptions import (
    LLMDataException,
    LLMMetricException,
    LLMModelException,
)
from llm_studio.src.utils.logging_utils import TqdmToLogger
from llm_studio.src.utils.utils import save_pickle

logger = logging.getLogger(__name__)


def unwrap_model(model: torch.nn.Module):
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    while isinstance(model, options):
        model = model.module

    return model


# TODO: currently not saving optimizer
def save_checkpoint(model: torch.nn.Module, path: str, cfg: Any):
    """Saves a model checkpoint if the path is provided.

    Args:
        model: model to save
        path: path to save the checkpoint to

    Returns:
        Dictionary with all the keys to save
    """

    model = unwrap_model(model)

    if hasattr(cfg.training, "lora") and cfg.training.lora:
        model.backbone.save_pretrained(path)

    checkpoint = {"model": model.state_dict()}

    if path is not None:
        torch.save(checkpoint, os.path.join(path, "checkpoint.pth"))


def load_model_weights(
    model: torch.nn.Module, model_weights: Dict, strict: bool, cfg: Any
):
    orig_num_items = len(model_weights)
    model_state_dict = model.state_dict()

    # needed to load models trained in int8 with other dtypes
    model_weights = {
        k: v
        if not (v.dtype is torch.int8 and cfg.architecture.backbone_dtype != "int8")
        else model_state_dict[k]
        for k, v in model_weights.items()
        if not ("SCB" in k and cfg.architecture.backbone_dtype != "int8")
    }

    # Need to ignore int8 weights so undo strict loading requirement
    if len(model_weights) != orig_num_items:
        strict = False

    model_weights = {re.sub(r"^module\.", "", k): v for k, v in model_weights.items()}
    model_weights = {k.replace("_orig_mod.", ""): v for k, v in model_weights.items()}
    try:
        model.load_state_dict(OrderedDict(model_weights), strict=True)
    except Exception as e:
        if strict:
            raise e
        else:
            if cfg.environment._local_rank == 0:
                logger.warning(
                    "Only a part of the pretrained weights was loaded. "
                    "Some layers can't be initialized with pretrained "
                    f"weights: {e}"
                )

            for layer_name in re.findall("size mismatch for (.*?):", str(e)):
                model_weights.pop(layer_name, None)
            model.load_state_dict(OrderedDict(model_weights), strict=False)
    return model


def load_checkpoint(
    cfg: Any, model: torch.nn.Module, strict: bool = True, weights_path: str = None
):
    """Load checkpoint

    Args:
        cfg: config file
        model: model to load weights to
        strict: whether to apply strict matching for weights
        weights_path: custom path to the weights.
            If None, cfg.architecture.pretrained_weights is used
    Returns:
        epoch: current epoch
    """

    if weights_path is None:
        weights_path = cfg.architecture.pretrained_weights

    d = torch.load(weights_path, map_location="cpu")

    model_weights = d["model"]
    model = load_model_weights(model, model_weights, strict, cfg)

    del model_weights

    if cfg.environment._local_rank == 0:
        logger.info(f"Weights loaded from: {weights_path}")


def wrap_model_distributed(model: torch.nn.Module, cfg: Any, fsdp: bool):
    if fsdp:
        auto_wrap_policy = None

        mixed_precision_policy = None
        dtype = None
        if cfg.environment.mixed_precision:
            dtype = torch.float16
        if dtype is not None:
            mixed_precision_policy = MixedPrecision(
                param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
            )
        model = FullyShardedDataParallel(
            model,
            # sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            # cpu_offload=CPUOffload(offload_params=True),
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            device_id=cfg.environment._local_rank,
            # use_orig_params=False
            limit_all_gathers=True,
        )
    else:
        find_unused_parameters = cfg.environment.find_unused_parameters
        if getattr(cfg.architecture, "gradient_checkpointing", None):
            find_unused_parameters = False
        model = DistributedDataParallel(
            model,
            device_ids=[cfg.environment._local_rank],
            find_unused_parameters=find_unused_parameters,
        )

    return model


def get_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Prepares Optimizer.

    Args:
        model: model
        cfg: input config

    Returns:
        Optimizer
    """

    no_decay = ["bias", "LayerNorm.weight"]
    differential_layers = cfg.training.differential_learning_rate_layers
    optimizer = Optimizers.get(cfg.training.optimizer)(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                    # and param.requires_grad
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                    # and param.requires_grad
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                    # and param.requires_grad
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                    # and param.requires_grad
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": 0,
            },
        ],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    return optimizer


def get_scheduler(
    cfg: Any, optimizer: torch.optim.Optimizer, epoch_steps: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Prepares Learning Rate Scheduler.

    Args:
        cfg: input config
        optimizer: model optimizer
        epoch_steps: total number of weight updates during the epoch

    Returns:
        Learning Rate Scheduler
    """

    scheduler = Schedulers.get(cfg.training.schedule)(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.warmup_epochs * epoch_steps,
        num_training_steps=cfg.training.epochs * epoch_steps,
    )

    return scheduler


def generate_experiment_name() -> str:
    """
    Generates a random human-readable experiment name in kebab-case.

    Returns:
        The random name.
    """
    return coolname.generate_slug(2)


def compute_metric(
    metric_func: Callable, cfg: Any, data: Any, df: Any
) -> Tuple[float, Any]:
    """Compute metric and return metric score (number) and full metric (number or dict)

    Args:
        metric_func: metric function
        cfg: input Config
        data: data Dict
        df: data DataFrame

    Returns:
        val_metric: single number score (using config threshold for threshold metrics)
        full_val_metric: for threshold metrics return dictionary where keys are
            different thresholds, values are metric scores, for regular metrics
            just return the metric score (same as val_metric)

    """
    try:
        full_val_metric = metric_func(cfg=cfg, results=data, val_df=df)
    except Exception:
        raise LLMMetricException()

    if type(full_val_metric) is dict:  # threshold dependent clf metrics
        if "argmax" in full_val_metric.keys():  # multiclass using argmax
            val_metric = full_val_metric["argmax"]
        elif hasattr(cfg.prediction, "probability_threshold"):
            # retrieve score using selected threhshold
            threshold = getattr(cfg.prediction, "probability_threshold")
            val_metric = full_val_metric[threshold]
        else:
            raise ValueError("Config prediction misses probability threshold.")
    else:
        val_metric = full_val_metric

    return val_metric, full_val_metric


def get_number_of_validation_epochs(training_epochs: int, evaluation_epochs: float):
    """
    Given the number of training epochs and the number of epochs between model
    evaluations, return the number of times the model is being evaluated during
    training

    Args:
        training_epochs: The number of epochs to train for
        evaluation_epochs: This is the number of epochs after which we want to
            evaluate our model

    Returns:
        num_val_epochs: The number of epochs to be evaluated during training.
    """
    return training_epochs // evaluation_epochs


def contains_nan(output: Dict):
    return (
        sum(
            [
                1
                for key, val in output.items()
                if isinstance(val, torch.Tensor)
                and torch.isnan(val.detach().cpu()).sum() > 0
            ]
        )
        > 0
    )


def run_inference(
    cfg: Any,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    mode: str,
) -> Dict[str, list]:
    """Runs inference

    Args:
        cfg: config
        model: model
        dataloader: dataloader
        mode: mode for inference

    Returns:
        Dictionary with output

    """

    # Store information for evaluation
    out = dict()

    if cfg.environment._local_rank == 0:
        logger.info(f"Starting {mode} inference")

    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    progress_bar = tqdm(
        total=len(dataloader),
        disable=cfg.environment._local_rank != 0,
        file=tqdm_out,
        ascii=True,
        desc=f"{mode} progress",
        mininterval=0,
    )

    log_update_steps = max(len(dataloader) // 20, 1)
    inf_it = iter(dataloader)
    for itr in range(len(dataloader)):
        try:
            data = next(inf_it)
        except Exception:
            raise LLMDataException("Data reading error. Skipping inference.")

        val_batch_size = get_inference_batch_size(cfg)
        cfg.environment._curr_val_step += val_batch_size * cfg.environment._world_size

        batch = cfg.dataset.dataset_class.batch_to_device(data, cfg.environment._device)

        calculate_loss = True
        if cfg.environment.mixed_precision:
            with torch.cuda.amp.autocast():
                output = model.forward(batch, calculate_loss=calculate_loss)
            if contains_nan(output):
                raise LLMModelException(
                    "NaN caught during mixed precision inference. "
                    "Please disable mixed precision inference. "
                    "Alternatively, reducing learning rate or "
                    "gradient clipping may help to stabilize training."
                )
        else:
            output = model.forward(batch, calculate_loss=calculate_loss)

        output = dataloader.dataset.postprocess_batch_predictions(  # type: ignore
            cfg=cfg, input_batch=batch, output=output
        )

        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu()

            # DefaultDict is not used as it adds extra keys during pickle.dump
            if key not in out:
                out[key] = [val]
            else:
                out[key] += [val]

        if cfg.environment._local_rank == 0:
            # Show logs each 5% of the inference
            if (itr + 1) % log_update_steps == 0 or itr == len(dataloader) - 1:
                progress_bar.set_description(f"{mode} progress", refresh=False)
                if (itr + 1) % log_update_steps == 0:
                    progress_bar.update(log_update_steps)
                else:
                    progress_bar.update(len(dataloader) % log_update_steps)

            cfg.logging._logger.log(
                "internal",
                "current_val_step",
                cfg.environment._curr_val_step,
                step=cfg.environment._curr_val_step,
            )

    progress_bar.close()
    del progress_bar
    out = cat_batches(out)

    return out


def save_predictions(cfg, val_data, val_dataloader, val_df, mode):
    val_data, val_df = val_dataloader.dataset.format_output(  # type: ignore
        cfg=cfg, df=val_df, output=val_data
    )
    raw_preds_name = os.path.join(cfg.output_directory, f"{mode}_raw_predictions.pkl")
    csv_preds_name = os.path.join(cfg.output_directory, f"{mode}_predictions.csv")
    save_pickle(raw_preds_name, val_data)
    val_df.to_csv(csv_preds_name, index=False)


def prepare_model_for_lora_training(model, layer_norm_names=["layer_norm"]):
    loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

        if loaded_in_8bit:
            # cast layer norm in fp32 for stability for 8bit models
            if param.ndim == 1 and any(
                layer_norm_name in name for layer_norm_name in layer_norm_names
            ):
                param.data = param.data.to(torch.float32)

    return model


def create_nlp_backbone(cfg, model_class=AutoModel, kwargs={}) -> Any:
    """
    Creates a backbone model for NLP tasks.
    This is needed for Gradient Checkpointing in DDP mode.
    """
    config = AutoConfig.from_pretrained(
        cfg.llm_backbone, trust_remote_code=cfg.environment.trust_remote_code
    )
    config.hidden_dropout_prob = cfg.architecture.intermediate_dropout
    config.attention_probs_dropout_prob = cfg.architecture.intermediate_dropout

    quantization_config = None
    if cfg.architecture.backbone_dtype == "int8":
        kwargs["device_map"] = {"": cfg.environment._device}
        kwargs["torch_dtype"] = torch.float16
        kwargs["load_in_8bit"] = True
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
        )
        # need to force pretrained
        cfg.architecture.pretrained = True
    else:
        kwargs["torch_dtype"] = getattr(torch, cfg.architecture.backbone_dtype)
    logger.info("dtype: {dtype}".format(dtype=kwargs["torch_dtype"]))

    if cfg.architecture.gradient_checkpointing:
        config.use_cache = False

    if cfg.architecture.pretrained:
        backbone = model_class.from_pretrained(
            cfg.llm_backbone,
            config=config,
            trust_remote_code=cfg.environment.trust_remote_code,
            quantization_config=quantization_config,
            **kwargs,
        )
    else:
        backbone = model_class.from_config(config, **kwargs)

    if cfg.tokenizer._vocab_length > config.vocab_size:
        logger.info(f"Resizing token embeddings to {cfg.tokenizer._vocab_length}")
        backbone.resize_token_embeddings(cfg.tokenizer._vocab_length)

    if cfg.training.lora:
        backbone = prepare_model_for_lora_training(backbone, layer_norm_names=[])
    else:
        if cfg.architecture.backbone_dtype != "float32":
            if cfg.environment.mixed_precision:
                logger.info("Disabling mixed precision as dtype not set to float32.")
                cfg.environment.mixed_precision = False
            if cfg.architecture.backbone_dtype != "bfloat16":
                logger.warning(
                    "Pure float16 or int8 training will "
                    "likely lead to unstable training without adapters."
                )

    if cfg.architecture.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()

    return backbone
