import codecs
import collections.abc
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from llm_studio.src.datasets.text_utils import get_texts, get_tokenizer

logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """Base PyTorch dataset for any problem type."""

    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        """
        Args:
            df: input DataFrame
            cfg: config with all the hyperparameters
            mode: dataset mode. One of {"train", "validation"}
        """

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()

        self.indices = np.arange(len(self.df))

        assert self.mode in [
            "train",
            "validation",
        ], f"There is no {self.mode} for the datasets"

        # Get the labels
        has_all_columns = cfg.dataset.answer_column in self.df.columns
        has_missing_values = False
        if has_all_columns:
            has_missing_values = (
                self.df.shape[0]
                != self.df[[cfg.dataset.answer_column]].dropna().shape[0]
            )

        if not has_all_columns or has_missing_values:
            if has_missing_values:
                message = (
                    f"The {self.mode} DataFrame"
                    f" column {cfg.dataset.answer_column}"
                    " contain missing values."
                )
            else:
                message = (
                    f"The {self.mode} DataFrame "
                    "does not contain the required column:"
                    f" {cfg.dataset.answer_column}."
                )

            raise ValueError(message)

        self.tokenizer = get_tokenizer(cfg)

        self.raw_prompts = get_texts(df, self.cfg, separator="")
        self.prompts = [self.parse_prompt(cfg, prompt) for prompt in self.raw_prompts]

        self.answers = (
            self.df[self.cfg.dataset.answer_column].astype(str).values.tolist()
        )

        self.parent_ids = None
        if self.cfg.dataset.parent_id_column != "None":
            if "id" not in self.df.columns:
                logger.warning(
                    f"When using parent column, the dataframe requires an 'id' column. "
                    f"Disabling functionality for mode {self.mode}."
                )
            else:
                self.parent_ids = self.df[self.cfg.dataset.parent_id_column].values
                self.df_id_to_idx = {v: k for k, v in enumerate(self.df["id"].values)}

                # limit chained samples to the longest chain
                if self.cfg.dataset.limit_chained_samples and self.mode == "train":
                    unique_parent_ids = set(self.parent_ids)
                    self.indices = self.indices[
                        [id not in unique_parent_ids for id in self.df["id"].values]
                    ]

        self.systems = None
        if self.cfg.dataset.system_column != "None":
            if self.cfg.training.use_rlhf:
                logger.warning(
                    f"RLHF is not compatible with system column. "
                    f"Disabling functionality for mode {self.mode}."
                )
            elif self.cfg.dataset.system_column not in self.df.columns:
                logger.warning(
                    f"System column {self.cfg.dataset.system_column} not found."
                    f"Disabling functionality for mode {self.mode}."
                )
            else:
                systems = (
                    self.df[self.cfg.dataset.system_column].astype(str).values.tolist()
                )
                self.systems = [self.parse_system(cfg, system) for system in systems]

        if self.cfg.environment._local_rank == 0:
            logger.info(f"Sample prompt: {self.prompts[0]}")

    @staticmethod
    def parse_prompt(cfg: Any, prompt: str):
        prompt = (
            f"{codecs.decode(cfg.dataset.text_prompt_start, 'unicode_escape')}{prompt}"
        )
        if cfg.dataset.add_eos_token_to_prompt:
            prompt += cfg._tokenizer_eos_token
        prompt = (
            f"{prompt}"
            f"{codecs.decode(cfg.dataset.text_answer_separator, 'unicode_escape')}"
        )
        return prompt

    @staticmethod
    def parse_system(cfg: Any, system: str):
        # no system tokens if empty
        if system == "":
            return system
        system = (
            f"{codecs.decode(cfg.dataset.text_system_start, 'unicode_escape')}{system}"
        )
        if cfg.dataset.add_eos_token_to_system:
            system += cfg._tokenizer_eos_token
        return system

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def get_input_columns(cfg: Any) -> Tuple[str, ...]:
        """Assigns the input columns

        Args:
            cfg: config

        """
        if isinstance(cfg.dataset.prompt_column, tuple):
            return cfg.dataset.prompt_column
        return (cfg.dataset.prompt_column,)

    @staticmethod
    def batch_to_device(
        batch: Union[Dict, List, torch.Tensor], device: str
    ) -> Union[Dict, List, torch.Tensor, str]:
        """Function to send the batch to the device specified

        Args:
            batch: input batch
            device: device to send the data to
        Returns:
            batch with the elements on the device specified
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, (list, tuple)) and all(
            isinstance(item, str) for item in batch
        ):
            # Do not move list of strings to device
            return batch
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: CustomDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        elif isinstance(batch, collections.abc.Sequence):
            return [CustomDataset.batch_to_device(value, device) for value in batch]
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, cfg: Any, mode: str) -> pd.DataFrame:
        """
        Preprocesses the input dataframe

        Args:
            df: the full training dataframe
            cfg: config
            mode: the mode. One of {"train", "validation"}
        Returns:
            the processed dataframe
        """

        def personalize(text):
            text = text.replace("Open Assistant", cfg.dataset.chatbot_name)
            text = text.replace("Open-Assistant", cfg.dataset.chatbot_name)
            text = text.replace("open-assistant", cfg.dataset.chatbot_name)
            text = text.replace("OpenAssistant", cfg.dataset.chatbot_name)
            text = text.replace("open assistant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assistand", cfg.dataset.chatbot_name)
            text = text.replace("Open Assitant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assistent", cfg.dataset.chatbot_name)
            text = text.replace("Open Assisstant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assitent", cfg.dataset.chatbot_name)
            text = text.replace("Open Assitiant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assistiant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assitan ", cfg.dataset.chatbot_name + " ")
            text = text.replace("Open Assistan ", cfg.dataset.chatbot_name + " ")
            text = text.replace("Open Asistant", cfg.dataset.chatbot_name)
            text = text.replace("Open Assiant", cfg.dataset.chatbot_name)
            text = text.replace("Assistant", cfg.dataset.chatbot_name)
            text = text.replace("LAION AI", cfg.dataset.chatbot_author)
            text = text.replace("LAION-AI", cfg.dataset.chatbot_author)
            text = text.replace("LAION,", cfg.dataset.chatbot_author + ",")
            text = text.replace("LAION.ai", cfg.dataset.chatbot_author)
            text = text.replace("LAION.", cfg.dataset.chatbot_author + ".")
            text = text.replace("LAION", cfg.dataset.chatbot_author)
            return text

        if cfg.dataset.personalize:
            for prompt_col in cfg.dataset.prompt_column:
                df[prompt_col] = df[prompt_col].apply(personalize)
            df[cfg.dataset.answer_column] = df[cfg.dataset.answer_column].apply(
                personalize
            )

        return df

    def get_train_collate_fn(self):
        """
        Returns train batch collate function for the PyTorch Dataloader.
        By default returns None that uses the default PyTorch collate
        """

        return None

    def get_validation_collate_fn(self):
        """
        Return validation batch collate function for the PyTorch Dataloader.
        By default returns None that uses the default PyTorch collate
        """

        return None

    def postprocess_batch_predictions(self, cfg: Any, output: Dict) -> Dict:
        if cfg.prediction.metric == "Perplexity":
            return output

        predicted_text = [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in output["predicted_answer_ids"]
        ]
        output["predicted_text"] = np.array(predicted_text)

        if not cfg.training.use_rlhf:
            del output["predicted_answer_ids"]
        else:
            output["predicted_answer_ids"].detach()

        return output

    @staticmethod
    def clean_output(
        output: Dict,
        prompts: List[str],
        cfg: Any,
    ):
        output["predicted_text"] = output["predicted_text"].tolist()
        for j in range(len(output["predicted_text"])):
            curr_text = output["predicted_text"][j].strip()
            for stop_token in cfg.tokenizer._stop_words:
                if curr_text.find(stop_token) != -1:
                    curr_text = curr_text[: curr_text.find(stop_token)]
            output["predicted_text"][j] = curr_text.strip()

        return output

    def postprocess_output(self, cfg, df: pd.DataFrame, output: Dict) -> Dict:
        if not cfg.prediction.metric == "Perplexity":
            output = self.clean_output(output, self.prompts, cfg)

        output["target_text"] = self.answers

        metric_func, _, _ = cfg.prediction.metric_class.get(cfg.prediction.metric)

        if "GPT" in cfg.prediction.metric:
            metrics, explanations = metric_func(
                cfg,
                output,
                df,
                raw_results=True,
            )
            output["explanations"] = explanations
        else:
            metrics = metric_func(
                cfg,
                output,
                df,
            )
        output["metrics"] = metrics

        return output

    def format_output(
        self, cfg, df: pd.DataFrame, output: Dict
    ) -> Tuple[Dict, pd.DataFrame]:
        output = {
            key: value
            for key, value in output.items()
            if key not in ["loss", "target", "losses"]
        }

        output.pop("target_text", None)

        if "predicted_text" in output.keys():
            output["predicted_text"] = np.array(output["predicted_text"])

        if isinstance(cfg.dataset.prompt_column, tuple):
            for col in cfg.dataset.prompt_column:
                output[col] = df[col].values
        else:
            output[cfg.dataset.prompt_column] = df[cfg.dataset.prompt_column].values

        if "predicted_text" in output.keys():
            df[f"pred_{cfg.dataset.answer_column}"] = output["predicted_text"]

        return output, df

    @classmethod
    def sanity_check(cls, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        """
        Quick check whether Dataframe and configurations are correctly set.
        """
        if (
            cfg.dataset.parent_id_column is not None
            and cfg.dataset.parent_id_column in df.columns
            and "id" in df.columns
        ):
            assert (
                df[cfg.dataset.parent_id_column] != df["id"]
            ).all(), "Parent id column is the same as id column for some rows"
            assert (df[cfg.dataset.parent_id_column].fillna("") == "").sum() > 0, (
                "Did not find any conversation start. "
                "Please ensure that some parent ids are empty."
            )

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        idx = self.indices[idx]

        sample = dict()
        system_encoding, prompt_encoding, answer_encoding = self._get_sample_encoding(
            idx
        )

        rlhf_is_in_training_mode = self.cfg.training.use_rlhf and self.mode == "train"
        if rlhf_is_in_training_mode:
            # ground truth answer not used in RLHF training
            encodings = [[system_encoding, prompt_encoding, torch.empty(0)]]
        else:
            encodings = [[system_encoding, prompt_encoding, answer_encoding]]

        encodings = self.get_parent_encodings(idx) + encodings

        # in case of chained samples, we only want to keep the first system encoding
        system_encoding = encodings[0][0]
        # remove system encodings from list of encodings to only keep prompt and answer
        encodings = [encoding[1:] for encoding in encodings]
        # concatenate system encoding with root prompt encoding
        encodings[0][0] = torch.cat([system_encoding, encodings[0][0]])

        input_ids = torch.cat([torch.cat(encoding) for encoding in encodings])
        if not rlhf_is_in_training_mode:  # no labels required for RLHF during training
            labels = input_ids.clone()

            if self.cfg.dataset.mask_prompt_labels:
                prompt_mask = torch.cat(
                    [
                        torch.cat(
                            [
                                torch.ones_like(prompt_encoding),
                                torch.zeros_like(answer_encoding),
                            ]
                        )
                        for prompt_encoding, answer_encoding in encodings
                    ]
                ).to(torch.bool)
                labels.masked_fill_(prompt_mask, -100)
            if self.cfg.dataset.add_eos_token_to_answer:
                # eos_token may be equal to pad_token. Add the label back manually.
                labels[-1] = self.tokenizer.eos_token_id

            if self.cfg.tokenizer.max_length < len(input_ids):
                labels = labels[-self.cfg.tokenizer.max_length :]

            sample["labels"] = torch.full((self.cfg.tokenizer.max_length,), -100)
            sample["labels"][-len(labels) :] = labels

        sample.update(
            self.pad_tokens(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_length=self.cfg.tokenizer.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        )

        # get answer encodings
        answer_input_ids = encodings[-1][1]
        answer_attention_mask = torch.ones_like(answer_input_ids)

        sample.update(
            self.pad_tokens(
                answer_input_ids,
                attention_mask=answer_attention_mask,
                max_length=self.cfg.tokenizer.max_length_answer,
                pad_token_id=self.tokenizer.pad_token_id,
                direction="right",
                prefix="answer_",
            )
        )

        # Remove last answer from encoding to create the prompt for inference
        encodings[-1][1] = torch.empty(0)
        prompt_input_ids = torch.cat([torch.cat(encoding) for encoding in encodings])
        prompt_attention_mask = torch.ones_like(prompt_input_ids)

        sample.update(
            self.pad_tokens(
                prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_length=self.cfg.tokenizer.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix="prompt_",
            )
        )

        # make sure system encoding is always prepended if max_length exceeded
        if sample["input_ids"][0] != self.tokenizer.pad_token_id:
            sample["input_ids"][: len(system_encoding)] = system_encoding
            if self.cfg.dataset.mask_prompt_labels:
                sample["labels"][: len(system_encoding)] = -100

        if sample["prompt_input_ids"][0] != self.tokenizer.pad_token_id:
            sample["prompt_input_ids"][: len(system_encoding)] = system_encoding

        if self.cfg.training.use_rlhf:
            sample["reward_model_prompt_text"] = (
                self.get_reward_model_parent_prompt_text(idx) + self.raw_prompts[idx]
            )
        return sample

    def _get_sample_encoding(self, idx) -> List:
        if self.systems is not None:
            system = self.systems[idx]
            system_encoding = self.encode(
                self.tokenizer, system, self.cfg.tokenizer.max_length_prompt, "right"
            )["input_ids"]
        else:
            system_encoding = torch.empty(0)
        prompt = self.prompts[idx]
        answer = self.answers[idx]

        prompt_encoding = self.encode(
            self.tokenizer, prompt, self.cfg.tokenizer.max_length_prompt, "left"
        )["input_ids"]
        if self.cfg.dataset.add_eos_token_to_answer:
            max_length_answer = self.cfg.tokenizer.max_length_answer - 1
        else:
            max_length_answer = self.cfg.tokenizer.max_length_answer
        answer_encoding = self.encode(
            self.tokenizer, answer, max_length_answer, "right"
        )["input_ids"]
        if self.cfg.dataset.add_eos_token_to_answer:
            answer_encoding = torch.cat(
                [
                    answer_encoding,
                    torch.Tensor([self.tokenizer.eos_token_id]),
                ],
                dim=0,
            )

        return [system_encoding, prompt_encoding, answer_encoding]

    def get_parent_ids(self, idx):
        max_loop = 1_000
        parent_idxs = []
        if self.parent_ids is not None:
            parent_idx = idx
            while (
                (parent_idx := self.df_id_to_idx.get(self.parent_ids[parent_idx], None))
            ) is not None:
                parent_idxs.append(parent_idx)
                max_loop -= 1
                if max_loop == 0:
                    raise ValueError(
                        f"Parent chain of sample with idx {idx} "
                        f"exceeds max loop count. "
                        f"Please ensure that parent chain is not circular."
                    )
        return parent_idxs[::-1]

    def get_parent_encodings(self, idx):
        parent_encodings = [
            self._get_sample_encoding(int(parent_idx))
            for parent_idx in self.get_parent_ids(idx)
        ]
        if self.mode == "train":
            # Note that if condition is called for each parent encoding,
            # thus the probability is not the same for each parent encoding.
            parent_encodings = [
                parent_encoding
                for parent_encoding in parent_encodings
                if not np.random.random()
                < self.cfg.augmentation.skip_parent_probability
            ]
            if np.random.random() < self.cfg.augmentation.random_parent_probability:
                rnd_idx = np.random.randint(len(self))
                parent_encodings.insert(0, self._get_sample_encoding(int(rnd_idx)))
        return parent_encodings

    def get_reward_model_parent_prompt_text(self, idx):
        return "".join(
            [
                self.raw_prompts[int(parent_idx)]
                + "<|endoftext|>"
                + self.answers[int(parent_idx)]
                + "<|endoftext|>"
                for parent_idx in self.get_parent_ids(idx)
            ]
        )

    def pad_tokens(
        self,
        input_ids,
        attention_mask,
        max_length,
        pad_token_id,
        direction="left",
        prefix="",
        system_ids=None,
    ):
        sample = {}

        if max_length < len(input_ids):
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]

        if len(input_ids) > 0:
            if direction == "left":
                sample[f"{prefix}input_ids"] = torch.full((max_length,), pad_token_id)
                sample[f"{prefix}input_ids"][-len(input_ids) :] = input_ids
                sample[f"{prefix}attention_mask"] = torch.zeros(max_length)
                sample[f"{prefix}attention_mask"][-len(input_ids) :] = attention_mask
            else:
                sample[f"{prefix}input_ids"] = torch.full((max_length,), pad_token_id)
                sample[f"{prefix}input_ids"][: len(input_ids)] = input_ids
                sample[f"{prefix}attention_mask"] = torch.zeros(max_length)
                sample[f"{prefix}attention_mask"][: len(input_ids)] = attention_mask
        else:
            # Pad everything if empty (continued pretraining)
            sample[f"{prefix}input_ids"] = torch.full((max_length,), pad_token_id)
            sample[f"{prefix}attention_mask"] = torch.zeros(max_length)

        return sample

    @staticmethod
    def encode(tokenizer, text: str, max_length: int, truncation_side: str) -> Dict:
        encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encodings["input_ids"] = encodings["input_ids"][0]
        encodings["attention_mask"] = encodings["attention_mask"][0]
        if truncation_side == "right":
            encodings["input_ids"] = encodings["input_ids"][:max_length]
            encodings["attention_mask"] = encodings["attention_mask"][:max_length]
        else:
            encodings["input_ids"] = encodings["input_ids"][-max_length:]
            encodings["attention_mask"] = encodings["attention_mask"][-max_length:]
        return encodings
