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


class ConversationChainHandler:
    """
    Partitions the dataset into conversation chains.
    The conversation chains consists of a list of conversations, where each conversation round consists of
    a triple of (system, prompt, answer).

    The conversation may be:
    - a single (system, prompt, answer) round,
      if the conversation is not chained
      (cfg.dataset.parent_id_column == "None" or there is no parent_id for the conversation)
    - a conversation potentially starting somewhere in the middle of the conversation,
      if the conversation is chained and limit_chained_samples is set to False
    - always a complete conversation, if the conversation is chained
      and limit_chained_samples is set to True
    """

    def __init__(self, df, cfg):
        if cfg.dataset.parent_id_column != "None":
            id2children_id = {
                parent_id: id
                for id, parent_id in zip(df["id"], df[cfg.dataset.parent_id_column])
            }
            if cfg.dataset.limit_chained_samples:
                conversation_start_ids = [
                    idx for idx in df["id"].values if idx not in id2children_id
                ]
            else:
                conversation_start_ids = df["id"].values

            conversation_ids_lists = [
                self.get_conversation_ids(id2children_id, conversation_start_id)
                for conversation_start_id in conversation_start_ids
            ]
            # map from df["id"] to enumeration index
            dataframeid2idx = {
                id: idx for idx, id in enumerate(df["id"].astype(str).tolist())
            }
            self.conversation_ids_lists = [
                [
                    dataframeid2idx[conversation_id]
                    for conversation_id in conversation_ids
                ]
                for conversation_ids in conversation_ids_lists
            ]
        else:
            # no parent id column, so each sample is a conversation chain
            self.conversation_ids_lists = [[idx] for idx in range(len(df))]

        self.prompts = get_texts(df, cfg, separator="")
        if cfg.dataset.answer_column not in df.columns:
            self.answers = df[cfg.dataset.answer_column].astype(str).tolist()
        else:
            self.answers = ["" for _ in range(len(self.prompts))]
        self.systems = ["" for _ in range(len(self.prompts))]

        if cfg.dataset.system_column != "None":
            if cfg.dataset.system_column not in df.columns:
                logger.warning(
                    f"System column {cfg.dataset.system_column} not found."
                    f"Disabling functionality."
                )
            else:
                self.systems = df[cfg.dataset.system_column].astype(str).tolist()

    def get_conversation_ids(self, id2children_id, start_id):
        loop_counter = 0  # prevent infinite loops in case of circular parent chains (dataframe issue)

        conversation_chain_ids = [start_id]
        current_id = start_id
        while current_id in id2children_id:
            loop_counter += 1
            current_id = id2children_id[current_id]
            conversation_chain_ids.append(current_id)
            if loop_counter > 1000:
                raise ValueError(
                    f"Parent chain of sample with idx {start_id} exceeds max loop count. "
                    f"Please ensure that parent chain is not circular."
                )
        return conversation_chain_ids

    def __len__(self):
        return len(self.conversation_ids_lists)

    def __getitem__(self, idx):
        """
        Gets a single conversation chain.
        Args:
            idx:

        Returns:

        """
        prompts = [self.prompts[i] for i in self.conversation_ids_lists[idx]]
        answers = [self.answers[i] for i in self.conversation_ids_lists[idx]]
        systems = [self.systems[i] for i in self.conversation_ids_lists[idx]]
        return {
            "prompts": prompts,
            "answers": answers,
            "systems": systems,
        }


class CustomDataset(Dataset):
    """Base PyTorch dataset for any problem type."""

    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        """
        Args:
            df: input DataFrame
            cfg: config with all the hyperparameters
            mode: dataset mode. One of {"train", "validation"}
        """
        assert mode in [
            "train",
            "validation",
        ], f"There is no {mode} for the datasets"

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()

        self.tokenizer = get_tokenizer(self.cfg)
        self.conversation_chain_handler = ConversationChainHandler(self.df, self.cfg)
        if cfg.environment._local_rank == 0:
            text_dict = self.conversation_chain_handler[0]
            self.parse_text_dict(text_dict)
            logger.info(f"Sample prompt: " f"{text_dict['prompts'][0]} ")

    def __len__(self) -> int:
        return len(self.conversation_chain_handler)

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        input_text_dict = self.conversation_chain_handler[idx]
        self.parse_text_dict(input_text_dict)

        sample = dict()
        encodings, system_encoding = self.get_encodings(input_text_dict=input_text_dict)

        input_ids = torch.cat([torch.cat(encoding) for encoding in encodings])
        sample.update(self.get_labels(encodings))
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
        return sample

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

    def parse_text_dict(self, input_text_dict):
        input_text_dict["systems"] = [
            self.parse_system(self.cfg, system) for system in input_text_dict["systems"]
        ]
        input_text_dict["prompt"] = [
            self.parse_prompt(self.cfg, prompt) for prompt in input_text_dict["prompt"]
        ]

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

        del output["predicted_answer_ids"]
        return output

    @staticmethod
    def clean_output(
        output: Dict,
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
            output = self.clean_output(output, cfg)

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

        assert cfg.dataset.answer_column in df.columns, (
            f"Answer column {cfg.dataset.answer_column} not found in the "
            f"{mode} DataFrame."
        )
        assert df.shape[0] == df[[cfg.dataset.answer_column]].dropna().shape[0], (
            f"The {mode} DataFrame"
            f" column {cfg.dataset.answer_column}"
            " contains missing values."
        )
        if cfg.dataset.parent_id_column != "None":
            assert (
                "id" in df.columns
            ), "When using parent column, the dataframe requires an 'id' column. "

    def get_labels(self, encodings):
        labels = torch.cat([torch.cat(encoding) for encoding in encodings]).clone()

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
        if self.cfg.tokenizer.max_length < len(labels):
            labels = labels[-self.cfg.tokenizer.max_length :]

        sample = dict(labels=torch.full((self.cfg.tokenizer.max_length,), -100))
        sample["labels"][-len(labels) :] = labels
        return sample

    def get_encodings(self, input_text_dict: Dict[str, List[str]]):
        """
        Get encodings for a single conversation history.
        Args:
            input_text_dict: A dictionary containing the input text for a single sample.
            Contains the keys "systems", "prompts", "answers". System may be an empty string.

        Returns:
            encodings: A list of encodings for the sample. Each encoding is a tuple of
            (prompt_encoding, answer_encoding). The encodings are sorted such that the start of
            the conversation is at the beginning of the list. If the system text is not empty,
            the first prompt encoding is the encoding of the system text and the first prompt.
            system_encoding: System encoding for the sample. The system encoding is taken from
            the first system text, as this marks the beginning of the conversation.
        """
        encodings = [
            self._get_sample_encoding(system, prompt, answer)
            for idx, (prompt, answer, system) in enumerate(
                zip(
                    input_text_dict["systems"],
                    input_text_dict["prompts"],
                    input_text_dict["answers"],
                )
            )
        ]

        if self.mode == "train":
            parent_encodings = encodings[:-1]
            # randomly skip parent
            parent_encodings = [
                encoding
                for idx, encoding in enumerate(parent_encodings)
                if np.random.random() > self.cfg.augmentation.skip_parent_probability
            ]
            # randomly replace parent with another parent
            if np.random.random() > self.cfg.augmentation.replace_parent_probability:
                idx = np.random.randint(len(self.conversation_chain_handler.prompts))
                parent_encodings = [
                    self._get_sample_encoding(
                        self.conversation_chain_handler.systems[idx],
                        self.conversation_chain_handler.prompts[idx],
                        self.conversation_chain_handler.answers[idx],
                    )
                ] + parent_encodings[1:]

            encodings = parent_encodings + [encodings[-1]]

        system_encoding = encodings[0][0]
        # remove system encodings from list of encodings to only keep prompt and answer
        encodings = [encoding[1:] for encoding in encodings]
        # concatenate system encoding with root prompt encoding
        encodings[0][0] = torch.cat([system_encoding, encodings[0][0]])
        return encodings, system_encoding

    def _get_sample_encoding(self, system: str, prompt: str, answer: str) -> List:
        if len(system) > 0:
            system_encoding = self.encode(
                self.tokenizer, system, self.cfg.tokenizer.max_length_prompt, "right"
            )["input_ids"]
        else:
            system_encoding = torch.empty(0)
        prompt_encoding = self.encode(
            self.tokenizer, prompt, self.cfg.tokenizer.max_length_prompt, "left"
        )["input_ids"]
        max_length_answer = self.cfg.tokenizer.max_length_answer - int(
            self.cfg.dataset.add_eos_token_to_answer
        )
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

    def get_chained_prompt_text_list(self, idx) -> List[str]:
        text_separator = "TEXT_SEPARATOR"
        text_dict = self.conversation_chain_handler[idx]
        # system_prompt == Start of conversation
        chat_history = "".join(
            [
                prompt + text_separator + answer + text_separator
                for prompt, answer in zip(
                    text_dict["prompt"][:-1], text_dict["answer"][:-1]
                )
            ]
        )

        # system prompt may be an empty string
        prompt_text = (
            text_dict["system_prompt"][0] + chat_history + text_dict["prompt"][-1]
        )
        return prompt_text.split(text_separator)

    def pad_tokens(
        self,
        input_ids,
        attention_mask,
        max_length,
        pad_token_id,
        direction="left",
        prefix="",
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
