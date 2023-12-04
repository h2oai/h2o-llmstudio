import logging
from typing import Any, Dict, Tuple

import pandas as pd
import torch

import llm_studio.src.datasets.text_causal_language_modeling_ds as text_causal_language_modeling_ds
from llm_studio.src.datasets.conversation_chain_handler import ConversationChainHandler
from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.utils.utils import PatchedAttribute

logger = logging.getLogger(__name__)


class CustomDataset(text_causal_language_modeling_ds.CustomDataset):
    """
    Dataset for DPO optimization.
    The data is assumed to be in (potential) hierarchical format.
    id and parent_id are not necessarily required.

    Example format from HH DPO dataset:

    Beginning of a chat-answer interaction (parent_id is not set):
        instruction                    What kind of noises did dinosaurs make?
        id                                610e4ad5-09c4-4055-9ff4-948fe6b4f832
        parent_id                                                         None
        chosen_response       Humans and dinosaurs didn’t live at the same t...
        rejected_response     Humans and dinosaurs didn’t live at the same t...
    Within a chat-answer interaction (parent_id points for the previous prompt-answer sample):
        instruction                                               yes they did
        output               to guess, and that would probably require lots...
        id                                573e8d77-550a-4889-8ff4-1e8d8944897c
        parent_id                         610e4ad5-09c4-4055-9ff4-948fe6b4f832
        chosen_response      to guess, and that would probably require lots...
        rejected_response    to guess, and that would probably require lots...
    Last question. Chosen and rejected responses are different:
        instruction          Do have a phone number or email address for hi...
        output
        id                                e0edeaf1-166d-4683-8609-dcba6fafc520
        parent_id                         e7e96d54-006d-4b34-a9ed-479c3ec3068c
        chosen_response       He doesn’t have a publicly available phone nu...
        rejected_response     If you want to contact Ryan Reynolds by phone...
    """

    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        assert (
            cfg.dataset.limit_chained_samples
        ), "Need to enable limit_chained_samples for dpo training"

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.tokenizer = get_tokenizer(self.cfg)

        with PatchedAttribute(
            cfg.dataset, "answer_column", cfg.dataset.chosen_response_column
        ):
            self.conversation_chain_handler_chosen = ConversationChainHandler(
                self.df, cfg
            )

        with PatchedAttribute(
            cfg.dataset, "answer_column", cfg.dataset.rejected_response_column
        ):
            self.conversation_chain_handler_rejected = ConversationChainHandler(
                self.df, cfg
            )

    def __len__(self) -> int:
        return len(self.conversation_chain_handler_chosen)

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        sample = {}
        for name, conversation_chain_handler in zip(
            ["chosen", "rejected"],
            [
                self.conversation_chain_handler_chosen,
                self.conversation_chain_handler_rejected,
            ],
        ):
            with PatchedAttribute(
                self, "conversation_chain_handler", conversation_chain_handler
            ):
                sample.update(
                    {
                        f"{name}_{key}": value
                        for key, value in super().__getitem__(idx).items()
                        if key
                        in ["input_ids", "attention_mask", "token_type_ids", "labels"]
                    }
                )

        # Used chosen responses for functionality related to generation
        with PatchedAttribute(
            self, "conversation_chain_handler", self.conversation_chain_handler_chosen
        ):
            sample.update(
                {
                    key: value
                    for key, value in super().__getitem__(idx).items()
                    if key
                    in [
                        "prompt_input_ids",
                        "prompt_attention_mask",
                        "prompt_token_type_ids",
                    ]
                }
            )

        return sample

    def get_labels(self, prompt_encodings, answer_encodings):
        """
        Mask all but the last answer.
        """
        labels = torch.cat(
            [
                torch.cat(
                    [
                        torch.full_like(
                            prompt_encoding,
                            fill_value=-100,
                        ),
                        torch.full_like(
                            answer_encoding,
                            fill_value=-100,
                        ),
                    ]
                )
                for prompt_encoding, answer_encoding in zip(
                    prompt_encodings, answer_encodings
                )
            ]
        ).clone()
        labels[-len(answer_encodings[-1]) :] = answer_encodings[-1]

        if self.cfg.dataset.add_eos_token_to_answer:
            # eos_token may be equal to pad_token. Add the label back manually.
            labels[-1] = self.tokenizer.eos_token_id
        if self.cfg.tokenizer.max_length < len(labels):
            labels = labels[-self.cfg.tokenizer.max_length :]

        sample = dict(labels=torch.full((self.cfg.tokenizer.max_length,), -100))
        sample["labels"][-len(labels) :] = labels
        return sample

    def postprocess_output(self, cfg, df: pd.DataFrame, output: Dict) -> Dict:
        with PatchedAttribute(
            self, "conversation_chain_handler", self.conversation_chain_handler_chosen
        ):
            return super().postprocess_output(cfg, df, output)

    def format_output(
        self, cfg, df: pd.DataFrame, output: Dict
    ) -> Tuple[Dict, pd.DataFrame]:
        with PatchedAttribute(
            cfg.dataset, "answer_column", cfg.dataset.chosen_answer_column
        ):
            with PatchedAttribute(
                self,
                "conversation_chain_handler",
                self.conversation_chain_handler_chosen,
            ):
                return super().format_output(cfg, df, output)

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
        for answer_column in [
            cfg.dataset.chosen_answer_column,
            cfg.dataset.rejected_answer_column,
        ]:
            assert answer_column in df.columns, (
                f"Answer column {answer_column} not found in the " f"{mode} DataFrame."
            )
            assert df.shape[0] == df[[answer_column]].dropna().shape[0], (
                f"The {mode} DataFrame"
                f" column {answer_column}"
                " contains missing values."
            )

        if cfg.dataset.parent_id_column != "None":
            assert (
                "id" in df.columns
            ), "When using parent column, the dataframe requires an 'id' column. "
