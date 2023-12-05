import logging
from typing import Any, Dict

import pandas as pd
import torch

import llm_studio.src.datasets.text_causal_language_modeling_ds as text_causal_language_modeling_ds  # noqa: [F401]
from llm_studio.src.datasets.conversation_chain_handler import ConversationChainHandler
from llm_studio.src.utils.utils import PatchedAttribute

logger = logging.getLogger(__name__)


class CustomDataset(text_causal_language_modeling_ds.CustomDataset):
    """
    Dataset for DPO optimization.
    The data is assumed to be in the same format as for causal language modeling,
    but an additional column with rejected answers is required.
    For chained conversations, rejected answers are equal normal answers up to the
    last answer. The last answer is the rejected answer.
    """

    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        assert (
            cfg.dataset.limit_chained_samples
        ), "Need to enable limit_chained_samples for dpo training"
        super().__init__(df=df, cfg=cfg, mode=mode)
        with PatchedAttribute(
            cfg.dataset, "answer_column", cfg.dataset.rejected_answer_column
        ):
            self.conversation_chain_handler_rejected = ConversationChainHandler(
                self.df, cfg
            )

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        chosen_sample = super().__getitem__(idx)
        keys = ["input_ids", "attention_mask", "token_type_ids", "labels"]
        prompt_keys = [
            "prompt_input_ids",
            "prompt_attention_mask",
            "prompt_token_type_ids",
        ]
        prompt_sample = {k: v for k, v in chosen_sample.items() if k in prompt_keys}
        chosen_sample = {
            f"chosen_{k}": v for k, v in chosen_sample.items() if k in keys
        }

        with PatchedAttribute(
            self, "conversation_chain_handler", self.conversation_chain_handler_rejected
        ):
            rejected_sample = {
                f"rejected_{k}": v
                for k, v in super().__getitem__(idx).items()
                if k in keys
            }

        sample = {**chosen_sample, **rejected_sample, **prompt_sample}
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
        try:
            labels[-len(answer_encodings[-1]) :] = answer_encodings[-1]
        except Exception as e:
            raise ValueError(
                f"Could not get labels for prompt_encodings={prompt_encodings} "
                f"and answer_encodings={answer_encodings}."
            ) from e

        if self.cfg.dataset.add_eos_token_to_answer:
            # eos_token may be equal to pad_token. Add the label back manually.
            labels[-1] = self.tokenizer.eos_token_id
        if self.cfg.tokenizer.max_length < len(labels):
            labels = labels[-self.cfg.tokenizer.max_length :]

        sample = dict(labels=torch.full((self.cfg.tokenizer.max_length,), -100))
        sample["labels"][-len(labels) :] = labels
        return sample

    @classmethod
    def sanity_check(cls, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        """
        Quick check whether Dataframe and configurations are correctly set.
        """
        super().sanity_check(df=df, cfg=cfg, mode=mode)
        assert cfg.dataset.rejected_answer_column in df.columns, (
            f"Answer column {cfg.dataset.rejected_answer_column} not found in the "
            f"{mode} DataFrame."
        )
