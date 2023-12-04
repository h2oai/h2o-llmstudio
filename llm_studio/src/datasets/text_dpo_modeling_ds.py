import logging
from typing import Any, Dict

import pandas as pd

import llm_studio.src.datasets.text_causal_language_modeling_ds as text_causal_language_modeling_ds

logger = logging.getLogger(__name__)


class CustomDataset(text_causal_language_modeling_ds.CustomDataset):
    """
    Dataset for DPO optimization.
    The data is assumed to be in hierarchical form of the following format:
    Beginning of a chat-answer interaction (parent_id is not set):
        instruction                    What kind of noises did dinosaurs make?
        output               Humans and dinosaurs didn’t live at the same t...
        id                                610e4ad5-09c4-4055-9ff4-948fe6b4f832
        parent_id                                                         None
        chosen_response                                                   None
        rejected_response                                                 None
    Within a chat-answer interaction (parent_id points for the previous prompt-answer sample):
        instruction                                               yes they did
        output               to guess, and that would probably require lots...
        id                                573e8d77-550a-4889-8ff4-1e8d8944897c
        parent_id                         610e4ad5-09c4-4055-9ff4-948fe6b4f832
        chosen_response                                                   None
        rejected_response                                                 None
    Last question. Output should be empty, chosen and rejected responses should be given:
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

        super().__init__(df=df, cfg=cfg, mode=mode)
        self.chosen_answers = [
            chosen_answer if chosen_answer else answer
            for chosen_answer, answer in zip(
                self.df[self.cfg.dataset.chosen_response_column].tolist(), self.answers
            )
        ]

        self.rejected_answers = [
            rejected_answer if rejected_answer else answer
            for rejected_answer, answer in zip(
                self.df[self.cfg.dataset.rejected_response_column].tolist(),
                self.answers,
            )
        ]
        self.tmp_answers = self.answers

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        sample = {}
        for name, answer in zip(
            ["chosen", "rejected"], [self.chosen_answers, self.rejected_answers]
        ):
            self.answers = answer
            sample.update(
                {
                    f"{name}_{key}": value
                    for key, value in super().__getitem__(idx).items()
                    if key
                    in ["input_ids", "attention_mask", "token_type_ids", "labels"]
                }
            )
        self.answers = self.tmp_answers
        # TODO: Why this?
        sample.update(super().__getitem__(idx))
        return sample

    # TODO: mask all answers except the last one
    # TODO: Metric with chained samples -> Maybe also implement this as a separate metric
