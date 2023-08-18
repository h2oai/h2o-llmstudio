import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from llm_studio.src.datasets.text_causal_language_modeling_ds import (
    CustomDataset as CausalLMCustomDataset,
)
from llm_studio.src.datasets.text_utils import get_texts

logger = logging.getLogger(__name__)


class CustomDataset(CausalLMCustomDataset):
    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        assert (
            cfg.dataset.system_column == "None"
        ), "RLHF is not compatible with system column."
        super().__init__(df, cfg, mode)
        self.raw_prompts = get_texts(df, self.cfg, separator="")

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        sample = super().__getitem__(idx)
        idx = self.indices[idx]
        sample["reward_model_prompt_text"] = self.get_reward_model_prompt_text(idx)
        return sample

    def get_labels(self, encodings):
        if self.mode == "train":  # no labels required for RLHF during training
            return dict()
        else:
            return super().get_labels(encodings)

    def get_encodings(self, idx):
        encodings, system_encoding = super().get_encodings(idx)
        # remove last ground truth answer,
        # as RLHF will generate the answer from the prompt
        encodings[-1][-1] = torch.empty(0)
        return encodings, system_encoding

    def get_reward_model_prompt_text(self, idx):
        return (
            "".join(
                [
                    self.raw_prompts[int(parent_idx)]
                    + "<|endoftext|>"
                    + self.answers[int(parent_idx)]
                    + "<|endoftext|>"
                    for parent_idx in self.get_parent_ids(idx)
                ]
            )
            + self.raw_prompts[idx]
        )

    def postprocess_batch_predictions(self, cfg: Any, output: Dict) -> Dict:
        if cfg.prediction.metric == "Perplexity":
            return output
        predicted_text = [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in output["predicted_answer_ids"]
        ]
        output["predicted_text"] = np.array(predicted_text)
        output["predicted_answer_ids"] = output["predicted_answer_ids"].detach()
        return output
