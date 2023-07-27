import logging
from typing import Any, Dict, List

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
        if self.cfg.dataset.system_column != "None":
            logger.warning(
                f"RLHF is not compatible with system column. "
                f"Disabling functionality for mode {self.mode}."
            )
            self.cfg.dataset.system_column = "None"

        super().__init__(df, cfg, mode)
        self.raw_prompts = get_texts(df, self.cfg, separator="")

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        idx = self.indices[idx]

        sample = dict()
        encodings, system_encoding = self.get_encodings(idx)
        # ground truth answer not used in RLHF training
        encodings[-1][-1] = torch.empty(0)

        input_ids = torch.cat([torch.cat(encoding) for encoding in encodings])

        rlhf_is_in_training_mode = self.cfg.training.use_rlhf and self.mode == "train"
        if not rlhf_is_in_training_mode:  # no labels required for RLHF during training
            sample.update(self.get_labels(encodings))

        self.pad_and_add_prompt_encoding(input_ids, encodings, sample, system_encoding)
        return sample

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
