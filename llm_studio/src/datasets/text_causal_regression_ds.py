import logging
from typing import Any

import numpy as np
import pandas as pd

from llm_studio.src.datasets.text_causal_language_modeling_ds import (
    CustomDataset as TextCausalLanguageModelingCustomDataset,
)
from llm_studio.src.utils.exceptions import LLMDataException

logger = logging.getLogger(__name__)


class CustomDataset(TextCausalLanguageModelingCustomDataset):
    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        super().__init__(df=df, cfg=cfg, mode=mode)
        self.answers_float = df[cfg.dataset.answer_column].astype(float).values

        if cfg.dataset.parent_id_column != "None":
            raise LLMDataException(
                "Parent ID column is not supported for regression datasets."
            )

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        sample["class_label"] = self.answers_float[idx]
        return sample

    def postprocess_output(self, cfg, df: pd.DataFrame, output: dict) -> dict:
        output["predictions"] = output["predictions"].float()
        preds = []
        for col in np.arange(len(cfg.dataset.answer_column)):
            preds.append(
                np.round(output["predictions"][:, col].cpu().numpy(), 3).astype(str)
            )
        preds = [",".join(pred) for pred in zip(*preds, strict=False)]
        output["predicted_text"] = preds
        return super().postprocess_output(cfg, df, output)

    def clean_output(self, output, cfg):
        return output

    @classmethod
    def sanity_check(cls, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        for answer_col in cfg.dataset.answer_column:
            assert answer_col in df.columns, (
                f"Answer column {answer_col} not found in the {mode} DataFrame."
            )
            assert df.shape[0] == df[answer_col].dropna().shape[0], (
                f"The {mode} DataFrame column {answer_col} contains missing values."
            )
