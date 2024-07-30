import logging
from typing import Any, Dict

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
        self.answers_float = df[cfg.dataset.answer_column].astype(float).values.tolist()

        if cfg.dataset.parent_id_column != "None":
            raise LLMDataException(
                "Parent ID column is not supported for regression datasets."
            )

    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)
        sample["class_label"] = self.answers_float[idx]
        return sample

    def postprocess_output(self, cfg, df: pd.DataFrame, output: Dict) -> Dict:
        output["logits"] = output["logits"].float()
        preds = output["logits"]
        preds = np.array(preds).astype(float).astype(str).reshape(-1)
        output["predicted_text"] = preds
        return super().postprocess_output(cfg, df, output)

    def clean_output(self, output, cfg):
        return output
