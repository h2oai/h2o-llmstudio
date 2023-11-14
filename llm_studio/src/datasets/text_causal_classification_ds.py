from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from llm_studio.src.datasets.text_causal_language_modeling_ds import (
    CustomDataset as TextCausalLanguageModelingCustomDataset,
)
from llm_studio.src.utils.exceptions import LLMDataException


class CustomDataset(TextCausalLanguageModelingCustomDataset):
    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        super().__init__(df=df, cfg=cfg, mode=mode)
        check_for_non_int_answers(cfg, df)
        self.answers_int = df[cfg.dataset.answer_column].astype(int).values.tolist()
        if 1 < cfg.dataset.num_classes <= max(self.answers_int):
            raise LLMDataException(
                "Number of classes is smaller than max label "
                f"{max(self.answers_int)}. Please increase the setting accordingly."
            )
        elif cfg.dataset.num_classes == 1 and max(self.answers_int) > 1:
            raise LLMDataException(
                "For binary classification, max label should be 1 but is "
                f"{max(self.answers_int)}."
            )
        if cfg.dataset.parent_id_column != "None":
            raise LLMDataException(
                "Parent ID column is not supported for classification datasets."
            )

    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)
        sample["class_label"] = self.answers_int[idx]
        return sample

    def postprocess_output(self, cfg, df: pd.DataFrame, output: Dict) -> Dict:
        if cfg.dataset.num_classes == 1:
            preds = output["logits"]
            preds = np.array((preds > 0.0)).astype(int).astype(str).reshape(-1)
        else:
            preds = output["logits"]
            preds = (
                np.array(torch.argmax(preds, dim=1))  # type: ignore[arg-type]
                .astype(str)
                .reshape(-1)
            )
        output["predicted_text"] = preds
        return super().postprocess_output(cfg, df, output)

    def clean_output(self, output, cfg):
        return output

    @classmethod
    def sanity_check(cls, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        # TODO: Dataset import in UI is currently using text_causal_language_modeling_ds
        check_for_non_int_answers(cfg, df)


def check_for_non_int_answers(cfg, df):
    answers_non_int = [
        x for x in df[cfg.dataset.answer_column].values if not is_castable_to_int(x)
    ]
    if len(answers_non_int) > 0:
        raise LLMDataException(
            f"Column {cfg.dataset.answer_column} contains non int items. "
            f"Sample values: {answers_non_int[:5]}."
        )


def is_castable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
