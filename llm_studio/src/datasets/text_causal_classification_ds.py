import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from llm_studio.src.datasets.text_causal_language_modeling_ds import (
    CustomDataset as TextCausalLanguageModelingCustomDataset,
)
from llm_studio.src.utils.exceptions import LLMDataException

logger = logging.getLogger(__name__)


class CustomDataset(TextCausalLanguageModelingCustomDataset):
    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        super().__init__(df=df, cfg=cfg, mode=mode)
        check_for_non_int_answers(cfg, df)
        self.answers_int = df[cfg.dataset.answer_column].astype(int).values
        print(self.answers_int.shape)
        max_value = np.max(self.answers_int)
        min_value = np.min(self.answers_int)
                            
        if 1 < cfg.dataset.num_classes <= max_value:
            raise LLMDataException(
                "Number of classes is smaller than max label "
                f"{max_value}. Please increase the setting accordingly."
            )
        elif cfg.dataset.num_classes == 1 and max_value> 1:
            raise LLMDataException(
                "For binary classification, max label should be 1 but is "
                f"{max_value}."
            )
        if min_value < 0:
            raise LLMDataException(
                "Labels should be non-negative but min label is "
                f"{min_value}."
            )
        if (
            min_value != 0
            or max_value != np.unique(self.answers_int).size - 1
        ):
            logger.warning(
                "Labels should start at 0 and be continuous but are "
                f"{sorted(np.unique(self.answers_int))}."
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
        output["logits"] = output["logits"].float()
        if len(cfg.dataset.answer_column) == 1:
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
        else:
            preds = output["logits"].cpu().numpy().astype(str)
        output["predicted_text"] = preds
        return super().postprocess_output(cfg, df, output)

    def clean_output(self, output, cfg):
        return output

    @classmethod
    def sanity_check(cls, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        # TODO: Dataset import in UI is currently using text_causal_language_modeling_ds
        check_for_non_int_answers(cfg, df)


def check_for_non_int_answers(cfg, df):
    answers_non_int = []
    for column in cfg.dataset.answer_column:
        answers_non_int.extend(
            x for x in df[column].values if not is_castable_to_int(x)
        )
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
