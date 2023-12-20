import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from tabulate import tabulate

from llm_studio.src.datasets.text_causal_language_modeling_ds import (
    CustomDataset as CustomCausalLLMDataset,
)

logger = logging.getLogger(__name__)


class CustomDataset(CustomCausalLLMDataset):
    """Dataset for KTO Language modeling."""

    def __init__(self, df: pd.DataFrame, cfg: Any, mode: str = "train"):
        """
        Args:
            df: input DataFrame
            cfg: config with all the hyperparameters
            mode: dataset mode. One of {"train", "validation"}
        """
        super().__init__(df=df, cfg=cfg, mode=mode)
        # split into preferred - non-preferred answers
        feedback_values = df[cfg.dataset.feedback_column].values
        feedback_values_merged = []
        for (
            conversation_chain_id
        ) in self.conversation_chain_handler.conversation_chain_ids:
            if len(set(feedback_values[conversation_chain_id])) > 1:
                raise ValueError(
                    "Found multiple feedback values in conversation chain {}".format(
                        tabulate(
                            self.df.iloc[conversation_chain_id],
                            headers=df.columns,
                            floatfmt=".5f",
                            showindex=True,
                            tablefmt="psql",
                        )
                    )
                )
            feedback_values_merged.append(feedback_values[conversation_chain_id][0])
        self.feedback_values = np.array(feedback_values_merged)

    def __getitem__(self, idx: int) -> Dict:
        """Reads a single text observation."""
        sample = super().__getitem__(idx)
        sample["feedback"] = self.feedback_values[idx]
        return sample
