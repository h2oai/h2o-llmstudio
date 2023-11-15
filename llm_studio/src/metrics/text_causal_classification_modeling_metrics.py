import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import softmax
from sklearn.metrics import log_loss, roc_auc_score

logger = logging.getLogger(__name__)


def accuracy_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    predicted_text = np.array([int(text) for text in results["predicted_text"]])
    target_text = np.array([int(text) for text in results["target_text"]])
    return (predicted_text == target_text).astype("float")


def auc_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    logits = np.array(results["logits"])
    target_text = np.array([int(text) for text in results["target_text"]])
    if cfg.dataset.num_classes > 1:
        target_text = np.eye(cfg.dataset.num_classes)[target_text]
    return roc_auc_score(target_text, logits, multi_class="ovr")


def logloss_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    logits = np.array(results["logits"])
    target_text = np.array([int(text) for text in results["target_text"]])
    if cfg.dataset.num_classes > 1:
        target_text = np.eye(cfg.dataset.num_classes)[target_text]
        logits = softmax(logits, axis=1)
    return log_loss(target_text, logits, eps=1e-7)


class Metrics:
    """
    Metrics factory. Returns:
        - metric value
        - should it be maximized or minimized
        - Reduce function

    Maximized or minimized is needed for early stopping (saving best checkpoint)
    Reduce function to generate a single metric value, usually "mean" or "none"
    """

    _metrics = {
        "AUC": (auc_score, "max", "mean"),
        "Accuracy": (accuracy_score, "max", "mean"),
        "LogLoss": (logloss_score, "min", "mean"),
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._metrics.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Metrics.

        Args:
            name: metrics name
        Returns:
            A class to build the Metrics
        """
        return cls._metrics.get(name, cls._metrics["LogLoss"])
