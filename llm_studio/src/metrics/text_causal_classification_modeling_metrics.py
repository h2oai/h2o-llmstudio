import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import softmax
from sklearn.metrics import log_loss, roc_auc_score

from llm_studio.python_configs.base import DefaultConfigProblemBase

logger = logging.getLogger(__name__)


def accuracy_score(
    cfg: DefaultConfigProblemBase,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    """Calculate accuracy score.

    Only considers the predicted value (results["predicted_text"]) and target value
    (results["target_text"]).
    It supports both binary and multiclass classification.

    Args:
        cfg: DefaultConfigProblemBase, ignored
        results: Dict, model results including 'predicted_text' and 'target_text'
        val_df: pd.DataFrame, validation dataframe
        raw_results: bool, ignored

    Returns:
        Numpy array of 0.0 or 1.0 for each sample

    Raises:
        ValueError: If input data is invalid or inconsistent
    """
    predicted_text = np.array([int(text) for text in results["predicted_text"]])
    target_text = np.array([int(text) for text in results["target_text"]])

    # Input validation
    if len(target_text) != len(predicted_text):
        raise ValueError(
            f"Length of target_text ({len(target_text)}) and predicted_text "
            f"({len(predicted_text)}) should be the same."
        )
    if len(target_text) == 0:
        raise ValueError("No data to calculate accuracy score")

    return (predicted_text == target_text).astype("float")


def auc_score(
    cfg: DefaultConfigProblemBase,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    """Calculate Area Under the ROC Curve (AUC) score.

    This function computes the AUC score using the predicted logits and target values.
    It supports both binary and multiclass classification.

    Args:
        cfg: DefaultConfigProblemBase, configuration
        results: Dict, model results including 'logits' and 'target_text'
        val_df: pd.DataFrame, ignored
        raw_results: bool, ignored

    Returns:
        float: AUC score for binary classification
        NDArray: AUC scores for multiclass classification (one-vs-rest)

    Raises:
        ValueError: If input data is invalid or inconsistent
    """
    logits = np.array(results["logits"])
    target_text = np.array([int(text) for text in results["target_text"]])

    # Input validation
    if len(target_text) != len(logits):
        raise ValueError(
            f"Length of target_text ({len(target_text)}) and logits ({len(logits)}) "
            "should be the same."
        )
    if len(target_text) == 0:
        raise ValueError("No data to calculate AUC score.")

    if cfg.dataset.num_classes > 1:
        target_text = np.eye(cfg.dataset.num_classes)[target_text]
    return roc_auc_score(target_text, logits, multi_class="ovr")


def logloss_score(
    cfg: DefaultConfigProblemBase,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    """Calculate the Log Loss (Cross-Entropy Loss) score.

    This function computes the log loss using the predicted logits and target values.
    It supports both binary and multiclass classification.

    Args:
        cfg: DefaultConfigProblemBase, configuration
        results: Dict, results from the model including 'logits' and 'target_text'
        val_df: pd.DataFrame, ignored
        raw_results: bool, ignored

    Returns:
        float: Log Loss score

    Raises:
        ValueError: If input data is invalid or inconsistent
    """
    logits = np.array(results["logits"])
    target_text = np.array([int(text) for text in results["target_text"]])

    # Input validation
    if len(target_text) != len(logits):
        raise ValueError(
            f"Length of target_text ({len(target_text)}) and logits ({len(logits)}) "
            "should be the same."
        )
    if len(target_text) == 0:
        raise ValueError("No data to calculate log loss.")

    if cfg.dataset.num_classes > 1:
        target_text = np.eye(cfg.dataset.num_classes)[target_text]
        logits = softmax(logits, axis=1)
    return log_loss(target_text, logits)


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
