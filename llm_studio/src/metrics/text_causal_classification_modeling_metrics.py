from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import softmax
from sklearn.metrics import log_loss, roc_auc_score

from llm_studio.python_configs.base import DefaultConfigProblemBase


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
    logits = np.array(results["logits"])
    target = np.array(
        [[int(t) for t in text.split(",")] for text in results["target_text"]]
    )

    # multi class or single binary classification
    if len(cfg.dataset.answer_column) == 1:
        if cfg.dataset.num_classes == 1:
            predicted = logits > 0.5
        else:
            predicted = np.argmax(softmax(logits, axis=-1), axis=-1)

    else:
        predicted = []
        for col in range(len(cfg.dataset.answer_column)):
            predicted.append(np.round(logits[:, col]))
        predicted = np.array(predicted).T

    # Input validation
    if len(target) != len(predicted):
        raise ValueError(
            f"Length of target ({len(target)}) and predicted ({len(predicted)}) "
            "should be the same."
        )
    if len(target) == 0:
        raise ValueError("No data to calculate accuracy score")

    return (predicted == target).mean(axis=1).reshape(-1).astype("float")


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
    target = np.array(
        [[int(t) for t in text.split(",")] for text in results["target_text"]]
    )

    # Input validation
    if len(target) != len(logits):
        raise ValueError(
            f"Length of target ({len(target)}) and logits ({len(logits)}) "
            "should be the same."
        )
    if len(target) == 0:
        raise ValueError("No data to calculate AUC score.")

    if target.shape[1] == 1 and cfg.dataset.num_classes > 1:
        target = np.eye(cfg.dataset.num_classes)[target.reshape(-1)]
    return roc_auc_score(target, logits, multi_class="ovr")


def logloss_score(
    cfg: DefaultConfigProblemBase,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    """Calculate the Log Loss (Cross-Entropy Loss) score.

    This function computes the log loss using the predicted probabilities and target values.
    It supports binary, multiclass, and multilabel classification.

    Args:
        cfg: DefaultConfigProblemBase, configuration
        results: Dict, model results including 'probabilities' and 'target_text'
        val_df: pd.DataFrame, ignored
        raw_results: bool, ignored

    Returns:
        float: Log Loss score

    Raises:
        ValueError: If input data is invalid or inconsistent
    """
    predictions = np.array(results["probabilities"])
    target = np.array(
        [[int(t) for t in text.split(",")] for text in results["target_text"]]
    )

    # Input validation
    if len(target) != len(predictions):
        raise ValueError(
            f"Length of target ({len(target)}) and predictions ({len(predictions)}) "
            "should be the same."
        )
    if len(target) == 0:
        raise ValueError("No data to calculate log loss.")

    # Handle multilabel case
    if len(cfg.dataset.answer_column) > 1:
        log_losses = []
        for col in range(len(cfg.dataset.answer_column)):
            log_losses.append(log_loss(target[:, col], predictions[:, col]))
        return np.mean(log_losses)

    # Handle binary and multiclass cases
    if cfg.dataset.num_classes > 1:
        target = np.eye(cfg.dataset.num_classes)[target.reshape(-1)]
    return log_loss(target, predictions)


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
