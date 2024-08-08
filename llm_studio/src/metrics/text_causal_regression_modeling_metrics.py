import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def mse_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    target = np.array(
        [[float(t) for t in text.split(",")] for text in results["target_text"]]
    )
    predictions = np.array(results["predictions"])

    if len(target) != len(predictions):
        raise ValueError(
            f"Length of target ({len(target)}) and predictions ({len(predictions)}) "
            "should be the same."
        )
    if len(target) == 0:
        raise ValueError("No data to calculate MSE score")

    return ((target - predictions) ** 2).mean(axis=1).reshape(-1).astype("float")


def mae_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    target = np.array(
        [[float(t) for t in text.split(",")] for text in results["target_text"]]
    )
    predictions = np.array(results["predictions"])

    if len(target) != len(predictions):
        raise ValueError(
            f"Length of target ({len(target)}) and predictions ({len(predictions)}) "
            "should be the same."
        )
    if len(target) == 0:
        raise ValueError("No data to calculate MAE score")

    return np.abs(target - predictions).mean(axis=1).reshape(-1).astype("float")


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
        "MSE": (mse_score, "min", "mean"),
        "MAE": (mae_score, "min", "mean"),
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
        return cls._metrics.get(name, cls._metrics["MSE"])
