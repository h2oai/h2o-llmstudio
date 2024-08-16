from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from llm_studio.src.metrics.text_causal_regression_modeling_metrics import (
    Metrics,
    mae_score,
    mse_score,
)


@pytest.fixture
def mock_val_df():
    return pd.DataFrame()


@pytest.fixture
def mock_cfg():
    return MagicMock()


def test_mse_score_single_value():
    results = {
        "predictions": [[1.0], [2.0], [3.0], [4.0]],
        "target_text": ["2.0", "2.0", "2.0", "2.0"],
    }
    cfg = MagicMock()
    val_df = pd.DataFrame()

    score = mse_score(cfg, results, val_df)

    expected = np.array([1.0, 0.0, 1.0, 4.0])
    np.testing.assert_almost_equal(score, expected)


def test_mse_score_multiple_values():
    results = {
        "predictions": [[1.0, 2.0], [3.0, 4.0]],
        "target_text": ["2.0,3.0", "3.0,3.0"],
    }
    cfg = MagicMock()
    val_df = pd.DataFrame()

    score = mse_score(cfg, results, val_df)

    expected = np.array([1.0, 0.5])
    np.testing.assert_almost_equal(score, expected)


def test_mae_score_single_value():
    results = {
        "predictions": [[1.0], [2.0], [3.0], [4.0]],
        "target_text": ["2.0", "2.0", "2.0", "2.0"],
    }
    cfg = MagicMock()
    val_df = pd.DataFrame()

    score = mae_score(cfg, results, val_df)

    expected = np.array([1.0, 0.0, 1.0, 2.0])
    np.testing.assert_almost_equal(score, expected)


def test_mae_score_multiple_values():
    results = {
        "predictions": [[1.0, 2.0], [3.0, 4.0]],
        "target_text": ["2.0,3.0", "3.0,3.0"],
    }
    cfg = MagicMock()
    val_df = pd.DataFrame()

    score = mae_score(cfg, results, val_df)

    expected = np.array([1.0, 0.5])
    np.testing.assert_almost_equal(score, expected)


def test_metrics_names():
    assert Metrics.names() == ["MAE", "MSE"]


def test_metrics_get_mse():
    metric = Metrics.get("MSE")
    assert metric[0] == mse_score
    assert metric[1] == "min"
    assert metric[2] == "mean"


def test_metrics_get_mae():
    metric = Metrics.get("MAE")
    assert metric[0] == mae_score
    assert metric[1] == "min"
    assert metric[2] == "mean"


def test_metrics_get_unknown():
    metric = Metrics.get("Unknown")
    assert metric[0] == mse_score
    assert metric[1] == "min"
    assert metric[2] == "mean"


def test_mse_score_empty_input():
    results = {"predictions": [], "target_text": []}
    cfg = MagicMock()
    val_df = pd.DataFrame()

    with pytest.raises(ValueError):
        mse_score(cfg, results, val_df)


def test_mae_score_empty_input():
    results = {"predictions": [], "target_text": []}
    cfg = MagicMock()
    val_df = pd.DataFrame()

    with pytest.raises(ValueError):
        mae_score(cfg, results, val_df)


def test_mse_score_ignore_raw_results(mock_cfg, mock_val_df):
    results = {"predictions": [[1.0], [2.0]], "target_text": ["2.0", "2.0"]}

    score_without_raw = mse_score(mock_cfg, results, mock_val_df)
    score_with_raw = mse_score(mock_cfg, results, mock_val_df, raw_results=True)

    np.testing.assert_array_equal(score_without_raw, score_with_raw)


def test_mae_score_ignore_raw_results(mock_cfg, mock_val_df):
    results = {"predictions": [[1.0], [2.0]], "target_text": ["2.0", "2.0"]}

    score_without_raw = mae_score(mock_cfg, results, mock_val_df)
    score_with_raw = mae_score(mock_cfg, results, mock_val_df, raw_results=True)

    np.testing.assert_array_equal(score_without_raw, score_with_raw)
