from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.metrics import log_loss, roc_auc_score

from llm_studio.src.metrics.text_causal_classification_modeling_metrics import (
    accuracy_score,
    auc_score,
    logloss_score,
)


@pytest.fixture
def mock_val_df():
    return pd.DataFrame()


def test_accuracy_score_binary_perfect_match(mock_val_df):
    results = {
        "predictions": [[1], [0], [1], [0]],
        "target_text": ["1", "0", "1", "0"],
    }
    cfg = MagicMock()

    score = accuracy_score(cfg, results, mock_val_df)

    assert np.array_equal(score, np.array([1.0, 1.0, 1.0, 1.0]))


def test_accuracy_score_binary_no_match(mock_val_df):
    results = {
        "predictions": [[1], [1], [1], [1]],
        "target_text": ["0", "0", "0", "0"],
    }
    cfg = MagicMock()

    score = accuracy_score(cfg, results, mock_val_df)

    assert np.array_equal(score, np.array([0.0, 0.0, 0.0, 0.0]))


def test_accuracy_score_binary_mixed_results(mock_val_df):
    results = {
        "predictions": [[1], [0], [1], [0]],
        "target_text": ["1", "1", "0", "0"],
    }
    cfg = MagicMock()

    score = accuracy_score(cfg, results, mock_val_df)

    assert np.array_equal(score, np.array([1.0, 0.0, 0.0, 1.0]))


def test_accuracy_score_multiclass_perfect_match(mock_val_df):
    results = {
        "predictions": [[0], [1], [2], [3], [4]],
        "target_text": ["0", "1", "2", "3", "4"],
    }
    cfg = MagicMock()

    score = accuracy_score(cfg, results, mock_val_df)

    assert np.array_equal(score, np.array([1.0, 1.0, 1.0, 1.0, 1.0]))


def test_accuracy_score_multiclass_no_match(mock_val_df):
    results = {
        "predictions": [[1], [2], [3], [4], [0]],
        "target_text": ["0", "1", "2", "3", "4"],
    }
    cfg = MagicMock()

    score = accuracy_score(cfg, results, mock_val_df)

    assert np.array_equal(score, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))


def test_accuracy_score_multiclass_mixed_results(mock_val_df):
    results = {
        "predictions": [[0], [1], [2], [2], [4]],
        "target_text": ["0", "1", "2", "3", "3"],
    }
    cfg = MagicMock()

    score = accuracy_score(cfg, results, mock_val_df)

    assert np.array_equal(score, np.array([1.0, 1.0, 1.0, 0.0, 0.0]))


def test_accuracy_score_invalid_input_empty(mock_val_df):
    results = {"predictions": [], "target_text": []}
    cfg = MagicMock()

    with pytest.raises(ValueError):
        accuracy_score(cfg, results, mock_val_df)


def test_accuracy_score_invalid_input_unequal_length(mock_val_df):
    results = {"predictions": [[1], [0]], "target_text": ["1", "0", "2"]}
    cfg = MagicMock()

    with pytest.raises(ValueError):
        accuracy_score(cfg, results, mock_val_df)


def test_accuracy_score_ignore_raw_results(mock_val_df):
    results = {"predictions": [[1], [0], [2]], "target_text": ["1", "1", "2"]}
    cfg = MagicMock()
    raw_results = True

    score = accuracy_score(cfg, results, mock_val_df, raw_results)

    assert np.array_equal(score, np.array([1.0, 0.0, 1.0]))


def test_accuracy_score_large_class_numbers(mock_val_df):
    results = {
        "predictions": [[10], [20], [30], [40], [50]],
        "target_text": ["10", "20", "30", "40", "60"],
    }
    cfg = MagicMock()

    score = accuracy_score(cfg, results, mock_val_df)

    assert np.array_equal(score, np.array([1.0, 1.0, 1.0, 1.0, 0.0]))


def test_auc_score_binary_classification(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 2
    results = {
        "logits": [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]],
        "target_text": ["1", "0", "1", "0"],
    }

    score = auc_score(cfg, results, mock_val_df)

    expected_score = roc_auc_score([1, 0, 1, 0], [0.9, 0.2, 0.7, 0.1])
    assert np.isclose(score, expected_score)


def test_auc_score_multiclass_classification(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 3
    results = {
        "logits": [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]],
        "target_text": ["1", "0", "2", "2"],
    }

    score = auc_score(cfg, results, mock_val_df)

    expected_score = roc_auc_score(
        np.eye(3)[[1, 0, 2, 2]], np.array(results["logits"]), multi_class="ovr"
    )
    assert np.allclose(score, expected_score)


def test_auc_score_invalid_input_empty(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 2
    results = {"logits": [], "target_text": []}

    with pytest.raises(ValueError):
        auc_score(cfg, results, mock_val_df)


def test_auc_score_invalid_input_unequal_length(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 2
    results = {"logits": [[0.1, 0.9], [0.8, 0.2]], "target_text": ["1", "2", "0", "2"]}

    with pytest.raises(ValueError):
        auc_score(cfg, results, mock_val_df)


def test_auc_score_ignore_val_df_and_raw_results(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 2
    results = {"logits": [[0.1, 0.9], [0.8, 0.2]], "target_text": ["1", "0"]}
    raw_results = True

    score = auc_score(cfg, results, "This should be ignored", raw_results)

    expected_score = roc_auc_score([1, 0], [0.9, 0.2])
    assert np.isclose(score, expected_score)


def test_auc_score_different_number_of_classes(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 4
    results = {
        "logits": [
            [0.1, 0.7, 0.1, 0.1],
            [0.6, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.2, 0.2, 0.3, 0.3],
        ],
        "target_text": ["1", "0", "2", "3"],
    }

    score = auc_score(cfg, results, mock_val_df)

    expected_score = roc_auc_score(
        np.eye(4)[[1, 0, 2, 3]], np.array(results["logits"]), multi_class="ovr"
    )
    assert np.allclose(score, expected_score)


def test_logloss_score_binary_classification(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 2
    cfg.dataset.answer_column = ["label"]
    results = {
        "probabilities": softmax(
            [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]], axis=1
        ),
        "target_text": ["1", "0", "1", "0"],
    }

    score = logloss_score(cfg, results, mock_val_df)

    expected_score = log_loss([1, 0, 1, 0], results["probabilities"])
    assert np.isclose(score, expected_score)


def test_logloss_score_multiclass_classification(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 3
    cfg.dataset.answer_column = ["label"]
    results = {
        "probabilities": softmax(
            [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]], axis=1
        ),
        "target_text": ["1", "0", "2", "2"],
    }

    score = logloss_score(cfg, results, mock_val_df)

    expected_score = log_loss(np.eye(3)[[1, 0, 2, 2]], results["probabilities"])
    assert np.isclose(score, expected_score)


def test_logloss_score_multilabel_classification(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 3
    cfg.dataset.answer_column = ["label1", "label2", "label3"]
    results = {
        "probabilities": [
            [0.1, 0.8, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.1, 0.8],
            [0.3, 0.3, 0.4],
        ],
        "target_text": ["1,0,1", "0,1,0", "1,1,0", "0,0,1"],
    }

    score = logloss_score(cfg, results, mock_val_df)

    expected_scores = []
    for i in range(3):
        expected_scores.append(
            log_loss(
                [int(t.split(",")[i]) for t in results["target_text"]],
                [p[i] for p in results["probabilities"]],
            )
        )
    expected_score = np.mean(expected_scores)
    assert np.isclose(score, expected_score)


def test_logloss_score_invalid_input_empty(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 2
    results = {"probabilities": [], "target_text": []}

    with pytest.raises(ValueError):
        logloss_score(cfg, results, mock_val_df)


def test_logloss_score_invalid_input_unequal_length(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 2
    results = {
        "probabilities": [[0.1, 0.9], [0.8, 0.2]],
        "target_text": ["1", "2", "0"],
    }

    with pytest.raises(ValueError):
        logloss_score(cfg, results, mock_val_df)


def test_logloss_score_ignore_val_df_and_raw_results(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 2
    cfg.dataset.answer_column = ["label"]
    results = {"probabilities": [[0.1, 0.9], [0.8, 0.2]], "target_text": ["1", "0"]}
    raw_results = True

    score = logloss_score(cfg, results, "This should be ignored", raw_results)

    expected_score = log_loss([1, 0], results["probabilities"])
    assert np.isclose(score, expected_score)


def test_logloss_score_extreme_probabilities(mock_val_df):
    cfg = MagicMock()
    cfg.dataset.num_classes = 2
    cfg.dataset.answer_column = ["label"]
    results = {
        "probabilities": [[0.0001, 0.9999], [0.9999, 0.0001]],
        "target_text": ["1", "0"],
    }

    score = logloss_score(cfg, results, mock_val_df)

    expected_score = log_loss([1, 0], results["probabilities"])
    assert np.isclose(score, expected_score)
