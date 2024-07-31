from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from llm_studio.src.metrics.text_causal_language_modeling_metrics import sacrebleu_score


@pytest.fixture
def mock_val_df():
    return pd.DataFrame()


def test_sacrebleu_score_perfect_match(mock_val_df):
    cfg = MagicMock()
    results = {
        "predicted_text": ["Hello world", "Python is great"],
        "target_text": ["Hello world", "Python is great"],
    }

    scores = sacrebleu_score(cfg, results, mock_val_df)

    assert np.allclose(scores, np.array([100.0, 100.0]))


def test_sacrebleu_score_partial_match(mock_val_df):
    cfg = MagicMock()
    results = {
        "predicted_text": ["Hello universe", "Python is awesome"],
        "target_text": ["Hello world", "Python is great"],
    }

    scores = sacrebleu_score(cfg, results, mock_val_df)

    assert np.allclose(scores, np.array([50.0, 55.03212081]))


def test_sacrebleu_score_no_match(mock_val_df):
    cfg = MagicMock()
    results = {
        "predicted_text": ["Goodbye universe", "What a day"],
        "target_text": ["Hello world", "Python is great"],
    }

    scores = sacrebleu_score(cfg, results, mock_val_df)

    assert np.allclose(scores, np.array([0.0, 0.0]))


def test_sacrebleu_score_all_empty_target(mock_val_df):
    cfg = MagicMock()
    results = {
        "predicted_text": ["Hello world", "Python is great"],
        "target_text": ["", ""],
    }

    scores = sacrebleu_score(cfg, results, mock_val_df)

    assert np.allclose(scores, np.array([0.0, 0.0]))


def test_sacrebleu_score_one_empty_target(mock_val_df):
    cfg = MagicMock()
    results = {
        "predicted_text": ["Hello world", "Python is great"],
        "target_text": ["", "Python is great"],
    }

    scores = sacrebleu_score(cfg, results, mock_val_df)

    assert np.allclose(scores, np.array([0.0, 100.0]))


def test_sacrebleu_score_invalid_input_empty(mock_val_df):
    cfg = MagicMock()
    results = {"predicted_text": [], "target_text": []}

    with pytest.raises(ValueError):
        sacrebleu_score(cfg, results, mock_val_df)


def test_sacrebleu_score_invalid_input_different_lengths(mock_val_df):
    cfg = MagicMock()
    results = {
        "predicted_text": ["Hello world", "Python", "is", "great"],
        "target_text": ["Hello universe", "Python is awesome"],
    }

    with pytest.raises(ValueError):
        sacrebleu_score(cfg, results, mock_val_df)
