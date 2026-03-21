import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from llm_studio.src.metrics.text_causal_language_modeling_metrics import (
    get_openai_client,
    sacrebleu_score,
)


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


# --- MiniMax provider tests ---


class TestGetOpenaiClient:
    """Tests for the get_openai_client factory function."""

    def test_default_openai_client(self):
        """Default provider creates a standard OpenAI client."""
        env = {
            "OPENAI_API_TYPE": "open_ai",
            "OPENAI_API_KEY": "sk-test-key",
        }
        with patch.dict(os.environ, env, clear=False):
            # Remove MINIMAX_API_KEY if present
            os.environ.pop("MINIMAX_API_KEY", None)
            client = get_openai_client()
            assert client.base_url.host == "api.openai.com"

    def test_minimax_client_via_api_type(self):
        """OPENAI_API_TYPE=minimax creates a MiniMax-backed client."""
        env = {
            "OPENAI_API_TYPE": "minimax",
            "MINIMAX_API_KEY": "mm-test-key",
        }
        with patch.dict(os.environ, env, clear=False):
            client = get_openai_client()
            assert "minimax" in client.base_url.host

    def test_minimax_auto_detect(self):
        """Auto-detect MiniMax when MINIMAX_API_KEY is set but OPENAI_API_KEY is not."""
        env = {
            "OPENAI_API_TYPE": "open_ai",
            "MINIMAX_API_KEY": "mm-test-key",
            "OPENAI_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            client = get_openai_client()
            assert "minimax" in client.base_url.host

    def test_openai_preferred_over_minimax_auto_detect(self):
        """When both OPENAI_API_KEY and MINIMAX_API_KEY are set, OpenAI is used."""
        env = {
            "OPENAI_API_TYPE": "open_ai",
            "OPENAI_API_KEY": "sk-test-key",
            "MINIMAX_API_KEY": "mm-test-key",
        }
        with patch.dict(os.environ, env, clear=False):
            client = get_openai_client()
            assert client.base_url.host == "api.openai.com"

    def test_minimax_explicit_overrides_openai_key(self):
        """OPENAI_API_TYPE=minimax takes precedence even when OPENAI_API_KEY is set."""
        env = {
            "OPENAI_API_TYPE": "minimax",
            "OPENAI_API_KEY": "sk-test-key",
            "MINIMAX_API_KEY": "mm-test-key",
        }
        with patch.dict(os.environ, env, clear=False):
            client = get_openai_client()
            assert "minimax" in client.base_url.host

    def test_minimax_custom_base_url(self):
        """Custom OPENAI_API_BASE is respected for MiniMax provider."""
        env = {
            "OPENAI_API_TYPE": "minimax",
            "MINIMAX_API_KEY": "mm-test-key",
            "OPENAI_API_BASE": "https://custom-proxy.example.com/v1",
        }
        with patch.dict(os.environ, env, clear=False):
            client = get_openai_client()
            assert "custom-proxy" in client.base_url.host

    def test_azure_client(self):
        """Azure provider creates an AzureOpenAI client."""
        from openai import AzureOpenAI

        env = {
            "OPENAI_API_TYPE": "azure",
            "OPENAI_API_KEY": "azure-key",
            "OPENAI_API_BASE": "https://my-endpoint.openai.azure.com",
            "OPENAI_API_VERSION": "2023-05-15",
        }
        with patch.dict(os.environ, env, clear=False):
            client = get_openai_client()
            assert isinstance(client, AzureOpenAI)
