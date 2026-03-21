"""Integration tests for MiniMax provider support.

These tests verify the end-to-end MiniMax integration, including
client creation, model configuration, and the evaluation pipeline.

Tests that require a live API key are skipped unless MINIMAX_API_KEY
is set in the environment.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from llm_studio.src.metrics.text_causal_language_modeling_metrics import (
    call_openai_api,
    get_openai_client,
)
from llm_studio.src.utils.utils import check_metric


@pytest.fixture
def minimax_env():
    """Environment with MiniMax API key and no OpenAI key."""
    env = {
        "OPENAI_API_TYPE": "open_ai",
        "OPENAI_API_KEY": "",
        "MINIMAX_API_KEY": "mm-integration-test-key",
    }
    with patch.dict(os.environ, env, clear=False):
        yield


@pytest.fixture
def minimax_explicit_env():
    """Environment with explicit MiniMax API type selection."""
    env = {
        "OPENAI_API_TYPE": "minimax",
        "MINIMAX_API_KEY": "mm-integration-test-key",
    }
    with patch.dict(os.environ, env, clear=False):
        yield


class TestMiniMaxIntegration:
    """Integration tests for MiniMax provider in the evaluation pipeline."""

    def test_minimax_auto_detect_creates_correct_client(self, minimax_env):
        """Auto-detection creates client pointing to MiniMax API."""
        client = get_openai_client()
        assert str(client.base_url).startswith("https://api.minimax.io/v1")

    def test_minimax_explicit_creates_correct_client(self, minimax_explicit_env):
        """Explicit OPENAI_API_TYPE=minimax creates client pointing to MiniMax API."""
        client = get_openai_client()
        assert str(client.base_url).startswith("https://api.minimax.io/v1")

    def test_check_metric_preserves_gpt_with_minimax(self, minimax_env):
        """check_metric keeps GPT metric when MiniMax key is available."""
        cfg = MagicMock()
        cfg.prediction.metric = "GPT"
        cfg = check_metric(cfg)
        assert cfg.prediction.metric == "GPT"

    @pytest.mark.skipif(
        not os.getenv("MINIMAX_API_KEY"),
        reason="MINIMAX_API_KEY not set",
    )
    def test_minimax_live_api_call(self):
        """Live test: call MiniMax API with a simple evaluation prompt."""
        env = {
            "OPENAI_API_TYPE": "minimax",
            "MINIMAX_API_KEY": os.environ["MINIMAX_API_KEY"],
        }
        with patch.dict(os.environ, env, clear=False):
            template = (
                "Rate the following answer on a scale of 1 to 10.\n"
                "Question: What is 2+2?\n"
                "Answer: 4\n"
                "Please output SCORE: followed by the numeric score."
            )
            score, explanation = call_openai_api(template, "MiniMax-M2.5-highspeed")
            assert isinstance(score, float)
            assert 0 <= score <= 10
            assert len(explanation) > 0
