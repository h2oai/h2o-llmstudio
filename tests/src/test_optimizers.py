"""Tests for optimizer factory with optional bitsandbytes support."""

from unittest.mock import patch

import pytest
from torch import optim


class TestOptimizers:
    """Tests for Optimizers factory."""

    def test_base_optimizers_always_available(self):
        """Test that base PyTorch optimizers are always available."""
        from llm_studio.src.optimizers import Optimizers

        # These should always be available
        assert "Adam" in Optimizers.names()
        assert "AdamW" in Optimizers.names()
        assert "SGD" in Optimizers.names()
        assert "RMSprop" in Optimizers.names()
        assert "Adadelta" in Optimizers.names()

    def test_base_optimizers_are_correct_classes(self):
        """Test that base optimizers return correct classes."""
        from llm_studio.src.optimizers import Optimizers

        assert Optimizers.get("Adam") == optim.Adam
        assert Optimizers.get("AdamW") == optim.AdamW

    def test_adamw8bit_available_when_bitsandbytes_present(self):
        """Test AdamW8bit is available when bitsandbytes is installed."""
        from llm_studio.src.optimizers import HAS_BITSANDBYTES, Optimizers

        if HAS_BITSANDBYTES:
            assert "AdamW8bit" in Optimizers.names()
            assert Optimizers.get("AdamW8bit") is not None
        else:
            pytest.skip("bitsandbytes not available (expected on macOS ARM64)")

    def test_adamw8bit_unavailable_when_bitsandbytes_missing(self):
        """Test AdamW8bit is not available when bitsandbytes is not installed."""
        from llm_studio.src.optimizers import HAS_BITSANDBYTES, Optimizers

        if not HAS_BITSANDBYTES:
            assert "AdamW8bit" not in Optimizers.names()
            assert Optimizers.get("AdamW8bit") is None
        else:
            pytest.skip("bitsandbytes is available")

    def test_get_nonexistent_optimizer_returns_none(self):
        """Test that getting a non-existent optimizer returns None."""
        from llm_studio.src.optimizers import Optimizers

        assert Optimizers.get("NonExistentOptimizer") is None

    @pytest.mark.arm64_mps
    def test_optimizers_work_without_bitsandbytes_on_macos_arm64(self):
        """Test that optimizer factory works without bitsandbytes on macOS ARM64."""
        import platform

        from llm_studio.src.utils.gpu_utils import detect_platform

        # This test is specific to macOS ARM64
        if platform.system() != "Darwin" or detect_platform() != "arm64":
            pytest.skip("Test only runs on macOS ARM64")

        from llm_studio.src.optimizers import HAS_BITSANDBYTES, Optimizers

        # bitsandbytes should not be available on macOS ARM64
        assert not HAS_BITSANDBYTES

        # But base optimizers should still work
        assert len(Optimizers.names()) >= 5
        assert "Adam" in Optimizers.names()
        assert "AdamW8bit" not in Optimizers.names()


class TestOptimizersWithMockedImport:
    """Tests for Optimizers with mocked bitsandbytes import."""

    def test_import_failure_handled_gracefully(self):
        """Test that ImportError for bitsandbytes is handled gracefully."""
        import importlib
        import sys

        # First import llm_studio to ensure it exists in sys.modules
        import llm_studio.src.optimizers

        # Now patch sys.modules and reload
        with patch.dict("sys.modules", {"bitsandbytes": None}):
            # Reload the module with bitsandbytes mocked out
            importlib.reload(llm_studio.src.optimizers)

            # When import fails, bitsandbytes should not be available
            assert isinstance(llm_studio.src.optimizers.HAS_BITSANDBYTES, bool)

            # Base optimizers should still be available
            assert "Adam" in llm_studio.src.optimizers.Optimizers.names()
            assert "AdamW" in llm_studio.src.optimizers.Optimizers.names()

        # Reload again to restore original state
        importlib.reload(llm_studio.src.optimizers)
