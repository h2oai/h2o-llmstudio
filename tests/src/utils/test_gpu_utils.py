"""Tests for GPU utility functions including platform and backend detection."""

from unittest.mock import MagicMock, patch

import pytest

from llm_studio.src.utils.gpu_utils import detect_gpu_backend, detect_platform


class TestDetectPlatform:
    """Tests for detect_platform function."""

    @patch("platform.machine")
    def test_detect_arm64(self, mock_machine):
        """Test ARM64 detection with 'arm64' machine type."""
        mock_machine.return_value = "arm64"
        assert detect_platform() == "arm64"

    @patch("platform.machine")
    def test_detect_aarch64(self, mock_machine):
        """Test ARM64 detection with 'aarch64' machine type."""
        mock_machine.return_value = "aarch64"
        assert detect_platform() == "arm64"

    @patch("platform.machine")
    def test_detect_arm64_uppercase(self, mock_machine):
        """Test ARM64 detection with uppercase 'ARM64'."""
        mock_machine.return_value = "ARM64"
        assert detect_platform() == "arm64"

    @patch("platform.machine")
    def test_detect_x86_64(self, mock_machine):
        """Test x86_64 detection."""
        mock_machine.return_value = "x86_64"
        assert detect_platform() == "x86_64"

    @patch("platform.machine")
    def test_detect_amd64(self, mock_machine):
        """Test x86_64 detection with 'AMD64' machine type."""
        mock_machine.return_value = "AMD64"
        assert detect_platform() == "x86_64"

    @patch("platform.machine")
    def test_detect_unknown_defaults_to_x86_64(self, mock_machine):
        """Test that unknown architectures default to x86_64."""
        mock_machine.return_value = "unknown_arch"
        assert detect_platform() == "x86_64"


class TestDetectGpuBackend:
    """Tests for detect_gpu_backend function."""

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_detect_cuda(self, mock_mps_available, mock_cuda_available):
        """Test CUDA backend detection when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = False
        assert detect_gpu_backend() == "cuda"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_detect_mps(self, mock_mps_available, mock_cuda_available):
        """Test MPS backend detection when only MPS is available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True
        assert detect_gpu_backend() == "mps"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_detect_cpu(self, mock_mps_available, mock_cuda_available):
        """Test CPU backend detection when no GPU is available."""
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        assert detect_gpu_backend() == "cpu"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_cuda_preferred_over_mps(self, mock_mps_available, mock_cuda_available):
        """Test that CUDA is preferred when both CUDA and MPS are available."""
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = True
        assert detect_gpu_backend() == "cuda"

    @patch("torch.cuda.is_available")
    @patch("torch.backends", new_callable=MagicMock)
    def test_mps_not_available_attribute(self, mock_backends, mock_cuda_available):
        """Test MPS detection when torch.backends.mps doesn't exist (older PyTorch)."""
        mock_cuda_available.return_value = False
        # Remove mps attribute to simulate older PyTorch versions
        delattr(mock_backends, "mps")
        assert detect_gpu_backend() == "cpu"


class TestCombinedPlatformAndBackend:
    """Integration tests for platform and backend detection combinations."""

    @patch("platform.machine")
    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_arm64_with_cuda(
        self, mock_mps_available, mock_cuda_available, mock_machine
    ):
        """Test ARM64 platform with CUDA backend (NVIDIA ARM64)."""
        mock_machine.return_value = "aarch64"
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = False

        platform = detect_platform()
        backend = detect_gpu_backend()

        assert platform == "arm64"
        assert backend == "cuda"

    @patch("platform.machine")
    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_arm64_with_mps(
        self, mock_mps_available, mock_cuda_available, mock_machine
    ):
        """Test ARM64 platform with MPS backend (Apple Silicon)."""
        mock_machine.return_value = "arm64"
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = True

        platform = detect_platform()
        backend = detect_gpu_backend()

        assert platform == "arm64"
        assert backend == "mps"

    @patch("platform.machine")
    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_x86_64_with_cuda(
        self, mock_mps_available, mock_cuda_available, mock_machine
    ):
        """Test x86_64 platform with CUDA backend (standard NVIDIA setup)."""
        mock_machine.return_value = "x86_64"
        mock_cuda_available.return_value = True
        mock_mps_available.return_value = False

        platform = detect_platform()
        backend = detect_gpu_backend()

        assert platform == "x86_64"
        assert backend == "cuda"

    @patch("platform.machine")
    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_arm64_cpu_only(
        self, mock_mps_available, mock_cuda_available, mock_machine
    ):
        """Test ARM64 platform with CPU-only (no GPU)."""
        mock_machine.return_value = "arm64"
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False

        platform = detect_platform()
        backend = detect_gpu_backend()

        assert platform == "arm64"
        assert backend == "cpu"
