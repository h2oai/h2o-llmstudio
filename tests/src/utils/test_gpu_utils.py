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


class TestCudaOutOfMemory:
    """Tests for CUDA out-of-memory detection across architectures."""

    def test_cuda_oom_standard_message(self):
        """Test detection of standard CUDA OOM error."""
        from llm_studio.src.utils.gpu_utils import is_cuda_out_of_memory

        exception = RuntimeError("CUDA out of memory")
        assert is_cuda_out_of_memory(exception)

    def test_cuda_oom_detailed_message(self):
        """Test detection of CUDA OOM with detailed message."""
        from llm_studio.src.utils.gpu_utils import is_cuda_out_of_memory

        exception = RuntimeError(
            "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.78 GiB total capacity)"
        )
        assert is_cuda_out_of_memory(exception)

    def test_cuda_oom_case_insensitive(self):
        """Test CUDA OOM detection is case insensitive."""
        from llm_studio.src.utils.gpu_utils import is_cuda_out_of_memory

        # Test various case combinations
        exception1 = RuntimeError("cuda out of memory")
        exception2 = RuntimeError("CUDA OUT OF MEMORY")
        exception3 = RuntimeError("Cuda Out Of Memory")

        assert is_cuda_out_of_memory(exception1)
        assert is_cuda_out_of_memory(exception2)
        assert is_cuda_out_of_memory(exception3)

    def test_cuda_oom_arm64_message(self):
        """Test CUDA OOM detection with ARM64-specific context.

        While CUDA error messages are standardized across architectures,
        this test validates the function works with ARM64-context messages.
        """
        from llm_studio.src.utils.gpu_utils import is_cuda_out_of_memory

        exception = RuntimeError(
            "CUDA out of memory on ARM64 device. Tried to allocate 1.50 GiB"
        )
        assert is_cuda_out_of_memory(exception)

    def test_non_runtime_error_not_detected(self):
        """Test that non-RuntimeError exceptions are not detected."""
        from llm_studio.src.utils.gpu_utils import is_cuda_out_of_memory

        exception = ValueError("CUDA out of memory")
        assert not is_cuda_out_of_memory(exception)

    def test_runtime_error_without_cuda_not_detected(self):
        """Test that RuntimeError without CUDA keyword is not detected."""
        from llm_studio.src.utils.gpu_utils import is_cuda_out_of_memory

        exception = RuntimeError("out of memory")
        assert not is_cuda_out_of_memory(exception)

    def test_runtime_error_without_oom_not_detected(self):
        """Test that RuntimeError with CUDA but without OOM is not detected."""
        from llm_studio.src.utils.gpu_utils import is_cuda_out_of_memory

        exception = RuntimeError("CUDA error: device-side assert triggered")
        assert not is_cuda_out_of_memory(exception)

    def test_runtime_error_empty_args_not_detected(self):
        """Test that RuntimeError with empty args is not detected."""
        from llm_studio.src.utils.gpu_utils import is_cuda_out_of_memory

        exception = RuntimeError()
        assert not is_cuda_out_of_memory(exception)

    def test_runtime_error_non_string_args_not_detected(self):
        """Test that RuntimeError with non-string args is not detected."""
        from llm_studio.src.utils.gpu_utils import is_cuda_out_of_memory

        exception = RuntimeError(12345)
        assert not is_cuda_out_of_memory(exception)


class TestMPSOutOfMemory:
    """Tests for MPS out-of-memory detection."""

    def test_mps_oom_error(self):
        """Test detection of MPS out of memory error."""
        from llm_studio.src.utils.gpu_utils import is_mps_out_of_memory

        exception = RuntimeError("MPS backend out of memory")
        assert is_mps_out_of_memory(exception)

    def test_mps_oom_failed_to_allocate(self):
        """Test detection of MPS failed to allocate error."""
        from llm_studio.src.utils.gpu_utils import is_mps_out_of_memory

        exception = RuntimeError("MPS failed to allocate memory")
        assert is_mps_out_of_memory(exception)

    def test_mps_oom_case_insensitive(self):
        """Test MPS OOM detection is case insensitive."""
        from llm_studio.src.utils.gpu_utils import is_mps_out_of_memory

        exception = RuntimeError("MPS backend Out Of Memory")
        assert is_mps_out_of_memory(exception)

    def test_non_mps_error_not_detected(self):
        """Test that non-MPS errors are not detected as MPS OOM."""
        from llm_studio.src.utils.gpu_utils import is_mps_out_of_memory

        exception = RuntimeError("Some other error")
        assert not is_mps_out_of_memory(exception)

    def test_cuda_error_not_detected_as_mps(self):
        """Test that CUDA errors are not detected as MPS OOM."""
        from llm_studio.src.utils.gpu_utils import is_mps_out_of_memory

        exception = RuntimeError("CUDA out of memory")
        assert not is_mps_out_of_memory(exception)

    def test_wrong_exception_type(self):
        """Test that non-RuntimeError exceptions are not detected."""
        from llm_studio.src.utils.gpu_utils import is_mps_out_of_memory

        exception = ValueError("MPS out of memory")
        assert not is_mps_out_of_memory(exception)


class TestUnifiedOOMError:
    """Tests for unified OOM error detection across all backends."""

    def test_cuda_oom_detected(self):
        """Test that CUDA OOM is detected by is_oom_error."""
        from llm_studio.src.utils.gpu_utils import is_oom_error

        exception = RuntimeError("CUDA out of memory")
        assert is_oom_error(exception)

    def test_mps_oom_detected(self):
        """Test that MPS OOM is detected by is_oom_error."""
        from llm_studio.src.utils.gpu_utils import is_oom_error

        exception = RuntimeError("MPS backend out of memory")
        assert is_oom_error(exception)

    def test_cpu_oom_detected(self):
        """Test that CPU OOM is detected by is_oom_error."""
        from llm_studio.src.utils.gpu_utils import is_oom_error

        exception = RuntimeError("DefaultCPUAllocator: can't allocate memory")
        assert is_oom_error(exception)

    def test_cudnn_error_detected(self):
        """Test that cuDNN errors are detected by is_oom_error."""
        from llm_studio.src.utils.gpu_utils import is_oom_error

        exception = RuntimeError("cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.")
        assert is_oom_error(exception)

    def test_non_oom_error_not_detected(self):
        """Test that non-OOM errors are not detected."""
        from llm_studio.src.utils.gpu_utils import is_oom_error

        exception = RuntimeError("Some other error")
        assert not is_oom_error(exception)


class TestSyncAcrossProcesses:
    """Tests for sync_across_processes with multiple backends."""

    @patch("torch.distributed.barrier")
    @patch("torch.distributed.all_gather")
    def test_sync_cuda_tensor(self, mock_all_gather, mock_barrier):
        """Test syncing CUDA tensors uses all_gather."""
        import torch

        from llm_studio.src.utils.gpu_utils import sync_across_processes

        # Create a mock CUDA tensor - use spec to pass isinstance check
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.is_cuda = True
        mock_tensor.is_mps = False

        # Mock torch.cat to return the tensor
        with patch("torch.cat", return_value=mock_tensor):
            with patch("torch.ones_like", return_value=mock_tensor):
                result = sync_across_processes(mock_tensor, world_size=2)

        # Verify all_gather was called (not all_gather_object)
        assert mock_all_gather.called
        assert result == mock_tensor

    @patch("torch.distributed.barrier")
    @patch("torch.distributed.all_gather")
    def test_sync_mps_tensor(self, mock_all_gather, mock_barrier):
        """Test syncing MPS tensors uses all_gather."""
        import torch

        from llm_studio.src.utils.gpu_utils import sync_across_processes

        # Create a mock MPS tensor - use spec to pass isinstance check
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.is_cuda = False
        mock_tensor.is_mps = True

        # Mock torch.cat to return the tensor
        with patch("torch.cat", return_value=mock_tensor):
            with patch("torch.ones_like", return_value=mock_tensor):
                result = sync_across_processes(mock_tensor, world_size=2)

        # Verify all_gather was called (not all_gather_object)
        assert mock_all_gather.called
        assert result == mock_tensor

    @patch("torch.distributed.barrier")
    @patch("torch.distributed.all_gather_object")
    def test_sync_cpu_tensor(self, mock_all_gather_object, mock_barrier):
        """Test syncing CPU tensors uses all_gather_object."""
        import torch

        from llm_studio.src.utils.gpu_utils import sync_across_processes

        # Create a mock CPU tensor - use spec to pass isinstance check
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.is_cuda = False
        mock_tensor.is_mps = False

        # Mock torch.cat to return the tensor
        with patch("torch.cat", return_value=mock_tensor):
            with patch("torch.ones_like", return_value=mock_tensor):
                result = sync_across_processes(mock_tensor, world_size=2)

        # Verify all_gather_object was called (not all_gather)
        assert mock_all_gather_object.called
        assert result == mock_tensor

    @patch("torch.distributed.barrier")
    @patch("torch.distributed.all_gather_object")
    def test_sync_numpy_array(self, mock_all_gather_object, mock_barrier):
        """Test syncing numpy arrays uses all_gather_object."""
        from llm_studio.src.utils.gpu_utils import sync_across_processes

        import numpy as np

        # Create a mock numpy array
        mock_array = np.array([1, 2, 3])

        # Mock np.concatenate to return the array
        with patch("numpy.concatenate", return_value=mock_array):
            with patch("numpy.ones_like", return_value=mock_array):
                result = sync_across_processes(mock_array, world_size=2)

        # Verify all_gather_object was called
        assert mock_all_gather_object.called
        assert isinstance(result, np.ndarray)
