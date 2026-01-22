from typing import Any, Literal

import platform

import numpy as np
import torch
import torch.distributed as dist


def detect_platform() -> Literal["arm64", "x86_64"]:
    """Detect the CPU architecture.

    Returns:
        'arm64' for ARM64 architecture (aarch64), 'x86_64' for x86_64 architecture
    """
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    elif machine in ("x86_64", "amd64"):
        return "x86_64"
    else:
        # Default to x86_64 for unknown architectures
        return "x86_64"


def detect_gpu_backend() -> Literal["cuda", "mps", "cpu"]:
    """Detect the available GPU backend.

    Returns:
        'cuda' if NVIDIA CUDA is available,
        'mps' if Apple Metal Performance Shaders is available,
        'cpu' if no GPU is available
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def sync_across_processes(
    t: torch.Tensor | np.ndarray, world_size: int, group: Any = None
) -> torch.Tensor | np.ndarray:
    """Concatenates tensors across processes.

    Supports CUDA, MPS, and CPU backends uniformly.

    Args:
        t: input tensor or numpy array
        world_size: world size
        group (ProcessGroup, optional): The process group to work on

    Returns:
        Tensor or numpy array concatenated across all processes
    """

    dist.barrier()
    ret: torch.Tensor | np.ndarray

    if isinstance(t, torch.Tensor):
        gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]

        # Use all_gather for GPU tensors (CUDA, MPS, etc.), all_gather_object for CPU
        # Check if tensor is on a GPU device (not just CUDA, but any GPU backend)
        is_gpu_tensor = t.is_cuda or (hasattr(t, "is_mps") and t.is_mps)

        if is_gpu_tensor:
            dist.all_gather(gather_t_tensor, t)
        else:
            dist.all_gather_object(gather_t_tensor, t, group=group)

        ret = torch.cat(gather_t_tensor)
    elif isinstance(t, np.ndarray):
        gather_t_array = [np.ones_like(t) for _ in range(world_size)]
        dist.all_gather_object(gather_t_array, t, group=group)
        ret = np.concatenate(gather_t_array)
    else:
        raise ValueError(f"Can't synchronize {type(t)}.")

    return ret


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    """Check if exception is a CUDA out of memory error.

    Works uniformly across x86_64 and ARM64 (aarch64) CUDA platforms.
    CUDA error messages are standardized across architectures.
    """
    if not isinstance(exception, RuntimeError):
        return False

    if len(exception.args) < 1 or not isinstance(exception.args[0], str):
        return False

    error_message = exception.args[0].lower()

    # Check for CUDA OOM patterns that work across all architectures
    has_cuda = "cuda" in error_message
    has_oom = "out of memory" in error_message

    return has_cuda and has_oom


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def is_mps_out_of_memory(exception: BaseException) -> bool:
    """Check if exception is an MPS (Metal Performance Shaders) out of memory error.

    MPS OOM errors on Apple Silicon typically manifest as RuntimeError with
    messages containing 'MPS' and memory-related keywords.
    """
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) >= 1
        and isinstance(exception.args[0], str)
        and "MPS" in exception.args[0]
        and ("out of memory" in exception.args[0].lower()
             or "failed to allocate" in exception.args[0].lower())
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_oom_error(exception: BaseException) -> bool:
    """Check if exception is an out-of-memory error across all backends.

    Supports CUDA, MPS, and CPU memory errors uniformly.
    """
    return (
        is_cuda_out_of_memory(exception)
        or is_cudnn_snafu(exception)
        or is_mps_out_of_memory(exception)
        or is_out_of_cpu_memory(exception)
    )
