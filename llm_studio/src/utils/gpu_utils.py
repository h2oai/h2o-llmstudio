from typing import Any, Union

import numpy as np
import torch


def sync_across_processes(
    t: Union[torch.Tensor, np.ndarray], world_size: int, group: Any = None
) -> Union[torch.Tensor, np.ndarray]:
    """Concatenates tensors across processes.

    Args:
        t: input tensor or numpy array
        world_size: world size
        group: The process group to work on

    Returns:
        Tensor or numpy array concatenated across all processes
    """

    torch.distributed.barrier()

    if isinstance(t, torch.Tensor):
        gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]

        if t.is_cuda:
            torch.distributed.all_gather(gather_t_tensor, t)
        else:
            torch.distributed.all_gather_object(gather_t_tensor, t, group=group)

        ret = torch.cat(gather_t_tensor)
    elif isinstance(t, np.ndarray):
        gather_t_array = [np.ones_like(t) for _ in range(world_size)]
        torch.distributed.all_gather_object(gather_t_array, t, group=group)
        ret = np.concatenate(gather_t_array)  # type: ignore
    else:
        raise ValueError(f"Can't synchronize {type(t)}.")

    return ret


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


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


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_oom_error(exception: BaseException) -> bool:
    return (
        is_cuda_out_of_memory(exception)
        or is_cudnn_snafu(exception)
        or is_out_of_cpu_memory(exception)
    )
