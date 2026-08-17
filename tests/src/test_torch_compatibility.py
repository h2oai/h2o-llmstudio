import shutil
import subprocess
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from deepspeed.ops.op_builder import FusedAdamBuilder
from deepspeed.ops.transformer.inference.triton.matmul_ext import is_nfs_path

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMTraining,
)
from llm_studio.src.utils.gpu_utils import sync_across_processes
from llm_studio.train import _initialize_distributed_environment


def test_deepspeed_native_ops_use_the_torch_cpp_standard():
    assert shutil.which("ninja") is not None
    cxx_args = FusedAdamBuilder().cxx_args()
    assert "-std=c++20" in cxx_args
    assert "-std=c++17" not in cxx_args


def test_deepspeed_nfs_check_uses_portable_df_output(monkeypatch, tmp_path):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "Filesystem Type 1024-blocks Used Available Capacity Mounted on\n/dev/test nfs 1 1 0 100% /tmp\n"

    monkeypatch.setattr(
        "deepspeed.ops.transformer.inference.triton.matmul_ext.subprocess.check_output",
        check_output,
    )

    assert is_nfs_path(tmp_path)
    assert calls == [
        (
            ["df", "-PT", str(tmp_path)],
            {"encoding": "utf-8", "stderr": subprocess.DEVNULL},
        )
    ]


def test_distributed_initialization_uses_local_rank_device():
    cfg = SimpleNamespace(
        environment=SimpleNamespace(_local_rank=1, use_deepspeed=False)
    )
    expected_device = torch.device("cuda:1")

    with (
        patch("torch.cuda.set_device") as set_device,
        patch("torch.distributed.init_process_group") as init_process_group,
        patch("torch.distributed.new_group", return_value="cpu-group"),
        patch("torch.distributed.get_world_size", return_value=4),
        patch("torch.distributed.get_rank", return_value=3),
    ):
        _initialize_distributed_environment(cfg)

    set_device.assert_called_once_with(expected_device)
    init_process_group.assert_called_once_with(
        backend="nccl", init_method="env://", device_id=expected_device
    )
    assert cfg.environment._device == "cuda:1"
    assert cfg.environment._cpu_comm == "cpu-group"
    assert cfg.environment._world_size == 4
    assert cfg.environment._rank == 3


def test_torch_compile_forward_and_backward():
    torch.manual_seed(0)
    eager_model = torch.nn.Linear(4, 3)
    compiled_source = torch.nn.Linear(4, 3)
    compiled_source.load_state_dict(eager_model.state_dict())
    compiled_model = torch.compile(compiled_source)
    eager_input = torch.randn(2, 4, requires_grad=True)
    compiled_input = eager_input.detach().clone().requires_grad_()

    eager_loss = eager_model(eager_input).square().mean()
    compiled_loss = compiled_model(compiled_input).square().mean()
    eager_loss.backward()
    compiled_loss.backward()

    torch.testing.assert_close(compiled_loss, eager_loss)
    torch.testing.assert_close(compiled_input.grad, eager_input.grad)
    torch.testing.assert_close(compiled_model.weight.grad, eager_model.weight.grad)
    torch.compiler.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pytorch_flash_sdpa_forward_and_backward():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    query, key, value = (
        torch.randn(
            2,
            4,
            16,
            32,
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        for _ in range(3)
    )

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
    output.float().square().mean().backward()

    assert output.shape == query.shape
    assert all(tensor.grad is not None for tensor in (query, key, value))
    assert all(torch.isfinite(tensor.grad).all() for tensor in (query, key, value))


def _run_distributed_sync(rank: int, world_size: int, init_method: str):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    cpu_group = dist.new_group(backend="gloo")
    try:
        actual = sync_across_processes(
            np.array([rank], dtype=np.int64), world_size, group=cpu_group
        )
        np.testing.assert_array_equal(actual, np.arange(world_size))
    finally:
        dist.destroy_process_group()


def test_distributed_gloo_group_sync(tmp_path):
    world_size = 2
    mp.start_processes(
        _run_distributed_sync,
        args=(world_size, f"file://{tmp_path / 'torch-distributed-store'}"),
        nprocs=world_size,
        join=True,
        start_method="fork",
    )


def test_config_only_offers_available_attention_backends():
    values = (
        ConfigNLPCausalLMTraining()._possible_values["attention_implementation"].values
    )
    assert {value for value, _ in values} == {"auto", "eager", "sdpa"}
