from types import SimpleNamespace

import pytest

from llm_studio.app_utils.utils import start_process


@pytest.mark.parametrize(
    "use_deepspeed, expected",
    [(False, "CUDA_VISIBLE_DEVICES=3,1"), (True, "localhost:3,1")],
)
def test_start_process_maps_gpu_ids_to_visible_devices(
    mocker, monkeypatch, use_deepspeed, expected
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,1")
    popen = mocker.patch("llm_studio.app_utils.utils.subprocess.Popen")
    cfg = SimpleNamespace(
        output_directory="output",
        environment=SimpleNamespace(use_deepspeed=use_deepspeed),
    )

    start_process(cfg, gpu_list=["1", "2"], process_queue=[], env_vars={})

    assert expected in popen.call_args.args[0]
