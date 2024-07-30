import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from llm_studio.src.utils.modeling_utils import (
    unwrap_model,
    check_disk_space,
    save_checkpoint,
    load_checkpoint,
)


def test_unwrap_model():
    # Create a dummy model
    model = torch.nn.Linear(10, 10)

    # Wrap it in DataParallel
    wrapped_model = torch.nn.DataParallel(model)
    assert wrapped_model != model
    assert isinstance(wrapped_model, torch.nn.DataParallel)

    # Test unwrapping
    unwrapped = unwrap_model(wrapped_model)
    assert unwrapped == model
    assert not isinstance(unwrapped, torch.nn.DataParallel)


@pytest.mark.parametrize(
    "dtype",
    [
        (torch.float32),
        (torch.float16),
        (torch.bfloat16),
        (torch.int8),
        (torch.uint8),
        (torch.int16),
    ],
)
@pytest.mark.parametrize(
    "free_space,should_raise",
    [
        (1e12, False),  # Plenty of space
        (1, True),  # Not enough space
    ],
)
def test_check_disk_space(free_space, dtype, should_raise):
    # Mock model and shutil
    model = MagicMock()
    model.parameters.return_value = [torch.ones(1000, 1000, dtype=dtype)]

    with patch("shutil.disk_usage", return_value=(0, 0, free_space)):
        if should_raise:
            with pytest.raises(ValueError):
                check_disk_space(model, "/dummy/path")
        else:
            check_disk_space(model, "/dummy/path")  # Should not raise


class DummyModel(torch.nn.Module):
    def __init__(self, use_classification_head=True):
        super(DummyModel, self).__init__()
        self.backbone = torch.nn.Linear(10, 10)
        if use_classification_head:
            self.classification_head = torch.nn.Linear(10, 10)


@pytest.mark.parametrize("use_classification_head", [True, False])
def test_save_checkpoint(use_classification_head):
    model = DummyModel(use_classification_head)
    cfg = MagicMock()
    cfg.environment.use_deepspeed = False
    cfg.environment._local_rank = 0
    cfg.training.lora = False

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, tmpdir, cfg)
        assert os.path.exists(os.path.join(tmpdir, "checkpoint.pth"))
        if use_classification_head:
            assert os.path.exists(os.path.join(tmpdir, "classification_head.pth"))
        else:
            assert not os.path.exists(os.path.join(tmpdir, "classification_head.pth"))


def test_load_checkpoint():
    model = DummyModel()
    cfg = MagicMock()
    cfg.architecture.pretrained_weights = "dummy_weights.pth"
    cfg.environment.use_deepspeed = False

    # Mock torch.load
    dummy_state_dict = {"model": model.state_dict()}
    with patch("torch.load", return_value=dummy_state_dict):
        load_checkpoint(cfg, model, strict=True)


def test_load_checkpoint_mismatch():
    model = DummyModel(use_classification_head=True)
    model_no_classification_head = DummyModel(use_classification_head=False)
    cfg = MagicMock()
    cfg.architecture.pretrained_weights = "dummy_weights.pth"
    cfg.environment.use_deepspeed = False

    # Mock torch.load
    dummy_state_dict = {"model": model_no_classification_head.state_dict()}
    with patch("torch.load", return_value=dummy_state_dict):
        with pytest.raises(RuntimeError):
            load_checkpoint(cfg, model, strict=True)
        load_checkpoint(cfg, model, strict=False)
