import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from llm_studio.src.utils.modeling_utils import (
    _get_text_config,
    _load_model_weights,
    check_disk_space,
    create_nlp_backbone,
    load_checkpoint,
    save_checkpoint,
    unwrap_model,
    update_backbone_config,
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
    cfg.environment._local_rank = 0

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
    cfg.environment._local_rank = 0

    # Mock torch.load
    dummy_state_dict = {"model": model_no_classification_head.state_dict()}
    with patch("torch.load", return_value=dummy_state_dict):
        with pytest.raises(RuntimeError):
            load_checkpoint(cfg, model, strict=True)
        load_checkpoint(cfg, model, strict=False)


def test_load_transformers_4_gpt_neox_checkpoint_key():
    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.lm_head = torch.nn.Linear(2, 2, bias=False)
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    legacy_weights = {"module.backbone.embed_out.weight": expected}
    cfg = SimpleNamespace(
        architecture=SimpleNamespace(backbone_dtype="float32"),
        environment=SimpleNamespace(_device=torch.device("cpu")),
    )

    _load_model_weights(model, legacy_weights, strict=True, cfg=cfg)

    assert torch.equal(model.backbone.lm_head.weight, expected)


def test_update_backbone_config_handles_missing_special_token_attributes():
    config = SimpleNamespace(eos_token_id=7, pad_token_id=8)
    tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0, bos_token_id=2)
    cfg = SimpleNamespace(
        llm_backbone="t5-small",
        architecture=SimpleNamespace(intermediate_dropout=0),
        environment=SimpleNamespace(_device=torch.device("cpu")),
        training=SimpleNamespace(lora=False),
    )

    with patch(
        "llm_studio.src.utils.modeling_utils.get_tokenizer",
        return_value=tokenizer,
    ):
        update_backbone_config(config, cfg)

    assert config.eos_token_id == tokenizer.eos_token_id
    assert config.pad_token_id == tokenizer.pad_token_id
    assert not hasattr(config, "bos_token_id")


def test_update_backbone_config_syncs_composite_text_config():
    text_config = SimpleNamespace(
        vocab_size=151936,
        eos_token_id=7,
        pad_token_id=8,
        bos_token_id=9,
        hidden_dropout_prob=0.0,
        pretraining_tp=2,
    )
    composite_config = SimpleNamespace(
        text_config=text_config,
        get_text_config=lambda: text_config,
    )
    tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0, bos_token_id=2)
    cfg = SimpleNamespace(
        llm_backbone="Qwen/Qwen3.8-27B",
        architecture=SimpleNamespace(intermediate_dropout=0.1),
        environment=SimpleNamespace(_device=torch.device("cpu")),
        training=SimpleNamespace(lora=True),
    )

    with patch(
        "llm_studio.src.utils.modeling_utils.get_tokenizer",
        return_value=tokenizer,
    ):
        returned = update_backbone_config(composite_config, cfg)

    assert returned is composite_config
    assert _get_text_config(composite_config) is text_config
    assert text_config.eos_token_id == tokenizer.eos_token_id
    assert text_config.pad_token_id == tokenizer.pad_token_id
    assert text_config.bos_token_id == tokenizer.bos_token_id
    assert text_config.hidden_dropout_prob == 0.1
    assert text_config.pretraining_tp == 1


def test_create_nlp_backbone_uses_text_config_for_composite_config():
    text_config = SimpleNamespace(
        vocab_size=100,
        eos_token_id=10,
        pad_token_id=11,
        bos_token_id=12,
    )
    composite_config = SimpleNamespace(
        text_config=text_config,
        get_text_config=lambda: text_config,
    )
    tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0, bos_token_id=2)
    fake_backbone = MagicMock()
    fake_backbone.config = composite_config
    fake_backbone.generation_config = SimpleNamespace(
        eos_token_id=10,
        pad_token_id=11,
        bos_token_id=12,
    )
    fake_backbone.named_parameters.return_value = []
    fake_model_class = MagicMock()
    fake_model_class.from_config.return_value = fake_backbone

    cfg = SimpleNamespace(
        llm_backbone="Qwen/Qwen3.8-27B",
        problem_type="text_causal_classification_modeling",
        tokenizer=SimpleNamespace(_vocab_length=128),
        architecture=SimpleNamespace(
            pretrained=False,
            backbone_dtype="float32",
            intermediate_dropout=0,
            gradient_checkpointing=False,
        ),
        environment=SimpleNamespace(
            trust_remote_code=False,
            huggingface_branch="main",
            gpus=[],
            _device=torch.device("cpu"),
            mixed_precision=False,
        ),
        training=SimpleNamespace(
            attention_implementation="auto",
            lora=False,
            freeze_layers=[],
        ),
        prediction=SimpleNamespace(),
    )

    with (
        patch(
            "llm_studio.src.utils.modeling_utils.AutoConfig.from_pretrained",
            return_value=composite_config,
        ),
        patch(
            "llm_studio.src.utils.modeling_utils.get_tokenizer",
            return_value=tokenizer,
        ),
    ):
        backbone, backbone_config = create_nlp_backbone(
            cfg, model_class=fake_model_class
        )

    assert backbone is fake_backbone
    assert backbone_config is text_config
    assert backbone_config.vocab_size == 100
    fake_backbone.resize_token_embeddings.assert_called_once_with(128)
    assert fake_backbone.generation_config.eos_token_id == 1
    assert fake_backbone.generation_config.pad_token_id == 0
    assert fake_backbone.generation_config.bos_token_id == 2
