import torch

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigProblemBase,
)
from llm_studio.src.models.text_causal_language_modeling_model import Model
from llm_studio.src.utils.modeling_utils import TokenStoppingCriteria, activate_neftune


def test_token_stopping_criteria():
    token_stopping_criteria = TokenStoppingCriteria(
        stop_word_ids=torch.tensor([0, 1, 2, 8]), prompt_input_ids_len=4
    )

    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ]
    ).long()

    # prompt input len is 4, so generated ids of last sample of the batch are
    # [9, 10, 11, 12, 13, 14], do not trigger stopping criteria
    assert not token_stopping_criteria(input_ids=input_ids, scores=None)

    token_stopping_criteria = TokenStoppingCriteria(
        stop_word_ids=torch.tensor([6]), prompt_input_ids_len=0
    )

    # first item reads [ 0,  1,  2,  3,  4,  5], so do not trigger stopping criteria
    assert not token_stopping_criteria(input_ids=input_ids[:, :6], scores=None)
    assert token_stopping_criteria(input_ids=input_ids[:, :7], scores=None)

    # Test stopping criteria with compound tokens
    token_stopping_criteria = TokenStoppingCriteria(
        stop_word_ids=torch.tensor([[6, 7]]), prompt_input_ids_len=0
    )

    assert not token_stopping_criteria(input_ids=input_ids[:, :6], scores=None)
    assert not token_stopping_criteria(input_ids=input_ids[:, :7], scores=None)
    assert token_stopping_criteria(input_ids=input_ids[:, :8], scores=None)

    # Test stopping criteria with stop word ids being longer than generated text
    token_stopping_criteria = TokenStoppingCriteria(
        stop_word_ids=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]),
        prompt_input_ids_len=0,
    )

    assert not token_stopping_criteria(input_ids=input_ids, scores=None)


def test_neftune_is_disabled_in_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ConfigProblemBase(llm_backbone="MaxJeblick/llama2-0b-unit-test")
    cfg.architecture.backbone_dtype = "float32"
    cfg.architecture.mixed_precision = False
    model = Model(cfg).eval().to(device)

    input_batch = {
        "input_ids": torch.randint(
            0,
            1000,
            (1, 10),
        ).to(device),
        "attention_mask": torch.ones(1, 10).to(device),
    }

    with torch.no_grad():
        outputs = model.backbone(**input_batch)

    activate_neftune(model.backbone, neftune_noise_alpha=10)
    assert model.backbone.get_input_embeddings().neftune_noise_alpha == 10

    with torch.no_grad():
        outputs_after_neftune = model.backbone(**input_batch)

    assert torch.allclose(outputs["logits"], outputs_after_neftune["logits"])

    # state dict does not contain neftune noise
    assert [key for key in model.state_dict() if "neftune" in key] == []
