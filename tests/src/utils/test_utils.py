import pytest

from llm_studio.python_configs.text_dpo_modeling_config import (
    ConfigNLPDPOLMDataset,
    ConfigProblemBase,
)
from llm_studio.src.utils.utils import PatchedAttribute


def test_patched_attribute():
    cfg = ConfigProblemBase(
        dataset=ConfigNLPDPOLMDataset(
            prompt_column=("prompt_column",),
            answer_column="answer_column",
            rejected_answer_column="rejected_answer_column",
            parent_id_column="None",
        )
    )
    with PatchedAttribute(cfg.dataset, "answer_column", "chosen_response"):
        assert cfg.dataset.answer_column == "chosen_response"

    with PatchedAttribute(cfg.dataset, "answer_column", "new_answer_column"):
        assert cfg.dataset.answer_column == "new_answer_column"

    assert cfg.dataset.answer_column == "answer_column"

    with PatchedAttribute(cfg.dataset, "new_property", "new_value"):
        assert cfg.dataset.new_property == "new_value"  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        cfg.dataset.new_property  # type: ignore[attr-defined]
