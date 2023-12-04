import pytest

from llm_studio.python_configs.text_dpo_modeling_config import ConfigProblemBase, ConfigNLPDPOLMDataset
from llm_studio.src.utils.utils import PatchedAttribute


def test_patched_attribute():
    cfg = ConfigProblemBase(
        dataset=ConfigNLPDPOLMDataset(
            prompt_column=("prompt_column",),
            chosen_response_column="chosen_response_column",
            rejected_response_column="rejected_response_column",
            parent_id_column="None",
        )
    )
    with PatchedAttribute(cfg.dataset, "answer_column", "chosen_response"):
        assert cfg.dataset.answer_column == "chosen_response"

    with PatchedAttribute(
        cfg.dataset, "chosen_response_column", "new_chosen_response_column"
    ):
        assert cfg.dataset.chosen_response_column == "new_chosen_response_column"

    assert cfg.dataset.chosen_response_column == "chosen_response_column"

    with PatchedAttribute(cfg.dataset, "new_property", "new_value"):
        assert cfg.dataset.new_property == "new_value"

    with pytest.raises(AttributeError):
        cfg.dataset.new_property
