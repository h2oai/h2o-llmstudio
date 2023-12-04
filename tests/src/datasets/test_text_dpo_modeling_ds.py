import numpy as np
import pandas as pd
import pytest
import torch
from tqdm import tqdm

from llm_studio.python_configs.text_dpo_modeling_config import (
    ConfigNLPDPOLMDataset,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_dpo_modeling_ds import CustomDataset, PatchedAttribute


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "prompt_column": [f"prompt {i}" for i in range(200)],
            "chosen_response_column": [f"chosen_response {i}" for i in range(200)],
            "rejected_response_column": [f"rejected_response {i}" for i in range(200)],
        }
    )


@pytest.fixture
def df_with_conversation_chain_ids():
    ids = [str(i + 1) for i in range(200)]

    parent_ids = np.array(ids, dtype=object).reshape(-1, 5)
    parent_ids[:, -1] = "None"
    parent_ids = np.roll(parent_ids, 1, 1).reshape(-1)

    # ids:          [0, 1, 2, 3, 4   ]
    # parent_ids:   [None, 0, 1, 2, 3]
    # conversation: 0 -> 1 -> 2 -> 3 -> 4
    chosen_responses = [
        f"chosen_response {idx}" if int(idx) % 5 == 0 else f"response {idx}"
        for idx in ids
    ]
    rejected_responses = [
        f"rejected_response {idx}" if int(idx) % 5 == 0 else f"response {idx}"
        for idx in ids
    ]
    return pd.DataFrame(
        {
            "prompt_column": [f"prompt {i}" for i in range(200)],
            "chosen_response_column": chosen_responses,
            "rejected_response_column": rejected_responses,
            "parent_id_column": parent_ids,
            "id": ids,
        }
    )


def test_dataset_conversation_chain_is_correct(df_with_conversation_chain_ids):
    cfg = ConfigProblemBase(
        dataset=ConfigNLPDPOLMDataset(
            prompt_column=("prompt_column",),
            chosen_response_column="chosen_response_column",
            rejected_response_column="rejected_response_column",
            parent_id_column="parent_id_column",
        )
    )
    dataset = CustomDataset(df_with_conversation_chain_ids, cfg, mode="train")

    ## Check for right formatting, e.g.:
    # dataset.conversation_chain_handler_chosen[0] ==
    # {
    #     "prompts": ["prompt 0", "prompt 1", "prompt 2", "prompt 3", "prompt 4"],
    #     "answers": [
    #         "response 1",
    #         "response 2",
    #         "response 3",
    #         "response 4",
    #         "chosen_response 5",
    #     ],
    #     "systems": ["", "", "", "", ""],
    # }

    for idx in range(200 // 5):
        for name, conversation_chain_handler in zip(
            ["chosen", "rejected"],
            [
                dataset.conversation_chain_handler_chosen,
                dataset.conversation_chain_handler_rejected,
            ],
        ):
            input_text_dict = conversation_chain_handler[idx]
            expected = {
                "prompts": [f"prompt {i}" for i in range(idx * 5, (idx + 1) * 5)],
                "answers": [
                    f"response {i + 1}" for i in range(idx * 5, (idx + 1) * 5 - 1)
                ]
                + [f"{name}_response {idx * 5 + 5}"],
                "systems": [""] * 5,
            }

            for key in expected:
                assert input_text_dict[key] == expected[key], (
                    input_text_dict[key],
                    expected[key],
                    name,
                )


def test_dataset_label_is_correct(df_with_conversation_chain_ids):
    cfg = ConfigProblemBase(
        dataset=ConfigNLPDPOLMDataset(
            prompt_column=("prompt_column",),
            chosen_response_column="chosen_response_column",
            rejected_response_column="rejected_response_column",
            parent_id_column="parent_id_column",
        )
    )
    dataset = CustomDataset(df_with_conversation_chain_ids, cfg, mode="train")

    for idx, item in enumerate(dataset):
        sample = dataset[idx]
        chosen_response = dataset.tokenizer.decode(
            sample["chosen_labels"][sample["chosen_labels"] != -100],
            skip_special_tokens=True,
        )
        rejected_response = dataset.tokenizer.decode(
            sample["rejected_labels"][sample["rejected_labels"] != -100],
            skip_special_tokens=True,
        )
        assert chosen_response == f"chosen_response {idx * 5 + 5}"
        assert rejected_response == f"rejected_response {idx * 5 + 5}"


def test_dataloader_has_correct_keys(df):
    cfg = ConfigProblemBase(
        dataset=ConfigNLPDPOLMDataset(
            prompt_column=("prompt_column",),
            chosen_response_column="chosen_response_column",
            rejected_response_column="rejected_response_column",
            parent_id_column="None",
        )
    )

    dataset = CustomDataset(df, cfg, mode="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        for key in batch:
            if idx != len(dataloader) - 1:
                assert batch[key].size(0) == 16, (
                    key,
                    batch[key].shape,
                )

            keys = [
                "chosen_input_ids",
                "chosen_attention_mask",
                "chosen_labels",
                "rejected_input_ids",
                "rejected_attention_mask",
                "rejected_labels",
                "prompt_input_ids",
                "prompt_attention_mask",
            ]
            assert set(batch.keys()) - set(keys) == set()
            assert set(keys) - set(batch.keys()) == set()


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
