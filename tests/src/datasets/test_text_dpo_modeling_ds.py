import numpy as np
import pandas as pd
import pytest
import torch
from tqdm import tqdm

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMTokenizer,
)
from llm_studio.python_configs.text_dpo_modeling_config import (
    ConfigDPODataset,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_dpo_modeling_ds import CustomDataset


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "prompt_column": [f"prompt {i}" for i in range(200)],
            "answer_column": [f"chosen_response {i}" for i in range(200)],
            "rejected_answer_column": [f"rejected_response {i}" for i in range(200)],
        }
    )


@pytest.fixture
def df_with_conversation_chain_ids():
    """
    Create a dataframe with conversation chain ids, e.g.:
          prompt_column      answer_column rejected_answer_column parent_id_column id
    0      prompt 1         response 1             response 1             None  1
    1      prompt 2         response 2             response 2                1  2
    2      prompt 3         response 3             response 3                2  3
    3      prompt 4         response 4             response 4                3  4
    4      prompt 5  chosen_response 5    rejected_response 5                4  5
    5      prompt 6         response 6             response 6             None  6
    """
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
            "prompt_column": [f"prompt {idx}" for idx in ids],
            "answer_column": chosen_responses,
            "rejected_answer_column": rejected_responses,
            "parent_id_column": parent_ids,
            "id": ids,
        }
    )


def test_dataset_conversation_chain_is_correct(df_with_conversation_chain_ids):
    cfg = ConfigProblemBase(
        dataset=ConfigDPODataset(
            prompt_column=("prompt_column",),
            answer_column="answer_column",
            rejected_answer_column="rejected_answer_column",
            parent_id_column="parent_id_column",
        )
    )
    dataset = CustomDataset(df_with_conversation_chain_ids, cfg, mode="train")

    # Check for right formatting, e.g.:
    # dataset.conversation_chain_handler_chosen[0] ==
    # {
    #     "prompts": ["prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5"],
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
                dataset.conversation_chain_handler,
                dataset.conversation_chain_handler_rejected,
            ],
        ):
            input_text_dict = conversation_chain_handler[idx]
            expected = {
                "prompts": [f"prompt {i + 1}" for i in range(idx * 5, (idx + 1) * 5)],
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
        dataset=ConfigDPODataset(
            prompt_column=("prompt_column",),
            answer_column="answer_column",
            rejected_answer_column="rejected_answer_column",
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
        prompt = dataset.tokenizer.decode(
            sample["prompt_input_ids"][sample["prompt_input_ids"] != 0],
            skip_special_tokens=True,
        )

        assert (
            prompt == f"<|prompt|>prompt {idx * 5 + 1}"
            f"<|answer|>response {idx * 5 + 1}"
            f"<|prompt|>prompt {idx * 5 + 2}"
            f"<|answer|>response {idx * 5 + 2}"
            f"<|prompt|>prompt {idx * 5 + 3}"
            f"<|answer|>response {idx * 5 + 3}"
            f"<|prompt|>prompt {idx * 5 + 4}"
            f"<|answer|>response {idx * 5 + 4}"
            f"<|prompt|>prompt {idx * 5 + 5}"
            "<|answer|>"
        )
        assert chosen_response == f"chosen_response {idx * 5 + 5}"
        assert rejected_response == f"rejected_response {idx * 5 + 5}"


def test_dataloader_has_correct_keys(df):
    cfg = ConfigProblemBase(
        dataset=ConfigDPODataset(
            prompt_column=("prompt_column",),
            answer_column="answer_column",
            rejected_answer_column="rejected_answer_column",
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


def test_empy_answer_dataset_throws_no_error(df):
    cfg = ConfigProblemBase(
        dataset=ConfigDPODataset(
            prompt_column=("prompt_column",),
            answer_column="answer_column",
            rejected_answer_column="rejected_answer_column",
            add_eos_token_to_answer=False,
            add_eos_token_to_prompt=False,
            add_eos_token_to_system=False,
        ),
    )
    for column in ["prompt_column", "answer_column", "rejected_answer_column"]:
        values = df[column].values
        df[column] = ""
        dataset = CustomDataset(df, cfg, mode="train")
        [dataset[i] for i in range(len(dataset))]
        df[column] = values


@pytest.fixture
def df_single_prompt():
    prompt = """when ordering your sandstones, you select which colour scale you would want.
 it could be e.g. a 100% from grey/sand mix, or 80% fra beige/yellow mixed with 20% from black/brown.
  This is all lower case. Can you fix that?"""
    system = """You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can.
While performing the task think step-by-step and justify your steps."""
    answer = """When ordering your sandstones, you select which color scale you would want. It could be, for example, a 100% from grey/sand mix, or 80% from beige/yellow mixed with 20% from black/brown.
Step 1: Capitalize the first letter of the sentence.
Step 2: Correct the spelling of "color" (assuming American English usage).
Step 3: Replace ", e.g." with "for example" to clarify the sentence.
Step 4: Capitalize "a" in "100% from a grey/sand mix"
Step 5: Ensure the proper usage of words and punctuation throughout the revised sentence."""
    return pd.DataFrame(
        {
            "prompt": [prompt],
            "system": [system],
            "answer": [answer],
            "rejected_answer": ["I cannot do that."],
        }
    )


def generate_causal_lm_model_input_ids(df):
    from llm_studio.python_configs.text_causal_language_modeling_config import (
        ConfigNLPCausalLMDataset,
    )
    from llm_studio.python_configs.text_causal_language_modeling_config import (
        ConfigProblemBase as ConfigCausalLMProblemBase,
    )
    from llm_studio.src.datasets.text_causal_language_modeling_ds import (
        CustomDataset as CausalLMCustomDataset,
    )

    cfg = ConfigCausalLMProblemBase(
        llm_backbone="h2oai/h2ogpt-4096-llama2-7b",
        dataset=ConfigNLPCausalLMDataset(
            system_column="system",
            prompt_column=("prompt",),
            answer_column="answer",
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length_prompt=256, max_length_answer=256, max_length=512
        ),
    )
    dataset = CausalLMCustomDataset(df, cfg, mode="train")
    return dataset[0]


def test_dataset_prompt_ids_are_the_same_as_for_causal_language_modeling(
    df_single_prompt,
):
    """
    DPO model should generate the same prompts as causal language modeling
    """
    generated_text_causal_lm = generate_causal_lm_model_input_ids(df_single_prompt)

    cfg = ConfigProblemBase(
        llm_backbone="h2oai/h2ogpt-4096-llama2-7b",
        dataset=ConfigDPODataset(
            system_column="system",
            prompt_column=("prompt",),
            answer_column="answer",
            rejected_answer_column="rejected_answer",
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length_prompt=256, max_length_answer=256, max_length=512
        ),
    )
    dataset = CustomDataset(df_single_prompt, cfg, mode="train")
    generated_text = dataset[0]

    for key in ["prompt_input_ids", "prompt_attention_mask"]:
        assert torch.all(
            generated_text_causal_lm[key] == generated_text[key]
        ), f"{key} is not the same"
