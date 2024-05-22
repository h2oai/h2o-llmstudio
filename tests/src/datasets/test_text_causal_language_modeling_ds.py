from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from llm_studio.app_utils.default_datasets import (
    prepare_default_dataset_causal_language_modeling,
)
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMTokenizer,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


def test_prepare_default_dataset(tmp_path):
    df = prepare_default_dataset_causal_language_modeling(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert set(df.keys()) == set(
        ["instruction", "output", "id", "parent_id", "lang", "rank"]
    )
    assert df.shape == (13026, 6)


def test_clean_output():
    output = {
        "predicted_text": np.array(
            [
                "This is a test",
                "This is a test <stop> This is a test",
                "This is a test <stop2> This is a test",
                "This is a test <stop3> <stop> This is a test",
                "<stop2> <stop> This is a test",
                "This is a test <stop>",
            ]
        )
    }

    cfg = mock.MagicMock()
    cfg.tokenizer._stop_words = ["<stop>", "<stop2>", "<stop3>"]

    predicted_text_clean = CustomDataset.clean_output(output=output, cfg=cfg)[
        "predicted_text"
    ]
    assert predicted_text_clean == [
        "This is a test",
        "This is a test",
        "This is a test",
        "This is a test",
        "",
        "This is a test",
    ]


def test_sanity_check_raises_error():
    mock_config = MagicMock()
    mock_config.dataset.parent_id_column = "parent_id"
    mock_config.dataset.answer_column = "answer"

    df_1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [2, None, 4, 1],
            "answer": ["a", "b", "c", "d"],
            "other_data": ["a", "b", "c", "d"],
        }
    )
    CustomDataset.sanity_check(df_1, mock_config)

    df_2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [None, None, None, None],
            "answer": ["a", "b", "c", "d"],
            "other_data": ["a", "b", "c", "d"],
        }
    )
    CustomDataset.sanity_check(df_2, mock_config)

    invalid_df_1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [1, 2, 3, 4],
            "answer": ["a", "b", "c", "d"],
            "other_data": ["a", "b", "c", "d"],
        }
    )
    with pytest.raises(
        AssertionError, match="Parent id column is the same as id column for some rows"
    ):
        CustomDataset.sanity_check(invalid_df_1, mock_config)

    invalid_df_2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [2, 3, 4, 1],
            "other_data": ["a", "b", "c", "d"],
        }
    )
    with pytest.raises(
        AssertionError,
        match="Did not find any conversation start. "
        "Please ensure that some parent ids are empty.",
    ):
        CustomDataset.sanity_check(invalid_df_2, mock_config)


@pytest.fixture
def mock_auto_tokenizer():
    # from
    # https://github.com/deepset-ai/haystack/blob/b5aef24a7ebac55cb4ba492baf81a85598700b94/test/conftest.py#L908
    with patch(
        "transformers.AutoTokenizer.from_pretrained", autospec=True
    ) as mock_from_pretrained:
        yield mock_from_pretrained


def test_init(mock_auto_tokenizer):
    df = pd.DataFrame(
        {
            "col_A": [1, 2, 3],
            "col_B": [4, 5, 6],
        }
    )
    cfg = mock.MagicMock()
    cfg.dataset.prompt_column = "col_A"
    cfg.dataset.answer_column = "col_B"
    cfg.dataset.parent_id_column = "None"
    cfg.dataset.system_column = "None"

    cfg.dataset.text_system_start = ""
    cfg.dataset.text_prompt_start = ""
    cfg.dataset.text_answer_separator = ""

    cfg.tokenizer.tokenizer_kwargs = '{"use_fast": true, "add_prefix_space": false}'

    dataset = CustomDataset(df, cfg)

    assert dataset.df.equals(df)
    assert dataset.mode == "train"


def test_getitem():
    df = pd.DataFrame(
        {
            "prompt": ["prompt 1", "prompt 2", "prompt 3"],
            "answer": ["answer 1", "answer 2", "answer 3"],
            "parent_id": [None, 0, 1],
            "system": ["system 1", "system 2", "system 3"],
            "id": [0, 1, 2],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            system_column="system",
            text_system_start="System:",
            text_prompt_start="Prompt:",
            text_answer_separator="Answer:",
            add_eos_token_to_answer=True,
            limit_chained_samples=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=513),
    )

    cfg.llm_backbone = "EleutherAI/pythia-2.8b-deduped"

    dataset = CustomDataset(df, cfg)
    assert len(dataset) == 1

    result = dataset[0]
    assert isinstance(result, dict)
    assert set(result.keys()) == {
        "labels",
        "input_ids",
        "attention_mask",
        "prompt_input_ids",
        "prompt_attention_mask",
        "answer_input_ids",
        "answer_attention_mask",
    }

    assert (
        dataset.tokenizer.decode(result["input_ids"], skip_special_tokens=True)
        == "System:system 1"
        "Prompt:prompt 1"
        "Answer:answer 1"
        "Prompt:prompt 2"
        "Answer:answer 2"
        "Prompt:prompt 3"
        "Answer:answer 3"
    )

    assert (
        dataset.tokenizer.decode(result["prompt_input_ids"], skip_special_tokens=True)
        == "System:system 1"
        "Prompt:prompt 1"
        "Answer:answer 1"
        "Prompt:prompt 2"
        "Answer:answer 2"
        "Prompt:prompt 3"
        "Answer:"
    )

    assert (
        dataset.tokenizer.decode(result["input_ids"], skip_special_tokens=False)
        == "<|endoftext|>" * 475 + "System:system 1"
        "<|endoftext|>"
        "Prompt:prompt 1"
        "<|endoftext|>"
        "Answer:answer 1"
        "<|endoftext|>"
        "Prompt:prompt 2"
        "<|endoftext|>"
        "Answer:answer 2"
        "<|endoftext|>"
        "Prompt:prompt 3"
        "<|endoftext|>"
        "Answer:answer 3"
        "<|endoftext|>"
    )

    assert result["input_ids"].shape == (513,)
    assert result["prompt_input_ids"].shape == (513,)


def test_getitem_no_chaining():
    df = pd.DataFrame(
        {
            "prompt": ["prompt 1", "prompt 2", "prompt 3"],
            "answer": ["answer 1", "answer 2", "answer 3"],
            "parent_id": [None, 0, 1],
            "system": ["system 1", "system 2", "system 3"],
            "id": [0, 1, 2],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="None",
            system_column="system",
            text_system_start="System:",
            text_prompt_start="Prompt:",
            text_answer_separator="Answer:",
            add_eos_token_to_answer=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=513),
    )

    cfg.llm_backbone = "EleutherAI/pythia-2.8b-deduped"

    dataset = CustomDataset(df, cfg)
    assert len(dataset) == 3

    for i in range(3):
        result = dataset[i]
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "labels",
            "input_ids",
            "attention_mask",
            "prompt_input_ids",
            "prompt_attention_mask",
            "answer_input_ids",
            "answer_attention_mask",
        }

        assert (
            dataset.tokenizer.decode(result["input_ids"], skip_special_tokens=True)
            == f"System:system {i+1}"
            f"Prompt:prompt {i+1}"
            f"Answer:answer {i+1}"
        )

        assert (
            dataset.tokenizer.decode(
                result["prompt_input_ids"], skip_special_tokens=True
            )
            == f"System:system {i+1}"
            f"Prompt:prompt {i+1}"
            "Answer:"
        )


def test_encode():
    df = pd.DataFrame(
        {
            "prompt": ["a", "a"],
            "answer": ["b", "b"],
            "parent_id": [None, 0],
            "id": [0, 1],
        }
    )

    cfg = ConfigProblemBase(
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            text_prompt_start="<|prompt|>",
            text_answer_separator="<|answer|>",
            add_eos_token_to_answer=True,
            limit_chained_samples=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(
            max_length=64,
            tokenizer_kwargs='{"use_fast": true, "add_prefix_space": false}',
        ),
    )

    cfg.llm_backbone = "h2oai/h2o-danube2-1.8b-base"

    dataset = CustomDataset(df, cfg)
    assert len(dataset) == 1

    result = dataset[0]
    out = dataset.tokenizer.decode(result["input_ids"]).replace("<unk>", "")
    assert out == "<|prompt|>a</s><|answer|>b</s><|prompt|>a</s><|answer|>b</s>"
