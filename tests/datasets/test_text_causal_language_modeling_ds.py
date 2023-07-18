from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset


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

    predicted_text_clean = CustomDataset.clean_output(
        output=output, prompts=None, cfg=cfg
    )["predicted_text"]
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
    mock_config.dataset.parent_id_column = 'parent_id'

    df_1 = pd.DataFrame({"id": [1, 2, 3, 4],
                         "parent_id": [2, None, 4, 1],
                         "other_data": ['a', 'b', 'c', 'd']})
    CustomDataset.sanity_check(df_1, mock_config)

    df_2 = pd.DataFrame({"id": [1, 2, 3, 4],
                         "parent_id": [None, None, None, None],
                         "other_data": ['a', 'b', 'c', 'd']})
    CustomDataset.sanity_check(df_2, mock_config)

    invalid_df_1 = pd.DataFrame({"id": [1, 2, 3, 4],
                                 "parent_id": [5, 6, 7, 8],
                                 "other_data": ['a', 'b', 'c', 'd']})
    with pytest.raises(AssertionError, match="Parent id column contains ids that are not in the dataset"):
        CustomDataset.sanity_check(invalid_df_1, mock_config)

    invalid_df_2 = pd.DataFrame({"id": [1, 2, 3, 4],
                                 "parent_id": [1, 2, 3, 4],
                                 "other_data": ['a', 'b', 'c', 'd']})
    with pytest.raises(AssertionError, match="Parent id column is the same as id column for some rows"):
        CustomDataset.sanity_check(invalid_df_2, mock_config)

    invalid_df_3 = pd.DataFrame({"id": [1, 2, 3, 4],
                                 "parent_id": [2, 3, 4, 1],
                                 "other_data": ['a', 'b', 'c', 'd']})
    with pytest.raises(AssertionError,
                       match="Did not find any conversation start. Please ensure that some parent ids are empty."):
        CustomDataset.sanity_check(invalid_df_3, mock_config)


@pytest.fixture
def mock_auto_tokenizer():
    # from
    # https://github.com/deepset-ai/haystack/blob/b5aef24a7ebac55cb4ba492baf81a85598700b94/test/conftest.py#L908
    with patch("transformers.AutoTokenizer.from_pretrained", autospec=True) as mock_from_pretrained:
        yield mock_from_pretrained


def test_init(mock_auto_tokenizer):
    df = pd.DataFrame({
        'col_A': [1, 2, 3],
        'col_B': [4, 5, 6],
    })
    cfg = mock.MagicMock()
    cfg.dataset.prompt_column = 'col_A'
    cfg.dataset.answer_column = 'col_B'
    cfg.dataset.parent_id_column = 'None'
    cfg.dataset.system_column = 'None'

    cfg.dataset.text_system_start = ""
    cfg.dataset.text_prompt_start = ""
    cfg.dataset.text_answer_separator = ""

    dataset = CustomDataset(df, cfg)

    assert dataset.df.equals(df)
    assert dataset.mode == 'train'
    assert all(dataset.indices == np.array([0, 1, 2]))
    assert all(dataset.raw_prompts == ["1", "2", "3"])
    assert dataset.answers == ["4", "5", "6"]


def test_get_parent_ids(mock_auto_tokenizer):
    df = pd.DataFrame({'prompt': ['prompt 1', 'prompt 2', 'prompt 3'],
                       'answer': ['answer 1', 'answer 2', 'answer 3'],
                       'parent_id': [None, 0, 1],
                       'id': [0, 1, 2]})

    cfg = mock.MagicMock()
    cfg.dataset.prompt_column = 'prompt'
    cfg.dataset.answer_column = 'answer'
    cfg.dataset.parent_id_column = 'parent_id'
    cfg.dataset.text_system_start = "System:"
    cfg.dataset.text_prompt_start = "Prompt:"
    cfg.dataset.text_answer_separator = "Answer:"

    dataset = CustomDataset(df, cfg)

    assert dataset.get_parent_ids(0) == []
    assert dataset.get_parent_ids(1) == [0]
    assert dataset.get_parent_ids(2) == [0, 1]


def test_getitem(mock_auto_tokenizer):
    df = pd.DataFrame({'prompt': ['prompt 1', 'prompt 2', 'prompt 3'],
                       'answer': ['answer 1', 'answer 2', 'answer 3'],
                       'parent_id': [None, 0, 1],
                       'system': ['system 1', 'system 2', 'system 3'],
                       'id': [0, 1, 2]})

    cfg = mock.MagicMock()
    cfg.dataset.prompt_column = 'prompt'
    cfg.dataset.answer_column = 'answer'
    cfg.dataset.parent_id_column = 'parent_id'
    cfg.dataset.system_column = 'system'
    cfg.dataset.text_system_start = "System:"
    cfg.dataset.text_prompt_start = "Prompt:"
    cfg.dataset.text_answer_separator = "Answer:"

    cfg.tokenizer.max_length = 256
    cfg.tokenizer.max_length_answer = 256
    cfg.tokenizer.max_length_prompt = 256
    cfg.dataset.add_eos_token_to_answer = True

    dataset = CustomDataset(df, cfg)

    def tokenize(text, **kwargs):
        words = text.split()
        return {'input_ids': torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10][:len(words)])[None],
                'attention_mask': torch.ones(len(words))[None]
                }

    dataset.tokenizer = mock.MagicMock(side_effect=tokenize)
    dataset.tokenizer.eos_token_id = 999

    # dataset._get_sample_encoding = mock.MagicMock(return_value=[torch.Tensor(), torch.Tensor(), torch.Tensor()])
    # dataset.get_parent_encodings = mock.MagicMock(return_value=[])
    # dataset.get_reward_model_parent_prompt_text = mock.MagicMock(return_value="")
    # dataset.pad_tokens = mock.MagicMock(return_value={'key': 'value'})

    result = dataset[0]
    assert isinstance(result, dict)
