import os
import random
import unittest
from unittest.mock import MagicMock

import pandas as pd
import pytest

from llm_studio.app_utils.utils import prepare_default_dataset
from llm_studio.src.datasets.conversation_chain_handler import ConversationChainHandler
from llm_studio.src.utils.data_utils import load_train_valid_data


@pytest.fixture
def cfg_mock():
    cfg = MagicMock()
    cfg.dataset.train_dataframe = "/path/to/train/data"
    cfg.dataset.validation_dataframe = "/path/to/validation/data"

    cfg.dataset.system_column = "None"
    cfg.dataset.prompt_column = "prompt"
    cfg.dataset.answer_column = "answer"

    cfg.dataset.validation_size = 0.2
    return cfg


@pytest.fixture
def read_dataframe_drop_missing_labels_mock(monkeypatch):
    data = {
        "prompt": [f"Prompt{i}" for i in range(100)],
        "answer": [f"Answer{i}" for i in range(100)],
        "id": list(range(100)),
    }
    df = pd.DataFrame(data)
    mock = MagicMock(return_value=df)
    monkeypatch.setattr(
        "llm_studio.src.utils.data_utils.read_dataframe_drop_missing_labels", mock
    )
    return mock


numbers = list(range(100))
random.shuffle(
    numbers,
)
groups = [numbers[n::13] for n in range(13)]


@pytest.fixture
def conversation_chain_ids_mock(monkeypatch):
    def mocked_init(self, *args, **kwargs):
        self.conversation_chain_ids = groups

    with unittest.mock.patch.object(
        ConversationChainHandler, "__init__", new=mocked_init
    ):
        yield


def test_get_data_custom_validation_strategy(
    cfg_mock, read_dataframe_drop_missing_labels_mock
):
    cfg_mock.dataset.validation_strategy = "custom"
    train_df, val_df = load_train_valid_data(cfg_mock)
    assert len(train_df), len(val_df) == 100


def test_get_data_automatic_split(
    cfg_mock, read_dataframe_drop_missing_labels_mock, conversation_chain_ids_mock
):
    cfg_mock.dataset.validation_strategy = "automatic"
    train_df, val_df = load_train_valid_data(cfg_mock)
    train_ids = set(train_df["id"].tolist())
    val_ids = set(val_df["id"].tolist())

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids) + len(val_ids) == 100

    shared_groups = [
        i for i in groups if not train_ids.isdisjoint(i) and not val_ids.isdisjoint(i)
    ]
    assert len(shared_groups) == 0


@pytest.mark.skip("slow test due to downloading oasst")
def test_oasst_data_automatic_split(tmp_path):
    prepare_default_dataset(tmp_path)

    cfg_mock = MagicMock()
    for file in os.listdir(tmp_path):
        if file.endswith(".pq"):
            cfg_mock.dataset.train_dataframe = os.path.join(tmp_path, file)

            cfg_mock.dataset.system_column = "None"
            cfg_mock.dataset.prompt_column = ("instruction",)
            cfg_mock.dataset.answer_column = "output"
            cfg_mock.dataset.parent_id_column = "parent_id"

            cfg_mock.dataset.validation_strategy = "automatic"

            for validation_size in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
                cfg_mock.dataset.validation_size = validation_size

                train_df, val_df = load_train_valid_data(cfg_mock)
                assert set(train_df["parent_id"].dropna().values).isdisjoint(
                    set(val_df["id"].dropna().values)
                )
                assert set(val_df["parent_id"].dropna().values).isdisjoint(
                    set(train_df["id"].dropna().values)
                )
