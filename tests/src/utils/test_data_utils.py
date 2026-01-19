import os
import pathlib
import random
import unittest
from unittest.mock import MagicMock

import pandas as pd
import pytest

from llm_studio.app_utils.default_datasets import (
    prepare_default_dataset_causal_language_modeling,
)
from llm_studio.src.datasets.conversation_chain_handler import ConversationChainHandler
from llm_studio.src.utils.data_utils import (
    is_valid_data_frame,
    load_train_valid_data,
    read_dataframe,
)


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


def test_oasst_data_automatic_split(tmp_path: pathlib.Path):
    prepare_default_dataset_causal_language_modeling(tmp_path)
    assert len(os.listdir(tmp_path)) > 0, tmp_path
    cfg_mock = MagicMock()
    for file in os.listdir(tmp_path):
        if file.endswith(".pq"):
            cfg_mock.dataset.train_dataframe = os.path.join(tmp_path, file)

            cfg_mock.dataset.system_column = "None"
            cfg_mock.dataset.prompt_column = ("instruction",)
            cfg_mock.dataset.answer_column = "output"
            cfg_mock.dataset.parent_id_column = "parent_id"
            cfg_mock.dataset.id_column = "id"
            cfg_mock.dataset.prompt_column_separator = "\n\n"

            cfg_mock.dataset.validation_strategy = "automatic"

            for validation_size in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
                cfg_mock.dataset.validation_size = validation_size

                train_df, val_df = load_train_valid_data(cfg_mock)
                assert set(train_df["parent_id"].dropna().values).isdisjoint(
                    set(val_df["id"].dropna().values)
                )
                assert set(val_df["parent_id"].dropna().values).isdisjoint(
                    set(train_df["id"].dropna().values)
                )
                assert (len(val_df) / (len(train_df) + len(val_df))) == pytest.approx(
                    validation_size, 0.05
                )


class TestReadDataframe:
    """Tests for read_dataframe function covering file ingestion."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "prompt": ["Hello", "World", "Test"],
                "answer": ["Hi", "Earth", "Check"],
                "id": [1, 2, 3],
            }
        )

    def test_read_csv_lowercase(self, tmp_path, sample_dataframe):
        """Test reading a CSV file with lowercase extension."""
        csv_path = tmp_path / "test.csv"
        sample_dataframe.to_csv(csv_path, index=False)

        df = read_dataframe(str(csv_path))
        assert len(df) == 3
        assert list(df.columns) == ["prompt", "answer", "id"]

    def test_read_csv_uppercase(self, tmp_path, sample_dataframe):
        """Test reading a CSV file with uppercase extension."""
        csv_path = tmp_path / "test.CSV"
        sample_dataframe.to_csv(csv_path, index=False)

        df = read_dataframe(str(csv_path))
        assert len(df) == 3
        assert list(df.columns) == ["prompt", "answer", "id"]

    def test_read_csv_mixed_case(self, tmp_path, sample_dataframe):
        """Test reading a CSV file with mixed case extension."""
        csv_path = tmp_path / "test.Csv"
        sample_dataframe.to_csv(csv_path, index=False)

        df = read_dataframe(str(csv_path))
        assert len(df) == 3

    def test_read_parquet_lowercase_pq(self, tmp_path, sample_dataframe):
        """Test reading a Parquet file with .pq extension."""
        pq_path = tmp_path / "test.pq"
        sample_dataframe.to_parquet(pq_path)

        df = read_dataframe(str(pq_path))
        assert len(df) == 3
        assert list(df.columns) == ["prompt", "answer", "id"]

    def test_read_parquet_uppercase_pq(self, tmp_path, sample_dataframe):
        """Test reading a Parquet file with .PQ extension."""
        pq_path = tmp_path / "test.PQ"
        sample_dataframe.to_parquet(pq_path)

        df = read_dataframe(str(pq_path))
        assert len(df) == 3

    def test_read_parquet_lowercase_parquet(self, tmp_path, sample_dataframe):
        """Test reading a Parquet file with .parquet extension."""
        parquet_path = tmp_path / "test.parquet"
        sample_dataframe.to_parquet(parquet_path)

        df = read_dataframe(str(parquet_path))
        assert len(df) == 3

    def test_read_parquet_uppercase_parquet(self, tmp_path, sample_dataframe):
        """Test reading a Parquet file with .PARQUET extension."""
        parquet_path = tmp_path / "test.PARQUET"
        sample_dataframe.to_parquet(parquet_path)

        df = read_dataframe(str(parquet_path))
        assert len(df) == 3

    def test_read_parquet_mixed_case(self, tmp_path, sample_dataframe):
        """Test reading a Parquet file with mixed case extension."""
        parquet_path = tmp_path / "test.Parquet"
        sample_dataframe.to_parquet(parquet_path)

        df = read_dataframe(str(parquet_path))
        assert len(df) == 3

    def test_read_json_returns_empty_dataframe(self):
        """Test that JSON files return an empty DataFrame (not supported)."""
        df = read_dataframe("somefile.json")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_json_uppercase_returns_empty_dataframe(self):
        """Test that JSON files with uppercase extension return empty DataFrame."""
        df = read_dataframe("somefile.JSON")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_read_empty_path_returns_empty_dataframe(self):
        """Test that empty path returns an empty DataFrame."""
        df = read_dataframe("")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_unsupported_extension_raises_error(self):
        """Test that unsupported file extensions raise ValueError."""
        with pytest.raises(ValueError, match="Could not determine type of file"):
            read_dataframe("somefile.xlsx")

    def test_unsupported_extension_txt_raises_error(self):
        """Test that .txt extension raises ValueError."""
        with pytest.raises(ValueError, match="Could not determine type of file"):
            read_dataframe("somefile.txt")

    def test_read_csv_with_n_rows_limit(self, tmp_path):
        """Test reading CSV with row limit."""
        df_large = pd.DataFrame(
            {"col1": range(100), "col2": [f"val{i}" for i in range(100)]}
        )
        csv_path = tmp_path / "large.csv"
        df_large.to_csv(csv_path, index=False)

        df = read_dataframe(str(csv_path), n_rows=10)
        assert len(df) == 10

    def test_read_parquet_with_n_rows_limit(self, tmp_path):
        """Test reading Parquet with row limit."""
        df_large = pd.DataFrame(
            {"col1": range(100), "col2": [f"val{i}" for i in range(100)]}
        )
        parquet_path = tmp_path / "large.parquet"
        df_large.to_parquet(parquet_path)

        df = read_dataframe(str(parquet_path), n_rows=10)
        assert len(df) == 10

    def test_fill_columns(self, tmp_path):
        """Test that fill_columns fills NaN values."""
        df_with_nan = pd.DataFrame(
            {"prompt": ["Hello", None, "Test"], "answer": ["Hi", "Earth", None]}
        )
        csv_path = tmp_path / "with_nan.csv"
        df_with_nan.to_csv(csv_path, index=False)

        df = read_dataframe(str(csv_path), fill_columns=["prompt", "answer"])
        assert df["prompt"].isna().sum() == 0
        assert df["answer"].isna().sum() == 0
        assert df.loc[1, "prompt"] == ""
        assert df.loc[2, "answer"] == ""

    def test_non_missing_columns_drops_rows(self, tmp_path):
        """Test that non_missing_columns drops rows with NaN in specified columns."""
        df_with_nan = pd.DataFrame(
            {"prompt": ["Hello", None, "Test"], "answer": ["Hi", "Earth", "Check"]}
        )
        csv_path = tmp_path / "with_nan.csv"
        df_with_nan.to_csv(csv_path, index=False)

        df = read_dataframe(str(csv_path), non_missing_columns=["prompt"])
        assert len(df) == 2
        assert "Hello" in df["prompt"].values
        assert "Test" in df["prompt"].values


class TestIsValidDataFrame:
    """Tests for is_valid_data_frame function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    def test_valid_csv_lowercase(self, tmp_path, sample_dataframe):
        """Test validation of CSV with lowercase extension."""
        csv_path = tmp_path / "test.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        assert is_valid_data_frame(str(csv_path)) is True

    def test_valid_csv_uppercase(self, tmp_path, sample_dataframe):
        """Test validation of CSV with uppercase extension."""
        csv_path = tmp_path / "test.CSV"
        sample_dataframe.to_csv(csv_path, index=False)
        assert is_valid_data_frame(str(csv_path)) is True

    def test_valid_parquet_lowercase_pq(self, tmp_path, sample_dataframe):
        """Test validation of Parquet with .pq extension."""
        pq_path = tmp_path / "test.pq"
        sample_dataframe.to_parquet(pq_path)
        assert is_valid_data_frame(str(pq_path)) is True

    def test_valid_parquet_uppercase_pq(self, tmp_path, sample_dataframe):
        """Test validation of Parquet with .PQ extension."""
        pq_path = tmp_path / "test.PQ"
        sample_dataframe.to_parquet(pq_path)
        assert is_valid_data_frame(str(pq_path)) is True

    def test_valid_parquet_lowercase_parquet(self, tmp_path, sample_dataframe):
        """Test validation of Parquet with .parquet extension."""
        parquet_path = tmp_path / "test.parquet"
        sample_dataframe.to_parquet(parquet_path)
        assert is_valid_data_frame(str(parquet_path)) is True

    def test_valid_parquet_uppercase_parquet(self, tmp_path, sample_dataframe):
        """Test validation of Parquet with .PARQUET extension."""
        parquet_path = tmp_path / "test.PARQUET"
        sample_dataframe.to_parquet(parquet_path)
        assert is_valid_data_frame(str(parquet_path)) is True

    def test_invalid_extension_returns_false(self, tmp_path):
        """Test that unsupported extensions return False."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("some content")
        assert is_valid_data_frame(str(txt_path)) is False

    def test_nonexistent_csv_returns_false(self, tmp_path):
        """Test that non-existent CSV returns False."""
        csv_path = tmp_path / "nonexistent.csv"
        assert is_valid_data_frame(str(csv_path)) is False

    def test_corrupted_parquet_returns_false(self, tmp_path):
        """Test that corrupted Parquet returns False."""
        parquet_path = tmp_path / "corrupted.parquet"
        parquet_path.write_bytes(b"\x00\x01\x02\x03")
        assert is_valid_data_frame(str(parquet_path)) is False
