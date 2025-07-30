import os

import pandas as pd
import pytest

from llm_studio.src.possible_values import (
    Columns,
    DatasetValue,
    Files,
    Number,
    String,
    _scan_dirs,
    _scan_files,
    strip_common_prefix,
)


# Helper function to create a temporary directory structure
@pytest.fixture
def temp_dir_structure(tmp_path):
    base_dir = tmp_path / "test_dir"
    base_dir.mkdir()
    (base_dir / "subdir1").mkdir()
    (base_dir / "subdir2").mkdir()
    (base_dir / "subdir1" / "subsubdir").mkdir()
    (base_dir / "file1.csv").touch()
    (base_dir / "file2.json").touch()
    (base_dir / "__meta_info__train.json").touch()
    (base_dir / "subdir1" / "file3.parquet").touch()
    (base_dir / "subdir1" / "__meta_info__train.pq.csv").touch()
    return base_dir


def test_scan_dirs(temp_dir_structure):
    dirs = _scan_dirs(str(temp_dir_structure))
    expected = [
        str(temp_dir_structure / "subdir1") + "/",
        str(temp_dir_structure / "subdir2") + "/",
        str(temp_dir_structure / "subdir1" / "subsubdir") + "/",
    ]
    assert set(dirs) == set(expected)


def test_scan_files(temp_dir_structure):
    files = _scan_files(str(temp_dir_structure))
    expected = [
        str(temp_dir_structure / "file1.csv"),
        str(temp_dir_structure / "file2.json"),
        str(temp_dir_structure / "subdir1" / "file3.parquet"),
    ]
    assert set(files) == set(expected)


def test_strip_common_prefix():
    paths = ["/a/b/c/file1.txt", "/a/b/c/file2.txt", "/a/b/d/file3.txt"]
    stripped = strip_common_prefix(paths)
    assert stripped == ("c/file1.txt", "c/file2.txt", "d/file3.txt")

    # Test with ignore_set
    paths_with_ignore = paths + ["ignore_this"]
    stripped_with_ignore = strip_common_prefix(
        paths_with_ignore, ignore_set={"ignore_this"}
    )
    assert stripped_with_ignore == (
        "c/file1.txt",
        "c/file2.txt",
        "d/file3.txt",
        "ignore_this",
    )


def test_strip_common_prefix_empty_paths():
    paths = []
    stripped = strip_common_prefix(paths)
    assert stripped == tuple([])


def test_number_slider():
    num = Number(min=0, max=10, step=0.5)
    assert num.min == 0
    assert num.max == 10
    assert num.step == 0.5


def test_number_spinbox():
    num = Number(min=0, step=0.5)
    assert num.min == 0
    assert num.max is None
    assert num.step == 0.5


def test_number_impossible_values():
    with pytest.raises(ValueError):
        Number(min=0, max=10, step="a")

    with pytest.raises(ValueError):
        Number(min=0, max="a", step=0.5)

    with pytest.raises(ValueError):
        Number(min="a", max=10, step=0.5)

    with pytest.raises(ValueError):
        Number(min=0, max=10)

    with pytest.raises(ValueError):
        Number(min=10, max=1, step=1)

    with pytest.raises(ValueError):
        Number(min=10, max=0, step=1)


def test_string_tuple_of_strings():
    s = String(
        values=("a", "b", "c"), allow_custom=True, placeholder="Select an option"
    )
    assert s.values == ("a", "b", "c")
    assert s.allow_custom is True
    assert s.placeholder == "Select an option"


def test_string_tuple_of_tuples():
    s = String(
        values=(("a", "hello there"), ("b", "hello there"), ("c", "hello there")),
        allow_custom=True,
        placeholder="Select an option",
    )
    assert s.values == (
        ("a", "hello there"),
        ("b", "hello there"),
        ("c", "hello there"),
    )
    assert s.allow_custom is True
    assert s.placeholder == "Select an option"


def test_string_impossible_values():
    with pytest.raises(ValueError):
        String(values=("a", "b", "c"), allow_custom="a")

    with pytest.raises(ValueError):
        String(values=("a", "b", "c"), placeholder=True)


class TestDatasetValue:
    def test_get_value(self):
        """
        Test that NotImplementedError is raised when get_value is called directly

        This is a base class and should not be used directly.
        get_value is an abstract method.
        """
        dataset_value = DatasetValue()
        with pytest.raises(NotImplementedError):
            dataset_value.get_value(None, None, None)

    @pytest.mark.parametrize(
        "current_values, possible_values, prefer_with, expected",
        [
            (["a", "b"], ["a", "b", "c"], None, ["a", "b"]),
            (["a", "d"], ["a", "b", "c"], None, ["a"]),
            ([], ["a", "b", "c"], None, ["a"]),
            (["d", "e"], ["a", "b", "c"], None, ["a"]),
            ([], [], None, [""]),
            (["a", "b"], [], None, [""]),
        ],
    )
    def test_compute_current_values_basic(
        self, current_values, possible_values, prefer_with, expected
    ):
        result = DatasetValue._compute_current_values(
            current_values, possible_values, prefer_with
        )
        assert result == expected

    def test_compute_current_values_with_prefer_function(self):
        current_values = []
        possible_values = ["a", "b", "c", "d"]

        def prefer_with(x):
            return x in ["b", "c"]

        result = DatasetValue._compute_current_values(
            current_values, possible_values, prefer_with
        )
        assert result == ["b", "c"]

    def test_compute_current_values_with_prefer_function_single_match(self):
        current_values = []
        possible_values = ["a", "b", "d"]

        def prefer_with(x):
            return x in ["b", "c"]

        result = DatasetValue._compute_current_values(
            current_values, possible_values, prefer_with
        )
        assert result == ["b"]

    def test_compute_current_values_prefer_function_no_match(self):
        current_values = []
        possible_values = ["a", "b", "c"]

        def prefer_with(x):
            return x == "d"

        result = DatasetValue._compute_current_values(
            current_values, possible_values, prefer_with
        )
        assert result == [
            "a"
        ]  # Should return first possible value when no preference matches

    def test_compute_current_values_all_filtered_out(self):
        current_values = ["d", "e"]
        possible_values = ["a", "b", "c"]

        result = DatasetValue._compute_current_values(current_values, possible_values)
        assert result == [
            "a"
        ]  # Should return first possible value when all current values are filtered out

    @pytest.mark.parametrize(
        "current_values, possible_values",
        [
            (["a", "a", "b"], ["a", "b", "c"]),
            (["a", "b", "a"], ["a", "b", "c"]),
            (["a", "b", "a"], ["a", "b", "c", "a"]),
            (["a", "b"], ["a", "a", "b"]),
        ],
    )
    def test_compute_current_values_duplicates(self, current_values, possible_values):
        with pytest.raises(ValueError):
            DatasetValue._compute_current_values(current_values, possible_values)

    def test_compute_current_values_type_check(self):
        current_values = ["a", "b"]
        possible_values = ["a", "b", "c"]
        result = DatasetValue._compute_current_values(current_values, possible_values)
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)


# Mock dataset for testing
@pytest.fixture
def mock_dataset(temp_dir_structure):
    return {
        "path": str(temp_dir_structure),
        "dataframe": pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [True, False, True]}
        ),
    }


@pytest.mark.parametrize(
    "filename", ["file1.csv", "file2.json", "subdir1/file3.parquet", "non-existant"]
)
def test_files(mock_dataset, filename):
    files = Files()
    result, value = files.get_value(mock_dataset, filename, str)
    assert isinstance(result, String)
    assert value == os.path.join(mock_dataset["path"], "file1.csv")


@pytest.mark.parametrize("col", ["col1", "col2", "col3", "non-existant"])
def test_columns(mock_dataset, col):
    cols = Columns()
    result, value = cols.get_value(mock_dataset, col, str)
    assert isinstance(result, String)
    assert set(result.values) == {"col1", "col2", "col3"}
    if col == "non-existant":
        assert value == "col1"
    else:
        assert value == col
