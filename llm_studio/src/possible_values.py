import os
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from pydantic.dataclasses import dataclass


def _scan_dirs(dirname: str) -> list[str]:
    """
    Recursively scans a directory for subfolders.

    Args:
        dirname (str): The directory to scan.

    Returns:
        List[str]: A list of subfolder paths, with '/' appended to each path.
    """

    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(_scan_dirs(dirname))
    subfolders = [x + "/" if x[-1] != "/" else x for x in subfolders]
    return subfolders


def _scan_files(
    dirname: str,
    extensions: tuple[str, ...] = (
        ".csv",
        ".CSV",
        ".pq",
        ".PQ",
        ".parquet",
        ".PARQUET",
    ),
) -> list[str]:
    """
    Scans a directory for files with given extension

    Excludes files starting with "__meta_info__".

    Args:
        dirname (str): The directory to scan.
        extensions (Tuple[str, ...]): File extensions to consider.

    Returns:
        List[str]: A sorted list of file paths matching the given extensions.
    """
    path_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(dirname)
        for filename in filenames
        if any([filename.endswith(ext) for ext in extensions])
        and not filename.startswith("__meta_info__")
    ]
    return sorted(path_list)


def strip_common_prefix(
    paths: Sequence[str], ignore_set: set[str] = set()
) -> tuple[str, ...]:
    """
    Strips the common prefix from all given paths.

    Args:
        paths (Sequence[str]): The paths to strip.
        ignore_set (Set[str]): Set of path names to ignore when computing the prefix.

    Returns:
        Tuple[str, ...]: A tuple of paths with common prefixes removed.
    """

    paths_to_check = [
        os.path.split(os.path.normpath(path))[0]
        for path in paths
        if path not in ignore_set
    ]

    if len(paths_to_check) == 0:
        return tuple(paths)

    prefix = os.path.commonpath(paths_to_check)
    stripped = tuple(
        [
            path if path in ignore_set else os.path.relpath(path, prefix)
            for path in paths
        ]
    )

    return stripped


class Value:
    """Base class for value types."""

    pass


@dataclass
class Number:
    """
    Represents a numeric range for a setting with optional constraints.

    Attributes:
        min (float | int): Minimum allowed value. Must be less than or equal to `max`.
        step (float | int]): Step size for value increments
        max (float | None): Maximum allowed value. Optional.
            If provided, the UI component will be rendered as a slider. Otherwise as \
                a spinbox.
    """

    min: float | int
    step: float | int
    max: float | int | None = None

    def __post_init__(self):
        if self.max is not None and self.min > self.max:
            raise ValueError(
                f"Expected `min <= max`, got min={self.min} > max={self.max}"
            )


@dataclass
class String:
    """
    Represents possible string values for a setting with optional constraints.

    Attributes:
        values (Tuple[str, ...] | Tuple[Tuple[str, str], ...]):
            Possible values for the string.
            - a tuple of tuples (value, name)
            - a tuple of strings. In that case the value will be used for name and value
        allow_custom (bool): Whether custom values are allowed. This will render a \
            combobox. If False (default), a dropdown will be rendered.
        placeholder (Optional[str]): Placeholder text for input fields.
    """

    values: tuple[str, ...] | tuple[tuple[str, str], ...]
    allow_custom: bool = False
    placeholder: str | None = None


class DatasetValue:
    """Base class for dataset-related values."""

    @abstractmethod
    def get_value(
        self, dataset: Any, value: Any, type_annotation: type
    ) -> tuple[String, Any]:
        """
        Abstract method to get the value for a dataset.

        Args:
            dataset (Any): The dataset object.
            value (Any): The current value.
            type_annotation (type): The expected type of the value.

        Returns:
            Tuple[String, Any]: A tuple containing the String object and the value.
        """
        raise NotImplementedError

    @staticmethod
    def _compute_current_values(
        current_values: list[str],
        possible_values: list[str],
        prefer_with: Callable[[str], bool] | None = None,
    ) -> list[str]:
        """
        Compute current values based on possible values and preferences.

        This method does not handle duplicate values and raises an error if either \
            `current_values` or `possible_values` contain duplicates.

        Args:
            current_values (List[str]): The preliminary current values.
            possible_values (List[str]): All possible values.
            prefer_with (Optional[Callable[[str], bool]]): Function determining which \
                values to prefer as default.

        Returns:
            List[str]: A list of computed current values.

        Raises:
            ValueError: If either `current_values` or `possible_values` contain \
                duplicate
        """

        if len(set(current_values)) != len(current_values):
            raise ValueError("Duplicate values in `current_values`")

        if len(set(possible_values)) != len(possible_values):
            raise ValueError("Duplicate values in `possible_values`")

        if len(possible_values) == 0:
            return [""]

        # allow only values which are in the possible values
        current_values = list(
            filter(lambda value: value in possible_values, current_values)
        )

        if len(current_values) == 0:
            # if the values are empty, take all the values where `prefer_with` is true
            for c in possible_values:
                if prefer_with is not None and prefer_with(c):
                    current_values.append(c)

            # if they are still empty, just take the first possible value
            if len(current_values) == 0:
                current_values = [possible_values[0]]

        return current_values


@dataclass
class Files(DatasetValue):
    """
    Represents a selection of files from a dataset.

    Used to select a file from a dataset for e.g. `train_dataframe`.

    Attributes:
        add_none (bool): Whether to add a "None" option.
        prefer_with (Optional[Callable[[str], bool]]): Function to determine preferred \
            values.
        prefer_none (bool): Whether to prefer "None" as the default option.
    """

    add_none: bool = False
    prefer_with: Callable[[str], bool] | None = None
    # For the case where no match found, whether to prioritize
    # selecting any file or selecting no file
    prefer_none: bool = True

    def get_value(
        self, dataset: Any, value: Any, type_annotation: type
    ) -> tuple[String, Any]:
        """
        Get the value for file selection.

        Args:
            dataset (Any): The dataset object.
            value (Any): The current value.
            type_annotation (type): The expected type of the value.

        Returns:
            Tuple[String, Any]: Tuple containing the String object and the current \
                value.
        """
        if dataset is None:
            return String(tuple()), value

        available_files = _scan_files(dataset["path"])
        if self.add_none is True:
            if self.prefer_none:
                available_files.insert(0, "None")
            else:
                available_files.insert(len(available_files), "None")

        if isinstance(value, str):
            value = [value]

        value = DatasetValue._compute_current_values(
            value, available_files, self.prefer_with
        )

        return (
            String(
                tuple(
                    zip(
                        available_files,
                        strip_common_prefix(available_files, ignore_set={"None"}),
                        strict=False,
                    )
                )
            ),
            value if type_annotation == tuple[str, ...] else value[0],
        )


@dataclass
class Columns(DatasetValue):
    """
    Represents a selection of columns from a dataset.

    Used to select a column from a dataset for e.g. `prompt_column`.

    Attributes:
        add_none (bool): Whether to add a "None" option.
        prefer_with (Optional[Callable[[str], bool]]): Function to determine preferred \
            values.
    """

    add_none: bool = False
    prefer_with: Callable[[str], bool] | None = None

    def get_value(
        self, dataset: Any, value: Any, type_annotation: type
    ) -> tuple[String, Any]:
        if dataset is None:
            return String(tuple()), value

        try:
            columns = list(dataset["dataframe"].columns)
        except KeyError:
            columns = []

        if self.add_none is True:
            columns.insert(0, "None")

        if isinstance(value, str):
            value = [value]
        if value is None:
            value = [columns[0]]

        value = DatasetValue._compute_current_values(value, columns, self.prefer_with)

        return (
            String(tuple(columns)),
            value if type_annotation == tuple[str, ...] else value[0],
        )
