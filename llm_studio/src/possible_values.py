import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union

from llm_studio.src.nesting import Dependency


def _scan_dirs(dirname) -> List[str]:
    """Scans a directory for subfolders

    Args:
        dirname: directory name

    Returns:
        List of subfolders

    """

    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(_scan_dirs(dirname))
    subfolders = [x + "/" if x[-1] != "/" else x for x in subfolders]
    return subfolders


def _scan_files(
    dirname, extensions: Tuple[str, ...] = (".csv", ".pq", ".parquet", ".json")
) -> List[str]:
    """Scans a directory for files with given extension

    Args:
        dirname: directory name
        extensions: extensions to consider

    Returns:
        List of files

    """
    path_list = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(dirname)
        for filename in filenames
        if any(map(filename.__contains__, extensions))
        and not filename.startswith("__meta_info__")
    ]
    return path_list


def strip_prefix(paths: Sequence[str], ignore_set: Set[str] = set()) -> Tuple[str, ...]:
    """
    Strips the common prefix of all the given paths.

    Args:
        paths: the paths to strip
        ignore_set: set of path names to ignore when computing the prefix.

    Returns:
        List with the same length as `paths` without common prefixes.
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
    pass


@dataclass
class Number:
    min: Optional[float] = None
    max: Optional[float] = None
    step: Union[str, float] = 1.0


@dataclass
class String:
    # Each element of the tuple can be either:
    # - a tuple of (value, name)
    # - a string. In that case the same value will be used for name and value
    values: Any = None
    allow_custom: bool = False
    placeholder: Optional[str] = None


class DatasetValue:
    pass

    @abstractmethod
    def get_value(
        self, dataset: Any, value: Any, type_annotation: type, mode: str
    ) -> Tuple[String, Any]:
        pass

    @staticmethod
    def _compute_current_values(
        current_values: List[str],
        possible_values: List[str],
        prefer_with: Optional[Callable[[str], bool]] = None,
    ) -> List[str]:
        """
        Compute current values.

        Args:
            current_values: The preliminary current values.
            possible_values: All possible values.
            prefer_with: Function determining which values to prefer as default.

        Returns:
            A list
        """
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
class Directories(DatasetValue):
    add_none: Union[bool, Callable[[str], bool]] = False
    prefer_with: Optional[Callable[[str], bool]] = None
    prefer_none: bool = True

    def get_value(self, dataset, value, type_annotation, mode) -> Tuple[String, Any]:
        if dataset is None:
            return String(tuple()), value

        available_dirs = _scan_dirs(dataset["path"])

        if (isinstance(self.add_none, bool) and self.add_none) or (
            callable(self.add_none) and self.add_none(mode)
        ):
            if self.prefer_none:
                available_dirs.insert(0, "None")
            else:
                available_dirs.insert(len(available_dirs), "None")

        if isinstance(value, str):
            value = [value]

        value = DatasetValue._compute_current_values(
            value, available_dirs, self.prefer_with
        )

        return (
            String(
                tuple(
                    zip(
                        available_dirs,
                        strip_prefix(available_dirs, ignore_set={"None"}),
                    )
                )
            ),
            value if type_annotation == Tuple[str, ...] else value[0],
        )


@dataclass
class Files(DatasetValue):
    add_none: Union[bool, Callable[[str], bool]] = False
    prefer_with: Optional[Callable[[str], bool]] = None
    # For the case where no match found, whether to prioritize
    # selecting any file or selecting no file
    prefer_none: bool = True

    def get_value(self, dataset, value, type_annotation, mode) -> Tuple[String, Any]:
        if dataset is None:
            return String(tuple()), value

        available_files = _scan_files(dataset["path"])

        if (isinstance(self.add_none, bool) and self.add_none) or (
            callable(self.add_none) and self.add_none(mode)
        ):
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
                        strip_prefix(available_files, ignore_set={"None"}),
                    )
                )
            ),
            value if type_annotation == Tuple[str, ...] else value[0],
        )


@dataclass
class Columns(DatasetValue):
    add_none: Union[bool, Callable[[str], bool]] = False
    prefer_with: Optional[Callable[[str], bool]] = None

    def get_value(self, dataset, value, type_annotation, mode) -> Tuple[String, Any]:
        if dataset is None:
            return String(tuple()), value

        try:
            columns = list(dataset["dataframe"].columns)
        except KeyError:
            columns = []

        if (isinstance(self.add_none, bool) and self.add_none) or (
            callable(self.add_none) and self.add_none(mode)
        ):
            columns.insert(0, "None")

        if isinstance(value, str):
            value = [value]
        if value is None:
            value = [columns[0]]

        value = DatasetValue._compute_current_values(value, columns, self.prefer_with)

        return (
            String(tuple(columns)),
            value if type_annotation == Tuple[str, ...] else value[0],
        )


@dataclass
class ColumnValue(DatasetValue):
    column: str
    default: List[str]
    prefer_with: Optional[Callable[[str], bool]] = None
    dependency: Optional[Dependency] = None

    def get_value(self, dataset, value, type_annotation, mode) -> Tuple[String, Any]:
        if dataset is None:
            return String(tuple()), value

        try:
            df = dataset["dataframe"]
        except KeyError:
            df = None

        if df is not None:
            if self.dependency is not None and not self.dependency.check(
                [dataset[self.dependency.key]]
            ):
                values = self.default
            elif self.column in df:
                values = [str(v) for v in sorted(list(df[self.column].unique()))]
            else:
                values = self.default
        else:
            values = self.default

        value = DatasetValue._compute_current_values(value, values, self.prefer_with)

        return (
            String(tuple(values)),
            value if type_annotation == Tuple[str, ...] else value[0],
        )
