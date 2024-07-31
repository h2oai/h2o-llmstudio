from collections import defaultdict
from typing import DefaultDict, List, Set, Union

from pydantic.dataclasses import dataclass


@dataclass
class Dependency:
    """
    Represents a dependency with a key, value, and set condition.

    Attributes:
        key (str): The key of the dependency (parent).
        value (Union[str, bool, int, None]): The value of the dependency to look for. \
            None for empty condition (dependency only needs to exist).
        is_set (bool): Whether the value of the dependency should be set (True) or not \
            set (False).
    """

    key: str
    value: Union[str, bool, int, None]
    is_set: bool

    def check(self, dependency_values: List[str]) -> bool:
        """
        Check if dependency is satisfied

        Args:
            dependency_values (List[str]): List of dependency values to check against.

        Returns:
            bool: True if the dependency is satisfied, False otherwise.
        """

        if self.value is None and self.is_set and len(dependency_values):
            return False
        elif self.value is None and not self.is_set and not len(dependency_values):
            return False
        elif self.is_set and self.value not in dependency_values:
            return False
        elif (
            not self.is_set
            and len([v for v in dependency_values if v != self.value]) == 0
        ):
            return False
        return True


class Nesting:
    """
    A tree-like structure to specify nested dependencies of type `Dependency`.

    This class maps dependencies of keys requiring any number dependencies of \
        type `Dependency`. It is primarily useful for specifying nested dependencies \
        of UI elements shown in Wave.

    Attributes:
        dependencies (DefaultDict[str, List[Dependency]]): A dictionary mapping keys \
            to their dependencies of type `Dependency`.
        triggers (Set[str]): A set of all dependency keys that can trigger changes.
    """

    def __init__(self) -> None:
        self.dependencies: DefaultDict[str, List[Dependency]] = defaultdict(list)
        self.triggers: Set[str] = set()

    def add(self, keys: List[str], dependencies: List[Dependency]) -> None:
        """
        Append dependencies of type `Dependency` for given keys.

        Args:
            keys (List[str]): Keys to add dependencies for.
            dependencies (List[Dependency]): The Dependencys to depend on.

        Raises:
            ValueError: If the input keys are not unique.
        """

        if len(set(keys)) != len(keys):
            raise ValueError("Nesting keys must be unique.")

        for dependency in dependencies:
            self.triggers.add(dependency.key)
            for key in set(keys):
                self.dependencies[key].append(dependency)
