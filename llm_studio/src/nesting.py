from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Optional, Set, Union


@dataclass
class Dependency:
    """Dependency class.

    Args:
        key: key of the dependency
        value: required value of the dependency, None for empty condition
        is_set: whether the dependency should be set, or not set
    """

    key: str
    value: Union[str, bool, int, None] = True
    is_set: bool = True

    def check(self, dependency_values: Optional[List[str]]) -> bool:
        """
        Check if dependency is satisfied

        Args:
            dependency_values: list of dependency values

        Returns:
            True if the dependency is satisfied, False otherwise
        """

        if dependency_values is None:
            dependency_values = []

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
    A tree-like structure to specify nested dependencies of key-value pairs
    In detail it maps dependencies of key requiring any number of key:value pairs

    Primarily useful for specifying nested dependencies of UI elements shown in Wave.
    """

    def __init__(self):
        self.dependencies: DefaultDict[str, List[Dependency]] = defaultdict(list)
        self.triggers: Set[str] = set()

    def add(self, keys: List[str], dependencies: List[Dependency]):
        """
        Append dependencies (key:value) for a given key

        Args:
            keys: keys to add dependencies for
            dependencies: key:value pairs to depend on
        """

        if len(set(keys)) != len(keys):
            raise ValueError("Nesting keys must be unique.")

        for dependency in dependencies:
            self.triggers.add(dependency.key)
            for key in set(keys):
                self.dependencies[key].append(dependency)
