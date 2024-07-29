from typing import Iterable, List, Optional


class Order:
    """
    A list-like structure to specify the order of items in a dictionary.
    The main characteristics are:
        - Append and insert only. Cannot remove elements. This is not strictly required
            by the use-case but probably good practice.
        - Elements must be unique. Inserting an element which is already in the list
            will throw an error.

    Primarily useful for specifying the order in which UI elements
    should be shown in Wave.
    """

    def __init__(self, keys: Optional[List[str]] = None):
        if keys is not None:
            self._list = list(keys)
        else:
            self._list = list()

    def _unique_guard(self, *keys: str):
        for key in keys:
            if key in self._list:
                raise ValueError(f"`{key}` is already in the list!")

    def append(self, key: str):
        """
        Append a key at the end of the list:

        Args:
            key: String to append.

        Raises:
            - `ValueError` if the key is already in the list.
        """

        self._unique_guard(key)

        self._list.append(key)

    def extend(self, keys: Iterable[str]):
        """
        Extend the list by multiple keys.

        Args:
            keys: Iterable of keys.

        Raises:
            - `ValueError` if one or more key is already in the list.
        """

        self._unique_guard(*keys)

        self._list.extend(keys)

    def insert(
        self, *keys: str, before: Optional[str] = None, after: Optional[str] = None
    ):
        """
        Insert one or more keys. Either `before` or `after`, but not both, must be set
        to determine position.

        Args:
            keys: One more keys to insert.
            after: A key immediately after which the `keys` will be inserted.
            before: A key immediately before which the `keys` are inserted.

        Raises:
            - `ValueError` if one or more key is already in the list.
            - `ValueError` if `before` / `after` does not exist in the list.
            - `ValueError` if an invalid combination of arguments is set.
        """

        self._unique_guard(*keys)

        if before is None and after is None:
            raise ValueError("Either `before` or `after` must be set.")

        if before and after:
            raise ValueError("Can't set `before` and `after` at the same time.")

        if before is not None:
            for key in keys:
                self._list.insert(self._list.index(before), key)

        if after is not None:
            for key in keys:
                self._list.insert(self._list.index(after) + 1, key)

    def __getitem__(self, idx):
        return self._list[idx]

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)
