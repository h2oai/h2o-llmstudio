"""Tiny persistent key-value store backed by SQLite + pickle.

This module replaces the previously used `diskcache.Cache` for the small
charts-cache use case. https://github.com/grantjenks/python-diskcache.
Only the subset of the API actually used by LLM Studio
is implemented:

    with Cache(directory) as cache:
        cache[key] = value
        value = cache[key]
        if key in cache:
            ...
        for key in cache:
            ...
        value = cache.get(key, default=None)

Values are pickled, so any picklable Python object can be stored. The store is
safe for a single writer and concurrent readers across processes (SQLite is
opened in WAL mode), which matches the original usage pattern: the training
process writes charts, the app process reads them.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from typing import Any, Iterator

DB_FILENAME = "kv.db"


class Cache:
    """Minimal persistent key-value store backed by SQLite.

    Mirrors the small subset of the ``diskcache.Cache`` API used in this
    project. The cache directory is created on demand. The underlying SQLite
    database lives at ``<directory>/kv.db``.
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._db_path = os.path.join(directory, DB_FILENAME)
        # check_same_thread=False mirrors diskcache, which allowed access from
        # multiple threads within the same process. Concurrency between
        # processes is handled by SQLite's WAL mode below.
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self._db_path, check_same_thread=False, timeout=30.0
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            "  key TEXT PRIMARY KEY,"
            "  value BLOB NOT NULL"
            ")"
        )
        self._conn.commit()

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
                self._conn = None

    # -- mapping API -------------------------------------------------------
    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Cache is closed")
        return self._conn

    def __setitem__(self, key: str, value: Any) -> None:
        conn = self._require_conn()
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        conn.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, sqlite3.Binary(blob)),
        )
        conn.commit()

    def __getitem__(self, key: str) -> Any:
        conn = self._require_conn()
        row = conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return pickle.loads(row[0])

    def __contains__(self, key: object) -> bool:
        conn = self._require_conn()
        row = conn.execute("SELECT 1 FROM kv WHERE key = ?", (key,)).fetchone()
        return row is not None

    def __iter__(self) -> Iterator[str]:
        conn = self._require_conn()
        # Materialize so callers can mutate the cache while iterating.
        rows = conn.execute("SELECT key FROM kv ORDER BY key").fetchall()
        return iter(row[0] for row in rows)

    def __len__(self) -> int:
        conn = self._require_conn()
        row = conn.execute("SELECT COUNT(*) FROM kv").fetchone()
        return int(row[0])

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
