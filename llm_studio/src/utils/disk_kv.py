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

The on-disk filename ``cache.db`` is kept identical to the one ``diskcache``
used, so the cache directory layout stays unchanged across the upgrade. Our
``kv`` table does not collide with diskcache's ``Cache``/``Settings`` tables.
If this class opens a database written by the legacy diskcache backend, a
one-shot migration copies the relevant entries into the new ``kv`` table so
that charts of pre-upgrade experiments remain readable.
"""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
import time
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DB_FILENAME = "cache.db"

# Diskcache value-mode constant for pickled entries (see diskcache/core.py).
# This is the only mode LLM Studio's LocalLogger ever wrote -- all values
# were Python dicts, all stored via pickle.
_DC_MODE_PICKLE = 4


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
        self._migrate_legacy_diskcache_if_needed()

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

    # -- legacy diskcache migration ---------------------------------------
    # TODO: remove everthing below in v1.16 or later
    def _migrate_legacy_diskcache_if_needed(self) -> None:
        """Copy entries from a legacy ``diskcache`` ``Cache`` table into ``kv``.

        Runs on open. No-op if ``kv`` already has any rows or if no legacy
        ``Cache`` table is present, so it is idempotent and cheap on the hot
        path. The old table (and any external ``*.val`` files) are left in
        place to allow rollback to a pre-upgrade version of LLM Studio if
        ever needed.

        Note on pickle: this code unpickles legacy values. That is acceptable
        because the cache directory is the same trusted experiment output
        directory the previous diskcache-based code already unpickled from --
        we do not introduce any new trust boundary here.
        """
        conn = self._require_conn()

        # Already migrated (or fresh-written by this class)?
        if conn.execute("SELECT 1 FROM kv LIMIT 1").fetchone() is not None:
            return

        # Legacy table present?
        has_legacy = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='Cache' LIMIT 1"
        ).fetchone()
        if not has_legacy:
            return

        logger.warning(
            "Migrating diskcache to new Cache. This will be deprecated in v1.16"
        )

        try:
            rows = conn.execute(
                "SELECT key, raw, mode, filename, value, expire_time FROM Cache"
            ).fetchall()
        except sqlite3.Error as exc:
            logger.warning(
                "Could not read legacy diskcache table at %s: %s",
                self._db_path,
                exc,
            )
            return

        now = time.time()
        migrated = 0
        skipped = 0
        for key, raw, mode, filename, value, expire_time in rows:
            if expire_time is not None and expire_time < now:
                continue
            # LLM Studio's LocalLogger only ever wrote (str key, dict value)
            # entries, so we only migrate raw=1 string keys with MODE_PICKLE
            # values. Anything else is skipped defensively.
            if not raw or not isinstance(key, str) or mode != _DC_MODE_PICKLE:
                skipped += 1
                continue
            try:
                decoded_value = _decode_legacy_pickle_value(
                    filename, value, self.directory
                )
            except Exception as exc:
                logger.warning(
                    "Skipping unreadable legacy cache entry %r in %s: %s",
                    key,
                    self._db_path,
                    exc,
                )
                skipped += 1
                continue
            blob = pickle.dumps(decoded_value, protocol=pickle.HIGHEST_PROTOCOL)
            conn.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, sqlite3.Binary(blob)),
            )
            migrated += 1

        conn.commit()
        if migrated or skipped:
            logger.info(
                "Migrated %d entr%s from legacy diskcache at %s (skipped %d).",
                migrated,
                "y" if migrated == 1 else "ies",
                self._db_path,
                skipped,
            )


def _decode_legacy_pickle_value(
    filename: str | None, value: Any, directory: str
) -> Any:
    """Decode a ``MODE_PICKLE`` value from a diskcache ``Cache`` row.

    The pickled bytes live either inline in the ``value`` column or in an
    external ``*.val`` file under the cache directory (diskcache spills
    values larger than 32 KiB by default).
    """
    if filename:
        full_path = _safe_join(directory, filename)
        with open(full_path, "rb") as fh:
            return pickle.load(fh)
    if isinstance(value, (bytes, memoryview)):
        return pickle.loads(bytes(value))
    raise ValueError("MODE_PICKLE row has neither inline blob nor filename")


def _safe_join(directory: str, relative: str) -> str:
    """Resolve ``directory/relative`` and refuse paths that escape ``directory``."""
    base = os.path.realpath(directory)
    target = os.path.realpath(os.path.join(base, relative))
    if not (target == base or target.startswith(base + os.sep)):
        raise ValueError(f"Refusing to read outside cache directory: {relative!r}")
    return target
