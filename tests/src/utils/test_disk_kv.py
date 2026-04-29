import os
import pickle
import sqlite3
import tempfile
import time

import numpy as np
import pytest

from llm_studio.src.utils.disk_kv import DB_FILENAME, Cache

# Diskcache value-mode constants, mirrored from disk_kv._DC_MODE_*.
_DC_MODE_RAW = 1
_DC_MODE_BINARY = 2
_DC_MODE_TEXT = 3
_DC_MODE_PICKLE = 4


def _create_legacy_diskcache_db(directory: str) -> sqlite3.Connection:
    """Create a minimal diskcache-compatible ``Cache`` table for tests.

    Mirrors the schema in ``diskcache.core`` (only the columns this project
    actually needs to read).
    """
    os.makedirs(directory, exist_ok=True)
    conn = sqlite3.connect(os.path.join(directory, DB_FILENAME))
    conn.execute(
        "CREATE TABLE Cache ("
        "  rowid INTEGER PRIMARY KEY,"
        "  key BLOB,"
        "  raw INTEGER,"
        "  store_time REAL,"
        "  expire_time REAL,"
        "  access_time REAL,"
        "  access_count INTEGER DEFAULT 0,"
        "  tag BLOB,"
        "  size INTEGER DEFAULT 0,"
        "  mode INTEGER DEFAULT 0,"
        "  filename TEXT,"
        "  value BLOB)"
    )
    return conn


def _insert_legacy_pickle_inline(
    conn: sqlite3.Connection,
    key: str,
    value: object,
    *,
    expire_time: float | None = None,
) -> None:
    blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    conn.execute(
        "INSERT INTO Cache(key, raw, mode, filename, value, expire_time) "
        "VALUES(?, 1, ?, NULL, ?, ?)",
        (key, _DC_MODE_PICKLE, sqlite3.Binary(blob), expire_time),
    )
    conn.commit()


def _insert_legacy_pickle_external(
    conn: sqlite3.Connection,
    directory: str,
    key: str,
    value: object,
    relative_filename: str,
) -> None:
    full_path = os.path.join(directory, relative_filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as fh:
        pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
    conn.execute(
        "INSERT INTO Cache(key, raw, mode, filename, value) VALUES(?, 1, ?, ?, NULL)",
        (key, _DC_MODE_PICKLE, relative_filename),
    )
    conn.commit()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_set_get(temp_dir):
    with Cache(temp_dir) as cache:
        cache["a"] = {"steps": [1, 2], "values": [0.1, 0.2]}
        assert cache["a"] == {"steps": [1, 2], "values": [0.1, 0.2]}


def test_overwrite(temp_dir):
    with Cache(temp_dir) as cache:
        cache["a"] = 1
        cache["a"] = 2
        assert cache["a"] == 2
        assert len(cache) == 1


def test_contains(temp_dir):
    with Cache(temp_dir) as cache:
        cache["a"] = 1
        assert "a" in cache
        assert "b" not in cache


def test_get_default(temp_dir):
    with Cache(temp_dir) as cache:
        assert cache.get("missing") is None
        assert cache.get("missing", 42) == 42
        cache["x"] = "value"
        assert cache.get("x") == "value"


def test_keyerror(temp_dir):
    with Cache(temp_dir) as cache:
        with pytest.raises(KeyError):
            _ = cache["nope"]


def test_iter_yields_keys(temp_dir):
    with Cache(temp_dir) as cache:
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3
        assert sorted(list(cache)) == ["a", "b", "c"]
        # The pattern used in the codebase: {key: cache.get(key) for key in cache}
        assert {key: cache.get(key) for key in cache} == {"a": 1, "b": 2, "c": 3}


def test_persistence_across_sessions(temp_dir):
    with Cache(temp_dir) as cache:
        cache["cfg"] = {"lr": 1e-4, "epochs": 3}
        cache["train"] = {"loss": {"steps": [1], "values": [0.5]}}
    # Re-open
    with Cache(temp_dir) as cache:
        assert cache["cfg"] == {"lr": 1e-4, "epochs": 3}
        assert cache["train"] == {"loss": {"steps": [1], "values": [0.5]}}


def test_creates_directory(tmp_path):
    target = tmp_path / "nested" / "charts_cache"
    with Cache(str(target)) as cache:
        cache["k"] = 1
    assert os.path.isfile(target / DB_FILENAME)


def test_stores_numpy_floats(temp_dir):
    # LocalLogger casts to float, but make sure picklable numpy types still work.
    with Cache(temp_dir) as cache:
        cache["x"] = float(np.float32(0.25))
        assert cache["x"] == 0.25


def test_use_after_close_raises(temp_dir):
    cache = Cache(temp_dir)
    cache["a"] = 1
    cache.close()
    with pytest.raises(RuntimeError):
        _ = cache["a"]


def test_concurrent_reader_sees_writes(temp_dir):
    # Mirrors usage: training process writes, app process reads.
    writer = Cache(temp_dir)
    reader = Cache(temp_dir)
    try:
        writer["train"] = {"loss": [0.5]}
        assert reader["train"] == {"loss": [0.5]}
        writer["train"] = {"loss": [0.5, 0.4]}
        assert reader["train"] == {"loss": [0.5, 0.4]}
    finally:
        writer.close()
        reader.close()


# -- legacy diskcache migration -------------------------------------------
def test_migrates_inline_pickle_entries(temp_dir):
    conn = _create_legacy_diskcache_db(temp_dir)
    _insert_legacy_pickle_inline(conn, "cfg", {"lr": 1e-4, "epochs": 3})
    _insert_legacy_pickle_inline(
        conn, "train", {"loss": {"steps": [1, 2], "values": [0.5, 0.4]}}
    )
    _insert_legacy_pickle_inline(conn, "val", {"acc": {"steps": [1], "values": [0.9]}})
    conn.close()

    with Cache(temp_dir) as cache:
        assert sorted(cache) == ["cfg", "train", "val"]
        assert cache["cfg"] == {"lr": 1e-4, "epochs": 3}
        assert cache["train"] == {"loss": {"steps": [1, 2], "values": [0.5, 0.4]}}
        assert cache["val"] == {"acc": {"steps": [1], "values": [0.9]}}


def test_migrates_external_pickle_files(temp_dir):
    conn = _create_legacy_diskcache_db(temp_dir)
    big_value = {"loss": {"steps": list(range(1000)), "values": [0.1] * 1000}}
    # diskcache stores large values in nested aa/bb/<hex>.val files
    _insert_legacy_pickle_external(
        conn, temp_dir, "train", big_value, os.path.join("aa", "bb", "deadbeef.val")
    )
    conn.close()

    with Cache(temp_dir) as cache:
        assert cache["train"] == big_value


def test_migration_skips_expired_entries(temp_dir):
    conn = _create_legacy_diskcache_db(temp_dir)
    _insert_legacy_pickle_inline(conn, "fresh", {"v": 1})
    _insert_legacy_pickle_inline(conn, "stale", {"v": 2}, expire_time=time.time() - 10)
    conn.close()

    with Cache(temp_dir) as cache:
        assert "fresh" in cache
        assert "stale" not in cache


def test_migration_is_idempotent(temp_dir):
    conn = _create_legacy_diskcache_db(temp_dir)
    _insert_legacy_pickle_inline(conn, "k", {"v": 1})
    conn.close()

    # First open performs migration.
    with Cache(temp_dir) as cache:
        assert cache["k"] == {"v": 1}
        # User updates value via the new code path.
        cache["k"] = {"v": 2}

    # Re-opening must not re-migrate and overwrite the user's update.
    with Cache(temp_dir) as cache:
        assert cache["k"] == {"v": 2}


def test_no_migration_when_no_legacy_table(temp_dir):
    # Fresh open (no legacy data) should not error and should be empty.
    with Cache(temp_dir) as cache:
        assert list(cache) == []
        cache["k"] = 1
        assert cache["k"] == 1


def test_migration_refuses_path_traversal(temp_dir, caplog):
    conn = _create_legacy_diskcache_db(temp_dir)
    # Malicious filename pointing outside the cache directory.
    conn.execute(
        "INSERT INTO Cache(key, raw, mode, filename, value) VALUES(?, 1, ?, ?, NULL)",
        ("evil", _DC_MODE_PICKLE, "../../../etc/passwd"),
    )
    conn.commit()
    conn.close()

    with caplog.at_level("WARNING"):
        with Cache(temp_dir) as cache:
            assert "evil" not in cache
    assert any("evil" in rec.message for rec in caplog.records)
