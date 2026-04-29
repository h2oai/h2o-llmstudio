import os
import tempfile

import numpy as np
import pytest

from llm_studio.src.utils.disk_kv import DB_FILENAME, Cache


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
