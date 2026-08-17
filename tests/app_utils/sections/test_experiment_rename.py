import asyncio
import os
from types import SimpleNamespace

import pandas as pd
import pytest

from llm_studio.app_utils.sections.experiment import experiment_rename_action
from llm_studio.src.utils.disk_kv import Cache

CHARTS_CACHE = "charts_cache"


def _run_async(coro):
    """Run a coroutine without clearing pytest's current event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_experiment(tmp_path, name: str):
    """Create an experiment directory with charts and a populated charts cache."""
    path = os.path.join(str(tmp_path), "output", "user", name)
    os.makedirs(path)

    df_paths = {}
    for mode in ["batch", "validation"]:
        parquet_path = os.path.join(path, f"{mode}_viz.parquet")
        pd.DataFrame({"Predicted Text": ["a", "b"]}).to_parquet(parquet_path)
        df_paths[mode] = parquet_path

    with Cache(os.path.join(path, CHARTS_CACHE)) as cache:
        # `df` values are absolute paths embedding the experiment name, while
        # `image`/`html` hold the data inline and must not be rewritten.
        cache["df"] = df_paths
        cache["image"] = {"train_data": "base64-encoded-png"}
        cache["html"] = {"train_data": f"<div>{name}</div>"}

    experiment = SimpleNamespace(id=1, name=name, path=path)
    return experiment


def _make_q():
    renames = []
    app_db = SimpleNamespace(
        rename_experiment=lambda *args: renames.append(args),
    )
    return SimpleNamespace(client=SimpleNamespace(app_db=app_db)), renames


def test_rename_persists_rewritten_chart_paths(tmp_path) -> None:
    """Renaming must write the new chart paths back through the cache.

    Regression test for #1088: the rewritten paths were assigned to a local
    dict, so they were discarded and the Insights tabs raised FileNotFoundError
    on the renamed experiment.
    """
    experiment = _make_experiment(tmp_path, "old-name")
    q, renames = _make_q()
    new_path = experiment.path.replace("old-name", "new-name")

    _run_async(experiment_rename_action(q, experiment, "new-name"))

    with Cache(os.path.join(new_path, CHARTS_CACHE)) as cache:
        charts = {key: cache[key] for key in cache}

    for mode in ["batch", "validation"]:
        expected = os.path.join(new_path, f"{mode}_viz.parquet")
        assert charts["df"][mode] == expected
        # the path the Insights tab reads must actually resolve on disk
        assert os.path.exists(charts["df"][mode])
        pd.read_parquet(charts["df"][mode])

    # only the `df` encoding stores paths, the others are left alone
    assert charts["image"] == {"train_data": "base64-encoded-png"}
    assert charts["html"] == {"train_data": "<div>old-name</div>"}

    assert renames == [(1, "new-name", new_path)]


def test_rename_without_charts_does_not_raise(tmp_path) -> None:
    """An experiment that has not produced charts yet is still renameable."""
    experiment = _make_experiment(tmp_path, "old-name")
    new_path = experiment.path.replace("old-name", "new-name")

    # mimic a queued experiment: the cache exists but holds no plot encodings
    db_path = os.path.join(experiment.path, CHARTS_CACHE, "cache.db")
    os.remove(db_path)
    with Cache(os.path.join(experiment.path, CHARTS_CACHE)) as cache:
        cache["cfg"] = {"experiment_name": "old-name"}

    q, renames = _make_q()
    _run_async(experiment_rename_action(q, experiment, "new-name"))

    with Cache(os.path.join(new_path, CHARTS_CACHE)) as cache:
        assert "df" not in cache
        assert cache["cfg"] == {"experiment_name": "old-name"}

    assert renames == [(1, "new-name", new_path)]


def test_rename_to_same_name_is_a_noop(tmp_path) -> None:
    experiment = _make_experiment(tmp_path, "old-name")
    q, renames = _make_q()

    _run_async(experiment_rename_action(q, experiment, "old-name"))

    assert os.path.exists(experiment.path)
    assert renames == []


@pytest.mark.parametrize("new_name", ["new-name", "old-name-2", "renamed"])
def test_rename_charts_resolve_for_various_names(tmp_path, new_name) -> None:
    experiment = _make_experiment(tmp_path, "old-name")
    q, _ = _make_q()
    new_path = experiment.path.replace("old-name", new_name)

    _run_async(experiment_rename_action(q, experiment, new_name))

    with Cache(os.path.join(new_path, CHARTS_CACHE)) as cache:
        for path in cache["df"].values():
            assert os.path.exists(path)
