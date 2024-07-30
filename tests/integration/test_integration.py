import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from transformers.testing_utils import execute_subprocess_async

from llm_studio.app_utils.default_datasets import (
    prepare_default_dataset_causal_language_modeling,
)


def get_experiment_status(path: str) -> str:
    """Get status information from experiment.

    Args:
        path: path to experiment folder
    Returns:
        experiment status
    """

    try:
        flag_json_path = os.path.join(path, "flags.json")
        if not os.path.exists(flag_json_path):
            return "none"
        with open(flag_json_path) as file:
            flags = json.load(file)
            status = flags.get("status", "none")
        return status
    except Exception:
        return "none"


@pytest.mark.parametrize(
    "config_name",
    [
        "test_causal_language_modeling_oasst_cfg",
        "test_sequence_to_sequence_modeling_oasst_cfg",
    ],
)
@pytest.mark.parametrize(
    "metric",
    [
        "Perplexity",
        "BLEU",
    ],
)
def test_oasst_training_gpu(tmp_path, config_name, metric):
    run_oasst(tmp_path, config_name, metric)


@pytest.mark.parametrize(
    "settings",
    [
        ["AUC", "test_causal_binary_classification_modeling_cfg"],
        ["LogLoss", "test_causal_multiclass_classification_modeling_cfg"],
    ],
)
def test_oasst_classification_training_gpu(tmp_path, settings):
    metric, config_name = settings
    run_oasst(
        tmp_path,
        config_name=config_name,
        metric=metric,
    )


@pytest.mark.parametrize(
    "settings",
    [
        ["MSE", "test_causal_regression_modeling_cfg"],
    ],
)
def test_oasst_regression_training_gpu(tmp_path, settings):
    metric, config_name = settings
    run_oasst(
        tmp_path,
        config_name=config_name,
        metric=metric,
    )


@pytest.mark.parametrize(
    "settings",
    [
        ["MSE", "test_causal_regression_modeling_cpu_cfg"],
    ],
)
def test_oasst_regression_training_cpu(tmp_path, settings):
    metric, config_name = settings
    run_oasst(
        tmp_path,
        config_name=config_name,
        metric=metric,
    )


@pytest.mark.parametrize(
    "settings",
    [
        ["AUC", "test_causal_binary_classification_modeling_cpu_cfg"],
        ["LogLoss", "test_causal_multiclass_classification_modeling_cpu_cfg"],
    ],
)
def test_oasst_classification_training_cpu(tmp_path, settings):
    metric, config_name = settings
    run_oasst(
        tmp_path,
        config_name=config_name,
        metric=metric,
    )


@pytest.mark.parametrize(
    "config_name",
    [
        "test_causal_language_modeling_oasst_cpu_cfg",
        "test_sequence_to_sequence_modeling_oasst_cpu_cfg",
    ],
)
@pytest.mark.parametrize(
    "metric",
    [
        "Perplexity",
        "BLEU",
    ],
)
def test_oasst_training_cpu(tmp_path, config_name, metric):
    run_oasst(tmp_path, config_name, metric)


def run_oasst(tmp_path, config_name, metric):
    """
    Test training on OASST dataset.

    Pytest keeps around the last 3 test runs in the tmp_path fixture.
    """
    prepare_default_dataset_causal_language_modeling(tmp_path)
    train_path = os.path.join(tmp_path, "train_full.pq")
    # create dummy labels for classification problem type,
    # unused for other problem types
    df = pd.read_parquet(train_path)
    df["multiclass_label"] = np.random.choice(["0", "1", "2"], size=len(df))
    df["binary_label"] = np.random.choice(["0", "1"], size=len(df))
    df["regression_label"] = np.random.uniform(0, 1, size=len(df))
    df.to_parquet(train_path)

    with open(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), f"{config_name}.yaml"
        ),
        "r",
    ) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # set paths and save in tmp folder
    cfg["dataset"]["train_dataframe"] = train_path
    cfg["output_directory"] = os.path.join(tmp_path, "output")
    # set metric
    cfg["prediction"]["metric"] = metric
    modifed_config_path = os.path.join(tmp_path, "cfg.yaml")
    with open(modifed_config_path, "w") as fp:
        yaml.dump(cfg, fp)

    # llm studio root directory.
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")
    )
    cmd = [
        f"{sys.executable}",
        os.path.join(root_dir, "train.py"),
        "-Y",
        f"{modifed_config_path}",
    ]
    execute_subprocess_async(cmd)
    assert os.path.exists(cfg["output_directory"])
    status = get_experiment_status(path=cfg["output_directory"])
    assert status == "finished"
    assert os.path.exists(os.path.join(cfg["output_directory"], "charts.db"))
    assert os.path.exists(os.path.join(cfg["output_directory"], "checkpoint.pth"))
    assert os.path.exists(os.path.join(cfg["output_directory"], "logs.log"))
    assert os.path.exists(
        os.path.join(cfg["output_directory"], "validation_predictions.csv")
    )
