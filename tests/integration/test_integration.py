import json
import os
import sys

import pytest
import torch
import yaml
from transformers.testing_utils import execute_subprocess_async

from llm_studio.app_utils.utils import prepare_default_dataset

need_gpus = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only test")
has_no_gpus = pytest.mark.skipif(torch.cuda.is_available(), reason="CPU only test")


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


@need_gpus
@pytest.mark.parametrize(
    "config_name",
    [
        "test_causal_language_modeling_oasst_cfg",
        "test_sequence_to_sequence_modeling_oasst_cfg",
        "test_rlhf_language_modeling_oasst_cfg",
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


@has_no_gpus
@pytest.mark.parametrize(
    "config_name",
    [
        "test_causal_language_modeling_oasst_cpu_cfg",
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
    prepare_default_dataset(tmp_path)
    train_path = os.path.join(tmp_path, "train_full.pq")
    with os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"{config_name}.yaml", "r"
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
    cmd = [
        f"{sys.executable}",
        "train.py",
        "-Y",
        f"{modifed_config_path}",
    ]
    execute_subprocess_async(cmd)
    assert os.path.exists(cfg["output_directory"])
    status = get_experiment_status(path=cfg["output_directory"])
    assert status == "finished"
    assert os.path.exists(os.path.join(cfg["output_directory"], "adapter_model.bin"))
    assert os.path.exists(os.path.join(cfg["output_directory"], "charts.db"))
    assert os.path.exists(os.path.join(cfg["output_directory"], "checkpoint.pth"))
    assert os.path.exists(os.path.join(cfg["output_directory"], "logs.log"))
    assert os.path.exists(
        os.path.join(cfg["output_directory"], "validation_predictions.csv")
    )
