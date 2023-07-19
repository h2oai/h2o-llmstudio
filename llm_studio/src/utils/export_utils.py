import json
import logging
import os
import zipfile
from typing import Optional

from llm_studio.src.utils.exceptions import LLMResourceException
from llm_studio.src.utils.utils import add_file_to_zip


def get_artifact_path_path(
    experiment_name: str, experiment_path: str, artifact_type: str
):
    """Get path to experiment artifact zipfile

    Args:
        experiment_name: name of the experiment
        experiment_path: path containing experiment related files
        artifact_type: type of the artifact

    Returns:
        Path to the zip file with experiment artifact
    """

    return os.path.join(experiment_path, f"{artifact_type}_{experiment_name}.zip")


def get_predictions_path(experiment_name: str, experiment_path: str):
    """Get path to experiment predictions"""

    return get_artifact_path_path(experiment_name, experiment_path, "preds")


def get_logs_path(experiment_name: str, experiment_path: str):
    """Get path to experiment logs"""

    return get_artifact_path_path(experiment_name, experiment_path, "logs")


def get_model_path(experiment_name: str, experiment_path: str):
    """Get path to experiment model"""

    return get_artifact_path_path(experiment_name, experiment_path, "model")


def check_available_space(output_folder: str, min_disk_space: Optional[float]):
    if not min_disk_space:
        return True

    stats = os.statvfs(output_folder)
    available_size = stats.f_frsize * stats.f_bavail

    if available_size < min_disk_space:
        error = (
            f"Not enough disk space. Available space is {get_size_str(available_size)}."
            f" Required space is {get_size_str(min_disk_space)}."
        )
        raise LLMResourceException(error)


def save_prediction_outputs(
    experiment_name: str,
    experiment_path: str,
):
    """Save experiment prediction

    Args:
        experiment_name: name of the experiment
        experiment_path: path containing experiment related files

    Returns:
        Path to the zip file with experiment predictions
    """

    zip_path = get_predictions_path(experiment_name, experiment_path)
    zf = zipfile.ZipFile(zip_path, "w")

    add_file_to_zip(zf=zf, path=f"{experiment_path}/validation_raw_predictions.pkl")
    add_file_to_zip(zf=zf, path=f"{experiment_path}/validation_predictions.csv")

    zf.close()
    return zip_path


def save_logs(experiment_name: str, experiment_path: str, logs: dict):
    """Save experiment logs

    Args:
        experiment_name: name of the experiment
        experiment_path: path containing experiment related files
        logs: dictionary with experiment charts

    Returns:
        Path to the zip file with experiment logs
    """

    cfg_path = os.path.join(experiment_path, "cfg.yaml")
    charts_path = f"{experiment_path}/charts_{experiment_name}.json"
    with open(charts_path, "w") as fp:
        json.dump(
            {k: v for k, v in logs.items() if k in ["meta", "train", "validation"]}, fp
        )

    zip_path = get_logs_path(experiment_name, experiment_path)
    zf = zipfile.ZipFile(zip_path, "w")
    zf.write(charts_path, os.path.basename(charts_path))
    zf.write(cfg_path, f"cfg_{experiment_name}.yaml")

    try:
        zf.write(
            f"{experiment_path}/logs.log",
            f"logs_{experiment_name}.log",
        )
    except FileNotFoundError:
        logging.warning("Log file is not available yet.")

    zf.close()

    return zip_path


def get_size_str(
    x, sig_figs=2, input_unit="B", output_unit="dynamic", show_unit=True
) -> str:
    """
    Convert a small input unit such as bytes to human readable format.

    Args:
        x: input value
        sig_figs: number of significant figures
        input_unit: input unit ("B", "KB", "MB", "GB", "TB"), default "B"
        output_unit: output unit ("B", "KB", "MB", "GB", "TB", "dynamic")
            default "dynamic"
        show_unit: whether to show the unit in the output string

    Returns:
        str: Human readable string
    """

    names = ["B", "KB", "MB", "GB", "TB"]
    names = names[names.index(input_unit) :]

    act_i = 0
    if output_unit == "dynamic":
        while x >= 1024 and act_i < len(names):
            x /= 1024
            act_i += 1
    else:
        target = names.index(output_unit)
        while act_i < target:
            x /= 1024
            act_i += 1

    ret_str = f"{str(round(x, sig_figs))}"
    if show_unit:
        ret_str += f" {names[act_i]}"

    return ret_str
