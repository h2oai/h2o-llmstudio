import logging
import os
from typing import Dict, List

import torch

from llm_studio.app_utils.config import default_cfg
from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src.utils.export_utils import get_size_str

logger = logging.getLogger(__name__)

__all__ = ["check_config_for_errors"]


def check_config_for_errors(cfg: DefaultConfigProblemBase) -> dict:
    """
    Checks the configuration for consistency.
        Parameters:
    - cfg (DefaultConfigProblemBase):
    The config object to be checked.

    Returns:
    A dictionary with two keys:
    - "title": A list of error titles.
    - "message": A list of error messages.
    """
    errors = check_for_common_errors(cfg)
    problem_type_errors = cfg.check()
    errors["title"].extend(problem_type_errors["title"])
    errors["message"].extend(problem_type_errors["message"])
    return errors


def check_for_common_errors(cfg: DefaultConfigProblemBase) -> dict:
    errors: Dict[str, List] = {"title": [], "message": []}
    if not len(cfg.environment.gpus) > 0:
        errors["title"] += ["No GPU selected"]
        errors["message"] += [
            "Please select at least one GPU to start the experiment! "
        ]

    if len(cfg.environment.gpus) > torch.cuda.device_count():
        errors["title"] += ["More GPUs selected than available"]
        errors["message"] += [
            f"There are {cfg.environment.gpus} GPUs selected but only "
            f"{torch.cuda.device_count()} GPUs available."
            "This error can happen when you start from an experiment configuration "
            "that was created on a different machine. Please deselect all GPUs and "
            "select the GPUs you want to use again. "
        ]

    stats = os.statvfs(".")
    available_size = stats.f_frsize * stats.f_bavail
    if available_size < default_cfg.min_experiment_disk_space:
        errors["title"] += ["Not enough disk space."]
        errors["message"] += [
            f"Not enough disk space. Available space is {get_size_str(available_size)}."
            f" Required space is "
            f"{get_size_str(default_cfg.min_experiment_disk_space)}. "
            "Experiment has not started. "
            "Please ensure that you have enough disk space before "
            "starting the experiment."
        ]

    # see create_nlp_backbone
    if (
        cfg.architecture.backbone_dtype in ["int4", "int8"]
        and not cfg.architecture.pretrained
    ):
        errors["title"] += ["Quantization without pretrained weights."]
        errors["message"] += [
            "Quantization is only supported for pretrained models. "
            "Please enable pretrained model or disable quantization."
        ]

    if (
        not cfg.training.lora
        and cfg.architecture.backbone_dtype not in ["bfloat16", "float32"]
        and cfg.training.epochs > 0
    ):
        errors["title"] += [f"Pure {cfg.architecture.backbone_dtype} training."]
        errors["message"] += [
            f"When not using LORA, {cfg.architecture.backbone_dtype} training will "
            "likely lead to unstable training. "
            "Please use LORA or set Backbone Dtype to bfloat16 or float32."
        ]

    if cfg.environment.use_deepspeed and cfg.architecture.backbone_dtype in [
        "int8",
        "int4",
    ]:
        errors["title"] += ["Deepspeed does not support quantization."]
        errors["message"] += [
            "Deepspeed do not support backbone type "
            f"{cfg.architecture.backbone_dtype}. "
            "Please set backbone type to float16 or bfloat16 for using deepspeed."
        ]
    if cfg.environment.use_deepspeed and len(cfg.environment.gpus) < 2:
        errors["title"] += ["Deepspeed not supported for single GPU."]
        errors["message"] += [
            "Deepspeed does not support single GPU training. "
            "Please select more than one GPU or disable deepspeed."
        ]
    return errors
