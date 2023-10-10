import logging
import os
from typing import Any, Dict, List

import torch

import llm_studio.python_configs.text_causal_language_modeling_config as text_causal_language_modeling_config
import llm_studio.python_configs.text_rlhf_language_modeling_config as text_rlhf_language_modeling_config
import llm_studio.python_configs.text_sequence_to_sequence_modeling_config as text_sequence_to_sequence_modeling_config
from llm_studio.app_utils.config import default_cfg
from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src.utils.export_utils import get_size_str

logger = logging.getLogger(__name__)

__all__ = ["check_config_for_consistency"]


def check_config_for_consistency(cfg: DefaultConfigProblemBase) -> dict:
    """
    Checks the configuration for consistency.
        Parameters:
    - cfg (AudioRegressionConfigProblemBase): The audio regression config object to be checked.

    Returns:
    - dict: The dictionary containing the result of the audio classification config check.

    """
    errors: Dict[str, List] = {"title": [], "message": []}
    common_errors = check_for_common_errors(cfg)
    extend_errors(errors, common_errors)
    specific_errors = check_for_problem_type_specific_errors(cfg)
    extend_errors(errors, specific_errors)
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

    if cfg.training.save_best_checkpoint and cfg.training.train_validation_data:
        errors["title"] += ["Save Best Checkpoint incompatible settings."]
        errors["message"] += [
            "Save Best Checkpoint is not compatible with "
            "Train Validation Data. "
            "Please set Save Best Checkpoint to False or disable "
            "Train Validation Data. "
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
            "starting the experiment. "
        ]

    # see create_nlp_backbone
    if (
        cfg.architecture.backbone_dtype in ["int4", "int8"]
        and not cfg.architecture.pretrained
    ):
        errors["title"] += ["Quantization without pretrained weights."]
        errors["message"] += [
            "Quantization is only supported for pretrained models. "
            "Please enable pretrained model or disable quantization. "
        ]

    if not cfg.training.lora and cfg.architecture.backbone_dtype != "float32":
        if cfg.environment.mixed_precision:
            errors["title"] += ["Mixed precision disabled."]
            errors["message"] += [
                "Mixed precision is disabled as dtype is not set to float32. "
                "Please enable mixed precision or set dtype to float32. "
            ]
        if cfg.architecture.backbone_dtype != "bfloat16":
            errors["title"] += ["Pure float16 or int8 training."]
            errors["message"] += [
                "Pure float16 or int8 training will "
                "likely lead to unstable training without adapters. "
                "Please use adapters or set dtype to float32. "
            ]

    return errors


def check_for_problem_type_specific_errors(cfg: DefaultConfigProblemBase) -> dict:
    """
    Check for errors for the specific problem type.

    Parameters:
        cfg (ConfigBase): The configuration object to be checked.

    Returns:
        dict: A dictionary containing the title and message of the error, if any.

    """
    config_check = ConfigCheckFactory.get(cfg.problem_type)
    if config_check:
        return config_check(cfg)
    return {"title": [], "message": []}


def extend_errors(original_errors: dict, new_errors: dict) -> None:
    """
    Extends the errors dictionary with new error values.

    :param original_errors: The original errors dictionary.
    :type original_errors: dict
    :param new_errors: The new errors dictionary.
    :type new_errors: dict
    :return: None
    """
    original_errors["title"].extend(new_errors.get("title", []))
    original_errors["message"].extend(new_errors.get("message", []))


def check_text_nlp_causal_model_cfg(
    cfg: text_causal_language_modeling_config.ConfigProblemBase,
) -> dict:
    errors: Dict[str, List] = {"title": [], "message": []}
    return errors


def check_text_rlhf_language_modeling_config(
    cfg: text_rlhf_language_modeling_config.ConfigProblemBase,
) -> dict:
    errors: Dict[str, List] = {"title": [], "message": []}
    if not cfg.training.lora:
        errors["title"] += ["LoRA must be True for RLHF"]
        errors["message"] += [
            "LoRA must be True for RLHF. "
            "Please set LoRA to True or change the problem type. "
        ]

    # see CustomDataset for RLHF
    if cfg.dataset.system_column != "None":
        errors["title"] += ["RLHF is not compatible with system column."]
        errors["message"] += [
            "RLHF is not compatible with system column. "
            "Please set system column to None or change the problem type. "
        ]
    if cfg.dataset.limit_chained_samples:
        errors["title"] += ["RLHF is not compatible with limit_chained_samples."]
        errors["message"] += [
            "RLHF is not compatible with limit_chained_samples. "
            "Please set limit_chained_samples to False or change the problem type. "
        ]
    if not cfg.dataset.mask_prompt_labels:
        errors["title"] += ["RLHF is not compatible with mask_prompt_labels."]
        errors["message"] += [
            "RLHF is not compatible with mask_prompt_labels. "
            "Please set mask_prompt_labels to True or change the problem type. "
        ]
    return errors


def check_text_sequence_to_sequence_modeling_config(
    cfg: text_sequence_to_sequence_modeling_config.ConfigProblemBase,
) -> dict:
    errors: Dict[str, List] = {"title": [], "message": []}
    return errors


class ConfigCheckFactory:
    """ConfigUpdater factory."""

    _config_checks = {
        "text_causal_language_modeling_config": check_text_nlp_causal_model_cfg,
        "text_rlhf_language_modeling_config": check_text_rlhf_language_modeling_config,
        "text_sequence_to_sequence_modeling_config": check_text_sequence_to_sequence_modeling_config,
    }

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to ConfigUpdater.
        Args:
            name: problem type name
        Returns:
            A class to build the ConfigUpdater
        """
        return cls._config_checks.get(name)
