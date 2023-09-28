import logging
from typing import List, Any

import torch

from llm_studio.python_configs.base import DefaultConfigProblemBase
import llm_studio.python_configs.text_causal_language_modeling_config as text_causal_language_modeling_config
import llm_studio.python_configs.text_rlhf_language_modeling_config as text_rlhf_language_modeling_config
import llm_studio.python_configs.text_sequence_to_sequence_modeling_config as text_sequence_to_sequence_modeling_config

logger = logging.getLogger(__name__)


def check_text_nlp_causal_model_cfg(
    cfg: text_causal_language_modeling_config.ConfigProblemBase,
) -> dict:
    return {}


def check_text_rlhf_language_modeling_config(
    cfg: text_rlhf_language_modeling_config.ConfigProblemBase,
) -> dict:
    return {}


def check_text_sequence_to_sequence_modeling_config(
    cfg: text_sequence_to_sequence_modeling_config.ConfigProblemBase,
) -> dict:
    return {}


class ConfigCheckFactory:
    """ConfigUpdater factory."""

    _config_checks = {
        "text_causal_language_modeling_config": check_text_nlp_causal_model_cfg,
        "text_rlhf_language_modeling_config": check_text_rlhf_language_modeling_config,
        "text_sequence_to_sequence_modeling_config": check_text_sequence_to_sequence_modeling_config,
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._config_checks.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to ConfigUpdater.
        Args:
            name: problem type name
        Returns:
            A class to build the ConfigUpdater
        """
        return cls._config_checks.get(name)


def check_cfg_for_conflicts(cfg: DefaultConfigProblemBase) -> dict:
    """
    Checks the config for conflicts and returns a dict with the issues.
    """
    cfg_issues = {}
    if len(cfg.environment.gpus) == 0:
        title = "No GPU selected."
        text = "Please select at least one GPU to start the experiment!"
        cfg_issues = {"title": title, "text": text}
    elif len(cfg.environment.gpus) > torch.cuda.device_count():
        title = "Too many GPUs selected."
        text = (
            "Please deselect all GPU and assign the GPUs you want to use! "
            "This issue can happen if you start from a config that was using more GPUs than you have available."
        )
        cfg_issues = {"title": title, "text": text}

    cfg_checker = ConfigCheckFactory.get(cfg.problem_type)
    if cfg_checker is not None:
        cfg_issues.update(cfg_checker(cfg))
    return cfg_issues
