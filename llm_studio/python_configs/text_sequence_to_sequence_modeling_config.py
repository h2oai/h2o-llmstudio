import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch

import llm_studio.src.datasets.text_causal_language_modeling_ds
from llm_studio.python_configs.base import DefaultConfig
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPAugmentation,
    ConfigNLPCausalLMArchitecture,
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMEnvironment,
    ConfigNLPCausalLMLogging,
    ConfigNLPCausalLMPrediction,
    ConfigNLPCausalLMTokenizer,
    ConfigNLPCausalLMTraining,
)
from llm_studio.src import possible_values
from llm_studio.src.augmentations.nlp_aug import BaseNLPAug
from llm_studio.src.loggers import Loggers
from llm_studio.src.losses import text_causal_language_modeling_losses
from llm_studio.src.metrics import text_causal_language_modeling_metrics
from llm_studio.src.models import (
    text_reward_model,
    text_sequence_to_sequence_modeling_model,
)
from llm_studio.src.nesting import Dependency
from llm_studio.src.optimizers import Optimizers
from llm_studio.src.plots import text_causal_language_modeling_plots
from llm_studio.src.schedulers import Schedulers
from llm_studio.src.utils.modeling_utils import generate_experiment_name


@dataclass
class ConfigNLPSeq2SeqDataset(ConfigNLPCausalLMDataset):
    text_system_start: str = ""
    text_prompt_start: str = ""
    text_answer_separator: str = ""

    limit_chained_samples: bool = False
    add_eos_token_to_system: bool = True
    add_eos_token_to_prompt: bool = True
    add_eos_token_to_answer: bool = True
    mask_prompt_labels: bool = True

    def __post_init__(self):
        self.prompt_column = (
            tuple(
                self.prompt_column,
            )
            if isinstance(self.prompt_column, str)
            else tuple(self.prompt_column)
        )
        super().__post_init__()

        self._visibility["limit_chained_samples"] = -1
        self._visibility["mask_prompt_labels"] = -1
        self._visibility["dataset_class"] = -1


@dataclass
class ConfigNLPSeq2SeqTraining(ConfigNLPCausalLMTraining):
    use_rlhf: bool = False

    def __post_init__(self):
        super().__post_init__()

        self._visibility["use_rlhf"] = -1


@dataclass
class ConfigNLPSeq2SeqArchitecture(ConfigNLPCausalLMArchitecture):
    model_class: Any = text_sequence_to_sequence_modeling_model.Model

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigProblemBase(DefaultConfig):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    _parent_experiment: str = ""
    llm_backbone: str = "t5-small"

    dataset: ConfigNLPSeq2SeqDataset = field(default_factory=ConfigNLPSeq2SeqDataset)
    tokenizer: ConfigNLPCausalLMTokenizer = field(
        default_factory=ConfigNLPCausalLMTokenizer
    )
    architecture: ConfigNLPSeq2SeqArchitecture = field(
        default_factory=ConfigNLPSeq2SeqArchitecture
    )
    training: ConfigNLPSeq2SeqTraining = field(default_factory=ConfigNLPSeq2SeqTraining)
    augmentation: ConfigNLPAugmentation = field(default_factory=ConfigNLPAugmentation)
    prediction: ConfigNLPCausalLMPrediction = field(
        default_factory=ConfigNLPCausalLMPrediction
    )
    environment: ConfigNLPCausalLMEnvironment = field(
        default_factory=ConfigNLPCausalLMEnvironment
    )
    logging: ConfigNLPCausalLMLogging = field(default_factory=ConfigNLPCausalLMLogging)

    def __post_init__(self):
        super().__post_init__()

        self._visibility["output_directory"] = -1

        self._possible_values["llm_backbone"] = possible_values.String(
            values=(
                "t5-small",
                "t5-base",
                "t5-large",
                "google/flan-t5-small",
                "google/flan-t5-base",
                "google/flan-t5-large",
                "google/flan-ul2",
            ),
            allow_custom=True,
        )

    @classmethod
    def from_dict(cls, cfg_dict: dict):
        return cls(
            output_directory=cfg_dict.get(
                "output_directory", ConfigProblemBase.output_directory
            ),
            experiment_name=cfg_dict.get("experiment_name", generate_experiment_name()),
            llm_backbone=cfg_dict.get("llm_backbone", ConfigProblemBase.llm_backbone),
            dataset=ConfigNLPSeq2SeqDataset.from_dict(cfg_dict.get("dataset", {})),
            tokenizer=ConfigNLPCausalLMTokenizer.from_dict(
                cfg_dict.get("tokenizer", {})
            ),
            augmentation=ConfigNLPAugmentation.from_dict(
                cfg_dict.get("augmentation", {})
            ),
            architecture=ConfigNLPSeq2SeqArchitecture.from_dict(
                cfg_dict.get("architecture", {})
            ),
            training=ConfigNLPSeq2SeqTraining.from_dict(cfg_dict.get("training", {})),
            prediction=ConfigNLPCausalLMPrediction.from_dict(
                cfg_dict.get("prediction", {})
            ),
            environment=ConfigNLPCausalLMEnvironment.from_dict(
                cfg_dict.get("environment", {})
            ),
            logging=ConfigNLPCausalLMLogging.from_dict(cfg_dict.get("logging", {})),
        )
