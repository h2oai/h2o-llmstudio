import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from llm_studio.python_configs.base import DefaultConfigProblemBase
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
from llm_studio.src.models import text_sequence_to_sequence_modeling_model
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


@dataclass
class ConfigNLPSeq2SeqArchitecture(ConfigNLPCausalLMArchitecture):
    model_class: Any = text_sequence_to_sequence_modeling_model.Model
    backbone_dtype: str = "bfloat16"

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigNLPSeq2SeqEnvironment(ConfigNLPCausalLMEnvironment):
    mixed_precision: bool = False

    _model_card_template: str = (
        "text_sequence_to_sequence_modeling_model_card_template.md"
    )
    _summary_card_template: str = (
        "text_sequence_to_sequence_modeling_experiment_summary_card_template.md"
    )

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigProblemBase(DefaultConfigProblemBase):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    llm_backbone: str = "t5-small"

    dataset: ConfigNLPSeq2SeqDataset = field(default_factory=ConfigNLPSeq2SeqDataset)
    tokenizer: ConfigNLPCausalLMTokenizer = field(
        default_factory=ConfigNLPCausalLMTokenizer
    )
    architecture: ConfigNLPSeq2SeqArchitecture = field(
        default_factory=ConfigNLPSeq2SeqArchitecture
    )
    training: ConfigNLPCausalLMTraining = field(
        default_factory=ConfigNLPCausalLMTraining
    )
    augmentation: ConfigNLPAugmentation = field(default_factory=ConfigNLPAugmentation)
    prediction: ConfigNLPCausalLMPrediction = field(
        default_factory=ConfigNLPCausalLMPrediction
    )
    environment: ConfigNLPSeq2SeqEnvironment = field(
        default_factory=ConfigNLPSeq2SeqEnvironment
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

    def check(self) -> Dict[str, List]:
        errors: Dict[str, List] = {"title": [], "message": [], "type": []}
        if self.prediction.temperature > 0 and not self.prediction.do_sample:
            errors["title"] += ["Do sample needs to be enabled for temperature > 0"]
            errors["message"] += [
                "Please enable do sample if you want to use temperature > 0."
            ]
            errors["type"].append("warning")
        return errors
