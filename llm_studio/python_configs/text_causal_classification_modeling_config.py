import os
from dataclasses import dataclass, field
from typing import Any, Tuple

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
from llm_studio.src.losses import text_causal_classification_modeling_losses
from llm_studio.src.metrics import text_causal_classification_modeling_metrics
from llm_studio.src.models import text_causal_classification_modeling_model
from llm_studio.src.utils.modeling_utils import generate_experiment_name


@dataclass
class ConfigNLPCausalClassificationDataset(ConfigNLPCausalLMDataset):
    system_column: str = "None"
    prompt_column: Tuple[str, ...] = ("instruction", "input")
    answer_column: str = "label"
    num_classes: int = 1
    parent_id_column: str = "None"

    text_system_start: str = ""
    text_prompt_start: str = ""
    text_answer_separator: str = ""

    add_eos_token_to_system: bool = False
    add_eos_token_to_prompt: bool = False
    add_eos_token_to_answer: bool = False

    _allowed_file_extensions: Tuple[str, ...] = ("csv", "pq", "parquet")

    def __post_init__(self):
        self.prompt_column = (
            tuple(
                self.prompt_column,
            )
            if isinstance(self.prompt_column, str)
            else tuple(self.prompt_column)
        )
        super().__post_init__()

        self._possible_values["num_classes"] = (1, 100, 1)

        self._visibility["personalize"] = -1
        self._visibility["chatbot_name"] = -1
        self._visibility["chatbot_author"] = -1
        self._visibility["mask_prompt_labels"] = -1
        self._visibility["add_eos_token_to_answer"] = -1


@dataclass
class ConfigNLPCausalClassificationTraining(ConfigNLPCausalLMTraining):
    loss_class: Any = text_causal_classification_modeling_losses.Losses
    loss_function: str = "BinaryCrossEntropyLoss"

    learning_rate: float = 0.0001
    differential_learning_rate_layers: Tuple[str, ...] = ("classification_head",)
    differential_learning_rate: float = 0.00001

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["loss_function"] = self.loss_class.names()

        self._possible_values[
            "differential_learning_rate_layers"
        ] = possible_values.String(
            values=("backbone", "embed", "classification_head"),
            allow_custom=False,
            placeholder="Select optional layers...",
        )


@dataclass
class ConfigNLPCausalClassificationTokenizer(ConfigNLPCausalLMTokenizer):
    max_length_prompt: int = 512
    max_length: int = 512

    def __post_init__(self):
        super().__post_init__()

        self._visibility["max_length_answer"] = -1


@dataclass
class ConfigNLPCausalClassificationArchitecture(ConfigNLPCausalLMArchitecture):
    model_class: Any = text_causal_classification_modeling_model.Model

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigNLPCausalClassificationPrediction(ConfigNLPCausalLMPrediction):
    metric_class: Any = text_causal_classification_modeling_metrics.Metrics
    metric: str = "AUC"

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["metric"] = self.metric_class.names()

        for k in [
            "min_length_inference",
            "max_length_inference",
            "do_sample",
            "num_beams",
            "temperature",
            "repetition_penalty",
            "stop_tokens",
            "top_k",
            "top_p",
        ]:
            self._visibility[k] = -1


@dataclass
class ConfigNLPCausalClassificationEnvironment(ConfigNLPCausalLMEnvironment):
    _model_card_template: str = "text_causal_classification_model_card_template.md"
    _summary_card_template: str = (
        "text_causal_classification_experiment_summary_card_template.md"
    )

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigProblemBase(DefaultConfigProblemBase):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    _parent_experiment: str = ""
    llm_backbone: str = "h2oai/h2ogpt-4096-llama2-7b"
    type: str = "causal_classification"

    dataset: ConfigNLPCausalClassificationDataset = field(
        default_factory=ConfigNLPCausalClassificationDataset
    )
    tokenizer: ConfigNLPCausalLMTokenizer = field(
        default_factory=ConfigNLPCausalLMTokenizer
    )
    architecture: ConfigNLPCausalClassificationArchitecture = field(
        default_factory=ConfigNLPCausalClassificationArchitecture
    )
    training: ConfigNLPCausalClassificationTraining = field(
        default_factory=ConfigNLPCausalClassificationTraining
    )
    augmentation: ConfigNLPAugmentation = field(default_factory=ConfigNLPAugmentation)
    prediction: ConfigNLPCausalClassificationPrediction = field(
        default_factory=ConfigNLPCausalClassificationPrediction
    )
    environment: ConfigNLPCausalClassificationEnvironment = field(
        default_factory=ConfigNLPCausalClassificationEnvironment
    )
    logging: ConfigNLPCausalLMLogging = field(default_factory=ConfigNLPCausalLMLogging)

    def __post_init__(self):
        super().__post_init__()

        self._visibility["output_directory"] = -1

        self._possible_values["llm_backbone"] = possible_values.String(
            values=(
                "h2oai/h2ogpt-4096-llama2-70b",
                "h2oai/h2ogpt-4096-llama2-70b-chat",
                "h2oai/h2ogpt-4096-llama2-13b",
                "h2oai/h2ogpt-4096-llama2-13b-chat",
                "h2oai/h2ogpt-4096-llama2-7b",
                "h2oai/h2ogpt-4096-llama2-7b-chat",
                "tiiuae/falcon-40b",
                "tiiuae/falcon-7b",
                "openlm-research/open_llama_13b",
                "openlm-research/open_llama_7b",
                "openlm-research/open_llama_3b",
                "EleutherAI/gpt-j-6B",
                "facebook/opt-125m",
            ),
            allow_custom=True,
        )
