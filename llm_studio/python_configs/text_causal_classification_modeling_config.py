import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import llm_studio.src.datasets.text_causal_classification_ds
import llm_studio.src.plots.text_causal_classification_modeling_plots
from llm_studio.python_configs.base import DefaultConfig, DefaultConfigProblemBase
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPAugmentation,
    ConfigNLPCausalLMArchitecture,
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMEnvironment,
    ConfigNLPCausalLMLogging,
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
    dataset_class: Any = (
        llm_studio.src.datasets.text_causal_classification_ds.CustomDataset
    )
    system_column: str = "None"
    prompt_column: Tuple[str, ...] = ("instruction", "input")
    answer_column: str = "label"
    num_classes: int = 1
    parent_id_column: str = "None"

    text_system_start: str = ""
    text_prompt_start: str = ""
    text_answer_separator: str = ""

    add_prompt_answer_tokens: bool = False

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

        self._visibility["system_column"] = -1
        self._visibility["parent_id_column"] = -1
        self._visibility["text_system_start"] = -1
        self._visibility["add_prompt_answer_tokens"] = -1
        self._visibility["add_eos_token_to_system"] = -1
        self._visibility["add_eos_token_to_answer"] = -1
        self._visibility["personalize"] = -1
        self._visibility["chatbot_name"] = -1
        self._visibility["chatbot_author"] = -1
        self._visibility["mask_prompt_labels"] = -1
        self._visibility["only_last_answer"] = -1


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

        self._possible_values["differential_learning_rate_layers"] = (
            possible_values.String(
                values=("backbone", "embed", "classification_head"),
                allow_custom=False,
                placeholder="Select optional layers...",
            )
        )


@dataclass
class ConfigNLPCausalClassificationTokenizer(ConfigNLPCausalLMTokenizer):
    max_length: int = 512

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigNLPCausalClassificationArchitecture(ConfigNLPCausalLMArchitecture):
    model_class: Any = text_causal_classification_modeling_model.Model

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigNLPCausalClassificationAugmentation(ConfigNLPAugmentation):
    skip_parent_probability: float = 0.0
    random_parent_probability: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self._visibility["skip_parent_probability"] = -1
        self._visibility["random_parent_probability"] = -1


@dataclass
class ConfigNLPCausalClassificationPrediction(DefaultConfig):
    metric_class: Any = text_causal_classification_modeling_metrics.Metrics
    metric: str = "AUC"
    batch_size_inference: int = 0

    def __post_init__(self):
        super().__post_init__()

        self._possible_values["metric"] = self.metric_class.names()
        self._possible_values["batch_size_inference"] = (0, 512, 1)

        self._visibility["metric_class"] = -1


@dataclass
class ConfigNLPCausalClassificationEnvironment(ConfigNLPCausalLMEnvironment):
    _model_card_template: str = "text_causal_classification_model_card_template.md"
    _summary_card_template: str = (
        "text_causal_classification_experiment_summary_card_template.md"
    )

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigNLPCausalClassificationLogging(ConfigNLPCausalLMLogging):
    plots_class: Any = (
        llm_studio.src.plots.text_causal_classification_modeling_plots.Plots
    )


@dataclass
class ConfigProblemBase(DefaultConfigProblemBase):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    llm_backbone: str = "h2oai/h2o-danube2-1.8b-chat"

    dataset: ConfigNLPCausalClassificationDataset = field(
        default_factory=ConfigNLPCausalClassificationDataset
    )
    tokenizer: ConfigNLPCausalClassificationTokenizer = field(
        default_factory=ConfigNLPCausalClassificationTokenizer
    )
    architecture: ConfigNLPCausalClassificationArchitecture = field(
        default_factory=ConfigNLPCausalClassificationArchitecture
    )
    training: ConfigNLPCausalClassificationTraining = field(
        default_factory=ConfigNLPCausalClassificationTraining
    )
    augmentation: ConfigNLPCausalClassificationAugmentation = field(
        default_factory=ConfigNLPCausalClassificationAugmentation
    )
    prediction: ConfigNLPCausalClassificationPrediction = field(
        default_factory=ConfigNLPCausalClassificationPrediction
    )
    environment: ConfigNLPCausalClassificationEnvironment = field(
        default_factory=ConfigNLPCausalClassificationEnvironment
    )
    logging: ConfigNLPCausalClassificationLogging = field(
        default_factory=ConfigNLPCausalClassificationLogging
    )

    def __post_init__(self):
        super().__post_init__()

        self._visibility["output_directory"] = -1

        self._possible_values["llm_backbone"] = possible_values.String(
            values=(
                "h2oai/h2o-danube2-1.8b-base",
                "h2oai/h2o-danube2-1.8b-chat",
                "h2oai/h2ogpt-4096-llama2-7b",
                "h2oai/h2ogpt-4096-llama2-7b-chat",
                "h2oai/h2ogpt-4096-llama2-13b",
                "h2oai/h2ogpt-4096-llama2-13b-chat",
                "h2oai/h2ogpt-4096-llama2-70b",
                "h2oai/h2ogpt-4096-llama2-70b-chat",
                "tiiuae/falcon-7b",
                "mistralai/Mistral-7B-v0.1",
                "HuggingFaceH4/zephyr-7b-beta",
                "google/gemma-2b",
                "google/gemma-7b",
                "stabilityai/stablelm-3b-4e1t",
                "microsoft/phi-2",
                "facebook/opt-125m",
            ),
            allow_custom=True,
        )

    def check(self) -> Dict[str, List]:
        errors: Dict[str, List] = {"title": [], "message": []}

        if self.training.loss_function == "CrossEntropyLoss":
            if self.dataset.num_classes == 1:
                errors["title"] += ["CrossEntropyLoss requires num_classes > 1"]
                errors["message"] += [
                    "CrossEntropyLoss requires num_classes > 1, "
                    "but num_classes is set to 1."
                ]
        elif self.training.loss_function == "BinaryCrossEntropyLoss":
            if self.dataset.num_classes != 1:
                errors["title"] += ["BinaryCrossEntropyLoss requires num_classes == 1"]
                errors["message"] += [
                    "BinaryCrossEntropyLoss requires num_classes == 1, "
                    "but num_classes is set to {}.".format(self.dataset.num_classes)
                ]
        if self.dataset.parent_id_column not in ["None", None]:
            errors["title"] += ["Parent ID column is not supported for classification"]
            errors["message"] += [
                "Parent ID column is not supported for classification datasets."
            ]

        return errors
