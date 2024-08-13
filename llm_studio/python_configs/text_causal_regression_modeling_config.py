import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import llm_studio.src.datasets.text_causal_regression_ds
import llm_studio.src.plots.text_causal_classification_modeling_plots
from llm_studio.python_configs.base import DefaultConfig, DefaultConfigProblemBase
from llm_studio.python_configs.text_causal_classification_modeling_config import (
    ConfigNLPCausalClassificationAugmentation as ConfigNLPCausalRegressionAugmentation,
)
from llm_studio.python_configs.text_causal_classification_modeling_config import (
    ConfigNLPCausalClassificationDataset,
)
from llm_studio.python_configs.text_causal_classification_modeling_config import (
    ConfigNLPCausalClassificationLogging as ConfigNLPCausalRegressionLogging,
)
from llm_studio.python_configs.text_causal_classification_modeling_config import (
    ConfigNLPCausalClassificationTokenizer as ConfigNLPCausalRegressionTokenizer,
)
from llm_studio.python_configs.text_causal_classification_modeling_config import (
    ConfigNLPCausalClassificationTraining,
)
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMArchitecture,
    ConfigNLPCausalLMEnvironment,
)
from llm_studio.src import possible_values
from llm_studio.src.losses import text_causal_regression_modeling_losses
from llm_studio.src.metrics import text_causal_regression_modeling_metrics
from llm_studio.src.models import text_causal_regression_modeling_model
from llm_studio.src.utils.modeling_utils import generate_experiment_name


@dataclass
class ConfigNLPCausalRegressionDataset(ConfigNLPCausalClassificationDataset):
    dataset_class: Any = llm_studio.src.datasets.text_causal_regression_ds.CustomDataset
    num_classes: int = 1

    def __post_init__(self):
        self.prompt_column = (
            tuple(
                self.prompt_column,
            )
            if isinstance(self.prompt_column, str)
            else tuple(self.prompt_column)
        )
        super().__post_init__()

        self._visibility["num_classes"] = -1


@dataclass
class ConfigNLPCausalRegressionTraining(ConfigNLPCausalClassificationTraining):
    loss_class: Any = text_causal_regression_modeling_losses.Losses
    loss_function: str = "MSELoss"

    learning_rate: float = 0.0001
    differential_learning_rate_layers: Tuple[str, ...] = ("regression_head",)
    differential_learning_rate: float = 0.00001

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["loss_function"] = self.loss_class.names()

        self._possible_values["differential_learning_rate_layers"] = (
            possible_values.String(
                values=("backbone", "embed", "regression_head"),
                allow_custom=False,
                placeholder="Select optional layers...",
            )
        )


@dataclass
class ConfigNLPCausalRegressionArchitecture(ConfigNLPCausalLMArchitecture):
    model_class: Any = text_causal_regression_modeling_model.Model

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigNLPCausalRegressionPrediction(DefaultConfig):
    metric_class: Any = text_causal_regression_modeling_metrics.Metrics
    metric: str = "MSE"
    batch_size_inference: int = 0

    def __post_init__(self):
        super().__post_init__()

        self._possible_values["metric"] = self.metric_class.names()
        self._possible_values["batch_size_inference"] = (0, 512, 1)

        self._visibility["metric_class"] = -1


@dataclass
class ConfigNLPCausalRegressionEnvironment(ConfigNLPCausalLMEnvironment):
    _model_card_template: str = "text_causal_regression_model_card_template.md"
    _summary_card_template: str = (
        "text_causal_regression_experiment_summary_card_template.md"
    )

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ConfigProblemBase(DefaultConfigProblemBase):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    llm_backbone: str = "h2oai/h2o-danube3-500m-chat"

    dataset: ConfigNLPCausalRegressionDataset = field(
        default_factory=ConfigNLPCausalRegressionDataset
    )
    tokenizer: ConfigNLPCausalRegressionTokenizer = field(
        default_factory=ConfigNLPCausalRegressionTokenizer
    )
    architecture: ConfigNLPCausalRegressionArchitecture = field(
        default_factory=ConfigNLPCausalRegressionArchitecture
    )
    training: ConfigNLPCausalRegressionTraining = field(
        default_factory=ConfigNLPCausalRegressionTraining
    )
    augmentation: ConfigNLPCausalRegressionAugmentation = field(
        default_factory=ConfigNLPCausalRegressionAugmentation
    )
    prediction: ConfigNLPCausalRegressionPrediction = field(
        default_factory=ConfigNLPCausalRegressionPrediction
    )
    environment: ConfigNLPCausalRegressionEnvironment = field(
        default_factory=ConfigNLPCausalRegressionEnvironment
    )
    logging: ConfigNLPCausalRegressionLogging = field(
        default_factory=ConfigNLPCausalRegressionLogging
    )

    def __post_init__(self):
        super().__post_init__()

        self._visibility["output_directory"] = -1

        self._possible_values["llm_backbone"] = possible_values.String(
            values=(
                "h2oai/h2o-danube3-500m-base",
                "h2oai/h2o-danube3-500m-chat",
                "h2oai/h2o-danube3-4b-base",
                "h2oai/h2o-danube3-4b-chat",
                "h2oai/h2o-danube2-1.8b-base",
                "h2oai/h2o-danube2-1.8b-chat",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "meta-llama/Meta-Llama-3.1-70B-Instruct",
                "mistralai/Mistral-7B-v0.3",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "google/gemma-2-2b-it",
                "google/gemma-2-9b-it",
                "microsoft/Phi-3-mini-4k-instruct",
                "microsoft/Phi-3-medium-4k-instruct",
                "Qwen/Qwen2-7B-Instruct",
                "Qwen/Qwen2-72B-Instruct",
            ),
            allow_custom=True,
        )

    def check(self) -> Dict[str, List]:
        errors: Dict[str, List] = {"title": [], "message": [], "type": []}

        if isinstance(self.dataset.answer_column, str):
            errors["title"].append("Invalid answer_column type")
            errors["message"].append(
                "Providing the answer_column as a string is deprecated. "
                "Please provide the answer_column as a list."
            )
            errors["type"].append("deprecated")
            self.dataset.answer_column = [self.dataset.answer_column]

        if self.dataset.parent_id_column not in ["None", None]:
            errors["title"] += ["Parent ID column is not supported for regression"]
            errors["message"] += [
                "Parent ID column is not supported for regression datasets."
            ]
            errors["type"].append("error")

        return errors
