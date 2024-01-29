import os
from dataclasses import dataclass, field
from typing import Any

import llm_studio.src.datasets.text_dpo_modeling_ds
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
from llm_studio.src.losses import text_dpo_modeling_losses
from llm_studio.src.models import text_dpo_modeling_model
from llm_studio.src.nesting import Dependency
from llm_studio.src.plots import text_dpo_modeling_plots
from llm_studio.src.utils.modeling_utils import generate_experiment_name


@dataclass
class ConfigDPODataset(ConfigNLPCausalLMDataset):
    dataset_class: Any = llm_studio.src.datasets.text_dpo_modeling_ds.CustomDataset
    # Always have full chat history.
    # Chosen/Rejected prompt are only at the end of a conversation.
    limit_chained_samples: bool = True
    mask_prompt_labels: bool = True

    answer_column: str = "chosen_response"
    rejected_answer_column: str = "rejected_response"

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["rejected_answer_column"] = possible_values.Columns(
            prefer_with=lambda column: column
            in ("rejected_answer", "rejected_response")
        )

        self._visibility["limit_chained_samples"] = -1
        self._visibility["mask_prompt_labels"] = -1
        self._order.insert("rejected_answer_column", after="answer_column")


@dataclass
class ConfigDPOTraining(ConfigNLPCausalLMTraining):
    learning_rate: float = 1e-4  # relatively high as we use LORA
    beta: float = 0.2
    gradient_clip: float = 10.0
    loss_class: Any = text_dpo_modeling_losses.Losses
    loss_function: str = "DPOLoss"
    optimizer: str = "AdamW"
    # Needs to be enabled as we need logits from original model, see forward pass
    lora: bool = True

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["beta"] = possible_values.Number(0.05, 0.5, 0.05)
        self._order.insert("beta", after="learning_rate")
        self._visibility["lora"] = -1


@dataclass
class ConfigDPOArchitecture(ConfigNLPCausalLMArchitecture):
    model_class: Any = text_dpo_modeling_model.Model


@dataclass
class ConfigDPOPLogging(ConfigNLPCausalLMLogging):
    plots_class: Any = text_dpo_modeling_plots.Plots


@dataclass
class ConfigProblemBase(DefaultConfigProblemBase):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    _parent_experiment: str = ""
    # 7b model may be unstable (NaN loss)
    llm_backbone: str = "h2oai/h2ogpt-4096-llama2-13b-chat"

    dataset: ConfigDPODataset = field(default_factory=ConfigDPODataset)
    tokenizer: ConfigNLPCausalLMTokenizer = field(
        default_factory=ConfigNLPCausalLMTokenizer
    )
    architecture: ConfigDPOArchitecture = field(default_factory=ConfigDPOArchitecture)
    training: ConfigDPOTraining = field(default_factory=ConfigDPOTraining)
    augmentation: ConfigNLPAugmentation = field(default_factory=ConfigNLPAugmentation)
    prediction: ConfigNLPCausalLMPrediction = field(
        default_factory=ConfigNLPCausalLMPrediction
    )
    environment: ConfigNLPCausalLMEnvironment = field(
        default_factory=ConfigNLPCausalLMEnvironment
    )
    logging: ConfigDPOPLogging = field(default_factory=ConfigDPOPLogging)

    def __post_init__(self):
        super().__post_init__()
        self._visibility["output_directory"] = -1
        self._possible_values["llm_backbone"] = possible_values.String(
            values=(
                "h2oai/h2ogpt-4096-llama2-7b",
                "h2oai/h2ogpt-4096-llama2-7b-chat",
                "h2oai/h2ogpt-4096-llama2-13b",
                "h2oai/h2ogpt-4096-llama2-13b-chat",
                "h2oai/h2ogpt-4096-llama2-70b",
                "h2oai/h2ogpt-4096-llama2-70b-chat",
                "tiiuae/falcon-7b",
                "tiiuae/falcon-40b",
                "mistralai/Mistral-7B-v0.1",
                "HuggingFaceH4/zephyr-7b-beta",
                "stabilityai/stablelm-3b-4e1t",
                "facebook/opt-125m",
            ),
            allow_custom=True,
        )
