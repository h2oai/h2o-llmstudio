import os
from dataclasses import dataclass, field
from typing import Any

import llm_studio.src.models.text_rlhf_language_modeling_model
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
from llm_studio.src.datasets.text_rlhf_modeling_ds import CustomDataset
from llm_studio.src.models import text_reward_model
from llm_studio.src.utils.modeling_utils import generate_experiment_name


@dataclass
class ConfigRLHFLMDataset(ConfigNLPCausalLMDataset):
    dataset_class: Any = CustomDataset

    def __post_init__(self):
        super().__post_init__()
        # RLHF is not compatible with system column.
        self.system_column = "None"
        self._visibility["system_column"] = -1


class LossClass:
    @classmethod
    def names(cls):
        return []


class ConfigRLHFLMAugmentation(ConfigNLPAugmentation):
    def __post_init__(self):
        super().__post_init__()
        self._visibility["skip_parent_probability"] = -1
        self._visibility["random_parent_probability"] = -1


@dataclass
class ConfigRLHFLMTraining(ConfigNLPCausalLMTraining):
    loss_class: Any = LossClass
    loss_function: str = "RLHF"
    adaptive_kl_control: bool = True
    initial_kl_coefficient: float = 0.2
    kl_target: float = 6.0
    kl_horizon: int = 10000
    advantages_gamma: float = 0.99
    advantages_lambda: float = 0.95
    ppo_clip_policy: float = 0.2
    ppo_clip_value: float = 0.2
    scaling_factor_value_loss: float = 0.1
    ppo_epochs: int = 4
    ppo_batch_size: int = 1
    ppo_generate_temperature: float = 1.0
    offload_reward_model: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.lora = True
        self._possible_values[
            "differential_learning_rate_layers"
        ] = possible_values.String(
            values=("backbone", "value_head"),
            allow_custom=False,
            placeholder="Select optional layers...",
        )

        self._possible_values["initial_kl_coefficient"] = (0.01, 0.5, 0.01)
        self._possible_values["kl_target"] = (0.1, 16, 0.1)
        self._possible_values["kl_horizon"] = (1000, 20000, 1000)
        self._possible_values["advantages_gamma"] = (0.800, 0.999, 0.001)
        self._possible_values["advantages_lambda"] = (0.8, 1.0, 0.01)
        self._possible_values["ppo_clip_policy"] = (0.1, 0.5, 0.05)
        self._possible_values["ppo_clip_value"] = (0.1, 0.5, 0.05)
        self._possible_values["scaling_factor_value_loss"] = (0.01, 1, 0.01)
        self._possible_values["ppo_epochs"] = (1, 16, 1)
        self._possible_values["ppo_generate_temperature"] = (0.1, 1.0, 0.1)
        self._possible_values["ppo_batch_size"] = (1, 256, 1)

        self._order.insert(
            "adaptive_kl_control",
            "advantages_gamma",
            "offload_reward_model",
            "kl_horizon",
            "ppo_generate_temperature",
            "kl_target",
            "scaling_factor_value_loss",
            "ppo_clip_value",
            "ppo_clip_policy",
            "initial_kl_coefficient",
            "advantages_lambda",
            "ppo_batch_size",
            "ppo_epochs",
            after="learning_rate",
        )

        self._visibility["lora"] = -1
        self._visibility["loss_function"] = -1


@dataclass
class ConfigRLHFLMArchitecture(ConfigNLPCausalLMArchitecture):
    model_class: Any = llm_studio.src.models.text_rlhf_language_modeling_model.Model
    reward_model_class: Any = text_reward_model.RewardModel

    def __post_init__(self):
        super().__post_init__()
        # RLHF is not supported with force_embedding_gradients.
        self.force_embedding_gradients = False
        self._visibility["reward_model_class"] = -1
        self._visibility["force_embedding_gradients"] = -1


@dataclass
class ConfigRLHFLMPrediction(ConfigNLPCausalLMPrediction):
    do_sample: bool = True
    repetition_penalty: float = 1.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        # These values are fixed for RLHF
        self._visibility["do_sample"] = -1
        self._visibility["repetition_penalty"] = -1
        self._visibility["top_p"] = -1
        self._visibility["top_k"] = -1


@dataclass
class ConfigRLHFLMEnvironment(ConfigNLPCausalLMEnvironment):
    def __post_init__(self):
        super().__post_init__()
        self._visibility["use_fsdp"] = -1


@dataclass
class ConfigProblemBase(DefaultConfigProblemBase):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    _parent_experiment: str = ""
    llm_backbone: str = "h2oai/h2ogpt-4096-llama2-7b"
    reward_model: str = "OpenAssistant/reward-model-deberta-v3-large-v2"

    dataset: ConfigRLHFLMDataset = field(default_factory=ConfigRLHFLMDataset)
    tokenizer: ConfigNLPCausalLMTokenizer = field(
        default_factory=ConfigNLPCausalLMTokenizer
    )
    architecture: ConfigRLHFLMArchitecture = field(
        default_factory=ConfigRLHFLMArchitecture
    )
    training: ConfigRLHFLMTraining = field(default_factory=ConfigRLHFLMTraining)
    augmentation: ConfigNLPAugmentation = field(default_factory=ConfigNLPAugmentation)
    prediction: ConfigRLHFLMPrediction = field(default_factory=ConfigRLHFLMPrediction)
    environment: ConfigRLHFLMEnvironment = field(
        default_factory=ConfigRLHFLMEnvironment
    )
    logging: ConfigNLPCausalLMLogging = field(default_factory=ConfigNLPCausalLMLogging)

    def __post_init__(self):
        super().__post_init__()

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

        self._possible_values["reward_model"] = possible_values.String(
            values=(
                "OpenAssistant/reward-model-deberta-v3-large-v2",
                "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
                "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
            ),
            # Custom models are not supported, as they would need to be implemented in
            # /src/models/text_reward_model.py
            allow_custom=False,
        )

        self._order.insert(
            "reward_model",
            after="llm_backbone",
        )
        self._visibility["output_directory"] = -1
