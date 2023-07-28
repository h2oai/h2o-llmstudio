from dataclasses import dataclass, field
from typing import Any

import llm_studio.src.models.text_rlhf_language_modeling_model
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
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigProblemBase,
)
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigProblemBase as TextCausalLMConfigProblemBase,
)
from llm_studio.src import possible_values
from llm_studio.src.datasets.text_rlhf_modeling_ds import CustomDataset
from llm_studio.src.metrics import text_causal_language_modeling_metrics
from llm_studio.src.models import text_reward_model
from llm_studio.src.utils.modeling_utils import generate_experiment_name


@dataclass
class ConfigRLHFLMDataset(ConfigNLPCausalLMDataset):
    dataset_class: Any = CustomDataset

    def __post_init__(self):
        super().__post_init__()
        # RLHF is not compatible with system column.
        self.system_column = "None"
        self._visibility["system_column"] = False


@dataclass
class ConfigRLHFLMTraining(ConfigNLPCausalLMTraining):
    reward_model: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
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
            values=("backbone", "embed", "value_head"),
            allow_custom=False,
            placeholder="Select optional layers...",
        )
        self._possible_values["reward_model"] = possible_values.String(
            values=(
                "OpenAssistant/reward-model-deberta-v3-large-v2",
                "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
                "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
            ),
            allow_custom=False,
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

        self._visibility["lora"] = -1


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
    metric_class: Any = text_causal_language_modeling_metrics.Metrics
    metric: str = "GPT"
    metric_gpt_model: str = "gpt-3.5-turbo-0301"

    min_length_inference: int = 2
    max_length_inference: int = 256
    batch_size_inference: int = 0

    do_sample: bool = True
    num_beams: int = 1
    temperature: float = 0.3
    repetition_penalty: float = 1.0
    stop_tokens: str = ""
    top_k: int = 0
    top_p: float = 1.0

    num_history: int = 4

    def __post_init__(self):
        super().__post_init__()
        # fixed for RLHF
        self._visibility["do_sample"] = -1
        self._visibility["repetition_penalty"] = -1
        self._visibility["top_p"] = -1
        self._visibility["top_k"] = -1


@dataclass
class ConfigProblemBase(TextCausalLMConfigProblemBase):
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
    environment: ConfigNLPCausalLMEnvironment = field(
        default_factory=ConfigNLPCausalLMEnvironment
    )
    logging: ConfigNLPCausalLMLogging = field(default_factory=ConfigNLPCausalLMLogging)

    @classmethod
    def from_dict(cls, cfg_dict: dict):
        return cls(
            output_directory=cfg_dict.get(
                "output_directory", ConfigProblemBase.output_directory
            ),
            experiment_name=cfg_dict.get("experiment_name", generate_experiment_name()),
            llm_backbone=cfg_dict.get("llm_backbone", ConfigProblemBase.llm_backbone),
            dataset=ConfigRLHFLMDataset.from_dict(cfg_dict.get("dataset", {})),
            tokenizer=ConfigNLPCausalLMTokenizer.from_dict(
                cfg_dict.get("tokenizer", {})
            ),
            augmentation=ConfigNLPAugmentation.from_dict(
                cfg_dict.get("augmentation", {})
            ),
            architecture=ConfigRLHFLMArchitecture.from_dict(
                cfg_dict.get("architecture", {})
            ),
            training=ConfigRLHFLMTraining.from_dict(cfg_dict.get("training", {})),
            prediction=ConfigRLHFLMPrediction.from_dict(cfg_dict.get("prediction", {})),
            environment=ConfigNLPCausalLMEnvironment.from_dict(
                cfg_dict.get("environment", {})
            ),
            logging=ConfigNLPCausalLMLogging.from_dict(cfg_dict.get("logging", {})),
        )
