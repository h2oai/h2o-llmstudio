import os
from dataclasses import dataclass, field
from typing import Any

import llm_studio.src.models.text_rlhf_language_modeling_model
from llm_studio.python_configs.base import DefaultConfig
from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMTraining,
    ConfigNLPCausalLMArchitecture,
    ConfigNLPCausalLMPrediction,
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMTokenizer,
    ConfigNLPAugmentation,
    ConfigNLPCausalLMEnvironment,
    ConfigNLPCausalLMLogging,
)
from llm_studio.src import possible_values
from llm_studio.src.datasets.text_rlhf_language_modeling_ds import CustomDataset
from llm_studio.src.metrics import text_causal_language_modeling_metrics
from llm_studio.src.models import text_reward_model
from llm_studio.src.utils.modeling_utils import generate_experiment_name


@dataclass
class ConfigRLHDataset(ConfigNLPCausalLMDataset):
    dataset_class: Any = CustomDataset


@dataclass
class ConfigRLHFLMTraining(ConfigNLPCausalLMTraining):
    lora: bool = True

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
    pretrained: bool = True

    backbone_dtype: str = "float16"
    gradient_checkpointing: bool = True
    intermediate_dropout: float = 0
    pretrained_weights: str = ""

    def __post_init__(self):
        super().__post_init__()
        self._visibility["reward_model_class"] = -1


@dataclass
class ConfigRLHFPrediction(ConfigNLPCausalLMPrediction):
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
        self._visibility["do_sample"] = -1
        self._visibility["repetition_penalty"] = -1
        self._visibility["top_p"] = -1
        self._visibility["top_k"] = -1


@dataclass
class ConfigProblemBase(DefaultConfig):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    _parent_experiment: str = ""
    llm_backbone: str = "EleutherAI/pythia-2.8b-deduped"

    dataset: ConfigRLHDataset = field(default_factory=ConfigRLHDataset)
    tokenizer: ConfigNLPCausalLMTokenizer = field(
        default_factory=ConfigNLPCausalLMTokenizer
    )
    architecture: ConfigRLHFLMArchitecture = field(
        default_factory=ConfigRLHFLMArchitecture
    )
    training: ConfigRLHFLMTraining = field(default_factory=ConfigRLHFLMTraining)
    augmentation: ConfigNLPAugmentation = field(default_factory=ConfigNLPAugmentation)
    prediction: ConfigRLHFPrediction = field(default_factory=ConfigRLHFPrediction)
    environment: ConfigNLPCausalLMEnvironment = field(
        default_factory=ConfigNLPCausalLMEnvironment
    )
    logging: ConfigNLPCausalLMLogging = field(default_factory=ConfigNLPCausalLMLogging)

    def __post_init__(self):
        super().__post_init__()

        self._visibility["output_directory"] = -1

        self._possible_values["llm_backbone"] = possible_values.String(
            values=(
                "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3",
                "h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b",
                "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2",
                "tiiuae/falcon-7b",
                "tiiuae/falcon-40b",
                "openlm-research/open_llama_3b",
                "openlm-research/open_llama_7b",
                "openlm-research/open_llama_13b",
                "EleutherAI/gpt-j-6B",
                "EleutherAI/gpt-neox-20b",
                "facebook/opt-125m",
                "facebook/opt-2.7b",
                "EleutherAI/pythia-1b-deduped",
                "EleutherAI/pythia-2.8b-deduped",
                "EleutherAI/pythia-6.9b-deduped",
                "EleutherAI/pythia-12b-deduped",
                "togethercomputer/GPT-NeoXT-Chat-Base-20B",
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
            dataset=ConfigRLHDataset.from_dict(cfg_dict.get("dataset", {})),
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
            prediction=ConfigRLHFPrediction.from_dict(cfg_dict.get("prediction", {})),
            environment=ConfigNLPCausalLMEnvironment.from_dict(
                cfg_dict.get("environment", {})
            ),
            logging=ConfigNLPCausalLMLogging.from_dict(cfg_dict.get("logging", {})),
        )
