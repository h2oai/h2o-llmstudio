import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Tuple

import torch

import llm_studio.src.datasets.text_causal_language_modeling_ds
from llm_studio.python_configs.base import DefaultConfig
from llm_studio.src import possible_values
from llm_studio.src.augmentations.nlp_aug import BaseNLPAug
from llm_studio.src.loggers import Loggers
from llm_studio.src.losses import text_causal_language_modeling_losses
from llm_studio.src.metrics import text_causal_language_modeling_metrics
from llm_studio.src.models import text_causal_language_modeling_model, text_reward_model
from llm_studio.src.nesting import Dependency
from llm_studio.src.optimizers import Optimizers
from llm_studio.src.plots import text_causal_language_modeling_plots
from llm_studio.src.schedulers import Schedulers
from llm_studio.src.utils.modeling_utils import generate_experiment_name


@dataclass
class ConfigNLPCausalLMDataset(DefaultConfig):
    dataset_class: Any = (
        llm_studio.src.datasets.text_causal_language_modeling_ds.CustomDataset
    )

    personalize: bool = False
    chatbot_name: str = "h2oGPT"
    chatbot_author: str = "H2O.ai"

    train_dataframe: str = "/path/to/train.csv"
    validation_strategy: str = "automatic"
    validation_dataframe: str = ""
    validation_size: float = 0.01

    data_sample: float = 1.0
    data_sample_choice: Tuple[str, ...] = ("Train", "Validation")

    system_column: str = "None"
    prompt_column: Tuple[str, ...] = ("instruction", "input")
    answer_column: str = "output"
    parent_id_column: str = "None"

    text_system_start: str = "<|system|>"
    text_prompt_start: str = "<|prompt|>"
    text_answer_separator: str = "<|answer|>"

    limit_chained_samples: bool = False
    add_eos_token_to_system: bool = True
    add_eos_token_to_prompt: bool = True
    add_eos_token_to_answer: bool = True
    mask_prompt_labels: bool = True

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

        self._possible_values["train_dataframe"] = possible_values.Files(
            prefer_with=lambda path: "train" in path
        )
        self._possible_values["validation_strategy"] = possible_values.String(
            values=(
                ("custom", "Custom holdout validation"),
                ("automatic", "Automatic holdout validation"),
            ),
            allow_custom=False,
        )
        self._possible_values["validation_dataframe"] = possible_values.Files(
            add_none=True, prefer_with=lambda path: "val" in path
        )
        self._possible_values["validation_size"] = (0.01, 0.95, 0.01)
        self._possible_values["data_sample"] = (0.01, 1, 0.01)
        self._possible_values["data_sample_choice"] = ["Train", "Validation"]
        self._possible_values["system_column"] = possible_values.Columns(
            prefer_with=lambda column: column in ("system",), add_none=True
        )
        self._possible_values["prompt_column"] = possible_values.Columns(
            prefer_with=lambda column: column in ("instruction", "prompt")
        )
        self._possible_values["answer_column"] = possible_values.Columns(
            prefer_with=lambda column: column in ("answer", "output")
        )
        self._possible_values["parent_id_column"] = possible_values.Columns(
            prefer_with=lambda column: column in ("parent",), add_none=True
        )

        self._nesting.add(
            ["chatbot_name", "chatbot_author"],
            [Dependency(key="personalize", value=True, is_set=True)],
        )

        self._nesting.add(
            ["validation_dataframe"],
            [Dependency(key="validation_strategy", value="custom", is_set=True)],
        )

        self._nesting.add(
            ["validation_size"],
            [Dependency(key="validation_strategy", value="automatic", is_set=True)],
        )

        self._nesting.add(
            ["data_sample_choice"],
            [Dependency(key="data_sample", value=1, is_set=False)],
        )

        self._nesting.add(
            ["limit_chained_samples"],
            [Dependency(key="parent_id_column", value="None", is_set=False)],
        )

        self._nesting.add(
            ["text_system_start", "add_eos_token_to_system"],
            [Dependency(key="system_column", value="None", is_set=False)],
        )

        self._visibility["dataset_class"] = -1


@dataclass
class ConfigNLPCausalLMTraining(DefaultConfig):
    loss_class: Any = text_causal_language_modeling_losses.Losses
    loss_function: str = "TokenAveragedCrossEntropy"
    optimizer: str = "AdamW"

    learning_rate: float = 0.0001
    differential_learning_rate_layers: Tuple[str, ...] = ()
    differential_learning_rate: float = 0.00001

    batch_size: int = 2
    drop_last_batch: bool = True
    epochs: int = 1
    schedule: str = "Cosine"
    warmup_epochs: float = 0.0

    weight_decay: float = 0.0
    gradient_clip: float = 0.0
    grad_accumulation: int = 1

    lora: bool = True
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = ""

    save_best_checkpoint: bool = False
    evaluation_epochs: float = 1.0
    evaluate_before_training: bool = False
    train_validation_data: bool = False

    use_rlhf: bool = False
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
        self._possible_values["loss_function"] = self.loss_class.names()
        self._possible_values["optimizer"] = Optimizers.names()

        self._possible_values["learning_rate"] = possible_values.Number(
            step=0.000001, min=0.000001
        )
        self._possible_values[
            "differential_learning_rate_layers"
        ] = possible_values.String(
            values=("backbone", "embed", "value_head"),
            allow_custom=False,
            placeholder="Select optional layers...",
        )
        self._possible_values["differential_learning_rate"] = self._possible_values[
            "learning_rate"
        ]
        self._possible_values["reward_model"] = possible_values.String(
            values=(
                "OpenAssistant/reward-model-deberta-v3-large-v2",
                "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
                "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
            ),
            allow_custom=False,
        )

        self._possible_values["batch_size"] = (1, 256, 1)
        self._possible_values["epochs"] = (0, 10, 1)
        self._possible_values["schedule"] = Schedulers.names()
        self._possible_values["warmup_epochs"] = (0.0, 5, 0.05)

        self._possible_values["weight_decay"] = possible_values.Number(step=1e-5, min=0)
        self._possible_values["gradient_clip"] = (0.0, 10.0, 0.1)
        self._possible_values["grad_accumulation"] = (1, 8, 1)

        self._possible_values["lora_r"] = (1, 256, 1)
        self._possible_values["lora_alpha"] = (1, 256, 1)
        self._possible_values["lora_dropout"] = (0.0, 0.5, 0.01)

        self._possible_values["evaluation_epochs"] = (0.01, 1, 0.01)

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

        self._visibility["loss_class"] = -1
        self._visibility["drop_last_batch"] = -1
        self._visibility["differential_learning_rate_layers"] = 1
        self._visibility["differential_learning_rate"] = 1
        self._visibility["ppo_batch_size"] = 1

        self._nesting.add(
            ["differential_learning_rate"],
            [
                Dependency(
                    key="differential_learning_rate_layers", value=None, is_set=False
                )
            ],
        )
        self._nesting.add(
            ["lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"],
            [Dependency(key="lora", value=False, is_set=False)],
        )
        self._nesting.add(
            ["train_validation_data"],
            [Dependency(key="save_best_checkpoint", value=False, is_set=True)],
        )
        self._nesting.add(
            [
                "reward_model",
                "differential_learning_rate",
                "adaptive_kl_control",
                "initial_kl_coefficient",
                "kl_target",
                "kl_horizon",
                "advantages_gamma",
                "advantages_lambda",
                "ppo_clip_policy",
                "ppo_clip_value",
                "ppo_generate_temperature",
                "scaling_factor_value_loss",
                "ppo_epochs",
                "ppo_batch_size",
                "offload_reward_model",
            ],
            [Dependency(key="use_rlhf", value=False, is_set=False)],
        )


@dataclass
class ConfigNLPCausalLMTokenizer(DefaultConfig):
    max_length_prompt: int = 256
    max_length_answer: int = 256
    max_length: int = 512
    add_prompt_answer_tokens: bool = False
    padding_quantile: float = 1.0
    use_fast: bool = True
    add_prefix_space: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["max_length_prompt"] = (32, 8192, 32)
        self._possible_values["max_length_answer"] = (32, 8192, 32)
        self._possible_values["max_length"] = (32, 8192, 32)
        self._possible_values["padding_quantile"] = (0, 1, 0.01)
        self._padding_side = "left"

        self._visibility["add_prefix_space"] = -1


@dataclass
class ConfigNLPCausalLMArchitecture(DefaultConfig):
    model_class: Any = text_causal_language_modeling_model.Model
    reward_model_class: Any = text_reward_model.RewardModel
    pretrained: bool = True

    backbone_dtype: str = "float16"
    gradient_checkpointing: bool = True
    force_embedding_gradients: bool = False
    intermediate_dropout: float = 0
    pretrained_weights: str = ""

    def __post_init__(self):
        super().__post_init__()

        self._possible_values["backbone_dtype"] = possible_values.String(
            values=("float32", "bfloat16", "float16", "int8", "int4"),
            allow_custom=False,
        )
        self._possible_values["intermediate_dropout"] = (0, 0.5, 0.05)

        self._nesting.add(
            ["force_embedding_gradients"],
            [Dependency(key="lora", value=False, is_set=False)],
        )

        self._visibility["model_class"] = -1
        self._visibility["reward_model_class"] = -1
        self._visibility["pretrained"] = -1


@dataclass
class ConfigNLPAugmentation(DefaultConfig):
    nlp_augmentations_class: Any = BaseNLPAug
    token_mask_probability: float = 0
    skip_parent_probability: float = 0
    random_parent_probability: float = 0

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["token_mask_probability"] = (0.0, 0.9, 0.05)
        self._possible_values["skip_parent_probability"] = (0.0, 1.0, 0.05)
        self._possible_values["random_parent_probability"] = (0.0, 1.0, 0.05)
        self._visibility["nlp_augmentations_class"] = -1


@dataclass
class ConfigNLPCausalLMPrediction(DefaultConfig):
    metric_class: Any = text_causal_language_modeling_metrics.Metrics
    metric: str = "GPT"
    metric_gpt_model: str = "gpt-3.5-turbo-0301"

    min_length_inference: int = 2
    max_length_inference: int = 256
    batch_size_inference: int = 0

    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 0.3
    repetition_penalty: float = 1.2
    stop_tokens: str = ""
    top_k: int = 0
    top_p: float = 1.0

    num_history: int = 4

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["metric"] = self.metric_class.names()

        self._possible_values["metric_gpt_model"] = possible_values.String(
            values=(
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-0613",
                "gpt-4-0314",
                "gpt-4-0613",
            ),
            allow_custom=True,
        )

        self._possible_values["batch_size_inference"] = (0, 512, 1)
        self._possible_values["min_length_inference"] = (0, 1024, 1)
        self._possible_values["max_length_inference"] = (1, 1024, 1)

        self._possible_values["num_beams"] = (1, 4, 1)
        self._possible_values["temperature"] = (0, 10, 0.05)
        self._possible_values["repetition_penalty"] = (1, 10, 0.05)
        self._possible_values["top_k"] = (0, 100, 1)
        self._possible_values["top_p"] = (0.5, 1, 0.05)
        self._possible_values["num_history"] = (1, 50, 1)

        self._visibility["metric_class"] = -1
        # possible values for num_history are only used in chatbot tab
        self._visibility["num_history"] = -1

        self._nesting.add(
            ["metric_gpt_model"],
            [Dependency(key="metric", value="GPT", is_set=True)],
        )


@dataclass
class ConfigNLPCausalLMEnvironment(DefaultConfig):
    gpus: Tuple[str, ...] = tuple(str(x) for x in range(torch.cuda.device_count()))

    mixed_precision: bool = True

    compile_model: bool = False
    use_fsdp: bool = False

    find_unused_parameters: bool = False
    trust_remote_code: bool = True
    huggingface_branch: str = "main"
    number_of_workers: int = 4
    seed: int = -1

    _seed: int = 0  # internal seed set in train.py (equals seed if seed is not -1)
    _distributed: bool = False
    _distributed_inference: bool = True
    _local_rank: int = 0
    _world_size: int = 1
    _curr_step: int = 0
    _curr_val_step: int = 0
    _rank: int = 0  # global rank
    _device: str = "cuda"
    _cpu_comm: Any = None
    _model_card_template: str = "text_causal_language_modeling_model_card_template.md"
    _summary_card_template: str = (
        "text_causal_language_modeling_experiment_summary_card_template.md"
    )

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["gpus"] = possible_values.String(
            values=tuple(
                [(str(x), f"GPU #{x+1}") for x in range(torch.cuda.device_count())]
            ),
            allow_custom=False,
        )

        self._possible_values["number_of_workers"] = (1, multiprocessing.cpu_count(), 1)
        self._possible_values["seed"] = possible_values.Number(step=1, min=-1)


@dataclass
class ConfigNLPCausalLMLogging(DefaultConfig):
    logger: str = "None"
    neptune_project: str = ""
    _neptune_debug: bool = False

    plots_class: Any = text_causal_language_modeling_plots.Plots

    # the actual logger, will be set dynamically at runtime
    _logger: Any = None

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["logger"] = Loggers.names()

        self._nesting.add(
            ["neptune_project"],
            [Dependency(key="logger", value="Neptune", is_set=True)],
        )

        self._visibility["plots_class"] = -1


@dataclass
class ConfigProblemBase(DefaultConfig):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    _parent_experiment: str = ""
    llm_backbone: str = "EleutherAI/pythia-2.8b-deduped"

    dataset: ConfigNLPCausalLMDataset = field(default_factory=ConfigNLPCausalLMDataset)
    tokenizer: ConfigNLPCausalLMTokenizer = field(
        default_factory=ConfigNLPCausalLMTokenizer
    )
    architecture: ConfigNLPCausalLMArchitecture = field(
        default_factory=ConfigNLPCausalLMArchitecture
    )
    training: ConfigNLPCausalLMTraining = field(
        default_factory=ConfigNLPCausalLMTraining
    )
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
            dataset=ConfigNLPCausalLMDataset.from_dict(cfg_dict.get("dataset", {})),
            tokenizer=ConfigNLPCausalLMTokenizer.from_dict(
                cfg_dict.get("tokenizer", {})
            ),
            augmentation=ConfigNLPAugmentation.from_dict(
                cfg_dict.get("augmentation", {})
            ),
            architecture=ConfigNLPCausalLMArchitecture.from_dict(
                cfg_dict.get("architecture", {})
            ),
            training=ConfigNLPCausalLMTraining.from_dict(cfg_dict.get("training", {})),
            prediction=ConfigNLPCausalLMPrediction.from_dict(
                cfg_dict.get("prediction", {})
            ),
            environment=ConfigNLPCausalLMEnvironment.from_dict(
                cfg_dict.get("environment", {})
            ),
            logging=ConfigNLPCausalLMLogging.from_dict(cfg_dict.get("logging", {})),
        )
