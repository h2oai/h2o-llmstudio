import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

import llm_studio.src.datasets.text_causal_language_modeling_ds
from llm_studio.python_configs.base import DefaultConfig, DefaultConfigProblemBase
from llm_studio.src import possible_values
from llm_studio.src.augmentations.nlp_aug import BaseNLPAug
from llm_studio.src.loggers import Loggers
from llm_studio.src.losses import text_causal_language_modeling_losses
from llm_studio.src.metrics import text_causal_language_modeling_metrics
from llm_studio.src.models import text_causal_language_modeling_model
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

    use_flash_attention_2: bool = False
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

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["loss_function"] = self.loss_class.names()
        self._possible_values["optimizer"] = Optimizers.names()

        self._possible_values["learning_rate"] = possible_values.Number(
            step=0.000001, min=0.000001
        )
        self._possible_values["differential_learning_rate_layers"] = (
            possible_values.String(
                values=("backbone", "embed"),
                allow_custom=False,
                placeholder="Select optional layers...",
            )
        )
        self._possible_values["differential_learning_rate"] = self._possible_values[
            "learning_rate"
        ]

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

        self._visibility["loss_class"] = -1
        self._visibility["drop_last_batch"] = -1
        self._visibility["differential_learning_rate_layers"] = 1
        self._visibility["differential_learning_rate"] = 1

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


@dataclass
class ConfigNLPCausalLMTokenizer(DefaultConfig):
    max_length_prompt: int = 256
    max_length_answer: int = 256
    max_length: int = 512
    add_prompt_answer_tokens: bool = False
    padding_quantile: float = 1.0
    use_fast: bool = True

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["max_length_prompt"] = (32, 1024 * 16, 32)
        self._possible_values["max_length_answer"] = (32, 1024 * 16, 32)
        self._possible_values["max_length"] = (32, 1024 * 16, 32)
        self._possible_values["padding_quantile"] = (0, 1, 0.01)
        self._padding_side = "left"


@dataclass
class ConfigNLPCausalLMArchitecture(DefaultConfig):
    model_class: Any = text_causal_language_modeling_model.Model
    pretrained: bool = True

    backbone_dtype: str = "int4"
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
        self._visibility["pretrained"] = -1


@dataclass
class ConfigNLPAugmentation(DefaultConfig):
    nlp_augmentations_class: Any = BaseNLPAug
    token_mask_probability: float = 0.0
    skip_parent_probability: float = 0.0
    random_parent_probability: float = 0.0
    neftune_noise_alpha: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self._possible_values["token_mask_probability"] = (0.0, 0.9, 0.05)
        self._possible_values["skip_parent_probability"] = (0.0, 1.0, 0.05)
        self._possible_values["random_parent_probability"] = (0.0, 1.0, 0.05)
        self._possible_values["neftune_noise_alpha"] = (0.0, 15, 0.05)
        self._visibility["nlp_augmentations_class"] = -1


@dataclass
class ConfigNLPCausalLMPrediction(DefaultConfig):
    metric_class: Any = text_causal_language_modeling_metrics.Metrics
    metric: str = "GPT"
    metric_gpt_model: str = "gpt-3.5-turbo-0301"
    metric_gpt_template: str = "general"

    min_length_inference: int = 2
    max_length_inference: int = 256
    max_time: float = 120.0
    batch_size_inference: int = 0

    do_sample: bool = False
    num_beams: int = 1
    temperature: float = 0.0
    repetition_penalty: float = 1.0
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
                "gpt-4-1106-preview",
            ),
            allow_custom=True,
        )
        self._possible_values["metric_gpt_template"] = possible_values.String(
            values=(f.split(".")[0] for f in os.listdir("prompts"))
        )

        self._possible_values["batch_size_inference"] = (0, 512, 1)
        self._possible_values["min_length_inference"] = (0, 1024, 1)
        self._possible_values["max_length_inference"] = (1, 4096, 1)
        self._possible_values["max_time"] = (1.0, 600.0, 1.0)

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
            ["metric_gpt_model", "metric_gpt_template"],
            [Dependency(key="metric", value="GPT", is_set=True)],
        )


@dataclass
class ConfigNLPCausalLMEnvironment(DefaultConfig):
    gpus: Tuple[str, ...] = tuple(str(x) for x in range(torch.cuda.device_count()))

    mixed_precision: bool = True

    compile_model: bool = False
    use_deepspeed: bool = False
    deepspeed_method: str = "ZeRO2"
    deepspeed_allgather_bucket_size: int = int(1e6)
    deepspeed_reduce_bucket_size: int = int(1e6)
    deepspeed_stage3_prefetch_bucket_size: int = int(1e6)
    deepspeed_stage3_param_persistence_threshold: int = int(1e6)
    #     deepspeed_offload_optimizer: bool = False
    #     deepspeed_stage3_max_live_parameters: int = 1e9
    #     deepspeed_stage3_max_reuse_distance: int = 1e9

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
        self._possible_values["deepspeed_method"] = ["ZeRO2", "ZeRO3"]
        self._possible_values["deepspeed_allgather_bucket_size"] = (
            possible_values.Number(step=1, min=1e6)
        )
        self._possible_values["deepspeed_reduce_bucket_size"] = possible_values.Number(
            step=1, min=1e6
        )
        self._possible_values["deepspeed_stage3_prefetch_bucket_size"] = (
            possible_values.Number(step=1, min=1e6)
        )
        self._possible_values["deepspeed_stage3_param_persistence_threshold"] = (
            possible_values.Number(step=1, min=1e6)
        )
        self._possible_values["deepspeed_stage3_max_live_parameters"] = (
            possible_values.Number(step=1, min=1e6)
        )
        self._possible_values["deepspeed_stage3_max_reuse_distance"] = (
            possible_values.Number(step=1, min=1e6)
        )
        self._nesting.add(
            [
                "deepspeed_method",
                "deepspeed_reduce_bucket_size",
            ],
            [Dependency(key="use_deepspeed", value=True, is_set=True)],
        )
        self._nesting.add(
            [
                "deepspeed_allgather_bucket_size",
            ],
            [
                Dependency(key="use_deepspeed", value=True, is_set=True),
                Dependency(key="deepspeed_method", value="ZeRO2", is_set=True),
            ],
        )
        self._nesting.add(
            [
                "deepspeed_stage3_prefetch_bucket_size",
                "deepspeed_stage3_param_persistence_threshold",
                # "deepspeed_offload_optimizer",
            ],
            [
                Dependency(key="use_deepspeed", value=True, is_set=True),
                Dependency(key="deepspeed_method", value="ZeRO3", is_set=True),
            ],
        )
        # self._nesting.add(
        #     [
        #         "deepspeed_stage3_max_live_parameters",
        #         "deepspeed_stage3_max_reuse_distance",
        #     ],
        #     [Dependency(key="deepspeed_offload_optimizer", value=False, is_set=False)],  # noqa: E501
        # )


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
class ConfigProblemBase(DefaultConfigProblemBase):
    output_directory: str = f"output/{os.path.basename(__file__).split('.')[0]}"
    experiment_name: str = field(default_factory=generate_experiment_name)
    _parent_experiment: str = ""
    llm_backbone: str = "h2oai/h2ogpt-4096-llama2-7b"

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
                "h2oai/h2o-danube-1.8b-base",
                "h2oai/h2o-danube-1.8b-chat",
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

    def check(self) -> Dict[str, List]:
        errors: Dict[str, List] = {"title": [], "message": []}
        if self.prediction.temperature > 0 and not self.prediction.do_sample:
            errors["title"] += ["Do sample needs to be enabled for temperature > 0"]
            errors["message"] += [
                "Please enable do sample if you want to use temperature > 0."
            ]
        if self.prediction.temperature == 0 and self.prediction.do_sample:
            errors["title"] += ["Temperature needs to be > 0 for do sample"]
            errors["message"] += [
                "Please increase temperature if you want to use do sample."
            ]
        return errors
