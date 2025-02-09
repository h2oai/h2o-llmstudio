import os
import socket
from types import SimpleNamespace

import toml
from huggingface_hub.constants import _is_true

toml_root_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
)
app_toml_filename = os.path.join(toml_root_dir, "pyproject.toml")

toml_loaded = toml.load(app_toml_filename)

version = toml_loaded["project"]["version"]


def get_size(x):
    try:
        if x.endswith("TB"):
            return float(x.replace("TB", "")) * (2**40)
        if x.endswith("GB"):
            return float(x.replace("GB", "")) * (2**30)
        if x.endswith("MB"):
            return float(x.replace("MB", "")) * (2**20)
        if x.endswith("KB"):
            return float(x.replace("KB", "")) * (2**10)
        if x.endswith("B"):
            return float(x.replace("B", ""))
        return 2**31
    except Exception:
        return 2**31


try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    host = s.getsockname()[0]
    s.close()
except OSError:
    host = "localhost"

port = "10101"
url = f"http://{host}:{port}/"


if os.getenv("H2O_LLM_STUDIO_DEFAULT_LM_MODELS"):
    default_causal_language_models = [
        mdl.strip() for mdl in os.getenv("H2O_LLM_STUDIO_DEFAULT_LM_MODELS").split(",")
    ]
else:
    default_causal_language_models = [
        "h2oai/h2o-danube3-500m-base",
        "h2oai/h2o-danube3-500m-chat",
        "h2oai/h2o-danube3-4b-base",
        "h2oai/h2o-danube3-4b-chat",
        "h2oai/h2o-danube2-1.8b-base",
        "h2oai/h2o-danube2-1.8b-chat",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
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
    ]

if os.getenv("H2O_LLM_STUDIO_DEFAULT_S2S_MODELS"):
    default_sequence_to_sequence_models = [
        mdl.strip() for mdl in os.getenv("H2O_LLM_STUDIO_DEFAULT_S2S_MODELS").split(",")
    ]
else:
    default_sequence_to_sequence_models = [
        "t5-small",
        "t5-base",
        "t5-large",
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-ul2",
    ]

default_cfg = {
    "url": url,
    "name": "H2O LLM Studio",
    "version": version,
    "github": "https://github.com/h2oai/h2o-llmstudio",
    "min_experiment_disk_space": get_size(
        os.getenv("MIN_DISK_SPACE_FOR_EXPERIMENTS", "2GB")
    ),
    "allowed_file_extensions": os.getenv(
        "ALLOWED_FILE_EXTENSIONS", ".zip,.csv,.pq,.parquet"
    ).split(","),
    "llm_studio_workdir": f"{os.getenv('H2O_LLM_STUDIO_WORKDIR', os.getcwd())}",
    "heap_mode": os.getenv("H2O_LLM_STUDIO_ENABLE_HEAP", "False") == "True",
    "data_folder": "data/",
    "output_folder": "output/",
    "cfg_file": "text_causal_language_modeling_config",
    "start_page": "home",
    "problem_types": [
        "text_causal_language_modeling_config",
        "text_causal_classification_modeling_config",
        "text_causal_regression_modeling_config",
        "text_sequence_to_sequence_modeling_config",
        "text_dpo_modeling_config",
    ],
    "default_causal_language_models": default_causal_language_models,
    "default_sequence_to_sequence_models": default_sequence_to_sequence_models,
    "problem_categories": ["text"],
    "dataset_keys": [
        "train_dataframe",
        "validation_dataframe",
        "system_column",
        "prompt_column",
        "rejected_prompt_column",
        "answer_column",
        "rejected_answer_column",
        "parent_id_column",
        "id_column",
    ],
    "dataset_trigger_keys": [
        "train_dataframe",
        "validation_dataframe",
        "parent_id_column",
    ],
    "dataset_extra_keys": [
        "validation_strategy",
        "data_sample",
        "data_sample_choice",
    ],
    "dataset_folder_keys": [
        "train_dataframe",
        "validation_dataframe",
    ],
    "user_settings": {
        "credential_saver": ".env File",
        "default_aws_bucket_name": f"{os.getenv('AWS_BUCKET', 'bucket_name')}",
        "default_aws_access_key": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "default_aws_secret_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        "default_azure_conn_string": "",
        "default_azure_container": "",
        "default_kaggle_username": "",
        "default_kaggle_secret_key": "",
        "set_max_epochs": 50,
        "set_max_batch_size": 256,
        "set_max_num_classes": 100,
        "set_max_max_length": 16384,
        "set_max_gradient_clip": 10,
        "set_max_lora_r": 256,
        "set_max_lora_alpha": 256,
        "gpu_used_for_download": "cuda:0",
        "gpu_used_for_chat": 1,
        "default_number_of_workers": 8,
        "default_logger": "None",
        "default_neptune_project": os.getenv("NEPTUNE_PROJECT", ""),
        "default_neptune_api_token": os.getenv("NEPTUNE_API_TOKEN", ""),
        "default_wandb_api_token": os.getenv("WANDB_API_KEY", ""),
        "default_wandb_project": os.getenv("WANDB_PROJECT", ""),
        "default_wandb_entity": os.getenv("WANDB_ENTITY", ""),
        "default_huggingface_api_token": os.getenv("HF_TOKEN", ""),
        "default_hf_hub_enable_hf_transfer": _is_true(
            os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
        ),
        "default_openai_azure": os.getenv("OPENAI_API_TYPE", "open_ai") == "azure",
        "default_openai_api_token": os.getenv("OPENAI_API_KEY", ""),
        "default_openai_api_base": os.getenv(
            "OPENAI_API_BASE", "https://example-endpoint.openai.azure.com"
        ),
        "default_openai_api_deployment_id": os.getenv(
            "OPENAI_API_DEPLOYMENT_ID", "deployment-name"
        ),
        "default_openai_api_version": os.getenv("OPENAI_API_VERSION", "2023-05-15"),
        "default_gpt_eval_max": os.getenv("GPT_EVAL_MAX", 100),
        "default_safe_serialization": True,
        "delete_dialogs": True,
        "chart_plot_max_points": 1000,
    },
}

default_cfg = SimpleNamespace(**default_cfg)
