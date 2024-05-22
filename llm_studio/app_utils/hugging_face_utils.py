import os

import accelerate
import einops
import huggingface_hub
import torch
import transformers

from llm_studio.app_utils.sections.chat import load_cfg_model_tokenizer
from llm_studio.app_utils.utils import hf_repo_friendly_name, save_hf_yaml, set_env
from llm_studio.src.utils.config_utils import (
    GENERATION_PROBLEM_TYPES,
    NON_GENERATION_PROBLEM_TYPES,
)
from llm_studio.src.utils.modeling_utils import check_disk_space


def get_model_card(cfg, model, repo_id) -> huggingface_hub.ModelCard:
    """
    Method to define the Model Card.

    It is possible to change the language, the library name, and the tags.
    These values will appear in the Model Card tab of Hugging Face.

    Parameters:
        cfg : Configuration parameters for the model card.
        model : The model for which the model card is being generated.
        repo_id : The ID of the target Hugging Face repository.

    Returns:
        huggingface_hub.ModelCard: The Model Card containing model information.
    """
    card_data = huggingface_hub.ModelCardData(
        language="en",
        library_name="transformers",
        tags=["gpt", "llm", "large language model", "h2o-llmstudio"],
    )
    cfg_kwargs = dict(
        text_prompt_start=cfg.dataset.text_prompt_start,
        text_answer_separator=cfg.dataset.text_answer_separator,
        trust_remote_code=cfg.environment.trust_remote_code,
        end_of_sentence=(
            cfg._tokenizer_eos_token if cfg.dataset.add_eos_token_to_prompt else ""
        ),
    )
    if cfg.problem_type not in NON_GENERATION_PROBLEM_TYPES:
        cfg_kwargs.update(
            dict(
                min_new_tokens=cfg.prediction.min_length_inference,
                max_new_tokens=cfg.prediction.max_length_inference,
                do_sample=cfg.prediction.do_sample,
                num_beams=cfg.prediction.num_beams,
                temperature=cfg.prediction.temperature,
                repetition_penalty=cfg.prediction.repetition_penalty,
            )
        )
        if cfg.dataset.system_column != "None":
            cfg_kwargs[
                "sample_messages"
            ] = """[
    {
        "role": "system",
        "content": "You are a friendly and polite chatbot.",
    },
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "I'm doing great, how about you?"},
    {"role": "user", "content": "Why is drinking water so healthy?"},
]"""
        else:
            cfg_kwargs[
                "sample_messages"
            ] = """[
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "I'm doing great, how about you?"},
    {"role": "user", "content": "Why is drinking water so healthy?"},
]"""

    card = huggingface_hub.ModelCard.from_template(
        card_data,
        template_path=os.path.join("model_cards", cfg.environment._model_card_template),
        base_model=cfg.llm_backbone,  # will be replaced in template if it exists
        repo_id=repo_id,
        model_architecture=model.backbone.__repr__(),
        config=cfg.__repr__(),
        transformers_version=transformers.__version__,
        einops_version=einops.__version__,
        accelerate_version=accelerate.__version__,
        torch_version=torch.__version__.split("+")[0],
        **cfg_kwargs,
    )
    return card


def get_chat_template(cfg):

    chat_template = """
{% for message in messages %}
chat_template_for_checking_system_role
chat_template_for_checking_alternating_roles
{% if message['role'] == 'user' %}
{{ 'text_prompt_start' + message['content'].strip() + eos_token_prompt }}
chat_template_for_system
{% elif message['role'] == 'assistant' %}
{{ 'text_answer_separator' + message['content'].strip() + eos_token_answer }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}{{ 'text_answer_separator' }}{% endif %}"""

    if cfg.dataset.system_column != "None":
        # If system role is supported
        chat_template = chat_template.replace(
            "chat_template_for_checking_system_role", ""
        )
        chat_template = chat_template.replace(
            "chat_template_for_checking_alternating_roles",
            """
{% if loop.index0 != 0 and message['role'] == 'system' %}
{{ raise_exception('Conversation roles must alternate system(optional)/user/assistant/user/assistant/...') }}"""  # noqa
            + """
{% elif messages[0]['role'] == 'system' and ((message['role'] == 'user' and (loop.index0 % 2 == 0)) or (message['role'] == 'assistant' and (loop.index0 % 2 == 1))) %}"""  # noqa
            + """
{{ raise_exception('Conversation roles must alternate system(optional)/user/assistant/user/assistant/...') }}"""  # noqa
            + """
{% elif messages[0]['role'] != 'system' and ((message['role'] == 'user' and (loop.index0 % 2 != 0)) or (message['role'] == 'assistant' and (loop.index0 % 2 != 1))) %}"""  # noqa
            + """
{{ raise_exception('Conversation roles must alternate system(optional)/user/assistant/user/assistant/...') }}"""  # noqa
            + """
{% endif %}""",
        )
        chat_template = chat_template.replace(
            "chat_template_for_system",
            """
{% elif message['role'] == 'system' %}
{{ 'text_system_start' + message['content'].strip() + eos_token_system }}""",
        )
        if cfg.dataset.add_eos_token_to_system:
            chat_template = chat_template.replace("eos_token_system", "eos_token")
        else:
            chat_template = chat_template.replace("+ eos_token_system", "")
    else:
        # If system role is NOT supported
        chat_template = chat_template.replace(
            "chat_template_for_checking_system_role",
            """
{% if message['role'] == 'system' %}
{{ raise_exception('System role not supported') }}
{% endif %}""",
        )
        chat_template = chat_template.replace(
            "chat_template_for_checking_alternating_roles",
            """
{% if ((message['role'] == 'user') != (loop.index0 % 2 == 0)) or ((message['role'] == 'assistant') != (loop.index0 % 2 == 1)) %}"""  # noqa
            + """
{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"""  # noqa
            + """
{% endif %}""",
        )
        chat_template = chat_template.replace("chat_template_for_system", "")

    if cfg.dataset.add_eos_token_to_prompt:
        chat_template = chat_template.replace("eos_token_prompt", "eos_token")
    else:
        chat_template = chat_template.replace("+ eos_token_prompt", "")
    if cfg.dataset.add_eos_token_to_answer:
        chat_template = chat_template.replace("eos_token_answer", "eos_token")
    else:
        chat_template = chat_template.replace("+ eos_token_answer", "")

    chat_template = (
        chat_template.replace("\n", "")
        .replace("text_system_start", cfg.dataset.text_system_start)
        .replace("text_prompt_start", cfg.dataset.text_prompt_start)
        .replace("text_answer_separator", cfg.dataset.text_answer_separator)
    )

    return chat_template


def publish_model_to_hugging_face(
    path_to_experiment: str,
    model_name: str,
    user_id: str = None,
    api_key: str = None,
    device: str = "cuda:0",
    safe_serialization: bool = True,
) -> None:
    """
    Method to publish the model to Hugging Face.

    Parameters:
        path_to_experiment: The file path of the fine-tuned model's files.
        device: The target device for running the model, either 'cpu', 'cpu_shard'
            or 'cuda:0'.
        user_id: The Hugging Face user ID.
        api_key: The Hugging Face API Key.
        model_name: The name of the model to be published on Hugging Face.
        safe_serialization: A flag indicating whether safe serialization should be used.

    Returns:
        None. The model is published to the specified Hugging Face repository.
    """

    # Check if the 'device' value is valid, raise an exception if not
    if device == "cpu" or device == "cpu_shard":
        pass  # 'cpu' is a valid value
    elif device.startswith("cuda:") and device[5:].isdigit():
        pass  # 'cuda:integer' format is valid
    else:
        raise ValueError(
            "Invalid device value. Use 'cpu', 'cpu_shard' or 'cuda:INTEGER'."
        )

    with set_env(HUGGINGFACE_TOKEN=api_key):
        cfg, model, tokenizer = load_cfg_model_tokenizer(
            path_to_experiment,
            merge=True,
            device=device,
        )

    check_disk_space(model.backbone, "./")

    # Check if the user is already logged in, and if not, prompt for API key
    if api_key:
        huggingface_hub.login(api_key)

    # If 'user_id' argument is blank, fetch 'user_id' from the logged-in user
    if user_id == "":
        user_id = huggingface_hub.whoami()["name"]

    repo_id = f"{user_id}/{hf_repo_friendly_name(model_name)}"

    # push tokenizer to hub
    if cfg.problem_type in GENERATION_PROBLEM_TYPES:
        tokenizer.chat_template = get_chat_template(cfg)
    tokenizer.push_to_hub(repo_id=repo_id, private=True)

    # push model card to hub
    card = get_model_card(cfg, model, repo_id)
    card.push_to_hub(
        repo_id=repo_id, repo_type="model", commit_message="Upload model card"
    )

    api = huggingface_hub.HfApi()

    # push classification head to hub
    if os.path.isfile(f"{path_to_experiment}/classification_head.pth"):
        api.upload_file(
            path_or_fileobj=f"{path_to_experiment}/classification_head.pth",
            path_in_repo="classification_head.pth",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload classification_head.pth",
        )

    # push config to hub
    api.upload_file(
        path_or_fileobj=os.path.join(path_to_experiment, "cfg.yaml"),
        path_in_repo="cfg.yaml",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload cfg.yaml",
    )

    # push model to hub
    model.backbone.push_to_hub(
        repo_id=repo_id,
        private=True,
        commit_message="Upload model",
        safe_serialization=safe_serialization,
    )

    # Storing HF attributes
    output_directory = cfg.output_directory
    save_hf_yaml(
        path=f"{output_directory.rstrip('/')}/hf.yaml",
        account_name=user_id,
        model_name=model_name,
        repo_id=repo_id,
    )
