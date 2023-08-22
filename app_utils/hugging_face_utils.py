import os

import accelerate
import einops
import huggingface_hub
import torch
import transformers
from jinja2 import Environment, FileSystemLoader

from app_utils.sections.chat import load_cfg_model_tokenizer
from app_utils.utils import hf_repo_friendly_name, save_hf_yaml, set_env
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
    card = huggingface_hub.ModelCard.from_template(
        card_data,
        template_path=os.path.join("model_cards", cfg.environment._model_card_template),
        base_model=cfg.llm_backbone,  # will be replaced in template if it exists
        repo_id=repo_id,
        model_architecture=model.backbone.__repr__(),
        config=cfg.__repr__(),
        use_fast=cfg.tokenizer.use_fast,
        min_new_tokens=cfg.prediction.min_length_inference,
        max_new_tokens=cfg.prediction.max_length_inference,
        do_sample=cfg.prediction.do_sample,
        num_beams=cfg.prediction.num_beams,
        temperature=cfg.prediction.temperature,
        repetition_penalty=cfg.prediction.repetition_penalty,
        text_prompt_start=cfg.dataset.text_prompt_start,
        text_answer_separator=cfg.dataset.text_answer_separator,
        trust_remote_code=cfg.environment.trust_remote_code,
        transformers_version=transformers.__version__,
        einops_version=einops.__version__,
        accelerate_version=accelerate.__version__,
        torch_version=torch.__version__.split("+")[0],
        end_of_sentence=cfg._tokenizer_eos_token
        if cfg.dataset.add_eos_token_to_prompt
        else "",
    )
    return card


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
        device: The target device for running the model, either 'cpu' or 'cuda:0'.
        user_id: The Hugging Face user ID.
        api_key: The Hugging Face API Key.
        model_name: The name of the model to be published on Hugging Face.
        safe_serialization: A flag indicating whether safe serialization should be used.

    Returns:
        None. The model is published to the specified Hugging Face repository.
    """

    # Check if the 'device' value is valid, raise an exception if not
    if device == "cpu":
        pass  # 'cpu' is a valid value
    elif device.startswith("cuda:") and device[5:].isdigit():
        pass  # 'cuda:integer' format is valid
    else:
        raise ValueError("Invalid device value. Use 'cpu' or 'cuda:INTEGER'.")

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
    tokenizer.push_to_hub(repo_id=repo_id, private=True)

    # push model card to hub
    card = get_model_card(cfg, model, repo_id)
    card.push_to_hub(
        repo_id=repo_id, repo_type="model", commit_message="Upload model card"
    )

    # push config to hub
    api = huggingface_hub.HfApi()
    api.upload_file(
        path_or_fileobj=os.path.join(path_to_experiment, "cfg.yaml"),
        path_in_repo="cfg.yaml",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload cfg.yaml",
    )

    # push model to hub
    model.backbone.config.custom_pipelines = {
        "text-generation": {
            "impl": "h2oai_pipeline.H2OTextGenerationPipeline",
            "pt": "AutoModelForCausalLM",
        }
    }

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

    # push pipeline to hub
    template_env = Environment(loader=FileSystemLoader(searchpath="llm_studio/src/"))

    pipeline_template = template_env.get_template("h2oai_pipeline_template.py")

    data = {
        "text_prompt_start": cfg.dataset.text_prompt_start,
        "text_answer_separator": cfg.dataset.text_answer_separator,
    }

    if cfg.dataset.add_eos_token_to_prompt:
        data.update({"end_of_sentence": cfg._tokenizer_eos_token})
    else:
        data.update({"end_of_sentence": ""})

    custom_pipeline = pipeline_template.render(data)

    custom_pipeline_path = os.path.join(path_to_experiment, "h2oai_pipeline.py")

    with open(custom_pipeline_path, "w") as f:
        f.write(custom_pipeline)

    api.upload_file(
        path_or_fileobj=custom_pipeline_path,
        path_in_repo="h2oai_pipeline.py",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload h2oai_pipeline.py",
    )
