import os

from llm_studio.src.utils.config_utils import load_config_yaml


def test_load_config_yaml():
    test_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    cfg_path = os.path.join(test_directory, "test_data/cfg.yaml")
    cfg = load_config_yaml(cfg_path)

    assert cfg.experiment_name == "test"
    assert cfg.llm_backbone == "EleutherAI/pythia-12b-deduped"
    assert cfg.output_directory == "output/user/test/"

    assert cfg.architecture.backbone_dtype == "float16"
    assert cfg.architecture.force_embedding_gradients is False
    assert cfg.architecture.gradient_checkpointing is False
    assert cfg.architecture.intermediate_dropout == 0.0

    assert cfg.augmentation.token_mask_probability == 0.0

    assert cfg.dataset.add_eos_token_to_answer is True
    assert cfg.dataset.add_eos_token_to_prompt is True
    assert cfg.dataset.answer_column == "output"
    assert cfg.dataset.data_sample == 0.1
    assert cfg.dataset.data_sample_choice == ["Train", "Validation"]
    assert cfg.dataset.mask_prompt_labels is False
    assert cfg.dataset.prompt_column == ("instruction",)
    assert cfg.dataset.text_answer_separator == "\\n"
    assert cfg.dataset.text_prompt_start == ""
    assert cfg.dataset.train_dataframe == "data/user/train/train.csv"
    assert cfg.dataset.validation_dataframe == "None"
    assert cfg.dataset.validation_size == 0.01
    assert cfg.dataset.validation_strategy == "automatic"

    assert cfg.environment.compile_model is False
    assert cfg.environment.find_unused_parameters is False
    assert cfg.environment.gpus == ["0"]
    assert cfg.environment.mixed_precision is True
    assert cfg.environment.number_of_workers == 8
    assert cfg.environment.seed == -1

    assert cfg.logging.logger == "None"
    assert cfg.logging.neptune_project == ""

    assert cfg.prediction.batch_size_inference == 0
    assert cfg.prediction.do_sample is False
    assert cfg.prediction.max_length_inference == 256
    assert cfg.prediction.min_length_inference == 2
    assert cfg.prediction.num_beams == 2
    assert cfg.prediction.repetition_penalty == 1.2
    assert cfg.prediction.stop_tokens == ""
    assert cfg.prediction.temperature == 0.3

    assert cfg.tokenizer.max_length == 144
    assert cfg.tokenizer.max_length_answer == 256
    assert cfg.tokenizer.max_length_prompt == 256
    assert cfg.tokenizer.padding_quantile == 1.0

    assert cfg.training.batch_size == 3
    assert cfg.training.epochs == 0
    assert cfg.training.evaluate_before_training is True
    assert cfg.training.evaluation_epochs == 1.0
    assert cfg.training.grad_accumulation == 1
    assert cfg.training.gradient_clip == 0.0
    assert cfg.training.learning_rate == 0.0001
    assert cfg.training.lora is True
    assert cfg.training.lora_alpha == 16
    assert cfg.training.lora_dropout == 0.05
    assert cfg.training.lora_r == 4
    assert cfg.training.lora_target_modules == ""
    assert cfg.training.optimizer == "AdamW"
    assert cfg.training.save_best_checkpoint is False
    assert cfg.training.schedule == "Cosine"
    assert cfg.training.train_validation_data is False
    assert cfg.training.warmup_epochs == 0.0
    assert cfg.training.weight_decay == 0.0
