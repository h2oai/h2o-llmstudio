import gc
import logging
import os
import re
import shutil
from collections import OrderedDict
from typing import Any, Dict

import coolname
import deepspeed
import numpy as np
import torch
import transformers
from deepspeed.runtime.dataloader import DeepSpeedDataLoader
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig, PeftModel, get_peft_model
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    BitsAndBytesConfig,
    GenerationMixin,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.pytorch_utils import Conv1D as Conv1DTransformer
from transformers.utils import logging as transformers_logging

from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.optimizers import Optimizers
from llm_studio.src.schedulers import Schedulers
from llm_studio.src.utils.config_utils import NON_GENERATION_PROBLEM_TYPES
from llm_studio.src.utils.data_utils import (
    OrderedDistributedSampler,
    batch_padding,
    cat_batches,
    get_inference_batch_size,
)
from llm_studio.src.utils.exceptions import LLMDataException, LLMModelException
from llm_studio.src.utils.logging_utils import TqdmToLogger
from llm_studio.src.utils.utils import save_pickle

logger = logging.getLogger(__name__)


def unwrap_model(model: torch.nn.Module):
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)

    while isinstance(model, options):
        model = model.module

    return model


def check_disk_space(model: torch.nn.Module, path: str):
    total, used, free = shutil.disk_usage(path)

    model_size_in_bytes = 0
    for param in model.parameters():
        n_params = param.ds_numel if hasattr(param, "ds_numel") else param.numel()
        if param.data.dtype in [torch.int8, torch.uint8]:
            model_size_in_bytes += n_params * 1
        elif param.data.dtype in [torch.float16, torch.bfloat16]:
            model_size_in_bytes += n_params * 2
        elif param.data.dtype == torch.float32:
            model_size_in_bytes += n_params * 4
        else:
            # If the data type is not supported, calculate it as float32.
            model_size_in_bytes += n_params * 4
            logger.warning(f"Unsupported data type: {param.data.dtype}")

    if model_size_in_bytes * 1.03 < free:  # leave a 3% margin here.
        logger.info(
            "Enough space available for saving model weights."
            f"Required space: {model_size_in_bytes * 1.03 / (1024 * 1024):.2f}MB, "
            f"Available space: {free / (1024 * 1024):.2f}MB."
        )
    else:
        raise ValueError(
            f"Not enough space available for saving model weights. "
            f"Required space: {model_size_in_bytes * 1.03 / (1024 * 1024):.2f}MB, "
            f"Available space: {free / (1024 * 1024):.2f}MB."
        )


# TODO: currently not saving optimizer
def save_checkpoint(model: torch.nn.Module, path: str, cfg: Any):
    """Saves a model checkpoint if the path is provided.

    Args:
        model: model to save
        path: path to save the checkpoint to

    Returns:
        Dictionary with all the keys to save
    """

    if cfg.environment.use_deepspeed:
        if path is not None:
            # gather model params from all ranks when using Deepspeed
            status = model.save_16bit_model(path, "checkpoint.pth")  # type: ignore
            if status:
                if cfg.environment._local_rank == 0:
                    checkpoint = {
                        "model": torch.load(
                            os.path.join(path, "checkpoint.pth"), map_location="cpu"
                        )
                    }
            else:
                logger.warning(
                    "deepspeed.save_16bit_model didn't save the model, since"
                    " stage3_gather_16bit_weights_on_model_save=False."
                    " Saving the full checkpoint instead"
                )
                model.save_checkpoint(  # type: ignore
                    os.path.join(path, "ds_checkpoint")
                )
                if cfg.environment._local_rank == 0:
                    # load to cpu
                    state_dict = get_fp32_state_dict_from_zero_checkpoint(
                        os.path.join(path, "ds_checkpoint")
                    )
                    # save as normal checkpoint that can be loaded by `load_state_dict`
                    checkpoint = {"model": state_dict}
                    torch.save(checkpoint, os.path.join(path, "checkpoint.pth"))
                    shutil.rmtree(os.path.join(path, "ds_checkpoint"))

    else:
        if cfg.environment._local_rank == 0:
            model = unwrap_model(model)
            checkpoint = {"model": model.state_dict()}
            if path is not None:
                torch.save(checkpoint, os.path.join(path, "checkpoint.pth"))

    if (
        cfg.environment._local_rank == 0
        and "classification_head.weight" in checkpoint["model"]
    ):
        torch.save(
            checkpoint["model"]["classification_head.weight"],
            os.path.join(path, "classification_head.pth"),
        )


def load_model_weights(
    model: torch.nn.Module, model_weights: Dict, strict: bool, cfg: Any
):
    orig_num_items = len(model_weights)
    model_state_dict = model.state_dict()

    # needed to load models trained in int4/int8 with other dtypes
    model_weights = {
        k: (
            v
            if not (
                cfg.architecture.backbone_dtype not in ("int4", "int8")
                and (v.dtype is torch.int8 or v.dtype is torch.uint8)
            )
            else model_state_dict[k]
        )
        for k, v in model_weights.items()
        if not (
            ("SCB" in k or "weight_format" in k or "quant_state" in k)
            and cfg.architecture.backbone_dtype not in ("int4", "int8")
        )
    }

    # Need to ignore int4/int8 weights so undo strict loading requirement
    if len(model_weights) != orig_num_items:
        strict = False

    model_weights = {re.sub(r"^module\.", "", k): v for k, v in model_weights.items()}
    model_weights = {k.replace("_orig_mod.", ""): v for k, v in model_weights.items()}

    # manual fix for int8 weights
    if cfg.architecture.backbone_dtype == "int8":
        model_weights = {
            k: v.to(cfg.environment._device) if "weight_format" not in k else v
            for k, v in model_weights.items()
        }

    try:
        model.load_state_dict(OrderedDict(model_weights), strict=True)
    except Exception as e:
        if strict:
            raise e
        else:
            if cfg.environment._local_rank == 0:
                logger.warning(
                    "Only a part of the pretrained weights was loaded. "
                    "Some layers can't be initialized with pretrained "
                    f"weights: {e}"
                )

            for layer_name in re.findall("size mismatch for (.*?):", str(e)):
                model_weights.pop(layer_name, None)
            model.load_state_dict(OrderedDict(model_weights), strict=False)
    return model


def load_checkpoint(
    cfg: Any, model: torch.nn.Module, strict: bool = True, weights_path: str = None
):
    """Load checkpoint

    Args:
        cfg: config file
        model: model to load weights to
        strict: whether to apply strict matching for weights
        weights_path: custom path to the weights.
            If None, cfg.architecture.pretrained_weights is used
    Returns:
        epoch: current epoch
    """

    if weights_path is None:
        weights_path = cfg.architecture.pretrained_weights

    model_weights = torch.load(weights_path, map_location="cpu")
    if "model" in model_weights.keys():
        model_weights = model_weights["model"]

    if cfg.environment.use_deepspeed:
        if cfg.training.lora:
            model.backbone.base_model.model = load_model_weights(  # type: ignore
                model.backbone.base_model.model,  # type: ignore
                model_weights,
                strict,
                cfg,
            )
        else:
            model.backbone = load_model_weights(
                model.backbone, model_weights, strict, cfg  # type: ignore
            )
    else:
        model = load_model_weights(model, model_weights, strict, cfg)

    del model_weights
    gc.collect()

    if cfg.environment._local_rank == 0:
        logger.info(f"Weights loaded from: {weights_path}")


def get_ds_config(cfg: Any):
    ds_config = {
        "fp16": {
            "enabled": True if cfg.architecture.backbone_dtype == "float16" else False,
            "loss_scale_window": 100,
        },
        "bf16": {
            "enabled": True if cfg.architecture.backbone_dtype == "bfloat16" else False,
            "loss_scale_window": 100,
        },
        # https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training
        "zero_force_ds_cpu_optimizer": False,
        "zero_optimization": {
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": cfg.environment.deepspeed_reduce_bucket_size,
            # zero3 offload cpu
            # "stage3_max_live_parameters": cfg.environment.deepspeed_stage3_max_live_parameters,  # noqa: E501
            # "stage3_max_reuse_distance": cfg.environment.deepspeed_stage3_max_reuse_distance,  # noqa: E501
            # zero++
            # "reduce_scatter": True,
            # "zero_quantized_weights": True,
            # "zero_hpz_partition_size": 16,
            # "zero_quantized_gradients": True,
        },
        "steps_per_print": 2000,
        "train_micro_batch_size_per_gpu": cfg.training.batch_size,
        "gradient_accumulation_steps": cfg.training.grad_accumulation,
        "wall_clock_breakdown": False,
    }

    if cfg.environment.deepspeed_method == "ZeRO2":
        ds_config["zero_optimization"]["stage"] = 2
        ds_config["zero_optimization"]["allgather_partitions"] = True
        ds_config["zero_optimization"][
            "allgather_bucket_size"
        ] = cfg.environment.deepspeed_allgather_bucket_size
    elif cfg.environment.deepspeed_method == "ZeRO3":
        ds_config["zero_optimization"]["stage"] = 3
        ds_config["zero_optimization"][
            "stage3_prefetch_bucket_size"
        ] = cfg.environment.deepspeed_stage3_prefetch_bucket_size
        ds_config["zero_optimization"][
            "stage3_param_persistence_threshold"
        ] = cfg.environment.deepspeed_stage3_param_persistence_threshold
        ds_config["zero_optimization"][
            "stage3_gather_16bit_weights_on_model_save"
        ] = True

    # TODO: Do not enable offload cpu for now.
    # if cfg.environment.deepspeed_offload_optimizer:
    #     ds_config["zero_optimization"]["offload_optimizer"] = {
    #         "device": "cpu",
    #         "pin_memory": True,
    #     }
    # TODO: RuntimeError: Tensors must be CUDA and dense
    # if cfg.environment.deepspeed_offload_param:
    #     ds_config["zero_optimization"]["offload_param"] =
    #         {"device": "cpu", "pin_memory": True}

    logger.info(f"DeepSpeed config: {ds_config}")

    return ds_config


def wrap_model_distributed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    cfg: Any,
):
    if cfg.environment.use_deepspeed:
        ds_config = get_ds_config(cfg)
        if not cfg.training.lora:
            ds_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
                model=model.backbone,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                training_data=train_dataloader.dataset,
                config_params=ds_config,
            )
            model.backbone = ds_engine
        else:
            ds_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
                model=model.backbone.base_model.model,  # type: ignore
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                training_data=train_dataloader.dataset,
                config_params=ds_config,
            )
            model.backbone.base_model.model = ds_engine  # type: ignore
        model.init_deepspeed()  # type: ignore
        val_dataloader = DeepSpeedDataLoader(
            val_dataloader.dataset,
            batch_size=val_dataloader.batch_size,
            local_rank=cfg.environment._local_rank,
            pin_memory=True,
            tput_timer=None,
            data_sampler=OrderedDistributedSampler(
                val_dataloader.dataset,
                num_replicas=cfg.environment._world_size,
                rank=cfg.environment._local_rank,
            ),
        )
    else:
        find_unused_parameters = cfg.environment.find_unused_parameters
        if getattr(cfg.architecture, "gradient_checkpointing", None):
            find_unused_parameters = False
        model = DistributedDataParallel(
            model,
            device_ids=[cfg.environment._local_rank],
            find_unused_parameters=find_unused_parameters,
        )

    return model, optimizer, train_dataloader, val_dataloader, lr_scheduler


def get_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Prepares Optimizer.

    Args:
        model: model
        cfg: input config

    Returns:
        Optimizer
    """
    no_decay = ["bias", "LayerNorm.weight"]
    differential_layers = cfg.training.differential_learning_rate_layers
    optimizer = Optimizers.get(cfg.training.optimizer)(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                    and param.requires_grad
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                    and param.requires_grad
                ],
                "lr": cfg.training.learning_rate,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                    and param.requires_grad
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                    and param.requires_grad
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": 0,
            },
        ],
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    return optimizer


def get_scheduler(
    cfg: Any, optimizer: torch.optim.Optimizer, epoch_steps: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Prepares Learning Rate Scheduler.

    Args:
        cfg: input config
        optimizer: model optimizer
        epoch_steps: total number of weight updates during the epoch

    Returns:
        Learning Rate Scheduler
    """

    scheduler = Schedulers.get(cfg.training.schedule)(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.warmup_epochs * epoch_steps,
        num_training_steps=cfg.training.epochs * epoch_steps,
    )

    return scheduler


def generate_experiment_name() -> str:
    """
    Generates a random human-readable experiment name in kebab-case.

    Returns:
        The random name.
    """
    return coolname.generate_slug(2)


def reduce_metric(output, reduce=None) -> float:
    """Reduces metric and return metric score (number)

    Args:
        output: output of the model
        reduce: how to reduce the metric over the sample dimension

    Returns:
        score: single number score (using config threshold for threshold metrics)
        or non-reduced array of scores per sample.
    """

    if reduce == "mean":
        score = np.mean(output["metrics"])
    else:
        raise NotImplementedError()

    return score


def get_number_of_validation_epochs(training_epochs: int, evaluation_epochs: float):
    """
    Given the number of training epochs and the number of epochs between model
    evaluations, return the number of times the model is being evaluated during
    training

    Args:
        training_epochs: The number of epochs to train for
        evaluation_epochs: This is the number of epochs after which we want to
            evaluate our model

    Returns:
        num_val_epochs: The number of epochs to be evaluated during training.
    """
    return training_epochs // evaluation_epochs


def contains_nan(output: Dict):
    return (
        sum(
            [
                1
                for key, val in output.items()
                if isinstance(val, torch.Tensor)
                and torch.isnan(val.detach().cpu()).sum() > 0
            ]
        )
        > 0
    )


def run_inference(
    cfg: Any,
    model: torch.nn.Module,
    dataloader,
    mode: str,
) -> Dict[str, list]:
    """Runs inference

    Args:
        cfg: config
        model: model
        dataloader: custom dataloader
        mode: mode for inference

    Returns:
        Dictionary with output

    """

    # Store information for evaluation
    out = dict()

    if cfg.environment._local_rank == 0:
        logger.info(f"Starting {mode} inference")

    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    progress_bar = tqdm(
        total=len(dataloader),
        disable=cfg.environment._local_rank != 0,
        file=tqdm_out,
        ascii=True,
        desc=f"{mode} progress",
        mininterval=0,
    )

    log_update_steps = max(len(dataloader) // 20, 1)
    inf_it = iter(dataloader)
    for itr in range(len(dataloader)):
        try:
            data = next(inf_it)
        except Exception:
            raise LLMDataException("Data reading error. Skipping inference.")

        val_batch_size = get_inference_batch_size(cfg)
        cfg.environment._curr_val_step += val_batch_size * cfg.environment._world_size

        batch = cfg.dataset.dataset_class.batch_to_device(data, cfg.environment._device)

        if cfg.environment.use_deepspeed:
            if (
                cfg.prediction.metric != "Perplexity"
                and cfg.problem_type not in NON_GENERATION_PROBLEM_TYPES
            ):
                output = {}
                output["predicted_answer_ids"] = (
                    model.generate(batch, cfg).detach().cpu()  # type: ignore
                )
            else:
                output = model.forward(batch)
        else:
            with autocast(
                enabled=cfg.environment.mixed_precision,
                dtype=get_torch_dtype(cfg.environment.mixed_precision_dtype),
            ):
                if (
                    cfg.prediction.metric != "Perplexity"
                    and cfg.problem_type not in NON_GENERATION_PROBLEM_TYPES
                ):
                    output = {}
                    output["predicted_answer_ids"] = (
                        unwrap_model(model).generate(batch, cfg).detach().cpu()
                    )
                else:
                    output = model.forward(batch)
        if contains_nan(output) and cfg.environment.mixed_precision:
            raise LLMModelException(
                "NaN caught during mixed precision inference. "
                "Please disable mixed precision inference. "
                "Alternatively, reducing learning rate or "
                "gradient clipping may help to stabilize training."
            )

        output = dataloader.dataset.postprocess_batch_predictions(output=output)

        if "predicted_answer_ids" in output.keys():
            del output["predicted_answer_ids"]

        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu()

            # DefaultDict is not used as it adds extra keys during pickle.dump
            if key not in out:
                out[key] = [val]
            else:
                out[key] += [val]

        if cfg.environment._local_rank == 0:
            # Show logs each 5% of the inference
            if (itr + 1) % log_update_steps == 0 or itr == len(dataloader) - 1:
                progress_bar.set_description(f"{mode} progress", refresh=False)
                if (itr + 1) % log_update_steps == 0:
                    progress_bar.update(log_update_steps)
                else:
                    progress_bar.update(len(dataloader) % log_update_steps)

            cfg.logging._logger.log(
                "internal",
                "current_val_step",
                cfg.environment._curr_val_step,
                step=cfg.environment._curr_val_step,
            )

        if cfg.environment._distributed:
            torch.distributed.barrier()

    progress_bar.close()
    del progress_bar
    out = cat_batches(out)

    return out


def save_predictions(cfg, val_data, val_dataloader, val_df, mode):
    val_data, val_df = val_dataloader.dataset.format_output(  # type: ignore
        cfg=cfg, df=val_df, output=val_data
    )
    raw_preds_name = os.path.join(cfg.output_directory, f"{mode}_raw_predictions.pkl")
    csv_preds_name = os.path.join(cfg.output_directory, f"{mode}_predictions.csv")
    save_pickle(raw_preds_name, val_data)
    val_df.to_csv(csv_preds_name, index=False)


def update_backbone_config(config: Any, cfg: Any):
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = cfg.architecture.intermediate_dropout
    if hasattr(config, "attention_probs_dropout_prob"):
        config.attention_probs_dropout_prob = cfg.architecture.intermediate_dropout
    if (
        not hasattr(config, "hidden_dropout_prob")
        and not hasattr(config, "attention_probs_dropout_prob")
        and cfg.architecture.intermediate_dropout > 0
    ):
        logger.warning(
            "Model config does not have dropout attributes. "
            f"Ignoring Intermediate Dropout = {cfg.architecture.intermediate_dropout}."
        )
        cfg.architecture.intermediate_dropout = 0

    tokenizer = get_tokenizer(cfg)

    if config.eos_token_id != tokenizer.eos_token_id:
        logger.warning(
            "EOS token id not matching between config and tokenizer. "
            "Overwriting with tokenizer id."
        )
        config.eos_token_id = tokenizer.eos_token_id
    if config.pad_token_id != tokenizer.pad_token_id:
        logger.warning(
            "PAD token id not matching between config and tokenizer. "
            "Overwriting with tokenizer id."
        )
        config.pad_token_id = tokenizer.pad_token_id
    # no warning needed as not used
    if config.bos_token_id != tokenizer.bos_token_id:
        config.bos_token_id = tokenizer.bos_token_id

    if "mpt-" in cfg.llm_backbone:
        config.init_device = cfg.environment._device

    # See: https://github.com/huggingface/transformers/pull/24906
    if hasattr(config, "pretraining_tp") and cfg.training.lora:
        logger.info("Setting pretraining_tp of model config to 1.")
        config.pretraining_tp = 1

    return config


def set_generation_config(backbone: torch.nn.Module, cfg_prediction: Any):
    backbone.generation_config.min_new_tokens = cfg_prediction.min_length_inference
    backbone.generation_config.max_new_tokens = cfg_prediction.max_length_inference
    backbone.generation_config.max_time = (
        cfg_prediction.max_time if cfg_prediction.max_time > 0 else None
    )
    backbone.generation_config.do_sample = cfg_prediction.do_sample
    backbone.generation_config.num_beams = cfg_prediction.num_beams
    backbone.generation_config.repetition_penalty = cfg_prediction.repetition_penalty
    if cfg_prediction.do_sample:
        backbone.generation_config.temperature = cfg_prediction.temperature
        backbone.generation_config.top_k = cfg_prediction.top_k
        backbone.generation_config.top_p = cfg_prediction.top_p
    backbone.generation_config.transformers_version = transformers.__version__
    return backbone


def create_nlp_backbone(cfg, model_class=AutoModel) -> Any:
    """
    Creates a backbone model for NLP tasks.
    This is needed for Gradient Checkpointing in DDP mode.
    """
    kwargs = dict()
    try:
        config = AutoConfig.from_pretrained(
            cfg.llm_backbone,
            trust_remote_code=cfg.environment.trust_remote_code,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            revision=cfg.environment.huggingface_branch,
        )
        kwargs["token"] = os.getenv("HUGGINGFACE_TOKEN")
    except TypeError:
        # TypeError: RWForCausalLM.__init__() got
        # an unexpected keyword argument 'token'
        config = AutoConfig.from_pretrained(
            cfg.llm_backbone,
            trust_remote_code=cfg.environment.trust_remote_code,
            revision=cfg.environment.huggingface_branch,
        )

    config = update_backbone_config(config, cfg)

    quantization_config = None
    if cfg.architecture.backbone_dtype == "int8" and len(cfg.environment.gpus):
        kwargs["device_map"] = {"": cfg.environment._device}  # type: ignore
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=0.0,
        )
        # need to force pretrained
        cfg.architecture.pretrained = True
        kwargs["torch_dtype"] = torch.float16  # type: ignore
    elif cfg.architecture.backbone_dtype == "int4" and len(cfg.environment.gpus):
        kwargs["device_map"] = {"": cfg.environment._device}  # type: ignore
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        # need to force pretrained
        cfg.architecture.pretrained = True
        kwargs["torch_dtype"] = torch.float16  # type: ignore
    elif len(cfg.environment.gpus) == 0 and cfg.architecture.backbone_dtype in [
        "int4",
        "int8",
    ]:
        logger.warning(
            "Quantization is not supported on CPU. "
            "Please run on GPU or disable quantization."
        )
        cfg.architecture.backbone_dtype = "float32"
    else:
        kwargs["torch_dtype"] = getattr(torch, cfg.architecture.backbone_dtype)

    logger.info(f"Using {cfg.architecture.backbone_dtype} for backbone")

    kwargs["trust_remote_code"] = cfg.environment.trust_remote_code

    if cfg.training.use_flash_attention_2:
        try:
            import flash_attn  # noqa: F401

            # see https://github.com/fxmarty/transformers/
            # blob/3f06a3a0aec8cc1ec3ad6bf66ebe277392c5ab37/
            # src/transformers/configuration_utils.py#L380
            config._attn_implementation_internal = "flash_attention_2"
            if cfg.environment._local_rank == 0:
                logger.info("Using Flash Attention 2.")
        except ImportError:
            if cfg.environment._local_rank == 0:
                logger.warning(
                    "Flash Attention 2.0 is not available. "
                    "Please consider to run 'make setup' to install it."
                )

    if cfg.architecture.pretrained:
        if cfg.environment._local_rank == 0:
            logger.info(f"Loading {cfg.llm_backbone}. This may take a while.")

        backbone = model_class.from_pretrained(
            cfg.llm_backbone,
            revision=cfg.environment.huggingface_branch,
            config=config,
            quantization_config=quantization_config,
            **kwargs,
        )
        if cfg.environment._local_rank == 0:
            logger.info(f"Loaded {cfg.llm_backbone}.")
    else:
        kwargs.pop("token", None)
        backbone = model_class.from_config(config, **kwargs)

    if cfg.tokenizer._vocab_length > config.vocab_size:
        if cfg.environment._local_rank == 0:
            logger.info(f"Resizing token embeddings to {cfg.tokenizer._vocab_length}")
        backbone.resize_token_embeddings(cfg.tokenizer._vocab_length)

    backbone.model_parallel = False

    if cfg.training.lora:
        # if used, gradient checkpointing will be enabled below
        loaded_in_kbit = getattr(backbone, "is_loaded_in_8bit", False) or getattr(
            backbone, "is_loaded_in_4bit", False
        )

        for name, param in backbone.named_parameters():
            # freeze base model's layers
            param.requires_grad = False

        # cast all non INT8 parameters to fp32
        if loaded_in_kbit:
            for param in backbone.parameters():
                if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                    param.data = param.data.to(torch.float32)
    else:
        if cfg.architecture.backbone_dtype != "float32":
            if cfg.environment.mixed_precision:
                logger.info("Disabling mixed precision as dtype not set to float32.")
                cfg.environment.mixed_precision = False
            if cfg.architecture.backbone_dtype != "bfloat16":
                logger.warning(
                    "Pure float16 or int8 training will "
                    "likely lead to unstable training without adapters."
                )

    if cfg.architecture.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()

    # initialize the generation config
    if backbone.generation_config.eos_token_id != config.eos_token_id:
        logger.warning(
            "EOS token id not matching between generation config and tokenizer. "
            "Overwriting with tokenizer id."
        )
        backbone.generation_config.eos_token_id = config.eos_token_id
    if backbone.generation_config.pad_token_id != config.pad_token_id:
        logger.warning(
            "PAD token id not matching between generation config and tokenizer. "
            "Overwriting with tokenizer id."
        )
        backbone.generation_config.pad_token_id = config.pad_token_id
    # no warning needed as not used
    if backbone.generation_config.bos_token_id != config.bos_token_id:
        backbone.generation_config.bos_token_id = config.bos_token_id

    if cfg.problem_type not in NON_GENERATION_PROBLEM_TYPES:
        backbone = set_generation_config(backbone, cfg.prediction)

    return backbone, config


# Adapted from https://github.com/huggingface/trl/blob/
# 2068fdcd931183b59110aa6dc99d8f5bb55c6f2d/trl/trainer/utils.py#L742
def activate_neftune(model, neftune_noise_alpha):
    r"""
    Activates the neftune as presented in this code:
    https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
    """
    backbone = unwrap_model(model).backbone
    if isinstance(backbone, PeftModel):
        embeddings = backbone.base_model.get_input_embeddings()
    else:
        embeddings = backbone.get_input_embeddings()

    embeddings.neftune_noise_alpha = neftune_noise_alpha
    embeddings.register_forward_hook(neftune_post_forward_hook)


def neftune_post_forward_hook(module, input, output):
    """
    Implements the NEFTune forward pass for the model using forward hooks.
    Note this works only for torch.nn.Embedding layers.
    This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Simply add it to your model as follows:
    ```python
    model = ...
    model.embed_tokens.neftune_noise_alpha = 0.1
    model.embed_tokens.register_forward_hook(neftune_post_forward_hook)
    ```

    Args:
        module (`torch.nn.Module`):
            The embedding module where the hook is attached. Note that you need to set
            `module.neftune_noise_alpha` to the desired noise alpha value.
        input (`torch.Tensor`):
            The input tensor to the model.
        output (`torch.Tensor`):
            The output tensor of the model (i.e. the embeddings).
    """
    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output


class TokenStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria based on tokens.
    Will stop generation when each generated sample contains at least one of the
    stop_word_ids.
    """

    def __init__(self, stop_word_ids, prompt_input_ids_len):
        super().__init__()
        self.prompt_input_ids_len = prompt_input_ids_len
        if stop_word_ids is None:
            stop_word_ids = []
        self.stop_word_ids = stop_word_ids

    def should_stop(
        self,
        generated_ids: torch.Tensor,
        stop_word_id: torch.Tensor,
    ):
        if len(stop_word_id.shape) == 0:
            return (
                torch.mean(((generated_ids == stop_word_id).sum(1) > 0).float()) == 1
            ).item()
        else:
            return (
                self.get_num_vector_found_in_matrix_rows(stop_word_id, generated_ids)
                == generated_ids.shape[0]
            )

    @staticmethod
    def get_num_vector_found_in_matrix_rows(vector, matrix):
        """
        Count the number of times a vector is found in a matrix row.
        If the vector is found in a row, the search stops and the next row is searched.
        """
        assert len(vector.shape) == 1
        assert len(matrix.shape) == 2

        found = 0
        for row in matrix:
            # stride through the vector
            for i in range(len(row) - len(vector) + 1):
                # check if the vector contains the tensor
                if torch.all(row[i : i + len(vector)] == vector):
                    found += 1
                    break

        return found

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor, **kwargs):
        generated_ids: torch.Tensor = input_ids[:, self.prompt_input_ids_len :]
        for stop_word_id in self.stop_word_ids:
            if self.should_stop(generated_ids, stop_word_id.to(generated_ids.device)):
                if generated_ids.shape[1] == 1:
                    logger.warning(
                        f"Stopping criteria triggered for {stop_word_id} at first "
                        "generated token."
                    )
                return True
        return False


class EnvVariableStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria based on env variable.
    Useful to force stopping within the app.
    """

    stop_streaming_env: str = "STOP_STREAMING"

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor, **kwargs):
        should_stop = self.stop_streaming_env in os.environ
        if should_stop:
            logger.info("Received signal to stop generating")
        return should_stop


def prepare_lora(cfg, backbone):
    target_modules = (
        [
            lora_target_module.strip()
            for lora_target_module in cfg.training.lora_target_modules.strip().split(  # noqa: E501
                ","
            )
        ]
        if cfg.training.lora_target_modules
        else None
    )

    if target_modules is None:
        target_modules = []
        for name, module in backbone.named_modules():
            if (
                isinstance(
                    module, (torch.nn.Linear, torch.nn.Conv1d, Conv1DTransformer)
                )
                and "head" not in name
            ):
                name = name.split(".")[-1]
                if name not in target_modules:
                    target_modules.append(name)

    if cfg.environment._local_rank == 0:
        logger.info(f"Lora module names: {target_modules}")

    lora_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.architecture.gradient_checkpointing:
        backbone.enable_input_require_grads()
    backbone = get_peft_model(backbone, lora_config)

    trainable_params, all_param = backbone.get_nb_trainable_parameters()
    if cfg.environment._local_rank == 0:
        logger.info(f"Trainable parameters count: {trainable_params}")
        logger.info(f"Total parameters count: {all_param}")
        logger.info(f"Trainable %: {100 * trainable_params / all_param:.4f}%")

    return backbone


def generate(backbone, batch, cfg, streamer, remove_prompt=True):
    mask_key = "prompt_attention_mask"
    pad_keys = [
        "prompt_input_ids",
        "prompt_attention_mask",
    ]
    batch = batch_padding(
        cfg,
        batch,
        training=False,
        mask_key=mask_key,
        pad_keys=pad_keys,
    )
    input_ids = batch["prompt_input_ids"]
    attention_mask = batch["prompt_attention_mask"]
    # Adding GenerationMixin type annotation for faster lookup
    generation_function: GenerationMixin.generate = backbone.generate
    verbosity = transformers_logging.get_verbosity()
    stopping_criteria = StoppingCriteriaList(
        [
            TokenStoppingCriteria(
                stop_word_ids=cfg.tokenizer._stop_words_ids,
                prompt_input_ids_len=input_ids.shape[1],
            ),
            EnvVariableStoppingCriteria(),
        ]
    )
    # force to use cache and disable gradient checkpointing if enabled
    backbone.config.use_cache = True
    if cfg.architecture.gradient_checkpointing:
        backbone.gradient_checkpointing_disable()
    transformers_logging.set_verbosity_error()
    output = generation_function(
        inputs=input_ids,
        attention_mask=attention_mask,
        generation_config=backbone.generation_config,
        stopping_criteria=stopping_criteria,
        renormalize_logits=True,
        return_dict_in_generate=False,
        use_cache=True,
        streamer=streamer,
    )
    transformers_logging.set_verbosity(verbosity)
    # enable checkpointing again
    if cfg.architecture.gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
    if remove_prompt:
        output = output[:, input_ids.shape[1] :]
    return output


def get_torch_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float32
