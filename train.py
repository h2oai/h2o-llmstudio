import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import gc
import logging
import sys
import time
from distutils import util
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.loggers import MainLogger
from llm_studio.src.trl.trainer import PPOTrainer
from llm_studio.src.utils.config_utils import (
    load_config_py,
    load_config_yaml,
    save_config_yaml,
)
from llm_studio.src.utils.data_utils import (
    get_data,
    get_inference_batch_size,
    get_train_dataloader,
    get_train_dataset,
    get_val_dataloader,
    get_val_dataset,
)
from llm_studio.src.utils.exceptions import LLMTrainingException
from llm_studio.src.utils.export_utils import save_prediction_outputs
from llm_studio.src.utils.gpu_utils import sync_across_processes
from llm_studio.src.utils.logging_utils import (
    TqdmToLogger,
    initialize_logging,
    log_plot,
    write_flag,
)
from llm_studio.src.utils.modeling_utils import (
    check_disk_space,
    get_number_of_validation_epochs,
    get_optimizer,
    get_scheduler,
    load_checkpoint,
    reduce_metric,
    run_inference,
    save_checkpoint,
    save_predictions,
    unwrap_model,
    wrap_model_distributed,
)
from llm_studio.src.utils.utils import kill_ddp_processes, set_environment, set_seed

logger = logging.getLogger(__name__)


def run_eval(
    cfg,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    val_df: pd.DataFrame,
    mode: str = "validation",
) -> Tuple:
    """Runs the evaluation loop.

    Args:
        cfg: config object
        model: trained model
        val_dataloader: validation Dataloader
        val_df: validation DataFrame
        mode: validation

    Returns:
        Validation loss
    """

    with torch.no_grad():
        model.eval()
        val_data: Dict[str, Any] = run_inference(
            cfg, model, val_dataloader, mode
        )  # type: ignore

    # Sync validation predictions across GPUs
    if cfg.environment._distributed and cfg.environment._distributed_inference:
        for key, value in val_data.items():
            val_data[key] = sync_across_processes(
                value, cfg.environment._world_size, group=cfg.environment._cpu_comm
            )

    torch.inference_mode(mode=True)
    # Drop any extra observations
    for k, v in val_data.items():
        val_data[k] = v[: len(val_dataloader.dataset)]  # type: ignore

    if cfg.environment._local_rank == 0:
        val_data = val_dataloader.dataset.postprocess_output(  # type: ignore
            cfg=cfg, df=val_df, output=val_data
        )

    val_loss = 0.0
    val_metric = 0.0
    if cfg.environment._local_rank == 0:
        # Calculate validation loss
        if "loss" in val_data:
            assert isinstance(val_data["loss"], torch.Tensor)
            val_losses = val_data["loss"].float().cpu().numpy()
            val_loss = np.mean(val_losses)
            logger.info(f"Mean {mode} loss: {val_loss:.5f}")
            cfg.logging._logger.log(
                mode, "loss", val_loss, step=cfg.environment._curr_step
            )

        # Calculate reduced validation metric
        _, _, reduce = cfg.prediction.metric_class.get(cfg.prediction.metric)
        val_metric = reduce_metric(val_data, reduce=reduce)

        logger.info(f"{mode.capitalize()} {cfg.prediction.metric}: {val_metric:.5f}")
        cfg.logging._logger.log(
            mode, cfg.prediction.metric, val_metric, step=cfg.environment._curr_step
        )

        # Log plots
        if val_df is not None:
            plot = cfg.logging.plots_class.plot_validation_predictions(
                val_outputs=val_data, cfg=cfg, val_df=val_df, mode="validation"
            )
            log_plot(cfg, plot, "validation_predictions")

        save_predictions(cfg, val_data, val_dataloader, val_df, mode)

    if cfg.environment._distributed:
        torch.distributed.barrier()

    torch.inference_mode(mode=False)

    return val_loss, val_metric


def run_train(
    cfg: Any,
    model: torch.nn.Module,
    train_dataloader,
    val_dataloader,
    val_df: pd.DataFrame,
):
    """Runs the training loop.

    Args:
        cfg: config object
        model: model
        train_dataloader: custom training Dataloader
        train_df: train DataFrame
        val_dataloader: custom validation Dataloader
        val_df: validation DataFrame

    Returns:
        Validation prediction output
        Validation loss
        Validation metric
        Last train batch
    """

    epoch_steps = len(train_dataloader)

    optimizer = get_optimizer(model=model, cfg=cfg)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer, epoch_steps=epoch_steps)

    scaler: GradScaler | ShardedGradScaler | None = None
    if cfg.environment.mixed_precision:
        if cfg.environment.use_fsdp:
            scaler = ShardedGradScaler()
        else:
            scaler = GradScaler()

    optimizer.zero_grad(set_to_none=True)

    # Prepare NLP Augmentation
    nlp_augment = None
    if hasattr(cfg.augmentation, "nlp_augmentations_class"):
        nlp_augment = cfg.augmentation.nlp_augmentations_class(cfg=cfg)

    start_epoch = 0

    _, metric_mode, _ = cfg.prediction.metric_class.get(cfg.prediction.metric)
    objective_op: Callable[[float, float], bool]
    if metric_mode == "max":
        best_val_metric = -np.inf
        objective_op = np.greater
    else:
        best_val_metric = np.inf
        objective_op = np.less

    if cfg.training.evaluate_before_training:
        val_loss, val_metric = run_eval(
            cfg=cfg, model=model, val_dataloader=val_dataloader, val_df=val_df
        )

    for epoch in range(start_epoch, cfg.training.epochs):
        set_seed(
            cfg.environment._seed
            + epoch * cfg.environment._world_size * cfg.environment.number_of_workers
            + cfg.environment._local_rank * cfg.environment.number_of_workers
        )
        if cfg.environment._local_rank == 0:
            logger.info(f"Training Epoch: {epoch + 1} / {cfg.training.epochs}")

        if cfg.environment._distributed and hasattr(
            train_dataloader.sampler, "set_epoch"
        ):
            train_dataloader.sampler.set_epoch(epoch)  # type: ignore

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        progress_bar = tqdm(
            total=epoch_steps,
            disable=cfg.environment._local_rank != 0,
            file=tqdm_out,
            ascii=True,
            desc="train loss",
            mininterval=0,
        )
        tr_it = iter(train_dataloader)

        losses = []
        model.train()

        log_update_steps = max(epoch_steps // 20, 1)
        evaluation_step = max(int(epoch_steps * cfg.training.evaluation_epochs), 1)
        for itr, data in enumerate(tr_it):
            cfg.environment._curr_step += (
                cfg.training.batch_size * cfg.environment._world_size
            )

            # Batch to device
            batch = cfg.dataset.dataset_class.batch_to_device(
                data, cfg.environment._device
            )

            # NLP augmentation
            if nlp_augment is not None:
                batch = nlp_augment(batch)

            # Plot first batch
            if epoch == 0 and itr == 0 and cfg.environment._local_rank == 0:
                plot = cfg.logging.plots_class.plot_batch(batch=batch, cfg=cfg)
                log_plot(cfg, plot, "train_data")

            # Forward pass
            with autocast(enabled=cfg.environment.mixed_precision):
                output_dict = model.forward(batch)

            loss = output_dict["loss"]
            if ~np.isfinite(loss.item()) and (epoch > start_epoch or itr > 20):
                raise LLMTrainingException(
                    "NaN caught in loss during training. "
                    "Please, reduce learning rate, change dtype, "
                    "or disable mixed precision. Alternatively, "
                    "gradient clipping may help to stabilize training."
                )
            losses.append(loss.item())

            # loss is a mean loss per batch/sample
            # as grad_accumulations sums up the gradients, this loss must be scaled
            # by the number of grad_accumulations, to have similar behavior for
            # BS * grad_accumulations = const.
            if cfg.training.grad_accumulation != 1:
                loss = loss / cfg.training.grad_accumulation

            # Backward pass
            if cfg.environment.mixed_precision:
                scaler.scale(loss).backward()  # type: ignore
                if itr % cfg.training.grad_accumulation == 0:
                    if cfg.training.gradient_clip > 0:
                        scaler.unscale_(optimizer)  # type: ignore
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.gradient_clip
                        )
                    scaler.step(optimizer)  # type: ignore
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if itr % cfg.training.grad_accumulation == 0:
                    if cfg.training.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.gradient_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if cfg.environment._distributed:
                torch.cuda.synchronize(device=cfg.environment._local_rank)

            if scheduler is not None:
                scheduler.step()

            if cfg.environment._local_rank == 0:
                cfg.logging._logger.log(
                    "train", "loss", losses[-1], step=cfg.environment._curr_step
                )
                cfg.logging._logger.log(
                    "meta",
                    "lr",
                    optimizer.param_groups[0]["lr"],
                    step=cfg.environment._curr_step,
                )
                if cfg.training.differential_learning_rate_layers:
                    cfg.logging._logger.log(
                        "meta",
                        "lr_diff",
                        optimizer.param_groups[2]["lr"],
                        step=cfg.environment._curr_step,
                    )

                cfg.logging._logger.log(
                    "internal",
                    "current_step",
                    cfg.environment._curr_step,
                    step=cfg.environment._curr_step,
                )

                # Show logs each 5% of the epoch (only if doing per epoch evaluation)
                if (itr + 1) % log_update_steps == 0 or itr == epoch_steps - 1:
                    progress_bar.set_description(
                        f"train loss: {np.mean(losses[-10:]):.2f}", refresh=False
                    )
                    if (itr + 1) % log_update_steps == 0:
                        progress_bar.update(log_update_steps)
                    else:
                        progress_bar.update(epoch_steps % log_update_steps)

                del output_dict

            # Validation loop
            if (itr + 1) % evaluation_step == 0:
                if cfg.training.evaluation_epochs == 1:
                    progress_bar.close()

                val_loss, val_metric = run_eval(
                    cfg=cfg, model=model, val_dataloader=val_dataloader, val_df=val_df
                )
                if cfg.environment._local_rank == 0:
                    if cfg.training.save_best_checkpoint:
                        if objective_op(val_metric, best_val_metric):
                            checkpoint_path = cfg.output_directory
                            logger.info(
                                f"Saving best model checkpoint: "
                                f"val_{cfg.prediction.metric} {best_val_metric:.5} -> "
                                f"{val_metric:.5} to {checkpoint_path}"
                            )
                            save_checkpoint(model=model, path=checkpoint_path, cfg=cfg)
                            best_val_metric = val_metric
                    else:
                        checkpoint_path = cfg.output_directory
                        logger.info(
                            f"Saving last model checkpoint: "
                            f"val_loss {val_loss:.5}, val_{cfg.prediction.metric} "
                            f"{val_metric:.5} to {checkpoint_path}"
                        )
                        save_checkpoint(model=model, path=checkpoint_path, cfg=cfg)

                model.train()

        progress_bar.close()
        del progress_bar

        if cfg.environment._distributed:
            torch.cuda.synchronize(device=cfg.environment._local_rank)
            torch.distributed.barrier()

        if cfg.environment._local_rank == 0:
            cfg.logging._logger.log(
                "internal", "epoch", epoch + 1, step=cfg.environment._curr_step
            )

    if cfg.environment._distributed:
        torch.distributed.barrier()

    return val_loss, val_metric


def run_train_rlhf(
    cfg: Any,
    model: torch.nn.Module,
    train_dataloader,
    val_dataloader,
    val_df: pd.DataFrame,
):
    """Runs the training loop.

    Args:
        cfg: config object
        model: model
        train_dataloader: custom training Dataloader
        train_df: train DataFrame
        val_dataloader: custom validation Dataloader
        val_df: validation DataFrame

    Returns:
        Validation prediction output
        Validation loss
        Validation metric
        Last train batch
    """

    epoch_steps = len(train_dataloader)

    optimizer = get_optimizer(model=model, cfg=cfg)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer, epoch_steps=epoch_steps)

    scaler: GradScaler | ShardedGradScaler | None = None
    if cfg.environment.mixed_precision:
        if cfg.environment.use_fsdp:
            scaler = ShardedGradScaler()
        else:
            scaler = GradScaler()

    optimizer.zero_grad(set_to_none=True)

    # Prepare NLP Augmentation
    nlp_augment = None
    if hasattr(cfg.augmentation, "nlp_augmentations_class"):
        nlp_augment = cfg.augmentation.nlp_augmentations_class(cfg=cfg)

    start_epoch = 0

    _, metric_mode, _ = cfg.prediction.metric_class.get(cfg.prediction.metric)
    objective_op: Callable[[float, float], bool]
    if metric_mode == "max":
        best_val_metric = -np.inf
        objective_op = np.greater
    else:
        best_val_metric = np.inf
        objective_op = np.less

    if cfg.training.evaluate_before_training:
        val_loss, val_metric = run_eval(
            cfg=cfg, model=model, val_dataloader=val_dataloader, val_df=val_df
        )

    with torch.device(cfg.environment._device):
        logger.info("Using RLHF - Loading reward model")
        reward_model = cfg.architecture.reward_model_class(cfg)
        reward_model.eval()

    if cfg.training.offload_reward_model:
        reward_model.to("cpu")
    else:
        reward_model.to(cfg.environment._device)

    # initialize trainer
    tokenizer = get_tokenizer(cfg)
    ppo_trainer = PPOTrainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scaler=scaler,
    )

    for epoch in range(start_epoch, cfg.training.epochs):
        set_seed(
            cfg.environment._seed
            + epoch * cfg.environment._world_size * cfg.environment.number_of_workers
            + cfg.environment._local_rank * cfg.environment.number_of_workers
        )
        if cfg.environment._local_rank == 0:
            logger.info(f"Training Epoch: {epoch + 1} / {cfg.training.epochs}")

        if cfg.environment._distributed and hasattr(
            train_dataloader.sampler, "set_epoch"
        ):
            train_dataloader.sampler.set_epoch(epoch)  # type: ignore

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        progress_bar = tqdm(
            total=epoch_steps,
            disable=cfg.environment._local_rank != 0,
            file=tqdm_out,
            ascii=True,
            desc="train loss",
            mininterval=0,
        )
        tr_it = iter(train_dataloader)

        losses = []
        model.train()

        log_update_steps = max(epoch_steps // 20, 1)
        evaluation_step = max(int(epoch_steps * cfg.training.evaluation_epochs), 1)
        # Round up to the nearest multiple of cfg.training.rollout_steps
        evaluation_step = (
            (evaluation_step + cfg.training.rollout_steps - 1)
            // cfg.training.rollout_steps
        ) * cfg.training.rollout_steps

        query_tensors = []
        response_tensors = []
        rewards = []

        for itr, data in enumerate(tr_it):
            cfg.environment._curr_step += (
                cfg.training.batch_size * cfg.environment._world_size
            )

            # Batch to device
            batch = cfg.dataset.dataset_class.batch_to_device(
                data, cfg.environment._device
            )

            # NLP augmentation
            if nlp_augment is not None:
                batch = nlp_augment(batch)

            # Plot first batch
            if epoch == 0 and itr == 0 and cfg.environment._local_rank == 0:
                plot = cfg.logging.plots_class.plot_batch(batch=batch, cfg=cfg)
                log_plot(cfg, plot, "train_data")

            with torch.no_grad():
                logger.debug("Rollout: Generating response from active model")
                output_dict = {}
                output_dict["predicted_answer_ids"] = (
                    unwrap_model(model)
                    .generate(batch, unwrap_model(model).cfg)
                    .detach()
                )
                output_dict = train_dataloader.dataset.postprocess_batch_predictions(
                    cfg=cfg, output=output_dict
                )

                logger.debug("Evaluation: Score from reward model")
                # tokenize prompt & output internally
                if cfg.training.offload_reward_model:
                    reward_model.to(cfg.environment._device)
                with autocast(enabled=cfg.environment.mixed_precision):
                    scores = reward_model.get_score(
                        batch["reward_model_prompt_text"],
                        output_dict["predicted_text"],
                    )
                if cfg.training.offload_reward_model:
                    reward_model.to("cpu")

            # score by reward model
            reward = [torch.tensor(score, dtype=torch.float32) for score in scores]

            # remove padding from query and response
            batch["input_ids"] = batch["input_ids"].detach().cpu()
            query_tensor = [
                input_ids[torch.where(att_mask == 1)[0].min() :]
                if len(torch.where(att_mask == 1)[0]) > 0
                else input_ids
                for input_ids, att_mask in zip(
                    batch["input_ids"].detach().cpu(), batch["attention_mask"]
                )
            ]
            pad_tok_id = (
                unwrap_model(model).backbone.config.pad_token_id
                or unwrap_model(model).backbone.config.eos_token_id
            )
            output_dict["predicted_answer_ids"] = (
                output_dict["predicted_answer_ids"].detach().cpu()
            )
            response_tensor = [
                predicted_answer_ids[
                    : torch.where(predicted_answer_ids == pad_tok_id)[0].min()
                ]
                if len(torch.where(predicted_answer_ids == pad_tok_id)[0]) > 0
                else predicted_answer_ids
                for predicted_answer_ids in output_dict["predicted_answer_ids"]
            ]

            del output_dict
            del batch

            query_tensors += query_tensor
            response_tensors += response_tensor
            rewards += reward

            if cfg.environment._distributed:
                torch.cuda.synchronize(device=cfg.environment._local_rank)
                torch.distributed.barrier()

            if (itr + 1) % cfg.training.rollout_steps == 0:
                output_dict = ppo_trainer.step(query_tensors, response_tensors, rewards)
                del query_tensors, response_tensors, rewards, scores

                query_tensors = []
                response_tensors = []
                rewards = []

                loss = output_dict["ppo/loss/total"]
                losses.append(loss)

                if cfg.environment._local_rank == 0:
                    for key in output_dict.keys():
                        if isinstance(output_dict[key], (float, int)) or (
                            isinstance(output_dict[key], np.ndarray)
                            and output_dict[key].size == 1
                        ):
                            if np.isfinite(output_dict[key]):
                                cfg.logging._logger.log(
                                    "train",
                                    key,
                                    output_dict[key],
                                    step=cfg.environment._curr_step,
                                )
                    cfg.logging._logger.log(
                        "train", "loss", losses[-1], step=cfg.environment._curr_step
                    )
                    cfg.logging._logger.log(
                        "meta",
                        "lr",
                        optimizer.param_groups[0]["lr"],
                        step=cfg.environment._curr_step,
                    )
                    if cfg.training.differential_learning_rate_layers:
                        cfg.logging._logger.log(
                            "meta",
                            "lr_diff",
                            optimizer.param_groups[2]["lr"],
                            step=cfg.environment._curr_step,
                        )

                    cfg.logging._logger.log(
                        "internal",
                        "current_step",
                        cfg.environment._curr_step,
                        step=cfg.environment._curr_step,
                    )

                    # Show logs each 5% of the epoch (only if doing per epoch eval)
                    if (itr + 1) % log_update_steps == 0 or itr == epoch_steps - 1:
                        progress_bar.set_description(
                            f"train loss: {np.mean(losses[-10:]):.2f}", refresh=False
                        )
                        if (itr + 1) % log_update_steps == 0:
                            progress_bar.update(log_update_steps)
                        else:
                            progress_bar.update(epoch_steps % log_update_steps)

                    del output_dict

            if cfg.environment._distributed:
                torch.cuda.synchronize(device=cfg.environment._local_rank)
                torch.distributed.barrier()

            # Validation loop
            if (itr + 1) % evaluation_step == 0:
                if cfg.training.evaluation_epochs == 1:
                    progress_bar.close()

                val_loss, val_metric = run_eval(
                    cfg=cfg, model=model, val_dataloader=val_dataloader, val_df=val_df
                )
                if cfg.environment._local_rank == 0:
                    if cfg.training.save_best_checkpoint:
                        if objective_op(val_metric, best_val_metric):
                            checkpoint_path = cfg.output_directory
                            logger.info(
                                f"Saving best model checkpoint: "
                                f"val_{cfg.prediction.metric} {best_val_metric:.5} -> "
                                f"{val_metric:.5} to {checkpoint_path}"
                            )
                            save_checkpoint(model=model, path=checkpoint_path, cfg=cfg)
                            best_val_metric = val_metric
                    else:
                        checkpoint_path = cfg.output_directory
                        logger.info(
                            f"Saving last model checkpoint: "
                            f"val_loss {val_loss:.5}, val_{cfg.prediction.metric} "
                            f"{val_metric:.5} to {checkpoint_path}"
                        )
                        save_checkpoint(model=model, path=checkpoint_path, cfg=cfg)

                model.train()

        progress_bar.close()
        del progress_bar

        if cfg.environment._distributed:
            torch.cuda.synchronize(device=cfg.environment._local_rank)
            torch.distributed.barrier()

        if cfg.environment._local_rank == 0:
            cfg.logging._logger.log(
                "internal", "epoch", epoch + 1, step=cfg.environment._curr_step
            )

    if cfg.environment._distributed:
        torch.distributed.barrier()

    return val_loss, val_metric


def run(cfg: Any) -> None:
    """Runs the routine.

    Args:
        cfg: config object with all the hyperparameters
    """

    os.makedirs(cfg.output_directory, exist_ok=True)

    # Force evaluation if user trains 0 epochs
    cfg.training.evaluate_before_training = (
        cfg.training.evaluate_before_training or cfg.training.epochs == 0
    )

    # Set the random seed for reproducibility
    # either random seed when user set it -1 or deterministic user chosen seed
    if cfg.environment.seed < 0:
        cfg.environment._seed = np.random.randint(1_000_000)
    else:
        cfg.environment._seed = cfg.environment.seed

    # Prepare environment
    if "WORLD_SIZE" in os.environ:
        cfg.environment._distributed = int(os.environ["WORLD_SIZE"]) > 1
    else:
        cfg.environment._distributed = False

    if cfg.environment._distributed:
        cfg.environment._local_rank = int(os.environ["LOCAL_RANK"])
        cfg.environment._device = "cuda:%d" % cfg.environment._local_rank
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        cfg.environment._cpu_comm = torch.distributed.new_group(backend="gloo")

        cfg.environment._world_size = torch.distributed.get_world_size()
        cfg.environment._rank = torch.distributed.get_rank()
        torch.cuda.set_device(cfg.environment._rank)
        logger.info(
            f"Training in distributed mode with multiple processes, "
            f"1 GPU per process. Process {cfg.environment._rank}, "
            f"total: {cfg.environment._world_size} "
            f"local rank: {cfg.environment._local_rank}."
        )

        # Sync the random seed
        cfg.environment._seed = int(
            sync_across_processes(
                np.array([cfg.environment._seed]),
                cfg.environment._world_size,
                group=cfg.environment._cpu_comm,
            )[0]
        )
    else:
        cfg.environment._local_rank = 0
        cfg.environment._device = "cuda:0"

    set_seed(cfg.environment._seed)
    if cfg.environment._local_rank == 0:
        logger.info(f"Global random seed: {cfg.environment._seed}")

    cfg = set_environment(cfg)

    # we need to get train dataframe and number of labels if not set or in training mode
    if cfg.environment._local_rank == 0:
        logger.info("Preparing the data...")
    train_df, val_df = get_data(cfg)

    if (
        len(val_df) > int(os.getenv("GPT_EVAL_MAX", 100))
        and "GPT" in cfg.prediction.metric
    ):
        logger.warning(
            f"More than {os.getenv('GPT_EVAL_MAX', 100)} validation records. "
            "Safeguarding against OpenAI API costs. Setting metric to BLEU. "
            "Change GPT_EVAL_MAX to run GPT validation."
        )
        cfg.prediction.metric = "BLEU"

    # prepare data
    if cfg.environment._local_rank == 0:
        logger.info("Preparing train and validation data")
    train_dataset = get_train_dataset(train_df=train_df, cfg=cfg)
    val_dataset = get_val_dataset(val_df=val_df, cfg=cfg)
    train_dataloader = get_train_dataloader(train_ds=train_dataset, cfg=cfg)
    val_dataloader = get_val_dataloader(val_ds=val_dataset, cfg=cfg)

    if cfg.environment._local_rank == 0:
        total_training_steps = (
            cfg.training.epochs
            * len(train_dataloader)
            * cfg.training.batch_size
            * cfg.environment._world_size
        )

        num_eval_epochs = get_number_of_validation_epochs(
            training_epochs=cfg.training.epochs,
            evaluation_epochs=cfg.training.evaluation_epochs,
        )
        val_batch_size = get_inference_batch_size(cfg)

        # if zero shot, validate once before training
        total_validation_steps = (
            len(val_dataloader)
            * (num_eval_epochs + int(cfg.training.evaluate_before_training))
            * val_batch_size
            * cfg.environment._world_size
        )

    # Prepare model
    with torch.device(cfg.environment._device):
        model = cfg.architecture.model_class(cfg)
        check_disk_space(model, cfg.output_directory)

        # load model weights
        if cfg.architecture.pretrained_weights != "":
            # Do not load strictly if continue training from the previous experiment
            load_checkpoint(cfg, model, strict=cfg.training.epochs == -1)
    model.to(cfg.environment._device)

    if getattr(cfg.architecture, "force_embedding_gradients"):
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                for param in module.parameters():
                    param.requires_grad = True
                    param.data = param.data.float()

    if cfg.environment._distributed:
        model = wrap_model_distributed(model, cfg, cfg.environment.use_fsdp)

    if cfg.environment.compile_model:
        if cfg.environment._distributed:
            model.module.backbone = torch.compile(model.module.backbone)
        else:
            model.backbone = torch.compile(model.backbone)

    # Force settings when saving best checkpoint
    if cfg.training.save_best_checkpoint:
        cfg.training.train_validation_data = False

    # reset steps
    cfg.environment._curr_step = 0
    cfg.environment._curr_val_step = 0

    gc.collect()

    global_start_time = time.time()
    if cfg.environment._local_rank == 0:
        # re-save cfg
        save_config_yaml(f"{cfg.output_directory}/cfg.yaml", cfg)

        cfg.logging._logger = MainLogger(cfg)

        cfg.logging._logger.log(
            "internal", "total_training_steps", total_training_steps, step=0
        )

        cfg.logging._logger.log(
            "internal", "total_validation_steps", total_validation_steps, step=0
        )

        cfg.logging._logger.log(
            "internal",
            "global_start_time",
            global_start_time,
            step=cfg.environment._curr_step,
        )
        # re-save config
        save_config_yaml(f"{cfg.output_directory}/cfg.yaml", cfg)

    train_function = (
        run_train_rlhf
        if cfg.problem_type == "text_rlhf_language_modeling"
        else run_train
    )
    val_loss, val_metric = train_function(
        cfg=cfg,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_df=val_df,
    )

    # reset external logging
    if cfg.environment._local_rank == 0:
        cfg.logging._logger.reset_external()

    experiment_path = f"{cfg.output_directory}"

    if cfg.environment._local_rank == 0:
        if cfg.training.epochs == 0:
            checkpoint_path = cfg.output_directory
            logger.info(
                f"Saving last model checkpoint: "
                f"val_loss {val_loss:.5}, val_{cfg.prediction.metric} "
                f"{val_metric:.5} to {checkpoint_path}"
            )
            save_checkpoint(model=model, path=checkpoint_path, cfg=cfg)

        save_config_yaml(f"{cfg.output_directory}/cfg.yaml", cfg)

        save_prediction_outputs(cfg.experiment_name, experiment_path)

        flag_path = os.path.join(cfg.output_directory, "flags.json")
        write_flag(flag_path, "status", "finished")
        time_took = time.time() - global_start_time
        if time_took > 86400:
            # if more than one day, show days
            # need to subtract 1 day from time_took since strftime shows day of year
            # which starts counting at 1
            time_took_formatted = time.strftime(
                "%-jd %H:%M:%S", time.gmtime(float(time_took - 86400))
            )
        else:
            time_took_formatted = time.strftime(
                "%H:%M:%S", time.gmtime(float(time_took))
            )
        write_flag(flag_path, "info", f"Runtime: {time_took_formatted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-C", "--config", help="config filename", default=argparse.SUPPRESS
    )
    parser.add_argument("-Y", "--yaml", help="yaml filename", default=argparse.SUPPRESS)
    parser_args, unknown = parser.parse_known_args(sys.argv)

    if "config" in parser_args:
        cfg = load_config_py(parser_args.config)
    elif "yaml" in parser_args:
        cfg = load_config_yaml(parser_args.yaml)
    else:
        raise ValueError("Please, provide a configuration file")

    extra_args = []
    for arg_orig in unknown:
        if arg_orig.startswith(("-", "--")):
            arg = arg_orig.replace("-", "").split(".")
            try:
                arg_type = getattr(cfg, arg[0]).get_annotations()[arg[1]]
            except (AttributeError, KeyError):
                continue
            if arg_type == bool:
                parser.add_argument(arg_orig, type=util.strtobool)
            else:
                parser.add_argument(arg_orig, type=arg_type)
            extra_args.append(arg)

    args = parser.parse_args()

    for arg in extra_args:
        value = getattr(args, ".".join(arg))
        setattr(getattr(cfg, arg[0]), arg[1], value)

    out_dir = cfg.output_directory
    os.makedirs(out_dir, exist_ok=True)

    initialize_logging(cfg)

    try:
        run(cfg=cfg)
    except Exception:
        logging.error("Exception occurred during the run:", exc_info=True)
        kill_ddp_processes()
