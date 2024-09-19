import os

from llm_studio.python_configs.cfg_checks import check_config_for_errors

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

import deepspeed
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.integrations import HfDeepSpeedConfig

from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src.loggers import MainLogger
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
    activate_neftune,
    check_disk_space,
    get_ds_config,
    get_number_of_validation_epochs,
    get_optimizer,
    get_scheduler,
    get_torch_dtype,
    load_checkpoint,
    run_inference,
    save_checkpoint,
    save_predictions,
    wrap_model_distributed,
)
from llm_studio.src.utils.utils import (
    check_metric,
    create_symlinks_in_parent_folder,
    kill_child_processes_and_current,
    kill_sibling_ddp_processes,
    set_seed,
)

logger = logging.getLogger(__name__)


def run_eval(
    cfg: DefaultConfigProblemBase,
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
        is_training = model.training
        model.eval()
        val_data: Dict[str, Any] = run_inference(
            cfg, model, val_dataloader, mode
        )  # type: ignore
        model.train(is_training)

    # Sync validation predictions across GPUs
    if cfg.environment._distributed and cfg.environment._distributed_inference:
        for key, value in val_data.items():
            val_data[key] = sync_across_processes(
                value, cfg.environment._world_size, group=cfg.environment._cpu_comm
            )

    if cfg.environment._local_rank != 0:
        # data has been synced, so we can return early on other ranks
        if cfg.environment._distributed:
            torch.distributed.barrier()
        return 0, 0

    # Drop any extra observations
    for k, v in val_data.items():
        val_data[k] = v[: len(val_dataloader.dataset)]  # type: ignore

    val_data = val_dataloader.dataset.postprocess_output(  # type: ignore
        cfg=cfg, df=val_df, output=val_data
    )
    val_loss = np.mean(val_data.get("loss", torch.tensor(0)).float().cpu().numpy())
    # postprocess_output only runs on rank 0 to save time/memory
    val_metric = np.mean(val_data["metrics"])
    logger.info(f"{mode.capitalize()} {cfg.prediction.metric}: {val_metric:.5f}")

    for key in val_data:
        if key.startswith("additional_log_") or key == "loss":
            value = np.mean(val_data[key].float().cpu().numpy())
            key = key.replace("additional_log_", "")
            logger.info(f"Mean {mode} {key}: {value:.5f}")
            cfg.logging._logger.log(
                mode,
                key,
                value,
                step=cfg.environment._curr_step / cfg.environment._step_log_denominator,
            )
    cfg.logging._logger.log(
        mode,
        cfg.prediction.metric,
        val_metric,
        step=cfg.environment._curr_step / cfg.environment._step_log_denominator,
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

    return val_loss, val_metric


def run_train(
    cfg: DefaultConfigProblemBase,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch_steps,
    train_dataloader,
    val_dataloader,
    val_df: pd.DataFrame,
):
    """Runs the training loop.

    Args:
        cfg: DefaultConfigProblemBase config object
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
    if (
        hasattr(cfg.augmentation, "neftune_noise_alpha")
        and cfg.augmentation.neftune_noise_alpha > 0
    ):
        activate_neftune(model, cfg.augmentation.neftune_noise_alpha)

    scaler: GradScaler | None = None
    if cfg.environment.mixed_precision:
        scaler = GradScaler(
            enabled=(cfg.environment.mixed_precision_dtype == "float16")
        )

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
        logger.info(f"Training Epoch: {epoch + 1} / {cfg.training.epochs}")

        if (
            cfg.environment._distributed
            and not cfg.environment.use_deepspeed
            and hasattr(train_dataloader.sampler, "set_epoch")
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
        logger.info(f"Evaluation step: {evaluation_step}")

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

            # only need to sync gradients at last step of grad accumulation
            model.require_backward_grad_sync = itr % cfg.training.grad_accumulation == 0

            # Forward pass
            with autocast(
                enabled=cfg.environment.mixed_precision,
                dtype=get_torch_dtype(cfg.environment.mixed_precision_dtype),
            ):
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
            if (
                cfg.environment.mixed_precision
                and len(cfg.environment.gpus)
                and not cfg.environment.use_deepspeed
            ):
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
                if cfg.environment.use_deepspeed:
                    model.backward(loss)  # type: ignore[operator]
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
                    "train",
                    "loss",
                    losses[-1],
                    step=cfg.environment._curr_step
                    / cfg.environment._step_log_denominator,
                )
                cfg.logging._logger.log(
                    "meta",
                    "lr",
                    optimizer.param_groups[0]["lr"],
                    step=cfg.environment._curr_step
                    / cfg.environment._step_log_denominator,
                )
                if cfg.training.differential_learning_rate_layers:
                    cfg.logging._logger.log(
                        "meta",
                        "lr_diff",
                        optimizer.param_groups[2]["lr"],
                        step=cfg.environment._curr_step
                        / cfg.environment._step_log_denominator,
                    )

                cfg.logging._logger.log(
                    "internal",
                    "current_step",
                    cfg.environment._curr_step,
                )
                for key in output_dict:
                    if key.startswith("additional_log_"):
                        cfg.logging._logger.log(
                            "train",
                            key.replace("additional_log_", ""),
                            output_dict[key].item(),
                            step=cfg.environment._curr_step
                            / cfg.environment._step_log_denominator,
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
                # TODO: Move back after fixing slow generation of deepspeed.
                if cfg.training.save_checkpoint == "last":
                    logger.info(
                        f"Saving last model checkpoint to {cfg.output_directory}"
                    )
                    save_checkpoint(model=model, path=cfg.output_directory, cfg=cfg)
                elif cfg.training.save_checkpoint == "each_evaluation_epoch":
                    checkpoint_path = os.path.join(
                        cfg.output_directory, f"epoch_{epoch}_step_{itr}"
                    )
                    logger.info(f"Saving model checkpoint to {checkpoint_path}")
                    save_checkpoint(model=model, path=checkpoint_path, cfg=cfg)
                    create_symlinks_in_parent_folder(checkpoint_path)

                val_loss, val_metric = run_eval(
                    cfg=cfg, model=model, val_dataloader=val_dataloader, val_df=val_df
                )

                if cfg.training.save_checkpoint == "best":
                    if objective_op(val_metric, best_val_metric):
                        logger.info(
                            f"Saving best model checkpoint: "
                            f"val_{cfg.prediction.metric} {best_val_metric:.5} -> "
                            f"{val_metric:.5} to {cfg.output_directory}"
                        )
                        save_checkpoint(model=model, path=cfg.output_directory, cfg=cfg)
                        best_val_metric = val_metric

                model.train()

        progress_bar.close()
        del progress_bar

        if cfg.environment._distributed:
            torch.cuda.synchronize(device=cfg.environment._local_rank)
            torch.distributed.barrier()

        if cfg.environment._local_rank == 0:
            cfg.logging._logger.log("internal", "epoch", epoch + 1)

    if cfg.environment._distributed:
        torch.distributed.barrier()

    return val_loss, val_metric


def run(cfg: DefaultConfigProblemBase) -> float:
    """Runs the routine.

    Args:
        cfg: DefaultConfigProblemBase config object with all the hyperparameters
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

    if (
        cfg.architecture.backbone_dtype in ["int8", "int4"]
        and cfg.environment.use_deepspeed
    ):
        raise ValueError(
            f"Deepspeed do not support backbone type {cfg.architecture.backbone_dtype}."
            + " Please set backbone type to float16 or bfloat16 for using deepspeed."
        )

    # Prepare environment
    if "WORLD_SIZE" in os.environ:
        cfg.environment._distributed = int(os.environ["WORLD_SIZE"]) > 1
        cfg.environment._local_rank = int(os.environ["LOCAL_RANK"])
    else:
        cfg.environment._distributed = False
        cfg.environment._local_rank = 0

    initialize_logging(cfg)

    # Check for errors in the configuration
    errors = check_config_for_errors(cfg)
    for i in range(len(errors["title"])):
        if errors["type"][i] == "error":
            logger.error(f"{errors['title'][i]}: {errors['message'][i]}")
        else:
            logger.warning(f"{errors['title'][i]}: {errors['message'][i]}")

    if any(error_type == "error" for error_type in errors["type"]):
        raise LLMTrainingException(
            "Configuration contains errors. Please fix them before proceeding."
        )

    if cfg.environment._distributed:
        cfg.environment._device = "cuda:%d" % cfg.environment._local_rank
        if cfg.environment.use_deepspeed:
            deepspeed.init_distributed()
        else:
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
        cfg.environment._device = (
            "cuda:0"
            if (torch.cuda.is_available() and len(cfg.environment.gpus) > 0)
            else "cpu"
        )
        if cfg.environment._device == "cpu":
            logger.warning("Training on CPU. This will be slow.")

    set_seed(cfg.environment._seed)
    logger.info(f"Problem Type: {cfg.problem_type}")
    logger.info(f"Global random seed: {cfg.environment._seed}")

    cfg = check_metric(cfg)

    # we need to get train dataframe and number of labels if not set or in training mode
    logger.info("Preparing the data...")
    train_df, val_df = get_data(cfg)

    # We allow system prompt column to be missing in validation DataFrame, but let us
    # assert that it does exist in the train DataFrame.
    if hasattr(cfg.dataset, "system_column") and cfg.dataset.system_column != "None":
        if cfg.dataset.system_column not in train_df.columns:
            raise LLMTrainingException(
                f"System column '{cfg.dataset.system_column}' not found in train "
                "DataFrame."
            )

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

        total_validation_steps = (
            len(val_dataloader)
            * (num_eval_epochs + int(cfg.training.evaluate_before_training))
            * val_batch_size
            * cfg.environment._world_size
        )

        if cfg.logging.log_step_size == "relative":
            cfg.environment._step_log_denominator = total_training_steps
        else:
            cfg.environment._step_log_denominator = 1

    # Prepare model and optimizer
    if cfg.environment.use_deepspeed:
        ds_config = get_ds_config(cfg)
        # keep this object alive.
        dschf = HfDeepSpeedConfig(ds_config)  # noqa: F841
    with torch.device(cfg.environment._device):
        model = cfg.architecture.model_class(cfg)
        check_disk_space(model, cfg.output_directory)

        # load model weights
        if cfg.architecture.pretrained_weights != "":
            # Do not load strictly if continue training from the previous experiment
            load_checkpoint(cfg, model, strict=cfg.training.epochs == -1)
    model.to(cfg.environment._device)

    epoch_steps = len(train_dataloader)
    optimizer = get_optimizer(model=model, cfg=cfg)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer, epoch_steps=epoch_steps)

    if cfg.environment._distributed:
        (
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            scheduler,
        ) = wrap_model_distributed(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            cfg=cfg,
        )

    if cfg.environment.compile_model:
        # deepspeed do not support torch.compile
        if cfg.environment.use_deepspeed:
            logger.warning(
                "Deepspeed is active, but it doesn't support torch.compile."
                "Skipping compilation for this experiment."
            )
        else:
            if cfg.environment._distributed:
                model.module.backbone = torch.compile(model.module.backbone)
            else:
                model.backbone = torch.compile(model.backbone)

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
            "internal", "total_training_steps", total_training_steps
        )

        cfg.logging._logger.log(
            "internal", "total_validation_steps", total_validation_steps
        )

        cfg.logging._logger.log(
            "internal",
            "global_start_time",
            global_start_time,
        )
        # re-save config
        save_config_yaml(f"{cfg.output_directory}/cfg.yaml", cfg)

    val_loss, val_metric = run_train(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_steps=epoch_steps,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_df=val_df,
    )

    # reset external logging
    if cfg.environment._local_rank == 0:
        cfg.logging._logger.reset_external()

    experiment_path = f"{cfg.output_directory}"

    if cfg.training.epochs == 0 and cfg.training.save_checkpoint != "disable":
        checkpoint_path = cfg.output_directory
        logger.info(f"Saving last model checkpoint to {checkpoint_path}")
        save_checkpoint(model=model, path=checkpoint_path, cfg=cfg)

    if cfg.environment._local_rank == 0:
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

    return val_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-C", "--config", help="config filename", type=(str), default=argparse.SUPPRESS
    )
    parser.add_argument(
        "-Y", "--yaml", help="yaml filename", type=(str), default=argparse.SUPPRESS
    )
    parser_args, unknown = parser.parse_known_args(sys.argv)

    if "config" in parser_args:
        logging.warning(
            "Using deprecated -C argument. Please use -Y instead to load yaml."
        )
        cfg: DefaultConfigProblemBase = load_config_py(parser_args.config)
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

    try:
        run(cfg=cfg)
    except Exception:
        logging.error("Exception occurred during the run:", exc_info=True)
        if ("WORLD_SIZE" in os.environ) and (int(os.environ["WORLD_SIZE"]) > 1):
            kill_sibling_ddp_processes()
        else:
            kill_child_processes_and_current()
