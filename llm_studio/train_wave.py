import os

# Set this before importing any other modules to be on the safe side
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import logging
import sys
import time

import psutil

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def check_for_done(process_queue):
    """Checks for finished process ids

    Args:
        process_queue: list of process ids
    Returns:
        (True, process_idx) if there is any finished process
        (False, False) if there is not finished processes
    """

    for i, pid in enumerate(process_queue):
        zombie = False
        try:
            p = psutil.Process(pid)
            zombie = p.status() == "zombie"
        except psutil.NoSuchProcess:
            pass
        if not psutil.pid_exists(pid) or zombie:
            return True, i

    return False, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-Y", "--yaml", help="yaml filename", type=(str), default=argparse.SUPPRESS
    )
    parser.add_argument(
        "-Q",
        "--process-queue",
        help="process queue to wait for",
        default=argparse.SUPPRESS,
    )
    parser_args, _ = parser.parse_known_args(sys.argv)

    process_queue = []
    if "process_queue" in parser_args and parser_args.process_queue != "":
        process_queue = [int(x) for x in parser_args.process_queue.split(",")]

    while True:
        if len(process_queue) == 0:
            break
        done, num = check_for_done(process_queue)
        if done:
            process_queue.pop(num)
        else:
            time.sleep(30)

    # delayed imports from llm_studio, only after we want to start training
    import subprocess

    import torch

    from llm_studio.src.utils.config_utils import load_config_yaml
    from llm_studio.src.utils.exceptions import (
        LLMAugmentationsException,
        LLMDataException,
        LLMMetricException,
        LLMModelException,
        LLMTrainingException,
    )
    from llm_studio.src.utils.gpu_utils import is_oom_error
    from llm_studio.src.utils.logging_utils import initialize_logging, write_flag
    from llm_studio.src.utils.utils import kill_child_processes_and_current
    from llm_studio.train import run

    cfg = load_config_yaml(parser_args.yaml)

    flag_path = os.path.join(cfg.output_directory, "flags{}.json")

    # Check if DDP
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == 0:
            write_flag(flag_path.format(""), "status", "running")
    else:
        write_flag(flag_path.format(""), "status", "running")
        local_rank = 0

    try:
        run(cfg=cfg)
    except Exception as exception:
        initialize_logging(cfg)
        write_flag(flag_path.format(local_rank), "status", "failed")
        if is_oom_error(exception):
            logging.error(
                "GPU Out-of-Memory (OOM) error occurred. "
                "Please, reduce the batch size, or input data size, "
                "or model size. Or try gradient checkpointing.",
                exc_info=True,
            )
            write_flag(flag_path.format(local_rank), "info", "OOM error")

            logging.info(
                "<pre>"
                + subprocess.check_output(["nvidia-smi"]).decode("utf-8")
                + "</pre>"
            )

            if torch.cuda.is_available():
                logging.info(
                    "<pre>" + torch.cuda.memory_summary().replace("-", "=") + "</pre>"
                )

        elif isinstance(exception, LLMDataException):
            logging.error(
                "Data error occurred during H2O LLM Studio run:", exc_info=True
            )
            write_flag(flag_path.format(local_rank), "info", "Data error")
        elif isinstance(exception, LLMTrainingException):
            logging.error(
                "Training error occurred during H2O LLM Studio run:", exc_info=True
            )
            write_flag(flag_path.format(local_rank), "info", "Training error")
        elif isinstance(exception, LLMMetricException):
            logging.error(
                "Validation metric failed. Please make sure selected validation "
                "metric is suitable for your current problem setup.",
                exc_info=True,
            )
            write_flag(flag_path.format(local_rank), "info", "Metric error")
        elif isinstance(exception, LLMAugmentationsException):
            logging.error(
                "Custom augmentations error occurred during " "H2O LLM Studio run:",
                exc_info=True,
            )
            write_flag(flag_path.format(local_rank), "info", "Augmentations error")
        elif isinstance(exception, LLMModelException):
            logging.error(
                "Model error occurred during H2O LLM Studio run:",
                exc_info=True,
            )
            write_flag(flag_path.format(local_rank), "info", "Model error")
        else:
            logging.error(
                "Exception occurred during H2O LLM Studio run:", exc_info=True
            )
            write_flag(flag_path.format(local_rank), "info", "See logs")

        # Clean up any potential processes for this experiment
        kill_child_processes_and_current()
