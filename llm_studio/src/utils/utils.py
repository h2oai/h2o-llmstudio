import logging
import os
import pickle
import random
import zipfile
from typing import Any, Tuple

import dill
import numpy as np
import openai
import psutil
import torch

# types which can be shown directly in the UI without any extra nesting
KNOWN_TYPE_ANNOTATIONS = [int, float, bool, str, Tuple[str, ...]]

logger = logging.getLogger(__name__)


def set_seed(seed: int = 1234) -> None:
    """Sets the random seed.

    Args:
        seed: seed value
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def set_environment(cfg):
    """Sets and checks environment settings"""

    os.environ["OPENAI_API_KEY"] = cfg.environment.openai_api_token
    openai.api_key = cfg.environment.openai_api_token

    if "GPT" in cfg.prediction.metric and cfg.environment.openai_api_token == "":
        logger.warning("No OpenAI API Key set. Setting metric to BLEU. ")
        cfg.prediction.metric = "BLEU"

    return cfg


def kill_child_processes(parent_pid: int) -> bool:
    """Killing a process and all its child processes

    Args:
        parent_pid: process id of parent

    Returns:
        True or False in case of success or failure
    """

    logger.debug(f"Killing process id: {parent_pid}")

    try:
        parent = psutil.Process(parent_pid)
        if parent.status() == "zombie":
            return False
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
        return True
    except psutil.NoSuchProcess:
        logger.warning(f"Cannot kill process id: {parent_pid}. No such process.")
        return False


def kill_ddp_processes() -> None:
    """
    Killing all DDP processes from a single process.
    Firstly kills all children of a single DDP process (dataloader workers)
    Then kills all other DDP processes
    Then kills main parent DDP process
    """

    pid = os.getpid()
    parent_pid = os.getppid()

    current_process = psutil.Process(pid)
    children = current_process.children(recursive=True)
    for child in children:
        child.kill()

    parent_process = psutil.Process(parent_pid)
    children = parent_process.children(recursive=True)[::-1]
    for child in children:
        if child.pid == pid:
            continue
        child.kill()
    parent_process.kill()
    current_process.kill()


def add_file_to_zip(zf: zipfile.ZipFile, path: str) -> None:
    """Adds a file to the existing zip. Does nothing if file does not exist.

    Args:
        zf: zipfile object to add to
        path: path to the file to add
    """

    try:
        zf.write(path, os.path.basename(path))
    except Exception:
        pass


def load_dill(path: str) -> Any:
    """Loads a dill file

    Args:
        path: path of file to load

    Returns:
        Loaded object
    """

    with open(path, "rb") as binary_file:
        f = dill.load(binary_file)
        return f


def save_pickle(path: str, obj: Any, protocol: int = 4) -> None:
    """Saves object as pickle file

    Args:
        path: path of file to save
        obj: object to save
        protocol: protocol to use when saving pickle
    """

    with open(path, "wb") as pickle_file:
        pickle.dump(obj, pickle_file, protocol=protocol)


def save_dill(path: str, obj: Any) -> None:
    """Saves object as dill file

    Args:
        path: path of file to save
        obj: object to save
    """

    with open(path, "wb") as pickle_file:
        dill.dump(obj, pickle_file)


def copy_config(cfg: Any) -> Any:
    """Makes a copy of the config

    Args:
        cfg: config object
    Returns:
        copy of the config
    """

    # cfg.environment._cpu_comm can't be copied
    cpu_comm = None
    if cfg.environment._cpu_comm is not None:
        cpu_comm = cfg.environment._cpu_comm
        cfg.environment._cpu_comm = None
    cfg = dill.copy(cfg)
    cfg.environment._cpu_comm = cpu_comm

    return cfg


def save_config(path: str, cfg: Any) -> None:
    """Saves config as dill file

    Args:
        path: path of file to save to
        cfg: config to save
    """

    # cfg.environment._cpu_comm can't be saved
    cpu_comm = None
    if cfg.environment._cpu_comm is not None:
        cpu_comm = cfg.environment._cpu_comm
        cfg.environment._cpu_comm = None
    save_dill(path, cfg)
    cfg.environment._cpu_comm = cpu_comm


class DisableLogger:
    def __init__(self, level: int = logging.INFO):
        self.level = level

    def __enter__(self):
        logging.disable(self.level)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)
