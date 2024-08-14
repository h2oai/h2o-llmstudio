import logging
import os
import pickle
import random
import zipfile
from typing import Any

import numpy as np
import psutil
import torch

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
    if "GPT" in cfg.prediction.metric and os.getenv("OPENAI_API_KEY", "") == "":
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


def kill_child_processes(current_pid, exclude=None) -> None:
    """
    Kill all child processes of specified PID
    Optionally, excludes one PID
    """
    current_process = psutil.Process(current_pid)
    children = current_process.children(recursive=True)[::-1]
    for child in children:
        if child.pid == exclude:
            continue
        child.kill()


def kill_child_processes_and_current() -> None:
    """
    Kill all child processes of the current process, then terminates itself
    """
    current_pid = os.getpid()
    kill_child_processes(current_pid)
    current_process = psutil.Process(current_pid)
    current_process.kill()


def kill_sibling_ddp_processes() -> None:
    """
    Killing all sibling DDP processes from a single DDP process.
    """
    pid = os.getpid()
    parent_pid = os.getppid()
    kill_child_processes(parent_pid, exclude=pid)
    current_process = psutil.Process(pid)
    current_process.kill()


def add_file_to_zip(zf: zipfile.ZipFile, path: str, folder=None) -> None:
    """Adds a file to the existing zip. Does nothing if file does not exist.

    Args:
        zf: zipfile object to add to
        path: path to the file to add
        folder: folder in the zip to add the file to
    """

    try:
        if folder is None:
            zip_path = os.path.basename(path)
        else:
            zip_path = os.path.join(folder, os.path.basename(path))
        zf.write(path, zip_path)
    except Exception:
        logger.warning(f"File {path} could not be added to zip.")


def save_pickle(path: str, obj: Any, protocol: int = 4) -> None:
    """Saves object as pickle file

    Args:
        path: path of file to save
        obj: object to save
        protocol: protocol to use when saving pickle
    """

    with open(path, "wb") as pickle_file:
        pickle.dump(obj, pickle_file, protocol=protocol)


class DisableLogger:
    def __init__(self, level: int = logging.INFO):
        self.level = level

    def __enter__(self):
        logging.disable(self.level)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


class PatchedAttribute:
    """
    Patches an attribute of an object for the duration of this context manager.
    Similar to unittest.mock.patch,
    but works also for properties that are not present in the original class

    >>> class MyObj:
    ...     attr = 'original'
    >>> my_obj = MyObj()
    >>> with PatchedAttribute(my_obj, 'attr', 'patched'):
    ...     print(my_obj.attr)
    patched
    >>> print(my_obj.attr)
    original
    >>> with PatchedAttribute(my_obj, 'new_attr', 'new_patched'):
    ...     print(my_obj.new_attr)
    new_patched
    >>> assert not hasattr(my_obj, 'new_attr')
    """

    def __init__(self, obj, attribute, new_value):
        self.obj = obj
        self.attribute = attribute
        self.new_value = new_value
        self.original_exists = hasattr(obj, attribute)
        if self.original_exists:
            self.original_value = getattr(obj, attribute)

    def __enter__(self):
        setattr(self.obj, self.attribute, self.new_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_exists:
            setattr(self.obj, self.attribute, self.original_value)
        else:
            delattr(self.obj, self.attribute)


def create_symlinks_in_parent_folder(directory):
    """Creates symlinks for each item in a folder in the parent folder

    Only creates symlinks for items at the root level of the directory.
    """

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")

    parent_directory = os.path.dirname(directory)

    for file in os.listdir(directory):
        src = os.path.join(directory, file)
        dst = os.path.join(parent_directory, file)
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
