import asyncio
import collections
import contextlib
import dataclasses
import glob
import json
import logging
import math
import os
import random
import re
import shutil
import socket
import string
import subprocess
import time
import uuid
import zipfile
from collections import defaultdict
from contextlib import closing
from functools import partial
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Type, Union

import GPUtil
import h2o_drive
import numpy as np
import pandas as pd
import psutil
import yaml
from azure.storage.filedatalake import DataLakeServiceClient
from boto3.session import Session
from botocore.handlers import disable_signing
from datasets import load_dataset
from h2o_wave import Choice, Q, ui
from pandas.core.frame import DataFrame
from sqlitedict import SqliteDict

from llm_studio.app_utils.db import Experiment
from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src import possible_values
from llm_studio.src.utils.config_utils import (
    _get_type_annotation_error,
    load_config_yaml,
    parse_cfg_dataclass,
    save_config_yaml,
)
from llm_studio.src.utils.data_utils import is_valid_data_frame, read_dataframe
from llm_studio.src.utils.export_utils import get_size_str
from llm_studio.src.utils.type_annotations import KNOWN_TYPE_ANNOTATIONS

from .config import default_cfg

logger = logging.getLogger(__name__)


class GridCheckError(Exception):
    pass


def get_user_id(q):
    return q.auth.subject


def get_user_name(q):
    return q.auth.username


def get_data_dir(q):
    return os.path.join(default_cfg.llm_studio_workdir, default_cfg.data_folder, "user")


def get_database_dir(q):
    return os.path.join(default_cfg.llm_studio_workdir, default_cfg.data_folder, "dbs")


def get_output_dir(q):
    return os.path.join(
        default_cfg.llm_studio_workdir, default_cfg.output_folder, "user"
    )


def get_download_dir(q):
    return os.path.join(
        default_cfg.llm_studio_workdir, default_cfg.output_folder, "download"
    )


def get_user_db_path(q):
    return os.path.join(get_database_dir(q), "user.db")


def get_usersettings_path(q):
    return os.path.join(get_database_dir(q), f"{get_user_id(q)}.settings")


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def start_process(
    cfg: DefaultConfigProblemBase, gpu_list: List, process_queue: List, env_vars: Dict
) -> subprocess.Popen:
    """Starts train.py for a given configuration setting

    Args:
        cfg: DefaultConfigProblemBase config
        gpu_list: list of GPUs to use for the training
        process_queue: list of processes to wait for before starting the training
        env_vars: dictionary of ENV variables to pass to the training process
    Returns:
        Process

    """

    num_gpus = len(gpu_list)
    config_name = os.path.join(cfg.output_directory, "cfg.yaml")
    env = {**os.environ, **env_vars}

    if num_gpus == 0:
        cmd = [
            "python",
            "train_wave.py",
            "-Y",
            config_name,
        ]
    # Do not delete for debug purposes
    # elif num_gpus == 1:
    #     cmd = [
    #         "env",
    #         f"CUDA_VISIBLE_DEVICES={','.join(gpu_list)}",
    #         "python",
    #         "-u",
    #         "train_wave.py",
    #         "-P",
    #         config_name,
    #     ]
    else:
        free_port = find_free_port()
        if cfg.environment.use_deepspeed:
            logger.info("Starting deepspeed...")
            cmd = [
                "env",
                "deepspeed",
                "--include",
                f"localhost:{','.join(gpu_list)}",
                "--master_port",
                f"{str(free_port)}",
                "train_wave.py",
                "-Y",
                config_name,
            ]
        else:
            logger.info("Starting torchrun...")
            cmd = [
                "env",
                f"CUDA_VISIBLE_DEVICES={','.join(gpu_list)}",
                "torchrun",
                f"--nproc_per_node={str(num_gpus)}",
                f"--master_port={str(free_port)}",
                "train_wave.py",
                "-Y",
                config_name,
            ]

    if len(process_queue) > 0:
        cmd.append("-Q")
        cmd.append(",".join([str(x) for x in process_queue]))

    p = subprocess.Popen(
        cmd,
        env=env,
    )

    logger.info(f"Percentage of RAM memory used: {psutil.virtual_memory().percent}")

    return p


def clean_macos_artifacts(path: str) -> None:
    """Cleans artifacts from MacOSX zip archives

    Args:
        path: path to the unzipped directory
    """

    shutil.rmtree(os.path.join(path, "__MACOSX/"), ignore_errors=True)

    for ds_store in glob.glob(os.path.join(path, "**/.DS_Store"), recursive=True):
        try:
            os.remove(ds_store)
        except OSError:
            pass


def s3_session(aws_access_key: str, aws_secret_key: str) -> Any:
    """Establishes s3 session

    Args:
        aws_access_key: s3 access key
        aws_secret_key: s3 secret key

    Returns:
        Session

    """

    session = Session(
        aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
    )
    s3 = session.resource("s3")
    # if no key is present, disable signing
    if aws_access_key == "" and aws_secret_key == "":
        s3.meta.client.meta.events.register("choose-signer.s3.*", disable_signing)

    return s3


def filter_valid_files(files) -> List[str]:
    valid_files = [
        file
        for file in files
        if any([file.endswith(ext) for ext in default_cfg.allowed_file_extensions])
    ]

    return valid_files


def s3_file_options(
    bucket: str, aws_access_key: str, aws_secret_key: str
) -> List[str] | Exception:
    """ "Returns all zip files in the target s3 bucket

    Args:
        bucket: s3 bucket name
        aws_access_key: s3 access key
        aws_secret_key: s3 secret key

    Returns:
        List of zip files in bucket or Exception in case of access error

    """

    try:
        bucket = bucket.replace("s3://", "")
        if bucket[-1] == os.sep:
            bucket = bucket[:-1]

        bucket_split = bucket.split(os.sep)
        bucket = bucket_split[0]
        s3 = s3_session(aws_access_key, aws_secret_key)
        s3_bucket = s3.Bucket(bucket)

        folder = "/".join(bucket_split[1:])

        files = []
        for s3_file in s3_bucket.objects.filter(Prefix=f"{folder}/"):
            if s3_file.key == f"{folder}/":
                continue

            files.append(s3_file.key)

        files = filter_valid_files(files)
        return files

    except Exception as e:
        logger.warning(f"Can't load S3 datasets list: {e}")
        return e


def convert_file_size(size: float):
    """Converts file size to human readable format

    Args:
        size: size in bytes

    Returns:
        size in readable format
    """

    if size == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size, 1024)))
    p = math.pow(1024, i)
    s = round(size / p, 2)
    return "%.2f %s" % (s, size_name[i])


class S3Progress:
    """Progress update for s3 downloads

    Source:
        https://stackoverflow.com/a/59843153/1281171

    """

    def __init__(self, q: Q, size: float) -> None:
        """Initialize

        Args:
            q: Q
            size: size of the file to download
        """

        self._q: Q = q
        self._size: float = size
        self._seen_so_far: float = 0.0
        self._percentage: float = 0.0

    def progress(self, bytes_amount: float):
        """Update progress

        Args:
            bytes_amount: amount of bytes downloaded
        """

        self._seen_so_far += bytes_amount
        self._percentage = (self._seen_so_far / self._size) * 100.0

    async def update_ui(self):
        """Update progress in UI"""

        self._q.page["meta"].dialog = ui.dialog(
            title="S3 file download in progress",
            blocking=True,
            items=[
                ui.progress(
                    label="Please be patient...",
                    caption=(
                        f"{convert_file_size(self._seen_so_far)} of "
                        f"{convert_file_size(self._size)} "
                        f"({self._percentage:.2f}%)"
                    ),
                    value=self._percentage / 100,
                )
            ],
        )
        await self._q.page.save()

    async def poll(self):
        """Update wave ui"""

        while self._percentage / 100 < 1:
            await self.update_ui()
            await self._q.sleep(0.1)
        await self.update_ui()


def s3_download_coroutine(q: Q, filename: str):
    download_folder = f"{get_data_dir(q)}/tmp"
    download_folder = get_valid_temp_data_folder(q, download_folder)

    if os.path.exists(download_folder):
        shutil.rmtree(download_folder)
    os.makedirs(download_folder, exist_ok=True)

    downloaded_zip = f"{download_folder}/{filename.split('/')[-1]}"

    q.page["dataset/import"] = ui.form_card(box="content", items=[])
    return downloaded_zip, download_folder


def extract_if_zip(file, actual_path):
    if file.endswith("zip"):
        with zipfile.ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(actual_path)

        os.remove(file)
        clean_macos_artifacts(actual_path)


async def s3_download(
    q: Q, bucket, filename, aws_access_key, aws_secret_key
) -> Tuple[str, str]:
    """Downloads a file from s3

    Args:
        q: Q
        bucket: s3 bucket name
        filename: filename to download
        aws_access_key: s3 access key
        aws_secret_key: s3 secret key

    Returns:
        Download location path
    """
    bucket = bucket.replace("s3://", "")
    if bucket[-1] == os.sep:
        bucket = bucket[:-1]

    bucket = bucket.split(os.sep)[0]

    s3 = s3_session(aws_access_key, aws_secret_key)

    file, s3_path = s3_download_coroutine(q, filename)

    progress = S3Progress(
        q, (s3.meta.client.head_object(Bucket=bucket, Key=filename))["ContentLength"]
    )

    poll_future = asyncio.create_task(progress.poll())

    def download_file():
        s3.Bucket(bucket).download_file(filename, file, Callback=progress.progress)

    await q.run(download_file)
    await poll_future

    extract_if_zip(file, s3_path)

    return s3_path, "".join(filename.split("/")[-1].split(".")[:-1])


def azure_file_options(conn_string: str, container: str) -> List[str]:
    """Returns all zip files in the target azure datalake container

    Args:
        conn_string: connection string
        container: container including sub-paths

    Returns:
        - List of files in storage or empty list in case of access error

    """

    try:
        service_client = DataLakeServiceClient.from_connection_string(  # type: ignore
            conn_string
        )

        container_split = container.split(os.sep)
        container = container_split[0]

        folder = "/".join(container_split[1:])

        file_system_client = service_client.get_file_system_client(
            file_system=container
        )

        files = file_system_client.get_paths(path=folder)
        files = next(files.by_page())  # type: ignore[arg-type]
        files = [x.name for x in files]  # type: ignore[assignment]
        return filter_valid_files(files)

    except Exception as e:
        logger.warning(f"Can't load Azure datasets list: {e}")
        return []


async def download_progress(q: Q, title, seen_so_far, total_len):
    if seen_so_far is not None and total_len is not None:
        percentage = seen_so_far / total_len
        value = percentage
        caption = (
            f"{convert_file_size(seen_so_far)} of "
            f"{convert_file_size(total_len)} "
            f"({percentage * 100:.2f}%)"
        )
    else:
        value = None
        caption = None

    q.page["meta"].dialog = ui.dialog(
        title=title,
        blocking=True,
        items=[ui.progress(label="Please be patient...", caption=caption, value=value)],
    )
    await q.page.save()


async def azure_download(
    q: Q, conn_string: str, container: str, filename: str
) -> Tuple[str, str]:
    """Downloads a file from azure

    Args:
        q: Q
        conn_string: connection string
        container: container
        filename: filename to download

    Returns:
        Download location path
    """

    service_client = DataLakeServiceClient.from_connection_string(  # type: ignore
        conn_string
    )

    container_split = container.split(os.sep)
    container = container_split[0]

    file_system_client = service_client.get_file_system_client(file_system=container)

    filename_split = filename.split(os.sep)
    folder = "/".join(filename_split[:-1])
    filename = filename_split[-1]

    rnd_folder = "".join(random.choice(string.digits) for i in range(10))
    azure_path = f"{get_data_dir(q)}/tmp_{rnd_folder}"
    azure_path = get_valid_temp_data_folder(q, azure_path)

    if os.path.exists(azure_path):
        shutil.rmtree(azure_path)
    os.makedirs(azure_path, exist_ok=True)

    file = f"{azure_path}/{filename}"

    file_client = file_system_client.get_file_client(f"{folder}/{filename}")

    download = file_client.download_file()

    blocks = download.chunks()

    seen_so_far = 0
    with open(file, "wb") as local_file:
        for block in blocks:
            local_file.write(block)

            seen_so_far += len(block)

            await download_progress(
                q,
                "Azure Datalake file download in progress",
                seen_so_far,
                len(blocks),  # type: ignore[arg-type]
            )

    extract_if_zip(file, azure_path)

    return azure_path, "".join(filename.split(".")[:-1])


async def local_download(q: Q, filename: str) -> Tuple[str, str]:
    """Downloads a file from local path

    Args:
        q: Q
        filename: filename to download

    Returns:
        Download location path
    """

    local_path = f"{get_data_dir(q)}/tmp"
    local_path = get_valid_temp_data_folder(q, local_path)

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)

    shutil.copy2(filename, local_path)

    zip_file = f"{local_path}/{filename.split('/')[-1]}"
    extract_if_zip(zip_file, local_path)

    return local_path, "".join(filename.split("/")[-1].split(".")[:-1])


async def kaggle_download(
    q: Q, command: str, kaggle_access_key: str, kaggle_secret_key: str
) -> Tuple[str, str]:
    """ "Downloads a file from kaggle

    Args:
        q: Q
        command: kaggle api command
        kaggle_access_key: kaggle access key
        kaggle_secret_key: kaggle secret key

    Returns:
        Download location path
    """

    kaggle_path = f"{get_data_dir(q)}/tmp"
    kaggle_path = get_valid_temp_data_folder(q, kaggle_path)

    if os.path.exists(kaggle_path):
        shutil.rmtree(kaggle_path)
    os.makedirs(kaggle_path, exist_ok=True)

    command_run = []
    if kaggle_access_key != "":
        command_run += ["env", f"KAGGLE_USERNAME={kaggle_access_key}"]
    if kaggle_secret_key != "":
        command_run += ["env", f"KAGGLE_KEY={kaggle_secret_key}"]
    command_run += command.split(" ") + ["-p", kaggle_path]
    subprocess.run(command_run)

    try:
        zip_file = f"{kaggle_path}/{command.split(' ')[-1].split('/')[-1]}.zip"
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(kaggle_path)
        os.remove(zip_file)
    except Exception:
        pass

    clean_macos_artifacts(kaggle_path)

    for f in glob.glob(kaggle_path + "/*"):
        if ".zip" in f and zip_file not in f:
            with zipfile.ZipFile(f, "r") as zip_ref:
                zip_ref.extractall(kaggle_path)

            clean_macos_artifacts(kaggle_path)

    return kaggle_path, "".join(command.split(" ")[-1].split("/")[-1])


async def huggingface_download(
    q: Q, huggingface_dataset: str, huggingface_split: str
) -> Tuple[str, str]:
    """Downloads a dataset from Hugging Face

    Args:
        q: Q
        huggingface_dataset: HF dataset
        huggingface_split: Dataset split

    Returns:
        Download location path
    """

    huggingface_path = f"{get_data_dir(q)}/tmp"
    huggingface_path = get_valid_temp_data_folder(q, huggingface_path)

    if os.path.exists(huggingface_path):
        shutil.rmtree(huggingface_path)
    os.makedirs(huggingface_path, exist_ok=True)

    token = q.client["dataset/import/huggingface_api_token"]
    if token == "":
        token = None

    # Download the dataset
    dataset = load_dataset(
        huggingface_dataset, split=huggingface_split, token=token
    ).to_pandas()
    filename = f"{huggingface_dataset.split('/')[-1]}_{huggingface_split}"
    dataset_path = os.path.join(huggingface_path, f"{filename}.pq")
    dataset.to_parquet(dataset_path, index=False)

    return huggingface_path, filename


async def h2o_drive_download(q: Q, filename) -> Tuple[str, str]:
    """Downloads a file from H2O Drive

    Args:
        q: Q
        filename: filename to download

    Returns:
        Download location path
    """
    drive = await h2o_drive.Drive(q.auth.access_token)
    my_home_space = drive.my_bucket().home()

    local_path = f"{get_data_dir(q)}/tmp"
    local_path = get_valid_temp_data_folder(q, local_path)

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)

    zip_file = f"{local_path}/{filename.split('/')[-1]}"

    await my_home_space.download_file(filename, zip_file)

    extract_if_zip(zip_file, local_path)

    return local_path, "".join(filename.split("/")[-1].split(".")[:-1])


async def h2o_drive_file_options(q: Q) -> List[str] | Exception:
    """ "Returns all zip files in the H2O Drive

    Args:

    Returns:
        List of zip files in bucket or Exception in case of access error

    """
    try:
        drive = await h2o_drive.Drive(q.auth.access_token)
        my_home_space = drive.my_bucket().home()

        files = []
        for h2o_drive_file in await my_home_space.list_objects():
            files.append(h2o_drive_file.key)

        files = filter_valid_files(files)
        return files

    except Exception as e:
        logger.warning(f"Can't connect to H2O Drive: {e}")
        return e


def clean_error(error: str):
    """Cleans some error messages

    Args:
        error: original error message

    Returns:
        Cleaned error message

    """

    if "UNIQUE constraint failed: datasets.name" in error:
        error = "Dataset name already exists, please choose a different one."
    elif "No such file or directory" in error:
        error = "Import failed."

    return error


def remove_model_type(problem_type: str) -> str:
    """Removes model type from problem type

    Args:
        problem_type: problem type

    Returns:
        Cleaned raw problem type

    """
    if "_config_" in problem_type:
        problem_type = problem_type.split("_config_")[0] + "_config"
    return problem_type


def add_model_type(problem_type: str, model_type: str) -> str:
    """Adds model type to problem type

    Args:
        problem_type: problem type
        model_type: model type

    Returns:
        problem type including model type

    """
    problem_type = remove_model_type(problem_type)
    if model_type != "":
        problem_type = f"{problem_type}_{model_type}"
    return problem_type


def get_problem_categories() -> List[Tuple[str, str]]:
    """Returns all available problem category choices

    Returns:
        List of tuples, each containing the raw problem category name
        and the problem category name as label.
    """

    problem_categories: List[Tuple[str, str]] = []
    for c in default_cfg.problem_categories:
        cc = (c, make_label(c))
        problem_categories.append(cc)
    return problem_categories


def get_problem_types(category: Optional[str] = None) -> List[Tuple[str, str]]:
    """Returns all problem type choices

    Args:
        category: optional category to filter for

    Returns:
        List of tuples, each containing the raw problem type name
        and the problem type name as label.
    """
    problem_types: List[Tuple[str, str]] = []
    for c in default_cfg.problem_types:
        if category is not None and not c.startswith(category):
            continue
        cc = (c, make_label("_".join(c.split("_")[1:]).replace("_config", "")))
        problem_types.append(cc)

    return problem_types


def get_model_types(problem_type: str) -> List[Tuple[str, str]]:
    """Returns all model types for a given problem type

    Args:
        problem_type: problem type name

    Returns:
        List of model types and their labels
    """

    model_types = []
    for c in sorted(os.listdir("llm_studio/python_configs")):
        if "_config_" not in c:
            continue
        if problem_type in c:
            c = c.replace(".py", "").split("_config_")[1]
            model_types.append((c, make_label(c[1:])))

    return model_types


def get_dataset(
    k: str,
    v: Any,
    q: Q,
    limit: Optional[List[str]] = None,
    pre: str = "experiment/start",
) -> Tuple[List[str], Any]:
    """
    Get the dataset and the preliminary default value for a setting.
    The default value may still be overridden by the `possible_values.DatasetValue`
    instances if it is not a valid choice.

    Args:
        k: key for the setting
        v: value for the setting
        q: Q
        limit: list of keys to limit
        pre: prefix for client key

    Returns:
        List of possible values, the preliminary default value.
    """

    if q.client[f"{pre}/dataset"] is None:
        dataset_id = 1
    else:
        dataset_id = int(q.client[f"{pre}/dataset"])

    dataset = q.client.app_db.get_dataset(dataset_id)

    if dataset is None:
        return None, ""

    dataset = dataset.__dict__

    dataset_cfg = load_config_yaml(dataset["config_file"]).dataset.__dict__

    for kk, vv in dataset_cfg.items():
        dataset[kk] = vv

    dataset["dataframe"] = q.client[f"{pre}/cfg/dataframe"]

    if q.client[f"{pre}/cfg_mode/from_dataset"] and (limit is None or k in limit):
        v = dataset[k] if k in dataset else v

    if limit is not None and k not in limit:
        return None, v

    # we need to not reset dataset settings when changing expert mode
    if q.client[f"{pre}/cfg_mode/from_dataset_args"]:
        v = q.client[f"{pre}/cfg/{k}"]

    return dataset, v


def get_ui_element(
    k: str,
    v: Any,
    poss_values: Any,
    type_annotation: Type,
    tooltip: str,
    password: bool,
    trigger: bool,
    q: Q,
    pre: str = "",
) -> Any:
    """Returns a single ui element for a given config entry

    Args:
        k: key
        v: value
        poss_values: possible values
        type_annotation: type annotation
        tooltip: tooltip
        password: flag for whether it is a password
        trigger: flag for triggering the element
        q: Q
        pre: optional prefix for ui key
        get_default: flag for whether to get the default values

    Returns:
        Ui element

    """
    assert type_annotation in KNOWN_TYPE_ANNOTATIONS

    # Overwrite current values with values from yaml
    if pre == "experiment/start/cfg/":
        if q.args["experiment/upload_yaml"] and "experiment/yaml_data" in q.client:
            if (k in q.client["experiment/yaml_data"].keys()) and (
                k != "experiment_name"
            ):
                q.client[pre + k] = q.client["experiment/yaml_data"][k]

    if type_annotation in (int, float):
        if not isinstance(poss_values, possible_values.Number):
            raise ValueError(
                "Type annotations `int` and `float` need a `possible_values.Number`!"
            )

        val = q.client[pre + k] if q.client[pre + k] is not None else v

        min_val = (
            type_annotation(poss_values.min) if poss_values.min is not None else None
        )
        max_val = (
            type_annotation(poss_values.max) if poss_values.max is not None else None
        )

        # Overwrite default maximum values with user_settings
        if f"set_max_{k}" in q.client:
            max_val = q.client[f"set_max_{k}"]

        if isinstance(poss_values.step, (float, int)):
            step_val = type_annotation(poss_values.step)
        elif poss_values.step == "decad" and val < 1:
            step_val = 10 ** -len(str(int(1 / val)))
        else:
            step_val = 1

        if min_val is None or max_val is None:
            t = [
                # TODO: spinbox `trigger` https://github.com/h2oai/wave/pull/598
                ui.spinbox(
                    name=pre + k,
                    label=make_label(k),
                    value=val,
                    # TODO: open issue in wave to make spinbox optionally unbounded
                    max=max_val if max_val is not None else 1e12,
                    min=min_val if min_val is not None else -1e12,
                    step=step_val,
                    tooltip=tooltip,
                )
            ]
        else:
            t = [
                ui.slider(
                    name=pre + k,
                    label=make_label(k),
                    value=val,
                    min=min_val,
                    max=max_val,
                    step=step_val,
                    tooltip=tooltip,
                    trigger=trigger,
                )
            ]
    elif type_annotation == bool:
        val = q.client[pre + k] if q.client[pre + k] is not None else v

        t = [
            ui.toggle(
                name=pre + k,
                label=make_label(k),
                value=val,
                tooltip=tooltip,
                trigger=trigger,
            )
        ]
    elif type_annotation in (str, Tuple[str, ...]):
        if poss_values is None:
            val = q.client[pre + k] if q.client[pre + k] is not None else v

            title_label = make_label(k)

            t = [
                ui.textbox(
                    name=pre + k,
                    label=title_label,
                    value=val,
                    required=False,
                    password=password,
                    tooltip=tooltip,
                    trigger=trigger,
                    multiline=False,
                )
            ]
        else:
            if isinstance(poss_values, possible_values.String):
                options = list(poss_values.values)
                allow_custom = poss_values.allow_custom
                placeholder = poss_values.placeholder
            else:
                options = list(poss_values)
                allow_custom = False
                placeholder = None

            is_tuple = type_annotation == Tuple[str, ...]

            v = q.client[pre + k] if q.client[pre + k] is not None else v
            if isinstance(v, str):
                v = [v]

            # `v` might be a tuple of strings here but Wave only accepts lists
            v = list(v)

            if allow_custom:
                if not all(isinstance(option, str) for option in options):
                    raise ValueError(
                        "Combobox cannot handle (value, name) pairs for options."
                    )
                if is_tuple:
                    choices = list(set(list(options) + list(v)))
                else:
                    if isinstance(v, list):
                        for option in v:
                            if option not in options:
                                options.append(option)
                        choices = list(options)
                    else:
                        raise ValueError("Expected a list.")

                t = [
                    ui.combobox(
                        name=pre + k,
                        label=make_label(k),
                        value=None if is_tuple else v[0],
                        values=v if is_tuple else None,
                        choices=choices,
                        tooltip=tooltip,
                        placeholder=placeholder,
                        trigger=trigger,
                    )
                ]
            else:
                choices = [
                    (
                        ui.choice(option, option)
                        if isinstance(option, str)
                        else ui.choice(option[0], option[1])
                    )
                    for option in options
                ]

                t = [
                    ui.dropdown(
                        name=pre + k,
                        label=make_label(k),
                        value=None if is_tuple else v[0],
                        values=v if is_tuple else None,
                        required=False,
                        choices=choices,
                        tooltip=tooltip,
                        placeholder=placeholder,
                        trigger=trigger,
                    )
                ]

    return t


def get_dataset_elements(cfg: DefaultConfigProblemBase, q: Q) -> List:
    """For a given configuration setting return the according dataset ui components.

    Args:
        cfg: DefaultConfigProblemBase configuration settings
        q: Q

    Returns:
        List of ui elements
    """

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()

    cfg_dict = {key: cfg_dict[key] for key in cfg._get_order()}

    items = []
    for k, v in cfg_dict.items():
        # Show some fields only during dataset import
        if k.startswith("_") or cfg._get_visibility(k) == -1:
            continue

        if not (
            check_dependencies(
                cfg=cfg, pre="dataset/import", k=k, q=q, dataset_import=True
            )
        ):
            continue
        tooltip = cfg._get_tooltips(k)

        trigger = False
        if k in default_cfg.dataset_trigger_keys or k == "data_format":
            trigger = True

        if type_annotations[k] in KNOWN_TYPE_ANNOTATIONS:
            if k in default_cfg.dataset_keys:
                dataset = cfg_dict.copy()
                dataset["path"] = q.client["dataset/import/path"]

                for kk, vv in q.client["dataset/import/cfg"].__dict__.items():
                    dataset[kk] = vv

                for trigger_key in default_cfg.dataset_trigger_keys:
                    if q.client[f"dataset/import/cfg/{trigger_key}"] is not None:
                        dataset[trigger_key] = q.client[
                            f"dataset/import/cfg/{trigger_key}"
                        ]
                if (
                    q.client["dataset/import/cfg/data_format"] is not None
                    and k == "data_format"
                ):
                    v = q.client["dataset/import/cfg/data_format"]

                dataset["dataframe"] = q.client["dataset/import/cfg/dataframe"]

                type_annotation = type_annotations[k]
                poss_values, v = cfg._get_possible_values(
                    field=k,
                    value=v,
                    type_annotation=type_annotation,
                    dataset_fn=lambda k, v: (
                        dataset,
                        dataset[k] if k in dataset else v,
                    ),
                )

                if k == "train_dataframe" and v != "None":
                    q.client["dataset/import/cfg/dataframe"] = read_dataframe(v)

                q.client[f"dataset/import/cfg/{k}"] = v

                t = get_ui_element(
                    k,
                    v,
                    poss_values,
                    type_annotation,
                    tooltip=tooltip,
                    password=False,
                    trigger=trigger,
                    q=q,
                    pre="dataset/import/cfg/",
                )
            else:
                t = []
        elif dataclasses.is_dataclass(v):
            elements_group = get_dataset_elements(cfg=v, q=q)
            t = elements_group
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

        items += t

    return items


def check_dependencies(
    cfg: DefaultConfigProblemBase, pre: str, k: str, q: Q, dataset_import: bool = False
):
    """Checks all dependencies for a given key

    Args:
        cfg: DefaultConfigProblemBase configuration settings
        pre: prefix for client keys
        k: key to be checked
        q: Q
        dataset_import: flag whether dependencies are checked in dataset import

    Returns:
        True if dependencies are met
    """

    dependencies = cfg._get_nesting_dependencies(k)

    if dependencies is None:
        dependencies = []
    # Do not respect some nesting during the dataset import
    if dataset_import:
        dependencies = [x for x in dependencies if x.key not in ["validation_strategy"]]
    # Do not respect some nesting during the create experiment
    else:
        dependencies = [x for x in dependencies if x.key not in ["data_format"]]

    if len(dependencies) > 0:
        all_deps = 0
        for d in dependencies:
            if (
                not dataset_import
                and q.client["experiment/start/grid_search"]
                and cfg._get_grid_search_values(d.key) is not None
            ):
                dependency_values = q.client[f"{pre}/cfg/{d.key}_grid_search"]
            else:
                if isinstance(q.client[f"{pre}/cfg/{d.key}"], (list, tuple)):
                    dependency_values = q.client[f"{pre}/cfg/{d.key}"]
                else:
                    dependency_values = [q.client[f"{pre}/cfg/{d.key}"]]

            all_deps += d.check(dependency_values)
        return all_deps == len(dependencies)

    return True


def is_visible(k: str, cfg: DefaultConfigProblemBase, q: Q) -> bool:
    """Returns a flag whether a given key should be visible on UI.

    Args:
        k: name of the hyperparameter
        cfg: configuration settings,
        q: Q
    Returns:
        List of ui elements
    """

    visibility = 1

    if visibility < cfg._get_visibility(k):
        return False

    return True


def get_grid_value(v: Any, type_annotation: Any) -> List[str]:
    """Handles .0 for floats in the grid search

    Args:
        v: value of the hyperparameter
        type_annotation: type of the parameter
    Returns:
        List with a grid search element
    """

    v: str = str(v)

    if type_annotation == float and v.endswith(".0"):
        v = v[:-2]

    return [v]


def get_ui_elements(
    cfg: DefaultConfigProblemBase,
    q: Q,
    limit: Optional[List[str]] = None,
    pre: str = "experiment/start",
) -> List:
    """For a given configuration setting return the according ui components.

    Args:
        cfg: configuration settings
        q: Q
        limit: optional list of keys to limit
        pre: prefix for client keys
        parent_cfg: parent config class.

    Returns:
        List of ui elements
    """
    items = []

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()

    cfg_dict = {key: cfg_dict[key] for key in cfg._get_order()}

    for k, v in cfg_dict.items():
        if ("api" in k) or ("secret" in k):
            password = True
        else:
            password = False

        if k.startswith("_") or cfg._get_visibility(k) < 0:
            if q.client[f"{pre}/cfg_mode/from_cfg"]:
                q.client[f"{pre}/cfg/{k}"] = v
            continue
        else:
            type_annotation = type_annotations[k]
            poss_values, v = cfg._get_possible_values(
                field=k,
                value=v,
                type_annotation=type_annotation,
                dataset_fn=partial(get_dataset, q=q, limit=limit, pre=pre),
            )

            if k in default_cfg.dataset_keys:
                # reading dataframe
                if k == "train_dataframe" and (v != ""):
                    q.client[f"{pre}/cfg/dataframe"] = read_dataframe(v, meta_only=True)
                q.client[f"{pre}/cfg/{k}"] = v
            elif k in default_cfg.dataset_extra_keys:
                _, v = get_dataset(k, v, q=q, limit=limit, pre=pre)
                q.client[f"{pre}/cfg/{k}"] = v
            elif q.client[f"{pre}/cfg_mode/from_cfg"]:
                q.client[f"{pre}/cfg/{k}"] = v
                q.client[f"{pre}/cfg/{k}_grid_search"] = get_grid_value(
                    v, type_annotation
                )
        # Overwrite current default values with user_settings
        if q.client[f"{pre}/cfg_mode/from_default"] and f"default_{k}" in q.client:
            q.client[f"{pre}/cfg/{k}"] = q.client[f"default_{k}"]

        if not (check_dependencies(cfg=cfg, pre=pre, k=k, q=q)):
            continue

        if not is_visible(k=k, cfg=cfg, q=q):
            if type_annotation not in KNOWN_TYPE_ANNOTATIONS:
                _ = get_ui_elements(cfg=v, q=q, limit=limit, pre=pre)
            elif q.client[f"{pre}/cfg_mode/from_cfg"]:
                q.client[f"{pre}/cfg/{k}"] = v

            continue

        tooltip = cfg._get_tooltips(k)

        trigger = False
        q.client[f"{pre}/trigger_ks"] = ["train_dataframe"]
        q.client[f"{pre}/trigger_ks"] += cfg._get_nesting_triggers()
        if k in q.client[f"{pre}/trigger_ks"]:
            trigger = True

        if (
            pre == "experiment/start"
            and q.client[f"{pre}/grid_search"]
            and cfg._get_grid_search_values(k)
        ):

            grid_name = f"{pre}/cfg/{k}_grid_search"
            add_choice = []

            # Value is not in the grid range, add to choices
            if v not in cfg._get_grid_search_values(k):
                add_choice = get_grid_value(v, type_annotation)

            v = get_grid_value(v, type_annotation)
            v = q.client[grid_name] if q.client[grid_name] is not None else v

            t = [
                ui.message_bar(
                    type="info",
                    text=f"**{make_label(k)}** is a grid search hyperparameter.",
                )
            ]

            # len(add_choice) == 1 added due to a strange bug, that iscustom will get
            # overwritten to None in special cases
            allow_custom = cfg._get_grid_search_iscustom(k) or len(add_choice) == 1

            if allow_custom:
                cust_choices: list[str] = [
                    str(c) for c in cfg._get_grid_search_values(k)
                ] + add_choice
                t += [
                    ui.combobox(
                        name=grid_name,
                        label=make_label(k) + " (grid search)",
                        values=v,
                        required=False,
                        choices=cust_choices,
                        tooltip=tooltip,
                        trigger=trigger,
                    )
                ]
            else:
                choices: list[Choice] = [
                    ui.choice(str(c), str(c)) for c in cfg._get_grid_search_values(k)
                ]

                t += [
                    ui.dropdown(
                        name=grid_name,
                        label=make_label(k) + " (grid search)",
                        values=v,
                        required=False,
                        choices=choices,
                        tooltip=tooltip,
                        trigger=trigger,
                    )
                ]
        elif type_annotation in KNOWN_TYPE_ANNOTATIONS:
            if limit is not None and k not in limit:
                continue

            t = get_ui_element(
                k=k,
                v=v,
                poss_values=poss_values,
                type_annotation=type_annotation,
                tooltip=tooltip,
                password=password,
                trigger=trigger,
                q=q,
                pre=f"{pre}/cfg/",
            )
        elif dataclasses.is_dataclass(v):
            if limit is not None and k in limit:
                elements_group = get_ui_elements(cfg=v, q=q, limit=None, pre=pre)
            else:
                elements_group = get_ui_elements(cfg=v, q=q, limit=limit, pre=pre)

            if k == "dataset" and pre != "experiment/start":
                # get all the datasets available
                df_datasets = q.client.app_db.get_datasets_df()
                if not q.client[f"{pre}/dataset"]:
                    if len(df_datasets) >= 1:
                        q.client[f"{pre}/dataset"] = str(df_datasets["id"].iloc[-1])
                    else:
                        q.client[f"{pre}/dataset"] = "1"

                elements_group = [
                    ui.dropdown(
                        name=f"{pre}/dataset",
                        label="Dataset",
                        required=True,
                        value=q.client[f"{pre}/dataset"],
                        choices=[
                            ui.choice(str(row["id"]), str(row["name"]))
                            for _, row in df_datasets.iterrows()
                        ],
                        trigger=True,
                        tooltip=tooltip,
                    )
                ] + elements_group

            if len(elements_group) > 0:
                t = [
                    ui.separator(
                        name=k + "_expander", label=make_label(k, appendix=" settings")
                    )
                ]
            else:
                t = []

            t += elements_group
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

        items += t

    q.client[f"{pre}/prev_dataset"] = q.client[f"{pre}/dataset"]

    return items


def parse_ui_elements(
    cfg: DefaultConfigProblemBase, q: Q, limit: List | str = "", pre: str = ""
) -> Any:
    """Sets configuration settings with arguments from app

    Args:
        cfg: configuration
        q: Q
        limit: optional list of keys to limit
        pre: prefix for keys

    Returns:
        Configuration with settings overwritten from arguments
    """

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()
    for k, v in cfg_dict.items():
        if k.startswith("_") or cfg._get_visibility(k) == -1:
            continue

        if (
            len(limit) > 0
            and k not in limit
            and type_annotations[k] in KNOWN_TYPE_ANNOTATIONS
        ):
            continue

        if (
            pre == "experiment/start/cfg/"
            and q.client["experiment/start/grid_search"]
            and cfg._get_grid_search_values(k)
        ):
            value = q.client[f"{pre}{k}_grid_search"]
            setattr(cfg, k, value)
        elif type_annotations[k] in KNOWN_TYPE_ANNOTATIONS:
            value = q.client[f"{pre}{k}"]

            if type_annotations[k] == Tuple[str, ...]:
                if isinstance(value, str):
                    value = [value]
                elif value is None:
                    value = ()
                else:
                    value = tuple(value)
            if isinstance(type_annotations[k], str) and isinstance(value, list):
                # fix for combobox outputting custom values as list in wave 0.22
                value = value[0]
            setattr(cfg, k, value)
        elif dataclasses.is_dataclass(v):
            setattr(cfg, k, parse_ui_elements(cfg=v, q=q, limit=limit, pre=pre))
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

    return cfg


def get_experiment_status(path: str) -> Tuple[str, str]:
    """Get status information from experiment.

    Args:
        path: path to experiment folder
    Returns:
        Tuple of experiment status and experiment info
    """

    try:
        flag_json_path = f"{path}/flags.json"
        if not os.path.exists(flag_json_path):
            logger.debug(f"File {flag_json_path} does not exist yet.")
            return "none", "none"
        with open(flag_json_path) as file:
            flags = json.load(file)
            status = flags.get("status", "none")
            info = flags.get("info", "none")

        # Collect failed statuses from all GPUs
        single_gpu_failures = []
        for flag_json_path in glob.glob(f"{path}/flags?*.json"):
            if os.path.exists(flag_json_path):
                with open(flag_json_path) as file:
                    flags = json.load(file)
                    status = flags.get("status", "none")
                    info = flags.get("info", "none")

                    if status == "failed":
                        single_gpu_failures.append(info)
        # Get the most detailed failure info
        if len(single_gpu_failures) > 0:
            detailed_gpu_failures = [x for x in single_gpu_failures if x != "See logs"]
            if len(detailed_gpu_failures) > 0:
                return "failed", detailed_gpu_failures[0]
            else:
                return "failed", single_gpu_failures[0]
        return status, info

    except Exception:
        logger.debug("Could not get experiment status:", exc_info=True)
        return "none", "none"


def get_experiments_status(df: DataFrame) -> Tuple[List[str], List[str]]:
    """For each experiment in given dataframe, return the status of the process

    Args:
        df: experiment dataframe

    Returns:
        A list with each status and a list with all infos
    """

    status_all = []
    info_all = []
    for idx, row in df.iterrows():
        status, info = get_experiment_status(row.path)

        if info == "none":
            info = ""
        info_all.append(info)

        pid = row.process_id

        zombie = False
        try:
            p = psutil.Process(pid)
            zombie = p.status() == "zombie"
        except psutil.NoSuchProcess:
            pass
        if not psutil.pid_exists(pid) or zombie:
            running = False
        else:
            running = True

        if running:
            if status == "none":
                status_all.append("queued")
            elif status == "running":
                status_all.append("running")
            elif status == "queued":
                status_all.append("queued")
            elif status == "finished":
                status_all.append("finished")
            elif status == "stopped":
                status_all.append("stopped")
            elif status == "failed":
                status_all.append("failed")
            else:
                status_all.append("finished")
        else:
            if status == "none":
                status_all.append("failed")
            elif status == "queued":
                status_all.append("failed")
            elif status == "running":
                status_all.append("failed")
            elif status == "finished":
                status_all.append("finished")
            elif status == "stopped":
                status_all.append("stopped")
            elif status == "failed":
                status_all.append("failed")
            else:
                status_all.append("failed")

    return status_all, info_all


def get_experiments_info(df: DataFrame, q: Q) -> DefaultDict:
    """For each experiment in given dataframe, return certain configuration settings

    Args:
        df: experiment dataframe
        q: Q

    Returns:
        A dictionary of lists of additional information
    """

    info = defaultdict(list)
    for _, row in df.iterrows():
        try:
            # load_config_yaml issues a warning if the yaml file contains keys
            # that are no longer part of the dataclass fields.
            # This can happen if the codebase has changed since the experiment was run.
            # Ignore those warnings here

            original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)
            try:
                cfg = load_config_yaml(f"{row.path}/cfg.yaml").__dict__
            finally:
                logging.getLogger().setLevel(original_level)

        except Exception:
            cfg = None

        metric = ""
        loss_function = ""

        if cfg is not None:
            try:
                metric = cfg["prediction"].metric
                loss_function = cfg["training"].loss_function
            except KeyError:
                metric = ""
                loss_function = ""

        charts_db_path = os.path.join(row.path, "charts.db")
        if os.path.exists(charts_db_path):
            with SqliteDict(charts_db_path) as logs:
                if "internal" in logs.keys():
                    if "current_step" in logs["internal"].keys():
                        curr_step = int(logs["internal"]["current_step"]["values"][-1])
                    else:
                        curr_step = 0

                    if "total_training_steps" in logs["internal"].keys():
                        total_training_steps = int(
                            logs["internal"]["total_training_steps"]["values"][-1]
                        )
                    else:
                        total_training_steps = 0

                    if "current_val_step" in logs["internal"].keys():
                        curr_val_step = int(
                            logs["internal"]["current_val_step"]["values"][-1]
                        )
                    else:
                        curr_val_step = 0

                    if "total_validation_steps" in logs["internal"].keys():
                        total_validation_steps = int(
                            logs["internal"]["total_validation_steps"]["values"][-1]
                        )
                    else:
                        total_validation_steps = 0

                    curr_total_step = curr_step + curr_val_step

                    total_steps = max(total_training_steps + total_validation_steps, 1)

                    if (
                        "global_start_time" in logs["internal"].keys()
                        and curr_total_step > 0
                    ):
                        elapsed = (
                            time.time()
                            - logs["internal"]["global_start_time"]["values"][-1]
                        )
                        remaining_steps = total_steps - curr_total_step
                        eta = elapsed * (remaining_steps / curr_total_step)
                        if eta == 0:
                            eta = ""
                        else:
                            # if more than one day, show days
                            # need to subtract 1 day from time_took since strftime shows
                            # day of year which starts counting at 1
                            if eta > 86400:
                                eta = time.strftime(
                                    "%-jd %H:%M:%S", time.gmtime(float(eta - 86400))
                                )
                            else:
                                eta = time.strftime("%H:%M:%S", time.gmtime(float(eta)))
                    else:
                        eta = "N/A"
                else:
                    # Default values for when charts.db is not available
                    # (experiment deleted manually)
                    eta = "N/A"
                    total_steps = 1
                    curr_total_step = 0

                if (
                    "validation" in logs
                    and metric in logs["validation"]
                    and logs["validation"][metric]["values"][-1] is not None
                ):
                    score_val = np.round(logs["validation"][metric]["values"][-1], 4)
                else:
                    score_val = ""

        else:
            logs = {}
            eta = "N/A"
            total_steps = 1
            curr_total_step = 0
            score_val = ""
            logger.info(f"Experiment path {charts_db_path} not found.")

        try:
            dataset = q.client.app_db.get_dataset(row.dataset).name
        except Exception:
            dataset = ""

        config_file = make_config_label(row.config_file)

        info["config_file"].append(config_file)
        info["dataset"].append(dataset)
        info["loss"].append(loss_function)
        info["metric"].append(metric)
        info["eta"].append(eta)
        info["val metric"].append(score_val)
        info["progress"].append(f"{np.round(curr_total_step / total_steps, 2)}")

        del cfg

    return info


def make_config_label(config_file: str) -> str:
    """Makes a label from a config file name

    Args:
        config_file: config file name

    Returns:
        Label
    """

    config_file = config_file.replace(".yaml", "")
    if "_config_" in config_file:
        config_file_split = config_file.split("_config_")
        config_file = (
            f"{make_label(config_file_split[0])} "
            f"({make_label(config_file_split[1][1:])})"
        )
    else:
        config_file = make_label(config_file.replace("_config", ""))

    return config_file


def get_datasets_info(df: DataFrame, q: Q) -> Tuple[DataFrame, DefaultDict]:
    """For each dataset in given dataframe, return certain configuration settings

    Args:
        df: dataset dataframe
        q: Q

    Returns:
        A dictionary of lists of additional information
    """

    info = defaultdict(list)
    for idx, row in df.iterrows():
        config_file = q.client.app_db.get_dataset(row.id).config_file
        path = row.path + "/"

        try:
            logging_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)
            cfg = load_config_yaml(config_file)
            logging.getLogger().setLevel(logging_level)
        except Exception as e:
            logger.warning(f"Could not load configuration from {config_file}. {e}")
            cfg = None

        if cfg is not None:
            cfg_dataset = cfg.dataset.__dict__

            config_file = make_config_label(row.config_file.replace(path, ""))

            info["problem type"].append(config_file)
            info["train dataframe"].append(
                cfg_dataset["train_dataframe"].replace(path, "")
            )
            info["validation dataframe"].append(
                cfg_dataset["validation_dataframe"].replace(path, "")
            )

            info["labels"].append(cfg.dataset.answer_column)

            del cfg, cfg_dataset
        else:
            df = df.drop(idx)

    return df, info


def get_experiments(
    q: Q,
    status: Union[Optional[str], Optional[List[str]]] = None,
) -> pd.DataFrame:
    """Return all experiments given certain restrictions

    Args:
        q: Q
        status: option to filter for certain experiment status
    Returns:
        experiment df
    """

    df = q.client.app_db.get_experiments_df()

    info = get_experiments_info(df, q)
    for k, v in info.items():
        df[k] = v

    df["status"], df["info"] = get_experiments_status(df)

    if status is not None:
        if type(status) is str:
            status = [status]
        df = df[df["status"].isin(status)]

    if len(df) > 0:
        # make sure progress is 100% for finished experiments
        df.loc[df.status == "finished", "progress"] = "1.0"

        df["info"] = np.where(
            (df["status"] == "running") & (df["eta"] != ""),
            df["eta"].apply(lambda x: f"ETA: {x}"),
            df["info"],
        )

    return df


def get_datasets(
    q: Q,
    show_experiment_datasets: bool = True,
) -> pd.DataFrame:
    """Return all datasets given certain restrictions

    Args:
        q: Q
        show_experiment_datasets: whether to also show datasets linked to experiments

    Returns:
        dataset df
    """

    df = q.client.app_db.get_datasets_df()

    df, info = get_datasets_info(df, q)
    for k, v in info.items():
        df[k] = v

    for type in ["train", "validation"]:
        col_name = f"{type}_rows"
        if col_name not in df:
            continue
        rows = df[col_name].astype(float).map("{:.0f}".format)
        del df[col_name]
        rows[rows == "nan"] = "None"

        if f"{type} dataframe" in df.columns:
            idx = df.columns.get_loc(f"{type} dataframe") + 1
            df.insert(idx, f"{type} rows", rows)

    if not show_experiment_datasets:
        experiment_datasets = get_experiments(q).dataset.unique()
        df = df.loc[~df["name"].isin(experiment_datasets)]

    return df


def filter_grid_search_combination(grid: Dict[str, Any], cfg: Any) -> Dict[str, Any]:
    """Filters grid search combination in order not to start multiple same experiments.

    Args:
        grid: grid combination from the full grid search
        cfg: configuration settings

    Returns:
        Filtered grid combination after checking the dependencies
    """

    grid = grid.copy()
    cfg_dict = cfg.__dict__

    for k, v in cfg_dict.items():
        if dataclasses.is_dataclass(v):
            grid = filter_grid_search_combination(grid=grid, cfg=v)

        if k in grid:
            dependencies = cfg._get_nesting_dependencies(k)
            if dependencies is None:
                continue

            if all(
                [d.key in grid and not d.check([grid[d.key]]) for d in dependencies]
            ):
                grid.pop(k)

    return grid


def get_grid_search(cfg: Any, q: Q, pre: str) -> Dict[str, List]:
    """Creates a dictionary with grid search values.

    Args:
        cfg: configuration settings
        q: Q
        pre: prefix for client keys
    """

    grid_search = {}
    cfg_dict = cfg.__dict__

    type_annotations = cfg.get_annotations()
    for k, v in cfg_dict.items():
        if k.startswith("_") or cfg._get_visibility(k) < 0:
            continue
        elif (
            pre == "experiment/start"
            and q.client["experiment/start/grid_search"]
            and cfg._get_grid_search_values(k)
        ):
            if type_annotations[k] == bool:
                grid_search[k] = [True if x == "True" else False for x in v]
            else:
                try:
                    grid_search[k] = [type_annotations[k](x) for x in v]
                except ValueError:
                    raise GridCheckError(f"{make_label(k)}")
        elif type_annotations[k] in KNOWN_TYPE_ANNOTATIONS:
            pass
        elif dataclasses.is_dataclass(v):
            grid_search.update(get_grid_search(cfg=v, q=q, pre=pre))
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

    return grid_search


def set_grid_to_cfg(cfg: Any, grid: Dict[str, List]) -> Any:
    """Sets individual run config for the Grid Search.

    Args:
        cfg: configuration settings
        grid: dictionary of a single grid search element
    Returns:
        config for the corresponding grid search run
    """

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()

    for k, v in cfg_dict.items():
        if k.startswith("_") or cfg._get_visibility(k) < 0:
            continue
        elif k in grid and cfg._get_grid_search_values(k):
            setattr(cfg, k, grid[k])
        elif type_annotations[k] in KNOWN_TYPE_ANNOTATIONS:
            pass
        elif dataclasses.is_dataclass(v):
            setattr(cfg, k, set_grid_to_cfg(cfg=v, grid=grid))
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

    return cfg


def start_experiment(
    cfg: DefaultConfigProblemBase, q: Q, pre: str, gpu_list: Optional[List] = None
) -> None:
    """Starts an experiment

    Args:
        cfg: DefaultConfigProblemBase configuration settings
        q: Q
        pre: prefix for client keys
        gpu_list: list of GPUs available
    """
    if gpu_list is None:
        gpu_list = cfg.environment.gpus

    # Get queue of the processes to wait for
    running_experiments = get_experiments(q=q)
    running_experiments = running_experiments[
        running_experiments.status.isin(["queued", "running"])
    ]
    all_process_queue = []
    for _, row in running_experiments.iterrows():
        for gpu_id in row["gpu_list"].split(","):
            if gpu_id in gpu_list:
                all_process_queue.append(row["process_id"])

    process_queue = list(set(all_process_queue))

    env_vars = {
        "NEPTUNE_API_TOKEN": q.client["default_neptune_api_token"],
        "WANDB_API_KEY": q.client["default_wandb_api_token"],
        "OPENAI_API_KEY": q.client["default_openai_api_token"],
        "GPT_EVAL_MAX": str(q.client["default_gpt_eval_max"]),
        "HF_HUB_ENABLE_HF_TRANSFER": str(q.client["default_hf_hub_enable_hf_transfer"]),
    }
    if q.client["default_openai_azure"]:
        env_vars.update(
            {
                "OPENAI_API_TYPE": "azure",
                "OPENAI_API_BASE": q.client["default_openai_api_base"],
                "OPENAI_API_VERSION": q.client["default_openai_api_version"],
                "OPENAI_API_DEPLOYMENT_ID": q.client[
                    "default_openai_api_deployment_id"
                ],
            }
        )
    if q.client["default_huggingface_api_token"]:
        env_vars.update(
            {"HUGGINGFACE_TOKEN": q.client["default_huggingface_api_token"]}
        )

    env_vars = {k: v or "" for k, v in env_vars.items()}

    cfg = copy_config(cfg, q)
    cfg.output_directory = f"{get_output_dir(q)}/{cfg.experiment_name}/"
    os.makedirs(cfg.output_directory)
    save_config_yaml(f"{cfg.output_directory}/cfg.yaml", cfg)

    # Start the training process
    p = start_process(
        cfg=cfg, gpu_list=gpu_list, process_queue=process_queue, env_vars=env_vars
    )

    logger.info(f"Process: {p.pid}, Queue: {process_queue}, GPUs: {gpu_list}")

    experiment = Experiment(
        name=cfg.experiment_name,
        mode="train",
        dataset=q.client[f"{pre}/dataset"],
        config_file=q.client[f"{pre}/cfg_file"],
        path=cfg.output_directory,
        seed=cfg.environment.seed,
        process_id=p.pid,
        gpu_list=",".join(gpu_list),
    )

    q.client.app_db.add_experiment(experiment)


def get_frame_stats(frame):
    non_numeric_cols = frame.select_dtypes(object).columns
    is_str_cols = [
        x
        for x in non_numeric_cols
        if frame[x].dropna().size and (frame[x].dropna().apply(type) == str).all()
    ]
    cols_to_drop = [x for x in non_numeric_cols if x not in is_str_cols]

    if len(cols_to_drop):  # drop array/list/non-str object columns
        frame = frame.drop(columns=cols_to_drop)
        non_numeric_cols = frame.select_dtypes(object).columns

    if len(frame.columns) == 0:
        return None

    numeric_cols = [col for col in frame if col not in non_numeric_cols]

    if len(non_numeric_cols) == 0 or len(numeric_cols) == 0:
        stats = frame.describe()
        if len(numeric_cols):
            stats = stats.round(decimals=3)
            stats.loc["unique"] = frame.nunique()  # unique is part of describe for str

    else:
        stats1 = frame[non_numeric_cols].describe()
        stats2 = frame[numeric_cols].describe().round(decimals=3)

        stats2.loc["unique"] = frame[numeric_cols].nunique()
        stats = (
            stats1.reset_index()
            .merge(stats2.reset_index(), how="outer", on="index")
            .fillna("")
        ).set_index("index")

    stats = stats.T.reset_index().rename(columns={"index": "column"})

    for col in ["count", "unique"]:
        if col in stats:
            stats[col] = stats[col].astype(int)

    return stats


def dir_file_table(current_path: str) -> pd.DataFrame:
    results = [".."]
    try:
        if os.path.isdir(current_path):
            files = os.listdir(current_path)
            files = sorted([f for f in files if not f.startswith(".")], key=str.lower)
            results.extend(files)
    except Exception:
        logger.error(f"Error while listing folder '{current_path}':", exc_info=True)

    return pd.DataFrame({current_path: results})


def get_download_link(q: Q, artifact_path):
    new_path = os.path.relpath(artifact_path, get_output_dir(q))
    new_path = os.path.join(get_download_dir(q), new_path)
    url_path = os.path.relpath(new_path, get_output_dir(q))

    if not os.path.exists(new_path):
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        os.symlink(os.path.abspath(artifact_path), os.path.abspath(new_path))

    # return a relative path so that downloads work when the instance is
    # behind a reverse proxy or being accessed by a public IP in a public
    # cloud.

    return url_path


def check_valid_upload_content(upload_path: str) -> Tuple[bool, str]:
    if upload_path.endswith("zip"):
        valid = zipfile.is_zipfile(upload_path)
        error = "" if valid else "File is not a zip file"
    else:
        valid = is_valid_data_frame(upload_path)
        error = "" if valid else "File does not have valid format"

    if not valid:
        os.remove(upload_path)

    return valid, error


def flatten_dict(d: collections.abc.MutableMapping) -> dict:
    """
    Adapted from https://stackoverflow.com/a/6027615
    Does not work with nesting and mutiple keys with the same name!

    Args:
        d: dict style object
    Return:
        A flattened dict
    """

    items: List[Tuple[Any, Any]] = []
    for k, v in d.items():
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v).items())
        else:
            items.append((k, v))
    return dict(items)


def get_unique_name(expected_name, existing_names, is_invalid_function=None):
    """
    Return a new name that does not exist in list of existing names

    Args:
        expected_name: preferred name
        existing_names: list of existing names
        is_invalid_function: optional callable, to determine if the new name is
            invalid
    Return:
        new name
    """

    new_name = expected_name
    cnt = 1

    while new_name in existing_names or (
        is_invalid_function is not None and is_invalid_function(new_name)
    ):
        new_name = f"{expected_name}.{cnt}"
        cnt += 1

    return new_name


def get_unique_dataset_name(q, dataset_name, include_all_folders=True):
    """
    Return a dataset name that does not exist yet

    Args:
        q: Q
        dataset_name: preferred dataset name
        include_all_folders: whether to also consider all (temp) dataset folders
    Return:
        new dataset_name
    """
    datasets_df = q.client.app_db.get_datasets_df()

    existing_names = datasets_df["name"].values.tolist()
    if include_all_folders:
        existing_names.extend(os.listdir(get_data_dir(q)))

    return get_unique_name(dataset_name, existing_names)


def get_valid_temp_data_folder(q: Q, folder_path: str) -> str:
    """
    Return new temporary data folder path not associated with any existing dataset

    Args:
        q: Q
        folder_path: original folder_path
    Return:
        new folder path not associated with any existing dataset
    """
    dirname = os.path.dirname(folder_path)
    basename = os.path.basename(folder_path)
    unique_name = get_unique_dataset_name(q, basename, include_all_folders=False)
    return os.path.join(dirname, unique_name)


def remove_temp_files(q: Q):
    """
    Remove any temp folders leftover from dataset import
    """

    datasets_df = q.client.app_db.get_datasets_df()
    all_files = glob.glob(os.path.join(get_data_dir(q), "*"))
    for file in all_files:
        if not any([path in file for path in datasets_df["path"].values]):
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)


def get_gpu_usage() -> float:
    usage: float = 0.0
    all_gpus: List[GPUtil.GPU] = GPUtil.getGPUs()
    for gpu in all_gpus:
        usage += float(gpu.load)

    usage /= len(all_gpus)
    return usage * 100.0


def get_single_gpu_usage(sig_figs: int = 1, highlight: Optional[str] = None):
    all_gpus = GPUtil.getGPUs()
    items = []
    for i, gpu in enumerate(all_gpus):
        gpu_load = f"{round(gpu.load * 100, sig_figs)}%"
        memory_used = get_size_str(
            gpu.memoryUsed, sig_figs=1, input_unit="MB", output_unit="GB"
        )
        memory_total = get_size_str(
            gpu.memoryTotal, sig_figs=1, input_unit="MB", output_unit="GB"
        )

        if highlight is not None:
            gpu_load = f"**<span style='color:{highlight}'>{gpu_load}</span>**"
            memory_used = f"**<span style='color:{highlight}'>{memory_used}</span>**"
            memory_total = f"**<span style='color:{highlight}'>{memory_total}</span>**"

        items.append(
            ui.text(
                f"GPU #{i + 1} - current utilization: {gpu_load} - "
                f"VRAM usage: {memory_used} / {memory_total} - {gpu.name}"
            )
        )
    return items


def copy_config(cfg: DefaultConfigProblemBase, q: Q) -> Any:
    """Makes a copy of the config

    Args:
        cfg: DefaultConfigProblemBase config object
    Returns:
        copy of the config
    """
    # make unique yaml file using uuid
    os.makedirs(get_output_dir(q), exist_ok=True)
    tmp_file = os.path.join(f"{get_output_dir(q)}/", str(uuid.uuid4()) + ".yaml")
    save_config_yaml(tmp_file, cfg)
    cfg = load_config_yaml(tmp_file)
    os.remove(tmp_file)
    return cfg


def make_label(title: str, appendix: str = "") -> str:
    """Cleans a label

    Args:
        title: title to clean
        appendix: optional appendix

    Returns:
        Cleaned label

    """
    label = " ".join(w.capitalize() for w in title.split("_")) + appendix
    label = label.replace("Llm", "LLM")
    return label


def get_cfg_list_items(cfg: DefaultConfigProblemBase) -> List:
    items = parse_cfg_dataclass(cfg)
    x = []
    for item in items:
        for k, v in item.items():
            x.append(ui.stat_list_item(label=make_label(k), value=str(v)))
    return x


# https://stackoverflow.com/questions/2059482/temporarily-modify-the-current-processs-environment
@contextlib.contextmanager
def set_env(**environ):
    """
    Temporarily set the process environment variables.

    >>> with set_env(PLUGINS_DIR='test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False

    :type environ: dict[str, unicode]
    :param environ: Environment variables to set
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def hf_repo_friendly_name(name: str) -> str:
    """
    Converts the given string into a huggingface-repository-friendly name.

    • Repo id must use alphanumeric chars or '-', '_', and '.' allowed.
    • '--' and '..' are forbidden
    • '-' and '.' cannot start or end the name
    • max length is 96
    """
    name = re.sub("[^0-9a-zA-Z]+", "-", name)
    name = name[1:] if name.startswith("-") else name
    name = name[:-1] if name.endswith("-") else name
    name = name[:96]
    return name


def save_hf_yaml(
    path: str, account_name: str, model_name: str, repo_id: Optional[str] = None
):
    with open(path, "w") as fp:
        yaml.dump(
            {
                "account_name": account_name,
                "model_name": model_name,
                "repo_id": repo_id if repo_id else f"{account_name}/{model_name}",
            },
            fp,
            indent=4,
        )
