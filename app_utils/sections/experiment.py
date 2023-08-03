import glob
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Set

import accelerate
import einops
import huggingface_hub
import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from h2o_wave import Q, data, ui
from jinja2 import Environment, FileSystemLoader
from sqlitedict import SqliteDict

from app_utils.config import default_cfg
from app_utils.sections.chat import chat_tab, load_cfg_model_tokenizer
from app_utils.sections.common import clean_dashboard
from app_utils.utils import (
    add_model_type,
    flatten_dict,
    get_cfg_list_items,
    get_data_dir,
    get_download_link,
    get_experiment_status,
    get_experiments,
    get_model_types,
    get_problem_categories,
    get_problem_types,
    get_ui_elements,
    get_unique_name,
    hf_repo_friendly_name,
    parse_ui_elements,
    remove_model_type,
    set_env,
    start_experiment,
)
from app_utils.wave_utils import busy_dialog, ui_table_from_df, wave_theme
from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.tooltips import tooltips
from llm_studio.src.utils.config_utils import (
    load_config_py,
    load_config_yaml,
    save_config_yaml,
)
from llm_studio.src.utils.exceptions import LLMResourceException
from llm_studio.src.utils.export_utils import (
    check_available_space,
    get_artifact_path_path,
    get_logs_path,
    get_model_path,
    get_predictions_path,
    get_size_str,
    save_logs,
    save_prediction_outputs,
)
from llm_studio.src.utils.logging_utils import write_flag
from llm_studio.src.utils.modeling_utils import check_disk_space, unwrap_model
from llm_studio.src.utils.plot_utils import PLOT_ENCODINGS
from llm_studio.src.utils.utils import add_file_to_zip, kill_child_processes

logger = logging.getLogger(__name__)


async def experiment_start(q: Q) -> None:
    """Display experiment start cards."""

    await clean_dashboard(q, mode="experiment_start", exclude=["experiment/start"])
    q.client["nav/active"] = "experiment/start"

    show_update_warnings = True
    is_create_experiment = False
    # reset certain configs if new experiment start session
    if (
        q.args["experiment/start"]
        or q.args["experiment/start_experiment"]
        or q.args["dataset/newexperiment"]
        or q.args["dataset/newexperiment/from_current"]
        or q.args["experiment/list/new"]
    ):
        q.client["experiment/start/cfg_experiment_prev"] = None
        q.client["experiment/start/cfg_file_prev"] = None
        q.client["experiment/start/prev_dataset"] = None
        q.client["experiment/start/cfg_sub"] = None
        show_update_warnings = False
        is_create_experiment = True

    # get all the datasets available
    df_datasets = q.client.app_db.get_datasets_df()
    # Hide inference only datasets
    df_datasets = df_datasets.loc[df_datasets["train_rows"].notna()]
    if (
        not q.client["experiment/start/dataset"]
        or q.client["experiment/start/dataset"] not in df_datasets.id.astype(str).values
    ):
        if len(df_datasets) >= 1:
            q.client["experiment/start/dataset"] = str(df_datasets["id"].iloc[-1])
        else:
            q.client["experiment/start/dataset"] = "1"

    warning_message = "Experiment settings might be updated after changing {}"

    items = [
        ui.separator(name="general_expander", label="General settings"),
        ui.dropdown(
            name="experiment/start/dataset",
            label="Dataset",
            required=True,
            value=q.client["experiment/start/dataset"],
            choices=[
                ui.choice(str(row["id"]), str(row["name"]))
                for _, row in df_datasets.iterrows()
            ],
            trigger=True,
            tooltip=tooltips["experiments_dataset"],
        ),
    ]

    if (
        show_update_warnings
        and q.client["experiment/start/dataset_prev"]
        != q.client["experiment/start/dataset"]
    ):
        items += [
            ui.message_bar(type="warning", text=warning_message.format("Dataset"))
        ]
        show_update_warnings = False

    if (
        q.client["experiment/start/cfg_file"] is None
        or q.client["experiment/start/dataset_prev"]
        != q.client["experiment/start/dataset"]
    ) and q.client["experiment/start/cfg_category"] != "experiment":
        dataset = q.client.app_db.get_dataset(q.client["experiment/start/dataset"])
        if dataset is not None:
            problem_type = dataset.config_file.replace(dataset.path + "/", "").replace(
                ".yaml", ""
            )
        else:
            problem_type = default_cfg.cfg_file
        q.client["experiment/start/cfg_file"] = problem_type
        q.client["experiment/start/cfg_category"] = problem_type.split("_")[0]

    if q.client["experiment/start/cfg_category"] == "experiment":
        q.client["experiment/start/cfg_file"] = "experiment"

    # get all experiments
    df_experiments = get_experiments(q, mode="train")

    # get all problem category choices
    choices_problem_categories = [
        ui.choice(name, label) for name, label in get_problem_categories()
    ]

    if len(df_experiments["id"]) > 0:
        choices_problem_categories += [ui.choice("experiment", "From Experiment")]

    # set default value of problem type if no match to category
    if (
        q.client["experiment/start/cfg_category"]
        not in q.client["experiment/start/cfg_file"]
    ):
        if q.client["experiment/start/cfg_category"] != "experiment":
            q.client["experiment/start/cfg_file"] = get_problem_types(
                category=q.client["experiment/start/cfg_category"]
            )[0][0]

    # get all problem type choices
    choices_problem_types = [
        ui.choice(name, label)
        for name, label in get_problem_types(
            category=q.client["experiment/start/cfg_category"]
        )
    ]

    # remove model type if present in cfg file name here
    q.client["experiment/start/cfg_file"] = remove_model_type(
        q.client["experiment/start/cfg_file"]
    )

    if len(df_experiments["id"]) > 0:
        if q.client["experiment/start/cfg_experiment"] is None:
            q.client["experiment/start/cfg_experiment"] = str(
                df_experiments["id"].iloc[0]
            )
        # Default pretrained from the previous experiment to False
        if (
            q.client["experiment/start/cfg_experiment_pretrained"] is None
            or is_create_experiment
        ):
            q.client["experiment/start/cfg_experiment_pretrained"] = False

    if q.client["experiment/start/cfg_category"] != "experiment":
        items += [
            ui.dropdown(
                name="experiment/start/cfg_file",
                label="Problem Type",
                required=True,
                choices=choices_problem_types,
                value=q.client["experiment/start/cfg_file"],
                trigger=True,
                tooltip=tooltips["experiments_problem_type"],
            )
        ]

    model_types = get_model_types(q.client["experiment/start/cfg_file"])
    if len(model_types) > 0:
        choices = [ui.choice(name, label) for name, label in model_types]
        if q.client["experiment/start/cfg_sub"] in [None, ""]:
            q.client["experiment/start/cfg_sub"] = model_types[0][0]
        items += [
            ui.dropdown(
                name="experiment/start/cfg_sub",
                label="Model Type",
                required=True,
                choices=choices,
                value=q.client["experiment/start/cfg_sub"],
                trigger=True,
            )
        ]
    else:
        q.client["experiment/start/cfg_sub"] = ""

    # add model type to cfg file name here
    q.client["experiment/start/cfg_file"] = add_model_type(
        q.client["experiment/start/cfg_file"], q.client["experiment/start/cfg_sub"]
    )

    if (
        show_update_warnings
        and q.client["experiment/start/cfg_file_prev"]
        != q.client["experiment/start/cfg_file"]
        and q.client["experiment/start/cfg_category"] != "experiment"
    ):
        items += [
            ui.message_bar(type="warning", text=warning_message.format("Problem Type"))
        ]
        show_update_warnings = False

    if q.client["experiment/start/cfg_category"] == "experiment":
        items += [
            ui.dropdown(
                name="experiment/start/cfg_experiment",
                label="Experiment",
                required=True,
                choices=[
                    ui.choice(str(row.id), row["name"])
                    for _, row in df_experiments.iterrows()
                ],
                value=q.client["experiment/start/cfg_experiment"],
                trigger=True,
            )
        ]

        if (
            show_update_warnings
            and q.client["experiment/start/cfg_experiment_prev"]
            != q.client["experiment/start/cfg_experiment"]
        ):
            items += [
                ui.message_bar(
                    type="warning", text=warning_message.format("previous Experiment")
                )
            ]

        # Show pretrained weights toggle only for successfully finished experiments
        if (
            df_experiments.loc[
                df_experiments.id == int(q.client["experiment/start/cfg_experiment"]),
                "status",
            ].values[0]
            == "finished"
        ):
            items += [
                ui.toggle(
                    name="experiment/start/cfg_experiment_pretrained",
                    label="Use previous experiment weights",
                    value=q.client["experiment/start/cfg_experiment_pretrained"],
                    trigger=True,
                )
            ]

    # only show yaml option, when not starting from another experiment
    if q.client["experiment/start/cfg_category"] != "experiment":
        items += [
            ui.toggle(
                name="experiment/start/from_yaml",
                label="Import config from YAML",
                value=False,
                trigger=True,
                tooltip=tooltips["experiments_import_config_from_yaml"],
            )
        ]

    if q.args["experiment/start/from_yaml"]:
        items += [
            ui.file_upload(
                name="experiment/upload_yaml",
                label="Upload!",
                multiple=False,
                file_extensions=["yaml"],
            )
        ]

    if q.args["experiment/upload_yaml"] is not None:
        # reset previous, so the UI will be reloaded
        q.client["experiment/start/cfg_file_prev"] = None
        await config_import_uploaded_file(q)

    logger.info(
        f"PREV {q.client['experiment/start/cfg_file_prev']} "
        f"{q.client['experiment/start/cfg_file']} "
        f"{q.client['experiment/start/dataset_prev']} "
        f"{q.client['experiment/start/dataset']} "
        f"{q.client['experiment/start/cfg_experiment_prev']} "
        f"{q.client['experiment/start/cfg_experiment']} "
    )

    # set mode to training
    q.client["experiment/start/cfg_mode/mode"] = "train"

    if q.client["experiment/start/cfg_category"] == "experiment":
        logger.info("Starting from experiment")

        # reset previous config file
        q.client["experiment/start/cfg_file_prev"] = None

        q.client["experiment/start/experiment"] = q.client.app_db.get_experiment(
            q.client["experiment/start/cfg_experiment"]
        )

        parent_path = os.path.dirname(q.client["experiment/start/experiment"].path)
        parent_exp_name = parent_path.split("/")[-1]
        parent_experiment = f"{parent_exp_name}"

        old_config = load_config_yaml(f"{parent_path}/cfg.yaml")
        old_config._parent_experiment = parent_experiment

        q.client["experiment/start/cfg"] = old_config

        # set pretrained weights
        if q.client["experiment/start/cfg_experiment_pretrained"]:
            prev_weights = os.path.join(
                q.client["experiment/start/experiment"].path,
                "checkpoint.pth",
            )
            if os.path.exists(prev_weights):
                q.client[
                    "experiment/start/cfg"
                ].architecture.pretrained_weights = prev_weights
                q.client["experiment/start/cfg"].architecture._visibility[
                    "pretrained_weights"
                ] = -1

        experiments_df = q.client.app_db.get_experiments_df()
        output_dir = os.path.abspath(
            os.path.join(q.client["experiment/start/cfg"].output_directory, "..")
        )
        q.client["experiment/start/cfg"].experiment_name = get_unique_name(
            q.client["experiment/start/cfg"].experiment_name,
            experiments_df["name"].values,
            lambda x: os.path.exists(os.path.join(output_dir, x)),
        )

        # Configuration flags:
        # from_dataset -- take the values from the dataset config
        # from_cfg -- take the values from the configuration file
        # from_default -- take the values from the the default settings
        # from_dataset_args -- take the values from the dataset's q.args
        # Otherwise -- take the values from the q.args (user input)

        # pick default values from config
        if (
            q.client["experiment/start/cfg_experiment_prev"]
            != q.client["experiment/start/cfg_experiment"]
        ):
            q.client["experiment/start/cfg_mode/from_dataset"] = False
            q.client["experiment/start/cfg_mode/from_cfg"] = True
            q.client["experiment/start/cfg_mode/from_dataset_args"] = False

            q.client["experiment/start/dataset"] = str(
                q.client["experiment/start/experiment"].dataset
            )

            items[1].dropdown.value = q.client["experiment/start/dataset"]
        # pick default values from config or dataset
        elif (
            q.client["experiment/start/dataset_prev"]
            != q.client["experiment/start/dataset"]
        ):
            q.client["experiment/start/cfg_mode/from_dataset"] = True
            q.client["experiment/start/cfg_mode/from_cfg"] = True
            q.client["experiment/start/cfg_mode/from_dataset_args"] = False
        # pick default values from args
        else:
            q.client["experiment/start/cfg_mode/from_dataset"] = False
            q.client["experiment/start/cfg_mode/from_cfg"] = False
            q.client["experiment/start/cfg_mode/from_dataset_args"] = True

        q.client["experiment/start/cfg_mode/from_default"] = False
        q.client["experiment/start/cfg_experiment_prev"] = q.client[
            "experiment/start/cfg_experiment"
        ]

    else:
        logger.info("Starting from CFG")

        # reset previous experiment
        q.client["experiment/start/cfg_experiment_prev"] = None

        # pick default values from dataset or config
        if (
            q.client["experiment/start/cfg_file_prev"]
            != q.client["experiment/start/cfg_file"]
        ) or (
            q.client["experiment/start/dataset_prev"]
            != q.client["experiment/start/dataset"]
        ):
            q.client["experiment/start/cfg_mode/from_dataset"] = True
            q.client["experiment/start/cfg_mode/from_cfg"] = True
            q.client["experiment/start/cfg_mode/from_default"] = True
            q.client["experiment/start/cfg_mode/from_dataset_args"] = False
        # pick default values from args
        else:
            q.client["experiment/start/cfg_mode/from_dataset"] = False
            q.client["experiment/start/cfg_mode/from_cfg"] = False
            q.client["experiment/start/cfg_mode/from_default"] = False
            q.client["experiment/start/cfg_mode/from_dataset_args"] = True

        q.client["experiment/start/cfg_file_prev"] = q.client[
            "experiment/start/cfg_file"
        ]

        config_path = (
            f"llm_studio/python_configs/{q.client['experiment/start/cfg_file']}"
        )

        q.client["experiment/start/cfg"] = load_config_py(
            config_path=config_path, config_name="ConfigProblemBase"
        )

    q.client["experiment/start/dataset_prev"] = q.client["experiment/start/dataset"]
    logger.info(f"From dataset {q.client['experiment/start/cfg_mode/from_dataset']}")
    logger.info(f"From cfg {q.client['experiment/start/cfg_mode/from_cfg']}")
    logger.info(f"From default {q.client['experiment/start/cfg_mode/from_default']}")
    logger.info(f"Config file: {q.client['experiment/start/cfg_file']}")

    option_items = get_ui_elements(cfg=q.client["experiment/start/cfg"], q=q)
    items.extend(option_items)

    if q.client["experiment/start/cfg_mode/from_cfg"]:
        q.page["experiment/start"] = ui.form_card(box="content", items=items)
    else:
        q.page["experiment/start"].items = items

    q.client.delete_cards.add("experiment/start")

    q.page["experiment/start/footer"] = ui.form_card(
        box="footer",
        items=[
            ui.inline(
                items=[
                    ui.button(
                        name="experiment/start/run",
                        label="Run experiment",
                        primary=True,
                    )
                ]
            )
        ],
    )
    q.client.delete_cards.add("experiment/start/footer")


async def experiment_run(q: Q, pre: str = "experiment/start") -> None:
    """Start an experiment.

    Args:
        q: Q
        pre: prefix for client key
    """

    logger.info("Starting experiment")
    logger.info(f"{pre}/cfg_file")
    logger.info(f"CFG: {q.client[f'{pre}/cfg_file']}")

    if q.client[f"{pre}/cfg_category"] == "experiment":
        q.client[f"{pre}/cfg_file"] = q.client[f"{pre}/experiment"].config_file

    cfg = q.client[f"{pre}/cfg"]
    cfg = parse_ui_elements(cfg=cfg, q=q, pre=f"{pre}/cfg/")
    cfg.experiment_name = cfg.experiment_name.replace("/", "-")

    stats = os.statvfs(".")
    available_size = stats.f_frsize * stats.f_bavail

    if available_size < default_cfg.min_experiment_disk_space:
        entity = "Experiment" if pre == "experiment/start" else "Prediction"
        q.client["experiment_halt_reason"] = (
            f"Not enough disk space. Available space is {get_size_str(available_size)}."
            f" Required space is "
            f"{get_size_str(default_cfg.min_experiment_disk_space)}. "
            f"{entity} has not started."
        )
        logger.error(q.client["experiment_halt_reason"])
        return

    start_experiment(cfg=cfg, q=q, pre=pre)


def get_experiment_table(
    q, df_viz, predictions, height="calc(100vh - 245px)", actions=None
):
    col_remove = [
        "id",
        "path",
        "mode",
        "seed",
        "process_id",
        "gpu_list",
        "loss",
        "eta",
        "epoch",
        "config_file",
    ]
    if predictions:
        col_remove += ["epoch", "val metric"]

    for col in col_remove:
        if col in df_viz:
            del df_viz[col]
    # df_viz = df_viz.rename(
    #     columns={"process_id": "pid", "config_file": "problem type"},
    # )
    # df_viz["problem type"] = df_viz["problem type"].str.replace("Text ", "")

    if actions == "experiment" and q.client["experiment/list/mode"] == "train":
        actions_dict = {
            "experiment/list/new": "New experiment",
            "experiment/list/rename": "Rename experiment",
            "experiment/list/stop/table": "Stop experiment",
            "experiment/list/delete/table/dialog": "Delete experiment",
        }
    else:
        actions_dict = {}

    min_widths = {
        "name": "350",
        "dataset": "150",
        # "problem type": "190",
        "metric": "75",
        "val metric": "102",
        "progress": "85",
        "status": "90",
        "info": "115",
        "actions": "5" if predictions else "5",
    }

    if predictions:
        for k, v in min_widths.items():
            min_widths[k] = str(int(np.ceil(int(v) * 1.05)))

    return ui_table_from_df(
        q=q,
        df=df_viz,
        name="experiment/list/table",
        sortables=["val metric"],
        filterables=["name", "dataset", "problem type", "metric", "status"],
        searchables=["name", "dataset"],
        numerics=["val metric"],
        tags=["status"],
        progresses=["progress"],
        min_widths=min_widths,
        link_col="name",
        height=height,
        actions=actions_dict,
    )


async def experiment_list(
    q: Q,
    reset: bool = True,
    allowed_statuses: Optional[List[str]] = None,
    actions: bool = True,
) -> None:
    """List all experiments."""

    if q.client["experiment/list/mode"] is None:
        q.client["experiment/list/mode"] = "train"

    if q.client["experiment/list/mode"] == "train":
        q.client["nav/active"] = "experiment/list"
    else:
        q.client["nav/active"] = "experiment/list_predictions"

    if reset:
        await clean_dashboard(q, mode="full")

        q.client["experiment/list/df_experiments"] = get_experiments(
            q,
            mode=q.client["experiment/list/mode"],
            status=allowed_statuses,
        )

        df_viz = q.client["experiment/list/df_experiments"].copy()

        table = get_experiment_table(
            q,
            df_viz,
            q.client["experiment/list/mode"] == "predict",
            actions="experiment" if actions else None,
        )

        message_bar = get_experiment_list_message_bar(q)

        items = [table, message_bar]

        q.page["experiment/list"] = ui.form_card(box="content", items=items)
        q.client.delete_cards.add("experiment/list")

    buttons = [
        ui.button(name="experiment/list/refresh", label="Refresh", primary=True),
        ui.button(
            name="experiment/list/compare",
            label="Compare experiments",
            primary=False,
        ),
        ui.button(name="experiment/list/stop", label="Stop experiments", primary=False),
        ui.button(
            name="experiment/list/delete", label="Delete experiments", primary=False
        ),
    ]

    q.page["dataset/display/footer"] = ui.form_card(
        box="footer", items=[ui.inline(items=buttons)]
    )
    q.client.delete_cards.add("dataset/display/footer")


def get_table_and_message_item_indices(q):
    table_item_idx, message_item_idx = 0, 1
    return table_item_idx, message_item_idx


async def experiment_compare(q: Q, selected_rows: list):
    if q.client["experiment/compare/tab"] is None:
        q.client["experiment/compare/tab"] = "experiment/compare/charts"
    if q.args["experiment/compare/charts"] is not None:
        q.client["experiment/compare/tab"] = "experiment/compare/charts"
    if q.args["experiment/compare/config"] is not None:
        q.client["experiment/compare/tab"] = "experiment/compare/config"

    experiment_ids = [
        q.client["experiment/list/df_experiments"]["id"].iloc[int(idx)]
        for idx in selected_rows
    ]

    await clean_dashboard(q, mode=q.client["experiment/compare/tab"])
    tabs = [
        ui.tab(name="experiment/compare/charts", label="Charts"),
        ui.tab(name="experiment/compare/config", label="Config"),
    ]
    q.page["experiment/compare/tab"] = ui.tab_card(
        box="nav2", link=True, items=tabs, value=q.client["experiment/compare/tab"]
    )
    q.client.delete_cards.add("experiment/compare/tab")

    if q.client["experiment/compare/tab"] == "experiment/compare/charts":
        charts = []
        experiment_names = []

        for experiment_id in experiment_ids:
            experiment = q.client.app_db.get_experiment(experiment_id)
            experiment_path = experiment.path
            charts.append(load_charts(experiment_path))
            current_name = f" {experiment.name}"
            experiment_names.append(current_name)

        await charts_tab(q, charts, experiment_names)

    elif q.client["experiment/compare/tab"] == "experiment/compare/config":
        if q.client["experiment/compare/diff_toggle"] is None:
            q.client["experiment/compare/diff_toggle"] = False

        settings = pd.DataFrame()
        for experiment_id in experiment_ids:
            experiment = q.client.app_db.get_experiment(experiment_id)
            experiment_path = experiment.path
            experiment_cfg = load_config_yaml(os.path.join(experiment_path, "cfg.yaml"))
            items = get_cfg_list_items(experiment_cfg)
            act_df = pd.Series({item.label: item.value for item in items})
            settings[experiment.name] = act_df

        settings.index.name = "setting"

        if q.client["experiment/compare/diff_toggle"]:
            val_counts = settings.T.nunique()
            drop_idx = val_counts[val_counts == 1].index.values
            settings = settings.drop(drop_idx)

        items = [
            ui.toggle(
                name="experiment/compare/diff_toggle",
                label="Show differences only",
                value=q.client["experiment/compare/diff_toggle"],
                trigger=True,
            ),
            ui_table_from_df(
                q=q,
                df=settings.reset_index(),
                name="experiment/compare/summary/table",
                link_col="setting",
                height="calc(100vh - 315px)",
            ),
        ]

        q.page["experiment/compare/config"] = ui.form_card(box="first", items=items)
        q.client.delete_cards.add("experiment/compare/config")

    buttons = [
        ui.button(name="experiment/compare", label="Refresh", primary=True),
        ui.button(name="experiment/list/current", label="Back", primary=False),
    ]
    q.page["experiment/compare/footer"] = ui.form_card(
        box="footer", items=[ui.inline(items=buttons)]
    )
    q.client.delete_cards.add("experiment/compare/footer")


async def experiment_rename_form(q: Q, error: str = "") -> None:
    experiment = q.client.app_db.get_experiment(q.client["experiment/rename/id"])

    experiment_name = experiment.name
    items = [
        ui.textbox(
            name="experiment/rename/name",
            label=f"New name for {experiment_name}",
            value=experiment_name,
            required=True,
        )
    ]

    if error:
        items.append(ui.message_bar(type="error", text=error))

    q.page["experiment/list"].items = items

    buttons = [
        ui.button(name="experiment/rename/action", label="Rename", primary=True),
        ui.button(name="experiment/list/current", label="Abort", primary=False),
    ]
    q.page["dataset/display/footer"] = ui.form_card(
        box="footer", items=[ui.inline(items=buttons)]
    )
    q.client.delete_cards.add("dataset/display/footer")


async def experiment_rename_ui_workflow(q: Q):
    selected_row = q.args["experiment/list/rename"]
    rename_id = q.client["experiment/list/df_experiments"]["id"].iloc[int(selected_row)]
    q.client["experiment/rename/id"] = rename_id
    await experiment_rename_form(q)


async def experiment_rename_action(q, experiment, new_name):
    """Rename experiment with `current_id` id in DB to `new_name`"""

    old_name = experiment.name
    old_path = experiment.path
    new_path = old_path.replace(old_name, new_name)

    if old_path != new_path:
        old_exp_path = f"{old_path}"
        exp_path = f"{new_path}"
        logger.info(f"Renaming {old_exp_path} to {exp_path}")
        shutil.move(os.path.abspath(old_exp_path), os.path.abspath(exp_path))

        for config_file in ["cfg.yaml"]:
            config_path = os.path.join(exp_path, config_file)
            if os.path.exists(config_path):
                experiment_cfg = load_config_yaml(config_path)
                experiment_cfg.experiment_name = new_name
                experiment_cfg.output_directory = new_path
                save_config_yaml(config_path, experiment_cfg)

        rename_files = ["preds"]
        for file in rename_files:
            old_file = get_artifact_path_path(old_name, exp_path, file)
            new_file = get_artifact_path_path(new_name, exp_path, file)
            if os.path.exists(old_file):
                logger.info(f"Renaming {old_file} to {new_file}")
                shutil.move(os.path.abspath(old_file), os.path.abspath(new_file))

        delete_files = ["logs"]  # will be generated on demand with updates
        for file in delete_files:
            file = get_artifact_path_path(old_name, exp_path, file)
            if os.path.exists(file):
                logger.info(f"Deleting {file}")
                os.remove(file)

        q.client.app_db.rename_experiment(experiment.id, new_name, new_path)


async def experiment_delete(q: Q, experiment_ids: List[int]) -> None:
    """Delete selected experiments.

    Args:
        q: Q
        experiment_ids: list of experiment ids to delete
    """

    for experiment_id in experiment_ids:
        experiment = q.client.app_db.get_experiment(experiment_id)
        q.client.app_db.delete_experiment(experiment.id)
        shutil.rmtree(f"{experiment.path}")


async def experiment_stop(q: Q, experiment_ids: List[int]) -> None:
    """Stop selected experiments.

    Args:
        q: Q
        experiment_ids: list of experiment ids to stop
    """

    for experiment_id in experiment_ids:
        experiment = q.client.app_db.get_experiment(experiment_id)

        try:
            ret = kill_child_processes(int(experiment.process_id))
            if ret:
                flag_path = os.path.join(experiment.path, "flags.json")
                write_flag(flag_path, "status", "stopped")
        except Exception as e:
            logger.error(f"Error while stopping the experiment: {e}")
            pass


def load_charts(experiment_path):
    try:
        with SqliteDict(os.path.join(experiment_path, "charts.db")) as charts:
            charts = dict(charts)
    except Exception:
        charts = {}
        logger.warning("Too early, wait for the charts to appear")

    return charts


async def experiment_display(q: Q) -> None:
    """Display a selected experiment."""

    experiment_id = q.client["experiment/list/df_experiments"]["id"].iloc[
        q.client["experiment/display/id"]
    ]
    q.client["experiment/display/experiment_id"] = experiment_id
    experiment = q.client.app_db.get_experiment(experiment_id)
    q.client["experiment/display/experiment"] = experiment

    q.client["experiment/display/experiment_path"] = experiment.path

    status, _ = get_experiment_status(experiment.path)

    charts = load_charts(q.client["experiment/display/experiment_path"])
    q.client["experiment/display/charts"] = charts

    if experiment.mode == "train":
        if q.client["experiment/display/tab"] is None:
            q.client["experiment/display/tab"] = "experiment/display/charts"
    else:
        if q.client["experiment/display/tab"] is None:
            q.client["experiment/display/tab"] = "experiment/display/summary"

    if q.args["experiment/display/charts"] is not None:
        q.client["experiment/display/tab"] = "experiment/display/charts"
    if q.args["experiment/display/summary"] is not None:
        q.client["experiment/display/tab"] = "experiment/display/summary"
    if q.args["experiment/display/train_data_insights"] is not None:
        q.client["experiment/display/tab"] = "experiment/display/train_data_insights"
    if q.args["experiment/display/validation_prediction_insights"] is not None:
        q.client[
            "experiment/display/tab"
        ] = "experiment/display/validation_prediction_insights"
    if q.args["experiment/display/config"] is not None:
        q.client["experiment/display/tab"] = "experiment/display/config"
    if q.args["experiment/display/deployment"] is not None:
        q.client["experiment/display/tab"] = "experiment/display/deployment"
    if q.args["experiment/display/logs"] is not None:
        q.client["experiment/display/tab"] = "experiment/display/logs"
    if q.args["experiment/display/chat"] is not None:
        q.client["experiment/display/tab"] = "experiment/display/chat"

    await clean_dashboard(q, mode=q.client["experiment/display/tab"])

    tabs = [
        ui.tab(name="experiment/display/charts", label="Charts"),
        ui.tab(name="experiment/display/summary", label="Summary"),
    ]
    # html for legacy experiments
    has_train_data_insights = any(
        [
            charts.get(plot_encoding, dict()).get("train_data") is not None
            for plot_encoding in PLOT_ENCODINGS
        ]
    )
    if has_train_data_insights:
        tabs += [
            ui.tab(
                name="experiment/display/train_data_insights",
                label="Train Data Insights",
            )
        ]
    has_validation_prediction_insights = any(
        [
            charts.get(plot_encoding, dict()).get("validation_predictions") is not None
            for plot_encoding in PLOT_ENCODINGS
        ]
    )
    if has_validation_prediction_insights:
        tabs += [
            ui.tab(
                name="experiment/display/validation_prediction_insights",
                label="Validation Prediction Insights",
            )
        ]

    tabs += [
        ui.tab(name="experiment/display/logs", label="Logs"),
        ui.tab(name="experiment/display/config", label="Config"),
    ]

    if status == "finished":
        tabs += [ui.tab(name="experiment/display/chat", label="Chat")]

    q.page["experiment/display/tab"] = ui.tab_card(
        box="nav2", link=True, items=tabs, value=q.client["experiment/display/tab"]
    )
    q.client.delete_cards.add("experiment/display/tab")

    if q.client["experiment/display/tab"] == "experiment/display/charts":
        await charts_tab(q, [charts], [""])
    elif q.client["experiment/display/tab"] in [
        "experiment/display/train_data_insights",
        "experiment/display/validation_prediction_insights",
    ]:
        await insights_tab(charts, q)
    elif q.client["experiment/display/tab"] in ["experiment/display/summary"]:
        await summary_tab(experiment_id, q)
    elif q.client["experiment/display/tab"] in ["experiment/display/config"]:
        await configs_tab(q)
    elif q.client["experiment/display/tab"] in ["experiment/display/logs"]:
        await logs_tab(q)
    elif q.client["experiment/display/tab"] in ["experiment/display/chat"]:
        await chat_tab(q)

    await q.page.save()

    buttons = [
        ui.button(name="experiment/display/refresh", label="Refresh", primary=True)
    ]

    buttons += [
        ui.button(
            name="experiment/display/download_logs",
            label="Download logs/config",
            primary=False,
        )
    ]

    if status == "finished":
        buttons += [
            ui.button(
                name="experiment/display/download_predictions",
                label="Download predictions",
                primary=False,
                disabled=False,
                tooltip=None,
            ),
            ui.button(
                name="experiment/display/download_model",
                label="Download model",
                primary=False,
                disabled=False,
                tooltip=None,
            ),
            ui.button(
                name="experiment/display/push_to_huggingface",
                label="Push checkpoint to huggingface",
                primary=False,
                disabled=False,
                tooltip=None,
            ),
        ]

    buttons += [ui.button(name="experiment/list/current", label="Back", primary=False)]

    q.page["experiment/display/footer"] = ui.form_card(
        box="footer",
        items=[
            ui.inline(items=buttons),
        ],
    )
    q.client.delete_cards.add("experiment/display/footer")


async def insights_tab(charts, q):
    if q.client["experiment/display/tab"] == "experiment/display/train_data_insights":
        key = "train_data"
    elif (
        q.client["experiment/display/tab"]
        == "experiment/display/validation_prediction_insights"
    ):
        key = "validation_predictions"
    for k1 in PLOT_ENCODINGS:
        if k1 not in charts:
            continue
        for k2, v2 in charts[k1].items():
            if k2 != key:
                continue
            if k1 == "html":
                q.page[f"experiment/display/charts/{k1}_{k2}"] = ui.markup_card(
                    box="first", title="", content=v2
                )
                q.client.delete_cards.add(f"experiment/display/charts/{k1}_{k2}")

                continue

            elif k1 == "image":
                q.page[f"experiment/display/charts/{k1}_{k2}"] = ui.image_card(
                    box="first", title="", type="png", image=v2
                )
                q.client.delete_cards.add(f"experiment/display/charts/{k1}_{k2}")
                continue

            elif k1 == "df":
                df = pd.read_parquet(v2)
                min_widths = {
                    col: "350" for col in df.columns if "text" in str(col).lower()
                }
                #
                if key == "train_data":
                    min_widths["Content"] = "800"
                q.page[f"experiment/display/charts/{k1}_{k2}"] = ui.form_card(
                    box="first",
                    items=[
                        ui_table_from_df(
                            q=q,
                            df=df,
                            name=f"experiment/display/charts/{k1}_{k2}",
                            sortables=[
                                col for col in df.columns if col.startswith("Metric")
                            ],
                            markdown_cells=[
                                col
                                for col in df.columns
                                if not col.startswith("Metric")
                            ],
                            searchables=list(df.columns),
                            downloadable=True,
                            resettable=True,
                            min_widths=min_widths,
                            height="calc(100vh - 245px)",
                            max_char_length=50_000,
                            cell_overflow="tooltip",
                        )
                    ],
                )
                q.client.delete_cards.add(f"experiment/display/charts/{k1}_{k2}")
                continue


async def summary_tab(experiment_id, q):
    experiment_df = get_experiments(q)
    input_dict = experiment_df[experiment_df.id == experiment_id].iloc[0].to_dict()
    cfg = load_config_yaml(
        os.path.join(q.client["experiment/display/experiment_path"], "cfg.yaml")
    )
    _ = get_tokenizer(cfg)

    # experiment card
    card_name = "experiment/display/summary/experiment"
    q.page[card_name] = ui.form_card(
        box=ui.box(zone="first"),
        items=[
            ui.separator("Experiment"),
            ui.stats(
                [
                    ui.stat(
                        value=cfg.experiment_name,
                        label="Name",
                    ),
                ],
                justify="between",
                inset=True,
            ),
            ui.stats(
                [
                    ui.stat(
                        value=input_dict["config_file"],
                        label="Problem Type",
                    ),
                ],
                justify="between",
                inset=True,
            ),
        ],
    )
    q.client.delete_cards.add(card_name)

    # datasets card
    card_name = "experiment/display/summary/datasets"
    q.page[card_name] = ui.form_card(
        box=ui.box(zone="first"),
        items=[
            ui.separator("Datasets"),
            ui.stats(
                [
                    ui.stat(
                        value=Path(cfg.dataset.train_dataframe).stem,
                        label="Training Dataset",
                    ),
                ],
                justify="between",
                inset=True,
            ),
            ui.stats(
                [
                    ui.stat(
                        value="-"
                        if cfg.dataset.validation_dataframe in ["", "None", None]
                        else Path(cfg.dataset.validation_dataframe).stem,
                        label="Validation Dataset",
                    ),
                ],
                justify="between",
                inset=True,
            ),
        ],
    )
    q.client.delete_cards.add(card_name)

    # score card
    card_name = "experiment/display/summary/score"
    q.page[card_name] = ui.form_card(
        box=ui.box(zone="first"),
        items=[
            ui.separator("Score"),
            ui.stats(
                [
                    ui.stat(
                        value=input_dict["metric"],
                        label="Metric",
                    ),
                ],
                justify="between",
                inset=True,
            ),
            ui.stats(
                [
                    ui.stat(
                        value="-"
                        if input_dict["val metric"] in ["", "None", None]
                        else str(input_dict["val metric"]),
                        label="Validation Score",
                    ),
                ],
                justify="between",
                inset=True,
            ),
        ],
    )
    q.client.delete_cards.add(card_name)

    # main configs card
    card_name = "experiment/display/summary/main_configs"
    q.page[card_name] = ui.form_card(
        box=ui.box(zone="second"),
        items=[
            ui.separator("Main Configurations"),
            ui.stats(
                [
                    ui.stat(
                        value=cfg.llm_backbone,
                        label="LLM Backbone",
                    ),
                    ui.stat(
                        value=str(cfg.training.lora),
                        label="Lora",
                    ),
                    ui.stat(
                        value=str(cfg.training.epochs),
                        label="Epochs",
                    ),
                    ui.stat(
                        value=str(cfg.training.batch_size),
                        label="Batch Size",
                    ),
                ],
                justify="between",
                inset=True,
            ),
            ui.stats(
                [
                    ui.stat(
                        value=str(input_dict["loss"]),
                        label="Loss Function",
                    ),
                    ui.stat(
                        value=cfg.architecture.backbone_dtype,
                        label="Backbone Dtype",
                    ),
                    ui.stat(
                        value=str(cfg.architecture.gradient_checkpointing),
                        label="Gradient Checkpointing",
                    ),
                    ui.stat(
                        value=input_dict["gpu_list"],
                        label="GPU List",
                    ),
                ],
                justify="between",
                inset=True,
            ),
        ],
    )
    q.client.delete_cards.add(card_name)

    # code card
    card_name = "experiment/display/summary/code"
    content = get_experiment_summary_code_card(cfg=cfg)
    q.page[card_name] = ui.markdown_card(
        box=ui.box(zone="third"),
        title="",
        content=content,
    )
    q.client.delete_cards.add(card_name)


async def configs_tab(q):
    experiment_cfg = load_config_yaml(
        os.path.join(q.client["experiment/display/experiment_path"], "cfg.yaml")
    )
    items = get_cfg_list_items(experiment_cfg)
    q.page["experiment/display/config"] = ui.stat_list_card(
        box="first", items=items, title=""
    )
    q.client.delete_cards.add("experiment/display/config")


async def logs_tab(q):
    logs_path = f"{q.client['experiment/display/experiment_path']}/logs.log"
    text = ""
    in_pre = 0
    # Read log file only if it already exists
    if os.path.exists(logs_path):
        with open(logs_path, "r") as f:
            for line in f.readlines():
                if in_pre == 0:
                    text += "<div>"
                if "INFO: Lock" in line:
                    continue
                # maximum line length
                n = 250
                chunks = [line[i : i + n] for i in range(0, len(line), n)]
                text += "</div><div>".join(chunks)

                # Check for formatted HTML text
                if "<pre>" in line:
                    in_pre += 1
                if "</pre>" in line:
                    in_pre -= 1
                if in_pre == 0:
                    text += "</div>"
    items = [ui.text(text)]
    q.page["experiment/display/logs"] = ui.form_card(box="first", items=items, title="")
    q.client.delete_cards.add("experiment/display/logs")


def subsample(key1, key2, value, max_plot_points=1000):
    act_plot_points = len(value["steps"])
    if act_plot_points > max_plot_points:
        stride = int(np.ceil(act_plot_points / max_plot_points))
        value["steps"] = value["steps"][::stride]
        value["values"] = value["values"][::stride]
        logger.info(
            f"{key1} {key2} sampled from size {act_plot_points} to size "
            f"{len(value['steps'])} using stride {stride}."
        )
    return value


def unite_validation_metric_charts(charts_list):
    unique_metrics = []
    for chart in charts_list:
        unique_metrics.extend(list(chart.get("validation", {}).keys()))

    unique_metrics = set([key for key in unique_metrics if key != "loss"])

    if len(unique_metrics) > 1:
        for chart in charts_list:
            if "validation" in chart:
                for key in unique_metrics:
                    if key in chart["validation"]:
                        chart["validation"]["metric"] = chart["validation"][key]
                        del chart["validation"][key]
    return charts_list


async def charts_tab(q, charts_list, legend_labels):
    charts_list = unite_validation_metric_charts(charts_list)

    box = ["first", "first", "second", "second"]
    cnt = 0
    for k1 in ["meta", "train", "validation"]:
        if all([k1 not in charts for charts in charts_list]):
            continue

        all_second_keys: Set = set()
        for charts in charts_list:
            if k1 in charts:
                all_second_keys = all_second_keys.union(set(charts[k1].keys()))

        # Always plot loss in the lower left corner
        if "loss" in all_second_keys:
            all_second_keys.remove("loss")
            list_all_second_keys = ["loss"] + list(all_second_keys)
        else:
            list_all_second_keys = list(all_second_keys)

        for k2 in list_all_second_keys:
            logger.info(f"{k1} {k2}")

            items = []

            tooltip = ""
            if k1 == "meta" and k2 == "lr":
                tooltip = "Current learning rate throughout the training process."
            elif k1 == "train" and k2 == "loss":
                tooltip = (
                    "Current training loss throughout the training process. "
                    "Loss is calculated as the average of the last ten batches."
                )
            elif k1 == "validation" and k2 == "loss":
                tooltip = (
                    "Current validation loss throughout the training process. "
                    "Loss is calculated as the average of all validation batches. "
                )
            elif k1 == "validation" and k2 != "loss":
                tooltip = (
                    "Current validation metric throughout the training process. "
                    "Metric is calculated on full validation set predictions."
                )
            else:
                continue

            title = f"{k1} {k2}".upper().replace("META LR", "LEARNING RATE")
            if k2 == "loss":
                title = title.replace("LOSS", "BATCH LOSS")

            items.append(ui.text(title, tooltip=tooltip))

            rows = []

            max_samples = q.client["chart_plot_max_points"]
            for charts, label in zip(charts_list, legend_labels):
                if k1 not in charts or k2 not in charts[k1]:
                    continue

                v2 = charts[k1][k2]
                v2 = subsample(k1, k2, v2, max_samples)

                if k2 == "lr" and "lr_diff" in charts["meta"]:
                    v3 = charts["meta"]["lr_diff"]
                    v3 = subsample("meta", "lr_diff", v3, max_samples)
                    rows.extend(
                        [
                            (v2["steps"][i], f"learning rate{label}", v2["values"][i])
                            for i in range(len(v2["values"]))
                        ]
                        + [
                            (
                                v3["steps"][i],
                                f"differential learning rate{label}",
                                v3["values"][i],
                            )
                            for i in range(len(v3["values"]))
                        ]
                    )
                    color = "=type"
                    fields = ["step", "type", k2]

                elif len(charts_list) > 1:
                    rows.extend(
                        [
                            (v2["steps"][i], label.strip(), v2["values"][i])
                            for i in range(len(v2["values"]))
                        ]
                    )
                    color = "=type"
                    fields = ["step", "type", "value"]
                else:
                    rows.extend(
                        [
                            (v2["steps"][i], v2["values"][i])  # type: ignore
                            for i in range(len(v2["values"]))
                        ]
                    )
                    color = wave_theme.color
                    fields = ["step", "value"]

            d = data(fields=fields, rows=rows, pack=True)

            viz = ui.visualization(
                plot=ui.plot(
                    [
                        ui.mark(
                            type="line",
                            x_title="step",
                            x_scale="linear",
                            y_scale="linear",
                            x="=step",
                            y="=value",
                            color=color,
                            y_min=0 if k1 == "meta" and k2 == "lr" else None,
                            color_range=wave_theme.color_range,
                        )
                    ]
                ),
                data=d,  # type: ignore
                interactions=["brush"],
                height="calc((100vh - 275px)*0.41)",
                width="560px",
            )

            items.append(viz)

            if k1 == "validation" and k2 == "loss" and np.sum(v2["values"]) == 0:
                items.append(
                    ui.message_bar(
                        type="info",
                        text="Validation batch loss cannot be \
                        calculated for this problem type.",
                    )
                )

            q.page[f"experiment/display/charts/{k1}_{k2}"] = ui.form_card(
                box=box[cnt], items=items
            )
            q.client.delete_cards.add(f"experiment/display/charts/{k1}_{k2}")

            cnt += 1


async def experiment_artifact_build_error_dialog(q: Q, error: str):
    q.page["meta"].dialog = ui.dialog(
        "Failed to build artifact", items=[ui.text(error)], closable=True
    )
    q.client["keep_meta"] = True


async def experiment_download_artifact(
    q: Q,
    get_artifact_path_fn: Callable[[str, str], str],
    save_artifact_fn: Callable[[str, str], str],
    additional_log: Optional[str] = "",
    min_disk_space: Optional[float] = 0.0,
):
    """Download specific artifact, if it does not exist, create it on demand

    Args:
        q: Q
        get_artifact_path_fn: function that returns path to the artifact
        save_artifact_fn: function that generates the artifact and returns its path
        additional_log: additional information to be logged
        min_disk_space: minimal disk available needed to generate artifact
    """

    experiment = q.client["experiment/display/experiment"]
    experiment_path = q.client["experiment/display/experiment_path"]

    zip_path = get_artifact_path_fn(experiment.name, experiment_path)

    if not os.path.exists(zip_path):
        try:
            check_available_space(experiment_path, min_disk_space)
        except LLMResourceException as e:
            error = f"Cannot create {os.path.basename(zip_path)}. {e}"
            await experiment_artifact_build_error_dialog(q, error)
            return

        logger.info(f"Creating {zip_path} on demand")
        zip_path = save_artifact_fn(experiment.name, experiment_path)

    if additional_log:
        logger.info(f"{additional_log}: {zip_path}")

    q.page["meta"].script = ui.inline_script(
        f'window.open("{get_download_link(q, zip_path)}", "_blank");'
    )
    await q.page.save()


async def experiment_download_predictions(q: Q):
    """Download experiment predictions."""
    await experiment_download_artifact(
        q, get_predictions_path, save_prediction_outputs, "Predictions path", None
    )


async def experiment_download_logs(q: Q):
    """Download experiment logs."""

    experiment = q.client["experiment/display/experiment"]
    experiment_path = q.client["experiment/display/experiment_path"]
    zip_path = get_logs_path(experiment.name, experiment_path)

    if not os.path.exists(zip_path):
        logs = q.client["experiment/display/charts"]
        logger.info(f"Creating {zip_path} on demand")
        zip_path = save_logs(experiment.name, experiment_path, logs)

    download_url = get_download_link(q, zip_path)
    logger.info(f"Logs URL: {download_url}")

    q.page["meta"].script = ui.inline_script(
        f'window.open("{download_url}", "_blank");'
    )
    await q.page.save()


async def config_import_uploaded_file(q: Q):
    """ "Importing a config file from drag and drop to the filesystem"""

    file_url = q.args["experiment/upload_yaml"][0]
    file_name = file_url.split("/")[-1]
    path = f"{get_data_dir(q)}/{file_name}"

    local_path = await q.site.download(file_url, path)

    await q.site.unload(q.args["experiment/upload_yaml"][0])

    with open(local_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data = flatten_dict(yaml_data)

    q.client["experiment/yaml_data"] = yaml_data


async def show_message(q, msg_key, page, idx, msg_type):
    info = q.client[msg_key]
    if info:
        q.page[page].items[idx].message_bar.text = info
        q.page[page].items[idx].message_bar.type = msg_type
        q.client[msg_key] = ""


def get_experiment_list_message_bar(q):
    if q.client["experiment_halt_reason"]:
        msg_bar = ui.message_bar(type="error", text=q.client["experiment_halt_reason"])
        del q.client["experiment_halt_reason"]

    elif q.client["force_disable_pipelines"]:
        msg_bar = ui.message_bar(type="info", text=q.client["force_disable_pipelines"])
        del q.client["force_disable_pipelines"]

    else:
        msg_bar = ui.message_bar(type="info", text="")

    return msg_bar


async def experiment_download_model(q: Q, error: str = ""):
    experiment = q.client["experiment/display/experiment"]
    experiment_path = q.client["experiment/display/experiment_path"]
    zip_path = get_model_path(experiment.name, experiment_path)

    if not os.path.exists(zip_path):
        logger.info(f"Creating {zip_path} on demand")
        cfg = load_config_yaml(os.path.join(experiment_path, "cfg.yaml"))

        device = "cuda"
        experiments = get_experiments(q)
        num_running_queued = len(
            experiments[experiments["status"].isin(["queued", "running"])]
        )
        if num_running_queued > 0 or (
            cfg.training.lora and cfg.architecture.backbone_dtype in ("int4", "int8")
        ):
            logger.info("Preparing model on CPU. This might slow down the progress.")
            device = "cpu"
        with set_env(HUGGINGFACE_TOKEN=q.client["default_huggingface_api_token"]):
            cfg, model, tokenizer = load_cfg_model_tokenizer(
                experiment_path, merge=True, device=device
            )

        model = unwrap_model(model)
        checkpoint_path = cfg.output_directory
        model.backbone.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

        card = get_model_card(cfg, model, repo_id="<path_to_local_folder>")
        card.save(os.path.join(experiment_path, "model_card.md"))

        logger.info(f"Creating Zip File at {zip_path}")
        zf = zipfile.ZipFile(zip_path, "w")

        FILES_TO_PUSH = [
            "vocab.json",
            "sentencepiece.bpe.model",
            "bpe_encoder.bin",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "merges.txt",
            "generation_config.json",
            "config.json",
            "added_tokens.json",
            "model_card.md",
        ]

        for file in FILES_TO_PUSH:
            path = os.path.join(experiment_path, file)
            if os.path.isfile(path):
                add_file_to_zip(zf=zf, path=path)

        # Add model weight files
        weight_files = glob.glob(os.path.join(checkpoint_path, "pytorch_model*.*"))
        for file in weight_files:
            add_file_to_zip(zf=zf, path=file)

        zf.close()

    download_url = get_download_link(q, zip_path)
    logger.info(f"Logs URL: {download_url}")

    q.page["meta"].script = ui.inline_script(
        f'window.open("{download_url}", "_blank");'
    )
    await q.page.save()


async def experiment_push_to_huggingface_dialog(q: Q, error: str = ""):
    if q.args["experiment/display/push_to_huggingface"] or error:
        devices = ["cpu", "cpu_shard"] + [
            f"cuda:{idx}" for idx in range(torch.cuda.device_count())
        ]
        default_device = "cuda:0"

        experiments = get_experiments(q)
        num_running_queued = len(
            experiments[experiments["status"].isin(["queued", "running"])]
        )
        if num_running_queued > 0:
            default_device = "cpu"

        try:
            huggingface_hub.login(q.client["default_huggingface_api_token"])
            user_id = huggingface_hub.whoami()["name"]
        except Exception:
            user_id = ""

        dialog_items = [
            ui.message_bar("error", error, visible=True if error else False),
            ui.textbox(
                name="experiment/display/push_to_huggingface/account_name",
                label="Account Name",
                value=user_id,
                width="500px",
                required=False,
                tooltip=(
                    "The account name on HF to push the model to. "
                    "Leaving it empty will push it to the default user account."
                ),
            ),
            ui.textbox(
                name="experiment/display/push_to_huggingface/model_name",
                label="Model Name",
                value=hf_repo_friendly_name(
                    q.client["experiment/display/experiment"].name
                ),
                width="500px",
                required=True,
                tooltip="The name of the model as shown on HF.",
            ),
            ui.dropdown(
                name="experiment/display/push_to_huggingface/device",
                label="Device for preparing the model",
                required=True,
                value=default_device,
                width="500px",
                choices=[ui.choice(str(d), str(d)) for d in devices],
                tooltip=(
                    "The local device to prepare the model before pushing it to HF. "
                    "CPU will never load the weights to the GPU, which can be useful "
                    "for large models, but will be significantly slower. "
                    "Cpu_shard will first load on CPU and then shard on all GPUs "
                    "before pushing to HF."
                ),
            ),
            ui.textbox(
                name="experiment/display/push_to_huggingface/api_key",
                label="Huggingface API Key",
                value=q.client["default_huggingface_api_token"],
                width="500px",
                password=True,
                required=True,
                tooltip="HF API key, needs write access.",
            ),
            ui.toggle(
                name="default_safe_serialization",
                label="Use Hugging Face safetensors for safe serialization",
                value=q.client["default_safe_serialization"],
            ),
            ui.buttons(
                [
                    ui.button(
                        name="experiment/display/push_to_huggingface_submit",
                        label="Export",
                        primary=True,
                    ),
                    ui.button(name="cancel", label="Cancel", primary=False),
                ]
            ),
        ]
    elif q.args["experiment/display/push_to_huggingface_submit"]:
        await busy_dialog(
            q=q,
            title="Exporting to HuggingFace",
            text="Model size can affect the export time significantly.",
        )

        experiment_path = q.client["experiment/display/experiment_path"]
        with set_env(HUGGINGFACE_TOKEN=q.client["default_huggingface_api_token"]):
            cfg, model, tokenizer = load_cfg_model_tokenizer(
                experiment_path,
                merge=True,
                device=q.client["experiment/display/push_to_huggingface/device"],
            )

        check_disk_space(model.backbone, "./")

        huggingface_hub.login(
            q.client["experiment/display/push_to_huggingface/api_key"]
        )

        user_id = q.client["experiment/display/push_to_huggingface/account_name"]
        if user_id == "":
            user_id = huggingface_hub.whoami()["name"]
        exp_name = q.client[
            "experiment/display/push_to_huggingface/model_name"
        ].replace(".", "-")
        repo_id = f"{user_id}/{exp_name}"

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
            path_or_fileobj=f"{experiment_path}/cfg.yaml",
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
            safe_serialization=q.client["default_safe_serialization"],
        )

        # Updating Config HF attributes & # re-save
        cfg.hf.account_name = user_id
        cfg.hf.model_name = exp_name
        save_config_yaml(f"{cfg.output_directory}/cfg.yaml", cfg)

        # push pipeline to hub
        template_env = Environment(
            loader=FileSystemLoader(searchpath="llm_studio/src/")
        )
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

        custom_pipeline_path = os.path.join(experiment_path, "h2oai_pipeline.py")
        with open(custom_pipeline_path, "w") as f:
            f.write(custom_pipeline)

        api.upload_file(
            path_or_fileobj=custom_pipeline_path,
            path_in_repo="h2oai_pipeline.py",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload h2oai_pipeline.py",
        )

        dialog_items = [
            ui.message_bar("success", "Success"),
            ui.buttons(
                [
                    ui.button(name="ok", label="OK", primary=True),
                ]
            ),
        ]

    dialog = ui.dialog(
        title="Push to HuggingFace Hub",
        items=dialog_items,
        closable=True,
        name="push_to_huggingface_dialog",
    )

    q.page["meta"].dialog = dialog
    q.client["keep_meta"] = True


def get_model_card(cfg, model, repo_id) -> huggingface_hub.ModelCard:
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


def get_experiment_summary_code_card(cfg) -> str:
    with open(
        os.path.join("model_cards", cfg.environment._summary_card_template), "r"
    ) as f:
        text = f.read()

    # Model repo
    text = text.replace(
        "{{repo_id}}", cfg.hf.repo_id if cfg.hf.repo_id else "account/model"
    )

    # Versions
    text = text.replace("{{transformers_version}}", transformers.__version__)
    text = text.replace("{{einops_version}}", einops.__version__)
    text = text.replace("{{accelerate_version}}", accelerate.__version__)
    text = text.replace("{{torch_version}}", torch.__version__)

    # Configs
    text = text.replace("{{text_prompt_start}}", str(cfg.dataset.text_prompt_start))
    text = text.replace(
        "{{text_answer_separator}}", str(cfg.dataset.text_answer_separator)
    )
    text = text.replace(
        "{{end_of_sentence}}",
        str(cfg._tokenizer_eos_token) if cfg.dataset.add_eos_token_to_prompt else "",
    )

    text = text.replace("{{min_new_tokens}}", str(cfg.prediction.min_length_inference))
    text = text.replace("{{max_new_tokens}}", str(cfg.prediction.max_length_inference))
    text = text.replace("{{use_fast}}", str(cfg.tokenizer.use_fast))
    text = text.replace("{{do_sample}}", str(cfg.prediction.do_sample))
    text = text.replace("{{num_beams}}", str(cfg.prediction.num_beams))
    text = text.replace("{{temperature}}", str(cfg.prediction.temperature))
    text = text.replace(
        "{{repetition_penalty}}", str(cfg.prediction.repetition_penalty)
    )

    return text
