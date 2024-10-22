import functools
import hashlib
import logging
import os
import re
import shutil
import textwrap
import time
import traceback
from typing import List, Optional

import pandas as pd
from h2o_wave import Q, ui
from h2o_wave.types import FormCard, ImageCard, MarkupCard, StatListItem, Tab

from llm_studio.app_utils.config import default_cfg
from llm_studio.app_utils.db import Dataset
from llm_studio.app_utils.sections.common import clean_dashboard
from llm_studio.app_utils.sections.experiment import experiment_start
from llm_studio.app_utils.sections.histogram_card import histogram_card
from llm_studio.app_utils.utils import (
    add_model_type,
    azure_download,
    azure_file_options,
    check_valid_upload_content,
    clean_error,
    dir_file_table,
    get_data_dir,
    get_dataset_elements,
    get_datasets,
    get_experiments_status,
    get_frame_stats,
    get_model_types,
    get_problem_types,
    get_unique_dataset_name,
    h2o_drive_download,
    h2o_drive_file_options,
    huggingface_download,
    kaggle_download,
    local_download,
    make_label,
    parse_ui_elements,
    remove_temp_files,
    s3_download,
    s3_file_options,
)
from llm_studio.app_utils.wave_utils import busy_dialog, ui_table_from_df
from llm_studio.src.datasets.conversation_chain_handler import get_conversation_chains
from llm_studio.src.tooltips import tooltips
from llm_studio.src.utils.config_utils import (
    load_config_py,
    load_config_yaml,
    save_config_yaml,
)
from llm_studio.src.utils.data_utils import (
    get_fill_columns,
    read_dataframe,
    read_dataframe_drop_missing_labels,
    sanity_check,
)
from llm_studio.src.utils.plot_utils import PlotData

logger = logging.getLogger(__name__)


def file_extension_is_compatible(q):
    cfg = q.client["dataset/import/cfg"]
    allowed_extensions = cfg.dataset._allowed_file_extensions

    is_correct_extension = []
    for mode in ["train", "validation"]:
        dataset_name = q.client[f"dataset/import/cfg/{mode}_dataframe"]

        if dataset_name is None or dataset_name == "None":
            continue
        is_correct_extension.append(dataset_name.endswith(allowed_extensions))
    return all(is_correct_extension)


async def dataset_import(
    q: Q,
    step: int,
    edit: Optional[bool] = False,
    error: Optional[str] = "",
    warning: Optional[str] = "",
    info: Optional[str] = "",
    allow_merge: bool = True,
) -> None:
    """Display dataset import cards.

    Args:
        q: Q
        step: current step of wizard
        edit: whether in edit mode
        error: optional error message
        warning: optional warning message
        info: optional info message
        allow_merge: whether to allow merging dataset when importing
    """

    await clean_dashboard(q, mode="full")
    q.client["nav/active"] = "dataset/import"
    if step == 1:  # select import data source
        q.page["dataset/import"] = ui.form_card(box="content", items=[])
        q.client.delete_cards.add("dataset/import")

        if q.client["dataset/import/source"] is None:
            q.client["dataset/import/source"] = "Upload"

        import_choices = [
            ui.choice("Upload", "Upload"),
            ui.choice("Local", "Local"),
            ui.choice("S3", "AWS S3"),
            ui.choice("Azure", "Azure Datalake"),
            ui.choice("H2O-Drive", "H2O-Drive"),
            ui.choice("Kaggle", "Kaggle"),
            ui.choice("Huggingface", "Hugging Face"),
        ]

        items = [
            ui.text_l("Import dataset"),
            ui.dropdown(
                name="dataset/import/source",
                label="Source",
                value=(
                    "Upload"
                    if q.client["dataset/import/source"] is None
                    else q.client["dataset/import/source"]
                ),
                choices=import_choices,
                trigger=True,
                tooltip="Source of dataset import",
            ),
        ]

        if q.client["dataset/import/source"] == "S3":
            if q.client["dataset/import/s3_bucket"] is None:
                q.client["dataset/import/s3_bucket"] = q.client[
                    "default_aws_bucket_name"
                ]
            if q.client["dataset/import/s3_access_key"] is None:
                q.client["dataset/import/s3_access_key"] = q.client[
                    "default_aws_access_key"
                ]
            if q.client["dataset/import/s3_secret_key"] is None:
                q.client["dataset/import/s3_secret_key"] = q.client[
                    "default_aws_secret_key"
                ]

            files: List[str] | Exception = s3_file_options(
                q.client["dataset/import/s3_bucket"],
                q.client["dataset/import/s3_access_key"],
                q.client["dataset/import/s3_secret_key"],
            )

            # Handle errors in S3 connection and display them nicely below
            if isinstance(files, Exception):
                warning = str(files)
                files = []

            if len(files) == 0:
                ui_filename = ui.textbox(
                    name="dataset/import/s3_filename",
                    label="File name",
                    value="",
                    required=True,
                    tooltip="File name to be imported",
                )
            else:
                default_file = files[0]
                ui_filename = ui.dropdown(
                    name="dataset/import/s3_filename",
                    label="File name",
                    value=default_file,
                    choices=[ui.choice(x, x.split("/")[-1]) for x in files],
                    required=True,
                    tooltip="File name to be imported",
                )

            items += [
                ui.textbox(
                    name="dataset/import/s3_bucket",
                    label="S3 bucket name",
                    value=q.client["dataset/import/s3_bucket"],
                    trigger=True,
                    required=True,
                    tooltip="S3 bucket name including relative paths",
                ),
                ui.textbox(
                    name="dataset/import/s3_access_key",
                    label="AWS access key",
                    value=q.client["dataset/import/s3_access_key"],
                    trigger=True,
                    required=True,
                    password=True,
                    tooltip="Optional AWS access key; empty for anonymous access.",
                ),
                ui.textbox(
                    name="dataset/import/s3_secret_key",
                    label="AWS secret key",
                    value=q.client["dataset/import/s3_secret_key"],
                    trigger=True,
                    required=True,
                    password=True,
                    tooltip="Optional AWS secret key; empty for anonymous access.",
                ),
                ui_filename,
            ]

        elif q.client["dataset/import/source"] == "Azure":
            if q.client["dataset/import/azure_conn_string"] is None:
                q.client["dataset/import/azure_conn_string"] = q.client[
                    "default_azure_conn_string"
                ]
            if q.client["dataset/import/azure_container"] is None:
                q.client["dataset/import/azure_container"] = q.client[
                    "default_azure_container"
                ]

            files = azure_file_options(
                q.client["dataset/import/azure_conn_string"],
                q.client["dataset/import/azure_container"],
            )

            if not files:
                ui_filename = ui.textbox(
                    name="dataset/import/azure_filename",
                    label="File name",
                    value="",
                    required=True,
                    tooltip="File name to be imported",
                )
            else:
                default_file = files[0]
                ui_filename = ui.dropdown(
                    name="dataset/import/azure_filename",
                    label="File name",
                    value=default_file,
                    choices=[ui.choice(x, x.split("/")[-1]) for x in files],
                    required=True,
                    tooltip="File name to be imported",
                )

            items += [
                ui.textbox(
                    name="dataset/import/azure_conn_string",
                    label="Datalake connection string",
                    value=q.client["dataset/import/azure_conn_string"],
                    trigger=True,
                    required=True,
                    password=True,
                    tooltip="Azure connection string to connect to Datalake storage",
                ),
                ui.textbox(
                    name="dataset/import/azure_container",
                    label="Datalake container name",
                    value=q.client["dataset/import/azure_container"],
                    trigger=True,
                    required=True,
                    tooltip="Azure Datalake container name including relative paths",
                ),
                ui_filename,
            ]

        elif q.client["dataset/import/source"] == "Upload":
            items += [
                ui.file_upload(
                    name="dataset/import/local_upload",
                    label="Upload!",
                    multiple=False,
                    file_extensions=default_cfg.allowed_file_extensions,
                )
            ]

        elif q.client["dataset/import/source"] == "Local":
            current_path = (
                q.client["dataset/import/local_path_current"]
                if q.client["dataset/import/local_path_current"] is not None
                else os.path.expanduser("~")
            )

            if q.args.__wave_submission_name__ == "dataset/import/local_path_list":
                idx = int(q.args["dataset/import/local_path_list"][0])
                options = q.client["dataset/import/local_path_list_last"]
                new_path = os.path.abspath(os.path.join(current_path, options[idx]))
                if os.path.exists(new_path):
                    current_path = new_path

            results_df = dir_file_table(current_path)
            files_list = results_df[current_path].tolist()
            q.client["dataset/import/local_path_list_last"] = files_list
            q.client["dataset/import/local_path_current"] = current_path

            items += [
                ui.textbox(
                    name="dataset/import/local_path",
                    label="File location",
                    value=current_path,
                    required=True,
                    tooltip="Location of file to be imported",
                ),
                ui_table_from_df(
                    q=q,
                    df=results_df,
                    name="dataset/import/local_path_list",
                    sortables=[],
                    searchables=[],
                    min_widths={current_path: "400"},
                    link_col=current_path,
                    height="calc(65vh)",
                ),
            ]

        elif q.client["dataset/import/source"] == "H2O-Drive":

            files = await h2o_drive_file_options(q)

            # Handle errors in h2o_drive connection and display them nicely below
            if isinstance(files, Exception):
                warning = str(files)
                files = []

            if len(files) == 0:
                ui_filename = ui.textbox(
                    name="dataset/import/h2o_drive_filename",
                    label="File name",
                    value="No files found",
                    required=True,
                    disabled=True,
                    tooltip="File name to be imported",
                )
            else:
                default_file = files[0]
                ui_filename = ui.dropdown(
                    name="dataset/import/h2o_drive_filename",
                    label="File name",
                    value=default_file,
                    choices=[ui.choice(x, x.split("/")[-1]) for x in files],
                    required=True,
                    tooltip="File name to be imported",
                )

            items += [ui_filename]

        elif q.client["dataset/import/source"] == "Kaggle":
            if q.client["dataset/import/kaggle_access_key"] is None:
                q.client["dataset/import/kaggle_access_key"] = q.client[
                    "default_kaggle_username"
                ]
            if q.client["dataset/import/kaggle_secret_key"] is None:
                q.client["dataset/import/kaggle_secret_key"] = q.client[
                    "default_kaggle_secret_key"
                ]

            items += [
                ui.textbox(
                    name="dataset/import/kaggle_command",
                    label="Kaggle API command",
                    placeholder="kaggle competitions download -c dataset",
                    required=True,
                    tooltip="Kaggle API command to be executed",
                ),
                ui.textbox(
                    name="dataset/import/kaggle_access_key",
                    label="Kaggle username",
                    value=q.client["dataset/import/kaggle_access_key"],
                    required=True,
                    password=False,
                    tooltip="Kaggle username for API authentication",
                ),
                ui.textbox(
                    name="dataset/import/kaggle_secret_key",
                    label="Kaggle secret key",
                    value=q.client["dataset/import/kaggle_secret_key"],
                    required=True,
                    password=True,
                    tooltip="Kaggle secret key for API authentication",
                ),
            ]
        elif q.client["dataset/import/source"] == "Huggingface":

            if q.client["dataset/import/huggingface_split"] is None:
                q.client["dataset/import/huggingface_split"] = "train"
            if q.client["dataset/import/huggingface_api_token"] is None:
                q.client["dataset/import/huggingface_api_token"] = q.client[
                    "default_huggingface_api_token"
                ]

            items += [
                ui.textbox(
                    name="dataset/import/huggingface_dataset",
                    label="Hugging Face dataset",
                    value=q.client["dataset/import/huggingface_dataset"],
                    required=True,
                    tooltip="Name of the Hugging Face dataset",
                ),
                ui.textbox(
                    name="dataset/import/huggingface_split",
                    label="Split",
                    value=q.client["dataset/import/huggingface_split"],
                    required=True,
                    password=False,
                    tooltip="Split of the dataset",
                ),
                ui.textbox(
                    name="dataset/import/huggingface_api_token",
                    label="Hugging Face API token",
                    value=q.client["dataset/import/huggingface_api_token"],
                    required=False,
                    password=True,
                    tooltip="Optional Hugging Face API token",
                ),
            ]

        allowed_types = ", ".join(default_cfg.allowed_file_extensions)
        allowed_types = " or".join(allowed_types.rsplit(",", 1))
        items += [
            ui.message_bar(type="info", text=info + f"Must be a {allowed_types} file."),
            ui.message_bar(type="error", text=error),
            ui.message_bar(type="warning", text=warning),
        ]

        q.page["dataset/import"].items = items

        buttons = [ui.button(name="dataset/list", label="Abort")]
        if q.client["dataset/import/source"] != "Upload":
            buttons.insert(
                0, ui.button(name="dataset/import/2", label="Continue", primary=True)
            )

        q.page["dataset/import/footer"] = ui.form_card(
            box="footer", items=[ui.inline(items=buttons, justify="start")]
        )
        q.client.delete_cards.add("dataset/import/footer")

        q.client["dataset/import/id"] = None
        q.client["dataset/import/cfg_file"] = None

    elif step == 2:  # download / import data from source
        q.page["dataset/import/footer"] = ui.form_card(box="footer", items=[])
        try:
            if not q.args["dataset/import/cfg_file"] and not edit:
                if q.client["dataset/import/source"] == "S3":
                    (
                        q.client["dataset/import/path"],
                        q.client["dataset/import/name"],
                    ) = await s3_download(
                        q,
                        q.client["dataset/import/s3_bucket"],
                        q.client["dataset/import/s3_filename"],
                        q.client["dataset/import/s3_access_key"],
                        q.client["dataset/import/s3_secret_key"],
                    )
                elif q.client["dataset/import/source"] == "Azure":
                    (
                        q.client["dataset/import/path"],
                        q.client["dataset/import/name"],
                    ) = await azure_download(
                        q,
                        q.client["dataset/import/azure_conn_string"],
                        q.client["dataset/import/azure_container"],
                        q.client["dataset/import/azure_filename"],
                    )
                elif q.client["dataset/import/source"] in ("Upload", "Local"):
                    (
                        q.client["dataset/import/path"],
                        q.client["dataset/import/name"],
                    ) = await local_download(q, q.client["dataset/import/local_path"])
                elif q.client["dataset/import/source"] == "H2O-Drive":
                    (
                        q.client["dataset/import/path"],
                        q.client["dataset/import/name"],
                    ) = await h2o_drive_download(
                        q, q.client["dataset/import/h2o_drive_filename"]
                    )
                elif q.client["dataset/import/source"] == "Kaggle":
                    (
                        q.client["dataset/import/path"],
                        q.client["dataset/import/name"],
                    ) = await kaggle_download(
                        q,
                        q.client["dataset/import/kaggle_command"],
                        q.client["dataset/import/kaggle_access_key"],
                        q.client["dataset/import/kaggle_secret_key"],
                    )
                elif q.client["dataset/import/source"] == "Huggingface":
                    (
                        q.client["dataset/import/path"],
                        q.client["dataset/import/name"],
                    ) = await huggingface_download(
                        q,
                        q.client["dataset/import/huggingface_dataset"],
                        q.client["dataset/import/huggingface_split"],
                    )

            # store if in edit mode
            q.client["dataset/import/edit"] = edit

            # clear dataset triggers from client
            for trigger_key in default_cfg.dataset_trigger_keys:
                if q.client[f"dataset/import/cfg/{trigger_key}"]:
                    del q.client[f"dataset/import/cfg/{trigger_key}"]

            await dataset_import(
                q,
                step=3,
                edit=edit,
                error=error,
                warning=warning,
                allow_merge=allow_merge,
            )
        except Exception:
            logger.error("Dataset error:", exc_info=True)
            error = (
                "Dataset import failed. Please make sure all required "
                "fields are filled correctly."
            )
            await clean_dashboard(q, mode="full")
            await dataset_import(q, step=1, error=str(error))

    elif step == 3:  # set dataset configuration
        q.page["dataset/import/footer"] = ui.form_card(box="footer", items=[])
        try:
            if not q.args["dataset/import/cfg_file"] and not edit:
                q.client["dataset/import/name"] = get_unique_dataset_name(
                    q, q.client["dataset/import/name"]
                )
            q.page["dataset/import"] = ui.form_card(box="content", items=[])
            q.client.delete_cards.add("dataset/import")

            wizard = q.page["dataset/import"]

            title = "Configure dataset"

            items = [
                ui.text_l(title),
                ui.textbox(
                    name="dataset/import/name",
                    label="Dataset name",
                    value=q.client["dataset/import/name"],
                    required=True,
                    tooltip=tooltips["experiments_dataset_name"],
                ),
            ]

            choices_problem_types = [
                ui.choice(name, label) for name, label in get_problem_types()
            ]

            if q.client["dataset/import/cfg_file"] is None:
                max_substring_len = 0
                for c in choices_problem_types:
                    problem_type_name = c.name.replace("_config", "")
                    if problem_type_name in q.client["dataset/import/name"]:
                        if len(problem_type_name) > max_substring_len:
                            q.client["dataset/import/cfg_file"] = c.name
                            q.client["dataset/import/cfg_category"] = c.name.split("_")[
                                0
                            ]
                            max_substring_len = len(problem_type_name)
            if q.client["dataset/import/cfg_file"] is None:
                q.client["dataset/import/cfg_file"] = default_cfg.cfg_file
                q.client["dataset/import/cfg_category"] = q.client[  # type: ignore
                    "dataset/import/cfg_file"
                ].split("_")[0]

            # set default value of problem type if no match to category
            if (
                q.client["dataset/import/cfg_category"]
                not in q.client["dataset/import/cfg_file"]
            ):
                q.client["dataset/import/cfg_file"] = get_problem_types(
                    category=q.client["dataset/import/cfg_category"]
                )[0][0]

            items += [
                ui.dropdown(
                    name="dataset/import/cfg_file",
                    label="Problem Type",
                    required=True,
                    choices=choices_problem_types,
                    value=q.client["dataset/import/cfg_file"],
                    trigger=True,
                    tooltip=tooltips["experiments_problem_type"],
                )
            ]
            model_types = get_model_types(q.client["dataset/import/cfg_file"])
            if len(model_types) > 0:
                # add model type to cfg file name here
                q.client["dataset/import/cfg_file"] = add_model_type(
                    q.client["dataset/import/cfg_file"], model_types[0][0]
                )
            if not edit:
                q.client["dataset/import/cfg"] = load_config_py(
                    config_path=(
                        f"llm_studio/python_configs/"
                        f"{q.client['dataset/import/cfg_file']}"
                    ),
                    config_name="ConfigProblemBase",
                )

            option_items = get_dataset_elements(cfg=q.client["dataset/import/cfg"], q=q)
            items.extend(option_items)
            items.append(ui.message_bar(type="error", text=error))
            items.append(ui.message_bar(type="warning", text=warning))
            if file_extension_is_compatible(q):
                ui_nav_name = "dataset/import/4/edit" if edit else "dataset/import/4"
                buttons = [
                    ui.button(name=ui_nav_name, label="Continue", primary=True),
                    ui.button(name="dataset/list", label="Abort"),
                ]
                if allow_merge:
                    datasets_df = q.client.app_db.get_datasets_df()
                    if datasets_df.shape[0]:
                        label = "Merge With Existing Dataset"
                        buttons.insert(1, ui.button(name="dataset/merge", label=label))
            else:
                problem_type = make_label(
                    re.sub("_config.*", "", q.client["dataset/import/cfg_file"])
                )
                items += [
                    ui.text(
                        "<b> The chosen file extensions is not "
                        f"compatible with {problem_type}.</b> "
                    )
                ]
                buttons = [
                    ui.button(name="dataset/list", label="Abort"),
                ]
            q.page["dataset/import/footer"] = ui.form_card(
                box="footer", items=[ui.inline(items=buttons, justify="start")]
            )

            wizard.items = items

            q.client.delete_cards.add("dataset/import/footer")

        except Exception as exception:
            logger.error("Dataset error:", exc_info=True)
            error = clean_error(str(exception))
            await clean_dashboard(q, mode="full")
            await dataset_import(q, step=1, error=str(error))

    elif step == 31:  # activities after change in Parent ID columns
        logger.info("Step 31")
        cfg = q.client["dataset/import/cfg"]
        cfg = parse_ui_elements(
            cfg=cfg, q=q, limit=default_cfg.dataset_keys, pre="dataset/import/cfg/"
        )
        q.client["dataset/import/cfg"] = cfg
        await dataset_import(q, 3, edit=True)
    elif step == 4:  # verify if dataset does not exist already
        dataset_name = q.client["dataset/import/name"]
        original_name = q.client["dataset/import/original_name"]  # used in edit mode
        valid_dataset_name = get_unique_dataset_name(q, dataset_name)
        if valid_dataset_name != dataset_name and not (
            q.client["dataset/import/edit"] and dataset_name == original_name
        ):
            err = f"Dataset <strong>{dataset_name}</strong> already exists"
            q.client["dataset/import/name"] = valid_dataset_name
            await dataset_import(q, 3, edit=edit, error=err)
        else:
            await dataset_import(q, 5, edit=edit)

    elif step == 5:  # visualize dataset
        header = "<h2>Sample Data Visualization</h2>"
        valid_visualization = False
        continue_visible = True
        try:
            cfg = q.client["dataset/import/cfg"]
            cfg = parse_ui_elements(
                cfg=cfg, q=q, limit=default_cfg.dataset_keys, pre="dataset/import/cfg/"
            )

            q.client["dataset/import/cfg"] = cfg

            await busy_dialog(
                q=q,
                title="Performing sanity checks on the data",
                text="Please be patient...",
            )
            # add one-second delay for datasets where sanity check is instant
            # to avoid flickering dialog
            time.sleep(1)
            sanity_check(cfg)

            plot = cfg.logging.plots_class.plot_data(cfg)
            text = (
                "Data Validity Check. Click <strong>Continue</strong> if the input "
                "data and labels appear correctly."
            )
            if plot.encoding == "image":
                plot_item = ui.image(title="", type="png", image=plot.data)
            elif plot.encoding == "html":
                plot_item = ui.markup(content=plot.data)
            elif plot.encoding == "df":
                df = pd.read_parquet(plot.data)
                df = df.iloc[:2000]
                min_widths = {"Content": "800"}
                plot_item = ui_table_from_df(
                    q=q,
                    df=df,
                    name="experiment/display/table",
                    markdown_cells=list(df.select_dtypes(include=["object"]).columns),
                    searchables=list(df.columns),
                    downloadable=False,
                    resettable=False,
                    min_widths=min_widths,
                    height="calc(100vh - 267px)",
                    max_char_length=5_000,
                    cell_overflow="tooltip",
                )
            else:
                raise ValueError(f"Unknown plot encoding `{plot.encoding}`")

            items = [ui.markup(content=header), ui.message_bar(text=text), plot_item]
            valid_visualization = True

            # await busy_dialog(
            #     q=q,
            #     title="Performing sanity checks on the data",
            #     text="Please be patient...",
            # )
            # add one-second delay for datasets where sanity check is instant
            # to avoid flickering dialog
            # Move sanity check to the beginning of step 5
            # time.sleep(1)
            # sanity_check(cfg)

        except AssertionError as exception:
            logger.error(f"Error while validating data: {exception}", exc_info=True)
            # Wrap the exception text to limit the line length to 100 characters
            wrapped_exception_lines = textwrap.fill(
                str(exception), width=100
            ).splitlines()

            # Join the wrapped exception lines with an extra newline to separate each
            wrapped_exception = "\n".join(wrapped_exception_lines)
            text = (
                "# Error while validating data\n"
                "Please review the error message below \n"
                "\n"
                "**Details of the Validation Error**:\n"
                "\n"
                f"{wrapped_exception}"
                "\n"
            )

            items = [
                ui.markup(content=header),
                ui.message_bar(text=text, type="error"),
                ui.expander(
                    name="expander",
                    label="Expand Error Traceback",
                    items=[ui.markup(f"<pre>{traceback.format_exc()}</pre>")],
                ),
            ]
            continue_visible = False
        except Exception as exception:
            logger.error(
                f"Error while plotting data preview: {exception}", exc_info=True
            )
            text = (
                "Error occurred while visualizing the data. Please go back and verify "
                "whether the problem type and other settings were set properly."
            )
            items = [
                ui.markup(content=header),
                ui.message_bar(text=text, type="error"),
                ui.expander(
                    name="expander",
                    label="Expand Error Traceback",
                    items=[ui.markup(f"<pre>{traceback.format_exc()}</pre>")],
                ),
            ]
            continue_visible = False

        buttons = [
            ui.button(
                name="dataset/import/6",
                label="Continue",
                primary=valid_visualization,
                visible=continue_visible,
            ),
            ui.button(
                name="dataset/import/3/edit",
                label="Back",
                primary=not valid_visualization,
            ),
            ui.button(name="dataset/list", label="Abort"),
        ]

        q.page["dataset/import"] = ui.form_card(box="content", items=items)
        q.client.delete_cards.add("dataset/import")

        q.page["dataset/import/footer"] = ui.form_card(
            box="footer", items=[ui.inline(items=buttons, justify="start")]
        )
        q.client.delete_cards.add("dataset/import/footer")

    elif step == 6:  # create dataset
        if q.client["dataset/import/name"] == "":
            await clean_dashboard(q, mode="full")
            await dataset_import(q, step=2, error="Please enter all required fields!")

        else:
            folder_name = q.client["dataset/import/path"].split("/")[-1]
            new_folder = q.client["dataset/import/name"]
            act_path = q.client["dataset/import/path"]
            new_path = new_folder.join(act_path.rsplit(folder_name, 1))

            try:
                shutil.move(q.client["dataset/import/path"], new_path)

                cfg = q.client["dataset/import/cfg"]

                # remap old path to new path
                for k in default_cfg.dataset_folder_keys:
                    old_path = getattr(cfg.dataset, k, None)
                    if old_path is not None:
                        setattr(
                            cfg.dataset,
                            k,
                            old_path.replace(q.client["dataset/import/path"], new_path),
                        )

                # change the default validation strategy if validation df set
                if cfg.dataset.validation_dataframe != "None":
                    cfg.dataset.validation_strategy = "custom"
                cfg_path = f"{new_path}/{q.client['dataset/import/cfg_file']}.yaml"
                save_config_yaml(cfg_path, cfg)

                train_rows = None
                if os.path.exists(cfg.dataset.train_dataframe):
                    train_rows = read_dataframe_drop_missing_labels(
                        cfg.dataset.train_dataframe, cfg
                    ).shape[0]
                validation_rows = None
                if os.path.exists(cfg.dataset.validation_dataframe):
                    validation_rows = read_dataframe_drop_missing_labels(
                        cfg.dataset.validation_dataframe, cfg
                    ).shape[0]

                dataset = Dataset(
                    id=q.client["dataset/import/id"],
                    name=q.client["dataset/import/name"],
                    path=new_path,
                    config_file=cfg_path,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                )
                if q.client["dataset/import/id"] is not None:
                    q.client.app_db.delete_dataset(dataset.id)
                q.client.app_db.add_dataset(dataset)
                await dataset_list(q)

            except Exception as exception:
                logger.error("Dataset error:", exc_info=True)
                q.client.app_db._session.rollback()
                error = clean_error(str(exception))
                await clean_dashboard(q, mode="full")
                await dataset_import(q, step=2, error=str(error))


async def dataset_merge(q: Q, step, error=""):
    if step == 1:  # Select which dataset to merge
        await clean_dashboard(q, mode="full")
        q.client["nav/active"] = "dataset/merge"

        q.page["dataset/merge"] = ui.form_card(box="content", items=[])
        q.client.delete_cards.add("dataset/merge")

        datasets_df = q.client.app_db.get_datasets_df()
        import_choices = [
            ui.choice(x["path"], x["name"]) for idx, x in datasets_df.iterrows()
        ]

        items = [
            ui.text_l("Merge current dataset with an existing dataset"),
            ui.dropdown(
                name="dataset/merge/target",
                label="Dataset",
                value=datasets_df.iloc[0]["path"],
                choices=import_choices,
                trigger=False,
                tooltip="Source of dataset import",
            ),
        ]

        if error:
            items.append(ui.message_bar(type="error", text=error))

        q.page["dataset/merge"].items = items

        buttons = [
            ui.button(name="dataset/merge/action", label="Merge", primary=True),
            ui.button(name="dataset/import/3", label="Back", primary=False),
            ui.button(name="dataset/list", label="Abort"),
        ]

        q.page["dataset/import/footer"] = ui.form_card(
            box="footer", items=[ui.inline(items=buttons, justify="start")]
        )
        q.client.delete_cards.add("dataset/import/footer")

    elif step == 2:  # copy file to dataset and go to edit dataset
        current_dir = q.client["dataset/import/path"]
        target_dir = q.args["dataset/merge/target"]

        if current_dir == target_dir:
            await dataset_merge(q, step=1, error="Cannot merge dataset with itself")
            return

        datasets_df = q.client.app_db.get_datasets_df().set_index("path")
        has_dataset_entry = current_dir in datasets_df.index

        if has_dataset_entry:
            experiment_df = q.client.app_db.get_experiments_df()
            source_id = int(datasets_df.loc[current_dir, "id"])
            has_experiment = any(experiment_df["dataset"].astype(int) == source_id)
        else:
            source_id = None
            has_experiment = False

        current_files = os.listdir(current_dir)
        current_files = [x for x in current_files if not x.endswith(".yaml")]
        target_files = os.listdir(target_dir)
        overlapping_files = list(set(current_files).intersection(set(target_files)))
        rename_map = {}

        for file in overlapping_files:
            tmp_str = file.split(".")
            if len(tmp_str) == 1:
                file_name, extension = file, ""
            else:
                file_name, extension = ".".join(tmp_str[:-1]), f".{tmp_str[-1]}"

            cnt = 1
            while f"{file_name}_{cnt}{extension}" in target_files:
                cnt += 1

            rename_map[file] = f"{file_name}_{cnt}{extension}"
            target_files.append(rename_map[file])

        if len(overlapping_files):
            warning = (
                f"Renamed {', '.join(rename_map.keys())} to "
                f"{', '.join(rename_map.values())} due to duplicated entries."
            )
        else:
            warning = ""

        for file in current_files:
            new_file = rename_map.get(file, file)
            src = os.path.join(current_dir, file)
            dst = os.path.join(target_dir, new_file)

            if has_experiment:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy(src, dst)
            else:
                shutil.move(src, dst)

        if not has_experiment:
            shutil.rmtree(current_dir)
            if has_dataset_entry:
                q.client.app_db.delete_dataset(source_id)

        dataset_id = int(datasets_df.loc[target_dir, "id"])
        await dataset_edit(q, dataset_id, warning=warning, allow_merge=False)


async def dataset_list_table(
    q: Q,
    show_experiment_datasets: bool = True,
) -> None:
    """Pepare dataset list form card

    Args:
        q: Q
        show_experiment_datasets: whether to also show datasets linked to experiments
    """

    q.client["dataset/list/df_datasets"] = get_datasets(
        q=q,
        show_experiment_datasets=show_experiment_datasets,
    )

    df_viz = q.client["dataset/list/df_datasets"].copy()

    columns_to_drop = [
        "id",
        "path",
        "config_file",
        "validation dataframe",
    ]

    df_viz = df_viz.drop(columns=columns_to_drop, errors="ignore")
    if "problem type" in df_viz.columns:
        df_viz["problem type"] = df_viz["problem type"].str.replace("Text ", "")

    widths = {
        "name": "200",
        "problem type": "210",
        "train dataframe": "190",
        "train rows": "120",
        "validation rows": "130",
        "labels": "120",
        "actions": "5",
    }

    actions_dict = {
        "dataset/newexperiment": "New experiment",
        "dataset/edit": "Edit dataset",
        "dataset/delete/dialog/single": "Delete dataset",
    }

    q.page["dataset/list"] = ui.form_card(
        box="content",
        items=[
            ui_table_from_df(
                q=q,
                df=df_viz,
                name="dataset/list/table",
                sortables=["train rows", "validation rows"],
                filterables=["name", "problem type"],
                searchables=[],
                min_widths=widths,
                link_col="name",
                height="calc(100vh - 267px)",
                actions=actions_dict,
            ),
            ui.message_bar(type="info", text=""),
        ],
    )
    q.client.delete_cards.add("dataset/list")


async def dataset_list(q: Q, reset: bool = True) -> None:
    """Display all datasets."""
    q.client["nav/active"] = "dataset/list"

    if reset:
        await clean_dashboard(q, mode="full")
        await dataset_list_table(q)

    q.page["dataset/display/footer"] = ui.form_card(
        box="footer",
        items=[
            ui.inline(
                items=[
                    ui.button(
                        name="dataset/import", label="Import dataset", primary=True
                    ),
                    ui.button(
                        name="dataset/list/delete",
                        label="Delete datasets",
                        primary=False,
                    ),
                ],
                justify="start",
            )
        ],
    )
    q.client.delete_cards.add("dataset/display/footer")
    remove_temp_files(q)

    await q.page.save()


async def dataset_newexperiment(q: Q, dataset_id: int):
    """Start a new experiment from given dataset."""

    dataset = q.client.app_db.get_dataset(dataset_id)

    q.client["experiment/start/cfg_file"] = dataset.config_file.split("/")[-1].replace(
        ".yaml", ""
    )
    q.client["experiment/start/cfg_category"] = q.client[
        "experiment/start/cfg_file"
    ].split("_")[0]
    q.client["experiment/start/dataset"] = str(dataset_id)

    await experiment_start(q)


async def dataset_edit(
    q: Q, dataset_id: int, error: str = "", warning: str = "", allow_merge: bool = True
):
    """Edit selected dataset.

    Args:
        q: Q
        dataset_id: dataset id to edit
        error: optional error message
        warning: optional warning message
        allow_merge: whether to allow merging dataset when editing
    """

    dataset = q.client.app_db.get_dataset(dataset_id)

    experiments_df = q.client.app_db.get_experiments_df()
    experiments_df = experiments_df[experiments_df["dataset"] == str(dataset_id)]
    statuses, _ = get_experiments_status(experiments_df)
    num_invalid = len([stat for stat in statuses if stat in ["running", "queued"]])

    if num_invalid:
        info = "s" if num_invalid > 1 else ""
        info_str = (
            f"Dataset <strong>{dataset.name}</strong> is linked to {num_invalid} "
            f"running or queued experiment{info}. Wait for them to finish or stop them "
            "first before editing the dataset."
        )
        q.page["dataset/list"].items[1].message_bar.text = info_str
        return

    q.client["dataset/import/id"] = dataset_id

    q.client["dataset/import/cfg_file"] = dataset.config_file.split("/")[-1].replace(
        ".yaml", ""
    )
    q.client["dataset/import/cfg_category"] = q.client["dataset/import/cfg_file"].split(
        "_"
    )[0]
    q.client["dataset/import/path"] = dataset.path
    q.client["dataset/import/name"] = dataset.name
    q.client["dataset/import/original_name"] = dataset.name
    q.client["dataset/import/cfg"] = load_config_yaml(dataset.config_file)

    if allow_merge and experiments_df.shape[0]:
        allow_merge = False

    await dataset_import(
        q=q, step=2, edit=True, error=error, warning=warning, allow_merge=allow_merge
    )


async def dataset_list_delete(q: Q):
    """Allow to select multiple datasets for deletion."""

    await dataset_list_table(q, show_experiment_datasets=False)

    q.page["dataset/list"].items[0].table.multiple = True

    info_str = "Only datasets not linked to experiments can be deleted."

    q.page["dataset/list"].items[1].message_bar.text = info_str

    q.page["dataset/display/footer"].items = [
        ui.inline(
            items=[
                ui.button(
                    name="dataset/delete/dialog", label="Delete datasets", primary=True
                ),
                ui.button(name="dataset/list/delete/abort", label="Abort"),
            ]
        )
    ]


async def dataset_delete(q: Q, dataset_ids: List[int]):
    """Delete selected datasets.

    Args:
        q: Q
        dataset_ids: list of dataset ids to delete
    """

    for dataset_id in dataset_ids:
        dataset = q.client.app_db.get_dataset(dataset_id)
        q.client.app_db.delete_dataset(dataset.id)

        try:
            shutil.rmtree(dataset.path)
        except OSError:
            pass


async def dataset_delete_single(q: Q, dataset_id: int):
    dataset = q.client.app_db.get_dataset(dataset_id)

    experiments_df = q.client.app_db.get_experiments_df()
    num_experiments = sum(experiments_df["dataset"] == str(dataset_id))
    if num_experiments:
        info = "s" if num_experiments > 1 else ""
        info_str = (
            f"Dataset <strong>{dataset.name}</strong> is linked to {num_experiments} "
            f"experiment{info}. Only datasets not linked to experiments can be deleted."
        )
        await dataset_list(q)
        q.page["dataset/list"].items[1].message_bar.text = info_str
    else:
        await dataset_delete(q, [dataset_id])
        await dataset_list(q)


async def dataset_display(q: Q) -> None:
    """Display a selected dataset."""

    dataset_id = q.client["dataset/list/df_datasets"]["id"].iloc[
        q.client["dataset/display/id"]
    ]
    dataset: Dataset = q.client.app_db.get_dataset(dataset_id)
    config_filename = dataset.config_file
    cfg = load_config_yaml(config_filename)
    dataset_filename = cfg.dataset.train_dataframe

    if (
        q.client["dataset/display/tab"] is None
        or q.args["dataset/display/data"] is not None
    ):
        q.client["dataset/display/tab"] = "dataset/display/data"

    if q.args["dataset/display/visualization"] is not None:
        q.client["dataset/display/tab"] = "dataset/display/visualization"

    if q.args["dataset/display/statistics"] is not None:
        q.client["dataset/display/tab"] = "dataset/display/statistics"

    if q.args["dataset/display/summary"] is not None:
        q.client["dataset/display/tab"] = "dataset/display/summary"

    await clean_dashboard(q, mode=q.client["dataset/display/tab"])

    items: List[Tab] = [
        ui.tab(name="dataset/display/data", label="Sample Train Data"),
        ui.tab(
            name="dataset/display/visualization", label="Sample Train Visualization"
        ),
        ui.tab(name="dataset/display/statistics", label="Train Data Statistics"),
        ui.tab(name="dataset/display/summary", label="Summary"),
    ]

    q.page["dataset/display/tab"] = ui.tab_card(
        box="nav2",
        link=True,
        items=items,
        value=q.client["dataset/display/tab"],
    )
    q.client.delete_cards.add("dataset/display/tab")

    if q.client["dataset/display/tab"] == "dataset/display/data":
        await show_data_tab(q=q, cfg=cfg, filename=dataset_filename)

    elif q.client["dataset/display/tab"] == "dataset/display/visualization":
        await show_visualization_tab(q, cfg)

    elif q.client["dataset/display/tab"] == "dataset/display/statistics":
        await show_statistics_tab(
            q, dataset_filename=dataset_filename, config_filename=config_filename
        )

    elif q.client["dataset/display/tab"] == "dataset/display/summary":
        await show_summary_tab(q, dataset_id)

    q.page["dataset/display/footer"] = ui.form_card(
        box="footer",
        items=[
            ui.inline(
                items=[
                    ui.button(
                        name="dataset/newexperiment/from_current",
                        label="Create experiment",
                        primary=False,
                        disabled=False,
                        tooltip=None,
                    ),
                    ui.button(name="dataset/list", label="Back", primary=False),
                ],
                justify="start",
            )
        ],
    )
    q.client.delete_cards.add("dataset/display/footer")


async def show_data_tab(q: Q, cfg, filename: str):
    fill_columns = get_fill_columns(cfg)
    df = read_dataframe(filename, n_rows=200, fill_columns=fill_columns)
    q.page["dataset/display/data"] = ui.form_card(
        box="first",
        items=[
            ui_table_from_df(
                q=q,
                df=df,
                name="dataset/display/data/table",
                sortables=list(df.columns),
                markdown_cells=None,  # render all cells as raw text
                height="calc(100vh - 267px)",
                cell_overflow="wrap",
            )
        ],
    )
    q.client.delete_cards.add("dataset/display/data")


async def show_visualization_tab(q: Q, cfg):
    try:
        plot = cfg.logging.plots_class.plot_data(cfg)
    except Exception as error:
        logger.error(f"Error while plotting data preview: {error}", exc_info=True)
        plot = PlotData("<h2>Error while plotting data.</h2>", encoding="html")
    card: ImageCard | MarkupCard | FormCard
    if plot.encoding == "image":
        card = ui.image_card(box="first", title="", type="png", image=plot.data)
    elif plot.encoding == "html":
        card = ui.markup_card(box="first", title="", content=plot.data)
    elif plot.encoding == "df":
        df = pd.read_parquet(plot.data)
        df = df.iloc[:2000]
        min_widths = {"Content": "800"}
        card = ui.form_card(
            box="first",
            items=[
                ui_table_from_df(
                    q=q,
                    df=df,
                    name="dataset/display/visualization/table",
                    markdown_cells=list(df.select_dtypes(include=["object"]).columns),
                    searchables=list(df.columns),
                    downloadable=True,
                    resettable=True,
                    min_widths=min_widths,
                    height="calc(100vh - 267px)",
                    max_char_length=50_000,
                    cell_overflow="tooltip",
                )
            ],
        )

    else:
        raise ValueError(f"Unknown plot encoding `{plot.encoding}`")
    q.page["dataset/display/visualization"] = card
    q.client.delete_cards.add("dataset/display/visualization")


async def show_summary_tab(q: Q, dataset_id):
    dataset_df = get_datasets(q)
    dataset_df = dataset_df[dataset_df.id == dataset_id]
    stat_list_items: List[StatListItem] = []
    for col in dataset_df.columns:
        if col in ["id", "config_file", "path", "process_id", "status"]:
            continue
        v = dataset_df[col].values[0]
        t: StatListItem = ui.stat_list_item(label=make_label(col), value=str(v))

        stat_list_items.append(t)
    q.page["dataset/display/summary"] = ui.stat_list_card(
        box="first", items=stat_list_items, title=""
    )
    q.client.delete_cards.add("dataset/display/summary")


async def show_statistics_tab(q: Q, dataset_filename, config_filename):
    cfg_hash = hashlib.md5(open(config_filename, "rb").read()).hexdigest()
    stats_dict = compute_dataset_statistics(dataset_filename, config_filename, cfg_hash)

    for chat_type in ["prompts", "answers"]:
        q.page[f"dataset/display/statistics/{chat_type}_histogram"] = histogram_card(
            x=stats_dict[chat_type],
            x_axis_description=f"text_length_{chat_type.capitalize()}",
            title=f"Text Length Distribution for {chat_type.capitalize()}"
            f" (split by whitespace)",
            histogram_box="first",
        )
        q.client.delete_cards.add(f"dataset/display/statistics/{chat_type}_histogram")

    q.page["dataset/display/statistics/full_conversation_histogram"] = histogram_card(
        x=stats_dict["complete_conversations"],
        x_axis_description="text_length_complete_conversations",
        title="Text Length Distribution for complete "
        "conversations (split by whitespace)",
        histogram_box="second",
    )
    q.client.delete_cards.add("dataset/display/statistics/full_conversation_histogram")

    if len(set(stats_dict["number_of_prompts"])) > 1:
        q.page["dataset/display/statistics/parent_id_length_histogram"] = (
            histogram_card(
                x=stats_dict["number_of_prompts"],
                x_axis_description="number_of_prompts",
                title="Distribution of number of prompt-answer turns per conversation.",
                histogram_box="second",
            )
        )
        q.client.delete_cards.add(
            "dataset/display/statistics/parent_id_length_histogram"
        )

    df_stats = stats_dict["df_stats"]
    if df_stats is None:
        component_items = [
            ui.text(
                "Dataset does not contain numerical or text features. "
                "No statistics available."
            )
        ]
    else:
        if df_stats.shape[1] > 5:  # mixed text and numeric
            widths = {col: "77" for col in df_stats}
        else:  # only text features
            widths = None
        component_items = [
            ui_table_from_df(
                q=q,
                df=df_stats,
                name="dataset/display/statistics/table",
                sortables=list(df_stats.columns),
                min_widths=widths,
                height="265px",
            )
        ]
    q.page["dataset/display/statistics"] = ui.form_card(
        box="third",
        items=component_items,
    )
    q.client.delete_cards.add("dataset/display/statistics")


@functools.lru_cache()
def compute_dataset_statistics(dataset_path: str, cfg_path: str, cfg_hash: str) -> dict:
    """
    Compute various statistics for a dataset.
    - text length distribution for prompts and answers
    - text length distribution for complete conversations
    - distribution of number of prompt-answer turns per conversation
    - statistics for non text features

    We use LRU caching to avoid recomputing the statistics for the same dataset.
    Thus, cfg_hash is used as a function argument to identify the dataset.
    """
    df_train = read_dataframe(dataset_path)
    cfg = load_config_yaml(cfg_path)
    conversations = get_conversation_chains(
        df=df_train, cfg=cfg, limit_chained_samples=True
    )
    stats_dict = {}
    for chat_type in ["prompts", "answers"]:
        text_lengths = [
            [len(text.split(" ")) for text in conversation[chat_type]]
            for conversation in conversations
        ]
        text_lengths = [item for sublist in text_lengths for item in sublist]
        stats_dict[chat_type] = text_lengths
    input_texts = []
    for conversation in conversations:
        input_text = conversation["systems"][0]
        prompts = conversation["prompts"]
        answers = conversation["answers"]
        for prompt, answer in zip(prompts, answers):
            input_text += prompt + answer
        input_texts += [input_text]
    stats_dict["complete_conversations"] = [
        len(text.split(" ")) for text in input_texts
    ]
    stats_dict["number_of_prompts"] = [
        len(conversation["prompts"]) for conversation in conversations
    ]
    stats_dict["df_stats"] = get_frame_stats(df_train)
    return stats_dict


async def dataset_import_uploaded_file(q: Q) -> None:
    local_path = await q.site.download(
        q.args["dataset/import/local_upload"][0],
        f"{get_data_dir(q)}/"
        f'{q.args["dataset/import/local_upload"][0].split("/")[-1]}',
    )
    await q.site.unload(q.args["dataset/import/local_upload"][0])
    valid, error = check_valid_upload_content(local_path)
    if valid:
        q.args["dataset/import/local_path"] = local_path
        q.client["dataset/import/local_path"] = q.args["dataset/import/local_path"]
        await dataset_import(q, step=2)
    else:
        await dataset_import(q, step=1, error=error)


async def dataset_delete_current_datasets(q: Q) -> None:
    dataset_ids = list(
        q.client["dataset/list/df_datasets"]["id"].iloc[
            list(map(int, q.client["dataset/list/table"]))
        ]
    )
    await dataset_delete(q, dataset_ids)
    await dataset_list(q)
