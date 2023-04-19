import logging
import os
import re
import shutil
import time
import traceback
from typing import List, Optional

from h2o_wave import Q, ui

from app_utils.config import default_cfg
from app_utils.db import Dataset
from app_utils.sections.experiment import experiment_start
from app_utils.utils import (
    add_model_type,
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
    kaggle_download,
    local_download,
    parse_ui_elements,
    remove_temp_files,
    s3_download,
    s3_file_options,
)
from app_utils.wave_utils import ui_table_from_df
from llm_studio.src.utils.config_utils import load_config, make_label
from llm_studio.src.utils.data_utils import (
    get_fill_columns,
    read_dataframe,
    read_dataframe_drop_missing_labels,
    sanity_check,
)
from llm_studio.src.utils.utils import load_config_yaml, save_config_yaml
from .common import clean_dashboard

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
            ui.choice("Kaggle", "Kaggle"),
        ]

        items = [
            ui.text_l("Import dataset"),
            ui.dropdown(
                name="dataset/import/source",
                label="Source",
                value="Upload"
                if q.client["dataset/import/source"] is None
                else q.client["dataset/import/source"],
                choices=import_choices,
                trigger=True,
                tooltip="Source of dataset import",
            ),
        ]

        if (
            q.client["dataset/import/source"] is None
            or q.client["dataset/import/source"] == "S3"
        ):
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

            files = s3_file_options(
                q.client["dataset/import/s3_bucket"],
                q.client["dataset/import/s3_access_key"],
                q.client["dataset/import/s3_secret_key"],
            )

            if not files:
                ui_filename = ui.textbox(
                    name="dataset/import/s3_filename",
                    label="File name",
                    value="",
                    required=True,
                    tooltip="File name to be imported",
                )
            else:
                if default_cfg.s3_filename in files:
                    default_file = default_cfg.s3_filename
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

            if q.args["dataset/import/local_path_list"]:
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
                    value=default_cfg.kaggle_command,
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
            box="footer", items=[ui.inline(items=buttons)]
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
                elif q.client["dataset/import/source"] in ("Upload", "Local"):
                    (
                        q.client["dataset/import/path"],
                        q.client["dataset/import/name"],
                    ) = await local_download(q, q.client["dataset/import/local_path"])
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

            model_types = get_model_types(q.client["dataset/import/cfg_file"])
            if len(model_types) > 0:
                # add model type to cfg file name here
                q.client["dataset/import/cfg_file"] = add_model_type(
                    q.client["dataset/import/cfg_file"], model_types[0][0]
                )
            if not edit:
                q.client["dataset/import/cfg"] = load_config(
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
                box="footer", items=[ui.inline(items=buttons)]
            )

            wizard.items = items

            q.client.delete_cards.add("dataset/import/footer")

        except Exception as exception:
            logger.error("Dataset error:", exc_info=True)
            error = clean_error(str(exception))
            await clean_dashboard(q, mode="full")
            await dataset_import(q, step=1, error=str(error))

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
        try:
            cfg = q.client["dataset/import/cfg"]
            cfg = parse_ui_elements(
                cfg=cfg, q=q, limit=default_cfg.dataset_keys, pre="dataset/import/cfg/"
            )

            q.client["dataset/import/cfg"] = cfg
            plot = cfg.logging.plots_class.plot_data(cfg)
            text = (
                "Data Validity Check. Click <strong>Continue</strong> if the input "
                "data and labels appear correctly."
            )
            if plot.encoding == "png":
                plot_item = ui.image(title="", type="png", image=plot.data)
            elif plot.encoding == "html":
                plot_item = ui.markup(content=plot.data)
            else:
                raise ValueError(f"Unknown plot encoding `{plot.encoding}`")

            items = [ui.markup(content=header), ui.message_bar(text=text), plot_item]
            valid_visualization = True

            q.page["meta"].dialog = ui.dialog(
                title="Performing sanity checks on the data",
                blocking=True,
                items=[ui.progress(label="Please be patient...")],
            )
            await q.page.save()
            # add one-second delay for datasets where sanity check is instant
            # to avoid flickering dialog
            time.sleep(1)
            sanity_check(cfg)

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

        buttons = [
            ui.button(
                name="dataset/import/6", label="Continue", primary=valid_visualization
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
            box="footer", items=[ui.inline(items=buttons)]
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
                save_config_yaml(f"{new_path}/{q.client['dataset/import/cfg_file']}.yaml", cfg)

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
                    config_file=f"{new_path}/{q.client['dataset/import/cfg_file']}.p",
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
            box="footer", items=[ui.inline(items=buttons)]
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
        current_files = [x for x in current_files if not x.endswith(".p")]
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
                searchables=[],
                filterables=["name", "problem type"],
                min_widths=widths,
                link_col="name",
                height="calc(100vh - 245px)",
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
                ]
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
        ".p", ""
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
        ".p", ""
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
    dataset = q.client.app_db.get_dataset(dataset_id)
    cfg = load_config_yaml(dataset.config_file)

    has_train_df = cfg.dataset.train_dataframe != "None"

    dataset = cfg.dataset.__dict__

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

    data_string = "Train" if has_train_df else "Test"

    items = [
        ui.tab(name="dataset/display/data", label=f"Sample {data_string} Data"),
        ui.tab(
            name="dataset/display/statistics", label=f"{data_string} Data Statistics"
        ),
        ui.tab(name="dataset/display/summary", label="Summary"),
    ]

    if has_train_df:
        items.insert(
            1,
            ui.tab(
                name="dataset/display/visualization", label="Sample Train Visualization"
            ),
        )

    q.page["dataset/display/tab"] = ui.tab_card(
        box="nav2",
        link=True,
        items=items,
        value=q.client["dataset/display/tab"],
    )
    q.client.delete_cards.add("dataset/display/tab")

    if q.client["dataset/display/tab"] == "dataset/display/data":
        fill_columns = get_fill_columns(cfg)
        df = read_dataframe(
            dataset["train_dataframe"], n_rows=200, fill_columns=fill_columns
        )

        q.page["dataset/display/data"] = ui.form_card(
            box="first",
            items=[
                ui_table_from_df(
                    q=q,
                    df=df,
                    name="dataset/display/data/table",
                    sortables=list(df.columns),
                    height="calc(100vh - 265px)",
                )
            ],
        )
        q.client.delete_cards.add("dataset/display/data")

    elif q.client["dataset/display/tab"] == "dataset/display/visualization":

        try:
            plot = cfg.logging.plots_class.plot_data(cfg)
        except Exception as error:
            logger.error(f"Error while plotting data preview: {error}", exc_info=True)
            plot = cfg.logging.plots_class.plot_empty(
                cfg, error="Error while plotting data."
            )

        if plot.encoding == "png":
            card = ui.image_card(box="first", title="", type="png", image=plot.data)
        elif plot.encoding == "html":
            card = ui.markup_card(box="first", title="", content=plot.data)
        else:
            raise ValueError(f"Unknown plot encoding `{plot.encoding}`")

        q.page["dataset/display/visualization"] = card
        q.client.delete_cards.add("dataset/display/visualization")

    elif q.client["dataset/display/tab"] == "dataset/display/statistics":
        stats = get_frame_stats(read_dataframe(dataset["train_dataframe"]))
        if stats is None:
            items = [
                ui.text(
                    "Dataset does not contain numerical or text features. "
                    "No statistics available."
                )
            ]
        else:
            if stats.shape[1] > 5:  # mixed text and numeric
                widths = {col: "77" for col in stats}
            else:  # only text features
                widths = None
            items = [
                ui_table_from_df(
                    q=q,
                    df=stats,
                    name="dataset/display/statistics/table",
                    sortables=list(stats.columns),
                    height="calc(100vh - 265px)",
                    min_widths=widths,
                )
            ]
        q.page["dataset/display/statistics"] = ui.form_card(box="first", items=items)
        q.client.delete_cards.add("dataset/display/statistics")

    elif q.client["dataset/display/tab"] == "dataset/display/summary":

        dataset_df = get_datasets(q)
        dataset_df = dataset_df[dataset_df.id == dataset_id]

        items = []

        for col in dataset_df.columns:
            if col in ["id", "config_file", "path", "process_id", "status"]:
                continue
            v = dataset_df[col].values[0]
            t = ui.stat_list_item(label=make_label(col), value=str(v))

            items.append(t)

        q.page["dataset/display/summary"] = ui.stat_list_card(
            box="first", items=items, title=""
        )

        q.client.delete_cards.add("dataset/display/summary")

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
                ]
            )
        ],
    )
    q.client.delete_cards.add("dataset/display/footer")


async def dataset_import_uploaded_file(q: Q):
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


async def dataset_delete_current_datasets(q: Q):
    dataset_ids = list(
        q.client["dataset/list/df_datasets"]["id"].iloc[
            list(map(int, q.client["dataset/list/table"]))
        ]
    )
    await dataset_delete(q, dataset_ids)
    await dataset_list(q)
