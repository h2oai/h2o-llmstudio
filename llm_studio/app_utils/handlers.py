import gc
import logging
from typing import List

import torch
from h2o_wave import Q

from llm_studio.app_utils.sections.chat import chat_tab
from llm_studio.app_utils.sections.chat_update import chat_update
from llm_studio.app_utils.sections.common import delete_dialog
from llm_studio.app_utils.sections.dataset import (
    dataset_delete_current_datasets,
    dataset_delete_single,
    dataset_display,
    dataset_edit,
    dataset_import,
    dataset_import_uploaded_file,
    dataset_list,
    dataset_list_delete,
    dataset_merge,
    dataset_newexperiment,
)
from llm_studio.app_utils.sections.experiment import (
    experiment_delete,
    experiment_display,
    experiment_download_adapter,
    experiment_download_logs,
    experiment_download_model,
    experiment_download_predictions,
    experiment_list,
    experiment_push_to_huggingface_dialog,
    experiment_rename_ui_workflow,
    experiment_run,
    experiment_start,
    experiment_stop,
)
from llm_studio.app_utils.sections.home import home
from llm_studio.app_utils.sections.project import (
    current_experiment_compare,
    current_experiment_list_compare,
    current_experiment_list_delete,
    current_experiment_list_stop,
    experiment_rename_action_workflow,
    list_current_experiments,
)
from llm_studio.app_utils.sections.settings import settings
from llm_studio.app_utils.setting_utils import (
    load_default_user_settings,
    load_user_settings_and_secrets,
    save_user_settings_and_secrets,
)
from llm_studio.app_utils.utils import add_model_type
from llm_studio.app_utils.wave_utils import report_error, wave_utils_handle_error

logger = logging.getLogger(__name__)


async def handle(q: Q) -> None:
    """Handles all requests in application and calls according functions."""

    # logger.info(f"args: {q.args}")
    # logger.info(f"events: {q.events}")

    if not (
        q.args.__wave_submission_name__ == "experiment/display/chat/chatbot"
        or q.args.__wave_submission_name__ == "experiment/display/chat/clear_history"
    ):
        if "experiment/display/chat/cfg" in q.client:
            del q.client["experiment/display/chat/cfg"]
        if "experiment/display/chat/model" in q.client:
            del q.client["experiment/display/chat/model"]
        if "experiment/display/chat/tokenizer" in q.client:
            del q.client["experiment/display/chat/tokenizer"]
        torch.cuda.empty_cache()
        gc.collect()

    try:
        if q.args.__wave_submission_name__ == "home":
            await home(q)
        elif q.args.__wave_submission_name__ == "settings":
            await settings(q)
        elif q.args.__wave_submission_name__ == "save_settings":
            logger.info("Saving user settings")
            await save_user_settings_and_secrets(q)
            await settings(q)
        elif q.args.__wave_submission_name__ == "load_settings":
            load_user_settings_and_secrets(q)
            await settings(q)
        elif q.args.__wave_submission_name__ == "restore_default_settings":
            load_default_user_settings(q)
            await settings(q)

        elif q.args.__wave_submission_name__ == "report_error":
            await report_error(q)

        elif q.args.__wave_submission_name__ == "dataset/import":
            await dataset_import(q, step=1)
        elif q.args.__wave_submission_name__ == "dataset/list":
            await dataset_list(q)
        elif q.args.__wave_submission_name__ == "dataset/list/delete/abort":
            q.page["dataset/list"].items[0].table.multiple = False
            await dataset_list(q, reset=True)
        elif q.args.__wave_submission_name__ == "dataset/list/abort":
            q.page["dataset/list"].items[0].table.multiple = False
            await dataset_list(q, reset=True)
        elif q.args.__wave_submission_name__ == "dataset/list/delete":
            await dataset_list_delete(q)
        elif q.args.__wave_submission_name__ == "dataset/delete/single":
            dataset_id = q.client["dataset/delete/single/id"]
            dataset_id = q.client["dataset/list/df_datasets"]["id"].iloc[dataset_id]
            await dataset_delete_single(q, int(dataset_id))
        elif q.args.__wave_submission_name__ == "dataset/delete/dialog/single":
            dataset_id = int(q.args["dataset/delete/dialog/single"])
            q.client["dataset/delete/single/id"] = dataset_id
            name = q.client["dataset/list/df_datasets"]["name"].iloc[dataset_id]

            if q.client["delete_dialogs"]:
                await delete_dialog(q, [name], "dataset/delete/single", "dataset")
            else:
                dataset_id = q.client["dataset/list/df_datasets"]["id"].iloc[dataset_id]
                await dataset_delete_single(q, int(dataset_id))

        elif q.args["dataset/delete/dialog"]:
            names = list(
                q.client["dataset/list/df_datasets"]["name"].iloc[
                    list(map(int, q.client["dataset/list/table"]))
                ]
            )

            if not names:
                return

            if q.client["delete_dialogs"]:
                await delete_dialog(q, names, "dataset/delete", "dataset")
            else:
                await dataset_delete_current_datasets(q)

        elif q.args.__wave_submission_name__ == "dataset/delete":
            await dataset_delete_current_datasets(q)
        elif q.args.__wave_submission_name__ == "dataset/edit":
            if q.client["dataset/list/df_datasets"] is not None:
                dataset_id = int(q.args["dataset/edit"])
                dataset_id = q.client["dataset/list/df_datasets"]["id"].iloc[dataset_id]
                await dataset_edit(q, int(dataset_id))
        elif q.args.__wave_submission_name__ == "dataset/newexperiment":
            if q.client["dataset/list/df_datasets"] is not None:
                dataset_id = int(q.args["dataset/newexperiment"])
                dataset_id = q.client["dataset/list/df_datasets"]["id"].iloc[dataset_id]
                await dataset_newexperiment(q, int(dataset_id))
        elif q.args.__wave_submission_name__ == "dataset/newexperiment/from_current":
            idx = q.client["dataset/display/id"]
            dataset_id = q.client["dataset/list/df_datasets"]["id"].iloc[idx]
            await dataset_newexperiment(q, dataset_id)

        elif q.args.__wave_submission_name__ == "dataset/list/table":
            q.client["dataset/display/id"] = int(q.args["dataset/list/table"][0])
            await dataset_display(q)

        elif q.args.__wave_submission_name__ == "dataset/display/visualization":
            await dataset_display(q)
        elif q.args.__wave_submission_name__ == "dataset/display/data":
            await dataset_display(q)
        elif q.args.__wave_submission_name__ == "dataset/display/statistics":
            await dataset_display(q)
        elif q.args["dataset/display/summary"]:
            await dataset_display(q)

        elif (
            q.args.__wave_submission_name__ == "experiment/start/run"
            or q.args.__wave_submission_name__ == "experiment/start/error/proceed"
            or q.args.__wave_submission_name__ == "experiment/start/gridsearch/proceed"
        ):
            # add model type to cfg file name here
            q.client["experiment/start/cfg_file"] = add_model_type(
                q.client["experiment/start/cfg_file"],
                q.client["experiment/start/cfg_sub"],
            )
            q.client.delete_cards.add("experiment/start")
            await experiment_run(q)
            q.client["experiment/list/mode"] = "train"

        elif (
            q.args.__wave_submission_name__ == "experiment/start_experiment"
            or q.args.__wave_submission_name__ == "experiment/list/new"
        ):
            if q.client["experiment/list/df_experiments"] is not None:
                selected_idx = int(q.args["experiment/list/new"])
                experiment_id = q.client["experiment/list/df_experiments"]["id"].iloc[
                    selected_idx
                ]

                q.client["experiment/start/cfg_category"] = "experiment"
                q.client["experiment/start/cfg_file"] = "experiment"
                q.client["experiment/start/cfg_experiment"] = str(experiment_id)

            await experiment_start(q)
        elif q.args.__wave_submission_name__ == "experiment/start":
            q.client["experiment/start/cfg_category"] = None
            q.client["experiment/start/cfg_file"] = None
            datasets_df = q.client.app_db.get_datasets_df()
            if datasets_df.shape[0] == 0:
                info = "Import dataset before you create an experiment. "
                await dataset_import(q, step=1, info=info)
            else:
                await experiment_start(q)

        elif q.args.__wave_submission_name__ == "experiment/display/download_logs":
            await experiment_download_logs(q)
        elif (
            q.args.__wave_submission_name__ == "experiment/display/download_predictions"
        ):
            await experiment_download_predictions(q)

        elif q.args.__wave_submission_name__ == "experiment/list":
            q.client["experiment/list/mode"] = "train"
            await experiment_list(q)
        elif q.args.__wave_submission_name__ == "experiment/list/current":
            await list_current_experiments(q)
        elif q.args.__wave_submission_name__ == "experiment/list/current/noreset":
            await list_current_experiments(q, reset=False)
        elif q.args.__wave_submission_name__ == "experiment/list/refresh":
            await experiment_list(q)
        elif q.args.__wave_submission_name__ == "experiment/list/abort":
            await list_current_experiments(q)
        elif q.args.__wave_submission_name__ == "experiment/list/stop":
            await current_experiment_list_stop(q)
        elif q.args.__wave_submission_name__ == "experiment/list/delete":
            await current_experiment_list_delete(q)
        elif q.args.__wave_submission_name__ == "experiment/list/rename":
            await experiment_rename_ui_workflow(q)
        elif q.args.__wave_submission_name__ == "experiment/list/compare":
            await current_experiment_list_compare(q)
        elif (
            q.args.__wave_submission_name__ == "experiment/stop"
            or q.args.__wave_submission_name__ == "experiment/list/stop/table"
        ):
            if q.args["experiment/list/stop/table"]:
                idx = int(q.args["experiment/list/stop/table"])
                selected_id = q.client["experiment/list/df_experiments"]["id"].iloc[idx]
                experiment_ids = [selected_id]
            else:
                selected_idxs = q.client["experiment/list/table"]
                experiment_ids = list(
                    q.client["experiment/list/df_experiments"]["id"].iloc[
                        list(map(int, selected_idxs))
                    ]
                )

            await experiment_stop(q, experiment_ids)
            await list_current_experiments(q)
        elif q.args.__wave_submission_name__ == "experiment/list/delete/table/dialog":
            idx = int(q.args["experiment/list/delete/table/dialog"])
            names = [q.client["experiment/list/df_experiments"]["name"].iloc[idx]]
            selected_id = q.client["experiment/list/df_experiments"]["id"].iloc[idx]
            q.client["experiment/delete/single/id"] = selected_id
            if q.client["delete_dialogs"]:
                await delete_dialog(
                    q, names, "experiment/list/delete/table", "experiment"
                )
            else:
                await experiment_delete_all_artifacts(q, [selected_id])

        elif q.args.__wave_submission_name__ == "experiment/delete/dialog":
            selected_idxs = q.client["experiment/list/table"]
            exp_df = q.client["experiment/list/df_experiments"]
            names = list(exp_df["name"].iloc[list(map(int, selected_idxs))])

            if not names:
                return

            if q.client["delete_dialogs"]:
                await delete_dialog(q, names, "experiment/delete", "experiment")
            else:
                experiment_ids = list(exp_df["id"].iloc[list(map(int, selected_idxs))])
                await experiment_delete_all_artifacts(q, experiment_ids)

        elif (
            q.args.__wave_submission_name__ == "experiment/delete"
            or q.args.__wave_submission_name__ == "experiment/list/delete/table"
        ):
            if q.args["experiment/list/delete/table"]:
                selected_id = q.client["experiment/delete/single/id"]
                experiment_ids = [selected_id]
            else:
                selected_idxs = q.client["experiment/list/table"]
                exp_df = q.client["experiment/list/df_experiments"]
                experiment_ids = list(exp_df["id"].iloc[list(map(int, selected_idxs))])

            await experiment_delete_all_artifacts(q, experiment_ids)

        elif q.args.__wave_submission_name__ == "experiment/rename/action":
            await experiment_rename_action_workflow(q)

        elif q.args.__wave_submission_name__ == "experiment/compare":
            await current_experiment_compare(q)
        elif q.args.__wave_submission_name__ == "experiment/compare/charts":
            await current_experiment_compare(q)
        elif q.args.__wave_submission_name__ == "experiment/compare/config":
            await current_experiment_compare(q)
        elif q.args.__wave_submission_name__ == "experiment/compare/diff_toggle":
            q.client["experiment/compare/diff_toggle"] = q.args[
                "experiment/compare/diff_toggle"
            ]
            await current_experiment_compare(q)

        elif q.args.__wave_submission_name__ == "experiment/list/table":
            q.client["experiment/display/id"] = int(q.args["experiment/list/table"][0])
            q.client["experiment/display/logs_path"] = None
            q.client["experiment/display/preds_path"] = None
            q.client["experiment/display/tab"] = None
            await experiment_display(q)

        elif q.args.__wave_submission_name__ == "experiment/display/refresh":
            await experiment_display(q)

        elif q.args.__wave_submission_name__ == "experiment/display/charts":
            await experiment_display(q)
        elif q.args.__wave_submission_name__ == "experiment/display/summary":
            await experiment_display(q)
        elif (
            q.args.__wave_submission_name__ == "experiment/display/train_data_insights"
        ):
            await experiment_display(q)
        elif (
            q.args.__wave_submission_name__
            == "experiment/display/validation_prediction_insights"
        ):
            await experiment_display(q)
        elif (
            q.args.__wave_submission_name__ == "experiment/display/push_to_huggingface"
        ):
            await experiment_push_to_huggingface_dialog(q)
        elif q.args.__wave_submission_name__ == "experiment/display/download_model":
            await experiment_download_model(q)
        elif q.args.__wave_submission_name__ == "experiment/display/download_adapter":
            await experiment_download_adapter(q)
        elif (
            q.args.__wave_submission_name__
            == "experiment/display/push_to_huggingface_submit"
        ):
            await experiment_push_to_huggingface_dialog(q)

        elif q.args.__wave_submission_name__ == "experiment/display/config":
            await experiment_display(q)
        elif q.args.__wave_submission_name__ == "experiment/display/logs":
            await experiment_display(q)
        elif q.args.__wave_submission_name__ == "experiment/display/chat":
            await experiment_display(q)

        elif q.args.__wave_submission_name__ == "experiment/display/chat/chatbot":
            await chat_update(q)
        elif q.args.__wave_submission_name__ == "experiment/display/chat/clear_history":
            await chat_tab(q, load_model=False)

        elif q.args.__wave_submission_name__ == "dataset/import/local_upload":
            await dataset_import_uploaded_file(q)
        elif q.args.__wave_submission_name__ == "dataset/import/local_path_list":
            await dataset_import(q, step=1)
        elif q.args.__wave_submission_name__ == "dataset/import/2":
            await dataset_import(q, step=2)
        elif q.args.__wave_submission_name__ == "dataset/import/3":
            await dataset_import(q, step=3)
        elif q.args.__wave_submission_name__ == "dataset/import/3/edit":
            await dataset_import(q, step=3, edit=True)
        elif q.args.__wave_submission_name__ == "dataset/import/4":
            await dataset_import(q, step=4)
        elif q.args.__wave_submission_name__ == "dataset/import/4/edit":
            await dataset_import(q, step=4, edit=True)
        elif q.args.__wave_submission_name__ == "dataset/import/6":
            await dataset_import(q, step=6)
        elif (
            q.args.__wave_submission_name__ == "dataset/import/source"
            and not q.args["dataset/list"]
        ):
            await dataset_import(q, step=1)
        elif q.args.__wave_submission_name__ == "dataset/merge":
            await dataset_merge(q, step=1)
        elif q.args.__wave_submission_name__ == "dataset/merge/action":
            await dataset_merge(q, step=2)

        elif q.args.__wave_submission_name__ == "dataset/import/cfg_file":
            await dataset_import(q, step=3)

        # leave at the end of dataset import routing,
        # would also be triggered if user clicks on
        # a continue button in the dataset import wizard
        elif q.args.__wave_submission_name__ == "dataset/import/cfg/train_dataframe":
            await dataset_import(q, step=3)

        elif q.args.__wave_submission_name__ == "experiment/start/cfg_file":
            q.client["experiment/start/cfg_file"] = q.args["experiment/start/cfg_file"]
            await experiment_start(q)
        elif q.args.__wave_submission_name__ == "experiment/start/dataset":
            await experiment_start(q)

        elif q.client["nav/active"] == "experiment/start":
            await experiment_start(q)

    except Exception as unknown_exception:
        logger.error("Unknown exception", exc_info=True)
        await wave_utils_handle_error(
            q,
            error=unknown_exception,
        )


async def experiment_delete_all_artifacts(q: Q, experiment_ids: List[int]):
    await experiment_stop(q, experiment_ids)
    await experiment_delete(q, experiment_ids)
    await list_current_experiments(q)
