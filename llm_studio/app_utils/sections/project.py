import logging
import os

from h2o_wave import Q, ui

from app_utils.sections.experiment import (
    experiment_compare,
    experiment_list,
    experiment_rename_action,
    experiment_rename_form,
    get_table_and_message_item_indices,
)
from app_utils.utils import get_experiments_status

logger = logging.getLogger(__name__)


async def experiment_rename_action_workflow(q: Q):
    if q.args["experiment/rename/name"]:
        q.client["experiment/rename/name"] = q.args["experiment/rename/name"]

    new_name = q.client["experiment/rename/name"]
    if new_name and new_name.strip():
        current_id = q.client["experiment/rename/id"]
        experiment = q.client.app_db.get_experiment(current_id)
        new_path = experiment.path.replace(experiment.name, new_name)

        experiment_df = q.client.app_db.get_experiments_df()
        experiment_df["status"], experiment_df["info"] = get_experiments_status(
            experiment_df
        )
        status = experiment_df.set_index("id").loc[experiment.id, "status"]

        if os.path.exists(new_path):
            error = f"Experiment <strong>{new_name}</strong> already exists."
            await experiment_rename_form(q, error=error)
        elif status in ["running", "queued"]:
            error = "Cannot rename running or queued experiments."
            await experiment_rename_form(q, error=error)
        else:
            await experiment_rename_action(q, experiment, new_name)
            await list_current_experiments(q)
    else:
        await experiment_rename_form(q, error="New name must be non-empty")


async def list_current_experiments(q, allowed_statuses=None, actions=True, reset=True):
    await experiment_list(
        q,
        allowed_statuses=allowed_statuses,
        reset=reset,
        actions=actions,
    )

    if not reset:  # in case of abort button disable multi-select
        table_item_idx, message_item_idx = get_table_and_message_item_indices(q)
        q.page["experiment/list"].items[table_item_idx].table.multiple = False


async def current_experiment_list_stop(q: Q) -> None:
    """Allow to select experiments to stop."""

    table_item_idx, message_item_idx = get_table_and_message_item_indices(q)
    stop_label = "Stop experiments"

    q.page["experiment/list"].items[table_item_idx].table.multiple = True
    q.page["dataset/display/footer"].items = [
        ui.inline(
            items=[
                ui.button(name="experiment/stop", label=stop_label, primary=True),
                ui.button(name="experiment/list/current/noreset", label="Abort"),
            ]
        )
    ]


async def current_experiment_list_delete(q: Q) -> None:
    """Allow to select experiments to delete."""

    table_item_idx, message_item_idx = get_table_and_message_item_indices(q)
    delete_label = "Delete experiments"

    q.page["experiment/list"].items[table_item_idx].table.multiple = True
    q.page["dataset/display/footer"].items = [
        ui.inline(
            items=[
                ui.button(
                    name="experiment/delete/dialog", label=delete_label, primary=True
                ),
                ui.button(name="experiment/list/current/noreset", label="Abort"),
            ]
        )
    ]


async def current_experiment_list_compare(q: Q) -> None:
    """Allow to select previous experiment to start new one."""

    table_item_idx, message_item_idx = get_table_and_message_item_indices(q)
    q.page["experiment/list"].items[table_item_idx].table.multiple = True
    q.page["dataset/display/footer"].items = [
        ui.inline(
            items=[
                ui.button(
                    name="experiment/compare",
                    label="Compare experiments",
                    primary=True,
                ),
                ui.button(name="experiment/list/current/noreset", label="Abort"),
            ]
        )
    ]


async def current_experiment_compare(q: Q) -> None:
    selected_rows = q.args["experiment/list/table"]
    if selected_rows:
        q.client["experiment/compare/selected"] = selected_rows
    elif q.client["experiment/compare/selected"]:
        selected_rows = q.client["experiment/compare/selected"]
    else:
        await list_current_experiments(q)
        return

    await experiment_compare(q, selected_rows)
