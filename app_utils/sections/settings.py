import multiprocessing

import torch
from h2o_wave import Q, ui

from app_utils.sections.common import clean_dashboard
from llm_studio.src.loggers import Loggers


async def settings(q: Q) -> None:
    await clean_dashboard(q, mode="full")
    q.client["nav/active"] = "settings"

    label_width = "250px"
    textbox_width = "350px"

    q.page["settings/content"] = ui.form_card(
        box="content",
        items=[
            ui.message_bar(
                type="info",
                text="Setting changes are directly applied for the \
                current session and can be made persistent by using the \
                'Save settings persistently' button below. To reload \
                the persistently saved settings, use the 'Load settings' button.",
            ),
            ui.separator("Appearance"),
            ui.inline(
                items=[
                    ui.label("Dark Mode", width=label_width),
                    ui.toggle(
                        name="theme_dark",
                        value=q.client["theme_dark"],
                        tooltip="Enables Dark Mode as theme.",
                        trigger=True,
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Delete Dialogs", width=label_width),
                    ui.toggle(
                        name="delete_dialogs",
                        value=q.client["delete_dialogs"],
                        trigger=False,
                        tooltip=(
                            "Whether to show delete dialogs before deleting "
                            "datasets or experiments."
                        ),
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Maximum Chart Points", width=label_width),
                    ui.spinbox(
                        name="chart_plot_max_points",
                        label=None,
                        min=1,
                        max=10000,
                        step=1000,
                        value=q.client["chart_plot_max_points"],
                        width=textbox_width,
                        trigger=False,
                        tooltip="Set the maximum number of points shown in the "
                        "experiment chart plots. Plots will be sub-sampled if "
                        "needed.",
                    ),
                ]
            ),
            ui.separator("Default Connector Settings"),
            ui.inline(
                items=[
                    ui.label("AWS S3 bucket name", width=label_width),
                    ui.textbox(
                        name="default_aws_bucket_name",
                        label=None,
                        value=q.client["default_aws_bucket_name"],
                        width=textbox_width,
                        trigger=False,
                        tooltip="Set the value for the AWS bucket for \
                            dataset import. S3 bucket name including relative paths.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("AWS access key", width=label_width),
                    ui.textbox(
                        name="default_aws_access_key",
                        label=None,
                        value=q.client["default_aws_access_key"],
                        width=textbox_width,
                        password=True,
                        trigger=False,
                        tooltip="Set the value for the AWS access key \
                            for dataset import.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("AWS secret key", width=label_width),
                    ui.textbox(
                        name="default_aws_secret_key",
                        label=None,
                        value=q.client["default_aws_secret_key"],
                        width=textbox_width,
                        password=True,
                        trigger=False,
                        tooltip="Set the value for the AWS secret key \
                            for dataset import.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Kaggle username", width=label_width),
                    ui.textbox(
                        name="default_kaggle_username",
                        label=None,
                        value=q.client["default_kaggle_username"],
                        width=textbox_width,
                        password=False,
                        trigger=False,
                        tooltip="Set the value for the Kaggle username \
                            for dataset import.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Kaggle secret key", width=label_width),
                    ui.textbox(
                        name="default_kaggle_secret_key",
                        label=None,
                        value=q.client["default_kaggle_secret_key"],
                        width=textbox_width,
                        password=True,
                        trigger=False,
                        tooltip="Set the value for the Kaggle secret key \
                            for dataset import.",
                    ),
                ]
            ),
            ui.separator("Default Experiment Settings"),
            ui.inline(
                items=[
                    ui.label("Gpu ID used for chat", width=label_width),
                    ui.spinbox(
                        name="gpu_id_for_chat",
                        label=None,
                        min=0,
                        max=torch.cuda.device_count(),
                        step=1,
                        value=q.client["gpu_id_for_chat"],
                        width=textbox_width,
                        trigger=False,
                        tooltip="Set the gpu id that is used for the chat window.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Number of Workers", width=label_width),
                    ui.spinbox(
                        name="default_number_of_workers",
                        label=None,
                        min=1,
                        max=multiprocessing.cpu_count(),
                        step=1,
                        value=q.client["default_number_of_workers"],
                        width=textbox_width,
                        trigger=False,
                        tooltip="Set the value for the number of workers \
                            sliders in the experiment setup.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Logger", width=label_width),
                    ui.dropdown(
                        name="default_logger",
                        value=q.client["default_logger"],
                        choices=[ui.choice(name, name) for name in Loggers.names()],
                        trigger=False,
                        width="100px",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Neptune Project", width=label_width),
                    ui.textbox(
                        name="default_neptune_project",
                        label=None,
                        value=q.client["default_neptune_project"],
                        width=textbox_width,
                        trigger=False,
                        tooltip="Set the value for the neptune project \
                            in the experiment setup.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Neptune API Token", width=label_width),
                    ui.textbox(
                        name="default_neptune_api_token",
                        label=None,
                        value=q.client["default_neptune_api_token"],
                        width=textbox_width,
                        password=True,
                        trigger=False,
                        tooltip="Set the value for the Neptune API token \
                            in the experiment setup.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Huggingface API Token", width=label_width),
                    ui.textbox(
                        name="default_huggingface_api_token",
                        label=None,
                        value=q.client["default_huggingface_api_token"],
                        width=textbox_width,
                        password=True,
                        trigger=False,
                        tooltip="Set the value for the Huggingface API token \
                            in the experiment setup.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("OpenAI API Token", width=label_width),
                    ui.textbox(
                        name="default_openai_api_token",
                        label=None,
                        value=q.client["default_openai_api_token"],
                        width=textbox_width,
                        password=True,
                        trigger=False,
                        tooltip="Set the value for the OpenAI API token \
                            in the experiment setup.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("GPT evaluation max samples", width=label_width),
                    ui.spinbox(
                        name="default_gpt_eval_max",
                        label=None,
                        value=q.client["default_gpt_eval_max"],
                        width=textbox_width,
                        min=1,
                        max=10000,
                        step=1,
                        trigger=False,
                        tooltip="Set the maximum samples for GPT evaluation. \
                            This is used to prevent unexpected high API costs. \
                            Increase at your own risk.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Use OpenAI API on Azure", width=label_width),
                    ui.toggle(
                        name="default_openai_azure",
                        value=q.client["default_openai_azure"],
                        tooltip=(
                            "Toggle to use Microsoft Azure Endpoints for the "
                            "OpenAI API."
                        ),
                        trigger=True,
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("OpenAI API Endpoint", width=label_width),
                    ui.textbox(
                        name="default_openai_api_base",
                        label=None,
                        value=q.client["default_openai_api_base"],
                        width=textbox_width,
                        password=False,
                        trigger=False,
                        tooltip=(
                            "Set the value for the OpenAI API endpoint. "
                            "Use when on Azure."
                        ),
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("OpenAI API Deployment ID", width=label_width),
                    ui.textbox(
                        name="default_openai_api_deployment_id",
                        label=None,
                        value=q.client["default_openai_api_deployment_id"],
                        width=textbox_width,
                        password=False,
                        trigger=False,
                        tooltip=(
                            "Set the value for the OpenAI API deployment ID. "
                            "Use when on Azure."
                        ),
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("OpenAI API version", width=label_width),
                    ui.textbox(
                        name="default_openai_api_version",
                        label=None,
                        value=q.client["default_openai_api_version"],
                        width=textbox_width,
                        password=False,
                        trigger=False,
                        tooltip=(
                            "Set the value for the OpenAI API version. "
                            "Use when on Azure."
                        ),
                    ),
                ]
            ),
            ui.separator("Experiment Maximum Settings"),
            ui.inline(
                items=[
                    ui.label("Number of Epochs", width=label_width),
                    ui.spinbox(
                        name="set_max_epochs",
                        label=None,
                        min=1,
                        max=2000,
                        step=1,
                        value=q.client["set_max_epochs"],
                        width=textbox_width,
                        trigger=False,
                        tooltip="Set the maximum value for the epoch slider \
                            in the experiment setup.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Batch Size", width=label_width),
                    ui.spinbox(
                        name="set_max_batch_size",
                        label=None,
                        min=1,
                        max=4096,
                        step=1,
                        value=q.client["set_max_batch_size"],
                        width=textbox_width,
                        trigger=False,
                        tooltip="Set the maximum value for the batch size slider \
                            in the experiment setup.",
                    ),
                ]
            ),
            ui.inline(
                items=[
                    ui.label("Gradient clip", width=label_width),
                    ui.spinbox(
                        name="set_max_gradient_clip",
                        label=None,
                        min=1,
                        max=16384,
                        step=1,
                        value=q.client["set_max_gradient_clip"],
                        width=textbox_width,
                        trigger=False,
                        tooltip="Set the maximum value for the gradient clip \
                            slider in the experiment setup.",
                    ),
                ]
            ),
        ],
    )

    q.client.delete_cards.add("settings/content")

    q.page["settings/footer"] = ui.form_card(
        box="footer",
        items=[
            ui.inline(
                items=[
                    ui.button(
                        name="save_settings",
                        label="Save settings persistently",
                        primary=True,
                    ),
                    ui.button(
                        name="load_settings", label="Load settings", primary=False
                    ),
                    ui.button(
                        name="restore_default_settings",
                        label="Restore default settings",
                        primary=False,
                    ),
                ]
            )
        ],
    )
    q.client.delete_cards.add("settings/footer")
