import psutil
import torch
from h2o_wave import Q, data, ui

from app_utils.sections.common import clean_dashboard
from app_utils.utils import (
    get_datasets,
    get_experiments,
    get_gpu_usage,
    get_single_gpu_usage,
)
from app_utils.wave_utils import ui_table_from_df, wave_theme
from llm_studio.src.utils.export_utils import get_size_str


async def home(q: Q) -> None:

    await clean_dashboard(q, mode="home")
    q.client["nav/active"] = "home"

    experiments = get_experiments(q)
    hdd = psutil.disk_usage("./data/")

    q.page["home/disk_usage"] = ui.tall_gauge_stat_card(
        box=ui.box("content", order=2, width="20%" if len(experiments) > 0 else "30%"),
        title="Disk usage",
        value=f"{hdd.percent:.2f} %",
        aux_value=f"{get_size_str(hdd.used, sig_figs=1)} /\
            {get_size_str(hdd.total, sig_figs=1)}",
        plot_color=wave_theme.get_primary_color(q),
        progress=hdd.percent / 100,
    )

    if len(experiments) > 0:
        num_finished = len(experiments[experiments["status"] == "finished"])
        num_running_queued = len(
            experiments[experiments["status"].isin(["queued", "running"])]
        )
        num_failed_stopped = len(
            experiments[experiments["status"].isin(["failed", "stopped"])]
        )

        q.page["home/experiments_stats"] = ui.form_card(
            box=ui.box("content", order=1, width="40%"),
            title="Experiments",
            items=[
                ui.visualization(
                    plot=ui.plot(
                        [ui.mark(type="interval", x="=status", y="=count", y_min=0)]
                    ),
                    data=data(
                        fields="status count",
                        rows=[
                            ("finished", num_finished),
                            ("queued + running", num_running_queued),
                            ("failed + stopped", num_failed_stopped),
                        ],
                        pack=True,
                    ),
                )
            ],
        )

    stats = []
    if torch.cuda.is_available():
        stats.append(ui.stat(label="Current GPU load", value=f"{get_gpu_usage():.1f}%"))
    stats += [
        ui.stat(label="Current CPU load", value=f"{psutil.cpu_percent()}%"),
        ui.stat(
            label="Memory usage",
            value=f"{get_size_str(psutil.virtual_memory().used, sig_figs=1)} /\
                    {get_size_str(psutil.virtual_memory().total, sig_figs=1)}",
        ),
    ]

    q.page["home/compute_stats"] = ui.tall_stats_card(
        box=ui.box("content", order=1, width="40%" if len(experiments) > 0 else "70%"),
        items=stats,
    )

    if torch.cuda.is_available():
        q.page["home/gpu_stats"] = ui.form_card(
            box=ui.box("expander", width="100%"),
            items=[
                ui.expander(
                    name="expander",
                    label="Detailed GPU stats",
                    items=get_single_gpu_usage(
                        highlight=wave_theme.get_primary_color(q)
                    ),
                    expanded=True,
                )
            ],
        )
        q.client.delete_cards.add("home/gpu_stats")

    q.client.delete_cards.add("home/compute_stats")
    q.client.delete_cards.add("home/disk_usage")
    q.client.delete_cards.add("home/experiments_stats")

    q.client["experiment/list/mode"] = "train"

    q.client["dataset/list/df_datasets"] = get_datasets(q)
    df_viz = q.client["dataset/list/df_datasets"].copy()
    df_viz = df_viz[df_viz.columns.intersection(["name", "problem type"])]

    if torch.cuda.is_available():
        table_height = "max(calc(100vh - 650px), 400px)"
    else:
        table_height = "max(calc(100vh - 540px), 400px)"

    q.page["dataset/list"] = ui.form_card(
        box="datasets",
        items=[
            ui.inline(
                [
                    ui.button(
                        name="dataset/list", icon="Database", label="", primary=True
                    ),
                    ui.label("List of Datasets"),
                ]
            ),
            ui_table_from_df(
                q=q,
                df=df_viz,
                name="dataset/list/table",
                sortables=[],
                searchables=[],
                min_widths={"name": "240", "problem type": "130"},
                link_col="name",
                height=table_height,
            ),
        ],
    )
    q.client.delete_cards.add("dataset/list")

    q.client["experiment/list/df_experiments"] = get_experiments(
        q, mode=q.client["experiment/list/mode"], status="finished"
    )

    df_viz = q.client["experiment/list/df_experiments"].copy()
    df_viz = df_viz.rename(columns={"process_id": "pid", "config_file": "problem type"})
    df_viz = df_viz[
        df_viz.columns.intersection(
            ["name", "dataset", "problem type", "metric", "val metric"]
        )
    ]

    q.page["experiment/list"] = ui.form_card(
        box="experiments",
        items=[
            ui.inline(
                [
                    ui.button(
                        name="experiment/list",
                        icon="FlameSolid",
                        label="",
                        primary=True,
                    ),
                    ui.label("List of Experiments"),
                ]
            ),
            ui_table_from_df(
                q=q,
                df=df_viz,
                name="experiment/list/table",
                min_widths={
                    # "id": "50",
                    "name": "115",
                    "dataset": "100",
                    "problem type": "120",
                    "metric": "70",
                    "val metric": "85",
                },
                link_col="name",
                numerics=["val metric"],
                sortables=["val metric"],
                height=table_height,
            ),
        ],
    )
