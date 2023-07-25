import subprocess
import sys
import traceback
from typing import TypedDict

import pandas as pd
from h2o_wave import Q, expando_to_dict, ui
from h2o_wave.types import Component

from app_utils.sections.common import clean_dashboard

from .config import default_cfg


class ThemeColors(TypedDict):
    light: dict
    dark: dict


class WaveTheme:
    _theme_colors: ThemeColors = {
        "light": {
            "primary": "#000000",
            "background_color": "#ffffff",
        },
        "dark": {
            "primary": "#FEC925",
            "background_color": "#121212",
        },
    }

    states = {
        "zombie": "#E0E0E0",
        "queued": "#B8B8B8",
        "running": "#FFE52B",
        "finished": "#92E95A",
        "failed": "#DA0000",
        "stopped": "#DA0000",
    }
    color = "#2196F3"
    color_range = "#2196F3 #CC7722 #2CA02C #D62728 #9467BD #17BECF #E377C2 #DDAA22"

    def __repr__(self) -> str:
        return "WaveTheme"

    def get_value_by_key(self, q: Q, key: str):
        value = (
            self._theme_colors["dark"][key]
            if q.client.theme_dark
            else self._theme_colors["light"][key]
        )
        return value

    def get_primary_color(self, q: Q):
        primary_color = self.get_value_by_key(q, "primary")
        return primary_color

    def get_background_color(self, q: Q):
        background_color = self.get_value_by_key(q, "background_color")
        return background_color


wave_theme = WaveTheme()


def ui_table_from_df(
    q: Q,
    df: pd.DataFrame,
    name: str,
    sortables: list = None,
    filterables: list = None,
    searchables: list = None,
    numerics: list = None,
    times: list = None,
    tags: list = None,
    progresses: list = None,
    min_widths: dict = None,
    max_widths: dict = None,
    link_col: str = None,
    multiple: bool = False,
    groupable: bool = False,
    downloadable: bool = False,
    resettable: bool = False,
    height: str = None,
    checkbox_visibility: str = None,
    actions: dict = None,
    # enables truncating the maximum length in characters of each cell on the
    # server-side. Wave truncates on the client-side anyway, so truncating on
    # the server side as well does not make a visible difference but we avoid
    # sending excessive amounts of data.
    max_char_length: int = 500,
) -> Component:
    """
    Convert a Pandas dataframe into Wave ui.table format.
    """

    df = df.reset_index(drop=True)

    if not sortables:
        sortables = []
    if not filterables:
        filterables = []
    if not searchables:
        searchables = []
    if not numerics:
        numerics = []
    if not times:
        times = []
    if not tags:
        tags = []
    if not progresses:
        progresses = []
    if not min_widths:
        min_widths = {}
    if not max_widths:
        max_widths = {}

    cell_types = {}
    for col in tags:
        cell_types[col] = ui.tag_table_cell_type(
            name="tags",
            tags=[
                ui.tag(label=state, color=wave_theme.states[state])
                for state in wave_theme.states
            ],
        )
    for col in progresses:
        cell_types[col] = ui.progress_table_cell_type(
            wave_theme.get_primary_color(q),
        )

    columns = [
        ui.table_column(
            name=str(x),
            label=str(x),
            sortable=True if x in sortables else False,
            filterable=True if x in filterables else False,
            searchable=True if x in searchables else False,
            data_type="number"
            if x in numerics
            else ("time" if x in times else "string"),
            cell_type=cell_types[x] if x in cell_types.keys() else None,
            min_width=min_widths[x] if x in min_widths.keys() else None,
            max_width=max_widths[x] if x in max_widths.keys() else None,
            link=True if x == link_col else False,
            cell_overflow="tooltip",
        )
        for x in df.columns.values
    ]

    if actions:
        commands = [ui.command(name=key, label=val) for key, val in actions.items()]
        action_column = ui.table_column(
            name="actions",
            label="action" if int(min_widths["actions"]) > 30 else "",
            cell_type=ui.menu_table_cell_type(name="commands", commands=commands),
            min_width=min_widths["actions"],
        )
        columns.append(action_column)

    rows = []

    for i, row in df.iterrows():
        cells = []

        for cell in row:
            str_repr = str(cell)

            if len(str_repr) >= max_char_length:
                str_repr = str_repr[:max_char_length] + "..."

            cells.append(str_repr)

        rows.append(ui.table_row(name=str(i), cells=cells))

    table = ui.table(
        name=name,
        columns=columns,
        rows=rows,
        multiple=multiple,
        groupable=groupable,
        downloadable=downloadable,
        resettable=resettable,
        height=height,
        checkbox_visibility=checkbox_visibility,
    )

    return table


def wave_utils_error_card(
    q: Q,
    box: str,
    app_name: str,
    github: str,
    q_app: dict,
    error: Exception,
    q_user: dict,
    q_client: dict,
    q_events: dict,
    q_args: dict,
) -> ui.FormCard:
    """
    Card for handling crash.
    """

    q_app_str = (
        "### q.app\n```"
        + "\n".join(
            [
                f"{k}: {v}"
                for k, v in q_app.items()
                if "_key" not in k and "_token not in k"
            ]
        )
        + "\n```"
    )
    q_user_str = (
        "### q.user\n```"
        + "\n".join(
            [
                f"{k}: {v}"
                for k, v in q_user.items()
                if "_key" not in k and "_token" not in k
            ]
        )
        + "\n```"
    )

    q_client_str = (
        "### q.client\n```"
        + "\n".join(
            [
                f"{k}: {v}"
                for k, v in q_client.items()
                if "_key" not in k and "_token" not in k
            ]
        )
        + "\n```"
    )
    q_events_str = (
        "### q.events\n```"
        + "\n".join(
            [
                f"{k}: {v}"
                for k, v in q_events.items()
                if "_key" not in k and "_token" not in k
            ]
        )
        + "\n```"
    )
    q_args_str = (
        "### q.args\n```"
        + "\n".join(
            [
                f"{k}: {v}"
                for k, v in q_args.items()
                if "_key" not in k and "_token" not in k
            ]
        )
        + "\n```"
    )

    type_, value_, traceback_ = sys.exc_info()
    stack_trace = traceback.format_exception(type_, value_, traceback_)
    git_version = subprocess.getoutput("git rev-parse HEAD")
    if not q.app.wave_utils_stack_trace_str:
        q.app.wave_utils_stack_trace_str = "### stacktrace\n" + "\n".join(stack_trace)

    card = ui.form_card(
        box=box,
        items=[
            ui.stats(
                items=[
                    ui.stat(
                        label="",
                        value="Oops!",
                        caption="Something went wrong",
                        icon="Error",
                        icon_color="#CDDD38",
                    )
                ],
                justify="center",
            ),
            ui.separator(),
            ui.text_l(content="<center>Apologies for the inconvenience!</center>"),
            ui.buttons(
                items=[
                    ui.button(name="home", label="Restart", primary=True),
                    ui.button(name="report_error", label="Report", primary=True),
                ],
                justify="center",
            ),
            ui.separator(visible=False),
            ui.text(
                content=f"""<center>
                    To report this error,
                    please open an issues on Github <a href={github}>{github}</a>
                    with the details below:</center>""",
                visible=False,
            ),
            ui.text_l(content=f"Report Issue: {app_name}", visible=False),
            ui.text_xs(content=q_app_str, visible=False),
            ui.text_xs(content=q_user_str, visible=False),
            ui.text_xs(content=q_client_str, visible=False),
            ui.text_xs(content=q_events_str, visible=False),
            ui.text_xs(content=q_args_str, visible=False),
            ui.text_xs(content=q.app.wave_utils_stack_trace_str, visible=False),
            ui.text_xs(content=f"### Error\n {error}", visible=False),
            ui.text_xs(content=f"### Git Version\n {git_version}", visible=False),
        ],
    )

    return card


async def wave_utils_handle_error(q: Q, error: Exception):
    """
    Handle any app error.
    """

    await clean_dashboard(q, mode="error")

    card_name = "wave_utils_error"

    q.page[card_name] = wave_utils_error_card(
        q,
        box="content",
        error=error,
        app_name=f"{default_cfg.name} at {default_cfg.url}",
        github=default_cfg.github,
        q_app=expando_to_dict(q.app),
        q_user=expando_to_dict(q.user),
        q_client=expando_to_dict(q.client),
        q_events=expando_to_dict(q.events),
        q_args=expando_to_dict(q.args),
    )
    q.client.delete_cards.add("wave_utils_error")

    await q.page.save()


async def report_error(q: Q):
    """
    Report error details.
    """
    card_name = "wave_utils_error"
    # Show card again. Required since card can be cleared
    await wave_utils_handle_error(
        q,
        error=q.app.wave_utils_error_str,
    )

    q.page[card_name].items[4].separator.visible = True
    q.page[card_name].items[5].text.visible = True
    q.page[card_name].items[6].text_l.visible = True
    q.page[card_name].items[7].text_xs.visible = True
    q.page[card_name].items[8].text_xs.visible = True
    q.page[card_name].items[9].text_xs.visible = True
    q.page[card_name].items[10].text_xs.visible = True
    q.page[card_name].items[11].text_xs.visible = True
    q.page[card_name].items[12].text_xs.visible = True
    q.page[card_name].items[13].text_xs.visible = True
    q.page[card_name].items[14].text_xs.visible = True

    await q.page.save()


async def busy_dialog(
    q: Q, title: str = "", text: str = "", force_wait: bool = False
) -> None:
    """Creates busy dialog"""

    q.page["meta"].dialog = ui.dialog(
        title=title,
        primary=True,
        items=[
            ui.progress(label=text),
        ],
        blocking=True,
    )
    await q.page.save()
    if force_wait:
        await q.sleep(1)
    q.page["meta"].dialog = None
