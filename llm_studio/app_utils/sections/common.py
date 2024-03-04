import hashlib
import logging
from typing import List

from h2o_wave import Q, ui

from llm_studio.app_utils.cards import card_zones
from llm_studio.app_utils.config import default_cfg

logger = logging.getLogger(__name__)


async def meta(q: Q) -> None:
    if q.client["keep_meta"]:  # Do not reset meta, keep current dialog opened
        q.client["keep_meta"] = False
        return

    zones = card_zones(mode=q.client["mode_curr"])

    if q.client["notification_bar"]:
        notification_bar = ui.notification_bar(
            type="warning",
            timeout=20,
            text=q.client["notification_bar"],
            position="top-right",
        )
    else:
        notification_bar = None

    # TODO remove `stylesheet` when wave makes message bars smaller
    q.page["meta"] = ui.meta_card(
        box="",
        title="H2O LLM Studio",
        layouts=[
            ui.layout(breakpoint="0px", width="1430px", zones=zones),
        ],
        scripts=[
            ui.script(source, asynchronous=True) for source in q.app["script_sources"]
        ],
        stylesheet=ui.inline_stylesheet(
            """
             .ms-MessageBar {
              padding-top: 3px;
              padding-bottom: 3px;
              min-height: 18px;
            }
            div[data-test="nav_bar"] .ms-Nav-groupContent {
              margin-bottom: 0;
            }

            div[data-test="experiment/display/deployment/top_right"],
            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child > div > div {
                display: flex;
            }

            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child,
            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child > div {
                display: flex;
                flex-grow: 1;
            }

            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child > div > div > div {
                display: flex;
                flex-grow: 1;
                flex-direction: column;
            }

            div[data-test="experiment/display/deployment/top_right"]
            div[data-visible="true"]:last-child > div > div > div > div {
                flex-grow: 1;
            }
            """
        ),
        script=None,
        notification_bar=notification_bar,
    )

    if q.client.theme_dark:
        q.page["meta"].theme = "h2o-dark"
    else:
        q.page["meta"].theme = "light"


def heap_analytics(
    userid, user_properties=None, event_properties=None
) -> ui.InlineScript:
    script = (
        "window.heap=window.heap||[],heap.load=function(e,t)"
        "{window.heap.appid=e,window.heap."
        'config=t=t||{};var r=document.createElement("script");'
        'r.type="text/javascript",'
        'r.async=!0,r.src="https://cdn.heapanalytics.com/js/heap-"+e+".js";'
        'var a=document.getElementsByTagName("script")[0];'
        "a.parentNode.insertBefore(r,a);"
        "for(var n=function(e){return function(){heap.push([e]."
        "concat(Array.prototype.slice.call(arguments,0)))}},"
        'p=["addEventProperties","addUserProperties","clearEventProperties","identify",'
        '"resetIdentity","removeEventProperty","setEventProperties","track",'
        '"unsetEventProperty"],o=0;o<p.length;o++)heap[p[o]]=n(p[o])};'
        'heap.load("1090178399");'
    )

    identity = hashlib.sha256(userid.encode()).hexdigest()
    script += f"heap.identify('{identity}');"

    if user_properties is not None:
        script += f"heap.addUserProperties({user_properties})"

    if event_properties is not None:
        script += f"heap.addEventProperties({event_properties})"

    return ui.inline_script(content=script)


async def interface(q: Q) -> None:
    """Display interface cards."""

    await meta(q)

    navigation_pages = ["Home", "Settings"]

    if q.client["init_interface"] is None:
        # to avoid flickering
        q.page["header"] = ui.header_card(
            box="header",
            title=default_cfg.name,
            image=q.app["icon_path"],
            subtitle=f"v{default_cfg.version}",
        )

        if q.app.heap_mode:
            logger.info("Heap on")
            q.page["meta"].script = heap_analytics(
                userid=q.auth.subject,
                event_properties=(
                    f"{{version: '{q.app.version}'" + f", product: '{q.app.name}'}}"
                ),
            )
            # execute the heap inline script once in the initialization
            await q.page.save()
        else:
            logger.info("Heap off")

        q.page["nav_bar"] = ui.nav_card(
            box="nav",
            items=[
                ui.nav_group(
                    "Navigation",
                    items=[
                        ui.nav_item(page.lower(), page) for page in navigation_pages
                    ],
                ),
                ui.nav_group(
                    "Datasets",
                    items=[
                        ui.nav_item(name="dataset/import", label="Import dataset"),
                        ui.nav_item(name="dataset/list", label="View datasets"),
                    ],
                ),
                ui.nav_group(
                    "Experiments",
                    items=[
                        ui.nav_item(name="experiment/start", label="Create experiment"),
                        ui.nav_item(name="experiment/list", label="View experiments"),
                    ],
                ),
            ],
            value=(
                default_cfg.start_page
                if q.client["nav/active"] is None
                else q.client["nav/active"]
            ),
        )
    else:
        # Only update menu properties to prevent from flickering
        q.page["nav_bar"].value = (
            default_cfg.start_page
            if q.client["nav/active"] is None
            else q.client["nav/active"]
        )

    q.client["init_interface"] = True


async def clean_dashboard(q: Q, mode: str = "full", exclude: List[str] = []):
    """Drop cards from Q page."""

    logger.info(q.client.delete_cards)
    for card_name in q.client.delete_cards:
        if card_name not in exclude:
            del q.page[card_name]

    q.page["meta"].layouts[0].zones = card_zones(mode=mode)
    q.client["mode_curr"] = mode
    q.client["notification_bar"] = None


async def delete_dialog(q: Q, names: List[str], action, entity):
    title = "Do you really want to delete "
    n_datasets = len(names)

    if n_datasets == 1:
        title = f"{title} {entity} {names[0]}?"
    else:
        title = f"{title} {n_datasets} {entity}s?"

    q.page["meta"].dialog = ui.dialog(
        f"Delete {entity}",
        items=[
            ui.text(title),
            ui.markup("<br>"),
            ui.buttons(
                [
                    ui.button(name=action, label="Delete", primary=True),
                    ui.button(name="abort", label="Abort", primary=False),
                ],
                justify="end",
            ),
        ],
    )
    q.client["keep_meta"] = True


async def info_dialog(q: Q, title: str, message: str):
    q.page["meta"].dialog = ui.dialog(
        title,
        items=[
            ui.text(message),
            ui.markup("<br>"),
            ui.buttons(
                [
                    ui.button(name="abort", label="Continue", primary=False),
                ],
                justify="end",
            ),
        ],
        blocking=True,
    )
    q.client["keep_meta"] = True


async def heap_redact(q: Q) -> None:
    if q.app.heap_mode:
        # Send the page to the browser, so the following js can be applied
        await q.page.save()

        # replace dataset names with ****
        q.page["meta"].script = ui.inline_script(
            """
document.querySelectorAll('div[data-automation-key="name"]').forEach(a => {
  a.setAttribute('data-heap-redact-text', '')
})

document.querySelector('div[data-test="datasets_table"] \
.ms-ScrollablePane--contentContainer').addEventListener('scroll', () => {
  window.setTimeout(() => {{
    document.querySelectorAll('div[data-automation-key="name"]').forEach(a => {
      a.setAttribute('data-heap-redact-text', '')
    })
  }}, 100)
})
    """
        )
