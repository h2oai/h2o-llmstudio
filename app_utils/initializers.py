import logging
import os
from tempfile import NamedTemporaryFile

from bokeh.resources import Resources as BokehResources
from h2o_wave import Q

from app_utils.sections.common import interface

from .config import default_cfg
from .db import Database
from .utils import get_data_dir, get_db_path, get_user_name, load_user_settings

logger = logging.getLogger(__name__)


async def initialize_client(q: Q) -> None:
    """Initialize the client."""

    logger.info(f"Initializing client {q.client.client_initialized}")

    if not q.client.client_initialized:

        q.client.delete_cards = set()
        q.client.delete_cards.add("init_app")

        os.makedirs(get_data_dir(q), exist_ok=True)

        os.makedirs(default_cfg.dbs_path, exist_ok=True)
        db_path = get_db_path(q)

        q.client.app_db = Database(db_path)

        logger.info(f"User name: {get_user_name(q)}")

        q.client.client_initialized = True

        q.client["mode_curr"] = "full"

        load_user_settings(q)

        await interface(q)

        q.args[default_cfg.start_page] = True

    return


async def initialize_app(q: Q) -> None:
    """
    Initialize the app.

    This function is called once when the app is started and stores values in q.app.
    """

    logger.info("Initializing app ...")

    icons_pth = "app_utils/static/"
    (q.app["icon_path"],) = await q.site.upload([f"{icons_pth}/icon.png"])

    script_sources = []

    with NamedTemporaryFile(mode="w", suffix=".min.js") as f:
        # write all Bokeh scripts to one file to make sure
        # they are loaded sequentially
        for js_raw in BokehResources(mode="inline").js_raw:
            f.write(js_raw)
            f.write("\n")

        (url,) = await q.site.upload([f.name])
        script_sources.append(url)

    q.app["script_sources"] = script_sources

    q.app["initialized"] = True

    logger.info("Initializing app ... done")
