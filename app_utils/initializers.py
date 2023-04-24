import logging
import os
import pickle
from tempfile import NamedTemporaryFile

import dill
from bokeh.resources import Resources as BokehResources
from h2o_wave import Q

from app_utils.sections.common import interface
from llm_studio.src.utils.config_utils import save_config_yaml

from .config import default_cfg
from .db import Database
from .utils import get_data_dir, get_db_path, get_user_name, load_user_settings, get_output_dir

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


def migrate_pickle_to_yaml(q: Q) -> None:
    data_dir = get_data_dir(q)
    output_dir = get_output_dir(q)

    for dir in [data_dir, output_dir]:
        if os.path.exists(dir):
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith(".p") and not os.path.exists(os.path.join(root, file.replace(".p", ".yml"))):
                        try:
                            with open(os.path.join(root, file), "rb") as f:
                                cfg = dill.load(f)
                                save_config_yaml(os.path.join(root, file.replace(".p", ".yml")), cfg)
                                logger.info(f"migrated {os.path.join(root, file)} to yaml")
                        except Exception as e:
                            logger.error(f"Could not migrate {os.path.join(root, file)} to yaml: {e}")


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

    migrate_pickle_to_yaml(q)

    q.app["initialized"] = True

    logger.info("Initializing app ... done")
