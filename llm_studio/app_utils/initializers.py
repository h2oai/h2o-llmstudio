import logging
import os
import shutil
from tempfile import NamedTemporaryFile

from bokeh.resources import Resources as BokehResources
from h2o_wave import Q

from app_utils.sections.common import interface
from llm_studio.src.utils.config_utils import load_config_py, save_config_yaml

from .config import default_cfg
from .db import Database, Dataset
from .utils import (
    get_data_dir,
    get_database_dir,
    get_download_dir,
    get_output_dir,
    get_user_db_path,
    get_user_name,
    load_user_settings,
    prepare_default_dataset,
)

logger = logging.getLogger(__name__)


def import_data(q: Q):
    """Imports default data"""

    try:
        if q.client.app_db.get_dataset(1) is None:
            logger.info("Downloading default dataset...")

            path = f"{get_data_dir(q)}/oasst"

            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

            df = prepare_default_dataset(path)

            cfg = load_config_py(
                config_path=os.path.join(
                    "llm_studio/python_configs", default_cfg.cfg_file
                ),
                config_name="ConfigProblemBase",
            )

            cfg.dataset.train_dataframe = os.path.join(path, "train_full.pq")
            cfg.dataset.prompt_column = ("instruction",)
            cfg.dataset.answer_column = "output"
            cfg.dataset.parent_id_column = "None"

            cfg_path = os.path.join(path, f"{default_cfg.cfg_file}.yaml")

            save_config_yaml(cfg_path, cfg)

            dataset = Dataset(
                id=1,
                name="oasst",
                path=path,
                config_file=cfg_path,
                train_rows=df.shape[0],
            )

            q.client.app_db.add_dataset(dataset)
    except Exception as e:
        q.client.app_db._session.rollback()
        logger.warning(f"Could not download default dataset: {e}")
        pass


async def initialize_client(q: Q) -> None:
    """Initialize the client."""

    logger.info(f"Initializing client {q.client.client_initialized}")

    if not q.client.client_initialized:
        q.client.delete_cards = set()
        q.client.delete_cards.add("init_app")

        os.makedirs(get_data_dir(q), exist_ok=True)
        os.makedirs(get_database_dir(q), exist_ok=True)
        os.makedirs(get_output_dir(q), exist_ok=True)
        os.makedirs(get_download_dir(q), exist_ok=True)

        db_path = get_user_db_path(q)

        q.client.app_db = Database(db_path)

        logger.info(f"User name: {get_user_name(q)}")

        q.client.client_initialized = True

        q.client["mode_curr"] = "full"

        import_data(q)

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
