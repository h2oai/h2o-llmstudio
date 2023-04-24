import logging
import os

import dill
from h2o_wave import Q

from app_utils.db import Database
from app_utils.utils import get_data_dir, get_output_dir
from llm_studio.src.utils.config_utils import save_config_yaml

logger = logging.getLogger(__name__)


async def migrate_app(q: Q) -> None:
    migrate_pickle_to_yaml(q)
    migrate_database(q)


def migrate_pickle_to_yaml(q: Q) -> None:
    data_dir = get_data_dir(q)
    output_dir = get_output_dir(q)

    for dir in [data_dir, output_dir]:
        if os.path.exists(dir):
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith(".p") and not os.path.exists(
                        os.path.join(root, file.replace(".p", ".yaml"))
                    ):
                        try:
                            with open(os.path.join(root, file), "rb") as f:
                                cfg = dill.load(f)
                                save_config_yaml(
                                    os.path.join(root, file.replace(".p", ".yaml")), cfg
                                )
                                logger.info(
                                    f"migrated {os.path.join(root, file)} to yaml"
                                )
                        except Exception as e:
                            logger.error(
                                f"Could not migrate {os.path.join(root, file)} "
                                f"to yaml: {e}"
                            )


def migrate_database(q: Q) -> None:
    db: Database = q.client.app_db
    for dataset_id in db.get_datasets_df()["id"]:
        dataset = db.get_dataset(dataset_id)
        dataset.config_file = dataset.config_file.replace(".p", ".yaml")
        db.update()
