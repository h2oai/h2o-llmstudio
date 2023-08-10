import logging
import os
import pickle
import traceback
from typing import List, Any

import keyring
import yaml
from h2o_wave import Q, ui

from app_utils.config import default_cfg
from app_utils.utils.utils import get_usersettings_path

logger = logging.getLogger(__name__)
PASSWORDS = ["token", "key"]


class KeyRingSaver:
    def __init__(self, namespace="h2o-llmstudio"):
        self.namespace = namespace

    def save(self, name, password):
        keyring.set_password(self.namespace, name, password)

    def load(self, name):
        return keyring.get_password(self.namespace, name)

    def delete(self, name):
        keyring.delete_password(self.namespace, name)


class NoSaver:
    def save(self, name, password):
        pass

    def load(self, name):
        pass

    def delete(self, name):
        pass


class EnvFileSaver:
    def __init__(self, filename):
        self.filename = filename

    def save(self, name, password):
        pass

    def load(self, name):
        pass

    def delete(self, name):
        pass


class Secrets:
    """Optimizers factory."""

    _secrets = {
        "Keyring": KeyRingSaver,
        "Do not save credentials": NoSaver,
        ".env File": EnvFileSaver,
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._secrets.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Optimizers.

        Args:
            name: optimizer name
        Returns:
            A class to build the Optimizer
        """
        return cls._secrets.get(name)


async def save_user_settings(q: Q):
    secret_name = q.client["credential_saver"] or default_cfg.user_settings["credential_saver"]
    secrets_handler = Secrets.get(secret_name)()

    can_save_secrets = True
    exception = None

    user_settings = {}
    for key in default_cfg.user_settings:
        if q.client[key] and any(password in key for password in PASSWORDS):
            try:
                clear_secrets(key, excludes=tuple(secret_name))
                secrets_handler.save(key, q.client[key])

            except Exception:
                exception = str(traceback.format_exc())
                can_save_secrets = False
                logger.error(f"Could not save password {key} to {secret_name}")
        else:
            user_settings.update({key: q.client[key]})

    # force dataset connector updated when the user decides to click on save
    q.client["dataset/import/s3_bucket"] = q.client["default_aws_bucket_name"]
    q.client["dataset/import/s3_access_key"] = q.client["default_aws_access_key"]
    q.client["dataset/import/s3_secret_key"] = q.client["default_aws_secret_key"]

    q.client["dataset/import/kaggle_access_key"] = q.client["default_kaggle_username"]
    q.client["dataset/import/kaggle_secret_key"] = q.client["default_kaggle_secret_key"]

    with open(get_usersettings_path(q), "w") as f:
        yaml.dump(user_settings, f)

    if not can_save_secrets:
        q.page["meta"].dialog = ui.dialog(
            title="Could not save secrets.",
            name="secrets_error",
            items=[
                ui.text(
                    f"The following error occurred when using {secret_name}: {exception}."
                ),
            ],
            closable=True,
        )
        q.client["keep_meta"] = True
        await q.page.save()


def load_user_settings(q: Q):
    secret_name = q.client["credential_saver"] or default_cfg.user_settings["credential_saver"]
    secrets_handler = Secrets.get(secret_name)()

    if os.path.isfile(get_usersettings_path(q)):
        logger.info("Reading settings")

        maybe_migrate_to_yaml(q)
        with open(get_usersettings_path(q), "r") as f:
            user_settings = yaml.load(f, Loader=yaml.FullLoader)
        for key in default_cfg.user_settings:
            if any(password in key for password in PASSWORDS):
                try:
                    q.client[key] = secrets_handler.load(key)
                except Exception as e:
                    logger.error(
                        f"Could not load password {key} from keyring due to {e}"
                    )
            else:
                q.client[key] = user_settings.get(key, default_cfg.user_settings[key])


def load_default_user_settings(q: Q):
    for key in default_cfg.user_settings:
        q.client[key] = default_cfg.user_settings[key]


def clear_secrets(name, excludes=tuple()):
    for secret_name in Secrets.names():
        if secret_name not in excludes:
            Secrets.get(secret_name)().delete(name)


def maybe_migrate_to_yaml(q):
    secret_name = q.client["credential_saver"] or default_cfg.user_settings["credential_saver"]
    secrets_handler = Secrets.get(secret_name)()

    usersettings_path = get_usersettings_path(q)
    try:
        with open(usersettings_path, "rb") as f:
            user_settings = pickle.load(f)

        if any(
            [any(password in key for password in PASSWORDS) for key in user_settings]
        ):
            logger.info("Migrating token to keyring")
            for key in list(user_settings.keys()):
                if any(password in key for password in PASSWORDS):
                    if isinstance(user_settings[key], str):
                        secrets_handler.save(key, user_settings[key])
                    del user_settings[key]
            with open(usersettings_path, "w") as f:
                yaml.dump(user_settings, f)
    except Exception as e:
        logger.info(f"Migrating of pickle usersettings {usersettings_path} failed.")
