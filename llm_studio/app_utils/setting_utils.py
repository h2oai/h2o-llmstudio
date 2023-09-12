import logging
import multiprocessing
import os
import pickle
import time
import traceback
from typing import Any, List

import keyring
import yaml
from h2o_wave import Q, ui
from keyring.errors import KeyringLocked, PasswordDeleteError

from llm_studio.app_utils.config import default_cfg
from llm_studio.app_utils.utils import get_database_dir, get_user_id

__all__ = [
    "load_user_settings_and_secrets",
    "load_default_user_settings",
    "save_user_settings_and_secrets",
    "Secrets",
]

logger = logging.getLogger(__name__)
PASSWORDS_PHRASES = ["token", "key"]
SECRET_KEYS = [
    key
    for key in default_cfg.user_settings
    if any(password in key for password in PASSWORDS_PHRASES)
]
USER_SETTING_KEYS = [key for key in default_cfg.user_settings if key not in SECRET_KEYS]


async def save_user_settings_and_secrets(q: Q):
    await _save_secrets(q)
    _save_user_settings(q)


def load_user_settings_and_secrets(q: Q):
    _maybe_migrate_to_yaml(q)
    _load_secrets(q)
    _load_user_settings(q)


def load_default_user_settings(q: Q):
    for key in default_cfg.user_settings:
        q.client[key] = default_cfg.user_settings[key]
        _clear_secrets(q, key)


class NoSaver:
    """
    Base class that provides methods for saving, loading, and deleting password entries.

    Attributes:
        username (str): The username associated with the password entries.
        root_dir (str): The root directory.

    Methods:
        save(name: str, password: str) -> None:
            Save a password entry with the given name and password.

        load(name: str) -> str:
            Load and return the password associated with the given name.

        delete(name: str) -> None:
            Delete the password entry with the given name.

    """

    def __init__(self, username: str, root_dir: str):
        self.username = username
        self.root_dir = root_dir

    def save(self, name: str, password: str):
        pass

    def load(self, name: str):
        pass

    def delete(self, name: str):
        pass


class KeyRingSaver(NoSaver):
    """
    A class for saving, loading, and deleting passwords using the keyring library.
    Some machines may not have keyring installed, so this class may not be available.
    """

    def __init__(self, username: str, root_dir: str):
        super().__init__(username, root_dir)
        self.namespace = f"{username}_h2o_llmstudio"

    def save(self, name: str, password: str):
        keyring.set_password(self.namespace, name, password)

    def load(self, name: str):
        return keyring.get_password(self.namespace, name)

    def delete(self, name: str):
        try:
            keyring.delete_password(self.namespace, name)
        except (KeyringLocked, PasswordDeleteError):
            pass
        except Exception as e:
            logger.warning(f"Error deleting password for keyring: {e}")


class EnvFileSaver(NoSaver):
    """
    This module provides the EnvFileSaver class, which is used to save, load,
    and delete name-password pairs in an environment file.
    Only use this class if you are sure that the environment file is secure.
    """

    @property
    def filename(self):
        return os.path.join(self.root_dir, f"{self.username}.env")

    def save(self, name: str, password: str):
        data = {}
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                data = yaml.safe_load(f)
        data[name] = password
        with open(self.filename, "w") as f:
            yaml.safe_dump(data, f)

    def load(self, name: str):
        with open(self.filename, "r") as f:
            data = yaml.safe_load(f)
            return data.get(name, None)

    def delete(self, name: str):
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                data = yaml.safe_load(f)
                if data and name in data:
                    del data[name]
            with open(self.filename, "w") as f:
                yaml.safe_dump(data, f)


def _test_keyring():
    try:
        keyring.get_password("service", "username")
    except Exception:
        time.sleep(4)


def check_if_keyring_works():
    """
    Test if keyring is working. On misconfigured machines,
    keyring may hang up to 2 minutes with the following error:
    jeepney.wrappers.DBusErrorResponse:
    [org.freedesktop.DBus.Error.TimedOut]
    ("Failed to activate service 'org.freedesktop.secrets': timed out (service_start_timeout=120000ms)",)

    To avoid waiting for 2 minutes, we test if keyring works in a separate process and kill it after 3 seconds.
    """

    p = multiprocessing.Process(target=_test_keyring)
    p.start()
    p.join(3)

    if p.is_alive():
        p.kill()
        return False
    return True


class Secrets:
    """
    Factory class to get the secrets' handler.
    """

    _secrets = {
        "Do not save credentials permanently": NoSaver,
        ".env File": EnvFileSaver,
    }
    if check_if_keyring_works():
        _secrets["Keyring"] = KeyRingSaver
    else:
        logger.warning(f"Error loading keyring. Disabling keyring save option.")

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._secrets.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._secrets.get(name)


def _save_user_settings(q: Q):
    user_settings = {key: q.client[key] for key in USER_SETTING_KEYS}
    with open(_get_usersettings_path(q), "w") as f:
        yaml.dump(user_settings, f)


def _load_user_settings(q: Q):
    if os.path.isfile(_get_usersettings_path(q)):
        logger.info("Reading user settings")
        with open(_get_usersettings_path(q), "r") as f:
            user_settings = yaml.load(f, Loader=yaml.FullLoader)
        for key in USER_SETTING_KEYS:
            q.client[key] = user_settings.get(key, default_cfg.user_settings[key])


async def _save_secrets(q: Q):
    secret_name, secrets_handler = _get_secrets_handler(q)
    for key in SECRET_KEYS:
        try:
            _clear_secrets(q, key, excludes=tuple(secret_name))
            if q.client[key]:
                secrets_handler.save(key, q.client[key])

        except Exception:
            exception = str(traceback.format_exc())
            logger.error(f"Could not save password {key} to {secret_name}")
            q.page["meta"].dialog = ui.dialog(
                title="Could not save secrets. "
                      "Please choose another Credential Handler.",
                name="secrets_error",
                items=[
                    ui.text(
                        f"The following error occurred when"
                        f" using {secret_name}: {exception}."
                    ),
                    ui.button(
                        name="settings/close_error_dialog", label="Close", primary=True
                    ),
                ],
                closable=True,
            )
            q.client["keep_meta"] = True
            await q.page.save()
            break
    else:  # if no exception
        # force dataset connector updated when the user decides to click on save
        q.client["dataset/import/s3_bucket"] = q.client["default_aws_bucket_name"]
        q.client["dataset/import/s3_access_key"] = q.client["default_aws_access_key"]
        q.client["dataset/import/s3_secret_key"] = q.client["default_aws_secret_key"]
        q.client["dataset/import/kaggle_access_key"] = q.client[
            "default_kaggle_username"
        ]
        q.client["dataset/import/kaggle_secret_key"] = q.client[
            "default_kaggle_secret_key"
        ]


def _load_secrets(q: Q):
    secret_name, secrets_handler = _get_secrets_handler(q)
    for key in SECRET_KEYS:
        try:
            q.client[key] = secrets_handler.load(key)
        except Exception:
            logger.error(f"Could not load password {key} from {secret_name}")


def _get_secrets_handler(q: Q):
    secret_name = (
            q.client["credential_saver"] or default_cfg.user_settings["credential_saver"]
    )
    secrets_handler = Secrets.get(secret_name)(
        username=get_user_id(q), root_dir=get_database_dir(q)
    )
    return secret_name, secrets_handler


def _clear_secrets(q: Q, name: str, excludes=tuple()):
    for secret_name in Secrets.names():
        if secret_name not in excludes:
            secrets_handler = Secrets.get(secret_name)(
                username=get_user_id(q), root_dir=get_database_dir(q)
            )

            secrets_handler.delete(name)


def _maybe_migrate_to_yaml(q: Q):
    """
    Migrate user settings from a pickle file to a YAML file.
    """
    # prior, we used to save the user settings in a pickle file
    old_usersettings_path = os.path.join(
        get_database_dir(q), f"{get_user_id(q)}.settings"
    )
    if not os.path.isfile(old_usersettings_path):
        return

    try:
        with open(old_usersettings_path, "rb") as f:
            user_settings = pickle.load(f)

        secret_name, secrets_handler = _get_secrets_handler(q)
        logger.info(f"Migrating token using {secret_name}")
        for key in SECRET_KEYS:
            if key in user_settings:
                secrets_handler.save(key, user_settings[key])

        with open(_get_usersettings_path(q), "w") as f:
            yaml.dump(
                {
                    key: value
                    for key, value in user_settings.items()
                    if key in USER_SETTING_KEYS
                },
                f,
            )
        os.remove(old_usersettings_path)
        logger.info(f"Successfully migrated tokens to {secret_name}. Old file deleted.")
    except Exception as e:
        logger.info(
            f"Could not migrate tokens. "
            f"Please delete {old_usersettings_path} and set your credentials again."
            f"Error: \n\n {e} {traceback.format_exc()}"
        )


def _get_usersettings_path(q: Q):
    return os.path.join(get_database_dir(q), f"{get_user_id(q)}.yaml")
