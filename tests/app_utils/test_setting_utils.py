from unittest import mock

from llm_studio.app_utils.config import default_cfg
from llm_studio.app_utils.setting_utils import (
    EnvFileSaver,
    KeyRingSaver,
    NoSaver,
    Secrets,
    load_default_user_settings,
)


def test_no_saver():
    saver = NoSaver("test_user", "/")
    assert saver.save("name", "password") is None
    assert saver.load("name") == ""
    assert saver.delete("name") is None


def test_keyring_saver(mocker):
    mocker.patch("keyring.set_password")
    mocker.patch("keyring.get_password", return_value="password")
    mocker.patch("keyring.delete_password")
    saver = KeyRingSaver("test_user", "/")
    saver.save("name", "password")
    assert saver.load("name") == "password"
    saver.delete("name")
    assert mocker.patch("keyring.delete_password").is_called


def test_env_file_saver(tmpdir):
    saver = EnvFileSaver("test_user", str(tmpdir))
    saver.save("name", "password")
    saver.save("name2", "password2")
    assert saver.load("name") == "password"
    saver.delete("name")
    assert saver.load("name") == ""
    assert saver.load("name2") == "password2"


def test_secrets_get():
    assert isinstance(Secrets.get("Do not save credentials permanently"), type)
    assert isinstance(Secrets.get("Keyring"), type)
    assert isinstance(Secrets.get(".env File"), type)


def test_load_default_user_settings(mocker):
    q = mock.MagicMock()
    q.client = dict()
    mocker.patch("llm_studio.app_utils.setting_utils._clear_secrets", return_value=None)
    load_default_user_settings(q)
    assert set(q.client.keys()) == set(default_cfg.user_settings.keys())
