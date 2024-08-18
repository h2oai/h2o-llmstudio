from unittest.mock import MagicMock, patch

import pytest

from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.python_configs.cfg_checks import (
    check_config_for_errors,
    check_for_common_errors,
)


class MockConfig(DefaultConfigProblemBase):
    def __init__(self):
        self.environment = MagicMock()
        self.architecture = MagicMock()
        self.training = MagicMock()

    def check(self):
        return {"title": [], "message": [], "type": []}


@pytest.fixture
def mock_config():
    return MockConfig()


def test_check_config_for_errors(mock_config):
    with patch(
        "llm_studio.python_configs.cfg_checks.check_for_common_errors"
    ) as mock_common_errors:
        mock_common_errors.return_value = {
            "title": ["Common Error"],
            "message": ["Common Error Message"],
            "type": ["error"],
        }

        result = check_config_for_errors(mock_config)

        assert "title" in result
        assert "message" in result
        assert "Common Error" in result["title"]
        assert "Common Error Message" in result["message"]


def test_check_for_common_errors_no_gpu(mock_config):
    mock_config.environment.gpus = []

    result = check_for_common_errors(mock_config)

    assert "No GPU selected" in result["title"]


def test_check_for_common_errors_too_many_gpus(mock_config):
    mock_config.environment.gpus = [0, 1, 2, 3]
    with patch("torch.cuda.device_count", return_value=2):
        result = check_for_common_errors(mock_config)

        assert "More GPUs selected than available" in result["title"]


@patch("os.statvfs")
def test_check_for_common_errors_disk_space(mock_statvfs, mock_config):
    mock_statvfs.return_value = MagicMock(
        f_frsize=4096, f_bavail=1000
    )  # Small disk space

    result = check_for_common_errors(mock_config)

    assert "Not enough disk space." in result["title"]


def test_check_for_common_errors_quantization_without_pretrained(mock_config):
    mock_config.architecture.backbone_dtype = "int4"
    mock_config.architecture.pretrained = False

    result = check_for_common_errors(mock_config)

    assert "Quantization without pretrained weights." in result["title"]
