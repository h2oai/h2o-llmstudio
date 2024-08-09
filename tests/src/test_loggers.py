import tempfile
from unittest.mock import MagicMock

import pytest

from llm_studio.src.loggers import DummyLogger, MainLogger


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir_path:
        yield temp_dir_path


@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    cfg.logging.logger = "None"
    cfg.logging._neptune_debug = False
    return cfg


def test_main_logger_initialization(mock_cfg, temp_dir):
    mock_cfg.output_directory = temp_dir
    logger = MainLogger(mock_cfg)

    # Has external and local logger
    assert "external" in logger.loggers.keys()
    assert "local" in logger.loggers.keys()

    # external logger is DummyLogger
    assert isinstance(logger.loggers["external"], DummyLogger)
