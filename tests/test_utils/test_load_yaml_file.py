import os

from llm_studio.src.utils.config_utils import load_config_yaml

load_config_yaml


def test_load_config_yaml():
    test_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(test_directory, "test_data/cfg.yaml")
    cfg = load_config_yaml(cfg_path)
    cfg
