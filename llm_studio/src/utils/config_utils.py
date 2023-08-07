import dataclasses
import importlib
from types import ModuleType
from typing import Any, Dict, List, Type

import yaml

from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src.utils.type_annotations import KNOWN_TYPE_ANNOTATIONS


def rreload(module):
    """Recursively reload modules.

    Args:
        module: module to reload
    """

    for attribute_name in dir(module):
        if "Config" in attribute_name:
            attribute1 = getattr(module, attribute_name)
            for attribute_name in dir(attribute1):
                attribute2 = getattr(attribute1, attribute_name)
                if type(attribute2) is ModuleType:
                    importlib.reload(attribute2)


def _load_cls(module_path: str, cls_name: str) -> Any:
    """Loads the python class.

    Args:
        module_path: path to the module
        cls_name: name of the class

    Returns:
        Loaded python class
    """

    module_path_fixed = module_path
    if module_path_fixed.endswith(".py"):
        module_path_fixed = module_path_fixed[:-3]
    module_path_fixed = module_path_fixed.replace("/", ".")

    module = importlib.import_module(module_path_fixed)
    module = importlib.reload(module)
    rreload(module)
    module = importlib.reload(module)

    assert hasattr(module, cls_name), "{} file should contain {} class".format(
        module_path, cls_name
    )

    cls = getattr(module, cls_name)

    return cls


def load_config_py(config_path: str, config_name: str = "Config"):
    """Loads the config class.

    Args:
        config_path: path to the config file
        config_name: name of the config class

    Returns:
        Loaded config class
    """

    return _load_cls(config_path, config_name)()


def _get_type_annotation_error(v: Any, type_annotation: Type) -> ValueError:
    return ValueError(
        f"Cannot show {v}: not a dataclass"
        f" and {type_annotation} is not a known type annotation."
    )


def convert_cfg_base_to_nested_dictionary(cfg: DefaultConfigProblemBase) -> dict:
    """Returns a grouped config settings dict for a given configuration

    Args:
        cfg: configuration
        q: Q

    Returns:
        Dict of configuration settings
    """

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()
    cfg_dict = {key: cfg_dict[key] for key in cfg._get_order()}

    grouped_cfg_dict = {}

    for k, v in cfg_dict.items():
        if k.startswith("_"):
            continue

        if any([x in k for x in ["api", "secret"]]):
            raise AssertionError(
                "Config item must not contain the word 'api' or 'secret'"
            )

        type_annotation = type_annotations[k]

        if type_annotation in KNOWN_TYPE_ANNOTATIONS:
            grouped_cfg_dict.update({k: v})
        elif dataclasses.is_dataclass(v):
            group_items = parse_cfg_dataclass(cfg=v)
            group_items = {
                k: list(v) if isinstance(v, tuple) else v
                for d in group_items
                for k, v in d.items()
            }
            grouped_cfg_dict.update({k: group_items})
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

    # not an explicit field in the config
    grouped_cfg_dict["problem_type"] = cfg.problem_type
    return grouped_cfg_dict


def get_parent_element(cfg):
    if hasattr(cfg, "_parent_experiment") and cfg._parent_experiment != "":
        key = "Parent Experiment"
        value = cfg._parent_experiment
        return {key: value}

    return None


def parse_cfg_dataclass(cfg) -> List[Dict]:
    """Returns all single config settings for a given configuration

    Args:
        cfg: configuration
    """

    items = []

    parent_element = get_parent_element(cfg)
    if parent_element:
        items.append(parent_element)

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()
    cfg_dict = {key: cfg_dict[key] for key in cfg._get_order()}

    for k, v in cfg_dict.items():
        if k.startswith("_"):
            continue

        if any([x in k for x in ["api"]]):
            continue

        type_annotation = type_annotations[k]

        if type_annotation in KNOWN_TYPE_ANNOTATIONS:
            if type_annotation == float:
                v = float(v)
            t = [{k: v}]
        elif dataclasses.is_dataclass(v):
            elements_group = parse_cfg_dataclass(cfg=v)
            t = elements_group
        else:
            continue

        items += t

    return items


def save_config_yaml(path: str, cfg: DefaultConfigProblemBase) -> None:
    """Saves config as yaml file

    Args:
        path: path of file to save to
        cfg: config to save
    """
    """
    Returns a dictionary representation of the config object.
    Protected attributes (starting with an underscore) are not included.
    Nested configs are converted to nested dictionaries.
    """
    cfg_dict = convert_cfg_base_to_nested_dictionary(cfg)
    with open(path, "w") as fp:
        yaml.dump(cfg_dict, fp, indent=4)


def load_config_yaml(path: str):
    """Loads config from yaml file

    Args:
        path: path of file to load from
    Returns:
        config object
    """
    with open(path, "r") as fp:
        cfg_dict = yaml.load(fp, Loader=yaml.FullLoader)

    problem_type = cfg_dict["problem_type"]
    module_name = f"llm_studio.python_configs.{problem_type}_config"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise NotImplementedError(f"Problem Type {problem_type} not implemented")
    return module.ConfigProblemBase.from_dict(cfg_dict)
