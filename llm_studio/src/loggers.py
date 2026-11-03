import dataclasses
import os
from typing import Any

import numpy as np

from llm_studio.src.utils.disk_kv import Cache
from llm_studio.src.utils.plot_utils import PLOT_ENCODINGS


def get_cfg(cfg: Any) -> dict:
    """Returns simplified config elements

    Args:
        cfg: configuration

    Returns:
        Dict of config elements
    """

    items: dict = {}
    type_annotations = cfg.get_annotations()

    cfg_dict = cfg.__dict__

    cfg_dict = {key: cfg_dict[key] for key in cfg._get_order(warn_if_unset=False)}

    for k, v in cfg_dict.items():
        if k.startswith("_") or cfg._get_visibility(k) < 0:
            continue

        if any([x in k for x in ["api", "secret", "key"]]):
            continue

        if dataclasses.is_dataclass(v):
            elements_group = get_cfg(cfg=v)
            t = elements_group
            items = {**items, **t}
        else:
            type_annotation = type_annotations[k]
            if type_annotation is float:
                items[k] = float(v)
            else:
                items[k] = v

    return items


class LocalLogger:
    def __init__(self, cfg: Any):
        self.logs = os.path.join(cfg.output_directory, "charts_cache")

        params = get_cfg(cfg)

        with Cache(self.logs) as cache:
            cache["cfg"] = params

    def log(self, subset: str, name: str, value: Any, step: int | None = None):
        with Cache(self.logs) as cache:
            if subset in PLOT_ENCODINGS:
                if subset not in cache:
                    subset_dict = dict()
                else:
                    subset_dict = cache[subset]
                subset_dict[name] = value
                cache[subset] = subset_dict
                return

            # https://github.com/h2oai/wave/issues/447
            if np.isnan(value):
                value = None
            else:
                value = float(value)

            if subset not in cache:
                subset_dict = dict()
            else:
                subset_dict = cache[subset]
            if name not in subset_dict:
                subset_dict[name] = {"steps": [], "values": []}

            subset_dict[name]["steps"].append(step)
            subset_dict[name]["values"].append(value)

            cache[subset] = subset_dict
