import dataclasses
import logging
import os
from typing import Any

import numpy as np
from diskcache import Cache

from llm_studio.src.utils.plot_utils import PLOT_ENCODINGS

logger = logging.getLogger(__name__)


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


class NeptuneLogger:
    def __init__(self, cfg: Any):
        import neptune as neptune
        from neptune.utils import stringify_unsupported

        self.logger = neptune.init_run(
            project=cfg.logging.neptune_project,
            api_token=os.getenv("NEPTUNE_API_TOKEN", ""),
            name=cfg.experiment_name,
            mode="debug" if cfg.logging._neptune_debug else "async",
            capture_stdout=False,
            capture_stderr=False,
            source_files=[],
        )

        self.logger["cfg"] = stringify_unsupported(get_cfg(cfg))

    def log(self, subset: str, name: str, value: Any, step: int | None = None):
        name = f"{subset}/{name}"
        self.logger[name].append(value, step=step)


class WandbLogger:
    def __init__(self, cfg: Any) -> None:
        os.environ["WANDB_DISABLE_CODE"] = "true"
        os.environ["WANDB_DISABLE_GIT"] = "true"
        os.environ["WANDB_ERROR_REPORTING"] = "false"
        os.environ["WANDB_CONSOLE"] = "off"
        os.environ["WANDB_IGNORE_GLOBS"] = "*.*"
        os.environ["WANDB_HOST"] = "H2O LLM Studio"

        import wandb

        self.logger = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.experiment_name,
            config=get_cfg(cfg),
            save_code=False,
        )

    def log(self, subset: str, name: str, value: Any, step: int | None = None):
        name = f"{subset}/{name}"
        if step is not None:
            # NOTE: Explicit cast is needed, as training sometimes reports float steps
            step = int(step)

        self.logger.log({name: value}, step=step)


class LocalLogger:
    def __init__(self, cfg: Any):
        logging.getLogger("diskcache").setLevel(logging.ERROR)

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


class DummyLogger:
    def __init__(self, cfg: Any | None = None):
        return

    def log(self, subset: str, name: str, value: Any, step: int | None = None):
        return


class MainLogger:
    """Main logger"""

    def __init__(self, cfg: Any):
        self.loggers = {
            "local": LocalLogger(cfg),
            "external": ExternalLoggers.get(cfg.logging.logger),
        }

        try:
            self.loggers["external"] = self.loggers["external"](cfg)
        except Exception as e:
            logger.warning(
                f"Error when initializing logger. "
                f"Disabling custom logging functionality. "
                f"Please ensure logger configuration is correct and "
                f"you have a stable Internet connection: {e}"
            )
            self.loggers["external"] = DummyLogger(cfg)

    def reset_external(self):
        self.loggers["external"] = DummyLogger()

    def log(self, subset: str, name: str, value: str | float, step: float = None):
        for k, logger in self.loggers.items():
            if "validation_predictions" in name and k == "external":
                continue
            if subset == "internal" and not isinstance(logger, LocalLogger):
                continue
            logger.log(subset=subset, name=name, value=value, step=step)


class ExternalLoggers:
    """ExternalLoggers factory."""

    _loggers = {"None": DummyLogger, "Neptune": NeptuneLogger, "W&B": WandbLogger}

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._loggers.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to ExternalLoggers.

        Args:
            name: external loggers name
        Returns:
            A class to build the ExternalLoggers
        """

        return cls._loggers.get(name, DummyLogger)
