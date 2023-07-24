import dataclasses
from functools import partial
from typing import Any, Optional, List, Tuple, Type, Union

from h2o_wave import Q, ui

from app_utils.default_config import default_cfg
from app_utils.utils import make_label, get_dataset
from llm_studio.src import possible_values
from llm_studio.src.utils.data_utils import read_dataframe
from llm_studio.src.utils.type_annotations import KNOWN_TYPE_ANNOTATIONS


def get_dataset_elements(cfg: Any, q: Q) -> List:
    """For a given configuration setting return the according dataset ui components.

    Args:
        cfg: configuration settings
        q: Q

    Returns:
        List of ui elements
    """

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()

    cfg_dict = {key: cfg_dict[key] for key in cfg._get_order()}

    items = []
    for k, v in cfg_dict.items():
        # Show some fields only during dataset import
        if k.startswith("_") or cfg._get_visibility(k) == -1:
            continue

        if not (
            _check_dependencies(
                cfg=cfg, pre="dataset/import", k=k, q=q, dataset_import=True
            )
        ):
            continue
        tooltip = cfg._get_tooltips(k)

        trigger = False
        if k in default_cfg.dataset_trigger_keys or k == "data_format":
            trigger = True

        if type_annotations[k] in KNOWN_TYPE_ANNOTATIONS:
            if k in default_cfg.dataset_keys:
                dataset = cfg_dict.copy()
                dataset["path"] = q.client["dataset/import/path"]

                for kk, vv in q.client["dataset/import/cfg"].__dict__.items():
                    dataset[kk] = vv

                for trigger_key in default_cfg.dataset_trigger_keys:
                    if q.client[f"dataset/import/cfg/{trigger_key}"] is not None:
                        dataset[trigger_key] = q.client[
                            f"dataset/import/cfg/{trigger_key}"
                        ]
                if (
                    q.client["dataset/import/cfg/data_format"] is not None
                    and k == "data_format"
                ):
                    v = q.client["dataset/import/cfg/data_format"]

                dataset["dataframe"] = q.client["dataset/import/cfg/dataframe"]

                type_annotation = type_annotations[k]
                poss_values, v = cfg._get_possible_values(
                    field=k,
                    value=v,
                    type_annotation=type_annotation,
                    mode="train",
                    dataset_fn=lambda k, v: (
                        dataset,
                        dataset[k] if k in dataset else v,
                    ),
                )

                if k == "train_dataframe" and v != "None":
                    q.client["dataset/import/cfg/dataframe"] = read_dataframe(v)

                q.client[f"dataset/import/cfg/{k}"] = v

                t = _get_ui_element(
                    k,
                    v,
                    poss_values,
                    type_annotation,
                    tooltip=tooltip,
                    password=False,
                    trigger=trigger,
                    q=q,
                    pre="dataset/import/cfg/",
                )
            else:
                t = []
        elif dataclasses.is_dataclass(v):
            elements_group = get_dataset_elements(cfg=v, q=q)
            t = elements_group
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

        items += t

    return items


def get_ui_elements(
    cfg: Any,
    q: Q,
    limit: Optional[List[str]] = None,
    pre: str = "experiment/start",
) -> List:
    """For a given configuration setting return the according ui components.

    Args:
        cfg: configuration settings
        q: Q
        limit: optional list of keys to limit
        pre: prefix for client keys
        parent_cfg: parent config class.

    Returns:
        List of ui elements
    """
    items = []

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()

    cfg_dict = {key: cfg_dict[key] for key in cfg._get_order()}

    for k, v in cfg_dict.items():
        if "api" in k:
            password = True
        else:
            password = False

        if k.startswith("_") or cfg._get_visibility(k) < 0:
            if q.client[f"{pre}/cfg_mode/from_cfg"]:
                q.client[f"{pre}/cfg/{k}"] = v
            continue
        else:
            type_annotation = type_annotations[k]
            poss_values, v = cfg._get_possible_values(
                field=k,
                value=v,
                type_annotation=type_annotation,
                mode=q.client[f"{pre}/cfg_mode/mode"],
                dataset_fn=partial(get_dataset, q=q, limit=limit, pre=pre),
            )

            if k in default_cfg.dataset_keys:
                # reading dataframe
                if k == "train_dataframe" and (v != ""):
                    q.client[f"{pre}/cfg/dataframe"] = read_dataframe(v, meta_only=True)
                q.client[f"{pre}/cfg/{k}"] = v
            elif k in default_cfg.dataset_extra_keys:
                _, v = get_dataset(k, v, q=q, limit=limit, pre=pre)
                q.client[f"{pre}/cfg/{k}"] = v
            elif q.client[f"{pre}/cfg_mode/from_cfg"]:
                q.client[f"{pre}/cfg/{k}"] = v
        # Overwrite current default values with user_settings
        if q.client[f"{pre}/cfg_mode/from_default"] and f"default_{k}" in q.client:
            q.client[f"{pre}/cfg/{k}"] = q.client[f"default_{k}"]

        if not (_check_dependencies(cfg=cfg, pre=pre, k=k, q=q)):
            continue

        if not _is_visible(k=k, cfg=cfg, q=q):
            if type_annotation not in KNOWN_TYPE_ANNOTATIONS:
                _ = get_ui_elements(cfg=v, q=q, limit=limit, pre=pre)
            elif q.client[f"{pre}/cfg_mode/from_cfg"]:
                q.client[f"{pre}/cfg/{k}"] = v

            continue

        tooltip = cfg._get_tooltips(k)

        trigger = False
        q.client[f"{pre}/trigger_ks"] = ["train_dataframe"]
        q.client[f"{pre}/trigger_ks"] += cfg._get_nesting_triggers()
        if k in q.client[f"{pre}/trigger_ks"]:
            trigger = True

        if type_annotation in KNOWN_TYPE_ANNOTATIONS:
            if limit is not None and k not in limit:
                continue

            t = _get_ui_element(
                k=k,
                v=v,
                poss_values=poss_values,
                type_annotation=type_annotation,
                tooltip=tooltip,
                password=password,
                trigger=trigger,
                q=q,
                pre=f"{pre}/cfg/",
            )
        elif dataclasses.is_dataclass(v):
            if limit is not None and k in limit:
                elements_group = get_ui_elements(cfg=v, q=q, limit=None, pre=pre)
            else:
                elements_group = get_ui_elements(cfg=v, q=q, limit=limit, pre=pre)

            if k == "dataset" and pre != "experiment/start":
                # get all the datasets available
                df_datasets = q.client.app_db.get_datasets_df()
                if not q.client[f"{pre}/dataset"]:
                    if len(df_datasets) >= 1:
                        q.client[f"{pre}/dataset"] = str(df_datasets["id"].iloc[-1])
                    else:
                        q.client[f"{pre}/dataset"] = "1"

                elements_group = [
                    ui.dropdown(
                        name=f"{pre}/dataset",
                        label="Dataset",
                        required=True,
                        value=q.client[f"{pre}/dataset"],
                        choices=[
                            ui.choice(str(row["id"]), str(row["name"]))
                            for _, row in df_datasets.iterrows()
                        ],
                        trigger=True,
                        tooltip=tooltip,
                    )
                ] + elements_group

            if len(elements_group) > 0:
                t = [
                    ui.separator(
                        name=k + "_expander", label=make_label(k, appendix=" settings")
                    )
                ]
            else:
                t = []

            t += elements_group
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

        items += t

    q.client[f"{pre}/prev_dataset"] = q.client[f"{pre}/dataset"]

    return items


def parse_ui_elements(
    cfg: Any, q: Q, limit: Union[List, str] = "", pre: str = ""
) -> Any:
    """Sets configuration settings with arguments from app

    Args:
        cfg: configuration
        q: Q
        limit: optional list of keys to limit
        pre: prefix for keys

    Returns:
        Configuration with settings overwritten from arguments
    """

    cfg_dict = cfg.__dict__
    type_annotations = cfg.get_annotations()
    for k, v in cfg_dict.items():
        if k.startswith("_") or cfg._get_visibility(k) == -1:
            continue

        if (
            len(limit) > 0
            and k not in limit
            and type_annotations[k] in KNOWN_TYPE_ANNOTATIONS
        ):
            continue

        elif type_annotations[k] in KNOWN_TYPE_ANNOTATIONS:
            value = q.client[f"{pre}{k}"]

            if type_annotations[k] == Tuple[str, ...]:
                if isinstance(value, str):
                    value = [value]
                value = tuple(value)
            if type_annotations[k] == str and type(value) == list:
                # fix for combobox outputting custom values as list in wave 0.22
                value = value[0]
            setattr(cfg, k, value)
        elif dataclasses.is_dataclass(v):
            setattr(cfg, k, parse_ui_elements(cfg=v, q=q, limit=limit, pre=pre))
        else:
            raise _get_type_annotation_error(v, type_annotations[k])

    return cfg


def _get_ui_element(
    k: str,
    v: Any,
    poss_values: Any,
    type_annotation: Type,
    tooltip: str,
    password: bool,
    trigger: bool,
    q: Q,
    pre: str = "",
) -> Any:
    """Returns a single ui element for a given config entry

    Args:
        k: key
        v: value
        poss_values: possible values
        type_annotation: type annotation
        tooltip: tooltip
        password: flag for whether it is a password
        trigger: flag for triggering the element
        q: Q
        pre: optional prefix for ui key
        get_default: flag for whether to get the default values

    Returns:
        Ui element

    """
    assert type_annotation in KNOWN_TYPE_ANNOTATIONS

    # Overwrite current values with values from yaml
    if pre == "experiment/start/cfg/":
        if q.args["experiment/upload_yaml"] and "experiment/yaml_data" in q.client:
            if (k in q.client["experiment/yaml_data"].keys()) and (
                k != "experiment_name"
            ):
                q.client[pre + k] = q.client["experiment/yaml_data"][k]

    if type_annotation in (int, float):
        if not isinstance(poss_values, possible_values.Number):
            raise ValueError(
                "Type annotations `int` and `float` need a `possible_values.Number`!"
            )

        val = q.client[pre + k] if q.client[pre + k] is not None else v

        min_val = (
            type_annotation(poss_values.min) if poss_values.min is not None else None
        )
        max_val = (
            type_annotation(poss_values.max) if poss_values.max is not None else None
        )

        # Overwrite default maximum values with user_settings
        if f"set_max_{k}" in q.client:
            max_val = q.client[f"set_max_{k}"]

        if isinstance(poss_values.step, (float, int)):
            step_val = type_annotation(poss_values.step)
        elif poss_values.step == "decad" and val < 1:
            step_val = 10 ** -len(str(int(1 / val)))
        else:
            step_val = 1

        if min_val is None or max_val is None:
            t = [
                # TODO: spinbox `trigger` https://github.com/h2oai/wave/pull/598
                ui.spinbox(
                    name=pre + k,
                    label=make_label(k),
                    value=val,
                    # TODO: open issue in wave to make spinbox optionally unbounded
                    max=max_val if max_val is not None else 1e12,
                    min=min_val if min_val is not None else -1e12,
                    step=step_val,
                    tooltip=tooltip,
                )
            ]
        else:
            t = [
                ui.slider(
                    name=pre + k,
                    label=make_label(k),
                    value=val,
                    min=min_val,
                    max=max_val,
                    step=step_val,
                    tooltip=tooltip,
                    trigger=trigger,
                )
            ]
    elif type_annotation == bool:
        val = q.client[pre + k] if q.client[pre + k] is not None else v

        t = [
            ui.toggle(
                name=pre + k,
                label=make_label(k),
                value=val,
                tooltip=tooltip,
                trigger=trigger,
            )
        ]
    elif type_annotation in (str, Tuple[str, ...]):
        if poss_values is None:
            val = q.client[pre + k] if q.client[pre + k] is not None else v

            title_label = make_label(k)

            t = [
                ui.textbox(
                    name=pre + k,
                    label=title_label,
                    value=val,
                    required=False,
                    password=password,
                    tooltip=tooltip,
                    trigger=trigger,
                    multiline=False,
                )
            ]
        else:
            if isinstance(poss_values, possible_values.String):
                options = poss_values.values
                allow_custom = poss_values.allow_custom
                placeholder = poss_values.placeholder
            else:
                options = poss_values
                allow_custom = False
                placeholder = None

            is_tuple = type_annotation == Tuple[str, ...]

            if is_tuple and allow_custom:
                raise TypeError(
                    "Multi-select (`Tuple[str, ...]` type annotation) and"
                    " `allow_custom=True` is not supported at the same time."
                )

            v = q.client[pre + k] if q.client[pre + k] is not None else v
            if isinstance(v, str):
                v = [v]

            # `v` might be a tuple of strings here but Wave only accepts lists
            v = list(v)

            if allow_custom:
                if not all(isinstance(option, str) for option in options):
                    raise ValueError(
                        "Combobox cannot handle (value, name) pairs for options."
                    )

                t = [
                    ui.combobox(
                        name=pre + k,
                        label=make_label(k),
                        value=v[0],
                        choices=list(options),
                        tooltip=tooltip,
                    )
                ]
            else:
                choices = [
                    ui.choice(option, option)
                    if isinstance(option, str)
                    else ui.choice(option[0], option[1])
                    for option in options
                ]

                t = [
                    ui.dropdown(
                        name=pre + k,
                        label=make_label(k),
                        value=None if is_tuple else v[0],
                        values=v if is_tuple else None,
                        required=False,
                        choices=choices,
                        tooltip=tooltip,
                        placeholder=placeholder,
                        trigger=trigger,
                    )
                ]

    return t


def _check_dependencies(cfg: Any, pre: str, k: str, q: Q, dataset_import: bool = False):
    """Checks all dependencies for a given key

    Args:
        cfg: configuration settings
        pre: prefix for client keys
        k: key to be checked
        q: Q
        dataset_import: flag whether dependencies are checked in dataset import

    Returns:
        True if dependencies are met
    """

    dependencies = cfg._get_nesting_dependencies(k)

    if dependencies is None:
        dependencies = []
    # Do not respect some nesting during the dataset import
    if dataset_import:
        dependencies = [x for x in dependencies if x.key not in ["validation_strategy"]]
    # Do not respect some nesting during the create experiment
    else:
        dependencies = [x for x in dependencies if x.key not in ["data_format"]]

    if len(dependencies) > 0:
        all_deps = 0
        for d in dependencies:
            if isinstance(q.client[f"{pre}/cfg/{d.key}"], (list, tuple)):
                dependency_values = q.client[f"{pre}/cfg/{d.key}"]
            else:
                dependency_values = [q.client[f"{pre}/cfg/{d.key}"]]

            all_deps += d.check(dependency_values)
        return all_deps > 0

    return True


def _is_visible(k: str, cfg: Any, q: Q) -> bool:
    """Returns a flag whether a given key should be visible on UI.

    Args:
        k: name of the hyperparameter
        cfg: configuration settings,
        q: Q
    Returns:
        List of ui elements
    """

    visibility = 1

    if visibility < cfg._get_visibility(k):
        return False

    return True


def _get_type_annotation_error(v: Any, type_annotation: Type) -> ValueError:
    return ValueError(
        f"Cannot show {v}: not a dataclass"
        f" and {type_annotation} is not a known type annotation."
    )
