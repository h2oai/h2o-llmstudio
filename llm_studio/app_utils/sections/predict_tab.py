import asyncio
import gc
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from h2o_wave import Q
from h2o_wave import data as chat_data
from h2o_wave import ui
from transformers import AutoTokenizer, TextStreamer

from llm_studio.app_utils.utils import (
    get_experiments,
    get_ui_elements,
    parse_ui_elements,
    set_env,
    start_experiment,
)
from llm_studio.python_configs.text_causal_classification_modeling_config import (
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.models.text_causal_language_modeling_model import Model
from llm_studio.src.utils.config_utils import load_config_yaml
from llm_studio.src.utils.modeling_utils import load_checkpoint

logger = logging.getLogger(__name__)


async def predict_classification_tab(q: Q):
    q.page["experiment/display/predict/settings"] = ui.form_card(
        box="second",
        items=[
            ui.expander(
                name="predict_settings",
                label="Predict Settings",
                items=[ui.progress(label="Loading model configuration...")],
                expanded=True,
            )
        ],
    )
    cfg: ConfigProblemBase = load_config_yaml(
        os.path.join(q.client["experiment/display/experiment_path"], "cfg.yaml")
    )
    q.client["experiment/display/predict/cfg"] = cfg

    option_items = get_ui_elements(
        cfg=q.client["experiment/display/predict/cfg"].prediction,
        q=q,
        pre="chat/cfg_predictions",
    )
    q.page["experiment/display/predict/settings"] = ui.form_card(
        box="second",
        items=[
            ui.expander(
                name="predict_settings",
                label="Predict Settings",
                items=option_items,
                expanded=True,
            ),
            ui.button("Predict", name="experiment/display/predict/start"),
        ],
    )
    q.client.delete_cards.add("experiment/display/predict/settings")


async def run_prediction(q: Q) -> None:
    """
    Update the chatbot with the new message.
    """
    cfg_prediction = parse_ui_elements(
        cfg=q.client["experiment/display/chat/cfg"].prediction,
        q=q,
        pre="chat/cfg_predictions/cfg/",
    )
    logger.info(f"Using prediction config: {cfg_prediction}")
    q.client["experiment/display/predict/cfg"].prediction = cfg_prediction

    cfg: ConfigProblemBase = q.client["experiment/display/predict/cfg"]
    start_experiment(cfg=cfg, q=q, pre=pre)
