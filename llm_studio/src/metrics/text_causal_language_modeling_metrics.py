import logging
import os
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import openai
import pandas as pd
import torch
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sacrebleu import BLEU
from sacrebleu.metrics.base import Metric
from tenacity import retry, stop_after_attempt, wait_random_exponential
from torch import nn
from tqdm import tqdm

from llm_studio.src.datasets.text_utils import get_texts
from llm_studio.src.utils.logging_utils import TqdmToLogger

logger = logging.getLogger(__name__)


def sacrebleu_score(
    cfg: Any, results: Dict, val_df: pd.DataFrame, metric: Metric
) -> NDArray:
    scores = []
    for predicted_text, target_text in zip(
        results["predicted_text"], results["target_text"]
    ):
        if target_text == "":
            score = 0.0
        else:
            score = metric.sentence_score(predicted_text, [target_text]).score
        scores.append(score)
    return np.array(scores)


@retry(
    reraise=True,
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(3),
)
def call_openai_api(template, model, deployment_id=None):
    response = openai.ChatCompletion.create(
        deployment_id=deployment_id,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and precise assistant "
                "for checking the quality of the answer.",
            },
            {
                "role": "user",
                "content": template,
            },
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    ret = response["choices"][0]["message"]["content"]
    try:
        score = float(ret.split("SCORE:")[-1].split()[0])
    except ValueError:
        raise ValueError(f"Could not parse score from response: {ret}")
    return score, ret


def rate_reply(filled_eval_template, model, deployment_id=None):
    try:
        return call_openai_api(filled_eval_template, model, deployment_id)
    except Exception as e:
        logger.warning(f"Exception caught in api call: {e}")
        return 0.0, ""


def gpt_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    raw_results: bool = False,
) -> Union[NDArray, Tuple[NDArray, List[str]]]:
    vdf = val_df.copy()
    vdf["_PROMPT"] = get_texts(val_df, cfg, separator="")
    vdf["_PREDICTED_TEXT"] = results["predicted_text"]
    vdf["_TARGET_TEXT"] = results["target_text"]

    if os.getenv("OPENAI_API_TYPE", "open_ai") == "azure":
        deployment_id = os.getenv("OPENAI_API_DEPLOYMENT_ID")
    else:
        deployment_id = None

    model = cfg.prediction.metric_gpt_model
    template_name = cfg.prediction.metric_gpt_template

    eval_template = open(f"prompts/{template_name}.txt", "r").read()
    vdf["filled_eval_template"] = [
        eval_template.format(**row) for _, row in vdf.iterrows()
    ]

    ret = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(rate_reply)(
            filled_eval_template,
            model,
            deployment_id=deployment_id,
        )
        for filled_eval_template in tqdm(
            vdf["filled_eval_template"].values,
            file=TqdmToLogger(logger, level=logging.INFO),
            desc=f"GPT eval {model} - {template_name}",
            total=len(vdf),
        )
    )
    scores = [x[0] for x in ret]
    explanations = [x[1] for x in ret]

    if raw_results:
        return np.array(scores), explanations
    return np.mean(scores)


class Perplexity(nn.Module):
    def __init__(self, cfg: Any, reduce: bool = True):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        perplexity = torch.exp(perplexity)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity


def perplexity(cfg: Any, results: Dict, val_df: pd.DataFrame):
    return results["perplexity"].detach().float().cpu().numpy()


class Metrics:
    """
    Metrics factory. Returns:
        - metric value
        - should it be maximized or minimized
        - Reduce function

    Maximized or minimized is needed for early stopping (saving best checkpoint)
    Reduce function to generate a single metric value, usually "mean" or "none"
    """

    _metrics = {
        "Perplexity": (perplexity, "min", "mean"),
        "BLEU": (
            partial(sacrebleu_score, metric=BLEU(effective_order=True)),
            "max",
            "mean",
        ),
        "GPT": (gpt_score, "max", "mean"),
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._metrics.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Metrics.

        Args:
            name: metrics name
        Returns:
            A class to build the Metrics
        """
        return cls._metrics.get(name, cls._metrics["BLEU"])
