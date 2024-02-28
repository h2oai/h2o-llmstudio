import logging
import os
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from openai import OpenAI, AzureOpenAI
import pandas as pd
import torch
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sacrebleu import BLEU
from sacrebleu.metrics.base import Metric
from torch import nn
from tqdm import tqdm

from llm_studio.src.datasets.text_utils import get_texts
from llm_studio.src.utils.logging_utils import TqdmToLogger

logger = logging.getLogger(__name__)


LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", 3))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", 60))


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


def call_openai_api(template, model, deployment_id=None):
    if os.getenv("OPENAI_API_TYPE", "open_ai") == "azure":
        client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            azure_deployment=os.getenv("OPENAI_API_DEPLOYMENT_ID"),
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
            api_version=os.getenv("OPENAI_API_VERSION", "2023-05-15"),
            # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
            azure_endpoint=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            max_retries=LLM_RETRY_ATTEMPTS,
            timeout=LLM_TIMEOUT,  # unit is seconds
        )
    else:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            max_retries=LLM_RETRY_ATTEMPTS,
            timeout=LLM_TIMEOUT,  # unit is seconds
        )
    response = client.chat.completions.create(
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
        score = float(ret.split("SCORE:")[-1].split()[0].split("/")[0])
    except ValueError:
        raise ValueError(f"Could not parse score from response: {ret}")
    return score, ret


def rate_reply(filled_eval_template, model):
    try:
        return call_openai_api(filled_eval_template, model)
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

    model = cfg.prediction.metric_gpt_model
    template_name = cfg.prediction.metric_gpt_template

    if template_name == "mt-bench":
        eval_template = open("prompts/mt-bench/general.txt", "r").read()
    else:
        eval_template = open(f"prompts/{template_name}.txt", "r").read()
    vdf["filled_eval_template"] = eval_template
    if template_name == "mt-bench":
        eval_template = open("prompts/mt-bench/reference.txt", "r").read()
        vdf.loc[
            vdf.category.isin(["math", "reasoning", "coding"]), "filled_eval_template"
        ] = eval_template

    vdf["filled_eval_template"] = vdf.apply(
        lambda row: row["filled_eval_template"].format(**row), axis=1
    )

    ret = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(rate_reply)(
            filled_eval_template,
            model,
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

    if template_name == "mt-bench":
        vdf["score"] = scores
        score_by_category = vdf.groupby("category").agg({"score": "mean"}).reset_index()
        logger.info(
            "MT-Bench scores by category:\n" + score_by_category.to_string(index=False)
        )

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
