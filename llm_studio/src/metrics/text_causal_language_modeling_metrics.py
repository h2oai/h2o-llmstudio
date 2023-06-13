import logging
from functools import partial
from typing import Any, Dict, List

import numpy as np
import openai
import pandas as pd
from joblib import Parallel, delayed
from sacrebleu import BLEU
from sacrebleu.metrics.base import Metric
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from llm_studio.src.datasets.text_utils import get_texts
from llm_studio.src.utils.logging_utils import TqdmToLogger

logger = logging.getLogger(__name__)


def sacrebleu_score(
    cfg: Any, results: Dict, val_df: pd.DataFrame, metric: Metric
) -> float:
    scores = []
    for predicted_text, target_text in zip(
        results["predicted_text"], results["target_text"]
    ):
        scores.append(metric.sentence_score(predicted_text, [target_text]).score)
    return np.array(scores)


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
def call_openai_api(template, model):
    response = openai.ChatCompletion.create(
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
    ret = ret.split("\n")
    score = ret[0]
    score = score.lower().replace("score:", "").strip()
    score = float(score)
    return score, " ".join(ret[1:]).strip()


def rate_reply(question, reference_answer, assistant_answer, model):
    # motivated by https://github.com/lm-sys/FastChat/tree/main/fastchat/eval
    template = open("prompts/eval_template.txt", "r").read()

    template = template.format(
        question=question,
        reference_answer=reference_answer,
        assistant_answer=assistant_answer,
    )

    try:
        return call_openai_api(template, model)
    except Exception:
        logger.warning("error in api call")
        return 0.0, ""


def gpt_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    model: str = "gpt-3.5-turbo",
    raw_results: bool = False,
) -> float:
    prompts = get_texts(val_df, cfg, separator="")

    ret = Parallel(n_jobs=8, backend="multiprocessing")(
        delayed(rate_reply)(prompt, target_text, predicted_text, model)
        for prompt, predicted_text, target_text in tqdm(
            zip(
                prompts,
                results["predicted_text"],
                results["target_text"],
            ),
            file=TqdmToLogger(logger, level=logging.INFO),
            desc="GPT eval",
            total=len(prompts),
        )
    )
    scores = [x[0] for x in ret]
    explanations = [x[1] for x in ret]

    if raw_results:
        return np.array(scores), explanations
    return np.mean(scores)


def perplexity(cfg: Any, results: Dict, val_df: pd.DataFrame):
    return results["perplexity"].detach().cpu().numpy()


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
        "GPT3.5": (partial(gpt_score, model="gpt-3.5-turbo"), "max", "mean"),
        "GPT4": (partial(gpt_score, model="gpt-4"), "max", "mean"),
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
        return cls._metrics.get(name)

    @classmethod
    def suitable_metrics(cls, cfg: Any, results: Dict, val_df: pd.DataFrame) -> Dict:
        """Access to All Suitable Metrics. For some problem types (e.g. classification)
        there might be metrics (e.g. Micro Averaged F1) that are only suitable in
        specific cases (multiclass not binary). There might also be additional
        metrics returned, which are not possible to select as validation metrics,
        e.g. threshold dependant metrics

        Returns:
            A dictionary of all suitable metrics for current problem setup
        """
        return cls._metrics

    @classmethod
    def all_metrics(cls) -> Dict:
        """Access to All Metrics. There might also be additional
        metrics returned, which are not possible to select as validation metrics,
        e.g. threshold dependant metrics

        Returns:
            A dictionary of all metrics (including not suitable metrics).
        """
        return cls._metrics
