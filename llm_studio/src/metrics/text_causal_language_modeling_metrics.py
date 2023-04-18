import logging
from functools import partial
from typing import Any, Dict, List

import numpy as np
import openai
import pandas as pd
from joblib import Parallel, delayed
from sacrebleu import BLEU
from sacrebleu.metrics.base import Metric

from llm_studio.src.datasets.text_utils import get_texts

logger = logging.getLogger(__name__)


def sacrebleu_score(
    cfg: Any, results: Dict, val_df: pd.DataFrame, metric: Metric
) -> float:
    scores = []
    for predicted_text, target_text in zip(
        results["predicted_text"], results["target_text"]
    ):
        scores.append(metric.sentence_score(predicted_text, [target_text]).score)
    return np.mean(scores)


def rate_reply(question, reference_answer, assistant_answer, model):
    # motivated by https://github.com/lm-sys/FastChat/tree/main/fastchat/eval
    template = open("prompts/eval_template.txt", "r").read()

    template = template.format(
        question=question,
        reference_answer=reference_answer,
        assistant_answer=assistant_answer,
    )

    for _ in range(3):
        try:
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
                temperature=0.1,
                max_tokens=1024,
            )
            ret = response["choices"][0]["message"]["content"]
            ret = ret.split("\n")
            score = ret[0]
            score = score.lower().replace("score:", "").strip()
            score = float(score)
            return score, " ".join(ret[1:]).strip()
        except Exception:
            pass

    logger.warning("error in api call")
    return 0.0, ""


def gpt_score(
    cfg: Any,
    results: Dict,
    val_df: pd.DataFrame,
    model: str = "gpt-3.5-turbo",
    raw_results: bool = False,
) -> float:
    if "metrics" in results:
        return np.mean(results["metrics"].detach().cpu().numpy())
    prompts = get_texts(val_df, cfg, separator="")

    ret = Parallel(n_jobs=len(prompts), backend="multiprocessing")(
        delayed(rate_reply)(prompt, target_text, predicted_text, model)
        for prompt, predicted_text, target_text in zip(
            prompts, results["predicted_text"], results["target_text"]
        )
    )
    scores = [x[0] for x in ret]
    explanations = [x[1] for x in ret]

    if raw_results:
        return scores, explanations
    return np.mean(scores)


class Metrics:
    """Metrics factory. Returns metric value and should it be maximized or minimized"""

    _metrics = {
        "BLEU": (partial(sacrebleu_score, metric=BLEU(effective_order=True)), "max"),
        "GPT3.5": (partial(gpt_score, model="gpt-3.5-turbo"), "max"),
        "GPT4": (partial(gpt_score, model="gpt-4"), "max"),
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
