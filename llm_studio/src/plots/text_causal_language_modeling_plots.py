from typing import Any, Dict

import pandas as pd
from bokeh.models import Div, Panel, Tabs

from llm_studio.src.datasets.text_utils import get_texts, get_tokenizer
from llm_studio.src.utils.data_utils import (
    read_dataframe_drop_missing_labels,
    sample_indices,
)
from llm_studio.src.utils.plot_utils import (
    PlotData,
    color_code_tokenized_text,
    get_best_and_worst_sample_idxs,
    get_line_separator_html,
    text_to_html,
    to_html,
)


class Plots:
    NUM_TEXTS: int = 20

    @classmethod
    def plot_batch(cls, batch, cfg) -> PlotData:
        tokenizer = get_tokenizer(cfg)

        texts = [
            tokenizer.decode(input_ids, skip_special_tokens=True)
            for input_ids in batch["input_ids"].detach().cpu().numpy()
        ]

        tokenized_texts = [
            color_code_tokenized_text(
                tokenizer.convert_ids_to_tokens(input_ids), tokenizer
            )
            for input_ids in batch["input_ids"].detach().cpu().numpy()
        ]

        input_ids_labels = batch["labels"].detach().cpu().numpy()
        input_ids_labels = [
            [input_id for input_id in input_ids if input_id != -100]
            for input_ids in input_ids_labels
        ]

        target_texts = [
            tokenizer.decode(input_ids, skip_special_tokens=True)
            for input_ids in input_ids_labels
        ]

        tokenized_target_texts = [
            color_code_tokenized_text(
                tokenizer.convert_ids_to_tokens(input_ids),
                tokenizer,
            )
            for input_ids in input_ids_labels
        ]
        if cfg.dataset.mask_prompt_labels:
            markup = ""
        else:
            markup = (
                """
            <div padding: 10px;">
            <p style="font-size: 20px;">
            <b>Note:</b> <br> Model is jointly trained on prompt + answer text. If you only want to use 
            the answer text as a target, restart the experiment and enable <i> Mask Prompt Labels </i>
            </p>
            </div>
            """
                + get_line_separator_html()
            )
        for i in range(len(tokenized_texts)):
            markup += f"<p><strong>Input Text: </strong>{texts[i]}</p>\n"
            markup += (
                f"<p><strong>Tokenized Input Text: </strong>{tokenized_texts[i]}</p>\n"
            )
            markup += f"<p><strong>Target Text: </strong>{target_texts[i]}</p>\n"
            markup += (
                f"<p><strong>Tokenized Target Text:"
                f" </strong>{tokenized_target_texts[i]}</p>\n"
            )
            markup += get_line_separator_html()
        return PlotData(markup, encoding="html")

    @classmethod
    def plot_data(cls, cfg) -> PlotData:

        df = read_dataframe_drop_missing_labels(cfg.dataset.train_dataframe, cfg)
        df = df.iloc[sample_indices(len(df), Plots.NUM_TEXTS)]

        input_texts = get_texts(df, cfg, separator="")

        if cfg.dataset.answer_column in df.columns:
            target_texts = df[cfg.dataset.answer_column].values
        else:
            target_texts = ""

        markup = ""
        for input_text, target_text in zip(input_texts, target_texts):
            markup += (
                f"<p><strong>Input Text: </strong>{text_to_html(input_text)}</p>\n"
            )
            markup += "<br/>"
            markup += (
                f"<p><strong>Target Text: </strong>{text_to_html(target_text)}</p>\n"
            )
            markup += "<br/>"
            markup += get_line_separator_html()
        return PlotData(markup, encoding="html")

    @classmethod
    def selection_validation_predictions(
        cls,
        val_outputs: Dict,
        cfg: Any,
        val_df: pd.DataFrame,
        true_labels: Any,
        pred_labels: Any,
        metrics: Any,
        sample_idx: Any,
    ) -> str:
        input_texts = get_texts(val_df, cfg, separator="")
        markup = ""

        for idx in sample_idx:
            input_text = input_texts[idx]
            markup += (
                f"<p><strong>Input Text: </strong>{text_to_html(input_text)}</p>\n"
            )

            if true_labels is not None:
                target_text = true_labels[idx]
                markup += "<br/>"
                markup += (
                    f"<p><strong>Target Text: "
                    f"</strong>{text_to_html(target_text)}</p>\n"
                )

            predicted_text = pred_labels[idx]
            markup += "<br/>"
            markup += (
                f"<p><strong>Predicted Text: </strong>"
                f"{text_to_html(predicted_text)}</p>\n"
            )

            if metrics is not None:
                markup += "<br/>"
                markup += (
                    f"<p><strong>{cfg.prediction.metric} Score: </strong>"
                    f"{metrics[idx]:.3f}"
                )

            if "explanations" in val_outputs:
                markup += "<br/>"
                markup += (
                    f"<p><strong>Explanation: </strong>"
                    f"{val_outputs['explanations'][idx]}"
                )

            if idx != sample_idx[-1]:
                markup += get_line_separator_html()

        return markup

    @classmethod
    def plot_validation_predictions(
        cls, val_outputs: Dict, cfg: Any, val_df: pd.DataFrame, mode: str
    ) -> PlotData:
        assert mode in ["validation"]

        metrics = val_outputs["metrics"].detach().cpu().numpy()
        best_samples, worst_samples = get_best_and_worst_sample_idxs(
            cfg, metrics, n_plots=min(cfg.logging.number_of_texts, len(val_df))
        )
        random_samples = sample_indices(len(val_df), len(best_samples))
        selection_plots = {
            title: cls.selection_validation_predictions(
                val_outputs=val_outputs,
                cfg=cfg,
                val_df=val_df,
                true_labels=(val_outputs["target_text"]),
                pred_labels=(val_outputs["predicted_text"]),
                metrics=metrics,
                sample_idx=indices,
            )
            for (indices, title) in [
                (random_samples, f"Random {mode} samples"),
                (best_samples, f"Best {mode} samples"),
                (worst_samples, f"Worst {mode} samples"),
            ]
        }

        tabs = [
            Panel(
                child=Div(
                    text=markup, sizing_mode="scale_width", style={"font-size": "105%"}
                ),
                title=title,
            )
            for title, markup in selection_plots.items()
        ]
        return PlotData(to_html(Tabs(tabs=tabs)), encoding="html")
