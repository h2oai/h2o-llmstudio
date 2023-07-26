import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from llm_studio.src.datasets.text_utils import get_texts, get_tokenizer
from llm_studio.src.utils.data_utils import (
    read_dataframe_drop_missing_labels,
    sample_indices,
)
from llm_studio.src.utils.plot_utils import (
    PlotData,
    format_for_markdown_visualization,
    get_line_separator_html,
    text_to_html,
)


class Plots:
    NUM_TEXTS: int = 20

    @classmethod
    def plot_batch(cls, batch, cfg) -> PlotData:
        tokenizer = get_tokenizer(cfg)

        df = pd.DataFrame(
            dict(
                texts=[
                    tokenizer.decode(input_ids, skip_special_tokens=False)
                    for input_ids in batch["input_ids"].detach().cpu().numpy()
                ]
            )
        )
        df["texts"] = df["texts"].apply(format_for_markdown_visualization)

        df["tokenized_texts"] = [
            f"```\n{np.array(tokenizer.convert_ids_to_tokens(input_ids))}\n```"
            for input_ids in batch["input_ids"].detach().cpu().numpy()
        ]

        if "labels" in batch.keys():
            input_ids_labels = batch["labels"].detach().cpu().numpy()
            input_ids_labels = [
                [input_id for input_id in input_ids if input_id != -100]
                for input_ids in input_ids_labels
            ]

            df["target_texts"] = [
                tokenizer.decode(input_ids, skip_special_tokens=False)
                for input_ids in input_ids_labels
            ]
            df["target_texts"] = df["target_texts"].apply(
                format_for_markdown_visualization
            )

            df["tokenized_target_texts"] = [
                f"```\n{np.array(tokenizer.convert_ids_to_tokens(input_ids))}\n```"
                for input_ids in input_ids_labels
            ]
        path = os.path.join(cfg.output_directory, "batch_viz.parquet")
        df.to_parquet(path)
        return PlotData(path, encoding="df")

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
    def plot_validation_predictions(
        cls, val_outputs: Dict, cfg: Any, val_df: pd.DataFrame, mode: str
    ) -> PlotData:
        assert mode in ["validation"]

        input_texts = get_texts(val_df, cfg, separator="")
        target_text = val_outputs["target_text"]
        if "predicted_text" in val_outputs.keys():
            predicted_text = val_outputs["predicted_text"]
        else:
            predicted_text = [
                "No predictions are generated for the selected metric"
            ] * len(target_text)

        df = pd.DataFrame(
            {
                "input_text": input_texts,
                "target_text": target_text,
                "predicted_text": predicted_text,
            }
        )
        df["input_text"] = df["input_text"].apply(format_for_markdown_visualization)
        df["target_text"] = df["target_text"].apply(format_for_markdown_visualization)
        df["predicted_text"] = df["predicted_text"].apply(
            format_for_markdown_visualization
        )

        if val_outputs.get("metrics") is not None:
            df["metrics"] = val_outputs["metrics"]
            df["metrics"] = df["metrics"].round(decimals=3)
        if val_outputs.get("explanations") is not None:
            df["explanations"] = val_outputs["explanations"]

        path = os.path.join(cfg.output_directory, f"{mode}_viz.parquet")
        df.to_parquet(path)
        return PlotData(data=path, encoding="df")
