import html
import os
from typing import Any, Dict

import pandas as pd

from llm_studio.src.dataset.text_utils import get_texts, get_tokenizer
from llm_studio.src.utils.data_utils import (
    read_dataframe_drop_missing_labels,
    sample_indices,
)
from llm_studio.src.utils.plot_utils import (
    PlotData,
    format_for_markdown_visualization,
    get_line_separator_html,
    list_to_markdown_representation,
)


class Plots:
    NUM_TEXTS: int = 20

    @classmethod
    def plot_batch(cls, batch, cfg) -> PlotData:
        tokenizer = get_tokenizer(cfg)

        df = pd.DataFrame(
            {
                "Prompt Text": [
                    tokenizer.decode(input_ids, skip_special_tokens=True)
                    for input_ids in batch["prompt_input_ids"].detach().cpu().numpy()
                ]
            }
        )
        df["Prompt Text"] = df["Prompt Text"].apply(format_for_markdown_visualization)
        if "labels" in batch.keys():
            df["Answer Text"] = [
                tokenizer.decode(
                    [label for label in labels if label != -100],
                    skip_special_tokens=True,
                )
                for labels in batch.get("labels", batch["input_ids"])
                .detach()
                .cpu()
                .numpy()
            ]
        tokens_list = [
            tokenizer.convert_ids_to_tokens(input_ids)
            for input_ids in batch["input_ids"].detach().cpu().numpy()
        ]
        masks_list = [
            [label != -100 for label in labels]
            for labels in batch.get("labels", batch["input_ids"]).detach().cpu().numpy()
        ]
        df["Tokenized Text"] = [
            list_to_markdown_representation(
                tokens, masks, pad_token=tokenizer.pad_token, num_chars=100
            )
            for tokens, masks in zip(tokens_list, masks_list)
        ]
        # limit to 2000 rows, still renders fast in wave
        df = df.iloc[:2000]

        # Convert into a scrollable table by transposing the dataframe
        df_transposed = pd.DataFrame(columns=["Sample Number", "Field", "Content"])
        has_answer = "Answer Text" in df.columns

        for i, row in df.iterrows():
            offset = 2 + int(has_answer)
            df_transposed.loc[i * offset] = [
                i,
                "Prompt Text",
                row["Prompt Text"],
            ]
            if has_answer:
                df_transposed.loc[i * offset + 1] = [
                    i,
                    "Answer Text",
                    row["Answer Text"],
                ]
            df_transposed.loc[i * offset + 1 + int(has_answer)] = [
                i,
                "Tokenized Text",
                row["Tokenized Text"],
            ]

        path = os.path.join(cfg.output_directory, "batch_viz.parquet")
        df_transposed.to_parquet(path)

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
            markup += f"<p><strong>Input Text: </strong>{html.escape(input_text)}</p>\n"
            markup += "\n"
            markup += (
                f"<p><strong>Target Text: </strong>{html.escape(target_text)}</p>\n"
            )
            markup += "\n"
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
                "Input Text": input_texts,
                "Target Text": target_text,
                "Predicted Text": predicted_text,
            }
        )
        df["Input Text"] = df["Input Text"].apply(format_for_markdown_visualization)
        df["Target Text"] = df["Target Text"].apply(format_for_markdown_visualization)
        df["Predicted Text"] = df["Predicted Text"].apply(
            format_for_markdown_visualization
        )

        if val_outputs.get("metrics") is not None:
            df[f"Metric ({cfg.prediction.metric})"] = val_outputs["metrics"]
            df[f"Metric ({cfg.prediction.metric})"] = df[
                f"Metric ({cfg.prediction.metric})"
            ].round(decimals=3)
        if val_outputs.get("explanations") is not None:
            df["Explanation"] = val_outputs["explanations"]

        path = os.path.join(cfg.output_directory, f"{mode}_viz.parquet")
        df.to_parquet(path)
        return PlotData(data=path, encoding="df")
