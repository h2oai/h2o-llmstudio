import hashlib
import os
from typing import Any, Dict

import pandas as pd

from llm_studio.src.datasets.conversation_chain_handler import get_conversation_chains
from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.utils.data_utils import read_dataframe_drop_missing_labels
from llm_studio.src.utils.plot_utils import (
    PlotData,
    format_for_markdown_visualization,
    list_to_markdown_representation,
)


class Plots:
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
        """
        Plots the data in a scrollable table.
        We limit the number of rows to max 600 to avoid rendering issues in Wave.
        As the data visualization is instantiated on every page load, we cache the
        data visualization in a parquet file.
        """
        config_id = (
            str(cfg.dataset.train_dataframe)
            + str(cfg.dataset.system_column)
            + str(cfg.dataset.prompt_column)
            + str(cfg.dataset.answer_column)
            + str(cfg.dataset.parent_id_column)
        )
        config_hash = hashlib.md5(config_id.encode()).hexdigest()
        path = os.path.join(
            os.path.dirname(cfg.dataset.train_dataframe),
            f"{config_hash}_data_viz.parquet",
        )
        if os.path.exists(path):
            return PlotData(path, encoding="df")

        df = read_dataframe_drop_missing_labels(cfg.dataset.train_dataframe, cfg)

        conversations = get_conversation_chains(df, cfg, limit_chained_samples=True)

        # Limit to max 15 prompt-conversation-answer rounds
        # This yields to max 5 * sum_{i=1}^{15} i = 600 rows in the DataFrame
        max_conversation_length = min(
            max([len(conversation["prompts"]) for conversation in conversations]), 15
        )

        conversations_to_display = []
        for conversation_length in range(1, max_conversation_length + 1):
            conversations_to_display += [
                conversation
                for conversation in conversations
                if len(conversation["prompts"]) == conversation_length
            ][:5]

        # Convert into a scrollable table by transposing the dataframe
        df_transposed = pd.DataFrame(columns=["Sample Number", "Field", "Content"])

        i = 0
        for sample_number, conversation in enumerate(conversations_to_display):
            if conversation["systems"][0] != "":
                df_transposed.loc[i] = [
                    sample_number,
                    "System",
                    conversation["systems"][0],
                ]
                i += 1
            for prompt, answer in zip(conversation["prompts"], conversation["answers"]):
                df_transposed.loc[i] = [
                    sample_number,
                    "Prompt",
                    prompt,
                ]
                i += 1
                df_transposed.loc[i] = [
                    sample_number,
                    "Answer",
                    answer,
                ]
                i += 1

        df_transposed["Content"] = df_transposed["Content"].apply(
            format_for_markdown_visualization
        )

        df_transposed.to_parquet(path)

        return PlotData(path, encoding="df")

    @classmethod
    def plot_validation_predictions(
        cls, val_outputs: Dict, cfg: Any, val_df: pd.DataFrame, mode: str
    ) -> PlotData:
        conversations = get_conversation_chains(
            val_df, cfg, limit_chained_samples=cfg.dataset.limit_chained_samples
        )

        target_texts = [conversation["answers"][-1] for conversation in conversations]

        input_texts = []
        for conversation in conversations:
            input_text = conversation["systems"][0]
            prompts = conversation["prompts"]
            answers = conversation["answers"]
            # exclude last answer
            answers[-1] = ""
            for prompt, answer in zip(prompts, answers):
                input_text += prompt + answer
            input_texts += [input_text]

        if "predicted_text" in val_outputs.keys():
            predicted_texts = val_outputs["predicted_text"]
        else:
            predicted_texts = [
                "No predictions are generated for the selected metric"
            ] * len(target_texts)

        df = pd.DataFrame(
            {
                "Input Text": input_texts,
                "Target Text": target_texts,
                "Predicted Text": predicted_texts,
            }
        )
        df["Input Text"] = df["Input Text"].apply(format_for_markdown_visualization)
        df["Target Text"] = df["Target Text"].apply(format_for_markdown_visualization)
        df["Predicted Text"] = df["Predicted Text"].apply(
            format_for_markdown_visualization
        )

        if val_outputs.get("metrics") is not None:
            metric_column_name = f"Metric ({cfg.prediction.metric})"
            df[metric_column_name] = val_outputs["metrics"]
            df[metric_column_name] = df[metric_column_name].round(decimals=3)
            if len(df) > 900:
                df.sort_values(by=metric_column_name, inplace=True)
                df = pd.concat(
                    [
                        df.iloc[:300],
                        df.iloc[300:-300].sample(n=300, random_state=42),
                        df.iloc[-300:],
                    ]
                ).reset_index(drop=True)

        elif len(df) > 900:
            df = df.sample(n=900, random_state=42).reset_index(drop=True)

        if val_outputs.get("explanations") is not None:
            df["Explanation"] = val_outputs["explanations"]

        path = os.path.join(cfg.output_directory, f"{mode}_viz.parquet")
        df.to_parquet(path)
        return PlotData(data=path, encoding="df")
