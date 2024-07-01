import hashlib
import os
from typing import Any, Dict, List

import pandas as pd

from llm_studio.src.datasets.conversation_chain_handler import get_conversation_chains
from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.plots.text_causal_language_modeling_plots import (
    create_batch_prediction_df,
    plot_validation_predictions,
)
from llm_studio.src.utils.data_utils import read_dataframe_drop_missing_labels
from llm_studio.src.utils.plot_utils import PlotData
from llm_studio.src.utils.utils import PatchedAttribute


class Plots:
    @classmethod
    def plot_batch(cls, batch, cfg) -> PlotData:
        tokenizer = get_tokenizer(cfg)
        df = create_batch_prediction_df(
            batch,
            tokenizer,
            ids_for_tokenized_text="chosen_input_ids",
            labels_column="chosen_labels",
        )
        path = os.path.join(cfg.output_directory, "batch_viz.parquet")
        df.to_parquet(path)
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
            + str(cfg.dataset.rejected_answer_column)
            + str(cfg.dataset.parent_id_column)
        )
        config_hash = hashlib.md5(config_id.encode()).hexdigest()
        path = os.path.join(
            os.path.dirname(cfg.dataset.train_dataframe),
            f"__meta_info__{config_hash}_data_viz.parquet",
        )
        if os.path.exists(path):
            return PlotData(path, encoding="df")

        df = read_dataframe_drop_missing_labels(cfg.dataset.train_dataframe, cfg)

        conversations_chosen = get_conversation_chains(
            df, cfg, limit_chained_samples=True
        )
        with PatchedAttribute(
            cfg.dataset, "answer_column", cfg.dataset.rejected_answer_column
        ):
            conversations_rejected = get_conversation_chains(
                df, cfg, limit_chained_samples=True
            )

        # Limit to max 15 prompt-conversation-answer rounds
        max_conversation_length = min(
            max(
                [len(conversation["prompts"]) for conversation in conversations_chosen]
            ),
            15,
        )

        conversations_to_display: List = []
        for conversation_length in range(1, max_conversation_length + 1):
            conversations_to_display += [
                (conversation_chosen, conversations_rejected)
                for conversation_chosen, conversations_rejected in zip(
                    conversations_chosen, conversations_rejected
                )
                if len(conversation_chosen["prompts"]) == conversation_length
            ][:5]

        # Convert into a scrollable table by transposing the dataframe
        df_transposed = pd.DataFrame(columns=["Sample Number", "Field", "Content"])

        i = 0
        for sample_number, (conversation_chosen, conversations_rejected) in enumerate(
            conversations_to_display
        ):
            if conversation_chosen["systems"][0] != "":
                df_transposed.loc[i] = [
                    sample_number,
                    "System",
                    conversation_chosen["systems"][0],
                ]
                i += 1
            for prompt, answer_chosen, answer_rejected in zip(
                conversation_chosen["prompts"],
                conversation_chosen["answers"],
                conversations_rejected["answers"],  # type: ignore
            ):
                df_transposed.loc[i] = [
                    sample_number,
                    "Prompt",
                    prompt,
                ]
                i += 1
                if answer_chosen == answer_rejected:
                    df_transposed.loc[i] = [
                        sample_number,
                        "Answer",
                        answer_chosen,
                    ]
                    i += 1
                else:
                    df_transposed.loc[i] = [
                        sample_number,
                        "Answer Chosen",
                        answer_chosen,
                    ]
                    i += 1
                    df_transposed.loc[i] = [
                        sample_number,
                        "Answer Rejected",
                        answer_rejected,
                    ]
                    i += 1

        df_transposed.to_parquet(path)
        return PlotData(path, encoding="df")

    @classmethod
    def plot_validation_predictions(
        cls, val_outputs: Dict, cfg: Any, val_df: pd.DataFrame, mode: str
    ) -> PlotData:
        return plot_validation_predictions(val_outputs, cfg, val_df, mode)
