import os

import pandas as pd
from datasets import load_dataset


def prepare_default_dataset_causal_language_modeling(path):
    if os.environ.get("H2O_LLM_STUDIO_DEMO_DATASETS") is None:
        ds = load_dataset("OpenAssistant/oasst2")
        train = ds["train"].to_pandas()
        val = ds["validation"].to_pandas()
        df = pd.concat([train, val], axis=0).reset_index(drop=True)
    else:
        df = pd.read_parquet(
            os.path.join(
                os.environ.get("H2O_LLM_STUDIO_DEMO_DATASETS"),
                "causal_language_modeling.pq",
            )
        )

    df_assistant = df[(df.role == "assistant")].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    df_assistant["output"] = df_assistant["text"].values

    inputs = []
    parent_ids = []
    for _, row in df_assistant.iterrows():
        input = df_prompter.loc[row.parent_id]
        inputs.append(input.text)
        parent_ids.append(input.parent_id)

    df_assistant["instruction"] = inputs
    df_assistant["parent_id"] = parent_ids

    df_assistant = df_assistant[
        ["instruction", "output", "message_id", "parent_id", "lang", "rank"]
    ].rename(columns={"message_id": "id"})

    df_assistant[(df_assistant["rank"] == 0.0) & (df_assistant["lang"] == "en")][
        ["instruction", "output", "id", "parent_id"]
    ].to_parquet(os.path.join(path, "train_full.pq"), index=False)

    df_assistant[df_assistant["lang"] == "en"][
        ["instruction", "output", "id", "parent_id"]
    ].to_parquet(os.path.join(path, "train_full_allrank.pq"), index=False)

    df_assistant[df_assistant["rank"] == 0.0][
        ["instruction", "output", "id", "parent_id"]
    ].to_parquet(os.path.join(path, "train_full_multilang.pq"), index=False)

    df_assistant[["instruction", "output", "id", "parent_id"]].to_parquet(
        os.path.join(path, "train_full_multilang_allrank.pq"), index=False
    )

    return df_assistant[(df_assistant["rank"] == 0.0) & (df_assistant["lang"] == "en")]


def prepare_default_dataset_dpo_modeling() -> pd.DataFrame:
    if os.environ.get("H2O_LLM_STUDIO_DEMO_DATASETS") is None:
        df = load_dataset("Intel/orca_dpo_pairs")["train"].to_pandas()
    else:
        df = pd.read_parquet(
            os.path.join(
                os.environ.get("H2O_LLM_STUDIO_DEMO_DATASETS"), "dpo_modeling.pq"
            )
        )
    return df


def prepare_default_dataset_classification_modeling() -> pd.DataFrame:
    if os.environ.get("H2O_LLM_STUDIO_DEMO_DATASETS") is None:
        df = load_dataset("stanfordnlp/imdb")["train"].to_pandas()
    else:
        df = pd.read_parquet(
            os.path.join(
                os.environ.get("H2O_LLM_STUDIO_DEMO_DATASETS"),
                "classification_modeling.pq",
            )
        )
    return df


def prepare_default_dataset_regression_modeling() -> pd.DataFrame:
    if os.environ.get("H2O_LLM_STUDIO_DEMO_DATASETS") is None:
        df = load_dataset("nvidia/HelpSteer2")["train"].to_pandas()
    else:
        df = pd.read_parquet(
            os.path.join(
                os.environ.get("H2O_LLM_STUDIO_DEMO_DATASETS"),
                "regression_modeling.pq",
            )
        )
    return df
