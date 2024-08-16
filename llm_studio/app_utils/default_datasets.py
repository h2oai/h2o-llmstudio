import os
import random
import re
import uuid

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def prepare_default_dataset_causal_language_modeling(path):
    ds = load_dataset("OpenAssistant/oasst2")
    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas()

    df = pd.concat([train, val], axis=0).reset_index(drop=True)

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
    df = load_dataset("Intel/orca_dpo_pairs")["train"].to_pandas()
    return df


def prepare_default_dataset_classification_modeling() -> pd.DataFrame:
    df = load_dataset("stanfordnlp/imdb")["train"].to_pandas()
    return df


def prepare_default_dataset_regression_modeling() -> pd.DataFrame:
    df = load_dataset("nvidia/HelpSteer2")["train"].to_pandas()
    return df


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def _parse_row(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response["chosen"].rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    prompt = prompt_and_response["chosen"][: search_term_idx + len(search_term)]

    chosen_response = prompt_and_response["chosen"][len(prompt) :]
    rejected_response = prompt_and_response["rejected"][len(prompt) :]

    return prompt, chosen_response, rejected_response


def _split_up_prompt(prompt):
    human_texts = re.findall(
        r"\n\nHuman:(.*?)(?=(\n\nAssistant:|$))", prompt, flags=re.DOTALL
    )
    assistant_texts = re.findall(
        r"\n\nAssistant:(.*?)(?=(\n\nHuman:|$))", prompt, flags=re.DOTALL
    )
    human_texts = [text[0].strip() for text in human_texts]
    assistant_texts = [text[0].strip() for text in assistant_texts]

    assert len(human_texts) == len(assistant_texts), prompt
    dialogue = list(zip(human_texts, assistant_texts))
    return dialogue


def prepare_hh_dpo_modeling(split: str) -> pd.DataFrame:
    """
    Adapted from
    https://github.com/eric-mitchell/direct-preference-optimization/blob/main/preference_datasets.py
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    rnd = random.Random()
    rnd.seed(123)
    dfs = []
    for row in tqdm(dataset):
        prompt, chosen_response, rejected_response = _parse_row(row)
        if len(rejected_response) == 0:
            # remove rejected answers that are empty
            continue

        parent_uuid = None
        parsed_texts = []
        for human_text, assistant_text in _split_up_prompt(prompt):
            random_uuid = str(uuid.UUID(int=rnd.getrandbits(128), version=4))
            parsed_texts += [
                [human_text, assistant_text, random_uuid, parent_uuid, None, None]
            ]
            parent_uuid = random_uuid

        parsed_texts[-1][-2] = chosen_response
        parsed_texts[-1][-1] = rejected_response
        df = pd.DataFrame(
            parsed_texts,
            columns=[
                "instruction",
                "output",
                "id",
                "parent_id",
                "chosen_response",
                "rejected_response",
            ],
        )
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    # merge output into chosen and rejected response
    df["chosen_response"] = df["chosen_response"].fillna(df["output"])
    df["rejected_response"] = df["rejected_response"].fillna(df["output"])
    del df["output"]
    return df
