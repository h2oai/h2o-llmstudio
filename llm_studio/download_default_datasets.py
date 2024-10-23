import os

import pandas as pd
from datasets import load_dataset


def download_default_datasets_to_local_folder() -> None:
    """
    Downloads the default datasets to a local folder.

    The temporary folder is given by the ENV var H2O_LLM_STUDIO_DEMO_DATASETS.
    If the ENV var is not set, this function will raise an error.
    The datasets are transformed to parquet format and saved in the folder.
    """

    path = os.environ.get("H2O_LLM_STUDIO_DEMO_DATASETS")
    if path is None:
        raise ValueError("H2O_LLM_STUDIO_DEMO_DATASETS is not set.")

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # Prepare Causal Language Modeling Dataset
    ds = load_dataset("OpenAssistant/oasst2")
    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas()
    df = pd.concat([train, val], axis=0).reset_index(drop=True)
    df.to_parquet(os.path.join(path, "causal_language_modeling.pq"), index=False)

    # Prepare DPO Modeling Dataset
    df = load_dataset("Intel/orca_dpo_pairs")["train"].to_pandas()
    df.to_parquet(os.path.join(path, "dpo_modeling.pq"), index=False)

    # Prepare Classification Modeling Dataset
    df = load_dataset("stanfordnlp/imdb")["train"].to_pandas()
    df.to_parquet(os.path.join(path, "classification_modeling.pq"), index=False)

    # Prepare Regression Modeling Dataset
    df = load_dataset("nvidia/HelpSteer2")["train"].to_pandas()
    df.to_parquet(os.path.join(path, "regression_modeling.pq"), index=False)


if __name__ == "__main__":
    download_default_datasets_to_local_folder()
