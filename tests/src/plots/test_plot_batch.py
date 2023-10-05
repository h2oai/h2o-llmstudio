from typing import Dict

import pandas as pd
from torch.utils.data import DataLoader

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigNLPCausalLMTokenizer,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_causal_language_modeling_ds import CustomDataset
from llm_studio.src.plots.text_causal_language_modeling_plots import Plots


def test_plot_batch():
    df = pd.DataFrame(
        {
            "prompt": ["prompt 1", "prompt 2", "prompt 3"],
            "answer": ["answer 1", "answer 2", "answer 3"],
            "parent_id": [None, 0, 1],
            "system": ["system 1", "system 2", "system 3"],
            "id": [0, 1, 2],
        }
    )

    cfg = ConfigProblemBase(
        output_directory="my_output_directory",
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="parent_id",
            system_column="system",
            text_system_start="System:",
            text_prompt_start="Prompt:",
            text_answer_separator="Answer:",
            add_eos_token_to_answer=True,
            limit_chained_samples=True,
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=512),
    )

    dataset = CustomDataset(df, cfg)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))

    df_dict: Dict[str, pd.DataFrame] = {}

    def mocked_to_parquet(df, path, *_args, **_kwargs):
        df_dict[path] = df

    pd.DataFrame.to_parquet = mocked_to_parquet

    plot = Plots.plot_batch(batch, cfg)

    df = df_dict[plot.data]
    df.head()
