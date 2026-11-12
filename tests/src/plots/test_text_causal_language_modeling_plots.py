import pandas as pd
import pytest

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMDataset,
    ConfigProblemBase,
)
from llm_studio.src.plots.text_causal_language_modeling_plots import (
    plot_validation_predictions,
)


@pytest.mark.parametrize("num_samples", [10, 1000])
def test_plot_validation_predictions_stores_num_samples(tmp_path, num_samples):
    val_df = pd.DataFrame(
        {
            "prompt": [f"prompt {i}" for i in range(num_samples)],
            "answer": [f"answer {i}" for i in range(num_samples)],
        }
    )
    cfg = ConfigProblemBase(
        output_directory=str(tmp_path),
        dataset=ConfigNLPCausalLMDataset(
            prompt_column=("prompt",),
            answer_column="answer",
            parent_id_column="None",
            system_column="None",
        ),
    )
    val_outputs = {"predicted_text": [f"pred {i}" for i in range(num_samples)]}

    plot = plot_validation_predictions(val_outputs, cfg, val_df, "validation")

    df = pd.read_parquet(plot.data)
    assert len(df) == min(num_samples, 900)
    assert df.attrs["num_samples"] == num_samples
