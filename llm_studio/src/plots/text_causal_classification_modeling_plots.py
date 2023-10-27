import os

from llm_studio.src.datasets.text_utils import get_tokenizer
from llm_studio.src.plots.text_causal_language_modeling_plots import (
    Plots as TextCausalLanguageModelingPlots,
)
from llm_studio.src.plots.text_causal_language_modeling_plots import (
    create_batch_prediction_df,
)
from llm_studio.src.utils.plot_utils import (
    PlotData,
)


class Plots(TextCausalLanguageModelingPlots):
    @classmethod
    def plot_batch(cls, batch, cfg) -> PlotData:
        tokenizer = get_tokenizer(cfg)
        df = create_batch_prediction_df(
            batch, tokenizer, ids_for_tokenized_text="prompt_input_ids"
        )
        path = os.path.join(cfg.output_directory, "batch_viz.parquet")
        df.to_parquet(path)
        return PlotData(path, encoding="df")
