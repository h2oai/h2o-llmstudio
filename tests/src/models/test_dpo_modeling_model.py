import pandas as pd
import pytest
import torch

from llm_studio.python_configs.text_dpo_modeling_config import (
    ConfigDPODataset,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_dpo_modeling_ds import CustomDataset
from llm_studio.src.models.text_dpo_modeling_model import Model

need_gpus = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU only test")


@pytest.fixture
def df():
    prompt = """when ordering your sandstones, you select which colour scale you would want.
 it could be e.g. a 100% from grey/sand mix, or 80% fra beige/yellow mixed with 20% from black/brown.
  This is all lower case. Can you fix that?"""
    system = """You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. 
While performing the task think step-by-step and justify your steps."""
    answer = """When ordering your sandstones, you select which color scale you would want. It could be, for example, a 100% from grey/sand mix, or 80% from beige/yellow mixed with 20% from black/brown.

Step 1: Capitalize the first letter of the sentence.

Step 2: Correct the spelling of "color" (assuming American English usage).

Step 3: Replace ", e.g." with "for example" to clarify the sentence.

Step 4: Capitalize "a" in "100% from a grey/sand mix"

Step 5: Ensure the proper usage of words and punctuation throughout the revised sentence."""
    return pd.DataFrame(
        {
            "prompt": [prompt],
            "system": [system],
            "answer": [answer],
            "rejected_answer": ["I cannot do that."],
        }
    )


def generate_causal_lm_model_text(df):
    from llm_studio.python_configs.text_causal_language_modeling_config import (
        ConfigNLPCausalLMDataset,
    )
    from llm_studio.python_configs.text_causal_language_modeling_config import (
        ConfigProblemBase as ConfigCausalLMProblemBase,
    )
    from llm_studio.src.datasets.text_causal_language_modeling_ds import (
        CustomDataset as CausalLMCustomDataset,
    )
    from llm_studio.src.models.text_causal_language_modeling_model import (
        Model as CausalLMModel,
    )

    cfg = ConfigCausalLMProblemBase(
        llm_backbone="h2oai/h2ogpt-4096-llama2-7b-chat",
        dataset=ConfigNLPCausalLMDataset(
            system_column="system",
            prompt_column=("prompt",),
            answer_column="answer_column",
        ),
    )
    dataset = CausalLMCustomDataset(df, cfg, mode="train")
    model = CausalLMModel(cfg).cuda().eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    batch = next(iter(dataloader))
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        generated_text = model.generate(batch, cfg)

    return generated_text


@need_gpus
def test_generation_works(df):
    cfg = ConfigProblemBase(
        llm_backbone="h2oai/h2ogpt-4096-llama2-7b-chat",
        dataset=ConfigDPODataset(
            system_column="system",
            prompt_column=("prompt",),
            answer_column="answer_column",
            rejected_answer_column="rejected_answer",
        ),
    )

    dataset = CustomDataset(df, cfg, mode="train")
    model = Model(cfg).cuda().eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    batch = next(iter(dataloader))
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        generated_text = model.generate(batch, cfg)

    generated_text_causal_lm = generate_causal_lm_model_text(df)
    assert (
        generated_text == generated_text_causal_lm
    ), "Generated text is not the same as from causal LM model:" "{}\n{}".format(
        generated_text, generated_text_causal_lm
    )
