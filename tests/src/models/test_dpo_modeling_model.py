import random
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import pandas as pd
import pytest
import torch
import torch.nn as nn

from llm_studio.python_configs.text_causal_language_modeling_config import (
    ConfigNLPCausalLMPrediction,
    ConfigNLPCausalLMTokenizer,
)
from llm_studio.python_configs.text_dpo_modeling_config import (
    ConfigDPODataset,
    ConfigProblemBase,
)
from llm_studio.src.datasets.text_dpo_modeling_ds import CustomDataset
from llm_studio.src.models.text_dpo_modeling_model import Model
from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.train import run_eval


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ConfigCausalLMProblemBase(
        llm_backbone="h2oai/llama2-0b-unit-test",
        dataset=ConfigNLPCausalLMDataset(
            system_column="system",
            prompt_column=("prompt",),
            answer_column="answer_column",
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=512),
    )
    cfg.architecture.backbone_dtype = "float32"

    dataset = CausalLMCustomDataset(df, cfg, mode="train")
    model = CausalLMModel(cfg).to(device).eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    batch_padding(
        cfg,
        batch,
        mask_key="prompt_attention_mask",
        pad_keys=[
            "prompt_input_ids",
            "prompt_attention_mask",
            "prompt_special_tokens_mask",
        ],
    )
    with torch.no_grad():
        generated_text = dataset.tokenizer.decode(model.generate(batch, cfg)[0])

    return generated_text


def test_generation_is_the_same_as_for_causal_language_modeling(df):
    """
    DPO model should generate the same output text as causal language modeling
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generated_text_causal_lm = generate_causal_lm_model_text(df)

    cfg = ConfigProblemBase(
        llm_backbone="h2oai/llama2-0b-unit-test",
        dataset=ConfigDPODataset(
            system_column="system",
            prompt_column=("prompt",),
            answer_column="answer_column",
            rejected_answer_column="rejected_answer",
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=512),
    )
    cfg.architecture.backbone_dtype = "float32"

    dataset = CustomDataset(df, cfg, mode="train")
    model = Model(cfg).eval().to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    batch_padding(
        cfg,
        batch,
        mask_key="prompt_attention_mask",
        pad_keys=[
            "prompt_input_ids",
            "prompt_attention_mask",
            "prompt_special_tokens_mask",
        ],
    )
    with torch.no_grad():
        generated_text = dataset.tokenizer.decode(model.generate(batch, cfg)[0])

    assert (
        generated_text == generated_text_causal_lm
    ), "Generated text is not the same as from causal LM model:" "{}\n{}".format(
        generated_text, generated_text_causal_lm
    )


@pytest.fixture
def df2():
    # create a list of all lowercase letters
    alphabet = [chr(i) for i in range(97, 123)]

    # create random strings from the alphabet
    prompts = ["".join(random.choice(alphabet) for _ in range(10)) for _ in range(10)]
    systems = ["".join(random.choice(alphabet) for _ in range(10)) for _ in range(10)]
    answers = ["".join(random.choice(alphabet) for _ in range(10)) for _ in range(10)]
    rejected_answers = [
        "".join(random.choice(alphabet) for _ in range(10)) for _ in range(10)
    ]

    return pd.DataFrame(
        {
            "prompt": prompts,
            "system": systems,
            "answer": answers,
            "rejected_answer": rejected_answers,
        }
    )


def test_dpo_perplexity_metric(tmp_path, df2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ConfigProblemBase(
        output_directory=str(tmp_path),
        llm_backbone="MaxJeblick/llama2-0b-unit-test",
        dataset=ConfigDPODataset(
            system_column="system",
            prompt_column=("prompt",),
            answer_column="answer_column",
            rejected_answer_column="answer_column",
        ),
        tokenizer=ConfigNLPCausalLMTokenizer(max_length=512),
        prediction=ConfigNLPCausalLMPrediction(metric="Perplexity"),
    )
    cfg.architecture.gradient_checkpointing = False
    cfg.environment._device = device  # type: ignore

    # bfloat16 is not supported on older GPUs
    cfg.environment.mixed_precision_dtype = "float16"

    dataset = CustomDataset(df2, cfg, mode="train")
    model = Model(cfg).eval().to(device)
    vocab_size = model.backbone.config.vocab_size

    class MockBackbone(nn.Module):
        """
        Chosen and rejected logits are the same
        Chosen reference and rejected reference logits are the same,
        but different from chosen and rejected logits.
        As answer_column and rejected_answer_column are the same,

          -> perplexity and rejection_perplexity should be the same
          -> chosen_rewards and rejected_rewards should be the same
          -> chosen_cross_entropy and rejected_cross_entropy should be the same
          -> reward margin should be 0
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seed = 0

        def disable_adapter(self):
            # mock lora adapter
            @contextmanager
            def flip_seed():
                self.seed = 1
                yield None
                self.seed = 0

            return flip_seed()

        def forward(self, input_ids, attention_mask):
            @dataclass
            class Result:
                bs, seq_len = input_ids.shape
                torch.manual_seed(self.seed)
                logits = torch.rand((bs, seq_len, vocab_size)).to(input_ids.device)

            result = Result()
            return result

    class ListLogger:
        def __init__(self):
            self.logs = {}

        def log(self, subset: str, name: str, value: str | float, step: float = None):
            self.logs[name] = self.logs.get(name, []) + [value]

    with patch.object(target=model, attribute="backbone", new_callable=MockBackbone):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        # mock cfg.logging._logger.log
        cfg.logging._logger = ListLogger()

        run_eval(
            cfg,
            model=model,
            val_dataloader=dataloader,
            val_df=df2,
            mode="validation",
        )

    log_dict = cfg.logging._logger.logs
    assert log_dict["Perplexity"] == log_dict["rejected_perplexity"]
    assert log_dict["chosen_rewards"] == log_dict["rejected_rewards"]
    assert (
        log_dict["chosen_cross_entropy_loss"] == log_dict["rejected_cross_entropy_loss"]
    )
    assert log_dict["reward_margin"] == [0] * len(log_dict["reward_margin"])
