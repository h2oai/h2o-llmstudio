from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXConfig,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
)
from transformers.utils import ModelOutput


class GPTNeoXRewardModelConfig(GPTNeoXConfig):
    model_type = "gpt_neox_reward_model"

    pooling: Literal["mean", "last"]

    def __init__(
        self,
        pooling: Literal["mean", "last"] = "last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pooling = pooling or "last"


@dataclass
class GPTNeoXRewardModelOutput(ModelOutput):
    """
    Reward model output.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Reward score
    """

    logits: torch.FloatTensor = None


class GPTNeoXRewardModel(GPTNeoXPreTrainedModel):
    config_class = GPTNeoXRewardModelConfig

    def __init__(self, config):
        if isinstance(config, GPTNeoXConfig):
            # When a normal GPTNeoX was loaded it will be converted into a reward model.
            # The direct `type(config) == GPTNeoXConfig` comparison is used (instead of
            # `isinstance()`) since the configuration class of the reward model is also
            # derived form `GPTNeoXConfig`.
            config = GPTNeoXRewardModelConfig.from_dict(config.to_dict())
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.pooling = config.pooling

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> GPTNeoXRewardModelOutput:
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pooling == "mean":
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                pooled = (hidden_states * attention_mask).sum(
                    dim=1
                ) / attention_mask.sum(dim=1)
        elif self.pooling == "last":
            if attention_mask is None:
                pooled = hidden_states[:, -1]
            else:
                last_idx = attention_mask.cumsum(dim=1).argmax(dim=1)
                pooled = hidden_states.gather(
                    1, last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
                ).squeeze(1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        logits = self.out_proj(pooled)

        if not return_dict:
            return (logits,) + outputs[1:]

        return GPTNeoXRewardModelOutput(logits=logits)


class RewardModel(nn.Module):
    def __init__(self, cfg):
        super(RewardModel, self).__init__()

        AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
        AutoModelForSequenceClassification.register(
            GPTNeoXRewardModelConfig, GPTNeoXRewardModel
        )

        self.cfg = cfg
        self.model_name = cfg.training.reward_model
        self.device = cfg.environment._device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, max_model_input_sizes=2048
        )

    def get_score(
        self,
        prompts=None,
        answers=None,
    ):
        scores = []
        for prompt, answer in zip(prompts, answers):
            if self.model_name == "OpenAssistant/reward-model-deberta-v3-large-v2":
                inputs = self.tokenizer(
                    " ".join(prompt.split("<|endoftext|>")),
                    answer,
                    return_tensors="pt",
                    max_length=2048,
                ).to(self.device)
            elif self.model_name in [
                "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
                "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
            ]:
                prompt = prompt.split("<|endoftext|>")

                input_text = ""

                for i, prompt_part in enumerate(prompt[::-1]):
                    if i % 2 == 0:
                        prefix = "<|prompter|>"
                    else:
                        prefix = "<|assistant|>"
                    input_text = f"{prefix}{prompt_part}<|endoftext|>" + input_text

                input_text = input_text + f"<|assistant|>{answer}<|endoftext|>"

                inputs = self.tokenizer(
                    input_text, return_tensors="pt", max_length=2048
                ).to(self.device)

            scores.append(self.model(**inputs).logits[0].cpu().detach().item())
            del inputs
        return scores
