import logging
from typing import Any, Dict

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from llm_studio.src.metrics.text_causal_language_modeling_metrics import Perplexity
from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import (
    create_nlp_backbone,
    generate,
    prepare_lora,
)

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """
    The ValueHead class implements a head for GPT2 that returns a scalar for each
    output token.

    Based on the implementation of trl library:
    https://github.com/lvwerra/trl/blob/main/trl/models/modeling_value_head.py
    """

    def __init__(self, config):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = 0.1
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = (
            nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        )

        # some models such as OPT have a projection layer before the word embeddings
        # e.g. OPT-350m
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        output = self.summary(output)
        return output


class Model(nn.Module):
    """
    Model for causal language modeling problem type.
    """

    def __init__(self, cfg: Any):
        """
        Args:
            cfg: config with all the hyperparameters
        """

        super(Model, self).__init__()

        self.cfg = cfg
        assert cfg.training.lora, "LoRA must be True for RLHF"

        self.backbone, self.backbone_config = create_nlp_backbone(
            cfg, model_class=AutoModelForCausalLM
        )

        self.backbone = prepare_lora(cfg=self.cfg, backbone=self.backbone)

        if self.cfg.prediction.metric == "Perplexity":
            self.perplexity = Perplexity(self.cfg, reduce=False)

        self.value_head = ValueHead(self.backbone_config)
        self.value_head.summary.bias.data.zero_()

    def forward(
        self,
        batch: Dict,
        padding: bool = True,
    ) -> Dict:
        # disable cache if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}
        mask_key = "attention_mask"
        pad_keys = [
            "input_ids",
            "attention_mask",
            "special_tokens_mask",
            "labels",
        ]

        if padding:
            batch = batch_padding(
                self.cfg,
                batch,
                self.training,
                mask_key=mask_key,
                pad_keys=pad_keys,
            )

        output = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )

        if self.cfg.prediction.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(output.logits, batch["labels"])

        if self.training:
            last_hidden_state = output.hidden_states[-1]

            # force upcast in fp32 if logits are in half-precision
            if output.logits.dtype != torch.float32:
                output.logits = output.logits.float()

            outputs["logits"] = output.logits
            outputs["value"] = self.value_head(last_hidden_state).squeeze(-1)

        # enable cache again if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs

    def generate(self, batch: Dict, cfg: Any, streamer=None):
        return generate(self.backbone, batch, cfg, streamer)
