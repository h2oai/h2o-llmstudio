import logging
from typing import Any, Dict

import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

from llm_studio.src.metrics.text_causal_language_modeling_metrics import Perplexity
from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import (
    create_nlp_backbone,
    generate,
    prepare_lora,
)

logger = logging.getLogger(__name__)


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
        self.backbone, self.backbone_config = create_nlp_backbone(
            cfg, model_class=AutoModelForSeq2SeqLM
        )

        if cfg.training.lora:
            self.backbone = prepare_lora(cfg, self.backbone)

        self.loss_fn = self.cfg.training.loss_class.get(
            self.cfg.training.loss_function
        )(self.cfg)

        if self.cfg.prediction.metric == "Perplexity":
            self.perplexity = Perplexity(self.cfg, reduce=False)

    def generate(self, batch: Dict, cfg: Any, streamer=None):
        return generate(
            backbone=self.backbone,
            batch=batch,
            cfg=cfg,
            streamer=streamer,
            remove_prompt=False,
        )

    def forward(
        self,
        batch: Dict,
        padding: bool = True,
    ) -> Dict:
        # disable cache if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}
        kwargs: Dict = {}

        if padding:
            mask_key = "prompt_attention_mask"
            pad_keys = [
                "prompt_input_ids",
                "prompt_attention_mask",
            ]

            batch = batch_padding(
                self.cfg,
                batch,
                self.training,
                mask_key=mask_key,
                pad_keys=pad_keys,
                padding_side=self.cfg.tokenizer._padding_side,
            )

            mask_key = "answer_attention_mask"
            pad_keys = [
                "answer_input_ids",
                "answer_attention_mask",
            ]

            batch = batch_padding(
                self.cfg,
                batch,
                self.training,
                mask_key=mask_key,
                pad_keys=pad_keys,
                padding_side="right",
            )

        labels = batch["answer_input_ids"]
        labels[batch["answer_attention_mask"] == 0] = -100

        output = self.backbone(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            labels=labels,
            **kwargs,
        )

        outputs["loss"] = output.loss

        if self.cfg.prediction.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(output.logits, batch["labels"])

        # enable cache again if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs
