import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM

from llm_studio.src.models.text_base_model import BaseModel
from llm_studio.src.utils.data_utils import batch_padding

logger = logging.getLogger(__name__)


class Model(BaseModel):
    """
    Model for causal language modeling problem type.
    """

    def __init__(self, cfg: Any):
        """
        Args:
            cfg: config with all the hyperparameters
        """

        super(Model, self).__init__(cfg, AutoModelForCausalLM)

    def generate(self, batch: Dict, cfg: Any, streamer=None):
        output = self.generate_output(batch, cfg, cut_input=True, streamer=streamer)

        return output

    def forward(
        self,
        batch: Dict,
        padding: bool = True,
    ) -> Dict:
        # disable cache if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}
        kwargs = {}

        if self.training and self.cfg.training.use_rlhf:
            kwargs["output_hidden_states"] = True

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
                padding_side=self.cfg.tokenizer._padding_side,
            )

        output = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs,
        )

        if "labels" in batch:
            loss = self.loss_fn(output.logits, batch["labels"])
            outputs["loss"] = loss

        if self.cfg.prediction.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(output.logits, batch["labels"])

        if self.training and self.cfg.training.use_rlhf:
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
