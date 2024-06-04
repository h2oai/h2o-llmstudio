import logging
from typing import Any, Dict

from torch import nn
from transformers import AutoModelForCausalLM

from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import (
    create_nlp_backbone,
    get_position_ids,
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
            cfg, model_class=AutoModelForCausalLM
        )

        if cfg.training.lora:
            self.backbone = prepare_lora(cfg, self.backbone)

        self.classification_head = nn.Linear(
            self.backbone_config.vocab_size, cfg.dataset.num_classes, bias=False
        )

        self.loss_fn = self.cfg.training.loss_class.get(
            self.cfg.training.loss_function
        )(self.cfg)

    def forward(
        self,
        batch: Dict,
        padding: bool = True,
    ) -> Dict:
        # disable cache if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}
        mask_key = "prompt_attention_mask"
        pad_keys = [
            "prompt_input_ids",
            "prompt_attention_mask",
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
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            position_ids=get_position_ids(batch["prompt_input_ids"]),
        )

        output.logits = self.classification_head(output[0][:, -1].float())

        if "labels" in batch:
            loss = self.loss_fn(
                output.logits, batch["class_label"].unsqueeze(1).float()
            )
            outputs["loss"] = loss

        outputs["logits"] = output.logits

        # enable cache again if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs
