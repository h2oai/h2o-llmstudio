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


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """
    Based upon the official implementation of DPO:
    https://github.com/eric-mitchell/direct-preference-optimization

    Compute the log probabilities of the given labels under the given logits.
    Args:
        logits:
            Logits of the model (unnormalized).
            Shape: (batch_size, sequence_length, vocab_size)
        labels:
            Labels for which to compute the log probabilities.
            Label tokens with a value of -100 are ignored.
            Shape: (batch_size, sequence_length)
        average_log_prob:
            If True, return the average log probability per (non-masked) token.
            Otherwise, return the sum of the
            log probabilities of the (non-masked) tokens.
    Returns:
        A tensor of shape (batch_size,) containing the average/sum
        log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens later
    # Needed to be able to apply torch.gather with index=labels.unsqueeze(2)
    labels[labels == -100] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


class Model(nn.Module):
    """
    Model for DPO language modeling problem type.
    """

    def __init__(self, cfg: Any):
        super().__init__()

        self.cfg = cfg
        self.backbone, self.backbone_config = create_nlp_backbone(
            cfg, model_class=AutoModelForCausalLM
        )

        assert cfg.training.lora, "Need to enable lora for dpo training"
        self.backbone = prepare_lora(cfg=cfg, backbone=self.backbone)

        self.loss_fn = self.cfg.training.loss_class.get(
            self.cfg.training.loss_function
        )(self.cfg)
        if self.cfg.prediction.metric == "Perplexity":
            self.perplexity = Perplexity(self.cfg, reduce=False)

    def generate(self, batch: Dict, cfg: Any, streamer=None):
        return generate(self.backbone, batch, cfg, streamer)

    def forward(
        self,
        batch: Dict,
        padding: bool = True,
    ) -> Dict:
        # disable cache if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}

        chosen_logits = None
        chosen_labels = None

        for answer in ["chosen", "rejected"]:
            if padding:
                batch = batch_padding(
                    self.cfg,
                    batch,
                    self.training,
                    mask_key=f"{answer}_attention_mask",
                    pad_keys=[
                        f"{answer}_input_ids",
                        f"{answer}_attention_mask",
                        f"{answer}_labels",
                    ],
                )
            logits = self.backbone(
                input_ids=batch[f"{answer}_input_ids"],
                attention_mask=batch[f"{answer}_attention_mask"],
            ).logits
            chosen_logits = logits.detach() if answer == "chosen" else chosen_logits
            chosen_labels = (
                batch[f"{answer}_labels"] if answer == "chosen" else chosen_labels
            )

            outputs[f"{answer}_logps"] = get_batch_logps(
                logits, batch[f"{answer}_labels"]
            )

            with self.backbone.disable_adapter():
                with torch.no_grad():
                    reference_logits = self.backbone(
                        input_ids=batch[f"{answer}_input_ids"],
                        attention_mask=batch[f"{answer}_attention_mask"],
                    ).logits
                    outputs[f"{answer}_reference_logps"] = get_batch_logps(
                        reference_logits, batch[f"{answer}_labels"]
                    )

        loss, chosen_rewards, rejected_rewards = self.loss_fn(
            policy_chosen_logps=outputs["chosen_logps"],
            policy_rejected_logps=outputs["rejected_logps"],
            reference_chosen_logps=outputs["chosen_reference_logps"],
            reference_rejected_logps=outputs["rejected_reference_logps"],
        )
        outputs["loss"] = loss

        # These values will be logged to Neptune, if enabled, see train.py
        outputs["chosen_rewards"] = chosen_rewards
        outputs["rejected_rewards"] = rejected_rewards
        outputs["reward_margin"] = chosen_rewards - rejected_rewards

        if not self.training and self.cfg.prediction.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(chosen_logits, chosen_labels)

        # enable cache again if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs
