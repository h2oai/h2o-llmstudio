import logging
from tkinter import NONE
from typing import Any, Dict

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from llm_studio.src.losses.text_causal_language_modeling_losses import (
    SampleAveragedCrossEntropyLoss,
)
from llm_studio.src.losses.text_dpo_modeling_losses import LOSS_REDUCTION
from llm_studio.src.metrics.text_causal_language_modeling_metrics import Perplexity
from llm_studio.src.utils.data_utils import batch_padding
from llm_studio.src.utils.modeling_utils import (
    create_nlp_backbone,
    forward,
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

    # shift labels and logits to account for next token prediction
    # See also text_causal_language_modeling_losses.py
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100

    # dummy token; we'll ignore the losses on these tokens when loss_mask is applied
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

        if cfg.training.lora:
            self.backbone = prepare_lora(cfg=cfg, backbone=self.backbone)

        self.loss_fn = self.cfg.training.loss_class.get(
            self.cfg.training.loss_function
        )(self.cfg)

        if self.loss_fn.requires_reference_model:
            if cfg.training.lora and not cfg.training.lora_unfreeze_layers:
                self.backbone_reference = None
            else:
                if cfg.environment._local_rank == 0:
                    logger.info("Duplicating backbone for reference model.")
                self.backbone_reference, _ = create_nlp_backbone(
                    cfg, model_class=AutoModelForCausalLM
                )
                for _, param in self.backbone_reference.named_parameters():
                    # freeze base model's layers
                    param.requires_grad = False
                self.backbone_reference = self.backbone_reference.eval()

        if self.cfg.prediction.metric == "Perplexity":
            self.perplexity = Perplexity(self.cfg, reduce=False)

    def generate(self, batch: Dict, cfg: Any, streamer=None):
        return generate(self.backbone, batch, cfg, streamer)

    def forward(
        self,
        batch: Dict,
        padding: bool = True,
    ) -> Dict:
        """
        Forward pass of DPO model.
        Runtime is 4 times slower than causal language modeling model
        as we need to compute
        - logits for chosen answer
        - logits for rejected answer
        - logits for chosen answer with reference model
        - logits for rejected answer with reference model
        """
        # disable cache if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = False

        outputs: Dict = {}

        logits_dict = {}
        labels_dict = {}

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
            logits = forward(
                self.backbone,
                input_ids=batch[f"{answer}_input_ids"],
                attention_mask=batch[f"{answer}_attention_mask"],
            ).logits

            logits_dict[answer] = logits
            labels_dict[answer] = batch[f"{answer}_labels"]

            outputs[f"{answer}_logps"] = get_batch_logps(
                logits,
                batch[f"{answer}_labels"],
                average_log_prob=LOSS_REDUCTION[self.cfg.training.loss_function],
            )

            if self.loss_fn.requires_reference_model:
                with torch.no_grad():
                    if self.backbone_reference:
                        with torch.no_grad():
                            reference_logits = forward(
                                self.backbone_reference,
                                input_ids=batch[f"{answer}_input_ids"],
                                attention_mask=batch[f"{answer}_attention_mask"],
                            ).logits
                    else:
                        with self.backbone.disable_adapter():
                            reference_logits = forward(
                                self.backbone,
                                input_ids=batch[f"{answer}_input_ids"],
                                attention_mask=batch[f"{answer}_attention_mask"],
                            ).logits

                    outputs[f"{answer}_reference_logps"] = get_batch_logps(
                        reference_logits,
                        batch[f"{answer}_labels"],
                        average_log_prob=LOSS_REDUCTION[self.cfg.training.loss_function],
                    )

        if self.loss_fn.requires_reference_model:
            loss, chosen_rewards, rejected_rewards = self.loss_fn(
                policy_chosen_logps=outputs["chosen_logps"],
                policy_rejected_logps=outputs["rejected_logps"],
                reference_chosen_logps=outputs["chosen_reference_logps"],
                reference_rejected_logps=outputs["rejected_reference_logps"],
            )
        else:
            loss, chosen_rewards, rejected_rewards = self.loss_fn(
                policy_chosen_logps=outputs["chosen_logps"],
                policy_rejected_logps=outputs["rejected_logps"],
            )
        outputs["loss"] = loss

        # These values will be logged to Neptune if enabled, see train.py
        outputs["additional_log_chosen_rewards"] = chosen_rewards.detach()
        outputs["additional_log_rejected_rewards"] = rejected_rewards.detach()
        # Reward margin should increase over time
        outputs["additional_log_reward_margin"] = (
            chosen_rewards - rejected_rewards
        ).detach()

        # log sample average cross entropy, perplexity metric is also sample averaged
        outputs["additional_log_chosen_cross_entropy_loss"] = (
            SampleAveragedCrossEntropyLoss(self.cfg)(
                logits_dict["chosen"], labels_dict["chosen"]
            ).detach()
        )
        outputs["additional_log_rejected_cross_entropy_loss"] = (
            SampleAveragedCrossEntropyLoss(self.cfg)(
                logits_dict["rejected"], labels_dict["rejected"]
            ).detach()
        )

        if not self.training and self.cfg.prediction.metric == "Perplexity":
            outputs["perplexity"] = self.perplexity(
                logits_dict["chosen"], labels_dict["chosen"]
            )
            outputs["additional_log_rejected_perplexity"] = self.perplexity(
                logits_dict["rejected"], labels_dict["rejected"]
            )

        # enable cache again if gradient checkpointing is enabled
        if self.cfg.architecture.gradient_checkpointing:
            self.backbone.config.use_cache = True

        return outputs
