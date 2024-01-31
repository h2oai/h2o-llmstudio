"""
Loss Implementation based upon
https://github.com/eric-mitchell/direct-preference-optimization
"""

import logging
from typing import Any, KeysView, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["Losses"]

logger = logging.getLogger(__name__)


class DPOLoss(nn.Module):
    """
    Implements
    "Direct Preference Optimization:
    Your Language Model is Secretly a Reward Model"
    from https://arxiv.org/abs/2305.18290
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        losses = self.get_losses(logits=pi_logratios - ref_logratios)
        chosen_rewards = (
            self.cfg.training.beta
            * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.cfg.training.beta
            * (policy_rejected_logps - reference_rejected_logps).detach()
        )

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def get_losses(self, logits):
        # The beta is a temperature parameter for the DPO loss,
        # typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0.
        # The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.

        # For now, set label_smoothing to 0 (original DPO loss).
        # See https://ericmitchell.ai/cdpo.pdf for more details
        label_smoothing = 0

        losses = (
            -F.logsigmoid(self.cfg.training.beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-self.cfg.training.beta * logits) * label_smoothing
        )
        return losses


class HingeLoss(DPOLoss):
    def get_losses(self, logits):
        losses = torch.relu(1 - self.cfg.training.beta * logits)
        return losses


class IPOLoss(DPOLoss):
    """
    Implements "A General Theoretical Paradigm
    to Understand Learning from Human Preferences"
    from https://arxiv.org/pdf/2310.12036.pdf
    """

    def get_losses(self, logits):
        # eqn (17) of the https://arxiv.org/pdf/2310.12036.pdf
        # where beta is the real, positive KL parameter for the IPO loss,
        # denoted by tau in the paper (see also eqn (6)).
        losses = (logits - 1 / (2 * self.cfg.training.beta)) ** 2
        return losses


class Losses:
    """Losses factory."""

    _losses = {
        "DPOLoss": DPOLoss,
        "HingeLoss": HingeLoss,
        "IPOLoss": IPOLoss,
    }

    @classmethod
    def names(cls) -> KeysView:
        return cls._losses.keys()

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Losses.
        Args:
            name: losses name
        Returns:
            A class to build the Losses
        """
        return cls._losses.get(name, DPOLoss)


# see https://github.com/huggingface/trl/commit/29d439a2043edf4455b05cae5a1e2ade69d22794
LOSS_REDUCTION = {"DPOLoss": False, "HingeLoss": True, "IPOLoss": True}
