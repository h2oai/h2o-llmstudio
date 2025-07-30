"""
Loss Implementation based upon
https://github.com/eric-mitchell/direct-preference-optimization
https://github.com/huggingface/trl
"""

import logging
from collections.abc import KeysView
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

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
        self.requires_reference_model = True

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
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


class DPOHingeLoss(DPOLoss):
    def get_losses(self, logits):
        losses = torch.relu(1 - self.cfg.training.beta * logits)
        return losses


class DPOIPOLoss(DPOLoss):
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


class KTOPairLoss(nn.Module):
    """
    Implements original paired KTO implementation
    Adopted from https://github.com/ContextualAI/HALOs
    and https://github.com/huggingface/trl
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.requires_reference_model = True

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (
            (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)
        )

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        losses = torch.cat(
            (
                1
                - F.sigmoid(self.cfg.training.beta * (chosen_logratios - rejected_KL)),
                1
                - F.sigmoid(self.cfg.training.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )

        chosen_rewards = (
            self.cfg.training.beta
            * (policy_chosen_logps - reference_chosen_logps).detach()
        ).float()
        rejected_rewards = (
            self.cfg.training.beta
            * (policy_rejected_logps - reference_rejected_logps).detach()
        ).float()

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


class CPOLoss(nn.Module):
    """
    Implements CPO Loss https://arxiv.org/abs/2401.08417
    Adopted from https://github.com/huggingface/trl
    """

    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.requires_reference_model = False

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ):
        logits = policy_chosen_logps - policy_rejected_logps

        losses = self.get_losses(logits)

        chosen_rewards = (self.cfg.training.beta * policy_chosen_logps.detach()).float()
        rejected_rewards = (
            self.cfg.training.beta * policy_rejected_logps.detach()
        ).float()

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def get_losses(self, logits):
        label_smoothing = 0

        losses = (
            -F.logsigmoid(self.cfg.training.beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-self.cfg.training.beta * logits) * label_smoothing
        )
        return losses


class SimPOLoss(CPOLoss):
    """
    Implements SimPO Loss https://arxiv.org/abs/2405.14734
    Adopted from https://github.com/princeton-nlp/SimPO
    and https://github.com/huggingface/trl
    """

    def get_losses(self, logits):
        label_smoothing = 0
        gamma = self.cfg.training.simpo_gamma
        gamma_logratios = gamma / self.cfg.training.beta
        logits = logits - gamma_logratios

        losses = (
            -F.logsigmoid(self.cfg.training.beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-self.cfg.training.beta * logits) * label_smoothing
        )
        return losses


class Losses:
    """Losses factory."""

    _losses = {
        "DPOLoss": DPOLoss,
        "DPOHingeLoss": DPOHingeLoss,
        "DPOIPOLoss": DPOIPOLoss,
        "KTOPairLoss": KTOPairLoss,
        "CPOLoss": CPOLoss,
        "SimPOLoss": SimPOLoss,
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
LOSS_REDUCTION = {
    "DPOLoss": False,
    "KTOPairLoss": False,
    "DPOHingeLoss": True,
    "DPOIPOLoss": True,
    "CPOLoss": False,
    "SimPOLoss": True,
}
