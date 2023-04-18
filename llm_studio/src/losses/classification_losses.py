import logging
from typing import Any, List

import torch.nn.functional as F

__all__ = ["Losses"]

import torch
from torch import nn

logger = logging.getLogger(__name__)


class DenseCrossEntropy(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x, target):
        return self.loss_fn(x, target)


class SigmoidFocalLoss(nn.Module):
    """
    Sigmoid focal loss for classification
    Source: https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html
    """

    def __init__(self, cfg: Any, gamma=2.0):
        super().__init__()
        self.cfg = cfg
        self.gamma = gamma

    def forward(self, x, target):
        p = torch.sigmoid(x)
        ce_loss = F.binary_cross_entropy_with_logits(x, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # if self.alpha >= 0:
        #     alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        #     loss = alpha_t * loss

        return loss.mean()


class SoftmaxFocalLoss(nn.Module):
    """
    Softmax focal loss for classification
    Source: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8   # noqa
    """

    def __init__(self, cfg: Any, gamma=2.0):
        super().__init__()
        self.cfg = cfg
        self.gamma = gamma

    def forward(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target.argmax(dim=1),
            reduction="mean",
        )


class ClassificationLoss(nn.modules.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        if cfg.environment._local_rank == 0:
            logger.info("Selecting BCE Loss.")
        self.loss_fn = nn.BCEWithLogitsLoss()  # type: ignore

    def forward(self, x, target):
        return self.loss_fn(x, target)


class Losses:
    """Losses factory."""

    _losses = {
        "Classification": ClassificationLoss,
        "BCE": BCEWithLogitsLoss,
        "CrossEntropy": DenseCrossEntropy,
        "SigmoidFocal": SigmoidFocalLoss,
        "SoftmaxFocal": SoftmaxFocalLoss,
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._losses.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Losses.

        Args:
            name: losses name
        Returns:
            A class to build the Losses
        """
        return cls._losses.get(name)
