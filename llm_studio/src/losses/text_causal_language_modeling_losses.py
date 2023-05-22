import logging
from typing import Any, List

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


class Losses:
    """Losses factory."""

    _losses = {
        "CrossEntropy": DenseCrossEntropy,
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
