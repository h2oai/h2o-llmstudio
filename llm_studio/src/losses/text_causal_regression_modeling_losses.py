import logging
from typing import Any, KeysView

from torch import Tensor, nn

logger = logging.getLogger(__name__)


class MSELoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.MSELoss()

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.loss_fn(logits, labels)


class MAELoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.L1Loss()

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.loss_fn(logits, labels)


class Losses:
    """Losses factory."""

    _losses = {
        "MSELoss": MSELoss,
        "MAELoss": MAELoss,
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
        return cls._losses.get(name, MSELoss)
