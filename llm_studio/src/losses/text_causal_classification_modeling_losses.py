import logging
from typing import Any, KeysView

from torch import nn

__all__ = ["Losses"]


logger = logging.getLogger(__name__)


class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels.reshape(-1).long())


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)


class Losses:
    """Losses factory."""

    _losses = {
        "CrossEntropyLoss": CrossEntropyLoss,
        "BinaryCrossEntropyLoss": BinaryCrossEntropyLoss,
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
        return cls._losses.get(name, CrossEntropyLoss)
