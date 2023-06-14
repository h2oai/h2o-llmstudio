import logging
from typing import Any, List

from torch import nn

__all__ = ["Losses"]


logger = logging.getLogger(__name__)


class TokenAveragedCrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        return self.loss_fn(shift_logits, shift_labels)


class SampleAveragedCrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = 0
        for i in range(labels.shape[0]):
            loss += self.loss_fn(shift_logits[i], shift_labels[i])
        loss /= labels.shape[0]
        return loss


class Losses:
    """Losses factory."""

    _losses = {
        "TokenAveragedCrossEntropy": TokenAveragedCrossEntropyLoss,
        "SampleAveragedCrossEntropy": SampleAveragedCrossEntropyLoss,
    }

    @classmethod
    def names(cls) -> List[str]:
        return cls._losses.keys()

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Losses.

        Args:
            name: losses name
        Returns:
            A class to build the Losses
        """
        return cls._losses.get(name)
