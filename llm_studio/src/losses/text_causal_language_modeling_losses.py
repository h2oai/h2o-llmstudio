import logging
from typing import Any, KeysView

__all__ = ["Losses"]
import torch
import torch.nn as nn

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


class LengthBasedTACE(nn.Module):
    def __init__(self, cfg: Any, length_penalty_coeff: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.length_penalty_coeff = length_penalty_coeff

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss = self.loss_fn(shift_logits, shift_labels)

        true_lengths = torch.sum(labels != 0, dim=-1).float()
        pred_lengths = torch.sum(torch.argmax(logits, dim=-1) != 0, dim=-1).float()

        length_ratio = true_lengths / (pred_lengths + 1e-8)

        length_penalty = torch.pow(length_ratio, self.length_penalty_coeff)
        normalized_loss = loss / length_penalty.mean()

        return normalized_loss


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


class LengthBasedSACE(nn.Module):
    def __init__(self, cfg: Any, length_penalty_coeff: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()
        self.length_penalty_coeff = length_penalty_coeff

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        total_loss = 0.0

        for i in range(labels.shape[0]):
            sample_logit = shift_logits[i].view(-1, shift_logits.size(-1))
            sample_label = shift_labels[i].view(-1)
            sample_loss = self.loss_fn(sample_logit, sample_label)

            true_length = torch.sum(labels[i] != 0).float()
            pred_length = torch.sum(torch.argmax(logits[i], dim=-1) != 0).float()
            length_ratio = true_length / (pred_length + 1e-8)
            length_penalty = torch.pow(length_ratio, self.length_penalty_coeff)

            total_loss += sample_loss / length_penalty

        average_loss = total_loss / labels.shape[0]
        return average_loss


class Losses:
    """Losses factory."""

    _losses = {
        "TokenAveragedCrossEntropy": TokenAveragedCrossEntropyLoss,
        "SampleAveragedCrossEntropy": SampleAveragedCrossEntropyLoss,
        "LengthBasedTACE": LengthBasedTACE,
        "LengthBasedSACE": LengthBasedSACE,
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
        return cls._losses.get(name, TokenAveragedCrossEntropyLoss)
