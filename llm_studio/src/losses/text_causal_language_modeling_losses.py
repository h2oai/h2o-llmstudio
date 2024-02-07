import logging
from typing import Any, KeysView

import torch
from torch import nn

__all__ = ["Losses"]


logger = logging.getLogger(__name__)


class TokenAveragedCrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, labels):
        logits = output.logits
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

    def forward(self, output, labels):
        logits = output.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = 0
        for i in range(labels.shape[0]):
            loss += self.loss_fn(shift_logits[i], shift_labels[i])
        loss /= labels.shape[0]
        return loss


class MoECrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()

    def load_balancing_loss_func(
        self,
        gate_logits: torch.Tensor,
        labels: torch.Tensor,
        num_experts: torch.Tensor = None,
        top_k: int = 2,
    ) -> float:
        """
        Adjusted from Transformers Library
        https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/mixtral/modeling_mixtral.py#L77 # noqa: E501
        Computes auxiliary load balancing loss as in Switch Transformer

        See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details.
        """
        if gate_logits is None or not isinstance(gate_logits, tuple):
            return 0

        if isinstance(gate_logits, tuple):
            compute_device = gate_logits[0].device
            # get the mask for ignored labels
            mask = labels.view(-1) != -100
            concatenated_gate_logits = torch.cat(
                [layer_gate[mask].to(compute_device) for layer_gate in gate_logits],
                dim=0,
            )

        routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)

        overall_loss = torch.sum(
            tokens_per_expert * router_prob_per_expert.unsqueeze(0)
        )
        return overall_loss * num_experts

    def forward(self, output, labels):
        logits = output.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss = self.loss_fn(shift_logits, shift_labels)
        aux_loss = self.load_balancing_loss_func(
            output.router_logits,
            labels,
            self.cfg.architecture._num_local_experts,
            self.cfg.architecture._num_experts_per_tok,
        )
        loss += self.cfg.training.router_aux_loss_coef * aux_loss
        return loss


class Losses:
    """Losses factory."""

    _losses = {
        "TokenAveragedCrossEntropy": TokenAveragedCrossEntropyLoss,
        "SampleAveragedCrossEntropy": SampleAveragedCrossEntropyLoss,
        "MoECrossEntropy": MoECrossEntropyLoss,
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
