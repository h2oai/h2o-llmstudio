from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.mixtral.modeling_mixtral import MixtralBLockSparseTop2MLP


class MixtralSparseMoeBlock(nn.Module):
    """
    Monkey patch for https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/mixtral/modeling_mixtral.py#L773  # noqa: E501
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.config = config

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList(
            [MixtralBLockSparseTop2MLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # random topk shuffle
        if self.training and self.config.random_expert_shuffle > 0:
            print("yay")
            num_rows = routing_weights.shape[0]
            num_shuffle = int(self.config.random_expert_shuffle * num_rows)

            # Randomly select rows
            rows_to_shuffle = torch.randint(low=0, high=num_rows, size=(num_shuffle,))

            # Shuffle the first dimension of selected rows
            shuffled_routing_weights = routing_weights.clone()
            for row in rows_to_shuffle:
                shuffled_routing_weights[row] = routing_weights[
                    row, torch.randperm(routing_weights.size(1))
                ].clone()
            routing_weights = shuffled_routing_weights

        if self.training:
            top_k = self.config.num_experts_per_tok_train
        else:
            top_k = self.config.num_experts_per_tok
        top_k = min(top_k, self.num_experts)

        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = (
                expert_layer(current_state)
                * routing_weights[top_x_list, idx_list, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


def monkey_patch_backbone(cfg: Any):
    if getattr(cfg.architecture, "moe_model", False):
        import transformers

        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = (
            MixtralSparseMoeBlock
        )
