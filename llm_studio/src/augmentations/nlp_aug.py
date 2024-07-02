import logging
from abc import abstractmethod
from typing import Any, Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)


class BaseNLPAug(nn.Module):
    """Base class for NLP augmentation"""

    def __init__(self, cfg: Any):
        """
        Args:
            cfg: config with all the hyperparameters
        """

        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, batch: Dict) -> Dict:
        """Augmenting

        Args:
            batch: current batch

        Returns:
            augmented batch
        """

        if self.cfg.augmentation.token_mask_probability > 0:
            input_ids = batch["input_ids"].clone()
            # special_mask = ~batch["special_tokens_mask"].clone().bool()
            mask = (
                torch.bernoulli(
                    torch.full(
                        input_ids.shape,
                        float(self.cfg.augmentation.token_mask_probability),
                    )
                )
                .to(input_ids.device)
                .bool()
                # & special_mask
            ).bool()
            input_ids[mask] = self.cfg.tokenizer._tokenizer_mask_token_id
            batch["input_ids"] = input_ids.clone()
            batch["attention_mask"][mask] = 0
            if batch["labels"].shape[1] == batch["input_ids"].shape[1]:
                batch["labels"][mask] = -100

        return batch
