import math
from typing import Any, List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_constant_schedule_with_warmup


def constant_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, **kwargs
) -> LambdaLR:
    return get_constant_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps
    )


# adjusted from transformers
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_learning_rate_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_learning_rate_ratio,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# adjusted from transformers
def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_learning_rate_ratio: float = 0.0,
    last_epoch: int = -1,
):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_learning_rate_ratio,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Schedulers:
    """Schedulers factory."""

    _schedulers = {
        "Cosine": get_cosine_schedule_with_warmup,
        "Linear": get_linear_schedule_with_warmup,
        "Constant": constant_schedule_with_warmup,
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._schedulers.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Schedulers.

        Args:
            name: scheduler name
        Returns:
            A class to build the Schedulers
        """
        return cls._schedulers.get(name)
