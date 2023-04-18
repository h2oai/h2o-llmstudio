from typing import Any, List

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

__all__ = ["Schedulers"]


def constant_schedule_with_warmup(optimizer, num_warmup_steps, **kwargs):
    return get_constant_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps
    )


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
