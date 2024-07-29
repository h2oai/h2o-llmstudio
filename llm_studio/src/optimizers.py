from functools import partial
from typing import Any, List

import bitsandbytes as bnb
from torch import optim


class Optimizers:
    """Optimizers factory."""

    _optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": partial(optim.SGD, momentum=0.9, nesterov=True),
        "RMSprop": partial(optim.RMSprop, momentum=0.9, alpha=0.9),
        "Adadelta": optim.Adadelta,
        "AdamW8bit": bnb.optim.Adam8bit,
    }

    @classmethod
    def names(cls) -> List[str]:
        return sorted(cls._optimizers.keys())

    @classmethod
    def get(cls, name: str) -> Any:
        """Access to Optimizers.

        Args:
            name: optimizer name
        Returns:
            A class to build the Optimizer
        """
        return cls._optimizers.get(name)
