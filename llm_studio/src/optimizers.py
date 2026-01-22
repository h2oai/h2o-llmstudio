from functools import partial
from typing import Any

from torch import optim

# bitsandbytes is optional (not available on macOS ARM64)
try:
    import bitsandbytes as bnb

    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


class Optimizers:
    """Optimizers factory."""

    _optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": partial(optim.SGD, momentum=0.9, nesterov=True),
        "RMSprop": partial(optim.RMSprop, momentum=0.9, alpha=0.9),
        "Adadelta": optim.Adadelta,
    }

    # Add 8-bit optimizer only if bitsandbytes is available
    if HAS_BITSANDBYTES:
        _optimizers["AdamW8bit"] = bnb.optim.Adam8bit

    @classmethod
    def names(cls) -> list[str]:
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
