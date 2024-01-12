import logging
from functools import partial, partialmethod
from typing import Any

import pytest

try:
    logging.TRACE  # type: ignore
except AttributeError:
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, "TRACE")
    logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
    logging.trace = partial(logging.log, logging.TRACE)


@pytest.fixture(scope="session")
def logger() -> Any:
    return logging.getLogger("ui-tests")
