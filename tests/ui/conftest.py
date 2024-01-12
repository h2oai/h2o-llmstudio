import logging
from functools import partial, partialmethod

import pytest

logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")
logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
logging.trace = partial(logging.log, logging.TRACE)


class CustomLogger(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.TRACE):
            self._log(logging.TRACE, msg, args, **kwargs)


@pytest.fixture(scope="session")
def logger() -> logging.Logger:
    return CustomLogger("ui-tests")
