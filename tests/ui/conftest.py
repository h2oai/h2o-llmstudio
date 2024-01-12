import logging

import pytest

TRACE = 5

# Add a custom log level
logging.addLevelName(TRACE, "TRACE")


# Define a custom log method for TRACE level
def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Attach the custom log method to the Logger class
logging.Logger.trace = trace

logging.TRACE = TRACE

# Set up a custom logger configuration
logging.basicConfig(
    level=logging.TRACE, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add levelName for TRACE
logging.addLevelName(TRACE, "TRACE")


@pytest.fixture(scope="session")
def logger() -> logging.Logger:
    return logging.getLogger("ui-tests")
