import logging
import os
from functools import partial, partialmethod

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")
logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
logging.trace = partial(logging.log, logging.TRACE)


@pytest.fixture(scope="session")
def logger() -> logging.Logger:
    return logging.getLogger("ui-tests")


@pytest.fixture(scope="session")
def app_address() -> str:
    try:
        address = os.environ.get("LLMSTUDIO_ADDRESS")
        if address:
            return address
        else:
            raise EnvironmentError(
                "LLMSTUDIO_ADDRESS is not set in the environment: `export LLMSTUDIO_ADDRESS=...`"
            )
    except Exception as e:
        return str(e)