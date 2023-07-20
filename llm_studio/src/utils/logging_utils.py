import io
import json
import logging
import os
import re
from typing import Any, Optional

from llm_studio.src.utils.plot_utils import PlotData

logger = logging.getLogger(__name__)


class IgnorePatchRequestsFilter(logging.Filter):
    def filter(self, record):
        log_message = record.getMessage()
        if re.search(r"HTTP Request: PATCH", log_message):
            return False  # Ignore the log entry
        return True  # Include the log entry


def initialize_logging(cfg: Optional[Any] = None, actual_logger=None):
    format = "%(asctime)s - %(levelname)s: %(message)s"

    if actual_logger is None:
        actual_logger = logging.root
        logging.getLogger("sqlitedict").setLevel(logging.ERROR)
    else:
        actual_logger.handlers.clear()

    actual_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(format)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(IgnorePatchRequestsFilter())
    actual_logger.addHandler(console_handler)

    if cfg is not None:
        logs_dir = f"{cfg.output_directory}/"
        os.makedirs(logs_dir, exist_ok=True)
        file_handler = logging.FileHandler(filename=f"{logs_dir}/logs.log")
        file_formatter = logging.Formatter(format)
        file_handler.setFormatter(file_formatter)
        actual_logger.addHandler(file_handler)


class TqdmToLogger(io.StringIO):
    """
    Outputs stream for TQDM.
    It will output to logger module instead of the StdOut.
    """

    logger: logging.Logger = None
    level: int = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip("\r\n\t [A")

    def flush(self):
        if self.buf != "":
            try:
                self.logger.log(self.level, self.buf)
            except NameError:
                pass


def write_flag(path: str, key: str, value: str):
    """Writes a new flag

    Args:
        path: path to flag json
        key: key of the flag
        value: values of the flag
    """

    logger.debug(f"Writing flag {key}: {value}")

    if os.path.exists(path):
        with open(path, "r+") as file:
            flags = json.load(file)
    else:
        flags = {}

    flags[key] = value

    with open(path, "w+") as file:
        json.dump(flags, file)


def log_plot(cfg: Any, plot: PlotData, type: str) -> None:
    """Logs a given plot

    Args:
        cfg: cfg
        plot: plot to log
        type: type of the plot

    """

    if plot.encoding == "png":
        cfg.logging._logger.log("image", type, plot.data)
    elif plot.encoding == "html":
        cfg.logging._logger.log("html", type, plot.data)
    elif plot.encoding == "df":
        cfg.logging._logger.log("df", type, plot.data)
    else:
        raise ValueError(f"Unknown {type} plot encoding `{plot.encoding}`")
