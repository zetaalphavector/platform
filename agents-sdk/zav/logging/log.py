import logging
import os
from typing import Optional

from pythonjsonlogger import jsonlogger
from zav.strtobool import strtobool

JSON_LOGGING = bool(strtobool(os.getenv("JSON_LOGGING", "true")))


def get_logger(logger: Optional[logging.Logger] = None):

    if not logger:
        logging.basicConfig()
        logger = logging.getLogger()
    if JSON_LOGGING:
        logger.handlers = []
        handler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        formatter = jsonlogger.JsonFormatter("%(levelname)%%(message)%", timestamp=True)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
