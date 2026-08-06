import logging
from typing import Callable, Any
from .colored_logging import setup_colored_logging, ColoredFormatter

# Set up colored logging by default
setup_colored_logging()

logger = logging.getLogger(__name__)

def log_title(title: str):
    title_len = len(title)
    print("", end="\n")
    logger.info(f"{'=' * title_len}")
    logger.info(f"{title}")
    logger.info(f"{'=' * title_len}")

def set_config(f: Callable[[Any], None]):
    f(logging)
