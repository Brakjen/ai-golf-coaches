import logging

# Configure logging for the entire package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    force=True,  # This ensures it works in Jupyter notebooks
)

logger = logging.getLogger(__name__)
logger.info("AI Golf Coaches package initialized")

from . import config, models, utils, youtube

__all__ = [
    "config",
    "models",
    "youtube",
    "utils",
]
