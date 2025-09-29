"""AI Golf Coaches - Golf instruction assistant using RAG and YouTube transcripts.

This package provides tools to build, query, and manage a golf instruction system
based on YouTube video transcripts from professional golf coaches.

Main modules:
- config: Configuration management
- models: Data models and types
- youtube: YouTube API integration and video catalog management
- transcripts: Transcript fetching and processing
- rag: Retrieval-Augmented Generation system for golf instruction
- cli: Command-line interface
- personalities: Coach personality definitions and prompting
"""

import logging

from . import config, models, transcripts, utils, youtube

# Configure logging for the entire package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    force=True,  # This ensures it works in Jupyter notebooks
)

logger = logging.getLogger(__name__)
logger.info("AI Golf Coaches package initialized")

__all__ = [
    "config",
    "models",
    "youtube",
    "utils",
    "transcripts",
]
