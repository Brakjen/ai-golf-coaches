from __future__ import annotations

"""Centralized application constants.

This module centralizes default values and tunables used across the codebase
to keep configuration in one place.
"""

# Transcript combining defaults
COMBINE_DEFAULT_SECONDS: float = 15.0

# Transcript fetching defaults
DEFAULT_LANGUAGES = ["en", "en-US", "en-GB"]

# Concurrency defaults
FETCH_MAX_WORKERS: int = 2

# Retry/backoff defaults (used with tenacity)
RETRY_MAX_ATTEMPTS: int = 3
RETRY_WAIT_MIN_SECONDS: float = 1.0
RETRY_WAIT_MAX_SECONDS: float = 30.0

# Catalog filtering heuristics
SHORTS_THRESHOLD_SECONDS: int = 180  # Under 3 minutes considered a Short

# Embedding and indexing defaults
DEFAULT_EMBEDDING_PROVIDER: str = "openai"
DEFAULT_OPENAI_EMBED_MODEL: str = "text-embedding-3-large"
INDEX_DIR_NAME: str = "index"

# QA extraction defaults
QA_DEFAULT_WINDOW_SECONDS: float = 600.0
QA_DEFAULT_OVERLAP_SECONDS: float = 60.0
QA_DEFAULT_MAX_WINDOW_CHARS: int = 0
QA_DEFAULT_MAX_OUTPUT_TOKENS: int = 6000
QA_DEFAULT_WORKERS: int = 3
