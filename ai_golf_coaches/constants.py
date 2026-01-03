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
