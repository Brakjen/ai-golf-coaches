from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import CatalogVideo


class YouTubeSettings(BaseModel):
    """YouTube-related configuration loaded from environment variables.

    Attributes:
        api_key (str): API key for YouTube Data API v3.

    """

    api_key: str = Field(..., description="YouTube Data API v3 key")


class ProxySettings(BaseModel):
    """Proxy configuration for outbound HTTP requests.

    Attributes:
        http (str | None): HTTP proxy URL (e.g., http://user:pass@host:port).
        https (str | None): HTTPS proxy URL.
        username (str | None): Proxy username for composing URLs when http/https not provided.
        password (str | None): Proxy password for composing URLs when http/https not provided.
        host (str | None): Proxy host (e.g., proxy.webshare.io).
        port (int | None): Proxy port number.
        scheme (str | None): Proxy scheme, defaults to 'http' if not set.
        max_parallel (int): Max concurrent outbound requests; keep small to avoid burning proxies.
        retry_backoff_seconds (float): Base backoff seconds for transient failures.

    """

    http: Optional[str] = None
    https: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    scheme: Optional[str] = None
    max_parallel: int = 2
    retry_backoff_seconds: float = 2.0


class OpenAISettings(BaseModel):
    """OpenAI-related configuration loaded from environment variables.

    Attributes:
        api_key (str): API key for OpenAI services.
        embedding_model (str | None): Default embedding model name.

    """

    api_key: str = Field(..., description="OpenAI API key")
    embedding_model: Optional[str] = Field(
        None, description="Default OpenAI embedding model name"
    )


class AppSettings(BaseSettings):
    """Top-level application settings with nested env parsing.

    Environment variables use double-underscore nested keys, e.g.:
    - YOUTUBE__API_KEY
    - PROXY__HTTP
    - PROXY__HTTPS
    - PROXY__MAX_PARALLEL

    Attributes:
        youtube (YouTubeSettings): YouTube API settings.
        proxy (ProxySettings): Proxy settings.
        openai (OpenAISettings | None): OpenAI API settings (optional, required for remote embeddings).

    """

    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")

    youtube: YouTubeSettings
    proxy: ProxySettings = ProxySettings()
    openai: Optional[OpenAISettings] = None


def load_channels_config(config_path: Path | str) -> Dict[str, dict]:
    """Load the channels configuration YAML file.

    Args:
        config_path (Path | str): Path to the YAML configuration containing channel entries.

    Returns:
        Dict[str, dict]: Mapping of canonical channel key to its settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.

    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Channels config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_channel_key(alias_or_key: str, channels: Dict[str, dict]) -> Optional[str]:
    """Resolve an input alias or key to the canonical channel key.

    Args:
        alias_or_key (str): User-provided alias or canonical key.
        channels (Dict[str, dict]): Loaded channels configuration.

    Returns:
        str | None: The canonical key if found; otherwise None.

    """
    candidate = alias_or_key.strip().lower()
    if candidate in channels:
        return candidate
    for key, cfg in channels.items():
        aliases = [a.lower() for a in (cfg.get("aliases") or [])]
        handle = str(cfg.get("handle", "")).lower()
        if candidate == handle or candidate in aliases:
            return key
    return None


def load_channel_catalog(
    alias_or_key: str,
    root: Path | str | None = None,
    channels: Dict[str, dict] | None = None,
    channels_config_path: Path | str = "config/channels.yaml",
) -> List[CatalogVideo]:
    """Load a channel's catalog.jsonl into typed `CatalogVideo` models.

    This helper accepts an alias, handle, or canonical channel key, resolves to
    the canonical key using the channels configuration, and reads the channel's
    `catalog.jsonl` from the workspace `data/<channel>/` directory. Each line is
    parsed and validated against the `CatalogVideo` data contract (Pydantic v2),
    returning a list of typed models.

    Args:
        alias_or_key (str): Alias, handle, or canonical channel key (e.g., "egs",
            "@elitegolfschools", or "elitegolfschools").
        root (Path | str | None): Optional workspace root. Defaults to the
            current working directory.
        channels (Dict[str, dict] | None): Optional pre-loaded channels config
            mapping. If not provided, the config is loaded from `channels_config_path`.
        channels_config_path (Path | str): YAML path used when `channels` is not
            provided.

    Returns:
        List[CatalogVideo]: Validated catalog entries.

    Raises:
        FileNotFoundError: If the catalog file does not exist.
        json.JSONDecodeError: If any line in the catalog is not valid JSON.

    """
    base = Path(root) if root is not None else Path.cwd()
    if channels is None:
        channels = load_channels_config(channels_config_path)
    resolved = (
        resolve_channel_key(alias_or_key, channels) or alias_or_key.strip().lower()
    )

    catalog_path = base / "data" / resolved / "catalog.jsonl"
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found for '{alias_or_key}' (resolved='{resolved}'): {catalog_path}"
        )

    items: List[CatalogVideo] = []
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(CatalogVideo.model_validate(obj))
    return items
