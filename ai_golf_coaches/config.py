from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    import pathlib


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

    """

    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")

    youtube: YouTubeSettings
    proxy: ProxySettings = ProxySettings()


def load_channels_config(config_path: pathlib.Path) -> Dict[str, dict]:
    """Load the channels configuration YAML file.

    Args:
        config_path (Path): Path to the YAML configuration containing channel entries.

    Returns:
        Dict[str, dict]: Mapping of canonical channel key to its settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.

    """
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
