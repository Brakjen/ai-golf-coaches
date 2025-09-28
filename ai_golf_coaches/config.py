"""Configuration management for AI Golf Coaches application."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChannelRef(BaseModel):
    """Reference to a YouTube channel using either channel ID or handle.

    Provides a flexible way to reference YouTube channels, supporting both
    the unique channel ID (UCxxxxx format) and user-friendly handles (@username).

    Attributes:
        channel_id: The unique identifier for a YouTube channel (starts with 'UC').
        handle: The custom handle for a YouTube channel (starts with '@').

    """

    channel_id: Optional[str] = Field(
        default=None,
        description="The unique identifier for a YouTube channel. If not provided, the system will attempt to find the channel ID using the provided username.",
    )
    handle: Optional[str] = Field(
        default=None,
        description="The custom handle for a YouTube channel, typically starting with '@'. If not provided, the system will attempt to find the channel ID using the provided username.",
    )

    def is_id(self) -> bool:
        """Check if this reference contains a valid channel ID.

        Returns:
            True if channel_id is a string starting with 'UC', False otherwise.

        """
        return isinstance(self.channel_id, str) and self.channel_id.startswith("UC")

    def ref_for_api(self) -> str:
        """Get the appropriate reference string for YouTube API calls.

        Returns the channel ID if available, otherwise returns the handle
        with proper '@' prefix formatting.

        Returns:
            str: String suitable for YouTube API channel identification.

        Raises:
            ValueError: If neither channel_id nor handle is provided.

        """
        if self.is_id():
            return self.channel_id  # type: ignore[return-value]  # is_id() ensures this is not None
        if self.handle:
            return self.handle if self.handle.startswith("@") else f"@{self.handle}"
        raise ValueError("Either channel_id or handle must be provided.")


class YouTubeSettings(BaseModel):
    """Configuration settings for YouTube API integration.

    Contains API credentials and channel references for golf coaching
    YouTube channels, along with operational parameters.

    Attributes:
        api_key: Secret API key for YouTube Data API v3 access.
        egs: Channel reference for Elite Golf Schools.
        milo: Channel reference for Milo Lines Golf.
        max_results_per_page: Maximum number of results per API page request.
        http_timeout_secs: HTTP timeout in seconds for API requests.

    """

    api_key: SecretStr
    egs: ChannelRef = ChannelRef(channel_id=None, handle="@elitegolfschools")
    milo: ChannelRef = ChannelRef(channel_id=None, handle="@milolinesgolf")
    max_results_per_page: int = 50
    http_timeout_secs: int = 20


class ProxyConfig(BaseModel):
    """Configuration for proxy server access.

    Contains credentials and connection details for accessing a proxy server,
    which may be used for routing API requests.

    Attributes:
        username: Proxy server username.
        password: Secret password for proxy server.
        port: Proxy server port number (default is 80).
        endpoint: Proxy server endpoint URL (default is 'proxy.webshare.io').

    """

    username: str
    password: SecretStr
    port: int = 80
    endpoint: str = "proxy.webshare.io"


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API access.

    Contains API credentials and model settings for OpenAI services
    used in the RAG system for golf instruction.

    Attributes:
        api_key: Secret API key for OpenAI API access.
        model: Default model to use for chat completions.
        temperature: Sampling temperature for response generation.
        max_tokens: Maximum tokens in response.

    """

    api_key: SecretStr
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1000


class Settings(BaseSettings):
    """Main application settings.

    Root configuration object that aggregates all application settings,
    with support for environment variable overrides using nested delimiters.

    Attributes:
        youtube: YouTube API and channel configuration.

    """

    youtube: YouTubeSettings = YouTubeSettings(
        api_key=SecretStr("api_key"),  # Will be overridden by env vars
        egs=ChannelRef(channel_id=None, handle="@elitegolfschools"),
        milo=ChannelRef(channel_id=None, handle="@milolinesgolf"),
    )

    proxy: ProxyConfig = ProxyConfig(
        username="proxy_user",  # Will be overridden by env vars
        password=SecretStr("proxy_password"),
    )

    openai: OpenAIConfig = OpenAIConfig(
        api_key=SecretStr("openai_api_key"),  # Will be overridden by env vars
    )

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings instance.

    Returns a singleton Settings object, creating it on first call
    and returning the cached instance on subsequent calls.

    Returns:
        Settings: Configured Settings instance with all application settings.

    """
    return Settings()
