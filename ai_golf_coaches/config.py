from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChannelRef(BaseModel):
    channel_id: Optional[str] = Field(
        default=None,
        description="The unique identifier for a YouTube channel. If not provided, the system will attempt to find the channel ID using the provided username.",
    )
    handle: Optional[str] = Field(
        default=None,
        description="The custom handle for a YouTube channel, typically starting with '@'. If not provided, the system will attempt to find the channel ID using the provided username.",
    )

    def is_id(self) -> bool:
        return isinstance(self.channel_id, str) and self.channel_id.startswith("UC")

    def ref_for_api(self) -> str:
        if self.is_id():
            return self.channel_id
        if self.handle:
            return self.handle if self.handle.startswith("@") else f"@{self.handle}"
        raise ValueError("Either channel_id or handle must be provided.")


class YouTubeSettings(BaseModel):
    api_key: SecretStr
    egs: ChannelRef = ChannelRef(channel_id=None, handle="@elitegolfschools")
    milo: ChannelRef = ChannelRef(channel_id=None, handle="@milolinesgolf")
    max_results_per_page: int = 50
    http_timeout_secs: int = 20


class Settings(BaseSettings):
    youtube: YouTubeSettings
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
