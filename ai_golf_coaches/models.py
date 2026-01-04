from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptChunk(BaseModel):
    """A single transcript chunk line stored in JSONL.

    Attributes:
        chunk (str): The textual content of the transcript segment.
        start (float): Start time in seconds from the beginning of the video.
        duration (float | None): Duration in seconds of the chunk, if provided by the source.

    """

    chunk: str = Field(..., description="Transcript text chunk")
    start: float = Field(..., description="Start time (seconds) from video start")
    duration: Optional[float] = Field(
        None, description="Chunk duration in seconds, if available"
    )


class CatalogVideo(BaseModel):
    """Metadata for a single YouTube video in our catalog.

    Attributes:
        video_id (str): The YouTube video ID.
        title (str): The video title.
        published_at (datetime): Timestamp when the video was published.
        duration_seconds (int): Video length in seconds.
        is_short (bool): Whether the video is short-form (heuristic threshold).
        is_livestream (bool): Whether the video is/was a livestream.
        is_podcast (bool): Whether the video is a podcast (simple title heuristic).
        channel_id (str): The YouTube channel ID this video belongs to.
        channel_title (str | None): The display title of the channel.

    """

    video_id: str
    title: str
    published_at: datetime
    duration_seconds: int
    is_short: bool = False
    is_livestream: bool = False
    is_podcast: bool = False
    channel_id: str
    channel_title: Optional[str] = None


class ChannelConfig(BaseModel):
    """Configuration for a channel including aliases.

    Attributes:
        handle (str): The channel handle (e.g., @elitegolfschools).
        channel_id (str): The YouTube channel ID.
        aliases (list[str]): Short aliases for CLI convenience.

    """

    handle: str
    channel_id: str
    aliases: List[str] = Field(default_factory=list)


class TranscriptFile(BaseModel):
    """Represents a transcript file for a video in memory.

    This is used internally for validation and writing JSONL output.

    Attributes:
        video_id (str): The YouTube video ID.
        channel_key (str): Canonical channel key (e.g., elitegolfschools).
        chunks (list[TranscriptChunk]): List of transcript chunks.
        generated_at (datetime): Timestamp when this transcript was generated.

    """

    video_id: str
    channel_key: str
    chunks: List[TranscriptChunk]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
