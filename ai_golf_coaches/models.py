from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, PositiveFloat


# Youtube: Catalog & Transcripts
class VideoMeta(BaseModel):
    video_id: str = Field(
        ..., description="The unique identifier for the YouTube video."
    )
    title: str
    description: Optional[str] = None
    published_at: Optional[str] = None  # ISO 8601 format
    channel_title: Optional[str] = None
    url: HttpUrl


class TranscriptLine(BaseModel):
    start: PositiveFloat
    duration: PositiveFloat
    text: str


class TransciptDoc(BaseModel):
    video_id: str
    language: Optional[str] = "en"
    lines: list[TranscriptLine] = Field(default_factory=list)
    text: Optional[str] = Field(
        default=None,
        description="Optional flattened text for the entire transcript for convenience.",
    )


class RecordStatus(str, Enum):
    ok = "ok"
    transcripts_disabled = "transcripts_didabled"
    not_found = "not_found"
    error = "error"


class VideoRecord(BaseModel):
    meta: VideoMeta
    status: RecordStatus
    transcript: Optional[TransciptDoc] = None
    error: Optional[str] = None
    schema_version: str = "v1"
