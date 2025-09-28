"""Data models for AI Golf Coaches application."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class VideoMeta(BaseModel):
    """Metadata for a YouTube video.

    This model contains basic information about a YouTube video including
    its identifier, title, description, publication date, and URL.

    Attributes:
        video_id: The unique identifier for the YouTube video.
        title: The video title.
        description: Optional video description.
        published_at: ISO 8601 formatted publication timestamp.
        channel_title: Name of the YouTube channel that published the video.
        url: The YouTube URL for the video.

    """

    video_id: str = Field(
        ..., description="The unique identifier for the YouTube video."
    )
    title: str
    description: Optional[str] = None
    published_at: Optional[str] = None  # ISO 8601 format
    channel_title: Optional[str] = None
    url: HttpUrl


class TranscriptLine(BaseModel):
    """A single line from a video transcript.

    Represents a segment of transcript text with timing information.

    Attributes:
        start: Start time of the transcript segment in seconds.
        duration: Duration of the transcript segment in seconds.
        text: The transcript text content for this segment.

    """

    start: float = Field(ge=0, description="Start time in seconds (>= 0)")
    duration: float = Field(gt=0, description="Duration in seconds (> 0)")
    text: str


class TranscriptDoc(BaseModel):
    """Complete transcript document for a video.

    Contains the full transcript for a video, including all individual
    transcript lines and optional flattened text representation.

    Attributes:
        video_id: The unique identifier for the YouTube video.
        language: Language code for the transcript (defaults to "en").
        lines: List of individual transcript line segments.
        text: Optional flattened text of the entire transcript.

    """

    video_id: str
    language: Optional[str] = "en"
    lines: list[TranscriptLine] = Field(default_factory=list)
    text: Optional[str] = Field(
        default=None,
        description="Optional flattened text for the entire transcript for convenience.",
    )


class RecordStatus(str, Enum):
    """Status enumeration for video processing records.

    Indicates the current state of video processing, including success,
    various failure modes, and error conditions.

    Attributes:
        ok: Processing completed successfully.
        transcripts_disabled: Video has transcripts disabled.
        not_found: Video was not found.
        error: An error occurred during processing.

    """

    ok = "ok"
    transcripts_disabled = "transcripts_disabled"
    not_found = "not_found"
    error = "error"


class VideoRecord(BaseModel):
    """Complete record for a processed video.

    Combines video metadata, processing status, transcript data,
    and any error information into a single record.

    Attributes:
        meta: Video metadata information.
        status: Current processing status.
        transcript: Optional transcript document if available.
        error: Optional error message if processing failed.
        schema_version: Version identifier for the record schema.

    """

    meta: VideoMeta
    status: RecordStatus
    transcript: Optional[TranscriptDoc] = None
    error: Optional[str] = None
    schema_version: str = "v1"
