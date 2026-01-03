from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterable, List, Optional

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApiException,
)
from youtube_transcript_api.proxies import WebshareProxyConfig

from .constants import (
    DEFAULT_LANGUAGES,
    RETRY_MAX_ATTEMPTS,
    RETRY_WAIT_MAX_SECONDS,
    RETRY_WAIT_MIN_SECONDS,
)
from .models import TranscriptChunk

if TYPE_CHECKING:
    import pathlib


def _build_ytt_api_from_env() -> YouTubeTranscriptApi:
    """Create a `YouTubeTranscriptApi` configured with Webshare rotating proxies.

    Reads `PROXY__USERNAME` and `PROXY__PASSWORD` from the environment and
    constructs a `WebshareProxyConfig` to enable rotating residential proxies.

    Returns:
        YouTubeTranscriptApi: Instance using Webshare proxies.

    Note:
        This implementation assumes Webshare credentials are present and does
        not implement a generic proxy fallback.

    """
    proxy_config = None
    username = os.getenv("PROXY__USERNAME")
    password = os.getenv("PROXY__PASSWORD")

    proxy_config = WebshareProxyConfig(
        proxy_username=username,
        proxy_password=password,
        proxy_port=80,
    )

    return YouTubeTranscriptApi(proxy_config=proxy_config)


@retry(
    wait=wait_random_exponential(
        multiplier=1, min=RETRY_WAIT_MIN_SECONDS, max=RETRY_WAIT_MAX_SECONDS
    ),
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    retry=retry_if_not_exception_type(
        (
            TranscriptsDisabled,
            NoTranscriptFound,
            VideoUnavailable,
            CouldNotRetrieveTranscript,
            YouTubeTranscriptApiException,
        )
    ),
    reraise=True,
)
def fetch_transcript_chunks(
    video_id: str,
    languages: Optional[List[str]] = None,
) -> List[TranscriptChunk]:
    """Fetch and convert transcript to a list of `TranscriptChunk`.

    Args:
        video_id (str): YouTube video ID.
        languages (list[str] | None): Preferred language codes.

    Returns:
        list[TranscriptChunk]: Validated transcript chunk objects.

    Raises:
        TranscriptsDisabled: If transcripts are disabled.
        NoTranscriptFound: If no transcript exists in preferred languages.
        VideoUnavailable: If the video is unavailable.

    """
    ytt_api = _build_ytt_api_from_env()
    langs = languages or list(DEFAULT_LANGUAGES)
    fetched = ytt_api.fetch(video_id=video_id, languages=langs)

    # Prefer raw dict conversion when available; otherwise iterate snippets
    try:
        raw = fetched.to_raw_data()
    except AttributeError:
        raw = [
            {
                "text": getattr(snip, "text", ""),
                "start": getattr(snip, "start", 0.0),
                "duration": getattr(snip, "duration", None),
            }
            for snip in fetched
        ]

    chunks: List[TranscriptChunk] = []
    for it in raw:
        text = (it.get("text") or "").strip()
        start_val = it.get("start", 0.0)
        start = float(start_val) if not isinstance(start_val, float) else start_val
        duration_val = it.get("duration")
        duration = (
            float(duration_val)
            if duration_val is not None and not isinstance(duration_val, float)
            else duration_val
        )
        chunks.append(TranscriptChunk(chunk=text, start=start, duration=duration))
    return chunks


def write_transcript_jsonl(
    output_path: pathlib.Path, chunks: Iterable[TranscriptChunk]
) -> None:
    """Write transcript chunks to a JSONL file, one chunk per line.

    Args:
        output_path (Path): Path to the JSONL file to create.
        chunks (Iterable[TranscriptChunk]): Iterable of validated transcript items.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            obj = {"chunk": chunk.chunk, "start": chunk.start}
            if chunk.duration is not None:
                obj["duration"] = chunk.duration
            f.write(json.dumps(obj) + "\n")


def combine_chunks(
    chunks: List[TranscriptChunk], window_seconds: float
) -> List[TranscriptChunk]:
    """Combine adjacent transcript chunks into windows of approximately `window_seconds`.

    This function assumes each input chunk has a non-None `duration`. Chunks are
    accumulated in order until the cumulative duration reaches or exceeds
    `window_seconds`, at which point a combined chunk is emitted. The final group
    may be shorter than the window.

    Args:
        chunks (list[TranscriptChunk]): Input transcript chunks in chronological order.
        window_seconds (float): Target window size in seconds (e.g., 15.0).

    Returns:
        list[TranscriptChunk]: Combined chunks with concatenated text, the start of
            the first chunk in each group, and the summed duration for the group.

    Raises:
        ValueError: If any input chunk has a missing `duration` when combining.

    """
    if window_seconds <= 0:
        return []

    combined: List[TranscriptChunk] = []
    acc_text: List[str] = []
    acc_start: Optional[float] = None
    acc_duration: float = 0.0

    for ch in chunks:
        if ch.duration is None:
            raise ValueError("combine_chunks requires all chunks to have a duration")
        if acc_start is None:
            acc_start = ch.start
        txt = (ch.chunk or "").strip()
        if txt:
            acc_text.append(txt)
        acc_duration += float(ch.duration)

        if acc_duration >= window_seconds:
            combined.append(
                TranscriptChunk(
                    chunk=" ".join(acc_text),
                    start=acc_start,
                    duration=acc_duration,
                )
            )
            # reset accumulator
            acc_text = []
            acc_start = None
            acc_duration = 0.0

    # flush any remainder
    if acc_start is not None:
        combined.append(
            TranscriptChunk(
                chunk=" ".join(acc_text),
                start=acc_start,
                duration=acc_duration if acc_duration > 0 else 0.0,
            )
        )

    return combined
