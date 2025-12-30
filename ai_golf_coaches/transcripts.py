from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential
from youtube_transcript_api import (
    YouTubeTranscriptApi,
)

from .models import TranscriptChunk

if TYPE_CHECKING:
    import pathlib


def _proxies_dict(
    http: Optional[str], https: Optional[str]
) -> Optional[Dict[str, str]]:
    """Construct a proxies dictionary for requests.

    Args:
        http (str | None): HTTP proxy URL, if available.
        https (str | None): HTTPS proxy URL, if available.

    Returns:
        dict[str, str] | None: Dict suitable for `proxies={'http': ..., 'https': ...}`.

    """
    proxies: Dict[str, str] = {}
    if http:
        proxies["http"] = http
    if https:
        proxies["https"] = https
    return proxies or None


@retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(5))
def fetch_transcript_chunks(
    video_id: str,
    http_proxy: Optional[str] = None,
    https_proxy: Optional[str] = None,
    languages: Optional[List[str]] = None,
) -> List[TranscriptChunk]:
    """Fetch transcript chunks for a video using public transcripts.

    Applies exponential backoff with jitter and uses proxies when provided.

    Args:
        video_id (str): YouTube video ID to fetch transcript for.
        http_proxy (str | None): HTTP proxy URL.
        https_proxy (str | None): HTTPS proxy URL.
        languages (list[str] | None): Preferred language codes; defaults to English fallback.

    Returns:
        list[TranscriptChunk]: Transcript chunks for the video.

    Raises:
        TranscriptsDisabled: If transcripts are disabled for the video.
        VideoUnavailable: If the video is unavailable.

    """
    proxies = _proxies_dict(http_proxy, https_proxy)
    langs = languages or ["en"]
    items = YouTubeTranscriptApi.get_transcript(
        video_id, proxies=proxies, languages=langs
    )
    chunks: List[TranscriptChunk] = []
    for it in items:
        text = it.get("text", "").strip()
        start = float(it.get("start", 0.0))
        duration = it.get("duration")
        chunks.append(
            TranscriptChunk(
                chunk=text,
                start=start,
                duration=(float(duration) if duration is not None else None),
            )
        )
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
