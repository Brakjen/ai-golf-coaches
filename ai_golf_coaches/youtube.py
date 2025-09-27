"""YouTube API integration and data management.

Provides functionality for:
- Fetching video catalogs from YouTube channels
- Managing local data storage for video metadata
- Handling YouTube API authentication and rate limiting
- Processing video transcripts and metadata
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, HttpUrl

from ai_golf_coaches.config import get_settings
from ai_golf_coaches.models import VideoMeta

if TYPE_CHECKING:
    import googleapiclient.discovery.Resource

# Set up logging
logger = logging.getLogger(__name__)

# IO paths


def channel_dir(name: str) -> Path:
    """Create and return the data directory path for a channel.

    Creates the directory structure if it doesn't exist.

    Args:
        name (str): Channel name or handle (without @).

    Returns:
        Path: Path object pointing to the channel's data directory.

    """
    d = Path("data") / "raw" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def catalog_path(name: str) -> Path:
    """Get the file path for a channel's video catalog.

    Args:
        name (str): Channel name or handle (without @).

    Returns:
        Path: Path object pointing to the channel's catalog JSON file.

    """
    return channel_dir(name) / "_catalog.json"


# YouTube API


def yt_client() -> googleapiclient.discovery.Resource:
    """Create and return a YouTube API client.

    Uses the API key from application settings to authenticate
    with the YouTube Data API v3.

    Returns:
        googleapiclient.discovery.Resource: Configured YouTube API client resource.

    """
    cfg = get_settings()
    return build(
        "youtube",
        "v3",
        developerKey=cfg.youtube.api_key.get_secret_value(),
        cache_discovery=False,
    )


def get_uploads_playlist_id(yt: Any, channel_id: str) -> str:
    """Get the uploads playlist ID for a YouTube channel.

    Args:
        yt (Any): YouTube API client instance.
        channel_id (str): YouTube channel ID (starts with 'UC').

    Returns:
        str: The playlist ID containing all uploaded videos for the channel.

    Raises:
        KeyError: If the channel is not found or has no uploads playlist.

    """
    r = yt.channels().list(part="contentDetails", id=channel_id).execute()
    return r["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_all_uploads(yt: Any, uploads_pid: str, page_size: int = 50) -> List[Dict]:
    """Fetch all videos from a YouTube channel's uploads playlist.

    Handles pagination and rate limiting automatically, retrying on
    temporary errors (403, 429, 500, 503).

    Args:
        yt (Any): YouTube API client instance.
        uploads_pid (str): Playlist ID for the channel's uploads.
        page_size (int, optional): Number of results per API page (max 50). Defaults to 50.

    Returns:
        List[Dict]: List of video items from the playlist.

    Raises:
        HttpError: For non-recoverable API errors.

    """
    items: List[Dict] = []
    token: Optional[str] = None
    while True:
        try:
            resp = (
                yt.playlistItems()
                .list(
                    part="snippet,contentDetails",
                    playlistId=uploads_pid,
                    maxResults=page_size,
                    pageToken=token,
                )
                .execute()
            )
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in (403, 429, 500, 503):  # <-- fix: 500 instead of 5009
                time.sleep(2)
                continue
            raise
        items += resp.get("items", [])
        token = resp.get("nextPageToken")
        if not token:
            break
    return items


_DUR_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _iso8601_to_seconds(s: str) -> int:
    """Convert ISO 8601 duration string to total seconds.

    Parses YouTube's PT format (e.g., 'PT1H23M45S') into seconds.

    Args:
        s (str): ISO 8601 duration string from YouTube API.

    Returns:
        int: Total duration in seconds, or 0 if parsing fails.

    """
    m = _DUR_RE.fullmatch(s or "")
    if not m:
        return 0
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    s_ = int(m.group(3) or 0)
    return h * 3600 + m_ * 60 + s_


def _videos_details_by_ids(yt: Any, ids: List[str]) -> Dict[str, Dict]:
    """Fetch detailed information for multiple videos in batches.

    Processes video IDs in batches of 50 (YouTube API limit) and
    includes rate limiting delays between requests.

    Args:
        yt (Any): YouTube API client instance.
        ids (List[str]): List of YouTube video IDs.

    Returns:
        Dict[str, Dict]: Dictionary mapping video IDs to their detailed information.

    """
    out: Dict[str, Dict] = {}
    for i in range(0, len(ids), 50):
        batch = ids[i : i + 50]
        resp = (
            yt.videos()
            .list(
                part="contentDetails,liveStreamingDetails,snippet",
                id=",".join(batch),
            )
            .execute()
        )
        for it in resp.get("items", []):
            out[it["id"]] = it
        time.sleep(0.05)
    return out


def build_catalog_for_channel_id(
    channel_id: str,
    page_size: int = 50,
    *,
    include_shorts: bool = False,
    duration_thrs_seconds: int = 360,
) -> List[VideoMeta]:
    """Build a complete video catalog for a YouTube channel.

    Fetches all videos from a channel, filters by duration, and returns
    structured metadata for each video.

    Args:
        channel_id (str): YouTube channel ID (starts with 'UC').
        page_size (int, optional): Number of results per API page (max 50). Defaults to 50.
        include_shorts (bool, optional): Whether to include YouTube Shorts in results. Defaults to False.
        duration_thrs_seconds (int, optional): Minimum duration threshold for filtering shorts. Defaults to 360.

    Returns:
        List[VideoMeta]: List of VideoMeta objects sorted by publication date (newest first).

    Raises:
        HttpError: If YouTube API requests fail.
        KeyError: If channel is not found.

    """
    logger.info(f"Building catalog for channel id: {channel_id}")
    yt = yt_client()
    pid = get_uploads_playlist_id(yt, channel_id)
    items = list_all_uploads(yt, pid, page_size=page_size)

    # Fetch durations to filter Shorts
    ids = [it["contentDetails"]["videoId"] for it in items]
    details = _videos_details_by_ids(yt, ids)

    metas: List[VideoMeta] = []
    for item in items:
        snip = item["snippet"]
        vid = item["contentDetails"]["videoId"]

        det = details.get(vid)
        if not det:
            continue  # rare, but skip if missing detail

        # Filter out Shorts unless explicitly included
        dur_iso = det.get("contentDetails", {}).get("duration", "PT0S")
        seconds = _iso8601_to_seconds(dur_iso)
        if not include_shorts and seconds < duration_thrs_seconds:
            continue

        metas.append(
            VideoMeta(
                video_id=vid,
                title=snip.get("title", ""),
                description=snip.get("description"),
                published_at=snip.get("publishedAt", ""),
                channel_title=snip.get("channelTitle"),
                url=HttpUrl(f"https://youtu.be/{vid}"),
            )
        )

    # Optional: newest first
    metas.sort(key=lambda m: m.published_at or "", reverse=True)
    return metas


def _dump_model(m: BaseModel) -> dict:
    """Convert a Pydantic model to a JSON-serializable dictionary.

    Handles both Pydantic v1 and v2 with proper JSON encoding
    for complex types like HttpUrl.

    Args:
        m (BaseModel): Pydantic model instance to serialize.

    Returns:
        dict: Dictionary representation suitable for JSON serialization.

    """
    # Pydantic v2
    if hasattr(m, "model_dump"):
        return m.model_dump(mode="json")  # ensures HttpUrl -> str, etc.
    # Pydantic v1
    return json.loads(m.json())  # uses pydantic encoders


def save_catalog(channel_handle: str, metas: List[VideoMeta]) -> Path:
    """Save video catalog to local storage.

    Serializes video metadata to JSON and saves to the channel's
    data directory.

    Args:
        channel_handle (str): Channel name or handle (@ prefix optional).
        metas (List[VideoMeta]): List of video metadata to save.

    Returns:
        Path: Path where the catalog was saved.

    """
    logger.info(
        f"Saving catalog for channel: {channel_handle} with {len(metas)} videos"
    )
    if channel_handle.startswith("@"):
        channel_handle = channel_handle[1:]
    path = catalog_path(channel_handle)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [_dump_model(m) for m in metas]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path


def load_catalog(channel_handle: str) -> List[VideoMeta]:
    """Load video catalog from local storage.

    Reads and deserializes video metadata from the channel's
    catalog JSON file.

    Args:
        channel_handle (str): Channel name or handle (@ prefix optional).

    Returns:
        List[VideoMeta]: List of VideoMeta objects, or empty list if catalog doesn't exist.

    Raises:
        JSONDecodeError: If the catalog file is corrupted.
        ValidationError: If the data doesn't match VideoMeta schema.

    """
    if channel_handle.startswith("@"):
        channel_handle = channel_handle[1:]
    path = catalog_path(channel_handle)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [VideoMeta(**row) for row in data]
