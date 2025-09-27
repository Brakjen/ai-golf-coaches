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
from typing import Any, Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, HttpUrl

from ai_golf_coaches.config import get_settings
from ai_golf_coaches.models import VideoMeta

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


def yt_client() -> Any:
    """Create and return a YouTube API client.

    Uses the API key from application settings to authenticate
    with the YouTube Data API v3.

    Returns:
        Any: Configured YouTube API client resource from googleapiclient.discovery.build().

    """
    cfg = get_settings()
    return build(
        "youtube",
        "v3",
        developerKey=cfg.youtube.api_key.get_secret_value(),
        cache_discovery=False,
    )


def get_uploads_playlist_id(yt: Any, channel_identifier: str) -> str:
    """Get the uploads playlist ID for a YouTube channel.

    Args:
        yt (Any): YouTube API client resource from googleapiclient.discovery.build().
        channel_identifier (str): YouTube channel ID (UC...) or handle (@...).

    Returns:
        str: The playlist ID containing all uploaded videos for the channel.

    Raises:
        ValueError: If the channel is not found.

    """
    if channel_identifier.startswith("@") or not channel_identifier.startswith("UC"):
        # It's a handle - use forHandle parameter
        handle = (
            channel_identifier
            if channel_identifier.startswith("@")
            else f"@{channel_identifier}"
        )
        r = yt.channels().list(part="contentDetails", forHandle=handle[1:]).execute()
    else:
        # It's a channel ID - use id parameter
        r = yt.channels().list(part="contentDetails", id=channel_identifier).execute()

    if not r.get("items"):
        raise ValueError(f"Channel not found: {channel_identifier}")

    return r["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_all_uploads(yt: Any, uploads_pid: str, page_size: int = 50) -> List[Dict]:
    """Fetch all videos from a YouTube channel's uploads playlist.

    Handles pagination and rate limiting automatically, retrying on
    temporary errors (403, 429, 500, 503).

    Args:
        yt (Any): YouTube API client resource from googleapiclient.discovery.build().
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
        yt (Any): YouTube API client resource from googleapiclient.discovery.build().
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


def resolve_channel_identifier(channel_identifier: str) -> tuple[str, str]:
    """Resolve a channel identifier to both ID and handle for API calls and logging.

    Takes either a channel ID or handle and resolves it using the configuration
    to return both the ID (for API calls) and handle (for logging).

    Args:
        channel_identifier (str): YouTube channel ID (UC...) or handle (@...).

    Returns:
        tuple[str, str]: A tuple of (channel_id, handle) where:
            - channel_id: The actual channel ID for API calls
            - handle: The human-readable handle for logging

    Raises:
        ValueError: If handle is provided but not found in configuration.

    """
    config = get_settings()

    if channel_identifier.startswith("@"):
        # It's a handle - look up in config
        handle = channel_identifier
        channel_ref = None

        # Check configured channels
        if config.youtube.egs.handle == handle:
            channel_ref = config.youtube.egs
        elif config.youtube.milo.handle == handle:
            channel_ref = config.youtube.milo

        if not channel_ref:
            available_handles = [config.youtube.egs.handle, config.youtube.milo.handle]
            raise ValueError(
                f"Handle '{handle}' not found in configuration. Available handles: {available_handles}"
            )

        # Get the actual channel ID for API calls
        if channel_ref.is_id():
            channel_id = channel_ref.channel_id
        else:
            # If no ID is stored, we need to resolve the handle via API
            channel_id = (
                channel_identifier  # Will be handled by get_uploads_playlist_id
            )

        return channel_id, handle

    else:
        # It's a channel ID - find corresponding handle for logging
        channel_id = channel_identifier
        handle = channel_id  # fallback

        # Look up handle in config
        if channel_id == config.youtube.egs.channel_id:
            handle = config.youtube.egs.handle or config.youtube.egs.ref_for_api()
        elif channel_id == config.youtube.milo.channel_id:
            handle = config.youtube.milo.handle or config.youtube.milo.ref_for_api()

        return channel_id, handle


def build_catalog_for_channel_id(
    channel_identifier: str, page_size: int = 50, *, duration_thrs_seconds: int = 360
) -> List[VideoMeta]:
    """Build a video catalog for a YouTube channel.

    Accepts either a channel ID (starting with 'UC') or a handle (starting with '@').
    If a handle is provided, looks up the corresponding channel configuration.
    Filters out YouTube Shorts based on duration threshold.

    Args:
        channel_identifier (str): YouTube channel ID (UC...) or handle (@...).
        page_size (int, optional): Number of results per API page (max 50). Defaults to 50.
        duration_thrs_seconds (int, optional): Minimum duration to exclude Shorts. Defaults to 360.

    Returns:
        List[VideoMeta]: List of video metadata objects for the channel.

    Raises:
        ValueError: If handle is provided but not found in configuration.

    """
    # Resolve channel identifier to ID and handle
    channel_id, handle = resolve_channel_identifier(channel_identifier)
    logger.info(f"Building catalog for channel: {handle} (ID: {channel_id})")

    yt = yt_client()
    pid = get_uploads_playlist_id(yt, channel_id)
    items = list_all_uploads(yt, pid, page_size=page_size)

    # Fetch durations to filter Shorts
    ids = [it["contentDetails"]["videoId"] for it in items]
    details = _videos_details_by_ids(yt, ids)
    logger.info(
        f"Found {len(items)} videos, filtering shorts (threshold: {duration_thrs_seconds}s)"
    )

    # Process videos and filter
    catalog: List[VideoMeta] = []
    shorts_count = 0

    for item in items:
        video_id = item["contentDetails"]["videoId"]
        detail = details.get(video_id)

        if not detail:
            logger.warning(f"No details found for video {video_id}, skipping")
            continue

        # Parse duration and filter Shorts
        duration_iso = detail["contentDetails"]["duration"]
        duration_seconds = _iso8601_to_seconds(duration_iso)

        if duration_seconds < duration_thrs_seconds:
            shorts_count += 1
            logger.debug(f"Skipping Short: {video_id} ({duration_seconds}s)")
            continue

        # Extract metadata
        snippet = item["snippet"]
        video_meta = VideoMeta(
            video_id=video_id,
            title=snippet["title"],
            description=snippet.get("description"),
            published_at=snippet.get("publishedAt"),
            channel_title=snippet.get("channelTitle"),
            url=HttpUrl(f"https://www.youtube.com/watch?v={video_id}"),
        )

        catalog.append(video_meta)

    logger.info(
        f"Catalog built: {len(catalog)} videos, {shorts_count} shorts filtered out"
    )
    return catalog


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


def save_catalog(channel_handle: str, catalog: List[VideoMeta]) -> Path:
    """Save video catalog to local storage.

    Serializes video metadata to JSON and saves to the channel's
    data directory.

    Args:
        channel_handle (str): Channel name or handle (@ prefix optional).
        catalog (List[VideoMeta]): List of video metadata to save.

    Returns:
        Path: Path where the catalog was saved.

    """
    logger.info(
        f"Saving catalog for channel: {channel_handle} with {len(catalog)} videos"
    )
    if channel_handle.startswith("@"):
        channel_handle = channel_handle[1:]
    path = catalog_path(channel_handle)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [_dump_model(m) for m in catalog]
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
