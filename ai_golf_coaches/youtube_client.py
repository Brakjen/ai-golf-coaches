from __future__ import annotations

from datetime import datetime, timezone
from typing import Generator, Iterable, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import ValidationError

from .constants import SHORTS_THRESHOLD_SECONDS
from .models import CatalogVideo


def _parse_duration_iso8601(duration: str) -> int:
    """Convert ISO8601 duration like 'PT1H2M10S' to seconds.

    Args:
        duration (str): ISO8601 duration string from YouTube API.

    Returns:
        int: Duration in seconds.

    """
    # Minimal parser without external deps
    total = 0
    num = ""
    for ch in duration:
        if ch.isdigit():
            num += ch
        elif ch in {"H", "M", "S"} and num:
            val = int(num)
            if ch == "H":
                total += val * 3600
            elif ch == "M":
                total += val * 60
            elif ch == "S":
                total += val
            num = ""
    return total


def fetch_channel_video_ids(
    youtube_api_key: str, channel_id: str
) -> Generator[str, None, None]:
    """Yield all video IDs for a channel ordered by date.

    Uses YouTube Data API `search.list` with pagination.

    Args:
        youtube_api_key (str): The YouTube API key.
        channel_id (str): Channel ID to enumerate videos for.

    Yields:
        str: Video IDs.

    """
    yt = build("youtube", "v3", developerKey=youtube_api_key)
    page_token = None
    while True:
        try:
            req = yt.search().list(
                channelId=channel_id,
                part="id,snippet",
                order="date",
                type="video",
                maxResults=50,
                pageToken=page_token,
            )
            resp = req.execute()
        except HttpError as e:
            raise RuntimeError(f"YouTube API error: {e}") from e

        for item in resp.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                yield vid

        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def fetch_channel_uploads_video_ids(youtube_api_key: str, channel_id: str) -> List[str]:
    """Return all video IDs from the channel's uploads playlist.

    This approach avoids `search.list` result caps and yields the full archive.

    Args:
        youtube_api_key (str): The YouTube API key.
        channel_id (str): The channel ID to enumerate uploads for.

    Returns:
        list[str]: All upload video IDs for the channel.

    """
    yt = build("youtube", "v3", developerKey=youtube_api_key)

    # Get uploads playlist ID
    try:
        ch_req = yt.channels().list(id=channel_id, part="contentDetails", maxResults=1)
        ch_resp = ch_req.execute()
    except HttpError as e:
        raise RuntimeError(f"YouTube API error: {e}") from e

    items = ch_resp.get("items", [])
    if not items:
        return []
    uploads_pl: Optional[str] = (
        items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    )
    if not uploads_pl:
        return []

    # Page through uploads playlist items
    out: List[str] = []
    page_token: Optional[str] = None
    while True:
        try:
            pl_req = yt.playlistItems().list(
                playlistId=uploads_pl,
                part="contentDetails",
                maxResults=50,
                pageToken=page_token,
            )
            pl_resp = pl_req.execute()
        except HttpError as e:
            raise RuntimeError(f"YouTube API error: {e}") from e

        for it in pl_resp.get("items", []):
            vid = it.get("contentDetails", {}).get("videoId")
            if vid:
                out.append(vid)

        page_token = pl_resp.get("nextPageToken")
        if not page_token:
            break
    return out


def fetch_videos_details(
    youtube_api_key: str, video_ids: Iterable[str]
) -> List[CatalogVideo]:
    """Fetch detailed metadata for given video IDs and produce catalog entries.

    Args:
        youtube_api_key (str): The YouTube API key.
        video_ids (Iterable[str]): Iterable of video IDs.

    Returns:
        list[CatalogVideo]: Catalog entries with derived flags and durations.

    """
    yt = build("youtube", "v3", developerKey=youtube_api_key)
    vids = list(video_ids)
    out: List[CatalogVideo] = []
    for i in range(0, len(vids), 50):
        batch = vids[i : i + 50]
        try:
            req = yt.videos().list(
                id=",".join(batch),
                part="contentDetails,liveStreamingDetails,snippet",
                maxResults=50,
            )
            resp = req.execute()
        except HttpError as e:
            raise RuntimeError(f"YouTube API error: {e}") from e

        for item in resp.get("items", []):
            vid = item.get("id")
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            live = item.get("liveStreamingDetails", {})

            title = snippet.get("title") or ""
            description = snippet.get("description") or ""
            published_at_raw = snippet.get("publishedAt")
            published_at = (
                datetime.fromisoformat(published_at_raw.replace("Z", "+00:00"))
                if published_at_raw
                else datetime.now(timezone.utc)
            )
            duration_iso = content.get("duration") or "PT0S"
            duration_seconds = _parse_duration_iso8601(duration_iso)

            is_short = duration_seconds < SHORTS_THRESHOLD_SECONDS  # Heuristic
            live_broadcast_content = snippet.get("liveBroadcastContent", "none")
            is_livestream = bool(live) or live_broadcast_content in {"live", "upcoming"}
            is_podcast = "podcast" in title.lower()

            channel_id = snippet.get("channelId") or ""
            channel_title = snippet.get("channelTitle")

            try:
                out.append(
                    CatalogVideo(
                        video_id=vid,
                        title=title,
                        description=description,
                        published_at=published_at,
                        duration_seconds=duration_seconds,
                        is_short=is_short,
                        is_livestream=is_livestream,
                        is_podcast=is_podcast,
                        channel_id=channel_id,
                        channel_title=channel_title,
                    )
                )
            except ValidationError:
                continue
    return out
