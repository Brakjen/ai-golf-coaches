from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, HttpUrl

from ai_golf_coaches.config import get_settings
from ai_golf_coaches.models import VideoMeta

# Set up logging
logger = logging.getLogger(__name__)

# IO paths


def channel_dir(name: str) -> Path:
    d = Path("data") / "raw" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def catalog_path(name: str) -> Path:
    return channel_dir(name) / "_catalog.json"


# YouTube API


def yt_client():
    cfg = get_settings()
    return build(
        "youtube",
        "v3",
        developerKey=cfg.youtube.api_key.get_secret_value(),
        cache_discovery=False,
    )


def get_uploads_playlist_id(yt, channel_id: str) -> str:
    r = yt.channels().list(part="contentDetails", id=channel_id).execute()
    return r["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_all_uploads(yt, uploads_pid: str, page_size: int = 50) -> List[Dict]:
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
    m = _DUR_RE.fullmatch(s or "")
    if not m:
        return 0
    h = int(m.group(1) or 0)
    m_ = int(m.group(2) or 0)
    s_ = int(m.group(3) or 0)
    return h * 3600 + m_ * 60 + s_


def _videos_details_by_ids(yt, ids: List[str]) -> Dict[str, Dict]:
    """Batch videos.list (50 per call). Returns dict[videoId] -> detail item."""
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
    # Pydantic v2
    if hasattr(m, "model_dump"):
        return m.model_dump(mode="json")  # ensures HttpUrl -> str, etc.
    # Pydantic v1
    return json.loads(m.json())  # uses pydantic encoders


def save_catalog(channel_handle: str, metas: List[VideoMeta]) -> Path:
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
    if channel_handle.startswith("@"):
        channel_handle = channel_handle[1:]
    path = catalog_path(channel_handle)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [VideoMeta(**row) for row in data]
