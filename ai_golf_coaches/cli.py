from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from .config import AppSettings, load_channels_config, resolve_channel_key
from .constants import COMBINE_DEFAULT_SECONDS, FETCH_MAX_WORKERS
from .transcripts import combine_chunks, fetch_transcript_chunks, write_transcript_jsonl
from .youtube_client import (
    fetch_channel_uploads_video_ids,
    fetch_videos_details,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"

# Defaults now centralized in constants.py


def _channel_paths(channel_key: str) -> Tuple[Path, Path]:
    """Return catalog file path and transcripts directory for a channel.

    Args:
        channel_key (str): Canonical key of the channel.

    Returns:
        Tuple[Path, Path]: Paths to catalog.jsonl and transcripts directory.

    """
    base = DATA_DIR / channel_key
    catalog = base / "catalog.jsonl"
    transcripts_dir = base / "transcripts"
    return catalog, transcripts_dir


def _load_channel_id(channel_key: str) -> str:
    """Load the YouTube channel ID from configuration.

    Args:
        channel_key (str): Canonical key of the channel.

    Returns:
        str: The YouTube channel ID.

    Raises:
        RuntimeError: If the channel or `channel_id` is missing.

    """
    channels = load_channels_config(CONFIG_DIR / "channels.yaml")
    entry = channels.get(channel_key)
    if not entry or not entry.get("channel_id"):
        raise RuntimeError(
            f"Channel '{channel_key}' missing or channel_id not configured in channels.yaml"
        )
    return str(entry["channel_id"])


def cmd_resolve(alias_or_key: str) -> int:
    """Resolve an alias or handle to the canonical channel key.

    Args:
        alias_or_key (str): Alias (e.g., 'egs', 'milo'), handle, or canonical key.

    Returns:
        int: Exit code (0 on success, non-zero on error).

    """
    if alias_or_key is None or str(alias_or_key).strip() == "":
        print("Missing required argument: alias_or_key. Try: aig resolve <alias>")
        return 2
    channels = load_channels_config(CONFIG_DIR / "channels.yaml")
    key = resolve_channel_key(alias_or_key, channels)
    if not key:
        print("No matching channel key found.")
        return 1
    print(key)
    return 0


def cmd_build_catalog(channel: str) -> int:
    """Build or refresh the catalog.jsonl for a channel.

    Currently includes all videos and flags livestreams when possible.

    Args:
        channel (str): Alias, handle, or canonical key for the channel.

    Returns:
        int: Exit code (0 on success, non-zero on error).

    """
    if channel is None or str(channel).strip() == "":
        print("Missing required argument: channel. Try: aig build-catalog <alias>")
        return 2
    settings = AppSettings()  # Loads env vars
    channels = load_channels_config(CONFIG_DIR / "channels.yaml")
    key = resolve_channel_key(channel, channels)
    if not key:
        print("Channel not found for provided alias/handle.")
        return 1

    channel_id = _load_channel_id(key)
    catalog_path, _ = _channel_paths(key)

    video_ids = list(
        fetch_channel_uploads_video_ids(settings.youtube.api_key, channel_id)
    )
    videos = fetch_videos_details(settings.youtube.api_key, video_ids)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("w", encoding="utf-8") as f:
        for v in videos:
            if v.is_short:
                continue  # Skip Shorts
            obj = {
                "video_id": v.video_id,
                "title": v.title,
                "published_at": v.published_at.isoformat(),
                "duration_seconds": v.duration_seconds,
                "is_livestream": v.is_livestream,
                "channel_id": v.channel_id,
                "channel_title": v.channel_title,
            }
            f.write("  " + json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Catalog written: {catalog_path}")
    return 0


def cmd_fetch_transcripts(
    channel: str,
    limit: Optional[int],
    combine: bool = False,
    combine_seconds: Optional[float] = None,
) -> int:
    """Fetch transcripts for catalog videos that don't have them yet.

    Applies proxy settings and backoff; stores to data/<channel>/transcripts/<video_id>.jsonl.

    Args:
        channel (str): Alias, handle, or canonical key for the channel.
        limit (Optional[int]): Max videos to process.
        combine (bool): Whether to combine chunks into windows before writing.
        combine_seconds (Optional[float]): Window size in seconds when combining.

    Returns:
        int: Exit code (0 on success, non-zero on error).

    """
    if channel is None or str(channel).strip() == "":
        print("Missing required argument: channel. Try: aig fetch-transcripts <alias>")
        return 2
    channels = load_channels_config(CONFIG_DIR / "channels.yaml")
    key = resolve_channel_key(channel, channels)
    if not key:
        print("Channel not found for provided alias/handle.")
        return 1

    catalog_path, transcripts_dir = _channel_paths(key)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if not catalog_path.exists():
        print("Catalog does not exist. Run build-catalog first.")
        return 1

    # Proxies are configured via Webshare credentials in transcripts._build_ytt_api_from_env()

    # Pre-compute catalog video ids and pending fetch targets
    all_ids = []
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            vid = entry.get("video_id")
            if vid:
                all_ids.append(vid)

    pending = []
    for vid in all_ids:
        out_path = transcripts_dir / f"{vid}.jsonl"
        if not out_path.exists():
            pending.append(vid)

    planned = pending[: limit if limit is not None else None]

    # Print status before starting (aligned labels and values)
    plan_window = (
        combine_seconds
        if combine and combine_seconds is not None
        else (COMBINE_DEFAULT_SECONDS if combine else None)
    )
    print("Starting transcript fetch:")
    items = [
        ("channel", key),
        ("catalog", str(catalog_path)),
        ("total", str(len(all_ids))),
        ("existing", str(len(all_ids) - len(pending))),
        ("to_fetch", str(len(planned))),
        ("combine", "true" if combine else "false"),
        ("window", str(plan_window) if combine else "N/A"),
    ]

    width = max(len(k) for k, _ in items)
    for k, v in items:
        print(f"{k:<{width}} : {v}")

    processed = 0
    # Fetch in parallel using a small thread pool (I/O bound work)
    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(fetch_transcript_chunks, vid): vid for vid in planned
        }
        for fut in as_completed(future_map):
            video_id = future_map[fut]
            out_path = transcripts_dir / f"{video_id}.jsonl"
            try:
                chunks = fut.result()
            except Exception as e:
                print(
                    f"FAIL video_id={video_id} status=? error={type(e).__name__}: {e}"
                )
                continue
            if combine:
                window = (
                    combine_seconds
                    if combine_seconds is not None
                    else COMBINE_DEFAULT_SECONDS
                )
                try:
                    chunks = combine_chunks(chunks, window)
                except Exception as e:
                    print(f"FAIL video_id={video_id} status=? error=CombineError: {e}")
                    continue
            write_transcript_jsonl(out_path, chunks)
            processed += 1
            print(f"OK video_id={video_id} | saved={out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with subcommands.

    Returns:
        argparse.ArgumentParser: Configured parser.

    """
    parser = argparse.ArgumentParser(
        prog="aig",
        description="AI Golf Coaches CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser(
        "resolve", help="Resolve an alias or handle to the canonical channel key"
    )
    p_resolve.add_argument("alias_or_key", help="Alias, handle, or canonical key")
    p_resolve.set_defaults(func=lambda args: cmd_resolve(args.alias_or_key))

    p_build = sub.add_parser(
        "build-catalog", help="Build or refresh the catalog.jsonl for a channel"
    )
    p_build.add_argument("channel", help="Alias, handle, or canonical key")
    p_build.set_defaults(func=lambda args: cmd_build_catalog(args.channel))

    p_fetch = sub.add_parser(
        "fetch-transcripts",
        help="Fetch transcripts for catalog videos that don't have them yet",
    )
    p_fetch.add_argument("channel", help="Alias, handle, or canonical key")
    p_fetch.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Max videos to process",
    )
    p_fetch.add_argument(
        "-c",
        "--combine",
        action="store_true",
        help="Combine chunks into windows before writing",
    )
    p_fetch.add_argument(
        "--combine-seconds",
        type=float,
        default=None,
        help=f"Window size in seconds when combining (default {COMBINE_DEFAULT_SECONDS} if --combine is set)",
    )
    p_fetch.set_defaults(
        func=lambda args: cmd_fetch_transcripts(
            args.channel,
            args.limit,
            args.combine,
            args.combine_seconds,
        )
    )

    return parser


def main() -> None:
    """Entrypoint for the argparse-based CLI."""
    parser = build_parser()
    args = parser.parse_args()
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
