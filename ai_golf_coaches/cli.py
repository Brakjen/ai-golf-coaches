from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import AppSettings, load_channels_config, resolve_channel_key
from .transcripts import fetch_transcript_chunks, write_transcript_jsonl
from .youtube_client import (
    fetch_channel_uploads_video_ids,
    fetch_videos_details,
)

app = typer.Typer(
    help="AI Golf Coaches CLI", rich_markup_mode=None, pretty_exceptions_enable=False
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"


def _channel_paths(channel_key: str) -> tuple[Path, Path]:
    base = DATA_DIR / channel_key
    catalog = base / "catalog.jsonl"
    transcripts_dir = base / "transcripts"
    return catalog, transcripts_dir


def _load_channel_id(channel_key: str) -> str:
    channels = load_channels_config(CONFIG_DIR / "channels.yaml")
    entry = channels.get(channel_key)
    if not entry or not entry.get("channel_id"):
        raise RuntimeError(
            f"Channel '{channel_key}' missing or channel_id not configured in channels.yaml"
        )
    return str(entry["channel_id"])


@app.command()
def resolve(
    alias_or_key: str = typer.Argument(..., help="Alias, handle, or canonical key"),
) -> None:
    """Resolve an alias or handle to the canonical channel key.

    Args:
        alias_or_key (str): Alias (e.g., 'egs', 'milo'), handle, or canonical key.

    Raises:
        typer.Exit: If the argument is missing or no channel matches.

    """
    if alias_or_key is None or str(alias_or_key).strip() == "":
        typer.echo("Missing required argument: alias_or_key. Try: aig resolve <alias>")
        raise typer.Exit(code=2)
    channels = load_channels_config(CONFIG_DIR / "channels.yaml")
    key = resolve_channel_key(alias_or_key, channels)
    if not key:
        typer.echo("No matching channel key found.")
        raise typer.Exit(code=1)
    typer.echo(key)


@app.command()
def build_catalog(
    channel: str = typer.Argument(..., help="Alias, handle, or canonical key"),
) -> None:
    """Build or refresh the catalog.jsonl for a channel.

    Currently includes all videos and flags livestreams when possible.

    Args:
        channel (str): Alias, handle, or canonical key for the channel.

    Raises:
        typer.Exit: If the argument is missing or the channel cannot be resolved.
        RuntimeError: If channel_id is missing from configuration.

    """
    if channel is None or str(channel).strip() == "":
        typer.echo("Missing required argument: channel. Try: aig build-catalog <alias>")
        raise typer.Exit(code=2)
    settings = AppSettings()  # Loads env vars
    channels = load_channels_config(CONFIG_DIR / "channels.yaml")
    key = resolve_channel_key(channel, channels)
    if not key:
        typer.echo("Channel not found for provided alias/handle.")
        raise typer.Exit(code=1)

    channel_id = _load_channel_id(key)
    catalog_path, _ = _channel_paths(key)

    # Use uploads playlist for complete coverage
    video_ids = list(
        fetch_channel_uploads_video_ids(settings.youtube.api_key, channel_id)
    )
    videos = fetch_videos_details(settings.youtube.api_key, video_ids)
    # No length filtering: include Shorts and long-form videos

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
            # Write UTF-8 directly to preserve characters in titles
            f.write("  " + json.dumps(obj, ensure_ascii=False) + "\n")
    typer.echo(f"Catalog written: {catalog_path}")


@app.command()
def fetch_transcripts(
    channel: str = typer.Argument(..., help="Alias, handle, or canonical key"),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Max videos to process"
    ),
) -> None:
    """Fetch transcripts for catalog videos that don't have them yet.

    Applies proxy settings and backoff; stores to data/<channel>/transcripts/<video_id>.jsonl.

    Args:
        channel (str): Alias, handle, or canonical key for the channel.
        limit (int | None): Optional cap on number of videos to process.

    Raises:
        typer.Exit: If channel cannot be resolved or catalog is missing.

    """
    if channel is None or str(channel).strip() == "":
        typer.echo(
            "Missing required argument: channel. Try: aig fetch-transcripts <alias>"
        )
        raise typer.Exit(code=2)
    settings = AppSettings()
    channels = load_channels_config(CONFIG_DIR / "channels.yaml")
    key = resolve_channel_key(channel, channels)
    if not key:
        typer.echo("Channel not found for provided alias/handle.")
        raise typer.Exit(code=1)

    catalog_path, transcripts_dir = _channel_paths(key)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Load catalog entries
    if not catalog_path.exists():
        typer.echo("Catalog does not exist. Run build_catalog first.")
        raise typer.Exit(code=1)

    processed = 0
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and processed >= limit:
                break
            entry = json.loads(line)
            video_id = entry.get("video_id")
            if not video_id:
                continue
            out_path = transcripts_dir / f"{video_id}.jsonl"
            if out_path.exists():
                continue
            try:
                chunks = fetch_transcript_chunks(
                    video_id,
                    http_proxy=settings.proxy.http,
                    https_proxy=settings.proxy.https,
                )
            except (Exception,) as e:  # Keep broad but non-fatal
                typer.echo(f"Failed transcript for {video_id}: {e}")
                continue
            write_transcript_jsonl(out_path, chunks)
            processed += 1
            typer.echo(f"Transcript saved: {out_path}")


def main() -> None:
    """Entrypoint for Typer CLI.

    Raises:
        SystemExit: Propagated by Typer on command completion or error.

    """
    app()


if __name__ == "__main__":
    main()
