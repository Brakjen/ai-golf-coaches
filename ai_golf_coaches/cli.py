from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from .agent import _format_video_recommendations, run_agent
from .config import AppSettings, load_channels_config, resolve_channel_key
from .constants import COMBINE_DEFAULT_SECONDS, FETCH_MAX_WORKERS
from .indexing import (
    build_faiss_index,
    build_hosted_vector_stores_longform,
    query_hosted_vector_store,
    query_index,
)
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
                "description": v.description or "",
                "published_at": v.published_at.isoformat(),
                "duration_seconds": v.duration_seconds,
                "is_livestream": v.is_livestream,
                "is_podcast": v.is_podcast,
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

    p_index = sub.add_parser(
        "build-index",
        help="Build a FAISS index for a channel using OpenAI embeddings",
    )
    p_index.add_argument("channel", help="Alias, handle, or canonical key")
    p_index.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model name (default: text-embedding-3-large)",
    )
    p_index.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding requests",
    )
    p_index.set_defaults(
        func=lambda args: (
            (
                lambda res: (
                    print(f"Indexed {res[0]} chunks\nindex={res[1]}\nmeta={res[2]}"),
                    0,
                )[1]
            )(build_faiss_index(args.channel, args.model, args.batch_size))
        )
    )

    p_query = sub.add_parser(
        "query-index",
        help="Query a channel index with a question and return top chunks",
    )
    p_query.add_argument("channel", help="Alias, handle, or canonical key")
    p_query.add_argument("question", help="Natural language query")
    p_query.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return",
    )
    p_query.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model name (default: text-embedding-3-large)",
    )
    p_query.set_defaults(
        func=lambda args: (
            (
                lambda res: (
                    [
                        (
                            print(
                                f"score={score:.4f} video_id={doc.video_id} start={doc.start:.2f} title={doc.title} livestream={doc.is_livestream}\ntext={doc.text}"
                            )
                        )
                        for doc, score in res
                    ],
                    0,
                )[1]
            )(query_index(args.channel, args.question, args.top_k, args.model))
        )
    )

    p_vs = sub.add_parser(
        "build-vector-stores",
        help="Build OpenAI-hosted vector stores (idempotent - reuses existing stores from config)",
    )
    p_vs.add_argument("channel", help="Alias, handle, or canonical key")
    p_vs.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated category list (default: all configured categories)",
    )
    p_vs.add_argument(
        "--include-livestreams",
        action="store_true",
        help="Include livestream chunks in longform ingestion",
    )
    p_vs.add_argument(
        "--max-part-mb",
        type=int,
        default=100,
        help="Max size per uploaded file part in MB (default: 100)",
    )
    p_vs.add_argument(
        "--build-all-store",
        action="store_true",
        default=True,
        help='Build "all" catch-all store for RAG fallback (default: true, NOT for static context)',
    )
    p_vs.add_argument(
        "--no-build-all-store",
        action="store_false",
        dest="build_all_store",
        help='Skip building "all" catch-all store',
    )

    def _cmd_build_vector_stores(args: argparse.Namespace) -> int:
        if args.channel is None or str(args.channel).strip() == "":
            print(
                "Missing required argument: channel. Try: aig build-vector-stores <alias>"
            )
            return 2
        # Hosted vector store ingestion must use a dedicated app-owned key.
        # Do not fall back to OPENAI__API_KEY (user completion key).
        _ = AppSettings()  # still validate env structure early
        if not os.getenv("OPENAI__API_KEY_STORAGE_INDEX"):
            print("OPENAI__API_KEY_STORAGE_INDEX not configured.")
            return 1

        cats = None
        if args.categories:
            cats = [c.strip() for c in str(args.categories).split(",") if c.strip()]

        vs_ids, missing = build_hosted_vector_stores_longform(
            args.channel,
            categories=cats,
            include_livestreams=bool(args.include_livestreams),
            max_part_mb=int(args.max_part_mb),
            build_all_store=bool(args.build_all_store),
        )

        print(
            "\n"
            + "=" * 70
            + "\nVector stores created/updated (paste into config/channels.yaml):"
        )
        print("Under vector_store_ids.longform:")
        for cat, vs_id in sorted(vs_ids.items()):
            print(f"  {cat}: {vs_id}")

        if "all" in vs_ids:
            print(
                '\nNote: "all" store is for RAG retrieval fallback only (NOT for static context)'
            )

        print(
            "\n\u2713 Command is idempotent - reuses existing stores from config and"
            "\n  refreshes their content. Safe to re-run when you add new transcripts."
        )

        any_missing = any(missing.get(cat) for cat in missing)
        if any_missing:
            print("\nMissing transcripts (video_id) by category:")
            for cat, vids in sorted(missing.items()):
                if vids:
                    print(f"  {cat}: {', '.join(vids)}")
        return 0

    p_vs.set_defaults(func=_cmd_build_vector_stores)

    p_query_vs = sub.add_parser(
        "query-vector-store",
        help="Query OpenAI-hosted vector store for relevant transcript chunks",
    )
    p_query_vs.add_argument("channel", help="Alias, handle, or canonical key")
    p_query_vs.add_argument("question", help="Natural language question")
    p_query_vs.add_argument("category", help="Category to query (e.g., full_swing)")
    p_query_vs.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of results to return (default: 20)",
    )

    def _cmd_query_vector_store(args: argparse.Namespace) -> int:
        if not args.channel or not args.question or not args.category:
            print("Missing required arguments: channel, question, category")
            return 2

        try:
            chunks = query_hosted_vector_store(
                args.channel, args.question, args.category, args.top_k
            )
        except Exception as e:
            print(f"Error: {e}")
            return 1

        if not chunks:
            print("No results found.")
            return 0

        print(f"Found {len(chunks)} results:\n")
        for i, chunk in enumerate(chunks, 1):
            timestamp_url = (
                f"https://youtube.com/watch?v={chunk.video_id}&t={int(chunk.start)}s"
            )
            score_str = (
                f" (score: {chunk.score:.4f})" if chunk.score is not None else ""
            )
            print(f"{i}. [{chunk.title or chunk.video_id}]({timestamp_url}){score_str}")
            print(f"   Time: {chunk.start:.2f}s - {chunk.end:.2f}s")
            print(f"   Livestream: {chunk.is_livestream}")
            print(
                f"   Text: {chunk.text[:200]}{'...' if len(chunk.text) > 200 else ''}"
            )
            print()

        return 0

    p_query_vs.set_defaults(func=_cmd_query_vector_store)

    p_agent = sub.add_parser(
        "agent",
        help="Run a minimal no-RAG agent using static context and channel instructions",
    )
    p_agent.add_argument("channel", help="Alias, handle, or canonical key")
    p_agent.add_argument("question", help="Natural language question")
    p_agent.add_argument(
        "--json",
        action="store_true",
        help="Output response as JSON instead of formatted text",
    )

    def _cmd_agent(args: argparse.Namespace) -> int:
        response = run_agent(args.channel, args.question)

        if args.json:
            # Output as JSON
            import json

            print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        else:
            # Output formatted for terminal
            print(response.response_text)
            if response.video_recommendations:
                video_section = _format_video_recommendations(
                    response.video_recommendations
                )
                print(video_section)

        return 0

    p_agent.set_defaults(func=_cmd_agent)

    return parser


def main() -> None:
    """Entrypoint for the argparse-based CLI."""
    parser = build_parser()
    args = parser.parse_args()
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
