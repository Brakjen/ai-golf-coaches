"""Command-line interface for AI Golf Coaches application.

This module provides the main CLI for managing golf instruction data,
building vector indices, and querying the RAG system. It supports
building catalogs, fetching transcripts, creating indices, and asking
questions to golf coaches.

Commands:
    build-catalog: Build video catalog from YouTube channels
    fetch-transcripts: Download transcripts for videos
    build-index: Create vector embeddings for RAG system
    status: Show current state of data and indices
    ask: Query the golf instruction system

The CLI is designed to be the primary interface for both data management
and end-user interactions with the golf coaching system.
"""

import argparse
import logging
from pathlib import Path

from ai_golf_coaches import rag, transcripts, youtube
from ai_golf_coaches.config import get_settings
from ai_golf_coaches.rag import EMBEDDING_MODELS, LLM_MODELS

logger = logging.getLogger(__name__)


def cmd_build_catalog(args: argparse.Namespace) -> int:
    """Build or update video catalog for a YouTube channel."""
    try:
        # Resolve channel from config if needed
        channel_identifier = args.channel

        if channel_identifier is None:
            raise ValueError("Channel must be specified with --channel")

        logger.info(f"Building catalog for channel: {channel_identifier}")

        # Build catalog
        catalog = youtube.build_catalog_for_channel_id(
            channel_identifier=channel_identifier,
            page_size=args.page_size,
            duration_thrs_seconds=args.min_duration,
        )

        # Save to local storage
        _, handle = youtube.resolve_channel_identifier(channel_identifier)
        saved_path = youtube.save_catalog(handle, catalog)

        logger.info(f"✅ Catalog saved: {saved_path}")
        logger.info(f"📊 Videos in catalog: {len(catalog)}")

        return 0

    except Exception as e:
        logger.error(f"❌ Error building catalog: {e}")
        return 1


def cmd_fetch_transcripts(args: argparse.Namespace) -> int:
    """Fetch transcripts for videos in a channel."""
    try:
        # Resolve channel from config if needed
        channel_identifier = args.channel

        if channel_identifier is None:
            raise ValueError("Channel must be specified with --channel")

        # Get channel handle for file operations
        _, handle = youtube.resolve_channel_identifier(channel_identifier)
        channel_name = handle.lstrip("@")

        logger.info(f"Fetching transcripts for channel: {handle}")

        # Load catalog
        catalog = youtube.load_catalog(channel_name)
        if not catalog:
            logger.error(
                f"❌ No catalog found for {channel_name}. Run 'build-catalog' first."
            )
            return 1

        logger.info(f"📚 Loaded catalog with {len(catalog)} videos")

        # Determine which videos to process
        if args.all:
            videos_to_process = catalog
            logger.info("🔄 Processing ALL videos in catalog")
        else:
            videos_to_process = youtube.get_missing_transcripts(channel_name, catalog)

        if not videos_to_process:
            logger.info("✅ No videos need transcript processing")
            return 0

        # Set up rate limiting config
        rate_config = transcripts.RateLimitConfig(
            min_delay=args.min_delay,
            max_delay=args.max_delay,
            max_workers=args.max_workers,
        )

        # Fetch transcripts
        fetcher = transcripts.TranscriptFetcher(rate_config)
        records = fetcher.fetch_transcripts_parallel(videos_to_process)

        # Save results
        saved_paths = youtube.save_video_records(channel_name, records)

        # Report results
        status_counts = {}
        for record in records:
            status = record.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        logger.info(f"✅ Processed {len(records)} videos")
        for status, count in status_counts.items():
            logger.info(f"   {status}: {count}")
        logger.info(f"💾 Saved {len(saved_paths)} video records")

        return 0

    except Exception as e:
        logger.error(f"❌ Error fetching transcripts: {e}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of catalogs and transcripts."""
    try:
        config = get_settings()

        # Get configured channels
        channels = []
        if config.youtube.egs.handle:
            channels.append(("EGS", config.youtube.egs.handle.lstrip("@")))
        if config.youtube.milo.handle:
            channels.append(("Milo", config.youtube.milo.handle.lstrip("@")))

        if not channels:
            logger.warning("No channels configured")
            return 1

        logger.info("📊 AI Golf Coaches Status")
        logger.info("=" * 40)

        for coach_name, channel_name in channels:
            logger.info(f"\n{coach_name} ({channel_name}):")

            # Check catalog
            catalog = youtube.load_catalog(channel_name)
            if catalog:
                logger.info(f"  📚 Catalog: {len(catalog)} videos")

                # Check transcripts
                records = youtube.load_channel_records(channel_name)
                record_map = {r.meta.video_id: r for r in records}

                transcript_stats = {
                    "ok": 0,
                    "error": 0,
                    "transcripts_disabled": 0,
                    "not_found": 0,
                    "missing": 0,
                }

                for video in catalog:
                    record = record_map.get(video.video_id)
                    if record:
                        status = record.status.value
                        transcript_stats[status] = transcript_stats.get(status, 0) + 1
                    else:
                        transcript_stats["missing"] += 1

                logger.info("  📝 Transcripts:")
                for status, count in transcript_stats.items():
                    if count > 0:
                        logger.info(f"    {status}: {count}")
            else:
                logger.info("  📚 Catalog: Not found")

        # Check index
        index_dir = Path("data/index/youtube")
        if index_dir.exists():
            logger.info(f"\n🔍 Vector Index: Available at {index_dir}")
        else:
            logger.info("\n🔍 Vector Index: Not built")

        return 0

    except Exception as e:
        logger.error(f"❌ Error checking status: {e}")
        return 1


def cmd_build_index(args: argparse.Namespace) -> int:
    """Build or rebuild the vector index for RAG queries."""
    try:
        logger.info("🔨 Building vector index with timed windowing approach...")

        if args.test_only:
            from ai_golf_coaches.rag import create_test_index

            create_test_index()
            logger.info("✅ Test index built successfully")
        else:
            from ai_golf_coaches.rag import create_index

            create_index()
            logger.info("✅ Full index built successfully")

        return 0

    except Exception as e:
        logger.error(f"❌ Error building index: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_ask(args: argparse.Namespace) -> int:
    """Ask a question to the AI golf coaches."""
    try:
        # Check if index exists
        index_dir = Path("data/index/youtube")
        if not index_dir.exists():
            logger.error("❌ Vector index not found. Build it first with:")
            logger.error("   aig build-catalog --channel @elitegolfschools")
            logger.error("   aig fetch-transcripts --channel @elitegolfschools")
            logger.error("   aig build-index")
            return 1

        index_type = "test" if args.test_index else "full"
        logger.info(
            f"🤔 Asking Coach {args.coach.upper() if args.coach != 'all' else 'AI Golf Coaches'} ({index_type} index): {args.question}"
        )

        # Get the answer from RAG
        response = rag.ask(
            args.question, coach=args.coach, use_test_index=args.test_index
        )

        # Print the response nicely formatted (keep prints for user output)
        print("\n" + "=" * 80)  # noqa: T201
        print(  # noqa: T201
            f"🏌️ AI Golf Coach ({args.coach.upper() if args.coach != 'all' else 'All Coaches'}) Response:"
        )
        print("=" * 80)  # noqa: T201
        print(response)  # noqa: T201
        print("=" * 80)  # noqa: T201

        return 0

    except Exception as e:
        logger.error(f"❌ Error asking question: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_test_models(args: argparse.Namespace) -> int:
    """Test different embedding and LLM models with a sample query."""
    from ai_golf_coaches.rag import EMBEDDING_MODELS, LLM_MODELS, setup_models, ask
    
    test_query = args.query or "How do I fix my slice off the tee?"
    coach = args.coach or "riley"
    
    logger.info(f"Testing models with query: '{test_query}'")
    logger.info(f"Coach: {coach}")
    
    # Test embedding model if specified
    if args.embedding_model:
        if args.embedding_model in EMBEDDING_MODELS:
            embedding_model = EMBEDDING_MODELS[args.embedding_model]
        else:
            embedding_model = args.embedding_model
        logger.info(f"Testing embedding model: {embedding_model}")
    else:
        embedding_model = None
    
    # Test LLM model if specified  
    if args.llm_model:
        if args.llm_model in LLM_MODELS:
            llm_model = LLM_MODELS[args.llm_model]
        else:
            llm_model = args.llm_model
        logger.info(f"Testing LLM model: {llm_model}")
    else:
        llm_model = None
    
    try:
        # Setup models and test
        setup_models(llm_model=llm_model, embedding_model=embedding_model)
        
        response = ask(test_query, coach=coach)
        print(f"\n🏌️ Response from {coach.upper()}:\n")  # noqa: T201
        print(response)  # noqa: T201
        print("\n" + "="*80 + "\n")  # noqa: T201
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error testing models: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cli() -> int:
    """Command-line interface for AI Golf Coaches data management."""
    parser = argparse.ArgumentParser(
        description="AI Golf Coaches - Data management and transcript processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build catalog for Elite Golf Schools
  aig build-catalog --channel @elitegolfschools

  # Fetch missing transcripts
  aig fetch-transcripts --channel @elitegolfschools

  # Build vector index for RAG queries
  aig build-index

  # Check status of all channels
  aig status

  # Ask the AI golf coaches questions
  aig ask "How do I fix my slice?"
  aig ask "What is the transition in golf swing?" --coach egs
  aig ask "How should I practice putting?" --coach milo
        """,
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="COMMAND"
    )

    # Build catalog command
    build_parser = subparsers.add_parser(
        "build-catalog",
        help="Build or update video catalog for a YouTube channel",
        description="Fetch video metadata from YouTube and save to local catalog",
    )
    build_parser.add_argument(
        "--channel",
        "-c",
        help="YouTube channel ID (UC...) or handle (@handle). Uses EGS if not specified.",
    )
    build_parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Number of results per API page (max 50, default: 50)",
    )
    build_parser.add_argument(
        "--min-duration",
        type=int,
        default=360,
        help="Minimum video duration in seconds to exclude Shorts (default: 360)",
    )
    build_parser.set_defaults(func=cmd_build_catalog)

    # Fetch transcripts command
    fetch_parser = subparsers.add_parser(
        "fetch-transcripts",
        help="Fetch transcripts for videos in a channel",
        description="Download transcripts for videos with rate limiting and proxy support",
    )
    fetch_parser.add_argument(
        "--channel",
        "-c",
        help="YouTube channel ID (UC...) or handle (@handle). Uses EGS if not specified.",
    )
    fetch_parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch transcripts for ALL videos, not just missing ones",
    )
    fetch_parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of concurrent workers (default: 3)",
    )
    fetch_parser.add_argument(
        "--min-delay",
        type=float,
        default=4.0,
        help="Minimum delay between requests in seconds (default: 4.0)",
    )
    fetch_parser.add_argument(
        "--max-delay",
        type=float,
        default=15.0,
        help="Maximum delay between requests in seconds (default: 15.0)",
    )
    fetch_parser.set_defaults(func=cmd_fetch_transcripts)

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of catalogs and transcripts",
        description="Display overview of available data for all configured channels",
    )
    status_parser.set_defaults(func=cmd_status)

    # Build index command
    index_parser = subparsers.add_parser(
        "build-index",
        help="Build or rebuild the vector index for RAG queries",
        description="Create vector embeddings from transcripts using timed windowing approach",
    )
    index_parser.add_argument(
        "--test-only",
        "-t",
        action="store_true",
        help="Build only the test index (faster, limited content for development)",
    )
    index_parser.set_defaults(func=cmd_build_index)

    # Ask command
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question to the AI golf coaches",
        description="Query the RAG system with golf instruction questions",
    )
    ask_parser.add_argument("question", help="Your golf instruction question")
    ask_parser.add_argument(
        "--coach",
        "-c",
        choices=["egs", "milo", "all"],
        default="all",
        help="Which coach to ask: egs (Elite Golf Schools), milo (Milo Lines Golf), or all (default: all)",
    )
    ask_parser.add_argument(
        "--test-index",
        "-t",
        action="store_true",
        help="Use the focused test index instead of full index (faster, limited content)",
    )
    ask_parser.set_defaults(func=cmd_ask)

    # Test models command
    test_parser = subparsers.add_parser(
        "test-models",
        help="Test different embedding and LLM models",
        description="Compare different models with a sample query to evaluate quality",
    )
    test_parser.add_argument(
        "--query",
        "-q", 
        default="How do I fix my slice off the tee?",
        help="Test query to use (default: 'How do I fix my slice off the tee?')",
    )
    test_parser.add_argument(
        "--coach",
        "-c",
        choices=["egs", "milo", "riley", "all"],
        default="riley",
        help="Which coach to ask (default: riley)",
    )
    test_parser.add_argument(
        "--embedding-model",
        "-e",
        choices=list(EMBEDDING_MODELS.keys()) + ["custom"],
        help=f"Embedding model to test: {', '.join(EMBEDDING_MODELS.keys())} or custom model name",
    )
    test_parser.add_argument(
        "--llm-model",
        "-l",
        choices=list(LLM_MODELS.keys()) + ["custom"],
        help=f"LLM model to test: {', '.join(LLM_MODELS.keys())} or custom model name",
    )
    test_parser.set_defaults(func=cmd_test_models)

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(level=logging.INFO, force=True)

    # Execute command
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(cli())
