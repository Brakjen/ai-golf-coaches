"""Lightweight RAG system for AI Golf Coaches using automatic methods."""

from __future__ import annotations

import contextlib
import logging
import os

# Optimize for multi-core performance (16 cores available)
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from ai_golf_coaches.config import get_settings
from ai_golf_coaches.personalities import get_coach_prompt

# Recommended embedding models (as of 2025)
EMBEDDING_MODELS = {
    "bge_m3": "BAAI/bge-m3",  # Best overall: multilingual, multi-granularity
    "bge_large": "BAAI/bge-large-en-v1.5",  # Best for English content
    "e5_base": "intfloat/multilingual-e5-base",  # Current default (good baseline)
    "openai_large": "text-embedding-3-large",  # OpenAI's latest (requires API)
    "openai_small": "text-embedding-3-small",  # OpenAI smaller but fast
}

# Recommended LLM models
LLM_MODELS = {
    "gpt4o": "gpt-4o",  # Latest GPT-4 Omni (best quality)
    "gpt4_turbo": "gpt-4-turbo",  # GPT-4 Turbo (good balance)
    "gpt4o_mini": "gpt-4o-mini",  # Current default (fast, cheaper)
}

logger = logging.getLogger(__name__)


WINDOW_SIZE_SEC = 150
WINDOW_OVERLAP_SEC = 20
PERSIST_DIR = Path("data/index/youtube")
TEST_PERSIST_DIR = Path("data/index/youtube_test")
CACHE_DIR = Path("data/cache")


@dataclass
class TimedWindow:
    """Represents a semantic window with precise timing from transcript segments.

    A TimedWindow combines multiple transcript lines into a semantically coherent
    segment with exact start and end times for precise video deep-linking.

    Attributes:
        start_time: Start time in seconds from video beginning
        end_time: End time in seconds from video beginning
        text: Combined text content from all lines in this window
        line_count: Number of transcript lines included in this window
        video_id: YouTube video identifier
        title: Video title for reference

    Notes:
        - Designed for 45-75 second semantic segments
        - Enables precise timestamp citations in responses
        - Maintains natural speech boundaries when possible

    """

    start_time: float
    end_time: float
    text: str
    line_count: int
    video_id: str
    title: str


def _ensure_cache_dir() -> None:
    """Ensure cache directory exists.

    Creates the cache directory if it doesn't exist, ignoring any errors.
    Used for caching philosophy summaries and other temporary data.
    """
    with contextlib.suppress(Exception):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extract_coach_philosophy(
    coach_dir: str = "elitegolfschools", max_lines: int = 50
) -> str:
    """Try to extract a concise 'philosophy' summary from the coach transcripts.

    This function scans transcript files for repeated high-level phrases and
    returns a short paragraph summarizing the coach's teaching philosophy.

    It's a heuristic extractor (not perfect) and is intended to enrich system
    prompts so the LLM can adopt consistent voice and emphasis.
    """
    _ensure_cache_dir()
    cache_file = CACHE_DIR / f"{coach_dir}_philosophy.txt"

    # Return cached if present
    if cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8")
        except Exception:
            pass

    import json
    from collections import Counter

    phrases = Counter()
    coach_path = Path("data/raw") / coach_dir
    if not coach_path.exists():
        return ""

    json_files = list(coach_path.glob("*.json"))
    json_files = [f for f in json_files if f.name != "_catalog.json"]

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            transcript = data.get("transcript", {})
            text = transcript.get("text", "")
            # collect sentence-level phrases
            for sent in text.split(". ")[:max_lines]:
                key = sent.strip().lower()
                if len(key) > 30:
                    phrases[key] += 1
        except Exception:
            continue

    # Take the most common long-ish phrases to build a short philosophy
    common = [p for p, _ in phrases.most_common(10) if len(p) > 40]
    if not common:
        return ""

    summary = " ".join(common[:3])
    # Persist the summary for future runs
    with contextlib.suppress(Exception):
        cache_file.write_text(summary, encoding="utf-8")

    return summary


def build_semantic_windows(
    lines: List[Dict[str, Any]],
    video_id: str,
    title: str,
    target_duration: float = 45.0,
    max_duration: float = 75.0,
) -> List[TimedWindow]:
    """Build semantic windows from transcript lines with precise timing.

    Args:
        lines: List of transcript line dicts with 'start', 'duration', 'text'
        video_id: Video identifier
        title: Video title
        target_duration: Target window duration in seconds
        max_duration: Maximum window duration before forced split

    Returns:
        List of TimedWindow objects with precise timing

    """
    if not lines:
        return []

    windows = []
    current_lines = []

    for line in lines:
        # Skip music/noise markers
        text = line.get("text", "").strip()
        if not text or text in ["[Music]", "[Applause]", "♪", "♫"]:
            continue

        current_lines.append(line)

        # Calculate current window duration
        if len(current_lines) > 1:
            window_start = current_lines[0]["start"]
            window_end = line["start"] + line.get("duration", 0)
            duration = window_end - window_start

            # Split if we've reached target duration or exceeded max
            if duration >= target_duration:
                # Build window from current lines
                start_time = current_lines[0]["start"]
                end_time = current_lines[-1]["start"] + current_lines[-1].get(
                    "duration", 0
                )
                combined_text = " ".join(
                    line["text"].strip()
                    for line in current_lines
                    if line["text"].strip()
                )

                if combined_text:  # Only add non-empty windows
                    windows.append(
                        TimedWindow(
                            start_time=start_time,
                            end_time=end_time,
                            text=combined_text,
                            line_count=len(current_lines),
                            video_id=video_id,
                            title=title,
                        )
                    )

                current_lines = []

    # Handle remaining lines
    if current_lines:
        start_time = current_lines[0]["start"]
        end_time = current_lines[-1]["start"] + current_lines[-1].get("duration", 0)
        combined_text = " ".join(
            line["text"].strip() for line in current_lines if line["text"].strip()
        )

        if combined_text:
            windows.append(
                TimedWindow(
                    start_time=start_time,
                    end_time=end_time,
                    text=combined_text,
                    line_count=len(current_lines),
                    video_id=video_id,
                    title=title,
                )
            )

    return windows


def setup_models(
    llm_model: str | None = None,
    embedding_model: str | None = None,
) -> None:
    """Configure LlamaIndex with enhanced models for better RAG performance.

    Sets up the global Settings object with state-of-the-art models.
    Supports multiple embedding providers and latest LLM models.

    Args:
        llm_model: OpenAI model name to use for text generation (optional, uses config default).
        embedding_model: Embedding model to use (optional, uses config default).

    """
    config = get_settings()
    api_key = config.openai.api_key.get_secret_value()
    
    # Use config defaults if not specified
    llm_model = llm_model or config.openai.model
    embedding_model = embedding_model or config.openai.embedding_model
    
    # Setup LLM
    Settings.llm = OpenAI(
        model=llm_model,
        temperature=0.3,  # Slightly higher for more elaboration
        api_key=api_key,
        max_tokens=config.openai.max_tokens,
    )
    
    # Setup embedding model based on provider
    if config.openai.embedding_provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        Settings.embed_model = OpenAIEmbedding(
            model=embedding_model,  # e.g., "text-embedding-3-large"
            api_key=api_key,
        )
        logger.info(f"Configured OpenAI embedding model: {embedding_model}")
    else:
        # Default to HuggingFace with better models
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            trust_remote_code=True,
            embed_batch_size=64,  # Optimized for 16 cores
            max_length=512,
        )
        logger.info(f"Configured HuggingFace embedding model: {embedding_model}")
    
    logger.info(f"Configured LLM: {llm_model}")


def create_index(
    coach: str = "all", data_path: str | Path = "data/raw"
) -> VectorStoreIndex:
    """Create and persist a vector index from transcript data.

    This function loads transcript documents from the specified data path,
    builds a vector index using the configured LLM and embedding models, and
    persists the index to disk for later use.

    Args:
        coach (str): Which coach's data to index ("egs", "milo", or "all").
        data_path (str | Path): Path to transcript data directory (default: "data/raw").

    Returns:
        VectorStoreIndex: The created vector index.

    """
    setup_models()

    if coach == "all":
        input_path = str(data_path)
    else:
        coach_mapping = {"egs": "elitegolfschools", "milo": "milolinesgolf"}
        coach_dir = coach_mapping.get(coach, coach)
        input_path = str(Path(data_path) / coach_dir)

    # Load and process documents with better chunking
    import json

    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    documents = []

    # Process each JSON file individually for better control
    json_files = list(Path(input_path).rglob("*.json"))
    json_files = [f for f in json_files if f.name != "_catalog.json"]

    logger.info(f"Processing {len(json_files)} transcript files for coach: {coach}")

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Extract video metadata
            meta = data.get("meta", {})
            video_id = meta.get("video_id", json_file.stem)
            title = meta.get("title", "Unknown Title")
            channel = meta.get("channel_title", "Unknown Channel")

            # Extract transcript lines for timed windows
            transcript = data.get("transcript", {})
            lines = transcript.get("lines", [])

            if not lines:
                logger.warning(f"No transcript lines found in {json_file.name}")
                continue

            # Build semantic windows with precise timing
            windows = build_semantic_windows(lines, video_id, title)

            # Create document for each window
            for i, window in enumerate(windows):
                # Create enhanced document with title and context
                enhanced_content = f"""
Video Title: {title}
Channel: {channel}
Video ID: {video_id}
Window: {i+1}/{len(windows)} ({window.start_time:.1f}s - {window.end_time:.1f}s)

Transcript Content:
{window.text}
                """.strip()

                # Create document with enhanced metadata including precise timing
                doc = Document(
                    text=enhanced_content,
                    metadata={
                        "video_id": video_id,
                        "title": title,
                        "channel": channel,
                        "file_path": str(json_file),
                        "file_name": json_file.name,
                        "start_time": window.start_time,
                        "end_time": window.end_time,
                        "window_index": i,
                        "total_windows": len(windows),
                        "line_count": window.line_count,
                        "coach": "EGS"
                        if "elitegolfschools" in str(json_file)
                        else "Milo",
                        "topic": title.lower(),  # For better topic matching
                    },
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue

    logger.info(f"Successfully processed {len(documents)} documents for coach: {coach}")

    # Use better chunking strategy
    node_parser = SentenceSplitter(
        chunk_size=1024,  # Larger chunks for better context
        chunk_overlap=200,  # More overlap to maintain context
    )

    # Create index with custom node parser
    index = VectorStoreIndex.from_documents(
        documents, node_parser=node_parser, show_progress=True
    )

    # Persist the index
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    logger.info(f"Persisted enhanced index to {PERSIST_DIR}")
    return index


def create_test_index(
    max_videos: int = 20, include_putting: bool = True
) -> VectorStoreIndex:
    """Create a smaller test index for development and debugging.

    This function creates a focused index using a subset of videos,
    making it faster to iterate during development.

    Args:
        max_videos: Maximum number of videos to include in test index.
        include_putting: Whether to include putting-related videos in test index.

    Returns:
        VectorStoreIndex: The test vector index.

    """
    setup_models()

    import json

    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    documents = []
    video_count = 0

    # Priority videos to include (putting videos)
    priority_videos = []
    if include_putting:
        priority_videos = ["69I1SF2-vc0.json", "ZRiAZp90Rag.json"]  # Putting videos

    # Get some videos from each coach for testing
    for coach_dir in ["elitegolfschools", "milolinesgolf"]:
        coach_path = Path("data/raw") / coach_dir
        if not coach_path.exists():
            continue

        json_files = list(coach_path.glob("*.json"))
        json_files = [f for f in json_files if f.name != "_catalog.json"]

        # Prioritize putting videos first, then add others
        priority_files = [f for f in json_files if f.name in priority_videos]
        other_files = [f for f in json_files if f.name not in priority_videos]

        # Combine with priority first
        ordered_files = priority_files + other_files
        videos_from_coach = min(max_videos // 2, len(ordered_files))

        for json_file in ordered_files[:videos_from_coach]:
            if video_count >= max_videos:
                break

            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Extract video metadata
                meta = data.get("meta", {})
                video_id = meta.get("video_id", json_file.stem)
                title = meta.get("title", "Unknown Title")
                channel = meta.get("channel_title", "Unknown Channel")

                # Extract transcript lines for timed windows
                transcript = data.get("transcript", {})
                lines = transcript.get("lines", [])

                if not lines:
                    continue

                # Build semantic windows with precise timing
                windows = build_semantic_windows(lines, video_id, title)

                # Create document for each window
                for i, window in enumerate(windows):
                    # Create enhanced document with title and context
                    enhanced_content = f"""
Video Title: {title}
Channel: {channel}
Video ID: {video_id}
Window: {i+1}/{len(windows)} ({window.start_time:.1f}s - {window.end_time:.1f}s)

Transcript Content:
{window.text}
                    """.strip()

                    # Create document with enhanced metadata including timing
                    doc = Document(
                        text=enhanced_content,
                        metadata={
                            "video_id": video_id,
                            "title": title,
                            "channel": channel,
                            "file_path": str(json_file),
                            "file_name": json_file.name,
                            "start_time": window.start_time,
                            "end_time": window.end_time,
                            "window_index": i,
                            "total_windows": len(windows),
                            "line_count": window.line_count,
                            "topic": title.lower(),
                            "coach": "EGS"
                            if "elitegolfschools" in str(json_file)
                            else "Milo",
                        },
                    )
                    documents.append(doc)

                video_count += 1

                # Show progress
                logger.info(
                    f"Added: {title[:60]}... ({coach_dir}) - {len(windows)} windows"
                )

            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
                continue

        if video_count >= max_videos:
            break

    logger.info(f"Created test dataset with {len(documents)} videos")

    # Use better chunking strategy
    node_parser = SentenceSplitter(
        chunk_size=1024,  # Larger chunks for better context
        chunk_overlap=200,  # More overlap to maintain context
    )

    # Create index with custom node parser
    index = VectorStoreIndex.from_documents(
        documents, node_parser=node_parser, show_progress=True
    )

    # Persist the test index
    index.storage_context.persist(persist_dir=str(TEST_PERSIST_DIR))
    logger.info(f"Persisted test index to {TEST_PERSIST_DIR}")
    return index


def load_index(use_test: bool = False) -> VectorStoreIndex:
    """Load a previously built index from disk.

    Args:
        use_test: If True, load the test index instead of the full index

    Returns:
        VectorStoreIndex: The loaded vector index.

    """
    persist_dir = TEST_PERSIST_DIR if use_test else PERSIST_DIR

    if not persist_dir.exists():
        index_type = "test" if use_test else "full"
        raise FileNotFoundError(
            f"No {index_type} index at {persist_dir}. Build it first."
        )

    setup_models()
    storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage)  # type: ignore[return-value]


def ask(
    question: str,
    coach: str = "all",
    data_path: str | Path = "data/raw",
    use_test_index: bool = False,
) -> str:
    """Ask a question to the specified AI golf coach.

    Args:
        question: Golf instruction question.
        coach: Which coach to ask ("egs", "milo", or "all").
        data_path: Path to transcript data directory (default: "data/raw").
        use_test_index: If True, use the focused test index instead of full index.

    Returns:
        str: The answer from the AI golf coach with citations.

    """
    index = load_index(use_test=use_test_index)

    # Create coach-specific filters
    coach_filters = None
    if coach == "egs":
        # Only include Elite Golf Schools content
        from llama_index.core.vector_stores import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        coach_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="file_path",
                    value="elitegolfschools",
                    operator=FilterOperator.TEXT_MATCH,
                )
            ]
        )
    elif coach == "milo":
        # Only include Milo Lines content
        from llama_index.core.vector_stores import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        coach_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="file_path",
                    value="milolinesgolf",
                    operator=FilterOperator.TEXT_MATCH,
                )
            ]
        )
    # If coach == "all", no filters applied
    # Note: Philosophy now handled in personalities.py

    # Get clean, personality-driven prompt (philosophy built into personalities.py)
    system_prompt = get_coach_prompt(coach)

    # Optimized for speed while maintaining quality
    query_engine = index.as_query_engine(
        system_prompt=system_prompt,
        similarity_top_k=8,  # Reduced for faster processing
        response_mode="compact",  # Faster single-pass generation
        filters=coach_filters,  # Apply coach-specific filtering
        streaming=True,  # Enable streaming responses
    )
    response = query_engine.query(question)

    # Deduplicate by video_id, keeping highest score per video
    by_video: Dict[str, Dict[str, Any]] = {}
    for node in response.source_nodes:
        video_id = node.metadata.get("video_id", "unknown")
        score = node.score or 0.0

        # Keep best scoring node per videoe
        if video_id not in by_video or score > by_video[video_id]["score"]:
            by_video[video_id] = {
                "node": node,
                "score": score,
            }

    # Filter by relevance threshold and select top 3 videos by score
    # Adjusted thresholds for BGE-M3 model (different similarity distribution than E5)
    relevant_videos = [v for v in by_video.values() if v["score"] >= 0.75]
    if len(relevant_videos) < 2:  # If too few high-confidence, lower threshold further
        relevant_videos = [v for v in by_video.values() if v["score"] >= 0.65]

    top_videos = sorted(relevant_videos, key=lambda d: d["score"], reverse=True)[:3]

    # Build citations and verbatim snippets from deduplicated videos
    citations = []
    verbatim_snippets = []

    for entry in top_videos:
        node = entry["node"]

        # Extract metadata (now reliable with timed windows)
        video_id = node.metadata.get("video_id", "unknown")
        title = node.metadata.get("title", f"Video {video_id}")
        start_time = node.metadata.get("start_time")
        end_time = node.metadata.get("end_time")
        coach = node.metadata.get("coach", "Unknown")

        # Map coach to full name
        if coach == "EGS":
            coach_name = "Elite Golf Schools (EGS)"
        elif coach == "Milo":
            coach_name = "Milo Lines Golf"
        else:
            coach_name = "Unknown Coach"

        # Extract verbatim snippet from transcript content
        content = node.text
        snippet = ""
        try:
            if "Transcript Content:" in content:
                snippet = content.split("Transcript Content:")[-1].strip()
            else:
                snippet = content.strip()

            # Clean and truncate snippet
            snippet = snippet.replace("\n", " ")
            if len(snippet) > 350:
                cutoff = snippet.rfind(".", 0, 350)
                if cutoff and cutoff > 100:
                    snippet = snippet[: cutoff + 1]
                else:
                    snippet = snippet[:350] + "..."
        except Exception:
            snippet = ""

        if snippet:
            verbatim_snippets.append(snippet)

        # Build timestamp URL and display using reliable metadata
        timestamp_url_param = ""
        timestamp_display = ""
        if start_time is not None:
            timestamp_url_param = f"&t={int(start_time)}s"
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp_display = f" at {minutes}:{seconds:02d}"
            if end_time is not None:
                end_minutes = int(end_time // 60)
                end_seconds = int(end_time % 60)
                timestamp_display += f"-{end_minutes}:{end_seconds:02d}"

        # Build citation with precise timing
        youtube_url = f"https://www.youtube.com/watch?v={video_id}{timestamp_url_param}"
        score = f"{entry['score']:.3f}"

        citation_lines = [
            f"📹 {title}",
            f"   Link: {youtube_url}",
            f"   Coach: {coach_name}",
            f"   Video ID: {video_id}",
        ]
        if timestamp_display:
            citation_lines.append(f"   Timestamp: {timestamp_display.strip()}")
        citation_lines.append(f"   Relevance: {score}")

        citations.append("\n".join(citation_lines))

    # Add verbatim snippets (short excerpts from top 3 distinct videos)
    full_response = str(response)
    if verbatim_snippets:
        full_response += (
            "\n\n"
            + "=" * 60
            + "\n📝 Verbatim excerpts (top 3 distinct videos):\n"
            + "=" * 60
            + "\n\n"
        )
        # Include numbered snippets from distinct videos
        for idx, s in enumerate(verbatim_snippets, 1):
            full_response += f"({idx}) {s}\n\n"

    # Add detailed citation information at the end
    if citations:
        full_response += (
            "\n\n"
            + "=" * 60
            + "\n🎯 **Video Sources & Citations**\n"
            + "=" * 60
            + "\n\n"
            + "\n\n".join(citations)
        )
        full_response += "\n\n💡 **Tip**: Click the video links above to watch the full explanations!"

    return full_response


def get_query_engine(index: VectorStoreIndex, coach: str = "all"):
    """Create a query engine from a pre-loaded index.
    
    Args:
        index: Pre-loaded VectorStoreIndex
        coach: Which coach to ask ("egs", "milo", or "all")
        
    Returns:
        Query engine ready for fast queries
    """
    # Create coach-specific filters
    coach_filters = None
    if coach == "egs":
        from llama_index.core.vector_stores import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )
        coach_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="file_path",
                    value="elitegolfschools",
                    operator=FilterOperator.TEXT_MATCH,
                )
            ]
        )
    elif coach == "milo":
        from llama_index.core.vector_stores import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )
        coach_filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="file_path",
                    value="milolinesgolf",
                    operator=FilterOperator.TEXT_MATCH,
                )
            ]
        )

    # Get system prompt
    system_prompt = get_coach_prompt(coach)

    return index.as_query_engine(
        system_prompt=system_prompt,
        similarity_top_k=8,
        response_mode="compact",
        filters=coach_filters,
        streaming=True,
    )


def ask_with_engine(question: str, query_engine, coach: str = "all") -> str:
    """Ask a question using a pre-created query engine (for session mode).
    
    Args:
        question: Golf instruction question
        query_engine: Pre-created query engine from get_query_engine()
        coach: Coach name for response formatting
        
    Returns:
        Formatted response with citations
    """
    response = query_engine.query(question)

    # Same response processing as ask() function
    from typing import Any, Dict
    by_video: Dict[str, Dict[str, Any]] = {}
    for node in response.source_nodes:
        video_id = node.metadata.get("video_id", "unknown")
        score = node.score or 0.0

        if video_id not in by_video or score > by_video[video_id]["score"]:
            by_video[video_id] = {
                "node": node,
                "score": score,
            }

    relevant_videos = [v for v in by_video.values() if v["score"] >= 0.75]
    if len(relevant_videos) < 2:
        relevant_videos = [v for v in by_video.values() if v["score"] >= 0.65]

    top_videos = sorted(relevant_videos, key=lambda d: d["score"], reverse=True)[:3]

    citations = []
    verbatim_snippets = []

    for entry in top_videos:
        node = entry["node"]
        video_id = node.metadata.get("video_id", "unknown")
        title = node.metadata.get("title", f"Video {video_id}")
        start_time = node.metadata.get("start_time")
        end_time = node.metadata.get("end_time")
        coach_meta = node.metadata.get("coach", "Unknown")

        if coach_meta == "EGS":
            coach_name = "Elite Golf Schools (EGS)"
        elif coach_meta == "Milo":
            coach_name = "Milo Lines Golf"
        else:
            coach_name = "Unknown Coach"

        content = node.text
        snippet = ""
        try:
            if "Transcript Content:" in content:
                snippet = content.split("Transcript Content:")[-1].strip()
            else:
                snippet = content.strip()

            snippet = snippet.replace("\n", " ")
            if len(snippet) > 350:
                cutoff = snippet.rfind(".", 0, 350)
                if cutoff and cutoff > 100:
                    snippet = snippet[: cutoff + 1]
                else:
                    snippet = snippet[:350] + "..."
        except Exception:
            snippet = ""

        if snippet:
            verbatim_snippets.append(snippet)

        timestamp_url_param = ""
        timestamp_display = ""
        if start_time is not None:
            timestamp_url_param = f"&t={int(start_time)}s"
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp_display = f" at {minutes}:{seconds:02d}"
            if end_time is not None:
                end_minutes = int(end_time // 60)
                end_seconds = int(end_time % 60)
                timestamp_display += f"-{end_minutes}:{end_seconds:02d}"

        youtube_url = f"https://www.youtube.com/watch?v={video_id}{timestamp_url_param}"
        score = f"{entry['score']:.3f}"

        citations.append(
            f"📹 **[{title}]({youtube_url})**{timestamp_display} - {coach_name} (Score: {score})"
        )

    full_response = str(response)

    if citations:
        full_response += (
            "\n\n---\n\n**Sources:**\n\n"
            + "\n\n".join(citations)
        )
        full_response += "\n\n💡 **Tip**: Click the video links above to watch the full explanations!"

    return full_response


# Convenience functions for each coach
def egs_coach(question: str) -> str:
    """Ask a question to the Elite Golf Schools (EGS) coach.

    Args:
        question: Golf instruction question

    Returns:
        str: EGS coach response with technical biomechanics focus

    Note:
        Convenience function that calls ask() with coach='egs'

    """
    return ask(question, coach="egs")


def milo_coach(question: str) -> str:
    """Ask a question to the Milo Lines Golf coach.

    Args:
        question: Golf instruction question

    Returns:
        str: Milo coach response with practical, feel-based focus

    Note:
        Convenience function that calls ask() with coach='milo'

    """
    return ask(question, coach="milo")


def golf_coach(question: str, coach: str = "all") -> str:
    """General golf coach function supporting multiple coaches.

    Args:
        question: Golf instruction question
        coach: Which coach to ask ('egs', 'milo', or 'all')

    Returns:
        str: Coach response with appropriate personality and expertise

    Note:
        General-purpose function that delegates to ask() with coach selection

    """
    return ask(question, coach)


def main() -> int:
    """Command-line interface for AI Golf Coaches."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask AI golf coaches questions about golf instruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ai_golf_coaches.rag "How do I fix my slice?"
  python -m ai_golf_coaches.rag "What's the best putting grip?" --coach milo
  python -m ai_golf_coaches.rag "Explain the X-Factor method" --coach egs
        """,
    )

    parser.add_argument("question", help="Golf instruction question to ask")

    parser.add_argument(
        "--coach",
        "-c",
        choices=["egs", "milo", "all"],
        default="all",
        help="Which coach to ask (default: all)",
    )

    parser.add_argument(
        "--data-path",
        "-d",
        default="data/raw",
        help="Path to transcript data directory (default: data/raw)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--llm-model",
        default="llama3.1",
        help="Ollama LLM model to use (default: llama3.1)",
    )

    parser.add_argument(
        "--embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model to use (default: nomic-embed-text)",
    )

    parser.add_argument(
        "--build", action="store_true", help="Build/persist the index and exit"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Setup models
    setup_models(args.llm_model)

    if args.build:
        create_index(coach=args.coach, data_path=args.data_path)
        logger.info("Index built and persisted. Exiting.")
        return 0

    # ask flow uses the persisted index
    logger.info(
        f"Asking Coach {args.coach.upper() if args.coach != 'all' else 'AI Golf Coaches'}"
    )
    answer = ask(args.question, coach=args.coach, data_path=args.data_path)
    logger.info(f"Answer: {answer}")
    return 0


if __name__ == "__main__":
    exit(main())
