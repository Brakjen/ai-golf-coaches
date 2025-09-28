"""Lightweight RAG system for AI Golf Coaches using automatic methods."""

from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from ai_golf_coaches.config import get_settings

logger = logging.getLogger(__name__)


WINDOW_SIZE_SEC = 150
WINDOW_OVERLAP_SEC = 20
PERSIST_DIR = Path("data/index/youtube")
TEST_PERSIST_DIR = Path("data/index/youtube_test")


def setup_models(llm_model: str = "gpt-4o-mini") -> None:
    """Configure LlamaIndex models: LLM via OpenAI, embeddings via HF.

    Args:
        llm_model: OpenAI LLM model name (default: "gpt-4o-mini").

    Returns:
        None

    """
    config = get_settings()
    api_key = config.openai.api_key.get_secret_value()
    Settings.llm = OpenAI(
        model=llm_model,
        temperature=0.2,
        api_key=api_key,
        max_tokens=config.openai.max_tokens,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base", trust_remote_code=True
    )

    logger.info(f"Configured OpenAI model: {config.openai.model}")
    logger.info("Configured embedding model: intfloat/multilingual-e5-base")


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

            # Extract transcript text
            transcript = data.get("transcript", {})
            transcript_text = transcript.get("text", "")

            if not transcript_text:
                logger.warning(f"No transcript text found in {json_file.name}")
                continue

            # Create enhanced document with title and context for better matching
            enhanced_content = f"""
Video Title: {title}
Channel: {channel}
Video ID: {video_id}
Topic Focus: {title}

Transcript Content:
{transcript_text}
            """.strip()

            # Create document with enhanced metadata
            doc = Document(
                text=enhanced_content,
                metadata={
                    "video_id": video_id,
                    "title": title,
                    "channel": channel,
                    "file_path": str(json_file),
                    "file_name": json_file.name,
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
    """Create a small test index with limited videos for quick testing.

    Args:
        max_videos: Maximum number of videos to include (default: 20)

    Returns:
        VectorStoreIndex: The test index

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

                # Extract transcript text
                transcript = data.get("transcript", {})
                transcript_text = transcript.get("text", "")

                if not transcript_text:
                    continue

                # Create enhanced document with title and context for better matching
                enhanced_content = f"""
Video Title: {title}
Channel: {channel}
Video ID: {video_id}
Topic Focus: {title}

Transcript Content:
{transcript_text}
                """.strip()

                # Create document with enhanced metadata
                doc = Document(
                    text=enhanced_content,
                    metadata={
                        "video_id": video_id,
                        "title": title,
                        "channel": channel,
                        "file_path": str(json_file),
                        "file_name": json_file.name,
                        "topic": title.lower(),
                        "coach": "EGS"
                        if "elitegolfschools" in str(json_file)
                        else "Milo",
                    },
                )
                documents.append(doc)
                video_count += 1

                # Show progress
                print(f"Added: {title[:60]}... ({coach_dir})")

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

    coach_prompts = {
        "egs": (
            "You are an Elite Golf Schools (EGS) instructor passionate about teaching the X-Factor method and biomechanically sound golf swings. "
            "Channel the enthusiastic, detailed teaching style of EGS coaches like TJ and Riley. "
            "Use ONLY the provided transcript context, but be comprehensive and thorough in your explanations. "
            "Your response style should be:\n"
            "- Detailed and educational, explaining the 'why' behind movements\n"
            "- Enthusiastic about proper biomechanics and body movement\n"
            "- Reference specific body parts, angles, and technical concepts from X-Factor method\n"
            "- Give step-by-step explanations and drills when appropriate\n"
            "- Use encouraging language like 'That's exactly right!' or 'Here's what we want to see'\n"
            "- Explain how movements connect to power, consistency, and injury prevention\n"
            "- Provide multiple examples or variations when the context allows\n"
            "CRITICAL: Include inline citations throughout your response. "
            "Format inline citations as: [EGS Video: VIDEO_ID] or [EGS Video: VIDEO_ID at MM:SS] "
            "Example: 'The X-Factor method emphasizes proper hip rotation [EGS Video: ABC123] to generate power...' "
            "Cite EVERY major point or technique you mention with the specific video source. "
            "If you don't know something from the context, say what you do know and suggest watching the referenced videos for full details. "
            "Be encouraging and educational like a golf coach."
        ),
        "milo": (
            "You are Milo Lines, the golf instructor known for your clear, practical teaching approach and excellent communication skills. "
            "Channel Milo's distinctive coaching personality and teaching methodology. "
            "Use ONLY the provided transcript context, but give detailed, practical explanations in Milo's style. "
            "Your response style should be:\n"
            "- Clear, practical explanations that golfers can immediately apply\n"
            "- Break down complex concepts into simple, understandable steps\n"
            "- Use Milo's characteristic phrases and teaching approach\n"
            "- Focus on feels, images, and practical drills that work on the course\n"
            "- Explain both what to do AND what NOT to do\n"
            "- Give multiple options or progressions when the context supports it\n"
            "- Use encouraging, conversational tone like you're coaching face-to-face\n"
            "- Connect technical concepts to real-world application and course management\n"
            "CRITICAL: Include inline citations throughout your response. "
            "Format inline citations as: [Milo Video: VIDEO_ID] or [Milo Video: VIDEO_ID at MM:SS] "
            "Example: 'For better chipping, focus on body rotation [Milo Video: XYZ789] rather than just using your hands...' "
            "Cite EVERY major point or technique you mention with the specific video source. "
            "If the transcript context doesn't provide a complete answer, explain what you can and encourage watching the full videos for comprehensive instruction. "
            "Be thorough and helpful - give golfers the detailed guidance they're seeking."
        ),
        "all": (
            "You are an AI golf coach with access to instruction from multiple expert instructors including Elite Golf Schools and Milo Lines Golf. "
            "Use the provided transcript context to give comprehensive, detailed answers that capture each coach's unique teaching style. "
            "When referencing different instructors, maintain their distinct voices and approaches. "
            "CRITICAL INSTRUCTION ATTRIBUTION RULES:\n"
            "- ALWAYS identify which coach is providing each piece of advice\n"
            "- Use phrases like: 'Elite Golf Schools teaches...', 'Milo Lines explains...', 'According to EGS...', 'Milo's approach is...'\n"
            "- When you have advice from multiple coaches, organize by coach or clearly separate their approaches\n"
            "- If transcript content mentions 'elitegolfschools' path, attribute to Elite Golf Schools/EGS\n"
            "- If transcript content mentions 'milolinesgolf' path, attribute to Milo Lines Golf\n"
            "- Never present advice without coach attribution - golfers want to know the source\n"
            "Your response should be:\n"
            "- Comprehensive and educational, drawing from all relevant context\n"
            "- Clearly structured by coach when multiple sources are used\n"
            "- Compare different perspectives or methods when multiple coaches address the same topic\n"
            "- Provide step-by-step explanations, drills, and practical application with coach attribution\n"
            "- Explain the biomechanics and reasoning behind recommendations\n"
            "- Give golfers multiple options or progressions to try, attributing each to its coach\n"
            "- Use encouraging, professional coaching language\n"
            "CRITICAL: Include inline citations throughout your response for EVERY major point. "
            "Format inline citations as: [EGS Video: VIDEO_ID], [Milo Video: VIDEO_ID], or [EGS Video: VIDEO_ID at MM:SS] "
            "Example: 'Elite Golf Schools teaches proper hip turn [EGS Video: ABC123], while Milo focuses on feel-based rotation [Milo Video: XYZ789].' "
            "Attribute each piece of advice immediately when you mention it - don't save citations for the end. "
            "If you need more context for a complete answer, explain what you do know and guide them to the most relevant videos. "
            "Be thorough - golfers want comprehensive instruction with clear source attribution and precise timestamps, not surface-level tips."
        ),
    }
    system_prompt = coach_prompts.get(coach, coach_prompts["all"])

    query_engine = index.as_query_engine(
        system_prompt=system_prompt,
        similarity_top_k=8,  # More context for richer responses
        response_mode="compact",  # Better for inline citations
        filters=coach_filters,  # Apply coach-specific filtering
    )
    response = query_engine.query(question)

    # Extract detailed citation information from source nodes
    citations = []
    for i, node in enumerate(response.source_nodes, 1):
        # Extract video ID from filename
        file_name = node.metadata.get("file_name", "")
        video_id = (
            file_name.replace(".json", "") if file_name.endswith(".json") else "Unknown"
        )

        # Identify coach from file path
        file_path = node.metadata.get("file_path", "")
        coach_name = "Unknown Coach"
        coach_short = "unknown"
        if "elitegolfschools" in file_path:
            coach_name = "Elite Golf Schools (EGS)"
            coach_short = "EGS"
        elif "milolinesgolf" in file_path:
            coach_name = "Milo Lines Golf"
            coach_short = "Milo"

        # Try to extract video title and timestamp from content if it contains JSON structure
        title = "Video"
        timestamp_seconds = None
        content = node.text

        if '"title":' in content:
            try:
                import re

                # Extract title from JSON content
                lines = content.split("\n")
                for line in lines:
                    if '"title":' in line:
                        title_part = line.split('"title":')[1].strip(' ",')
                        if title_part:
                            title = (
                                title_part.split('"')[1]
                                if '"' in title_part
                                else title_part[:50]
                            )
                            break

                # Extract timestamp from transcript lines - look for "start": value
                start_match = re.search(r'"start":\s*([0-9.]+)', content)
                if start_match:
                    timestamp_seconds = float(start_match.group(1))

            except:
                pass

        # Format timestamp for YouTube URL and display
        timestamp_url_param = ""
        timestamp_display = ""
        if timestamp_seconds is not None:
            # Convert seconds to YouTube URL format (&t=XXXs) and display format (MM:SS)
            timestamp_url_param = f"&t={int(timestamp_seconds)}s"
            minutes = int(timestamp_seconds // 60)
            seconds = int(timestamp_seconds % 60)
            timestamp_display = f" at {minutes}:{seconds:02d}"

        # Get relevance score
        score = f"{node.score:.3f}"

        # Create citation with coach attribution and timestamp
        youtube_url = f"https://www.youtube.com/watch?v={video_id}{timestamp_url_param}"

        # Create display with timestamp if available
        title_display = title[:80] if title != "Video" else f"Video {video_id}"

        citation = f"📹 **{coach_short}**: [{title_display}]({youtube_url})\n"
        citation += f"   Coach: {coach_name}\n"
        citation += f"   Video ID: {video_id}\n"
        if timestamp_display:
            citation += f"   Timestamp: {timestamp_display.strip()}\n"
        citation += f"   Relevance: {score}\n"

        citations.append(citation)

    # Add detailed citation information at the end
    full_response = str(response)
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


# Convenience functions for each coach
def egs_coach(question: str) -> str:
    """Ask Elite Golf Schools (X-Factor method) a question."""
    return ask(question, coach="egs")


def milo_coach(question: str) -> str:
    """Ask Milo Lines Golf a question."""
    return ask(question, coach="milo")


def golf_coach(question: str, coach: str = "all") -> str:
    """Ask your AI golf coach a question.

    Args:
        question: Golf instruction question.
        coach: Which coach to ask ("egs", "milo", or "all").

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
