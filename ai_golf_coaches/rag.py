"""Lightweight RAG system for AI Golf Coaches using automatic methods."""

from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)


WINDOW_SIZE_SEC = 150
WINDOW_OVERLAP_SEC = 20
PERSIST_DIR = Path("data/index/youtube")


def setup_models(llm_model: str = "llama3.2:1b") -> None:
    """Configure LlamaIndex models: LLM via Ollama, embeddings via HF.

    Args:
        llm_model: Ollama LLM model name (default: "llama3.2:1b").

    Returns:
        None

    """
    Settings.llm = Ollama(model=llm_model, request_timeout=120.0, temperature=0.2)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base", embed_batch_size=64
    )


def create_index(
    coach: str = "all", data_path: str | Path = "data/raw"
) -> VectorStoreIndex:
    setup_models()

    if coach == "all":
        input_path = str(data_path)
    else:
        coach_mapping = {"egs": "elitegolfschools", "milo": "milolinesgolf"}
        coach_dir = coach_mapping.get(coach, coach)
        input_path = str(Path(data_path) / coach_dir)

    reader = SimpleDirectoryReader(
        input_dir=input_path, required_exts=[".json"], recursive=True
    )
    documents = reader.load_data()
    logger.info(f"Auto-loaded {len(documents)} documents for coach: {coach}")

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    # 🔑 persist once
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    logger.info(f"Persisted index to {PERSIST_DIR}")
    return index


def load_index() -> VectorStoreIndex:
    """Load a previously built index from disk."""
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(f"No index at {PERSIST_DIR}. Run with --build first.")
    setup_models()
    storage = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
    return load_index_from_storage(storage)  # type: ignore[return-value]


def ask(question: str, coach: str = "all", data_path: str | Path = "data/raw") -> str:
    """Ask a question to the specified AI golf coach.

    Args:
        question: Golf instruction question.
        coach: Which coach to ask ("egs", "milo", or "all").
        data_path: Path to transcript data directory (default: "data/raw").

    Returns:
        str: The answer from the AI golf coach.

    """
    index = load_index()

    coach_prompts = {
        "egs": (
            "You are answering as Elite Golf Schools (EGS). "
            "Answer ONLY from the provided context. If unknown, say you don't know. "
            "Cite sources."
        ),
        "milo": (
            "You are answering as Milo Lines Golf. "
            "Answer ONLY from the provided context. If unknown, say you don't know. "
            "Cite sources."
        ),
        "all": (
            "You are an AI golf coach with multiple instructors. "
            "Answer ONLY from the provided context. If unknown, say you don't know. "
            "Cite the instructor/source."
        ),
    }
    system_prompt = coach_prompts.get(coach, coach_prompts["all"])

    query_engine = index.as_query_engine(
        system_prompt=system_prompt,
        similarity_top_k=6,
        response_mode="compact",
    )
    response = query_engine.query(question)
    return str(response)


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
        print("✅ Index built and persisted.")
        return 0

    # ask flow uses the persisted index
    print(
        f"\n🏌️  Asking {args.coach.upper() if args.coach != 'all' else 'AI Golf Coaches'}: {args.question}"
    )
    print("=" * 60)
    answer = ask(args.question, coach=args.coach, data_path=args.data_path)
    print(answer)


if __name__ == "__main__":
    exit(main())
