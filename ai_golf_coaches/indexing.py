from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
from pydantic import BaseModel

from .config import AppSettings, load_channels_config, resolve_channel_key
from .constants import (
    DEFAULT_OPENAI_EMBED_MODEL,
    INDEX_DIR_NAME,
)
from .models import TranscriptChunk


class IndexedChunk(BaseModel):
    """A single chunk prepared for indexing.

    Attributes:
        text (str): Chunk text content.
        video_id (str): YouTube video ID the chunk belongs to.
        start (float): Start time in seconds.
        end (float): End time in seconds.
        title (str | None): Video title from catalog.
        is_livestream (bool): Whether the video is/was a livestream.
        channel_key (str): Canonical channel key.

    """

    text: str
    video_id: str
    start: float
    end: float
    title: Optional[str] = None
    is_livestream: bool = False
    channel_key: str


@dataclass
class IndexPaths:
    """Paths for a channel's index files."""

    base_dir: Path
    faiss_path: Path
    meta_path: Path


def _channel_paths(root: Path, channel_key: str) -> Tuple[Path, Path]:
    """Return paths to catalog.jsonl and transcripts/ for a channel.

    Args:
        root (Path): Repository root.
        channel_key (str): Canonical channel key.

    Returns:
        Tuple[Path, Path]: (catalog.jsonl path, transcripts/ directory path).

    """
    base = root / "data" / channel_key
    catalog = base / "catalog.jsonl"
    transcripts_dir = base / "transcripts"
    return catalog, transcripts_dir


def _index_paths(root: Path, channel_key: str) -> IndexPaths:
    """Return paths for a channel's index files.

    Args:
        root (Path): Repository root.
        channel_key (str): Canonical channel key.

    Returns:
        IndexPaths: Paths for the index files.

    """
    base = root / "data" / channel_key / INDEX_DIR_NAME
    return IndexPaths(base, base / "faiss.index", base / "meta.jsonl")


def _load_catalog_index(catalog_path: Path) -> Dict[str, dict]:
    """Load catalog entries keyed by video_id.

    Args:
        catalog_path (Path): Path to catalog.jsonl.

    Returns:
        dict[str, dict]: Mapping of video_id to parsed catalog entry dict.

    """
    out: Dict[str, dict] = {}
    if not catalog_path.exists():
        return out
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            vid = obj.get("video_id")
            if vid:
                out[str(vid)] = obj
    return out


def iter_indexed_chunks(root: Path, channel_key: str) -> Iterable[IndexedChunk]:
    """Yield indexed chunks for a channel by reading transcripts and catalog metadata.

    Args:
        root (Path): Repository root.
        channel_key (str): Canonical channel key.

    Yields:
        IndexedChunk: Prepared chunk with metadata.

    """
    catalog_path, transcripts_dir = _channel_paths(root, channel_key)
    catalog_map = _load_catalog_index(catalog_path)

    if not transcripts_dir.exists():
        return

    for fpath in sorted(transcripts_dir.glob("*.jsonl")):
        video_id = fpath.stem
        meta = catalog_map.get(video_id, {})
        title = meta.get("title")
        is_livestream = bool(meta.get("is_livestream", False))
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    chunk = TranscriptChunk(**obj)
                except Exception:
                    continue
                start = float(chunk.start)
                dur = float(chunk.duration) if chunk.duration is not None else 0.0
                end = start + dur if dur > 0 else start
                yield IndexedChunk(
                    text=chunk.chunk,
                    video_id=video_id,
                    start=start,
                    end=end,
                    title=title,
                    is_livestream=is_livestream,
                    channel_key=channel_key,
                )


def _openai_embed_batch(texts: List[str], model: str, api_key: str) -> np.ndarray:
    """Embed a batch of texts using OpenAI embeddings API.

    Args:
        texts (List[str]): Input texts.
        model (str): Embedding model name.
        api_key (str): OpenAI API key.

    Returns:
        np.ndarray: Array of shape (N, D) of embeddings.

    """
    # Lazy import to avoid mandatory dependency if using local models later
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenAI client not available; install 'openai' package"
        ) from e

    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts)
    vecs: List[List[float]] = [d.embedding for d in resp.data]
    arr = np.array(vecs, dtype=np.float32)
    # Normalize for cosine similarity with inner product index
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return arr


def build_faiss_index(
    channel: str,
    model: Optional[str] = None,
    batch_size: int = 256,
) -> Tuple[int, Path, Path]:
    """Build a FAISS index for a channel using OpenAI embeddings.

    Args:
        channel (str): Alias, handle, or canonical key for the channel.
        model (Optional[str]): Embedding model; defaults to text-embedding-3-large.
        batch_size (int): Number of chunks per embedding request.

    Returns:
        Tuple[int, Path, Path]: Number of indexed chunks, path to index, path to meta file.

    """
    root = Path(__file__).resolve().parent.parent
    settings = AppSettings()
    channels_yaml = Path(root / "config" / "channels.yaml")
    channels = load_channels_config(channels_yaml)
    key = resolve_channel_key(channel, channels)
    if not key:
        raise RuntimeError("Channel not found for provided alias/handle.")

    if not settings.openai or not settings.openai.api_key:
        raise RuntimeError("OPENAI__API_KEY not configured.")

    embed_model = model or settings.openai.embedding_model or DEFAULT_OPENAI_EMBED_MODEL

    idx_paths = _index_paths(root, key)
    idx_paths.base_dir.mkdir(parents=True, exist_ok=True)

    # Collect chunks and stream through embedding in batches
    chunks_iter = iter_indexed_chunks(root, key)
    texts: List[str] = []
    metas: List[IndexedChunk] = []

    index: Optional[faiss.Index] = None
    total = 0

    with idx_paths.meta_path.open("w", encoding="utf-8") as meta_f:

        def flush_batch() -> None:
            nonlocal index, total
            if not texts:
                return
            vecs = _openai_embed_batch(texts, embed_model, settings.openai.api_key)
            if index is None:
                dim = vecs.shape[1]
                index = faiss.IndexFlatIP(dim)
            index.add(vecs)
            # Write metas aligned to vectors
            for m in metas:
                meta_f.write(json.dumps(m.model_dump(), ensure_ascii=False) + "\n")
            total += len(texts)
            texts.clear()
            metas.clear()

        for ch in chunks_iter:
            texts.append(ch.text)
            metas.append(ch)
            if len(texts) >= batch_size:
                flush_batch()
        flush_batch()

    if index is None:
        raise RuntimeError("No transcript chunks found to index.")

    faiss.write_index(index, str(idx_paths.faiss_path))
    return total, idx_paths.faiss_path, idx_paths.meta_path


def query_index(
    channel: str, question: str, top_k: int = 5, model: Optional[str] = None
) -> List[Tuple[IndexedChunk, float]]:
    """Query a channel index with a question and return top chunks.

    Args:
        channel (str): Alias, handle, or canonical key.
        question (str): Natural language question.
        top_k (int): Number of results to return.
        model (Optional[str]): Embedding model; defaults to text-embedding-3-large.

    Returns:
        List[Tuple[IndexedChunk, float]]: List of (chunk, score) pairs.

    """
    root = Path(__file__).resolve().parent.parent
    settings = AppSettings()
    channels_yaml = Path(root / "config" / "channels.yaml")
    channels = load_channels_config(channels_yaml)
    key = resolve_channel_key(channel, channels)
    if not key:
        raise RuntimeError("Channel not found for provided alias/handle.")

    if not settings.openai or not settings.openai.api_key:
        raise RuntimeError("OPENAI__API_KEY not configured.")

    embed_model = model or settings.openai.embedding_model or DEFAULT_OPENAI_EMBED_MODEL
    idx_paths = _index_paths(root, key)

    if not idx_paths.faiss_path.exists() or not idx_paths.meta_path.exists():
        raise RuntimeError("Index not built for channel; run build-index first.")

    index = faiss.read_index(str(idx_paths.faiss_path))
    qvec = _openai_embed_batch([question], embed_model, settings.openai.api_key)
    D, I = index.search(qvec, top_k)  # noqa: E741, N806
    # Load meta lines and map by line number
    metas: List[IndexedChunk] = []
    with idx_paths.meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                metas.append(IndexedChunk(**obj))
            except Exception:
                metas.append(
                    IndexedChunk(text="", video_id="", start=0, end=0, channel_key=key)
                )

    results: List[Tuple[IndexedChunk, float]] = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(metas):
            continue
        results.append((metas[idx], float(score)))
    return results
