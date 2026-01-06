from __future__ import annotations

import contextlib
import json
import os
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


class HostedLongformLine(BaseModel):
    """A prepared longform line for hosted vector store ingestion.

    This model represents a *single* searchable line of text that includes a
    compact metadata prefix. We intentionally include metadata in-band so that
    later retrieval can show sources without relying on vector-store metadata
    filters.

    Attributes:
        category (str): One of the app categories (e.g., "full_swing").
        text (str): The final line content that will be embedded and searched.

    """

    category: str
    text: str


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


def _load_category_video_ids(channel_entry: dict) -> Dict[str, List[str]]:
    """Load per-category video id lists from a channel entry.

    Args:
        channel_entry (dict): A single channel entry from channels.yaml.

    Returns:
        Dict[str, List[str]]: Mapping of category key to a list of video IDs.

    """
    raw = channel_entry.get("constant_context_videos") or {}
    if isinstance(raw, dict):
        out: Dict[str, List[str]] = {}
        for cat, vids in raw.items():
            if not vids:
                continue
            out[str(cat)] = [str(v) for v in vids]
        return out
    return {}


def _format_hosted_line(chunk: IndexedChunk, category: str) -> HostedLongformLine:
    """Format an indexed chunk into a single searchable line.

    Args:
        chunk (IndexedChunk): Parsed transcript chunk with metadata.
        category (str): Category label for this chunk.

    Returns:
        HostedLongformLine: Prepared line with in-band metadata.

    """
    title = chunk.title or ""
    title_clean = " ".join(str(title).split())
    text_clean = " ".join(str(chunk.text).split())
    prefix = (
        f"[channel={chunk.channel_key} category={category} video_id={chunk.video_id} "
        f"start={chunk.start:.2f} end={chunk.end:.2f} livestream={str(chunk.is_livestream).lower()} "
        f'title="{title_clean}"]'
    )
    return HostedLongformLine(category=category, text=f"{prefix} {text_clean}".strip())


def _write_category_text_files(
    *,
    root: Path,
    channel_key: str,
    category: str,
    video_ids: List[str],
    include_livestreams: bool,
    max_part_bytes: int,
) -> Tuple[List[Path], List[str]]:
    """Write one or more plain-text files for a category, splitting by size.

    Each output file contains one prepared searchable line per transcript chunk.
    Files are split at approximately `max_part_bytes` to avoid OpenAI upload
    limits.

    Args:
        root (Path): Repository root.
        channel_key (str): Canonical channel key.
        category (str): Category label.
        video_ids (List[str]): Video IDs to include.
        include_livestreams (bool): Whether to include livestream chunks.
        max_part_bytes (int): Max size per part file in bytes.

    Returns:
        Tuple[List[Path], List[str]]: (paths_to_written_parts, missing_video_ids)

    """
    out_dir = (
        root / ".cache" / "vector_store_uploads" / channel_key / "longform" / category
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_path, transcripts_dir = _channel_paths(root, channel_key)
    catalog_map = _load_catalog_index(catalog_path)

    written: List[Path] = []
    missing: List[str] = []

    part_idx = 1
    current_path = out_dir / f"{channel_key}_longform_{category}_part{part_idx}.txt"
    current_size = 0
    f = current_path.open("w", encoding="utf-8")

    def rotate_file() -> None:
        nonlocal part_idx, current_path, current_size, f
        f.close()
        written.append(current_path)
        part_idx += 1
        current_path = out_dir / f"{channel_key}_longform_{category}_part{part_idx}.txt"
        current_size = 0
        f = current_path.open("w", encoding="utf-8")

    try:
        for video_id in video_ids:
            tpath = transcripts_dir / f"{video_id}.jsonl"
            if not tpath.exists():
                missing.append(video_id)
                continue

            meta = catalog_map.get(video_id, {})
            title = meta.get("title")
            is_livestream = bool(meta.get("is_livestream", False))
            if is_livestream and not include_livestreams:
                continue

            with tpath.open("r", encoding="utf-8") as tf:
                for line in tf:
                    try:
                        obj = json.loads(line)
                        chunk = TranscriptChunk(**obj)
                    except Exception:
                        continue

                    start = float(chunk.start)
                    dur = float(chunk.duration) if chunk.duration is not None else 0.0
                    end = start + dur if dur > 0 else start
                    indexed = IndexedChunk(
                        text=chunk.chunk,
                        video_id=video_id,
                        start=start,
                        end=end,
                        title=title,
                        is_livestream=is_livestream,
                        channel_key=channel_key,
                    )
                    hosted = _format_hosted_line(indexed, category)
                    out_line = hosted.text + "\n"
                    b = out_line.encode("utf-8")
                    if current_size + len(b) > max_part_bytes and current_size > 0:
                        rotate_file()
                    f.write(out_line)
                    current_size += len(b)
    finally:
        f.close()

    # Only add the last file if it has content
    if current_path.exists() and current_path.stat().st_size > 0:
        written.append(current_path)
    else:
        with contextlib.suppress(Exception):
            current_path.unlink(missing_ok=True)  # type: ignore[arg-type]

    return written, missing


def _write_all_category_text_files(
    *,
    root: Path,
    channel_key: str,
    include_livestreams: bool,
    max_part_bytes: int,
) -> Tuple[List[Path], List[str]]:
    """Write text files for the 'all' catch-all category (all transcripts).

    Args:
        root (Path): Repository root.
        channel_key (str): Canonical channel key.
        include_livestreams (bool): Whether to include livestream chunks.
        max_part_bytes (int): Max size per part file in bytes.

    Returns:
        Tuple[List[Path], List[str]]: (paths_to_written_parts, missing_video_ids)

    """
    out_dir = (
        root / ".cache" / "vector_store_uploads" / channel_key / "longform" / "all"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog_path, transcripts_dir = _channel_paths(root, channel_key)
    catalog_map = _load_catalog_index(catalog_path)

    written: List[Path] = []
    missing: List[str] = []

    part_idx = 1
    current_path = out_dir / f"{channel_key}_longform_all_part{part_idx}.txt"
    current_size = 0
    f = current_path.open("w", encoding="utf-8")

    def rotate_file() -> None:
        nonlocal part_idx, current_path, current_size, f
        f.close()
        written.append(current_path)
        part_idx += 1
        current_path = out_dir / f"{channel_key}_longform_all_part{part_idx}.txt"
        current_size = 0
        f = current_path.open("w", encoding="utf-8")

    try:
        # Iterate through all transcript files
        if not transcripts_dir.exists():
            return [], []

        for tpath in sorted(transcripts_dir.glob("*.jsonl")):
            video_id = tpath.stem
            meta = catalog_map.get(video_id, {})
            title = meta.get("title")
            is_livestream = bool(meta.get("is_livestream", False))

            if is_livestream and not include_livestreams:
                continue

            with tpath.open("r", encoding="utf-8") as tf:
                for line in tf:
                    try:
                        obj = json.loads(line)
                        chunk = TranscriptChunk(**obj)
                    except Exception:
                        continue

                    start = float(chunk.start)
                    dur = float(chunk.duration) if chunk.duration is not None else 0.0
                    end = start + dur if dur > 0 else start
                    indexed = IndexedChunk(
                        text=chunk.chunk,
                        video_id=video_id,
                        start=start,
                        end=end,
                        title=title,
                        is_livestream=is_livestream,
                        channel_key=channel_key,
                    )
                    # For "all" store, use "all" as category label
                    hosted = _format_hosted_line(indexed, "all")
                    out_line = hosted.text + "\n"
                    b = out_line.encode("utf-8")
                    if current_size + len(b) > max_part_bytes and current_size > 0:
                        rotate_file()
                    f.write(out_line)
                    current_size += len(b)
    finally:
        f.close()

    # Only add the last file if it has content
    if current_path.exists() and current_path.stat().st_size > 0:
        written.append(current_path)
    else:
        with contextlib.suppress(Exception):
            current_path.unlink(missing_ok=True)  # type: ignore[arg-type]

    return written, missing


def build_hosted_vector_stores_longform(
    channel: str,
    categories: Optional[List[str]] = None,
    include_livestreams: bool = False,
    max_part_mb: int = 100,
    build_all_store: bool = True,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Build OpenAI-hosted vector stores for longform transcript chunks.

    Creates one vector store per category for the selected channel and uploads
    one or more text files containing chunk lines for that category.

    Optionally creates an 'all' catch-all store containing all transcript chunks
    regardless of category, intended for RAG retrieval fallback (NOT for static
    context, which would exceed 128k token limits).

    This function uses the app-owned OpenAI key from environment (AppSettings)
    and is intended to be run as an offline ingestion step.

    Args:
        channel (str): Alias, handle, or canonical channel key.
        categories (List[str] | None): If provided, only ingest these categories.
        include_livestreams (bool): Whether to include livestream chunks.
        max_part_mb (int): Max size per uploaded part file in MB.
        build_all_store (bool): Whether to build an 'all' catch-all store for retrieval fallback.

    Returns:
        Tuple[Dict[str, str], Dict[str, List[str]]]:
            (vector_store_ids_by_category, missing_video_ids_by_category)

    Raises:
        RuntimeError: If OpenAI settings or API key missing.
        KeyError: If channel cannot be resolved.

    """
    root = Path(__file__).resolve().parent.parent
    channels_yaml = Path(root / "config" / "channels.yaml")
    channels_cfg = load_channels_config(channels_yaml)
    channel_key = resolve_channel_key(channel, channels_cfg)
    if not channel_key:
        raise KeyError(f"Channel not found for alias/handle: {channel}")

    api_key = os.getenv("OPENAI__API_KEY_STORAGE_INDEX")
    if not api_key:
        raise RuntimeError("OPENAI__API_KEY_STORAGE_INDEX not configured.")

    entry = channels_cfg.get(channel_key) or {}
    cat_map = _load_category_video_ids(entry)
    if not cat_map:
        raise RuntimeError(
            f"No constant_context_videos categories configured for channel '{channel_key}'."
        )

    selected_categories = categories or sorted(cat_map.keys())
    selected_categories = [
        c.strip().lower().replace(" ", "_") for c in selected_categories
    ]

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenAI client not available; install 'openai'.") from e

    client = OpenAI(api_key=api_key)
    vs_ids: Dict[str, str] = {}
    missing_by_cat: Dict[str, List[str]] = {}

    max_part_bytes = int(max_part_mb * 1024 * 1024)

    for cat in selected_categories:
        vids = cat_map.get(cat) or []
        if not vids:
            continue

        # Write text files to upload
        part_paths, missing = _write_category_text_files(
            root=root,
            channel_key=channel_key,
            category=cat,
            video_ids=vids,
            include_livestreams=include_livestreams,
            max_part_bytes=max_part_bytes,
        )
        missing_by_cat[cat] = missing

        if not part_paths:
            continue

        # Create vector store
        store_name = f"aigolfcoaches-{channel_key}-longform-{cat}"
        vector_store = client.beta.vector_stores.create(name=store_name)
        vs_id = str(vector_store.id)
        vs_ids[cat] = vs_id

        # Upload parts
        for p in part_paths:
            with p.open("rb") as fh:
                client.beta.vector_stores.files.upload_and_poll(
                    vector_store_id=vs_id,
                    file=fh,
                )

    # Build "all" catch-all store if requested (for RAG fallback, NOT static context)
    if build_all_store:
        part_paths_all, missing_all = _write_all_category_text_files(
            root=root,
            channel_key=channel_key,
            include_livestreams=include_livestreams,
            max_part_bytes=max_part_bytes,
        )
        missing_by_cat["all"] = missing_all

        if part_paths_all:
            store_name_all = f"aigolfcoaches-{channel_key}-longform-all"
            vector_store_all = client.beta.vector_stores.create(name=store_name_all)
            vs_id_all = str(vector_store_all.id)
            vs_ids["all"] = vs_id_all

            for p in part_paths_all:
                with p.open("rb") as fh:
                    client.beta.vector_stores.files.upload_and_poll(
                        vector_store_id=vs_id_all,
                        file=fh,
                    )

    return vs_ids, missing_by_cat


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
