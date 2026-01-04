from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import load_channels_config, resolve_channel_key


def _repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Path: Absolute path to the repo root.

    """
    return Path(__file__).parent.parent.resolve()


def get_git_root() -> Path:
    """Get the root directory of the git repository.

    Returns:
        Path: The root directory of the git repository.

    """
    return _repo_root()


def find_transcript_file(video_id: str) -> Optional[Path]:
    """Find the transcript JSONL file for a given video ID across data/*/transcripts.

    Args:
        video_id (str): YouTube video ID.

    Returns:
        Path | None: Path to the transcript file if found; otherwise None.

    """
    root = _repo_root()
    data_dir = root / "data"
    if not data_dir.exists():
        return None
    # Search one level of channel directories
    for channel_dir in data_dir.iterdir():
        tdir = channel_dir / "transcripts"
        if tdir.is_dir():
            candidate = tdir / f"{video_id}.jsonl"
            if candidate.exists():
                return candidate
    return None


def read_transcript_text(transcript_path: Path) -> str:
    """Read a transcript JSONL file and join all chunk texts.

    Args:
        transcript_path (Path): Path to the transcript JSONL file.

    Returns:
        str: Concatenated transcript text.

    """
    parts = []
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                txt = str(obj.get("chunk", "")).strip()
                if txt:
                    parts.append(txt)
            except Exception:
                continue
    return " ".join(parts)


def count_tokens_for_text(text: str, encoding: Optional[str] = None) -> int:
    """Count tokens for text using tiktoken encodings.

    Attempts to use the GPT-4o encoding ("o200k_base"). Falls back to
    "cl100k_base" if unavailable.

    Args:
        text (str): Input text to tokenize.
        encoding (str | None): Optional explicit encoding name.

    Returns:
        int: Number of tokens.

    Raises:
        RuntimeError: If tiktoken is not installed.

    """
    try:
        import tiktoken  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "tiktoken is required for token counting. Install it via poetry."
        ) from e

    enc_name = encoding
    if enc_name is None:
        # Prefer GPT-4o encoding if available
        try:
            tiktoken.get_encoding("o200k_base")
            enc_name = "o200k_base"
        except Exception:
            enc_name = "cl100k_base"

    enc = tiktoken.get_encoding(enc_name)
    return len(enc.encode(text))


def transcript_token_count(
    video_id: str, encoding: Optional[str] = None
) -> Tuple[int, Optional[Path]]:
    """Compute token count for a transcript by video ID.

    Searches under data/*/transcripts/<video_id>.jsonl, concatenates all chunk
    texts, and tokenizes with tiktoken.

    Args:
        video_id (str): YouTube video ID.
        encoding (str | None): Optional encoding name (e.g., 'o200k_base').

    Returns:
        tuple[int, Path | None]: (token_count, path) where path is the located
        transcript file or None if not found.

    """
    fpath = find_transcript_file(video_id)
    if not fpath:
        return 0, None
    text = read_transcript_text(fpath)
    tokens = count_tokens_for_text(text, encoding=encoding)
    return tokens, fpath


def prepare_constant_context_text(
    alias_or_key: str,
    root: Path | str | None = None,
    channels: Dict[str, dict] | None = None,
    channels_config_path: Path | str = "config/channels.yaml",
    ensure_all: bool = True,
) -> Tuple[str, List[str]]:
    """Build a master constant-context string from channel-defined videos.

    This generates the constant context on-the-fly. It reads the channel's
    `constant_context_videos` from the channels configuration, locates each
    transcript JSONL under data/<channel>/transcripts, joins all transcript
    chunks with spaces, collapses whitespace so each transcript becomes a single
    line, and concatenates the per-video lines with newlines.

    Args:
        alias_or_key (str): Channel alias, handle, or canonical key (e.g., 'egs').
        root (Path | str | None): Optional workspace root. Defaults to the
            current working directory.
        channels (Dict[str, dict] | None): Optional preloaded channels config
            mapping to avoid re-reading YAML from disk.
        channels_config_path (Path | str): Path to the channels YAML when
            `channels` is not provided.
        ensure_all (bool): When True, raises an error if any listed transcript
            is missing. When False, returns a best-effort master string and the
            list of missing IDs.

    Returns:
        Tuple[str, List[str]]: A 2-tuple where the first element is the
        newline-joined master string (one line per transcript), and the second
        element is the list of video IDs whose transcripts were not found.

    Raises:
        KeyError: If the channel key cannot be resolved in the configuration.
        FileNotFoundError: If `ensure_all` is True and any transcript is missing.

    """
    base = Path(root) if root is not None else Path.cwd()
    cfg = channels or load_channels_config(base / channels_config_path)
    key = resolve_channel_key(alias_or_key, cfg) or alias_or_key.strip().lower()
    entry = cfg.get(key)
    if not entry:
        raise KeyError(f"Channel configuration not found for: {alias_or_key}")

    video_ids: List[str] = list(entry.get("constant_context_videos") or [])
    lines: List[str] = []
    missing: List[str] = []

    for vid in video_ids:
        fpath = find_transcript_file(vid)
        if not fpath:
            missing.append(vid)
            continue
        text = read_transcript_text(fpath)
        single_line = " ".join(text.split()).strip()
        if single_line:
            lines.append(single_line)

    if ensure_all and missing:
        raise FileNotFoundError(f"Missing transcripts for: {', '.join(missing)}")

    master = "\n".join(lines)
    return master, missing
