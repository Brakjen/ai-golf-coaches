from __future__ import annotations

import contextlib
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set

from pydantic import BaseModel, Field, ValidationError

from .config import (
    AppSettings,
    load_channel_catalog,
    load_channels_config,
    resolve_channel_key,
)
from .constants import (
    QA_DEFAULT_MAX_OUTPUT_TOKENS,
    QA_DEFAULT_MAX_WINDOW_CHARS,
    QA_DEFAULT_OVERLAP_SECONDS,
    QA_DEFAULT_WINDOW_SECONDS,
    QA_DEFAULT_WORKERS,
)
from .models import QARecord, TranscriptChunk

PROMPT_VERSION = "v1"


class QAPair(BaseModel):
    """Structured QA pair extracted from a transcript segment.

    Attributes:
        question (str): Viewer question text.
        answer (str): Coach answer text.
        question_start (float): Start time for the question (seconds).
        question_end (float | None): End time for the question (seconds), if known.
        answer_start (float): Start time for the answer (seconds).
        answer_end (float | None): End time for the answer (seconds), if known.

    """

    question: str
    answer: str
    question_start: float
    question_end: Optional[float] = None
    answer_start: float
    answer_end: Optional[float] = None


class QABatch(BaseModel):
    """Container for QA pairs returned by the extractor model.

    Attributes:
        pairs (list[QAPair]): Extracted QA pairs.

    """

    pairs: List[QAPair] = Field(default_factory=list)


@dataclass(frozen=True)
class TranscriptWindow:
    """A transcript window with a time range and formatted text.

    Attributes:
        start (float): Window start time in seconds.
        end (float): Window end time in seconds.
        text (str): Timestamped transcript text.

    """

    start: float
    end: float
    text: str


def _normalize_text(text: str) -> str:
    """Normalize text by trimming and collapsing whitespace.

    Args:
        text (str): Input text.

    Returns:
        str: Normalized text.

    """
    return re.sub(r"\s+", " ", text or "").strip()


def _chunk_end(chunks: Sequence[TranscriptChunk], idx: int) -> float:
    """Infer the end time for a transcript chunk.

    Args:
        chunks (Sequence[TranscriptChunk]): Transcript chunks.
        idx (int): Index of the chunk.

    Returns:
        float: End time in seconds.

    """
    ch = chunks[idx]
    if ch.duration is not None:
        return float(ch.start) + float(ch.duration)
    if idx + 1 < len(chunks):
        return float(chunks[idx + 1].start)
    return float(ch.start)


def _format_window_text(chunks: Iterable[TranscriptChunk], max_chars: int) -> str:
    """Format transcript chunks with timestamps for model input.

    Args:
        chunks (Iterable[TranscriptChunk]): Transcript chunks.
        max_chars (int): Maximum output length in characters.

    Returns:
        str: Formatted transcript segment.

    """
    lines: List[str] = []
    total = 0
    for ch in chunks:
        text = _normalize_text(ch.chunk)
        if not text:
            continue
        line = f"[t={ch.start:.2f}] {text}"
        if max_chars and total + len(line) + 1 > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def iter_transcript_windows(
    chunks: Sequence[TranscriptChunk],
    window_seconds: float = QA_DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = QA_DEFAULT_OVERLAP_SECONDS,
    max_chars: int = QA_DEFAULT_MAX_WINDOW_CHARS,
) -> Iterator[TranscriptWindow]:
    """Yield sliding transcript windows aligned to chunk starts.

    Args:
        chunks (Sequence[TranscriptChunk]): Transcript chunks in chronological order.
        window_seconds (float): Window duration in seconds.
        overlap_seconds (float): Window overlap in seconds.
        max_chars (int): Max characters per window.

    Yields:
        TranscriptWindow: A timestamped transcript window.

    """
    if not chunks or window_seconds <= 0:
        return

    step = window_seconds - overlap_seconds
    if step <= 0:
        step = window_seconds

    idx = 0
    while idx < len(chunks):
        window_start = float(chunks[idx].start)
        window_end = window_start + window_seconds

        window_chunks: List[TranscriptChunk] = []
        j = idx
        while j < len(chunks):
            ch = chunks[j]
            if ch.start < window_end:
                window_chunks.append(ch)
                j += 1
            else:
                break

        text = _format_window_text(window_chunks, max_chars)
        if text:
            yield TranscriptWindow(start=window_start, end=window_end, text=text)

        next_start = window_start + step
        while idx < len(chunks) and chunks[idx].start < next_start:
            idx += 1


def load_transcript_chunks(transcript_path: Path) -> List[TranscriptChunk]:
    """Load transcript chunks from a JSONL file.

    Args:
        transcript_path (Path): Path to transcript JSONL file.

    Returns:
        list[TranscriptChunk]: Parsed and validated transcript chunks.

    """
    chunks: List[TranscriptChunk] = []
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                chunks.append(TranscriptChunk.model_validate(obj))
            except Exception:
                continue
    return chunks


def _strip_code_fence(raw: str) -> str:
    """Strip markdown code fences from a JSON payload.

    Args:
        raw (str): Raw response text.

    Returns:
        str: Cleaned response text.

    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _parse_llm_json(raw: str) -> dict:
    """Parse a model JSON response into a dict with a pairs key.

    Args:
        raw (str): Raw model response text.

    Returns:
        dict: Parsed JSON object with a "pairs" list.

    """
    cleaned = _strip_code_fence(raw or "")
    data: object
    try:
        data = json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
            except Exception:
                data = []
        else:
            data = []

        if not data:
            pairs = _salvage_pairs_from_raw(cleaned)
            return {"pairs": pairs}

    if isinstance(data, list):
        return {"pairs": data}
    if isinstance(data, dict):
        if "pairs" in data and isinstance(data["pairs"], list):
            return data
        if "items" in data and isinstance(data["items"], list):
            data["pairs"] = data["items"]
            return data
        data["pairs"] = []
        return data
    return {"pairs": []}


def _salvage_pairs_from_raw(raw: str) -> List[dict]:
    """Salvage complete pair objects from a truncated JSON response.

    This attempts to recover fully-formed objects inside the "pairs" array
    by scanning for balanced braces outside of strings.

    Args:
        raw (str): Raw model response text.

    Returns:
        list[dict]: Recovered pair objects.

    """
    if not raw:
        return []
    key_idx = raw.find('"pairs"')
    if key_idx == -1:
        key_idx = raw.find('"items"')
    if key_idx == -1:
        return []
    arr_idx = raw.find("[", key_idx)
    if arr_idx == -1:
        return []

    pairs: List[dict] = []
    in_string = False
    escape = False
    depth = 0
    obj_start: Optional[int] = None

    i = arr_idx + 1
    while i < len(raw):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and obj_start is not None:
                        obj_text = raw[obj_start : i + 1]
                        with contextlib.suppress(Exception):
                            pairs.append(json.loads(obj_text))
                        obj_start = None
            elif ch == "]":
                break
        i += 1

    return pairs


def _write_debug_text(path: Path, text: str) -> None:
    """Write debug text to disk, ensuring parent directory exists.

    Args:
        path (Path): Output file path.
        text (str): Text content to write.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_prompt(window_text: str) -> tuple[str, str]:
    """Build system and user prompts for QA extraction.

    Args:
        window_text (str): Timestamped full transcript text.

    Returns:
        tuple[str, str]: (system_prompt, user_prompt)

    """
    system_prompt = (
        "You are a strict JSON-only extraction engine. "
        "Return only valid JSON with no extra text."
    )
    user_prompt = (
        "Full transcript from a golf livestream Q&A. Each line begins with "
        "[t=SECONDS] indicating when the words were spoken.\n\n"
        "Task: extract viewer questions and the coach's corresponding answers.\n"
        "Rules:\n"
        "- A QA pair requires a clear viewer question and a matching answer.\n"
        "- Only include golf instruction questions (swing, technique, drills, strategy, equipment).\n"
        "- Skip admin/support/logistics/platform questions (site access, uploads, scheduling, availability, greetings).\n"
        "- If the answer is missing or the coach never answers, skip the question.\n"
        "- Use timestamps to set question_start/question_end/answer_start/answer_end.\n"
        "- If you cannot determine an end time, set it to null.\n"
        "- Clean up stutters, repeated words, and verbal tics. Slight paraphrasing is allowed if meaning is unchanged.\n"
        "- Output numbers as JSON numbers (not strings).\n\n"
        "Return JSON exactly in this format:\n"
        "{\n"
        '  "pairs": [\n'
        "    {\n"
        '      "question": "string",\n'
        '      "answer": "string",\n'
        '      "question_start": 123.45,\n'
        '      "question_end": 130.0,\n'
        '      "answer_start": 130.2,\n'
        '      "answer_end": 200.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        'If there are no QA pairs, return {"pairs": []}.\n\n'
        "Transcript:\n"
        f"{window_text}"
    )
    return system_prompt, user_prompt


def extract_qa_pairs_from_window(
    window_text: str,
    model: str,
    max_output_tokens: int = QA_DEFAULT_MAX_OUTPUT_TOKENS,
    debug_dir: Optional[Path] = None,
    debug_tag: Optional[str] = None,
) -> List[QAPair]:
    """Call the model to extract QA pairs from a transcript window.

    Args:
        window_text (str): Timestamped transcript text.
        model (str): OpenAI model name.
        max_output_tokens (int): Max output tokens for the model.
        debug_dir (Path | None): Optional directory to write debug artifacts.
        debug_tag (str | None): Optional tag for debug file naming.

    Returns:
        list[QAPair]: Extracted QA pairs.

    Raises:
        RuntimeError: If OpenAI client not available or API key missing.

    """
    if not window_text.strip():
        return []

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenAI client not available; install 'openai'.") from e

    settings = AppSettings()
    if not settings.openai or not settings.openai.api_key:
        raise RuntimeError("OPENAI__API_KEY not configured.")

    system_prompt, user_prompt = _build_prompt(window_text)
    client = OpenAI(api_key=settings.openai.api_key)

    tag = debug_tag or "qa_extract"
    if debug_dir is not None:
        _write_debug_text(
            debug_dir / f"{tag}_prompt.txt",
            system_prompt + "\n\n" + user_prompt,
        )

    temperature = 1.0 if model == "gpt-5" else 0.2
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_output_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content or ""
    if debug_dir is not None:
        _write_debug_text(debug_dir / f"{tag}_raw.txt", content)
    parsed = _parse_llm_json(content)
    try:
        batch = QABatch.model_validate(parsed)
    except ValidationError as e:
        if debug_dir is not None:
            _write_debug_text(debug_dir / f"{tag}_parse_error.txt", str(e))
        return []
    if debug_dir is not None:
        _write_debug_text(
            debug_dir / f"{tag}_parsed.jsonl",
            "\n".join(
                json.dumps(pair.model_dump(mode="json"), ensure_ascii=False)
                for pair in batch.pairs
            ),
        )
    return batch.pairs


def _qa_id(
    channel_key: str,
    video_id: str,
    question_start: float,
    answer_start: float,
    question: str,
    answer: str,
) -> str:
    """Build a deterministic QA identifier from key fields.

    Args:
        channel_key (str): Canonical channel key.
        video_id (str): YouTube video ID.
        question_start (float): Question start timestamp.
        answer_start (float): Answer start timestamp.
        question (str): Question text.
        answer (str): Answer text.

    Returns:
        str: Deterministic QA identifier.

    """
    raw = "|".join(
        [
            channel_key,
            video_id,
            f"{question_start:.2f}",
            f"{answer_start:.2f}",
            _normalize_text(question),
            _normalize_text(answer),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"qa_{digest}"


def _qa_output_path(
    root: Path, channel_key: str, video_id: Optional[str] = None
) -> Path:
    """Resolve output path for QA JSONL output.

    Args:
        root (Path): Repository root.
        channel_key (str): Canonical channel key.
        video_id (str | None): YouTube video ID when using per-video mode.

    Returns:
        Path: Output JSONL file path.

    Raises:
        ValueError: If per-video mode is used without a video_id.

    """
    base = root / "data" / channel_key / "qa"
    if not video_id:
        return base / "qa_pairs.jsonl"
    return base / f"{video_id}.jsonl"


def _load_existing_qa_ids(path: Path) -> Set[str]:
    """Load QA ids from an existing JSONL file.

    Args:
        path (Path): Path to an existing QA JSONL file.

    Returns:
        set[str]: Set of QA ids found in the file.

    """
    if not path.exists():
        return set()
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qa_id = str(obj.get("qa_id") or "")
                if qa_id:
                    ids.add(qa_id)
            except Exception:
                continue
    return ids


def _load_existing_video_ids(path: Path) -> Set[str]:
    """Load video ids from an existing QA JSONL file.

    Args:
        path (Path): Path to an existing QA JSONL file.

    Returns:
        set[str]: Set of video ids found in the file.

    """
    if not path.exists():
        return set()
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                vid = str(obj.get("video_id") or "")
                if vid:
                    ids.add(vid)
            except Exception:
                continue
    return ids


def _write_qa_records(path: Path, records: Iterable[QARecord], append: bool) -> int:
    """Write QA records to a JSONL file.

    Args:
        path (Path): Output JSONL file path.
        records (Iterable[QARecord]): QA records to write.
        append (bool): Append to file if True, else overwrite.

    Returns:
        int: Number of records written.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    written = 0
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            obj = rec.model_dump(mode="json")
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
    return written


def _dedupe_records(
    records: Sequence[QARecord], time_precision: int = 1
) -> List[QARecord]:
    """Deduplicate QA records by time keys.

    Args:
        records (Sequence[QARecord]): Records to deduplicate.
        time_precision (int): Decimal places to round timestamps for deduping.

    Returns:
        list[QARecord]: Deduplicated records.

    """
    seen: Set[tuple[str, str, float, float]] = set()
    unique: List[QARecord] = []
    for rec in records:
        key = (
            rec.channel_key,
            rec.video_id,
            round(rec.question_start, time_precision),
            round(rec.answer_start, time_precision),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique


def extract_qa_for_video(
    video_id: str,
    channel_key: str,
    transcripts_dir: Path,
    model: str,
    window_seconds: float,
    overlap_seconds: float,
    max_window_chars: int,
    max_output_tokens: int,
    max_windows: Optional[int] = None,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> List[QARecord]:
    """Extract QA records for a single video transcript in one pass.

    Args:
        video_id (str): YouTube video ID.
        channel_key (str): Canonical channel key.
        transcripts_dir (Path): Directory containing transcript JSONL files.
        model (str): OpenAI model name.
        window_seconds (float): Unused in full-transcript mode (kept for compatibility).
        overlap_seconds (float): Unused in full-transcript mode (kept for compatibility).
        max_window_chars (int): Max characters to include (0 = no limit).
        max_output_tokens (int): Max output tokens for the model.
        max_windows (int | None): Optional cap on number of chunks to include.
        debug (bool): Whether to write debug artifacts.
        debug_dir (Path | None): Optional directory for debug artifacts.

    Returns:
        list[QARecord]: Extracted QA records.

    """
    transcript_path = transcripts_dir / f"{video_id}.jsonl"
    if not transcript_path.exists():
        return []

    chunks = load_transcript_chunks(transcript_path)
    if not chunks:
        return []

    if max_windows is not None:
        cap = max(max_windows, 0)
        if cap == 0:
            return []
        chunks = chunks[:cap]

    transcript_text = _format_window_text(chunks, max_window_chars)
    if not transcript_text:
        return []

    transcript_start = float(chunks[0].start)
    transcript_end = _chunk_end(chunks, len(chunks) - 1)

    records: List[QARecord] = []
    seen_ids: Set[str] = set()

    effective_debug_dir = debug_dir
    if debug and effective_debug_dir is None:
        effective_debug_dir = transcripts_dir.parent / "qa" / "debug"

    pairs = extract_qa_pairs_from_window(
        transcript_text,
        model=model,
        max_output_tokens=max_output_tokens,
        debug_dir=effective_debug_dir if debug else None,
        debug_tag=video_id,
    )
    for pair in pairs:
        question = _normalize_text(pair.question)
        answer = _normalize_text(pair.answer)
        if not question or not answer:
            continue
        qa_id = _qa_id(
            channel_key,
            video_id,
            pair.question_start,
            pair.answer_start,
            question,
            answer,
        )
        if qa_id in seen_ids:
            continue
        seen_ids.add(qa_id)
        records.append(
            QARecord(
                qa_id=qa_id,
                channel_key=channel_key,
                video_id=video_id,
                question=question,
                answer=answer,
                question_start=pair.question_start,
                question_end=pair.question_end,
                answer_start=pair.answer_start,
                answer_end=pair.answer_end,
                window_start=transcript_start,
                window_end=transcript_end,
                model=model,
                prompt_version=PROMPT_VERSION,
                extracted_at=datetime.now(tz="utc"),
            )
        )

    return records


def run_qa_extraction(
    channel: str,
    video_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    include_non_livestreams: bool = False,
    model: str = "gpt-5",
    window_seconds: float = QA_DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = QA_DEFAULT_OVERLAP_SECONDS,
    max_window_chars: int = QA_DEFAULT_MAX_WINDOW_CHARS,
    max_output_tokens: int = QA_DEFAULT_MAX_OUTPUT_TOKENS,
    output_mode: str = "per-video",
    append: bool = False,
    skip_existing: bool = False,
    dry_run: bool = False,
    max_windows: Optional[int] = None,
    workers: int = QA_DEFAULT_WORKERS,
    debug: bool = False,
) -> None:
    """Run QA extraction and write outputs based on configured options.

    Args:
        channel (str): Alias, handle, or canonical channel key.
        video_ids (Sequence[str] | None): Optional explicit video IDs to process.
        limit (int | None): Optional max number of videos to process.
        include_non_livestreams (bool): Include non-livestream videos when True.
        model (str): OpenAI model name.
        window_seconds (float): Unused in full-transcript mode (kept for compatibility).
        overlap_seconds (float): Unused in full-transcript mode (kept for compatibility).
        max_window_chars (int): Max characters to include (0 = no limit).
        max_output_tokens (int): Max output tokens for the model.
        output_mode (str): "master", "per-video", or "both".
        append (bool): Append to existing output files when True.
        skip_existing (bool): Skip videos that already have QA outputs.
        dry_run (bool): Print plan without calling the LLM when True.
        max_windows (int | None): Optional cap on number of chunks to include.
        workers (int): Number of worker threads to use.
        debug (bool): Write prompt/response debug artifacts when True.

    Raises:
        KeyError: If the channel cannot be resolved from configuration.
        FileNotFoundError: If transcripts directory does not exist.
        ValueError: If output_mode is invalid.

    """
    base = Path.cwd()
    channels = load_channels_config(base / "config" / "channels.yaml")
    channel_key = resolve_channel_key(channel, channels)
    if not channel_key:
        raise KeyError(f"Channel not found for alias/handle: {channel}")

    if output_mode not in {"master", "per-video", "both"}:
        raise ValueError("output_mode must be 'master', 'per-video', or 'both'")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    transcripts_dir = base / "data" / channel_key / "transcripts"
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_dir}")

    candidate_ids: List[str] = []
    if video_ids:
        candidate_ids = [str(v).strip() for v in video_ids if str(v).strip()]
    else:
        catalog = load_channel_catalog(channel_key, root=base, channels=channels)
        for entry in catalog:
            if not include_non_livestreams and not entry.is_livestream:
                continue
            candidate_ids.append(entry.video_id)

    if limit is not None:
        candidate_ids = candidate_ids[: max(limit, 0)]

    use_master = output_mode in {"master", "both"}
    use_per_video = output_mode in {"per-video", "both"}

    qa_dir = base / "data" / channel_key / "qa"
    master_path = _qa_output_path(base, channel_key) if use_master else None
    debug_dir = qa_dir / "debug" if debug else None

    if use_master and workers > 1:
        print(
            "WARNING: workers>1 is only supported for per-video output; using 1 worker."
        )
        workers = 1

    master_append = append or skip_existing
    existing_master_ids: Set[str] = set()
    existing_master_video_ids: Set[str] = set()
    if use_master and master_path is not None and master_append:
        existing_master_ids = _load_existing_qa_ids(master_path)
        if skip_existing:
            existing_master_video_ids = _load_existing_video_ids(master_path)

    if use_master and master_path is not None and not master_append and not dry_run:
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.open("w", encoding="utf-8").close()

    if dry_run:
        planned: List[str] = []
        for vid in candidate_ids:
            transcript_path = transcripts_dir / f"{vid}.jsonl"
            if not transcript_path.exists():
                continue
            per_video_path = _qa_output_path(base, channel_key, vid)
            needs_master = use_master and (
                not skip_existing or vid not in existing_master_video_ids
            )
            needs_per_video = use_per_video and (
                not skip_existing or not per_video_path.exists()
            )
            if needs_master or needs_per_video:
                planned.append(vid)

        print("QA extraction plan:")
        print(f"  channel  : {channel_key}")
        print(f"  videos   : {len(planned)}")
        print(f"  mode     : {output_mode}")
        print(f"  model    : {model}")
        print("  input    : full-transcript")
        if max_window_chars > 0:
            print(f"  max_chars: {max_window_chars}")
        else:
            print("  max_chars: none")
        print(f"  max_tokens: {max_output_tokens}")
        if max_windows is not None:
            print(f"  max_chunks: {max_windows}")
        if use_master and master_path is not None:
            print(f"  master_output: {master_path}")
        if use_per_video:
            print(f"  per_video_output: {qa_dir}")
        if debug_dir is not None:
            print(f"  debug_dir: {debug_dir}")
        print(f"  workers  : {workers}")
        return

    if use_per_video and not use_master and workers > 1:
        todo: List[str] = []
        for vid in candidate_ids:
            transcript_path = transcripts_dir / f"{vid}.jsonl"
            if not transcript_path.exists():
                continue
            per_video_path = _qa_output_path(base, channel_key, vid)
            if skip_existing and per_video_path.exists():
                continue
            todo.append(vid)

        total_written_per_video = 0
        videos_processed = 0

        def _run_one(video_id: str) -> tuple[str, List[QARecord]]:
            recs = extract_qa_for_video(
                video_id=video_id,
                channel_key=channel_key,
                transcripts_dir=transcripts_dir,
                model=model,
                window_seconds=window_seconds,
                overlap_seconds=overlap_seconds,
                max_window_chars=max_window_chars,
                max_output_tokens=max_output_tokens,
                max_windows=max_windows,
                debug=debug,
                debug_dir=debug_dir,
            )
            recs = _dedupe_records(recs)
            return video_id, recs

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_run_one, vid): vid for vid in todo}
            for fut in as_completed(future_map):
                vid = future_map[fut]
                try:
                    video_id, records = fut.result()
                except Exception as e:
                    print(f"FAIL video_id={vid} error={type(e).__name__}: {e}")
                    continue

                per_video_path = _qa_output_path(base, channel_key, video_id)
                per_video_append = append
                existing_per_video_ids: Set[str] = set()
                if per_video_append and per_video_path.exists():
                    existing_per_video_ids = _load_existing_qa_ids(per_video_path)
                filtered = [r for r in records if r.qa_id not in existing_per_video_ids]
                if filtered:
                    total_written_per_video += _write_qa_records(
                        per_video_path, filtered, append=per_video_append
                    )
                videos_processed += 1

        print("QA extraction complete:")
        print(f"  videos_processed : {videos_processed}")
        print(f"  per_video_written: {total_written_per_video}")
        print(f"  per_video_output : {qa_dir}")
        return

    total_written_master = 0
    total_written_per_video = 0
    videos_processed = 0

    for vid in candidate_ids:
        transcript_path = transcripts_dir / f"{vid}.jsonl"
        if not transcript_path.exists():
            continue

        per_video_path = _qa_output_path(base, channel_key, vid)
        needs_master = use_master and (
            not skip_existing or vid not in existing_master_video_ids
        )
        needs_per_video = use_per_video and (
            not skip_existing or not per_video_path.exists()
        )
        if not (needs_master or needs_per_video):
            continue

        print(f"Processing video_id={vid}")
        records = extract_qa_for_video(
            video_id=vid,
            channel_key=channel_key,
            transcripts_dir=transcripts_dir,
            model=model,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            max_window_chars=max_window_chars,
            max_output_tokens=max_output_tokens,
            max_windows=max_windows,
            debug=debug,
            debug_dir=debug_dir,
        )
        records = _dedupe_records(records)

        if needs_master and master_path is not None:
            filtered = [r for r in records if r.qa_id not in existing_master_ids]
            if filtered:
                total_written_master += _write_qa_records(
                    master_path, filtered, append=True
                )
                for r in filtered:
                    existing_master_ids.add(r.qa_id)

        if needs_per_video:
            per_video_append = append
            existing_per_video_ids: Set[str] = set()
            if per_video_append and per_video_path.exists():
                existing_per_video_ids = _load_existing_qa_ids(per_video_path)
            filtered = [r for r in records if r.qa_id not in existing_per_video_ids]
            if filtered:
                total_written_per_video += _write_qa_records(
                    per_video_path, filtered, append=per_video_append
                )

        videos_processed += 1

    print("QA extraction complete:")
    print(f"  videos_processed : {videos_processed}")
    if use_master and master_path is not None:
        print(f"  master_written   : {total_written_master}")
        print(f"  master_output    : {master_path}")
    if use_per_video:
        print(f"  per_video_written: {total_written_per_video}")
        print(f"  per_video_output : {qa_dir}")
