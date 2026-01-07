from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .config import AppSettings, load_channels_config, resolve_channel_key
from .utils import prepare_constant_context_text

if TYPE_CHECKING:
    from .indexing import IndexedChunk


class VideoSegment(BaseModel):
    """A continuous segment of a video with relevant content.

    Attributes:
        start_seconds (int): Start time in seconds.
        end_seconds (int): End time in seconds.
        avg_relevance_score (float | None): Average relevance score for this segment.

    """

    start_seconds: int
    end_seconds: int
    avg_relevance_score: Optional[float] = None


class VideoRecommendation(BaseModel):
    """A recommended video with relevant segments.

    Attributes:
        video_id (str): YouTube video ID.
        title (str): Video title.
        url (str): Full YouTube URL.
        segments (list[VideoSegment]): List of relevant segments in the video.

    """

    video_id: str
    title: str
    url: str
    segments: List[VideoSegment] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Structured response from the golf coach agent.

    Attributes:
        response_text (str): The main agent response (markdown formatted).
        category (str): The classified question category.
        video_recommendations (list[VideoRecommendation]): RAG-retrieved video recommendations.

    """

    response_text: str
    category: str
    video_recommendations: List[VideoRecommendation] = Field(default_factory=list)


def _clip_text_to_tokens(
    text: str, max_tokens: int, encoding: str | None = None
) -> str:
    """Clip a text to a maximum token budget using tiktoken.

    Attempts to use GPT-4o encoding ("o200k_base"), falling back to
    "cl100k_base" if unavailable.

    Args:
        text (str): Input text to clip.
        max_tokens (int): Maximum number of tokens allowed.
        encoding (str | None): Optional explicit encoding name.

    Returns:
        str: Possibly truncated text that fits within the token budget.

    """
    try:
        import tiktoken  # type: ignore
    except Exception:  # pragma: no cover
        # If tiktoken isn't available, return the original text
        return text

    enc_name = encoding
    if enc_name is None:
        try:
            tiktoken.get_encoding("o200k_base")
            enc_name = "o200k_base"
        except Exception:
            enc_name = "cl100k_base"

    enc = tiktoken.get_encoding(enc_name)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    clipped = enc.decode(tokens[:max_tokens])
    return clipped


def _build_messages(instructions: str, context: str, question: str) -> Tuple[list, int]:
    """Construct chat messages and estimate token usage for clipping.

    Args:
        instructions (str): System instructions content.
        context (str): Static context assembled from transcripts.
        question (str): User question.

    Returns:
        Tuple[list, int]: (messages, approximate_total_token_count)

    """
    system_base = (
        instructions.strip() if instructions else "You are a helpful golf coach."
    )
    system = (
        system_base
        + "\n\nCRITICAL formatting rules:"
        + "\n- Respond in GitHub-flavored Markdown only."
        + "\n- Use headings (##, ###), bullet points, and bold where helpful."
        + "\n- Do not output plain text or non-markdown preambles."
    )
    user_content = (
        "Context (from channel transcripts):\n"
        + context.strip()
        + "\n\n"
        + "Question:\n"
        + question.strip()
        + "\n\nFormat: Markdown only (use headings, bullets, bold)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    # Approximate token count by concatenating contents; used upstream for clipping
    approx_tokens = len((system + "\n" + user_content).split())
    return messages, approx_tokens


def _ensure_markdown(text: str) -> str:
    """Ensure the output is at least minimally formatted as Markdown.

    If no common Markdown structures are detected, wrap with a heading.

    Args:
        text (str): Model output text.

    Returns:
        str: Markdown-safe output.

    """
    if not text:
        return text
    has_heading = any(h in text for h in ("# ", "## ", "### "))
    has_list = any(s in text for s in ("\n- ", "\n* ", "\n1. "))
    if has_heading or has_list:
        return text
    return f"## Answer\n\n{text}"


def _build_video_recommendations(
    chunks: List[IndexedChunk],
) -> List[VideoRecommendation]:
    """Build structured video recommendations from retrieved chunks.

    Groups chunks by video_id and consolidates nearby timestamps into segments.

    Args:
        chunks (List[IndexedChunk]): Retrieved chunks from vector store.

    Returns:
        List[VideoRecommendation]: Structured video recommendations.

    """
    if not chunks:
        return []

    # Group chunks by video_id while preserving order of first appearance
    video_map: Dict[str, List[IndexedChunk]] = defaultdict(list)
    video_order: List[str] = []
    for chunk in chunks:
        if chunk.video_id not in video_map:
            video_order.append(chunk.video_id)
        video_map[chunk.video_id].append(chunk)

    recommendations: List[VideoRecommendation] = []

    for video_id in video_order:
        video_chunks = video_map[video_id]
        first_chunk = video_chunks[0]
        title = first_chunk.title or video_id
        video_url = f"https://youtube.com/watch?v={video_id}"

        # Sort chunks by start time
        sorted_chunks = sorted(video_chunks, key=lambda c: c.start)

        # Group nearby timestamps into ranges (within 30 seconds)
        timestamp_groups: List[List[IndexedChunk]] = []
        current_group: List[IndexedChunk] = []

        for chunk in sorted_chunks:
            if not current_group:
                current_group.append(chunk)
            else:
                # If this chunk starts within 30s of the last chunk's end, add to current group
                last_chunk = current_group[-1]
                if chunk.start - last_chunk.end <= 30:
                    current_group.append(chunk)
                else:
                    # Start new group
                    timestamp_groups.append(current_group)
                    current_group = [chunk]

        # Don't forget the last group
        if current_group:
            timestamp_groups.append(current_group)

        # Build segments from groups
        segments: List[VideoSegment] = []
        for group in timestamp_groups:
            start_t = int(group[0].start)
            end_t = int(group[-1].end)

            # Calculate average score for the group
            scores = [c.score for c in group if c.score is not None]
            avg_score = sum(scores) / len(scores) if scores else None

            segments.append(
                VideoSegment(
                    start_seconds=start_t,
                    end_seconds=end_t,
                    avg_relevance_score=avg_score,
                )
            )

        recommendations.append(
            VideoRecommendation(
                video_id=video_id,
                title=title,
                url=video_url,
                segments=segments,
            )
        )

    return recommendations


def _format_video_recommendations(recommendations: List[VideoRecommendation]) -> str:
    """Format video recommendations as markdown for display.

    Args:
        recommendations (List[VideoRecommendation]): Structured video recommendations.

    Returns:
        str: Markdown-formatted video recommendations section.

    """
    if not recommendations:
        return ""

    lines = ["\n\n---\n\n## 📺 Relevant Videos\n"]

    for rec in recommendations:
        lines.append(f"\n### [{rec.title}]({rec.url})")

        # Format segments
        if (
            len(rec.segments) == 1
            and rec.segments[0].start_seconds == rec.segments[0].end_seconds
        ):
            # Single timestamp
            seg = rec.segments[0]
            score_str = (
                f" (relevance: {seg.avg_relevance_score:.2f})"
                if seg.avg_relevance_score
                else ""
            )
            timestamp_url = f"{rec.url}&t={seg.start_seconds}s"
            lines.append(
                f"- [Jump to {seg.start_seconds}s]({timestamp_url}){score_str}"
            )
        else:
            # Multiple segments
            lines.append("\n**Relevant segments:**")
            for seg in rec.segments:
                score_str = (
                    f" (avg relevance: {seg.avg_relevance_score:.2f})"
                    if seg.avg_relevance_score
                    else ""
                )
                timestamp_url = f"{rec.url}&t={seg.start_seconds}s"
                if seg.start_seconds == seg.end_seconds:
                    # Single point
                    lines.append(
                        f"- [{seg.start_seconds}s]({timestamp_url}){score_str}"
                    )
                else:
                    # Range
                    lines.append(
                        f"- [{seg.start_seconds}s - {seg.end_seconds}s]({timestamp_url}){score_str}"
                    )

    return "\n".join(lines)


def run_agent(
    channel_alias: str,
    question: str,
    category: str | None = None,
    model: str = "gpt-4o",
    include_video_recommendations: bool = True,
    rag_top_k: int = 20,
) -> AgentResponse:
    """Run a minimal, no-RAG agent with static context and per-channel instructions.

    Loads per-channel `instructions` from channels.yaml, classifies the question
    into a category (if not provided), assembles category-specific static context
    from the channel's `constant_context_videos` transcripts, clips context to fit
    within a 128k-capable window, and calls an OpenAI chat model with hardcoded
    parameters.

    Optionally queries hosted vector stores for relevant video recommendations
    that are appended to the response as a separate section.

    Args:
        channel_alias (str): Alias, handle, or canonical key for the channel.
        question (str): Natural language question to ask the agent.
        category (str | None): Optional pre-classified category to avoid re-classification.
        model (str): OpenAI model to use (default: gpt-4o).
        include_video_recommendations (bool): Whether to query RAG for video recommendations (default: True).
        rag_top_k (int): Number of RAG results to retrieve (default: 20).

    Returns:
        AgentResponse: Structured response with text and video recommendations.

    Raises:
        RuntimeError: If OpenAI settings or API key are missing.
        KeyError: If the channel cannot be resolved from configuration.

    """
    # Resolve channels and instructions
    root = Path(__file__).resolve().parent.parent
    channels = load_channels_config(root / "config" / "channels.yaml")
    channel_key = resolve_channel_key(channel_alias, channels)
    if not channel_key:
        raise KeyError(f"Channel not found for alias/handle: {channel_alias}")

    entry = channels.get(channel_key) or {}
    instructions: str = str(entry.get("instructions") or "")

    # Classify question to select appropriate context category if not provided
    if category is None:
        from .classifier import classify_question_category

        pred = classify_question_category(question, channel_alias=channel_alias)
        category = pred.category

    # Prepare static context; tolerate missing transcripts for best-effort context
    context, missing = prepare_constant_context_text(
        channel_key, root=root, channels=channels, ensure_all=False, category=category
    )

    # Clip context to a conservative budget, reserving headroom for instructions and output
    # Target ~110k tokens for context; model window 128k
    context = _clip_text_to_tokens(context, max_tokens=125_000)

    messages, _ = _build_messages(instructions, context, question)

    # OpenAI client
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenAI client not available; install 'openai'.") from e

    settings = AppSettings()
    if not settings.openai or not settings.openai.api_key:
        raise RuntimeError("OPENAI__API_KEY not configured.")

    client = OpenAI(api_key=settings.openai.api_key)

    # Set parameters - gpt-5 requires temperature=1, others use 0.7
    max_output_tokens = 1024 * 4
    temperature = 1.0 if model == "gpt-5" else 0.7

    # Call chat completions with standard parameters
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_output_tokens,
        temperature=temperature,
    )

    text = resp.choices[0].message.content or ""
    text = _ensure_markdown(text)

    # Retrieve video recommendations if enabled and vector stores are available
    video_recommendations: List[VideoRecommendation] = []
    if include_video_recommendations and category:
        try:
            from .indexing import query_hosted_vector_store

            chunks = query_hosted_vector_store(
                channel_alias, question, category, top_k=rag_top_k
            )
            video_recommendations = _build_video_recommendations(chunks)
        except Exception as e:
            # Log error to stderr but don't fail the entire agent response
            import sys

            print(
                f"WARNING: Could not retrieve video recommendations: {e}",
                file=sys.stderr,
            )

    return AgentResponse(
        response_text=text,
        category=category,
        video_recommendations=video_recommendations,
    )


def summarize_for_header(text: str, max_chars: int = 120) -> str:
    """Summarize a response concisely for UI headers.

    Aims for a short 5–7 word phrase via a fast, low-cost
    OpenAI model. Does not hard-enforce word count; relies
    on prompt guidance. Falls back to a local first-line
    snippet if the API call fails.

    Args:
        text (str): Full response text (Markdown allowed).
        max_chars (int): Target maximum characters for the summary.

    Returns:
        str: Concise summary suitable for expander titles.

    """
    src = text.strip()
    if not src:
        return "Response"

    try:
        from openai import OpenAI  # type: ignore

        settings = AppSettings()
        client = OpenAI(api_key=settings.openai.api_key)

        prompt = (
            "Summarize the assistant response into a VERY SHORT phrase (ideally 5–7 words) for a collapsible header.\n"
            + "- Plain text only (no quotes/markdown).\n"
            + "- Specific and helpful; prefer nouns/verbs.\n"
            + "- Avoid punctuation and filler words.\n\n"
            + src
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise summarizer."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=64,
            temperature=0.2,
        )
        summary = (resp.choices[0].message.content or "").strip()
        if summary:
            # Soft character cap to avoid long titles
            if len(summary) > max_chars:
                summary = summary[:max_chars].rstrip()
            return summary
    except Exception:
        # Fall through to local first-sentence fallback
        pass

    # Local fallback: first sentence/line, lightly cap to ~7 words
    import re

    s = re.sub(r"^[#>*\-\s]+", "", src, flags=re.MULTILINE)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    first = lines[0] if lines else src
    parts = re.split(r"(?<=[.!?])\s+", first)
    sent = parts[0].strip() if parts else first.strip()
    words = sent.split()
    summary = " ".join(words[:7]) if words else sent
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip()
    return summary or "Response"
