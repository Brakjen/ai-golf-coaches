from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .config import AppSettings, load_channels_config, resolve_channel_key
from .utils import prepare_constant_context_text


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


def run_agent(channel_alias: str, question: str) -> str:
    """Run a minimal, no-RAG agent with static context and per-channel instructions.

    Loads per-channel `instructions` from channels.yaml, assembles static context
    from the channel's `constant_context_videos` transcripts, clips context to fit
    within a 128k-capable window, and calls an OpenAI chat model with hardcoded
    parameters.

    Args:
        channel_alias (str): Alias, handle, or canonical key for the channel.
        question (str): Natural language question to ask the agent.

    Returns:
        str: The agent's plain text response.

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

    # Prepare static context; tolerate missing transcripts for best-effort context
    context, missing = prepare_constant_context_text(
        channel_key, root=root, channels=channels, ensure_all=False
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

    # Hardcoded chat parameters per minimal test requirement
    model = "gpt-5"  # 128k-capable placeholder
    max_output_tokens = 1024 * 4

    # Call chat completions
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_output_tokens,
        reasoning_effort="medium",
    )

    text = resp.choices[0].message.content or ""
    return _ensure_markdown(text)


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
