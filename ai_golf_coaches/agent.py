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
    system = instructions.strip() if instructions else "You are a helpful golf coach."
    user_content = (
        "Context (from channel transcripts):\n"
        + context.strip()
        + "\n\n"
        + "Question:\n"
        + question.strip()
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    with open("context", "w") as f:
        f.write(context)

    # Approximate token count by concatenating contents; used upstream for clipping
    approx_tokens = len((system + "\n" + user_content).split())
    return messages, approx_tokens


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
        reasoning_effort="high",
    )

    text = resp.choices[0].message.content or ""
    return text
