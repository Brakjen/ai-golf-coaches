from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from ai_golf_coaches.agent import (
    AgentResponse,
    _build_messages,  # reuse to mirror agent prompt construction
    _clip_text_to_tokens,  # reuse to mirror agent context clipping
    run_agent,
    summarize_for_header,
)
from ai_golf_coaches.config import load_channels_config, resolve_channel_key
from ai_golf_coaches.utils import prepare_constant_context_text

# Model pricing per 1M tokens (input, output) in USD
MODEL_PRICING = {
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (2.50, 10.00),
    "gpt-5": (5.00, 15.00),
}

# Model info with two-word summaries
MODEL_INFO = {
    "gpt-4o-mini": "Super Fast",
    "gpt-4o": "Balanced",
    "gpt-5": "Most Capable",
}


# Relevance thresholds (RAG similarity score 0..1)
RELEVANCE_GREEN_MIN = 0.70
RELEVANCE_YELLOW_MIN = 0.50


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_channel_keys() -> List[str]:
    """Load enabled channel keys from config.

    Returns:
        List[str]: Sorted list of enabled channel keys.

    """
    cfg_path = _repo_root() / "config" / "channels.yaml"
    channels: Dict[str, dict] = load_channels_config(cfg_path)
    # Filter to only enabled channels
    enabled = [key for key, cfg in channels.items() if cfg.get("enabled", True)]
    return sorted(enabled)


def _get_secret_storage_key() -> Optional[str]:
    """Fetch the storage API key from Streamlit secrets if present."""
    try:
        return st.secrets.get("OPENAI__API_KEY_STORAGE_INDEX")
    except Exception:
        return None


def _ensure_rag_key() -> Tuple[bool, Optional[str]]:
    """Ensure a storage key is available for RAG and return status + source."""
    env_key = os.getenv("OPENAI__API_KEY_STORAGE_INDEX")
    if env_key:
        return True, "env"

    secret_key = _get_secret_storage_key()
    if secret_key:
        os.environ["OPENAI__API_KEY_STORAGE_INDEX"] = secret_key
        return True, "secrets"

    user_key = st.session_state.get("openai_api_key")
    if user_key:
        os.environ["OPENAI__API_KEY_STORAGE_INDEX"] = user_key
        return True, "user"

    return False, None


def _verify_chat_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """Verify chat completions access with a minimal request."""
    if not api_key:
        return False, "Missing API key"
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "ping"},
                {"role": "user", "content": "ping"},
            ],
            max_completion_tokens=1,
            temperature=0,
        )
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _verify_rag_key(api_key: str) -> Tuple[bool, Optional[str]]:
    """Verify vector store access with a minimal retrieve call."""
    if not api_key:
        return False, "Missing storage key"
    try:
        from openai import OpenAI  # type: ignore

        root = _repo_root()
        channels = load_channels_config(root / "config" / "channels.yaml")
        entry = channels.get("elitegolfschools", {})
        vs_ids = entry.get("vector_store_ids", {}).get("longform", {})
        target_id = vs_ids.get("all") or vs_ids.get("full_swing")
        if not target_id:
            return False, "No vector store ID configured"

        client = OpenAI(api_key=api_key)
        vector_stores = None
        if hasattr(client, "beta") and hasattr(client.beta, "vector_stores"):
            vector_stores = client.beta.vector_stores
        elif hasattr(client, "vector_stores"):
            vector_stores = client.vector_stores
        if vector_stores is None:
            return False, "OpenAI client too old for vector store APIs"
        vector_stores.retrieve(target_id)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _assistant_avatar(channel_key: str) -> str:
    """Return avatar for assistant messages based on selected channel.

    Prefers a local image; falls back to an emoji.

    Args:
        channel_key (str): Canonical channel key (e.g., "elitegolfschools").

    Returns:
        str: Path to local image or emoji string.

    """
    assets_dir = Path(__file__).resolve().parent / "ai_golf_coaches" / "assets"
    if channel_key == "elitegolfschools":
        icon = assets_dir / "icon_riley.png"
        if icon.exists():
            return str(icon)
    if channel_key == "milolinesgolf":
        icon = assets_dir / "icon_milo.png"
        if icon.exists():
            return str(icon)
    return "🏌️"


def _user_avatar_path() -> str:
    """Return the static user avatar path (fallback to emoji if missing)."""
    icon = _repo_root() / "ai_golf_coaches" / "assets" / "icon_user.png"
    return str(icon) if icon.exists() else "👤"


def _one_line_summary(text: str, max_len: int = 120) -> str:
    """Return a short, one-sentence summary suitable for a header.

    Strips common Markdown markers, takes the first sentence or line,
    and trims to a maximum length.

    Args:
        text (str): Full assistant response in Markdown.
        max_len (int): Maximum characters for the summary.

    Returns:
        str: Single-line summary.

    """
    s = text.strip()
    if not s:
        return "Response"
    s = re.sub(r"^[#>*\-\s]+", "", s, flags=re.MULTILINE)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return "Response"
    first = lines[0]
    parts = re.split(r"(?<=[.!?])\s+", first)
    sent = parts[0].strip() if parts else first
    if len(sent) < 20 and len(lines) > 1:
        combo = f"{first} {lines[1]}".strip()
        parts2 = re.split(r"(?<=[.!?])\s+", combo)
        sent = parts2[0].strip() if parts2 else combo
    if len(sent) > max_len:
        sent = sent[:max_len].rstrip()
        sent = re.sub(r"\s+\S*$", "", sent)
        sent += "…"
    return sent or "Response"


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken with a sensible default encoding.

    Tries "o200k_base" then falls back to "cl100k_base".

    Args:
        text (str): Input text to tokenize.

    Returns:
        int: Number of tokens.

    """
    try:
        import tiktoken  # type: ignore
    except Exception:
        return 0
    try:
        enc = tiktoken.get_encoding("o200k_base")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def _youtube_embed_url(video_id: str, start_seconds: Optional[int] = None) -> str:
    """Build a YouTube embed URL for a video and optional start time.

    Args:
        video_id (str): YouTube video ID.
        start_seconds (int | None): Start time in seconds.

    Returns:
        str: Embed URL.

    """
    start_q = f"&start={int(start_seconds)}" if start_seconds is not None else ""
    return (
        f"https://www.youtube.com/embed/{video_id}"
        f"?modestbranding=1&rel=0&playsinline=1{start_q}"
    )


def _youtube_watch_url(video_id: str, start_seconds: Optional[int] = None) -> str:
    """Build a standard YouTube watch URL for a video and optional start time.

    Args:
        video_id (str): YouTube video ID.
        start_seconds (int | None): Start time in seconds.

    Returns:
        str: Watch URL.

    """
    if start_seconds is not None and start_seconds >= 0:
        return f"https://www.youtube.com/watch?v={video_id}&t={int(start_seconds)}s"
    return f"https://www.youtube.com/watch?v={video_id}"


def _relevance_indicator(score: Optional[float]) -> str:
    """Return a compact relevance indicator for a score.

    Thresholds:
        - Green: >= 0.70
        - Yellow: 0.50 - 0.69...
        - Red: < 0.50

    Args:
        score (float | None): Similarity score.

    Returns:
        str: A short indicator like "🟢".

    """
    if score is None:
        return "⚪"
    s = float(score)
    if s >= RELEVANCE_GREEN_MIN:
        return "🟢"
    if s >= RELEVANCE_YELLOW_MIN:
        return "🟡"
    return "🔴"


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Fetch a value from dict-like or attribute-like objects.

    Args:
        obj (Any): Object to read from.
        key (str): Dict key or attribute name.
        default (Any): Default if not found.

    Returns:
        Any: Value if present, else default.

    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_video_recommendations(recs: Any) -> List[Dict[str, Any]]:
    """Normalize agent video recommendations into a renderer-friendly structure.

    Produces:
        [
          {
            "video_id": str,
            "title": str | None,
            "segments": [
               {"start_seconds": int, "end_seconds": int | None, "score": float | None}
            ]
          },
          ...
        ]

    Accepts pydantic objects or dicts.

    Args:
        recs (Any): AgentResponse.video_recommendations-like value.

    Returns:
        List[Dict[str, Any]]: Normalized list.

    """
    if not recs:
        return []

    items: List[Any] = recs if isinstance(recs, list) else [recs]
    out: List[Dict[str, Any]] = []

    for rec in items:
        video_id = _safe_get(rec, "video_id")
        if not video_id:
            continue
        title = _safe_get(rec, "title")

        raw_segments = _safe_get(rec, "segments") or []
        segments_out: List[Dict[str, Any]] = []
        if isinstance(raw_segments, list):
            for seg in raw_segments:
                start_val = _safe_get(seg, "start_seconds")
                end_val = _safe_get(seg, "end_seconds")
                score_val = (
                    _safe_get(seg, "avg_relevance_score")
                    if _safe_get(seg, "avg_relevance_score") is not None
                    else _safe_get(seg, "score")
                )

                try:
                    start_seconds = (
                        int(float(start_val)) if start_val is not None else 0
                    )
                except Exception:
                    start_seconds = 0
                try:
                    end_seconds = int(float(end_val)) if end_val is not None else None
                except Exception:
                    end_seconds = None
                try:
                    score = float(score_val) if score_val is not None else None
                except Exception:
                    score = None

                segments_out.append(
                    {
                        "start_seconds": max(0, start_seconds),
                        "end_seconds": end_seconds,
                        "score": score,
                    }
                )

        # Sort best-first; None scores last
        segments_out.sort(
            key=lambda x: (
                -(x["score"] if x["score"] is not None else -1.0),
                x["start_seconds"],
            )
        )

        out.append({"video_id": video_id, "title": title, "segments": segments_out})

    return out


def _video_relevance_score(segments: List[Dict[str, Any]]) -> Optional[float]:
    """Compute a single relevance score for a video from its segment scores.

    We still want to leverage segment-level similarity without showing timestamps.
    A simple and robust proxy is the max score across all matched segments.

    Note:
        Segment objects may provide `avg_relevance_score`; during normalization we map
        that value into the segment's `score` field.

    Args:
        segments (List[Dict[str, Any]]): Normalized segments with optional `score`.

    Returns:
        Optional[float]: Best segment score, or None if no valid scores exist.

    """
    scores: List[float] = [
        float(seg["score"]) for seg in segments if seg.get("score") is not None
    ]
    if not scores:
        return None
    return max(scores)


def _render_video_recommendations(
    video_recommendations: Any, *, rag_enabled: bool, **_kwargs: Any
) -> None:
    """Render embedded videos with clickable timestamp buttons inside the expander.

    Notes:
        Timestamp-based seeking (via buttons) was intentionally removed because it
        felt slow/fiddly due to iframe reloads in Streamlit. We instead embed the
        recommended videos and show a per-video relevance indicator derived from
        segment scores.

    Args:
        video_recommendations (Any): Raw recommendations from AgentResponse.
        rag_enabled (bool): Flag to indicate whether RAG is enabled or not.

    Returns:
        None

    """
    normalized = _normalize_video_recommendations(video_recommendations)
    if not normalized:
        if not rag_enabled:
            st.info(
                "Video recommendations are disabled. Set "
                "`OPENAI__API_KEY_STORAGE_INDEX` to enable RAG."
            )
        return

    st.markdown("### Watch")
    st.caption("Relevance: 🟢 ≥ 0.70, 🟡 0.50–0.69, 🔴 < 0.50.")

    for rec in normalized:
        video_id: str = rec["video_id"]
        title: str = rec.get("title") or f"Video {video_id}"
        segments: List[Dict[str, Any]] = rec.get("segments") or []

        video_score = _video_relevance_score(segments)

        header = f"{_relevance_indicator(video_score)} - {title}"
        with st.expander(header, expanded=False):
            cols = st.columns([5, 2])
            with cols[0]:
                st.subheader(title)
            with cols[1]:
                st.markdown(
                    (
                        '<div style="text-align:right;">'
                        f'<a href="{_youtube_watch_url(video_id, start_seconds=None)}" '
                        'target="_blank" '
                        'style="display:inline-block;padding:0.4rem 0.75rem;'
                        "border-radius:0.5rem;border:1px solid #3a3f4b;"
                        "background:#2a2f3a;color:#e6e8ee;text-decoration:none;"
                        'font-size:0.9rem;">Open on YouTube</a>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

            st.caption(f"{_relevance_indicator(video_score)} Relevance")
            st.video(_youtube_watch_url(video_id, start_seconds=None), width="stretch")


def _calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate cost in USD for a model call based on token counts.

    Args:
        input_tokens (int): Number of input tokens (system + user).
        output_tokens (int): Number of output tokens (assistant).
        model (str): Model identifier (e.g., 'gpt-4o-mini').

    Returns:
        float: Cost in USD.

    """
    pricing = MODEL_PRICING.get(model, (0.0, 0.0))
    input_cost = (input_tokens / 1_000_000) * pricing[0]
    output_cost = (output_tokens / 1_000_000) * pricing[1]
    return input_cost + output_cost


def _compute_turn_token_usage(
    channel_key: str, question: str, reply: str
) -> Dict[str, int | float]:
    """Compute tokens for system+user (instructions+context+question) and assistant reply.

    Mirrors the prompt construction in `ai_golf_coaches.agent.run_agent` for accuracy,
    then tokenizes with tiktoken and returns a breakdown plus total and percent of 128k.
    Also calculates cost based on model pricing.

    Args:
        channel_key (str): Canonical channel key.
        question (str): The user question just asked.
        reply (str): The assistant reply content.

    Returns:
        Dict[str, int | float]: Keys: system_tokens, user_tokens, assistant_tokens,
            total_tokens, percent_128k, main_cost, classifier_cost, total_cost.

    """
    from ai_golf_coaches.classifier import classify_question_category

    root = Path(__file__).resolve().parent
    channels = load_channels_config(root / "config" / "channels.yaml")
    entry = channels.get(channel_key) or {}
    instructions: str = str(entry.get("instructions") or "")

    # Classify to match agent's category-specific context selection
    pred = classify_question_category(question, channel_alias=channel_key)
    category = pred.category

    # Prepare and clip context as the agent does
    context, _missing = prepare_constant_context_text(
        channel_key, root=root, channels=channels, ensure_all=False, category=category
    )
    context = _clip_text_to_tokens(context, max_tokens=125_000)

    messages, _ = _build_messages(instructions, context, question)
    system_text = messages[0]["content"]
    user_text = messages[1]["content"]

    system_tokens = _count_tokens(system_text)
    user_tokens = _count_tokens(user_text)
    assistant_tokens = _count_tokens(reply)
    total = system_tokens + user_tokens + assistant_tokens
    percent = min(100.0, (total / 128_000) * 100.0)

    # Calculate costs
    # Get the model used for this turn from session state, default to gpt-4o
    main_model = st.session_state.get("selected_model", "gpt-4o")
    main_cost = _calculate_cost(
        system_tokens + user_tokens, assistant_tokens, main_model
    )
    # Classification call (gpt-4o-mini default, ~16 tokens output, estimate ~200 input)
    classifier_cost = _calculate_cost(200, 16, "gpt-4o-mini")
    # Summarization call for header (gpt-4o-mini, estimate ~500 input, 64 output)
    summary_cost = _calculate_cost(500, 64, "gpt-4o-mini")
    total_cost = main_cost + classifier_cost + summary_cost

    return {
        "system_tokens": system_tokens,
        "user_tokens": user_tokens,
        "assistant_tokens": assistant_tokens,
        "total_tokens": total,
        "percent_128k": percent,
        "main_cost": main_cost,
        "classifier_cost": classifier_cost,
        "summary_cost": summary_cost,
        "total_cost": total_cost,
    }


def _render_context_meter(
    placeholder: st.delta_generator.DeltaGenerator, usage: Dict[str, int | float] | None
) -> None:
    """Render or update the context window meter in the sidebar.

    Args:
        placeholder: A Streamlit placeholder/container created with `st.sidebar.empty()`.
        usage (dict | None): Token usage breakdown or None if not yet available.

    Returns:
        None

    """
    with placeholder.container():
        pct = (
            int(round(float(usage.get("percent_128k", 0.0))))
            if usage and isinstance(usage, dict)
            else 0
        )
        st.progress(pct, text="Context Window (128k token limit")


def _render_session_cost(
    placeholder: st.delta_generator.DeltaGenerator, cumulative_cost: float
) -> None:
    """Render or update the session cost metric in the sidebar.

    Args:
        placeholder: A Streamlit placeholder/container created with `st.sidebar.empty()`.
        cumulative_cost (float): Total accumulated cost for the session.

    Returns:
        None

    """
    with placeholder.container():
        cap = 10.0
        pct = min(100.0, (cumulative_cost / cap) * 100.0) if cap > 0 else 0.0
        st.progress(int(round(pct)), text="Session Cost ($10 limit)")


def _show_welcome_screen() -> bool:
    """Display a quick guide screen when explicitly requested.

    Returns:
        bool: True if the app should continue rendering, False if guide is open.

    """
    if not st.session_state.get("show_help", False):
        return True

    # Find logo path
    logo_path = (
        Path(__file__).resolve().parent / "ai_golf_coaches" / "assets" / "logo.png"
    )

    # Display logo (moved to bottom)

    if st.button("Close", width="stretch", type="secondary"):
        st.session_state["show_help"] = False
        st.rerun()

    # Tabbed content
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📖 Quick Start", "🎥 Data Sources", "📚 Video Library", "ℹ️ About"]
    )

    with tab1:
        st.markdown("""
        ## Quick Start

        ### Getting Started
        1. **Enter your OpenAI API key** when prompted
        2. **Ask your golf question** in the chat input
        3. **Review answers and recommended videos**

        ### Features
        - **Category detection** to load the most relevant EGS context
        - **Context usage meter** for the 128k window
        - **Session cost meter** with a $10 guide rail
        - **Collapsible answers** for easy scanning
        - **Expandable video recommendations** with relevance indicators

        ### Tips
        - Ask one focused question at a time
        - Use the sidebar to update your API key
        - Your API key is stored only in session memory
        """)

    with tab2:
        st.markdown("""
        ## Data Sources

        This coach learns from curated YouTube instruction by:

        ### Elite Golf Schools (@elitegolfschools)
        - **Instructor**: Riley Andrews and team
        - **Focus**: Fundamentals, swing mechanics, healthy movement
        - **Videos**: Long‑form instruction across categories

        ### How It Works
        1. We fetch transcripts from selected videos
        2. Videos are grouped by category
        3. Your question is classified to pick the best category context
        4. The AI answers using that coach’s knowledge and tone

        ### What We Don’t Include
        - Shorts
        - Podcast episodes
        - Live Q&A sessions (for now)
        """)

    with tab3:
        st.markdown("## Video Library")
        st.caption("Browse the curated EGS videos used for each context category.")

        cfg_path = Path(__file__).resolve().parent / "config" / "channels.yaml"
        channels_cfg = load_channels_config(cfg_path)

        # Load channel config to get video IDs per category
        try:
            channel_entry = channels_cfg.get("elitegolfschools", {})
            context_videos = channel_entry.get("constant_context_videos", {})

            if isinstance(context_videos, dict):
                # Load catalog to get video titles
                try:
                    from ai_golf_coaches.config import load_channel_catalog

                    catalog = load_channel_catalog("elitegolfschools")
                    video_titles = {v.video_id: v.title for v in catalog}
                except Exception:
                    video_titles = {}

                # Display videos per category
                for category, video_ids in context_videos.items():
                    if not video_ids:
                        continue

                    # Category header
                    cat_display = category.replace("_", " ").title()
                    st.markdown(f"### {cat_display} ({len(video_ids)} videos)")

                    # Create grid of videos (3 per row for compact display)
                    for i in range(0, len(video_ids), 3):
                        cols = st.columns(3)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(video_ids):
                                video_id = video_ids[idx]
                                title = video_titles.get(video_id, f"Video {video_id}")
                                with col:
                                    # Show title (truncated)
                                    title_short = (
                                        title[:40] + "..." if len(title) > 40 else title
                                    )
                                    st.caption(title_short)
                                    # Embed video with smaller size
                                    video_url = (
                                        f"https://www.youtube.com/embed/{video_id}"
                                    )
                                    st.markdown(
                                        f'<iframe width="100%" height="150" '
                                        f'src="{video_url}" '
                                        f'frameborder="0" allow="accelerometer; autoplay; '
                                        f"clipboard-write; encrypted-media; gyroscope; "
                                        f'picture-in-picture" allowfullscreen></iframe>',
                                        unsafe_allow_html=True,
                                    )
            else:
                st.info("No categorized videos found for this channel.")

        except Exception as e:
            st.error(f"Failed to load video library: {type(e).__name__}: {e}")

    with tab4:
        st.markdown("""
        ## About This App

        ### Technical Details
        - **Model**: GPT‑5 (128k context window)
        - **Classification**: GPT‑4o‑mini for fast category detection
        - **Context**: Static EGS context loaded by category
        - **Video Recs**: Enabled when a storage key is available

        ### Architecture
        - Static context compiled from curated EGS videos
        - Category‑specific subsets for relevance
        - Token counting for transparency

        ### Privacy
        - API keys stored only in session memory
        - No conversation history saved to disk
        - All processing happens via OpenAI APIs

        ### Version
        - **v0.0.1** — Proof of Concept
        """)

    # Video demonstration (moved to bottom)
    with st.expander("Don't disturb the shaft", expanded=False):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                '<iframe width="100%" height="400" '
                'src="https://www.youtube.com/embed/pQY9jbMh4hg" '
                'frameborder="0" allow="accelerometer; autoplay; '
                "clipboard-write; encrypted-media; gyroscope; "
                'picture-in-picture" allowfullscreen></iframe>',
                unsafe_allow_html=True,
            )

    # Display logo at the bottom of the guide
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if logo_path.exists():
            st.image(str(logo_path), width="stretch")
        else:
            st.markdown(
                "<h1 style='text-align: center;'>🏌️ AI Golf Coach</h1>",
                unsafe_allow_html=True,
            )

    if st.button("Close", width="stretch", type="secondary", key="close_help_bottom"):
        st.session_state["show_help"] = False
        st.rerun()

    return False


@st.dialog("Enter OpenAI API Key")
def _api_key_dialog() -> None:
    """Prompt the user to enter or update their OpenAI API key."""
    api_val = st.text_input(
        "OpenAI API Key",
        type="password",
        key="pending_openai_key",
        label_visibility="collapsed",
        placeholder="Enter API key...",
    )
    with st.columns(1)[0]:
        if st.button("Save", width="stretch", type="primary"):
            if api_val:
                st.session_state["openai_api_key"] = api_val.strip()
                os.environ["OPENAI__API_KEY"] = st.session_state["openai_api_key"]
                chat_ok, chat_err = _verify_chat_key(st.session_state["openai_api_key"])
                st.session_state["chat_key_verified"] = chat_ok
                st.session_state["chat_key_error"] = chat_err

                rag_ok, rag_err = _verify_rag_key(
                    os.getenv("OPENAI__API_KEY_STORAGE_INDEX", "")
                )
                st.session_state["rag_key_verified"] = rag_ok
                st.session_state["rag_key_error"] = rag_err
                st.session_state["show_key_dialog"] = False
                st.rerun()
            else:
                st.warning("Please enter a key.")


def main() -> None:
    """Generate the Streamlit app UI and handle interactions.

    Returns:
        None

    """
    st.set_page_config(page_title="AI Golf Coach (PoC)", page_icon="🏌️", layout="wide")

    # Show welcome screen on first load or when help is requested
    if not _show_welcome_screen():
        # User is viewing welcome screen, don't show main app
        return

    logo_path = (
        Path(__file__).resolve().parent
        / "ai_golf_coaches"
        / "assets"
        / "logo_small.png"
    )
    # Initialize session state for API key input
    if "pending_openai_key" not in st.session_state:
        st.session_state["pending_openai_key"] = ""
    if "show_key_dialog" not in st.session_state:
        st.session_state["show_key_dialog"] = False

    # Sidebar: API key setup (top) + fixed coach
    if "channel" not in st.session_state:
        st.session_state["channel"] = "elitegolfschools"
    with st.sidebar.container():
        # `st.logo()` can be a bit noisy with reruns in some environments.
        # Use a simple sidebar image instead.
        if logo_path.exists():
            st.image(str(logo_path), width="stretch")
        if st.button(
            "📖 Quick Guide", width="stretch", type="secondary", key="avatar_info_btn"
        ):
            st.session_state["show_help"] = True
            st.rerun()
        if st.button("Edit API Key", width="stretch", type="secondary"):
            st.session_state["show_key_dialog"] = True
            st.rerun()
        rag_enabled, _rag_source = _ensure_rag_key()
        st.session_state["rag_enabled"] = rag_enabled

        chat_enabled = bool(st.session_state.get("openai_api_key"))
        chat_verified = st.session_state.get("chat_key_verified") is True
        rag_verified = st.session_state.get("rag_key_verified") is True

        chat_light = "🟢" if chat_verified else "🔴"
        rag_light = "🟢" if rag_verified else "🔴"
        chat_status = (
            "verified"
            if chat_verified
            else ("configured (not verified)" if chat_enabled else "missing")
        )
        rag_status = (
            "verified"
            if rag_verified
            else ("configured (not verified)" if rag_enabled else "missing")
        )
        st.caption(f"{chat_light} Chat completions: {chat_status}")
        st.caption(f"{rag_light} Video recommendations: {rag_status}")

        if st.session_state.get("chat_key_error"):
            st.caption(f"Chat verify error: {st.session_state['chat_key_error']}")
        if st.session_state.get("rag_key_error"):
            st.caption(f"RAG verify error: {st.session_state['rag_key_error']}")

        # Context window usage (top of sidebar) with live-updatable placeholder
        _meter_ph = st.empty()
        _render_context_meter(_meter_ph, st.session_state.get("last_token_usage"))

        # Session cost tracking with live-updatable placeholder
        _cost_ph = st.empty()
        cumulative_cost = st.session_state.get("cumulative_cost", 0.0)
        _render_session_cost(_cost_ph, cumulative_cost)

        # Fixed channel for now
        channel = "elitegolfschools"
        st.session_state["channel"] = channel

        # Always use gpt-5 (model selection disabled)
        st.session_state["selected_model"] = "gpt-5"
        selected_model = "gpt-5"

    # Show API key dialog on first load if missing or when requested
    if not st.session_state.get("openai_api_key"):
        st.session_state["show_key_dialog"] = True
    if st.session_state.get("show_key_dialog"):
        _api_key_dialog()

    # Require API key before proceeding to chat
    if not st.session_state.get("openai_api_key"):
        st.info("Enter your OpenAI API key to start.")
        return

    rag_enabled = bool(st.session_state.get("rag_enabled"))

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict(role, content)]

    pending_prompt: Optional[str] = st.session_state.get("pending_prompt")
    if pending_prompt and not st.session_state.get("awaiting_response"):
        st.session_state.messages.append({"role": "user", "content": pending_prompt})
        st.session_state["awaiting_response"] = True

    # Render history (assistant messages collapsible with summary titles)
    last_assistant_idx = max(
        (
            i
            for i, m in enumerate(st.session_state.messages)
            if m.get("role") == "assistant"
        ),
        default=-1,
    )
    assistant_num = 0
    for i, msg in enumerate(st.session_state.messages):
        user_avatar = _user_avatar_path() if msg["role"] == "user" else None
        # Preserve assistant icon based on the coach used when the message was created
        assistant_channel = msg.get("channel") or channel
        assistant_avatar = _assistant_avatar(assistant_channel)
        avatar = user_avatar if msg["role"] == "user" else assistant_avatar
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant":
                assistant_num += 1
                title = (
                    msg.get("summary")
                    or _one_line_summary(msg["content"])
                    or f"Response {assistant_num}"
                )
                # Append category if available and not already in title
                cat = msg.get("category")
                if cat and f"(Category: {cat})" not in title:
                    title = f"{title} (Category: {cat})"
                with st.expander(title, expanded=(i == last_assistant_idx)):
                    tab_answer, tab_videos = st.tabs(["Answer", "Videos"])
                    with tab_answer:
                        st.markdown(msg["content"])  # nosec - display only
                    with tab_videos:
                        _render_video_recommendations(
                            msg.get("video_recommendations"),
                            message_id=str(msg.get("id") or f"idx_{i}"),
                            rag_enabled=rag_enabled,
                        )
            else:
                st.markdown(msg["content"])  # nosec - display only

    # Process a queued prompt after rendering history so the user sees it first.
    if pending_prompt and st.session_state.get("awaiting_response"):
        assistant_avatar = _assistant_avatar(channel)
        with st.chat_message("assistant", avatar=assistant_avatar):
            with st.spinner("Hang tight while I dig around in Riley's brain..."):
                # Classify question first to avoid duplicate calls
                from ai_golf_coaches.classifier import classify_question_category

                try:
                    pred = classify_question_category(
                        pending_prompt, channel_alias=channel
                    )
                    question_category = pred.category
                except Exception:
                    question_category = None

                try:
                    agent_result = run_agent(
                        channel,
                        pending_prompt,
                        category=question_category,
                        model=selected_model,
                        include_video_recommendations=rag_enabled,
                    )
                except Exception as e:  # display-friendly error
                    agent_result = (
                        f"**Error:** {type(e).__name__}: {e}\n\n"
                        "- Enter your OpenAI API key when prompted.\n"
                        "- If the model is unavailable, update ai_golf_coaches/agent.py to a supported model (e.g., gpt-4o)."
                    )

            reply_text: str
            video_recs: Any = None
            if isinstance(agent_result, AgentResponse):
                reply_text = agent_result.response_text
                video_recs = [
                    vr.model_dump() for vr in agent_result.video_recommendations
                ]
            else:
                reply_text = str(agent_result)

            try:
                title = summarize_for_header(reply_text)
            except Exception:
                title = _one_line_summary(reply_text)

            if question_category:
                title = f"{title} (Category: {question_category})"

            message_id: str = str(uuid.uuid4())
            st.session_state.messages.append(
                {
                    "id": message_id,
                    "role": "assistant",
                    "content": reply_text,
                    "summary": title,
                    "channel": channel,
                    "category": question_category,
                    "video_recommendations": video_recs,
                }
            )

            # After saving, compute and display token usage for this turn
            try:
                channels = load_channels_config(
                    _repo_root() / "config" / "channels.yaml"
                )
                channel_key = resolve_channel_key(channel, channels) or channel
                st.session_state["last_token_usage"] = _compute_turn_token_usage(
                    channel_key, pending_prompt, reply_text
                )
                turn_cost = st.session_state["last_token_usage"].get("total_cost", 0.0)
                st.session_state["cumulative_cost"] = (
                    st.session_state.get("cumulative_cost", 0.0) + turn_cost
                )
                _render_context_meter(
                    _meter_ph, st.session_state.get("last_token_usage")
                )
                _render_session_cost(_cost_ph, st.session_state["cumulative_cost"])
            except Exception:
                pass

        st.session_state.pop("pending_prompt", None)
        st.session_state.pop("awaiting_response", None)
        st.rerun()

    # Input section at bottom - either show transcription editor or normal chat input
    if (
        "pending_transcription" in st.session_state
        and st.session_state["pending_transcription"]
    ):
        # Show editable transcription with Send/Cancel buttons
        st.markdown("---")
        transcribed = st.text_area(
            "Edit transcription if needed:",
            value=st.session_state["pending_transcription"],
            height=100,
            key="transcription_editor",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "✓ Send", width="stretch", type="primary", key="transcription_send_btn"
            ):
                prompt = transcribed.strip()
                st.session_state.pop("pending_transcription", None)
                st.session_state.pop("voice_input", None)
                if prompt:
                    st.session_state["pending_prompt"] = prompt
                    st.session_state["pending_prompt_rendered"] = False
                    st.rerun()
        with col2:
            if st.button(
                "✕ Cancel",
                width="stretch",
                type="secondary",
                key="transcription_cancel_btn",
            ):
                st.session_state.pop("pending_transcription", None)
                st.session_state.pop("voice_input", None)
                st.rerun()
    else:
        # Normal input: chat input + audio button
        col1, col2 = st.columns([20, 1])
        with col1:
            prompt = st.chat_input("Ask your golf question...")
        with col2:
            audio_input = st.audio_input(
                "🎤", label_visibility="collapsed", key="voice_input"
            )

        # Process audio transcription if provided
        if audio_input is not None:
            try:
                from openai import OpenAI

                with st.spinner("🎧 Transcribing..."):
                    client = OpenAI(api_key=st.session_state.get("openai_api_key"))

                    # Save audio bytes to temporary file for Whisper API
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp_file:
                        tmp_file.write(audio_input.getvalue())
                        tmp_file_path = tmp_file.name

                    try:
                        with open(tmp_file_path, "rb") as audio_file:
                            transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                language="en",
                            )

                        transcribed_text = transcript.text.strip()
                        if transcribed_text:
                            # Store transcription for user review
                            st.session_state["pending_transcription"] = transcribed_text
                            st.rerun()
                        else:
                            st.warning("⚠️ No speech detected")
                    finally:
                        # Clean up temporary file
                        import os as _os

                        if _os.path.exists(tmp_file_path):
                            _os.unlink(tmp_file_path)
            except Exception as e:
                st.error(f"⚠️ Transcription failed: {type(e).__name__}: {e}")

    # If a prompt was entered in the chat input, queue it and rerun so the
    # new messages render above the input widget.
    if prompt:
        st.session_state["pending_prompt"] = prompt
        st.session_state["pending_prompt_rendered"] = False
        st.rerun()


if __name__ == "__main__":
    main()
