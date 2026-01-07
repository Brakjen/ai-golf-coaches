from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

import streamlit as st

from ai_golf_coaches.agent import (
    AgentResponse,
    _build_messages,  # reuse to mirror agent prompt construction
    _clip_text_to_tokens,  # reuse to mirror agent context clipping
    _format_video_recommendations,
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


def _save_avatar_image(img_bytes: bytes) -> str:
    """Persist an uploaded/captured image to a local path for avatar usage.

    Args:
        img_bytes (bytes): Raw image bytes.

    Returns:
        str: Absolute file path to the saved image.

    """
    cache_dir = _repo_root() / ".streamlit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "user_avatar.png"
    out_path.write_bytes(img_bytes)
    return str(out_path)


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
        st.header("Context Window")
        if usage and isinstance(usage, dict):
            pct = int(round(float(usage.get("percent_128k", 0.0))))
            st.progress(pct, text=f"{pct}% of 128k window")
            st.caption(
                f"~{int(usage.get('total_tokens', 0))} tokens used (incl. output)"
            )
            # Show cost breakdown
            turn_cost = float(usage.get("total_cost", 0.0))
            st.caption(f"Last turn: ${turn_cost:.4f}")
        else:
            st.progress(0, text="0% of 128k window")
            st.caption("Ask a question to estimate usage.")


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
        st.divider()
        st.metric("Session Cost", f"${cumulative_cost:.4f}")
        if cumulative_cost > 0:
            st.caption("Total spend this session")


def _show_welcome_screen() -> bool:
    """Display a welcome screen with logo, instructions, and data info.

    Returns True if user clicked "I am ready to start", False otherwise.
    Uses session state to track if it should be shown.

    Returns:
        bool: True if dismissed and should show main app.

    """
    if "welcome_dismissed" not in st.session_state:
        st.session_state["welcome_dismissed"] = False

    # Check if user wants to reopen via help button
    show_welcome = not st.session_state["welcome_dismissed"] or st.session_state.get(
        "show_help", False
    )

    if not show_welcome:
        return True

    # Find logo path
    logo_path = (
        Path(__file__).resolve().parent / "ai_golf_coaches" / "assets" / "logo.png"
    )

    # Display logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            st.markdown(
                "<h1 style='text-align: center;'>🏌️ AI Golf Coach</h1>",
                unsafe_allow_html=True,
            )

    # Video demonstration
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

    st.markdown("---")

    # Tabbed content
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📖 Instructions", "🎥 Data Sources", "📚 Video Library", "ℹ️ About"]
    )

    with tab1:
        st.markdown("""
        ## How to Use This App

        ### Getting Started
        1. **Add your OpenAI API key** in the sidebar (required)
        2. **Select a coach** (Riley from Elite Golf Schools or Milo Lines)
        3. **Ask your golf question** in the chat input at the bottom
        4. **Get personalized advice** based on the coach's teaching style

        ### Features
        - **Category Detection**: Your question is automatically classified into one of five categories:
          - Full Swing
          - Short Game
          - Putting
          - Ball Control
          - Mental Game
          - Dynamic Exercises
        - **Context Window**: Track how much of the 128k token context is being used
        - **Cost Tracking**: Monitor per-turn and session costs in real-time
        - **Response History**: All responses are saved and collapsible for easy review

        ### Tips
        - Be specific in your questions for better answers
        - Questions are analyzed to load relevant context from that category
        - Your API key is only stored for this session and never saved
        """)

    with tab2:
        st.markdown("""
        ## Data Sources

        This AI coach learns from real YouTube instructional videos. We've curated content from:

        ### Elite Golf Schools (@elitegolfschools)
        - **Instructor**: Riley Andrews and team
        - **Focus**: Technical fundamentals, swing mechanics, healthy movement, comprehensive instruction
        - **Videos**: 40+ long-form instructional videos across all categories

        ### Milo Lines Golf (@milolinesgolf)
        - **Instructor**: Milo Lines
        - **Focus**: Modern coaching methods, feel-based instruction
        - **Videos**: Curated selection of instructional content

        ### How It Works
        1. We fetch transcripts from select YouTube videos using the YouTube Data API
        2. We have selected subsets of videos per category to provide relevant context for each question type
        3. Your question is classified on-the-fly to select the most relevant context
        4. The AI responds using the instructor's teaching style and knowledge

        ### What We Don't Include
        - Short-form videos (Shorts, defined as videos shorted than 3 minutes)
        - Live Q&A sessions (for now - more to come)
        - Podcast episodes
        """)

    with tab3:
        st.markdown("## Video Library")
        st.caption(
            "Browse the curated videos used for each context category. "
            "These are the actual videos the AI learns from."
        )

        # Channel selector (only show enabled channels)
        cfg_path = Path(__file__).resolve().parent / "config" / "channels.yaml"
        channels_cfg = load_channels_config(cfg_path)
        enabled_channels = [
            k for k, v in channels_cfg.items() if v.get("enabled", True)
        ]

        if not enabled_channels:
            st.warning("No channels are currently enabled.")
            return True

        lib_channel = st.radio(
            "Select Channel",
            options=enabled_channels,
            format_func=lambda k: "Elite Golf Schools (Riley)"
            if k == "elitegolfschools"
            else "Milo Lines Golf",
            horizontal=True,
            key="lib_channel_selector",
        )

        # Load channel config to get video IDs per category
        try:
            channel_entry = channels_cfg.get(lib_channel, {})
            context_videos = channel_entry.get("constant_context_videos", {})

            if isinstance(context_videos, dict):
                # Load catalog to get video titles
                try:
                    from ai_golf_coaches.config import load_channel_catalog

                    catalog = load_channel_catalog(lib_channel)
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
                    st.markdown("---")
            else:
                st.info("No categorized videos found for this channel.")

        except Exception as e:
            st.error(f"Failed to load video library: {type(e).__name__}: {e}")

    with tab4:
        st.markdown("""
        ## About This App

        ### Technical Details
        - **Model**: GPT-5 (128k context window)
        - **Classification**: GPT-4o-mini for fast question categorization
        - **Context**: Static context loaded per category (full swing loads the most videos into context and is therefore the most expensive)
        - **RAG**: Not yet implemented (pure static context for now)

        ### Architecture
        - Static context prepared from pre-selected videos
        - Category-specific video subsets
        - Per-channel instructions to capture teaching style
        - Token counting with tiktoken for transparency

        ### Privacy
        - API keys stored only in session memory
        - No conversation history is saved to disk
        - All processing happens via OpenAI's API

        ### Version
        - **v0.0.1** - Proof of Concept
        - Static context demo, no RAG
        """)

    st.markdown("---")

    # Dismiss button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(
            "I am ready to start ⛳", use_container_width=True, type="primary"
        ):
            st.session_state["welcome_dismissed"] = True
            st.session_state["show_help"] = False
            st.rerun()

    return False


@st.dialog("Capture Avatar Photo")
def _avatar_capture_dialog() -> None:
    """Dialog for capturing user avatar photo."""
    st.write("Use your camera to capture a custom avatar image.")
    captured = st.camera_input("Capture photo", label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save", use_container_width=True, type="primary"):
            if captured is not None:
                try:
                    path = _save_avatar_image(captured.getvalue())
                    st.session_state["user_avatar_path"] = path
                    st.success("Avatar saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
            else:
                st.warning("Capture a photo first")
    with col2:
        if st.button("Cancel", use_container_width=True, type="secondary"):
            st.rerun()


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

    st.title("AI Golf Coach — Simple PoC")
    st.caption("Static-context agent demo. No RAG yet.")

    logo_path = (
        Path(__file__).resolve().parent
        / "ai_golf_coaches"
        / "assets"
        / "logo_small.png"
    )
    # `st.logo()` can be a bit noisy with reruns in some environments.
    # Use a simple sidebar image instead.
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)

    # Initialize session state for API key input
    if "pending_openai_key" not in st.session_state:
        st.session_state["pending_openai_key"] = ""

    # Sidebar: API key setup (top) + compact coach selector
    channel_keys = _load_channel_keys()
    if "channel" not in st.session_state:
        st.session_state["channel"] = (
            "elitegolfschools"
            if "elitegolfschools" in channel_keys
            else channel_keys[0]
        )
    with st.sidebar:
        # OpenAI API key
        def clear_api_key() -> None:
            """Callback to clear API key input and session state."""  # noqa: D401
            st.session_state["pending_openai_key"] = ""
            st.session_state.pop("openai_api_key", None)
            os.environ.pop("OPENAI__API_KEY", None)

        api_col1, api_col2 = st.columns([3, 1])
        with api_col1:
            api_val = st.text_input(
                "OpenAI API Key",
                type="password",
                key="pending_openai_key",
                label_visibility="collapsed",
                placeholder="Enter API key...",
            )
        with api_col2:
            set_clicked = st.button("Set", use_container_width=True)
        if st.button(
            "Clear", use_container_width=True, on_click=clear_api_key, type="secondary"
        ):
            pass

        if set_clicked and api_val:
            st.session_state["openai_api_key"] = api_val.strip()
            os.environ["OPENAI__API_KEY"] = st.session_state["openai_api_key"]
            st.success("✓ Key set")

        st.caption(
            "✅ Ready"
            if st.session_state.get("openai_api_key")
            else "❗ API key required"
        )

        # Context window usage (top of sidebar) with live-updatable placeholder
        _meter_ph = st.empty()
        _render_context_meter(_meter_ph, st.session_state.get("last_token_usage"))

        # Session cost tracking with live-updatable placeholder
        _cost_ph = st.empty()
        cumulative_cost = st.session_state.get("cumulative_cost", 0.0)
        _render_session_cost(_cost_ph, cumulative_cost)

        st.divider()
        # Coach selector
        current = st.session_state.get("channel", "elitegolfschools")

        if not channel_keys:
            st.warning("No channels enabled.")
        else:
            try:
                current_idx = (
                    channel_keys.index(current) if current in channel_keys else 0
                )
            except ValueError:
                current_idx = 0

            choice = st.radio(
                "Coach",
                options=channel_keys,
                index=current_idx,
                format_func=lambda k: "Riley (EGS)"
                if k == "elitegolfschools"
                else "Milo (MLG)",
                horizontal=True,
                label_visibility="collapsed",
            )
            if choice != current:
                st.session_state["channel"] = choice

        # Assign selected channel for use below
        channel = st.session_state["channel"]

        # Always use gpt-5 (model selection disabled)
        st.session_state["selected_model"] = "gpt-5"
        selected_model = "gpt-5"

        st.divider()
        # Avatar controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📸 Photo", use_container_width=True, help="Capture avatar"):
                _avatar_capture_dialog()
        with col2:
            if st.button("Reset", use_container_width=True, type="secondary"):
                st.session_state.pop("user_avatar_path", None)
        with col3:
            if st.button("📖 Info", use_container_width=True, type="secondary"):
                st.session_state["show_help"] = True
                st.rerun()

    # Require API key before proceeding to chat
    if not st.session_state.get("openai_api_key"):
        st.info("Add your OpenAI API key in the sidebar to start.")
        return

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict(role, content)]

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
        user_custom = st.session_state.get("user_avatar_path")
        user_avatar = user_custom if (msg["role"] == "user" and user_custom) else "👤"
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
                    st.markdown(msg["content"])  # nosec - display only
            else:
                st.markdown(msg["content"])  # nosec - display only

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
            if st.button("✓ Send", use_container_width=True, type="primary"):
                prompt = transcribed.strip()
                st.session_state.pop("pending_transcription", None)
                st.session_state.pop("voice_input", None)
                # Will process below
            else:
                prompt = None
        with col2:
            if st.button("✕ Cancel", use_container_width=True, type="secondary"):
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

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        user_custom = st.session_state.get("user_avatar_path")
        user_avatar = user_custom if user_custom else "👤"
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)

        # Classify question first to avoid duplicate calls
        from ai_golf_coaches.classifier import classify_question_category

        try:
            pred = classify_question_category(prompt, channel_alias=channel)
            question_category = pred.category
        except Exception:
            question_category = None

        # Call agent with pre-classified category and selected model
        with st.chat_message("assistant", avatar=_assistant_avatar(channel)):
            try:
                agent_result = run_agent(
                    channel, prompt, category=question_category, model=selected_model
                )
            except Exception as e:  # display-friendly error
                agent_result = (
                    f"**Error:** {type(e).__name__}: {e}\n\n"
                    "- Paste your OpenAI API key in the sidebar.\n"
                    "- If the model is unavailable, update ai_golf_coaches/agent.py to a supported model (e.g., gpt-4o)."
                )

            reply_text: str
            if isinstance(agent_result, AgentResponse):
                reply_text = agent_result.response_text
                if agent_result.video_recommendations:
                    reply_text = (
                        reply_text
                        + "\n\n"
                        + _format_video_recommendations(
                            agent_result.video_recommendations
                        )
                    )
            else:
                reply_text = str(agent_result)
            try:
                title = summarize_for_header(reply_text)
            except Exception:
                title = _one_line_summary(reply_text)
            # Include category in header if available
            if question_category:
                title = f"{title} (Category: {question_category})"
            with st.expander(title, expanded=True):
                st.markdown(reply_text)

        # Save assistant message to history first
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply_text,
                "summary": title,
                "channel": channel,
                "category": question_category,
            }
        )

        # After saving, compute and display token usage for this turn without rerun
        try:
            # Resolve canonical channel key for accurate config lookup
            channels = load_channels_config(_repo_root() / "config" / "channels.yaml")
            channel_key = resolve_channel_key(channel, channels) or channel
            st.session_state["last_token_usage"] = _compute_turn_token_usage(
                channel_key, prompt, reply_text
            )
            # Accumulate session cost
            turn_cost = st.session_state["last_token_usage"].get("total_cost", 0.0)
            st.session_state["cumulative_cost"] = (
                st.session_state.get("cumulative_cost", 0.0) + turn_cost
            )
            # Update both context meter and session cost displays
            _render_context_meter(_meter_ph, st.session_state.get("last_token_usage"))
            _render_session_cost(_cost_ph, st.session_state["cumulative_cost"])
        except Exception:
            # Best-effort; silently skip if counting fails
            pass


if __name__ == "__main__":
    main()
