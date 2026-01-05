from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

import streamlit as st

from ai_golf_coaches.agent import run_agent, summarize_for_header
from ai_golf_coaches.config import load_channels_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_channel_keys() -> List[str]:
    cfg_path = _repo_root() / "config" / "channels.yaml"
    channels: Dict[str, dict] = load_channels_config(cfg_path)
    return sorted(channels.keys())


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


def main() -> None:
    """Generate the Streamlit app UI and handle interactions.

    Returns:
        None

    """
    st.set_page_config(page_title="AI Golf Coach (PoC)", page_icon="🏌️", layout="wide")
    st.title("AI Golf Coach — Simple PoC")
    st.caption("Static-context agent demo. No RAG yet.")

    # Sidebar: API key setup (top) + compact coach selector
    channel_keys = _load_channel_keys()
    if "channel" not in st.session_state:
        st.session_state["channel"] = (
            "elitegolfschools"
            if "elitegolfschools" in channel_keys
            else channel_keys[0]
        )
    with st.sidebar:
        # OpenAI key first for clarity
        st.header("OpenAI Access")
        st.caption("Provide your own API key. Stored only in this session.")
        api_col1, api_col2 = st.columns([3, 1])
        with api_col1:
            api_val = st.text_input(
                "OpenAI API Key",
                type="password",
                key="pending_openai_key",
                help="Your key is used client-side in this session only.",
            )
        with api_col2:
            set_clicked = st.button("Use", use_container_width=True)
        clear_clicked = st.button("Clear Key", use_container_width=True)

        if set_clicked and api_val:
            st.session_state["openai_api_key"] = api_val.strip()
            # Set env var so AppSettings picks it up
            os.environ["OPENAI__API_KEY"] = st.session_state["openai_api_key"]
            st.success("API key set for this session.")
        if clear_clicked:
            st.session_state.pop("openai_api_key", None)
            os.environ.pop("OPENAI__API_KEY", None)
            st.info("API key cleared.")

        if st.session_state.get("openai_api_key"):
            st.caption("✅ API key is set for this session.")
        else:
            st.caption("❗ Required to ask questions.")

        st.divider()
        st.header("Coach")
        # Compact selector without additional icons/sections
        current = st.session_state.get("channel", "elitegolfschools")
        choice = st.radio(
            "Select Coach",
            options=["elitegolfschools", "milolinesgolf"],
            index=0 if current == "elitegolfschools" else 1,
            format_func=lambda k: "Riley (EGS)"
            if k == "elitegolfschools"
            else "Milo (MLG)",
            horizontal=True,
        )
        if choice != current:
            st.session_state["channel"] = choice

        # Assign selected channel for use below
        channel = st.session_state["channel"]

        st.divider()
        st.header("User Avatar")
        st.caption("Capture a custom user icon.")
        captured = st.camera_input("Capture from camera")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use default"):
                st.session_state.pop("user_avatar_path", None)
        with col2:
            if st.button("Clear custom avatar"):
                st.session_state.pop("user_avatar_path", None)

        # Persist captured input if provided
        if captured is not None:
            try:
                path = _save_avatar_image(captured.getvalue())
                st.session_state["user_avatar_path"] = path
                st.success("Custom user avatar set.")
            except Exception as e:
                st.error(f"Failed to set avatar: {type(e).__name__}: {e}")

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
                with st.expander(title, expanded=(i == last_assistant_idx)):
                    st.markdown(msg["content"])  # nosec - display only
            else:
                st.markdown(msg["content"])  # nosec - display only

    # Chat input
    prompt = st.chat_input("Ask your golf question...")
    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        user_custom = st.session_state.get("user_avatar_path")
        user_avatar = user_custom if user_custom else "👤"
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)

        # Call agent
        with st.chat_message("assistant", avatar=_assistant_avatar(channel)):
            try:
                reply = run_agent(channel, prompt)
            except Exception as e:  # display-friendly error
                reply = (
                    f"**Error:** {type(e).__name__}: {e}\n\n"
                    "- Paste your OpenAI API key in the sidebar.\n"
                    "- If the model is unavailable, update ai_golf_coaches/agent.py to a supported model (e.g., gpt-4o)."
                )
            try:
                title = summarize_for_header(reply)
            except Exception:
                title = _one_line_summary(reply)
            with st.expander(title, expanded=True):
                st.markdown(reply)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "summary": title,
                "channel": channel,
            }
        )


if __name__ == "__main__":
    main()
