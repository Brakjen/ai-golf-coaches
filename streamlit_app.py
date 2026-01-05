from __future__ import annotations

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

    # Sidebar: channel selector
    channel_keys = _load_channel_keys()
    if "channel" not in st.session_state:
        st.session_state["channel"] = (
            "elitegolfschools"
            if "elitegolfschools" in channel_keys
            else channel_keys[0]
        )
    with st.sidebar:
        st.header("Coach")
        assets_dir = Path(__file__).resolve().parent / "ai_golf_coaches" / "assets"
        riley_icon = assets_dir / "icon_riley.png"
        milo_icon = assets_dir / "icon_milo.png"

        col1, col2 = st.columns(2)
        with col1:
            if riley_icon.exists():
                st.image(str(riley_icon), caption="Riley (EGS)", width=96)
            if st.button("Select Riley", use_container_width=True):
                st.session_state["channel"] = "elitegolfschools"
        with col2:
            if milo_icon.exists():
                st.image(str(milo_icon), caption="Milo (MLG)", width=96)
            if st.button("Select Milo", use_container_width=True):
                st.session_state["channel"] = "milolinesgolf"

        # Show current selection with icon
        st.divider()
        st.subheader("Selected Coach")
        sel = st.session_state["channel"]
        if sel == "elitegolfschools" and riley_icon.exists():
            st.image(str(riley_icon), width=64)
            st.caption("Elite Golf Schools — Riley Andrews")
        elif sel == "milolinesgolf" and milo_icon.exists():
            st.image(str(milo_icon), width=64)
            st.caption("Milo Lines Golf — Milo Lines")
        else:
            st.markdown("🏌️")
        st.markdown("Environment variable required: `OPENAI__API_KEY`")

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
        assistant_avatar = _assistant_avatar(channel)
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
                reply = f"**Error:** {type(e).__name__}: {e}\n\n- Ensure `OPENAI__API_KEY` is set.\n- If the model is unavailable, update `ai_golf_coaches/agent.py` to a supported model (e.g., `gpt-4o`)."
            try:
                title = summarize_for_header(reply)
            except Exception:
                title = _one_line_summary(reply)
            with st.expander(title, expanded=True):
                st.markdown(reply)
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "summary": title}
        )


if __name__ == "__main__":
    main()
