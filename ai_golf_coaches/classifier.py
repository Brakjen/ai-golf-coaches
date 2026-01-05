from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .config import AppSettings, load_channels_config, resolve_channel_key

DEFAULT_CATEGORIES: List[str] = [
    "full swing",
    "short game",
    "mental game",
    "ball control",
    "putting",
    "dynamic exercises",
]


class CategoryPrediction(BaseModel):
    """Classification result for a golf question.

    Attributes:
        category (str): One of the provided choices (or default categories).
        choices (list[str]): The list of categories considered during classification.

    """

    category: str = Field(..., description="Predicted category label")
    choices: List[str] = Field(default_factory=list, description="Candidate labels")


def _load_channel_categories(channel_alias: Optional[str]) -> Optional[List[str]]:
    """Attempt to load category choices from channels.yaml for a given channel.

    This is best-effort. If the file or structure doesn't provide categories
    for the channel, returns None and the caller should fall back to defaults.

    Expected structures (any of the following):
    - channels[channel_key]["categories"]: list[str]
    - channels[channel_key]["category_labels"]: list[str]

    Args:
        channel_alias (str | None): Alias, handle, or canonical channel key.

    Returns:
        list[str] | None: Category labels if found; otherwise None.

    """
    if not channel_alias:
        return None
    try:
        channels: Dict[str, dict] = load_channels_config("config/channels.yaml")
        key = resolve_channel_key(channel_alias, channels) or channel_alias
        entry = channels.get(key, {})
        for k in ("categories", "category_labels"):
            vals = entry.get(k)
            if isinstance(vals, list) and all(isinstance(v, str) for v in vals):
                # Normalize labels to lower-case for matching
                return [v.strip() for v in vals if v and isinstance(v, str)]
    except Exception:
        return None
    return None


def classify_question_category(
    question: str,
    channel_alias: Optional[str] = None,
    choices: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> CategoryPrediction:
    """Classify a golf question into a single category using OpenAI.

    Uses a fast, low-cost chat model and a strict instruction to return exactly
    one label. If `choices` is not provided, attempts to load categories from
    channels.yaml for the specified channel; otherwise falls back to a default
    set: full swing, short game, mental game, ball control, putting.

    Args:
        question (str): The user's question to classify.
        channel_alias (str | None): Alias/handle/canonical key for the channel to
            optionally source category labels from configuration.
        choices (list[str] | None): Explicit category options to constrain output.
        model (str): OpenAI chat model to use (fast/cheap recommended).
        temperature (float): Sampling temperature for the classification.

    Returns:
        CategoryPrediction: Predicted category and the choices considered.

    Raises:
        RuntimeError: If OpenAI is not configured.

    """
    # Resolve category choices
    labels = [c.strip() for c in (choices or []) if c and isinstance(c, str)]
    if not labels:
        loaded = _load_channel_categories(channel_alias)
        labels = loaded if loaded else DEFAULT_CATEGORIES

    # Instruction: require exact label match from the provided options
    label_list = ", ".join(f"'{c}'" for c in labels)
    system = (
        "You are a strict single-label classifier for golf questions.\n"
        "Return ONLY one label from the provided options.\n"
        "Output the label string with no punctuation, no quotes, no explanation.\n"
        "If ambiguous, ALWAYS select full swing."
    )
    user = (
        "Categories: "
        + label_list
        + "\n\nQuestion:\n"
        + question.strip()
        + "\n\nAnswer with exactly one label from the list."
    )

    settings = AppSettings()
    if not settings.openai or not settings.openai.api_key:
        raise RuntimeError("OPENAI__API_KEY not configured.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenAI client not available; install 'openai'.") from e

    client = OpenAI(api_key=settings.openai.api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_completion_tokens=16,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Normalize and validate against choices
    normalized = raw.strip().strip("\"'").lower()
    # Match ignoring case; prefer exact match otherwise fallback by simple heuristic
    match = None
    for c in labels:
        if normalized == c.lower():
            match = c
            break
    if not match:
        # Simple heuristic: pick label with maximum token overlap
        import re

        toks = set(re.findall(r"[a-zA-Z]+", normalized))
        if not toks:
            match = labels[0]
        else:
            best = (0, labels[0])
            for c in labels:
                ctoks = set(re.findall(r"[a-zA-Z]+", c.lower()))
                score = len(toks & ctoks)
                if score > best[0]:
                    best = (score, c)
            match = best[1]

    return CategoryPrediction(category=match, choices=labels)
