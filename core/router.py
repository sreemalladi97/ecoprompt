"""
EcoPrompt - core/router.py
3-tier routing matrix for Groq models (all free tier)
"""

# Lets `str | None` / `tuple[str, str]` type hints below work on Python 3.9
# (this project's venv), which doesn't support `|` union syntax natively.
from __future__ import annotations

import logging

logger = logging.getLogger("ecoprompt")

# ── Model tiers ───────────────────────────────────────────────────────────────
# Each tier is an ordered list: [primary, fallback, ...]. The primary is tried
# first; fallbacks only kick in if the primary errors out (e.g. rate-limited
# or decommissioned by Groq — this already happened once, see check_models.py).
# All models below are free-tier on Groq's developer plan as of Jul 2026.
TIER_MODELS = {
    "simple": [
        "openai/gpt-oss-20b",   # fast, cheap — replaces llama-3.1-8b-instant
        "groq/compound-mini",   # fallback: lightweight general-purpose system
    ],
    "medium": [
        "qwen/qwen3.6-27b",             # balanced reasoning
        "qwen/qwen3-32b",               # fallback: same family, slightly larger
    ],
    "complex": [
        "openai/gpt-oss-120b",          # heavy lifting (code, math, long context)
        "meta-llama/llama-4-scout-17b-16e-instruct",  # fallback: different family
    ],
}

# ── Keyword signals ───────────────────────────────────────────────────────────
COMPLEX_SIGNALS = [
    "def ", "class ", "function(", "```", "SELECT ", "FROM ",  # code
    "debug", "refactor", "implement", "system architecture", "design the architecture",  # engineering
    "analyze", "compare", "evaluate", "critique",               # deep reasoning
    "step by step", "explain in detail", "write a",             # long-form
]

MEDIUM_SIGNALS = [
    "explain", "summarize", "how does",
    "describe in detail", "outline", "pros and cons",
    "difference between", "why does", "when should",
    "compare", "give me an overview",
]

# ── Classifier ────────────────────────────────────────────────────────────────
def _first_match(signals: list, haystack: str) -> str | None:
    """Returns the first signal string found in haystack, or None."""
    return next((sig for sig in signals if sig in haystack), None)


def classify_with_reason(messages: list) -> tuple[str, str]:
    """
    Returns (tier, reason). Tier is 'simple', 'medium', or 'complex'.
    Reason is a short human-readable explanation of which signal decided
    it — surfaced in the tester UI so a mis-route is visible and
    debuggable instead of a black box. Checks the last user message
    (most relevant signal).
    """
    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break

    word_count = len(user_text.split())
    lower = user_text.lower()

    # Length alone can force a tier upgrade
    if word_count > 300:
        return "complex", f"length > 300 words ({word_count})"

    if word_count > 80:
        # Still check signals before committing to medium
        matched = _first_match(COMPLEX_SIGNALS, user_text)
        if matched:
            return "complex", f"length > 80 words ({word_count}) + complex keyword '{matched.strip()}'"
        return "medium", f"length > 80 words ({word_count}), no complex keyword"

    # Short prompts: check signals
    matched = _first_match(COMPLEX_SIGNALS, user_text)
    if matched:
        return "complex", f"complex keyword '{matched.strip()}'"

    matched = _first_match(MEDIUM_SIGNALS, lower)
    if matched:
        return "medium", f"medium keyword '{matched.strip()}'"

    return "simple", "no signals matched (default tier)"


def classify(messages: list) -> str:
    """
    Returns just the tier ('simple', 'medium', or 'complex').
    Kept for backward compatibility — use classify_with_reason() if you
    also want to know which signal decided it.
    """
    tier, _ = classify_with_reason(messages)
    return tier


# ── Public entry point ────────────────────────────────────────────────────────
def get_candidates(tier: str, requested_model: str) -> list:
    """
    Returns the ordered list of models to try for a given tier.
    First entry is the primary; the rest are automatic fallbacks used only
    if the primary request fails.
    """
    return list(TIER_MODELS.get(tier, [requested_model]))


def route(messages: list, requested_model: str) -> str:
    """
    Returns the primary model name to try first.
    Kept for backward compatibility — callers that need the full fallback
    chain should use get_candidates() instead.
    Falls back to the requested model if classification fails.
    """
    try:
        tier = classify(messages)
        candidates = get_candidates(tier, requested_model)
        chosen = candidates[0]
        logger.info(f"Router: {requested_model} → {chosen} (tier={tier}, {len(candidates)} candidate(s))")
        return chosen
    except Exception as e:
        logger.warning(f"Router fallback to requested model: {e}")
        return requested_model
