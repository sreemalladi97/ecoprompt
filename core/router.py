"""
EcoPrompt - core/router.py
3-tier routing matrix for Groq models (all free tier)
"""

import logging

logger = logging.getLogger("ecoprompt")

# ── Model tiers ───────────────────────────────────────────────────────────────
TIER_MODELS = {
    "simple":  "openai/gpt-oss-20b",    # fast, cheap — replaces llama-3.1-8b-instant
    "medium":  "qwen/qwen3.6-27b",      # balanced reasoning
    "complex": "openai/gpt-oss-120b",   # heavy lifting (code, math, long context)
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
def classify(messages: list) -> str:
    """
    Returns 'simple', 'medium', or 'complex' based on prompt content and length.
    Checks the last user message (most relevant signal).
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
        return "complex"
    if word_count > 80:
        # Still check signals before committing to medium
        if any(sig in user_text for sig in COMPLEX_SIGNALS):
            return "complex"
        return "medium"

    # Short prompts: check signals
    if any(sig in user_text for sig in COMPLEX_SIGNALS):
        return "complex"
    if any(sig in lower for sig in MEDIUM_SIGNALS):
        return "medium"

    return "simple"


# ── Public entry point ────────────────────────────────────────────────────────
def route(messages: list, requested_model: str) -> str:
    """
    Returns the model name to actually send to Groq.
    Falls back to the requested model if classification fails.
    """
    try:
        tier = classify(messages)
        chosen = TIER_MODELS[tier]
        logger.info(f"Router: {requested_model} → {chosen} (tier={tier})")
        return chosen
    except Exception as e:
        logger.warning(f"Router fallback to requested model: {e}")
        return requested_model
