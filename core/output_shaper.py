"""
EcoPrompt - Output Token Shaping
Everything else in this proxy shrinks what gets SENT (compression,
caching, routing). This is headroom's "output shaper" idea applied to
what comes BACK: nudge the model toward terser answers and dial down
reasoning effort, but only on requests already classified "simple" by
the router — those are routine asks that don't need deliberation, so
there's a real signal to act on rather than guessing.
"""

TERSE_SYSTEM_NOTE = (
    "Be concise. Answer directly - no preamble, no restating the "
    "question, no unsolicited caveats or summaries. Skip filler."
)

# Only requests the router already classified as routine get shaped.
# Complex-tier requests are left untouched — a caller routed to the
# powerful model wants full deliberation, not a truncated answer.
SHAPED_TIERS = {"simple"}


def is_shaped(route_tier: str) -> bool:
    return route_tier in SHAPED_TIERS


def inject_terse_note(messages: list) -> list:
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + TERSE_SYSTEM_NOTE
    else:
        messages.insert(0, {"role": "system", "content": TERSE_SYSTEM_NOTE})
    return messages


def dial_reasoning_effort(body: dict, model: str) -> None:
    """
    Groq's gpt-oss models accept a reasoning_effort knob (low/medium/high).
    On a routine request there's no reason to pay for high-effort chain-of-
    thought, so mutate body in place to dial it down. No-op for models that
    don't support the parameter.
    """
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
