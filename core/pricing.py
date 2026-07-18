"""
EcoPrompt - Model Pricing
Real per-model USD rates (per 1M tokens), confirmed directly against
Groq's published pricing page (groq.com/pricing) rather than guessed.

Only models with a rate Groq actually publishes are listed. Everything
else — groq/compound-mini (a compound AI system whose cost passes
through to whichever underlying models/tools it invokes at runtime, not
a fixed per-token rate) and any preview/unlisted model — returns None
from get_price() rather than a made-up number, so a missing rate shows
up as "unknown" instead of silently wrong.

These are Groq's on-demand (pay-as-you-go) rates. If your traffic is
actually running on Groq's free developer tier (as this project's
routed models are, per core/router.py), real billed cost is $0 — the
numbers here are only useful as an illustrative "what this would have
cost on-demand" estimate, not an actual invoice.

Last verified against groq.com/pricing: 2026-07. Re-verify periodically
the same way scripts/check_models.py catches decommissioned models —
Groq's rates and model lineup both change.
"""

# model -> (input_price_per_million_tokens, output_price_per_million_tokens), in USD
PRICING_PER_MILLION_TOKENS = {
    "openai/gpt-oss-20b":  (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "qwen/qwen3.6-27b":    (0.60, 3.00),
}


def get_price(model: str):
    """
    Returns (input_price, output_price) per 1M tokens for `model`, or
    None if Groq hasn't published a fixed per-token rate for it.
    """
    return PRICING_PER_MILLION_TOKENS.get(model)


def estimate_input_cost_usd(model: str, tokens: int):
    """
    USD cost of `tokens` input tokens on `model` at Groq's on-demand
    rate, or None if the model's price is unknown. Used to value
    compression savings, which only ever reduce input tokens.
    """
    price = get_price(model)
    if price is None:
        return None
    input_price, _ = price
    return (tokens / 1_000_000) * input_price
