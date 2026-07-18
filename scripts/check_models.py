#!/usr/bin/env python3
"""
EcoPrompt - scripts/check_models.py

Compares the models configured in core/router.py against Groq's live
model catalog. Flags:
  - CONFIGURED models that Groq no longer serves (these will silently
    fall through to the next fallback, or fail outright if they were
    the last candidate in a tier) — fix these immediately.
  - NEW models available on Groq that aren't in our config yet — these
    are for manual review only. We do NOT auto-add them: model quality
    and routing-signal fit need to be checked by hand before they're
    trusted with real traffic (see the "Routing signals need
    calibration" lesson in project notes).

Usage:
    python scripts/check_models.py

Exit code is 1 if any configured model is missing from Groq's catalog
(useful for wiring this into a scheduled/cron check), 0 otherwise.

Requires GROQ_API_KEY in the environment or a .env file at the project root.
"""

import os
import sys

import httpx

# Make the project root importable so we can read the live TIER_MODELS
# config directly, instead of hardcoding a second copy of it here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.router import TIER_MODELS  # noqa: E402

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"


def fetch_live_models(api_key: str) -> set:
    """Returns the set of model IDs Groq currently serves."""
    resp = httpx.get(
        GROQ_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return {m["id"] for m in data.get("data", [])}


def configured_models() -> dict:
    """Returns {model_id: [tiers it's used in]} from TIER_MODELS."""
    seen = {}
    for tier, models in TIER_MODELS.items():
        for model in models:
            seen.setdefault(model, []).append(tier)
    return seen


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set (check your .env file).")
        sys.exit(2)

    try:
        live_models = fetch_live_models(api_key)
    except httpx.HTTPError as e:
        print(f"ERROR: couldn't reach Groq's /models endpoint: {e}")
        sys.exit(2)

    configured = configured_models()
    missing = {m: tiers for m, tiers in configured.items() if m not in live_models}
    new_available = sorted(live_models - set(configured.keys()))

    print(f"Checked {len(configured)} configured model(s) against "
          f"{len(live_models)} live Groq model(s).\n")

    if missing:
        print("MISSING — configured but no longer served by Groq (fix now):")
        for model, tiers in missing.items():
            print(f"  - {model}  (used in: {', '.join(tiers)})")
        print()
    else:
        print("All configured models are still live. Nothing broken.\n")

    if new_available:
        print("NEW — available on Groq but not in our config (review before adding):")
        for model in new_available:
            print(f"  - {model}")
        print("\n  Don't auto-add these. Test quality/format against each tier's")
        print("  use case first, same as we did for the current lineup.\n")
    else:
        print("No new models beyond what's already configured or missing.\n")

    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
