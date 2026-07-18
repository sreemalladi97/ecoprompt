# EcoPrompt
test 
> Smart middleware proxy that slashes AI token usage and cloud bills.

## What it does

```
Your App → EcoPrompt Proxy → AI Model (Groq, free tier)
                ↓
         [Token Compressor]   strip fluff before sending (originals recoverable via /v1/retrieve/{id})
         [Semantic Cache]     answer repeats for free (retrieve-and-rerank + cross-encoder validation)
         [Routing Matrix]     3-tier model routing, each tier with an automatic fallback model
```

## Live deployment

A shareable, always-on version is deployed at **https://ecoprompt-nu.vercel.app**:

- [Dashboard](https://ecoprompt-nu.vercel.app/dashboard) — usage stats, routing breakdown
- [Tester](https://ecoprompt-nu.vercel.app/test) — send prompts, see how they're routed, bring your own free [Groq API key](https://console.groq.com/keys)

New to the project? See **[GETTING_STARTED.md](./GETTING_STARTED.md)** for a full walkthrough.

**Note:** semantic caching and token compression are intentionally inactive on the Vercel deployment (those dependencies are too heavy for a serverless cold start). Routing works fully there. Run locally for the complete feature set.

## Status

- [x] **Step 1** — Proxy shell (passthrough + observability)
- [x] **Step 2** — Token compressor (LLMLingua)
- [x] **Step 3** — Semantic cache (embeddings + Qdrant), upgraded with retrieve-and-rerank and cross-encoder validation
- [x] **Step 4** — Routing matrix (complexity classifier), with per-tier fallback models and a `reason` explaining every routing decision
- [x] **Step 5** — Dashboard UI
- [x] **Step 6** — Request tester UI (`/test`) with routing transparency and markdown rendering
- [x] **Step 7** — Live deployment on Vercel
- [x] **Step 8** — Reversible compression (`/v1/retrieve/{id}`) so a compressed request's original text is recoverable within a TTL, and `/stats` now reads from the persisted SQLite log so it survives restarts

## Quick Start

```bash
# 1. Install deps (base + the ML packages needed for caching/compression)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-local.txt

# 2. Configure
echo "GROQ_API_KEY=your_key_here" > .env
# Get a free key at console.groq.com/keys — no credit card required

# 3. Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Point your app at the proxy instead of Groq directly
# Change: https://api.groq.com/openai/v1/chat/completions
# To:     http://localhost:8000/v1/chat/completions
```

Each caller supplies their own Groq API key in the request's `Authorization` header, same as calling Groq directly, the proxy doesn't require a server-side key to run. Locally, if a request omits the header, the proxy falls back to `GROQ_API_KEY` from `.env` (handy so the [tester](#endpoints) doesn't need the key re-pasted every time) — this fallback is local-dev-only: the public Vercel deployment has no `.env`, so every caller there still brings their own key and nobody shares quota.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check + current routing tier config |
| `GET /stats` | Token usage, cache hits, routing breakdown, fallback usage (persisted across restarts) |
| `GET /dashboard` | Live dashboard UI |
| `GET /test` | Browser-based request tester |
| `POST /v1/chat/completions` | Drop-in OpenAI-compatible proxy |
| `GET /v1/retrieve/{compression_id}` | Recover the pre-compression original for a request, within the TTL |

## Reversible compression

Compression is lossy, so whenever it actually saves tokens, the pre-compression messages are kept for a short TTL (default 1 hour, configurable via `ECOPROMPT_CCR_TTL_SECONDS`) under the request's id. The response header `x-ecoprompt-compression-id` (present whenever compression ran and helped) is the key to pass to `GET /v1/retrieve/{compression_id}` to get the original messages back. Entries expire and are purged automatically after the TTL. Ported from [headroom](https://github.com/chopratejas/headroom)'s "Cache & Context Retrieval" concept, scaled down to ecoprompt's single-table SQLite footprint.

## Routing

Prompts are classified as `simple`, `medium`, or `complex` based on length and keyword signals (see `core/router.py`). Each tier has a primary model and one automatic fallback on Groq's free tier, if the primary errors out or gets decommissioned, the fallback takes over automatically. Run `python scripts/check_models.py` periodically to catch decommissioned models before they cause a problem in production.

## Pricing / `estimated_savings_usd`

`tokens_saved_by_compression` is valued at each model's actual published Groq on-demand rate (see `core/pricing.py`), not one flat guessed rate — a request routed to `gpt-oss-120b` and one routed to `qwen3.6-27b` are priced differently because they really do cost differently. Only models with a rate Groq actually publishes are priced; anything else (e.g. `groq/compound-mini`, a compound system whose cost passes through to whatever it invokes at runtime rather than a fixed per-token rate, or an unlisted preview model) is excluded from the dollar total and named in `savings_excluded_unpriced_models` instead of being guessed at.

Two caveats worth knowing: this is Groq's **on-demand** rate — since the models this project routes to are actually on Groq's free developer tier, real billed cost is $0 regardless of what's shown here; treat the number as "what this would've cost on-demand," not an invoice. And prices/model lineups both drift over time, so re-verify `core/pricing.py` against [groq.com/pricing](https://groq.com/pricing) periodically, the same way `scripts/check_models.py` catches decommissioned models.

## Built on

- [FastAPI](https://fastapi.tiangolo.com) — Proxy framework
- [Groq](https://groq.com) — Free-tier inference for all routed models
- [LLMLingua](https://github.com/microsoft/LLMLingua) — Prompt compression (local dev only)
- [sentence-transformers](https://www.sbert.net) — Embeddings + cross-encoder validation for the semantic cache (local dev only)
- [Qdrant](https://qdrant.tech) — Vector cache (local dev only)
