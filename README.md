# EcoPrompt

> Smart middleware proxy that slashes AI token usage and cloud bills.

## What it does

```
Your App → EcoPrompt Proxy → AI Model (Groq, free tier)
                ↓
         [Token Compressor]   strip fluff before sending
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

Each caller supplies their own Groq API key in the request's `Authorization` header, same as calling Groq directly, the proxy doesn't hold or require a server-side key to run.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check + current routing tier config |
| `GET /stats` | Token usage, cache hits, routing breakdown, fallback usage |
| `GET /dashboard` | Live dashboard UI |
| `GET /test` | Browser-based request tester |
| `POST /v1/chat/completions` | Drop-in OpenAI-compatible proxy |

## Routing

Prompts are classified as `simple`, `medium`, or `complex` based on length and keyword signals (see `core/router.py`). Each tier has a primary model and one automatic fallback on Groq's free tier, if the primary errors out or gets decommissioned, the fallback takes over automatically. Run `python scripts/check_models.py` periodically to catch decommissioned models before they cause a problem in production.

## Built on

- [FastAPI](https://fastapi.tiangolo.com) — Proxy framework
- [Groq](https://groq.com) — Free-tier inference for all routed models
- [LLMLingua](https://github.com/microsoft/LLMLingua) — Prompt compression (local dev only)
- [sentence-transformers](https://www.sbert.net) — Embeddings + cross-encoder validation for the semantic cache (local dev only)
- [Qdrant](https://qdrant.tech) — Vector cache (local dev only)
