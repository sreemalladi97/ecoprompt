# EcoPrompt

> Smart middleware proxy that slashes AI token usage and cloud bills.

## What it does

```
Your App → EcoPrompt Proxy → AI Model (OpenAI / Anthropic / Ollama / etc.)
                ↓
         [Token Compressor]   strip fluff before sending
         [Semantic Cache]     answer repeats for free
         [Routing Matrix]     cheap model for simple tasks
```

## Roadmap

- [x] **Step 1** - Proxy shell (passthrough + observability)
- [ ] **Step 2** - Token compressor (LLMLingua)
- [ ] **Step 3** - Semantic cache (embeddings + Qdrant)
- [ ] **Step 4** - Routing matrix (complexity classifier)
- [ ] **Step 5** - Dashboard UI

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Point your app at the proxy instead of OpenAI
# Change: https://api.openai.com/v1/chat/completions
# To:     http://localhost:8000/v1/chat/completions
```

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /stats` | Token usage, cache hits, savings |
| `POST /v1/chat/completions` | Drop-in OpenAI-compatible proxy |

## Built on

- [FastAPI](https://fastapi.tiangolo.com) - Proxy framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Multi-model routing (Step 4)
- [LLMLingua](https://github.com/microsoft/LLMLingua) - Prompt compression (Step 2)
- [Qdrant](https://qdrant.tech) - Vector cache (Step 3)
# ecoprompt
