"""
EcoPrompt - Developer Middleware Proxy
Step 1: Proxy Shell (passthrough with observability)
"""

import time
import uuid
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
from utils.logger import log_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecoprompt")

app = FastAPI(
    title="EcoPrompt",
    description="Smart middleware proxy to slash AI token usage and cost",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Stats endpoint (grows as we add caching/routing)
# ---------------------------------------------------------------------------

@app.get("/stats")
async def stats():
    return {
        "requests_proxied": app.state.request_count,
        "tokens_in": app.state.tokens_in,
        "tokens_out": app.state.tokens_out,
        "cache_hits": app.state.cache_hits,
        "estimated_savings_usd": round(app.state.estimated_savings, 4),
    }


# ---------------------------------------------------------------------------
# Core proxy: intercept /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model", "unknown")
    messages = body.get("messages", [])
    logger.info(f"[{request_id}] Incoming request | model={model} | messages={len(messages)}")

    # -----------------------------------------------------------------------
    # PHASE 1 (current): Pure passthrough to OpenAI-compatible endpoint
    # PHASE 2: Token compressor hooks in here
    # PHASE 3: Semantic cache check hooks in here
    # PHASE 4: Routing matrix hooks in here
    # -----------------------------------------------------------------------

    upstream_url = "https://api.openai.com/v1/chat/completions"

    # Forward original headers (strip hop-by-hop, keep Authorization)
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "transfer-encoding"}
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            upstream = await client.post(upstream_url, json=body, headers=forward_headers)
            upstream.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"[{request_id}] Upstream error: {e.response.status_code}")
            return JSONResponse(status_code=e.response.status_code, content=e.response.json())
        except httpx.RequestError as e:
            logger.error(f"[{request_id}] Connection error: {e}")
            raise HTTPException(status_code=502, detail="Upstream connection failed")

    response_data = upstream.json()
    latency_ms = round((time.time() - start) * 1000)

    # Extract usage stats
    usage = response_data.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)

    # Update global counters
    app.state.request_count += 1
    app.state.tokens_in += tokens_in
    app.state.tokens_out += tokens_out

    log_request(
        request_id=request_id,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        cache_hit=False,
        source="upstream",
    )

    logger.info(
        f"[{request_id}] Done | tokens_in={tokens_in} tokens_out={tokens_out} "
        f"latency={latency_ms}ms"
    )

    return JSONResponse(content=response_data, status_code=upstream.status_code)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    app.state.request_count = 0
    app.state.tokens_in = 0
    app.state.tokens_out = 0
    app.state.cache_hits = 0
    app.state.estimated_savings = 0.0
    logger.info("EcoPrompt proxy started on http://0.0.0.0:8000")
