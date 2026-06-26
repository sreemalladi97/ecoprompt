"""
EcoPrompt - Developer Middleware Proxy
Step 2: Token Compressor (LLMLingua-2) wired in
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
    version="0.2.0",
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.0", "features": ["token_compression"]}


@app.get("/stats")
async def stats():
    return {
        "requests_proxied": app.state.request_count,
        "tokens_in": app.state.tokens_in,
        "tokens_out": app.state.tokens_out,
        "tokens_saved_by_compression": app.state.tokens_saved,
        "cache_hits": app.state.cache_hits,
        "estimated_savings_usd": round(app.state.estimated_savings, 4),
    }


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
    logger.info(f"[{request_id}] Incoming | model={model} | messages={len(messages)}")

    # PHASE 2: Token compression via LLMLingua-2
    compression_stats = {"tokens_saved": 0, "compression_ratio": 1.0}
    compress = request.headers.get("x-ecoprompt-compress", "true").lower() != "false"

    if compress:
        try:
            from core.compressor import compress_messages
            compressed_messages, compression_stats = compress_messages(messages)
            body["messages"] = compressed_messages
            logger.info(
                f"[{request_id}] Compressed | "
                f"saved={compression_stats['tokens_saved']} tokens | "
                f"ratio={compression_stats['compression_ratio']}x"
            )
        except Exception as e:
            logger.warning(f"[{request_id}] Compression skipped: {e}")

    # PHASE 3 (coming): Semantic cache check hooks in here
    # PHASE 4 (coming): Routing matrix hooks in here

    upstream_url = "https://api.openai.com/v1/chat/completions"
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "transfer-encoding", "x-ecoprompt-compress"}
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

    usage = response_data.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)
    tokens_saved = compression_stats.get("tokens_saved", 0)

    app.state.request_count += 1
    app.state.tokens_in += tokens_in
    app.state.tokens_out += tokens_out
    app.state.tokens_saved += tokens_saved
    app.state.estimated_savings += (tokens_saved / 1000) * 0.01

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
        f"saved={tokens_saved} latency={latency_ms}ms"
    )

    response = JSONResponse(content=response_data, status_code=upstream.status_code)
    response.headers["x-ecoprompt-tokens-saved"] = str(tokens_saved)
    response.headers["x-ecoprompt-compression-ratio"] = str(compression_stats.get("compression_ratio", 1.0))
    return response


@app.on_event("startup")
async def startup():
    app.state.request_count = 0
    app.state.tokens_in = 0
    app.state.tokens_out = 0
    app.state.tokens_saved = 0
    app.state.cache_hits = 0
    app.state.estimated_savings = 0.0
    logger.info("EcoPrompt v0.2.0 started - Token compression enabled")
