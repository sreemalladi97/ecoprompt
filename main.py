"""
EcoPrompt - Developer Middleware Proxy
Step 4: Routing Matrix + multi-provider support
"""

import os
import time
import uuid
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from utils.logger import log_request
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecoprompt")

app = FastAPI(
    title="EcoPrompt",
    description="Smart middleware proxy to slash AI token usage and cost",
    version="0.4.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROVIDERS = {
    "groq":   "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}


def detect_provider(model: str, auth_header: str):
    api_key = auth_header.replace("Bearer ", "").strip() if auth_header else ""
    if model.startswith("groq/") or api_key.startswith("gsk_"):
        clean_model = model.replace("groq/", "")
        return PROVIDERS["groq"], api_key, clean_model
    return PROVIDERS["openai"], api_key, model


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.4.1",
        "features": ["token_compression", "semantic_cache", "routing_matrix"],
    }


@app.get("/stats")
async def stats():
    total = app.state.request_count
    hits = app.state.cache_hits
    hit_rate = round((hits / total * 100), 1) if total > 0 else 0.0
    return {
        "requests_proxied": total,
        "tokens_in": app.state.tokens_in,
        "tokens_out": app.state.tokens_out,
        "tokens_saved_by_compression": app.state.tokens_saved,
        "cache_hits": hits,
        "cache_hit_rate_pct": hit_rate,
        "routes_to_cheap_model": app.state.cheap_routes,
        "routes_to_powerful_model": app.state.powerful_routes,
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
    auth_header = request.headers.get("authorization", "")

    # MEMORY: Check for remember command
    try:
        from core.memory import handle_memory_command, memory_inject
        was_command, confirmation = handle_memory_command(messages)
        if was_command:
            return JSONResponse(content={
                "choices": [{"message": {"role": "assistant", "content": confirmation}}]
            })
        # Inject memory context into messages before cache/AI
        messages = memory_inject(messages)
        body["messages"] = messages
    except Exception as e:
        logger.warning(f"[{request_id}] Memory skipped: {e}")

    logger.info(f"[{request_id}] Incoming | model={model} | messages={len(messages)}")

    upstream_url, api_key, clean_model = detect_provider(model, auth_header)
    body["model"] = clean_model

    # PHASE 3: Semantic cache
    use_cache = request.headers.get("x-ecoprompt-cache", "true").lower() != "false"
    if use_cache:
        try:
            from core.cache import cache_lookup
            cached = cache_lookup(messages)
            if cached:
                app.state.cache_hits += 1
                app.state.request_count += 1
                latency_ms = round((time.time() - start) * 1000)
                log_request(request_id=request_id, model=model, tokens_in=0,
                            tokens_out=0, latency_ms=latency_ms, cache_hit=True, source="cache")
                logger.info(f"[{request_id}] Cache HIT | latency={latency_ms}ms")
                response = JSONResponse(content=cached)
                response.headers["x-ecoprompt-cache"] = "hit"
                return response
        except Exception as e:
            logger.warning(f"[{request_id}] Cache lookup skipped: {e}")

    # PHASE 2: Token compression
    compression_stats = {"tokens_saved": 0, "compression_ratio": 1.0}
    compress = request.headers.get("x-ecoprompt-compress", "true").lower() != "false"
    if compress:
        try:
            from core.compressor import compress_messages
            compressed_messages, compression_stats = compress_messages(messages)
            body["messages"] = compressed_messages
            logger.info(f"[{request_id}] Compressed | saved={compression_stats['tokens_saved']} tokens")
        except Exception as e:
            logger.warning(f"[{request_id}] Compression skipped: {e}")

    # PHASE 4: Routing
    routed_model = clean_model
    if not model.startswith("groq/"):
        use_router = request.headers.get("x-ecoprompt-route", "true").lower() != "false"
        if use_router:
            try:
                from core.router import route
                routed_model = route(messages, clean_model)
                body["model"] = routed_model
                if "mini" in routed_model:
                    app.state.cheap_routes += 1
                else:
                    app.state.powerful_routes += 1
            except Exception as e:
                logger.warning(f"[{request_id}] Routing skipped: {e}")

    forward_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    logger.info(f"[{request_id}] Sending to {upstream_url} | model={body['model']}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            upstream = await client.post(upstream_url, json=body, headers=forward_headers)
            upstream.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"[{request_id}] Upstream error: {e.response.status_code} {e.response.text}")
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

    if use_cache:
        try:
            from core.cache import cache_store
            cache_store(messages, response_data)
        except Exception as e:
            logger.warning(f"[{request_id}] Cache store skipped: {e}")

    log_request(request_id=request_id, model=routed_model, tokens_in=tokens_in,
                tokens_out=tokens_out, latency_ms=latency_ms, cache_hit=False, source="upstream")

    logger.info(
        f"[{request_id}] Done | model={routed_model} tokens_in={tokens_in} "
        f"tokens_out={tokens_out} saved={tokens_saved} latency={latency_ms}ms"
    )

    response = JSONResponse(content=response_data, status_code=upstream.status_code)
    response.headers["x-ecoprompt-cache"] = "miss"
    response.headers["x-ecoprompt-routed-model"] = routed_model
    response.headers["x-ecoprompt-tokens-saved"] = str(tokens_saved)
    return response


@app.on_event("startup")
async def startup():
    app.state.request_count = 0
    app.state.tokens_in = 0
    app.state.tokens_out = 0
    app.state.tokens_saved = 0
    app.state.cache_hits = 0
    app.state.cheap_routes = 0
    app.state.powerful_routes = 0
    app.state.estimated_savings = 0.0
    logger.info("EcoPrompt v0.4.1 started - Compression + Cache + Routing + Multi-provider")
