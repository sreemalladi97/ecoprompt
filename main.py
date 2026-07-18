"""
EcoPrompt - Developer Middleware Proxy
Step 4: Routing Matrix + multi-provider support
"""

import os
import re
import time
import uuid
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from utils.logger import log_request
from dotenv import load_dotenv
load_dotenv()

DASHBOARD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
TESTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tester.html")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecoprompt")

app = FastAPI(
    title="EcoPrompt",
    description="Smart middleware proxy to slash AI token usage and cost",
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    # allow_headers controls what the browser may SEND; it does nothing for
    # what JS is allowed to READ back. Without expose_headers, browsers
    # silently hide any response header outside a tiny default safelist,
    # which is why the tester showed "unknown"/"none" for tier, model, and
    # cache despite the response itself being correct.
    expose_headers=[
        "x-ecoprompt-cache",
        "x-ecoprompt-routed-model",
        "x-ecoprompt-route-tier",
        "x-ecoprompt-route-reason",
        "x-ecoprompt-tokens-saved",
        "x-ecoprompt-compression-id",
        "x-ecoprompt-compression-skipped",
        "x-ecoprompt-output-shaped",
        "x-ecoprompt-style",
    ],
)

PROVIDERS = {
    "groq":   "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}


def detect_provider(model: str, auth_header: str):
    api_key = auth_header.replace("Bearer ", "").strip() if auth_header else ""
    if not api_key:
        # Local-dev convenience only: fall back to a server-side key from
        # .env when the caller doesn't supply one. Never required — the
        # public/Vercel deployment has no .env, so every caller there must
        # still bring their own key and nobody's quota is shared.
        api_key = os.environ.get("GROQ_API_KEY", "")
    if model.startswith("groq/") or api_key.startswith("gsk_"):
        clean_model = model.replace("groq/", "")
        return PROVIDERS["groq"], api_key, clean_model
    return PROVIDERS["openai"], api_key, model


def apply_reasoning_params(body: dict, model: str) -> dict:
    """
    Several of our routed models (qwen/qwen3*, openai/gpt-oss-*) are
    "reasoning" models that emit their step-by-step thinking by default —
    Qwen inline as <think>...</think> in the answer text, GPT-OSS in a
    separate `reasoning` field. Neither is useful to whatever app is
    calling this proxy, and generating it burns extra tokens/latency for
    nothing. Ask the model to skip it, per-family since the parameter
    name differs and the two are mutually exclusive on Groq's API.
    """
    if model.startswith("qwen/"):
        body["reasoning_format"] = "hidden"
    elif model.startswith("openai/gpt-oss"):
        body["include_reasoning"] = False
    return body


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(content: str) -> str:
    """
    Defensive cleanup in case a <think> block slips through anyway
    (e.g. a model/provider that doesn't honor apply_reasoning_params, or
    a future model we haven't special-cased yet). Removes complete
    <think>...</think> blocks, and if one never closed — e.g. the
    response got cut off by max_completion_tokens mid-thought — drops
    everything from that point on rather than showing a dangling tag.
    """
    if not content:
        return content
    cleaned = _THINK_BLOCK_RE.sub("", content)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.strip()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.5.0",
        "features": [
            "token_compression", "semantic_cache", "routing_matrix",
            "reversible_compression", "output_shaping", "lazy_mode_style",
            "content_aware_compression",
        ],
        "routing_tiers": {
            "simple":  ["openai/gpt-oss-20b", "groq/compound-mini"],
            "medium":  ["qwen/qwen3.6-27b", "qwen/qwen3-32b"],
            "complex": ["openai/gpt-oss-120b", "meta-llama/llama-4-scout-17b-16e-instruct"],
        },
    }


@app.get("/dashboard")
async def dashboard():
    if not os.path.exists(DASHBOARD_PATH):
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(DASHBOARD_PATH, media_type="text/html")


@app.get("/test")
async def tester():
    if not os.path.exists(TESTER_PATH):
        raise HTTPException(status_code=404, detail="tester.html not found")
    return FileResponse(TESTER_PATH, media_type="text/html")


@app.get("/stats")
async def stats():
    # Prefer the persisted SQLite log so stats survive restarts and
    # serverless cold starts. Vercel's filesystem is read-only at runtime,
    # so this falls back to the in-memory counters there — same
    # graceful-degradation pattern as the cache/compressor/memory subsystems.
    try:
        from utils.logger import get_summary
        return get_summary()
    except Exception as e:
        logger.warning(f"Persisted stats unavailable, using in-memory counters: {e}")

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
        # 3-tier routing breakdown
        "routes_to_cheap_model":    app.state.simple_routes,
        "routes_to_medium_model":   app.state.medium_routes,
        "routes_to_powerful_model": app.state.complex_routes,
        "fallback_model_used":     app.state.fallback_used,
        "estimated_savings_usd": round(app.state.estimated_savings, 4),
        "savings_excluded_unpriced_models": sorted(app.state.unpriced_models_seen),
    }


@app.get("/v1/retrieve/{compression_id}")
async def retrieve_original(compression_id: str):
    """
    Reversible-compression lookup (headroom's CCR pattern, ported to
    ecoprompt's scale): compression is lossy, so the pre-compression
    messages for a given request are kept around for a short TTL. Pass the
    request's x-ecoprompt-compression-id header (present whenever
    compression actually ran and saved tokens) to pull the original back.
    """
    from core.reversible import retrieve_original as lookup
    result = lookup(compression_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Not found or expired")
    return result


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

    # STYLE: Optional lazy/minimal-code system-prompt injection (ponytail-
    # inspired, opt-in via x-ecoprompt-style: lazy). Applied before the
    # cache lookup so a lazy-mode request never hits/pollutes the cache
    # with an answer generated under a different style.
    lazy_mode = False
    try:
        from core.style import is_lazy_mode, inject_lazy_mode
        lazy_mode = is_lazy_mode(request.headers.get("x-ecoprompt-style", ""))
        if lazy_mode:
            messages = inject_lazy_mode(messages)
            body["messages"] = messages
    except Exception as e:
        logger.warning(f"[{request_id}] Style injection skipped: {e}")

    logger.info(f"[{request_id}] Incoming | model={model} | messages={len(messages)}")

    upstream_url, api_key, clean_model = detect_provider(model, auth_header)
    body["model"] = clean_model

    # PHASE 3: Semantic cache
    use_cache = (
        request.headers.get("x-ecoprompt-cache", "true").lower() != "false"
        and not lazy_mode
    )
    if use_cache:
        try:
            from core.cache import cache_lookup
            cached = cache_lookup(messages)
            if cached:
                app.state.cache_hits += 1
                app.state.request_count += 1
                latency_ms = round((time.time() - start) * 1000)
                try:
                    log_request(request_id=request_id, model=model, tokens_in=0,
                                tokens_out=0, latency_ms=latency_ms, cache_hit=True, source="cache")
                except Exception as log_err:
                    # SQLite logging is a nice-to-have, not required for a
                    # correct response (e.g. Vercel's filesystem is
                    # read-only at runtime, so this always fails there —
                    # that shouldn't turn a cache hit into a wasted
                    # duplicate upstream call).
                    logger.warning(f"[{request_id}] Request logging skipped: {log_err}")
                logger.info(f"[{request_id}] Cache HIT | latency={latency_ms}ms")
                response = JSONResponse(content=cached)
                response.headers["x-ecoprompt-cache"] = "hit"
                return response
        except Exception as e:
            logger.warning(f"[{request_id}] Cache lookup skipped: {e}")

    # PHASE 2: Token compression
    compression_stats = {"tokens_saved": 0, "compression_ratio": 1.0}
    compression_id = None
    compress = request.headers.get("x-ecoprompt-compress", "true").lower() != "false"
    if compress:
        try:
            from core.compressor import compress_messages
            compressed_messages, compression_stats = compress_messages(messages)
            body["messages"] = compressed_messages
            skipped = compression_stats.get("content_type_skipped") or {}
            skip_note = f" | skipped (non-prose): {skipped}" if skipped else ""
            logger.info(f"[{request_id}] Compressed | saved={compression_stats['tokens_saved']} tokens{skip_note}")

            # Compression is lossy — keep the pre-compression messages
            # retrievable for a short TTL (headroom's CCR pattern) so a
            # caller can pull the original back via /v1/retrieve/{id}.
            if compression_stats.get("tokens_saved", 0) > 0:
                try:
                    from core.reversible import store_original
                    store_original(request_id, messages)
                    compression_id = request_id
                except Exception as ccr_err:
                    logger.warning(f"[{request_id}] Reversible-compression store skipped: {ccr_err}")
        except Exception as e:
            logger.warning(f"[{request_id}] Compression skipped: {e}")

    # PHASE 4: 3-tier Routing
    routed_model = clean_model
    route_tier = "none"
    route_reason = "routing disabled (x-ecoprompt-route: false)"
    candidates = [clean_model]
    use_router = request.headers.get("x-ecoprompt-route", "true").lower() != "false"
    if use_router:
        try:
            from core.router import get_candidates, classify_with_reason
            route_tier, route_reason = classify_with_reason(messages)
            candidates = get_candidates(route_tier, clean_model)

            if route_tier == "simple":
                app.state.simple_routes += 1
            elif route_tier == "medium":
                app.state.medium_routes += 1
            else:
                app.state.complex_routes += 1

        except Exception as e:
            route_reason = f"routing failed, used requested model instead ({e})"
            logger.warning(f"[{request_id}] Routing skipped: {e}")

    # PHASE 5: Output-token shaping — routine (simple-tier) requests only.
    # Terser prompt + lower reasoning effort cuts what comes back, not just
    # what gets sent.
    output_shaped = False
    shape_output = request.headers.get("x-ecoprompt-shape-output", "true").lower() != "false"
    if shape_output:
        try:
            from core.output_shaper import is_shaped, inject_terse_note
            if is_shaped(route_tier):
                body["messages"] = inject_terse_note(body["messages"])
                output_shaped = True
        except Exception as e:
            logger.warning(f"[{request_id}] Output shaping skipped: {e}")

    forward_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    logger.info(f"[{request_id}] Sending to {upstream_url} | candidates={candidates} | tier={route_tier} ({route_reason})")

    # Try each candidate model in order. Only the primary is tried under
    # normal conditions; later candidates are used if the primary fails
    # (e.g. rate-limited, or removed by the provider without our knowledge —
    # run scripts/check_models.py periodically to catch that ahead of time).
    upstream = None
    last_error = None
    fallback_used_this_request = False
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, candidate_model in enumerate(candidates):
            body["model"] = candidate_model
            apply_reasoning_params(body, candidate_model)
            if output_shaped:
                try:
                    from core.output_shaper import dial_reasoning_effort
                    dial_reasoning_effort(body, candidate_model)
                except Exception as e:
                    logger.warning(f"[{request_id}] Reasoning-effort dial skipped: {e}")
            try:
                upstream = await client.post(upstream_url, json=body, headers=forward_headers)
                upstream.raise_for_status()
                routed_model = candidate_model
                if i > 0:
                    app.state.fallback_used += 1
                    fallback_used_this_request = True
                    logger.warning(f"[{request_id}] Used fallback model {candidate_model} (primary {candidates[0]} failed)")
                break
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"[{request_id}] {candidate_model} failed ({e.response.status_code})"
                                + (" — trying next fallback" if i < len(candidates) - 1 else " — no fallbacks left"))
                upstream = None
                continue
            except httpx.RequestError as e:
                last_error = e
                logger.error(f"[{request_id}] Connection error on {candidate_model}: {e}")
                upstream = None
                continue

        if upstream is None:
            if isinstance(last_error, httpx.HTTPStatusError):
                logger.error(f"[{request_id}] All candidates failed: {last_error.response.status_code} {last_error.response.text}")
                return JSONResponse(status_code=last_error.response.status_code, content=last_error.response.json())
            logger.error(f"[{request_id}] All candidates failed: {last_error}")
            raise HTTPException(status_code=502, detail="Upstream connection failed for all candidate models")

    response_data = upstream.json()
    latency_ms = round((time.time() - start) * 1000)

    # Belt-and-suspenders: strip any leftover <think> reasoning that made
    # it into the answer text despite apply_reasoning_params() above.
    try:
        choice = response_data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        if msg.get("content"):
            original = msg["content"]
            msg["content"] = strip_reasoning(original)
            if msg["content"] != original.strip():
                logger.info(f"[{request_id}] Stripped inline reasoning from response")
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"[{request_id}] Reasoning strip skipped: {e}")

    usage = response_data.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0)
    tokens_out = usage.get("completion_tokens", 0)
    tokens_saved = compression_stats.get("tokens_saved", 0)

    app.state.request_count += 1
    app.state.tokens_in += tokens_in
    app.state.tokens_out += tokens_out
    app.state.tokens_saved += tokens_saved

    if tokens_saved > 0:
        # Priced at the model actually used for this request, not a flat
        # guessed rate — see core/pricing.py. Unpriced models contribute
        # nothing to the dollar total rather than a made-up number, but
        # are tracked so /stats can say so instead of silently omitting them.
        from core.pricing import estimate_input_cost_usd
        cost = estimate_input_cost_usd(routed_model, tokens_saved)
        if cost is not None:
            app.state.estimated_savings += cost
        else:
            app.state.unpriced_models_seen.add(routed_model)

    if use_cache:
        try:
            from core.cache import cache_store
            cache_store(messages, response_data)
        except Exception as e:
            logger.warning(f"[{request_id}] Cache store skipped: {e}")

    try:
        log_request(request_id=request_id, model=routed_model, tokens_in=tokens_in,
                    tokens_out=tokens_out, latency_ms=latency_ms, cache_hit=False, source="upstream",
                    tokens_saved=tokens_saved, route_tier=route_tier, fallback_used=fallback_used_this_request)
    except Exception as log_err:
        # Same reasoning as the cache-hit path above: a logging failure
        # (e.g. read-only filesystem on Vercel) must not turn an
        # otherwise-successful upstream response into a 500 error.
        logger.warning(f"[{request_id}] Request logging skipped: {log_err}")

    logger.info(
        f"[{request_id}] Done | model={routed_model} tier={route_tier} "
        f"tokens_in={tokens_in} tokens_out={tokens_out} saved={tokens_saved} latency={latency_ms}ms"
    )

    response = JSONResponse(content=response_data, status_code=upstream.status_code)
    response.headers["x-ecoprompt-cache"] = "miss"
    response.headers["x-ecoprompt-routed-model"] = routed_model
    response.headers["x-ecoprompt-route-tier"] = route_tier
    response.headers["x-ecoprompt-route-reason"] = route_reason
    response.headers["x-ecoprompt-tokens-saved"] = str(tokens_saved)
    if compression_id:
        response.headers["x-ecoprompt-compression-id"] = compression_id
    content_type_skipped = compression_stats.get("content_type_skipped") or {}
    if content_type_skipped:
        response.headers["x-ecoprompt-compression-skipped"] = ",".join(
            f"{k}:{v}" for k, v in sorted(content_type_skipped.items())
        )
    response.headers["x-ecoprompt-output-shaped"] = "true" if output_shaped else "false"
    if lazy_mode:
        response.headers["x-ecoprompt-style"] = "lazy"
    return response


@app.on_event("startup")
async def startup():
    app.state.request_count = 0
    app.state.tokens_in = 0
    app.state.tokens_out = 0
    app.state.tokens_saved = 0
    app.state.cache_hits = 0
    app.state.simple_routes = 0
    app.state.medium_routes = 0
    app.state.complex_routes = 0
    app.state.fallback_used = 0
    app.state.estimated_savings = 0.0
    app.state.unpriced_models_seen = set()
    logger.info("EcoPrompt v0.5.0 started - 3-tier routing (simple/medium/complex) with per-tier fallbacks")
