"""
EcoPrompt - Semantic Cache
Uses sentence-transformers to embed prompts and Qdrant to find similar past answers.
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger("ecoprompt.cache")

_embedder = None
_qdrant = None
_validator = None

# Cache statistics — tracks hits and misses for the dashboard
_stats = {
    "hits": 0,
    "misses": 0,
    "stores": 0,
}

COLLECTION_NAME = "ecoprompt_cache"
SIMILARITY_THRESHOLD = 0.85


def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Loading sentence-transformers model (first time only)...")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder loaded.")
    return _embedder


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        _qdrant = QdrantClient(path="./qdrant_storage")

        existing = [c.name for c in _qdrant.get_collections().collections]
        if COLLECTION_NAME not in existing:
            _qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    return _qdrant

def get_validator():
    global _validator
    if _validator is None:
        logger.info("Loading cross-encoder validator model...")
        from sentence_transformers import CrossEncoder
        _validator = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Validator loaded.")
    return _validator

def validate_answer(question: str, answer: str, qdrant_score: float = 0.0) -> bool:
    """
    Checks whether a cached answer actually answers the given question.
    Uses a cross-encoder to score relevance between question and answer.

    Returns True if the answer is relevant, False if it should be rejected.
    """
    try:
        validator = get_validator()

        # Cross-encoder scores the question+answer pair together
        score = validator.predict([(question, answer)])

        # Score is a raw logit — higher means more relevant
        # Threshold of 0 is a safe starting point based on ms-marco model behavior
        is_valid = float(score[0]) > 5.0 or (float(score[0]) > 3.0 and qdrant_score > 0.92)

        logger.info(f"Validator score: {score[0]:.3f} → {'VALID ✅' if is_valid else 'REJECTED ❌'}")
        return is_valid

    except Exception as e:
        logger.warning(f"Validation failed, defaulting to accept: {e}")
        return True  # if validator breaks, don't block the cache from working

def _messages_to_text(messages: list) -> str:
    # Only use user messages for cache key — ignore injected system context
    # This ensures memory injection doesn't break cache matching
    user_messages = [
        m.get("content", "") for m in messages
        if m.get("role") == "user"
    ]
    return " ".join(user_messages).strip()


def _make_id(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def cache_lookup(messages: list) -> Optional[dict]:
    try:
        text = _messages_to_text(messages)
        embedder = get_embedder()
        qdrant = get_qdrant()
        vector = embedder.encode(text).tolist()

        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=1,
            score_threshold=SIMILARITY_THRESHOLD,
        ).points

        if results:
            cached_response = results[0].payload.get("response")

            # Extract the answer text for validation
            answer_text = ""
            if isinstance(cached_response, dict):
                answer_text = cached_response.get("content", "")
            elif isinstance(cached_response, str):
                answer_text = cached_response

            # Validate that the cached answer actually answers the new question
            if validate_answer(text, answer_text, qdrant_score=results[0].score):
                _stats["hits"] += 1
                logger.info(f"Cache HIT (score={results[0].score:.3f})")
                return cached_response
            else:
                _stats["misses"] += 1
                logger.info("Cache HIT rejected by validator — sending to LLM")
                return None

        _stats["misses"] += 1
        logger.info("Cache MISS")
        return None

    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
        return None


def cache_store(messages: list, response: dict):
    try:
        text = _messages_to_text(messages)

        # Extract answer text for quality check
        answer_text = ""
        if isinstance(response, dict):
            answer_text = response.get("content", "")
        elif isinstance(response, str):
            answer_text = response

        # Validate answer quality before storing
        # Use a lower threshold here than lookup — we want to be lenient
        # about storing but strict about retrieving
        if answer_text and not validate_answer(text, answer_text, qdrant_score=1.0):
            logger.warning("Answer quality too low — skipping cache store")
            _stats["rejected_stores"] = _stats.get("rejected_stores", 0) + 1
            return

        embedder = get_embedder()
        qdrant = get_qdrant()
        vector = embedder.encode(text).tolist()
        point_id = _make_id(text)

        from qdrant_client.models import PointStruct
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"response": response, "prompt": text[:500]},
                )
            ],
        )
        logger.info(f"Cache stored (id={point_id})")
        _stats["stores"] += 1

    except Exception as e:
        logger.warning(f"Cache store failed: {e}")

def get_cache_stats() -> dict:
    """
    Returns current cache performance statistics.
    Used by the dashboard to display hit/miss rates.
    """
    total = _stats["hits"] + _stats["misses"]
    hit_rate = round(_stats["hits"] / total * 100, 1) if total > 0 else 0.0

    return {
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "stores": _stats["stores"],
        "rejected_stores": _stats.get("rejected_stores", 0),
        "total_requests": total,
        "hit_rate_percent": hit_rate,
    }