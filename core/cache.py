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
RETRIEVAL_THRESHOLD = 0.65  # wide net for candidates
VALIDATOR_THRESHOLD = 0.75  # strict gate for acceptance


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
        _validator = CrossEncoder("cross-encoder/quora-roberta-base")
        logger.info("Validator loaded.")
    return _validator

def validate_prompt_match(new_question: str, cached_question: str, qdrant_score: float = 0.0) -> bool:
    """
    Checks whether the new question and cached question mean the same thing.
    Uses a duplicate-question cross-encoder trained specifically for this task.
    Returns True if questions are equivalent, False if they should not share a cached answer.
    """
    try:
        validator = get_validator()
        score = float(validator.predict([(new_question, cached_question)])[0])
        is_valid = score >= VALIDATOR_THRESHOLD
        logger.info(f"Prompt validator score: {score:.3f} → {'VALID ✅' if is_valid else 'REJECTED ❌'}")
        return is_valid
    except Exception as e:
        logger.warning(f"Prompt validation failed, defaulting to reject: {e}")
        return False

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

def _normalize(text: str) -> str:
    """Normalize text for exact match comparison."""
    return " ".join(text.lower().strip().split())

def cache_lookup(messages: list) -> Optional[dict]:
    try:
        text = _messages_to_text(messages)
        embedder = get_embedder()
        qdrant = get_qdrant()
        vector = embedder.encode(text).tolist()

        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=5,
            score_threshold=RETRIEVAL_THRESHOLD,
        ).points

        for result in results:
            cached_response = result.payload.get("response")
            cached_prompt = result.payload.get("prompt", "")

            # Exact normalized match — skip validator entirely
            if _normalize(text) == _normalize(cached_prompt):
                _stats["hits"] += 1
                logger.info("Cache HIT — exact match, skipping validator")
                return cached_response

            # Validate question equivalence
            if validate_prompt_match(text, cached_prompt, qdrant_score=result.score):
                _stats["hits"] += 1
                logger.info(f"Cache HIT (qdrant={result.score:.3f})")
                return cached_response

        _stats["misses"] += 1
        logger.info("Cache MISS")
        return None
    
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
        return None


def cache_store(messages: list, response: dict):
    try:
        text = _messages_to_text(messages)

        # Only skip storing if answer is empty
        answer_text = ""
        if isinstance(response, dict):
            # Try direct content first
            answer_text = response.get("content", "")
            # If empty, try nested Groq/OpenAI response structure
            if not answer_text:
                try:
                    answer_text = response["choices"][0]["message"]["content"]
                except (KeyError, IndexError):
                    pass
        elif isinstance(response, str):
            answer_text = response

        if not answer_text.strip():
            logger.warning("Empty answer — skipping cache store")
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
                    payload={"response": response, "prompt": text},
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