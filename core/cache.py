"""
EcoPrompt - Semantic Cache
Uses sentence-transformers to embed prompts and Qdrant to find similar past answers.
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger("ecoprompt.cache")

# Lazy-load heavy dependencies
_embedder = None
_qdrant = None

COLLECTION_NAME = "ecoprompt_cache"
SIMILARITY_THRESHOLD = 0.95  # 0.0 to 1.0 — higher = stricter match required


def get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Loading sentence-transformers model (first time only, ~30s)...")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder loaded.")
    return _embedder


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        # Local in-memory Qdrant (no server needed, free)
        _qdrant = QdrantClient(":memory:")

        # Create collection if it doesn't exist
        collections = [c.name for c in _qdrant.get_collections().collections]
        if COLLECTION_NAME not in collections:
            _qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    return _qdrant


def _messages_to_text(messages: list) -> str:
    """Flatten messages into a single string for embedding."""
    return " ".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in messages
    ).strip()


def _make_id(text: str) -> int:
    """Make a numeric ID from a string (Qdrant requires integer IDs)."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def cache_lookup(messages: list) -> Optional[dict]:
    """
    Search the cache for a semantically similar past request.
    Returns the cached response if found, None otherwise.
    """
    try:
        text = _messages_to_text(messages)
        embedder = get_embedder()
        qdrant = get_qdrant()

        vector = embedder.encode(text).tolist()

        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=1,
            score_threshold=SIMILARITY_THRESHOLD,
        )

        if results:
            logger.info(f"Cache HIT (score={results[0].score:.3f})")
            return results[0].payload.get("response")

        logger.info("Cache MISS")
        return None

    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")
        return None


def cache_store(messages: list, response: dict):
    """
    Store a prompt+response pair in the cache.
    """
    try:
        text = _messages_to_text(messages)
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

    except Exception as e:
        logger.warning(f"Cache store failed: {e}")
