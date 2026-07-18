"""
EcoPrompt - Reversible Compression (CCR-lite)
Ported from headroom's "Cache & Context Retrieval" idea: compression is
lossy by nature, so keep the pre-compression original around for a short
window and let a caller pull it back via a request id if they need the
full text (e.g. to show a user, or re-send uncompressed on a retry).

Stored in the same SQLite file as the request log, scoped to one table.
Rows are TTL'd, not kept forever — this is a short-lived undo buffer, not
an audit trail.
"""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("ecoprompt.reversible")

DB_PATH = Path(__file__).parent.parent / "ecoprompt.db"
DEFAULT_TTL_SECONDS = int(os.environ.get("ECOPROMPT_CCR_TTL_SECONDS", 3600))


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_store():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS compressions (
            id             TEXT PRIMARY KEY,
            original_json  TEXT,
            created_at     REAL,
            expires_at     REAL
        )
    """)
    conn.commit()
    conn.close()


def _purge_expired(conn):
    conn.execute("DELETE FROM compressions WHERE expires_at < ?", (time.time(),))


def store_original(request_id: str, original_messages: list, ttl_seconds: int = None) -> None:
    """
    Save the pre-compression messages under request_id so they can be
    retrieved later. Best-effort: a storage failure should never break
    the actual proxied response (same graceful-degradation pattern used
    for the cache/compressor/memory subsystems).
    """
    ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_TTL_SECONDS
    try:
        init_store()
        conn = _get_conn()
        now = time.time()
        _purge_expired(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO compressions (id, original_json, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (request_id, json.dumps(original_messages), now, now + ttl),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to store original for retrieval (id={request_id}): {e}")


def retrieve_original(request_id: str) -> dict:
    """
    Returns {"messages": [...], "expires_at": <epoch>} if the id exists
    and hasn't expired, else None. Expired-but-present rows are purged
    lazily on read.
    """
    try:
        init_store()
        conn = _get_conn()
        _purge_expired(conn)
        conn.commit()
        row = conn.execute(
            "SELECT original_json, expires_at FROM compressions WHERE id = ?",
            (request_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return {
            "messages": json.loads(row["original_json"]),
            "expires_at": row["expires_at"],
        }
    except Exception as e:
        logger.warning(f"Failed to retrieve original (id={request_id}): {e}")
        return None
