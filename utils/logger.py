"""
EcoPrompt - Request Logger
Writes structured logs to SQLite for the /stats dashboard.
"""

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "ecoprompt.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id          TEXT PRIMARY KEY,
            timestamp   REAL,
            model       TEXT,
            tokens_in   INTEGER,
            tokens_out  INTEGER,
            latency_ms  INTEGER,
            cache_hit   INTEGER,
            source      TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_request(
    request_id: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    cache_hit: bool,
    source: str,
):
    init_db()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO requests (id, timestamp, model, tokens_in, tokens_out, latency_ms, cache_hit, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (request_id, time.time(), model, tokens_in, tokens_out, latency_ms, int(cache_hit), source),
    )
    conn.commit()
    conn.close()


def get_summary():
    init_db()
    conn = _get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*)          AS total_requests,
            SUM(tokens_in)    AS total_tokens_in,
            SUM(tokens_out)   AS total_tokens_out,
            SUM(cache_hit)    AS total_cache_hits,
            AVG(latency_ms)   AS avg_latency_ms
        FROM requests
    """).fetchone()
    conn.close()
    return dict(row)
