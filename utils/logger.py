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
    # Columns added after the original release — wrapped individually so a
    # pre-existing ecoprompt.db (missing them) gets migrated in place instead
    # of requiring a manual reset. SQLite has no "ADD COLUMN IF NOT EXISTS".
    for column, coltype, default in [
        ("tokens_saved", "INTEGER", 0),
        ("route_tier", "TEXT", "'none'"),
        ("fallback_used", "INTEGER", 0),
    ]:
        try:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {column} {coltype} DEFAULT {default}")
        except sqlite3.OperationalError:
            pass  # column already exists
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
    tokens_saved: int = 0,
    route_tier: str = "none",
    fallback_used: bool = False,
):
    init_db()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO requests
            (id, timestamp, model, tokens_in, tokens_out, latency_ms, cache_hit, source,
             tokens_saved, route_tier, fallback_used)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (request_id, time.time(), model, tokens_in, tokens_out, latency_ms, int(cache_hit), source,
         tokens_saved, route_tier, int(fallback_used)),
    )
    conn.commit()
    conn.close()


def get_summary():
    """
    Aggregate stats from the persisted SQLite log, so /stats survives
    process restarts and serverless cold starts instead of relying solely
    on in-memory counters that reset every time.
    """
    init_db()
    conn = _get_conn()
    totals = conn.execute("""
        SELECT
            COUNT(*)                                   AS requests_proxied,
            COALESCE(SUM(tokens_in), 0)                AS tokens_in,
            COALESCE(SUM(tokens_out), 0)                AS tokens_out,
            COALESCE(SUM(tokens_saved), 0)              AS tokens_saved_by_compression,
            COALESCE(SUM(cache_hit), 0)                 AS cache_hits,
            COALESCE(SUM(fallback_used), 0)             AS fallback_model_used
        FROM requests
    """).fetchone()

    tier_rows = conn.execute("""
        SELECT route_tier, COUNT(*) AS n FROM requests GROUP BY route_tier
    """).fetchall()

    # Tokens saved at Groq's real per-model rate, not one flat guessed
    # rate — compression only ever reduces input tokens, so this is
    # valued at each model's actual input price. Grouped by model since
    # different requests can route to different models.
    savings_rows = conn.execute("""
        SELECT model, COALESCE(SUM(tokens_saved), 0) AS tokens_saved
        FROM requests
        WHERE tokens_saved > 0
        GROUP BY model
    """).fetchall()
    conn.close()

    from core.pricing import estimate_input_cost_usd

    estimated_savings_usd = 0.0
    unpriced_models = set()
    for row in savings_rows:
        cost = estimate_input_cost_usd(row["model"], row["tokens_saved"])
        if cost is None:
            unpriced_models.add(row["model"])
            continue
        estimated_savings_usd += cost

    tier_counts = {row["route_tier"]: row["n"] for row in tier_rows}
    total = totals["requests_proxied"]
    hits = totals["cache_hits"]

    return {
        "requests_proxied": total,
        "tokens_in": totals["tokens_in"],
        "tokens_out": totals["tokens_out"],
        "tokens_saved_by_compression": totals["tokens_saved_by_compression"],
        "cache_hits": hits,
        "cache_hit_rate_pct": round((hits / total * 100), 1) if total > 0 else 0.0,
        "routes_to_cheap_model": tier_counts.get("simple", 0),
        "routes_to_medium_model": tier_counts.get("medium", 0),
        "routes_to_powerful_model": tier_counts.get("complex", 0),
        "fallback_model_used": totals["fallback_model_used"],
        # Illustrative only — priced at Groq's on-demand rate. If your
        # traffic actually runs on Groq's free developer tier, real
        # billed cost is $0 regardless of this number.
        "estimated_savings_usd": round(estimated_savings_usd, 4),
        # Models that had savings but no confirmed published Groq rate,
        # so they're excluded from estimated_savings_usd above rather
        # than guessed at — see core/pricing.py.
        "savings_excluded_unpriced_models": sorted(unpriced_models),
    }
