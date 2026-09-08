"""On-disk cache for model responses and embeddings, plus a daily call guard.

Three jobs, all of which the assignment cares about:

1. Reproducibility. Re-running the evaluation should produce byte-identical
   results without re-billing. `KIVI_OFFLINE=true` turns a cache miss into an
   error instead of an API call, so a replay is provably free of new inference.
2. Quota protection. Free-tier limits are per-project-per-model and small; a
   500-record ingest will blow through them if every re-run pays full price.
3. Honest cost accounting. A cache hit reports zero *new* cost while still
   reporting what the call originally cost.

The cache lives in its own tables in the main SQLite file. It creates them on
demand so Phase 1 does not depend on the Phase 2 migration runner; the migration
declares the same tables with `IF NOT EXISTS`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import struct
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key   TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- 'generate' | 'embed'
    response    TEXT NOT NULL,          -- JSON payload
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL,                   -- NULL when the model's price is unknown
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (content_hash, model, dim)
);

CREATE TABLE IF NOT EXISTS api_call_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    day        TEXT NOT NULL,           -- UTC date, YYYY-MM-DD
    model      TEXT NOT NULL,
    kind       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_api_call_log_day_model
    ON api_call_log (day, model);
"""


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def cache_key(kind: str, model: str, payload: Any) -> str:
    """Stable content hash. Sorted keys so dict ordering cannot cause a miss."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(f"{kind}|{model}|{blob}".encode("utf-8")).hexdigest()
    return digest


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LLMCache:
    """SQLite-backed cache. Safe to construct repeatedly; cheap to open."""

    def __init__(self, db_path: Path, enabled: bool = True) -> None:
        self.db_path = db_path
        self.enabled = enabled
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        # Without this the busy timeout is 0 and any contention fails instantly
        # with "database is locked". The cache writes its call log on its own
        # connection while the caller may be mid-write on theirs - which is
        # exactly what happens when a tool records a memory and the next model
        # call logs itself. WAL allows one writer plus readers; the timeout is
        # what makes a second writer wait its turn instead of dying.
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # --- generation responses ------------------------------------------------

    def get_response(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response, model, tokens_in, tokens_out, cost_usd"
                " FROM llm_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "response": json.loads(row["response"]),
            "model": row["model"],
            "tokens_in": row["tokens_in"],
            "tokens_out": row["tokens_out"],
            "cost_usd": row["cost_usd"],
        }

    def put_response(
        self,
        key: str,
        *,
        model: str,
        kind: str,
        response: Any,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float | None,
    ) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache"
                " (cache_key, model, kind, response, tokens_in, tokens_out, cost_usd, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    model,
                    kind,
                    json.dumps(response, ensure_ascii=False),
                    tokens_in,
                    tokens_out,
                    cost_usd,
                    _utcnow(),
                ),
            )

    # --- embeddings ----------------------------------------------------------

    def get_embedding(self, text: str, model: str, dim: int) -> list[float] | None:
        if not self.enabled:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT vector FROM embed_cache"
                " WHERE content_hash = ? AND model = ? AND dim = ?",
                (content_hash(text), model, dim),
            ).fetchone()
        return unpack_vector(row["vector"]) if row else None

    def put_embedding(
        self, text: str, model: str, dim: int, vector: list[float]
    ) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embed_cache"
                " (content_hash, model, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                (content_hash(text), model, dim, pack_vector(vector), _utcnow()),
            )

    # --- quota guard ---------------------------------------------------------

    def calls_today(self, model: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM api_call_log WHERE day = ? AND model = ?",
                (_today(), model),
            ).fetchone()
        return int(row["n"])

    def record_call(self, model: str, kind: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_call_log (day, model, kind, created_at)"
                " VALUES (?, ?, ?, ?)",
                (_today(), model, kind, _utcnow()),
            )

    # --- introspection, used by `kivi doctor` --------------------------------

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            responses = conn.execute("SELECT COUNT(*) AS n FROM llm_cache").fetchone()["n"]
            embeddings = conn.execute("SELECT COUNT(*) AS n FROM embed_cache").fetchone()["n"]
            today = conn.execute(
                "SELECT model, COUNT(*) AS n FROM api_call_log WHERE day = ?"
                " GROUP BY model ORDER BY n DESC",
                (_today(),),
            ).fetchall()
        return {
            "cached_responses": responses,
            "cached_embeddings": embeddings,
            "calls_today": {row["model"]: row["n"] for row in today},
        }
