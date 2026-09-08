"""Episode embeddings.

Vectors are written to `episode_vectors`, an ordinary table, which is always the
source of truth. When sqlite-vec is available a `vec0` index is built over the
same rows as an accelerator. If the extension cannot load - some Python builds
have extension loading compiled out - search falls back to numpy over the same
table. At this corpus size that is a few milliseconds, so the fallback is a
performance difference, never a correctness one.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from typing import Callable

import numpy as np

from kivi.db.connection import vec_available
from kivi.llm.cache import pack_vector, unpack_vector
from kivi.llm.gemini import GeminiClient

log = logging.getLogger(__name__)

BATCH = 100


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def embedding_text(row: sqlite3.Row) -> str:
    """What actually gets embedded.

    The formatted output, not the raw ASR: it is what the person meant, with the
    recogniser's mangling of names already corrected. Raw ASR stays available to
    BM25, which is where a mis-heard spelling is sometimes exactly the useful key.
    """
    return row["formatted"] or row["raw_asr"] or ""


def ensure_vec_index(conn: sqlite3.Connection, dim: int) -> bool:
    """Create the vec0 index if sqlite-vec loaded. Returns whether it exists."""
    if not vec_available(conn):
        return False
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS episode_vec"
            f" USING vec0(embedding float[{dim}])"
        )
        conn.commit()
    except sqlite3.OperationalError as exc:
        log.info("could not create vec0 index (%s); using numpy fallback", exc)
        return False
    return True


def embed_episodes(
    conn: sqlite3.Connection,
    client: GeminiClient,
    *,
    batch: int = BATCH,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Embed every episode that does not have a current vector yet.

    Idempotent and resumable: a run that dies halfway leaves the vectors it
    already wrote, and the next run picks up only what is missing.
    """
    model = client.settings.embed_model
    dim = client.settings.embed_dim

    rows = conn.execute(
        "SELECT e.rowid AS rid, e.id, e.raw_asr, e.formatted FROM episodes e"
        " LEFT JOIN episode_vectors v"
        "   ON v.episode_id = e.id AND v.model = ? AND v.dim = ?"
        " WHERE v.episode_id IS NULL",
        (model, dim),
    ).fetchall()

    total = len(rows)
    if not total:
        return {"embedded": 0, "total": 0}

    has_vec = ensure_vec_index(conn, dim)
    now = _utcnow()
    done = 0

    for start in range(0, total, batch):
        chunk = rows[start : start + batch]
        vectors = client.embed(
            [embedding_text(row) for row in chunk],
            task_type="RETRIEVAL_DOCUMENT",
        )
        conn.executemany(
            "INSERT OR REPLACE INTO episode_vectors"
            " (episode_id, model, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
            [
                (row["id"], model, dim, pack_vector(vec), now)
                for row, vec in zip(chunk, vectors)
            ],
        )
        if has_vec:
            conn.executemany(
                "INSERT OR REPLACE INTO episode_vec (rowid, embedding) VALUES (?, ?)",
                [
                    (row["rid"], pack_vector(vec))
                    for row, vec in zip(chunk, vectors)
                ],
            )
        conn.commit()
        done += len(chunk)
        if progress:
            progress(done, total)

    return {"embedded": done, "total": total, "vec_index": int(has_vec)}


def memory_embedding_text(row: sqlite3.Row) -> str:
    """What a memory looks like to the embedder.

    Subject, body and aliases together: an entity found by its alias ("the DSPM
    work") should match as readily as by its canonical name, and the alias only
    exists in the text if we put it there.
    """
    parts = [row["subject"]]
    if row["entity_type"]:
        parts.append(f"({row['entity_type']})")
    if row["body"]:
        parts.append(row["body"])
    try:
        aliases = json.loads(row["aliases_json"] or "[]")
    except (TypeError, ValueError):
        aliases = []
    if aliases:
        parts.append("also known as " + ", ".join(aliases))
    return " - ".join(str(p) for p in parts if p)


def embed_memories(
    conn: sqlite3.Connection,
    client: GeminiClient,
    *,
    batch: int = BATCH,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Embed active memories that do not have a current vector."""
    model = client.settings.embed_model
    dim = client.settings.embed_dim

    rows = conn.execute(
        "SELECT m.id, m.subject, m.body, m.entity_type, m.aliases_json FROM memories m"
        " LEFT JOIN memory_vectors v"
        "   ON v.memory_id = m.id AND v.model = ? AND v.dim = ?"
        " WHERE v.memory_id IS NULL AND m.status = 'active'",
        (model, dim),
    ).fetchall()

    total = len(rows)
    if not total:
        return {"embedded": 0, "total": 0}

    now = _utcnow()
    done = 0
    for start in range(0, total, batch):
        chunk = rows[start : start + batch]
        vectors = client.embed(
            [memory_embedding_text(row) for row in chunk],
            task_type="RETRIEVAL_DOCUMENT",
        )
        conn.executemany(
            "INSERT OR REPLACE INTO memory_vectors"
            " (memory_id, model, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
            [
                (row["id"], model, dim, pack_vector(vec), now)
                for row, vec in zip(chunk, vectors)
            ],
        )
        conn.commit()
        done += len(chunk)
        if progress:
            progress(done, total)

    return {"embedded": done, "total": total}


def rebuild_vec_index(conn: sqlite3.Connection, dim: int) -> int:
    """Repopulate the vec0 index from episode_vectors. Used after a restore."""
    if not ensure_vec_index(conn, dim):
        return 0
    conn.execute("DELETE FROM episode_vec")
    rows = conn.execute(
        "SELECT e.rowid AS rid, v.vector FROM episode_vectors v"
        " JOIN episodes e ON e.id = v.episode_id WHERE v.dim = ?",
        (dim,),
    ).fetchall()
    conn.executemany(
        "INSERT INTO episode_vec (rowid, embedding) VALUES (?, ?)",
        [(row["rid"], row["vector"]) for row in rows],
    )
    conn.commit()
    return len(rows)


def embedding_coverage(conn: sqlite3.Connection, model: str, dim: int) -> tuple[int, int]:
    """(embedded, total) episodes, for `kivi doctor` and the eval report."""
    total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    embedded = conn.execute(
        "SELECT COUNT(*) FROM episode_vectors WHERE model = ? AND dim = ?",
        (model, dim),
    ).fetchone()[0]
    return int(embedded), int(total)


def load_matrix(
    conn: sqlite3.Connection, model: str, dim: int, episode_ids: list[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """Load vectors as a matrix for numpy search, optionally restricted to a set."""
    if episode_ids is None:
        rows = conn.execute(
            "SELECT episode_id, vector FROM episode_vectors WHERE model = ? AND dim = ?",
            (model, dim),
        ).fetchall()
    elif not episode_ids:
        return [], np.zeros((0, dim), dtype=np.float32)
    else:
        placeholders = ",".join("?" * len(episode_ids))
        rows = conn.execute(
            f"SELECT episode_id, vector FROM episode_vectors"
            f" WHERE model = ? AND dim = ? AND episode_id IN ({placeholders})",
            (model, dim, *episode_ids),
        ).fetchall()

    if not rows:
        return [], np.zeros((0, dim), dtype=np.float32)

    ids = [row["episode_id"] for row in rows]
    matrix = np.array(
        [unpack_vector(row["vector"]) for row in rows], dtype=np.float32
    )
    return ids, matrix


def cosine_ranking(
    query: list[float], ids: list[str], matrix: np.ndarray, limit: int
) -> list[tuple[str, float]]:
    """Exact cosine similarity. Used as the fallback and for filtered searches."""
    if matrix.shape[0] == 0:
        return []
    q = np.asarray(query, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    m_norms = np.linalg.norm(matrix, axis=1)
    denom = m_norms * q_norm
    denom[denom == 0] = 1e-9
    scores = (matrix @ q) / denom
    top = np.argsort(-scores)[:limit]
    return [(ids[i], float(scores[i])) for i in top]
