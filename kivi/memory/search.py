"""Retrieval over memories, plus one-hop graph expansion.

Two layers are searched for a reason. Memories are deduplicated and precise -
one row instead of forty episodes saying the same thing. Episodes are complete
and verbatim, and they are the *evidence*. The rule that keeps this honest:

    a memory is an index into episodes, never a replacement for them.

An answer is grounded in episodes even when a memory is what found them, which
is what makes provenance real - and what makes abstention meaningful, because a
memory whose source episodes are gone cannot be cited and so cannot support an
answer.

The graph is never searched, only traversed. An edge has no text to embed. Its
job is the case pure similarity handles badly: a fact spread over several
episodes, where each one is equally similar to the question and they compete for
the same slots.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import time
from dataclasses import dataclass, field

from kivi.llm.gemini import GeminiClient
from kivi.memory import policy
from kivi.retrieval.embed import cosine_ranking
from kivi.retrieval.search import RRF_K, fts_query
from kivi.llm.cache import unpack_vector

import numpy as np


@dataclass
class MemoryHit:
    memory_id: str
    type: str
    subject: str
    body: str
    confidence: float
    status: str
    score: float
    via: str                       # 'direct' | 'graph:<relation>'
    sources: list[str] = field(default_factory=list)
    due_at: str | None = None
    pinned: bool = False


@dataclass
class RecallResult:
    hits: list[MemoryHit]
    episode_ids: list[str]
    latency_ms: int
    expanded: int
    query: str

    def memory_ids(self) -> list[str]:
        return [hit.memory_id for hit in self.hits]


def _load_memory_vectors(
    conn: sqlite3.Connection, model: str, dim: int
) -> tuple[list[str], np.ndarray]:
    rows = conn.execute(
        "SELECT v.memory_id, v.vector FROM memory_vectors v"
        " JOIN memories m ON m.id = v.memory_id"
        " WHERE v.model = ? AND v.dim = ? AND m.status = 'active'",
        (model, dim),
    ).fetchall()
    if not rows:
        return [], np.zeros((0, dim), dtype=np.float32)
    ids = [row["memory_id"] for row in rows]
    matrix = np.array([unpack_vector(row["vector"]) for row in rows], dtype=np.float32)
    return ids, matrix


def _lexical_memories(
    conn: sqlite3.Connection, query: str, limit: int
) -> list[tuple[str, float]]:
    match = fts_query(query)
    if not match:
        return []
    try:
        rows = conn.execute(
            "SELECT m.id AS id, bm25(memories_fts) AS score"
            " FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid"
            " WHERE memories_fts MATCH ? AND m.status = 'active'"
            " ORDER BY score LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(row["id"], -float(row["score"])) for row in rows]


def _fetch(conn: sqlite3.Connection, memory_ids: list[str]) -> dict[str, sqlite3.Row]:
    if not memory_ids:
        return {}
    placeholders = ",".join("?" * len(memory_ids))
    return {
        row["id"]: row
        for row in conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", memory_ids
        ).fetchall()
    }


def expand_graph(
    conn: sqlite3.Connection, memory_ids: list[str], limit: int = 12
) -> list[tuple[str, str, float]]:
    """One hop out from the given memories. Returns (memory_id, relation, weight).

    Undirected in effect: asking about DSPM should reach the people working on
    it, and asking about a person should reach their projects, so both
    directions of each edge are followed.
    """
    if not memory_ids:
        return []
    placeholders = ",".join("?" * len(memory_ids))
    rows = conn.execute(
        f"""
        SELECT dst_id AS id, relation, weight FROM edges
         WHERE src_id IN ({placeholders}) AND dst_type = 'memory'
        UNION ALL
        SELECT src_id AS id, relation, weight FROM edges
         WHERE dst_id IN ({placeholders}) AND src_type = 'memory'
        """,
        (*memory_ids, *memory_ids),
    ).fetchall()

    seen = set(memory_ids)
    out: list[tuple[str, str, float]] = []
    for row in sorted(rows, key=lambda r: -r["weight"]):
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        out.append((row["id"], row["relation"], float(row["weight"])))
        if len(out) >= limit:
            break
    return out


def sources_for(conn: sqlite3.Connection, memory_ids: list[str]) -> dict[str, list[str]]:
    """Provenance behind each memory: dictations AND things said to Kivi.

    This used to filter on `source_kind = 'episode'`, which meant a fact the
    person asserted directly had complete provenance in storage and none once
    retrieved. Under the citation guard that is worse than cosmetic: an answer
    resting on such a fact has nothing it is allowed to cite, so it abstains on
    something the person told Kivi themselves.
    """
    if not memory_ids:
        return {}
    placeholders = ",".join("?" * len(memory_ids))
    rows = conn.execute(
        f"SELECT memory_id, source_kind, source_id FROM memory_sources"
        f" WHERE memory_id IN ({placeholders})",
        memory_ids,
    ).fetchall()
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row["memory_id"], []).append(row["source_id"])
    return out


def provenance_times(
    conn: sqlite3.Connection, memory_id: str
) -> tuple[str | None, str | None]:
    """(latest episode time, latest statement time) behind one memory.

    Recency is what decides a conflict, so both clocks have to be readable
    independently. Episodes carry the moment the person spoke; statements carry
    the moment they told Kivi.
    """
    episode = conn.execute(
        "SELECT MAX(e.ts) FROM memory_sources s JOIN episodes e ON e.id = s.source_id"
        " WHERE s.memory_id = ? AND s.source_kind = 'episode'",
        (memory_id,),
    ).fetchone()[0]
    statement = conn.execute(
        "SELECT MAX(st.ts) FROM memory_sources s"
        " JOIN statements st ON st.id = s.source_id"
        " WHERE s.memory_id = ? AND s.source_kind = 'statement'",
        (memory_id,),
    ).fetchone()[0]
    return episode, statement


def recall(
    conn: sqlite3.Connection,
    client: GeminiClient,
    query: str,
    *,
    limit: int = 8,
    types: list[str] | None = None,
    use_graph: bool = True,
    graph_limit: int = 12,
) -> RecallResult:
    """Search memories, then optionally widen by one graph hop.

    `use_graph=False` is the ablation arm: the assignment asks whether the graph
    earns its place, and that is only answerable by running both.
    """
    started = time.perf_counter()
    model = client.settings.embed_model
    dim = client.settings.embed_dim

    ids, matrix = _load_memory_vectors(conn, model, dim)
    rankings: dict[str, list[tuple[str, float]]] = {}
    if len(ids):
        vector = client.embed_one(query, task_type="RETRIEVAL_QUERY")
        rankings["vector"] = cosine_ranking(vector, ids, matrix, limit * 3)
    rankings["lexical"] = _lexical_memories(conn, query, limit * 3)

    fused: dict[str, float] = {}
    for ranking in rankings.values():
        for position, (memory_id, _) in enumerate(ranking, start=1):
            fused[memory_id] = fused.get(memory_id, 0.0) + 1.0 / (RRF_K + position)

    direct = sorted(fused.items(), key=lambda item: -item[1])[:limit]
    direct_ids = [item[0] for item in direct]

    expanded: list[tuple[str, str, float]] = []
    if use_graph and direct_ids:
        expanded = expand_graph(conn, direct_ids, limit=graph_limit)

    all_ids = direct_ids + [item[0] for item in expanded]
    rows = _fetch(conn, all_ids)
    provenance = sources_for(conn, all_ids)

    hits: list[MemoryHit] = []
    for memory_id, score in direct:
        row = rows.get(memory_id)
        if row is None or (types and row["type"] not in types):
            continue
        hits.append(_hit(row, score, "direct", provenance.get(memory_id, [])))

    for memory_id, relation, weight in expanded:
        row = rows.get(memory_id)
        if row is None or row["status"] != "active":
            continue
        if types and row["type"] not in types:
            continue
        # Graph hits rank below every direct hit: they are context reached by
        # association, not something the question actually asked for.
        hits.append(
            _hit(row, 0.0001 * weight, f"graph:{relation}", provenance.get(memory_id, []))
        )

    episode_ids: list[str] = []
    for hit in hits:
        for episode_id in hit.sources:
            if episode_id not in episode_ids:
                episode_ids.append(episode_id)

    return RecallResult(
        hits=hits,
        episode_ids=episode_ids,
        latency_ms=int((time.perf_counter() - started) * 1000),
        expanded=len(expanded),
        query=query,
    )


def _hit(row: sqlite3.Row, score: float, via: str, sources: list[str]) -> MemoryHit:
    return MemoryHit(
        memory_id=row["id"],
        type=row["type"],
        subject=row["subject"],
        body=row["body"] or "",
        confidence=float(row["confidence"]),
        status=row["status"],
        score=score,
        via=via,
        sources=sources,
        due_at=row["due_at"],
        pinned=bool(row["pinned"]),
    )


def mark_used(conn: sqlite3.Connection, memory_ids: list[str]) -> None:
    """Record that a memory was actually used in an answer.

    The long-term shape of the store depends on this: memories that are never
    retrieved decay in ranking, so the set self-prunes toward what earns its
    place rather than growing without limit.
    """
    if not memory_ids:
        return
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.executemany(
        "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
        [(now, memory_id) for memory_id in memory_ids],
    )
    conn.commit()
