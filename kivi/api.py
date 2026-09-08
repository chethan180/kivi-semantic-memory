"""HTTP backend for the Kivi interface.

The frontend is plain HTML, CSS and JavaScript served from here, so the whole
product is one process and one command with no Node in the review path.

Naming note: the interface speaks in product terms - a *record* is a dictation,
a *fact* is something Kivi has learned - while the store underneath uses
`episodes` and `memories`. The translation lives in this file rather than
leaking either vocabulary into the other.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kivi import db as kivi_db
from kivi.config import get_settings
from kivi.llm.gemini import GeminiClient
from kivi.memory import policy

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Kivi", docs_url="/api/docs", openapi_url="/api/openapi.json")
_settings = get_settings()
_client = GeminiClient(_settings)


def db() -> sqlite3.Connection:
    """A fresh connection per request; uvicorn serves them on a thread pool."""
    return kivi_db.connect(_settings.resolved_db_path, check_same_thread=False)


def _statement(row: sqlite3.Row) -> str:
    """A memory in plain language, the way the person should read it."""
    subject = row["subject"] or ""
    body = (row["body"] or "").strip()
    if not body:
        return subject
    if body.lower().startswith(subject.lower()):
        return body
    return f"{subject}: {body}"


# ---------------------------------------------------------------- health ----


@app.get("/api/health")
def health() -> dict[str, Any]:
    conn = db()
    try:
        records = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        facts = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        ).fetchone()[0]
        unprocessed = conn.execute(
            "SELECT COUNT(*) FROM episodes e LEFT JOIN"
            " (SELECT DISTINCT subject_id FROM traces WHERE kind = 'extract') t"
            " ON t.subject_id = e.id WHERE t.subject_id IS NULL"
        ).fetchone()[0]
        return {"records": records, "facts": facts, "unprocessed": unprocessed}
    finally:
        conn.close()


# ------------------------------------------------------------------- ask ----


class AskBody(BaseModel):
    question: str
    session_id: str | None = None


@app.post("/api/ask")
def ask(body: AskBody) -> dict[str, Any]:
    """Answer a question inside a conversation.

    A session_id is always established, so follow-ups ("and what about Abhi?")
    can see what was just asked. That history is scoped strictly to this
    conversation: a different session sees none of it. Anything that should
    outlive the conversation becomes a fact, which every session can read.
    """
    from kivi.agent import ask as run_ask
    from kivi.agent.loop import ensure_session

    question = (body.question or "").strip()
    if not question:
        raise HTTPException(400, "empty question")

    conn = db()
    try:
        session_id = ensure_session(conn, body.session_id)
        answer = run_ask(conn, _client, question, session_id=session_id)
        return answer.to_dict()
    finally:
        conn.close()


@app.post("/api/sessions")
def new_session() -> dict[str, Any]:
    """Start a fresh conversation with no inherited context."""
    from kivi.agent.loop import ensure_session

    conn = db()
    try:
        return {"session_id": ensure_session(conn, None)}
    finally:
        conn.close()


@app.get("/api/sessions/{session_id}")
def session_turns(session_id: str) -> dict[str, Any]:
    """Replay one conversation. Only ever this one."""
    conn = db()
    try:
        rows = conn.execute(
            "SELECT idx, role, content, citations_json, trace_id FROM turns"
            " WHERE session_id = ? ORDER BY idx",
            (session_id,),
        ).fetchall()
        return {
            "session_id": session_id,
            "turns": [
                {
                    "idx": r["idx"], "role": r["role"], "content": r["content"],
                    "citations": json.loads(r["citations_json"] or "[]"),
                    "trace_id": r["trace_id"],
                }
                for r in rows
            ],
        }
    finally:
        conn.close()


# --------------------------------------------------------------- records ----


@app.get("/api/records")
def records(
    limit: int = 40, offset: int = 0, app_name: str | None = None, q: str | None = None
) -> dict[str, Any]:
    from kivi.retrieval.search import Filters, search

    conn = db()
    try:
        apps = [
            row["app"] for row in conn.execute(
                "SELECT DISTINCT app FROM episodes WHERE app IS NOT NULL ORDER BY app"
            ).fetchall()
        ]

        if q:
            result = search(
                conn, _client, q, limit=min(limit, 60),
                filters=Filters(app=app_name),
            )
            rows = [
                {
                    "record_id": hit.episode_id, "ts": hit.ts, "app": hit.app,
                    "destination": None, "formatted_text": hit.formatted,
                    "raw_asr": hit.raw_asr,
                }
                for hit in result.hits
            ]
            return {"records": rows, "apps": apps}

        sql = "SELECT id, ts, app, formatted, raw_asr FROM episodes WHERE 1=1"
        params: list = []
        if app_name:
            sql += " AND app = ?"
            params.append(app_name.lower())
        sql += " ORDER BY ts_epoch DESC LIMIT ? OFFSET ?"
        params += [limit, offset]

        rows = [
            {
                "record_id": row["id"], "ts": row["ts"], "app": row["app"],
                "destination": None, "formatted_text": row["formatted"],
                "raw_asr": row["raw_asr"],
            }
            for row in conn.execute(sql, params).fetchall()
        ]
        return {"records": rows, "apps": apps}
    finally:
        conn.close()


@app.get("/api/records/{record_id}")
def record_detail(record_id: str) -> dict[str, Any]:
    conn = db()
    try:
        # A citation may point at a dictation or at something the person told
        # Kivi directly. Both open the same drawer; the `hey_kivi` app value is
        # what the interface keys on to say "you said this to Kivi directly"
        # rather than presenting it as a recorded dictation.
        if record_id.startswith("say_"):
            said = conn.execute(
                "SELECT s.id, s.text, s.ts, s.intent FROM statements s WHERE s.id = ?",
                (record_id,),
            ).fetchone()
            if said is None:
                raise HTTPException(404, f"no statement {record_id}")
            learned = [
                {"fact_id": m["id"], "statement": _statement(m), "status": m["status"]}
                for m in conn.execute(
                    "SELECT DISTINCT m.id, m.subject, m.body, m.status"
                    " FROM memory_sources s JOIN memories m ON m.id = s.memory_id"
                    " WHERE s.source_id = ? AND s.source_kind = 'statement'",
                    (record_id,),
                ).fetchall()
            ]
            return {
                "record": {
                    "record_id": said["id"], "ts": said["ts"], "app": "hey_kivi",
                    "destination": None, "formatted_text": said["text"],
                    "raw_asr": said["text"], "speech_act": said["intent"],
                },
                "learned": learned,
                "ignored": [],
            }

        row = conn.execute(
            "SELECT id, ts, app, formatted, raw_asr FROM episodes WHERE id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, f"no record {record_id}")

        learned = [
            {
                "fact_id": m["id"],
                "statement": _statement(m),
                "status": m["status"],
            }
            for m in conn.execute(
                "SELECT DISTINCT m.id, m.subject, m.body, m.status"
                " FROM memory_sources s JOIN memories m ON m.id = s.memory_id"
                " WHERE s.source_id = ? AND s.source_kind = 'episode'",
                (record_id,),
            ).fetchall()
        ]

        ignored = [
            {
                "reason_code": (c["reason"] or "").split(":")[0].split(" - ")[0][:40],
                "reason_detail": c["reason"],
                "subject": c["subject"],
            }
            for c in conn.execute(
                "SELECT subject, reason FROM candidates"
                " WHERE episode_id = ? AND status = 'rejected'",
                (record_id,),
            ).fetchall()
        ]

        return {
            "record": {
                "record_id": row["id"], "ts": row["ts"], "app": row["app"],
                "destination": None, "formatted_text": row["formatted"],
                "raw_asr": row["raw_asr"], "speech_act": None,
            },
            "learned": learned,
            "ignored": ignored,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------- memory ----


@app.get("/api/memory")
def memory() -> dict[str, Any]:
    conn = db()
    try:
        facts = []
        for row in conn.execute(
            "SELECT m.*, (SELECT COUNT(*) FROM memory_sources s"
            "              WHERE s.memory_id = m.id) AS sources"
            " FROM memories m WHERE m.status = 'active' AND m.type != 'preference'"
            " ORDER BY m.type, m.subject"
        ).fetchall():
            revisions = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE type = ? AND subject = ?",
                (row["type"], row["subject"]),
            ).fetchone()[0]
            facts.append({
                "fact_id": row["id"],
                "statement": _statement(row),
                "claim_key": f"{row['type']}:{row['subject']}",
                "anchor_name": (row["entity_type"] or row["type"]).title(),
                "status": row["status"],
                "source_count": row["sources"],
                "revisions": revisions,
                "last_changed": row["valid_from"],
            })

        preferences = [
            {
                "pref_id": row["id"],
                "statement": _statement(row),
                # A preference the person stated outright is applied; one merely
                # inferred from behaviour stays a candidate until they say so.
                "state": "active" if row["confidence"] >= 0.85 else "candidate",
                "origin": "explicit" if row["confidence"] >= 0.85 else "inferred",
                "evidence_count": row["sources"],
            }
            for row in conn.execute(
                "SELECT m.*, (SELECT COUNT(*) FROM memory_sources s"
                "              WHERE s.memory_id = m.id) AS sources"
                " FROM memories m"
                " WHERE m.type = 'preference' AND m.status IN ('active', 'candidate')"
                " ORDER BY m.confidence DESC"
            ).fetchall()
        ]

        needs_decision = [
            {"fact_id": row["memory_id"], "statement": row["question"]}
            for row in conn.execute(
                "SELECT memory_id, question FROM clarifications WHERE status = 'open'"
            ).fetchall()
        ]

        return {
            "facts": facts,
            "preferences": preferences,
            "needs_decision": needs_decision,
        }
    finally:
        conn.close()


@app.get("/api/memory/facts/{fact_id}")
def fact_detail(fact_id: str) -> dict[str, Any]:
    conn = db()
    try:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (fact_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"no fact {fact_id}")

        # Walk the supersession chain so "how this changed" is real history
        # rather than a single current value presented as if it never moved.
        chain: list[sqlite3.Row] = []
        seen: set[str] = set()
        cursor: sqlite3.Row | None = row
        while cursor is not None and cursor["id"] not in seen:
            seen.add(cursor["id"])
            chain.append(cursor)
            parent = cursor["supersedes_id"]
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (parent,)
            ).fetchone() if parent else None

        revisions = []
        for item in chain:
            source = conn.execute(
                "SELECT source_id AS episode_id, quote FROM memory_sources"
                " WHERE memory_id = ? AND source_kind = 'episode'"
                " LIMIT 1",
                (item["id"],),
            ).fetchone()
            revisions.append({
                "value_display": _statement(item),
                "status": item["status"],
                "valid_from": item["valid_from"],
                "valid_to": item["valid_to"],
                "source_record_id": source["episode_id"] if source else "",
                "evidence_span": (source["quote"] if source else "") or "",
            })

        sources = [
            {
                "record_id": s["episode_id"], "ts": s["ts"], "app": s["app"],
                "formatted_text": s["formatted"], "evidence_span": s["quote"] or "",
                "contribution": "supports",
            }
            for s in conn.execute(
                "SELECT s.source_id AS episode_id, s.quote, e.ts, e.app, e.formatted"
                " FROM memory_sources s JOIN episodes e ON e.id = s.source_id"
                " WHERE s.memory_id = ? ORDER BY e.ts_epoch DESC LIMIT 25",
                (fact_id,),
            ).fetchall()
        ]

        return {
            "fact": {
                "fact_id": row["id"],
                "statement": _statement(row),
                "claim_key": f"{row['type']}:{row['subject']}",
                "status": row["status"],
            },
            "revisions": revisions,
            "sources": sources,
        }
    finally:
        conn.close()


@app.delete("/api/memory/facts/{fact_id}")
def forget_fact(fact_id: str) -> dict[str, Any]:
    conn = db()
    try:
        cur = conn.execute(
            "UPDATE memories SET status = 'forgotten', valid_to = ? WHERE id = ?",
            (dt.datetime.now(dt.timezone.utc).isoformat(), fact_id),
        )
        if not cur.rowcount:
            raise HTTPException(404, f"no fact {fact_id}")
        conn.commit()
        # The dictations behind it are deliberately untouched: the person said
        # what they said, and forgetting a conclusion is not rewriting history.
        return {"ok": True, "forgotten": fact_id}
    finally:
        conn.close()


@app.post("/api/memory/preferences/{pref_id}/toggle")
def toggle_preference(pref_id: str) -> dict[str, Any]:
    conn = db()
    try:
        row = conn.execute(
            "SELECT id, confidence, pinned FROM memories WHERE id = ?", (pref_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(404, f"no preference {pref_id}")
        applying = row["confidence"] >= 0.85
        conn.execute(
            "UPDATE memories SET confidence = ?, pinned = ?, source = 'user'"
            " WHERE id = ?",
            (0.4 if applying else 1.0, 0 if applying else 1, pref_id),
        )
        conn.commit()
        return {"ok": True, "state": "candidate" if applying else "active"}
    finally:
        conn.close()


# --------------------------------------------------------------- inspect ----


@app.get("/api/inspect/summary")
def inspect_summary() -> dict[str, Any]:
    conn = db()
    try:
        counts = {
            "records": conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
            "facts": conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status = 'active'"
            ).fetchone()[0],
            "fact_revisions": conn.execute(
                "SELECT COUNT(*) FROM memories WHERE status = 'superseded'"
            ).fetchone()[0],
            "entities": conn.execute(
                "SELECT COUNT(*) FROM memories WHERE type = 'entity' AND status = 'active'"
            ).fetchone()[0],
            "preferences": conn.execute(
                "SELECT COUNT(*) FROM memories WHERE type = 'preference' AND status = 'active'"
            ).fetchone()[0],
            "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        }
        reasons = [
            {"reason_code": (r["reason"] or "").split(":")[0].split(" - ")[0][:48],
             "n": r["n"]}
            for r in conn.execute(
                "SELECT reason, COUNT(*) AS n FROM candidates WHERE status = 'rejected'"
                " GROUP BY reason ORDER BY n DESC LIMIT 20"
            ).fetchall()
        ]
        operations = [
            {"op": r["status"], "n": r["n"]}
            for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM memories GROUP BY status"
            ).fetchall()
        ]
        usage = [
            {
                "purpose": r["kind"], "calls": r["n"],
                "tokens_in": r["ti"] or 0, "tokens_out": r["to_"] or 0,
                "avg_latency_ms": int(r["lat"] or 0),
                "cost_usd": round(r["cost"] or 0.0, 6),
            }
            for r in conn.execute(
                "SELECT kind, COUNT(*) AS n, SUM(tokens_in) AS ti,"
                " SUM(tokens_out) AS to_, AVG(latency_ms) AS lat,"
                " SUM(cost_usd) AS cost FROM traces GROUP BY kind ORDER BY cost DESC"
            ).fetchall()
        ]
        return {
            "counts": counts, "reasons": reasons, "operations": operations,
            "usage_by_purpose": usage,
            "db_kib": int(_settings.resolved_db_path.stat().st_size / 1024),
            "policy": policy.describe(),
        }
    finally:
        conn.close()


@app.get("/api/inspect/decisions")
def inspect_decisions(outcome: str = "rejected", limit: int = 60) -> dict[str, Any]:
    conn = db()
    try:
        rows = conn.execute(
            "SELECT episode_id, subject, reason, stance FROM candidates"
            " WHERE status = ? ORDER BY last_seen DESC LIMIT ?",
            (outcome, limit),
        ).fetchall()
        return {
            "decisions": [
                {
                    "record_id": r["episode_id"] or "",
                    "reason_code": (r["reason"] or "").split(":")[0].split(" - ")[0][:40],
                    "reason_detail": f"{r['subject']} — {r['reason']}",
                }
                for r in rows
            ]
        }
    finally:
        conn.close()


@app.get("/api/inspect/traces")
def inspect_traces(limit: int = 15) -> dict[str, Any]:
    conn = db()
    try:
        rows = conn.execute(
            "SELECT t.id, t.decision, t.reason, t.latency_ms, t.cost_usd,"
            "       a.query, a.abstained, a.abstain_reason"
            " FROM traces t LEFT JOIN answers a ON a.trace_id = t.id"
            " WHERE t.kind = 'answer' ORDER BY t.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {
            "traces": [
                {
                    "trace_id": r["id"],
                    "query": r["query"] or "",
                    "abstained": bool(r["abstained"]),
                    "abstain_reason": r["abstain_reason"] or "",
                    "latency_json": json.dumps({"total": r["latency_ms"]}),
                    "cost_usd": r["cost_usd"] or 0.0,
                }
                for r in rows
            ]
        }
    finally:
        conn.close()


@app.get("/api/inspect/traces/{trace_id}")
def inspect_trace(trace_id: str) -> dict[str, Any]:
    conn = db()
    try:
        row = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"no trace {trace_id}")
        try:
            detail = json.loads(row["candidates_json"] or "{}")
        except ValueError:
            detail = {}
        explanations = [
            f"{call['tool']}({', '.join(f'{k}={v!r}' for k, v in (call.get('args') or {}).items())})"
            for call in detail.get("tool_calls", [])
        ]
        return {
            "trace_id": row["id"],
            "decision": row["decision"],
            "reason": row["reason"],
            "constraints": {"explanations": explanations},
            "latency": {"total": row["latency_ms"]},
            "tokens": {"in": row["tokens_in"], "out": row["tokens_out"]},
            "cost_usd": row["cost_usd"],
            "detail": detail,
        }
    finally:
        conn.close()


# -------------------------------------------------------------- frontend ----

if FRONTEND.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND / "index.html")
