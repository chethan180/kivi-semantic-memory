"""The ingest-side memory pipeline: salience -> extraction -> promotion.

Deliberately not agentic. One structured call per batch, a fixed sequence, no
model discretion about control flow. Across hundreds of records an agent loop
would be slower, unrepeatable and uncacheable, and the evaluation depends on
this stage producing the same result twice.

Every episode leaves a trace row, including the ones nothing was learned from.
"An engineer can inspect why memory did or did not affect a result" is only true
if the negative cases are recorded too.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

from kivi.llm.gemini import GeminiClient
from kivi.memory import policy
from kivi.memory.extract import BATCH_SIZE, Extraction, extract_batch, salient
from kivi.memory.promote import Decision, build_edges, promote, stage_relations

log = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    episodes: int = 0
    skipped: int = 0
    candidates: int = 0
    promoted: int = 0
    pending: int = 0
    rejected: int = 0
    edges: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cached_batches: int = 0
    profiles: int = 0
    styles_read: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.episodes} episodes, {self.skipped} skipped at Gate 1, "
            f"{self.candidates} candidates -> {self.promoted} promoted, "
            f"{self.pending} pending, {self.rejected} rejected"
        )


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _trace(
    conn: sqlite3.Connection,
    kind: str,
    subject_id: str,
    decision: str,
    reason: str,
    *,
    candidates: list | None = None,
    model: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float | None = None,
    latency_ms: int = 0,
    cached: bool = False,
) -> None:
    trace_id = "tr_" + hashlib.sha256(
        f"{kind}|{subject_id}|{_utcnow()}".encode()
    ).hexdigest()[:16]
    conn.execute(
        "INSERT INTO traces"
        " (id, kind, subject_id, input_json, candidates_json, decision, reason,"
        "  model, tokens_in, tokens_out, cost_usd, latency_ms, cached, created_at)"
        " VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trace_id, kind, subject_id,
            json.dumps(candidates or [], ensure_ascii=False, default=str),
            decision, reason, model, tokens_in, tokens_out, cost_usd,
            latency_ms, int(cached), _utcnow(),
        ),
    )


def _decisions_json(decisions: list[Decision]) -> list[dict]:
    return [
        {
            "subject": d.subject,
            "type": d.type,
            "status": d.status,
            "reason": d.reason,
            "evidence": d.evidence_count,
            "memory_id": d.memory_id,
            "superseded": d.superseded_id,
        }
        for d in decisions
    ]


def extract_all(
    conn: sqlite3.Connection,
    client: GeminiClient,
    *,
    limit: int | None = None,
    source: str | None = None,
    batch_size: int = BATCH_SIZE,
    model: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PipelineReport:
    """Run the pipeline over every episode not yet processed.

    Idempotent: an episode with an existing `extract` trace is skipped, so a run
    that dies halfway can simply be run again.
    """
    where = ["t.subject_id IS NULL"]
    params: list = []
    if source:
        where.append("e.source = ?")
        params.append(source)
    sql = (
        "SELECT e.id, e.ts, e.app, e.formatted, e.raw_asr FROM episodes e"
        " LEFT JOIN (SELECT DISTINCT subject_id FROM traces WHERE kind = 'extract') t"
        "   ON t.subject_id = e.id"
        f" WHERE {' AND '.join(where)}"
        " ORDER BY e.ts_epoch"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()

    report = PipelineReport()
    if not rows:
        return report

    # Gate 1, before any token is spent.
    recent: set[str] = set()
    keep: list[sqlite3.Row] = []
    for row in rows:
        report.episodes += 1
        ok, reason = salient(row["formatted"], recent)
        recent.add((row["formatted"] or "").strip().lower())
        if len(recent) > policy.DUPLICATE_WINDOW:
            recent.pop()
        if ok:
            keep.append(row)
        else:
            report.skipped += 1
            _trace(conn, "extract", row["id"], "skipped", f"Gate 1: {reason}")
    conn.commit()

    total = len(keep)
    for start in range(0, total, batch_size):
        batch = keep[start : start + batch_size]
        began = time.perf_counter()
        extractions, usage = extract_batch(client, batch, model=model)
        latency = int((time.perf_counter() - began) * 1000)

        if usage is not None:
            report.tokens_in += usage.tokens_in
            report.tokens_out += usage.tokens_out
            report.cost_usd += usage.cost_usd or 0.0
            report.cached_batches += int(usage.cached)

        all_relations = []
        for position, row in enumerate(batch):
            extraction = extractions.get(position, Extraction(episode_id=row["id"]))
            report.candidates += len(extraction.candidates)
            decisions = promote(conn, extraction.candidates)
            all_relations.extend(extraction.relations)

            for decision in decisions:
                if decision.status == "promoted":
                    report.promoted += 1
                elif decision.status == "pending":
                    report.pending += 1
                else:
                    report.rejected += 1
                    tag = decision.reason.split(":")[0].split(" - ")[0][:60]
                    report.reject_reasons[tag] = report.reject_reasons.get(tag, 0) + 1

            summary = (
                "no candidates" if not decisions
                else ", ".join(f"{d.status}:{d.subject}" for d in decisions[:6])
            )
            _trace(
                conn, "extract", row["id"],
                "learned" if any(d.status == "promoted" for d in decisions) else "nothing learned",
                summary,
                candidates=_decisions_json(decisions),
                model=usage.model if usage else None,
                tokens_in=(usage.tokens_in // len(batch)) if usage else 0,
                tokens_out=(usage.tokens_out // len(batch)) if usage else 0,
                cost_usd=((usage.cost_usd or 0.0) / len(batch)) if usage else None,
                latency_ms=latency // max(len(batch), 1),
                cached=bool(usage.cached) if usage else False,
            )

        stage_relations(conn, all_relations)
        conn.commit()
        if progress:
            progress(min(start + batch_size, total), total)

    # Build the graph only now. An entity's second episode may arrive hundreds of
    # records after the relation that mentioned it, so edges cannot be resolved
    # until every promotion that is going to happen has happened.
    report.edges = build_edges(conn)
    conn.commit()

    # Personalization is part of ingestion, not a separate chore someone has to
    # remember to run. Two different costs, so two different cadences:
    #
    #   counted features - free, no model call, recomputed every run
    #   model-read style - one call for everyone, only when a profile has gained
    #                      enough new messages to be worth re-reading
    report.profiles, report.styles_read = _learn_styles(conn, client, model=model)
    return report


# Re-read a person's style when their message count crosses one of these. Early
# points are close together because the first few messages change the picture a
# lot; later ones spread out because the fortieth message rarely does.
STYLE_CHECKPOINTS = (2, 4, 8, 16, 32, 64, 128)


def _learn_styles(
    conn: sqlite3.Connection, client: GeminiClient, model: str | None = None
) -> tuple[int, int]:
    """Refresh writing profiles. Returns (profiles built, descriptions re-read)."""
    from kivi.memory import style, style_llm

    profiles = style.rebuild_profiles(conn)
    if not profiles:
        return 0, 0

    # Only re-read when someone has crossed a checkpoint since last time. Without
    # this the model call happens on every ingest batch, which is both expensive
    # and pointless - a style does not change because three more messages arrived.
    due = [
        p for p in profiles
        if p.active and _crossed_checkpoint(conn, p.recipient_norm, p.n_samples)
    ]
    if not due:
        return len(profiles), 0

    learned = style_llm.learn_styles(conn, client, model=model)
    written = style_llm.save(conn, learned)
    for profile in due:
        conn.execute(
            "UPDATE recipient_styles SET last_read_at_n = ? WHERE recipient_norm = ?",
            (profile.n_samples, profile.recipient_norm),
        )
    conn.commit()
    return len(profiles), written


def _crossed_checkpoint(
    conn: sqlite3.Connection, recipient_norm: str, n_samples: int
) -> bool:
    row = conn.execute(
        "SELECT last_read_at_n FROM recipient_styles WHERE recipient_norm = ?",
        (recipient_norm,),
    ).fetchone()
    last = int(row["last_read_at_n"] or 0) if row else 0
    return any(last < point <= n_samples for point in STYLE_CHECKPOINTS)


def process_episode(
    conn: sqlite3.Connection, client: GeminiClient, episode_id: str
) -> PipelineReport:
    """Run the pipeline over one episode. Used by the UI and by tests."""
    row = conn.execute(
        "SELECT id, ts, app, formatted, raw_asr FROM episodes WHERE id = ?",
        (episode_id,),
    ).fetchone()
    if row is None:
        raise KeyError(episode_id)

    report = PipelineReport(episodes=1)
    ok, reason = salient(row["formatted"], set())
    if not ok:
        report.skipped = 1
        _trace(conn, "extract", episode_id, "skipped", f"Gate 1: {reason}")
        conn.commit()
        return report

    extractions, usage = extract_batch(client, [row])
    extraction = extractions.get(0, Extraction(episode_id=episode_id))
    report.candidates = len(extraction.candidates)
    decisions = promote(conn, extraction.candidates)
    for decision in decisions:
        if decision.status == "promoted":
            report.promoted += 1
        elif decision.status == "pending":
            report.pending += 1
        else:
            report.rejected += 1
    stage_relations(conn, extraction.relations)
    report.edges = build_edges(conn)
    _trace(
        conn, "extract", episode_id,
        "learned" if report.promoted else "nothing learned",
        ", ".join(f"{d.status}:{d.subject}" for d in decisions) or "no candidates",
        candidates=_decisions_json(decisions),
        model=usage.model if usage else None,
        tokens_in=usage.tokens_in if usage else 0,
        tokens_out=usage.tokens_out if usage else 0,
        cost_usd=usage.cost_usd if usage else None,
        cached=bool(usage.cached) if usage else False,
    )
    conn.commit()
    return report
