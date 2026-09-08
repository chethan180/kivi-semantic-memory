"""Retrieval-only evaluation.

Scored against the planted ground truth, before any memory or model reasoning is
involved. That separation matters: if Hey Kivi later answers a question wrongly,
this tells us whether the evidence was even retrievable. A failure here is a
retrieval bug; a failure there with the evidence present is a reasoning bug.

Reported per question kind, because the kinds fail differently. A distributed
fact needs all three of its episodes; a commitment needs one.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kivi.llm.gemini import GeminiClient
from kivi.retrieval.search import Filters, search


@dataclass
class QuestionResult:
    qid: str
    kind: str
    question: str
    supporting: list[str]
    retrieved: list[str]
    hit_at: dict[int, bool]
    recall: float
    latency_ms: int
    first_rank: int | None


@dataclass
class RetrievalEval:
    corpus: str
    config: str
    questions: list[QuestionResult] = field(default_factory=list)

    def summary(self, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, Any]:
        scored = [q for q in self.questions if q.supporting]
        if not scored:
            return {}
        latencies = sorted(q.latency_ms for q in self.questions)

        def pct(p: float) -> int:
            if not latencies:
                return 0
            idx = min(int(len(latencies) * p), len(latencies) - 1)
            return latencies[idx]

        out: dict[str, Any] = {
            "corpus": self.corpus,
            "config": self.config,
            "questions": len(scored),
            "recall": round(statistics.fmean(q.recall for q in scored), 3),
            "mrr": round(
                statistics.fmean(
                    (1.0 / q.first_rank) if q.first_rank else 0.0 for q in scored
                ),
                3,
            ),
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
        }
        for k in ks:
            out[f"hit@{k}"] = round(
                statistics.fmean(1.0 if q.hit_at.get(k) else 0.0 for q in scored), 3
            )
        return out

    def by_kind(self, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, dict[str, Any]]:
        kinds: dict[str, list[QuestionResult]] = {}
        for question in self.questions:
            if question.supporting:
                kinds.setdefault(question.kind, []).append(question)
        out: dict[str, dict[str, Any]] = {}
        for kind, items in sorted(kinds.items()):
            row: dict[str, Any] = {"n": len(items)}
            for k in ks:
                row[f"hit@{k}"] = round(
                    statistics.fmean(1.0 if q.hit_at.get(k) else 0.0 for q in items), 3
                )
            row["recall"] = round(statistics.fmean(q.recall for q in items), 3)
            out[kind] = row
        return out


def run_retrieval_eval(
    conn: sqlite3.Connection,
    client: GeminiClient,
    truth_path: Path,
    *,
    limit: int = 10,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    use_vector: bool = True,
    use_bm25: bool = True,
    use_recency: bool = True,
    config_name: str = "hybrid",
    source: str | None = None,
    progress=None,
) -> RetrievalEval:
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    corpus = truth.get("profile", truth_path.stem)

    # Map the corpus's external ids onto episode ids.
    filters = Filters(source=source) if source else Filters()
    id_map = {
        row["external_id"]: row["id"]
        for row in conn.execute(
            "SELECT external_id, id FROM episodes WHERE external_id IS NOT NULL"
        ).fetchall()
    }

    result = RetrievalEval(corpus=corpus, config=config_name)
    questions = truth.get("questions", [])

    for i, question in enumerate(questions):
        supporting = [
            id_map[e] for e in question.get("supporting", []) if e in id_map
        ]
        started = time.perf_counter()
        found = search(
            conn, client, question["question"], limit=limit, filters=filters,
            use_vector=use_vector, use_bm25=use_bm25, use_recency=use_recency,
        )
        latency = int((time.perf_counter() - started) * 1000)
        retrieved = found.ids()

        hit_at: dict[int, bool] = {}
        for k in ks:
            hit_at[k] = any(e in retrieved[:k] for e in supporting)

        first_rank = None
        for position, episode_id in enumerate(retrieved, start=1):
            if episode_id in supporting:
                first_rank = position
                break

        recall = (
            len(set(retrieved) & set(supporting)) / len(supporting)
            if supporting else 0.0
        )

        result.questions.append(
            QuestionResult(
                qid=question["qid"],
                kind=question["kind"],
                question=question["question"],
                supporting=supporting,
                retrieved=retrieved,
                hit_at=hit_at,
                recall=recall,
                latency_ms=latency,
                first_rank=first_rank,
            )
        )
        if progress:
            progress(i + 1, len(questions))

    return result


def write_results(result: RetrievalEval, out_dir: Path) -> Path:
    """Persist per-question detail, so a number can always be traced to a case."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"retrieval_{result.corpus}_{result.config}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for question in result.questions:
            handle.write(
                json.dumps(
                    {
                        "qid": question.qid,
                        "kind": question.kind,
                        "question": question.question,
                        "supporting": question.supporting,
                        "retrieved": question.retrieved,
                        "hit@1": question.hit_at.get(1),
                        "hit@5": question.hit_at.get(5),
                        "hit@10": question.hit_at.get(10),
                        "recall": round(question.recall, 3),
                        "first_rank": question.first_rank,
                        "latency_ms": question.latency_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path
