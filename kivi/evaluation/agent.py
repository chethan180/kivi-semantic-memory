"""Hey Kivi, end to end, on the planted questions.

The retrieval and memory evaluations stop short of the thing a person actually
sees: the answer. This one runs the whole turn - plan, tools, synthesis, the
citation guard - and scores what comes out against the planted ground truth.

  answerable     the answer must cite real records AND contain the planted
                 expected answer: every expected name for "who is working on X",
                 most of the expected words otherwise. Uncited or refused fails.
  false premise  the history holds no answer, so the answer must cite nothing.
                 A citation there means Kivi is presenting something the planted
                 truth says was never said.

A stricter second number is kept beside the first: whether the answer cited one
of the exact dictations the generator planted. It is reported, not used to pass,
and the reason is worth stating because the first version of this evaluation
used it and was wrong. The corpora restate facts in filler dictations, and the
review database holds all three planted corpora at once - so a correct, cited
answer often comes from a different dictation, or a different corpus, than the
planted one. Scored by planted ids alone, 31 of 40 correct-looking answers
failed. Some questions are also asked of more than one corpus with a different
right answer in each ("Who is working on DSPM?"); in a merged database no answer
can satisfy both, so those are flagged `ambiguous` and counted apart.

Every question writes one row: the question, the plan, every tool call, the
answer, what it cited and whether that traced to the planted records, whether
it refused, and latency, tokens and cost.

Sampled by default. Every planted question is about 300 turns and several
dollars; all false-premise questions plus ten of each answerable kind is 56
turns and about $1.75. The sample is deterministic, so two runs compare.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kivi.llm.gemini import GeminiClient

FALSE_PREMISE = "false_premise"
DISTRIBUTED = "distributed_fact"

# How much of the expected answer must appear. A distributed fact is the whole
# point of its question - every name has to be found - so it needs all of them.
# Elsewhere the expected text is a paraphrase ("put the ask in the first line of
# an email"), so most of its content words is the fair bar.
NEEDED = {DISTRIBUTED: 1.0}
NEEDED_DEFAULT = 0.6

_STOP = frozenset(
    "the a an to of in on for and my by with is it be at or i me your you that "
    "this are was were from as".split()
)


@dataclass
class AgentCase:
    corpus: str
    qid: str
    kind: str
    question: str
    supporting: list[str]
    expected: str | None = None
    ambiguous: bool = False


@dataclass
class AgentResult:
    corpus: str
    qid: str
    kind: str
    question: str
    expected: str | None
    ambiguous: bool
    supporting: list[str]
    plan_kind: str
    plan_reason: str
    tools: list[dict[str, Any]]
    answer: str
    citations: list[str]
    grounded: list[str]
    mentions: float
    dropped: list[str]
    abstained: bool
    abstain_reason: str
    ok: bool
    why: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_usd: float
    trace_id: str


@dataclass
class AgentEval:
    results: list[AgentResult] = field(default_factory=list)
    db_bytes_before: int = 0
    db_bytes_after: int = 0

    def by_kind(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for kind in sorted({r.kind for r in self.results}):
            rows = [r for r in self.results if r.kind == kind]
            out[kind] = {
                "n": len(rows),
                "passed": sum(r.ok for r in rows),
                "rate": round(sum(r.ok for r in rows) / len(rows), 3),
                "cited_planted": sum(bool(r.grounded) for r in rows),
                "ambiguous": sum(r.ambiguous for r in rows),
                "abstained": sum(r.abstained for r in rows),
                "p50_ms": _pct([r.latency_ms for r in rows], 0.5),
                "cost_usd": round(sum(r.cost_usd for r in rows), 4),
            }
        return out

    def summary(self) -> dict[str, Any]:
        if not self.results:
            return {}
        answerable = [r for r in self.results if r.kind != FALSE_PREMISE]
        clear = [r for r in answerable if not r.ambiguous]
        premises = [r for r in self.results if r.kind == FALSE_PREMISE]
        latencies = [r.latency_ms for r in self.results]
        return {
            "questions": len(self.results),
            "answerable_passed": sum(r.ok for r in answerable),
            "answerable": len(answerable),
            "answerable_rate": _rate(answerable),
            "unambiguous_passed": sum(r.ok for r in clear),
            "unambiguous": len(clear),
            "unambiguous_rate": _rate(clear),
            "cited_planted": sum(bool(r.grounded) for r in answerable),
            "cited_planted_rate": (
                round(sum(bool(r.grounded) for r in answerable) / len(answerable), 3)
                if answerable else 0.0
            ),
            "false_premise_refused": sum(r.ok for r in premises),
            "false_premise": len(premises),
            "false_premise_rate": _rate(premises),
            "p50_ms": _pct(latencies, 0.5),
            "p95_ms": _pct(latencies, 0.95),
            "tokens_in": sum(r.tokens_in for r in self.results),
            "tokens_out": sum(r.tokens_out for r in self.results),
            "cost_usd": round(sum(r.cost_usd for r in self.results), 4),
            "cost_per_question_usd": round(
                statistics.fmean(r.cost_usd for r in self.results), 5
            ),
            "db_growth_bytes": self.db_bytes_after - self.db_bytes_before,
        }


def _rate(rows: list[AgentResult]) -> float:
    return round(sum(r.ok for r in rows) / len(rows), 3) if rows else 0.0


def _pct(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p), len(ordered) - 1)]


# ---------------------------------------------------------------------------
# Choosing the questions
# ---------------------------------------------------------------------------


def load_cases(
    conn: sqlite3.Connection,
    corpus_dir: Path,
    *,
    per_kind: int | None = 10,
    false_premise: int | None = None,
) -> list[AgentCase]:
    """The planted questions to run, spread evenly across corpora.

    `per_kind` caps each answerable kind; `false_premise` caps the questions
    with no answer. None means all of them. Answerable questions whose
    supporting dictations were never imported are skipped - with no evidence in
    the database, a refusal would be correct and scoring it a failure would be
    testing the import, not Kivi.
    """
    id_map = {
        row["external_id"]: row["id"]
        for row in conn.execute(
            "SELECT external_id, id FROM episodes WHERE external_id IS NOT NULL"
        ).fetchall()
    }

    pools: dict[str, dict[str, list[AgentCase]]] = {}
    answers_by_text: dict[str, set[str]] = {}
    for truth_path in sorted(Path(corpus_dir).glob("*_ground_truth.json")):
        corpus = truth_path.stem.replace("_ground_truth", "")
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        for question in truth.get("questions", []):
            kind = question["kind"]
            expected = question.get("expected")
            if expected:
                answers_by_text.setdefault(
                    question["question"].strip().lower(), set()
                ).add(str(expected).strip().lower())
            supporting = [
                id_map[e] for e in question.get("supporting", []) if e in id_map
            ]
            if kind != FALSE_PREMISE and not supporting:
                continue
            pools.setdefault(kind, {}).setdefault(corpus, []).append(
                AgentCase(corpus, question["qid"], kind, question["question"],
                          supporting, expected)
            )

    cases: list[AgentCase] = []
    for kind in sorted(pools):
        by_corpus = [
            sorted(items, key=lambda c: c.qid)
            for _, items in sorted(pools[kind].items())
        ]
        cap = false_premise if kind == FALSE_PREMISE else per_kind
        cases += _round_robin(by_corpus, cap)

    for case in cases:
        case.ambiguous = len(answers_by_text.get(case.question.strip().lower(), ())) > 1
    return cases


def _round_robin(pools: list[list[AgentCase]], cap: int | None) -> list[AgentCase]:
    """Take one from each corpus in turn, so no single corpus fills the sample."""
    queues = [list(pool) for pool in pools]
    picked: list[AgentCase] = []
    while any(queues) and (cap is None or len(picked) < cap):
        for queue in queues:
            if queue and (cap is None or len(picked) < cap):
                picked.append(queue.pop(0))
    return picked


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def grounded_citations(
    conn: sqlite3.Connection, citations: list[str], supporting: list[str]
) -> list[str]:
    """Citations that trace to a planted dictation, directly or via a memory."""
    wanted = set(supporting)
    hits: list[str] = []
    for ref in citations:
        if ref in wanted:
            hits.append(ref)
        elif ref.startswith("mem_"):
            sources = conn.execute(
                "SELECT source_id FROM memory_sources"
                " WHERE memory_id = ? AND source_kind = 'episode'",
                (ref,),
            ).fetchall()
            if any(row[0] in wanted for row in sources):
                hits.append(ref)
    return hits


def mentions_expected(answer: str, expected: str | None, kind: str) -> float:
    """How much of the planted answer the reply contains, 0 to 1.

    Every name, for a distributed fact. Otherwise the expected text's content
    words, each matched on its first five letters so "email" finds "emails" and
    "send" finds "sending". Plain word matching, on purpose: it is checkable by
    reading, and it cannot be talked round the way a model judge can.
    """
    if not expected:
        return 0.0
    text = answer.lower()
    if kind == DISTRIBUTED:
        names = [n.strip().lower() for n in expected.split(",") if n.strip()]
        if not names:
            return 0.0
        return sum(
            1 for n in names if re.search(rf"\b{re.escape(n)}\b", text)
        ) / len(names)
    words = [
        w for w in re.findall(r"[a-z0-9]+", expected.lower())
        if len(w) >= 3 and w not in _STOP
    ]
    if not words:
        return 0.0
    return sum(
        1 for w in words if re.search(rf"\b{re.escape(w[:5])}", text)
    ) / len(words)


def score(
    case: AgentCase, answer: Any, grounded: list[str], mentions: float
) -> tuple[bool, str]:
    """Pass or fail, with the reason in words a reviewer can check."""
    if case.kind == FALSE_PREMISE:
        if answer.citations:
            return False, "cited records for something the history does not contain"
        return True, "made no supported claim"
    if answer.abstained:
        return False, "refused a question the history answers"
    if not answer.citations:
        return False, "answered without citing anything"
    if not case.expected:
        # No planted answer to compare against: the planted records are all
        # there is to go on.
        return (bool(grounded),
                "cited a planted dictation" if grounded
                else "cited only records that are not the planted ones")
    needed = NEEDED.get(case.kind, NEEDED_DEFAULT)
    if mentions < needed:
        return False, f"cited records, but the answer lacks the expected '{case.expected}'"
    return True, (
        "cited its sources and gave the expected answer"
        + ("" if grounded else ", from records other than the planted ones")
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_agent_eval(
    conn: sqlite3.Connection,
    client: GeminiClient,
    cases: list[AgentCase],
    *,
    db_path: Path | None = None,
    progress=None,
) -> AgentEval:
    from kivi.agent import ask

    evaluation = AgentEval()
    if db_path is not None and db_path.exists():
        evaluation.db_bytes_before = db_path.stat().st_size

    for i, case in enumerate(cases):
        started = time.perf_counter()
        # No session: each question is asked cold, as a reviewer would ask it,
        # so an earlier question cannot leak context into a later one.
        answer = ask(conn, client, case.question)
        latency = int((time.perf_counter() - started) * 1000)
        evaluation.results.append(
            result_for(conn, case, answer, latency)
        )
        if progress:
            progress(i + 1, len(cases))

    if db_path is not None and db_path.exists():
        evaluation.db_bytes_after = db_path.stat().st_size
    return evaluation


def result_for(
    conn: sqlite3.Connection, case: AgentCase, answer: Any, latency_ms: int
) -> AgentResult:
    """Score one answer. Separate from running it, so stored answers can be
    re-scored without paying for them again."""
    grounded = grounded_citations(conn, list(answer.citations), case.supporting)
    mentions = mentions_expected(answer.answer or "", case.expected, case.kind)
    ok, why = score(case, answer, grounded, mentions)
    plan = getattr(answer, "plan", {}) or {}
    return AgentResult(
        corpus=case.corpus, qid=case.qid, kind=case.kind,
        question=case.question, expected=case.expected, ambiguous=case.ambiguous,
        supporting=case.supporting,
        plan_kind=str(plan.get("kind", "")), plan_reason=str(plan.get("reason", "")),
        tools=[{"tool": c["tool"], "args": c["args"]} for c in answer.tool_calls],
        answer=answer.answer, citations=list(answer.citations),
        grounded=grounded, mentions=round(mentions, 3),
        dropped=list(answer.dropped_citations),
        abstained=bool(answer.abstained), abstain_reason=answer.abstain_reason or "",
        ok=ok, why=why, latency_ms=latency_ms,
        tokens_in=answer.tokens_in, tokens_out=answer.tokens_out,
        cost_usd=round(answer.cost_usd or 0.0, 6), trace_id=answer.trace_id or "",
    )


def write_results(evaluation: AgentEval, out_dir: Path) -> tuple[Path, Path]:
    """One row per question, plus the summary, so every number traces to a case."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "agent.jsonl"
    with open(rows_path, "w", encoding="utf-8") as handle:
        for result in evaluation.results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    summary_path = out_dir / "agent_summary.json"
    summary_path.write_text(
        json.dumps(
            {"summary": evaluation.summary(), "by_kind": evaluation.by_kind()},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return rows_path, summary_path
