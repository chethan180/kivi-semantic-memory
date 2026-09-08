"""Memory-stage evaluation.

Three questions, each with a number:

1. Did the stance guard hold? Episodes planted as being ABOUT other people must
   produce no memory. With context modelling deliberately not built, this single
   check covers the system's entire privacy claim, so it is reported first and
   any failure is listed individually rather than summarised away.

2. Does the store stay small? Active memories should grow sublinearly with
   episodes. Linear growth means the gate is not gating.

3. Does the graph earn its place? Run recall with expansion on and off and
   compare, rather than asserting that a graph helps.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kivi.llm.gemini import GeminiClient
from kivi.memory import policy
from kivi.memory.search import recall


@dataclass
class StanceReport:
    """Leaks are split by severity, because they are not the same failure.

    A `content` leak means something sensitive about a third party - their
    health, their promotion, their money - became a memory. That is the failure
    the guard exists to prevent.

    A `name_only` leak means the episode was sensitive but the sole thing learned
    was that a person or project exists. Nothing sensitive was stored. Reporting
    these together as one number would overstate the harm; reporting only the
    first would hide that the boundary is fuzzy. Both are shown.
    """

    protected_episodes: int = 0
    content_leaks: list[dict[str, Any]] = field(default_factory=list)
    name_only_leaks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            self.protected_episodes + len(self.content_leaks) + len(self.name_only_leaks)
        )

    @property
    def content_precision(self) -> float:
        """The number that matters: no sensitive content stored."""
        if not self.total:
            return 1.0
        return (self.total - len(self.content_leaks)) / self.total

    @property
    def strict_precision(self) -> float:
        """Nothing at all learned from a sensitive episode."""
        return self.protected_episodes / self.total if self.total else 1.0


def evaluate_stance_guard(
    conn: sqlite3.Connection, truth_paths: list[Path]
) -> StanceReport:
    """Every episode marked must-not-learn should have produced zero memories."""
    report = StanceReport()
    external: list[str] = []
    for path in truth_paths:
        truth = json.loads(path.read_text(encoding="utf-8"))
        external.extend(truth.get("must_not_learn_episodes", []))
    if not external:
        return report

    placeholders = ",".join("?" * len(external))
    rows = conn.execute(
        f"""
        SELECT e.id, e.external_id, e.formatted,
               (SELECT COUNT(*) FROM memory_sources s WHERE s.source_id = e.id) AS leaked
          FROM episodes e
         WHERE e.external_id IN ({placeholders})
        """,
        external,
    ).fetchall()

    for row in rows:
        if not row["leaked"]:
            report.protected_episodes += 1
            continue

        memories = conn.execute(
            "SELECT DISTINCT m.id, m.type, m.subject, m.body FROM memory_sources s"
            " JOIN memories m ON m.id = s.memory_id WHERE s.source_id = ?",
            (row["id"],),
        ).fetchall()

        # Sensitive if anything beyond a bare entity name was stored: a
        # preference or commitment drawn from someone else's situation, or an
        # entity whose own text trips the ignore list.
        sensitive = [
            m for m in memories
            if m["type"] != "entity"
            or policy.ignore_match(f"{m['subject']} {m['body'] or ''}") is not None
        ]
        entry = {
            "episode": row["external_id"],
            "text": row["formatted"][:160],
            "memories": [
                {"id": m["id"], "type": m["type"], "subject": m["subject"]}
                for m in memories
            ],
        }
        if sensitive:
            report.content_leaks.append(entry)
        else:
            report.name_only_leaks.append(entry)
    return report


def memory_growth(conn: sqlite3.Connection, buckets: int = 10) -> list[dict[str, Any]]:
    """Active memories as a function of episodes ingested.

    Measured by creation order rather than replayed, which is an approximation:
    it shows the shape of growth, not what the store looked like at that instant.
    """
    total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    if not total:
        return []
    rows = conn.execute(
        "SELECT e.ts_epoch AS ts,"
        "       (SELECT COUNT(DISTINCT s.memory_id) FROM memory_sources s"
        "         JOIN episodes e2 ON e2.id = s.source_id"
        "        WHERE e2.ts_epoch <= e.ts_epoch) AS memories"
        "  FROM episodes e ORDER BY e.ts_epoch"
    ).fetchall()
    step = max(len(rows) // buckets, 1)
    out = []
    for i in range(step - 1, len(rows), step):
        out.append({"episodes": i + 1, "memories": int(rows[i]["memories"])})
    return out


@dataclass
class GraphAblation:
    question: str
    expected: list[str]
    found_without: int
    found_with: int
    expanded: int


def evaluate_graph(
    conn: sqlite3.Connection,
    client: GeminiClient,
    truth_paths: list[Path],
    *,
    limit: int = 8,
) -> list[GraphAblation]:
    """Does one graph hop recover facts that pure similarity splits up?

    Scored on distributed_fact questions only. Those are the case the graph was
    added for: several episodes each naming one member of a set, none naming the
    whole set, all equally similar to the question and so competing for the same
    retrieval slots.
    """
    out: list[GraphAblation] = []
    for path in truth_paths:
        truth = json.loads(path.read_text(encoding="utf-8"))
        plants = {p["plant_id"]: p for p in truth.get("plants", [])}
        for question in truth.get("questions", []):
            if question["kind"] != "distributed_fact":
                continue
            plant = plants.get(question.get("plant_id") or "")
            if not plant:
                continue
            members = [m.lower() for m in plant.get("detail", {}).get("members", [])]
            if not members:
                continue

            without = recall(conn, client, question["question"], limit=limit, use_graph=False)
            with_graph = recall(conn, client, question["question"], limit=limit, use_graph=True)

            def covered(result) -> int:
                text = " ".join(
                    f"{h.subject} {h.body}" for h in result.hits
                ).lower()
                return sum(1 for m in members if m in text)

            out.append(
                GraphAblation(
                    question=question["question"],
                    expected=members,
                    found_without=covered(without),
                    found_with=covered(with_graph),
                    expanded=with_graph.expanded,
                )
            )
    return out
