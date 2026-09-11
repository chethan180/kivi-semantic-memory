"""The answer-level evaluation: which questions it picks, and how it scores.

What Kivi answers is tested live by `kivi eval-agent`, which costs money. These
pin the parts that do not need a model - choosing the sample and judging an
answer - so the scoring rule is fixed by a test, not by whoever reads the output.
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from kivi.config import Settings
from kivi.db import connect, migrate
from kivi.evaluation.agent import (
    AgentCase, grounded_citations, load_cases, score,
)


@pytest.fixture()
def conn(tmp_path):
    settings = Settings(GEMINI_API_KEY="stub", KIVI_DB_PATH=str(tmp_path / "e.db"))
    connection = connect(settings.resolved_db_path)
    migrate(connection)
    now = dt.datetime.now(dt.timezone.utc)
    for i in range(6):
        connection.execute(
            "INSERT INTO episodes (id, external_id, content_hash, ts, ts_epoch, app,"
            " raw_asr, formatted, meta_json, source, created_at, tz_offset_min,"
            " local_hour, local_date)"
            " VALUES (?, ?, ?, ?, ?, 'slack', ?, ?, '{}', 'test', ?, 0, 9, ?)",
            (f"ep_e{i:06d}", f"x{i}", f"h{i}", now.isoformat(), int(now.timestamp()),
             f"raw {i}", f"text {i}", now.isoformat(), now.date().isoformat()),
        )
    connection.commit()
    yield connection
    connection.close()


def _truth(tmp_path, corpus, questions):
    path = tmp_path / f"{corpus}_ground_truth.json"
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")


def test_the_sample_takes_every_false_premise_and_spreads_the_rest(conn, tmp_path):
    for corpus, offset in (("alpha", 0), ("beta", 3)):
        _truth(tmp_path, corpus, [
            {"qid": f"{corpus}-c{i}", "kind": "commitment", "question": f"q{i}",
             "supporting": [f"x{offset + i}"]} for i in range(3)
        ] + [
            {"qid": f"{corpus}-f", "kind": "false_premise", "question": "none",
             "supporting": []},
        ])

    cases = load_cases(conn, tmp_path, per_kind=4, false_premise=None)
    commitments = [c for c in cases if c.kind == "commitment"]
    premises = [c for c in cases if c.kind == "false_premise"]

    assert len(premises) == 2, "a false-premise question was left out"
    assert len(commitments) == 4
    assert [c.corpus for c in commitments] == ["alpha", "beta", "alpha", "beta"], \
        "one corpus filled the sample"
    assert cases == load_cases(conn, tmp_path, per_kind=4, false_premise=None), \
        "the sample changed between runs"


def test_a_question_whose_evidence_was_never_imported_is_not_asked(conn, tmp_path):
    """Refusing it would be correct, so counting that as a failure tests the import."""
    _truth(tmp_path, "alpha", [
        {"qid": "a1", "kind": "commitment", "question": "q", "supporting": ["missing"]},
    ])
    assert load_cases(conn, tmp_path) == []


def test_citing_a_memory_drawn_from_the_evidence_counts_as_grounded(conn):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memories (id, type, subject, canonical, confidence, status,"
        " pinned, source, created_at, valid_from)"
        " VALUES ('mem_e1', 'entity', 'DSPM', 'DSPM', 0.75, 'active', 0,"
        " 'extraction', ?, ?)",
        (now, now),
    )
    conn.execute(
        "INSERT INTO memory_sources (memory_id, source_kind, source_id, quote,"
        " char_start, char_end, created_at)"
        " VALUES ('mem_e1', 'episode', 'ep_e000002', 'q', 0, 0, ?)",
        (now,),
    )
    conn.commit()

    grounded = grounded_citations(conn, ["mem_e1", "ep_e000005"], ["ep_e000002"])
    assert grounded == ["mem_e1"], "the memory's evidence was not followed"


CASE = AgentCase("alpha", "a1", "commitment", "what did I promise?", ["ep_e000001"],
                 expected="send the deck")
PREMISE = AgentCase("alpha", "f1", "false_premise", "what about Tokyo?", [])


@pytest.mark.parametrize("case, answer, grounded, mentions, ok", [
    (CASE, SimpleNamespace(citations=["ep_e000001"], abstained=False), ["ep_e000001"], 1.0, True),
    # The first version of this evaluation failed exactly this: a correct,
    # cited answer drawn from a dictation other than the planted one.
    (CASE, SimpleNamespace(citations=["ep_e000005"], abstained=False), [], 1.0, True),
    (CASE, SimpleNamespace(citations=["ep_e000005"], abstained=False), [], 0.0, False),
    (CASE, SimpleNamespace(citations=[], abstained=True), [], 0.0, False),
    (CASE, SimpleNamespace(citations=[], abstained=False), [], 1.0, False),
    (PREMISE, SimpleNamespace(citations=[], abstained=True), [], 0.0, True),
    (PREMISE, SimpleNamespace(citations=[], abstained=False), [], 0.0, True),
    (PREMISE, SimpleNamespace(citations=["ep_e000003"], abstained=False), [], 0.0, False),
], ids=[
    "cited the planted record and gave the answer",
    "gave the answer from a different real record",
    "cited records but gave the wrong answer",
    "refused an answerable question",
    "right words, no citation",
    "refused a false premise",
    "declined a false premise in its own words",
    "cited evidence for a false premise",
])
def test_scoring(case, answer, grounded, mentions, ok):
    passed, why = score(case, answer, grounded, mentions)
    assert passed is ok, why
    assert why, "every verdict must carry a reason"


@pytest.mark.parametrize("answer, expected, kind, want", [
    ("Sanjay, Priya and Umar are on DSPM.", "Sanjay, Priya, Umar", "distributed_fact", 1.0),
    ("Sanjay and Priya are on DSPM.", "Sanjay, Priya, Umar", "distributed_fact", 2 / 3),
    ("Always put the ask in the first line of your emails.",
     "put the ask in the first line of an email", "explicit_preference", 1.0),
    ("You said you would be sending the deck on Friday.", "send the deck", "commitment", 1.0),
    ("Priya is working on Beacon.", "Atlas", "knowledge_update", 0.0),
])
def test_the_expected_answer_is_matched_by_words(answer, expected, kind, want):
    from kivi.evaluation.agent import mentions_expected

    assert mentions_expected(answer, expected, kind) == pytest.approx(want)


def test_a_question_asked_of_two_corpora_with_different_answers_is_flagged(conn, tmp_path):
    """No answer can satisfy both once the corpora share a database."""
    _truth(tmp_path, "alpha", [{"qid": "a", "kind": "knowledge_update",
                                "question": "What is Priya working on now?",
                                "expected": "Atlas", "supporting": ["x0"]}])
    _truth(tmp_path, "beta", [{"qid": "b", "kind": "knowledge_update",
                               "question": "What is Priya working on now?",
                               "expected": "Beacon", "supporting": ["x3"]}])

    cases = load_cases(conn, tmp_path)
    assert [c.ambiguous for c in cases] == [True, True]
