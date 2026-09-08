"""Recency decides which evidence wins, and a newer dictation asks.

The rule under test:

    statement newer than the dictation  ->  the statement stands, silently
    dictation newer, and contradicting  ->  ask; change nothing until answered

Asking is only correct in the second case. A later dictation disagreeing with an
earlier instruction is genuinely ambiguous - the person may have changed their
mind, or may just have dictated a message that mentions the subject. Silently
keeping the old fact ignores evidence; silently overwriting discards an explicit
instruction. Neither is defensible, so nothing changes and a question is raised.

One thing these tests pin that is easy to get wrong: the conflict check sits
BEHIND the ordinary evidence threshold. A single contradicting dictation does not
raise a question - it is one occurrence, and interrupting someone over one
sentence is the same mistake as letting one angry message rewrite their tone.
The question is raised once the contradiction has corroboration.
"""

from __future__ import annotations

import datetime as dt

import pytest

from kivi.db import connect, migrate
from kivi.memory.extract import Candidate
from kivi.memory.promote import promote
from kivi.memory.search import provenance_times, sources_for


@pytest.fixture()
def conn(tmp_path):
    connection = connect(tmp_path / "conflict.db")
    migrate(connection)
    yield connection
    connection.close()


def _statement(conn, sid: str, when: str, text: str = "told Kivi") -> None:
    conn.execute(
        "INSERT INTO statements (id, session_id, turn_id, text, intent, ts,"
        " ts_epoch, created_at) VALUES (?, NULL, NULL, ?, 'add', ?, ?, ?)",
        (sid, text, when, int(dt.datetime.fromisoformat(when).timestamp()), when),
    )
    conn.commit()


def _episode(conn, eid: str, when: str, text: str) -> None:
    conn.execute(
        "INSERT INTO episodes (id, content_hash, ts, ts_epoch, raw_asr, formatted,"
        " meta_json, source, created_at) VALUES (?, ?, ?, ?, ?, ?, '{}', 'test', ?)",
        (eid, eid, when, int(dt.datetime.fromisoformat(when).timestamp()),
         text, text, when),
    )
    conn.commit()


def _candidate(**kw) -> Candidate:
    base = dict(
        type="entity", subject="Rohan", body="frontend engineer on DSPM",
        stance="asserted", explicit=True, quote="Rohan is on DSPM",
        entity_type="person", first_person=False, due_date=None, episode_id="",
    )
    base.update(kw)
    return Candidate(**base)


EARLY = "2026-09-01T10:00:00+00:00"
LATE = "2026-09-10T10:00:00+00:00"


# --- provenance is no longer dropped ----------------------------------------


def test_statement_provenance_survives_retrieval(conn):
    """The bug: a fact the person asserted had provenance in storage and none
    when recalled, which under the citation guard forces an abstention on
    something they said themselves."""
    _statement(conn, "say_1", EARLY)
    promote(conn, [_candidate()], from_user=True, statement_id="say_1")
    memory_id = conn.execute(
        "SELECT id FROM memories WHERE subject = 'Rohan'"
    ).fetchone()["id"]

    assert sources_for(conn, [memory_id])[memory_id] == ["say_1"]


def test_both_kinds_of_provenance_come_back(conn):
    _statement(conn, "say_1", EARLY)
    _episode(conn, "ep_1", EARLY, "Rohan is on DSPM")
    promote(conn, [_candidate()], from_user=True, statement_id="say_1")
    memory_id = conn.execute(
        "SELECT id FROM memories WHERE subject = 'Rohan'"
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO memory_sources (memory_id, source_kind, source_id, quote,"
        " char_start, char_end, created_at) VALUES (?, 'episode', 'ep_1', '', 0, 0, ?)",
        (memory_id, EARLY),
    )
    conn.commit()

    assert set(sources_for(conn, [memory_id])[memory_id]) == {"say_1", "ep_1"}
    episode_ts, statement_ts = provenance_times(conn, memory_id)
    assert episode_ts == EARLY and statement_ts == EARLY


# --- the rule ---------------------------------------------------------------


def test_older_dictation_loses_silently(conn):
    """Instruction is the more recent evidence, so no question is asked."""
    _statement(conn, "say_1", LATE)
    _episode(conn, "ep_o1", EARLY, "Rohan has moved to Beacon")
    _episode(conn, "ep_o2", "2026-09-02T10:00:00+00:00", "Rohan is on Beacon")
    promote(conn, [_candidate()], from_user=True, statement_id="say_1")

    promote(conn, [_candidate(body="engineer on Beacon", episode_id="ep_o1")])
    decisions = promote(
        conn, [_candidate(body="engineer on Beacon", episode_id="ep_o2")]
    )
    assert decisions[0].status == "rejected"
    assert "more recently" in decisions[0].reason
    assert conn.execute(
        "SELECT COUNT(*) FROM clarifications WHERE status = 'open'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT body FROM memories WHERE subject = 'Rohan' AND status = 'active'"
    ).fetchone()["body"] == "frontend engineer on DSPM"


def test_newer_dictation_asks_instead_of_deciding(conn):
    _statement(conn, "say_1", EARLY)
    _episode(conn, "ep_n1", LATE, "Rohan has moved to Beacon")
    _episode(conn, "ep_n2", "2026-09-11T10:00:00+00:00", "Rohan is on Beacon")
    promote(conn, [_candidate()], from_user=True, statement_id="say_1")

    # One contradicting dictation is an occurrence, not evidence.
    first = promote(conn, [_candidate(body="engineer on Beacon", episode_id="ep_n1")])
    assert first[0].status == "pending"
    assert conn.execute(
        "SELECT COUNT(*) FROM clarifications WHERE status = 'open'"
    ).fetchone()[0] == 0

    decisions = promote(
        conn, [_candidate(body="engineer on Beacon", episode_id="ep_n2")]
    )
    assert decisions[0].status == "pending"
    assert "asked rather than decided" in decisions[0].reason

    question = conn.execute(
        "SELECT question FROM clarifications WHERE status = 'open'"
    ).fetchone()
    assert question is not None
    assert "Rohan" in question["question"]

    # Crucially, nothing was changed while waiting for the answer.
    assert conn.execute(
        "SELECT body FROM memories WHERE subject = 'Rohan' AND status = 'active'"
    ).fetchone()["body"] == "frontend engineer on DSPM"


def test_agreeing_dictation_raises_nothing(conn):
    """Only a CONTRADICTION is worth interrupting someone about."""
    _statement(conn, "say_1", EARLY)
    _episode(conn, "ep_a1", LATE, "Rohan is on DSPM")
    _episode(conn, "ep_a2", "2026-09-11T10:00:00+00:00", "Rohan is on DSPM")
    promote(conn, [_candidate()], from_user=True, statement_id="say_1")

    promote(conn, [_candidate(episode_id="ep_a1")])
    decisions = promote(conn, [_candidate(episode_id="ep_a2")])
    assert decisions[0].status == "rejected"
    assert "agrees" in decisions[0].reason
    assert conn.execute(
        "SELECT COUNT(*) FROM clarifications WHERE status = 'open'"
    ).fetchone()[0] == 0


def test_the_same_question_is_not_raised_twice(conn):
    """Every later dictation on the subject must not add another question."""
    _statement(conn, "say_1", EARLY)
    _episode(conn, "ep_a", LATE, "Rohan moved to Beacon")
    _episode(conn, "ep_b", "2026-09-11T10:00:00+00:00", "Rohan is on Beacon")
    _episode(conn, "ep_c", "2026-09-12T10:00:00+00:00", "Rohan on Beacon again")
    promote(conn, [_candidate()], from_user=True, statement_id="say_1")

    promote(conn, [_candidate(body="engineer on Beacon", episode_id="ep_a")])
    promote(conn, [_candidate(body="engineer on Beacon", episode_id="ep_b")])
    promote(conn, [_candidate(body="engineer on Beacon", episode_id="ep_c")])

    assert conn.execute(
        "SELECT COUNT(*) FROM clarifications WHERE status = 'open'"
    ).fetchone()[0] == 1


def test_answering_resolves_the_question(conn):
    """The person answers in their own words; the question closes."""
    from kivi.agent.tools import Toolbox

    _statement(conn, "say_1", EARLY)
    _episode(conn, "ep_r1", LATE, "Rohan has moved to Beacon")
    _episode(conn, "ep_r2", "2026-09-11T10:00:00+00:00", "Rohan is on Beacon")
    promote(conn, [_candidate()], from_user=True, statement_id="say_1")
    promote(conn, [_candidate(body="engineer on Beacon", episode_id="ep_r1")])
    promote(conn, [_candidate(body="engineer on Beacon", episode_id="ep_r2")])
    assert conn.execute(
        "SELECT COUNT(*) FROM clarifications WHERE status = 'open'"
    ).fetchone()[0] == 1

    box = Toolbox(conn, client=None, utterance="Rohan is on Beacon now")
    closed = box._resolve_clarifications("Rohan", [])
    assert len(closed) == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM clarifications WHERE status = 'open'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM clarifications WHERE status = 'resolved'"
    ).fetchone()[0] == 1
