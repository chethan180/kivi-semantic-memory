"""Conversation scoping.

The rule these tests defend: a question can see the earlier turns of its OWN
conversation and nothing else. Follow-ups need that history to resolve at all
("and what about Atlas?"), but a second conversation inheriting it would mean
one person's stray phrasing silently steering an unrelated answer.

Anything that should outlive a conversation is promoted to a fact, which is
global, versioned, and correctable. Transcripts are working context; facts are
knowledge. These tests pin the boundary.
"""

from __future__ import annotations

import pytest

from kivi.agent.loop import append_turn, ensure_session, _history
from kivi.db import connect, migrate


@pytest.fixture()
def conn(tmp_path):
    connection = connect(tmp_path / "sessions.db")
    migrate(connection)
    yield connection
    connection.close()


def test_ensure_session_creates_and_is_idempotent(conn):
    first = ensure_session(conn, None)
    assert first.startswith("ses_")
    assert ensure_session(conn, first) == first
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_turns_are_ordered_within_a_session(conn):
    session = ensure_session(conn, None)
    append_turn(conn, session, "user", "first")
    append_turn(conn, session, "assistant", "second")
    append_turn(conn, session, "user", "third")

    rows = conn.execute(
        "SELECT idx, content FROM turns WHERE session_id = ? ORDER BY idx",
        (session,),
    ).fetchall()
    assert [r["idx"] for r in rows] == [0, 1, 2]
    assert [r["content"] for r in rows] == ["first", "second", "third"]


def test_history_is_scoped_to_one_session(conn):
    """The central guarantee: no conversation reads another's turns."""
    a = ensure_session(conn, None)
    b = ensure_session(conn, None)
    assert a != b

    append_turn(conn, a, "user", "Who is working on DSPM?")
    append_turn(conn, a, "assistant", "Priya and Abhi.")
    append_turn(conn, b, "user", "Something unrelated")

    history_a = _history(conn, a)
    history_b = _history(conn, b)

    assert len(history_a) == 2
    assert len(history_b) == 1
    joined_b = " ".join(t["parts"][0]["text"] for t in history_b)
    assert "DSPM" not in joined_b
    assert "Priya" not in joined_b


def test_history_is_oldest_first_and_role_mapped(conn):
    session = ensure_session(conn, None)
    append_turn(conn, session, "user", "q1")
    append_turn(conn, session, "assistant", "a1")

    history = _history(conn, session)
    assert [t["role"] for t in history] == ["user", "model"]
    assert history[0]["parts"][0]["text"] == "q1"


def test_history_is_bounded(conn):
    """A long conversation must not become an unbounded second store."""
    session = ensure_session(conn, None)
    for i in range(30):
        append_turn(conn, session, "user", f"turn {i}")

    history = _history(conn, session, max_turns=8)
    assert len(history) == 8
    # The bound keeps the most RECENT turns, not the oldest.
    assert history[-1]["parts"][0]["text"] == "turn 29"


def test_indices_do_not_collide_across_sessions(conn):
    a = ensure_session(conn, None)
    b = ensure_session(conn, None)
    append_turn(conn, a, "user", "a0")
    append_turn(conn, b, "user", "b0")
    append_turn(conn, a, "user", "a1")

    idx_a = [r["idx"] for r in conn.execute(
        "SELECT idx FROM turns WHERE session_id = ? ORDER BY idx", (a,)
    ).fetchall()]
    idx_b = [r["idx"] for r in conn.execute(
        "SELECT idx FROM turns WHERE session_id = ? ORDER BY idx", (b,)
    ).fetchall()]
    assert idx_a == [0, 1]
    assert idx_b == [0]


def test_session_title_comes_from_the_first_question(conn):
    session = ensure_session(conn, None)
    append_turn(conn, session, "user", "Who is working on DSPM?")
    append_turn(conn, session, "user", "And Atlas?")
    title = conn.execute(
        "SELECT title FROM sessions WHERE id = ?", (session,)
    ).fetchone()["title"]
    assert title == "Who is working on DSPM?"
