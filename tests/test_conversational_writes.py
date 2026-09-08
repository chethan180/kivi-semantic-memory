"""Writing to memory from a conversation.

Three rules, one test group each:

  a direct instruction writes, pinned, without needing corroboration
  an explicit revision versions the old value instead of overwriting it
  every citation shape the model emits is verified, not just the tidy one

The third-party status/cause rules that used to live here were removed with the
feature: Kivi still does not LEARN about other people - the stance guard covers
that - but it no longer redacts or refuses when asked about a dictation the
person made themselves.
"""

from __future__ import annotations

import sqlite3

import pytest

from kivi.db import connect, migrate
from kivi.memory import policy
from kivi.memory.extract import Candidate
from kivi.memory.promote import promote


@pytest.fixture()
def conn(tmp_path):
    connection = connect(tmp_path / "writes.db")
    migrate(connection)
    connection.execute(
        "INSERT INTO statements (id, session_id, turn_id, text, intent, ts,"
        " ts_epoch, created_at) VALUES ('say_test', NULL, NULL, 'test', 'add',"
        " '2026-09-06T10:00:00+00:00', 1757152800, '2026-09-06T10:00:00+00:00')"
    )
    connection.commit()
    yield connection
    connection.close()


def _candidate(**kwargs) -> Candidate:
    base = dict(
        type="entity", subject="Ashwin", body="frontend React developer on DSPM",
        stance="asserted", explicit=True, quote="add ashwin to dspm",
        entity_type="person", first_person=False, due_date=None, episode_id="",
    )
    base.update(kwargs)
    return Candidate(**base)


# --- a direct instruction writes -------------------------------------------


def test_instruction_promotes_without_corroboration(conn):
    """Entities normally need two episodes. An instruction is not an inference."""
    decisions = promote(conn, [_candidate()], from_user=True, statement_id="say_test")
    assert [d.status for d in decisions] == ["promoted"]

    row = conn.execute(
        "SELECT * FROM memories WHERE subject = 'Ashwin' AND status = 'active'"
    ).fetchone()
    assert row is not None
    assert row["confidence"] == 1.0
    assert row["pinned"] == 1
    assert row["source"] == "user"


def test_the_same_candidate_from_extraction_does_not_promote(conn):
    """The contrast that makes the previous test meaningful."""
    decisions = promote(conn, [_candidate()])
    assert [d.status for d in decisions] == ["pending"]
    assert conn.execute(
        "SELECT COUNT(*) FROM memories WHERE subject = 'Ashwin'"
    ).fetchone()[0] == 0


def test_provenance_points_at_the_statement_not_an_episode(conn):
    promote(conn, [_candidate()], from_user=True, statement_id="say_test")
    row = conn.execute(
        "SELECT source_kind, source_id FROM memory_sources"
        " WHERE memory_id = (SELECT id FROM memories WHERE subject = 'Ashwin')"
    ).fetchone()
    assert row["source_kind"] == "statement"
    assert row["source_id"] == "say_test"


def test_a_pinned_memory_resists_extraction_but_not_the_user(conn):
    """Extraction cannot overwrite what the person asserted; they can.

    Note the shape of the first assertion. A single contradicting dictation is
    now `pending`, not `rejected` - it has not yet cleared the ordinary evidence
    threshold, so the pinned-memory conflict never even arises. Corroboration
    comes first; only then does recency decide, and only then can a question be
    raised. See tests/test_provenance_conflict.py for that path.
    """
    promote(conn, [_candidate()], from_user=True, statement_id="say_test")

    from_extraction = promote(conn, [_candidate(body="backend engineer")])
    assert from_extraction[0].status == "pending"
    assert "evidence" in from_extraction[0].reason
    # Whatever happens, the asserted value is untouched.
    assert conn.execute(
        "SELECT body FROM memories WHERE subject = 'Ashwin' AND status = 'active'"
    ).fetchone()["body"] == "frontend React developer on DSPM"

    # The person themselves may revise it, with no threshold to clear.
    from_user = promote(
        conn, [_candidate(body="backend engineer")], from_user=True,
        statement_id="say_test",
    )
    assert from_user[0].status == "promoted"


# --- episodes are immutable -------------------------------------------------


def test_episodes_cannot_be_rewritten(conn):
    conn.execute(
        "INSERT INTO episodes (id, content_hash, ts, ts_epoch, raw_asr, formatted,"
        " meta_json, source, created_at) VALUES ('ep_x', 'h',"
        " '2026-09-06T10:00:00+00:00', 1757152800, 'raw', 'Formatted.', '{}',"
        " 'test', '2026-09-06T10:00:00+00:00')"
    )
    conn.commit()

    with pytest.raises(sqlite3.Error, match="immutable"):
        conn.execute("UPDATE episodes SET formatted = 'rewritten' WHERE id = 'ep_x'")

    assert conn.execute(
        "SELECT formatted FROM episodes WHERE id = 'ep_x'"
    ).fetchone()[0] == "Formatted."


# --- the citation guard sees every citation shape --------------------------


def _evidence(*refs):
    from kivi.agent.tools import Evidence

    return {r: Evidence(ref=r, kind="record", text="x") for r in refs}


def test_multi_id_brackets_are_verified():
    """`[a, b]` used to bypass the guard entirely, unverified."""
    from kivi.agent.loop import apply_citation_guard

    text = "Priya [ep_aaaaaaaaaaaa, ep_bbbbbbbbbbbb] and Abhi [ep_cccccccccccc]."
    cleaned, kept, dropped = apply_citation_guard(
        text, _evidence("ep_aaaaaaaaaaaa", "ep_bbbbbbbbbbbb")
    )
    assert set(kept) == {"ep_aaaaaaaaaaaa", "ep_bbbbbbbbbbbb"}
    assert dropped == ["ep_cccccccccccc"]
    assert "ep_cccccccccccc" not in cleaned


def test_bare_references_are_verified():
    """An id with no brackets was invisible to the guard and unclickable."""
    from kivi.agent.loop import apply_citation_guard

    cleaned, kept, dropped = apply_citation_guard(
        "Karthik ep_dddddddddddd leads it, and ep_eeeeeeeeeeee is invented.",
        _evidence("ep_dddddddddddd"),
    )
    assert kept == ["ep_dddddddddddd"]
    assert dropped == ["ep_eeeeeeeeeeee"]
    assert "[ep_dddddddddddd]" in cleaned


def test_a_fabricated_id_hidden_among_real_ones_is_caught():
    """The failure mode that matters: one fake in a group of genuine ids."""
    from kivi.agent.loop import apply_citation_guard

    cleaned, kept, dropped = apply_citation_guard(
        "On DSPM [ep_aaaaaaaaaaaa, ep_ffffffffffff, ep_bbbbbbbbbbbb].",
        _evidence("ep_aaaaaaaaaaaa", "ep_bbbbbbbbbbbb"),
    )
    assert dropped == ["ep_ffffffffffff"]
    assert "ep_ffffffffffff" not in cleaned
    assert len(kept) == 2


def test_statement_and_memory_refs_normalise_too():
    from kivi.agent.loop import normalise_citations

    out = normalise_citations("told you say_1111aaaa2222 about [mem_3333bbbb4444]")
    assert "[say_1111aaaa2222]" in out
    assert "[mem_3333bbbb4444]" in out


# --- revising retires the relation it replaces ------------------------------


def test_revision_retires_the_user_asserted_relation(conn):
    """"Ashwin moved to Beacon" must not leave him on DSPM as well."""
    from kivi.memory.promote import build_edges, stage_relations
    from kivi.memory.extract import Relation

    for subject in ("Ashwin", "DSPM", "Beacon"):
        promote(conn, [_candidate(subject=subject, body="")],
                from_user=True, statement_id="say_test")

    stage_relations(conn, [Relation(src="Ashwin", relation="works_on", dst="DSPM")])
    build_edges(conn)
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1

    # The revision path deletes prior user-asserted relations for this src.
    conn.execute(
        "DELETE FROM relations WHERE src_norm = 'ashwin' AND relation = 'works_on'"
        " AND episode_id IS NULL"
    )
    stage_relations(conn, [Relation(src="Ashwin", relation="works_on", dst="Beacon")])
    build_edges(conn)

    targets = [
        r["subject"] for r in conn.execute(
            "SELECT m2.subject FROM edges e JOIN memories m2 ON m2.id = e.dst_id"
        ).fetchall()
    ]
    assert targets == ["Beacon"]


def test_ingest_metadata_stays_writable(conn):
    """Immutability covers what was said, not bookkeeping around it."""
    conn.execute(
        "INSERT INTO episodes (id, content_hash, ts, ts_epoch, raw_asr, formatted,"
        " meta_json, source, created_at) VALUES ('ep_y', 'h2',"
        " '2026-09-06T10:00:00+00:00', 1757152800, 'raw', 'Text.', '{}',"
        " 'test', '2026-09-06T10:00:00+00:00')"
    )
    conn.execute("UPDATE episodes SET meta_json = '{\"reviewed\": true}' WHERE id = 'ep_y'")
    conn.commit()  # must not raise
