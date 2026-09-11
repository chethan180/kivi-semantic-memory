"""The "What Kivi knows" page, through the HTTP API it is built on.

Tested by the endpoint rather than in a browser: every card, button and drawer on
the page is one fetch to one of these, and the page renders exactly the fields
they return. Three of them were wrong when first checked end to end:

  - "Where from" on a fact the person told Kivi showed no evidence at all, and a
    "said in" button with an empty id, because the drawer read dictations only.
  - "Needs your decision" would have listed Kivi's own yes/no questions from a
    conversation as "You've said two different things".
  - "Stop doing this" lowered a preference's confidence, but drafting applied
    every active preference regardless, so the switch changed only a label.
"""

from __future__ import annotations

import datetime as dt

import pytest

from kivi.config import Settings
from kivi.db import connect, migrate
from kivi.memory import policy

NOW = dt.datetime.now(dt.timezone.utc)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import kivi.api as api

    settings = Settings(GEMINI_API_KEY="stub", KIVI_DB_PATH=str(tmp_path / "api.db"))
    conn = connect(settings.resolved_db_path)
    migrate(conn)
    now = NOW.isoformat()

    conn.execute(
        "INSERT INTO episodes (id, content_hash, ts, ts_epoch, app, raw_asr, formatted,"
        " meta_json, source, created_at, tz_offset_min, local_hour, local_date)"
        " VALUES ('ep_api000001', 'h1', ?, ?, 'slack', 'priya runs dspm',"
        " 'Priya runs DSPM.', '{}', 'test', ?, 0, 9, ?)",
        (now, int(NOW.timestamp()), now, NOW.date().isoformat()),
    )
    for mid, subject, body, source, confidence in (
        ("mem_api_dspm", "DSPM", "", "extraction", 0.75),
        ("mem_api_dev", "Dev", "owner of DSPM", "user", 1.0),
        ("mem_api_pref", "Slack messages", "keep them short", "extraction", 0.9),
    ):
        conn.execute(
            "INSERT INTO memories (id, type, subject, body, entity_type, canonical,"
            " confidence, status, pinned, source, created_at, valid_from)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'active', 0, ?, ?, ?)",
            (mid, "preference" if mid == "mem_api_pref" else "entity", subject, body,
             None if mid == "mem_api_pref" else "project", subject, confidence,
             source, now, now),
        )
    conn.execute(
        "INSERT INTO memory_sources (memory_id, source_kind, source_id, quote,"
        " char_start, char_end, created_at)"
        " VALUES ('mem_api_dspm', 'episode', 'ep_api000001', 'DSPM', 0, 0, ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO statements (id, session_id, turn_id, text, intent, ts, ts_epoch,"
        " created_at) VALUES ('say_api000001', NULL, NULL,"
        " 'assign dev as the owner of dspm', 'add', ?, ?, ?)",
        (now, int(NOW.timestamp()), now),
    )
    conn.execute(
        "INSERT INTO memory_sources (memory_id, source_kind, source_id, quote,"
        " char_start, char_end, created_at)"
        " VALUES ('mem_api_dev', 'statement', 'say_api000001',"
        " 'assign dev as the owner of dspm', 0, 0, ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO sessions (id, title, started_at, last_active_at)"
        " VALUES ('ses_api', NULL, ?, ?)",
        (now, now),
    )
    for cid, kind, question in (
        ("clr_dictation", "dictation", "You told me X, but a later dictation says Y."),
        ("clr_confirm", "confirm", "Dev is recorded as A. Replace that with B?"),
    ):
        conn.execute(
            "INSERT INTO clarifications (id, session_id, memory_id, question, status,"
            " created_at, kind) VALUES (?, 'ses_api', 'mem_api_dev', ?, 'open', ?, ?)",
            (cid, question, now, kind),
        )
    conn.commit()
    conn.close()

    monkeypatch.setattr(api, "_settings", settings)
    return TestClient(api.app)


def test_where_from_shows_what_the_person_told_kivi(client):
    detail = client.get("/api/memory/facts/mem_api_dev").json()

    assert detail["sources"], "a fact the person taught Kivi showed no evidence"
    assert detail["sources"][0]["contribution"] == "you told Kivi"
    assert detail["sources"][0]["record_id"] == "say_api000001"
    assert detail["revisions"][0]["source_record_id"] == "say_api000001", \
        "the 'said in' button had no record to open"
    # ...and that id opens in the same drawer as a dictation does.
    record = client.get("/api/records/say_api000001").json()["record"]
    assert record["app"] == "hey_kivi"


def test_an_extracted_fact_still_lists_its_dictations(client):
    detail = client.get("/api/memory/facts/mem_api_dspm").json()
    assert [s["record_id"] for s in detail["sources"]] == ["ep_api000001"]
    assert detail["revisions"][0]["source_record_id"] == "ep_api000001"


def test_needs_your_decision_lists_dictation_conflicts_only(client):
    decisions = client.get("/api/memory").json()["needs_decision"]
    assert [d["statement"] for d in decisions] == [
        "You told me X, but a later dictation says Y."
    ], "a question Kivi asked in a conversation reached the decisions list"


def test_the_preference_switch_moves_the_line_drafting_uses(client):
    def state():
        prefs = client.get("/api/memory").json()["preferences"]
        return next(p["state"] for p in prefs if p["pref_id"] == "mem_api_pref")

    assert state() == "active"
    assert client.post("/api/memory/preferences/mem_api_pref/toggle").json()["state"] == "candidate"
    assert state() == "candidate"

    import kivi.api as api
    conn = connect(api._settings.resolved_db_path)
    confidence = conn.execute(
        "SELECT confidence FROM memories WHERE id = 'mem_api_pref'"
    ).fetchone()[0]
    conn.close()
    assert confidence < policy.APPLY_PREFERENCE_AT, \
        "switched off in the interface but still above the line drafting applies"

    assert client.post("/api/memory/preferences/mem_api_pref/toggle").json()["state"] == "active"


def test_forget_removes_the_fact_and_keeps_the_dictation(client):
    assert client.delete("/api/memory/facts/mem_api_dspm").status_code == 200
    facts = client.get("/api/memory").json()["facts"]
    assert "mem_api_dspm" not in [f["fact_id"] for f in facts]
    assert client.get("/api/records/ep_api000001").status_code == 200, \
        "forgetting a conclusion removed the dictation behind it"
    assert client.delete("/api/memory/facts/mem_nope").status_code == 404
