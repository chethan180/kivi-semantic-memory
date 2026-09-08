"""Every tool is actually called, with a stub model.

These exist because `compose` shipped with `NameError: name 'json' is not
defined` on its happy path. Nothing caught it: the dispatcher wraps each handler
in try/except so a crashing tool returns `{"error": ...}` and the turn carries
on, which is right for keeping a conversation alive and terrible for noticing
that a tool has never worked. The failure surfaced as a wrong answer to a user,
several steps downstream, with memory ids embedded in the message.

So: call every handler, assert no error came back. No network - the client is a
stub - so these are fast enough to run on every change, which is the only kind of
test that would have caught this one.
"""

from __future__ import annotations

import datetime as dt

import pytest

from kivi.agent.tools import DECLARATIONS, Toolbox
from kivi.config import Settings
from kivi.db import connect, migrate


class StubClient:
    """Returns canned text. The point is exercising our code, not the model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls: list[str] = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)

        class Result:
            text = "Hi Amma, I will be late today."
            tokens_in = 10
            tokens_out = 8
            cost_usd = 0.0
            cached = False
            model = "stub"

        return Result()

    def embed_one(self, text, **kwargs):
        return [0.1] * self.settings.embed_dim

    def embed(self, texts, **kwargs):
        return [[0.1] * self.settings.embed_dim for _ in texts]


@pytest.fixture()
def box(tmp_path):
    settings = Settings(GEMINI_API_KEY="stub", KIVI_DB_PATH=str(tmp_path / "tools.db"))
    conn = connect(settings.resolved_db_path)
    migrate(conn)

    now = dt.datetime.now(dt.timezone.utc)
    for i in range(3):
        conn.execute(
            "INSERT INTO episodes (id, content_hash, ts, ts_epoch, app, raw_asr,"
            " formatted, meta_json, source, created_at, tz_offset_min, local_hour,"
            " local_date, recipient, recipient_norm)"
            " VALUES (?, ?, ?, ?, 'whatsapp', ?, ?, '{}', 'test', ?, 0, 9, ?,"
            " 'Amma', 'amma')",
            (f"ep_test{i:06d}", f"h{i}", now.isoformat(), int(now.timestamp()),
             f"hi amma message {i}", f"Hi Amma, message {i}.", now.isoformat(),
             now.date().isoformat()),
        )
    conn.commit()

    from kivi.memory.style import rebuild_profiles
    from kivi.retrieval.embed import embed_episodes

    client = StubClient(settings)
    # Embed them: vector search is the default ranker, so an unembedded fixture
    # would make every search return nothing and the tool would look broken when
    # it was only unindexed.
    embed_episodes(conn, client)
    rebuild_profiles(conn)

    yield Toolbox(conn, client, utterance="tell amma i will be late")
    conn.close()


# --- structure --------------------------------------------------------------


def test_every_declared_tool_has_a_handler(box):
    """A declared tool with no handler answers `unknown tool` at runtime only."""
    for declaration in DECLARATIONS:
        result = box.run(declaration["name"], {})
        assert "unknown tool" not in str(result), f"{declaration['name']} has no handler"


def test_an_undeclared_tool_is_refused(box):
    assert "unknown tool" in str(box.run("teleport", {}))


# --- each tool actually runs ------------------------------------------------


def test_search_dictations_runs(box):
    result = box.run("search_dictations", {"query": "message"})
    assert "error" not in result
    assert result["found"] >= 1


def test_recall_runs(box):
    result = box.run("recall", {"topic": "Amma"})
    assert "error" not in result


def test_compose_runs_and_records_the_message(box):
    """The exact path that shipped broken."""
    result = box.run(
        "compose", {"to": "Amma", "message": "I will be late", "app": "whatsapp"}
    )
    assert "error" not in result, result.get("error")
    assert result["draft"]
    assert result["saved_as"].startswith("cmp_")

    row = box.conn.execute(
        "SELECT recipient, text FROM composed_messages WHERE id = ?",
        (result["saved_as"],),
    ).fetchone()
    assert row is not None
    assert row["recipient"] == "Amma"


def test_a_composed_message_never_lands_in_the_dictations(box):
    """`episodes` means what this person actually dictated.

    Kivi's own output living there made every query that reads episodes
    responsible for filtering it out, and the cost of one query forgetting is
    the style profile learning from Kivi's prediction of the voice rather than
    the voice - confirming itself, and drifting.
    """
    from kivi.memory.style import rebuild_profiles

    before = rebuild_profiles(box.conn)[0].n_samples
    for _ in range(3):
        box.run("compose", {"to": "Amma", "message": "running late"})
    after = rebuild_profiles(box.conn)[0].n_samples

    assert after == before
    assert box.conn.execute("SELECT COUNT(*) FROM composed_messages").fetchone()[0] == 3
    assert box.conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE source = 'composed'"
    ).fetchone()[0] == 0


def test_redraft_runs(box):
    result = box.run(
        "redraft", {"record_ids": ["ep_test000000"], "target": "a meeting"}
    )
    assert "error" not in result
    assert result["draft"]


def test_redraft_without_ids_says_so(box):
    result = box.run("redraft", {"record_ids": [], "target": "a meeting"})
    assert "error" in result
    assert "search_dictations" in result["error"]


def test_remember_add_runs(box):
    result = box.run("remember", {
        "action": "add", "type": "entity", "subject": "Kiran",
        "entity_type": "person", "body": "backend engineer on Atlas",
        "relations": [{"src": "Kiran", "relation": "works_on", "dst": "Atlas"}],
    })
    assert "error" not in result, result.get("error")
    assert result["ok"] is True


def test_remember_forget_runs(box):
    box.run("remember", {"action": "add", "type": "entity", "subject": "Kiran",
                         "body": "engineer"})
    memory_id = box.conn.execute(
        "SELECT id FROM memories WHERE subject = 'Kiran'"
    ).fetchone()["id"]

    result = box.run("remember", {"action": "forget", "memory_id": memory_id})
    assert "error" not in result
    assert box.conn.execute(
        "SELECT status FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()["status"] == "forgotten"


# --- failures must be visible, not silent -----------------------------------


def test_a_failing_tool_is_recorded_and_rolled_back(box):
    """A crash returns an error AND releases the write lock.

    An unreleased transaction here surfaced as "database is locked" on the next
    model call - a failure that looks like a database fault and is really a tool
    that threw halfway through a write.
    """
    result = box.run("remember", {"action": "nonsense"})
    assert "error" in result
    box.conn.execute("SELECT 1")  # would raise if the lock were still held


def test_every_tool_call_is_logged_for_the_trace(box):
    box.run("recall", {"topic": "Amma"})
    box.run("search_dictations", {"query": "message"})
    assert [c["tool"] for c in box.calls] == ["recall", "search_dictations"]
    assert all("result" in c for c in box.calls)
