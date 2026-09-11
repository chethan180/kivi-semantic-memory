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


# --- preferences: the switch in the interface is the switch drafting obeys ---
#
# "Stop doing this" in What Kivi knows lowers a preference's confidence. Drafting
# used to apply every active preference regardless, so the switch changed a
# label and nothing else - and inferred preferences the page described as "not
# being applied yet" were being applied.


def _preference(conn, memory_id, subject, body, confidence):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO memories (id, type, subject, body, canonical, confidence,"
        " status, pinned, source, created_at, valid_from)"
        " VALUES (?, 'preference', ?, ?, ?, ?, 'active', 0, 'extraction', ?, ?)",
        (memory_id, subject, body, subject, confidence, now, now),
    )
    conn.commit()


@pytest.mark.parametrize("tool, args", [
    ("redraft", {"record_ids": ["ep_test000000"], "target": "a note for the team"}),
    ("compose", {"to": "Amma", "message": "I will be late"}),
], ids=["redraft", "compose"])
def test_a_switched_off_preference_is_not_applied(box, tool, args):
    _preference(box.conn, "mem_pref_on", "Slack messages", "keep them short", 0.9)
    _preference(box.conn, "mem_pref_off", "Notion notes", "use bullet points", 0.4)

    box.run(tool, args)
    prompt = "\n".join(box.client.calls)

    assert "keep them short" in prompt, "a preference that is on was not applied"
    assert "use bullet points" not in prompt, "a switched-off preference was applied"


# --- search: the person's own words -----------------------------------------
#
# "who is running dspm" was searched by the agent as "DSPM", and the dictation
# saying who runs it never reached the results. The person's own words are now
# searched too. These pin the merge, not the ranking - search itself is tested
# in test_retrieval.py - so `search` is replaced with one that answers by caller.


def _hit(episode_id):
    from kivi.retrieval.search import Hit

    return Hit(
        episode_id=episode_id, score=0.0, ts="2026-09-02T03:35:00+00:00",
        app="gmail", formatted=f"text of {episode_id}", raw_asr="",
        local_ts="2026-09-02 09:05", local_hour=9,
    )


@pytest.fixture()
def scripted_search(monkeypatch):
    from types import SimpleNamespace

    import kivi.agent.tools as tools

    calls: list[dict] = []
    agent = [_hit(f"ep_agent{i:02d}") for i in range(11)]
    own = [_hit("ep_agent03"), _hit("ep_own01"), _hit("ep_own02"),
           _hit("ep_own03"), _hit("ep_own04")]

    def fake(conn, client, query, *, limit=10, filters=None, use_bm25=False, **kwargs):
        calls.append({"query": query, "limit": limit, "bm25": use_bm25,
                      "app": filters.app if filters else None})
        return SimpleNamespace(hits=(own if use_bm25 else agent)[:limit])

    monkeypatch.setattr(tools, "search", fake)
    return calls


def test_the_persons_own_words_are_searched_alongside_the_agents_query(
    box, scripted_search
):
    box.utterance = "who is running dspm"
    result = box.run("search_dictations", {"query": "DSPM"})
    ids = [d["id"] for d in result["dictations"]]

    assert ids[:11] == [f"ep_agent{i:02d}" for i in range(11)], "agent's results moved"
    assert ids[11:] == ["ep_own01", "ep_own02", "ep_own03"], "more than 3 added"
    assert len(ids) == len(set(ids)), "a result the agent already had was repeated"
    assert "ep_own01" in box.evidence, "an added result could not be cited"
    assert [c["query"] for c in scripted_search] == ["DSPM", "who is running dspm"]
    assert scripted_search[1]["bm25"] is True


def test_the_persons_words_are_searched_once_per_turn(box, scripted_search):
    """The agent may search several times; the person's question has not changed."""
    box.utterance = "who is running dspm"
    box.run("search_dictations", {"query": "DSPM"})
    box.run("search_dictations", {"query": "DSPM owner"})

    assert len([c for c in scripted_search if c["bm25"]]) == 1


def test_the_persons_words_are_searched_under_the_same_filters(box, scripted_search):
    """"Around 5 PM in Slack" must narrow this search too, never widen it."""
    box.utterance = "polish my slack update from 5 pm"
    box.run("search_dictations", {"query": "update", "app": "slack"})
    box.run("search_dictations", {"query": "update", "app": "gmail"})

    assert [c["app"] for c in scripted_search if c["bm25"]] == ["slack", "gmail"]


def test_the_agent_query_gets_eleven_results_and_nothing_extra_without_words(
    box, scripted_search
):
    box.utterance = ""
    box.run("search_dictations", {"query": "DSPM"})

    assert [c["limit"] for c in scripted_search] == [11]


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
