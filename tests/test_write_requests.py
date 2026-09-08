"""Asking Kivi to WRITE something, end to end through the loop.

Two separate failures produced the same visible symptom - "I don't have anything
about that in your dictations" in reply to "write a message to Vikram" - and
fixing the first did not fix the symptom, which is why both are pinned here.

  routing   the prompt's request categories had drifted into contradiction, so
            the model treated a write request as a question and searched the
            dictations for a message that by definition did not exist yet.

  the guard the citation guard then abstained on the composed message even once
            routing was right. A message Kivi writes has nothing to cite - it is
            new text, not a claim about the person's history - but it reads as an
            assertion, so the uncited-claim branch threw it away and substituted
            the refusal. Every write request failed this way, including the ones
            that had routed correctly.

The loop tests below are scripted rather than live: a stub client returns a fixed
functionCall, so what is under test is our own control flow. Whether Gemini
actually picks `compose` is a separate question about the prompt, and lives in
the live test at the bottom - gated, because the default suite must stay fast and
free to run on every change.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pytest

from kivi.agent.loop import ask, is_write_request
from kivi.config import Settings
from kivi.db import connect, migrate


class ScriptedClient:
    """Plays back a fixed sequence of model turns.

    `generate` first serves the structured planning call, then serves both the
    synthesis step and `compose`'s own drafting call. The distinction matters:
    compose asks for a message, synthesis asks for an answer, and planning must
    return the JSON decision that drives the loop.
    """

    def __init__(
        self, settings: Settings, *, calls, draft, synthesis, plan=None
    ) -> None:
        self.settings = settings
        self._calls = list(calls)
        self._draft = draft
        self._synthesis = synthesis
        self._plan = plan if plan is not None else {
            "kind": "write", "reason": "stub", "recipient": "Vikram",
            "message": "finish the work in 2 days",
        }
        self.converse_count = 0
        self.prompts: list[str] = []
        self.planning_prompts: list[str] = []

    class _Result:
        def __init__(self, text: str) -> None:
            self.text = text
            self.tokens_in = 10
            self.tokens_out = 10
            self.cost_usd = 0.0
            self.cached = False
            self.model = "stub"

        def json(self):
            return json.loads(self.text)

    def converse(self, contents, **kwargs):
        self.converse_count += 1
        # Only the first round emits tool calls; a second round with the same
        # calls would loop forever against a stub that cannot change its mind.
        parts = (
            [{"functionCall": c} for c in self._calls]
            if self.converse_count == 1 else [{"text": ""}]
        )
        return {"content": {"role": "model", "parts": parts}}, self._Result("")

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if kwargs.get("schema") is not None:
            self.planning_prompts.append(prompt)
            return self._Result(json.dumps(self._plan))
        text = self._draft if prompt.startswith("Write a message") else self._synthesis
        return self._Result(text)

    def embed_one(self, text, **kwargs):
        return [0.1] * self.settings.embed_dim

    def embed(self, texts, **kwargs):
        return [[0.1] * self.settings.embed_dim for _ in texts]


@pytest.fixture()
def db(tmp_path):
    settings = Settings(GEMINI_API_KEY="stub", KIVI_DB_PATH=str(tmp_path / "w.db"))
    conn = connect(settings.resolved_db_path)
    migrate(conn)

    now = dt.datetime.now(dt.timezone.utc)
    for i in range(3):
        conn.execute(
            "INSERT INTO episodes (id, content_hash, ts, ts_epoch, app, raw_asr,"
            " formatted, meta_json, source, created_at, tz_offset_min, local_hour,"
            " local_date, recipient, recipient_norm)"
            " VALUES (?, ?, ?, ?, 'slack', ?, ?, '{}', 'test', ?, 0, 9, ?,"
            " 'Vikram', 'vikram')",
            (f"ep_w{i:06d}", f"wh{i}", now.isoformat(), int(now.timestamp()),
             f"vikram status {i}", f"Vikram, status update {i}.", now.isoformat(),
             now.date().isoformat()),
        )
    conn.commit()

    # Embed them. Vector search is the default ranker, so an unembedded fixture
    # makes every search return nothing - and a turn with no evidence abstains
    # for that reason instead of the one under test, which is how this fixture
    # first passed a guard test that was not actually exercising the guard.
    from kivi.retrieval.embed import embed_episodes

    embed_episodes(conn, ScriptedClient(settings, calls=[], draft="", synthesis=""))

    yield conn, settings
    conn.close()


def _client(settings, *, calls, draft="Vikram, this needs to be done in 2 days.",
            synthesis=None, plan=None):
    return ScriptedClient(
        settings, calls=calls, draft=draft,
        synthesis=synthesis if synthesis is not None else draft,
        plan=plan,
    )


COMPOSE_CALL = {
    "name": "compose",
    "args": {"to": "Vikram", "message": "finish the work in 2 days"},
}


# --- deciding before querying, in code --------------------------------------


@pytest.mark.parametrize("utterance", [
    "write a message to vikram to finish the work in 2 days",
    "send a message to my mother i will be late today",
    "send a message to appa asking if he took his medicine",
    "tell rahul the match is on saturday",
    "ask sanjay whether the release shipped",
    "message priya that the deck is ready",
    "remind arjun about the deposit",
    "draft an email to nikita about the offsite",
])
def test_these_are_recognised_as_write_requests(utterance):
    """The model routed these inconsistently, so the words decide instead.

    "send a message to appa asking if he took his medicine" went to `compose` on
    one run and to `recall` + `search_dictations` on the next - the word "asking"
    makes a request to write look like a question. A misroute tells the person
    "I don't have anything about that in your dictations", which is a refusal to
    write something new because it has not been written before.
    """
    assert is_write_request(utterance), utterance


@pytest.mark.parametrize("utterance", [
    "what did i tell bushan about the schema review",
    "did i send amma anything yesterday",
    "who is working on DSPM?",
    "what slack message did I send around 5 PM yesterday",
    "tell me who is on the DSPM project",
    "remind me what i said to priya",
    "show me my messages to rahul",
    "add ashwin to DSPM as a frontend react developer",
    "always keep my slack messages short",
])
def test_these_are_not_write_requests(utterance):
    """The false positives are what make forcing a tool safe.

    Forcing `compose` on a question would be a worse failure than the one being
    fixed - it would answer "who is on DSPM?" by writing a message to somebody.
    First person ("tell me", "remind me") and a leading question word are the two
    signals that separate them, and both are checked before the write pattern.
    """
    assert not is_write_request(utterance), utterance


# --- the regression ---------------------------------------------------------


def test_a_composed_message_reaches_the_person(db):
    """The bug: routing was right, and the guard discarded the result anyway."""
    conn, settings = db
    result = ask(conn, _client(settings, calls=[COMPOSE_CALL]),
                 "write a message to vikram to finish the work in 2 days")

    assert not result.abstained, result.abstain_reason
    assert "2 days" in result.answer
    assert "don't have anything" not in result.answer
    assert result.plan == {
        "kind": "write",
        "reason": "stub",
        "recipient": "Vikram",
        "message": "finish the work in 2 days",
        "failed": False,
    }


def test_the_planning_decision_drives_the_write_flow(db):
    """The scripted client must exercise the real planning call, not fallback."""
    conn, settings = db
    client = _client(
        settings,
        calls=[COMPOSE_CALL],
        plan={
            "kind": "write",
            "reason": "the request asks for a new message",
            "recipient": "Vikram",
            "message": "finish the work in 2 days",
        },
    )

    result = ask(conn, client, "write a message to vikram to finish the work in 2 days")

    assert client.planning_prompts == [
        "Request: write a message to vikram to finish the work in 2 days"
    ]
    assert result.plan["failed"] is False
    assert result.plan["kind"] == "write"
    assert "compose" in [call["tool"] for call in result.tool_calls]


class _RecordingClient(ScriptedClient):
    """Also keeps what the tool loop was actually sent."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.contents: list = []
        self.allowed: list = []

    def converse(self, contents, **kwargs):
        # A copy: the loop appends to this same list as the turn proceeds, so
        # holding the reference would record what it grew into rather than what
        # was sent.
        self.contents.append(list(contents))
        self.allowed.append(kwargs.get("allowed_tools"))
        return super().converse(contents, **kwargs)


def test_the_plan_is_given_to_the_tool_loop_not_just_recorded(db):
    """A decision the next step cannot see is not a decision.

    The plan has to reach the model that picks the tool, or classification
    happens twice - once here and once implicitly - and the two can disagree.
    """
    conn, settings = db
    client = _RecordingClient(
        settings, calls=[COMPOSE_CALL], draft="Vikram, finish in 2 days.",
        synthesis="Vikram, finish in 2 days.",
    )
    ask(conn, client, "write a message to vikram to finish the work in 2 days")

    sent = client.contents[0][-1]["parts"][0]["text"]
    assert "YOU HAVE ALREADY DECIDED THIS IS A WRITE" in sent
    assert "finish the work in 2 days" in sent


def test_a_question_plan_leaves_the_tool_choice_open(db):
    """Forcing must apply only to writes.

    Forcing `compose` on a question would be a worse bug than the one being
    fixed: "who is on DSPM?" would be answered by writing somebody a message.
    """
    conn, settings = db
    client = _RecordingClient(
        settings,
        calls=[{"name": "search_dictations", "args": {"query": "vikram"}}],
        draft="", synthesis="Vikram sent a status update [ep_w000000].",
        plan={"kind": "question", "reason": "asks what is known",
              "recipient": "", "message": ""},
    )
    result = ask(conn, client, "what did vikram say")

    assert all(a is None for a in client.allowed), client.allowed
    assert result.plan["kind"] == "question"
    assert "compose" not in [c["tool"] for c in result.tool_calls]


def test_a_failed_plan_falls_back_to_the_word_test(db):
    """A planning outage costs routing quality, never the fix itself.

    If planning failing meant routing reverted to the model's own judgement,
    the original bug would come back exactly when the system was already
    degraded. The deterministic recogniser covers that window.
    """
    from kivi.llm.gemini import LLMError

    class NoPlanning(_RecordingClient):
        def generate(self, prompt, **kwargs):
            if kwargs.get("schema") is not None:
                raise LLMError("planning is down")
            return super().generate(prompt, **kwargs)

    conn, settings = db
    client = NoPlanning(
        settings, calls=[COMPOSE_CALL], draft="Vikram, finish in 2 days.",
        synthesis="Vikram, finish in 2 days.",
    )
    result = ask(conn, client, "write a message to vikram to finish the work in 2 days")

    assert result.plan["failed"] is True
    assert client.allowed[0] == ["compose"], "fallback did not force the write"
    assert not result.abstained


def test_the_plan_is_written_to_the_trace(db):
    """Why an answer came out this way must be answerable after the fact.

    Before this, the only record of how a request was read was which tool ran
    first - so a misrouted turn looked identical to a turn that found nothing.
    """
    conn, settings = db
    result = ask(conn, _client(settings, calls=[COMPOSE_CALL]), "write to vikram")

    stored = json.loads(conn.execute(
        "SELECT candidates_json FROM traces WHERE id = ?", (result.trace_id,)
    ).fetchone()[0])
    assert stored["plan"]["kind"] == "write"
    assert stored["plan"]["reason"]


def test_the_composed_message_is_recorded_outside_the_dictations(db):
    """Kept, because it would have been sent - but not as something they said."""
    conn, settings = db
    ask(conn, _client(settings, calls=[COMPOSE_CALL]), "write to vikram")

    row = conn.execute(
        "SELECT recipient, text, style_samples FROM composed_messages"
    ).fetchone()
    assert row is not None
    assert row["recipient"] == "Vikram"
    assert conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE source = 'composed'"
    ).fetchone()[0] == 0


def test_blunt_content_is_written_not_refused(db):
    """It is their message to send. Kivi is not the arbiter of its tone."""
    conn, settings = db
    client = _client(
        settings,
        calls=[{"name": "compose",
                "args": {"to": "Vikram", "message": "he is fired"}}],
        draft="Vikram, we're letting you go, effective today.",
    )
    result = ask(conn, client, "send a message to vikram that he is fired")

    assert not result.abstained, result.abstain_reason
    assert "letting you go" in result.answer


def test_an_instruction_that_changes_nothing_says_so(db):
    """"Already recorded" is an outcome, not an absence of one.

    "add ashwin to DSPM" when Ashwin was already recorded promoted nothing, so
    the turn had no evidence and no artifact and fell through to "I don't have
    anything about that in your dictations" - a search-failed message for a turn
    that searched nothing, about a fact Kivi did in fact hold.
    """
    conn, settings = db

    def fresh():
        # A new client per turn: the stub only emits tool calls on its first
        # round, so reusing one would make the second turn call nothing at all
        # and abstain for a reason that has nothing to do with this test.
        return _client(
            settings,
            calls=[{"name": "remember", "args": {
                "action": "add", "type": "entity", "subject": "Ashwin",
                "entity_type": "person", "body": "frontend React developer on DSPM",
            }}],
            draft="", synthesis="Already recorded - Ashwin is on DSPM.",
            plan={"kind": "instruction", "reason": "states a new fact",
                  "recipient": "", "message": ""},
        )

    ask(conn, fresh(), "add ashwin to DSPM as a frontend react developer")

    # Twice: the second is the no-op that used to produce the wrong refusal.
    result = ask(conn, fresh(), "add ashwin to DSPM as a frontend react developer")

    assert not result.abstained, result.abstain_reason
    assert "don't have anything" not in result.answer


def test_the_remember_outcome_reaches_the_synthesis_step(db):
    """Whatever happened has to be sayable, including "nothing"."""
    conn, settings = db
    client = _client(
        settings,
        calls=[{"name": "remember", "args": {
            "action": "add", "type": "entity", "subject": "Kiran",
            "entity_type": "person", "body": "backend engineer",
        }}],
        draft="", synthesis="Recorded.",
        plan={"kind": "instruction", "reason": "states a fact",
              "recipient": "", "message": ""},
    )
    ask(conn, client, "kiran is a backend engineer")

    closing = [p for p in client.prompts if "WHAT KIVI RECORDED" in p]
    assert closing, "the outcome never reached synthesis"


# --- the guard still guards -------------------------------------------------


def test_an_uncited_claim_with_no_artifact_still_abstains(db):
    """The exemption is for written messages, not a hole in the guard.

    Same shape as the compose case - an assertion carrying no citation - but
    reached through a search, where a citation is exactly what should have been
    there. This must still abstain, or the fix above has quietly disabled the
    guarantee it was meant to preserve.
    """
    conn, settings = db
    client = _client(
        settings,
        calls=[{"name": "search_dictations", "args": {"query": "vikram"}}],
        synthesis="Vikram is running two days behind on the migration.",
    )
    result = ask(conn, client, "how is vikram doing?")

    assert result.abstained
    assert "cited nothing" in result.abstain_reason


def test_a_fabricated_citation_is_stripped_from_a_composed_message(db):
    """Artifacts skip the claim check, not the verification.

    A composed message needs no citation, but any id that does appear in one
    must still resolve - and an unresolvable id inside a message is worse than
    elsewhere, because the person is about to send it to somebody.
    """
    conn, settings = db
    client = _client(
        settings, calls=[COMPOSE_CALL],
        synthesis="Vikram, finish in 2 days. [ep_deadbeef1234]",
    )
    result = ask(conn, client, "write to vikram")

    assert not result.abstained
    assert "ep_deadbeef1234" not in result.answer
    assert "ep_deadbeef1234" in result.dropped_citations


def test_a_composed_message_carries_no_citation_at_all(db):
    """Ids in a message get sent to whoever it is addressed to.

    Observed: "Amma, I will be late today. [mem_d0b...] [mem_63e...]" - the ids
    were real and verifiable, so no guard objected, and they were still wrong to
    be there. A style profile shapes the wording; it is not a claim the message
    is making, so there is nothing for it to cite.
    """
    conn, settings = db
    client = _client(
        settings, calls=[COMPOSE_CALL],
        synthesis="Vikram, finish in 2 days.",
    )
    result = ask(conn, client, "write to vikram")

    assert result.citations == []
    assert "[" not in result.answer


def test_the_draft_survives_a_synthesis_step_that_returns_nothing(db):
    """A written message must not depend on a later model call to survive.

    Observed: compose produced "Amma, I will be late today.", synthesis then
    returned nothing, and the turn abstained on a finished draft sitting in the
    artifact. Asking a second call to faithfully reproduce text we already hold
    is risk taken for no gain.
    """
    conn, settings = db
    client = _client(settings, calls=[COMPOSE_CALL],
                     draft="Vikram, finish in 2 days.", synthesis="")
    result = ask(conn, client, "write to vikram")

    assert not result.abstained, result.abstain_reason
    assert result.answer == "Vikram, finish in 2 days."


def test_a_paraphrased_draft_is_replaced_by_the_real_one(db):
    """Synthesis may frame the message; it may not rewrite it.

    The draft is what was composed in this person's voice and saved as a
    dictation. A reworded version in the reply would be a different message from
    the one on the record.
    """
    conn, settings = db
    client = _client(
        settings, calls=[COMPOSE_CALL],
        draft="Vikram, finish in 2 days.",
        synthesis="I told Vikram you need it wrapped up by Thursday.",
    )
    result = ask(conn, client, "write to vikram")

    assert "Vikram, finish in 2 days." in result.answer


def test_a_record_id_never_survives_into_a_composed_message(db):
    """The person forwards this. An internal id would go with it.

    Observed: "...please finish the remediation tasks in 7 days
    ep_4b504036cb67e229. Thanks." The agent had searched before composing and
    pasted a retrieved id into the message body. Nothing caught it, because the
    draft bypasses the citation guard by design - the guard verifies claims, and
    a written message makes none.
    """
    conn, settings = db
    client = _client(
        settings, calls=[COMPOSE_CALL],
        draft="Vikram, finish in 2 days ep_4b504036cb67e229. Thanks.",
    )
    result = ask(conn, client, "write to vikram")

    assert "ep_4b504036cb67e229" not in result.answer
    assert result.answer.startswith("Vikram, finish in 2 days.")

    saved = conn.execute("SELECT text FROM composed_messages").fetchone()["text"]
    assert "ep_4b504036cb67e229" not in saved, "leaked id was stored too"


def test_the_draft_is_held_to_the_persons_words_not_the_agents(db):
    """The invention came back through the tool ARGUMENT, not the draft.

    "send message to finish tasks in the project in 7 days" produced a message
    about an asset discovery phase and compliance gaps. Compose had not invented
    them - the agent had, into the `message` argument, after reading DSPM
    dictations earlier in the turn. The guard was checking the draft against an
    argument that was itself already embellished.

    Their literal utterance is the only thing in the turn that cannot have been,
    so it goes in the prompt as what the content is held to.
    """
    conn, settings = db
    client = _client(
        settings,
        calls=[{"name": "compose", "args": {
            "to": "Vikram",
            "message": "given the asset discovery phase is complete, finish the "
                       "remediation tasks in 7 days",
        }}],
    )
    ask(conn, client, "send message to finish tasks in the project in 7 days")

    drafting = [p for p in client.prompts if p.startswith("Write a message")][0]
    assert "send message to finish tasks in the project in 7 days" in drafting
    assert "word for word" in drafting


# --- style must not become content ------------------------------------------


def test_the_drafting_prompt_forbids_inventing_content_for_a_style_rule(db):
    """The Vikram case: a learned rule demanded content the instruction lacked.

    Two rules were learned from 22 messages - "3-4 complex sentences" and "focus
    on DSPM, compliance and audits" - and asked to say only "finish the work in
    2 days", the model invented an audit and a data security review to satisfy
    them. Both are content instructions, and a style rule that dictates content
    will always force invention when the instruction is shorter than the rule.

    Fixed in two places. Here: the drafting prompt must state which side wins.
    """
    conn, settings = db
    conn.execute(
        "UPDATE recipient_styles SET rules_json = ? WHERE recipient_norm = 'vikram'",
        ('["Write 3-4 dense sentences about DSPM and compliance."]',),
    )
    conn.commit()

    client = _client(settings, calls=[COMPOSE_CALL])
    ask(conn, client, "write to vikram")

    drafting = [p for p in client.prompts if p.startswith("Write a message")][0]
    assert "CONTENT WINS" in drafting
    assert "DROPPED, not satisfied by invention" in drafting
    assert "must trace back to the instruction" in drafting


def test_style_learning_is_told_to_keep_topics_out_of_rules():
    """The root cause, fixed where the rules are written rather than applied.

    The learning prompt asked for "what subjects come up with this person", and
    the model dutifully emitted it as a `rules` entry - an imperative that later
    gets applied to someone else's content. Topic belongs in `distinctive`,
    which is descriptive and never followed.
    """
    from kivi.memory.style_llm import SYSTEM

    assert "NEVER WHAT THEY WRITE ABOUT" in SYSTEM
    assert "distinctive" in SYSTEM
    assert "never as a floor" in SYSTEM


# --- personalization is loaded by the tool, not by the planner ---------------


def test_compose_looks_up_the_recipient_style_itself(db):
    """Routing is a model decision; style is a lookup keyed on the recipient.

    Asserted because it is the reason there is no separate planning round trip:
    the model needs only the name to route, and the profile is addressable by
    that name afterwards. If this lookup ever moved into the prompt, every turn
    would pay for it.
    """
    from kivi.memory.style import rebuild_profiles

    conn, settings = db
    rebuild_profiles(conn)
    client = _client(settings, calls=[COMPOSE_CALL])
    ask(conn, client, "write to vikram")

    drafting = [p for p in client.prompts if p.startswith("Write a message")]
    assert drafting, "compose never called the model"
    assert "VIKRAM" in drafting[0]


# --- does the real model route it correctly? --------------------------------


@pytest.mark.skipif(
    not os.getenv("KIVI_LIVE_TESTS"),
    reason="live model call; set KIVI_LIVE_TESTS=1 to run",
)
def test_the_real_model_routes_a_write_request_to_compose(db):
    """The prompt half of the bug, which no stub can catch.

    Kept out of the default run deliberately: it costs a call and a few seconds,
    and the suite is only useful if it is cheap enough to run on every change.
    """
    from kivi.llm.gemini import GeminiClient

    conn, _ = db
    result = ask(conn, GeminiClient(Settings()),
                 "write a message to vikram to finish the work in 2 days")

    tools = [c["tool"] for c in result.tool_calls]
    assert "compose" in tools, tools
    assert "search_dictations" not in tools, "treated a write request as a question"
