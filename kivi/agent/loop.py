"""Hey Kivi: a bounded tool loop.

Bounded, not free-running. The assignment grades latency, and an agent allowed to
decide for itself when to stop has no latency you can quote. Four tool calls
across two rounds covers the worked example - search, then redraft what was found
- without letting a confused turn spiral.

Two rules make the answers trustworthy, and both are code rather than prompt:

  Citation guard   every reference in the answer must correspond to something a
                   tool actually returned this turn. A reference to anything else
                   is stripped, and an answer left with no support becomes an
                   abstention.

  Abstention       having no evidence is a correct outcome with its own path
                   through the code, not a phrasing the model is asked to prefer.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from kivi.agent.plan import Plan, make_plan
from kivi.agent.tools import DECLARATIONS, Evidence, Toolbox
from kivi.llm.gemini import GeminiClient, LLMError, function_calls

MAX_ROUNDS = 2
MAX_TOOL_CALLS = 4

CITATION_RE = re.compile(r"\[([a-z]{2,4}_[a-z0-9]+)\]")

# A reference id wherever it appears - inside a multi-id bracket, or bare.
_REF = r"(?:ep|mem|say|tr|ans)_[a-z0-9]{6,}"
_BRACKET_GROUP = re.compile(r"\[\s*(" + _REF + r"(?:\s*,\s*" + _REF + r")+)\s*\]")
_BARE_REF = re.compile(r"(?<![\[\w])(" + _REF + r")(?![\w\]])")


def normalise_citations(text: str) -> str:
    """Rewrite every citation into the single `[id]` form the guard checks.

    The model does not reliably emit one id per bracket. It writes
    `[ep_a, ep_b]` and sometimes a bare `ep_c` with no brackets at all, and
    neither form matched the guard's pattern - so those references were passed
    through completely unverified while looking exactly like verified ones. A
    fabricated id sitting inside a multi-id bracket would have reached the
    person with the guard reporting nothing wrong.

    Normalising first means the guard sees every reference, whatever shape the
    model chose. It also makes them all clickable, since the interface matches
    the same single-id form.
    """
    text = _BRACKET_GROUP.sub(
        lambda m: " ".join(f"[{ref.strip()}]" for ref in m.group(1).split(",")), text
    )
    return _BARE_REF.sub(lambda m: f"[{m.group(1)}]", text)

SYSTEM_TEMPLATE = """You are Kivi. You do three things for one person: answer
questions from their dictations, write messages for them, and keep track of what
they tell you.

TODAY IS {today} ({weekday}). Yesterday was {yesterday}.
Resolve every relative date the person uses - "yesterday", "last Tuesday", "this
morning" - into a YYYY-MM-DD calendar date yourself, and pass that date to the
tools. Never pass a relative phrase; the tools do not understand one.

You are talking to that person. Be direct and brief. Write the way a capable
colleague would answer across a desk - no preamble, no restating the question,
no "Based on your dictations". Just the answer.

HOW TO WORK
- Use the tools to find evidence before answering. Do not answer from memory of
  this conversation alone when a tool could check.
- `recall` for what is known about a person, project, preference or promise.
- `search_dictations` for what was actually said, or to find something to work
  from. It filters by app and by local hour, so "around 5 PM yesterday in Slack"
  is one call.
- `redraft` only after you have the ids of what to rewrite.

CITATIONS
- Cite every claim with the id of the thing it came from, in square brackets:
  [ep_1a2b3c] for a dictation, [mem_4d5e6f] for something Kivi has learned.
- Only cite ids that appeared in a tool result this turn. Never invent one.
- Put the citation right after the claim it supports.

WHEN THE ANSWER IS NOT THERE
- If the dictations do not contain the answer, say so plainly and stop. Do not
  offer a guess, a likely answer, or a reason it might be one thing or another.
- A question can be about something real and still have no answer here. Saying
  "you haven't mentioned that" is a correct, useful reply.
- Never answer a question whose premise you have not verified.

WHAT KIND OF REQUEST IS THIS? Decide before you touch a tool.

There are THREE kinds, and mistaking one for another is the most common way to
get this wrong.

1. A REQUEST TO WRITE. "Write a message to Vikram saying the work must be done
   in two days." "Tell Amma I'll be late." "Send Rahul something about Saturday."
   Call `compose` with who it is for and what it must say. Then give them the
   message.

   A message that ASKS the recipient something is still a message to write.
   "Ask Appa if he took his medicine", "check with Sanjay whether the release
   shipped" - the question is inside the message, addressed to them, not to
   you. Route it to `compose` like any other.

   These are NOT questions. Do not search their dictations to find out whether
   they already said it - they are asking you to write something new, and
   whether they have written it before is irrelevant. Answering "you haven't
   mentioned that" to a request to write is always wrong.

   Never write the message yourself in your reply. `compose` is what knows how
   this person writes to each recipient; a message you write directly is in
   nobody's voice and is not saved.

   Write what they asked, in their voice. It is their message to send. Do not
   soften it, refuse it, or add caveats because the content seems blunt.

2. A QUESTION about what is known or what was said. "Who is on DSPM?" "What did
   I tell Bushan?" Search, recall, answer with citations, and abstain if the
   history does not contain it.

3. An INSTRUCTION that states a fact or asks you to change what Kivi believes:
  "add Ashwin to DSPM as a frontend React developer", "Priya is on Atlas now",
   "always keep my Slack messages short", "forget that". Call `remember`, then
   confirm in one line what you recorded, citing the [say_...] id.

   Never call `remember` for a question, even one that mentions something in
   passing. "What did I tell Bushan about the schema review he was blocked on?"
   is a question - the part about being blocked is context for the search, not a
   fact to store. Writing down everything said in passing would fill their
   memory with chatter.

   If `remember` returns needs_confirmation, do NOT write. Tell them what Kivi
   currently believes and what they have just said, and ask which is right."""


# "send/write/message/text/tell/ask/remind <someone>", where someone is not the
# person themselves. The trailing name is required: "tell me who is on DSPM" and
# "remind me what I said" are questions, and excluding the first person is what
# separates them.
_WRITE_REQUEST = re.compile(
    r"\b(?:send|write|draft|compose|message|text|tell|ask|remind|reply|respond)\b"
    r"(?:\s+(?:a|an|the|my|another))?"
    r"(?:\s+(?:message|note|mail|email|text|reply|sms))?"
    r"(?:\s+(?:to|for))?\s+"
    r"(?!me\b|myself\b|us\b|kivi\b)[a-z]",
    re.IGNORECASE,
)


def is_write_request(utterance: str) -> bool:
    """Whether this is unambiguously "write X a message", from the words alone.

    Routing was left entirely to the model, and at this it is unreliable in a
    way that is worse than being wrong consistently: "send a message to appa
    asking if he took his medicine" routed to `compose` on one run and to
    `recall` + `search_dictations` on the next, because the word "asking" makes
    a write request look like a question. When it misroutes, the person is told
    "I don't have anything about that in your dictations" - a refusal to write
    something new on the grounds that it has not been written before, which is
    never a sensible answer.

    So the unambiguous cases are decided here instead, in code, and the tool
    choice is forced. This is deliberately narrow: it recognises an imperative
    aimed at a named recipient and nothing subtler. Everything it does not match
    still goes to the model with a free choice, which is where genuinely
    ambiguous requests belong.

    A separate planning round trip would also have worked, and was rejected -
    it costs a call on every turn to decide something that, in the cases that
    were actually failing, is visible in the text.
    """
    text = utterance.strip()
    if not text or text.endswith("?"):
        return False
    # A leading question word means they are asking about a message, not asking
    # for one: "what did I tell Rahul", "did you send Amma anything".
    if re.match(r"^\s*(what|who|when|where|why|how|did|do|does|was|were|is|are|"
                r"can|could|should|would|has|have|had)\b", text, re.IGNORECASE):
        return False
    return bool(_WRITE_REQUEST.search(text))


def grounding(conn: sqlite3.Connection) -> str:
    """Facts about the corpus itself, injected rather than looked up.

    None of this is worth a tool. A tool call costs a round trip and one of a
    strictly limited budget, and every one of these is needed by almost any
    temporal or app-filtered question - so paying for them on demand is paying
    repeatedly for something that could be free. A `get_current_time` tool is the
    clearest case: it can never fail, never varies by argument, and answering it
    from context is strictly better than answering it from a round trip.

    The corpus bounds matter for honesty, not just convenience. Asked about
    March when the dictations begin in June, the model replied "you haven't
    mentioned that in March" - true, and misleading. Knowing the range lets it
    say the useful thing instead: there is nothing from March at all.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n, MIN(local_date) AS lo, MAX(local_date) AS hi,"
        " MIN(tz_offset_min) AS tz FROM episodes"
    ).fetchone()
    if not row or not row["n"]:
        return "There are no dictations stored yet."

    apps = [
        r["app"] for r in conn.execute(
            "SELECT app, COUNT(*) AS n FROM episodes WHERE app IS NOT NULL"
            " GROUP BY app ORDER BY n DESC"
        ).fetchall()
    ]
    offset = int(row["tz"] or 0)
    sign = "+" if offset >= 0 else "-"
    tz = f"UTC{sign}{abs(offset) // 60:02d}:{abs(offset) % 60:02d}"

    # Open clarifications ride along in the grounding rather than being a queue
    # the person has to go and process. The PDF is explicit that they must not
    # become the administrator of the system, so the question surfaces when the
    # subject comes up in conversation and not before.
    pending = conn.execute(
        "SELECT c.id, c.question, m.subject FROM clarifications c"
        " LEFT JOIN memories m ON m.id = c.memory_id"
        " WHERE c.status = 'open' ORDER BY c.created_at DESC LIMIT 5"
    ).fetchall()
    questions = ""
    if pending:
        questions = (
            "\n\nUNRESOLVED - a dictation has contradicted something they told "
            "you directly. Do NOT raise these out of nowhere. When the subject "
            "comes up, ask, and use `remember` with their answer:\n"
            + "\n".join(f"  - {r['question']}" for r in pending)
        )

    return (
        f"ABOUT THIS PERSON'S DICTATIONS\n"
        f"- {row['n']:,} dictations, from {row['lo']} to {row['hi']} inclusive.\n"
        f"  There is nothing outside that range. If they ask about a date "
        f"before {row['lo']}, say the dictations do not go back that far rather "
        f"than that they never mentioned it.\n"
        f"- Their local timezone is {tz}. All times you are shown are already "
        f"local; use them as-is.\n"
        f"- The only apps that exist are: {', '.join(apps)}.\n"
        f"  Map what they say onto these ({'texted/messaged' } -> whatsapp or "
        f"messages, 'emailed' -> gmail, 'ticket' -> jira). If unsure which, "
        f"omit the app filter rather than guessing one that does not exist."
        + questions
    )


def system_prompt(
    now: dt.datetime | None = None, conn: sqlite3.Connection | None = None
) -> str:
    """The instructions, with today's date and the corpus bounds filled in.

    Without the date the model cannot resolve "yesterday" at all - it has no
    clock - so it either guessed or dropped the time filter, and the
    assignment's own worked example ("the dictation I did around 5 PM
    yesterday") returned nothing.
    """
    now = now or dt.datetime.now()
    base = SYSTEM_TEMPLATE.format(
        today=now.date().isoformat(),
        weekday=now.strftime("%A"),
        yesterday=(now.date() - dt.timedelta(days=1)).isoformat(),
    )
    if conn is not None:
        return f"{base}\n\n{grounding(conn)}"
    return base


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool = False
    abstain_reason: str = ""
    citations: list[str] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    used_facts: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    trace_id: str = ""
    session_id: str = ""
    rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "citations": self.citations,
            "dropped_citations": self.dropped_citations,
            "sources": self.sources,
            "used_facts": self.used_facts,
            "latency_ms": self.latency_ms,
            "cost_usd": round(self.cost_usd, 6),
            "trace_id": self.trace_id,
            "rounds": self.rounds,
            "plan": self.plan,
            "tool_calls": [c["tool"] for c in self.tool_calls],
        }


NO_EVIDENCE = (
    "I don't have anything about that in your dictations."
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def apply_citation_guard(
    text: str, evidence: dict[str, Evidence]
) -> tuple[str, list[str], list[str]]:
    """Strip references the tools did not actually return.

    A citation that cannot be followed is decoration; worse, it is decoration
    that looks like proof. Anything the model cites that was not retrieved this
    turn is removed from the visible answer and reported separately, so a
    fabricated reference becomes a visible event rather than a silent one.
    """
    text = normalise_citations(text)
    kept: list[str] = []
    dropped: list[str] = []

    def replace(match: re.Match) -> str:
        ref = match.group(1)
        if ref in evidence:
            if ref not in kept:
                kept.append(ref)
            return match.group(0)
        if ref not in dropped:
            dropped.append(ref)
        return ""

    cleaned = CITATION_RE.sub(replace, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip(), kept, dropped


def ensure_session(conn: sqlite3.Connection, session_id: str | None = None) -> str:
    """Get or create a conversation. Returns its id."""
    now = _now()
    if session_id:
        row = conn.execute(
            "SELECT id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE sessions SET last_active_at = ? WHERE id = ?", (now, session_id)
            )
            conn.commit()
            return session_id
    new_id = session_id or "ses_" + hashlib.sha256(
        f"{now}|{id(conn)}".encode()
    ).hexdigest()[:16]
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, title, started_at, last_active_at)"
        " VALUES (?, NULL, ?, ?)",
        (new_id, now, now),
    )
    conn.commit()
    return new_id


def append_turn(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    *,
    citations: list[str] | None = None,
    trace_id: str | None = None,
) -> None:
    """Record one turn. Ordering is per session, so sessions cannot interleave."""
    idx = conn.execute(
        "SELECT COALESCE(MAX(idx), -1) + 1 FROM turns WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO turns (id, session_id, idx, role, content, tool_calls_json,"
        " citations_json, trace_id, created_at) VALUES (?, ?, ?, ?, ?, '[]', ?, ?, ?)",
        (
            f"turn_{session_id[4:]}_{idx}", session_id, idx, role, content,
            json.dumps(citations or []), trace_id, _now(),
        ),
    )
    conn.execute(
        "UPDATE sessions SET last_active_at = ?,"
        " title = COALESCE(title, ?) WHERE id = ?",
        (_now(), content[:70] if role == "user" else None, session_id),
    )
    conn.commit()


def _history(conn: sqlite3.Connection, session_id: str, max_turns: int = 8) -> list[dict]:
    """Prior turns of THIS conversation only, oldest first.

    Scoped strictly by session_id, and that scoping is the point. A follow-up
    like "and what about Abhi?" only makes sense against what was just asked, so
    the turns have to be here - but a second conversation must not inherit them.
    Anything worth carrying between conversations belongs in memory, where it is
    subject to the gate, versioned, and correctable; conversation history is
    working context and nothing more.

    Bounded by turn count as well, so a long conversation cannot quietly become
    an unbounded, ungoverned store of its own.
    """
    rows = conn.execute(
        "SELECT role, content FROM turns WHERE session_id = ?"
        " ORDER BY idx DESC LIMIT ?",
        (session_id, max_turns),
    ).fetchall()
    return [
        {"role": "model" if row["role"] == "assistant" else "user",
         "parts": [{"text": row["content"]}]}
        for row in reversed(rows)
    ]


def ask(
    conn: sqlite3.Connection,
    client: GeminiClient,
    question: str,
    *,
    session_id: str | None = None,
    model: str | None = None,
) -> Answer:
    started = time.perf_counter()
    result = Answer(question=question, answer="")
    box = Toolbox(conn, client, session_id=session_id, utterance=question)
    system = system_prompt(conn=conn)

    contents: list[dict[str, Any]] = []
    history_text = ""
    plan = Plan()
    if session_id:
        session_id = ensure_session(conn, session_id)
        prior = _history(conn, session_id)
        contents.extend(prior)
        if prior:
            history_text = "Earlier in this conversation:\n" + "\n".join(
                f"  {'Kivi' if turn['role'] == 'model' else 'You'}: "
                f"{turn['parts'][0]['text'][:300]}"
                for turn in prior
            ) + "\n\n"
    # Decide what is being asked BEFORE any tool runs, and commit to it. The
    # decision used to be implicit in whichever tool got called first, which
    # meant it could not be inspected and was made inconsistently on the exact
    # request shape it most needed to get right.
    plan = make_plan(client, question, history=history_text)
    result.plan = plan.to_dict()
    result.tokens_in += plan.tokens_in
    result.tokens_out += plan.tokens_out
    result.cost_usd += plan.cost_usd

    guidance = plan.as_prompt()
    contents.append({
        "role": "user",
        "parts": [{"text": f"{guidance}\n\n{question}" if guidance else question}],
    })

    tool_calls_made = 0
    final_text = ""

    failed = False
    for round_no in range(MAX_ROUNDS):
        result.rounds = round_no + 1
        remaining = MAX_TOOL_CALLS - tool_calls_made
        if remaining <= 0:
            break

        # Force `compose` on the first round of a request the plan called a
        # write. Only the first: once the message exists the model is free
        # again, so a follow-up in the same turn is not locked into writing
        # another one.
        #
        # `is_write_request` is the fallback for a failed planning call, not a
        # second opinion - a planning outage should cost routing quality, not
        # reintroduce the bug that was being fixed.
        wants_write = (
            plan.kind == "write" if not plan.failed else is_write_request(question)
        )
        forced = (
            ["compose"] if round_no == 0 and not box.artifacts and wants_write
            else None
        )

        try:
            candidate, usage = client.converse(
                contents, system=system, tools=DECLARATIONS,
                tool_mode="AUTO", allowed_tools=forced, model=model,
            )
        except LLMError as exc:
            result.abstained = True
            result.abstain_reason = f"the model call failed: {exc}"
            result.answer = "Something went wrong reaching the model."
            failed = True
            break

        result.tokens_in += usage.tokens_in
        result.tokens_out += usage.tokens_out
        result.cost_usd += usage.cost_usd or 0.0

        calls = function_calls(candidate)[:remaining]
        if not calls:
            # Prose from inside the tool loop is only trusted when no evidence
            # was gathered at all - a greeting, or a request for clarification.
            # Once tools have returned something, the answer is always written
            # by the synthesis step below, against a fixed evidence set. Letting
            # the loop answer instead produced confident, uncited prose that the
            # citation guard then had to reject, turning good evidence into an
            # abstention.
            if not box.evidence:
                final_text = usage.text.strip()
            break

        contents.append(candidate.get("content") or {"role": "model", "parts": []})
        response_parts = []
        for call in calls:
            tool_calls_made += 1
            output = box.run(call.get("name", ""), call.get("args") or {})
            response_parts.append({
                "functionResponse": {
                    "name": call.get("name", ""),
                    "response": {"result": output},
                }
            })
        contents.append({"role": "user", "parts": response_parts})

    # Closing step: a plain generation over the evidence gathered, with no tools
    # in the payload at all.
    #
    # Neither omitting `tools` nor setting functionCallingConfig mode to NONE
    # actually stops Gemini calling tools once the conversation already contains
    # functionCall parts - it returns another call and no prose, and the turn
    # ends with nothing to show. Synthesising separately sidesteps that, and is
    # the better shape regardless: the answer is written from a fixed evidence
    # set, which is precisely the set the citation guard then checks it against.
    if not failed and not final_text:
        try:
            usage = client.generate(
                _synthesis_prompt(
                    question, box.evidence, history_text, box.artifacts
                ),
                system=system,
                model=model or client.settings.gen_model_heavy,
                temperature=0.0,
                max_output_tokens=2048,
            )
            result.tokens_in += usage.tokens_in
            result.tokens_out += usage.tokens_out
            result.cost_usd += usage.cost_usd or 0.0
            result.rounds += 1
            final_text = usage.text.strip()
        except LLMError as exc:
            result.abstained = True
            result.abstain_reason = f"the model call failed: {exc}"
            result.answer = "Something went wrong reaching the model."

    result.tool_calls = box.calls

    # --- the two guarantees --------------------------------------------------

    if not result.abstained:
        cleaned, kept, dropped = apply_citation_guard(final_text, box.evidence)
        result.citations = kept
        result.dropped_citations = dropped

        drafted = next(
            (a["text"].strip() for a in box.artifacts
             if a.get("kind") == "compose" and a.get("text", "").strip()),
            "",
        )

        if not box.evidence and not box.artifacts:
            # Nothing was retrieved. That is an abstention either way, but the
            # wording still matters: "your dictations only go back to 22 June"
            # is a far better refusal than "I don't have that", and the model
            # can produce it from the corpus bounds in its context. Keep its own
            # words when they are genuinely a refusal; substitute the generic
            # line only when it asserted something instead.
            result.abstained = True
            result.abstain_reason = "no dictation or memory matched this question"
            result.answer = (
                cleaned if cleaned and not _looks_like_a_claim(cleaned)
                else NO_EVIDENCE
            )
        elif drafted and drafted not in cleaned:
            # The message exists. It was written, in this person's voice, and
            # saved - so the turn cannot end without them seeing it, whatever
            # the synthesis step did or failed to do.
            #
            # Observed twice: synthesis returned nothing at all for "send a
            # message to my mother", and the turn abstained on a draft sitting
            # complete in the artifact; and elsewhere it paraphrased a draft
            # instead of passing it through. Asking a second model call to
            # faithfully reproduce text we already hold is a risk taken for no
            # gain, so the code guarantees it instead of the prompt requesting
            # it. Framing prose from synthesis is kept when it is there.
            preamble = cleaned.strip()
            result.answer = f"{preamble}\n\n{drafted}" if preamble else drafted
        elif not cleaned:
            result.abstained = True
            result.abstain_reason = "the answer had no support once unverifiable references were removed"
            result.answer = NO_EVIDENCE
        # A composed or redrafted message is written text, not a claim about
        # what happened - there is nothing for it to cite, and demanding a
        # citation abstained on every single write request. "Hi Vikram, please
        # wrap this up in two days" reads as an assertion to
        # `_looks_like_a_claim` because it IS one; it is just not a claim ABOUT
        # the person's history, which is the only kind this guard governs. The
        # guard still applies in full to the evidence-backed path.
        elif not kept and not box.artifacts and _looks_like_a_claim(cleaned):
            result.abstained = True
            result.abstain_reason = (
                "the answer cited nothing that was actually retrieved"
            )
            result.answer = NO_EVIDENCE
        else:
            result.answer = cleaned

    result.sources = [
        {"ref": e.ref, "kind": e.kind, "text": e.text, "ts": e.ts, "app": e.app}
        for e in box.evidence.values()
    ]
    result.used_facts = [
        {
            "fact_id": e.ref,
            "statement": e.text,
            "status": "active",
            "detail": f"{e.detail.get('type', '')} · found {e.detail.get('via', 'direct')}",
        }
        for e in box.evidence.values()
        if e.kind == "fact"
    ]

    if not result.abstained:
        box.note_usage()

    result.latency_ms = int((time.perf_counter() - started) * 1000)
    result.trace_id = _write_trace(conn, result, session_id)
    result.session_id = session_id or ""

    # Persist both turns so the NEXT question in this conversation can see them.
    # Written after the answer, not before, so a failed turn does not leave a
    # dangling question in the transcript.
    if session_id:
        append_turn(conn, session_id, "user", question)
        append_turn(
            conn, session_id, "assistant", result.answer,
            citations=result.citations, trace_id=result.trace_id,
        )
    return result


def _synthesis_prompt(
    question: str,
    evidence: dict[str, Evidence],
    history: str,
    artifacts: list[dict[str, Any]] | None = None,
) -> str:
    """Build the closing prompt from exactly the evidence that was retrieved."""
    artifacts = artifacts or []
    if not evidence and not artifacts:
        return (
            f"{history}Question: {question}\n\n"
            "Nothing was found in this person's dictations or memory for this "
            "question. Tell them plainly that you don't have it. Do not guess."
        )

    facts = [e for e in evidence.values() if e.kind == "fact"]
    records = [e for e in evidence.values() if e.kind == "record"]

    lines = [history, f"Question: {question}", ""]
    if facts:
        lines.append("WHAT KIVI HAS LEARNED (cite as [id]):")
        for item in facts:
            lines.append(f"  [{item.ref}] {item.text}")
        lines.append("")
    if records:
        lines.append("DICTATIONS FOUND (times are the person's LOCAL time):")
        for item in sorted(records, key=lambda e: e.ts or "", reverse=True):
            when = (item.ts or "")[:16].replace("T", " ")
            lines.append(f"  [{item.ref}] {when} · {item.app or '—'} · {item.text}")
        lines.append("")
    for artifact in artifacts:
        if artifact.get("kind") == "compose":
            lines.append(f"MESSAGE ALREADY WRITTEN for {artifact['to']}:")
            lines.append(artifact["text"])
            lines.append("")
            lines.append(
                "Give them this message EXACTLY as written. Do not rewrite it "
                "and do not summarise it.\n"
                "Cite NOTHING here - not on the message, not after it. This is "
                "new text Kivi wrote, so there is no source for it, and any id "
                "you add sits in something they are about to send to another "
                "person. The memories and dictations listed above shaped how it "
                "is worded; they are not claims being made, so they are not "
                "cited. One short line of your own before the message is enough."
            )
            lines.append("")
            continue

        if artifact.get("kind") == "remember":
            lines.append(f"WHAT KIVI RECORDED: {artifact['outcome']}")
            lines.append(
                "Tell them this in one short line. If nothing was written "
                "because Kivi already had it, say so plainly - do not say you "
                "have no record of it, because you do."
                + (f" Cite [{artifact['statement_id']}]."
                   if artifact.get("written") else "")
            )
            lines.append("")
            continue

        if artifact.get("kind") == "redraft":
            lines.append(f"REWRITE ALREADY PREPARED for '{artifact['target']}':")
            lines.append(artifact["text"])
            if artifact.get("preferences_applied"):
                lines.append(
                    "  (applied: " + "; ".join(artifact["preferences_applied"][:4]) + ")"
                )
            lines.append("")
            lines.append(
                "The person asked for this rewrite. Give it to them as the body "
                "of your reply - do not summarise it, describe it, or withhold "
                "it. Cite the dictation it came from AFTER the rewritten text, "
                "never inside it: this is text they will send, and an id in the "
                "middle of it would be sent along with everything else."
            )
            lines.append("")

    lines.append(
        "Answer the question using only what is listed above. Cite the id in "
        "square brackets after each claim. Times shown are the person's own "
        "local time, so a dictation listed at 16:59 IS the one they mean by "
        "'around 5 PM'. If what is listed does not actually answer the "
        "question, say you don't have it - do not stretch weak evidence into "
        "an answer."
    )
    return "\n".join(lines)


def _looks_like_a_claim(text: str) -> bool:
    """Whether an uncited answer is asserting something or declining to.

    A refusal needs no citation. An assertion does. Without this check, an
    answer that cites nothing because it invented everything would pass the
    guard simply by having had its fabricated references stripped.
    """
    lowered = text.lower()
    refusals = (
        "don't have", "do not have", "haven't mentioned", "have not mentioned",
        "nothing about", "no mention", "isn't anything", "is not anything",
        "couldn't find", "could not find", "not in your",
        # Refusals that explain the boundary rather than just denying, which are
        # the ones worth preserving verbatim.
        "only go back", "don't go back", "do not go back", "outside", "no record",
        "nothing from", "no dictations from", "don't cover", "do not cover",
    )
    return not any(phrase in lowered for phrase in refusals)


def _write_trace(
    conn: sqlite3.Connection, answer: Answer, session_id: str | None = None
) -> str:
    trace_id = "tr_" + hashlib.sha256(
        f"{answer.question}|{_now()}".encode()
    ).hexdigest()[:16]
    conn.execute(
        "INSERT INTO traces"
        " (id, kind, subject_id, input_json, candidates_json, decision, reason,"
        "  model, tokens_in, tokens_out, cost_usd, latency_ms, cached, created_at)"
        " VALUES (?, 'answer', NULL, ?, ?, ?, ?, NULL, ?, ?, ?, ?, 0, ?)",
        (
            trace_id,
            json.dumps({"question": answer.question}, ensure_ascii=False),
            json.dumps(
                {
                    "plan": answer.plan,
                    "tool_calls": answer.tool_calls,
                    "citations": answer.citations,
                    "dropped": answer.dropped_citations,
                    "rounds": answer.rounds,
                },
                ensure_ascii=False, default=str,
            ),
            "abstained" if answer.abstained else "answered",
            answer.abstain_reason or f"{len(answer.citations)} citations",
            answer.tokens_in, answer.tokens_out, answer.cost_usd,
            answer.latency_ms, _now(),
        ),
    )
    conn.execute(
        "INSERT INTO answers"
        " (id, session_id, query, answer, abstained, abstain_reason,"
        "  citations_json, trace_id, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "ans_" + trace_id[3:], session_id, answer.question, answer.answer,
            int(answer.abstained), answer.abstain_reason,
            json.dumps(answer.citations), trace_id, _now(),
        ),
    )
    conn.commit()
    return trace_id
