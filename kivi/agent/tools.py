"""Hey Kivi's tools. Four, and no more.

The assignment is explicit that a narrow set used convincingly beats a broad
collection used superficially, and its own worked example - "find the dictation I
did around 5 PM yesterday in Slack and polish it for the meeting I'm walking
into" - needs exactly two of these: a filtered search, then a redraft of what it
found. That example is the design proof, so the tool list is built around it
rather than around everything the store could expose.

Every tool returns evidence with a stable reference. Those references are what
the answer cites, and what the citation guard later checks the answer against.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from kivi.llm.gemini import GeminiClient
from kivi.memory import policy
from kivi.memory.extract import Candidate, Relation
from kivi.memory.promote import build_edges, promote, stage_relations
from kivi.memory import style
from kivi.memory.search import mark_used, recall
from kivi.retrieval.search import Filters, Hit, search

# A record id anywhere in text, bracketed or bare.
_RECORD_ID = re.compile(r"\[?\b(?:ep|mem|say|tr|ans|cmp)_[a-z0-9]{6,}\b\]?")


def _strip_record_ids(text: str) -> str:
    """Remove any internal id from text that is about to be sent to somebody.

    A composed message is the one place an id cannot be tolerated: the person
    forwards it to their mother or their manager, and a leaked `ep_4b5040...`
    goes with it. The agent had searched before composing and pasted a retrieved
    id into the message body - "...please finish the remediation tasks in 7 days
    ep_4b504036cb67e229. Thanks."

    It is stripped in code rather than asked for in a prompt because the draft
    deliberately bypasses the citation guard: the guard exists to verify claims,
    and a written message makes none, so nothing else was checking. Prompts are
    requests; this needs to be a guarantee.
    """
    cleaned = _RECORD_ID.sub("", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"\s+([.,;:!?])", r"\1", cleaned).strip()

log = logging.getLogger(__name__)


@dataclass
class Evidence:
    """One retrievable thing the answer is allowed to cite."""

    ref: str
    kind: str          # 'record' (a dictation) | 'fact' (a memory)
    text: str
    ts: str | None = None
    app: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


# How many dictations one search hands the agent, and how many more may be added
# from searching the person's own words. Both measured, not chosen: see _search.
SEARCH_LIMIT = 11
OWN_WORDS_EXTRA = 3

DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "search_dictations",
        "description": (
            "Search everything the person has dictated. Use this for anything "
            "about what they actually said, when they said it, or to find a "
            "specific message to work from. Supports filtering by app and by "
            "time, including the local hour of day."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in natural language.",
                },
                "app": {
                    "type": "string",
                    "description": "Restrict to one app, e.g. slack, gmail, whatsapp.",
                },
                "on_date": {
                    "type": "string",
                    "description": "A single calendar day, YYYY-MM-DD, in the "
                                   "person's own local time. Use this for "
                                   "'yesterday', 'on Tuesday', 'on the 5th'. "
                                   "Today's date is given in your instructions.",
                },
                "after_date": {
                    "type": "string",
                    "description": "Inclusive start of a range, YYYY-MM-DD local.",
                },
                "before_date": {
                    "type": "string",
                    "description": "Inclusive end of a range, YYYY-MM-DD local.",
                },
                "hour_from": {
                    "type": "integer",
                    "description": "Local hour of day, 0-23. For 'around 5 PM' pass "
                                   "17; a single hour is widened by an hour either "
                                   "side automatically.",
                },
                "hour_to": {
                    "type": "integer",
                    "description": "Local hour of day, 0-23. Omit if you passed a "
                                   "single hour_from.",
                },
                "limit": {"type": "integer", "description": "Default 11, max 20."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Look up what Kivi has learned about a person, project or topic - "
            "durable facts, stated preferences, and outstanding commitments. Use "
            "this for 'who is working on X', 'how do I like my emails written', "
            "'what did I promise'. Returns the dictations each memory came from."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Person, project or topic."},
                "types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["entity", "preference", "commitment"]},
                    "description": "Optional filter on kinds of memory.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "redraft",
        "description": (
            "Rewrite dictations the person already made, for a new context or "
            "audience. Call search_dictations first to find them, then pass their "
            "ids. Applies any writing preferences Kivi has learned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "record_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ids returned by search_dictations.",
                },
                "target": {
                    "type": "string",
                    "description": "What it is being rewritten for, e.g. 'a stand-up "
                                   "update for the steering meeting'.",
                },
                "to": {
                    "type": "string",
                    "description": "Who it is being written TO, if the person named "
                                   "someone - e.g. 'Amma', 'Sanjay'. Kivi has learned "
                                   "how they write to each person and will match it.",
                },
            },
            "required": ["record_ids", "target"],
        },
    },
    {
        "name": "compose",
        "description": (
            "Write a NEW message on the person's behalf, in the voice they use "
            "with whoever it is for. Use this when they describe a message they "
            "want written rather than pointing at one that already exists: "
            "'tell Amma I'll be late', 'write to Sanjay that the release "
            "slipped', 'message Rahul about the match'.\n"
            "Use `redraft` instead when they want an EXISTING dictation rewritten.\n"
            "The result is saved as a dictation, because in the real product it "
            "would be sent to the app it was written for."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Who it is for, e.g. 'Amma', 'Sanjay'. Kivi has "
                                   "learned how they write to each person.",
                },
                "message": {
                    "type": "string",
                    "description": "What it needs to say, in the person's own words "
                                   "from their instruction. Do not embellish.",
                },
                "app": {
                    "type": "string",
                    "description": "Where it is going: whatsapp, slack, gmail, ...",
                },
            },
            "required": ["to", "message"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Change what Kivi believes, when the person TELLS you something "
            "rather than asks you something.\n"
            "Use action='add' when they state a new fact or give an instruction: "
            "'add Ashwin to DSPM as a frontend React developer', 'Priya is on "
            "Atlas now', 'always keep my Slack messages short'.\n"
            "'Change the DSPM owner from Priya to Rahul' is also 'add', with the "
            "NEW value: subject 'Rahul', body 'leads DSPM, taking over as lead "
            "from Priya', "
            "and a works_on relation to DSPM. It does not matter whether Kivi "
            "already had the old value - it may exist only in their dictations.\n"
            "Use 'forget' to retire a memory, 'confirm' to pin one so extraction "
            "cannot overwrite it.\n"
            "IMPORTANT: only call this for an explicit instruction or statement. "
            "Do NOT call it for something mentioned in passing while asking a "
            "question - 'what did I tell Bushan about the schema review he was "
            "blocked on?' is a question, and writes nothing. "
            "The person's dictations are never modified by this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "revise", "forget", "confirm"],
                },
                "memory_id": {
                    "type": "string",
                    "description": "A mem_ id. Required for forget/confirm/revise.",
                },
                "type": {
                    "type": "string",
                    "enum": list(policy.MEMORY_TYPES),
                    "description": "For 'add': what kind of thing this is.",
                },
                "subject": {
                    "type": "string",
                    "description": "For 'add': who or what it is about, e.g. 'Ashwin'.",
                },
                "body": {
                    "type": "string",
                    "description": "For 'add': the fact itself, e.g. 'frontend React "
                                   "developer on DSPM'.",
                },
                "entity_type": {
                    "type": "string",
                    "enum": list(policy.ENTITY_TYPES),
                    "description": "For an entity: person, project, org, place, thing, term.",
                },
                "relations": {
                    "type": "array",
                    "description": "For 'add': links to other things, so the graph "
                                   "learns them too.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "src": {"type": "string"},
                            "relation": {
                                "type": "string", "enum": list(policy.RELATIONS)
                            },
                            "dst": {"type": "string"},
                        },
                        "required": ["src", "relation", "dst"],
                    },
                },
                "due_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD, for a commitment.",
                },
            },
            "required": ["action"],
        },
    },
]


class Toolbox:
    """Executes tool calls and accumulates the evidence they returned."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        client: GeminiClient,
        session_id: str | None = None,
        utterance: str = "",
    ) -> None:
        self.conn = conn
        self.client = client
        self.session_id = session_id
        # What the person actually typed, kept verbatim so a memory written from
        # it quotes them rather than the model's paraphrase of them.
        self.utterance = utterance
        self.evidence: dict[str, Evidence] = {}
        self.used_memory_ids: list[str] = []
        self.calls: list[dict[str, Any]] = []
        # Things a tool PRODUCED rather than retrieved - a redraft, the result of
        # a correction. Not evidence, so not citable, but the final answer has to
        # see them or the work is computed and then silently discarded.
        self.artifacts: list[dict[str, Any]] = []
        # The person's own words, searched once per set of filters per turn: the
        # agent may search several times, and that search's answer does not change.
        self._own_words_cache: dict[str, list[Hit]] = {}

    # -- dispatch ------------------------------------------------------------

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = {
            "search_dictations": self._search,
            "recall": self._recall,
            "redraft": self._redraft,
            "compose": self._compose,
            "remember": self._remember,
        }.get(name)
        if handler is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            payload = dict(args or {})
            if name == "remember":
                payload.setdefault("_utterance", self.utterance)
            result = handler(payload)
        except Exception as exc:  # a tool failure must not kill the turn
            # Roll back before swallowing it. A write tool that throws partway
            # leaves an open transaction on this connection, and because the
            # model cache writes its ledger on a *separate* connection, every
            # later call then fails with "database is locked" - a failure that
            # looks like a database problem and is really an unreleased lock.
            try:
                self.conn.rollback()
            except Exception:
                pass
            log.warning("tool %s failed: %s", name, exc, exc_info=True)
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self.calls.append({"tool": name, "args": args, "result": result})
        return result

    # -- tools ---------------------------------------------------------------

    def _search(self, args: dict[str, Any]) -> dict[str, Any]:
        filters = Filters(app=(args.get("app") or None))

        # Dates are matched against the speaker's own local date, not UTC. A
        # 16:59 IST dictation is 11:29 UTC the same day, but a late-evening one
        # would land on the following UTC day and "yesterday" would miss it.
        if args.get("on_date"):
            filters.local_date = str(args["on_date"])[:10]
        else:
            if args.get("after_date"):
                filters.after = _local_day_start(str(args["after_date"])[:10])
            if args.get("before_date"):
                filters.before = _local_day_end(str(args["before_date"])[:10])

        hour_from = args.get("hour_from")
        hour_to = args.get("hour_to")
        if hour_from is not None and hour_to is None:
            # People say "5 PM" for 16:59 as readily as for 17:30, so a single
            # hour is a point they remember, not a boundary they measured.
            # Widening by an hour either side is the difference between finding
            # the message and telling them it does not exist.
            hour_from, hour_to = (int(hour_from) - 1) % 24, (int(hour_from) + 1) % 24
        if hour_from is not None:
            filters.local_hour_min = int(hour_from)
        if hour_to is not None:
            filters.local_hour_max = int(hour_to)

        limit = min(int(args.get("limit") or SEARCH_LIMIT), 20)
        result = search(self.conn, self.client, str(args.get("query") or ""),
                        limit=limit, filters=filters)
        hits = list(result.hits)

        # The agent's query is its own rewrite of the question, and a rewrite can
        # drop the word the question turns on. "who is running dspm" was searched
        # as "DSPM" - about 124 matching dictations - and the one saying who runs
        # it never reached the results, so Kivi said it had never been mentioned.
        # The person's own words keep that word, so they are searched too, with
        # bm25 beside the vectors because a literal match on "running" is what
        # finds it. These only ever add to the agent's results, never replace.
        #
        # Measured on the 270 planted questions: top-8 alone found an answer for
        # 0.822, top-11 alone 0.889, top-11 plus these extras 0.915 - never worse
        # than top-11 on any question. And top-11 alone still missed the DSPM
        # case, because all it ever saw was "DSPM".
        seen = {hit.episode_id for hit in hits}
        hits += [
            hit for hit in self._own_words(filters) if hit.episode_id not in seen
        ][:OWN_WORDS_EXTRA]

        out = []
        for hit in hits:
            self.evidence[hit.episode_id] = Evidence(
                ref=hit.episode_id, kind="record", text=hit.formatted,
                ts=hit.when, app=hit.app,
            )
            out.append({
                "id": hit.episode_id,
                # Local time, always. The model compares this against what the
                # person said ("4:59 PM"), so handing it UTC makes it reject
                # correct results as mismatches.
                "when_local": hit.when,
                "app": hit.app, "text": hit.formatted,
            })
        if out:
            return {
                "found": len(out), "dictations": out,
                "note": f"cite these by id, e.g. [{out[0]['id']}]",
            }

        # An empty result is where a search agent gives up, and the reply it
        # produces - "you never mentioned that" - is often wrong: the filters
        # excluded the answer rather than the corpus lacking it. Say which
        # filter did the excluding so the model can retry instead of concluding.
        return {"found": 0, "dictations": [], "diagnosis": self._why_empty(args, filters)}

    def _own_words(self, filters: Filters) -> list[Hit]:
        """The person's literal question, searched under the agent's filters.

        Same filters, so "around 5 PM yesterday in Slack" narrows this search
        exactly as it narrows the agent's - it must never widen what was asked.
        """
        text = self.utterance.strip()
        if not text:
            return []
        key = repr(sorted(vars(filters).items()))
        if key not in self._own_words_cache:
            self._own_words_cache[key] = search(
                self.conn, self.client, text, limit=8, filters=filters,
                use_bm25=True,
            ).hits
        return self._own_words_cache[key]

    def _why_empty(self, args: dict[str, Any], filters: Filters) -> dict[str, Any]:
        """Relax one filter at a time and report what each would have matched."""
        def count(where_filters: Filters) -> int:
            where, params = where_filters.where()
            return int(
                self.conn.execute(
                    f"SELECT COUNT(*) FROM episodes e WHERE {where}", params
                ).fetchone()[0]
            )

        total = int(self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])
        notes: list[str] = []

        if filters.app:
            relaxed = Filters(**{**vars(filters), "app": None})
            n = count(relaxed)
            known = [
                r["app"] for r in self.conn.execute(
                    "SELECT DISTINCT app FROM episodes WHERE app IS NOT NULL"
                ).fetchall()
            ]
            if filters.app not in known:
                notes.append(
                    f"app={filters.app!r} does not exist. Real apps: {', '.join(known)}."
                )
            elif n:
                notes.append(f"dropping the app filter would match {n} dictations.")

        if filters.local_hour_min is not None:
            relaxed = Filters(**{
                **vars(filters), "local_hour_min": None, "local_hour_max": None
            })
            n = count(relaxed)
            if n:
                notes.append(f"dropping the hour filter would match {n} on that day.")

        if filters.local_date or filters.after or filters.before:
            relaxed = Filters(**{
                **vars(filters),
                "local_date": None, "after": None, "before": None,
            })
            n = count(relaxed)
            bounds = self.conn.execute(
                "SELECT MIN(local_date) AS lo, MAX(local_date) AS hi FROM episodes"
            ).fetchone()
            asked = filters.local_date or (
                filters.after.date().isoformat() if filters.after else None
            )
            if asked and bounds["lo"] and not (bounds["lo"] <= asked <= bounds["hi"]):
                notes.append(
                    f"{asked} is outside the recorded range "
                    f"({bounds['lo']} to {bounds['hi']}) - there are no dictations "
                    f"from then at all, which is different from never mentioning it."
                )
            elif n:
                notes.append(f"dropping the date filter would match {n} dictations.")

        return {
            "total_dictations": total,
            "why_empty": notes or ["no dictation matched this wording; try different words"],
            "advice": "Retry with the filter named above removed before concluding "
                      "that nothing was said.",
        }

    def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        result = recall(
            self.conn, self.client, str(args.get("topic") or ""),
            limit=6, types=args.get("types") or None,
        )
        memories = []
        from kivi.memory.search import provenance_times

        for hit in result.hits:
            text = f"{hit.subject}{': ' + hit.body if hit.body else ''}"
            # A fact the person stated to Kivi carries the date they said it.
            # Synthesis needs that to let a correction outrank the older
            # dictations it corrects - "Rahul leads DSPM" and "Priya is running
            # point on DSPM" are otherwise two claims with no way to rank them.
            told = provenance_times(self.conn, hit.memory_id)[1]
            if told:
                text += f" (you told Kivi on {told[:10]})"
            self.evidence[hit.memory_id] = Evidence(
                ref=hit.memory_id, kind="fact", text=text,
                detail={"type": hit.type, "confidence": hit.confidence,
                        "via": hit.via, "told": told},
            )
            self.used_memory_ids.append(hit.memory_id)
            memories.append({
                "id": hit.memory_id, "type": hit.type, "subject": hit.subject,
                "detail": hit.body or "", "confidence": round(hit.confidence, 2),
                "found_via": hit.via, "due": hit.due_at,
                **({"you_told_kivi_on": told[:10]} if told else {}),
            })

        # Attach the dictations behind those memories. A memory is an index into
        # episodes, not a replacement for them: the answer should be able to cite
        # what was actually said.
        sources = []
        if result.episode_ids:
            placeholders = ",".join("?" * len(result.episode_ids[:12]))
            for row in self.conn.execute(
                f"SELECT id, ts, ts_epoch, tz_offset_min, app, formatted FROM episodes"
                f" WHERE id IN ({placeholders}) ORDER BY ts_epoch DESC",
                result.episode_ids[:12],
            ).fetchall():
                when = _local_when(row)
                self.evidence[row["id"]] = Evidence(
                    ref=row["id"], kind="record", text=row["formatted"],
                    ts=when, app=row["app"],
                )
                sources.append({
                    "id": row["id"], "when_local": when,
                    "app": row["app"], "text": row["formatted"],
                })

        return {
            "found": len(memories), "memories": memories,
            "supporting_dictations": sources,
            "note": "cite memories as [mem_...] and dictations as [ep_...]",
        }

    def _redraft(self, args: dict[str, Any]) -> dict[str, Any]:
        ids = [str(i) for i in (args.get("record_ids") or [])][:8]
        if not ids:
            return {"error": "no record_ids given; call search_dictations first"}
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, ts, ts_epoch, tz_offset_min, app, formatted FROM episodes"
            f" WHERE id IN ({placeholders}) ORDER BY ts_epoch",
            ids,
        ).fetchall()
        if not rows:
            return {"error": "none of those ids exist"}

        # Only preferences the person stated, or switched on - see
        # policy.APPLY_PREFERENCE_AT for why this line is shared with the UI.
        prefs = self.conn.execute(
            "SELECT subject, body FROM memories"
            " WHERE type = 'preference' AND status = 'active' AND confidence >= ?"
            " ORDER BY confidence DESC LIMIT 6",
            (policy.APPLY_PREFERENCE_AT,),
        ).fetchall()
        preference_lines = [
            f"- {row['body'] or row['subject']}" for row in prefs
        ] or ["- (none learned yet)"]

        # How this person writes to THIS person. The general preferences above
        # are how they write in an app; this is how they write to a human, and
        # it is the more specific of the two, so it goes last and wins.
        profile = style.load_profile(self.conn, str(args.get("to") or "")) \
            if args.get("to") else None
        style_lines: list[str] = []
        if profile and profile.active:
            style_lines = style.instructions(profile)

        source_text = "\n".join(
            f"[{r['id']}] {r['formatted']}" for r in rows
        )
        for row in rows:
            # Local time, like every other tool. Writing UTC here silently
            # overwrote the local timestamp search had already recorded for the
            # same episode - the tools disagreed about when a thing happened,
            # and the last writer won.
            self.evidence[row["id"]] = Evidence(
                ref=row["id"], kind="record", text=row["formatted"],
                ts=_local_when(row), app=row["app"],
            )

        prompt = [
            f"Rewrite the following dictations for: {args.get('target')}",
            "",
            source_text,
            "",
            "Writing preferences this person has stated:",
            *preference_lines,
        ]
        if style_lines:
            prompt += [
                "",
                f"HOW THIS PERSON WRITES TO {profile.display.upper()} "
                f"(learned from {profile.n_samples} of their own messages - "
                f"follow these exactly, they override the general preferences "
                f"above):",
                *(f"- {rule}" for rule in style_lines),
            ]
        prompt += [
            "",
            "Return only the rewritten text. Keep every fact; add nothing that is "
            "not in the source.",
        ]

        result = self.client.generate(
            "\n".join(prompt),
            system="You rewrite a person's own dictated notes for a new context. "
                   "You never invent facts and you never soften what they said. "
                   "When told how they write to a particular person, you match "
                   "that exactly - the greeting, the length, the register.",
            temperature=0.3,
        )
        draft = result.text.strip()
        applied = [row["body"] or row["subject"] for row in prefs]
        artifact = {
            "kind": "redraft",
            "target": str(args.get("target") or ""),
            "text": draft,
            "based_on": ids,
            "preferences_applied": applied,
        }
        if profile and profile.active:
            artifact["style_for"] = profile.display
            artifact["style_rules"] = style_lines
            artifact["style_samples"] = profile.n_samples
            artifact["style_evidence"] = profile.episode_ids[:6]
        self.artifacts.append(artifact)

        out: dict[str, Any] = {
            "draft": draft,
            "based_on": ids,
            "preferences_applied": applied,
        }
        if profile and profile.active:
            # Reported back so the answer can say WHY it is written this way.
            # A style applied invisibly is the expectancy violation that makes
            # people distrust memory products.
            out["written_for"] = profile.display
            out["style_learned_from"] = profile.n_samples
            out["style_rules"] = style_lines
        elif args.get("to"):
            out["note"] = (
                f"No learned style for {args.get('to')!r} yet - not enough "
                f"messages to them. Wrote in the general style instead."
            )
        return out

    def _compose(self, args: dict[str, Any]) -> dict[str, Any]:
        """Write a new message in the voice this person uses with the recipient.

        The draft is kept in `composed_messages`, NOT in `episodes`. In a product
        with connectors it would have been sent, so the record is worth having -
        but `episodes` means what this person actually dictated, and this is what
        Kivi wrote in a voice Kivi predicted.

        Reading one back as evidence of how they write would be the profile
        confirming itself, and their voice would drift toward Kivi's idea of it.
        Keeping the two in separate tables makes that structurally impossible
        rather than relying on every future query remembering to filter.
        """
        to = str(args.get("to") or "").strip()
        message = str(args.get("message") or "").strip()
        if not to or not message:
            return {"error": "need both 'to' and 'message'"}

        profile = style.load_profile(self.conn, to)
        rules = _style_rules(self.conn, profile)

        prefs = self.conn.execute(
            "SELECT subject, body FROM memories"
            " WHERE type = 'preference' AND status = 'active' AND confidence >= ?"
            " ORDER BY confidence DESC LIMIT 5",
            (policy.APPLY_PREFERENCE_AT,),
        ).fetchall()

        prompt = [f"Write a message to {to}.", "", f"It needs to say: {message}", ""]
        if self.utterance:
            # The person's own words, not the agent's summary of them.
            #
            # `message` is written by the agent, and when the turn has already
            # searched for something the agent writes what it just read. Asked
            # to "send message to finish tasks in the project in 7 days" with
            # DSPM dictations in context, it passed a `message` carrying an
            # asset discovery phase and compliance gaps, and this tool rendered
            # them faithfully - the guard below was checking the draft against
            # an argument that was already embellished. Their literal request is
            # the only thing in the turn that cannot have been embellished, so
            # it is what the content is held to.
            prompt += [
                f"THEIR ACTUAL REQUEST, word for word: {self.utterance!r}",
                "Everything the message asserts must come from THIS. Where the "
                "line above says more than they did, follow this instead.",
                "",
            ]
        if rules:
            prompt.append(
                f"HOW THIS PERSON WRITES TO {to.upper()} - learned from "
                f"{profile.n_samples} of their own messages. Follow it exactly, "
                f"including any rule about which situation calls for what:"
            )
            prompt += [f"- {r}" for r in rules]
            prompt.append("")
        if prefs:
            prompt.append("Their general writing preferences:")
            prompt += [f"- {p['body'] or p['subject']}" for p in prefs]
            prompt.append("")
        prompt.append(
            "Write only the message itself, as they would send it. No preamble, "
            "no quotes, no explanation.\n"
            "Do not add FACTS they did not give you - no news, promises, times "
            "or details of their own.\n"
            "But the style above is not added detail, it is how this person "
            "writes: apply it fully. If they address the recipient by a name or "
            "nickname, use it. If they open or close a certain way, do that. A "
            "message that carries the content but not the voice has failed.\n"
            "\n"
            "WHERE THE TWO CONFLICT, CONTENT WINS - and a rule that cannot be "
            "followed is DROPPED, not satisfied by invention. Check the message "
            "before you send it: every fact, reason, deadline, project and "
            "excuse in it must trace back to the instruction above. If you "
            "wrote a clause the instruction did not give you, delete it.\n"
            "Keep whether it ASKS or TELLS. If they are asking the recipient "
            "something, the message must be a question to that person - "
            "'Did you take your medicine?', never 'You took your medicine.', "
            "which asserts the very thing they wanted to find out.\n"
            "This is a limit on FACTS, not on wording. The instruction is a note "
            "about what to say, not the text to send: write it out properly - "
            "capitalised, punctuated, a complete sentence phrased the way this "
            "person would phrase it. 'finish the work in 2 days' becomes "
            "'Please finish the work in 2 days.' Pasting the note in unchanged "
            "is as wrong as embellishing it.\n"
            "A rule asking for several sentences, a dense paragraph, or "
            "connectives like 'Because' or 'Therefore' cannot be followed when "
            "the instruction says only one thing - there is nothing to connect. "
            "Skip that rule and write the one thing. A correct short message "
            "beats a well-shaped message that says something untrue.\n"
            "Greetings, sign-offs, nicknames, register and word choice are "
            "voice: apply those always. Anything that would be NEW INFORMATION "
            "to the recipient is not voice, whatever a rule seems to ask for."
        )

        result = self.client.generate(
            "\n".join(prompt),
            system="You write messages on someone's behalf, in their own voice, "
                   "to a specific person. You never invent facts they did not "
                   "give you, and you match how they write to that person.",
            # Low, because this is rule-following rather than writing. At 0.4
            # the same request produced "Hi Amma, please prepare lunch by 12."
            # on one run and "Prepare lunch by 12." on the next, dropping the
            # most distinctive habit she has. Variety is not a virtue here - the
            # person wants their own voice, and their own voice is consistent.
            temperature=0.15,
            max_output_tokens=512,
        )
        draft = _strip_record_ids(result.text.strip())
        composed_id = self._save_composed(
            draft, to, str(args.get("app") or ""), rules,
            profile.n_samples if profile else 0,
        )

        self.artifacts.append({
            "kind": "compose", "to": to, "text": draft,
            "composed_id": composed_id, "style_rules": rules,
            "style_samples": profile.n_samples if profile else 0,
        })

        out: dict[str, Any] = {
            "draft": draft, "to": to, "saved_as": composed_id,
            "note": "kept in composed_messages, not in the dictations - Kivi "
                    "wrote it, so it is never read back as evidence of how this "
                    "person writes",
        }
        if rules:
            out["style_learned_from"] = profile.n_samples
            out["style_rules"] = rules[:6]
        else:
            out["style_note"] = (
                f"no learned style for {to!r} yet - too few messages to them, so "
                f"this is written in the general style"
            )
        return out

    def _save_composed(
        self, text: str, to: str, app: str, rules: list[str], samples: int
    ) -> str:
        """Record what Kivi wrote, in its own table - never in `episodes`.

        The rules that produced it are stored alongside, so a draft the person
        disliked can be traced to the habit Kivi thought it was following rather
        than leaving them to guess why it came out that way.
        """
        now = dt.datetime.now().astimezone()
        composed_id = "cmp_" + hashlib.sha256(
            f"{now.isoformat()}|{to}|{text}".encode("utf-8")
        ).hexdigest()[:16]
        offset = int((now.utcoffset() or dt.timedelta()).total_seconds() // 60)
        self.conn.execute(
            "INSERT OR IGNORE INTO composed_messages"
            " (id, ts, ts_epoch, app, recipient, recipient_norm, text,"
            "  session_id, style_rules_json, style_samples, tz_offset_min,"
            "  local_hour, local_date, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                composed_id,
                now.astimezone(dt.timezone.utc).isoformat(), int(now.timestamp()),
                (app or "").lower() or None,
                to, style.normalise_recipient(to), text,
                self.session_id,
                json.dumps(rules, ensure_ascii=False), samples,
                offset, now.hour, now.date().isoformat(), now.isoformat(),
            ),
        )
        self.conn.commit()
        return composed_id

    def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args.get("action") or "")
        if action == "add":
            out = self._add(args)
        elif action in {"forget", "confirm", "revise"}:
            out = self._change(action, args)
        else:
            out = {"error": f"unknown action {action!r}"}

        # Every outcome of an instruction has to reach the reply, including the
        # ones where nothing was written. A turn that ends with no artifact is
        # answered as if a search had failed - "you haven't mentioned that" -
        # which is wrong for a request that was never a search.
        if out.get("needs_confirmation"):
            self.artifacts.append({
                "kind": "remember", "written": False, "statement_id": "",
                "ask": out["ask"],
                "outcome": f"Nothing written yet - waiting for their answer. {out['ask']}",
            })
        elif "error" in out:
            # Marked failed so the loop still owes a real attempt: a bad mem_ id
            # on the first round should not count as having carried it out.
            self.artifacts.append({
                "kind": "remember", "written": False, "statement_id": "",
                "failed": True,
                "outcome": "Nothing was changed - Kivi could not tell which "
                           f"memory that refers to ({out['error']}).",
            })
        return out

    def _add(self, args: dict[str, Any], supersedes: str | None = None) -> dict[str, Any]:
        """Write something the person told Kivi directly.

        Goes through the same gate as extraction - the status/cause rule and the
        third-party refusals still apply - but with the evidence threshold
        lifted, because an instruction is not an inference to be corroborated.
        """
        subject = str(args.get("subject") or "").strip()
        if not subject:
            return {"error": "nothing to remember: no subject given"}

        mtype = str(args.get("type") or "entity")
        body = str(args.get("body") or "").strip()

        # Contradiction check BEFORE writing: a new fact lands silently, but one
        # that displaces something Kivi is confident about is worth a question.
        existing = self.conn.execute(
            "SELECT id, subject, body, confidence FROM memories"
            " WHERE type = ? AND LOWER(subject) = LOWER(?) AND status = 'active'",
            (mtype, subject),
        ).fetchone()
        if (
            existing is not None
            and body
            and (existing["body"] or "").strip()
            and (existing["body"] or "").strip().lower() != body.lower()
            and float(existing["confidence"]) >= 0.75
        ):
            # The exact write that carries out the change if they agree. A
            # revise retires the old value first, so it does not collide with it
            # again - calling `add` a second time can never get past this check.
            proposal: dict[str, Any] = {
                "action": "revise", "memory_id": existing["id"],
                "type": mtype, "subject": subject, "body": body,
            }
            for key in ("entity_type", "relations", "due_date"):
                if args.get(key):
                    proposal[key] = args[key]
            # A yes/no question, because "which is correct?" cannot be answered
            # "yes" - and "yes" is what people say.
            question = (
                f"{existing['subject']} is recorded as \"{existing['body']}\". "
                f"Replace that with \"{body}\"?"
            )
            self._hold_for_confirmation(existing["id"], question, proposal)
            return {
                "needs_confirmation": True,
                "memory_id": existing["id"],
                "currently": f"{existing['subject']}: {existing['body']}",
                "proposed": f"{subject}: {body}",
                "ask": question,
                "if_they_agree": proposal,
                "note": "Write nothing yet. Ask exactly this yes/no question. If "
                        "they agree next turn, call remember with if_they_agree.",
            }

        statement_id = self._write_statement(
            args.get("_utterance") or f"{subject}: {body}".strip(": "), "add"
        )
        candidate = Candidate(
            type=mtype,
            subject=subject,
            body=body,
            stance="asserted",
            explicit=True,
            quote=str(args.get("_utterance") or body or subject),
            entity_type=str(args.get("entity_type") or "") or None,
            first_person=False,
            due_date=str(args.get("due_date") or "") or None,
            episode_id="",
        )
        decisions = promote(
            self.conn, [candidate], from_user=True, statement_id=statement_id,
            supersedes=supersedes,
        )

        relations = [
            Relation(
                src=str(r.get("src") or ""), relation=str(r.get("relation") or ""),
                dst=str(r.get("dst") or ""), episode_id="",
            )
            for r in (args.get("relations") or [])
            if r.get("src") and r.get("dst") and r.get("relation") in policy.RELATIONS
        ]
        if relations:
            if supersedes:
                # A revision replaces what the person previously asserted, so the
                # relation it asserted goes too. Without this, saying "Ashwin
                # moved to Beacon" leaves the old works_on->DSPM edge in place and
                # the graph reports him on both. Only user-asserted relations are
                # retired (episode_id IS NULL); ones drawn from dictations are
                # historical evidence and stay.
                for relation in relations:
                    self.conn.execute(
                        "DELETE FROM relations WHERE src_norm = ? AND relation = ?"
                        " AND episode_id IS NULL",
                        (policy.normalise_entity(relation.src), relation.relation),
                    )
            stage_relations(self.conn, relations)

        promoted = [d for d in decisions if d.status == "promoted"]
        refused = [d for d in decisions if d.status == "rejected"]

        # Rebuild whenever an entity row changed, not only when relations were
        # given. Edges bind to row ids, and promoting "DSPM: run by Rahul"
        # supersedes the old DSPM row - so without a rebuild every works_on edge
        # still pointed at the retired row, and "who is working on DSPM" lost
        # the whole team it used to reach through the graph.
        if relations or (mtype == "entity" and promoted):
            build_edges(self.conn)
        self.conn.commit()

        # Answering settles the question. Resolved here rather than by a separate
        # step, because a clarification that stays open after the person has
        # answered it would be asked again - which is exactly the nagging the
        # design is meant to avoid.
        resolved = self._resolve_clarifications(subject, [d.memory_id for d in promoted])
        if promoted:
            self.evidence[statement_id] = Evidence(
                ref=statement_id, kind="record",
                text=f"You told Kivi: {candidate.quote}",
            )
        out: dict[str, Any] = {
            "ok": bool(promoted),
            "remembered": [
                {"memory_id": d.memory_id, "subject": d.subject, "reason": d.reason}
                for d in promoted
            ],
            "refused": [{"subject": d.subject, "reason": d.reason} for d in refused],
            "relations_written": len(relations),
            "statement_id": statement_id,
            "note": f"cite this as [{statement_id}] - it renders as 'you told Kivi'",
        }
        if resolved:
            out["resolved_questions"] = resolved

        # Say what happened, even when nothing was written.
        #
        # Evidence is only registered on promotion, so "add Ashwin to DSPM" when
        # Ashwin is already recorded left the turn with no evidence and no
        # artifact - and it fell through to "I don't have anything about that in
        # your dictations". Nothing was being looked up, so that answer was both
        # wrong and baffling. An instruction always produces an outcome worth
        # reporting, including the outcome "already known".
        if promoted:
            outcome = "Recorded: " + "; ".join(
                f"{d.subject}" + (f" - {d.reason}" if d.reason else "")
                for d in promoted
            )
        elif refused:
            outcome = (
                "Nothing new was written. " + "; ".join(
                    f"{d.subject}: {d.reason}" for d in refused
                )
            )
        else:
            outcome = "Nothing new was written - Kivi already had this."
        self.artifacts.append({
            "kind": "remember", "outcome": outcome,
            "statement_id": statement_id, "written": bool(promoted),
        })
        return out

    def _hold_for_confirmation(
        self, memory_id: str, question: str, proposal: dict[str, Any]
    ) -> None:
        """Keep a proposed change until the person answers, in this conversation.

        Without this the question existed only inside the turn that asked it, and
        a "yes" on the next turn had nothing to agree to. One open question per
        memory per conversation: asking again replaces the proposal rather than
        stacking a second one.
        """
        if not self.session_id:
            return
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        payload = json.dumps(proposal, ensure_ascii=False)
        updated = self.conn.execute(
            "UPDATE clarifications SET question = ?, proposed_json = ?, created_at = ?"
            " WHERE session_id = ? AND memory_id = ? AND kind = 'confirm'"
            " AND status = 'open'",
            (question, payload, now, self.session_id, memory_id),
        ).rowcount
        if not updated:
            cid = "clr_" + hashlib.sha256(
                f"{self.session_id}|{memory_id}|{now}".encode("utf-8")
            ).hexdigest()[:16]
            self.conn.execute(
                "INSERT INTO clarifications (id, session_id, memory_id, question,"
                " status, created_at, kind, proposed_json)"
                " VALUES (?, ?, ?, ?, 'open', ?, 'confirm', ?)",
                (cid, self.session_id, memory_id, question, now, payload),
            )
        self.conn.commit()

    def _resolve_clarifications(
        self, subject: str, memory_ids: list[str | None]
    ) -> list[str]:
        """Close any open question this answer settles.

        Matched on the memory the question was about, or on the subject name -
        the person answers "he's on Beacon now", not "resolve clar_abc123".
        """
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        ids = [m for m in memory_ids if m]
        placeholders = ",".join("?" * len(ids)) if ids else "NULL"
        rows = self.conn.execute(
            "SELECT c.id, c.question FROM clarifications c"
            " LEFT JOIN memories m ON m.id = c.memory_id"
            " WHERE c.status = 'open'"
            "   AND (LOWER(COALESCE(m.subject, '')) = LOWER(?)"
            f"        OR c.memory_id IN ({placeholders}))",
            (subject, *ids),
        ).fetchall()

        closed: list[str] = []
        for row in rows:
            self.conn.execute(
                "UPDATE clarifications SET status = 'resolved', resolved_at = ?"
                " WHERE id = ?",
                (now, row["id"]),
            )
            closed.append(row["question"])
        if closed:
            self.conn.commit()
        return closed

    def _change(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        memory_id = str(args.get("memory_id") or "")
        row = self.conn.execute(
            "SELECT id, subject, body FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return {"error": f"no memory {memory_id}"}

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        if action == "forget":
            self.conn.execute(
                "UPDATE memories SET status = 'forgotten', valid_to = ? WHERE id = ?",
                (now, memory_id),
            )
            message = "Forgotten. Your dictations are untouched."
        elif action == "confirm":
            self.conn.execute(
                "UPDATE memories SET pinned = 1, confidence = 1.0, source = 'user'"
                " WHERE id = ?",
                (memory_id,),
            )
            message = "Confirmed and pinned, so extraction will not overwrite it."
        else:  # revise: retire the old value, then add the new one
            self.conn.execute(
                "UPDATE memories SET status = 'superseded', valid_to = ? WHERE id = ?",
                (now, memory_id),
            )
            self.conn.commit()
            # Pass the retired id through so the new version points back at it.
            # Without the link the two rows are unrelated, and the interface can
            # no longer show what the value used to be or when it changed.
            return self._add(
                {**args, "subject": args.get("subject") or row["subject"]},
                supersedes=memory_id,
            )

        utterance = args.get("_utterance") or f"{action} {row['subject']}"
        statement_id = self._write_statement(utterance, action)
        # An answer settles any open question about this memory, whichever way
        # it went - "no, keep it" closes a confirmation as surely as "yes".
        self._resolve_clarifications(row["subject"], [memory_id])
        self.conn.commit()

        # Forget and confirm used to return with no artifact and no evidence, so
        # a successful "forget that" reached the closing step with nothing to
        # report and was answered "I don't have that" - the same wrong reply as
        # an unrecorded instruction, for the opposite reason.
        self.evidence[statement_id] = Evidence(
            ref=statement_id, kind="record", text=f"You told Kivi: {utterance}",
        )
        self.artifacts.append({
            "kind": "remember", "outcome": f"{message} ({row['subject']})",
            "statement_id": statement_id, "written": True,
        })
        return {
            "ok": True, "memory": row["subject"], "message": message,
            "statement_id": statement_id,
            "note": f"cite this as [{statement_id}] - it renders as 'you told Kivi'",
        }

    def _write_statement(self, text: str, intent: str) -> str:
        """Record what the person said, so the memory it wrote can be cited.

        Deliberately not an episode. Episodes are what was dictated and are
        immutable; a chat instruction is a different kind of thing and gets its
        own table. The `say_` prefix makes the interface render the citation as
        "you told Kivi" rather than as a record id.
        """
        now = dt.datetime.now(dt.timezone.utc)
        sid = "say_" + hashlib.sha256(
            f"{text}|{now.isoformat()}".encode("utf-8")
        ).hexdigest()[:16]
        self.conn.execute(
            "INSERT OR IGNORE INTO statements"
            " (id, session_id, turn_id, text, intent, ts, ts_epoch, created_at)"
            " VALUES (?, ?, NULL, ?, ?, ?, ?, ?)",
            (
                sid, self.session_id, str(text)[:2000], intent,
                now.isoformat(), int(now.timestamp()), now.isoformat(),
            ),
        )
        return sid

    # -- bookkeeping ---------------------------------------------------------

    def note_usage(self) -> None:
        """Record that these memories were actually used to answer something."""
        mark_used(self.conn, self.used_memory_ids)


def _style_rules(conn: sqlite3.Connection, profile) -> list[str]:
    """The rules to write by, model-read ones first.

    Both kinds exist for a reason. The counted features are cheap, always
    current, and explainable ("no greeting in 12 of 14 messages"). The model's
    rules catch what no detector was written for - idiom, when a nickname is used
    instead of a name, how directness shifts. When both are present the model's
    win, because they are the ones that carry conditions.
    """
    if profile is None or not profile.active:
        return []
    row = conn.execute(
        "SELECT rules_json FROM recipient_styles WHERE recipient_norm = ?",
        (profile.recipient_norm,),
    ).fetchone()
    if row:
        try:
            learned = [r for r in json.loads(row["rules_json"] or "[]") if r]
        except (TypeError, ValueError):
            learned = []
        if learned:
            return learned
    return style.instructions(profile)


def _local_when(row) -> str:
    """Render an episode row at the speaker's own wall clock."""
    offset = dt.timedelta(minutes=int(row["tz_offset_min"] or 0))
    local = dt.datetime.fromtimestamp(int(row["ts_epoch"]), dt.timezone.utc) + offset
    return local.strftime("%Y-%m-%d %H:%M")


def _local_day_start(date_str: str) -> dt.datetime:
    return dt.datetime.fromisoformat(f"{date_str}T00:00:00").replace(
        tzinfo=dt.timezone.utc
    ) - dt.timedelta(hours=14)  # widest plausible offset, so no day is clipped


def _local_day_end(date_str: str) -> dt.datetime:
    return dt.datetime.fromisoformat(f"{date_str}T23:59:59").replace(
        tzinfo=dt.timezone.utc
    ) + dt.timedelta(hours=14)
