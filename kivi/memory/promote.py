"""Gate 3: promotion. No model calls happen in this file.

Every decision here is a rule, which is what makes "why was this not remembered?"
answerable. Rejections are written to `candidates` with a reason rather than
dropped, so what the system deliberately ignored is a table you can query.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Iterable

from kivi.memory import policy
from kivi.memory.extract import Candidate, Relation


@dataclass
class Decision:
    candidate_key: str
    subject: str
    type: str
    status: str          # promoted | pending | rejected
    reason: str
    memory_id: str | None = None
    evidence_count: int = 1
    superseded_id: str | None = None


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def candidate_key(candidate: Candidate) -> str:
    """Identity of a candidate across episodes.

    Two episodes mentioning "DSPM" must land on the same key, which is what makes
    repetition-based promotion work at all. For entities the subject is
    additionally stripped of role words, so "Beacon", "the Beacon project" and
    "Beacon integration" accumulate evidence together instead of splitting it
    three ways and none of them ever reaching the threshold.
    """
    if candidate.type == "entity":
        subject = policy.normalise_entity(candidate.subject)
    else:
        subject = " ".join(candidate.subject.lower().split())
    return f"{candidate.type}:{subject}"


def _memory_id(key: str, salt: str = "") -> str:
    return "mem_" + hashlib.sha256(f"{key}|{salt}".encode("utf-8")).hexdigest()[:16]


def _candidate_id(key: str) -> str:
    return "cand_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def screen(candidate: Candidate, from_user: bool = False) -> tuple[bool, str]:
    """Hard refusals, applied before any evidence is counted.

    Checked here in code as well as described to the extractor in the prompt: a
    prompt instruction is a request, a code path is a guarantee.

    `from_user` marks a direct instruction in conversation. It relaxes the stance
    test - by definition the person is asserting - but it does NOT relax the
    third-party rules. You may tell Kivi anything about yourself; nobody gets to
    hand it someone else's medical history.
    """
    if not from_user and candidate.stance not in policy.PROMOTABLE_STANCES:
        return False, (
            f"stance={candidate.stance}: the speaker was not asserting this about "
            f"their own world"
        )

    # Kivi still does not LEARN about other people - that is what stance covers.
    # It no longer redacts or refuses when asked about a dictation, though: the
    # words are the person's own record, and hiding them back from the person who
    # said them was the wrong trade.
    rule = policy.ignore_match(f"{candidate.subject} {candidate.body}")
    if rule is not None:
        return False, f"ignore_list:{rule.key} - {rule.why}"

    if candidate.type not in policy.PROMOTION:
        return False, f"unknown memory type {candidate.type!r}"

    spec = policy.PROMOTION[candidate.type]
    if spec.requires_first_person and not candidate.first_person:
        return False, "commitment without a first-person subject: not the user's promise"
    if spec.requires_time_anchor and not candidate.due_date:
        return False, "commitment without a resolvable date: nothing to expire"

    return True, "passed screening"


def required_evidence(candidate: Candidate) -> int:
    spec = policy.PROMOTION[candidate.type]
    return (
        spec.evidence_required_if_explicit
        if candidate.explicit
        else spec.evidence_required
    )


def confidence_for(candidate: Candidate) -> float:
    spec = policy.PROMOTION[candidate.type]
    if candidate.explicit:
        return spec.confidence
    # Inferred from behaviour rather than stated: capped permanently. A guess
    # about a person should never be able to present itself as a fact.
    return min(spec.confidence, spec.inferred_confidence_cap)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _upsert_candidate(
    conn: sqlite3.Connection,
    candidate: Candidate,
    key: str,
    status: str,
    reason: str,
    memory_id: str | None,
) -> int:
    """Record the candidate and return its accumulated evidence count.

    Evidence is counted per distinct episode: the same episode restating a thing
    twice is one piece of evidence, not two.
    """
    cid = _candidate_id(key)
    now = _utcnow()
    row = conn.execute(
        "SELECT evidence_count, status FROM candidates WHERE id = ?", (cid,)
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO candidates"
            " (id, type, subject, body, entity_type, stance, explicit, quote,"
            "  episode_id, evidence_count, status, reason, memory_id,"
            "  first_seen, last_seen)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cid, candidate.type, candidate.subject, candidate.body,
                candidate.entity_type, candidate.stance, int(candidate.explicit),
                candidate.quote,
                # NULL, not "", for a candidate with no episode behind it. An
                # empty string is a value, and the foreign key to episodes
                # rejects it; only NULL means "no episode".
                candidate.episode_id or None,
                1, status, reason, memory_id, now, now,
            ),
        )
        return 1

    # Count the evidence rows rather than incrementing a running total.
    #
    # The previous version asked "have I seen this episode before?" and added one
    # if not - but `_record_evidence` runs FIRST and had already inserted that
    # exact row, so the answer was always yes and the count never moved past 1.
    # Every threshold in the system silently became unreachable: entities needing
    # two episodes never promoted from extraction, and inferred preferences
    # needing three never promoted at all.
    #
    # Deriving the count from the table cannot drift out of step with it.
    count = int(
        conn.execute(
            "SELECT COUNT(DISTINCT episode_id) FROM candidate_evidence"
            " WHERE candidate_id = ?",
            (cid,),
        ).fetchone()[0]
    ) or int(row["evidence_count"])
    conn.execute(
        "UPDATE candidates SET evidence_count = ?, status = ?, reason = ?,"
        " memory_id = COALESCE(?, memory_id), last_seen = ? WHERE id = ?",
        (count, status, reason, memory_id, now, cid),
    )
    return count


def _record_evidence(
    conn: sqlite3.Connection,
    key: str,
    candidate: Candidate,
    statement_id: str | None = None,
) -> None:
    """Evidence rows are keyed on episodes; a statement has no episode.

    Conversational candidates skip this table entirely - they are promoted on
    the instruction alone, so there is no counter to accumulate - and their
    provenance is written straight to memory_sources instead.
    """
    if statement_id is not None or not candidate.episode_id:
        return
    conn.execute(
        "INSERT OR IGNORE INTO candidate_evidence"
        " (candidate_id, episode_id, quote, created_at) VALUES (?, ?, ?, ?)",
        (_candidate_id(key), candidate.episode_id, candidate.quote, _utcnow()),
    )


def _active_memory(conn: sqlite3.Connection, ctype: str, subject: str) -> sqlite3.Row | None:
    """Find the live memory a candidate refers to, matching the way keys match."""
    if ctype == "entity":
        target = policy.normalise_entity(subject)
        for row in conn.execute(
            "SELECT * FROM memories WHERE type = 'entity' AND status = 'active'"
            " ORDER BY created_at DESC"
        ).fetchall():
            if policy.normalise_entity(row["subject"]) == target:
                return row
        return None
    return conn.execute(
        "SELECT * FROM memories WHERE type = ? AND LOWER(subject) = LOWER(?)"
        " AND status = 'active' ORDER BY created_at DESC LIMIT 1",
        (ctype, subject),
    ).fetchone()


def _surface_forms(conn: sqlite3.Connection, key: str) -> list[str]:
    """Every spelling seen for a candidate, longest-first.

    The canonical name is the SHORTEST distinct form: speech adds role words
    ("the Beacon project") rather than removing them, so the shortest surviving
    form is almost always the actual name.
    """
    rows = conn.execute(
        "SELECT DISTINCT surface FROM candidate_surface WHERE candidate_id = ?",
        (_candidate_id(key),),
    ).fetchall()
    return sorted({row["surface"] for row in rows}, key=lambda s: (len(s), s))


def _record_surface(conn: sqlite3.Connection, key: str, subject: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO candidate_surface (candidate_id, surface, created_at)"
        " VALUES (?, ?, ?)",
        (_candidate_id(key), subject.strip(), _utcnow()),
    )


def _write_memory(
    conn: sqlite3.Connection,
    candidate: Candidate,
    key: str,
    confidence: float,
    supersedes: str | None = None,
    pinned: bool = False,
    source: str = "extraction",
) -> str:
    now = _utcnow()
    # Ids are derived from the subject, so a second memory about the same
    # subject collides with the first - and INSERT OR REPLACE then overwrites it
    # in place. That silently destroys history: the old value vanishes instead of
    # being superseded, and "this changed on Tuesday because you said X" has
    # nothing left to show. Salt whenever any row already holds this id.
    mid = _memory_id(key)
    if supersedes or conn.execute(
        "SELECT 1 FROM memories WHERE id = ?", (mid,)
    ).fetchone():
        mid = _memory_id(key, now)

    subject = candidate.subject
    aliases: list[str] = []
    if candidate.type == "entity":
        forms = _surface_forms(conn, key)
        if forms:
            # Shortest form wins as the canonical name; the rest become aliases.
            subject = forms[0]
            aliases = [f for f in forms[1:] if f.lower() != subject.lower()]

    conn.execute(
        "INSERT OR REPLACE INTO memories"
        " (id, type, subject, body, entity_type, canonical, aliases_json,"
        "  confidence, status, pinned, source, use_count, created_at,"
        "  valid_from, supersedes_id, due_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, 0,"
        "         ?, ?, ?, ?)",
        (
            mid, candidate.type, subject, candidate.body,
            candidate.entity_type, subject,
            json.dumps(aliases, ensure_ascii=False), confidence,
            int(pinned), source,
            now, now, supersedes, candidate.due_date,
        ),
    )
    return mid


def _link_sources(
    conn: sqlite3.Connection,
    memory_id: str,
    key: str,
    statement_id: str | None = None,
    quote: str | None = None,
) -> None:
    """Attach provenance: every episode that was evidence, or the statement.

    Provenance is polymorphic now. A memory the person asserted points at what
    they said to Kivi, not at a fabricated dictation - so "because you told me"
    is checkable in exactly the same way "because you dictated this" is.
    """
    now = _utcnow()
    if statement_id is not None:
        conn.execute(
            "INSERT OR IGNORE INTO memory_sources"
            " (memory_id, source_kind, source_id, quote, char_start, char_end, created_at)"
            " VALUES (?, 'statement', ?, ?, 0, 0, ?)",
            (memory_id, statement_id, quote, now),
        )
        return

    rows = conn.execute(
        "SELECT episode_id, quote FROM candidate_evidence WHERE candidate_id = ?",
        (_candidate_id(key),),
    ).fetchall()
    conn.executemany(
        "INSERT OR IGNORE INTO memory_sources"
        " (memory_id, source_kind, source_id, quote, char_start, char_end, created_at)"
        " VALUES (?, 'episode', ?, ?, 0, 0, ?)",
        [(memory_id, row["episode_id"], row["quote"], now) for row in rows],
    )


def _supersede(conn: sqlite3.Connection, old_id: str) -> None:
    conn.execute(
        "UPDATE memories SET status = 'superseded', valid_to = ? WHERE id = ?",
        (_utcnow(), old_id),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def promote(
    conn: sqlite3.Connection,
    candidates: Iterable[Candidate],
    *,
    from_user: bool = False,
    statement_id: str | None = None,
    supersedes: str | None = None,
) -> list[Decision]:
    """Apply Gate 3 to a batch of candidates. Returns one decision each.

    `from_user=True` means these came from a direct instruction in conversation
    rather than from extraction over a dictation. Such candidates bypass the
    evidence thresholds and land pinned: the thresholds exist to filter inference
    from noisy speech - two episodes is what stopped a mis-heard "Abbey" becoming
    a colleague - and an instruction from the person is not inference. Making
    someone say "add Ashwin" twice would be absurd.
    """
    decisions: list[Decision] = []

    for candidate in candidates:
        key = candidate_key(candidate)

        allowed, reason = screen(candidate, from_user=from_user)
        if not allowed:
            _upsert_candidate(conn, candidate, key, "rejected", reason, None)
            _record_evidence(conn, key, candidate, statement_id)
            decisions.append(
                Decision(key, candidate.subject, candidate.type, "rejected", reason)
            )
            continue

        _record_evidence(conn, key, candidate, statement_id)
        _record_surface(conn, key, candidate.subject)
        count = _upsert_candidate(conn, candidate, key, "pending", "accumulating", None)
        needed = 1 if from_user else required_evidence(candidate)

        if count < needed:
            reason = (
                f"{count}/{needed} episodes of evidence"
                f" ({'stated' if candidate.explicit else 'inferred'})"
            )
            conn.execute(
                "UPDATE candidates SET reason = ? WHERE id = ?",
                (reason, _candidate_id(key)),
            )
            decisions.append(
                Decision(key, candidate.subject, candidate.type, "pending", reason, None, count)
            )
            continue

        existing = _active_memory(conn, candidate.type, candidate.subject)
        # An instruction is not a guess: it lands at full confidence, and pinned
        # so a later extraction pass cannot quietly overwrite what was asked for.
        confidence = 1.0 if from_user else confidence_for(candidate)

        if existing is not None:
            if existing["pinned"] and not from_user:
                # A dictation contradicts something the person told Kivi
                # directly. Which wins is decided by RECENCY, not by which
                # source is nominally more authoritative.
                same_value = (
                    (existing["body"] or "").strip().lower()
                    == (candidate.body or "").strip().lower()
                )
                if same_value:
                    reason = "dictation agrees with what the person already said"
                    decisions.append(
                        Decision(key, candidate.subject, candidate.type, "rejected",
                                 reason, existing["id"], count)
                    )
                    continue

                episode_ts = _episode_time(conn, candidate.episode_id)
                statement_ts = _latest_statement_time(conn, existing["id"])

                if statement_ts and episode_ts and episode_ts > statement_ts:
                    # The dictation is NEWER than the instruction. Genuinely
                    # ambiguous - they may have changed their mind, or may just
                    # have mentioned the subject in passing. Both silent options
                    # are wrong, so neither is taken: the fact stands untouched
                    # and a question is raised for the conversation to settle.
                    _raise_clarification(conn, existing, candidate)
                    reason = (
                        "a later dictation disagrees with what you told Kivi - "
                        "asked rather than decided"
                    )
                    decisions.append(
                        Decision(key, candidate.subject, candidate.type, "pending",
                                 reason, existing["id"], count)
                    )
                    continue

                # The instruction is the more recent evidence. It stands.
                reason = (
                    "you told Kivi this more recently than the dictation says "
                    "otherwise"
                )
                decisions.append(
                    Decision(key, candidate.subject, candidate.type, "rejected",
                             reason, existing["id"], count)
                )
                continue
            same = (existing["body"] or "").strip() == (candidate.body or "").strip()
            if same:
                conn.execute(
                    "UPDATE memories SET confidence = MAX(confidence, ?) WHERE id = ?",
                    (confidence, existing["id"]),
                )
                _link_sources(conn, existing["id"], key, statement_id, candidate.quote)
                decisions.append(
                    Decision(key, candidate.subject, candidate.type, "promoted",
                             "reinforced an existing memory", existing["id"], count)
                )
                continue
            # Contradiction: version it, never overwrite in place.
            mid = _write_memory(
                conn, candidate, key, confidence, supersedes=existing["id"],
                pinned=from_user, source="user" if from_user else "extraction",
            )
            _supersede(conn, existing["id"])
            _link_sources(conn, mid, key, statement_id, candidate.quote)
            conn.execute(
                "UPDATE candidates SET status = 'promoted', reason = ?, memory_id = ?"
                " WHERE id = ?",
                ("superseded an earlier value", mid, _candidate_id(key)),
            )
            decisions.append(
                Decision(key, candidate.subject, candidate.type, "promoted",
                         f"superseded {existing['id']}", mid, count, existing["id"])
            )
            continue

        mid = _write_memory(
            conn, candidate, key, confidence, supersedes=supersedes,
            pinned=from_user, source="user" if from_user else "extraction",
        )
        _link_sources(conn, mid, key, statement_id, candidate.quote)
        reason = (
            "you told Kivi this directly"
            if from_user
            else f"promoted on {count}/{needed} episodes at confidence {confidence:.2f}"
        )
        conn.execute(
            "UPDATE candidates SET status = 'promoted', reason = ?, memory_id = ?"
            " WHERE id = ?",
            (reason, mid, _candidate_id(key)),
        )
        decisions.append(
            Decision(key, candidate.subject, candidate.type, "promoted", reason, mid, count)
        )

    return decisions


def _episode_time(conn: sqlite3.Connection, episode_id: str | None) -> str | None:
    if not episode_id:
        return None
    row = conn.execute(
        "SELECT ts FROM episodes WHERE id = ?", (episode_id,)
    ).fetchone()
    return row["ts"] if row else None


def _latest_statement_time(conn: sqlite3.Connection, memory_id: str) -> str | None:
    """When the person most recently told Kivi this, if they ever did."""
    return conn.execute(
        "SELECT MAX(st.ts) FROM memory_sources s"
        " JOIN statements st ON st.id = s.source_id"
        " WHERE s.memory_id = ? AND s.source_kind = 'statement'",
        (memory_id,),
    ).fetchone()[0]


def _raise_clarification(
    conn: sqlite3.Connection, existing: sqlite3.Row, candidate: Candidate
) -> str | None:
    """Record a question for Hey Kivi to ask, without changing anything.

    Deliberately not resolved here. Ingest is batch over hundreds of records with
    nobody to answer; the conversation is the only place a question can actually
    be put to someone. This writes the question and stops.
    """
    question = (
        f"You told me {existing['subject']} is "
        f"\"{existing['body'] or existing['subject']}\", but a more recent "
        f"dictation says \"{candidate.body or candidate.quote}\". Which is right?"
    )
    already = conn.execute(
        "SELECT id FROM clarifications WHERE memory_id = ? AND status = 'open'",
        (existing["id"],),
    ).fetchone()
    if already:
        return already["id"]

    cid = "clar_" + hashlib.sha256(
        f"{existing['id']}|{candidate.body}|{_utcnow()}".encode()
    ).hexdigest()[:16]
    conn.execute(
        "INSERT INTO clarifications"
        " (id, session_id, turn_id, candidate_id, memory_id, question, status,"
        "  created_at) VALUES (?, NULL, NULL, ?, ?, ?, 'open', ?)",
        (cid, _candidate_id(candidate_key(candidate)), existing["id"], question,
         _utcnow()),
    )
    return cid


def stage_relations(conn: sqlite3.Connection, relations: Iterable[Relation]) -> int:
    """Record observed relations. They become edges later, in build_edges()."""
    now = _utcnow()
    staged = 0
    for relation in relations:
        conn.execute(
            "INSERT OR IGNORE INTO relations"
            " (src_norm, dst_norm, src_raw, dst_raw, relation, episode_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                policy.normalise_entity(relation.src),
                policy.normalise_entity(relation.dst),
                relation.src, relation.dst, relation.relation,
                # NULL for a relation asserted in conversation: it has no
                # episode, and "" would fail the foreign key.
                relation.episode_id or None,
                now,
            ),
        )
        staged += 1
    return staged


def build_edges(conn: sqlite3.Connection) -> int:
    """Turn staged relations into graph edges between promoted entities.

    Run after all promotion is done. Only entities that cleared the gate get
    nodes - an edge pointing at something Kivi refused to remember would
    reintroduce it through the back door, and the graph would quietly become a
    second, ungoverned memory store.

    Edge weight is the number of distinct episodes asserting the relation, which
    is why this rebuilds from scratch rather than incrementing in place.
    """
    now = _utcnow()
    entities = {
        policy.normalise_entity(row["subject"]): row["id"]
        for row in conn.execute(
            "SELECT id, subject FROM memories"
            " WHERE type = 'entity' AND status = 'active'"
        ).fetchall()
    }
    if not entities:
        return 0

    # Weight is how much corroboration a relation has. Counting distinct episodes
    # alone gives a user-asserted relation a weight of ZERO - its episode_id is
    # NULL, so COUNT(DISTINCT episode_id) counts nothing - and since graph
    # expansion orders by weight, the one relation the person stated outright
    # ranked below every relation merely inferred from speech. Exactly backwards.
    #
    # A direct assertion is worth more than a few passing mentions, so it is
    # scored as such rather than merely rescued from zero.
    rows = conn.execute(
        "SELECT src_norm, dst_norm, relation,"
        "       COUNT(DISTINCT episode_id)"
        "       + CASE WHEN SUM(CASE WHEN episode_id IS NULL THEN 1 ELSE 0 END) > 0"
        "              THEN 3.0 ELSE 0 END AS weight"
        " FROM relations GROUP BY src_norm, dst_norm, relation"
    ).fetchall()

    conn.execute("DELETE FROM edges")
    written = 0
    for row in rows:
        src = entities.get(row["src_norm"])
        dst = entities.get(row["dst_norm"])
        if not src or not dst or src == dst:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO edges"
            " (src_type, src_id, dst_type, dst_id, relation, weight, created_at, updated_at)"
            " VALUES ('memory', ?, 'memory', ?, ?, ?, ?, ?)",
            (src, dst, row["relation"], float(row["weight"]), now, now),
        )
        written += 1
    return written


def expire_commitments(conn: sqlite3.Connection, today: str | None = None) -> int:
    """Retire commitments whose date has passed. Expired, never deleted."""
    if not policy.EXPIRE_COMMITMENTS:
        return 0
    today = today or dt.date.today().isoformat()
    cur = conn.execute(
        "UPDATE memories SET status = 'expired', valid_to = ?"
        " WHERE type = 'commitment' AND status = 'active'"
        "   AND due_at IS NOT NULL AND due_at < ?",
        (_utcnow(), today),
    )
    return max(cur.rowcount, 0)
