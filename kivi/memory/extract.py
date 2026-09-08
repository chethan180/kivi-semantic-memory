"""Gate 1 (salience) and Gate 2 (candidate extraction).

The model's job here is linguistic only: identify what kind of thing was said,
who or what it was about, and - critically - whether the speaker was asserting
something about their own world or relaying someone else's. It does not decide
whether anything is worth remembering. That is Gate 3, in promote.py, and it is
deterministic.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from kivi.llm.gemini import GeminiClient, LLMError, LLMResult
from kivi.memory import policy

log = logging.getLogger(__name__)

BATCH_SIZE = 8


@dataclass
class Candidate:
    type: str
    subject: str
    body: str
    stance: str
    explicit: bool
    quote: str
    entity_type: str | None = None
    first_person: bool = False
    due_date: str | None = None
    episode_id: str = ""


@dataclass
class Relation:
    src: str
    relation: str
    dst: str
    episode_id: str = ""


@dataclass
class Extraction:
    episode_id: str
    candidates: list[Candidate] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    skipped_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gate 1: salience
# ---------------------------------------------------------------------------


def salient(text: str, recent: set[str]) -> tuple[bool, str | None]:
    """Cheap pre-filter. Returns (keep, reason_if_dropped).

    Note what this does NOT test: whether the utterance is in the first person.
    An earlier draft dropped anything without "I", which would have discarded
    almost every entity statement in the corpus - "Umar is working on DSPM" is
    third-person in grammar but first-hand in stance. First-person is a condition
    on *commitments* specifically, applied in Gate 3, not a global filter.
    """
    stripped = text.strip()
    if len(stripped.split()) < policy.MIN_WORDS:
        return False, f"too short (<{policy.MIN_WORDS} words)"
    if stripped.lower() in recent:
        return False, "duplicate of a recent dictation"
    return True, None


# ---------------------------------------------------------------------------
# Gate 2: candidate extraction
# ---------------------------------------------------------------------------

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": list(policy.MEMORY_TYPES)},
                                "subject": {"type": "string"},
                                "body": {"type": "string"},
                                "entity_type": {
                                    "type": "string", "enum": list(policy.ENTITY_TYPES)
                                },
                                "stance": {"type": "string", "enum": list(policy.STANCES)},
                                "explicit": {"type": "boolean"},
                                "first_person": {"type": "boolean"},
                                "due_date": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["type", "subject", "stance", "explicit", "quote"],
                        },
                    },
                    "relations": {
                        "type": "array",
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
                },
                "required": ["idx", "candidates", "relations"],
            },
        }
    },
    "required": ["results"],
}


def _system_prompt() -> str:
    ignore = "\n".join(f"  - {r.key}: {r.why}" for r in policy.IGNORE_LIST)
    return f"""You read one person's voice dictations and identify what a memory
system could learn from them. You do NOT decide what is worth keeping - a
separate deterministic stage does that. Your job is to describe accurately.

For each dictation, extract candidates of these types:

  entity      a person, project, org, place, thing or term that recurs in this
              person's world. subject = its canonical name.
  preference  how this person wants their writing or work done.
  commitment  something this person undertook to do, with a deadline.

For every candidate you MUST set `stance`, which is the most important field:

  asserted    the SPEAKER is stating something about their own world, work,
              plans or preferences. Only this stance can ever be remembered.
  transcribed the speaker is dictating content that is ABOUT someone else, or
              relaying what a third party said or disclosed. Notes about a
              patient, a client, a colleague's review, a customer's incident.
  reported    hearsay: the speaker passing on something they were told, where
              they are not the source and not the subject.
  unclear     you genuinely cannot tell.

Getting stance right matters more than finding every candidate. When a dictation
is a person writing ABOUT another person, the stance is `transcribed`, even if
the sentence is grammatically simple and sounds like a fact.

Also set:
  explicit      true if the person said it outright ("keep my messages short"),
                false if you are inferring it from how they wrote.
  first_person  true if the SPEAKER is the one who acts or undertakes.
  due_date      YYYY-MM-DD if a deadline is stated or clearly implied, else omit.
  quote         the VERBATIM span from the dictation that supports this. Copy it
                exactly. Do not paraphrase. This is the provenance.

Extract `relations` between named entities when the dictation states one:
{', '.join(policy.RELATIONS)}.

These subjects must NEVER be extracted as candidates, about anyone including the
speaker:
{ignore}

If a dictation contains nothing durable - a grocery list, a one-word reply, a
room number - return an empty candidates array for it. That is a correct and
common answer. Do not invent memories to fill space.

Return one result object per input idx."""


def _batch_prompt(episodes: list[sqlite3.Row]) -> str:
    lines = []
    for i, row in enumerate(episodes):
        lines.append(f"--- idx {i} | app: {row['app']} | {row['ts'][:16]} ---")
        lines.append(row["formatted"])
        lines.append("")
    return "\n".join(lines)


def _parse(payload: Any, episodes: list[sqlite3.Row]) -> dict[int, Extraction]:
    out: dict[int, Extraction] = {}
    for item in payload.get("results", []):
        try:
            idx = int(item["idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= idx < len(episodes):
            continue
        episode_id = episodes[idx]["id"]
        extraction = Extraction(episode_id=episode_id)

        for raw in item.get("candidates") or []:
            ctype = (raw.get("type") or "").strip()
            subject = (raw.get("subject") or "").strip()
            stance = (raw.get("stance") or "unclear").strip()
            if ctype not in policy.MEMORY_TYPES or not subject:
                continue
            if stance not in policy.STANCES:
                stance = "unclear"
            entity_type = (raw.get("entity_type") or "").strip() or None
            if entity_type not in policy.ENTITY_TYPES:
                entity_type = None
            due = (raw.get("due_date") or "").strip() or None
            extraction.candidates.append(
                Candidate(
                    type=ctype,
                    subject=subject,
                    body=(raw.get("body") or "").strip(),
                    stance=stance,
                    explicit=bool(raw.get("explicit")),
                    quote=(raw.get("quote") or "").strip(),
                    entity_type=entity_type,
                    first_person=bool(raw.get("first_person")),
                    due_date=due,
                    episode_id=episode_id,
                )
            )

        for raw in item.get("relations") or []:
            relation = (raw.get("relation") or "").strip()
            src = (raw.get("src") or "").strip()
            dst = (raw.get("dst") or "").strip()
            if relation in policy.RELATIONS and src and dst and src != dst:
                extraction.relations.append(
                    Relation(src=src, relation=relation, dst=dst, episode_id=episode_id)
                )

        out[idx] = extraction
    return out


def extract_batch(
    client: GeminiClient, episodes: list[sqlite3.Row], model: str | None = None
) -> tuple[dict[int, Extraction], LLMResult | None]:
    """Run Gate 2 over a batch. Returns extractions keyed by position in `episodes`."""
    if not episodes:
        return {}, None
    try:
        result = client.generate(
            _batch_prompt(episodes),
            system=_system_prompt(),
            schema=SCHEMA,
            model=model,
            temperature=0.0,
            max_output_tokens=8192,
        )
        payload = result.json()
    except LLMError as exc:
        log.warning("extraction batch failed (%d episodes): %s", len(episodes), exc)
        return {}, None
    return _parse(payload, episodes), result
