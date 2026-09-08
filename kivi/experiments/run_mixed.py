"""One mixed dictation log, three people, three kinds of habit - and the
resulting profiles written to the database.

The log interleaves all three recipients in time, which is what a real dictation
stream looks like. Separating them is the learner's job, not the fixture's.

Learning is sequential per person, in three passes, so the report shows what was
known after a third of their messages, two thirds and all of them. Profiles are
persisted to `recipient_styles` through the normal pipeline - ingest, then
counted features, then the model's reading - rather than being kept in memory for
the duration of a demo.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kivi.experiments import learn_prompt
from kivi.experiments.patterns import PEOPLE, PersonSpec, classify
from kivi.llm.gemini import GeminiClient, LLMError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
LOG_PATH = Path("data/experiments/mixed_log.jsonl")
PARTS = 3

GEN_SYSTEM = """You write realistic voice dictations - messages one person sends to
someone specific in their life.

For each item produce TWO versions:

1. `raw_asr` - what a speech recogniser outputs: all lowercase, no punctuation,
   natural disfluencies (um, uh, so, yeah).
2. `formatted` - what a good dictation product writes: correct punctuation and
   capitalisation, disfluencies removed, same content and tone.

The style instruction on each item is ABSOLUTE - it is the entire point of the
data. A message that ignores it is useless.

Vary sentence shape heavily between items. Never reuse a phrasing pattern. Write
only the message; never explain yourself."""

GEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "raw_asr": {"type": "string"},
                    "formatted": {"type": "string"},
                },
                "required": ["idx", "raw_asr", "formatted"],
            },
        }
    },
    "required": ["records"],
}

DRAFT_SYSTEM = """You write a short message on behalf of someone, to a specific
person, in exactly the way that person writes to them.

You are given what has been learned about how they write to this person. Follow
it literally, including any rule that depends on the situation - work out which
situation this message is before you choose how to write it.

Write only the message. No quotes, no preamble."""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(client: GeminiClient, seed: int = 5) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    start = dt.datetime(2026, 7, 1, tzinfo=IST)
    records: list[dict[str, Any]] = []

    for person in PEOPLE:
        plan = [
            (i, rng.choice(person.situations)) for i in range(person.volume)
        ]
        if person.outlier is not None:
            # Dropped in the middle third, so the final pass can show whether it
            # stayed an exception or wrongly hardened into a rule.
            plan[person.volume // 2] = (person.volume // 2, person.outlier)
        for offset in range(0, len(plan), 9):
            chunk = plan[offset : offset + 9]
            lines = [
                f"Write {len(chunk)} dictations to {person.name}, {person.who}.",
                "",
            ]
            for idx, situation in chunk:
                lines.append(
                    f"- idx {idx} | topic: {situation.topic}\n"
                    f"  style: {situation.instruction}"
                )
            try:
                payload = client.generate(
                    "\n".join(lines), system=GEN_SYSTEM, schema=GEN_SCHEMA,
                    temperature=0.95, max_output_tokens=4096,
                ).json()
            except LLMError:
                continue

            planned = dict(chunk)
            for entry in payload.get("records", []):
                try:
                    idx = int(entry["idx"])
                except (KeyError, TypeError, ValueError):
                    continue
                situation = planned.get(idx)
                raw = (entry.get("raw_asr") or "").strip()
                formatted = (entry.get("formatted") or "").strip()
                if not situation or not raw or not formatted:
                    continue
                ts = start + dt.timedelta(
                    days=idx, hours=rng.randint(7, 21), minutes=rng.randint(0, 59)
                )
                records.append({
                    "id": f"mix-{person.name.lower()}-{idx:03d}",
                    "ts": ts.isoformat(),
                    "app": person.app,
                    "recipient": person.name,
                    "raw_asr": raw,
                    "formatted": formatted,
                    "meta": {
                        "experiment": "mixed_patterns",
                        "relation": person.relation,
                        "branch": situation.branch,
                        "kind": person.kind,
                    },
                })

    # Interleaved by time, as a real stream would be.
    records.sort(key=lambda r: r["ts"])
    return records


def write_log(records: list[dict[str, Any]], path: Path = LOG_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_log(path: Path = LOG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fidelity(person: PersonSpec, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Did the planted rule actually reach the text? The gate before learning."""
    # The outlier is excluded: it is a deliberate one-off that does not belong to
    # either arm of the rule, so scoring it as a miss would understate how
    # faithfully the rule itself was planted.
    mine = [
        r for r in records
        if r["recipient"] == person.name and r["meta"]["branch"] != "outlier"
    ]
    correct = sum(
        1 for r in mine if classify(person, r["formatted"]) == r["meta"]["branch"]
    )
    return {
        "n": len(mine),
        "correct": correct,
        "rate": correct / len(mine) if mine else 0.0,
    }


# ---------------------------------------------------------------------------
# Learning and probing
# ---------------------------------------------------------------------------


@dataclass
class Stage:
    person: str
    stage: int
    seen: int
    profile: dict[str, Any]
    drafts: list[dict[str, Any]] = field(default_factory=list)

    def _sub(self, unseen: bool) -> list[dict[str, Any]]:
        return [d for d in self.drafts if d["unseen"] is unseen]

    @property
    def seen_score(self) -> float:
        s = self._sub(False)
        return sum(d["ok"] for d in s) / len(s) if s else 0.0

    @property
    def unseen_score(self) -> float:
        s = self._sub(True)
        return sum(d["ok"] for d in s) / len(s) if s else 0.0


def _draft(
    client: GeminiClient, person: PersonSpec, profile: dict[str, Any], instruction: str
) -> str:
    prompt = (
        f"HOW THIS PERSON WRITES TO {person.name.upper()} ({person.who}), "
        f"learned from their own messages:\n"
        f"{json.dumps(profile, indent=2, ensure_ascii=False)}\n\n"
        f"Now write this message: {instruction}"
    )
    try:
        return client.generate(
            prompt, system=DRAFT_SYSTEM, temperature=0.4, max_output_tokens=300
        ).text.strip()
    except LLMError as exc:
        return f"<error: {exc}>"


def learn_person(
    client: GeminiClient,
    person: PersonSpec,
    records: list[dict[str, Any]],
    *,
    parts: int = PARTS,
) -> list[Stage]:
    mine = [r for r in records if r["recipient"] == person.name]
    size = max(len(mine) // parts, 1)
    chunks = [
        mine[i * size : (i + 1) * size if i < parts - 1 else len(mine)]
        for i in range(parts)
    ]

    stages: list[Stage] = []
    profile: dict[str, Any] | None = None
    seen = 0

    for index, chunk in enumerate(chunks, start=1):
        if not chunk:
            continue
        texts = [r["formatted"] for r in chunk]
        label = f"{person.name} ({person.who})"
        prompt = (
            learn_prompt.first_pass_prompt(label, texts)
            if profile is None
            else learn_prompt.revision_prompt(label, profile, texts, seen)
        )
        # A truncated response is the common failure here, not a wrong one: the
        # schema is nested and the model runs out of output tokens mid-string.
        # Left unhandled it returns an EMPTY profile, the drafter gets no rules,
        # and the stage scores near zero - which reads exactly like the model
        # having failed to learn. That produced a 100% -> 33% -> 100% curve that
        # was entirely an artefact.
        #
        # So: retry with more room, and if it still fails, carry the previous
        # profile forward rather than replacing knowledge with nothing.
        learned: dict[str, Any] | None = None
        for max_tokens in (6144, 12288):
            try:
                learned = client.generate(
                    prompt, system=learn_prompt.SYSTEM, schema=learn_prompt.SCHEMA,
                    model=client.settings.gen_model_heavy,
                    temperature=0.1, max_output_tokens=max_tokens,
                ).json()
                break
            except LLMError:
                continue

        if learned is None:
            profile = dict(profile or {})
            profile["stage_note"] = (
                "model response could not be parsed; carried the previous "
                "profile forward rather than discarding what was known"
            )
        else:
            profile = learned

        seen += len(chunk)
        stage = Stage(person=person.name, stage=index, seen=seen, profile=profile)

        for instruction, expected in person.seen_probes:
            text = _draft(client, person, profile, instruction)
            got = classify(person, text)
            stage.drafts.append({
                "instruction": instruction, "expected": expected, "got": got,
                "ok": got == expected, "unseen": False, "text": text,
            })
        for instruction, expected in person.unseen_probes:
            text = _draft(client, person, profile, instruction)
            got = classify(person, text)
            stage.drafts.append({
                "instruction": instruction, "expected": expected, "got": got,
                "ok": got == expected, "unseen": True, "text": text,
            })

        stages.append(stage)

    return stages


def persist(conn: sqlite3.Connection, person: PersonSpec, profile: dict[str, Any]) -> bool:
    """Write the learned profile onto the recipient's row.

    Goes through `recipient_styles` rather than anywhere bespoke, so what the
    experiment produces is the same thing the product reads.
    """
    key = person.name.strip().lower()
    rules = [
        f"When {r.get('when')}, {r.get('then')}"
        for r in profile.get("conditional_rules", [])
    ] + list(profile.get("constants", []))
    forms = "; ".join(
        f"{f.get('form')} when {f.get('when')}"
        for f in profile.get("address_forms", [])
    )
    cur = conn.execute(
        "UPDATE recipient_styles SET summary = ?, distinctive = ?, rules_json = ?,"
        " relation = ?, updated_at = ? WHERE recipient_norm = ?",
        (
            profile.get("summary", ""), forms,
            json.dumps(rules, ensure_ascii=False),
            person.relation, dt.datetime.now(dt.timezone.utc).isoformat(), key,
        ),
    )
    conn.commit()
    return cur.rowcount > 0
