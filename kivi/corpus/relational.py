"""Generate dictations addressed to specific people, in that person's style.

The model writes the surface text; the style it must write in is specified here,
in Python, per recipient. That separation is what makes the evaluation mean
anything: the profile Kivi *learns* is scored against the style we *asked for*,
not against a description the model also produced.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import random
from pathlib import Path
from typing import Any

from kivi.corpus.recipients import RECIPIENTS, Recipient, expected_profile
from kivi.llm.gemini import GeminiClient, LLMError

log = logging.getLogger(__name__)

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
BATCH = 10

SYSTEM = """You write realistic voice-dictation data: messages one person dictates
to a specific other person.

For each item produce TWO versions:

1. `raw_asr` - what a speech recogniser outputs. All lowercase, NO punctuation at
   all, natural disfluencies (um, uh, so, yeah), names sometimes mangled
   phonetically.
2. `formatted` - what a good dictation product writes instead: correct
   punctuation and capitalisation, disfluencies removed, names spelled right.
   Same content, same voice, same length.

THE STYLE RULES FOR EACH ITEM ARE ABSOLUTE. If the item says to open with a
particular greeting, `formatted` must open with exactly that. If it says no
greeting, it must start straight into the content. If it gives a length, stay
inside it. These are the point of the exercise - a message that ignores its
style rule is useless data.

Write only the message itself. Never describe what you are doing."""

SCHEMA: dict[str, Any] = {
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


def _spec(recipient: Recipient, topic: str, idx: int) -> str:
    lo, hi = recipient.words
    lines = [
        f"- idx {idx} | to: {recipient.name} ({recipient.who})",
        f"  topic: {topic}",
        f"  length: between {lo} and {hi} words",
        f"  style: {recipient.register}",
    ]
    if recipient.greeting:
        lines.append(f'  MUST open with exactly: "{recipient.greeting}"')
    else:
        lines.append("  MUST NOT open with any greeting - start with the content")
    if recipient.signoff:
        lines.append(f'  MUST end with a sign-off like: "{recipient.signoff}"')
    else:
        lines.append("  MUST NOT add any sign-off")
    return "\n".join(lines)


def _timestamps(rng: random.Random, recipient: Recipient, count: int) -> list[dt.datetime]:
    start = dt.datetime(2026, 6, 22, tzinfo=IST)
    hours = (9, 10, 11, 12, 14, 15, 16, 17, 18) if recipient.relation == "boss" \
        else (7, 8, 9, 13, 18, 19, 20, 21, 22)
    out = []
    for _ in range(count):
        day = start + dt.timedelta(days=rng.randint(0, 74))
        out.append(day.replace(hour=rng.choice(hours), minute=rng.randint(0, 59),
                               second=rng.randint(0, 59)))
    out.sort()
    return out


def generate_for(
    recipient: Recipient,
    client: GeminiClient,
    *,
    seed: int = 11,
    progress=None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + len(recipient.name))
    stamps = _timestamps(rng, recipient, recipient.volume)
    records: list[dict[str, Any]] = []

    for start in range(0, recipient.volume, BATCH):
        idxs = list(range(start, min(start + BATCH, recipient.volume)))
        specs = [
            _spec(recipient, rng.choice(recipient.topics), i) for i in idxs
        ]
        prompt = (
            f"Write {len(idxs)} dictations, each addressed to {recipient.name}.\n\n"
            + "\n".join(specs)
        )
        try:
            result = client.generate(
                prompt, system=SYSTEM, schema=SCHEMA,
                temperature=0.9, max_output_tokens=8192,
            )
            payload = result.json()
        except LLMError as exc:
            log.warning("batch for %s failed: %s", recipient.name, exc)
            continue

        for item in payload.get("records", []):
            try:
                idx = int(item["idx"])
            except (KeyError, TypeError, ValueError):
                continue
            raw = (item.get("raw_asr") or "").strip()
            formatted = (item.get("formatted") or "").strip()
            if not raw or not formatted or idx >= len(stamps):
                continue
            records.append({
                "id": f"rel-{recipient.name.lower()}-{idx:04d}",
                "ts": stamps[idx].isoformat(),
                "app": rng.choice(recipient.apps),
                "style_id": "casual" if recipient.relation != "boss" else "formal",
                "duration_ms": max(1500, len(raw) * 55),
                "raw_asr": raw,
                "formatted": formatted,
                "recipient": recipient.name,
                "meta": {
                    "corpus": "relational",
                    "recipient": recipient.name,
                    "relation": recipient.relation,
                    "language": "en-IN",
                },
            })
        if progress:
            progress(len(records))

    return records


def generate_all(
    client: GeminiClient, *, seed: int = 11, out_dir: Path | None = None, progress=None
) -> tuple[Path, Path, list[dict[str, Any]]]:
    everything: list[dict[str, Any]] = []
    for recipient in RECIPIENTS:
        everything.extend(
            generate_for(recipient, client, seed=seed, progress=progress)
        )

    out_dir = out_dir or Path("data/corpus")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "relational_episodes.jsonl"
    truth_path = out_dir / "relational_ground_truth.json"

    everything.sort(key=lambda r: r["ts"])
    with open(corpus_path, "w", encoding="utf-8") as handle:
        for record in everything:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for record in everything:
        counts[record["recipient"]] = counts.get(record["recipient"], 0) + 1

    truth_path.write_text(
        json.dumps(
            {
                "profile": "relational",
                "seed": seed,
                "record_count": len(everything),
                "recipients": [
                    {**expected_profile(r), "generated": counts.get(r.name, 0)}
                    for r in RECIPIENTS
                ],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return corpus_path, truth_path, everything
