"""One person, learned in three passes.

The question this answers: is the model actually reading the messages, or is it
restating a pattern that was handed to it? A static habit ("always open with Hi
Amma") cannot tell those apart, because a rigid instruction survives a round trip
whether or not anything was learned.

So the habit planted here is CONDITIONAL and never stated to the learner:

    affectionate messages  ->  "Mama"
    routine / logistical   ->  "Amma"

Nothing tells the learner that two address forms exist, that they alternate, or
what governs the choice. It has to notice. And the test is generative rather than
descriptive: asked to write "I love you", a system that learned the rule produces
"I love you, Mama" - a system that memorised the commonest opener produces
"Amma".

The corpus is fed in three chronological parts, and the profile is revised after
each, so the transcript shows what was known after 15 messages, 30 and 45. A
capability that only appears at 45 is a different claim from one that appears at
15, and reporting a single number would hide which one is true.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kivi.llm.gemini import GeminiClient, LLMError

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
LOG_PATH = Path("data/experiments/amma_log.jsonl")
PARTS = 3

# The rule. Present in the data, never shown to the learner.
AFFECTIONATE = (
    "telling her you miss her",
    "saying you love her before a trip",
    "thanking her for something she cooked and sent",
    "comforting her after she sounded worried on a call",
    "wishing her on her birthday",
    "apologising for not calling for a few days",
    "telling her you are proud of her for something",
    "saying you are thinking of her today",
    "reassuring her that you are eating properly",
    "telling her the house feels empty without her cooking",
    "wishing her well before her hospital check-up",
    "saying goodnight affectionately",
)

ROUTINE = (
    "telling her what time you will reach home",
    "asking whether she took her tablets",
    "asking what she cooked today",
    "telling her the electrician is coming tomorrow",
    "asking her to keep the gas cylinder booking receipt",
    "confirming the train timing for next week",
    "asking if the maid came today",
    "telling her you will call after your meeting",
    "asking her to send the address for the courier",
    "reminding her about the temple visit on Friday",
    "telling her you paid the electricity bill",
    "asking whether Appa reached the bank",
    "confirming a doctor appointment time",
    "asking her to switch off the motor",
)

SYSTEM = """You write realistic voice dictations that one person sends to their mother.

For each item produce TWO versions:

1. `raw_asr` - what a speech recogniser outputs: all lowercase, no punctuation at
   all, natural disfluencies (um, uh, so, ya).
2. `formatted` - what a good dictation product writes: correct punctuation and
   capitalisation, disfluencies removed, same content and same warmth.

CRITICAL - how she is addressed:
- When the message is affectionate, emotional, tender, apologetic or celebratory,
  address her as "Mama".
- When the message is routine, practical or logistical, address her as "Amma".
- Vary WHERE the name appears - sometimes it opens the message, sometimes it sits
  mid-sentence, sometimes it closes. Do not put it in the same position every time.

Keep messages short, 8 to 22 words, in plain simple language with no work jargon.
This is a real family conversation, so vary the phrasing heavily - never reuse a
sentence pattern from an earlier item.

Write only the message. Never explain what you are doing."""

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


@dataclass
class Item:
    idx: int
    topic: str
    register: str      # affectionate | routine
    expected_form: str  # Mama | Amma


def plan(count: int = 45, seed: int = 3) -> list[Item]:
    """Which messages exist, and which address form each should carry."""
    rng = random.Random(seed)
    items: list[Item] = []
    for i in range(count):
        # Roughly 40% affectionate, shuffled through so neither form clusters at
        # one end of the timeline - a run of one kind would let the profile look
        # correct for the wrong reason.
        if rng.random() < 0.4:
            items.append(Item(i, rng.choice(AFFECTIONATE), "affectionate", "Mama"))
        else:
            items.append(Item(i, rng.choice(ROUTINE), "routine", "Amma"))
    return items


def generate(client: GeminiClient, count: int = 45, seed: int = 3) -> list[dict[str, Any]]:
    items = plan(count, seed)
    rng = random.Random(seed)
    start = dt.datetime(2026, 7, 1, tzinfo=IST)
    records: list[dict[str, Any]] = []

    for offset in range(0, len(items), 9):
        chunk = items[offset : offset + 9]
        lines = [f"Write {len(chunk)} dictations to the person's mother.", ""]
        for item in chunk:
            lines.append(
                f"- idx {item.idx} | register: {item.register} | "
                f"address her as \"{item.expected_form}\" | topic: {item.topic}"
            )
        try:
            payload = client.generate(
                "\n".join(lines), system=SYSTEM, schema=SCHEMA,
                temperature=0.95, max_output_tokens=4096,
            ).json()
        except LLMError:
            continue

        for entry in payload.get("records", []):
            try:
                idx = int(entry["idx"])
            except (KeyError, TypeError, ValueError):
                continue
            match = next((i for i in items if i.idx == idx), None)
            raw = (entry.get("raw_asr") or "").strip()
            formatted = (entry.get("formatted") or "").strip()
            if not match or not raw or not formatted:
                continue
            ts = start + dt.timedelta(
                days=idx, hours=rng.randint(7, 21), minutes=rng.randint(0, 59)
            )
            records.append({
                "id": f"amma-{idx:03d}",
                "ts": ts.isoformat(),
                "app": "whatsapp",
                "recipient": "Amma",
                "raw_asr": raw,
                "formatted": formatted,
                "meta": {
                    "experiment": "amma_address_form",
                    "register": match.register,
                    "expected_form": match.expected_form,
                },
            })

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


def split(records: list[dict[str, Any]], parts: int = PARTS) -> list[list[dict[str, Any]]]:
    """Chronological thirds. Order matters: this simulates time passing."""
    size = len(records) // parts
    out = []
    for i in range(parts):
        lo = i * size
        hi = (i + 1) * size if i < parts - 1 else len(records)
        out.append(records[lo:hi])
    return out


def verify(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the planted rule actually made it into the text.

    Generation is not obedient, and an experiment run on a corpus that does not
    contain the rule would measure nothing. This is the gate before learning.
    """
    stats = {"total": len(records), "correct": 0, "wrong": 0, "neither": 0,
             "mama": 0, "amma": 0}
    for record in records:
        text = record["formatted"].lower()
        has_mama, has_amma = "mama" in text, "amma" in text
        expected = record["meta"]["expected_form"].lower()
        stats["mama"] += int(has_mama)
        stats["amma"] += int(has_amma and not has_mama)
        if not has_mama and not has_amma:
            stats["neither"] += 1
        elif (expected == "mama" and has_mama) or (
            expected == "amma" and has_amma and not has_mama
        ):
            stats["correct"] += 1
        else:
            stats["wrong"] += 1
    stats["fidelity"] = (
        stats["correct"] / stats["total"] if stats["total"] else 0.0
    )
    return stats
