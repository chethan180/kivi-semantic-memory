"""Turn a corpus plan into dictation records.

The model's only job is to write plausible speech. It is told what the dictation
must convey; it never decides what is true. Records are produced in batches and
cached, so regenerating the same corpus costs nothing and yields identical text.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from kivi.corpus.plan import CorpusPlan, RecordSpec, build_plan
from kivi.corpus.profiles import PROFILES, ContextProfile
from kivi.llm.gemini import GeminiClient, LLMError

log = logging.getLogger(__name__)

BATCH_SIZE = 10

SYSTEM = """You write realistic voice-dictation training data.

For each item you produce TWO versions of the same utterance:

1. `raw_asr` - what a speech recogniser would output. All lowercase. No
   punctuation at all. Include natural disfluencies (um, uh, so, like, i mean,
   false starts). Proper nouns and acronyms are frequently MANGLED the way a
   recogniser mangles them - spell them phonetically when the item tells you how.
   Numbers and dates are often written as words.

2. `formatted` - what a good dictation product would write instead: correct
   punctuation and capitalisation, disfluencies removed, proper nouns and
   acronyms spelled correctly, but the SAME content and the same voice. Do not
   add information that was not spoken.

Rules:
- Write as the persona speaking in the first person, unless the item explicitly
  says the content is about someone else.
- Vary length. Some utterances are four words. Some run to several sentences.
- Never invent a different fact from the one the item asks for.
- NEVER open with "Dear ...", "Hi team", "Hello", or any salutation, and never
  close with a sign-off, unless the item explicitly asks for one. This is
  dictation into an app, not a letter. Start with the content.
- Return one object per item, with the item's `idx`."""

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


def _asr_hints(profile: ContextProfile) -> str:
    pairs = [f'"{p.name}" -> "{p.asr_error}"' for p in profile.people]
    pairs += [f'"{p.name}" -> "{p.asr_error}"' for p in profile.projects]
    return "; ".join(pairs)


def _batch_prompt(profile: ContextProfile, specs: list[RecordSpec]) -> str:
    lines = [
        f"PERSONA: {profile.persona}",
        "",
        f"ASR MANGLINGS to use in raw_asr (use them most of the time, not every time): "
        f"{_asr_hints(profile)}",
        "",
        f"Write {len(specs)} dictations. For each, produce raw_asr and formatted.",
        "",
    ]
    for spec in specs:
        lines.append(
            f"- idx {spec.idx} | app: {spec.app} | style: {spec.style} | "
            f"{spec.ts.strftime('%A %d %B %Y, %H:%M')}"
        )
        lines.append(f"  content: {spec.instruction}")
    return "\n".join(lines)


def _realise_batch(
    client: GeminiClient, profile: ContextProfile, specs: list[RecordSpec]
) -> dict[int, dict[str, str]]:
    result = client.generate(
        _batch_prompt(profile, specs),
        system=SYSTEM,
        schema=SCHEMA,
        temperature=0.9,
        max_output_tokens=8192,
    )
    payload = result.json()
    out: dict[int, dict[str, str]] = {}
    for item in payload.get("records", []):
        try:
            idx = int(item["idx"])
        except (KeyError, TypeError, ValueError):
            continue
        raw = (item.get("raw_asr") or "").strip()
        formatted = (item.get("formatted") or "").strip()
        if raw and formatted:
            out[idx] = {"raw_asr": raw, "formatted": formatted}
    return out


def realise(
    plan: CorpusPlan,
    profile: ContextProfile,
    client: GeminiClient,
    *,
    workers: int = 4,
    progress=None,
) -> list[dict[str, Any]]:
    """Generate text for every spec. Returns records in the plan's order."""
    batches = [
        plan.specs[i : i + BATCH_SIZE] for i in range(0, len(plan.specs), BATCH_SIZE)
    ]
    texts: dict[int, dict[str, str]] = {}
    failures = 0

    def run(batch: list[RecordSpec]) -> dict[int, dict[str, str]]:
        try:
            return _realise_batch(client, profile, batch)
        except LLMError as exc:
            log.warning("batch failed (%d specs): %s", len(batch), exc)
            return {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for partial in pool.map(run, batches):
            texts.update(partial)
            if progress:
                progress(len(texts))

    records: list[dict[str, Any]] = []
    for spec in plan.specs:
        text = texts.get(spec.idx)
        if text is None:
            failures += 1
            continue
        records.append(
            {
                "id": spec.external_id,
                "ts": spec.ts.isoformat(),
                "app": spec.app,
                "style_id": spec.style,
                "duration_ms": max(1200, len(text["raw_asr"]) * 55),
                "raw_asr": text["raw_asr"],
                "formatted": text["formatted"],
                "meta": {
                    "corpus": plan.profile_key,
                    "domain": spec.domain,
                    "planted_kind": spec.kind,
                    "plant_id": spec.plant_id,
                    "must_not_learn": spec.must_not_learn,
                    "language": "en-IN",
                },
            }
        )

    if failures:
        log.warning("%d specs produced no text and were dropped", failures)
    return records


def write_corpus(
    records: list[dict[str, Any]], plan: CorpusPlan, out_dir: Path
) -> tuple[Path, Path]:
    """Write the JSONL corpus and its ground truth side by side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / f"{plan.profile_key}_episodes.jsonl"
    truth_path = out_dir / f"{plan.profile_key}_ground_truth.json"

    with open(corpus_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    present = {record["id"] for record in records}
    truth = {
        "profile": plan.profile_key,
        "seed": plan.seed,
        "record_count": len(records),
        "plants": [
            {**asdict(plant), "episodes": [e for e in plant.episodes if e in present]}
            for plant in plan.plants
        ],
        "questions": [
            {**asdict(question),
             "supporting": [e for e in question.supporting if e in present]}
            for question in plan.questions
        ],
        "must_not_learn_episodes": sorted(
            record["id"] for record in records if record["meta"]["must_not_learn"]
        ),
    }
    truth_path.write_text(
        json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return corpus_path, truth_path


def generate(
    profile_key: str,
    count: int,
    client: GeminiClient,
    *,
    seed: int = 7,
    out_dir: Path | None = None,
    workers: int = 4,
    progress=None,
) -> tuple[Path, Path, list[dict[str, Any]], CorpusPlan]:
    profile = PROFILES[profile_key]
    plan = build_plan(profile, count, seed=seed)
    records = realise(plan, profile, client, workers=workers, progress=progress)
    out_dir = out_dir or Path("data/corpus")
    corpus_path, truth_path = write_corpus(records, plan, out_dir)
    return corpus_path, truth_path, records, plan
