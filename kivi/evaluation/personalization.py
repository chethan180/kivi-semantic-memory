"""Does knowing how you write to someone actually change what Kivi writes?

Scored by counting, not by asking a model whether the output feels personalized.
Every check below is a function over the produced text:

    greeting   does it open with the exact opener this person uses?
    length     is it inside the band they write in?
    signoff    present or absent, as they do it?

Run twice per recipient - once with the learned profile applied, once without -
so the number reported is a difference rather than an absolute that could be
explained by the model simply writing well.

Adherence is reported against how many dictations each profile was built from,
because that curve is the honest form of the claim: the capability is real where
there is evidence for it and unreliable where there is not, and a single
headline figure would hide exactly that.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kivi.corpus.recipients import RECIPIENTS, Recipient
from kivi.llm.gemini import GeminiClient
from kivi.memory import style

# A neutral note, deliberately in nobody's voice, so every recipient's draft is
# produced from the same starting point and the only variable is the profile.
SOURCE = (
    "the deployment is delayed by two days, the fix is being tested tonight, "
    "and I will confirm the new date tomorrow morning"
)

TASKS = (
    "tell them about the delay",
    "let them know what is happening and when you will confirm",
    "pass on the delay and the plan",
)


@dataclass
class DraftScore:
    recipient: str
    n_samples: int
    with_style: bool
    text: str
    greeting_ok: bool
    length_ok: bool
    signoff_ok: bool
    words: int

    @property
    def score(self) -> float:
        return (self.greeting_ok + self.length_ok + self.signoff_ok) / 3.0


@dataclass
class PersonalizationEval:
    scores: list[DraftScore] = field(default_factory=list)

    def by_recipient(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name in {s.recipient for s in self.scores}:
            on = [s for s in self.scores if s.recipient == name and s.with_style]
            off = [s for s in self.scores if s.recipient == name and not s.with_style]
            if not on or not off:
                continue
            out[name] = {
                "n_samples": on[0].n_samples,
                "with_style": round(statistics.fmean(s.score for s in on), 3),
                "without_style": round(statistics.fmean(s.score for s in off), 3),
                "greeting_on": round(statistics.fmean(s.greeting_ok for s in on), 3),
                "greeting_off": round(statistics.fmean(s.greeting_ok for s in off), 3),
                "length_on": round(statistics.fmean(s.length_ok for s in on), 3),
                "length_off": round(statistics.fmean(s.length_ok for s in off), 3),
                "mean_words_on": round(statistics.fmean(s.words for s in on), 1),
                "mean_words_off": round(statistics.fmean(s.words for s in off), 1),
            }
        return out

    def overall(self) -> dict[str, float]:
        on = [s.score for s in self.scores if s.with_style]
        off = [s.score for s in self.scores if not s.with_style]
        if not on or not off:
            return {}
        return {
            "with_style": round(statistics.fmean(on), 3),
            "without_style": round(statistics.fmean(off), 3),
            "delta": round(statistics.fmean(on) - statistics.fmean(off), 3),
        }


def score_draft(text: str, recipient: Recipient) -> tuple[bool, bool, bool, int]:
    """Check one draft against the style this recipient is actually written in."""
    stripped = (text or "").strip()
    words = len(stripped.split())
    f = style.features(stripped)

    if recipient.greeting:
        # The opener has to be the one they use, not merely some greeting.
        greeting_ok = bool(f.greeting) and recipient.greeting.split()[0].lower() \
            in f.greeting.lower()
        if greeting_ok and len(recipient.greeting.split()) > 1:
            greeting_ok = recipient.name.lower() in stripped[:40].lower()
    else:
        greeting_ok = f.greeting is None

    lo, hi = recipient.words
    # A band with slack: the target is a habit, not a character budget.
    length_ok = (lo * 0.6) <= words <= (hi * 1.6)

    signoff_ok = bool(f.signoff) if recipient.signoff else (f.signoff is None)
    return greeting_ok, length_ok, signoff_ok, words


def _draft(
    client: GeminiClient,
    recipient: Recipient,
    task: str,
    rules: list[str],
) -> str:
    prompt = [
        f"Rewrite this note as a message to {recipient.name}: {task}.",
        "",
        f"Note: {SOURCE}",
    ]
    if rules:
        prompt += [
            "",
            f"How this person writes to {recipient.name} "
            f"(learned from their own messages - follow exactly):",
            *(f"- {r}" for r in rules),
        ]
    prompt += ["", "Return only the message."]
    result = client.generate(
        "\n".join(prompt),
        system="You rewrite a person's notes as messages they would send. "
               "When told how they write to someone, you match it exactly.",
        temperature=0.3,
        max_output_tokens=1024,
    )
    return result.text.strip()


def run_personalization_eval(
    conn: sqlite3.Connection,
    client: GeminiClient,
    *,
    drafts_per_recipient: int = 3,
    progress=None,
) -> PersonalizationEval:
    evaluation = PersonalizationEval()
    total = len(RECIPIENTS) * drafts_per_recipient * 2
    done = 0

    for recipient in RECIPIENTS:
        profile = style.load_profile(conn, recipient.name)
        rules = style.instructions(profile) if profile else []
        n_samples = profile.n_samples if profile else 0

        for i in range(drafts_per_recipient):
            task = TASKS[i % len(TASKS)]
            for with_style in (False, True):
                text = _draft(client, recipient, task, rules if with_style else [])
                greeting_ok, length_ok, signoff_ok, words = score_draft(text, recipient)
                evaluation.scores.append(
                    DraftScore(
                        recipient=recipient.name, n_samples=n_samples,
                        with_style=with_style, text=text,
                        greeting_ok=greeting_ok, length_ok=length_ok,
                        signoff_ok=signoff_ok, words=words,
                    )
                )
                done += 1
                if progress:
                    progress(done, total)

    return evaluation


def write_results(evaluation: PersonalizationEval, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "personalization.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for s in evaluation.scores:
            handle.write(json.dumps({
                "recipient": s.recipient, "n_samples": s.n_samples,
                "with_style": s.with_style, "greeting_ok": s.greeting_ok,
                "length_ok": s.length_ok, "signoff_ok": s.signoff_ok,
                "words": s.words, "score": round(s.score, 3), "text": s.text,
            }, ensure_ascii=False) + "\n")
    return path
