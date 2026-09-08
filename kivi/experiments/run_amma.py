"""Run the three-pass experiment and report what changed at each stage.

The test is generative, not descriptive. A profile that *says* "uses Mama when
affectionate" has still proved nothing until a draft written from it actually
says "Mama" for "I love you" and "Amma" for the gas cylinder. So each stage ends
by writing messages and checking which form came out.

Two probes matter most:

    "I love you"                    -> expect Mama
    "tell her I'll be home by 8"    -> expect Amma

The second is the control. A system that simply learned "say Mama" scores full
marks on the first probe and fails the second, and only running both separates
learning the rule from memorising the commoner token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from kivi.experiments import amma, learn_prompt
from kivi.llm.gemini import GeminiClient, LLMError

# (instruction, expected address form, why this probe exists)
#
# Split into two groups on purpose. The `seen` probes use situations that appear
# in the corpus topics, so passing them shows the rule was applied. The `unseen`
# probes use situations that appear NOWHERE in the generation topics, so passing
# THOSE is the only evidence that the rule generalised rather than that
# topic->form pairs were memorised.
#
# The first version of this file used only seen probes and scored 6/6 at every
# stage, which looked like a result and was mostly an artefact.
SEEN_PROBES: tuple[tuple[str, str, str], ...] = (
    ("write I love you", "Mama", "affection - the headline case"),
    ("tell her I will be home by 8", "Amma", "control - routine"),
    ("ask if she took her tablets", "Amma", "control - practical"),
)

UNSEEN_PROBES: tuple[tuple[str, str, str], ...] = (
    ("tell her I got the promotion at work today", "Mama",
     "unseen: celebration - not in any topic list"),
    ("ask her for the wifi password at home", "Amma",
     "unseen: trivial logistics"),
    ("tell her I dreamt about her last night", "Mama",
     "unseen: tender, no overlap with training topics"),
    ("tell her the flight on Tuesday got cancelled", "Amma",
     "unseen: factual travel news"),
    ("tell her I am really proud of how she handled everything", "Mama",
     "unseen: pride, phrased unlike anything in the corpus"),
    ("ask her to keep the parcel with the neighbour", "Amma",
     "unseen: errand"),
)

PROBES = SEEN_PROBES + UNSEEN_PROBES

DRAFT_SYSTEM = """You write a short WhatsApp message on behalf of someone, to their
mother, in exactly the way that person writes to her.

You are given what has been learned about how they write to her. Follow it
literally, including any rule about which name to use in which situation. Decide
which situation this message is before you choose.

Write only the message. One or two sentences. No quotes, no preamble."""


@dataclass
class StageResult:
    stage: int
    messages_seen: int
    profile: dict[str, Any]
    drafts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def correct(self) -> int:
        return sum(1 for d in self.drafts if d["ok"])

    @property
    def score(self) -> float:
        return self.correct / len(self.drafts) if self.drafts else 0.0

    def _subset(self, unseen: bool) -> list[dict[str, Any]]:
        return [d for d in self.drafts if d["unseen"] is unseen]

    @property
    def seen_score(self) -> float:
        subset = self._subset(False)
        return sum(d["ok"] for d in subset) / len(subset) if subset else 0.0

    @property
    def unseen_score(self) -> float:
        """The number that actually matters."""
        subset = self._subset(True)
        return sum(d["ok"] for d in subset) / len(subset) if subset else 0.0

    def forms(self) -> list[str]:
        out = []
        for entry in self.profile.get("address_forms", []):
            out.append(f"{entry.get('form')} ({entry.get('when')})")
        return out


def _draft(client: GeminiClient, profile: dict[str, Any], instruction: str) -> str:
    prompt = (
        "HOW THIS PERSON WRITES TO THEIR MOTHER (learned from their own "
        f"messages):\n{json.dumps(profile, indent=2, ensure_ascii=False)}\n\n"
        f"Now write this message: {instruction}"
    )
    try:
        return client.generate(
            prompt, system=DRAFT_SYSTEM, temperature=0.4, max_output_tokens=256
        ).text.strip()
    except LLMError as exc:
        return f"<error: {exc}>"


def _check(text: str, expected: str) -> tuple[bool, str]:
    lowered = text.lower()
    has_mama = "mama" in lowered
    # "amma" is a substring of nothing here, but "mama" is not a substring of
    # "amma" either - they are distinct tokens, so plain containment is safe.
    has_amma = "amma" in lowered
    if has_mama and not has_amma:
        got = "Mama"
    elif has_amma and not has_mama:
        got = "Amma"
    elif has_mama and has_amma:
        got = "both"
    else:
        got = "neither"
    return got == expected, got


def run(
    client: GeminiClient,
    records: list[dict[str, Any]],
    *,
    parts: int = amma.PARTS,
    progress=None,
) -> list[StageResult]:
    chunks = amma.split(records, parts)
    results: list[StageResult] = []
    profile: dict[str, Any] | None = None
    seen = 0

    for stage, chunk in enumerate(chunks, start=1):
        texts = [r["formatted"] for r in chunk]

        if profile is None:
            prompt = learn_prompt.first_pass_prompt("Amma (their mother)", texts)
        else:
            prompt = learn_prompt.revision_prompt(
                "Amma (their mother)", profile, texts, seen
            )

        try:
            profile = client.generate(
                prompt,
                system=learn_prompt.SYSTEM,
                schema=learn_prompt.SCHEMA,
                model=client.settings.gen_model_heavy,
                temperature=0.1,
                max_output_tokens=4096,
            ).json()
        except LLMError as exc:
            profile = {"error": str(exc), "address_forms": [],
                       "conditional_rules": [], "constants": [], "summary": ""}

        seen += len(chunk)
        result = StageResult(stage=stage, messages_seen=seen, profile=profile)

        for instruction, expected, why in PROBES:
            text = _draft(client, profile, instruction)
            ok, got = _check(text, expected)
            result.drafts.append({
                "instruction": instruction, "expected": expected, "got": got,
                "ok": ok, "why": why, "text": text,
                "unseen": (instruction, expected, why) in UNSEEN_PROBES,
            })

        results.append(result)
        if progress:
            progress(stage, parts)

    return results
