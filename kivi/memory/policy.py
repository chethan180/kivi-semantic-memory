"""Memory policy - the only file the product position should need to change.

Everything here is a judgement about what Kivi ought to remember. Everything
elsewhere in `kivi/memory/` is mechanism that works whatever these values are.
The split is deliberate: the position determines the policy, and the policy is
one file, so a change of position is an edit here rather than a rewrite.

The values below are PROVISIONAL. They are defensible on their own terms, but
they have not yet been reconciled with a written product position, and the README
must say so rather than present them as settled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# What kinds of thing may become a memory at all
# ---------------------------------------------------------------------------

MEMORY_TYPES = ("entity", "preference", "commitment")

ENTITY_TYPES = ("person", "project", "org", "place", "thing", "term")

RELATIONS = ("works_on", "works_with", "part_of", "related_to", "mentioned_with")

# ---------------------------------------------------------------------------
# Gate 1: salience. Cheap, deterministic, runs before any token is spent.
# ---------------------------------------------------------------------------

# Below this, an utterance cannot carry durable meaning worth a model call.
MIN_WORDS = 5

# Identical formatted text seen within this window is a repeat, not new evidence.
DUPLICATE_WINDOW = 50

# ---------------------------------------------------------------------------
# Gate 3: promotion. The LLM proposes; these rules dispose.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionRule:
    """When a candidate of this type is allowed to become a memory."""

    # Episodes that must independently support it before promotion.
    evidence_required: int
    # Lower bar when the person states it outright rather than it being inferred.
    evidence_required_if_explicit: int
    # Confidence granted on promotion.
    confidence: float
    # Ceiling for candidates that were inferred from behaviour rather than said.
    inferred_confidence_cap: float
    # Extra conditions, enforced in promote.py.
    requires_first_person: bool = False
    requires_time_anchor: bool = False


PROMOTION: dict[str, PromotionRule] = {
    # Entities need two distinct episodes ALWAYS - there is deliberately no
    # single-episode shortcut, even for a name the speaker defines outright.
    #
    # This is what filters out speech-recognition manglings, and it is not a
    # theoretical concern: a first run promoted "Abbey" as a person alongside
    # "Abhi", because the recogniser mis-heard the name once and the extractor
    # dutifully reported it. A real colleague recurs; a mishearing does not.
    # Requiring corroboration costs one episode of latency and removes an entire
    # class of confident nonsense.
    "entity": PromotionRule(
        evidence_required=2,
        evidence_required_if_explicit=2,
        confidence=0.75,
        inferred_confidence_cap=0.6,
    ),
    # A stated preference is a direct instruction and deserves to be obeyed at
    # once. A preference merely *observed* in behaviour is a guess about a person,
    # so it needs repetition and can never become confident.
    "preference": PromotionRule(
        evidence_required=3,
        evidence_required_if_explicit=1,
        confidence=0.9,
        inferred_confidence_cap=0.5,
    ),
    # A commitment is actionable, so a single clear statement is enough - but only
    # if the speaker committed themselves, and only if there is a date to expire.
    "commitment": PromotionRule(
        evidence_required=1,
        evidence_required_if_explicit=1,
        confidence=0.85,
        inferred_confidence_cap=0.5,
        requires_first_person=True,
        requires_time_anchor=True,
    ),
}

# ---------------------------------------------------------------------------
# Entity resolution
#
# Speech names the same thing many ways: "Beacon", "the Beacon project", "Beacon
# integration". Left alone the store fills with near-duplicate entities that
# split the evidence for one real thing across several rows, so none of them
# ever reaches the promotion threshold and the graph gets four nodes where it
# needs one.
#
# The rule is deliberately shallow - strip role words from the edges of a name
# and compare what is left. It will merge two genuinely different things that
# differ only by a suffix. That is the acceptable direction of error here: an
# over-merged entity is visible and correctable, while a fragmented one silently
# starves every threshold in the system.
# ---------------------------------------------------------------------------

ENTITY_NOISE_WORDS = frozenset("""
project workstream work stream initiative programme program integration
migration rollout launch console dashboard service platform system tool
team squad group effort track piece feature epic ticket sprint
the a an our my
""".split())


def normalise_entity(subject: str) -> str:
    """Reduce an entity name to its identity for matching purposes."""
    words = [
        w for w in "".join(
            c if c.isalnum() or c.isspace() else " " for c in subject.lower()
        ).split()
    ]
    while words and words[0] in ENTITY_NOISE_WORDS:
        words.pop(0)
    while words and words[-1] in ENTITY_NOISE_WORDS:
        words.pop()
    return " ".join(words) if words else subject.strip().lower()


# ---------------------------------------------------------------------------
# The stance guard
#
# With context modelling deliberately not built, this is the entire mechanism
# preventing Kivi from learning about third parties. It is a single LLM-extracted
# field, which makes it the system's main privacy assumption and its most
# load-bearing failure point. It has its own evaluation slice for that reason.
# ---------------------------------------------------------------------------

STANCES = ("asserted", "transcribed", "reported", "unclear")

# Only these stances may ever produce a memory.
PROMOTABLE_STANCES = frozenset({"asserted"})

# ---------------------------------------------------------------------------
# The ignore list
#
# Categories Kivi refuses to remember about anyone, including the user. Enforced
# twice: described to the extractor in Gate 2, and checked again in Gate 3,
# because a prompt instruction is a request and a code path is a guarantee.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IgnoreRule:
    key: str
    why: str
    terms: tuple[str, ...] = field(default=())


IGNORE_LIST: tuple[IgnoreRule, ...] = (
    IgnoreRule(
        "health",
        "medical information is not ours to hold, about the user or anyone else",
        ("diagnosis", "diagnosed", "medication", "prescription", "symptom",
         "illness", "disease", "surgery", "therapy", "blood pressure",
         "depression", "anxiety", "cancer", "pregnan", "clinic", "hospital"),
    ),
    IgnoreRule(
        "personal_relationships",
        "the state of someone's marriage or family is not product signal",
        ("divorce", "affair", "breakup", "separated from", "custody"),
    ),
    IgnoreRule(
        "money_personal",
        "salary and personal finances invite inference far beyond dictation",
        ("salary", "compensation", "in debt", "loan", "borrow money", "bonus"),
    ),
    IgnoreRule(
        "employment_status",
        "someone quietly interviewing elsewhere must not become a stored fact",
        ("interviewing at", "quitting", "resigning", "fired", "laid off",
         "passed over", "performance review", "on a pip"),
    ),
    IgnoreRule(
        "traits_and_mood",
        "character judgements are inferences about a person, not facts about work",
        ("lazy", "difficult to work with", "struggling", "unhappy", "burnt out",
         "not a team player", "incompetent"),
    ),
    IgnoreRule(
        "third_party_confidential",
        "a customer's breach is their information, disclosed to us in confidence",
        ("breach", "under investigation", "regulatory review", "lawsuit",
         "litigation", "nda"),
    ),
)

IGNORE_KEYS = tuple(rule.key for rule in IGNORE_LIST)


def ignore_match(text: str) -> IgnoreRule | None:
    """Deterministic backstop for the ignore list.

    Keyword matching is crude and will over-reject. That is the intended
    direction of error: a memory not formed is a missed convenience, a memory
    wrongly formed about someone's health is a breach of the product's promise.
    """
    lowered = text.lower()
    for rule in IGNORE_LIST:
        if any(term in lowered for term in rule.terms):
            return rule
    return None


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

# Commitments become `expired` once their date passes. They are never deleted -
# the promise was really made, and the episode that records it stays true.
EXPIRE_COMMITMENTS = True

# A memory never retrieved in this many days is down-weighted in ranking. It is
# not removed: absence of use is not evidence of falsehood.
STALE_AFTER_DAYS = 120

# A preference is APPLIED to drafts only at or above this confidence. A stated
# preference is promoted at 0.9 and an inferred one is capped at 0.5, so this is
# the line between "you told Kivi" and "Kivi noticed" - and the interface's
# "Stop doing this" drops a preference below it. One number, read by the drafting
# tools and the interface alike, so the switch the person sees is the switch the
# drafting obeys. It used to be read by the interface only: drafting applied
# every active preference, and "Stop doing this" changed nothing but a label.
APPLY_PREFERENCE_AT = 0.85


def describe() -> dict[str, object]:
    """Machine-readable policy, for the eval report and the inspection UI."""
    return {
        "memory_types": list(MEMORY_TYPES),
        "entity_types": list(ENTITY_TYPES),
        "relations": list(RELATIONS),
        "min_words": MIN_WORDS,
        "promotable_stances": sorted(PROMOTABLE_STANCES),
        "promotion": {
            name: {
                "evidence_required": rule.evidence_required,
                "evidence_required_if_explicit": rule.evidence_required_if_explicit,
                "confidence": rule.confidence,
                "inferred_confidence_cap": rule.inferred_confidence_cap,
                "requires_first_person": rule.requires_first_person,
                "requires_time_anchor": rule.requires_time_anchor,
            }
            for name, rule in PROMOTION.items()
        },
        "ignore_list": [
            {"key": rule.key, "why": rule.why} for rule in IGNORE_LIST
        ],
    }
