"""Three people, three different KINDS of conditional habit, one mixed log.

The Amma experiment showed a lexical rule can be learned: which word is used to
address someone, conditional on feeling. That is the easiest kind to spot, since
the signal is a token that is literally present or absent.

These two are deliberately harder and deliberately different, because a prompt
tuned until it finds one kind of pattern proves only that it finds that kind:

    Amma    LEXICAL     which name, conditional on emotional register
    Sanjay  STRUCTURAL  whether a proposal is attached, conditional on bad news
    Priya   PRAGMATIC   whether the ask is hedged, conditional on urgency

Sanjay's rule leaves no distinctive vocabulary to latch onto - it is about what
the message must CONTAIN. Priya's is about softening, which is a register move
rather than a content one. If all three are recovered, the learner is reading
behaviour rather than matching a template.

Everything is written into ONE log, interleaved in time, exactly as a real
dictation stream would be. Separating the recipients is the learner's problem.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Situation:
    """One kind of message, and which side of the conditional it falls on."""

    topic: str
    branch: str          # which arm of the rule
    instruction: str     # given to the generator, never to the learner


@dataclass(frozen=True)
class PersonSpec:
    name: str
    relation: str
    who: str
    app: str
    kind: str            # lexical | structural | pragmatic
    rule_summary: str    # for the report only - never shown to the learner
    situations: tuple[Situation, ...]
    seen_probes: tuple[tuple[str, str], ...]    # (instruction, expected branch)
    unseen_probes: tuple[tuple[str, str], ...]
    volume: int
    # Planted ONCE, midway through the log. A single out-of-character message
    # that must NOT become the default: one bad day is not a habit, and a system
    # that rewrites someone's voice from one angry sentence is unusable.
    outlier: Situation | None = None


# ---------------------------------------------------------------------------
# Amma - lexical: which name, conditional on feeling
# ---------------------------------------------------------------------------

AMMA = PersonSpec(
    name="Amma",
    relation="family",
    who="their mother",
    app="whatsapp",
    kind="lexical",
    rule_summary='"Mama" when affectionate, "Amma" when routine',
    situations=(
        Situation("telling her you miss her", "Mama",
                  'address her as "Mama"'),
        Situation("saying you love her before a trip", "Mama",
                  'address her as "Mama"'),
        Situation("thanking her for food she sent", "Mama",
                  'address her as "Mama"'),
        Situation("comforting her after she sounded worried", "Mama",
                  'address her as "Mama"'),
        Situation("apologising for not calling", "Mama",
                  'address her as "Mama"'),
        Situation("wishing her before a hospital check-up", "Mama",
                  'address her as "Mama"'),
        Situation("telling her what time you reach home", "Amma",
                  'address her as "Amma"'),
        Situation("asking whether she took her tablets", "Amma",
                  'address her as "Amma"'),
        Situation("asking what she cooked", "Amma",
                  'address her as "Amma"'),
        Situation("telling her the electrician is coming", "Amma",
                  'address her as "Amma"'),
        Situation("confirming a train timing", "Amma",
                  'address her as "Amma"'),
        Situation("asking her to switch off the motor", "Amma",
                  'address her as "Amma"'),
    ),
    seen_probes=(
        ("write I love you", "Mama"),
        ("tell her I will be home by 8", "Amma"),
    ),
    unseen_probes=(
        ("tell her I dreamt about her last night", "Mama"),
        ("tell her the flight on Tuesday got cancelled", "Amma"),
        ("tell her I am proud of how she handled everything", "Mama"),
        ("ask her to keep the parcel with the neighbour", "Amma"),
        # Stability probes: asked AFTER a single angry message has been seen.
        # If one bad day rewrote the profile, these are where it shows.
        ("write I love you", "Mama"),
        ("ask her whether the milk was delivered", "Amma"),
    ),
    volume=45,
    outlier=Situation(
        "losing your temper because she called during a meeting",
        "outlier",
        "This one message is ANGRY and out of character - short, sharp, "
        "frustrated, no affection at all. Address her as 'Amma'. This is a bad "
        "day, not how they normally write to her.",
    ),
)

# ---------------------------------------------------------------------------
# Sanjay - structural: bad news always arrives with a proposal attached
#
# No vocabulary marks this. The rule is about what the message must contain,
# which is why it is a harder thing to notice than a name changing.
# ---------------------------------------------------------------------------

SANJAY = PersonSpec(
    name="Sanjay",
    relation="boss",
    who="their engineering manager",
    app="slack",
    kind="structural",
    rule_summary="bad news always carries a proposed course of action; good news is one bare line",
    situations=(
        Situation("DSPM slipping by two weeks", "problem",
                  "This is BAD NEWS. State the problem and the impact, then ALWAYS "
                  "propose a specific course of action in the same message "
                  "(phrased like 'I propose we...' or 'My plan is to...'). Never "
                  "report a problem without a proposal."),
        Situation("a schema review blocking two engineers", "problem",
                  "This is BAD NEWS. State it plainly, then propose what to do."),
        Situation("a customer escalation on Acme", "problem",
                  "This is BAD NEWS. State the impact, then propose a course of action."),
        Situation("losing an engineer to another team", "problem",
                  "This is BAD NEWS. State it, then propose how to cover the gap."),
        Situation("a security patch delaying the release", "problem",
                  "This is BAD NEWS. State it, then propose the way forward."),
        Situation("the connector work finishing early", "progress",
                  "This is GOOD NEWS or routine progress. ONE short line, purely "
                  "factual. Do NOT propose anything, do not add next steps."),
        Situation("staging deploy went clean", "progress",
                  "GOOD NEWS. One short factual line only. No proposal."),
        Situation("Northwind signed off", "progress",
                  "GOOD NEWS. One short factual line only. No proposal."),
        Situation("the sprint closing on time", "progress",
                  "GOOD NEWS. One short factual line only. No proposal."),
    ),
    seen_probes=(
        ("tell him the release is delayed by a week", "problem"),
        ("tell him the migration finished", "progress"),
    ),
    unseen_probes=(
        ("tell him our main database is close to running out of storage", "problem"),
        ("tell him the new dashboard shipped to production", "progress"),
        ("tell him two people are off sick and the sprint is at risk", "problem"),
        ("tell him the API latency improved after the cache change", "progress"),
    ),
    volume=42,
)

# ---------------------------------------------------------------------------
# Priya - pragmatic: hedging appears only when nothing is urgent
# ---------------------------------------------------------------------------

PRIYA = PersonSpec(
    name="Priya",
    relation="colleague",
    who="a close colleague they work with daily",
    app="slack",
    kind="pragmatic",
    rule_summary="asks are hedged when not urgent, and completely unhedged when they are",
    situations=(
        Situation("asking her to review a pull request", "relaxed",
                  "NOT urgent. Soften the ask with a hedge such as 'when you get "
                  "a chance', 'no rush', or 'whenever you have a minute'."),
        Situation("asking for her thoughts on a design", "relaxed",
                  "NOT urgent. Soften the ask with a hedge like 'no rush' or "
                  "'whenever suits you'."),
        Situation("asking her to look at some test data", "relaxed",
                  "NOT urgent. Include a softening hedge."),
        Situation("asking her to update a document", "relaxed",
                  "NOT urgent. Include a softening hedge."),
        Situation("a production incident needing her now", "urgent",
                  "URGENT. No hedging at all, no softeners, no 'when you get a "
                  "chance'. Direct and immediate."),
        Situation("a customer demo starting in ten minutes", "urgent",
                  "URGENT. Completely direct, no softening language."),
        Situation("a broken build blocking everyone", "urgent",
                  "URGENT. Completely direct, no softening."),
        Situation("a data issue about to reach a customer", "urgent",
                  "URGENT. Completely direct, no softening."),
    ),
    seen_probes=(
        ("ask her to review my pull request", "relaxed"),
        ("tell her the payments service is down and I need her now", "urgent"),
    ),
    unseen_probes=(
        ("ask her to take a look at the onboarding copy sometime", "relaxed"),
        ("tell her customer data is leaking into the logs right now", "urgent"),
        ("ask her opinion on renaming the settings page", "relaxed"),
        ("tell her the release train leaves in fifteen minutes and the tests are red",
         "urgent"),
    ),
    volume=40,
)

PEOPLE: tuple[PersonSpec, ...] = (AMMA, SANJAY, PRIYA)
BY_NAME = {p.name.lower(): p for p in PEOPLE}


# ---------------------------------------------------------------------------
# Detectors. Deterministic, so the model never scores its own output.
# ---------------------------------------------------------------------------

# A proposal is a call to action, however it is phrased. The first version of
# this list missed "Let's ..." and "I say we ...", which is how most of the
# corpus actually proposes things, and scored the data at 76% when it was very
# nearly perfect - the detector was wrong, not the data. It also included
# "I'll" and "I will", which are plain future tense and appear just as often in
# a progress update, so they were removed.
_PROPOSAL_MARKERS = (
    "i propose", "my plan is", "plan is to", "i suggest", "i'd suggest",
    "i would suggest", "proposing", "we should", "let's ", "lets ",
    "i say we", "shall we", "i'd like to", "i would like to",
    "next step is", "aim for", "suggest we", "recommend",
)
_HEDGE_MARKERS = (
    "when you get a chance", "no rush", "whenever you have", "whenever suits",
    "sometime", "when you can", "no hurry", "if you get a minute",
    "at some point", "whenever you get",
    # Conditional framing is the commonest way this corpus softens an ask, and
    # leaving it out scored genuinely hedged messages as urgent.
    "if you get a chance", "if you happen to", "if you have time",
    "free time", "do you mind", "whenever you're free", "whenever you are free",
    "when you have a moment", "if you can spare",
)


def classify(person: PersonSpec, text: str) -> str:
    """Which branch of this person's rule does the text fall on?"""
    lowered = (text or "").lower()

    if person.kind == "lexical":
        has_mama, has_amma = "mama" in lowered, "amma" in lowered
        if has_mama and not has_amma:
            return "Mama"
        if has_amma and not has_mama:
            return "Amma"
        return "both" if has_mama else "neither"

    if person.kind == "structural":
        return "problem" if any(m in lowered for m in _PROPOSAL_MARKERS) else "progress"

    return "relaxed" if any(m in lowered for m in _HEDGE_MARKERS) else "urgent"
