"""Deterministic corpus planning.

Every record's *meaning* is decided here, in seeded Python, before any model is
called. That is what makes the evaluation scoreable: we know which episodes
carry a planted fact, which must produce no memory at all, and which questions
have no answer anywhere in the corpus.

Planted case types, and what each one tests:

  distributed_fact    an answer split across 3+ episodes, in none of them whole
  knowledge_update    a fact that changes mid-corpus; the old value must retire
  explicit_preference stated outright once; should be learned from one episode
  inferred_preference behaviour repeated 3+ times; low confidence at best
  commitment          a promise with a date; expires rather than being deleted
  third_party         about someone else - must produce NO memory (stance test)
  noise               trivial; searchable forever, never remembered
  filler              ordinary traffic, so planted cases are not the whole corpus

A question set is planned alongside, including false-premise questions whose
correct answer is an abstention.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Any

from kivi.corpus.profiles import DOMAIN_APPS, ContextProfile

# Proportions of the corpus given to each planted type. The remainder is filler.
MIX: dict[str, float] = {
    "distributed_fact": 0.06,
    "knowledge_update": 0.04,
    "explicit_preference": 0.02,
    "inferred_preference": 0.06,
    "commitment": 0.06,
    "third_party": 0.10,
    "noise": 0.10,
}

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


@dataclass
class RecordSpec:
    """One dictation to be written. `instruction` is what it must convey."""

    idx: int
    external_id: str
    kind: str
    domain: str
    app: str
    style: str
    ts: dt.datetime
    instruction: str
    plant_id: str | None = None
    must_not_learn: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plant:
    """A planted fact, and the episodes that carry it."""

    plant_id: str
    kind: str
    summary: str
    expected_memory: str | None
    episodes: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    qid: str
    question: str
    kind: str            # one of the planted kinds, or 'false_premise'
    expected: str | None  # None means the correct behaviour is abstention
    plant_id: str | None = None
    supporting: list[str] = field(default_factory=list)


@dataclass
class CorpusPlan:
    profile_key: str
    seed: int
    specs: list[RecordSpec]
    plants: list[Plant]
    questions: list[Question]


def _choose_app(rng: random.Random, profile: ContextProfile, domain: str) -> str:
    if profile.domain != "mixed":
        return rng.choice(profile.apps)
    allowed = [a for a in DOMAIN_APPS[domain] if a in profile.apps]
    return rng.choice(allowed or list(profile.apps))


def _timestamps(rng: random.Random, profile: ContextProfile, count: int) -> list[dt.datetime]:
    """Spread records over ~10 weeks, clustered into plausible hours."""
    start = dt.datetime(2026, 6, 22, tzinfo=IST)
    stamps: list[dt.datetime] = []
    for _ in range(count):
        day = start + dt.timedelta(days=rng.randint(0, 69))
        is_weekend = day.weekday() >= 5
        # Re-roll weekend days down to the profile's weekend ratio.
        while is_weekend and rng.random() > profile.weekend_ratio:
            day = start + dt.timedelta(days=rng.randint(0, 69))
            is_weekend = day.weekday() >= 5
        stamps.append(
            day.replace(
                hour=rng.choice(profile.active_hours),
                minute=rng.randint(0, 59),
                second=rng.randint(0, 59),
            )
        )
    stamps.sort()
    return stamps


def build_plan(profile: ContextProfile, count: int, seed: int = 7) -> CorpusPlan:
    rng = random.Random(seed)

    kinds: list[str] = []
    for kind, share in MIX.items():
        kinds += [kind] * int(count * share)
    kinds += ["filler"] * (count - len(kinds))
    rng.shuffle(kinds)

    office = [p for p in profile.people if p.name in {
        "Umar", "Abhi", "Priya", "Karthik", "Meera", "Sanjay", "Nikhil"}]
    house = [p for p in profile.people if p not in office]
    office_projects = [p for p in profile.projects if p.kind in {"project", "customer"}]
    house_projects = [p for p in profile.projects if p.kind in {"plan", "chore", "place"}]

    def domain_for(kind: str) -> str:
        if profile.domain != "mixed":
            return profile.domain
        return rng.choice(["office", "house"])

    specs: list[RecordSpec] = []
    plants: dict[str, Plant] = {}
    questions: list[Question] = []

    # One entry per episode to be written. A single planted case can contribute
    # several - a distributed fact needs three - so this list ends up longer than
    # `count` and is trimmed back to size below.
    pending: list[dict[str, Any]] = []

    def add(kind: str, plant_id: str | None, instruction: str,
            must_not_learn: bool, domain: str) -> None:
        pending.append({
            "kind": kind, "plant_id": plant_id, "instruction": instruction,
            "must_not_learn": must_not_learn, "domain": domain,
        })

    def cast(domain: str):
        people = office if domain == "office" else house
        projects = office_projects if domain == "office" else house_projects
        if not people:
            people = list(profile.people)
        if not projects:
            projects = list(profile.projects)
        return people, projects

    counters = {kind: 0 for kind in MIX}

    for kind in kinds:
        if kind == "filler":
            add("filler", None, "", False, domain_for(kind))
            continue

        domain = domain_for(kind)
        people, projects = cast(domain)
        n = counters[kind]
        counters[kind] += 1

        if kind == "distributed_fact":
            # One fact, three episodes, no single one complete.
            project = projects[n % len(projects)]
            members = rng.sample(people, min(3, len(people)))
            pid = f"dist_{n}"
            plants[pid] = Plant(
                pid, kind,
                f"Who is working on {project.name}",
                f"{', '.join(p.name for p in members)} are on {project.name}",
                detail={"project": project.name, "members": [p.name for p in members]},
            )
            for member in members:
                add(kind, pid,
                    f"mention only that {member.name} ({member.role}) is working on "
                    f"{project.name} ({project.blurb}). Do not mention anyone else "
                    f"working on it.",
                    False, domain)
            questions.append(Question(
                f"q_{pid}", f"Who is working on {project.name}?", kind,
                ", ".join(p.name for p in members), pid,
            ))
            continue

        if kind == "knowledge_update":
            person = rng.choice(people)
            old, new = rng.sample(projects, 2) if len(projects) > 1 else (projects[0], projects[0])
            pid = f"upd_{n}"
            plants[pid] = Plant(
                pid, kind,
                f"{person.name} moved from {old.name} to {new.name}",
                f"{person.name} is on {new.name} (was {old.name})",
                detail={"person": person.name, "from": old.name, "to": new.name},
            )
            add(kind, pid, f"state that {person.name} is working on {old.name}.", False, domain)
            add(kind, pid, f"state that {person.name} is working on {old.name} again, in passing.", False, domain)
            add(kind, pid, f"state that {person.name} has now MOVED from {old.name} to {new.name}.", False, domain)
            questions.append(Question(
                f"q_{pid}", f"What is {person.name} working on now?", kind, new.name, pid,
            ))
            continue

        if kind == "explicit_preference":
            pref = rng.choice(_EXPLICIT_PREFS[domain])
            pid = f"pref_{n}"
            plants[pid] = Plant(pid, kind, pref["summary"], pref["memory"], detail=pref)
            add(kind, pid, f"state this preference outright: {pref['say']}", False, domain)
            questions.append(Question(f"q_{pid}", pref["question"], kind, pref["memory"], pid))
            continue

        if kind == "inferred_preference":
            beh = _INFERRED_PREFS[domain][n % len(_INFERRED_PREFS[domain])]
            pid = f"inf_{n}"
            plants[pid] = Plant(pid, kind, beh["summary"], beh["memory"], detail=beh)
            for _ in range(3):
                add(kind, pid, beh["instruction"], False, domain)
            continue

        if kind == "commitment":
            person = rng.choice(people)
            thing = rng.choice(_COMMITMENT_THINGS[domain])
            pid = f"cmt_{n}"
            plants[pid] = Plant(
                pid, kind, f"promised {thing} to {person.name}",
                f"send/do {thing} for {person.name}",
                detail={"person": person.name, "thing": thing},
            )
            add(kind, pid,
                f"say in the FIRST PERSON that you told {person.name} you would {thing}, "
                f"and give a specific day of the week as the deadline.",
                False, domain)
            questions.append(Question(
                f"q_{pid}", f"What did I promise {person.name}?", kind, thing, pid,
            ))
            continue

        if kind == "third_party":
            person = rng.choice(people)
            sensitive = rng.choice(_THIRD_PARTY[domain])
            pid = f"tp_{n}"
            plants[pid] = Plant(
                pid, kind,
                f"sensitive detail about {person.name}: {sensitive}",
                None,  # nothing should be learned
                detail={"person": person.name, "detail": sensitive},
            )
            add(kind, pid,
                f"You are transcribing or relaying information ABOUT someone else, not "
                f"about yourself. Report that {person.name} {sensitive}. Write it as "
                f"notes or a message you are drafting about them.",
                True, domain)
            continue

        if kind == "noise":
            add(kind, None, rng.choice(_NOISE[domain]), True, domain)
            continue

    # Planted cases needing several episodes overshoot `count`; fillers absorb the
    # difference in both directions so the corpus is exactly the size requested.
    # Only fillers are ever dropped - trimming a planted case would leave a plant
    # whose evidence is incomplete, and silently corrupt the ground truth.
    while len(pending) > count:
        fillers = [i for i, item in enumerate(pending) if item["kind"] == "filler"]
        if not fillers:
            break  # planted cases alone exceed count; keep them all
        pending.pop(fillers[-1])
    while len(pending) < count:
        add("filler", None, "", False, domain_for("filler"))

    # Shuffle so the parts of a multi-episode plant end up spread across the corpus
    # rather than adjacent - an answer distributed over three consecutive records
    # would not test distributed retrieval at all.
    rng.shuffle(pending)
    stamps = _timestamps(rng, profile, len(pending))

    for position, item in enumerate(pending):
        kind = item["kind"]
        domain = item["domain"]
        instruction = item["instruction"]

        if kind == "filler":
            people, projects = cast(domain)
            topic = rng.choice(profile.topics)
            who = rng.choice(people)
            what = rng.choice(projects)
            instruction = (
                f"{topic}, involving {who.name} ({who.role}) and/or "
                f"{what.name} ({what.blurb}). Ordinary day-to-day content."
            )

        spec = RecordSpec(
            idx=position,
            external_id=f"{profile.key}-{position:04d}",
            kind=kind,
            domain=domain,
            app=_choose_app(rng, profile, domain),
            style=rng.choice(profile.styles),
            ts=stamps[position],
            instruction=instruction,
            plant_id=item["plant_id"],
            must_not_learn=item["must_not_learn"],
        )
        specs.append(spec)
        if spec.plant_id and spec.plant_id in plants:
            plants[spec.plant_id].episodes.append(spec.external_id)

    for question in questions:
        if question.plant_id and question.plant_id in plants:
            question.supporting = list(plants[question.plant_id].episodes)

    questions.extend(_false_premise_questions(profile, rng))

    return CorpusPlan(profile.key, seed, specs, list(plants.values()), questions)


def _false_premise_questions(profile: ContextProfile, rng: random.Random) -> list[Question]:
    """Questions whose correct answer is 'I don't have that'.

    Deliberately plausible: they name real entities from the corpus but ask about
    something never said. A system that pattern-matches on the entity and
    confabulates the rest will fail exactly here.
    """
    templates = {
        "office": [
            "What did {person} say about their salary?",
            "Which university did {person} go to?",
            "What was the budget approved for {project}?",
            "What did I decide about the {project} pricing model?",
        ],
        "house": [
            "What is {person}'s blood group?",
            "How much did we finally pay for {project}?",
            "What did {person} say about moving house?",
            "Which airline did we book for {project}?",
        ],
    }
    people = list(profile.people)
    projects = list(profile.projects)
    out: list[Question] = []
    domains = ["office", "house"] if profile.domain == "mixed" else [profile.domain]
    for domain in domains:
        for i, template in enumerate(templates[domain]):
            text = template.format(
                person=rng.choice(people).name, project=rng.choice(projects).name
            )
            out.append(Question(f"q_fp_{domain}_{i}", text, "false_premise", None))
    return out


_HOUSE_HINTS = (
    "Amma", "Deepa", "Ishaan", "Nair", "Lakshmi", "Kerala", "kitchen",
    "school", "car service", "grocery", "uncle",
)

_EXPLICIT_PREFS = {
    "office": [
        {
            "say": "keep my Slack messages short, I hate long paragraphs there",
            "summary": "prefers short Slack messages",
            "memory": "keep Slack messages short",
            "question": "How do I like my Slack messages written?",
        },
        {
            "say": "always put the ask in the first line of an email",
            "summary": "ask goes in the first line of emails",
            "memory": "put the ask in the first line of an email",
            "question": "How should my emails be structured?",
        },
        {
            "say": "never use exclamation marks in customer emails, it reads as unserious",
            "summary": "no exclamation marks in customer email",
            "memory": "no exclamation marks in customer emails",
            "question": "What punctuation do I avoid with customers?",
        },
    ],
    "house": [
        {
            "say": "always add the shop name when you write a grocery list for me",
            "summary": "grocery lists include the shop name",
            "memory": "include the shop name on grocery lists",
            "question": "How do I want grocery lists written?",
        },
        {
            "say": "put family messages in plain simple language, no jargon",
            "summary": "plain language for family messages",
            "memory": "plain simple language for family messages",
            "question": "How should messages to family be written?",
        },
    ],
}

_INFERRED_PREFS = {
    "office": [
        {
            "instruction": "an email that starts immediately with the point - no 'Hi team', "
                           "no greeting of any kind, no sign-off.",
            "summary": "tends to omit greetings in email",
            "memory": "omits greetings in email (inferred)",
        },
        {
            "instruction": "a status update written as three short dashed bullet points, "
                           "never as a paragraph.",
            "summary": "tends to write status updates as bullets",
            "memory": "writes status updates as bullets (inferred)",
        },
    ],
    "house": [
        {
            "instruction": "a message that ends abruptly with no sign-off and no emoji.",
            "summary": "tends to end family messages abruptly",
            "memory": "no sign-offs in family messages (inferred)",
        },
    ],
}

_COMMITMENT_THINGS = {
    "office": [
        "send the deck", "share the scoping doc", "review the schema PR",
        "send the revised timeline", "write up the call notes",
    ],
    "house": [
        "book the tickets", "pick up the medicines", "call the plumber back",
        "pay the school fees", "collect the car",
    ],
}

_THIRD_PARTY = {
    "office": [
        "has been struggling with deadlines and mentioned personal health issues",
        "is quietly interviewing at another company",
        "was passed over for promotion and is unhappy about it",
        "disclosed that their team had a security breach last year",
        "is going through a difficult divorce",
    ],
    "house": [
        "has been advised to start blood pressure medication",
        "is worried about their diagnosis and has not told anyone",
        "has been having money trouble and asked to borrow",
        "failed two subjects and does not want it discussed",
    ],
}

_NOISE = {
    "office": [
        "a four-word note with no context, like a room number or a wifi password",
        "a single word confirming something, nothing else",
        "a string of numbers being read out, like a ticket ID",
        "a two-word reminder with no subject",
    ],
    "house": [
        "a bare grocery list of four items, no sentence around it",
        "a short list of two errands, no context",
        "a single word reply, nothing else",
        "a phone number being read out loud",
    ],
}
