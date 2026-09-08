"""Context profiles: office, house, and the mixed corpus.

Three corpora rather than one, because we deliberately chose not to model
contexts in the store. The single-domain corpora are controls; `mixed` is the
realistic case and the one that will show cross-context bleed if it exists -
a work question surfacing a family entity, or the reverse. Having all three
makes that measurable instead of a matter of opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Person:
    name: str
    role: str
    asr_error: str  # how a recogniser plausibly mangles the name


@dataclass(frozen=True)
class Project:
    name: str
    kind: str
    blurb: str
    asr_error: str


@dataclass(frozen=True)
class ContextProfile:
    key: str
    label: str
    persona: str
    domain: str                       # office | house | mixed
    apps: tuple[str, ...]
    styles: tuple[str, ...]
    people: tuple[Person, ...] = ()
    projects: tuple[Project, ...] = ()
    topics: tuple[str, ...] = ()
    # Hours of day (local) that dictations plausibly happen in this context.
    active_hours: tuple[int, ...] = tuple(range(9, 20))
    weekend_ratio: float = 0.1


OFFICE_PEOPLE = (
    Person("Umar", "backend engineer, owns the connector work", "oomar"),
    Person("Abhi", "engineer, schema and data modelling", "abbey"),
    Person("Priya", "product manager, enterprise accounts", "preeya"),
    Person("Karthik", "engineer, on the reporting surface", "kartik"),
    Person("Meera", "designer", "meera"),
    Person("Sanjay", "engineering manager", "sanjay"),
    Person("Nikhil", "solutions architect", "nikil"),
)

OFFICE_PROJECTS = (
    Project("DSPM", "project", "Data Security Posture Management, the flagship workstream", "dee ess pee em"),
    Project("Atlas", "project", "the internal name for the migration workstream", "at less"),
    Project("Beacon", "project", "the alerting and notification service", "beacon"),
    Project("Acme", "customer", "a large enterprise customer under regulatory review", "acme"),
    Project("Northwind", "customer", "a mid-market customer evaluating the product", "north wind"),
)

HOUSE_PEOPLE = (
    Person("Amma", "the user's mother", "amma"),
    Person("Deepa", "the user's partner", "deepa"),
    Person("Ishaan", "the user's son, age 9", "ishan"),
    Person("Dr. Nair", "the family doctor", "doctor nair"),
    Person("Ravi uncle", "a relative", "ravi uncle"),
    Person("Lakshmi", "the neighbour", "lakshmi"),
)

HOUSE_PROJECTS = (
    Project("the Kerala trip", "plan", "a family holiday being planned for December", "kerala trip"),
    Project("the kitchen repair", "chore", "an ongoing plumbing and cabinet repair", "kitchen repair"),
    Project("Ishaan's school", "place", "school admin, PTMs, fees, and events", "ishans school"),
    Project("the car service", "chore", "the car's overdue service and insurance renewal", "car service"),
)

OFFICE_TOPICS = (
    "a sprint status update",
    "a decision made in a design review",
    "notes taken during a customer call",
    "a follow-up email to a colleague",
    "a bug being triaged",
    "a scoping or estimation discussion",
    "feedback on a document someone shared",
    "a hiring or interview debrief",
    "a note about a deadline moving",
    "a summary written after a steering meeting",
    "a question posted to the team channel",
    "a short reply agreeing to something",
)

HOUSE_TOPICS = (
    "a grocery or shopping list",
    "a reminder about an appointment",
    "coordinating school pickup or drop-off",
    "arranging a repair or a service visit",
    "a message to a family member about plans",
    "a note about a bill or renewal",
    "planning part of a holiday",
    "a quick message saying you are running late",
    "a note about medicines or a check-up",
    "weekend plans with family",
)

_OFFICE = ContextProfile(
    key="office",
    label="Office",
    domain="office",
    persona=(
        "Ravi Menon, a product lead at a data-security company in Bangalore. "
        "Dictates constantly between meetings: status updates, customer call notes, "
        "emails to the team, and quick Slack replies."
    ),
    apps=("slack", "gmail", "jira", "notion", "notes"),
    styles=("casual", "formal", "bullet"),
    people=OFFICE_PEOPLE,
    projects=OFFICE_PROJECTS,
    topics=OFFICE_TOPICS,
    active_hours=(9, 10, 11, 12, 14, 15, 16, 17, 18, 19),
    weekend_ratio=0.05,
)

_HOUSE = ContextProfile(
    key="house",
    label="House",
    domain="house",
    persona=(
        "Ravi Menon at home in Bangalore. Dictates to his family WhatsApp threads, "
        "to a notes app, and to reminders: errands, school logistics, appointments, "
        "repairs, and holiday planning."
    ),
    apps=("whatsapp", "notes", "reminders", "messages"),
    styles=("casual", "bullet"),
    people=HOUSE_PEOPLE,
    projects=HOUSE_PROJECTS,
    topics=HOUSE_TOPICS,
    active_hours=(7, 8, 9, 13, 18, 19, 20, 21, 22),
    weekend_ratio=0.35,
)

_MIXED = ContextProfile(
    key="mixed",
    label="Mixed",
    domain="mixed",
    persona=(
        "Ravi Menon across a whole day: product lead at a data-security company in "
        "Bangalore, and a parent running a household. Work and home dictations land "
        "in the same stream, sometimes in the same sentence."
    ),
    apps=("slack", "gmail", "jira", "notion", "notes", "whatsapp", "reminders", "messages"),
    styles=("casual", "formal", "bullet"),
    people=OFFICE_PEOPLE + HOUSE_PEOPLE,
    projects=OFFICE_PROJECTS + HOUSE_PROJECTS,
    topics=OFFICE_TOPICS + HOUSE_TOPICS,
    active_hours=(7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21),
    weekend_ratio=0.2,
)

PROFILES: dict[str, ContextProfile] = {
    "office": _OFFICE,
    "house": _HOUSE,
    "mixed": _MIXED,
}

# Apps that belong to each domain, used by the mixed profile to keep a work
# dictation out of WhatsApp and a school reminder out of Jira.
DOMAIN_APPS: dict[str, tuple[str, ...]] = {
    "office": ("slack", "gmail", "jira", "notion", "notes"),
    "house": ("whatsapp", "notes", "reminders", "messages"),
}
