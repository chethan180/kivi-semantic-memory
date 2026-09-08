"""Seven people the persona writes to, each with a consistent, distinct style.

The styles here are the ground truth for the personalization evaluation. They are
specified as concrete, countable habits - a literal greeting, a length band, a
formality register - because the whole point of the exercise is that adherence
can be scored by a function rather than judged by a model.

Two per relation, so the evaluation can show that Kivi learns *the person* and
not merely *the category*: writing to your manager is not the same as writing to
your manager's manager, and two friends are not interchangeable. If profiles
collapsed to one style per relation, that would be a finding worth reporting too.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Recipient:
    name: str
    relation: str          # boss | friend | family
    who: str               # for the generator prompt
    apps: tuple[str, ...]
    greeting: str | None   # the literal opener, or None for no greeting
    signoff: str | None
    words: tuple[int, int]  # target length band
    register: str          # instruction to the generator
    topics: tuple[str, ...]
    # How many dictations to generate. Deliberately uneven: the evaluation
    # reports adherence against sample count, and that curve needs points
    # spread along it rather than clustered at one volume.
    volume: int


RECIPIENTS: tuple[Recipient, ...] = (
    # ---- bosses -----------------------------------------------------------
    Recipient(
        name="Sanjay",
        relation="boss",
        who="your direct manager, an engineering manager you respect and speak to daily",
        apps=("slack", "gmail"),
        greeting=None,
        signoff=None,
        words=(18, 34),
        register=(
            "Punctual and complete. No greeting, no sign-off, no slang. Lead with "
            "status or the ask, give a number or a date, close with what you need "
            "from him. Full sentences, no contractions."
        ),
        topics=(
            "a status update on DSPM with a concrete date",
            "flagging a slip and what you propose to do about it",
            "asking for a decision on scope",
            "confirming a priority before you commit the team",
            "a short summary after a steering meeting",
            "asking him to unblock something",
        ),
        volume=60,
    ),
    Recipient(
        name="Vikram",
        relation="boss",
        who="your manager's manager, a director you write to rarely and carefully",
        apps=("gmail",),
        greeting="Hi Vikram",
        signoff="thanks",
        words=(45, 80),
        register=(
            "Formal and considered, noticeably more careful than how you write to "
            "your own manager. Always greet him by name. Give context before the "
            "ask because he does not have the day-to-day. Complete sentences, "
            "no contractions, no slang, sign off with thanks."
        ),
        topics=(
            "a quarterly summary of where DSPM stands",
            "escalating a dependency that needs his help",
            "responding to a question he asked in a review",
            "flagging a risk early with context and a recommendation",
            "asking for headcount or budget with justification",
        ),
        volume=22,
    ),
    # ---- friends ----------------------------------------------------------
    Recipient(
        name="Rahul",
        relation="friend",
        who="a close school friend you message constantly in shorthand",
        apps=("whatsapp",),
        greeting=None,
        signoff=None,
        words=(4, 12),
        register=(
            "Very short and clipped. No greeting. Heavy contractions, lowercase "
            "feel, casual filler like 'yeah', 'bro', 'lol'. Often a fragment "
            "rather than a sentence. Never formal."
        ),
        topics=(
            "cricket match plans",
            "rescheduling a meetup",
            "reacting to something funny",
            "asking if he is free this weekend",
            "complaining lightly about work with no detail",
            "sending a quick yes or no",
        ),
        volume=55,
    ),
    Recipient(
        name="Nikita",
        relation="friend",
        who="a friend from your first job who you catch up with properly, less often",
        apps=("whatsapp", "messages"),
        greeting="Hey Nikita",
        signoff=None,
        words=(22, 45),
        register=(
            "Warm and chatty, and clearly longer than how you write to Rahul. "
            "Greet her by name. Ask a question back most of the time. Use "
            "contractions and the occasional exclamation mark. Friendly but in "
            "full sentences."
        ),
        topics=(
            "catching up after a few weeks",
            "asking how a new job is going",
            "planning dinner with a few people",
            "sharing news about work without jargon",
            "checking in after she mentioned something stressful",
        ),
        volume=30,
    ),
    # ---- family -----------------------------------------------------------
    Recipient(
        name="Amma",
        relation="family",
        who="your mother, who you message most days",
        apps=("whatsapp",),
        greeting="Hi Amma",
        signoff=None,
        words=(12, 25),
        register=(
            "Always open with 'Hi Amma'. Warm, simple, plain language with no "
            "work jargon whatsoever. Short sentences. Often about food, health "
            "check-ins, travel or visiting. Frequently ends with a question or "
            "a reassurance. Exclamation marks are in character."
        ),
        topics=(
            "telling her when you will reach home",
            "asking whether she took her tablets",
            "about a temple visit or a festival",
            "asking what she cooked",
            "reassuring her that work is fine",
            "arranging a call at the weekend",
        ),
        volume=48,
    ),
    Recipient(
        name="Appa",
        relation="family",
        who="your father, who you message less often and more practically",
        apps=("whatsapp", "messages"),
        greeting=None,
        signoff=None,
        words=(8, 18),
        register=(
            "Practical and brief, noticeably less warm than how you write to your "
            "mother, and with no greeting. Mostly logistics: money, the car, "
            "documents, travel timings. Statements rather than questions. No "
            "exclamation marks."
        ),
        topics=(
            "the car service or insurance renewal",
            "transferring money or a bill",
            "train or flight timings",
            "a document that needs signing",
            "confirming a plan in one line",
        ),
        volume=26,
    ),
    Recipient(
        name="Arjun",
        relation="family",
        who="your younger brother, who you talk to like a friend but about family things",
        apps=("whatsapp",),
        greeting=None,
        signoff=None,
        words=(6, 16),
        register=(
            "Casual and blunt, like a friend, but the subject is usually family. "
            "Heavy contractions, teasing tone, fragments. No greeting, no "
            "sign-off. Sometimes a single word."
        ),
        topics=(
            "coordinating something for your parents",
            "splitting a cost for a family thing",
            "teasing him about something",
            "asking if he has called home",
            "plans for a family trip",
        ),
        volume=34,
    ),
)


BY_NAME = {r.name.lower(): r for r in RECIPIENTS}


def expected_profile(recipient: Recipient) -> dict[str, object]:
    """The ground truth a learned profile is scored against."""
    lo, hi = recipient.words
    return {
        "name": recipient.name,
        "relation": recipient.relation,
        "greeting": recipient.greeting,
        "expects_greeting": recipient.greeting is not None,
        "signoff": recipient.signoff,
        "expects_signoff": recipient.signoff is not None,
        "words_min": lo,
        "words_max": hi,
        "volume": recipient.volume,
    }
