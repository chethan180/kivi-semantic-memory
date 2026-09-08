"""How this person writes to each person they write to.

Deliberately deterministic. Every feature here is counted, not judged, which
means a style profile can be explained to the person exactly ("you opened with
'Hi Amma' in 12 of 14 messages") and checked by a test rather than by asking a
model whether the output feels right.

The threshold is the whole design. One message to someone is not a habit; it is
an occurrence. A style is only applied once the same behaviour has been seen at
least twice, and confidence keeps rising with evidence after that - which is
also why the evaluation reports adherence against sample count rather than a
single headline number.
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

# A habit needs corroboration. Below this a profile is recorded but not applied.
MIN_SAMPLES = 2

# Above this the profile is treated as reliable; the eval reports the curve
# between the two so the claim is bounded by evidence rather than asserted.
CONFIDENT_SAMPLES = 8

_GREETING_RE = re.compile(
    r"^\s*((?:hi|hey|hello|dear|yo|hiya|good morning|good afternoon|good evening|"
    r"morning|namaste|hii+)\b[^.!?\n,]{0,24})",
    re.IGNORECASE,
)
_SIGNOFF_RE = re.compile(
    r"(thanks|thank you|cheers|regards|best|talk soon|love you|love|see you|ttyl|"
    r"take care)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_CONTRACTION_RE = re.compile(r"\b\w+'(s|t|re|ll|ve|d|m)\b", re.IGNORECASE)

# Words that mark register. Crude on purpose: a formality score that needs a
# model to compute could not be explained to the person or asserted in a test.
_FORMAL_MARKERS = frozenset("""
please kindly regarding accordingly furthermore however therefore requested
confirm confirming advise advised shall would could appreciate awaiting
attached herewith pursuant ensure

""".split())
_CASUAL_MARKERS = frozenset("""
yeah yep nope gonna wanna kinda sorta lol haha btw ok okay cool sure
tbh ping quick heads-up ya na re da bro dude
""".split())


@dataclass
class StyleFeatures:
    """What one dictation shows about how its author writes."""

    words: int = 0
    sentences: int = 0
    greeting: str | None = None
    signoff: str | None = None
    contractions: int = 0
    exclamations: int = 0
    questions: int = 0
    formal_hits: int = 0
    casual_hits: int = 0


def features(text: str) -> StyleFeatures:
    """Measure one message. No model, no judgement."""
    stripped = (text or "").strip()
    if not stripped:
        return StyleFeatures()

    words = stripped.split()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()]
    lowered = {w.strip(".,!?;:").lower() for w in words}

    greeting_match = _GREETING_RE.match(stripped)
    signoff_match = _SIGNOFF_RE.search(stripped)

    return StyleFeatures(
        words=len(words),
        sentences=max(len(sentences), 1),
        greeting=greeting_match.group(1).strip() if greeting_match else None,
        signoff=signoff_match.group(1).strip().lower() if signoff_match else None,
        contractions=len(_CONTRACTION_RE.findall(stripped)),
        exclamations=stripped.count("!"),
        questions=stripped.count("?"),
        formal_hits=len(lowered & _FORMAL_MARKERS),
        casual_hits=len(lowered & _CASUAL_MARKERS),
    )


@dataclass
class StyleProfile:
    recipient_norm: str
    display: str
    relation: str = "unknown"
    n_samples: int = 0
    greeting: str | None = None
    greeting_rate: float = 0.0
    signoff: str | None = None
    signoff_rate: float = 0.0
    mean_words: float = 0.0
    mean_sentence: float = 0.0
    contraction_rate: float = 0.0
    exclamation_rate: float = 0.0
    question_rate: float = 0.0
    formality: float = 0.0
    episode_ids: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.n_samples >= MIN_SAMPLES

    @property
    def strength(self) -> float:
        """0..1. How much evidence stands behind this profile."""
        if self.n_samples <= 0:
            return 0.0
        return min(1.0, self.n_samples / CONFIDENT_SAMPLES)

    def describe(self) -> str:
        """The profile in the person's own terms, for the interface."""
        bits: list[str] = []
        if self.greeting and self.greeting_rate >= 0.5:
            bits.append(f'usually opens with "{self.greeting}"')
        elif self.greeting_rate < 0.2:
            bits.append("no greeting")
        if self.signoff and self.signoff_rate >= 0.4:
            bits.append(f'signs off "{self.signoff}"')
        bits.append(f"about {self.mean_words:.0f} words")
        if self.formality >= 0.6:
            bits.append("formal")
        elif self.formality <= 0.35:
            bits.append("casual")
        if self.exclamation_rate >= 0.4:
            bits.append("warm, uses exclamations")
        return ", ".join(bits)


def build_profile(
    recipient_norm: str, display: str, rows: Iterable[sqlite3.Row]
) -> StyleProfile:
    """Aggregate one recipient's dictations into a profile."""
    profile = StyleProfile(recipient_norm=recipient_norm, display=display)
    greetings: Counter[str] = Counter()
    signoffs: Counter[str] = Counter()
    total_words = total_sentences = 0
    contraction_msgs = exclam_msgs = question_msgs = 0
    formal = casual = 0

    for row in rows:
        f = features(row["formatted"])
        if not f.words:
            continue
        profile.n_samples += 1
        profile.episode_ids.append(row["id"])
        total_words += f.words
        total_sentences += f.sentences
        if f.greeting:
            greetings[f.greeting.lower()] += 1
        if f.signoff:
            signoffs[f.signoff] += 1
        contraction_msgs += 1 if f.contractions else 0
        exclam_msgs += 1 if f.exclamations else 0
        question_msgs += 1 if f.questions else 0
        formal += f.formal_hits
        casual += f.casual_hits

    n = profile.n_samples
    if not n:
        return profile

    if greetings:
        # Store the spelling actually used most, not a lowercased key: "Hi Amma"
        # is the thing to reproduce, and casing is part of the habit.
        top, count = greetings.most_common(1)[0]
        profile.greeting = top
        profile.greeting_rate = count / n
    profile.signoff_rate = (sum(signoffs.values()) / n) if signoffs else 0.0
    if signoffs:
        profile.signoff = signoffs.most_common(1)[0][0]

    profile.mean_words = total_words / n
    profile.mean_sentence = total_words / max(total_sentences, 1)
    profile.contraction_rate = contraction_msgs / n
    profile.exclamation_rate = exclam_msgs / n
    profile.question_rate = question_msgs / n

    # Formality from things that are actually counted, not from word lists.
    #
    # The first version scored it as formal_words / (formal + casual), which put
    # a mother at 0.77 because "please" and "confirm" appear in ordinary family
    # messages, and a father at 1.00 on a handful of words. Marker vocabularies
    # are too sparse to divide by. Contractions, sentence length and slang are
    # present in every message and move together with register.
    slang_rate = casual / max(total_words, 1) * 100
    profile.formality = _clamp(
        0.5
        + 0.25 * (1.0 - min(profile.contraction_rate * 2, 1.0))   # contractions -> casual
        + 0.20 * min(profile.mean_sentence / 22.0, 1.0)            # long sentences -> formal
        - 0.35 * min(slang_rate / 4.0, 1.0)                        # slang -> casual
        - 0.15 * min(profile.exclamation_rate, 1.0)                # exclamations -> casual
    )
    return profile


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Relation inference
#
# Learned from how the person writes, not configured. The signals are weak
# individually and only combined into a label when they agree; "unknown" is a
# perfectly good answer and better than a confident guess about who someone is.
# ---------------------------------------------------------------------------

_RELATION_HINTS: dict[str, tuple[str, ...]] = {
    "family": ("amma", "appa", "mom", "mum", "mother", "father", "dad", "bro",
               "brother", "sister", "akka", "anna", "chithi", "mama", "uncle",
               "aunty", "grandma", "grandpa", "home", "dinner", "temple"),
    "boss": ("approve", "approval", "sign off", "sign-off", "headcount", "review",
             "escalate", "escalation", "one on one", "1:1", "priorit", "roadmap",
             "budget", "stakeholder", "steering"),
    "friend": ("beer", "match", "movie", "weekend", "trip", "game", "party",
               "dinner plan", "catch up", "gym", "cricket", "football"),
    "colleague": ("standup", "stand-up", "pr", "ticket", "deploy", "sprint",
                  "bug", "merge", "branch", "review the", "schema", "endpoint"),
}


# Kinship words. A name that IS one of these settles the question outright,
# which no amount of topic-word counting reliably does.
_KINSHIP = frozenset("""
amma appa ma pa mom mum mother father dad daddy papa
anna akka thambi thangai bro brother sis sister
chithi mama uncle aunty aunt grandma grandpa paati thatha
""".split())

# Vocabulary that only appears in working life.
_WORK_WORDS = frozenset("""
sprint standup stand-up deploy deployment release ticket pr merge branch
schema endpoint api roadmap headcount stakeholder escalate escalation
review approval sign-off signoff milestone backlog scope staging incident
quarter dspm atlas beacon northwind acme
""".split())


def infer_relation(rows: Iterable[sqlite3.Row], profile: StyleProfile) -> tuple[str, float]:
    """Guess how this person is related, and how sure that guess is.

    Decided in stages rather than by summing weak hints. Counting topic words
    across every relation at once got 3 of 7 right: a friend who mentions dinner
    scored as family, and a manager scored as a peer because both talk about
    work. Each stage below answers one question with the strongest evidence
    available for it, and 'unknown' is preferred to a confident bad guess.

    This is the least reliable thing in the module and is reported as such. Note
    that no drafting behaviour depends on it - the style profile drives that -
    so a wrong label mislabels a row in the interface rather than changing how
    anything is written.
    """
    rows = list(rows)
    if not rows:
        return "unknown", 0.0

    name = profile.recipient_norm.split()[0] if profile.recipient_norm else ""
    text = " ".join((r["formatted"] or "").lower() for r in rows)
    words = set(text.replace(",", " ").replace(".", " ").split())
    apps = Counter((r["app"] or "").lower() for r in rows)

    # 1. Kinship in the name is decisive. You do not call a colleague "Amma".
    if name in _KINSHIP:
        return "family", 0.95

    # 2. Work vocabulary separates working life from personal life cleanly.
    work_hits = len(words & _WORK_WORDS)
    work_apps = sum(apps[a] for a in ("slack", "jira", "notion", "gmail"))
    is_work = work_hits >= 3 or work_apps > len(rows) / 2

    if is_work:
        # 3. Boss versus peer is deference, not topic. Writing at length, with a
        #    greeting and a sign-off, to someone you rarely message, is how
        #    people write upward.
        deference = (
            (profile.greeting_rate >= 0.5)
            + (profile.signoff_rate >= 0.4)
            + (profile.mean_words >= 40)
            + (profile.formality >= 0.6)
        )
        if deference >= 3:
            return "boss", min(0.6 + 0.1 * deference, 0.95)
        return "colleague", 0.7

    # 4. Personal, but no kinship word: a friend. Family members who are not
    #    addressed by a kinship name are the known failure mode here.
    kinship_mentions = len(words & _KINSHIP)
    if kinship_mentions >= 2:
        return "family", 0.55
    return "friend", 0.6


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalise_recipient(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def rebuild_profiles(conn: sqlite3.Connection) -> list[StyleProfile]:
    """Recompute every recipient profile from the episodes.

    Idempotent and cheap, because it is a view over evidence. Anything that
    cannot be rederived from dictations does not belong in the table.
    """
    # Reading `episodes` is now sufficient to exclude Kivi's own output: a
    # composed message goes to `composed_messages` and never lands here. That
    # used to be an `AND source != 'composed'` repeated in four queries, which
    # is a guard only until someone writes a fifth query without it - and the
    # cost of forgetting is Kivi learning this person's voice from its own
    # predictions of it, drifting toward its idea of them.
    recipients = conn.execute(
        "SELECT recipient_norm, MAX(recipient) AS display, COUNT(*) AS n"
        " FROM episodes WHERE recipient_norm IS NOT NULL AND recipient_norm != ''"
        " GROUP BY recipient_norm ORDER BY n DESC"
    ).fetchall()

    profiles: list[StyleProfile] = []
    now = _utcnow()
    for entry in recipients:
        rows = conn.execute(
            "SELECT id, formatted, app FROM episodes WHERE recipient_norm = ?"
            " ORDER BY ts_epoch",
            (entry["recipient_norm"],),
        ).fetchall()
        profile = build_profile(entry["recipient_norm"], entry["display"], rows)
        profile.relation, relation_conf = infer_relation(rows, profile)

        # ON CONFLICT DO UPDATE, not INSERT OR REPLACE.
        #
        # REPLACE deletes the whole row and writes a new one, which silently
        # discarded every column this function does not set - the model's summary,
        # its rules, and the checkpoint marker. Counted features recompute on
        # every ingest batch, so the expensive model-read style was being wiped
        # minutes after it was produced, and the profiles read as "not read yet"
        # while the calls had genuinely been made and paid for.
        #
        # Only the counted columns are touched here. The learned ones belong to
        # style_llm and are left alone.
        conn.execute(
            "INSERT INTO recipient_styles"
            " (recipient_norm, display, relation, relation_conf, n_samples,"
            "  greeting, greeting_rate, signoff, signoff_rate, mean_words,"
            "  mean_sentence, contraction_rate, exclamation_rate, question_rate,"
            "  formality, active, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(recipient_norm) DO UPDATE SET"
            "   display = excluded.display,"
            "   relation_conf = excluded.relation_conf,"
            "   n_samples = excluded.n_samples,"
            "   greeting = excluded.greeting,"
            "   greeting_rate = excluded.greeting_rate,"
            "   signoff = excluded.signoff,"
            "   signoff_rate = excluded.signoff_rate,"
            "   mean_words = excluded.mean_words,"
            "   mean_sentence = excluded.mean_sentence,"
            "   contraction_rate = excluded.contraction_rate,"
            "   exclamation_rate = excluded.exclamation_rate,"
            "   question_rate = excluded.question_rate,"
            "   formality = excluded.formality,"
            "   active = excluded.active,"
            "   updated_at = excluded.updated_at",
            (
                profile.recipient_norm, profile.display, profile.relation,
                relation_conf, profile.n_samples, profile.greeting,
                profile.greeting_rate, profile.signoff, profile.signoff_rate,
                profile.mean_words, profile.mean_sentence,
                profile.contraction_rate, profile.exclamation_rate,
                profile.question_rate, profile.formality,
                int(profile.active), now,
            ),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO style_evidence (recipient_norm, episode_id, created_at)"
            " VALUES (?, ?, ?)",
            [(profile.recipient_norm, eid, now) for eid in profile.episode_ids],
        )
        profiles.append(profile)

    conn.commit()
    return profiles


def _resolve_by_description(conn: sqlite3.Connection, key: str):
    """Find a recipient from a word describing them: "mother" -> Amma.

    "send a message to my mother" reaches compose as to='mother'. There is no
    recipient called mother, so the profile lookup failed and the message was
    written in the generic voice - losing the one habit ("Hi Amma,") that makes
    it sound like her. The tool reported the miss in `style_note` and composed
    anyway, so nothing surfaced except a flatter message.

    The terms are read from what the model already wrote about each person -
    Amma's summary says "checks in on your mother's cooking", Appa's says
    "communicates with your father" - rather than from a hardcoded list of
    kinship words, which would need extending for every language and family this
    is used in.

    A match must be UNIQUE to count. "boss" describes both Sanjay and Vikram, so
    it resolves to neither and the generic voice is the honest outcome; guessing
    would put words in this person's mouth to somebody they did not name.
    """
    words = [w for w in re.findall(r"[a-z]{3,}", key) if w not in _NOT_A_NAME]
    if not words:
        return None

    matches = []
    for row in conn.execute(
        "SELECT * FROM recipient_styles WHERE n_samples >= 2"
    ).fetchall():
        haystack = " ".join(
            str(row[col] or "").lower()
            for col in ("summary", "distinctive", "relation_reason", "relation")
        )
        if any(re.search(rf"\b{re.escape(w)}('s)?\b", haystack) for w in words):
            matches.append(row)

    return matches[0] if len(matches) == 1 else None


# Words that describe the request rather than the person, so they never select a
# recipient on their own.
_NOT_A_NAME = {
    "the", "and", "for", "not", "you", "your", "our", "his", "her", "its",
    "message", "msg", "note", "mail", "email", "text", "reply", "send", "write",
}


def load_profile(conn: sqlite3.Connection, name: str) -> StyleProfile | None:
    """Fetch a recipient's profile by name, tolerating how it was typed."""
    key = normalise_recipient(name)
    row = conn.execute(
        "SELECT * FROM recipient_styles WHERE recipient_norm = ?", (key,)
    ).fetchone()
    if row is None:
        # "my amma" / "amma s" - a containment match, since the person will not
        # type the canonical key. Only helps when the name is actually in there.
        row = conn.execute(
            "SELECT * FROM recipient_styles WHERE ? LIKE '%' || recipient_norm || '%'"
            " ORDER BY LENGTH(recipient_norm) DESC LIMIT 1",
            (key,),
        ).fetchone()
    if row is None:
        row = _resolve_by_description(conn, key)
    if row is None:
        return None

    profile = StyleProfile(
        recipient_norm=row["recipient_norm"], display=row["display"],
        relation=row["relation"], n_samples=row["n_samples"],
        greeting=row["greeting"], greeting_rate=row["greeting_rate"],
        signoff=row["signoff"], signoff_rate=row["signoff_rate"],
        mean_words=row["mean_words"], mean_sentence=row["mean_sentence"],
        contraction_rate=row["contraction_rate"],
        exclamation_rate=row["exclamation_rate"],
        question_rate=row["question_rate"], formality=row["formality"],
    )
    profile.episode_ids = [
        r["episode_id"] for r in conn.execute(
            "SELECT episode_id FROM style_evidence WHERE recipient_norm = ? LIMIT 12",
            (profile.recipient_norm,),
        ).fetchall()
    ]
    return profile


def instructions(profile: StyleProfile) -> list[str]:
    """Turn a profile into concrete drafting rules.

    Rules, not adjectives. "Open with 'Hi Amma'" is followable and checkable;
    "write warmly" is neither.
    """
    if not profile.active:
        return []

    rules: list[str] = []
    if profile.greeting and profile.greeting_rate >= 0.5:
        rules.append(f'Open with "{profile.greeting}".')
    elif profile.greeting_rate <= 0.15:
        rules.append("Do not open with a greeting; start with the point.")

    if profile.signoff and profile.signoff_rate >= 0.4:
        rules.append(f'Close with "{profile.signoff}".')
    elif profile.signoff_rate <= 0.15:
        rules.append("Do not add a sign-off.")

    target = max(6, round(profile.mean_words))
    rules.append(f"Aim for about {target} words; stay close to that length.")

    if profile.formality >= 0.6:
        rules.append("Keep the register formal and complete; avoid slang.")
    elif profile.formality <= 0.35:
        rules.append("Keep it casual and conversational.")

    if profile.contraction_rate >= 0.5:
        rules.append("Use contractions.")
    elif profile.contraction_rate <= 0.15:
        rules.append("Avoid contractions.")

    if profile.exclamation_rate >= 0.4:
        rules.append("A warm exclamation mark is in character.")
    elif profile.exclamation_rate <= 0.1:
        rules.append("No exclamation marks.")

    return rules
