"""Let the model read the messages and say how this person writes.

Why this exists alongside the counted features in `style.py`: hand-written
detectors only find what someone thought to look for. A greeting regex and a
formality formula miss idiom, humour, what topics get raised with whom, how
directness shifts, and every other thing that makes writing to your brother
different from writing to your father. Enumerating those was never going to work.

The split that stays:

    the model LEARNS the style      - it can see what no rule was written for
    counted features SCORE it       - so the evaluation is not the model
                                      grading its own output

All seven recipients go in ONE call rather than one call each. That is cheaper,
but the real reason is contrast: asked to describe seven people together, the
model has to say what makes each one *different*, and difference is the whole
claim. Descriptions produced in isolation drift toward the same generic advice.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from kivi.llm.gemini import GeminiClient, LLMError

log = logging.getLogger(__name__)

# Enough to show a habit, few enough to keep one call cheap. Sampled across the
# length range rather than taken from the top, so a person's short replies and
# their longer messages are both represented.
SAMPLES_PER_RECIPIENT = 8

SYSTEM = """You study how one person writes to different people in their life.

You are given samples of their real messages, grouped by who each was written
to. Your job is to describe how they write TO EACH PERSON, precisely enough that
someone could imitate it.

What matters is DIFFERENCE. These people are being described together so you can
say what separates them. If two descriptions could be swapped without anyone
noticing, both are wrong. Look for:

- the exact opener, if there is a consistent one, and whether there is none
- length, and how much it varies
- register: blunt, warm, deferential, teasing, transactional
- contractions, slang, filler, emoji, exclamation marks
- sentence shape: fragments, or full sentences
- what they habitually do at the end - a question, a sign-off, nothing
- what subjects come up with this person and not others (for `distinctive`
  only - see below)

Write `rules` as instructions a writer could follow literally. Good: "Open with
exactly 'Hi Amma'." / "Keep to 6-10 words, usually a fragment." / "Never use a
greeting; start with the status." Bad: "Be warm." / "Write casually." An
adjective is not a rule.

RULES DESCRIBE HOW THEY WRITE, NEVER WHAT THEY WRITE ABOUT. This is the one
thing to get right, and the test is mechanical:

    Could this rule be followed on a message that says ONE thing?

If following it would require the writer to supply a fact, a reason, or a second
point that nobody gave them, the rule is broken. A rule is applied later to a
message whose content someone else has already decided, and a rule the content
cannot satisfy gets satisfied by invention instead.

  Never: "Focus on DSPM, compliance and audits."         (demands a subject)
  Never: "Write 3-4 dense sentences."                    (demands a length)
  Never: "Use transitions like 'Because' or 'Therefore'" (demands a second
                                                          clause to connect)
  Never: "Write a single dense multi-sentence paragraph." (demands both)

Rules like those produced "Hi Vikram. Because the aforementioned assignments are
still pending, please finish the work in 2 days. Thanks." from an instruction
that said only "finish the work in 2 days". The pending assignments do not
exist. A connective with nothing to connect will always be filled in.

Write them CONDITIONALLY instead, so a one-line message simply skips them:

  Instead: "When there is more than one point, run them together in one
            paragraph rather than separate lines."
  Instead: "Where clauses are joined, join them formally ('Therefore', 'As')
            rather than with 'and' or a dash."

Put the topics in `distinctive`, where it is a description of what you observed
and not an instruction to follow.

Length is a rule only as a CEILING or a shape ("rarely more than one line",
"usually a fragment"), never as a floor. A short instruction must be allowed to
stay a short message.

Ground every rule in what you actually see. If the samples do not show a
consistent habit, do not invent one - say the habit is inconsistent instead.
Fewer, accurate rules beat more, invented ones."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "profiles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": ["boss", "colleague", "friend", "family", "unknown"],
                    },
                    "relation_reason": {"type": "string"},
                    "summary": {
                        "type": "string",
                        "description": "One sentence the person would recognise as "
                                       "true about themselves.",
                    },
                    "distinctive": {
                        "type": "string",
                        "description": "What separates this from how they write to "
                                       "everyone else in the list.",
                    },
                    "rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3-6 literally followable drafting rules.",
                    },
                },
                "required": ["recipient", "relation", "summary", "distinctive", "rules"],
            },
        }
    },
    "required": ["profiles"],
}


@dataclass
class LearnedStyle:
    recipient: str
    relation: str = "unknown"
    relation_reason: str = ""
    summary: str = ""
    distinctive: str = ""
    rules: list[str] = field(default_factory=list)


def _sample(conn: sqlite3.Connection, recipient_norm: str, n: int) -> list[str]:
    """Take messages spread across the length range, not just the first n."""
    rows = conn.execute(
        "SELECT formatted FROM episodes WHERE recipient_norm = ?"
        " AND formatted IS NOT NULL AND formatted != ''"
        " ORDER BY LENGTH(formatted)",
        (recipient_norm,),
    ).fetchall()
    if not rows:
        return []
    if len(rows) <= n:
        return [r["formatted"] for r in rows]
    step = len(rows) / n
    return [rows[min(int(i * step), len(rows) - 1)]["formatted"] for i in range(n)]


def learn_styles(
    conn: sqlite3.Connection,
    client: GeminiClient,
    *,
    samples: int = SAMPLES_PER_RECIPIENT,
    model: str | None = None,
) -> list[LearnedStyle]:
    """One call, every recipient, contrastive descriptions."""
    recipients = conn.execute(
        "SELECT recipient_norm, MAX(recipient) AS display, COUNT(*) AS n"
        " FROM episodes WHERE recipient_norm IS NOT NULL AND recipient_norm != ''"
        " GROUP BY recipient_norm HAVING n >= 2 ORDER BY n DESC"
    ).fetchall()
    if not recipients:
        return []

    blocks: list[str] = []
    for row in recipients:
        messages = _sample(conn, row["recipient_norm"], samples)
        if not messages:
            continue
        blocks.append(
            f"=== TO {row['display']} ({row['n']} messages total, "
            f"{len(messages)} shown) ===\n"
            + "\n".join(f"  - {m}" for m in messages)
        )

    prompt = (
        "Here are one person's messages, grouped by who they were written to.\n"
        "Describe how they write to each of these people, and what makes each "
        "one different from the others.\n\n" + "\n\n".join(blocks)
    )

    # Output size scales with the number of people, and a truncated response
    # fails as a JSON parse error - which looks like the model refusing to
    # learn rather than running out of room. Eight recipients overran 4096 and
    # produced zero profiles after the calls had already been paid for. Retry
    # with more room before giving up.
    payload = None
    for max_tokens in (8192, 16384):
        try:
            payload = client.generate(
                prompt, system=SYSTEM, schema=SCHEMA,
                model=model or client.settings.gen_model_heavy,
                temperature=0.2, max_output_tokens=max_tokens,
            ).json()
            break
        except LLMError as exc:
            log.warning(
                "style learning at %d tokens failed: %s", max_tokens, exc
            )
    if payload is None:
        return []

    out: list[LearnedStyle] = []
    for item in payload.get("profiles", []):
        name = (item.get("recipient") or "").strip()
        if not name:
            continue
        out.append(
            LearnedStyle(
                recipient=name,
                relation=(item.get("relation") or "unknown").strip(),
                relation_reason=(item.get("relation_reason") or "").strip(),
                summary=(item.get("summary") or "").strip(),
                distinctive=(item.get("distinctive") or "").strip(),
                rules=[str(r).strip() for r in (item.get("rules") or []) if str(r).strip()],
            )
        )
    return out


def save(conn: sqlite3.Connection, learned: list[LearnedStyle]) -> int:
    """Store alongside the counted features, not instead of them."""
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    written = 0
    for item in learned:
        key = " ".join(item.recipient.strip().lower().split())
        cur = conn.execute(
            "UPDATE recipient_styles SET summary = ?, distinctive = ?,"
            " rules_json = ?, relation = ?, relation_reason = ?, updated_at = ?"
            " WHERE recipient_norm = ?",
            (
                item.summary, item.distinctive,
                json.dumps(item.rules, ensure_ascii=False),
                item.relation, item.relation_reason, now, key,
            ),
        )
        written += cur.rowcount
    conn.commit()
    return written
