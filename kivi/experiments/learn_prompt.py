"""The prompt that learns how someone writes to one other person.

Two things it has to do that a naive "describe the style" prompt does not.

**Find conditional rules.** Real habits are not constant. Someone addresses their
mother as "Amma" about the gas cylinder and "Mama" when they miss her. A prompt
that asks for "the greeting" gets whichever is commoner and silently discards the
rule. So the schema has a place for `when -> then` and the instructions ask for
it explicitly.

**Not invent conditional rules.** The obvious failure of asking for conditions is
that the model finds them everywhere. The few-shot examples exist for that: they
include a father who is addressed exactly one way regardless of feeling, and a
colleague with no address form at all. Both are worked examples of the correct
answer being "no, this does not vary", so the model has seen that answer be
right before it is asked.

The few-shot cases are deliberately drawn from OTHER relationships than the one
being learned - boss, colleague, father - so nothing in them can be copied
straight into an answer about a mother.
"""

from __future__ import annotations

from typing import Any

FEW_SHOT = """
Here are three worked examples from other people this person writes to. Study the
shape of the answers, not their content - you will be asked about someone else.

--- EXAMPLE 1: messages to Appa (their father) ---
  "Car service is booked for Saturday morning, Appa."
  "Appa, the insurance renewal is due on the 14th."
  "Reached safely. Appa, tell Amma I will call tonight."
  "Appa, I transferred the money. Check once."
  "Missing you both a lot. Appa, take care of your knee."
CORRECT ANALYSIS:
  address_forms: always "Appa", with no variation
  conditional_rules: none - the last message is affectionate and still uses
    "Appa", so the form does not track feeling here
  other_rules: no greeting word; short factual sentences; often an imperative
    at the end
NOTE why this matters: it would be easy to assume a family member must have a
tender variant. This one does not. Report what is there.

--- EXAMPLE 2: messages to Priya (a colleague) ---
  "Pushed the fix, can you review before standup?"
  "That schema change is going to break the connector."
  "yeah agreed, let's do it after the release"
  "Can you take the Acme call? I am double booked."
CORRECT ANALYSIS:
  address_forms: none - she is never addressed by name at all
  conditional_rules: register shifts with the topic - lowercase and clipped when
    agreeing or making plans, full sentences when raising a technical problem
  other_rules: no greeting, no sign-off; frequently ends on a question

--- EXAMPLE 3: messages to Sanjay (their manager) ---
  "DSPM enterprise has slipped two weeks. Schema review was the blocker."
  "Sanjay, I need a decision on scope before I commit the team."
  "Confirming Thursday works. I will send the summary after."
CORRECT ANALYSIS:
  address_forms: "Sanjay" sometimes, no name other times
  conditional_rules: the name appears when asking him for something, and is
    dropped when merely reporting status
  other_rules: no greeting, no sign-off; leads with the fact; complete sentences,
    no contractions
NOTE: this IS a real conditional rule, and it is about function - asking versus
telling - not about emotion. Conditions are not always emotional.
""".strip()

SYSTEM = f"""You work out how one person writes to one specific other person, by
reading their messages.

You are shown real messages, in the order they were sent. Describe the habits
precisely enough that someone else could write a new message that this person
would accept as their own.

WHAT TO LOOK FOR

1. ADDRESS FORMS. How do they refer to this person - by name, by a nickname, by
   a kinship word, or not at all? If MORE THAN ONE form appears, that is the most
   important thing on this page. Work out what governs the choice: the emotional
   register, the topic, whether they are asking or telling, the time of day.
   State it as a condition.

2. CONDITIONAL RULES generally. Anything of the form "when the message is X, they
   do Y". Length, warmth, directness, and punctuation can all shift with the
   situation.

3. CONSTANTS. Habits that hold regardless: greeting or none, sign-off or none,
   typical length, sentence shape, contractions, exclamation marks.

EVIDENCE BEFORE BELIEF - the most important instruction here

A person has a bad day. They snap at someone once, or write one unusually curt
message, or use an endearment they have never used before. That is an EVENT, not
a habit, and treating it as one would change how every future message to that
person is written on the strength of a single sentence.

- Behaviour seen ONCE goes in `exceptions`, never in `conditional_rules` and
  never in `constants`. Record it, say it was seen once, and move on.
- Behaviour needs to appear at least TWICE, in different messages, before it can
  be called a rule.
- An established rule is NOT overturned by one message that contradicts it. Keep
  the rule, and note the contradiction as an exception. If it happens again, then
  revise the rule.
- One angry message does not make someone's tone angry. One affectionate message
  does not make it affectionate. Describe the pattern, not the last thing you read.

RULES ABOUT YOUR ANSWER

- Only claim a conditional rule when the messages show BOTH sides of it. If every
  message uses one form, say the form is constant. Inventing a condition is worse
  than reporting none, because it will be followed.
- Quote the messages that made you say each thing. A rule you cannot point at is
  a guess.
- Say what you are unsure about. `uncertain` is a real field and a short list
  there is a good answer when the evidence is thin.
- Write rules a person could follow literally. "Use 'Mama' when the message is
  affectionate, 'Amma' otherwise" is followable. "Be warm" is not.

{FEW_SHOT}
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "address_forms": {
            "type": "array",
            "description": "Every distinct way they refer to this person.",
            "items": {
                "type": "object",
                "properties": {
                    "form": {"type": "string"},
                    "when": {
                        "type": "string",
                        "description": "The condition, or 'always' if it does not vary.",
                    },
                    "examples": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["form", "when"],
            },
        },
        "conditional_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "when": {"type": "string"},
                    "then": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["when", "then"],
            },
        },
        "constants": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Habits that hold regardless of the message.",
        },
        "summary": {
            "type": "string",
            "description": "One sentence the person would recognise as true.",
        },
        "exceptions": {
            "type": "array",
            "description": "Behaviour seen ONCE. Recorded, deliberately not a rule.",
            "items": {
                "type": "object",
                "properties": {
                    "observed": {"type": "string"},
                    "times_seen": {"type": "integer"},
                    "note": {
                        "type": "string",
                        "description": "Why this is not yet a habit.",
                    },
                },
                "required": ["observed", "times_seen"],
            },
        },
        "uncertain": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What the evidence does not yet settle.",
        },
    },
    "required": ["address_forms", "conditional_rules", "constants", "summary"],
}


def first_pass_prompt(recipient: str, messages: list[str]) -> str:
    listing = "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(messages))
    return (
        f"These are {len(messages)} messages this person sent to {recipient}, "
        f"oldest first.\n\n{listing}\n\n"
        f"How does this person write to {recipient}?"
    )


def revision_prompt(
    recipient: str, previous: dict[str, Any], messages: list[str], seen: int
) -> str:
    """Revise an existing understanding with newly arrived messages.

    Incremental rather than a re-read: the model gets what it concluded before
    plus what has happened since. That is cheaper, but more importantly it is the
    honest shape of memory - and it lets the transcript show a belief being
    revised, which a fresh analysis each time would hide.
    """
    import json as _json

    listing = "\n".join(f"  {i + 1}. {m}" for i, m in enumerate(messages))
    return (
        f"Earlier you read {seen} messages this person sent to {recipient} and "
        f"concluded:\n\n{_json.dumps(previous, indent=2, ensure_ascii=False)}\n\n"
        f"Here are {len(messages)} NEW messages that arrived since, oldest first:\n\n"
        f"{listing}\n\n"
        "Revise your understanding using everything now known - the earlier "
        "conclusion plus these.\n\n"
        f"Be conservative. You are revising a description built on {seen} "
        f"messages using {len(messages)} new ones, so the earlier conclusion "
        "carries more weight than what you have just read.\n"
        "- Keep every rule that still holds.\n"
        "- If something you listed in `exceptions` has now happened AGAIN, "
        "promote it to a rule - that is what the threshold is for.\n"
        "- If ONE new message contradicts an established rule, keep the rule and "
        "record the contradiction in `exceptions`. Do not rewrite a habit "
        "because of a single message.\n"
        "- Only change a rule outright when several new messages disagree with "
        "it.\n"
        "- If more evidence has settled something in `uncertain`, move it out."
    )
