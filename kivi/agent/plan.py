"""Decide what kind of request this is, before touching a tool.

Routing used to be implicit: the model read the instructions and its first
function call WAS the decision. That failed in a way worth describing, because
it is the argument for this file existing.

"Send a message to Appa asking if he took his medicine" routed to `compose` on
one run and to `recall` + `search_dictations` on the next - the word "asking"
makes a request to write look like a question. When it misrouted, the person was
told "I don't have anything about that in your dictations": a refusal to write
something new on the grounds that it had not been written before, which is never
a sensible answer. Nothing in the trace explained it either, since the only
record of the decision was the tool that got called.

Separating the decision from the action fixes both halves. The model commits to
a reading of the request first, in its own words; the tool choice then follows
from that commitment instead of standing in for it. And because the plan is
stored, a wrong answer can be traced to the point where it went wrong - the
reading, the retrieval, or the writing - rather than inferred from the outcome.

The cost is one call on the light model before each turn. That is real, and it
buys a decision the previous design was getting wrong intermittently on one of
the two things this product does.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from kivi.llm.gemini import GeminiClient, LLMError

log = logging.getLogger(__name__)

SYSTEM = """You read one request and decide what is being asked, before anything
is looked up. You do not answer it. You do not search. You classify and plan.

THE THREE KINDS

write        They want a message written to somebody else. "Write to Vikram
             that the work is due in two days." "Tell Amma I'll be late." "Ask
             Appa if he took his medicine." "Remind Arjun about the deposit."

             A message that ASKS the recipient something is still `write` - the
             question inside it is addressed to them, not to Kivi. This is the
             case most often got wrong.

             It is `write` however blunt the content is. Whether to send it is
             their decision, not yours.

question     They want to know something from their own dictations or from what
             Kivi has learned. "Who is on DSPM?" "What did I tell Bushan?"
             "What did I send at 5 PM yesterday?"

             "Tell me who is on DSPM" is a question, not a write: the recipient
             is Kivi itself. So is "remind me what I said to Priya".

instruction  They are telling Kivi something to remember, or to change what it
             believes. "Add Ashwin to DSPM as a frontend developer." "Priya is
             on Atlas now." "Always keep my Slack messages short." "Forget that."

             A fact mentioned in passing inside a question is NOT an
             instruction. "What did I tell Bushan about the review he was
             blocked on?" is a question; being blocked is context for the
             search.

chat        Anything else - a greeting, a thank-you, something too vague to act
             on yet.

FOR A WRITE

`recipient` is who the message goes to, as they referred to them: "Amma",
"Vikram", "my mother". Do not resolve it to a real name if they used a
relationship word - that is looked up later.

`message` is what the message must say, using THEIR words. Strip the framing
("send a message to X saying...") and keep the content. Add nothing. If they
said "finish the work in 2 days", that is the message - not a fuller, more
professional version of it, and not anything about why.

Keep whether it ASKS or TELLS. "Ask Appa if he took his medicine" is a question
being put to Appa, so `message` is "did you take your medicine?" - not "you took
your medicine", which asserts the thing they wanted to find out.

`reason` is one short sentence on why this kind and not another. It is shown to
an engineer working out why an answer came out the way it did, so be specific
about what in the wording decided it."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["write", "question", "instruction", "chat"]},
        "reason": {"type": "string"},
        "recipient": {"type": "string", "description": "Write only; else empty."},
        "message": {
            "type": "string",
            "description": "Write only: what it must say, in their words.",
        },
    },
    "required": ["kind", "reason"],
}


@dataclass
class Plan:
    kind: str = "question"
    reason: str = ""
    recipient: str = ""
    message: str = ""
    failed: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "reason": self.reason,
            "recipient": self.recipient, "message": self.message,
            "failed": self.failed,
        }

    def as_prompt(self) -> str:
        """What the tool loop is told it has already decided."""
        if self.failed:
            return ""
        lines = [f"YOU HAVE ALREADY DECIDED THIS IS A {self.kind.upper()}.",
                 f"Why: {self.reason}"]
        if self.kind == "write":
            lines += [
                f"Recipient: {self.recipient}",
                f"It must say: {self.message}",
                "Call `compose` with exactly that. Do not search their "
                "dictations first - they are asking for something new, and "
                "whether they have said it before does not matter.",
            ]
        elif self.kind == "question":
            lines.append(
                "Find the evidence and answer it, or say plainly that it is not "
                "there. Do not write a message to anybody."
            )
        elif self.kind == "instruction":
            lines.append("Record it with `remember`, then confirm in one line.")
        return "\n".join(lines)


def make_plan(
    client: GeminiClient,
    question: str,
    *,
    history: str = "",
    model: str | None = None,
) -> Plan:
    """Classify the request. Never raises - a failed plan just means no plan.

    Falling back to an unplanned turn rather than an error is deliberate: the
    tool loop worked without this for most requests, and a planning outage
    should degrade routing, not take the product down.
    """
    prompt = f"{history}Request: {question}"
    try:
        result = client.generate(
            prompt, system=SYSTEM, schema=SCHEMA,
            model=model or client.settings.gen_model,
            temperature=0.0, max_output_tokens=1024,
        )
        payload = result.json()
    except (LLMError, json.JSONDecodeError, ValueError) as exc:
        log.warning("planning failed, continuing unplanned: %s", exc)
        return Plan(failed=True)

    return Plan(
        kind=str(payload.get("kind") or "question").strip().lower(),
        reason=str(payload.get("reason") or "").strip(),
        recipient=str(payload.get("recipient") or "").strip(),
        message=str(payload.get("message") or "").strip(),
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd or 0.0,
    )
