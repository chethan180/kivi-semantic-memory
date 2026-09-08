"""Per-token pricing, so reported cost is real rather than decorative.

Prices are USD per 1M tokens, paid-tier standard, taken from
https://ai.google.dev/gemini-api/docs/pricing as of 2026-09-05.

An unknown model returns `None` rather than 0.0. Silently reporting zero cost for
a model we have no price for would be a lie in the evaluation report, and cost is
one of the things the assignment explicitly asks us to measure.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_AS_OF = "2026-09-05"
PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens."""

    input_per_m: float
    output_per_m: float


# Generation models. Long-context surcharge tiers (Pro models above 200k tokens)
# are not modelled; nothing in this system sends prompts that large.
GENERATION_PRICES: dict[str, ModelPrice] = {
    "gemini-3.8-flash": ModelPrice(0.75, 3.75),
    "gemini-3.7-flash": ModelPrice(0.75, 3.75),
    "gemini-3.6-flash": ModelPrice(0.75, 3.75),
    "gemini-3.5-flash": ModelPrice(1.50, 9.00),
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50),
    "gemini-3.1-flash-lite": ModelPrice(0.25, 1.50),
    "gemini-3.1-pro-preview": ModelPrice(2.00, 12.00),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40),
    "gemini-2.5-pro": ModelPrice(1.25, 10.00),
}

# Embedding models are charged on input only.
EMBEDDING_PRICES: dict[str, ModelPrice] = {
    "gemini-embedding-001": ModelPrice(0.15, 0.0),
    "gemini-embedding-2": ModelPrice(0.20, 0.0),
}


def _normalise(model: str) -> str:
    return model.removeprefix("models/").strip()


def lookup(model: str) -> ModelPrice | None:
    name = _normalise(model)
    if name in GENERATION_PRICES:
        return GENERATION_PRICES[name]
    if name in EMBEDDING_PRICES:
        return EMBEDDING_PRICES[name]
    # Tolerate `-preview`/date suffixes by falling back to the longest known prefix.
    candidates = [
        (known, price)
        for known, price in (GENERATION_PRICES | EMBEDDING_PRICES).items()
        if name.startswith(known)
    ]
    if candidates:
        return max(candidates, key=lambda item: len(item[0]))[1]
    return None


def estimate_cost(model: str, tokens_in: int, tokens_out: int = 0) -> float | None:
    """USD for a single call, or None when the model's price is unknown."""
    price = lookup(model)
    if price is None:
        return None
    return (tokens_in / 1_000_000) * price.input_per_m + (
        tokens_out / 1_000_000
    ) * price.output_per_m
