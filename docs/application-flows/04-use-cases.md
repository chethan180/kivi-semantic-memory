# End-to-end use cases

Use these as a demo/test script; exact record IDs and answers depend on corpus data.

## A. Ingest and build memory

Run the CLI ingest/extract/embed workflow on JSONL. Records are validated and idempotently saved in `episodes`; FTS triggers update immediately. Missing vectors use Gemini embeddings. The pipeline skips short/duplicate records, extracts schema JSON for the rest, then deterministic policy promotes only safe, adequately supported facts with source quotes. It rebuilds graph edges and recipient profiles; checkpointed style reads are the only optional LLM post-processing. Verify `/api/health` and Inspect show no unprocessed records.

## B. Durable recall

**Ask:** “Who is working on DSPM?”

Planner => `question`; tool loop => `recall`; memory vector/FTS ranking plus an optional graph hop finds facts; `memory_sources` loads original episodes; synthesis answers with citations. Click a citation to verify it opens the dictation that supports the claim.

## C. Time-filtered retrieval and redraft

**Ask:** “Find the Slack message I dictated around 5 PM yesterday and polish it for my steering meeting.”

`search_dictations` applies app/date/local-hour SQL filters before embedding and ranking. A single hour expands to a ±1-hour local window. It then calls `redraft` on returned IDs; Gemini rewrites only those records, applying stored preferences and a recipient profile if one was named. Verify no invented facts and that the cited Slack record matches the requested local time.

## D. New message

**Ask:** “Ask Appa if he took his medicine.”

The structured plan is `write`, so the first tool call is forced to `compose`. Compose loads Appa’s profile and general preferences, gives Gemini the literal request to constrain content, strips record IDs, and stores the result in `composed_messages`. It returns the exact draft without citations. Verify a later style-learning pass cannot use that generated message as a sample.

## E. Direct instruction and conflict

**Ask:** “Add Ashwin to DSPM as a frontend React developer.”

Planner => `instruction`; `remember(add)` stores the literal request as a `statement`, then uses deterministic promotion with direct-user provenance. If an active, high-confidence Ashwin fact contradicts it, the tool creates a clarification rather than replacing it. Otherwise it adds memory/relations. Ask the DSPM recall question again to verify retrieval.

## F. Safe no-result path

**Ask:** “What did I say to Priya in 2019?” when the corpus has no 2019 data.

Search returns no rows and `_why_empty()` runs SQL counts with filters relaxed; it can report that the date is outside corpus bounds. With no evidence or artifact, synthesis is restricted to a refusal and the citation/claim guard returns a safe abstention. Verify that it does not invent a fact or cite an unknown ID.
