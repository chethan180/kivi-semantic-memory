# End-to-end use cases

Use these as a demo/test script; exact record IDs and answers depend on corpus data.

## A. Ingest and build memory

Run the CLI ingest/extract/embed workflow on JSONL. Records are validated and idempotently saved in `episodes`; FTS triggers update immediately. Missing vectors use Gemini embeddings. The pipeline skips short/duplicate records, extracts schema JSON for the rest, then deterministic policy promotes only safe, adequately supported facts with source quotes. It rebuilds graph edges and recipient profiles; checkpointed style reads are the only optional LLM post-processing. Verify `/api/health` and Inspect show no unprocessed records.

## B. Durable recall

**Ask:** "Who is working on DSPM?"

Planner => `question`; tool loop => `recall`; memory vector/FTS ranking plus an optional graph hop finds facts; `memory_sources` loads original episodes; synthesis answers with citations. Click a citation to verify it opens the dictation that supports the claim.

## C. Time-filtered retrieval and redraft

**Ask:** "Find the Slack message I dictated around 5 PM yesterday and polish it for my steering meeting."

`search_dictations` applies app/date/local-hour SQL filters before embedding and ranking — to the agent's query and to the person's own words alike. A single hour expands to a ±1-hour local window. It then calls `redraft` on returned IDs; Gemini rewrites only those records, applying switched-on preferences and a recipient profile if one was named. Verify no invented facts and that the cited Slack record matches the requested local time.

## D. New message

**Ask:** "Ask Appa if he took his medicine."

The structured plan is `write`, so the first tool call is forced to `compose`. Compose loads Appa's profile and switched-on preferences, gives Gemini the literal request to constrain content, strips record IDs, and stores the result in `composed_messages`. It returns the exact draft without citations. Verify a later style-learning pass cannot use that generated message as a sample.

## E. Direct instruction, and a conflict answered "yes"

**Ask:** "Add Ashwin to DSPM as a frontend React developer."

Planner => `instruction`; `remember(add)` stores the literal request as a `statement` and promotes it with direct-user provenance, pinned at confidence 1.0. Ask the DSPM question again: Ashwin now appears.

**Then ask:** "Change the manager from Priya to Ashwin."

Ashwin is already recorded as a frontend developer, so this is a conflict. Nothing is written. Kivi stores the proposed change as a `confirm` clarification for this conversation and asks: *"Ashwin is recorded as 'frontend React developer on DSPM'. Replace that with 'manager of DSPM, taking over from Priya'?"*

**Answer:** "yes". The planner sees the pending question and classifies the reply as an instruction; the stored `revise` runs; the old value is superseded, not deleted; the question closes. "Who is running DSPM" now answers Ashwin. Answering "no, keep it" instead keeps the old value; ignoring the question drops it after one turn.

## F. Safe no-result path

**Ask:** "What did I say to Priya in 2019?" when the corpus has no 2019 data.

Search returns no rows and `_why_empty()` runs SQL counts with filters relaxed; it can report that the date is outside corpus bounds. With no evidence or artifact, synthesis is restricted to a refusal and the citation/claim guard returns a safe abstention. Verify that it does not invent a fact or cite an unknown ID.

## G. A question the agent's rewrite would have missed

**Ask:** "Who is running DSPM?"

The agent tends to search the bare topic, `"DSPM"`, which matches about 124 dictations; none of the top results says who runs it. The person's own words are searched as well, with vector + BM25, and the literal match on "running" brings back "Priya is running point on DSPM" (2 Sept) among up to three added results. Synthesis sees the dates, so it answers Priya rather than June's "Sanjay is leading". Verify the answer cites the September dictation.

## H. Control from "What Kivi knows"

Open the page. Facts are grouped by what they are about, preferences sit under *How you like things done*, and dictation conflicts under *Needs your decision*.

- **Where from** on a fact the person told Kivi lists that statement first (*you told Kivi*), then any dictations.
- **Forget this** marks the fact forgotten; it disappears from the page and from recall, including graph hops. The dictation behind it stays.
- **Stop doing this** drops a preference below `APPLY_PREFERENCE_AT`; drafting stops applying it from the next message. **Start doing this** puts it back.

## I. Score the answers

**Run:** `kivi eval-agent`

Runs Hey Kivi on the planted questions — all 16 with no answer in the history, plus ten of each answerable kind — and scores each answer: an answerable question must cite real records and contain the planted expected answer; a question with no answer must cite nothing. Citing the exact planted dictation is reported as a stricter second number, and questions whose right answer differs between corpora are flagged `ambiguous`. Every question writes a row to `eval/results/agent.jsonl`. Open a failed row to see the plan, every tool call, and what was cited.
