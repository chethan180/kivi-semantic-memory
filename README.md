# Kivi — semantic memory

Kivi is a voice-first assistant that remembers what a person said about their
work — not what a model guessed. It learns **how someone deals with the world,
not how they talk to Kivi**: who they work with, what they promised, and how
they write to each person in their life.

- **[Product positioning statement](docs/positioning-statement.md)** (100 words)
- **[Product vision document](docs/vision-document.md)** (600 words)
- **[RUN.md](RUN.md)** — the primary review method, start to finish

**To run it:** see [RUN.md](RUN.md). One `pip install`, one `uvicorn` command,
one SQLite file. No Docker, no Node, no database server.

---

## What the person experiences

The interface is Hey Kivi. There is no memory settings page and no queue of rows
to triage — the assignment is explicit that the person must stay in control
without becoming the administrator, so control happens in conversation, at the
moment memory is used.

| The person says | What happens |
|---|---|
| *"who is working on DSPM?"* | Names assembled from dictations that never appear together. Each carries a citation that opens the dictation behind it. |
| *"send a message to my mother to prepare lunch by 12"* | A draft that opens "Hi Amma", because 138 dictations to her do. Nothing in it is invented beyond the instruction. |
| *"write to Vikram that the deadline moved"* | ~52 words and formal. The same request to Rahul produces ~7 words and no greeting. |
| *"what did I promise Nikhil about the audit?"* | *"I don't have anything about that in your dictations."* No such promise exists, so none is invented. |
| *"add Ashwin to DSPM"* | Written immediately, pinned, cited as **"you told Kivi"**. The next question includes him. |
| *"forget that"* | Correction by sentence. |

## The five capabilities, and the ones refused

Five tools, chosen because the use cases need them and no more:
`search_dictations`, `recall`, `redraft`, `compose`, `remember`.

Deliberately **not** built, each for a stated reason:

| Not built | Why |
|---|---|
| A profile of the person (communication style, learning style, friction points) | Reading character out of how someone talks to their assistant is the easiest impressive thing to build here, and it is what turns memory into surveillance. Kivi records what a person does in the world, not what kind of person they are. |
| A searchable archive of past Hey Kivi conversations | Transcripts are session-scoped working context. Anything worth carrying between conversations is promoted to a fact, where it is versioned, correctable and provenanced. A chat archive would be exactly the ungoverned second store that boundary exists to prevent. |
| Context partitioning (work / family / health) | Skipped under the assignment's own instruction to build the smallest set of capabilities worth using. The privacy guarantee rests on the `stance` guard instead — Kivi does not learn about third parties in the first place. Named as a limitation below rather than left unmentioned. |
| Self-evolving memory that rewrites itself | Merging memories with a model loses detail on every pass. Profiles here are **recomputed from evidence**, never re-authored, so they cannot drift out of agreement with what was actually said. |
| Memory inside ordinary dictation | Dictation must stay predictable. It writes down what was said; memory is not allowed to quietly improve it. The one thing dictation contributes is noticing who a message was addressed to. |

## Architecture

```
                 ┌───────────── frontend/ (plain HTML/CSS/JS) ─────────────┐
                 │                    served by                            │
                 └──────────────── kivi/api.py (FastAPI) ──────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
            INGEST (not agentic)                        HEY KIVI (agentic, bounded)
                    │                                           │
   episodes ──► Gate 1 salience (no LLM)              plan.py — classify the request
                    │                                     write | question | instruction
              Gate 2 extraction (LLM)                        │
                    │  type, subject, stance,            bounded tool loop
                    │  quote, explicit, first_person      4 calls / 2 rounds
                    ▼                                          │
              Gate 3 promotion (NO LLM — pure policy)     ┌─────┴──────┐
                    │                                     │  5 tools   │
       ┌────────────┼────────────┐                        └─────┬──────┘
       ▼            ▼            ▼                              ▼
   memories    candidates      edges                   citation guard (deterministic)
   (96 active) (148 refused,   (graph as a                      │
                with reasons)   retrieval aid)             abstain, or answer
```

**Two layers, and the boundary between them is the design.**

- **`episodes`** — every dictation, stored losslessly, never judged, always
  searchable. Immutable, enforced by a trigger.
- **`memories`** — a small curated set that exists only to change future
  behaviour. 1,973 dictations produced **96** active memories.

*Searchable is not the same as remembered.* A grocery list stays findable
forever and is never promoted to a belief.

### The gate: the LLM proposes, deterministic policy disposes

If the LLM is the gate, a rejection cannot be explained or tested. So the model
does only the linguistic work, and promotion is pure code
([`kivi/memory/policy.py`](kivi/memory/policy.py),
[`promote.py`](kivi/memory/promote.py)):

| Type | Promotes when |
|---|---|
| `entity` | seen in ≥2 distinct episodes, **or** explicitly defined in one |
| `preference` | stated outright → 1 episode. Inferred from behaviour → ≥3, and capped at low confidence permanently |
| `commitment` | 1 episode, but requires a resolvable date **and** a first-person subject |

**Rejected candidates are kept, with their reason.** This is the assignment's
"what it deliberately ignored", as a queryable table rather than a claim in a
README: `kivi candidates`.

### `stance` — the privacy mechanism

Every extracted candidate carries a `stance`: is the speaker **asserting**
something about their own world, or **transcribing** someone else's? Only
`asserted` can ever be promoted. A doctor's dictation about a patient, a
lawyer's about a client, an email about a colleague's review — the speaker is
the source, not the subject, and none of it becomes memory.

Enforced twice: described to the extractor in Gate 2, and checked again in code
at Gate 3, *because a prompt instruction is a request and a code path is a
guarantee.* Backed by a six-category ignore list (health, relationships, money,
employment status, character judgements, third-party confidential) that
over-rejects on purpose.

### Personalisation — memory that changes behaviour

The clearest thing semantic memory can *do* for a writing product, as opposed to
merely knowing things.

- **Counted features** ([`style.py`](kivi/memory/style.py)) — greeting rate,
  sign-off rate, mean words, contractions, formality. Deterministic, no model.
- **Model-read rules** ([`style_llm.py`](kivi/memory/style_llm.py)) — all
  recipients described in **one** call, so the model must say what makes each
  one *different*.

The split matters: **the model learns the style, counted features score it.** If
the model both wrote the rules and judged adherence, the evaluation would be
grading its own homework.

Three guards, all load-bearing:

- Style decides **wording only**. Rules that demand a subject, a length or a
  second clause are rejected at the learning prompt, because a rule the content
  cannot satisfy gets satisfied by invention instead.
- Profiles are **derived, never authored** — a materialised view over episodes.
- Kivi's own drafts live in `composed_messages`, **not** `episodes`, so it can
  never learn the person's voice from its own imitation of it.

### Retrieval

RRF fusion over vector search (`sqlite-vec`, with automatic numpy fallback),
FTS5 `bm25()`, and recency, plus structured filters on time, app and entity.
Then **one hop** of graph expansion — the only mechanism that reliably handles
facts distributed across dictations that never co-occur.

The graph is a **retrieval aid, never a truth store**. Edge weight is
co-occurrence count; nothing is ever asserted to the user from an edge alone, so
a wrong edge degrades ranking rather than correctness.

### Inspecting why memory did or did not affect a result

`traces` records every ingest and every answer: what was considered, what was
rejected and why, the model, tokens, cost and latency. 2,066 rows. One
mechanism serving both the inspection requirement and every number below.

---

## Results

All figures from the committed runs in [`eval/results/`](eval/results/).
Reproduce with the four commands in [RUN.md §8](RUN.md).

### Selectivity — the store stays small

| | |
|---|---|
| Dictations ingested | **1,973** (2026-06-22 → 2026-09-05, 8 apps) |
| Active memories | **96** — 63 entities, 22 preferences, 11 commitments |
| Ratio | **4.9%** — sublinear, as required |
| Candidates considered | 454 → 136 promoted, 148 rejected, 170 still accumulating |
| Provenance rows | 1,897 |
| Graph edges | 165 |

Largest rejection reasons, all queryable with `kivi candidates`:

| n | Reason |
|---:|---|
| 122 | only 1 of 2 required episodes of evidence |
| 89 | commitment with no resolvable date — nothing to expire |
| 25 | commitment with no first-person subject — not the user's promise |
| **24** | **`stance=transcribed`** — the speaker was not asserting this about their own world |

### Retrieval — and an honest negative

Hit rate over 286 planted questions across three corpora, with ablations:

| Config | hit@1 | hit@5 | hit@10 | recall | p50 |
|---|---:|---:|---:|---:|---:|
| **vector-only** | **0.206** | **0.647** | **0.832** | **0.635** | 20 ms |
| hybrid (vector + bm25) | 0.119 | 0.556 | 0.801 | 0.601 | 21 ms |
| hybrid + recency | 0.119 | 0.549 | 0.801 | 0.601 | 21 ms |
| bm25-only | 0.098 | 0.350 | 0.531 | 0.330 | 0 ms |

**Hybrid retrieval lost.** Adding BM25 to vector search made every metric worse
— hit@1 nearly halved. Spoken dictation is short, paraphrased and full of ASR
variance, which is close to the worst case for lexical matching, and RRF gives
the weak ranker equal standing with the strong one.

This was expected to go the other way. The finding is reported rather than
buried, and the code follows it: **`--bm25` is off by default** in `kivi search`
and the ranker stays available for the ablation. Recency changed nothing at all.

### Personalisation

| | Without style | With style | Δ |
|---|---:|---:|---:|
| Adherence (21 drafts each, 7 recipients) | 0.540 | **1.000** | **+0.460** |

Learned profiles, all from dictations alone:

| To | Relation | n | Greeting learned | Mean words |
|---|---|---:|---|---:|
| Amma | family | 138 | "hi amma" | 16.1 |
| Sanjay | boss | 102 | "hey sanjay" | 22.3 |
| Rahul | friend | 55 | "hey rahul" | **6.6** |
| Priya | colleague | 40 | "hey priya" | 16.6 |
| Arjun | family | 34 | **none** | 11.3 |
| Nikita | friend | 30 | "hey nikita" | 24.1 |
| Appa | family | 26 | **none** | 12.9 |
| Vikram | boss | 22 | "hi vikram" | **52.5** |

Rahul at 6.6 words against Vikram at 52.5 is an 8× spread from the same writer.
Arjun and Amma are both family and differ on whether there is a greeting at all,
so what was learned is the **person**, not the category.

**This +0.460 is weaker evidence than it looks, and is reported as such.** The
corpus generator was told to give Amma the greeting "Hi Amma"; the generator
complied; the learner detected it; the scorer checked for it. Every link in that
chain restates the same specification. It demonstrates that a deterministic
instruction survives a round trip through a language model. It does *not* yet
demonstrate that style can be learned from data where nobody planted the answer.

What would settle it is a discrimination test — hide the recipient, identify who
a held-out message was written to from the profiles alone, against a chance
floor of 14.3% across seven people. That is the most important thing this
evaluation does not yet do, and it is named in Limitations rather than glossed.

Two experiments do push past the spec, in
[`eval/results/exp_amma.json`](eval/results/) and `exp_mixed.json`: habits that
are **conditional** ("Mama" when affectionate, "Amma" when routine) and never
described to the learner, scored on unseen prompts.

### Behaviour, latency and cost

| | |
|---|---|
| Answers logged | 93, of which **7 abstained** |
| Composed messages | 65, stored outside `episodes` |
| Conversational writes | 7 statements, cited `[say_…]` |
| Retrieval p50 / p95 | 20 ms / ~60 ms |
| End-to-end, cached question | < 1 s |
| End-to-end, message compose | 5–7 s (planner adds ~1 s) |
| Total development spend | **$5.62** across 1,554 API calls |
| Database growth | 1,973 episodes → 34 MB including vectors and cache |

Tests: **126 passed, 1 skipped** (the skip is a live API call, enabled with
`KIVI_LIVE_TESTS=1`).

---

## Limitations

Stated with evidence rather than omitted.

1. **The personalisation result is partly circular.** See above. The
   discrimination metric that would settle it is designed but not built.
2. **`stance` is a single point of failure.** With context partitioning
   deliberately not built, the entire "Kivi does not learn about third parties"
   guarantee rests on one LLM-extracted field, backed by a keyword ignore list.
   It holds on the planted third-party episodes, but it is one mechanism where a
   careful system would have two.
3. **No context modelling.** "Mom" and "DSPM" share one flat store, so a work
   query can surface a personal entity on semantic match.
4. **Sanjay is the manager and classifies as `colleague`.** This person writes
   to their manager with no greeting and 22 words — stylistically
   indistinguishable from a peer. Relation inference is 6/7 and this is the
   miss. Left unpatched deliberately: style reveals how someone is *treated*,
   not who they are, and no drafting behaviour depends on the label.
5. **Retrieval order is the model's choice.** `recall` and `search_dictations`
   are both tools and nothing enforces facts-first. A mandatory ordering would
   add a wasted round trip to the assignment's own worked example, which needs a
   dictation rather than a fact. A choice, not an omission.
6. **Kivi's own replies ignore what it knows about how the person writes.** It
   applies preferences when drafting *their* messages and then answers *them* in
   whatever voice the model picks. Two writing surfaces, one personalised.
7. **The corpus is synthetic and single-user**, generated by the same model
   family that reads it.

## Use of AI

- **Part One** — the [positioning statement](docs/positioning-statement.md) and
  [vision document](docs/vision-document.md) are the author's own thinking and
  own words, as the assignment requires.
- **Part Two** — written with AI assistance (Claude) throughout: implementation,
  tests, prompt iteration, and this README. Every architectural decision,
  refusal and trade-off recorded here was made by the author and is defended in
  the vision document.
- **The corpus** was generated with Gemini from a seeded Python plan. The plan —
  what each record must convey — is decided in deterministic code *before* any
  model call, so the ground truth is independent of what the model chose to say.
  See [`kivi/corpus/`](kivi/corpus/).
- **The system itself** uses Gemini for four things only: candidate extraction,
  style reading, request planning, and answering. Every promotion, rejection,
  citation check and abstention is deterministic code.

## Repository map

```
docs/positioning-statement.md   Part One — 100 words
docs/vision-document.md         Part One — 600 words
docs/application-flows/         how ingestion and Q&A actually work, step by step
RUN.md                          the primary review method

kivi/api.py                     FastAPI backend; also serves the frontend
kivi/cli.py                     every command in RUN.md
frontend/                       the interface — plain HTML, CSS, JS
ui/app.py                       Streamlit database-inspection harness (not the product)

kivi/agent/plan.py              classify the request before any tool runs
kivi/agent/loop.py              bounded tool loop, citation guard, abstention
kivi/agent/tools.py             the five tools

kivi/memory/extract.py          Gate 1 salience, Gate 2 extraction (the stance prompt)
kivi/memory/policy.py           the promotion rules and ignore list, as data
kivi/memory/promote.py          Gate 3 — deterministic promotion
kivi/memory/style.py            counted style features
kivi/memory/style_llm.py        model-read style rules
kivi/memory/search.py           memory recall + one-hop graph expansion

kivi/retrieval/search.py        hybrid episode retrieval, RRF fusion
kivi/llm/gemini.py              Gemini over raw httpx, with caching and cost accounting
kivi/db/migrations/             10 migrations, schema v10

data/corpus/                    1,775 records across four corpora, with ground truth
data/seed/                      small fixtures for the worked cases
eval/                           the evaluation and its committed results
tests/                          126 tests
```
