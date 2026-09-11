# Implementation reference: LLMs, regex, and SQL

## LLM inventory

| Code | Purpose | Output |
| --- | --- | --- |
| `retrieval/embed.py` | document and query embeddings | vectors |
| `memory/extract.py` | candidate, stance, quote, relation extraction | schema JSON |
| `memory/style_llm.py` | recipient-specific style rules | schema JSON |
| `agent/plan.py` | request classification before tools | schema JSON |
| `agent/loop.py` | tool selection and final synthesis | function calls / cited text |
| `agent/tools.py:redraft` | rewrite existing dictations | draft artifact |
| `agent/tools.py:compose` | new message in recipient voice | draft artifact |
| `evaluation/agent.py` | runs the whole Hey Kivi turn on each planted question | one scored row per question |

`GeminiClient` centralizes REST retries (429/5xx), quota checks, caching, token and cost accounting. The configured light model handles planning/extraction; the heavy model handles function calling and answer synthesis. Every 429 is retried, including a spending-cap 429 that cannot succeed on retry, so a capped key makes each call wait through its backoff before failing.

## Regex and keyword inventory

Rules that read sentences without a model. The ones marked **decides** change what Kivi keeps or says.

| Code | Purpose |
| --- | --- |
| `agent/loop.py:_WRITE_REQUEST` + leading-question match | fallback write detection, only when planning fails |
| `agent/loop.py:_looks_like_a_claim` | **decides** whether an uncited answer is a refusal or a claim. A phrase list: it misreads some claims containing "outside" or "no record" as refusals |
| `memory/policy.py:ignore_match` | **decides** the ignore list: plain substring match on keywords, so "nda" matches in "Monday" and blocks ordinary commitments |
| `memory/policy.py:MIN_WORDS`, exact-duplicate window | **decides** Gate 1 salience |
| `memory/policy.py:normalise_entity` / `ENTITY_NOISE_WORDS` | **decides** which entity mentions merge ("Beacon migration" = "Beacon") |
| `agent/loop.py` citation patterns | normalize/verify bracketed and bare internal references |
| `agent/tools.py:_RECORD_ID` | remove internal IDs from outbound message text |
| `retrieval/search.py:_FTS_SPECIALS` | sanitize natural language into safe FTS5 terms |
| `memory/style.py` greeting/sign-off/contraction patterns | measurable writing-style features |
| `memory/style.py` `_KINSHIP`, escaped word-boundary search | relation inference; description-based recipient resolution ("my mother" → Amma) |
| `db/migrate.py` filename regex | numbered migration validation |

Input record parsing is Pydantic plus field aliases, not regex.

## Main SQL map

| Stage | Tables | Operation |
| --- | --- | --- |
| ingest | `episodes`, `episodes_fts` | `INSERT OR IGNORE`; FTS triggers |
| embedding | `episode_vectors`, `memory_vectors`, `episode_vec` | select missing model/dimension vectors; BLOB inserts |
| memory queue | `episodes`, `traces` | `LEFT JOIN` to find no-extract-trace rows |
| promotion | `candidates`, `memories`, `memory_sources`, `relations`, `edges`, `clarifications` | upsert, provenance, supersession, graph build; a newer dictation against a pinned fact raises a `dictation` clarification |
| sessions | `sessions`, `turns` | load ordered last 8; append final turns |
| search | `episodes`, `episodes_fts`, `episode_vectors` | SQL filters, then the agent's query by vector (top 11) and the person's words by vector + BM25 (up to 3 more) |
| recall | `memories`, `memories_fts`, `memory_vectors`, `edges`, `memory_sources`, `statements` | rank, expand, provenance; told facts dated |
| compose | `composed_messages` | generated draft saved separately |
| preferences | `memories` (type `preference`) | applied to drafts only at confidence ≥ `policy.APPLY_PREFERENCE_AT`; the page's switch moves a preference across that line |
| instruction | `statements`, `memories`, `clarifications` | direct provenance, policy promotion; a conflict stores a `confirm` clarification with `proposed_json`, the exact revise to run on "yes" (migration 011) |
| audit | `traces`, `answers`, `api_call_log` | plan/tools/citations/outcome/cost/latency |
| evaluation | `answers`, `traces` → `eval/results/agent.jsonl`, `agent_summary.json` | one row per planted question: plan, tools, answer, citations, grounding, refusal, latency, tokens, cost |

FastAPI serves the browser frontend and exposes health, ask, sessions, record, memory, preference/forget, and inspection routes. The browser uses `fetch`, maintains a session ID, and renders citations as record/fact drawers. On "What Kivi knows", *Where from* lists what the person told Kivi first, then the dictations; *Needs your decision* shows dictation conflicts only. The separate Streamlit app is an inspection harness.
