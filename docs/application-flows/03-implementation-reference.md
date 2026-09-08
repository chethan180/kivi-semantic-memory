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

`GeminiClient` centralizes REST retries (429/5xx), quota checks, caching, token and cost accounting. The configured light model handles planning/extraction; the heavy model handles function calling and answer synthesis.

## Regex inventory

| Code | Purpose |
| --- | --- |
| `agent/loop.py:_WRITE_REQUEST` + leading-question match | fallback write detection only when planning fails |
| `agent/loop.py` citation patterns | normalize/verify bracketed and bare internal references |
| `agent/tools.py:_RECORD_ID` | remove internal IDs from outbound message text |
| `retrieval/search.py:_FTS_SPECIALS` | sanitize natural language into safe FTS5 terms |
| `memory/style.py` greeting/sign-off/contraction patterns | measurable writing-style features |
| `memory/style.py` escaped word-boundary search | description-based recipient resolution |
| `db/migrate.py` filename regex | numbered migration validation |

Input record parsing is Pydantic plus field aliases, not regex. Salience is exact-text/word-count policy, not language inference.

## Main SQL map

| Stage | Tables | Operation |
| --- | --- | --- |
| ingest | `episodes`, `episodes_fts` | `INSERT OR IGNORE`; FTS triggers |
| embedding | `episode_vectors`, `memory_vectors`, `episode_vec` | select missing model/dimension vectors; BLOB inserts |
| memory queue | `episodes`, `traces` | `LEFT JOIN` to find no-extract-trace rows |
| promotion | `candidates`, `memories`, `memory_sources`, `relations`, `edges` | upsert, provenance, supersession, graph build |
| sessions | `sessions`, `turns` | load ordered last 8; append final turns |
| search | `episodes`, `episodes_fts`, `episode_vectors` | SQL filters then rank/fetch |
| recall | `memories`, `memories_fts`, `memory_vectors`, `edges`, `memory_sources` | rank, expand, provenance |
| compose | `composed_messages` | generated draft saved separately |
| instruction | `statements`, `memories`, `clarifications` | direct provenance, policy promotion, conflict question |
| audit | `traces`, `answers`, `api_call_log` | plan/tools/citations/outcome/cost/latency |

FastAPI serves the browser frontend and exposes health, ask, sessions, record, memory, preference/forget, and inspection routes. The browser uses `fetch`, maintains a session ID, and renders citations as record/fact drawers. The separate Streamlit app is an inspection harness.
