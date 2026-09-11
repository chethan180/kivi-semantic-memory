# RUN.md

## Primary review method

**A completely local application.** A Python FastAPI process serves both the
HTTP API and the web interface on `http://127.0.0.1:8000`. Storage is a single
SQLite file at `data/kivi.db`. The only external dependency is the Google Gemini
API.

Nothing is hosted. No Docker, no Node, no database server. `pip install` and one
`uvicorn` command is the whole setup.

The database is **not** committed — it is 34 MB of derived state. It is rebuilt
from the committed corpus in step 4 below. Everything needed to rebuild it is in
this repository.

---

## 1. Required runtimes and versions

| | |
|---|---|
| Python | **3.12 or newer.** Tested on 3.14.6. The pinned set resolves on 3.12 for Linux, macOS and Windows, and on 3.13 for Linux. **3.11 will not work**: the pinned numpy 2.5.2 has no 3.11 build. |
| pip | any recent version |
| OS | Windows, macOS or Linux. Developed on Windows 11. |

No other runtime is required. `sqlite3` ships with Python; `sqlite-vec` is
installed by pip and, if the platform cannot load SQLite extensions, the system
falls back automatically to exact numpy cosine search and says so in
`kivi doctor`.

## 2. Required environment variables

Copy `.env.example` to `.env` and fill in the one required value:

```bash
cp .env.example .env          # macOS / Linux / Git Bash
copy .env.example .env        # Windows cmd
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | **yes** | — | Google Gemini key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)). The only credential the system needs. |
| `KIVI_GEN_MODEL` | no | `gemini-3.5-flash-lite` | High-volume work: per-record extraction. |
| `KIVI_GEN_MODEL_HEAVY` | no | `gemini-3.5-flash` | Hey Kivi answering and tool selection. |
| `KIVI_EMBED_MODEL` | no | `gemini-embedding-001` | Embeddings for hybrid retrieval. |
| `KIVI_EMBED_DIM` | no | `768` | Embedding dimensionality. |
| `KIVI_DB_PATH` | no | `./data/kivi.db` | SQLite file location. |
| `KIVI_LLM_CACHE` | no | `true` | Cache responses and embeddings on disk by content hash. |
| `KIVI_OFFLINE` | no | `false` | Replay-only mode: serve from cache, raise on a miss. Zero spend. |
| `KIVI_DAILY_CALL_LIMIT` | no | `0` | Refuse new API calls after N per model per day. `0` disables. |
| `KIVI_REQUEST_TIMEOUT` | no | `60` | Seconds. |
| `KIVI_MAX_RETRIES` | no | `4` | Retries on transient API failure. |
| `KIVI_LOG_LEVEL` | no | `INFO` | Python log level. |

`.env` is gitignored. No credential is committed.

## 3. Install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.lock.txt
pip install -e .
```

`requirements.lock.txt` holds the exact versions this submission was tested
against. `requirements.txt` holds the same set as version ranges if you prefer
to resolve fresh. `pip install -e .` puts the `kivi` command on your PATH; if you
would rather not install the package, every command below also works as
`python -m kivi.cli <command>`.

Verify the environment before going further:

```bash
kivi doctor
```

This reports the key, the models, the schema version, embedding coverage,
whether `sqlite-vec` loaded, and the cache state. It **exits non-zero** if
anything would stop the system from running, so it can be used as a gate.

## 4. Create, migrate and seed the database

```bash
kivi init-db                                             # create + migrate (11 migrations)
kivi import data/corpus/office_episodes.jsonl         --source office
kivi import data/corpus/house_episodes.jsonl          --source house
kivi import data/corpus/mixed_episodes.jsonl          --source mixed
kivi import data/corpus/relational_episodes.jsonl     --source relational
kivi import data/experiments/amma_log.jsonl           --source experiment
kivi import data/experiments/mixed_log.jsonl          --source experiment
kivi import data/seed/sample_episodes.jsonl           --source seed
kivi import data/seed/six_oclock_test.jsonl           --source seed
kivi import data/seed/rohan_leave_test.jsonl          --source seed
kivi import data/seed/bushan_test.jsonl               --source seed
kivi embed                                               # 1,973 embeddings
kivi extract                                          # the memory pipeline
kivi learn-style                                      # per-recipient writing profiles
```

Import is idempotent — re-running inserts nothing new. `embed` and `extract` are
both resumable: an episode that already has a vector, or an `extract` trace, is
skipped, so a run that dies partway can simply be repeated.

All ten imports are needed. The four corpora carry the planted ground truth;
the experiment and seed files carry the per-person writing history and several
dictations the interactions in §7 depend on — "Priya is running point on DSPM" is
one. Skip them and the database, the style profiles and the README's numbers
will not match.

**Expected cost and time.** A full rebuild is roughly **15 minutes** and
**under $2** at the default models. A clean-clone rehearsal extracted 300
dictations in 99 seconds. Cumulative spend across all development was
$5.62 over 1,554 API calls; `kivi cost` reports it at any time.

**For a faster look**, extract over a slice instead:

```bash
kivi extract --limit 300
```

Retrieval, Hey Kivi and the interface all work on the partial memory set; only
the memory-stage evaluation numbers will differ from those reported in the
README.

Confirm the seed worked:

```bash
kivi stats        # row counts per table, episode span, app mix
```

Expect exactly **1,973 episodes** and **schema v11**.

## 5. Start every required process

**One process. That is all.**

```bash
uvicorn kivi.api:app --host 127.0.0.1 --port 8000
```

This serves the JSON API and the web interface together. There is no separate
frontend build step — the interface is plain HTML, CSS and JavaScript in
`frontend/`, mounted by the same process.

*Optional, not required for review:* `streamlit run ui/app.py` opens a separate
database-inspection harness. It is a debugging surface, not the product.

## 6. What to open

**<http://127.0.0.1:8000>**

The API's own documentation is at <http://127.0.0.1:8000/api/docs> if you want
to drive it directly.

## 7. Primary interactions to try

Type these into Hey Kivi in the interface. Each exercises a different claim.

| Ask | What it should demonstrate |
|---|---|
| `who is working on DSPM?` | A fact assembled from several dictations that never appear together, via one graph hop. Every name carries a `[ep_…]` citation you can click. |
| `what did I say about the DSPM migration?` | Search over the dictations themselves: vector search on the agent's query, plus up to three results from searching the person's own words with vector + BM25. |
| `send a message to my mother to prepare lunch by 12` | Per-recipient voice. The draft opens "Hi Amma" because 138 dictations to her do; nothing in the message is invented beyond the instruction. |
| `write to Vikram that the deadline moved two days` | The same machinery, a different person: ~52 words and formal, against Rahul's ~7. |
| `ask Appa if he took his medicine` | The message asks rather than asserts, and Appa gets no greeting because the person never uses one with him. |
| `what did I promise Nikhil about the audit?` | **Abstention on a near miss.** No promise exists — the closest dictation only says Nikhil must review scoping docs before a regulatory audit. Kivi replies that no promise was mentioned, rather than turning that task into one. |
| `add Ashwin to DSPM as a frontend developer` | A conversational write. Lands pinned at confidence 1.0, cited `[say_…]` and rendered as "you told Kivi". |
| `who is working on DSPM?` *(ask again)* | Ashwin now appears — memory changed behaviour within the session. |
| `change the DSPM owner from Priya to Rahul` | A correction to something that only ever existed in dictations. Recorded as "Rahul leads DSPM", cited `[say_…]`. |
| `who is running the DSPM project?` | Rahul, citing what you told Kivi. A direct correction outranks the older dictations — but only on the point it corrects, so Priya stays on the team. |
| `forget that` | Correction by sentence, not by administering a list. |

Two things to watch for while doing this:

- **Citations resolve.** Click one and the source drawer opens on the dictation,
  with the quoted span. A citation to something no tool returned is stripped by
  a deterministic guard before you ever see it, and the answer abstains instead.
- **The memory panel** shows what was learned, what was refused and why.

## 8. Run the candidate evaluation

Four commands, each scoring a different stage. Run them in this order:

```bash
kivi corpus-report          # is the corpus varied enough to be a test at all
kivi eval-retrieval         # hit@k, MRR, latency, with vector/bm25/hybrid ablation
kivi eval-memory            # stance guard, store growth, graph ablation
kivi eval-personalization   # does learned style change what gets written
kivi eval-agent             # Hey Kivi end to end: answers, citations, refusals (~$1.75)
```

`eval-agent` is the one that scores what a person actually sees. It runs Hey
Kivi on the planted questions — all 16 that have no answer in the history, plus
ten of each answerable kind, 56 in all — and scores each answer. An answerable
question must cite real records **and** contain the planted expected answer; a
question with no answer must cite nothing. Whether it cited the exact planted
dictation is reported beside that, and questions whose right answer differs
between corpora are flagged `ambiguous` (see the README's results). Every question writes a row to `eval/results/agent.jsonl` with the
plan, every tool call, the answer, its citations, whether it refused, latency,
tokens and cost. `--per-kind 0` runs every question (about 300, several dollars).

Each prints a table and writes per-case JSONL to `eval/results/`. They are
separable on purpose: a failure in `eval-retrieval` is a retrieval bug, while a
failure later with the evidence present is a reasoning bug.

Two further experiments, slower and more expensive, test whether *conditional*
habits can be learned from messages alone:

```bash
kivi exp-amma      # a lexical rule: which form of address, by message type
kivi exp-mixed     # three people, three kinds of habit, none described to the learner
```

The full test suite:

```bash
pytest -q          # 169 passed, 1 skipped
```

The skipped test is a live end-to-end call, enabled with `KIVI_LIVE_TESTS=1`.

## 9. Importing another corpus

The import format is JSONL, one record per line:

```json
{
  "id": "optional-source-id",
  "ts": "2026-08-14T17:03:00Z",
  "app": "slack",
  "raw_asr": "what the recogniser heard",
  "formatted": "What Kivi wrote.",
  "style_id": "optional",
  "duration_ms": 4200,
  "recipient": "optional - who it was addressed to",
  "meta": {}
}
```

**Required:** `ts`, `raw_asr`, `formatted`. Everything else is optional.
`app` is lowercased. `recipient` is what the per-person writing profiles are
built from; records without it still work, they simply contribute nothing to
style learning. Unknown keys in `meta` are preserved verbatim.

Records are deduplicated by `id` when present, otherwise by a content hash, so
importing the same file twice inserts nothing.

To import and process a new corpus:

```bash
kivi import /path/to/your_corpus.jsonl --source theirs
kivi embed
kivi extract
kivi learn-style
```

Use `--strict` to abort on the first malformed record rather than skipping it
and reporting at the end.

**If your records are in a different shape**, map them to the above with a short
script — the reader is `kivi/records.py` and it validates against exactly these
fields. No adapter framework is involved.

## 10. Where to inspect results and memory state

**In the interface**, at <http://127.0.0.1:8000>: the memory panel lists what was
learned with its provenance, and every answer's citations open the dictation
behind them.

**On disk:** `eval/results/` holds per-question JSONL from every evaluation run,
committed so the reported numbers can be checked against their inputs.

**From the terminal:**

```bash
kivi stats                       # row counts, episode span, app mix
kivi memories                    # what was learned, with provenance counts
kivi memories --type preference  # or entity, commitment
kivi candidates                  # what was CONSIDERED and REFUSED, with reasons
kivi candidates --status all
kivi policy                      # the promotion rules and ignore list in force
kivi cost                        # tokens and spend, per model
kivi recall "DSPM" --sources     # memory search with the episodes behind it
kivi search "deadline" --explain # episode retrieval with per-ranker positions
kivi ask "who is on DSPM?" -v    # Hey Kivi from the terminal, with tool calls
```

**Directly in SQLite**, if you would rather query it:

```bash
sqlite3 data/kivi.db "SELECT type, subject, confidence FROM memories WHERE status='active';"
sqlite3 data/kivi.db "SELECT reason, COUNT(*) FROM candidates WHERE status='rejected' GROUP BY reason;"
sqlite3 data/kivi.db "SELECT kind, COUNT(*), SUM(tokens_in), SUM(cost_usd) FROM traces GROUP BY kind;"
```

`traces` records every ingest and every answer — what was considered, what was
rejected, the reason, the model, tokens, cost and latency. It is the answer to
"why did memory affect this result, or not".

## 11. Resetting

```bash
kivi reset          # empty episodes, memories, traces and conversations
kivi reset --yes    # skip the confirmation prompt
```

The model response and embedding cache is **kept** by default, so a reset does
not re-bill the corpus on the next run. To clear that too:

```bash
kivi reset --include-cache --yes
```

To go all the way back to nothing, delete the file and re-run step 4:

```bash
rm data/kivi.db
kivi init-db
```

---

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `GEMINI_API_KEY is not set` | `.env` missing or empty. Step 2. |
| `kivi: command not found` | `pip install -e .` was skipped. Use `python -m kivi.cli <command>` instead. |
| `sqlite-vec unavailable` in `doctor` | Harmless. The build of Python cannot load SQLite extensions; exact numpy cosine search is used instead. Correctness is unchanged, and at this corpus size so is speed. |
| `N migration(s) pending` | Run `kivi init-db`. |
| `none imported yet` | Step 4. |
| `X/Y episodes embedded` | Run `kivi embed`. It is resumable. |
| Rate limits during `extract` | Free-tier quota is per-model per-project. Re-run the same command — it skips what is already done. Or set `KIVI_DAILY_CALL_LIMIT` to stop cleanly rather than erroring. |
