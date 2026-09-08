-- Kivi core schema.
--
-- Two layers, and the boundary between them is the whole design:
--   episodes  every dictation, stored losslessly, never judged, always searchable
--   memories  a small curated set that exists only to change future behaviour
--
-- Everything else here exists to make that boundary inspectable: candidates
-- records what was considered and refused, memory_sources records why a memory
-- exists, and traces records why any decision went the way it did.

-- ---------------------------------------------------------------------------
-- Episodes: the lossless layer
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS episodes (
    id            TEXT PRIMARY KEY,          -- ep_<hash>, stable across re-imports
    external_id   TEXT UNIQUE,               -- id from the source corpus, if any
    content_hash  TEXT NOT NULL,             -- dedupe key when external_id is absent
    ts            TEXT NOT NULL,             -- ISO-8601 UTC, human readable
    ts_epoch      INTEGER NOT NULL,          -- seconds, for range filters
    app           TEXT,                      -- slack, gmail, notes, ...
    raw_asr       TEXT NOT NULL,             -- what the recogniser heard
    formatted     TEXT NOT NULL,             -- what Kivi wrote
    style_id      TEXT,
    duration_ms   INTEGER,
    meta_json     TEXT NOT NULL DEFAULT '{}',
    source        TEXT NOT NULL DEFAULT 'import',
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes (ts_epoch DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_app_ts ON episodes (app, ts_epoch DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_content_hash ON episodes (content_hash);

-- Lexical half of hybrid retrieval. External-content FTS5: the index stores no
-- copy of the text, it points back at episodes by rowid.
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5 (
    raw_asr,
    formatted,
    content='episodes',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS episodes_fts_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts (rowid, raw_asr, formatted)
    VALUES (new.rowid, new.raw_asr, new.formatted);
END;

CREATE TRIGGER IF NOT EXISTS episodes_fts_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts (episodes_fts, rowid, raw_asr, formatted)
    VALUES ('delete', old.rowid, old.raw_asr, old.formatted);
END;

CREATE TRIGGER IF NOT EXISTS episodes_fts_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts (episodes_fts, rowid, raw_asr, formatted)
    VALUES ('delete', old.rowid, old.raw_asr, old.formatted);
    INSERT INTO episodes_fts (rowid, raw_asr, formatted)
    VALUES (new.rowid, new.raw_asr, new.formatted);
END;

-- Vectors live in an ordinary table so they are never trapped inside an
-- extension-only structure. A vec0 index over these rows is an accelerator
-- built in Phase 4 when sqlite-vec is available; numpy reads the same rows when
-- it is not.
CREATE TABLE IF NOT EXISTS episode_vectors (
    episode_id TEXT PRIMARY KEY REFERENCES episodes (id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Candidates: what was considered, including what was refused
--
-- This table is the answer to "what did it deliberately ignore". A rejected
-- candidate is kept, with its reason, rather than dropped - which is also what
-- makes repetition-based promotion possible.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS candidates (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,     -- entity | preference | commitment
    subject         TEXT NOT NULL,
    body            TEXT,
    entity_type     TEXT,              -- person | project | org | place | thing | term
    stance          TEXT NOT NULL,     -- asserted | transcribed | reported | unclear
    explicit        INTEGER NOT NULL DEFAULT 0,
    quote           TEXT,              -- verbatim span that produced it
    episode_id      TEXT REFERENCES episodes (id) ON DELETE CASCADE,
    evidence_count  INTEGER NOT NULL DEFAULT 1,
    status          TEXT NOT NULL,     -- pending | promoted | rejected
    reason          TEXT,              -- why it was rejected, or why promoted
    memory_id       TEXT,              -- set once promoted
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates (status, type);
CREATE INDEX IF NOT EXISTS idx_candidates_subject ON candidates (type, subject);
CREATE INDEX IF NOT EXISTS idx_candidates_episode ON candidates (episode_id);

-- ---------------------------------------------------------------------------
-- Memories: the curated layer
--
-- Versioned, never overwritten in place. A correction writes a new row and
-- points it at the one it replaces, so "this changed on Tuesday because you
-- said X" is answerable from the data.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,       -- entity | preference | commitment
    subject       TEXT NOT NULL,
    body          TEXT,
    entity_type   TEXT,
    canonical     TEXT,                -- preferred spelling for entities
    aliases_json  TEXT NOT NULL DEFAULT '[]',
    confidence    REAL NOT NULL DEFAULT 0.5,
    status        TEXT NOT NULL DEFAULT 'active',  -- active | superseded | expired | forgotten
    pinned        INTEGER NOT NULL DEFAULT 0,      -- user-confirmed; extraction may not overwrite
    source        TEXT NOT NULL DEFAULT 'extraction',  -- extraction | user
    use_count     INTEGER NOT NULL DEFAULT 0,
    last_used_at  TEXT,
    created_at    TEXT NOT NULL,
    valid_from    TEXT NOT NULL,
    valid_to      TEXT,
    supersedes_id TEXT REFERENCES memories (id) ON DELETE SET NULL,
    due_at        TEXT                 -- commitments only
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories (status, type);
CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories (type, subject);
CREATE INDEX IF NOT EXISTS idx_memories_due ON memories (due_at)
    WHERE due_at IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5 (
    subject,
    body,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts (rowid, subject, body)
    VALUES (new.rowid, new.subject, new.body);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, subject, body)
    VALUES ('delete', old.rowid, old.subject, old.body);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts (memories_fts, rowid, subject, body)
    VALUES ('delete', old.rowid, old.subject, old.body);
    INSERT INTO memories_fts (rowid, subject, body)
    VALUES (new.rowid, new.subject, new.body);
END;

CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id  TEXT PRIMARY KEY REFERENCES memories (id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL
);

-- Provenance. A table, not a log line: every memory points at the episodes and
-- the exact spans that produced it, so any answer can be traced to what was said.
CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id  TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    episode_id TEXT NOT NULL REFERENCES episodes (id) ON DELETE CASCADE,
    quote      TEXT,
    char_start INTEGER,
    char_end   INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (memory_id, episode_id, char_start)
);

CREATE INDEX IF NOT EXISTS idx_memory_sources_episode ON memory_sources (episode_id);

-- ---------------------------------------------------------------------------
-- Graph: a retrieval aid, never a truth store
--
-- Weight is co-occurrence count. Traversal only widens the candidate set for
-- retrieval; nothing is ever asserted to the user from an edge alone, so a wrong
-- edge degrades ranking rather than correctness.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS edges (
    src_type   TEXT NOT NULL,    -- memory | episode
    src_id     TEXT NOT NULL,
    dst_type   TEXT NOT NULL,
    dst_id     TEXT NOT NULL,
    relation   TEXT NOT NULL,    -- works_on | works_with | part_of | related_to | mentioned_with
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (src_type, src_id, dst_type, dst_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_edges_src ON edges (src_id, relation);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges (dst_id, relation);

-- ---------------------------------------------------------------------------
-- Conversation. Distinct from semantic memory: turns are working context for
-- one conversation, memories are durable and cross-session.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    title          TEXT,
    started_at     TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    idx             INTEGER NOT NULL,
    role            TEXT NOT NULL,    -- user | assistant
    content         TEXT NOT NULL,
    tool_calls_json TEXT NOT NULL DEFAULT '[]',
    citations_json  TEXT NOT NULL DEFAULT '[]',
    trace_id        TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (session_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, idx);

-- Kivi asks only in conversation, never during ingest. A resolved clarification
-- is an authoritative memory write.
CREATE TABLE IF NOT EXISTS clarifications (
    id           TEXT PRIMARY KEY,
    session_id   TEXT REFERENCES sessions (id) ON DELETE CASCADE,
    turn_id      TEXT REFERENCES turns (id) ON DELETE SET NULL,
    candidate_id TEXT REFERENCES candidates (id) ON DELETE SET NULL,
    memory_id    TEXT REFERENCES memories (id) ON DELETE SET NULL,
    question     TEXT NOT NULL,
    answer       TEXT,
    status       TEXT NOT NULL DEFAULT 'open',   -- open | resolved | dismissed
    created_at   TEXT NOT NULL,
    resolved_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_clarifications_status ON clarifications (status);

-- ---------------------------------------------------------------------------
-- Traces: one mechanism, two requirements
--
-- Answers "why did memory affect this result, or not", and supplies every
-- number in the evaluation report.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS traces (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,    -- ingest | extract | retrieve | answer
    subject_id      TEXT,             -- episode id, session id, ...
    input_json      TEXT,
    candidates_json TEXT,             -- what was considered, with scores
    decision        TEXT,
    reason          TEXT,
    model           TEXT,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL,             -- NULL when the model price is unknown
    latency_ms      INTEGER,
    cached          INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_kind ON traces (kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_traces_subject ON traces (subject_id);

CREATE TABLE IF NOT EXISTS answers (
    id             TEXT PRIMARY KEY,
    session_id     TEXT REFERENCES sessions (id) ON DELETE CASCADE,
    turn_id        TEXT REFERENCES turns (id) ON DELETE SET NULL,
    query          TEXT NOT NULL,
    answer         TEXT,
    abstained      INTEGER NOT NULL DEFAULT 0,
    abstain_reason TEXT,
    citations_json TEXT NOT NULL DEFAULT '[]',
    trace_id       TEXT REFERENCES traces (id) ON DELETE SET NULL,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_answers_session ON answers (session_id);

-- ---------------------------------------------------------------------------
-- Model cache. Declared here so the schema is complete in one place; also
-- created on demand by kivi/llm/cache.py, hence IF NOT EXISTS on both sides.
-- Deliberately NOT cleared by `kivi reset`, so a reset does not re-bill the
-- corpus. `kivi reset --include-cache` clears it too.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key  TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    kind       TEXT NOT NULL,
    response   TEXT NOT NULL,
    tokens_in  INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_usd   REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embed_cache (
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (content_hash, model, dim)
);

CREATE TABLE IF NOT EXISTS api_call_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    day        TEXT NOT NULL,
    model      TEXT NOT NULL,
    kind       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_api_call_log_day_model ON api_call_log (day, model);
