-- Things the person told Kivi directly, and provenance that can point at them.
--
-- Episodes are a record of what was actually dictated. Typing an instruction
-- into a chat box is not a dictation, and writing one into `episodes` to make it
-- citable would corrupt the one table whose meaning depends on being untouched.
-- So statements get their own table, and provenance becomes polymorphic.
--
-- The `say_` id prefix is load-bearing: the interface already renders citations
-- beginning with it as "you told Kivi" rather than as a record id, so a fact the
-- person asserted reads differently from one Kivi inferred.

CREATE TABLE IF NOT EXISTS statements (
    id         TEXT PRIMARY KEY,          -- say_<hash>
    session_id TEXT REFERENCES sessions (id) ON DELETE SET NULL,
    turn_id    TEXT REFERENCES turns (id) ON DELETE SET NULL,
    text       TEXT NOT NULL,             -- what the person actually said
    intent     TEXT NOT NULL,             -- add | revise | forget | confirm
    ts         TEXT NOT NULL,
    ts_epoch   INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_statements_session ON statements (session_id);
CREATE INDEX IF NOT EXISTS idx_statements_ts ON statements (ts_epoch DESC);

-- ---------------------------------------------------------------------------
-- Polymorphic provenance
--
-- A memory may now be grounded in a dictation or in something the person told
-- Kivi. Rebuilt rather than extended because `episode_id` was NOT NULL and part
-- of the primary key; existing rows migrate as source_kind='episode'.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory_sources_new (
    memory_id   TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL,            -- 'episode' | 'statement'
    source_id   TEXT NOT NULL,
    quote       TEXT,
    char_start  INTEGER DEFAULT 0,
    char_end    INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (memory_id, source_kind, source_id, char_start)
);

INSERT OR IGNORE INTO memory_sources_new
    (memory_id, source_kind, source_id, quote, char_start, char_end, created_at)
SELECT memory_id, 'episode', episode_id, quote,
       COALESCE(char_start, 0), COALESCE(char_end, 0), created_at
  FROM memory_sources;

DROP TABLE memory_sources;
ALTER TABLE memory_sources_new RENAME TO memory_sources;

CREATE INDEX IF NOT EXISTS idx_memory_sources_source
    ON memory_sources (source_kind, source_id);
CREATE INDEX IF NOT EXISTS idx_memory_sources_memory
    ON memory_sources (memory_id);

-- ---------------------------------------------------------------------------
-- Episodes are immutable
--
-- Stated everywhere in the design; now enforced. Memory is allowed to be wrong
-- and be corrected. What the person said is not up for revision, and a product
-- that quietly edits its own record of your speech has nothing left to be
-- trusted about. Status changes belong on `memories`, never here.
--
-- Ingest metadata stays writable so backfills remain possible; the spoken
-- content and its timestamp do not.
-- ---------------------------------------------------------------------------

-- Note: the message must be a single string literal. SQLite has no implicit
-- string concatenation, so splitting it across lines is a syntax error.
CREATE TRIGGER IF NOT EXISTS episodes_are_immutable
BEFORE UPDATE OF raw_asr, formatted, ts, ts_epoch, local_hour, local_date
ON episodes
BEGIN
    SELECT RAISE(ABORT, 'episodes are immutable: what was said cannot be rewritten - correct the memory derived from it instead');
END;
