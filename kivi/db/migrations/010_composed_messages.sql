-- Composed messages leave `episodes` and get a table of their own.
--
-- They were stored as episodes with source='composed', on the reasoning that a
-- message which would have been sent belongs in the record of what this person
-- wrote. The record is worth keeping; putting it in `episodes` was wrong.
--
-- `episodes` means "what this person actually dictated". A composed message is
-- what KIVI wrote, in a voice Kivi predicted, and mixing the two makes every
-- query that reads episodes responsible for remembering to exclude it. Style
-- learning already carried `AND source != 'composed'` in four separate places
-- for exactly this reason - a filter that has to be repeated is a filter that
-- will eventually be forgotten, and forgetting it means Kivi learning this
-- person's voice from its own output.
--
-- A separate table makes the exclusion structural instead. Search, style
-- learning, extraction and the evaluation read `episodes` and cannot see these
-- rows however they are written.

CREATE TABLE IF NOT EXISTS composed_messages (
    id          TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    ts_epoch    INTEGER NOT NULL,
    app         TEXT,
    recipient   TEXT,
    recipient_norm TEXT,
    text        TEXT NOT NULL,
    session_id  TEXT,
    style_rules_json TEXT,
    style_samples    INTEGER NOT NULL DEFAULT 0,
    tz_offset_min    INTEGER NOT NULL DEFAULT 0,
    local_hour  INTEGER,
    local_date  TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS composed_recipient
    ON composed_messages (recipient_norm, ts_epoch);

-- Move anything already written under the old scheme, then take it out of
-- episodes. The AFTER DELETE trigger on episodes keeps the FTS index in step;
-- the vector rows have to be removed by hand because episode_vectors is an
-- ordinary table with no cascade.
INSERT OR IGNORE INTO composed_messages
    (id, ts, ts_epoch, app, recipient, recipient_norm, text, tz_offset_min,
     local_hour, local_date, created_at)
SELECT id, ts, ts_epoch, app, recipient, recipient_norm, formatted,
       tz_offset_min, local_hour, local_date, created_at
FROM episodes WHERE source = 'composed';

DELETE FROM episode_vectors
WHERE episode_id IN (SELECT id FROM episodes WHERE source = 'composed');

DELETE FROM episodes WHERE source = 'composed';
