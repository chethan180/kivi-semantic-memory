-- Preserve the speaker's local wall-clock time.
--
-- 001 stored timestamps normalised to UTC, which loses the offset they were
-- spoken in. "Find the dictation I did around 5 PM yesterday" - the assignment's
-- own worked example - means 5 PM where the person was standing, so the local
-- hour has to survive ingest. UTC stays the ordering key; these are for filters
-- a human would recognise.

ALTER TABLE episodes ADD COLUMN tz_offset_min INTEGER NOT NULL DEFAULT 0;
ALTER TABLE episodes ADD COLUMN local_hour INTEGER;
ALTER TABLE episodes ADD COLUMN local_date TEXT;

CREATE INDEX IF NOT EXISTS idx_episodes_local_hour ON episodes (local_hour);
CREATE INDEX IF NOT EXISTS idx_episodes_local_date ON episodes (local_date);
