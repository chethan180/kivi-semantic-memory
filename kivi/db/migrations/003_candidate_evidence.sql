-- Evidence for a candidate, one row per supporting episode.
--
-- Promotion thresholds are counted in DISTINCT EPISODES, not mentions: one
-- dictation that says "DSPM, the DSPM project, DSPM again" is a single piece of
-- evidence, and a counter incremented per mention would promote it immediately.
-- This table is also what supplies provenance the moment a candidate is
-- promoted - the episodes are already recorded, so memory_sources is a copy
-- rather than a re-derivation.

CREATE TABLE IF NOT EXISTS candidate_evidence (
    candidate_id TEXT NOT NULL,
    episode_id   TEXT NOT NULL REFERENCES episodes (id) ON DELETE CASCADE,
    quote        TEXT,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (candidate_id, episode_id)
);

CREATE INDEX IF NOT EXISTS idx_candidate_evidence_candidate
    ON candidate_evidence (candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_evidence_episode
    ON candidate_evidence (episode_id);
