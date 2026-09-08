-- Every spelling seen for a candidate.
--
-- Entities are matched on a normalised form, but the surface forms are what the
-- person actually said and are worth keeping: they become the memory's aliases,
-- so a later question phrased "the Beacon project" matches an entity stored as
-- "Beacon", and the interface can show which wording produced the memory.

CREATE TABLE IF NOT EXISTS candidate_surface (
    candidate_id TEXT NOT NULL,
    surface      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (candidate_id, surface)
);

CREATE INDEX IF NOT EXISTS idx_candidate_surface_candidate
    ON candidate_surface (candidate_id);
