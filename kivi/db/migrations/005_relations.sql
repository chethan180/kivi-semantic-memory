-- Staged relations, before they become graph edges.
--
-- An edge may only connect two entities that actually cleared the gate. But
-- promotion needs two episodes, so at the moment a relation is observed its
-- endpoints are usually still candidates. Writing edges immediately therefore
-- dropped almost all of them: the relation was seen in batch 3, the entities
-- promoted in batch 40, and nothing ever went back to reconsider it.
--
-- Relations are staged here as they are observed and converted to edges in a
-- final pass, once every entity that is going to promote has done so.

CREATE TABLE IF NOT EXISTS relations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    src_norm   TEXT NOT NULL,     -- normalised entity name, matches promotion
    dst_norm   TEXT NOT NULL,
    src_raw    TEXT NOT NULL,
    dst_raw    TEXT NOT NULL,
    relation   TEXT NOT NULL,
    episode_id TEXT REFERENCES episodes (id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    UNIQUE (src_norm, dst_norm, relation, episode_id)
);

CREATE INDEX IF NOT EXISTS idx_relations_src ON relations (src_norm);
CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations (dst_norm);
