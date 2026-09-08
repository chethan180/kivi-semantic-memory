-- Who a dictation was addressed to, and how this person writes to them.
--
-- The product claim this supports: you do not write one way, you write a
-- different way to each person. "Hi Amma" to your mother, clipped and punctual
-- to your manager, shorthand to a close colleague. That difference is stable,
-- it is visible in the dictations, and reproducing it is the clearest thing
-- semantic memory can actually DO for a writing product - as opposed to merely
-- knowing things.
--
-- Style is DERIVED, never authored. Every field below is recomputable from the
-- episodes, so this table is a materialised view over evidence rather than a
-- second store that can drift out of agreement with what was actually said.

ALTER TABLE episodes ADD COLUMN recipient TEXT;
ALTER TABLE episodes ADD COLUMN recipient_norm TEXT;

CREATE INDEX IF NOT EXISTS idx_episodes_recipient ON episodes (recipient_norm);

CREATE TABLE IF NOT EXISTS recipient_styles (
    recipient_norm   TEXT PRIMARY KEY,
    display          TEXT NOT NULL,
    -- boss | colleague | friend | family | unknown. Learned from how they are
    -- written to and about, not declared in configuration.
    relation         TEXT NOT NULL DEFAULT 'unknown',
    relation_conf    REAL NOT NULL DEFAULT 0.0,
    n_samples        INTEGER NOT NULL DEFAULT 0,

    -- The observable habits. Each is a rate or an average over n_samples, so a
    -- profile built on two dictations is visibly weaker than one built on forty
    -- and the interface can say so instead of pretending otherwise.
    greeting         TEXT,      -- the actual opener used most often, e.g. "Hi Amma"
    greeting_rate    REAL NOT NULL DEFAULT 0.0,
    signoff          TEXT,
    signoff_rate     REAL NOT NULL DEFAULT 0.0,
    mean_words       REAL NOT NULL DEFAULT 0.0,
    mean_sentence    REAL NOT NULL DEFAULT 0.0,
    contraction_rate REAL NOT NULL DEFAULT 0.0,
    exclamation_rate REAL NOT NULL DEFAULT 0.0,
    question_rate    REAL NOT NULL DEFAULT 0.0,
    formality        REAL NOT NULL DEFAULT 0.0,   -- 0 casual .. 1 formal

    -- Set once n_samples clears the threshold; below it the profile exists but
    -- is not applied, so a single message never becomes "how you write to them".
    active           INTEGER NOT NULL DEFAULT 0,
    memory_id        TEXT REFERENCES memories (id) ON DELETE SET NULL,
    updated_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recipient_styles_relation
    ON recipient_styles (relation, active);

-- Which episodes a profile was built from. Provenance for a *behaviour* rather
-- than for a fact: "I greeted her that way because you did, in these fourteen
-- messages" is checkable, and a style applied without that is just a guess the
-- person cannot audit.
CREATE TABLE IF NOT EXISTS style_evidence (
    recipient_norm TEXT NOT NULL,
    episode_id     TEXT NOT NULL REFERENCES episodes (id) ON DELETE CASCADE,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (recipient_norm, episode_id)
);

CREATE INDEX IF NOT EXISTS idx_style_evidence_recipient
    ON style_evidence (recipient_norm);
