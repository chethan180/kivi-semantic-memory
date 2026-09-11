-- A question Kivi asks in conversation before overwriting what it believes.
--
-- "Change the manager from Priya to Ashwin", with Ashwin already recorded as a
-- frontend developer, is a real conflict, so Kivi asks. But the question used to
-- live only inside the turn that asked it: the proposed change was stored
-- nowhere, so the person's "yes" on the next turn had nothing to agree to and the
-- change was lost. It is stored now, with the exact write to make if they agree.
--
-- `kind` keeps these apart from the dictation-versus-instruction conflicts raised
-- during ingest, which surface whenever their subject comes up. A confirmation
-- belongs to one conversation and one turn, and must never be presented as "a
-- dictation has contradicted you".

ALTER TABLE clarifications ADD COLUMN kind TEXT NOT NULL DEFAULT 'dictation';
ALTER TABLE clarifications ADD COLUMN proposed_json TEXT;
