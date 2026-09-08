-- How many messages a recipient's style had been read at, last time.
--
-- Personalization now runs inside ingestion, and the counted features are free
-- to recompute every batch. Having the model re-read someone's messages is not,
-- so it happens only when their message count crosses a checkpoint (2, 4, 8, 16,
-- 32, 64, 128). Without this marker the re-read would fire on every batch, which
-- is both expensive and pointless: a person's style does not change because
-- three more messages arrived.
--
-- The checkpoints double as the learning curve the evaluation reports, so the
-- cost control and the measurement are the same mechanism.

ALTER TABLE recipient_styles ADD COLUMN last_read_at_n INTEGER NOT NULL DEFAULT 0;
