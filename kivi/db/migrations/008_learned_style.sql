-- The model's reading of how this person writes, alongside the counted features.
--
-- Both are kept deliberately. The counted columns in 007 are what the evaluation
-- scores against and what lets the interface say "in 12 of 14 messages"; these
-- are what actually drive drafting, because a rule written by something that
-- read the messages catches idiom and register that no hand-written detector
-- was ever going to find.

ALTER TABLE recipient_styles ADD COLUMN summary TEXT;
ALTER TABLE recipient_styles ADD COLUMN distinctive TEXT;
ALTER TABLE recipient_styles ADD COLUMN rules_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE recipient_styles ADD COLUMN relation_reason TEXT;
