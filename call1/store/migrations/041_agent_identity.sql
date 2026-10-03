-- 041_agent_identity: contract 1.1.0 agent identity on the results area's read projections.
--
-- CallMetadata gained agent_display_name and agent_extension. The call record and the review-queue
-- items carry them next to agent_id, as they already carry agent_id, so the call list and detail,
-- the escalation list and the review queue can show "Name (ext)" (calls.agent_label). Existing rows
-- get NULL, which is exact: no 1.0.x registration could carry either field (CallMetadata forbids
-- unknown fields), so there is nothing to backfill. A later re-registration that names them fills
-- them in (reads.register_conversation -> projections.on_call_metadata_updated).
ALTER TABLE results_calls ADD COLUMN agent_display_name TEXT;
ALTER TABLE results_calls ADD COLUMN agent_extension TEXT;
ALTER TABLE results_review_items ADD COLUMN agent_display_name TEXT;
ALTER TABLE results_review_items ADD COLUMN agent_extension TEXT;
