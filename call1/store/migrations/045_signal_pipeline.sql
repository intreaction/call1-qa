-- Adopt the current signal process on both fresh and upgraded Stores. Historical
-- taxonomy versions, pinned snapshots and results remain immutable. This schema
-- backfill preserves the publication record/version; new snapshots are minted
-- using the changed settings identity after startup.
UPDATE results_signal_taxonomy
SET settings_json = json_set(settings_json, '$.pipeline', 'v2', '$.v1_fallback', json('false'));
