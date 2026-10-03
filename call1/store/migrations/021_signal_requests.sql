-- 021_signal_requests: Contact Signals v2 in the queue area (contract 1.3.0; docs/ContactSignalsV2.md
-- sections 7.5 and 7.6). Previews and compares, backfills, and the reanalysis-request additions:
-- priority (claims order by priority DESC, requested_at, id), rescore_signals, the resolved taxonomy
-- version and pipeline, and the preview links. Taxonomy versions are read through results/api.py.

-- A preview (the editor's "Test on recent calls") or a compare (a compare backfill, or a shadow-mode
-- companion). Its calls are the contact_signals_preview requests naming it (signal_preview_id).
CREATE TABLE q_signal_previews (
    id                    TEXT PRIMARY KEY,
    source                TEXT NOT NULL CHECK (source IN ('preview', 'compare')),
    taxonomy_version      INTEGER,
    taxonomy_digest       TEXT NOT NULL,
    options_trimmed_json  TEXT NOT NULL DEFAULT '[]',
    created_at            TEXT NOT NULL,
    created_by_account_id TEXT,
    idempotency_scope     TEXT,
    idempotency_key       TEXT,
    request_digest        TEXT,
    UNIQUE (idempotency_scope, idempotency_key)
);

CREATE TABLE q_signal_backfills (
    id                    TEXT PRIMARY KEY,
    mode                  TEXT NOT NULL CHECK (mode IN ('rescore', 'compare')),
    rescore_signals       INTEGER NOT NULL DEFAULT 0,
    created_after         TEXT NOT NULL,
    created_before        TEXT,
    max_calls             INTEGER NOT NULL,
    taxonomy_version      INTEGER NOT NULL,
    calls_matched         INTEGER NOT NULL,
    requests_created      INTEGER NOT NULL,
    calls_skipped         INTEGER NOT NULL,
    preview_id            TEXT,
    created_at            TEXT NOT NULL,
    created_by_account_id TEXT NOT NULL,
    idempotency_scope     TEXT NOT NULL,
    idempotency_key       TEXT NOT NULL,
    request_digest        TEXT NOT NULL,
    UNIQUE (idempotency_scope, idempotency_key)
);

ALTER TABLE q_reanalysis_requests ADD COLUMN priority INTEGER NOT NULL DEFAULT 0;
ALTER TABLE q_reanalysis_requests ADD COLUMN rescore_signals INTEGER NOT NULL DEFAULT 0;
ALTER TABLE q_reanalysis_requests ADD COLUMN signal_taxonomy_version INTEGER;
ALTER TABLE q_reanalysis_requests ADD COLUMN signal_pipeline TEXT;
ALTER TABLE q_reanalysis_requests ADD COLUMN signal_backfill_id TEXT;
ALTER TABLE q_reanalysis_requests ADD COLUMN signal_preview_id TEXT;
ALTER TABLE q_reanalysis_requests ADD COLUMN signal_taxonomy_snapshot_artifact_id TEXT;
ALTER TABLE q_reanalysis_requests ADD COLUMN preview_result_artifact_id TEXT;
CREATE INDEX q_reanalysis_claim_order ON q_reanalysis_requests (status, priority DESC, requested_at, id);
CREATE INDEX q_reanalysis_preview ON q_reanalysis_requests (signal_preview_id);
