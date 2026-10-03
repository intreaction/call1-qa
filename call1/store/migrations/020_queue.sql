-- 020_queue: owned by the queue area (call1/store/queue/). Numbers 020-029 belong to the queue area.
--
-- Conversations, artifacts (per-slot versions, linking, supersession, orphans), job graphs, jobs,
-- dependency edges, attempts (claims), completion/failure/release receipts keyed by completion
-- key, usage rows (one per attempt), reanalysis requests and their claims, hardware profiles and
-- catalog snapshots. This is the processing queue; the human review queue lives in the results
-- area and shares nothing with these tables. Other areas reference these rows by ID only.

CREATE TABLE q_conversations (
    id                            TEXT PRIMARY KEY,
    dedup_identity                TEXT NOT NULL UNIQUE,
    ingestion_kind                TEXT NOT NULL,
    source_json                   TEXT NOT NULL,
    call_id                       TEXT UNIQUE,
    call_metadata_json            TEXT,
    created_at                    TEXT NOT NULL,
    registered_by_installation_id TEXT
);

-- A graph request, scoped by (installation, conversation, idempotency_key). A draft-test graph
-- names the qa_draft_test request it fulfils.
CREATE TABLE q_graphs (
    id                    TEXT PRIMARY KEY,
    conversation_id       TEXT NOT NULL REFERENCES q_conversations(id),
    installation_id       TEXT NOT NULL,
    idempotency_key       TEXT NOT NULL,
    request_digest        TEXT NOT NULL,
    reason                TEXT NOT NULL,
    reanalysis_request_id TEXT,
    draft_test_request_id TEXT,
    created_at            TEXT NOT NULL,
    UNIQUE (installation_id, conversation_id, idempotency_key)
);
CREATE INDEX q_graphs_conversation ON q_graphs (conversation_id, created_at, id);

CREATE TABLE q_jobs (
    id                        TEXT PRIMARY KEY,
    conversation_id           TEXT NOT NULL REFERENCES q_conversations(id),
    graph_id                  TEXT NOT NULL REFERENCES q_graphs(id),
    ref                       TEXT NOT NULL,
    installation_id           TEXT NOT NULL,
    idempotency_key           TEXT NOT NULL,
    definition_digest         TEXT NOT NULL,
    job_type                  TEXT NOT NULL,
    execution_class           TEXT NOT NULL,
    status                    TEXT NOT NULL CHECK (status IN ('BLOCKED', 'QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELLED', 'WAITING_PROVIDER')),
    priority                  INTEGER NOT NULL,
    attempt_count             INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    claim_count               INTEGER NOT NULL DEFAULT 0 CHECK (claim_count >= attempt_count),
    max_attempts              INTEGER NOT NULL CHECK (max_attempts >= 1),
    retry_generation          INTEGER NOT NULL DEFAULT 0,
    next_run_at               TEXT,
    wait_cause                TEXT,
    queued_at                 TEXT,
    cancel_requested          INTEGER NOT NULL DEFAULT 0,
    error_code                TEXT,
    error_detail              TEXT,
    inputs_json               TEXT NOT NULL,
    resolved_inputs_json      TEXT NOT NULL DEFAULT '[]',
    selection_json            TEXT,
    resource_estimate_json    TEXT NOT NULL,
    parameters_json           TEXT NOT NULL,
    memory_slot               TEXT NOT NULL,
    outbound_connection_ref   TEXT,
    route_class               TEXT,
    catalog_entry_id          TEXT,
    catalog_entry_version     INTEGER,
    outputs_json              TEXT NOT NULL DEFAULT '[]',
    result_json               TEXT,
    result_version            INTEGER,
    lease_worker_id           TEXT,
    lease_installation_id     TEXT,
    lease_attempt_number      INTEGER,
    lease_granted_at          TEXT,
    lease_expires_at          TEXT,
    lease_seconds             INTEGER,
    lease_claim_token_hash    TEXT,
    draft_test_request_id     TEXT,
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL,
    completed_at              TEXT,
    UNIQUE (installation_id, conversation_id, idempotency_key),
    CHECK ((status = 'RUNNING') = (lease_claim_token_hash IS NOT NULL))
);
CREATE INDEX q_jobs_ready ON q_jobs (status, priority DESC, created_at, id);
CREATE INDEX q_jobs_conversation ON q_jobs (conversation_id, created_at, id);
CREATE INDEX q_jobs_graph ON q_jobs (graph_id);
CREATE INDEX q_jobs_lease ON q_jobs (status, lease_expires_at);
CREATE UNIQUE INDEX q_jobs_active_claim ON q_jobs (lease_claim_token_hash) WHERE lease_claim_token_hash IS NOT NULL;

-- Which jobs a graph lists (JobGraph.jobs): the jobs it created, jobs an idempotent job key
-- resolved to, and follow-on jobs created by its completions.
CREATE TABLE q_graph_jobs (
    graph_id TEXT NOT NULL REFERENCES q_graphs(id),
    position INTEGER NOT NULL,
    ref      TEXT NOT NULL,
    job_id   TEXT NOT NULL REFERENCES q_jobs(id),
    PRIMARY KEY (graph_id, position)
);
CREATE INDEX q_graph_jobs_job ON q_graph_jobs (job_id);

-- Dependency edges: 'requires' (upstream must SUCCEED) and 'after' (upstream must be terminal).
CREATE TABLE q_edges (
    job_id          TEXT NOT NULL REFERENCES q_jobs(id),
    upstream_job_id TEXT NOT NULL REFERENCES q_jobs(id),
    kind            TEXT NOT NULL CHECK (kind IN ('requires', 'after')),
    PRIMARY KEY (job_id, upstream_job_id)
);
CREATE INDEX q_edges_upstream ON q_edges (upstream_job_id);

-- Every claim of a job (jobs.Attempt). attempt_number is never reused; a released claim does not
-- count as an attempt and has no usage row.
CREATE TABLE q_attempts (
    job_id                 TEXT NOT NULL REFERENCES q_jobs(id),
    attempt_number         INTEGER NOT NULL CHECK (attempt_number >= 1),
    status                 TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled', 'lease_expired', 'released')),
    counts_as_attempt      INTEGER NOT NULL,
    started_at             TEXT NOT NULL,
    ended_at               TEXT,
    worker_id              TEXT NOT NULL,
    installation_id        TEXT NOT NULL,
    hardware_profile_id    TEXT NOT NULL,
    claim_token_hash       TEXT NOT NULL UNIQUE,
    slot_offer_index       INTEGER NOT NULL,
    lease_granted_at       TEXT NOT NULL,
    lease_expires_at       TEXT NOT NULL,
    queue_wait_seconds     REAL,
    error_code             TEXT,
    error_detail           TEXT,
    provenance_json        TEXT,
    usage_record_id        TEXT,
    resource_estimate_json TEXT NOT NULL,
    PRIMARY KEY (job_id, attempt_number)
);

-- Complete, fail and release share one completion-key space per job; one call ends a claim.
CREATE TABLE q_receipts (
    job_id         TEXT NOT NULL REFERENCES q_jobs(id),
    completion_key TEXT NOT NULL,
    receipt_id     TEXT NOT NULL UNIQUE,
    operation      TEXT NOT NULL CHECK (operation IN ('complete', 'fail', 'release')),
    request_digest TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    receipt_json   TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (job_id, completion_key)
);

CREATE TABLE q_artifacts (
    id                       TEXT PRIMARY KEY,
    conversation_id          TEXT REFERENCES q_conversations(id),
    kind                     TEXT NOT NULL,
    slot                     TEXT NOT NULL DEFAULT '',
    content_type             TEXT NOT NULL,
    size_bytes               INTEGER NOT NULL,
    checksum                 TEXT NOT NULL,
    content_contract         TEXT NOT NULL,
    sensitivity              TEXT NOT NULL,
    producing_job_id         TEXT REFERENCES q_jobs(id),
    producing_attempt_number INTEGER,
    labels_json              TEXT NOT NULL DEFAULT '{}',
    linked                   INTEGER NOT NULL DEFAULT 0,
    linked_by_receipt_id     TEXT,
    version                  INTEGER,
    storage                  TEXT NOT NULL CHECK (storage IN ('inline', 'object')),
    committed_at             TEXT NOT NULL,
    superseded_by            TEXT,
    upload_id                TEXT,
    CHECK ((linked = 1) = (version IS NOT NULL))
);
CREATE UNIQUE INDEX q_artifacts_slot_version ON q_artifacts (conversation_id, kind, slot, version) WHERE version IS NOT NULL;
CREATE INDEX q_artifacts_conversation ON q_artifacts (conversation_id, committed_at, id);
CREATE INDEX q_artifacts_natural ON q_artifacts (conversation_id, kind, slot, checksum);
CREATE INDEX q_artifacts_producer ON q_artifacts (producing_job_id, producing_attempt_number);
CREATE INDEX q_artifacts_unlinked ON q_artifacts (linked, committed_at);
CREATE INDEX q_artifacts_checksum ON q_artifacts (checksum);

-- One usage row per attempt (usage.UsageRecord as JSON plus the columns reports filter on).
CREATE TABLE q_usage_records (
    id                TEXT PRIMARY KEY,
    job_id            TEXT NOT NULL REFERENCES q_jobs(id),
    attempt_number    INTEGER NOT NULL,
    conversation_id   TEXT NOT NULL,
    recorded_at       TEXT NOT NULL,
    route_class       TEXT,
    purpose           TEXT,
    outcome           TEXT NOT NULL,
    recorded_by       TEXT NOT NULL,
    late_input_digest TEXT,
    record_json       TEXT NOT NULL,
    UNIQUE (job_id, attempt_number)
);
CREATE INDEX q_usage_time ON q_usage_records (recorded_at, id);
CREATE INDEX q_usage_conversation ON q_usage_records (conversation_id, recorded_at, id);

CREATE TABLE q_reanalysis_requests (
    id                         TEXT PRIMARY KEY,
    conversation_id            TEXT NOT NULL REFERENCES q_conversations(id),
    call_id                    TEXT NOT NULL,
    kind                       TEXT NOT NULL,
    status                     TEXT NOT NULL CHECK (status IN ('pending', 'claimed', 'fulfilled', 'rejected')),
    rubric_json                TEXT,
    draft_rubric_json          TEXT,
    speaker_correction_json    TEXT,
    note                       TEXT,
    requested_by_account_id    TEXT,
    requested_at               TEXT NOT NULL,
    idempotency_scope          TEXT NOT NULL,
    idempotency_key            TEXT NOT NULL,
    request_digest             TEXT NOT NULL,
    claimed_by_installation_id TEXT,
    claim_worker_id            TEXT,
    claim_token_hash           TEXT,
    claim_expires_at           TEXT,
    graph_id                   TEXT,
    draft_result_artifact_id   TEXT,
    rejected_reason            TEXT,
    updated_at                 TEXT NOT NULL,
    UNIQUE (idempotency_scope, idempotency_key)
);
CREATE INDEX q_reanalysis_status ON q_reanalysis_requests (status, requested_at, id);
CREATE INDEX q_reanalysis_call ON q_reanalysis_requests (call_id, requested_at, id);
CREATE INDEX q_reanalysis_conversation ON q_reanalysis_requests (conversation_id, status);

CREATE TABLE q_hardware_profiles (
    id            TEXT PRIMARY KEY,
    fingerprint   TEXT NOT NULL UNIQUE,
    fields_json   TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL
);

-- One read-only catalog snapshot per Process installation, replaced whole on each publish.
CREATE TABLE q_catalog_snapshots (
    installation_id TEXT PRIMARY KEY,
    snapshot_json   TEXT NOT NULL,
    digest          TEXT NOT NULL,
    published_at    TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
