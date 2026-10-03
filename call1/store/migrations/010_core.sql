-- 010_core: owned by the Store core (call1/store/*.py outside the area packages).
-- Store metadata, the change feed, the audit log and object-upload sessions.

CREATE TABLE store_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Change feed (events.ChangeEvent). seq is assigned inside the writing transaction; SQLite
-- serializes writers, so seq order is commit order. Cursors are "<feed_epoch>-<seq>".
CREATE TABLE change_events (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at     TEXT NOT NULL,
    kind            TEXT NOT NULL,
    resource_id     TEXT NOT NULL,
    conversation_id TEXT,
    call_id         TEXT,
    version         INTEGER,
    status          TEXT
);
CREATE INDEX change_events_kind_seq ON change_events (kind, seq);
CREATE INDEX change_events_occurred ON change_events (occurred_at);

-- Audit log (events.AuditEvent), hash-chained: event_digest = canonical_digest(body), and each
-- body carries the previous event's digest.
CREATE TABLE audit_events (
    sequence              INTEGER PRIMARY KEY,
    id                    TEXT NOT NULL UNIQUE,
    occurred_at           TEXT NOT NULL,
    actor_json            TEXT NOT NULL,
    actor_kind            TEXT NOT NULL,
    action                TEXT NOT NULL,
    target_kind           TEXT NOT NULL,
    target_id             TEXT NOT NULL,
    details_json          TEXT NOT NULL,
    previous_event_digest TEXT,
    event_digest          TEXT NOT NULL
);
CREATE INDEX audit_events_action ON audit_events (action, sequence);
CREATE INDEX audit_events_target ON audit_events (target_kind, target_id, sequence);
CREATE INDEX audit_events_occurred ON audit_events (occurred_at);

-- Object-upload sessions behind artifacts.UploadGrant. The bytes land in objects/staging, are
-- verified at commit and moved into the content-addressed store. The area that created the grant
-- keeps its own descriptor in metadata_json (for example the reserved artifact ID).
CREATE TABLE object_uploads (
    upload_id         TEXT PRIMARY KEY,
    token_hash        TEXT NOT NULL,
    purpose           TEXT NOT NULL,
    required_scope    TEXT,
    installation_id   TEXT,
    expected_checksum TEXT NOT NULL,
    expected_size     INTEGER NOT NULL,
    content_type      TEXT NOT NULL,
    metadata_json     TEXT NOT NULL DEFAULT '{}',
    status            TEXT NOT NULL CHECK (status IN ('pending', 'received', 'committed')),
    received_checksum TEXT,
    received_size     INTEGER,
    created_at        TEXT NOT NULL,
    expires_at        TEXT NOT NULL,
    committed_at      TEXT
);
CREATE INDEX object_uploads_expires ON object_uploads (status, expires_at);
