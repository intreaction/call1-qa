-- 042_signals: Contact Signals v2 in the results area (contract 1.3.0; docs/ContactSignalsV2.md section 7.6).
--
-- The signal taxonomy (one document, versioned whole: a singleton record plus immutable versions),
-- the settings, alert rules, reviewer hit feedback, and the projections of each call's published
-- contact_signals versions (outcomes, hits, hit fields). The contact_signals artifact is the source
-- of truth; the projection tables hold IDs, numbers, enum and boolean values only, never text
-- (quotes, string/number/amount/date field values stay in the masked artifact and pass the Masker).

-- The singleton record: the current version, the optimistic-concurrency token shared by taxonomy
-- and settings saves, and the settings (SignalSettings JSON).
CREATE TABLE results_signal_taxonomy (
    id                    INTEGER PRIMARY KEY CHECK (id = 1),
    current_version       INTEGER NOT NULL,
    record_version        INTEGER NOT NULL,
    settings_json         TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    updated_by_account_id TEXT
);

-- Immutable published versions. Only the text tombstone (redactSignalTaxonomyText) rewrites
-- taxonomy_json, and it keeps the digest.
CREATE TABLE results_signal_taxonomy_versions (
    version                 INTEGER PRIMARY KEY,
    digest                  TEXT NOT NULL,
    taxonomy_json           TEXT NOT NULL,
    published_at            TEXT NOT NULL,
    published_by_account_id TEXT,
    notes                   TEXT,
    text_redacted           INTEGER NOT NULL DEFAULT 0,
    redacted_at             TEXT,
    redacted_by_account_id  TEXT
);

CREATE TABLE results_signal_alert_rules (
    rule_id               TEXT PRIMARY KEY,
    rule_json             TEXT NOT NULL,
    enabled               INTEGER NOT NULL,
    record_version        INTEGER NOT NULL,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    updated_by_account_id TEXT NOT NULL
);

-- One row per projected contact_signals version: the metrics denominators.
CREATE TABLE results_signal_outcomes (
    call_id          TEXT NOT NULL,
    signals_version  INTEGER NOT NULL,
    pipeline         TEXT NOT NULL,
    taxonomy_version INTEGER,
    completeness     TEXT NOT NULL,
    hit_count        INTEGER NOT NULL,
    projected_at     TEXT NOT NULL,
    PRIMARY KEY (call_id, signals_version)
);

-- One row per hit (v1 hits project with category_id = kind and no subcategory). No quote.
CREATE TABLE results_signal_hits (
    call_id            TEXT NOT NULL,
    signals_version    INTEGER NOT NULL,
    hit_id             TEXT NOT NULL,
    category_id        TEXT NOT NULL,
    subcategory_id     TEXT,
    turn_id            INTEGER,
    speaker            TEXT NOT NULL,
    start              REAL NOT NULL,
    "end"              REAL NOT NULL,
    confidence         REAL NOT NULL,
    category_digest    TEXT,
    subcategory_digest TEXT,
    PRIMARY KEY (call_id, signals_version, hit_id)
);
CREATE INDEX results_signal_hits_by_node ON results_signal_hits (category_id, subcategory_id, call_id);
CREATE INDEX results_signal_hits_by_call ON results_signal_hits (call_id, signals_version);

-- Stage-3 fields per hit: status, and the value only for enum and boolean fields (the admin's own
-- enum list). Other types store presence only.
CREATE TABLE results_signal_hit_fields (
    call_id         TEXT NOT NULL,
    signals_version INTEGER NOT NULL,
    hit_id          TEXT NOT NULL,
    field_id        TEXT NOT NULL,
    field_type      TEXT NOT NULL,
    status          TEXT NOT NULL,
    value_enum      TEXT,
    value_bool      INTEGER,
    PRIMARY KEY (call_id, signals_version, hit_id, field_id)
);

-- Reviewer verdicts, keyed on the stable hit ID so they survive threshold edits and republishes.
CREATE TABLE results_signal_feedback (
    call_id                  TEXT NOT NULL,
    hit_id                   TEXT NOT NULL,
    category_verdict         TEXT,
    subcategory_id           TEXT,
    subcategory_digest       TEXT,
    subcategory_verdict      TEXT,
    corrected_subcategory_id TEXT,
    note                     TEXT,
    account_id               TEXT NOT NULL,
    feedback_version         INTEGER NOT NULL,
    updated_at               TEXT NOT NULL,
    PRIMARY KEY (call_id, hit_id)
);

ALTER TABLE results_calls ADD COLUMN signals_version INTEGER;
ALTER TABLE results_review_items ADD COLUMN signals_version INTEGER;
ALTER TABLE results_review_items ADD COLUMN trigger_alert_rule_ids_json TEXT NOT NULL DEFAULT '[]';

-- Install-time version 1: the eight built-in categories only (signals.builtin_signal_taxonomy();
-- tests/store/test_signals_taxonomy.py checks the JSON and digest). Settings are the defaults (v1).
INSERT INTO results_signal_taxonomy_versions (version, digest, taxonomy_json, published_at, published_by_account_id, notes) VALUES (1, 'sha256:1e4c75cca6816b99dd2a684089381b0ef5b07debde6f2085f5a2b76403bf5090', '{"categories":[{"active":true,"builtin":true,"category_id":"intent","description":null,"examples":[],"fields":[],"gloss":"Caller says what they want or why they called","name":"Caller objective","narrow_quote":false,"speaker":"CALLER","subcategories":[],"subcategory_threshold":null,"threshold":null},{"active":true,"builtin":true,"category_id":"issue","description":null,"examples":[],"fields":[],"gloss":"Caller describes the problem, fee or dispute","name":"Reported issue","narrow_quote":false,"speaker":"CALLER","subcategories":[],"subcategory_threshold":null,"threshold":null},{"active":true,"builtin":true,"category_id":"friction","description":null,"examples":[],"fields":[],"gloss":"Caller describes an obstacle, repeat failure or frustration","name":"Friction point","narrow_quote":false,"speaker":"CALLER","subcategories":[],"subcategory_threshold":null,"threshold":null},{"active":true,"builtin":true,"category_id":"fix_proposed","description":null,"examples":[],"fields":[],"gloss":"Agent proposes a fix or workaround","name":"Proposed fix","narrow_quote":false,"speaker":"AGENT","subcategories":[],"subcategory_threshold":null,"threshold":null},{"active":true,"builtin":true,"category_id":"agent_reports_completed","description":null,"examples":[],"fields":[],"gloss":"Agent reports that an action is done","name":"Agent completed","narrow_quote":false,"speaker":"AGENT","subcategories":[],"subcategory_threshold":null,"threshold":null},{"active":true,"builtin":true,"category_id":"caller_confirms_resolved","description":null,"examples":[],"fields":[],"gloss":"Caller confirms the problem is solved","name":"Caller confirmed","narrow_quote":false,"speaker":"CALLER","subcategories":[],"subcategory_threshold":null,"threshold":null},{"active":true,"builtin":true,"category_id":"caller_reports_unresolved","description":null,"examples":[],"fields":[],"gloss":"Caller says the problem persists","name":"Still unresolved","narrow_quote":false,"speaker":"CALLER","subcategories":[],"subcategory_threshold":null,"threshold":null},{"active":true,"builtin":true,"category_id":"deferred","description":null,"examples":[],"fields":[],"gloss":"Agent defers work or promises a callback","name":"Deferred","narrow_quote":false,"speaker":"AGENT","subcategories":[],"subcategory_threshold":null,"threshold":null}]}', strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z', NULL, 'Install-time version 1: the built-in categories only.');
INSERT INTO results_signal_taxonomy (id, current_version, record_version, settings_json, updated_at, updated_by_account_id) VALUES (1, 1, 1, '{"fallback_extraction_entry_id":null,"pipeline":"v1","v1_fallback":true}', strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z', NULL);
