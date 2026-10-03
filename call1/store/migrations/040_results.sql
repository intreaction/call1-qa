-- 040_results: the results area (call1/store/results/). Numbers 040-049 belong to it.
--
-- Call records and their result projections, the human review state and the human review queue
-- (never the processing queue), reviewer profiles, rubrics, and the semantic-search index.
-- Other areas' rows (conversations, artifacts, jobs, graphs, accounts) are referenced by ID only:
-- no FOREIGN KEY leaves this area. Timestamps are db.ts() text (UTC, fixed width).

-- One row per call conversation. Media fields stay NULL until validation_vad publishes; the QA
-- columns mirror the current qa publication (the machine evaluation version) for lists/metrics.
CREATE TABLE results_calls (
    call_id               TEXT PRIMARY KEY,
    conversation_id       TEXT NOT NULL UNIQUE,
    agent_id              TEXT NOT NULL,
    external_call_ref     TEXT,
    caller_reference      TEXT,
    recorded_at           TEXT,
    created_at            TEXT NOT NULL,
    legacy_call           INTEGER NOT NULL DEFAULT 0,
    duration_seconds      REAL,
    channels              INTEGER,
    channel_layout        TEXT,
    sample_rate           INTEGER,
    codec                 TEXT,
    silence_ratio         REAL,
    overtalk_duration     REAL,
    avg_agent_tone        REAL,
    avg_caller_sentiment  REAL,
    evaluation_version    INTEGER,
    rubric_id             TEXT,
    rubric_version        INTEGER,
    overall_score         REAL,
    passed                INTEGER,
    critical_failure      INTEGER,
    requires_human_review INTEGER NOT NULL DEFAULT 0,
    low_confidence        INTEGER NOT NULL DEFAULT 0,
    evaluated_at          TEXT,
    updated_at            TEXT NOT NULL
);
CREATE INDEX results_calls_by_created ON results_calls (created_at DESC, call_id DESC);
CREATE INDEX results_calls_by_agent ON results_calls (agent_id);
CREATE INDEX results_calls_by_rubric ON results_calls (rubric_id);

-- Every linked job output the completion hook saw (draft tests excluded), so the views can be
-- composed from committed artifacts without reading another area's tables.
CREATE TABLE results_artifacts (
    artifact_id     TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    kind            TEXT NOT NULL,
    slot            TEXT NOT NULL,
    version         INTEGER NOT NULL,
    checksum        TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    job_id          TEXT,
    graph_id        TEXT,
    committed_at    TEXT NOT NULL,
    UNIQUE (conversation_id, kind, slot, version)
);

-- Published versions per result group. version is the publishing artifact's slot version.
CREATE TABLE results_group_versions (
    conversation_id  TEXT NOT NULL,
    kind             TEXT NOT NULL,
    version          INTEGER NOT NULL,
    call_id          TEXT,
    artifact_id      TEXT NOT NULL,
    checksum         TEXT NOT NULL,
    job_id           TEXT,
    graph_id         TEXT,
    graph_created_at TEXT,
    state            TEXT NOT NULL,
    partial_reason   TEXT,
    committed_at     TEXT NOT NULL,
    PRIMARY KEY (conversation_id, kind, version)
);

-- Per-verdict rows of every QA version (metrics and review agreement).
CREATE TABLE results_verdicts (
    call_id            TEXT NOT NULL,
    evaluation_version INTEGER NOT NULL,
    criterion_id       TEXT NOT NULL,
    criterion_name     TEXT NOT NULL,
    status             TEXT NOT NULL,
    confidence         REAL NOT NULL,
    PRIMARY KEY (call_id, evaluation_version, criterion_id)
);

-- Semantic-search vectors of the newest embeddings artifact per conversation.
CREATE TABLE results_search_vectors (
    conversation_id TEXT NOT NULL,
    call_id         TEXT NOT NULL,
    turn_id         INTEGER NOT NULL,
    artifact_id     TEXT NOT NULL,
    scheme          TEXT NOT NULL,
    dimensions      INTEGER NOT NULL,
    vector          BLOB NOT NULL,
    PRIMARY KEY (conversation_id, turn_id)
);
CREATE INDEX results_search_vectors_by_scheme ON results_search_vectors (scheme, dimensions);

-- Masked copies of source audio (content-addressed objects), per source and mute intervals.
CREATE TABLE results_masked_audio (
    source_checksum  TEXT NOT NULL,
    intervals_digest TEXT NOT NULL,
    masked_checksum  TEXT NOT NULL,
    content_type     TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (source_checksum, intervals_digest)
);

-- Human decisions: one review record per call (CallReviewState).
CREATE TABLE results_review_state (
    call_id                        TEXT PRIMARY KEY,
    review_version                 INTEGER NOT NULL DEFAULT 0,
    reviewed_evaluation_version    INTEGER,
    staleness                      TEXT NOT NULL DEFAULT 'current',
    escalation_status              TEXT NOT NULL DEFAULT 'NONE',
    escalation_evaluation_version  INTEGER,
    escalation_resolved_by         TEXT,
    escalation_resolved_at         TEXT,
    reviewer_notes                 TEXT,
    retained_by                    TEXT,
    retained_at                    TEXT,
    updated_at                     TEXT NOT NULL
);

CREATE TABLE results_verdict_overrides (
    id                 TEXT PRIMARY KEY,
    call_id            TEXT NOT NULL,
    criterion_id       TEXT NOT NULL,
    evaluation_version INTEGER NOT NULL,
    original_status    TEXT NOT NULL,
    status             TEXT NOT NULL,
    reason_code        TEXT,
    reviewer_notes     TEXT,
    account_id         TEXT NOT NULL,
    review_version     INTEGER NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE INDEX results_verdict_overrides_by_call ON results_verdict_overrides (call_id, evaluation_version, created_at);

CREATE TABLE results_review_history (
    id                 TEXT PRIMARY KEY,
    call_id            TEXT NOT NULL,
    kind               TEXT NOT NULL,
    account_id         TEXT,
    review_version     INTEGER NOT NULL,
    evaluation_version INTEGER,
    payload_json       TEXT NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE INDEX results_review_history_by_call ON results_review_history (call_id, created_at, id);

-- The human review queue (not the processing job queue) and its rules.
CREATE TABLE results_queue_rules (
    id                    TEXT PRIMARY KEY,
    rank                  INTEGER NOT NULL,
    enabled               INTEGER NOT NULL,
    rule_json             TEXT NOT NULL,
    rule_version          INTEGER NOT NULL,
    updated_at            TEXT NOT NULL,
    updated_by_account_id TEXT
);

CREATE TABLE results_review_items (
    id                     TEXT PRIMARY KEY,
    call_id                TEXT NOT NULL,
    conversation_id        TEXT NOT NULL,
    rule_id                TEXT NOT NULL,
    rule_name              TEXT NOT NULL,
    stream                 TEXT NOT NULL,
    reason                 TEXT NOT NULL,
    urgency_score          REAL NOT NULL,
    evaluation_version     INTEGER NOT NULL,
    stale                  INTEGER NOT NULL DEFAULT 0,
    superseded_by_item_id  TEXT,
    status                 TEXT NOT NULL,
    item_version           INTEGER NOT NULL,
    assigned_to_account_id TEXT,
    assigned_display_name  TEXT,
    created_at             TEXT NOT NULL,
    assigned_at            TEXT,
    started_at             TEXT,
    resolved_at            TEXT,
    resolved_by_account_id TEXT,
    reviewer_notes         TEXT,
    agent_id               TEXT,
    overall_score          REAL,
    critical_failure       INTEGER NOT NULL DEFAULT 0,
    duration_seconds       REAL,
    UNIQUE (call_id, rule_id, evaluation_version)
);
CREATE INDEX results_review_items_by_created ON results_review_items (created_at DESC, id DESC);
CREATE INDEX results_review_items_by_status ON results_review_items (status, assigned_to_account_id);
CREATE INDEX results_review_items_by_call ON results_review_items (call_id);

-- Work-distribution attributes of reviewer accounts (accounts themselves belong to auth).
CREATE TABLE results_reviewer_profiles (
    account_id            TEXT PRIMARY KEY,
    skills_json           TEXT NOT NULL DEFAULT '[]',
    capacity_weight       REAL NOT NULL DEFAULT 1.0,
    accepting_assignments INTEGER NOT NULL DEFAULT 1,
    updated_at            TEXT NOT NULL
);

-- Rubrics: one row per rubric, immutable published versions, and at most one draft.
CREATE TABLE results_rubrics (
    rubric_id           TEXT PRIMARY KEY,
    current_version     INTEGER,
    last_draft_revision INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE results_rubric_versions (
    rubric_id               TEXT NOT NULL,
    version                 INTEGER NOT NULL,
    digest                  TEXT NOT NULL,
    definition_json         TEXT NOT NULL,
    status                  TEXT NOT NULL,
    published_at            TEXT NOT NULL,
    published_by_account_id TEXT,
    notes                   TEXT,
    PRIMARY KEY (rubric_id, version),
    FOREIGN KEY (rubric_id) REFERENCES results_rubrics (rubric_id)
);

CREATE TABLE results_rubric_drafts (
    rubric_id             TEXT PRIMARY KEY,
    definition_json       TEXT NOT NULL,
    draft_revision        INTEGER NOT NULL,
    based_on_version      INTEGER,
    updated_at            TEXT NOT NULL,
    updated_by_account_id TEXT NOT NULL,
    FOREIGN KEY (rubric_id) REFERENCES results_rubrics (rubric_id)
);

-- Seeds: the pre-split default rubric (call1.pipeline.evaluator.DEFAULT_RUBRIC, contract shape;
-- tests/store/test_results_rubrics.py checks the equivalence and digest) and the pre-split
-- default review-queue rules (the PRO1_AUTOMATED rule is not in v1: contract open question 4).
INSERT INTO results_rubrics (rubric_id, current_version, last_draft_revision, created_at, updated_at) VALUES ('call1_standard_v2', 1, 0, strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z', strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z');
INSERT INTO results_rubric_versions (rubric_id, version, digest, definition_json, status, published_at, published_by_account_id, notes) VALUES ('call1_standard_v2', 1, 'sha256:6544ac771f609983374c939a3d057263f58d34163e8676d71c28b8e8b3f78b91', '{"category":"GENERAL","criteria":[{"category":"COMPLIANCE","check":{"aggregation":"mean","check_type":"semantic_judgement","comparison":"gte","escalation_model_id":null,"escalation_when":["needs_review","invalid_answer"],"fail_when":"The agent explicitly contradicts a supplied recording policy. Missing or ambiguous wording alone requires review.","legacy_rule":null,"metric":"text_polarity","metric_threshold":0.0,"min_coverage":0.8,"min_samples":2,"not_applicable_when":"The agent explicitly establishes that recording does not occur on this call.","pass_when":"The agent explicitly tells the caller during the opening that this call is or may be recorded. A negated statement is not a disclosure.","pattern":null,"phrases":[],"policy_context":null,"primary_model_id":null,"requires_policy":false,"response_phrases":[],"speaker":"AGENT","threshold":80,"trigger_phrases":[],"window_seconds":null},"criterion_id":"REG-01","critical":true,"description":"Agent must notify the caller at the beginning of the call that the interaction is being recorded.","name":"Call Recording Disclosure","parameters":null,"rule_type":null,"weight":25.0},{"category":"SECURITY","check":{"aggregation":"mean","check_type":"semantic_judgement","comparison":"gte","escalation_model_id":null,"escalation_when":["needs_review","invalid_answer"],"fail_when":"A supplied verification policy clearly applies and protected account information is explicitly shared before its required steps. Without policy or a clear sequence choose needs_review.","legacy_rule":null,"metric":"text_polarity","metric_threshold":0.0,"min_coverage":0.8,"min_samples":2,"not_applicable_when":"The conversation explicitly concerns general public information only, with no protected account access or transaction.","pass_when":"The required verification policy is supplied and the conversation demonstrates its completion before protected account information is discussed. Asking to verify or mentioning an identifier is insufficient. Without that policy choose needs_review.","pattern":null,"phrases":[],"policy_context":null,"primary_model_id":null,"requires_policy":true,"response_phrases":[],"speaker":"AGENT","threshold":80,"trigger_phrases":[],"window_seconds":null},"criterion_id":"SEC-01","critical":true,"description":"Agent must authenticate caller identity (e.g. account number, DOB, SSN, or PIN) prior to discussing account data.","name":"Caller ID & Verification","parameters":null,"rule_type":null,"weight":30.0},{"category":"COMPLIANCE","check":{"aggregation":"mean","check_type":"semantic_judgement","comparison":"gte","escalation_model_id":null,"escalation_when":["needs_review","invalid_answer"],"fail_when":"An explicitly supplied required term contradicts what the agent states for this transaction. Missing policy or applicability requires review.","legacy_rule":null,"metric":"text_polarity","metric_threshold":0.0,"min_coverage":0.8,"min_samples":2,"not_applicable_when":"The conversation explicitly establishes that no transaction or applicable terms are involved.","pass_when":"The applicable transaction and required disclosures are supplied, and the agent communicates each required item. Mentioning fees, policy, or cancellation alone is insufficient.","pattern":null,"phrases":[],"policy_context":null,"primary_model_id":null,"requires_policy":true,"response_phrases":[],"speaker":"AGENT","threshold":80,"trigger_phrases":[],"window_seconds":null},"criterion_id":"COMP-01","critical":false,"description":"Agent must disclose applicable transaction terms, fees, or dispute rights.","name":"Mandatory Regulatory Disclosures","parameters":null,"rule_type":null,"weight":25.0},{"category":"ETIQUETTE","check":{"aggregation":"mean","check_type":"semantic_judgement","comparison":"gte","escalation_model_id":null,"escalation_when":["needs_review","invalid_answer"],"fail_when":"The agent explicitly refuses further assistance or uses a clearly unprofessional closing at the end. An incomplete ending or uncertain call boundary requires review.","legacy_rule":null,"metric":"text_polarity","metric_threshold":0.0,"min_coverage":0.8,"min_samples":2,"not_applicable_when":"The conversation explicitly ends through an agreed transfer to another agent before a closing is appropriate.","pass_when":"At the end of the conversation the agent both offers further assistance and closes professionally. An opening greeting or mid-call offer is insufficient.","pattern":null,"phrases":[],"policy_context":null,"primary_model_id":null,"requires_policy":false,"response_phrases":[],"speaker":"AGENT","threshold":80,"trigger_phrases":[],"window_seconds":null},"criterion_id":"ETIQ-01","critical":false,"description":"Agent must ask if there is anything else they can help with and close the call professionally.","name":"Professional Closing & Resolution","parameters":null,"rule_type":null,"weight":20.0}],"description":"Class QA template requiring contextual evidence and human review when policy or applicability is unknown. Not a compliance certification.","name":"Call1 Standard Contact Center QA & Compliance — Contextual v2","pass_threshold":80.0,"rubric_id":"call1_standard_v2"}', 'active', strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z', NULL, 'Seeded from the pre-split default rubric (call1.pipeline.evaluator.DEFAULT_RUBRIC).');
INSERT INTO results_queue_rules (id, rank, enabled, rule_json, rule_version, updated_at, updated_by_account_id) VALUES ('rule-triage-critical-lowconf', 10, 1, '{"critical_failure_only":true,"description":"Machine-flagged calls: critical compliance breaches or low-confidence verdicts. Routed to the reviewer with the smallest open backlog.","distribution_strategy":"LEAST_OUTSTANDING","enabled":true,"id":"rule-triage-critical-lowconf","low_confidence_only":true,"name":"Triage: Critical Breaches & Low Confidence","rank":10,"sampling_rate":0.0,"stream":"TRIAGE","target_agents":[],"target_domains":[],"target_skills":[]}', 1, strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z', NULL);
INSERT INTO results_queue_rules (id, rank, enabled, rule_json, rule_version, updated_at, updated_by_account_id) VALUES ('rule-audit-sample-drift', 50, 1, '{"critical_failure_only":false,"description":"Random 10% sample of confident passed calls to measure false negatives and model drift. Distributed round-robin across active reviewers.","distribution_strategy":"ROUND_ROBIN","enabled":true,"id":"rule-audit-sample-drift","low_confidence_only":false,"name":"Model Drift Audit Sample","rank":50,"sampling_rate":0.1,"stream":"AUDIT_SAMPLE","target_agents":[],"target_domains":[],"target_skills":[]}', 1, strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z', NULL);
INSERT INTO results_queue_rules (id, rank, enabled, rule_json, rule_version, updated_at, updated_by_account_id) VALUES ('rule-mandate-dispute-escalation', 20, 1, '{"critical_failure_only":false,"description":"Business mandate: dispute and escalation calls must be reviewed by a reviewer with the ESCALATIONS skill.","distribution_strategy":"SKILL_MATCHED","enabled":true,"id":"rule-mandate-dispute-escalation","low_confidence_only":false,"name":"Dispute & Escalation Mandate","rank":20,"sampling_rate":0.0,"stream":"MANDATE","target_agents":[],"target_domains":[],"target_skills":["ESCALATIONS"]}', 1, strftime('%Y-%m-%dT%H:%M:%S', 'now') || '.000000Z', NULL);
