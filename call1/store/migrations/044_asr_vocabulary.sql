-- 044_asr_vocabulary: the ASR vocabulary for dual transcription (contract 1.3.0, team decision 33;
-- docs/DualAsr.md section 4).
--
-- One singleton document: the admin-edited settings (enabled, the customer's own terms, the pack
-- terms switched off) and the installed industry pack (AsrVocabularyPack JSON, installed only by
-- Store's local apply-vocabulary-seed command). record_version is the optimistic-concurrency token
-- of saveAsrVocabulary; 0 until the first save or pack install. The effective terms, their digest
-- and whether dual transcription is active are derived on every read (call1/store/results/vocabulary.py),
-- never stored. Terms are business vocabulary validated to hold no digit; audit details and change
-- events never carry them.

CREATE TABLE results_asr_vocabulary (
    id                       INTEGER PRIMARY KEY CHECK (id = 1),
    record_version           INTEGER NOT NULL CHECK (record_version >= 0),
    enabled                  INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    customer_terms_json      TEXT NOT NULL,
    disabled_pack_terms_json TEXT NOT NULL,
    pack_json                TEXT,
    updated_at               TEXT,
    updated_by_account_id    TEXT
);

INSERT INTO results_asr_vocabulary (id, record_version, enabled, customer_terms_json, disabled_pack_terms_json, pack_json, updated_at, updated_by_account_id)
VALUES (1, 0, 1, '[]', '[]', NULL, NULL, NULL);
