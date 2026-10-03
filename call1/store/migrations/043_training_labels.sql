-- On-device training labels (contract 1.3.0, team decision 28; docs/OnDeviceTraining.md section 2.1).
--
-- An append-only log of the reviewer labels Process trains the customer's LoRA from, on the device.
-- One row per QA verdict override, Contact Signals v2 hit feedback save and speaker correction,
-- written in the label write's own transaction (call1/store/results/training_labels.py). seq is the
-- listTrainingLabels cursor. Rows are never rewritten: a later label for the same subject is a new
-- row, and a cleared signal feedback is a withdrawn row.
--
-- The row holds IDs, enums and versions only: no transcript text, quote, reviewer note, signal note
-- or reviewer identity. label_json is the kind's contract model (training.QaVerdictLabel,
-- SignalHitLabel or SpeakerRoleLabel). source_artifact_id is the artifact the label judged (the
-- qa_scorecard publication, the contact_signals publication, or the speaker_attribution); the
-- source job and the other source artifacts are resolved at read time through the queue area's API.

CREATE TABLE results_training_labels (
    seq                INTEGER PRIMARY KEY AUTOINCREMENT,
    kind               TEXT NOT NULL CHECK (kind IN ('qa_verdict', 'signal_hit', 'speaker_role')),
    subject            TEXT NOT NULL,
    call_id            TEXT NOT NULL,
    conversation_id    TEXT NOT NULL,
    source_artifact_id TEXT NOT NULL,
    label_json         TEXT NOT NULL,
    withdrawn          INTEGER NOT NULL DEFAULT 0 CHECK (withdrawn IN (0, 1)),
    recorded_at        TEXT NOT NULL
);
CREATE INDEX results_training_labels_by_subject ON results_training_labels (subject, seq);
CREATE INDEX results_training_labels_by_kind ON results_training_labels (kind, seq);
