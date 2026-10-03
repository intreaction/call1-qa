# Store results area

The results area of Call1 Store (`call1/store/results/`, migrations `040`–`049`: `040_results.sql`; `041_agent_identity.sql`, which adds the contract 1.1.0 agent display name and extension to the call record and review-queue items; `042_signals.sql`, the contract 1.3.0 Contact Signals v2 tables; `043_training_labels.sql`, the contract 1.3.0 on-device training label log; and `044_asr_vocabulary.sql`, the contract 1.3.0 ASR vocabulary). It owns the call
records and result projections Evaluate reads, semantic search, the human review decisions, the human
review queue and its rules, reviewer profiles, rubrics, metrics, (contract 1.3.0) the Contact Signals v2 taxonomy, alert rules,
hit feedback and signal metrics, (contract 1.3.0, decision 28) the reviewer-label log for on-device training, and (contract 1.3.0, decision 33) the ASR vocabulary for dual transcription. It serves 54 contract operations.
The table in `call1/store/routing.py` (`_RESULTS`) lists them all. The area runs no model and never
reads another area's tables.

## Modules

| Module | What it does |
|---|---|
| `projections.py` | The hooks the queue area calls inside its transactions: `on_conversation_registered`, `on_call_metadata_updated` (a re-registration changed the call metadata, contract 1.1.0), `apply_completion` (step 7) and `on_job_failed` |
| `records.py` | Call records, the linked-artifact index, published group versions, `result_groups` (the one place `derive_result_state` runs), and the transcript, evaluation, summary and contact-signal views |
| `review_state.py` | The call-level review: expected-version checks, the stale-write rule, review history |
| `review_queue.py` | The human review queue: rule matching (the pre-split `QueueManager` rules), distribution, supersession |
| `rubric_store.py` | Drafts (`draft_revision`), immutable published versions, retirement, snapshot content |
| `search.py` | Semantic search with the shared search embedder (`call1/embedding.py`, contract 1.2.0) |
| `masking.py`, `audio.py` | Masking of reviewer reads, and audio playback with muting and HTTP Range |
| `metrics.py` | Executive, per-rubric and review-agreement metrics. `total_hours_audited` keeps 6 decimals, so short calls are not rounded away |
| `content.py` | Reads committed artifact bytes by checksum. The hooks get no `Store`, so it finds `objects/` beside `store.db` |
| `signal_store.py` | Contact Signals v2 (1.3.0): the taxonomy record and immutable versions, save-time rules (caps from the effective `ContractParameters`, definition-text detectors, no delete), settings, redaction, alert rules, the seed applier |
| `signals.py` | The signal projection, the one alert predicate (`alert_condition_sql`), the contact-signals view's read-time context, call-list fields and filters, hit feedback, and the preview diff |
| `api.py` | What other areas call: `rubric_version`, `rubric_snapshot_content`, `draft_snapshot_content`, `result_groups`, and (1.3.0) `signal_taxonomy_version`, `current_signal_taxonomy`, `signal_settings`, `check_signal_taxonomy`, `signal_backfill_candidates`, `published_signal_pipeline`, `masked_signal_result`, `signal_preview_diff` |
| `training_labels.py` | On-device training (1.3.0, decision 28): the label log's writes (called from the override, feedback and speaker-correction handlers in their transactions), the one-time backfill at Store start, `listTrainingLabels` reads, and source resolution through `queue.api` |
| `vocabulary.py` | The ASR vocabulary (1.3.0, decision 33): the singleton record, save rules (caps from the effective `ContractParameters`, pack-only switches, PII detectors), the pack install |
| `http_*.py` | The HTTP handlers, registered on `routes.router` (`http_training.py`: `listTrainingLabels`; `http_vocabulary.py`: `getAsrVocabulary`, `saveAsrVocabulary`) |

## How a completion is projected

`apply_completion` does nothing for a draft-test graph. For any other completion it runs these steps
in a SAVEPOINT inside the queue's completion transaction:

1. It indexes every linked output in `results_artifacts`. The read views are built from this index.
2. It fills the call record: `validation_report` and `vad_metrics` give the media fields,
   `tone_blocks` gives `avg_agent_tone`, and `text_sentiment` gives `avg_caller_sentiment`.
3. `embeddings` of any scheme replace the conversation's search vectors (search ranks only its own scheme).
4. When the completion carries `result`, the publishing artifact's slot version becomes the group's
   version. That version goes into `results_group_versions` and back into the receipt as
   `result_version`.
5. When the published group is QA, the call's score columns and verdict rows are refreshed, and the
   stale-write rule runs against the review and the review queue (see below).
6. It appends change events: `call`, `result`, `review` and `review_queue`.

Result states are not cached. `result_groups` combines the published versions with the queue's
`group_stakes`, `reanalysis_pending_groups`, `failure_codes` and `reanalysis_request_for_group` at
read time. `on_job_failed` therefore only writes a `result` change event.

## A call metadata update (contract 1.1.0)

`on_call_metadata_updated` runs inside the queue's registration transaction when a re-registration
changed a call's metadata. It rewrites the call record's agent, external call reference, caller
reference and `recorded_at`, and the agent fields of every review-queue item of the call (resolved
ones too; `item_version` does not change, because this is not a queue transition), then appends one
`call` change event with status `metadata_updated`. It touches no result, review version, review
state or queue status. Queue rules are not re-run: a changed `agent_id` affects `target_agents`
matching from the next QA version on.

## Reviews and the review queue

- **Version checks.** Every review write checks the call's `review_version`. A mismatch is 409
  `review_version_conflict` with `current_version`. A write that judges the machine result must name
  the current evaluation version, or it gets 409 `conflict` with `current_evaluation_version`.
- **What a new QA version does.**
  - The review becomes `stale` when any decision referenced an older version. This does not bump
    `review_version`, because only human writes do.
  - Escalation becomes `PENDING` when the new scorecard requires review and none was open. It returns
    to `NONE` when the new scorecard does not require review.
  - Every unresolved queue item becomes `SUPERSEDED` and points at its replacement.
- **Replacement items.** The replacement is the new version's item for the same rule. When that rule
  no longer matches, the item is carried forward anyway, keeping its assignee, so work a person was
  given never disappears silently.
- **Rules.** Rules are evaluated when QA commits. TRIAGE rules use the critical-failure and
  low-confidence flags (confidence below 0.70, or FLAGGED). AUDIT_SAMPLE is a deterministic hash draw
  per call, rule and version. MANDATE uses the target agents or the dispute vocabulary. CALIBRATION
  matches every call. When `target_domains` is set, the rubric's category must be one of them.
- **Distribution.** UNASSIGNED_CLAIM leaves the item in the pool. ROUND_ROBIN and SKILL_MATCHED
  cycle through reviewers by name. LEAST_OUTSTANDING picks the smallest open backlog per capacity
  weight.
- **Seeds.** `040` seeds the three pre-split default rules. The PRO1_AUTOMATED rule is left out, per
  contract open question 4.

## Rubrics

`040` seeds version 1 of `call1_standard_v2`. It is the pre-split
`call1.pipeline.evaluator.DEFAULT_RUBRIC` in the contract shape, with the legacy `rule_type` and
`parameters` dropped because `check` is canonical. `test_results_rubrics.py` pins the definition and
its digest.

Each rubric has one draft. `draft_revision` never repeats within a rubric. Publishing consumes the
draft, and publishing with no criteria (or zero total weight) fails with `validation_failed`.
Retiring marks the current version `retired`, which hides the rubric from `listRubrics` unless
`include_retired` is set.

## Masking (a documented default)

Admin state is deferred, so Store applies the contract default `mask_reviewer_reads = true`. The
value rules come from `call1.redaction`: privacy-sensitive numeric entities from the `enrichment`
artifact, plus the SSN, card, phone, account and PIN patterns. Every occurrence is masked. These
fields are masked:

- transcript text and word timestamps
- verdict evidence, reasoning and model attempts, including a draft test's scorecard
  (`getDraftTestResult`, through `results.api.masked_scorecard`; since contract 1.2.0)
- summary text
- contact-signal quotes
- search hits

Audio is muted over the same spans:

- A linked `redacted_audio` is served as is.
- PCM WAV is muted in Python.
- Other containers are muted with `ffmpeg` when it is installed.
- When muting is impossible, playback fails closed with 503 `store_unavailable` (not retryable). The
  unmasked original is never served.

### Model PII findings and the fail-closed rule (contract 1.2.0, decision 19)

Process's `enrichment` job also writes `pii_findings`: the model PII layer's spans (names,
addresses, emails, URLs, secrets, account numbers, phones; never dates, never the agent's own name)
for one transcript revision. Store runs no model for this. It adds the findings' texts to the rule
values above for every masked field and for audio muting, but only when the findings name the
current published transcript (`PiiFindingsContent.transcript.checksum`).

While no findings exist for the current transcript revision (enrichment still running, failed with
`model_unavailable`, or a call from before 1.2.0), reads fail closed. This was chosen as the least
disruptive option: no new result state, and every route keeps its shape.

- `getTranscript` still answers 200 with timing, speakers, sentiment and tone, but every turn's
  `text` is `""`, `word_timestamps` is null and `text_withheld` is `true`.
- Evaluation evidence and reasoning, summary text and contact-signal quotes read `[REDACTED]`.
- Semantic search skips the call.
- `getCallAudio` answers 503 `store_unavailable`, retryable, `details.reason` `pii_findings_pending`.
  This also applies before a transcript exists.
- The transcript result group reads `partial`, with `partial_reason` `records.PII_PENDING_REASON`
  while an `enrichment` job can still finish, else `records.PII_MISSING_REASON`. To fix a call
  without findings, retry its `enrichment` job or request a `full` reanalysis.

## Semantic search

Since contract 1.2.0 (decision 18) Store embeds each query with `call1/embedding.py`, the module
Process embeds turns with: `nvidia/Nemotron-3-Embed-1B-BF16` at a pinned revision (scheme
`nemotron-3-embed-1b@c0c9fea`, 2048 dimensions, queries as `query: <text>`), or the deterministic
fake (`fake-embedding-v1`) on fake-handler stacks. This embedder is the only model Store runs. It
loads on the first query (about 3.4 s on CPU in bfloat16), stays resident (about 2.3 GB), and is
shared by request threads behind a lock; warm queries take about 0.1 s. Startup never needs it.

- Only vectors of the configured scheme are ranked (numpy cosine; negatives count as 0). Calls
  indexed under another scheme, such as `hashing-projection-v1` from before 1.2.0, are counted in
  `calls_needing_reembedding`; reanalysis kind `embeddings` re-embeds one.
- Without the weights, or when loading fails, search answers 503 `search_unavailable` (not
  retryable, `details.reason` `not_installed` or `load_failed`). It never falls back to a scheme.
- `GET /status/detail` reports `search_embedder` (`not_installed`, `installed`, `loaded`, `failed`,
  `fake`) without loading it, and `python -m call1.store serve` prints the same on start-up.
- `CALL1_EMBEDDING_BACKEND` (`nemotron` or `fake`, default: `fake` when `CALL1_PROCESS_HANDLERS=fake`,
  else `nemotron`), `CALL1_EMBEDDING_PATH` (default `data/models/nemotron-3-embed-1b`),
  `CALL1_EMBEDDING_DTYPE` and `CALL1_EMBEDDING_DEVICE` must match Process's.

A Process `embeddings` job links its artifact in slot `""` with the scheme it embedded under.

## Contact Signals v2 (contract 1.3.0)

Design: [docs/ContactSignalsV2.md](../../../docs/Architecture.md) sections 7, 9 and 14. Signals are
open core and unscored: nothing here touches a scorecard, `overall_score` or the review version.

**The taxonomy** (`signal_store.py`, tables `results_signal_taxonomy` and
`results_signal_taxonomy_versions`) is one document, versioned whole. `042_signals.sql` seeds version 1
with the eight built-ins only (`signals.builtin_signal_taxonomy()`), `record_version` 1 and the default
settings (`pipeline: v1`). `saveSignalTaxonomy` returns the record unchanged when the digest equals
the current version's, otherwise publishes N+1. Its save validator refuses, with `validation_failed`
and `details.field` naming the path (never the value):

- the section 9.6 caps, read from Store's *effective* `ContractParameters`
  (`CALL1_STORE_PARAMETERS`), so a tighter cap needs no schema change (`details.cap`, `limit`, `actual`);
- definition text a rule-based detector matches (`call1.redaction.find_pii`: SSN, card, phone,
  account number, PIN, digit runs; `details.reason: sensitive_text`, `details.detector`);
- a published category or subcategory missing from the save (`details.reason: node_removed`):
  there is no delete, `active: false` retires a node, so versions, results, feedback and rules keep
  resolving it. (Section 7.2 "No delete"; the contract models do not check it across versions.)

The contract models refuse built-in edits, reserved IDs and forbidden PII classes. A stale
`expected_record_version` is 409 `signal_taxonomy_conflict` with `current_version` and
`record_version`; settings share that token. `redactSignalTaxonomyText` stores
`signals.redact_signal_taxonomy_text` of a non-current version (numbered `[REDACTED <n>]`, built-in
constants kept) and keeps its digest. Audit details and change events carry versions, digests and
changed-node paths, never text.

**Seeds.** `python -m call1.store apply-signals-seed call1/store/seeds/signals_retail_v1.json
[--pipeline v2]` applies a `SignalTaxonomySave` file as an audited save (installer actor, no account);
a seed already current is a no-op. `--demo` uses it (decision 22); installs start with the built-ins
only. `python -m call1.store signals-pipeline v2` sets only the pipeline.

**Projection** (`signals.project`, called from `projections._publish` for the `contact_signals`
group, inside the completion transaction): skip when `results_calls.signals_version` is at or above
the version; insert `results_signal_outcomes`, `results_signal_hits` and `results_signal_hit_fields`;
set `signals_version`; call `review_queue.on_new_signals`; append one `signal_alert`
(`fired:<rule_id>`) event per enabled rule that matches now and did not match the previous projected
version. v1 results project with `category_id = kind` and no subcategory. **No text is copied**: no
quotes, and for fields only the status plus enum and boolean values. A multi-segment hit (decision 25)
is one row under its anchor's hit ID, spanning `start` to `span_end`, so it counts once in
`hit_count`, alerts, filters and metrics. Results published before 1.3.0
are projected by `python -m call1.store project-signals` (idempotent, bounded by `--limit`, audited as
`signal_backfill_requested` with `mode: project`, the nearest contract action) or at their next publish.

**Alerts** are evaluated at read time over the projection tables by one predicate
(`signals.alert_condition_sql`), used by the view's `alerts`, the call list's `signal_alerts` and
`signal_alert` filter, metrics, the queue facts and the fired events. A disabled rule, or one whose
node is inactive (`node_active: false`), matches nothing. Rule edits never fire events for older calls.

**Reads.** `getContactSignals` masks quotes (a multi-segment hit's part quotes too, decision 25),
string and date field values, surface text and evidence with the call's `Masker`, and reads `[REDACTED]` with `text_withheld: true` while the PII findings of
the current transcript are pending. It adds `taxonomy_status` (`signals.signal_taxonomy_status` against
the current version; v1 results are never outdated), `feedback`, `alerts`, and for admins
(`manage_signals`) `comparison_preview_id` (the newest compare with a result, from the queue API).
`CallListItem` gains `contact_signals_state`, `signal_categories` and `caller_needs` (current hits on
active nodes) and `signal_alerts`; `listCalls` filters on `signal_category`, `signal_subcategory` and
`signal_alert`.

**Feedback** (`saveSignalHitFeedback`, `results_signal_feedback`) is keyed on (call, hit ID), so it
survives threshold edits, sibling changes and republishes; a save replaces the verdicts. Store records
the subcategory the verdict judged (ID and digest) from the current hit. Metrics count a subcategory
verdict only while it still names the hit's subcategory and digest ("judged an earlier subcategory").
When a rescore merges a judged hit into another hit's `parts` (decision 25), its feedback row keeps
its own ID and the view still lists it (`signals.feedback_hit_ids`: every hit ID plus each part's
earlier ID); Evaluate shows it as a judgement of an earlier segmentation, not the merged hit's.
Audit `signal_hit_reviewed`; change event `review` with status `signal_feedback`.

**Preview diff** (`signals.preview_diff`): a preview hit pairs with the published hit covering any
of its segments (anchor or part, by category, turn and block), so a span that became a part is not
"removed". A pair is `relabelled`, else `fields_changed`, else `segments_changed`; no subcategory and
`other` compare equal.

**Metrics** (`metrics.signal_metrics`): each call at its current `signals_version`, windowed like the
other metrics. A built-in category's denominator is every projected call (v1 or v2), a custom
category's the calls whose current result is v2. `share_pct` is a subcategory's share of the
category's subcategorized hits; precision is null under 5 judged hits; field distributions cover enum
and boolean fields only. Without `include_inactive`, retired nodes and disabled or inactive rules are
left out.

**Review queue** (section 9.3). `ReviewQueueRule.target_signal_alerts` is checked before the stream
(like `target_domains`); the `SIGNAL` stream's base condition is always true, its reason reads
"Signal: <rule name> (<speaker>, m:ss)" and its urgency is 50. `on_new_evaluation` reads the call's
current alerts into `ScoreFacts` and keeps its per-version short-circuit. `on_new_signals` needs an
evaluation version (decision 22, Q2), pre-checks (call, rule, evaluation version) in the same
transaction and never supersedes anything. Items of alert-targeting rules record `signals_version`
and `trigger_alert_rule_ids`; an item whose alert stops matching stays.

## On-device training labels (contract 1.3.0, decision 28)

docs/OnDeviceTraining.md section 2. `results_training_labels` (`043`) is an append-only log; `seq`
is the cursor. Each row holds IDs, enums and versions only: no transcript text, quote, reviewer note,
signal note or reviewer identity.

- **Writes**, each in the label write's own transaction (a rolled-back write appends nothing):
  `overrideVerdict` appends a `qa_verdict` row that judged the scorecard publication at
  `evaluation_version`; `saveSignalHitFeedback` appends a `signal_hit` row for a v2 hit only (the
  published content names its `category_id`), with the anchor's and parts' `(turn_id, block)` read
  from the current publication, and a `withdrawn` row when both verdicts are cleared;
  `correctSpeaker` appends a `speaker_role` row that judged the model's attribution (the newest
  `speaker_attribution` not written by a reviewer correction, else the newest), with that
  artifact's cluster for the turn. The subject is `training_label_subject(...)`; a speaker subject
  is the cluster when `apply_to_cluster` and a cluster was resolved, else `t<turn_id>`. A row that
  would not validate as a `TrainingLabel` is logged and skipped, and never fails the reviewer's
  write. A call with no speaker attribution, and a v1 hit, log nothing.
- **Backfill** (`training_labels.backfill`, run by `Store.open`): once, guarded by the `store_meta`
  key `training_labels_backfilled`, in timestamp order: every `results_verdict_overrides` row, every
  `speaker_correction` history row (the attribution committed at or before it), and one row per
  current `results_signal_feedback` row at the newest projected version that has the hit.
- **Reads** (`listTrainingLabels`, `training:read` only). Items in `seq` order; `count_after` is
  counted before paging and `high_water` is the log's max `seq`. `source_job_id` and `sources`
  are resolved at read time through `queue.api` (`get_artifact`, `get_job`): for QA, the scorecard
  job's pinned `assessment:<criterion>` names the `qa_criterion` job, whose pinned inputs,
  `assessment` and `prompt_input` outputs, and the scorecard's `escalation:<criterion>` input are
  the sources; for signals, the merge job's pinned inputs plus the `contact_signals` artifact; for
  speakers, the attribution job's `transcript` plus the attribution, and the `enrichment` output of
  the job that made the chosen findings. `pii_findings` is the newest indexed findings made from the
  label's transcript revision, omitted when there is none. A source that cannot be resolved, or
  lacks a required role, leaves `sources: []` and `source_job_id: null` (Process skips it as
  `source_unavailable`); a withdrawn row resolves nothing. When `after` is beyond `high_water` (a
  cursor from another dataset), the page is empty and `next_after` is `high_water`.
- **Audit.** A read with `limit > 0` appends `training_labels_read` in its own transaction (actor:
  the Process installation; target `installation`; details: `after`, `next_after`, `items`,
  `withdrawn` and a count per kind). A count-only read (`limit 0`) is not audited.

## ASR vocabulary (contract 1.3.0, decision 33)

docs/DualAsr.md sections 4 and 8. `results_asr_vocabulary` (`044`) is one row: `record_version`,
`enabled`, the customer's terms, the disabled pack terms, the installed pack JSON and updated at/by.
Nothing derived is stored: `vocabulary.record` computes `effective_terms`, `effective_digest` and
`active` with `call1.contracts.vocabulary` on every read.

- **Save** (`saveAsrVocabulary`): the expected version first (409 `conflict`,
  `details.current_version`), then `check_settings`: `too_many_terms` (`max_vocabulary_terms`),
  `unknown_pack_term`, the term rule again, and `find_pii` over every term (`pii_detected`, with the
  detector label). Details name `field`, `reason` and `index`, never the term. Unchanged settings
  return the record as is. Audit `asr_vocabulary_saved` (versions, old and new digest, enabled,
  active, pack ID and version, counts per source; `change` `saved`) and change event
  `asr_vocabulary` `saved`.
- **Install** (`install_pack`, host command `apply-vocabulary-seed`): `check_pack`
  (`max_vocabulary_pack_terms`, detectors), then replace the pack in one transaction and keep only the
  disabled terms the new pack has. The same pack again is a no-op. Installer actor; `change`
  `pack_installed`; change event `pack_installed`.
- **View** (`records.vocabulary_correction_view`): the transcript's `vocabulary_correction` for
  reviewers. Withheld text withholds every replacement. Otherwise a replacement is withheld when its
  turn is missing, one of its view words is `[REDACTED]`, or its raw characters overlap a span the
  Masker replaces (`Masker.spans`). `heard` goes through `Masker.text` and is null when that changed
  it or it holds a digit. `char_start`/`char_end` are shifted by the masks before them and kept only
  when the masked text holds the same characters there. `candidate_text` never leaves Store, and the
  raw `asr_base_transcript` and `asr_vocabulary_pass` artifacts are indexed but never projected.

## Tests

`tests/store/test_results_*.py` and `tests/store/test_reviews_*.py` cover this area:

- `test_results_harness.py` provides `FakeQueue` and `FakeAccounts`, which monkeypatch `queue.api`
  and `auth.api`, the way the brief allows.
- `test_results_end_to_end.py` drives the real queue and auth areas over HTTP instead.
- `test_training_labels.py` (1.3.0) covers the label log (docs/OnDeviceTraining.md section 7.3,
  S1–S8) through the real queue, results and auth areas: one call with a transcript, a diarization
  attribution, PII findings, a QA criterion, escalation and scorecard, and a v2 merge with a
  multi-segment hit.
- `test_asr_vocabulary.py` (1.3.0, decision 33) covers the vocabulary document and every refusal,
  the pack install, the retail seed and `--demo`, the graph cap, completion with the `asr` job's
  optional outputs, and the masked view (withheld text, a masked term, a masked or digit `heard`,
  offsets), plus one call end to end through the real queue.
