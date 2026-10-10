# The Store contract (v1.4.0)

This directory is the only thing the three tracks share. `split/store` implements it, and
`split/process` and `split/evaluate` build clients against it. None of the three imports another's
code. It fixes the Store HTTP API, the shared schemas and artifact content, the principals and
their rights, the data-custody and release-trust records, the queue semantics, and the rules that
every layout must obey: single computer, LAN, and the v2 hosted layouts.

Revision round 1 (2026-09-25) applied the independent reviewers' findings, and revision round 2
(the same day) applied the verifiers' findings V1–V8. The contract version stays 1.0.0 because
nothing has been merged yet. Version 1.0.1 (2026-09-25) adds delivery-stage tags and records John's
decision on open question 12; no schema changed.

Version 1.1.0 (2026-09-25) applies the team decisions of that day on agent identity and
re-uploads. It is a minor version because every change is additive: optional fields, one audit
action, one route rule for a case that used to be a silent no-op. A 1.0.x client keeps working.

- **Agent identity.** `CallMetadata` gains optional `agent_display_name` (1–100 characters,
  trimmed) and `agent_extension` (1–20 dial-string characters). `agent_id` stays and remains the
  key for filters, metrics and queue rules. The call list, call detail, review-queue items and
  escalations carry both new fields. Every client labels an agent with `calls.agent_label()`:
  `display_name (extension)` when a display name is present, otherwise `agent_id`, with
  ` (extension)` appended whenever there is an extension. Call-list text search also matches the
  display name and extension.
- **Re-registration updates call metadata.** Registering a source identity that already exists,
  with different `call_metadata`, now updates the call's metadata instead of dropping it. See
  "Conversation re-registration" under "Idempotency". `ConversationRegistered` gains
  `metadata_updated` (default `false`) and `updated_fields`. `AuditAction` gains
  `call_metadata_updated`. `registerConversation` is now marked audited.
- **Summaries are unchanged.** They keep the legacy behavior: masked values, and categories may
  be named. This is recorded as an AI risk, not a contract change.

Version 1.2.0 (2026-09-25) applies team decision 18 (and, below, decision 19): semantic search moves from the model-free
`hashing-projection-v1` scheme to a real embedding model, and Store embeds queries with it. It is a
minor version because every change is additive: one error code, one reanalysis kind, optional
response fields, one optional status section. A 1.1.x client keeps working.

- **The search embedder.** `nvidia/Nemotron-3-Embed-1B-BF16` (OpenMDW-1.1) at revision
  `c0c9fea93ea424587517f2c59e20db9f1d6bf615`: a bidirectional encoder, attention-mask mean pooling
  over every token, L2-normalized, 2048 dimensions. Turns are embedded as `passage: <text>` and
  queries as `query: <text>`. Its scheme is `nemotron-3-embed-1b@c0c9fea`. Process and Store both
  embed with `call1/embedding.py`, so a query and the turn vectors share one space. The
  `embeddings` job is now a model stage (catalog entry `nemotron-3-embed-1b`, adapter
  `call1.torch.nemotron_embed`); `JobTypeRule` is unchanged because the job type already froze an
  `embeddings` selection.
- **Store runs this one model.** A deliberate, narrow exception to "Store has no model runtime":
  Store loads only the search embedder, locally, lazily on the first query, and keeps it resident.
  Never ASR, diarization, text generation or a remote model.
- **Scheme matching.** `semanticSearch` ranks only turn vectors whose `EmbeddingsContent.scheme`
  equals Store's configured embedder's. Older artifacts (`hashing-projection-v1`) are kept and
  indexed but never ranked. `SemanticSearchResponse` gains `embedding_scheme` and
  `calls_needing_reembedding` (calls in scope indexed under another scheme).
- **Re-embedding.** `ReanalysisKind` gains `embeddings`: one `embeddings` job on the call's linked
  transcript and attribution. It affects no result group (`REANALYSIS_KIND_AFFECTS` is empty), like
  the embeddings job itself, which counts only in `pending_work`.
- **Unavailable search.** When the embedder is not installed or fails to load, `semanticSearch`
  answers 503 with the new `ErrorCode.search_unavailable` (not retryable; `details.reason` is
  `not_installed`, `load_failed` or `misconfigured`, `details.embedding_scheme` names the scheme). It
  never falls back to another scheme.
- **Status.** `StoreStatus` gains optional `search_embedder` (`SearchEmbedderStatus`: scheme, model,
  revision, dimensions, state `not_installed`/`installed`/`loaded`/`failed`/`fake`, detail). Reading
  it never loads the model.
- **Test embedder.** Fake-handler stacks use a deterministic, model-free embedder with its own scheme,
  `fake-embedding-v1`. It is selected by `CALL1_EMBEDDING_BACKEND=fake`, or automatically when
  `CALL1_PROCESS_HANDLERS=fake`; Store and Process must resolve the same backend.

1.2.0 also carries team decision 19's model PII layer onto reviewer reads (additive: one artifact
kind, one output role on an existing job type, one optional view field):

- **`pii_findings` artifacts.** `ArtifactKind.pii_findings` (`pii_findings.v1`,
  `PiiFindingsContent`, sensitivity fixed at `raw`) holds the model PII layer's findings for one
  transcript revision: per turn, spans with `start`/`end` (character offsets into that turn's
  text), a `category` from the masked set (`account_number`, `private_address`, `private_email`,
  `private_person`, `private_phone`, `private_url`, `secret`; never `private_date`) and the matched
  `text`, plus `transcript` (the `ArtifactRef` of the transcript revision it was made from),
  `detector` (`openai/privacy-filter`, or `stub` on fake-handler stacks) and `detector_revision`.
  Process has already applied the masking policy: a span naming only the agent is never listed.
- **Produced by `enrichment`.** `JobTypeRule` for `enrichment` gains a second output role,
  `pii_findings`, so the model runs once per transcript revision in the job that already extracts
  the rule-level entities. The job also takes the `speaker_attribution` input when there is one,
  so agent self-introductions are recognized. Every masked text-model job (`qa_criterion` and its
  escalation, `summary_segment`, `summary_assembly`, the contact-signal passes) pins it as input
  role `pii_findings` and uses it instead of loading the model again. A missing model fails
  `enrichment` with `model_unavailable` (fail closed), which leaves those jobs dead-blocked.
- **Store masks with them, and only for the current revision.** Every reviewer read (transcript
  text and word timestamps, evaluation evidence and reasoning, a draft test's scorecard, summary,
  contact-signal quotes, search hits) masks the union of the rule values and the findings' texts, and audio is muted over
  both. Findings count only when `PiiFindingsContent.transcript.checksum` is the published
  transcript's. Until they exist, Store **fails closed**: `TranscriptView.text_withheld` (new,
  default `false`) is `true`, every turn's `text` is empty and `word_timestamps` null; other text
  fields read `[REDACTED]`; semantic search returns no hits from the call; `getCallAudio` answers
  503 `store_unavailable` (retryable, `details.reason` `pii_findings_pending`). The transcript
  result group reads `partial` meanwhile, with a `partial_reason` saying whether masking is still
  running or did not finish, rather than a new state. Store runs no model for this.

Version 1.3.0 (2026-09-25) applies team decisions 21 and 22: **Contact Signals v2**
(`docs/ContactSignalsV2.md` section 7). Contact signals become a three-stage cascade over ~7 s
segments: stage 1 picks a category per segment, stage 2 a subcategory per isolated span, and stage 3
optionally extracts admin-defined fields. Admins extend the vocabulary with subcategories, custom
categories, fields and alert rules, which drive review-queue rules, metrics, filters and the change
feed. Everything is open core; nothing reads a Pro1 connection or entitlement, and signals never
change a scorecard.

- **Additive for Evaluate and for the OpenAPI shape.** Every change is a new route, enum value,
  optional field, error code or contract parameter. A v2 hit is still a `ContactSignalView` with the
  v1 fields filled (`id` is the hit ID, `label` the category name, `confidence` 1 − p(Not)); a custom
  category's hit uses the new `ContactSignalKind.custom` with `label` set to its name, so a 1.2.x
  client that renders `CONTACT_SIGNAL_LABEL[kind] ?? label` shows the admin's name. Evaluate's
  *build* still needs two `Record`/`satisfies` maps updated (`CONTACT_SIGNAL_LABEL` and
  `RULE_FIELDS`).
- **Process and Store upgrade together.** A 1.2.x Process cannot work against a 1.3.0 Store, and no
  request filter fixes that: contract models reject unknown fields (`extra="forbid"`), Process
  validates every claim response, and every 1.3.0 `ReanalysisRequest` carries new fields (`priority`
  is always present). So 1.3.0 requires Process and Store at 1.3.x together, which is how the
  appliance ships them. `ReanalysisClaimRequest.kinds` is a plain filter for a worker that serves
  only some kinds; absent means every kind.
- **`signals.py` (new).** The taxonomy (`SignalTaxonomy`: the eight fixed built-ins of
  `BUILTIN_SIGNAL_CATEGORIES` plus custom categories, subcategories and `SignalField`s), settings
  (`pipeline` v1/shadow/v2, `v1_fallback`, the stage-3 fallback entry), immutable published versions
  and their record (`record_version` concurrency; a stale save is 409 `signal_taxonomy_conflict`), the
  snapshot content (`signal_taxonomy_snapshot.v1`, Store-minted), alert rules, hit feedback, the
  read-time `SignalTaxonomyStatus`, previews and backfills. Pure functions: `taxonomy_digest`,
  `category_digest` (ID, gloss, speaker: hit identity), `subcategory_digest`, `stage1_digest` per
  speaker scope, `stage2_digest`, `stage3_digest` (thresholds are in none of them), the cap check
  `signal_taxonomy_cap_violations`, the text paths Store's detectors scan
  (`signal_taxonomy_text_paths`), the redaction tombstone `redact_signal_taxonomy_text` (each custom
  text path becomes `[REDACTED <n>]`, numbered from 1 in text-path order, built-in constants kept, so
  a redacted version keeps satisfying the name and enum-value uniqueness rules; a redacted version's
  taxonomy must be that function's fixed point), the snapshot reuse check
  `signal_taxonomy_snapshot_current`, `signal_taxonomy_status` (what a taxonomy edit outdates) and the
  alert-condition checks. Node IDs match `^[a-z0-9][a-z0-9_-]{0,39}$`; `none` (stage 1's "none of
  these" option) and `custom` (the kind of every custom-category hit) are reserved category IDs
  (`RESERVED_CATEGORY_IDS`), `other` and `not` reserved subcategory IDs and `quote` a reserved field
  ID. Forbidden field PII classes (caller name,
  account, card, phone, email, address, URL, secret, government ID) are refused; an opaque order or
  receipt number is `none` (decision 22). Examples are plain one-line strings of 1–120 characters.
- **Stage artifacts (`contents.py`).** `signal_categories.v1` (segment grid, sparse stage-1
  probabilities, spans, skip counts, thresholds, `stage1_digests`; derived, no text),
  `signal_subcategories.v1` (per-span decisions; derived) and `signal_extraction.v1` (fields and
  narrowed quotes; masked). `SignalStageProvenance.masked` is always true: v2 purposes are masked on
  every route (`catalog.ALWAYS_MASKED_PURPOSES`). Offsets on a v2 hit are in the **masked** turn
  text. Hit IDs are `<category_id>.<category_digest[:12]>.<transcript_checksum[:8]>.t<turn_id>b<block>`
  (`contents.signal_hit_id`), so threshold, name, description and sibling edits keep IDs and
  feedback.
- **The result.** `ContactSignalsContent` gains `pipeline`, `pipeline_note`, `taxonomy`, `stages`,
  `segmentation` and `stage1_digests`. A v2 result lists `stages` (categorize, subcategorize, and
  extract when planned) instead of `passes`; it exists only when categorize produced its output, and
  a complete one includes every planned stage. `calls.ContactSignalsView` adds read-time context:
  `taxonomy_status`, `feedback`, `alerts`, `text_withheld` and, for admins, `comparison_preview_id`.
- **Multi-segment signals (decision 25).** `ContactSignalView` gains two additive, v2-only fields:
  `parts` (a list of `SignalHitPart`: `turn_id`, `block`, `start`, `end`, `quote`, `char_start`,
  `char_end`) and `span_end` (the last part's end, null when there are no parts). The merge folds
  one speaker's consecutive hits with the same category and stage-2 outcome into the first (the
  anchor keeps the hit ID, quote, offsets and span, so feedback stays joined); each later span is a
  part whose quote was re-verified against its own masked turn, and Store masks it again on read.
  Validators: `span_end` is set exactly when there are parts, is at or after `end` and every
  part's end; parts start in call order after the anchor; no (turn, block) repeats. Added to the
  unreleased 1.3.0 in a dedicated contract edit; the version is unchanged. Three more additive,
  defaulted fields follow from it: `ExtractedFieldView.turn_id` names a part's turn when a merged
  hit's field (and its offsets) came from that part; `SignalPreviewDiff.segments_changed` lists
  hits whose label and fields held but whose segments changed (previews pair hits by any segment
  they cover, so a span that became a part is not "removed"); and `ContactSignalsView.feedback`
  also carries feedback saved on a part's earlier hit ID (the anchor ID with the part's
  `t<turn>b<block>`), shown in Evaluate as a judgement of an earlier segmentation.
- **Jobs.** `contact_signals_categorize` (`primary_host`, purpose `signal_category`),
  `contact_signals_subcategorize` (`primary_host`, `signal_subcategory`) and
  `contact_signals_extract` (`llm_route`, `signal_extraction`), all in the `contact_signals` group;
  the existing `contact_signals_merge` still publishes, so there is still one publisher per group.
  `JobTypeRule.needs_signal_taxonomy` marks the three: each freezes `parameters.signals`
  (`SignalJobParameters`: taxonomy digest, `stage1_mode`, `span_keys`, `fallback_entry_id`,
  `preview_id`) and pins one `signal_taxonomy_snapshot` under input role `taxonomy`, which Store
  checks against the digest at graph creation. `parameters.window` is never set on a v2 job. A
  categorize job in `rederive` mode (a threshold-only edit) runs no model and takes no selection.
- **Reanalysis.** `ReanalysisKind.contact_signals_preview` is the second draft-test kind
  (`DRAFT_TEST_KINDS`): previews and v1/v2 comparisons run into draft slots, affect no group, and are
  refused by `requestReanalysis`. `ReanalysisRequest` gains `priority` (+5 previews, −10 backfills
  and compares, 0 otherwise; claims order by priority, then age), `rescore_signals`,
  `signal_taxonomy_version`, `signal_pipeline`, `signal_backfill_id`, `signal_preview_id`,
  `signal_taxonomy_snapshot_artifact_id` and `preview_result_artifact_id`. A new `contact_signals`
  request widens a pending, unclaimed one instead of returning 409.
- **Everything else.** `ArtifactKind` gains the snapshot (Store-minted, derived) and the three stage
  kinds; `ModelPurpose` gains `signal_category`, `signal_subcategory` and `signal_extraction`;
  `CallListItem` gains `contact_signals_state`, `signal_categories`, `caller_needs` and
  `signal_alerts`, and `CallListQuery` the `signal_category`, `signal_subcategory` and `signal_alert`
  filters; `ReviewStream.SIGNAL`, `ReviewQueueRule.target_signal_alerts` (a SIGNAL rule names at
  least one) and `ReviewQueueItem.signals_version`/`trigger_alert_rule_ids`; `ChangeKind` gains
  `signal_taxonomy` (Process and every reviewer role), `signal_alert_rule` and `signal_alert`;
  `Permission.manage_signals` (admin only, decision 22 Q1); `ErrorCode.signal_taxonomy_conflict`;
  seven audit actions; and eight `ContractParameters` caps (`max_custom_signal_categories` 8,
  `max_active_subcategories` 12, `max_fields_per_path` 12, `max_option_gloss_chars` 40,
  `max_signal_alert_rules` 50, `signal_preview_max_calls` 10, `signal_backfill_max_calls` 500,
  `max_extraction_spans_per_call` 64, increased from 24 for longer calls). Store's save validator reads the caps, so S0 can tighten one
  by changing a parameter; the Pydantic bounds on the models are fixed outer ceilings.
- **14 routes, all Stage 2:** `getSignalTaxonomy`, `listSignalTaxonomyVersions`,
  `getSignalTaxonomyVersion`, `saveSignalTaxonomy`, `saveSignalSettings`, `redactSignalTaxonomyText`,
  `listSignalAlertRules`, `saveSignalAlertRule`, `saveSignalHitFeedback` and `getSignalMetrics`
  (results area); `mintSignalTaxonomySnapshot`, `createSignalPreview`, `getSignalPreview` and
  `createSignalBackfill` (queue area).

**1.3.0 addition: on-device training labels** (team decision 28, `docs/OnDeviceTraining.md`
section 2). Added to the unreleased 1.3.0 in a dedicated contract edit; the version is unchanged.
The customer's LoRA is trained on the device from its own reviewers' labels, is open core and free,
and neither the labels nor the adapter leave the customer's hardware. Every change is additive:

- **`training.py` (new).** `TrainingLabelKind` (`qa_verdict`, `signal_hit`, `speaker_role`), the
  per-kind labels (`QaVerdictLabel`, `SignalHitLabel` with its `SignalSpanAt` segments,
  `SpeakerRoleLabel`), `TrainingSourceRef`, `TrainingLabel`, `TrainingLabelQuery` and
  `TrainingLabelPage`. Labels carry **IDs, enums and versions only**: no transcript text, quote,
  reviewer note, signal note or reviewer identity. Process rebuilds each prompt from the source
  artifacts with the engines' own prompt builders and masking. `training_label_subject` is the
  normative subject (`qa:<call_id>:<criterion_id>`, `signal:<call_id>:<hit_id>`,
  `speaker:<call_id>:<cluster or t<turn_id>>`); the newest `seq` per subject wins, and a withdrawn
  row (a `signal_hit` whose verdicts were both cleared) removes it and resolves no sources.
  `TRAINING_SOURCE_ROLES` lists each kind's required and optional source roles, and
  `TRAINING_SOURCE_ROLE_KINDS` the artifact kind of each role. `EXCLUDED_QA_REASON_CODES`
  (`transcription_error`, `speaker_misattributed`, `policy_exception`) are logged but never trained
  on.
- **One route, Stage 2, results area:** `GET /training/labels` (`listTrainingLabels`), Process
  service key with the new scope **`training:read`** only (no reviewer-session access). The cursor
  is the log's `seq` (`after`); `limit=0` returns only `count_after` and `high_water` and is not
  audited, because the Process console polls it. Every read with `limit > 0` appends the new audit
  action **`training_labels_read`** (after, next_after, item count and counts per kind; never
  content). Store resolves `source_job_id` and `sources` at read time; an unresolvable source
  leaves `sources` empty.
- **`JobListQuery.memory_slot`** filters `listJobs` by memory slot. Process's training start check
  asks for `local_memory` jobs that are queued or running, with `limit=1`.
- **Keys.** `training:read` joins the default Process scopes. An older key without it gets 403
  `insufficient_scope` on this route only; reissue it with `--scope training:read`.

**1.3.0 addition: dual transcription vocabulary** (team decision 33, `docs/DualAsr.md`). Added
to the unreleased 1.3.0 in a dedicated contract edit; the version is unchanged. Parakeet stays the
transcript; Whisper Small, prompted with the customer's vocabulary, only finds candidates, and a
deterministic rule merge replaces the Parakeet words a candidate overlaps in time when the two
sound and spell alike. It never inserts. Open core: nothing reads an entitlement; the industry
pack's terms are subscription content, the customer's own terms work without one. Every change is
additive:

- **`vocabulary.py` (new).** The term rule (`vocabulary_term_problem`: 1–60 characters of letters,
  spaces and `' ’ & . -`, at most six words, starts with a letter, **no digit**, so no term can
  carry a number, email, URL or phone number and a correction can never reintroduce a masked
  value), `normalize_vocabulary_term`, `vocabulary_term_key` (term identity: "Wi-Fi" = "wifi"),
  `AsrVocabularyPack` (an installed industry pack, read-only in Evaluate), `AsrVocabularySettings`
  (`enabled`, default true; `customer_terms`; `disabled_pack_terms`), `AsrVocabularyRecord` (the
  singleton document with derived `effective_terms`, `effective_digest` and `active`),
  `AsrVocabularySave`, `VocabularyMergeRule` (`vocab-merge-rule-v1`: Double Metaphone ≥ 0.70,
  character similarity ≥ 0.60, word-count delta ≤ 1, 0.3 s time slack) and
  `AsrVocabularyParameters`. Pure functions `effective_vocabulary` (pack terms not disabled, then
  customer terms not already present), `vocabulary_digest` and `vocabulary_active` (enabled and
  non-empty).
- **Jobs.** `JobParameters.asr_vocabulary` (asr only): the effective terms, their digest (checked
  by the model and by Store at graph creation), the frozen `candidate_entry` (new
  `ModelPurpose.asr_vocabulary`), the glossary prompt limit and the rule. `JobTypeRule` gains
  **`optional_outputs`**: roles a completion links at most once besides the required ones, never
  publishing and never bindable as an upstream input. `asr` declares `base_transcript`
  (`asr_base_transcript`, `transcript.v1`, the Parakeet transcript) and `vocabulary_pass`
  (`asr_vocabulary_pass`, `asr_vocabulary_pass.v1`, the raw prompted Whisper pass); both kinds are
  fixed at `raw` sensitivity and never projected to reviewers. `transcript` stays the one required,
  publishing output: the merged transcript, so enrichment, PII findings and masking all run on the
  merged text.
- **Content.** `TranscriptContent.vocabulary_correction` (optional, null without a vocabulary):
  `VocabularyCorrection` with `status` `applied` or `base_only` (the vocabulary pass failed or was
  unavailable; `note` and `failure_code` say why; never a failed call), the digest, engines, rule,
  candidate count and `replacements` (`TranscriptReplacement`: turn, word and character span in the
  merged turn, the term and its source, `heard` (Parakeet's words), `candidate_text` (Whisper's),
  both time spans and both similarities).
- **View.** `TranscriptView.vocabulary_correction` (`VocabularyCorrectionView`): status, note,
  counts and `TranscriptReplacementView`s with offsets into the masked view text; `heard` is masked
  and null whenever masking changed it or it holds a digit, and a replacement whose term overlaps a
  masked span is withheld.
- **Routes (Stage 2, tag `vocabulary`).** `getAsrVocabulary` (admin with the new
  **`manage_vocabulary`** permission, or Process `jobs:write`) and `saveAsrVocabulary` (admin,
  expected record version, audited **`asr_vocabulary_saved`**, 409 `conflict`). Packs are installed
  by Store's local `apply-vocabulary-seed` command (`--demo` applies
  `call1/store/seeds/asr_vocabulary_retail_v1.json`). `ChangeKind.asr_vocabulary` reaches Process
  and admins. Two caps: `max_vocabulary_terms` 500 and `max_vocabulary_pack_terms` 2000.

Version 1.4.0 (2026-09-27) adds the **Contact Signals rules engine** (`docs/SignalsEmbeddings.md`,
arm R2: rules decide the category and subcategory, Gemma extracts fields and optionally checks). It is
a minor version because every change is additive and defaulted: with no recipe, or with the default
`detection: model`, every taxonomy, digest, job and result is what 1.3.0 produced. Process and Store
still upgrade together (1.3.0 rule); a 1.3.x Evaluate keeps working (the major is unchanged).

- **Recipes (`signals.py`).** `SignalCategory.recipe` (`SignalRecipe`, optional): `engine`
  (`rules`, or `gemma` to keep today's stages for the category), `filter` (a `SignalRuleExpr` tree of
  `all`/`any`/`not` over `SignalRule`s, depth at most 3), `lexicon` (`SignalLexicon`: `syntax`
  `words` or `regex`, `phrases`, `negation_veto_words` 0-6), `lexicon_weight` (0-1), `threshold`
  (0.05-0.95, default 0.375), `check` (`none` or `gemma`) and a display-only `origin`. The rule
  types are the closed `SignalRuleType` enum (`contents.py`): `similar_to_examples`
  (`min_share`: kNN share of the category at least this), `phrase` (a lexicon matches; with no
  `phrases` it tests the recipe's own lexicon), `speaker` (AGENT or CALLER; must agree with the
  category's scope) and `call_position` (`start_from`/`start_to`: where the segment starts, as a
  fraction of the call). A segment fires when the filter passes and `kNN share + lexicon_weight *
  match` reaches `threshold`; at most two categories fire per segment, largest margin first; the
  subcategory is the kNN vote. Regex phrases are limited to a safe subset
  (`lexicon_phrase_problem`: no look-arounds, named groups, inline flags, back-references or
  unbounded repeats around repeats) and always match case-insensitively. `SignalTaxonomy.rules`
  (`SignalRulesConfig`, optional) pins the example bank (`SignalExampleBankRef`: `bank_id` and the
  pack's `canonical_digest`) and the kNN settings (`SignalKnnSettings`: `k` 10, `temperature` 0.1,
  `taxonomy_example_weight` 2.0). `SignalSettings.detection` (`model` default, or `rules`) is the
  legacy field; `rules_categories(taxonomy, settings)` follows each active category's recipe engine,
  regardless of that field. Lexicon
  phrases are definition text: `signal_taxonomy_text_paths` lists them (after each category's
  subcategories), Store's detectors check them, and the redaction tombstone replaces them.
  `BUILTIN_EDITABLE_FIELDS` gains `recipe`. **Digests:** `taxonomy_digest` leaves `rules` and each
  `recipe` out while they are null, so every earlier version, snapshot and hit keeps its digest; the
  new `recipe_digest(category)` is what rule decisions record. `category_digest` (hit identity) and
  the stage digests are unchanged, so a recipe edit is not "outdated" in `SignalTaxonomyStatus`;
  switching engines needs a `rescore_signals` backfill (docs/SignalsEmbeddings.md section 9.2 status).
- **Stage artifacts (`contents.py`).** `SignalCategoriesContent` gains `rules`
  (`SignalRulesProvenance`: engine version, embedder scheme, bank reference and sizes, kNN settings,
  the rules, checked and Gemma categories, recipe digests, per-category funnel counts
  `SignalRuleCounts`, timings) and `rule_decisions` (`SignalRuleDecision` per rules span: the
  strongest segment, score parts, threshold, `SignalRuleOutcome` per rule, the three nearest bank
  entries as `SignalNeighbour`, the kNN subcategory and its share, and whether a check was asked
  for). The categorize provenance of a run in which rules decided is `catalog_entry_id`
  `call1-signal-rules` (`SIGNAL_RULES_ENTRY_ID`). `SpanSubcategoryDecision` gains `source`
  (`engine` default, or `rules`: passed through from the kNN vote with no model) and `checked`
  (Gemma confirmed or rejected a rule-decided span). No text in any of them.
- **The hit.** `ContactSignalView.why` (`SignalHitWhy`, optional, v2 only): `category_source` and
  `subcategory_source` (`rules` or `gemma`), `check` (`confirmed`, or null) and `rule` (the
  `SignalRuleDecision`, set exactly for rule-decided hits). Set on every hit of a result in which
  the rules engine ran; null otherwise. Evaluate reads it through `getContactSignals`; no new route.
- **Jobs are unchanged.** The rules run inside `contact_signals_categorize` (the embedder under
  `inference_lock`) and the pass-through inside `contact_signals_subcategorize`; the doc's separate
  `contact_signals_candidates` job, `segment_vectors` and the Store-owned bank routes are not built
  (docs/SignalsEmbeddings.md status). Two caps: `max_signal_recipe_rules` 8 and
  `max_signal_lexicon_phrases` 24.

| File | What it holds |
| --- | --- |
| `common.py` | Base model (`extra="forbid"`, response defaults required), ID and digest types, **canonical JSON (RFC 8785) and `canonical_digest`**, the fixed-hostname rule, principals, scopes, roles, pagination, the named timing parameters (`ContractParameters`) |
| `errors.py` | API error codes with their HTTP status; the closed set of safe per-attempt `JobErrorCode`s, their retry classes, `PROVIDER_FAILURE_CODES` and `PRO1_ATTESTATION_CLASS_CODES` |
| `contents.py` | **The JSON content of every machine-produced artifact kind** (transcript, speaker attribution, VAD, validation report, tone, sentiment, embeddings, enrichment, prompt input, QA assessment, verdicts and scorecard, summary segments, syntheses and summary, contact-signal passes and merge, the v2 signal stages (1.3.0), migration record) and the shared vocabulary (speaker roles, verdict statuses, result kinds and states; since 1.3.0 also signal node IDs, field types, the taxonomy reference, span keys and **`signal_hit_id`**) |
| `signals.py` | (1.3.0) Contact Signals v2: the taxonomy and its built-ins, settings, versions and record, the snapshot content, **the digest functions** (`category_digest`, `stage1_digest`, `stage2_digest`, `stage3_digest`), **`signal_taxonomy_cap_violations`**, **`signal_taxonomy_text_paths`**, **`signal_taxonomy_status`**, alert rules and their checks, hit feedback, previews and backfills |
| `custody.py` | Route classes and their rules as data, the pure Call1-operated host rule, the per-job `RouteRecord`, the Pro1 attestation record and failed-attempt evidence, key-source and key-release records with Store's preconditions, the Pro1 connection record and its transitions |
| `catalog.py` | Catalog references, the `FrozenSelection` a job carries, the read-only catalog snapshot Process publishes |
| `usage.py` | Usage records (one per attempt, abandoned rows on lease expiry), token counts with their source, hardware profiles, the report, CSV columns, the admin price table, per-route medians |
| `artifacts.py` | Immutable checksummed artifacts, per-slot versions, link state and orphans, draft-test slots, global and Store-minted kinds, content contracts and their models, **`canonical_content`**, sensitivity, upload and download grants |
| `calls.py` | Conversation registration and the re-registration rule (**`merge_call_metadata`**), the agent label (**`agent_label`**), result groups, **`result_state_inputs`** and **`derive_result_state`**, the call, transcript, evaluation, summary and contact-signal projections Evaluate renders |
| `jobs.py` | The processing queue: statuses and the transition table as data, job types with outputs and groups, dependency edges (`requires`, `after`) and upstream-output inputs, graphs, slot offers, claims, leases, heartbeats, completion order, failure, release, retry, cancel, attempts, group progress, reanalysis requests and draft tests |
| `rubrics.py` | Drafts, immutable published versions and their digests, the rubric snapshot content and the request Store mints it from, draft references; field shapes mirror the pre-split rubric |
| `reviews.py` | The human review queue (rules, items, transitions including SUPERSEDED), verdict overrides, escalation resolution, the stale-write rule, retention, review history |
| `auth.py` | Accounts, WebAuthn credentials and ceremonies (including step-up add-authenticator), invitations, setup codes, break-glass, sessions and the cookie, roles and permissions as data, Process installations and service keys, the console credential |
| `release_trust.py` | Release approvals, pending releases and declines, log scan state, the updater's package flow and verification contract, the attestation policy and platform pins, trust-anchor bundles, the egress allowlist |
| `admin.py` | The audited `AdminState` (settings-only sections, provider endpoints as full URLs) and its change request, TLS and relying-party state, anonymous health and detailed status, backup manifest and restore preflight |
| `events.py` | Audit events (hash-chained), the change feed with its epoch, and the kinds each principal receives |
| `training.py` | (1.3.0) On-device training labels: the label kinds and models, the page and query, **`training_label_subject`**, the source roles per kind, and the QA override reasons never trained on |
| `vocabulary.py` | (1.3.0) The ASR vocabulary for dual transcription: the term rule (**`vocabulary_term_problem`**, no digits), term identity, the industry pack, settings and record, **`effective_vocabulary`**, **`vocabulary_digest`**, **`vocabulary_active`**, the merge rule and the frozen `asr` job parameters |
| `metrics.py` | Executive, per-rubric and review-agreement metrics; since 1.3.0 signal metrics (`SignalMetrics`, **`signal_precision_pct`**) |
| `api.py` | The route table (`ROUTES`) and `build_app()`, which emits OpenAPI from stub handlers and embeds the content schemas and the `x-call1` data tables |
| `openapi.json` | Generated; committed; `python -m call1.contracts.generate --check` fails on drift |
| `constraints.txt` | The Python generator pins as a pip constraints file (`pip install -r requirements.txt -c call1/contracts/constraints.txt`) |
| `../../frontend/src/contracts/store-v1.ts` | Generated TypeScript types for Evaluate (`openapi-typescript`, pinned); `index.ts` beside it adds `Schema`, `Input`, `Output`, `RequestBody`, `ResponseBody` |
| `../../tests/test_contracts.py` | Drift tests, invariant tests, round-trips |

## Principals

| Principal | How it authenticates | What it may do |
| --- | --- | --- |
| **Process service key** (`process_service_key`) | `Authorization: Bearer c1sk_<prefix>_<secret>` over HTTPS. Store keeps `sha256(token)` only, checks one `ServiceScope` per route, and derives the caller's installation from the key: every `installation_id` in a path or body must equal it (403 `forbidden`). Keys are issued to a registered `ProcessInstallation`. | Register sources, upload and read artifacts, create graphs, claim, heartbeat, complete, fail and release jobs, attach late usage, write hardware profiles, claim and reject reanalysis requests, upload attestation evidence and record key releases, record its own log-scan state, report Pro1 verifications, register a shipped trust-anchor bundle as *pending*, publish its catalog snapshot, read admin state, pending releases, approvals, the key-release log, usage medians and the change feed, and (1.3.0) the reviewer-label log for on-device training (`training:read`). Never an admin-state section, approval, decline, adoption, review or identity write. |
| **Reviewer session** (`reviewer_session`) | WebAuthn sign-in creates a server-side session. The browser holds the `__Host-call1_session` cookie (Secure, HttpOnly, SameSite=Strict, Path=/, no Domain; Store keeps its hash) and sends `X-Call1-CSRF` on every state-changing request. Roles: `reviewer` < `supervisor` < `admin`. `ROLE_PERMISSIONS` in `auth.py` is the permission table; each route names the permission it needs, and some add an object rule (`x-call1-object-rule`). | Everything Evaluate does, by role. Admin actions (identity, installations and service keys, trust state, releases, updates, usage report and price table, audit, backup) need `admin`. |
| **Anonymous** | None. | `GET /status` (health only), `GET /contract`, and the first steps of the enrollment and sign-in ceremonies, which carry a single-use invitation token or setup code. |
| **Console credential** | Loopback-only, to Process, stored hashed in Process's protected configuration. | Process operations only (`ConsoleOperation`: import, pipeline, retry/cancel, model defaults, Process-local runtime tuning, viewing). It is **not a Store principal**: no route accepts it. It cannot set provider endpoints, credential names, opt-ins or any trust setting, which are Store admin state that Process only reads. The console's retry and cancel go through Process's own service key (`jobs:control`). |

The split builds passkey sessions from the start, as CLAUDE.md's track list requires. No
shared-token principal exists on `/store/v1`. The pre-split shared admin and reviewer tokens keep
working only on the legacy `/api/v1` app behind its flag. The plan's Stage 2 wording ("audited
since Stage 2 under the shared admin token") therefore does not apply to `/store/v1`: every
audited admin action names the individual admin account from the first release (see "Docs to
reconcile").

Passkey-only means exactly that: there is no password endpoint, no reset and no fallback. Sign-in is
account-first (email, then `allowCredentials`), so non-discoverable credentials on FIDO2 security keys
work. Platform passkeys are allowed where the fleet allows them, `userVerification` is `required`,
and attestation is `none`. Recovery is an admin re-invite, which revokes every credential and
session of the account.

## Versioning and change rules

- `CONTRACT_VERSION` is semver. A patch adds documentation only. A minor adds optional fields, new
  routes, new enum values or new error codes. A major changes or removes anything a client relies
  on, and moves the API prefix (`/store/v1` → `/store/v2`).
- **Contract changes happen only in dedicated contract commits**, on a contract branch, touching this
  directory, the generated files and `tests/test_contracts.py`, and nothing else. Each track then
  picks the commit up. A track that needs a change proposes it; it never patches its own copy.
- `openapi.json` and `store-v1.ts` are generated and committed. Regenerate with
  `python -m call1.contracts.generate`. The generator refuses to run under versions other than
  `api.GENERATOR_PINS` (FastAPI 0.141.x, Pydantic 2.13.x, openapi-typescript 7.13.0), which are also
  recorded in `x-call1.generator` and, for pip, in `constraints.txt`. The repo has no CI yet. Any
  CI added later should run `--check` with `CALL1_REQUIRE_TS_CHECK=1`, so that a missing TypeScript
  generator fails the check instead of skipping it.
- The byte-for-byte drift tests (`openapi.json`, `store-v1.ts`, `--check`, the pins) **skip** under
  other generator versions, because the difference would be generator noise rather than contract
  drift. A newer upstream FastAPI or Pydantic therefore never breaks
  `pip install -r requirements.txt && pytest` for a track. With `CALL1_REQUIRE_GENERATOR_PINS=1` (or
  `CALL1_REQUIRE_TS_CHECK=1`) they fail instead. The CLAUDE.md install command applies `constraints.txt`, so
  the drift tests run by default.
- Response schemas mark every field Store always sends as required. A model used in both requests
  and responses therefore appears as `Name-Input` and `Name-Output`, and `Output<'Name'>` in
  `index.ts` resolves either spelling.
- Clients read `GET /store/v1/contract` at start and refuse to run against a different major.
  Store reports its effective `ContractParameters` there; clients never hardcode timings.
- Enum values are strings, and clients treat unknown values from a newer minor as "unknown", never
  as an error. The exception is job, review-queue, Pro1-connection and reanalysis statuses, which
  are closed sets.

## Canonical JSON and digests

Every digest the contract computes over JSON is `common.canonical_digest(value)`:
`sha256:` + hex SHA-256 of the **RFC 8785 (JCS)** serialization of the JSON value. For a contract
model, the JSON is `model_dump(mode="json")` with defaults and explicit nulls included. This covers
inline artifact checksums (Store stores the canonical bytes, and `size_bytes` is their length),
`RubricVersionRef.digest`, `AdminState.attestation_policy_digest` (the `policy_version` on every Pro1
attempt and key release), hardware fingerprints (`hardware_fingerprint`), audit-event digests
(`audit_event_digest`), and request digests for idempotent replay. Integers beyond 2^53, NaN and
infinities have no canonical form and are rejected. TypeScript clients use any RFC 8785
implementation. `test_canonical_json_matches_rfc8785_vectors` pins the RFC's own vectors.

**JSON artifact content has exactly one form.** An inline payload, or the bytes of an uploaded JSON
artifact, must already be the content model's full dump: `model_dump(mode="json")`, every default
and null present, timestamps as the model emits them (`Z`). `artifacts.canonical_content` checks
this and returns the bytes Store stores. Store never fills defaults or re-serializes, so the
producer's `canonical_digest(payload)` is the artifact checksum, and a reader always gets the
complete document the serialization-mode schemas describe. A payload in any other form is
`validation_failed`; a declared checksum that differs is `checksum_mismatch`.

## Transport, hostname, TLS, status

- HTTPS everywhere, including loopback. Plaintext remote requests are rejected.
- The **Store hostname** is fixed at install: a name in the customer's own DNS zone (for example
  `qa.example.com`). `validate_store_hostname` rejects IP addresses, `localhost`, single labels,
  `.local` names and anything under a Call1-owned domain. It is the WebAuthn **relying-party ID**,
  and the **allowed origins** are `https://<hostname>[:port]` only (`WebAuthnRelyingParty`).
  Changing it orphans every credential, so there is no hostname-migration flow, and setup warns
  about this.
- **Path A** (primary): the installer generates a Call1 CA name-constrained (`nameConstraints`,
  permitted `dNSName`) to exactly the Store hostname and issues the server certificate from it.
  The CA key is a separate file used only for issuance and renewal. **Path B**: the customer's own
  certificate and key for the hostname are imported. `TlsState` reports which, with expiry, health
  and the renewal owner. Untrusted, expired or mismatched certificates block traffic, with no
  bypass.
- `GET /status` is anonymous and returns only `StoreHealth`: contract info, hostname, relying party,
  TLS health and server time. `GET /status/detail` (an admin session or `admin-state:read`) returns
  the build, schema version, admin-state version, **feed epoch** and latest cursor.
- Location neutrality: the Store URL is a deployment setting. Process pulls and exposes no inbound
  endpoint that Store depends on. Artifacts are addressed by ID and checksum, transfer URLs are
  short-lived grants that point at Store, and nothing names a path, host or site of the other
  application. `test_no_field_assumes_shared_files_or_colocation` enforces the field-level part.
  HostedV2.md 1.4 lists the additive fields v2 will add without changing any shape here.

## Error model

Every non-2xx response is `ErrorResponse {code, message, details, retryable, request_id}`.
`ERROR_HTTP_STATUS` maps each `ErrorCode` to one status. The route table lists the codes each route
can return (`x-call1-errors`). Every session route can return `session_expired` and
`account_disabled`. Messages are safe text. These are the conflicts a client must handle:

| Code | When | `details` |
| --- | --- | --- |
| `review_version_conflict` (409) | A review write's `expected_version` is not the call's `review_version` | `current_version` |
| `conflict` (409) | A review write names a non-current machine version; an approval names no pending release; a key release fails a precondition; an adoption names the wrong pending digest | `current_evaluation_version`, `reason` |
| `state_version_conflict` (409) | An admin-state, approval, decline, install or block-clear request's expected version is stale | `current_state_version` or `current_connection_version` |
| `rubric_version_conflict` (409) | A draft save, publish or draft test is based on a stale draft revision or current version | `current_version`, `draft_revision` |
| `claim_token_stale` (409) | Heartbeat, completion, failure or release with a token that is not the job's active claim, after the completion-key lookup found no match | `current_attempt_number`, `status`, `your_attempt_outcome` (succeeded, failed, lease_expired, released, cancelled) |
| `job_cancelling` (409) | A completion arrived after `cancel_requested` was set | none |
| `completion_key_reused` / `idempotency_key_reused` (409) | Same key, different request digest | `original_receipt_id` or `original_id` |
| `cursor_expired` (410) | A change cursor older than `CHANGE_FEED_RETENTION` | `oldest_cursor` |
| `cursor_unknown` (410) | A cursor from another feed epoch (before a restore) or ahead of the latest | `feed_epoch` |

Per-attempt failures use the closed `JobErrorCode` set. `JOB_ERROR_CLASSES` says what Store does.
`transient` retries with backoff while attempts remain. `configuration` and `definitive` stop the
job. A definitive Pro1 code also makes Store block the Pro1 connection **in the same `/fail`
transaction**, and the receipt says so. `terminal` is an outcome (cancelled, validation rejected).
`PRO1_ATTESTATION_CLASS_CODES` never trigger the question-level provider-failure escalation.

## Pagination, cursors, change feed

List routes take `limit` (1–200, default 50) and an opaque `page_token`, and return
`Page[T] {items, next_page_token}`. Ordering is stable per route: newest first for calls, queue
items and audit; `priority desc, created_at asc, id asc` for jobs.

`GET /store/v1/changes?after=<cursor>` returns `ChangeEvent`s (kind, IDs, version, status, never
content) in cursor order, with `next_cursor`, `latest_cursor` and `feed_epoch`. The rules:

- Store assigns cursors in commit order, so no event becomes visible below a cursor already served.
- `next_cursor` is the scan position reached. It is at or after the request cursor even when a
  `kinds` filter matched nothing, so a filtered reader never stalls.
- Store filters by principal (`events.CHANGE_KINDS_BY_PRINCIPAL`). Process gets job, job-group,
  reanalysis, admin-state, Pro1-connection, catalog, rubric and (1.3.0) signal-taxonomy events.
  Reviewers get call, result, review, review-queue, rubric, reanalysis and job-group events and
  (1.3.0) the signal-taxonomy, signal-alert-rule and signal-alert kinds, plus job and catalog events
  from supervisor up. Admins get everything.
- Cursors are resumable for `CHANGE_FEED_RETENTION` (7 days) within one **feed epoch**. Every
  restore starts a new epoch. A cursor from another epoch is 410 `cursor_unknown`, and the client
  re-snapshots. Process also re-checks every claim and spooled receipt when the epoch changes.

Every read snapshot (`CallDetail`) and every write receipt carries the cursor it was taken at, typed
`ChangeCursor`. Polling the feed is the v1 transport; a later push channel carries only cursors,
never data. A `QueueWakeup` carries only a `job_id`: losing or duplicating it cannot lose or
duplicate a job, and a worker must find queued work after a restart without one.

## Idempotency

Each route declares one mode (`x-call1-idempotency`):

| Mode | Rule |
| --- | --- |
| `natural` | The resource's own identity deduplicates. For conversations: the source `(bucket, key, etag)` or content digest (a re-registration can still update call metadata, below). For job outputs: `(producing job, the attempt the claim token belongs to, kind, slot, checksum)`. For other conversation artifacts: `(conversation, kind, slot, checksum)`. For rubric snapshots: `(conversation, rubric_id, version)`. For signal taxonomy snapshots (1.3.0): `(conversation, version, current settings)`. For global artifacts: checksum. For upload commits: the upload ID. Also hardware `fingerprint`, key-release `session_id`, log scan `(installation, log_id)`, and a late usage report per attempt. A replay returns the existing record (`created: false` where the response says so). A deliberate rerun is a different job, so it always gets new artifacts and, once linked, new versions. |
| `body_key` | `idempotency_key` (graphs, each job definition) or `completion_key` (complete, fail and release share one key space, and one call ends a claim). Store looks the key up **before** it checks the claim token: the same request digest returns the original receipt with `replayed: true`, whatever the lease state; a different digest is 409. Keys are scoped to the installation and, for jobs, to the conversation. |
| `header` | `Idempotency-Key` is required (reanalysis requests and draft tests from Evaluate), scoped to the session and call. |
| `expected_version` | The body carries the version it was read against. Store applies the write only if it still matches, in one transaction, and returns the new version. |

A stable key per logical job definition means duplicate S3 notifications produce one graph.

### Conversation re-registration (1.1.0)

`POST /conversations` deduplicates by source identity, but it is not a pure replay when the
metadata differs. Re-uploading the same recording with different call metadata updates the
call's metadata:

1. Store finds the conversation by `source.dedup_identity` and computes
   `calls.merge_call_metadata(stored, incoming)`. Only the `CallMetadata` fields present in the
   request body replace the stored ones. Absent fields keep their stored value, so a re-upload
   that names no agent never resets a known agent to `Unknown`. An explicit `null` clears an
   optional field.
2. If nothing changed, the call is a pure replay: `created: false`, `metadata_updated: false`, no
   audit event and no change event.
3. If something changed, Store does all of this in one transaction: it saves the merged metadata;
   refreshes every projection that shows it (the call list and detail, review-queue items and
   escalations); writes an audit event `call_metadata_updated` (actor: the Process installation;
   target: the call, or the conversation when it has no call; details: `conversation_id` and
   `updated_fields`, the changed field names only, never values); and emits a `call` change
   event with `status: metadata_updated`. The response carries the updated conversation,
   `created: false`, `metadata_updated: true` and `updated_fields`.
4. **Nothing is reprocessed.** Store creates no graph and no reanalysis request, marks no result
   stale, and leaves `review_version` alone. Metadata is not an analysis input, except
   `agent_channel`, which Process reads when it builds a graph. A changed `agent_channel` affects
   only graphs built after the update. To re-analyze with it, a reviewer requests reanalysis.
5. Concurrent re-registrations of one identity apply in commit order, field by field; the last
   one wins.

The source itself (`ingestion_kind`, `source`) never changes on re-registration. Only
`call_metadata` does.

## Artifacts, slots and content

- Every JSON kind has a content model in `contents.py` (or `rubrics.RubricSnapshotContent`),
  mapped by `ARTIFACT_CONTENT_MODELS` and emitted as an OpenAPI component schema. Store validates
  inline payloads against it. Audio, the Pro1 evidence bundle and the trust-anchor bundle are opaque
  bytes. **A content model never carries its own Store-assigned identity** (artifact ID, version,
  call ID). It may reference other, already committed artifacts by `ArtifactRef` and name job IDs,
  which exist before the producing job runs.
- `EvaluationView`, `SummaryView` and `ContactSignalsView` are their content model plus
  `call_id`, `artifact_id` and `version`. `TranscriptView` is composed by Store from the linked
  transcript, speaker-attribution, text-sentiment, tone and VAD artifacts.
- **Slots and versions.** Versions and supersession are per `(conversation, kind, slot)`. The slot
  is `""` for a per-call singleton, the criterion ID for a QA assessment, `segment:<n>` for a
  summary segment, `<pass>:<window>` for a contact-signal pass, `rubric:<rubric_id>:v<version>`
  for a rubric snapshot, and (1.3.0) `signals:v<version>` for a signal taxonomy snapshot. The three
  v2 stage artifacts are per-call singletons (slot `""`). A live slot is at most `SLOT_MAX_LENGTH` (128) characters. A job output is
  **linked** only in the completion transaction that references it; only then does it get a
  version and supersede the previous linked artifact of its slot. An artifact with no producing job
  (source, rubric snapshot, migrated, global) is linked at commit. A committed artifact no
  completion links is an orphan: it is hidden from listings unless `include_unlinked`, supersedes
  nothing, and is deleted after `ORPHAN_ARTIFACT_RETENTION`.
- **Draft-test slots.** Every artifact of a draft-test graph, and the draft snapshot Store mints
  for it, lives in `draft:<request_id>:<slot>` (`artifacts.draft_test_slot`). The prefix is
  reserved: Store refuses a draft-test output outside its request's prefix, and any artifact that
  does not belong to that draft test inside it (`validation_failed` at creation, `graph_invalid`
  at completion). A draft test therefore never
  versions or supersedes the call's live assessments, scorecard or snapshots. Listings leave these
  slots out unless `include_draft_tests` (or a `draft:` slot filter) asks for them.
- **Rubric snapshots are Store's** (`STORE_MINTED_KINDS`). Before creating a graph, Process calls
  `POST /conversations/{id}/rubric-snapshots` (`mintRubricSnapshot`, `jobs:write`) with
  `{rubric_id, version}`. Store copies that published version (active or retired) into a linked
  `rubric_snapshot` in slot `rubric:<rubric_id>:v<version>`, idempotently per conversation and
  version. A draft test's snapshot is minted by `testRubricDraft`. Process never uploads one:
  `InlineArtifactCreate` and `UploadGrantRequest` refuse the kind. Every QA job pins its snapshot
  under input role `rubric` (`RUBRIC_INPUT_ROLE`), and graph creation checks that it names the job's
  rubric, version or draft revision, and digest (`graph_invalid`).
- **Signal taxonomy snapshots are Store's too** (1.3.0). Process calls
  `POST /conversations/{id}/signal-taxonomy-snapshots` (`mintSignalTaxonomySnapshot`, `jobs:write`)
  with `{version}`; Store copies that published, unredacted version and the current settings into a
  linked `signal_taxonomy_snapshot` in slot `signals:v<version>` (sensitivity `derived`). A repeat
  returns the linked snapshot only while it still matches the version and the current settings
  (`signals.signal_taxonomy_snapshot_current`); after a settings change Store links a new version in
  the same slot, so a reanalysis never runs with stale settings. A preview's
  snapshot is minted by `createSignalPreview` in `draft:<request_id>:signals:preview`. Every v2 job
  pins one under input role `taxonomy` (`signals.SIGNAL_TAXONOMY_INPUT_ROLE`) and freezes its digest
  in `parameters.signals.taxonomy_digest`; graph creation checks the two (`graph_invalid`).
- **Global artifacts** (`attestation_evidence`, `trust_anchor_bundle`) belong to no conversation.
  They are uploaded once through their own routes, addressed by digest, and readable by an admin
  with `read_audit` (the evidence viewer and `scripts/verify_attestation.py`).
- A call's media fields (`CallRecordView`) come from the `validation_report` and `vad_metrics`
  outputs of `validation_vad` and stay null until they publish.

## Dependencies and inputs

- A job depends on another through an edge of one of two kinds. **`requires`**
  (`requires_refs`, `requires_job_ids`): the upstream must SUCCEED. **`after`** (`after_refs`,
  `after_job_ids`): the upstream must be terminal, whatever the outcome. Merges and assemblies that
  can publish a partial result (contact signals) use `after` edges on the passes they combine.
- An input is either a committed artifact pinned by checksum (`JobInput.artifact`) or **an upstream
  job's output by role** (`JobInput.upstream = {ref | job_id, output_role}`). Inputs from `after`
  edges are `optional`. The output role must be one the upstream's type declares
  (`JOB_TYPE_RULES[...].outputs`). When Store releases a dependent, it resolves every upstream
  input to the linked output artifact and pins its checksum (`Job.resolved_inputs`).
  `ClaimedJob.inputs` carries the resolved artifacts, and `ClaimedJob.upstream` carries every
  upstream's outcome.
- A completion must link exactly one committed artifact per declared output role.
- A follow-on dependency (`NewDependency`) can also bind an input. With `input_role` and
  `output_role`, Store appends `JobInput{role: input_role, upstream: {job_id: <new job>,
  output_role}}` to the dependent in the completion transaction and resolves and pins it on release.
  The role must be new on the dependent (`graph_invalid`), and the output role one the new job's
  type declares. A triggered escalation uses this, so the scorecard receives the escalation's
  `assessment` in `resolved_inputs` and `ClaimedJob.inputs` like every criterion's.
- A dependent on a `requires` edge whose upstream is FAILED or CANCELLED stays BLOCKED
  (**dead-blocked**; `BlockingReason.dead`) until that upstream is retried. It is never failed for
  it. Store releases `after`-edge dependents when the upstream becomes terminal, including inside
  the failure transaction (`FailureReceipt.released_job_ids`). Cancel cascades to dependents by
  default; with `cascade: false`, `after`-edge dependents are released instead.
- `JobGraph` and `Job` expose every edge. There are no cross-conversation edges.

## Lease and claim semantics

Named parameters (`ContractParameters`, reported at `/contract`): `LEASE_DURATION` 300 s,
`HEARTBEAT_INTERVAL` 100 s, `LEASE_GRACE` 30 s, `MAX_CLAIM_BATCH` 16, `DEFAULT_MAX_ATTEMPTS` 3,
`RETRY_ATTEMPT_GRANT` 1, `RETRY_BACKOFF_BASE` 30 s × `RETRY_BACKOFF_FACTOR` 2 up to
`RETRY_BACKOFF_MAX` 3600 s, `REANALYSIS_CLAIM_LEASE` 300 s, `ORPHAN_ARTIFACT_RETENTION` 24 h.

1. **Admission (Store side).** On every claim call, whoever asks, and whenever admin state or the
   Pro1 connection changes, Store fails at admission (`QUEUED → FAILED`, `admission_rejected`, no
   attempt consumed) every QUEUED job whose frozen route's opt-in is disabled or whose connection
   is no longer in admin state (`route_disabled`). It does the same for every `call1_confidential`
   job while the Pro1 connection is BLOCKED (`pro1_connection_blocked`). A `pending_verification`
   connection only holds Pro1 jobs (`waiting_reason: pro1_pending_verification`). The fix is an
   admin change followed by a manual retry.
2. A job is **eligible** when it is QUEUED, `next_run_at` has passed, every `requires` upstream
   SUCCEEDED and every `after` upstream is terminal, its inputs are resolved, and the worker's
   `WorkerCapabilities` cover its `execution_class` (`primary_host` jobs go only to the designated
   primary host), route class and frozen catalog entry.
3. **Slot offers.** Process reserves slots before asking and sends them as
   `slot_offers: [{memory_slot, outbound_connection_ref?, count}]`. Store returns at most `count`
   jobs per offer, each needing exactly that slot and, for outbound, exactly that connection.
   `ClaimedJob.slot_offer_index` names the offer it consumed. Process releases unused slots, and
   Store never holds a lease for a slot.
4. **Claim** is atomic. Store orders eligible jobs call by call: the `conversation_id` affinity hint
   first, then the priority of the reanalysis request the job's graph fulfils (0 without one),
   then the oldest graph, then the job's own `priority desc, created_at asc, id asc`. It sets
   RUNNING, increments `attempt_count` and `claim_count`, records `LeaseInfo` with
   `sha256(claim_token)` and a never-reused `attempt_number`, and returns the job, its resolved
   inputs, upstream outcomes, `final_attempt` and the token. Two workers never receive the same
   active claim.
5. **Release** (`POST /jobs/{id}/release`). Before any inference starts, Process can end a claim
   without consuming an attempt. `requeue` (slot lost, input fetch failed:
   `RELEASE_REQUEUE_CODES`) returns the job to QUEUED with `not_before`. `reject` (credential
   missing, key anchor unavailable, model not installed or unqualified, context limit:
   `RELEASE_REJECT_CODES`) fails it. Either way `attempt_count` goes back down, the claim is
   recorded as a `released` attempt that does not count, and there is no usage row. Transient Pro1
   conditions (unreachable, revocation unavailable, release unapproved) use `/fail`, so they back
   off and exhaust attempts as Pro1ConfidentialInference.md §4 says.
6. **Heartbeat** renews `expires_at` and returns `cancel_requested`. A token that is not the active
   claim is `claim_token_stale`.
7. **Expiry.** Past `expires_at + LEASE_GRACE`, Store requeues the job with backoff if attempts
   remain, or fails it with `lease_expired`. If cancel was requested, the job becomes CANCELLED
   (`lease_expired_after_cancel`). In every case Store synthesizes the attempt's usage row with
   outcome `abandoned`. The old token is stale from then on. A worker that spooled output during a
   Store outage must re-check its claim before publishing and cannot overwrite a newer attempt; it
   may attach its measurements once to its own abandoned row
   (`POST /jobs/{id}/attempts/{n}/usage`).
8. **Completion** runs `jobs.COMPLETION_TRANSACTION_STEPS` in one transaction, in this order:
   1. Look up the completion key (a replay returns the stored receipt).
   2. Check the active claim, then `cancel_requested`, which gives 409 `job_cancelling`.
   3. Verify the outputs (in a draft-test graph, their `draft:` slots) and the provenance, and that
      the usage outcome is one the job type allows. A `failed` outcome is accepted only on the
      claim's final attempt with a `PROVIDER_FAILURE_CODES` code.
   4. Link the outputs (versions per slot).
   5. Write the usage row, provenance and receipt.
   6. Mark the job SUCCEEDED.
   7. Project the result, except in a draft-test graph (no projection, no review-queue items).
   8. Insert follow-on jobs, then `add_dependencies` and their input bindings. A dependent must be
      BLOCKED and directly require the completing job.
   9. Compute the follow-ons' initial status. They may require the completing job, which is
      already SUCCEEDED.
   10. Release dependents, write change events and return the receipt.

   Object uploads are outside that transaction; an unlinked upload is an orphan.
9. **Failure** records the usage row (its outcome is `usage_outcome_for(error_code)`, and its
   error code equals the failure's) and the provenance, then applies `JOB_ERROR_CLASSES`. Workers
   never report `lease_expired`. **Cancel**: a RUNNING job learns of it on heartbeat or as
   `job_cancelling`, then fails with `cancelled`. **Manual retry** re-queues one FAILED job, or
   blocks it until its upstream is satisfied, adds `RETRY_ATTEMPT_GRANT` attempts and bumps
   `retry_generation`; nothing else reruns. CANCELLED is final; rerun as a new graph
   (`retry_group`).
10. `attempt_count` counts inference executions started, which are claims not released.
    `WAITING_PROVIDER` is reserved and unreachable in v1. `JOB_TRANSITIONS` is the complete state
    machine, and anything else is `invalid_transition`.

## QA assessments that cannot answer

The pre-split router recorded a provider failure or an invalid answer as a FLAGGED verdict, and
escalated when the criterion's `escalation_when` listed that trigger. The split keeps that
behaviour:

- An answer that fails schema or quote validation is an outcome, not a job failure. The
  `qa_criterion` or `qa_escalation` job **completes** with a FLAGGED `qa_assessment`
  (`trigger: invalid_answer`) and usage outcome `validation_rejected`.
- A provider fault in `PROVIDER_FAILURE_CODES` (`provider_error`, `provider_timeout`,
  `pro1_unreachable`, `pro1_service_error`) is reported with `/fail` while attempts remain. On the
  job's final attempt (`ClaimedJob.final_attempt`), the job instead completes with a FLAGGED
  assessment (`trigger: provider_error`, the attempt's `error_code` set) and usage outcome `failed`.
  No other code may complete with `failed`.
- **On the Pro1 route** the codes are `pro1_unreachable` and `pro1_service_error`
  (`PRO1_PROVIDER_FAILURE_CODES`). A FLAGGED Pro1 completion uses one of them, never
  `provider_error` or `provider_timeout`, and a completion on any other route never uses them.
  Pro1ConfidentialInference.md §4 keeps the configured escalation for exactly these two. The FLAGGED completion keeps its evidence:
  `provenance.pro1_failure` with the same code, plus the attestation record and key-release
  reference for `pro1_service_error`, which comes from a sealed response. A `pro1_unreachable`
  attempt that never reached verification has no attestation record.
- In both cases the completion carries `follow_on` with a `qa_escalation` job and the scorecard's
  new dependency, bound to the escalation's `assessment` output, when the trigger is in the
  criterion's `escalation_when`, so the scorecard still assembles with the criterion flagged for
  review.
- Attestation-class codes (`PRO1_ATTESTATION_CLASS_CODES`) never escalate. The job FAILS, the
  scorecard is dead-blocked, and QA shows `failed` ("Needs attention") until the admin fixes the
  cause and the criterion is retried.
- `JobTypeRule.completion_outcomes` lists the usage outcomes a completion of each type may carry.
  Store rejects others.

## Result groups and derived states

Every job type counts toward one result group (`JobTypeRule.group`), or toward `pending_work` only
(embeddings). Exactly one type per group publishes (`JobTypeRule.publishes`). Store builds each
group's inputs with `calls.result_state_inputs` and derives its state with
`calls.derive_result_state`, the normative rules, and Evaluate and Process only render it.
`result_state_inputs` drops **draft-test graphs** first, so a draft test that runs, fails or succeeds
never makes the call's QA stale, failed or pending and never sets its `failure_code`. Since 1.3.0
a `contact_signals_preview` graph is a draft-test graph too (`jobs.DRAFT_TEST_KINDS`).

**Contact signals v2 (1.3.0).** The three v2 stage jobs (`contact_signals_categorize`,
`_subcategorize`, `_extract`) count toward the `contact_signals` group, and `contact_signals_merge`
stays its only publisher, so v1 and v2 results share one group and one version history. The merge
takes the stages on `after` edges (a missing stage makes the result `partial`, naming it) but
`requires` `pii_findings` in v2, because it re-verifies every quote against the masked turn. So a
failed categorize fails the merge (`input_unavailable`), and a failed `enrichment` dead-blocks every
v2 job and the merge: the group reads `failed` until enrichment is retried. Editing the taxonomy
never makes the group `stale`; Store labels an older result with `ContactSignalsView.taxonomy_status`
instead. Draft-test
jobs are also left out of `PendingWorkIndicator`, `GroupProgress` and `settled`; their progress is
`DraftTestResult.state`.

1. Nothing was ever requested or published for the group: `disabled`.
2. A version is published. The state is `stale` while a reanalysis affecting the group is pending
   or its newer graph's publisher is in progress; otherwise it is that version's `available` or
   `partial`. A newer graph that ends without a result leaves the old version shown with
   `failure_code` set.
3. Nothing is published. The state is `failed` once the newest publisher is FAILED, CANCELLED or
   dead-blocked ("Needs attention"); otherwise it is `pending` ("Analyzing").

`jobs.REANALYSIS_KIND_AFFECTS` says which groups a request makes stale from the moment it is
created. A kind lists only groups whose publisher the fulfilling graph reruns, because stale ends
through that publisher. `qa_draft_test` affects none. `speaker_correction` does not list the
transcript, because `asr` is not rerun. Its graph's `speaker_attribution` job carries the correction
in `parameters.speaker_correction`. It is then a code stage with no model selection, and it writes a
`reviewer_correction` attribution. `TranscriptView.speaker_attribution_version` changes when that
attribution is linked, and until then Evaluate shows the correction as pending from the request. A conversation is **settled**, the plan's "persisted
terminal state", when every job is terminal or dead-blocked. Every group is then `available`,
`partial`, `failed` or `disabled` (`SETTLED_RESULT_STATES`, `PendingWorkIndicator.settled`).

## Reanalysis and draft tests

- `POST /calls/{id}/reanalysis-requests` (header idempotency) creates a durable request for new
  machine results, never a draft test. Process claims requests under `REANALYSIS_CLAIM_LEASE`.
  **The graph that fulfils a request carries its claim token** (`JobGraphRequest`), and Store
  creates the graph and fulfils the request in one transaction. A stale claim is refused, so one
  request yields at most one graph. `REANALYSIS_TRANSITIONS` includes `claimed → pending` on claim
  expiry. There is no separate fulfil route.
- **Draft tests** are `POST /rubrics/{id}/draft/tests` (supervisor, `manage_rubrics`). Store
  snapshots the stored draft into a `rubric_snapshot` artifact (slot
  `draft:<request_id>:rubric:<rubric_id>:r<revision>`) in the same transaction and creates a
  `qa_draft_test` request carrying a `DraftRubricRef`. QA jobs in that graph freeze
  `parameters.draft_rubric` instead of `rubric`, write their outputs in `draft:<request_id>:` slots,
  and their completions carry `result: null`. Store records the scorecard as
  `draft_result_artifact_id` and never projects it as the call's QA: no projection, no review-queue
  items, no supersession of the call's artifacts, and no effect on the call's result states
  (`result_state_inputs`). Rubric Studio reads it at `GET /reanalysis-requests/{id}/draft-result`.
- **Signal previews and comparisons** (1.3.0) are the second draft-test kind,
  `contact_signals_preview`. `POST /signals/previews` mints a preview snapshot of the (unsaved)
  taxonomy per call in `draft:<request_id>:signals:preview` and creates one request each (priority
  +5); a compare backfill or a shadow-mode companion creates them against the published version
  (priority −10). The merge output is recorded as `preview_result_artifact_id` and read through
  `GET /signals/previews/{id}`, never projected.
- **Speaker corrections** (`POST /calls/{id}/speaker-corrections`) create a `speaker_correction`
  request. Process freezes its `SpeakerCorrection` on the graph's `speaker_attribution` job and
  reruns the downstream publishers the kind lists.

## Data custody and route classes

`ROUTE_CLASS_RULES` is the admission table as data. Every model-backed job freezes a
`FrozenSelection` whose `RouteRecord` names the route class, provider type, destination host,
admin-state connection and masking setting. Every attempt records the route it used.

**The host rule.** `custody.call1_operated_host(host, pro1_endpoint_hosts)` is the Stage 0 rule
stated as a pure function with no environment lookups. It matches a host equal to or under
`call1.cc` or the configured Pro1 endpoint host, a missing host, and any public IP literal, IPv4 or
IPv6. It handles bracketed IPv6. `test_route_host_rule_is_the_stage0_guard` runs 19 hosts, including
public and private IPv6, with and without `CALL1_PRO1_ENDPOINT`, and proves that it equals
`call1.question_models.is_call1_operated` on the identical host. The Pro1 endpoint host comes from
admin state (`Pro1OptIn.endpoint_base_url`), so Store, Process and Evaluate evaluate the same rule
(`x-call1.call1_operated_domains` is exported for Evaluate's forms). The contract no longer imports
Process code. When Stage 2 adds settings validation, `is_call1_operated()` should delegate to this
function, so that literally one function exists (see "Docs reconciled at WF-A").

`RouteRecord` validation enforces these rules:

- MLX runs in-process, and appliance Ollama is on loopback.
- Nothing else may be on loopback.
- A non-appliance route names its admin-state connection.
- A Call1-operated host is allowed only on `call1_confidential`.
- An unmasked non-appliance route cites the audit event that allowed it.
- ML stages (`primary_host`) use the appliance route, so audio never leaves the site.

At graph creation Store also checks each route against admin state: the Pro1 endpoint host, the
connection's host and its credential name (`route_not_permitted`).

**Endpoints are admin state.** `ByokProvider.base_url`, `CustomerLanHost.base_url` and
`Pro1OptIn.endpoint_base_url` are full non-secret HTTPS base URLs (scheme, port, path). They carry
no credentials, query or fragment, and never point at loopback. They change only through the audited
`route_opt_ins` section (`endpoint_changed`). Process connects only to those URLs, with the
credential variable named there. `RouteOptIns` also rejects a BYOK or LAN host under the Pro1
endpoint's domain.

A `call1_confidential` attempt that passed verification carries a `Pro1AttestationRecord`: evidence
digest and artifact, policy version, trust-anchor bundle digest, approval, release ID, manifest
digest, SVN, log entry, checkpoint, revocation sequence, platform, GPU claims, session, key ID,
key-release reference, model ID and weights digest. It also carries the matching `key_release_ref`.
A failed Pro1 attempt carries `Pro1AttemptFailureEvidence`: the failed check, the code, the policy
version, and the rejected evidence and parsed release when available. It also has the full record
when the failure came after key release. Its `error_code` equals the `/fail` request's (or, on a
FLAGGED completion, the usage row's), and every `pro1_*` failure carries it. An attempt that a
cancel, crash, input, resource or publication failure stopped before verification records
`failed_check: interrupted` with that code (`PRO1_INTERRUPTION_CODES`). After verification, the
attestation record is enough. No other route carries either. Usage rows for Pro1 carry
`session_setup_seconds` against the session's first attempt.

## Key release and release trust

- **Key release.** Before the wrapped session key leaves the appliance, Process uploads the
  evidence bundle once (`POST /attestation-evidence/uploads`, by digest). It then writes
  `KeyReleaseInput` (`POST /key-releases`, audited). Store checks `KEY_RELEASE_PRECONDITIONS` in
  order, each with its error and `details.reason`: the opt-in is enabled, the connection is not
  blocked, the approval is active and matches the release and manifest, the policy version is
  current, the key source matches `KeyManagerConfig`, the evidence is stored, and the session is
  released only once. These checks are defense in depth; Process enforces the same conditions
  before it wraps. No field anywhere holds a key.
- **Pro1 connection.** `custody.Pro1Connection` is its own record with its own
  `connection_version`, outside `AdminState`. Store is its only writer, and
  `PRO1_CONNECTION_TRANSITIONS` is the whole machine:
  - `ready → blocked`: Store does this inside `/fail` on a definitive Pro1 code, or on a Process
    verification report with one.
  - `blocked → pending_verification`: only through the admin's audited `clear-block`.
  - `pending_verification → ready`: only on a passing `Pro1VerificationReport` from Process, for
    example the attested connection test.
  - `ready → pending_verification`: when the route is disabled.

  Enabling or disabling the route never clears a block. Process reports verification outcomes as
  events and never sets a status.
- **Approvals.** `ReleaseApprovalRequest` names a pending release, `(kind, release_id,
  manifest_digest)`, that Process's log scan recorded. It carries no evidence of its own; Store
  copies the evidence the scan recorded. Anything else is 409 `conflict`, `not_pending`.
  `ReleaseApproval` stores the approved manifest's SHA-256, log ID and index, checkpoint, SVN and
  that evidence. The verifier accepts a Pro1 release, and the updater an appliance package, only
  when the offered manifest digest equals an approval's; a matching release ID is not enough. An
  admin can also **decline** a pending release (`POST /admin/releases/pending/decline`, audited).
  Approvals, withdrawals, declines, installs and every admin-state section change are
  expected-version writes by an `admin` session, and each writes an audit event.
- **Release floors.** The Pro1 floor lives only in `AttestationPolicy.minimum_release_svn`, so the
  policy digest covers it. The appliance floor lives only in
  `AdminState.minimum_appliance_build_svn`. The `attestation_policy` section's payload
  (`AttestationPolicySettings`) cannot change the floor. An approval may raise its kind's floor.
  The `minimum_release_svn` section sets either floor, and lowering one needs `confirm_lower`. The
  Pro1 session lifetime lives only in the policy.
- **Policy pins.** `AttestationPolicy.platform_identity` pins the expected SEV-SNP ID-key and
  author-key digests, Azure `vm-configuration` claims, TDX MRCONFIGID, MROWNER and MROWNERCONFIG,
  and the confidential VM firmware PCRs 0–7 (in the policy or via the trust-anchor bundle). All of
  them fall under the policy digest recorded per attempt.
- **Log scan state.** Each installation records its own scan (`PUT /release-trust/log-scans/{log_id}`,
  keyed by installation and log): last verified tree size, checkpoint digest and cosignature time,
  revocation sequence and issue time, pending releases with their evidence, revocations seen, and
  an equivocation flag that is audited.
- **Trust anchors.** Process uploads the bundle its build shipped as a `trust_anchor_bundle`
  artifact and registers it (`POST /release-trust/trust-anchor-bundles`, audited
  `trust_anchor_pending`). That fills `trust_anchors.pending`, which is inert: it does not bump
  `state_version`, and the verifier keeps using the adopted bundle, including its log key and URL.
  An admin adopts it through the `trust_anchors` section, naming the `pending_digest` they
  reviewed. Process fetches the adopted bundle by artifact ID and checksum and refuses any other
  bytes.
- **Updater.** The updater is the installed build's verifier, running on the Store host inside
  Store:
  1. The admin stages a downloaded package (`POST /admin/updates/packages`) and uploads its bytes.
  2. The commit runs the verifier before any package code runs, and Store writes the
     `UpdaterVerificationRecord`. No route accepts a verdict from a browser or from Process.
  3. The admin installs (`…/install`, audited), which is allowed only while the latest verdict is
     `accept` and the approval is still active.

  `RunningBuild` at startup flags a build no admin approved. Installing on other Process hosts in a
  multi-host layout is Stage 5 packaging.
- **Admin state.** `AdminState` holds the attestation policy, the appliance floor, trust anchors,
  the key manager, route opt-ins and endpoints, masking and the notification relay. It changes only
  through `POST /admin/state/changes`, one section per request, audited. Section payloads are
  settings only. Who changed a section, when, and the audit event are in `section_changes`, which
  Store writes.
- **UI placement.** The approval screen, the policy editor, trust-anchor adoption, pending-release
  declines, the key-release log and clear-block are admin actions in **Evaluate's admin area**,
  because the plan says the console can display these but not change them. Process Models settings
  *displays* the connection state, pending releases with their evidence, approvals and the
  key-release log, which it reads with `admin-state:read`. It links out with **Open in Evaluate**.
- **Egress allowlist.** `EgressAllowlist` entries say what each destination is for, when it is
  needed and whether it carries customer data. Only the Pro1 front end (as ciphertext), BYOK and
  customer-LAN model hosts, and the customer's SMTP relay (reviewer emails and invitation links)
  carry customer data. The relay may use `smtp+starttls` (587) or `smtps` (465).

## Usage

- Every attempt has exactly one usage row: written by completion or failure, or synthesized by
  Store as `abandoned` on lease expiry. A released claim is not an attempt. The outcome follows
  from the ending call: `succeeded` on completion (plus `validation_rejected` or `failed` for QA
  assessments recorded as FLAGGED); on failure, `usage_outcome_for(error_code)`.
- `POST /admin/usage/report` gives the rollup, with optional cost **estimates** from the price
  table. `POST /admin/usage/records.csv` exports the raw rows (`USAGE_CSV_COLUMNS`, RFC 4180), so a
  rollup recomputed from the CSV equals the report. `GET/PUT /admin/usage/price-table` holds the
  admin's dated prices in Store: per catalog entry, hardware profile or route class; per 1k tokens,
  per inference hour, or per provider billing unit. Nothing is hardcoded, and every dollar figure
  is labeled an estimate.
- `GET /usage/medians` (`usage:read`, or an admin) gives Process Models settings its measured
  medians per catalog entry, route and hardware profile. It returns aggregates only.

## Reviews and expected versions

`CallReviewState.reviewed_score` is an optional additive read model: the pinned rubric scored
with the latest current-version override per criterion. It is absent when no matching overrides
exist. Call-list scores and rubric metrics use the same projection. The immutable evaluation
artifact remains the original machine assessment; reanalysis resets the projection and older
decisions remain in history without affecting the new score.

Machine results and human decisions are separate versioned records. `EvaluationView.version` is the
machine version; `CallReviewState.review_version` is bumped by every human write. Each write
(`VerdictOverride`, `EscalationResolution`, `ResolveRequest`, `RetainReviewRequest`,
`SpeakerCorrectionRequest`) carries its expected version. A stale one is `review_version_conflict`.

**The stale-write rule.** Every write names the machine version it judged, and it must be the
call's current evaluation version, or Store answers 409 `conflict` with
`current_evaluation_version`. When a new machine version commits, Store:

- marks every unresolved queue item of that call SUPERSEDED (with `stale: true` and
  `superseded_by_item_id`);
- creates the items the rules call for against the new version;
- marks resolved items `stale: true`;
- marks the call's review `stale` if any decision referenced the old version.

A supervisor may then `retain` the call-level decisions. Nothing is resolved against a non-current
version. Resolving an item assigned to someone else needs `resolve_any_review`. Per-criterion
overrides may carry an `OverrideReasonCode`, which is optional in v1, so no contract change is
needed after cutover.

The human queue (`/review-queue/...`, `reviews.py`) and the processing queue (`/jobs/...`,
`jobs.py`) share no type, status value, transition or endpoint.

## Identity details

- **Sessions.** Authorization is evaluated on every request from the account's current role and
  status. Disabling an account, re-inviting it and restoring under a new hostname revoke its
  sessions in the same transaction, and a demotion applies from the next request. An admin can
  revoke one lost credential (`DELETE /admin/accounts/{id}/authenticators/{authenticator_id}`).
  `session_id` is a non-secret handle, never the cookie value.
- **CSRF.** `SessionInfo.csrf_token` is session-bound, not one-time. Sign-in and enrollment return
  it, and so does `GET /auth/session` (same origin; SameSite=Strict and CORS keep other sites from
  reading it), so a reload or a new tab recovers it. Keep it in memory, never in web storage.
- **Add another authenticator** is a step-up ceremony. Begin returns an assertion challenge for the
  account's existing credentials and the registration options. Finish verifies a fresh
  user-verified assertion from an existing credential within `CHALLENGE_LIFETIME`, and only then
  the new registration.
- **WebAuthn payloads** are exactly what `@simplewebauthn/browser` (v13/v14) returns, including
  `authenticatorData`, `publicKey`, `publicKeyAlgorithm` and open-ended transport strings.
  `test_simplewebauthn_browser_payloads_validate_unchanged` pins a v14 fixture.
- **Anti-enumeration.** Sign-in begin never returns an empty `allowCredentials`. An unknown,
  disabled or re-invite-pending email gets deterministic decoys with the same shape and timing.
  Finish accepts only a credential of the account bound to the ceremony.
- **Authenticator paths** use `{authenticator_id}` (`WebAuthnCredentialRecord.id`), never the raw
  WebAuthn credential ID.
- **Setup codes.** The host command takes `SetupCodeIssueRequest` (purpose, email, display name,
  optional break-glass `target_account_id`), so enrollment needs only the code. `first_admin`
  creates an admin account and is allowed only while no active admin exists. `break_glass` either
  creates an admin account or re-enrolls the named one; at redemption Store revokes that account's
  credentials and sessions and sets its role to admin. Both are audited `break_glass_used`, and no
  other account is touched. No HTTP route issues codes.
- **Installations and keys.** An admin registers a `ProcessInstallation` (at most one primary host),
  then issues keys to it. Rotation issues a replacement with the same installation and scopes and
  never adds scopes. The old key gets `superseded_by_key_id` and `grace_until`, and leases, claim
  tokens, job IDs and sessions are unaffected. Retiring an installation revokes all its keys.

## Backup and restore

The metadata database and the object store are one logical dataset. `BackupManifest` records the
hostname and relying-party ID, schema and contract versions, the database digest, the object
manifest digest and the feed epoch. It also states that credential records, audit events, usage
records and, on Path A, the CA key are inside. `POST /admin/backup/restore-preflight` applies the
rule. Restoring under the same hostname keeps every enrollment. Under a different hostname, every
credential is revoked, every account becomes `reinvite_required`, and the result says so before
anything is written. Every restore does four things: it verifies each object's checksum against the
database, invalidates sessions, starts a new feed epoch, and drops leases and claim tokens from after
the backup. Backup and restore themselves run offline on the Store host, as today.

## Sequence walkthroughs

**Ingest → graph → claim → heartbeat → complete → dependents released**

1. Process receives an S3 event and calls `POST /conversations` with the `s3_event` source. A
   duplicate notification returns the same conversation with `created: false`. If the duplicate
   carries different call metadata, Store updates the metadata and says so with
   `metadata_updated: true`, without reprocessing (see "Conversation re-registration").
2. `POST /conversations/{id}/artifacts/uploads` (kind `source_audio`, sensitivity `raw`, checksum)
   returns an `UploadGrant`. Process PUTs the bytes to the grant URL, and
   `POST /artifact-uploads/{id}/commit` verifies size and checksum and returns the committed,
   linked `Artifact`. Ingestion is acknowledged only after step 3.
3. `POST /conversations/{id}/job-graphs` with `idempotency_key`, `reason: ingest` and job
   definitions:
   - `validation_vad` and `asr`, with input `audio` as the pinned source artifact (`primary_host`);
   - `speaker_attribution`, `acoustic_tone`, `text_sentiment` and `enrichment` requiring `asr`,
     with input `transcript = {upstream: {ref: asr, output_role: transcript}}`;
   - one `qa_criterion` per semantic criterion, with a frozen `FrozenSelection`,
     `parameters.rubric` set to the immutable rubric version, its `rubric` input pinned to the
     snapshot from `POST /conversations/{id}/rubric-snapshots`, and `criterion_id`; plus
     `qa_deterministic`;
   - a `qa_scorecard` requiring all of them, with each assessment as an upstream-output input
     (for example role `assessment:greeting`);
   - `summary_segment` jobs and their `summary_synthesis` and `summary_assembly`;
   - the two contact-signal passes and a `contact_signals_merge` with `after` edges and optional
     pass inputs.

   Store validates refs, output roles, cycles and checksums. It returns the graph with its edges,
   `validation_vad` and `asr` QUEUED, and the rest BLOCKED.
4. A worker reserves slots and calls `POST /jobs/claim` with its `WorkerCapabilities` and
   `slot_offers`. Store returns `ClaimedJob`s with claim tokens, resolved inputs, upstream outcomes
   and the offer each consumed. The worker fetches input bytes via
   `GET /artifacts/{id}/content` (or a content grant). If the slot is lost before inference, it
   calls `/release` with `requeue`.
5. Every `HEARTBEAT_INTERVAL` the worker calls `POST /jobs/{id}/heartbeat`.
6. On success the worker inlines or uploads the `transcript` output (slot `""`). It then calls
   `POST /jobs/{id}/complete` with `completion_key`, the output, the usage row, provenance and
   `result: {kind: transcript, state: available}`. Store links the output, publishes the transcript
   projection (the call becomes selectable in Evaluate while QA is still `pending`), resolves and
   pins the dependents' `transcript` inputs, releases them, and returns the receipt with
   `released_job_ids` and the change cursor. If the response is lost, the same call with the same
   body returns the same receipt with `replayed: true`, even though the lease is gone.
7. A `qa_criterion` completion whose trigger fired includes `follow_on`: the `qa_escalation`
   definition, which may require the completing job, and
   `add_dependencies: [{dependent_job_id: <scorecard>, requires_ref: "esc", input_role:
   "escalation:greeting", output_role: "assessment"}]`. The scorecard is still BLOCKED on the
   completing criterion, so the edge and the input binding land before the scorecard can be
   released.
8. The `qa_scorecard` completion publishes `result: qa`. Store creates the review-queue items the
   rules call for, referencing `evaluation_version`, and Evaluate stops showing `Analyzing`. The
   conversation is settled once every job is terminal.

**Review expected-version conflict**

1. Two reviewers open the same call; both read `CallDetail.review_version = 3`.
2. Reviewer A posts `POST /calls/{id}/verdicts/greeting` with `expected_version: 3` and
   `evaluation_version: 2`. Store applies it and returns `review_version: 4`.
3. Reviewer B posts with `expected_version: 3`. Store answers 409 `review_version_conflict` with
   `details.current_version = 4`. B's client re-reads, shows A's change, and B decides again with
   `expected_version: 4`. Nothing was overwritten.
4. If reanalysis committed machine version 3 in between, B's `evaluation_version: 2` is 409
   `conflict` with `current_evaluation_version: 3`. The review shows `staleness: stale`, and B's
   open queue item was SUPERSEDED by a new item for version 3.

**Passkey enrollment**

1. An admin calls `POST /admin/invitations` (`out_of_band`). The response carries `invitation_url`
   once; Store keeps `token_hash` and writes `invitation_issued`. Re-invites name
   `reinvite_of_account_id` and revoke that account's credentials and sessions on issue.
2. The invitee opens the link on the fixed Store hostname, and Evaluate calls
   `POST /auth/enroll/begin` with the token. It receives `CredentialCreationOptions`: `rp.id` is
   the hostname, `userVerification` is `required`, `attestation` is `none`, and the challenge is
   single-use.
3. The browser runs `startRegistration` (security key or platform passkey), and Evaluate posts the
   result unchanged to `POST /auth/enroll/finish`. Store verifies challenge, origin, RP ID,
   signature and the user-verified flag, stores the credential record, marks the invitation
   redeemed, writes the audit event, and returns the account, the credential and `signed_in`
   (`SessionInfo`, including the CSRF token). `prompt_second_authenticator` stays true until a
   second authenticator exists.
4. Later sign-in: `POST /auth/sign-in/begin` with the email returns `allowCredentials` (or decoys).
   `POST /auth/sign-in/finish` verifies the assertion, the account binding and the counter, and
   creates the server-side session. After a reload, `GET /auth/session` returns the session and its
   CSRF token. The first administrator follows the same steps with a `setup_code` printed on the
   Store host for an email and name given to the host command.

**Pro1 attested job with key release**

1. Admission. The job's `FrozenSelection.route` is `call1_confidential` with connection `pro1`.
   Store fails it at admission if the route is disabled (`route_disabled`) or the connection is
   blocked (`pro1_connection_blocked`), and holds it while the connection is
   `pending_verification`. Otherwise it is offered to a worker that lists the route class, the
   qualified entry and an `outbound` offer for `pro1`. If the key anchor is unavailable, the worker
   releases with `reject` and `pro1_key_unavailable`.
2. Process requests evidence naming its approved `(release_id, manifest_digest)` pairs (from
   `GET /admin/releases/approvals`). It verifies the evidence against the policy in
   `GET /admin/state` and the adopted trust anchors, and uploads the bundle once
   (`POST /attestation-evidence/uploads`).
3. Process obtains a session key from its `KeySource` and calls `POST /key-releases`. Store checks
   `KEY_RELEASE_PRECONDITIONS`, records the release and writes `key_released`. Only then is the key
   wrapped to the verified release key and sent.
4. Sealed requests and responses follow, and the worker heartbeats as usual.
5. `POST /jobs/{id}/complete` carries `provenance.route`, `provenance.attestation` (every field of
   `Pro1AttestationRecord`) and `key_release_ref`. It also carries a usage row with `session_id`,
   `session_setup_seconds`, provider-reported tokens and any billing units. The contract rejects a
   Pro1 completion without them.
6. A transient `pro1_unreachable` or `pro1_service_error` is reported with `/fail` and retried
   while attempts remain. On a QA assessment's final attempt it completes FLAGGED instead, with
   `provenance.pro1_failure` and the configured escalation (see "QA assessments that cannot
   answer").
7. A definitive failure (`pro1_attestation_invalid`, `pro1_measurement_rejected`, ...) is reported
   with `POST /jobs/{id}/fail`, with `provenance.pro1_failure`. In that transaction Store marks the
   job FAILED with no retry, blocks the connection, audits it and returns
   `pro1_connection_blocked: true`. New Pro1 jobs then fail at admission. An admin calls
   `POST /admin/pro1/clear-block` in Evaluate, and the connection moves to `pending_verification`.
   Process runs the attested connection test and reports it
   (`POST /release-trust/pro1-connection/verifications`), and a pass makes it `ready`. At session
   end Process calls `POST /key-releases/{id}/close`.

## Traceability

| Stage 1 item / team decision / finding | Contract element | Test |
| --- | --- | --- |
| Store API: conversations, artifacts, jobs, dependencies, leases and claim tokens, heartbeat, completion receipts, retry, usage rows, audit events, reanalysis requests, change cursor, review versions | `calls.py`, `artifacts.py`, `contents.py`, `jobs.py` (incl. `NewDependency` input bindings), `usage.py`, `events.py`, `reviews.py`; routes under `/conversations`, `/artifacts`, `/jobs`, `/reanalysis-requests`, `/changes`, `/calls/{id}/review` | `test_every_route_declares_principals_errors_and_lives_under_prefix`, `test_job_graph_rejects_cycles_unknown_refs_and_bad_selections`, `test_dependents_bind_upstream_outputs_by_role`, `test_merges_use_after_edges_and_optional_inputs`, `test_completion_request_carries_usage_provenance_and_follow_on` (incl. the escalation's input binding), `test_completion_transaction_order`, `test_release_ends_a_claim_without_consuming_an_attempt`, `test_claims_consume_bounded_slot_offers`, `test_every_described_transition_is_in_the_table`, `test_reanalysis_requests_have_a_state_machine_and_one_graph`, `test_change_cursors_are_typed_and_resume_without_gaps`, round-trips |
| Artifact content both sides must agree on; canonical JSON; one checksum per payload | `contents.py`, `ARTIFACT_CONTENT_MODELS`, `common.canonical_json`/`canonical_digest`, `artifacts.canonical_content` | `test_every_json_artifact_kind_has_a_content_model_without_store_ids`, `test_views_are_content_plus_store_ids`, `test_inline_artifacts_hash_canonical_payloads`, `test_inline_payloads_are_the_full_canonical_dump`, `test_canonical_json_matches_rfc8785_vectors`, `test_digest_fields_are_canonical_digests`, `test_artifacts_version_per_slot_only_when_linked` |
| Result states (Analyzing, Partial, Stale, Needs attention), job accounting; draft tests and speaker corrections kept out of the wrong groups | `calls.result_state_inputs` (drops draft-test graphs), `calls.derive_result_state`, `JobTypeRule.group`/`publishes`, `REANALYSIS_KIND_AFFECTS`, `SETTLED_RESULT_STATES`, draft-test slots, `JobParameters.speaker_correction` | `test_result_state_derivation`, `test_result_state_rule_is_settled_and_reanalysis_scoped`, `test_draft_tests_never_change_the_calls_result_state`, `test_speaker_corrections_are_frozen_on_a_code_stage_job`, `test_every_job_type_has_outputs_a_group_and_each_group_one_publisher`, `test_contact_signal_partials_say_what_is_missing`, `test_call_record_media_fields_wait_for_validation` |
| Rubrics and their snapshots; draft tests; human review queue separate from the job queue; metrics; admin | `rubrics.py` (`RubricSnapshotRequest`, `RUBRIC_INPUT_ROLE`), `artifacts.STORE_MINTED_KINDS`, `reviews.py`, `metrics.py`, `admin.py`; routes `/rubrics`, `/conversations/{id}/rubric-snapshots`, `/review-queue`, `/metrics`, `/admin` | `test_rubric_snapshots_are_minted_by_store`, `test_job_queue_and_review_queue_share_no_type_status_or_endpoint`, `test_review_and_admin_writes_require_reviewer_sessions`, `test_review_queue_transitions_are_closed`, `test_draft_tests_never_publish_and_need_manage_rubrics`, `test_qa_assessments_record_invalid_answers_and_provider_failures_as_flagged` |
| Process service-key scopes, hashing, rotation; installation identity | `ServiceScope`, `ServiceKeyRecord` (`key_hash`, `superseded_by_key_id`, `grace_until`), `ServiceKeyIssued` (one-time), `ServiceKeyRotate`, `ProcessInstallation`, per-route scopes, `OWN_INSTALLATION` object rules | `test_service_key_hashing_scopes_and_rotation`, `test_every_scope_and_permission_is_used`, `test_process_routes_take_one_scope_and_act_as_their_installation`, `test_no_field_carries_a_secret_value` (the key is never on a stored record) |
| Reviewer auth: passkey-only, setup code, invitations, add-authenticator, re-invite, audited break-glass, server-side sessions, roles | `auth.py` (ceremony models, step-up, `SetupCodeIssueRequest`/`SetupCodeRecord`, `BreakGlassRecord`, `SessionInfo` with CSRF, `SESSION_COOKIE`, `ROLE_PERMISSIONS`), routes `/auth/...`, `/admin/invitations`, `/admin/accounts/.../authenticators/...` | `test_passkey_only_ceremonies`, `test_simplewebauthn_browser_payloads_validate_unchanged`, `test_csrf_token_is_session_bound_and_recoverable`, `test_add_authenticator_is_a_step_up_ceremony`, `test_authenticator_paths_use_store_handles`, `test_sign_in_never_reveals_whether_an_account_exists`, `test_setup_codes_carry_the_identity_they_enroll`, `test_session_routes_can_report_expiry_and_disablement`, `test_session_rules_only_name_permissions_the_role_holds` |
| Loopback console credential and its scope | `ConsoleOperation` (no provider-connection or trust writes), `ConsoleCredentialDescriptor` (`store_rights: none`); no Store route accepts `console_credential` | `test_console_credential_is_never_a_store_principal`, `test_console_cannot_set_provider_connections` |
| TLS trust model (Path A / Path B), fixed hostname, RP ID, allowed origins; status exposure | `validate_store_hostname`, `TlsTrustPath`, `TlsState`, `WebAuthnRelyingParty`, `StoreHealth`/`StoreStatus` | `test_hostname_rule_rejects`, `test_hostname_rule_accepts_customer_dns_names_and_binds_rp_and_origins`, `test_tls_paths`, `test_anonymous_status_carries_no_build_or_activity` |
| Data custody: route classes, per-job route record, Pro1 evidence fields (failed, interrupted and FLAGGED attempts); nothing reopens a plaintext Call1 route (decisions 4, 8) | `RouteClass`, `ROUTE_CLASS_RULES`, `RouteRecord`, `call1_operated_host` (pure, IPv6-aware), endpoints as admin-state URLs, `Pro1AttestationRecord`, `Pro1AttemptFailureEvidence` (`PRO1_INTERRUPTION_CODES`), `AttemptProvenance`, `PRO1_PROVIDER_FAILURE_CODES` | `test_route_record_enforces_route_classes`, `test_pro1_provider_failures_complete_flagged_with_their_evidence`, `test_pro1_failure_evidence_matches_the_failure_and_covers_interruptions`, `test_route_host_rule_is_the_stage0_guard` (IPv4, IPv6, with and without `CALL1_PRO1_ENDPOINT`), `test_host_rule_is_environment_independent`, `test_admin_endpoints_are_full_urls_and_reject_call1_hosts`, `test_pro1_attempts_record_evidence_whether_they_succeed_or_fail`, `test_attestation_evidence_is_global_and_readable_by_auditors` |
| Key release: key-release service, key-source interface (local v1; optional customer KMS/HSM anchor), reference per attempt (decision 6) | `KeySourceKind`, `KeySourceRef`, `KeyReleaseInput/Record`, `KEY_RELEASE_PRECONDITIONS`, `KeyManagerConfig`, `Pro1AttestationRecord.key_release_ref`, route `/key-releases` | `test_key_source_and_key_manager_never_hold_values`, `test_key_release_preconditions_are_listed_with_errors`, `test_pro1_attempts_record_evidence_whether_they_succeed_or_fail` |
| Release trust: approval records (manifest SHA-256 + log index), declines, log scan state, updater contract, admin state changed only by audited actions, Pro1 connection block, egress allowlist (decision 7) | `ReleaseApproval`/`ReleaseApprovalRequest` (names a pending release), `ReleaseDeclineRequest`, `LogScanState`, `UpdatePackage`/`UpdaterVerification` (Store-internal), `AttestationPolicy` (+ pins, the only Pro1 floor), `TrustAnchorState`/`TrustAnchorBundleSubmit`, `AdminState`/`AdminStateChange` (settings only), `Pro1Connection` + `PRO1_CONNECTION_TRANSITIONS`, `EgressAllowlist` | `test_admin_state_and_trust_changes_are_audited_admin_actions`, `test_admin_changes_carry_settings_only`, `test_admin_state_change_is_one_section_at_a_time`, `test_approvals_name_an_observed_pending_release_and_can_be_declined`, `test_updater_verification_needs_every_check_and_an_approval`, `test_updater_verdicts_have_no_write_route`, `test_minimum_release_svn_has_one_home_each`, `test_attestation_policy_pins_platform_identity`, `test_trust_anchor_bundles_are_artifacts_process_registers_and_admins_adopt`, `test_pro1_connection_has_one_writer_and_a_sticky_block`, `test_egress_entries_label_customer_data` |
| Usage-record and hardware-profile schemas; usage report, CSV, price table, medians | `UsageRecordInput/Record` (incl. `abandoned`), `TokenCount` with source, `HardwareProfileFields/Input`, `USAGE_CSV_COLUMNS`, `PriceTable`, `UsageMedians` | `test_usage_and_hardware_rules`, `test_failure_and_completion_usage_outcomes_match_the_ending_call`, `test_lease_expiry_produces_one_abandoned_usage_row`, `test_usage_report_has_csv_a_price_table_and_medians` |
| Status set with `WAITING_PROVIDER` reserved | `JobStatus`, `RESERVED_STATUSES`, `JOB_TRANSITIONS` | `test_status_set_reserves_waiting_provider` |
| Backup/restore rule | `BackupManifest` (feed epoch), `RestorePreflightRequest/Result` | `test_restore_rule_same_hostname_keeps_enrollments_and_starts_a_new_epoch` |
| Location neutrality vs HostedV2.md (decision 1) | No path fields; artifacts by ID + checksum; grants point at Store; pull-only; `StoreStatus` carries no data location | `test_no_field_assumes_shared_files_or_colocation` |
| Change feed filtering by principal | `CHANGE_KINDS_BY_PRINCIPAL` | `test_change_feed_is_filtered_by_principal` |
| Open core, nothing gated (decision 9); Pro1 never a default (decision 10) | No entitlement, licence or Pro1-credential check field; `call1_confidential.default_enabled = false`; no cap or seat field | `test_pro1_attempts_record_evidence_whether_they_succeed_or_fail` (asserts default off); the absence of entitlement fields is checked by code search, not by a test |
| v1 routes only (decisions 2, 3) | `ProviderType` = mlx, ollama, pro1, byok | none (the enum is closed) |
| Per-customer fine-tunes (decision 11) | Enter the catalog as entries with `weights_digest`; no contract change needed | none |
| Generated artifacts reproducible; unpinned upstream releases do not break the suite | `GENERATOR_PINS`, `constraints.txt`, `generate.py --check`, `CALL1_REQUIRE_TS_CHECK`, `CALL1_REQUIRE_GENERATOR_PINS` | `test_committed_openapi_matches_regeneration`, `test_committed_typescript_matches_regeneration`, `test_generate_check_cli_passes`, `test_generator_versions_match_pins`, `test_drift_tests_skip_under_unpinned_generators_unless_required` |
| Contact Signals v2 (decisions 21, 22; ContactSignalsV2.md section 7): taxonomy with fixed built-ins, caps as parameters, digests and hit identity, masked-only stages, v2 result validator, draft-test previews, claim filter, admin-only management, one publisher per group | `signals.py`, `contents.py` stage artifacts, `JobTypeRule.needs_signal_taxonomy`, `SignalJobParameters`, `DRAFT_TEST_KINDS`, `ReanalysisClaimRequest.kinds`, `ContractParameters` caps, `Permission.MANAGE_SIGNALS`, `ErrorCode.SIGNAL_TAXONOMY_CONFLICT`, the three signal `ChangeKind`s, 14 Stage-2 routes | `test_signal_*` and `test_contract_1_3_0_*` in `tests/test_contracts.py` (round-trips, taxonomy validators, caps, digests, provenance, the v2 result validator, draft-test kinds, claim filter, permissions and feed, one publisher per group, a v2 graph, the retail seed) |

## Migration notes (for the Stage 3 importer)

- `ingest_jobs` rows become a `Conversation` with `source.legacy_ingest_job_id`. `calls` become a
  `CallRecordView` with `legacy_call: true`; its groups have no graph, so they are `available` or
  `disabled`. `customer_phone` is not carried: the contract has only a masked `caller_reference`.
- Legacy transcripts, scorecards and summaries become linked artifacts of their content kinds
  (`producing_job_id` null, version 1). Anything a content model has no field for goes on a
  `migration_record` artifact.
- `evaluations.review_version` becomes `CallReviewState.review_version`. `evaluation_history`
  becomes `ReviewHistoryEntry` with `kind: legacy_action`, and `payload.actor_label` holds
  shared-token actors (`ActorKind.LEGACY_SHARED_TOKEN`). No per-person attribution is invented.
- `queue_rules.priority` becomes `ReviewQueueRule.rank`, and `stream_type` becomes `stream`.
  `PRO1_AUTOMATED` becomes `UNASSIGNED_CLAIM`, with the original value on the `migration_record`
  artifact. `review_queue.RELEASED` rows become `PENDING`, unassigned, with a history entry.
- `auditors` become `ReviewerAccount` (`legacy_auditor_id`, status `reinvite_required` until
  enrolled) plus `ReviewerProfile`.
- `QuestionModel` IDs become catalog entry IDs (`legacy_question_model_id`). BYOK entries become
  `ByokProvider` with `base_url` equal to the legacy endpoint. The rubric's `primary_model_id` and
  `escalation_model_id` keep their values.
- `RubricCriterion.criterion_id` must match `rubrics.CRITERION_ID_PATTERN` (slot-safe: no `:`, at
  most 128 characters), because the ID becomes the QA assessment slot. Every seeded legacy ID fits.
  The importer rejects and reports a customer-authored legacy ID that does not fit, and never
  rewrites it silently.
- `RubricCheck` fields are identical to the legacy model. `RubricDefinition.category` must be one of
  `RubricCategory`.

## Docs reconciled at WF-A

The revision rounds found these passages disagreeing with the contract and the plan. The WF-A
doc-update step applied them in the same commit as the contract. They are kept here as a record:

- `docs/Pro1ConfidentialInference.md` §2.5 ("approves it in Process Models settings") and §3.1 (the
  Settings and UI row). Under the plan's precedence ("the console can display them but not change
  them"; approvals are Store admin actions in the browser), approval, the policy editor, anchor
  adoption, declines, the key-release log and clear-block are Evaluate admin-area actions. Process
  Models displays status, pending releases and evidence, and links out.
- `docs/Pro1ConfidentialInference.md` §4: say that "fail at admission" means Store fails QUEUED
  jobs (`admission_rejected`) and that a key-anchor outage is a pre-inference release (`reject`),
  neither of which consumes an attempt. The same wording appears in §2.4 ("Pro1 jobs then fail at
  admission with `pro1_key_unavailable`") and in §9's crypto-shredding test.
- `docs/Pro1ConfidentialInference.md` §4, rule 6 ("The job shows `FAILED`"): on a QA assessment's
  final attempt, `pro1_unreachable` and `pro1_service_error` complete the job with a FLAGGED
  assessment and the configured escalation, which matches the section's own "Escalation"
  paragraph. Every other Pro1 failure shows `FAILED`.
- `docs/AsyncJobPipelinePlan.md` Stage 2 and Stage 4: "audited since Stage 2 under the shared admin
  token" and "Legacy token sign-in stays unchanged in this stage" apply to the legacy app only.
  `/store/v1` has passkey sessions from the start (CLAUDE.md track list).
- `docs/AsyncJobPipelinePlan.md` Stage 2, settings validation: `is_call1_operated()` should
  delegate to `call1.contracts.custody.call1_operated_host`, passing the Pro1 endpoint host, so the
  guard and the contract are literally one function.
- `CLAUDE.md` "Commands": install with `-c call1/contracts/constraints.txt` so the drift tests run
  instead of skipping. `requirements.txt` keeps its ranges.

## Delivery stages

Every route carries `x-call1-stage`: the stage of `docs/AsyncJobPipelinePlan.md` that first
implements it (`api.DELIVERY_STAGES`, `api.STAGE_BY_OPERATION`). Tracks build the Stage 2 set
first and leave later routes unimplemented (they may return 501 until their stage).

| Stage | Routes | What |
| --- | --- | --- |
| 2 | all except those below | Queue, artifacts, results, reviews, rubrics, metrics, change feed, catalog, usage, custody and Pro1 release trust, and reviewer passkey sessions with admin identity on `/store/v1`. There is no shared-token principal on `/store/v1`, so passkey sessions ship in Stage 2 (localhost is a WebAuthn secure context). The 14 Contact Signals v2 routes (1.3.0) ship on the Stage 2 substrate too. |
| 4 | `getTlsState`, `getBackupManifest`, `restorePreflight` | Cross-device trust: Call1 CA or customer certificate, the fixed hostname, same-hostname restore |
| 5 | update packages (5 routes), `listUpdaterVerifications`, `getEgressAllowlist` | The installed updater and the installer's egress allowlist |

## Open questions for John

These are decisions the sources left open, with the default the contract takes. Item 12 is the only
one with no buildable default; every other item has a default the tracks can build against.

1. **Three reviewer roles.** The sources name "roles" and mention admin, reviewer and supervisor
   actions without fixing the set. The contract defines `reviewer < supervisor < admin`, with the
   permission table in `auth.py` (supervisors assign work, resolve escalations, retain stale
   reviews, and manage rubrics and queue rules). Collapsing to two roles is a minor change.
2. **Job retry and cancel in v1 are Process-only.** Per the plan, the console reaches them through
   Process's service key (`jobs:control`). HostedV2 will add an admin-session principal on the same
   routes, which is additive. Say if you want the admin session allowed from v1.
3. **Verdict overrides by reviewers.** The legacy reviewer token could override verdicts and
   resolve escalations. The contract keeps overrides for `reviewer` and moves escalation resolution
   to `supervisor`.
4. **`PRO1_AUTOMATED` review distribution is dropped** from v1, because the frontier review offering
   is future work. The importer maps it to `UNASSIGNED_CLAIM`.
5. **Store admin rotates service keys over HTTP.** The token is returned exactly once in
   `ServiceKeyIssued`, and the installer still issues the first key on the Store host. If you
   prefer host-only issuance, the route becomes CLI-only and the model stays.
6. **Invitation delivery in v1 is link-only** (out of band or the customer's SMTP relay). The
   HostedV2 second-channel short code is not in v1; adding it is an optional field.
7. **`caller_reference` replaces `customer_phone`.** The contract carries only an already-masked
   caller identifier. If reviewers need the raw number, that is a masking decision (AI risks).
8. **Rubric drafts are single-writer** (one draft per rubric, `draft_revision` concurrency).
   Multi-draft editing would be a minor change.
9. **Draft rubric testing** is an asynchronous `qa_draft_test` request created by
   `POST /rubrics/{id}/draft/tests`. Its scorecard is readable only through the draft-result route
   and never becomes the call's QA. The legacy `test_draft` endpoint was synchronous.
10. **Default timings** (`ContractParameters`) are proposals: 300 s leases, 8 h idle and 12 h
    absolute sessions, 7-day invitations, 30-minute setup codes, 24 h rotation grace, 7-day
    change-feed retention and 24 h orphan retention. The 10-minute Pro1 session lifetime is now
    `AttestationPolicy.session_lifetime_seconds`.
11. **Audit hash chain.** `AuditEvent` carries `previous_event_digest` and `event_digest`, so the log
    is tamper-evident and HostedV2's exported audit root has something to anchor to. It costs
    nothing in v1; drop it if you consider it scope creep.
12. **Semantic search: decided, option (a), then reversed in 1.2.0.** Until 1.2.0 Store embedded
    queries with the deterministic, model-free `hashing-projection-v1` scheme and the `embeddings`
    job was a code stage. Decision 18 replaced it with a model-based scheme under its own name
    (`nemotron-3-embed-1b@c0c9fea`), as this item anticipated, and made Store run that one embedder.
    See "Version 1.2.0".
13. **QA provider failures and invalid answers complete as FLAGGED** assessments (with escalation
    when configured), rather than failing the job, to keep the pre-split router's output. See "QA
    assessments that cannot answer". Attestation-class failures still fail the job.
14. **A completion after cancel is refused** (`job_cancelling`), and the worker reports `cancelled`.
    The alternative is letting the completion win and deferring the cascade.
15. **Stale review writes are rejected, not reconciled.** New machine versions supersede unresolved
    queue items, and new items are created for the new version.
16. **Stale starts when a reanalysis request is created**, not when Process claims it, so Evaluate
    labels the old result at once.
17. **Break-glass on an existing account revokes its credentials at redemption**, not at issue.
18. **A speaker correction does not mark the transcript `Stale`.** Its graph reruns the tone,
    sentiment, QA, summary and contact-signal publishers, which go stale, but not `asr`. Evaluate
    shows the correction as pending from the request until the new attribution is linked. The
    alternative is a transcript group that also stays stale while its graph's
    `speaker_attribution` job runs, which needs a second "underway" input to
    `derive_result_state`.
