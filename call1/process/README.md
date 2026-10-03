# Call1 Process

Process ingests recordings, builds each conversation's Stage 2 job graph, and runs the worker loop
that claims those jobs from Store, executes them and publishes the results. It also serves the
Process operator console on loopback. It reaches Store only through the `/store/v1` HTTP API with
its own service key, and it never imports `call1.store`, `call1.db` or `call1.ingest`.
`tests/test_split_boundaries.py` enforces that in a fresh interpreter and with a static scan.
The architecture is in [docs/SplitBuild.md](../../docs/Architecture.md); the interface is the
frozen contract in [`call1/contracts/`](../contracts/README.md) (v1.4.0).

## Run it

```bash
# on the Store host, once: register this installation and write Process's config (mode 0600)
python -m call1.store issue-service-key --installation mac-mini --scope jobs:control \
  --scope calls:write --scope artifacts:read --scope artifacts:write --scope jobs:write \
  --scope jobs:claim --scope reanalysis:claim --scope changes:read --scope hardware:write \
  --scope catalog:publish --scope usage:read --scope admin-state:read --scope training:read

python -m call1.process serve                 # console + API on http://127.0.0.1:8020, worker loop
python -m call1.process ingest call.wav       # register, upload and create the graph (--wait 120 to follow it)
python -m call1.process ingest call.wav --agent-id a-104 --agent-name Samantha --agent-extension 104
python -m call1.process drain                 # run every ready job once, then exit (no HTTP)
python -m call1.process console-token --rotate
python -m call1.process status                # Store reachability and contract version, as JSON

python -m call1.launch                        # Store and Process together on one computer (below)
python -m call1.launch --demo                 # DEMO MODE for a class demo: data/demo/, five sample calls (below)
```

Without `--scope`, `issue-service-key` leaves out `jobs:control`; everything works except console
retry and cancel, which then answer 403 `insufficient_scope` with instructions.

**Agent identity and re-uploads (contract 1.1.0).** An ingest names the agent with any of
`agent_id` (the stable key filters, metrics and queue rules use), `agent_display_name` and
`agent_extension`; Evaluate shows "Name (ext)" (`calls.agent_label`). Process registers only the
metadata the caller supplied (`CallMetadata` built from the given values, sent with
`exclude_unset`), so a re-upload that names no agent never resets a known one to "Unknown".
Ingesting the same recording again with different details updates the call's metadata: Store
audits it and emits a change event, and the receipt says `metadata_updated: true` with
`updated_fields` (the CLI also prints a note). Nothing is reprocessed: the existing graph is
returned, and a changed `agent_channel` only affects graphs built later. The source identity
includes its kind, so a CLI import (`local_import`) and an API upload (`api_upload`) of the same
bytes are separate calls. Process reads no object-store (S3) metadata.

`CALL1_PROCESS_HANDLERS=fake` selects the fake handlers (tests, machines without Apple Silicon).
The default is `real`: it loads `call1/process/handlers/real/` when that package exists. A job type
with no handler is not offered to Store, so its jobs wait for a worker that has one. Real mode never
falls back to fake content. The console lists the missing types.

### Configuration

The config file lives at `CALL1_PROCESS_CONFIG` (default `data/process/config.json`, JSON, mode
0600). `config.py` documents every key. These are the main ones:

| Key | Default | Meaning |
|---|---|---|
| `store_url`, `installation_id`, `service_key_id`, `service_key` | written by `issue-service-key` | The Store connection. `https://` always, or `http://localhost` for a dev-mode Store (`127.0.0.1` and `[::1]` work too: dev-mode grant URLs name `localhost`, and the client treats the loopback names as one host when the port matches) |
| `bind` / `port` | `127.0.0.1` / `8020` | Loopback only; any other bind is refused |
| `data_dir` | `data/process` | `scratch/` (bounded), `spool/` (pending completions, and the outputs of deferred publications under `spool/outputs/`) and `conversations.jsonl` (the ingest ledger) |
| `handlers` | `real` | `fake` or `real` (`CALL1_PROCESS_HANDLERS` overrides) |
| `rubric_id` | `call1_standard_v2` | The published rubric new recordings are scored with (its current version) |
| `model_defaults` | catalog defaults | Purpose to catalog entry ID |
| `escalation_entry_id` | none | The QA escalation model when a criterion names none. None means no escalation, as before the split |
| `slots` | `cpu_io 4, torch 1, mlx 1, outbound 2` | Local resource slots; `mlx` is always 1 |
| `stages` | all on | `summary`, `contact_signals`, `embeddings` |
| `mask_model_text` | `store` | Masking of text-model prompts (QA, summaries, contact signals) on the appliance route: `store` follows Store's text masking, `on` / `off` override it (see "Masking") |
| `scratch_retention_hours` / `scratch_max_bytes` | 24 / 5 GiB | Scratch retention |
| `console_credential` | issued on first `serve` | `{credential_hash, created_at, rotated_at}`; only the hash is stored |
| `training` | off | On-device training (see "On-device training"): `enabled`, `schedule {frequency, weekday, time}`, `min_new_labels`, `max_duration_minutes`, `only_when_idle`, and config-only knobs. The console's Settings tab writes it |

Environment overrides: `CALL1_PROCESS_HANDLERS`, `CALL1_PROCESS_DATA`, `CALL1_PROCESS_PORT`,
`CALL1_PROCESS_BIND`, `CALL1_PROCESS_WORKER_ID`, `CALL1_PROCESS_STORE_URL`, `CALL1_PROCESS_MASK_MODEL_TEXT`,
`CALL1_MODELS_DIR` (models root, default `data/models`), `CALL1_MODEL_MANIFEST`, `CALL1_FAKE_BEHAVIOR`,
`CALL1_PROCESS_TRAINER` (`fake` or `mlx_lm`), `CALL1_FAKE_TRAINER_SECONDS` / `CALL1_FAKE_TRAINER_EXIT` and
`CALL1_FAKE_TRAINING_OUTCOMES` (fake-mode training only).

### The one-computer launcher

`python -m call1.launch` runs `python -m call1.store migrate`. When Process's config has no
service key yet, it also runs `issue-service-key` with the Stage 2 scopes plus `jobs:control` and
prints the `setup-code` command for the first admin. It then starts `call1.store serve` and
`call1.process serve` as subprocesses. Their output is prefixed `[store]` / `[process]` on the
terminal and appended to `data/logs/launch.log`, which is mode 0600 with credentials redacted.
Ctrl-C, SIGTERM or either app exiting stops both gracefully. The launcher imports neither app.
Ports come from `CALL1_STORE_PORT` and `CALL1_PROCESS_PORT` (8010 and 8020).

### Demo mode (`python -m call1.launch --demo`)

**DEMO MODE is for a class demo on this computer only, never for real calls.** It is a layer on top
of the normal launcher; the passkey code is untouched and keeps working beside it.

- **Separate data.** Everything lives under `data/demo/` (`store/`, `process/`, `logs/`), so
  `data/store` and `data/process` are never touched. `--reset` wipes the demo root first (it
  refuses a directory that is not a demo root). `--demo-root` or `CALL1_DEMO_ROOT` moves it.
- **Store demo sign-in.** Store runs with `CALL1_STORE_DEMO=1` in dev mode (plain
  `http://localhost`, loopback bind): labelled persona sign-in without a passkey. Any
  `CALL1_STORE_PUBLIC_URL`, TLS or hostname setting in the shell is dropped for the demo.
- **Handlers.** The real handlers with `CALL1_BACKEND=mlx` on an Apple-Silicon Mac when
  `data/models` (or `CALL1_MODELS_DIR`) holds weights; otherwise the fake handlers, and the banner
  says why. `--handlers fake|real` overrides the choice.
- **Contact Signals v2.** Right after the migration, before either app starts, `python -m
  call1.store apply-signals-seed call1/store/seeds/signals_retail_v1.json --pipeline v2` publishes
  the retail seed taxonomy (decision 22) and sets `pipeline: v2`, so every seeded call runs the three
  v2 stages (Gemma or category-specific rules with real handlers; see the rules-engine section). The seed is applied once per demo root
  (`signals-seeded.json`); later starts only run `signals-pipeline v2`, keeping taxonomy edits.
  `CALL1_SIGNALS_PIPELINE=v1|shadow` overrides the pipeline. On fake handlers the launcher maps
  `call_05_pii_heavy`'s checksum to the `cancel` fake script in `CALL1_FAKE_SCRIPTS` (keeping any
  mapping already set), for the ContactSignalsV2 §15 step. The banner's "Contact Signals:" line says
  which pipeline and engine run.
- **Seeding.** Once Store answers `/store/v1/status` and Process reports `running`, the launcher
  uploads the five `sample_audio/call_0*.wav` recordings through `POST /process/api/recordings`.
  Agent identity comes from `sample_audio/manifest.json` when it names one, else the legacy seed's
  agents: Samantha (104) on calls 01, 03 and 05, Bob (202) on 02 and 04. `external_call_ref` is the
  manifest's call ID and `agent_channel` its agent channel on stereo calls. A demo root that is
  already seeded (`demo-seeded.json`) is not seeded again; ingest is idempotent anyway.
- **URLs.** A fresh console credential is issued at every start, so the Process console URL
  carries it (`#console_token=…`; the log file redacts it). Evaluate (`/`), the Store console
  (`/console/`) and the Process console open in the default browser (macOS `open`); `--no-open`
  skips that.
- **Progress.** Each call's progress line is printed when it changes, until every call has
  settled. The apps' per-request access lines go to `data/demo/logs/launch.log` only, so the
  terminal stays readable.
- It refuses to start when either port is taken (for example by the regular launcher on
  8010/8020); pick others with `CALL1_STORE_PORT` / `CALL1_PROCESS_PORT`.

## Module map

| Module | What it does |
|---|---|
| `config.py` | `ProcessConfig`: file, environment, validation (loopback bind, HTTPS or dev-mode Store) |
| `store_client.py` | `StoreClient`: typed contract calls, retries with backoff, contract-major check, error envelope, dev-mode grant tolerance, checksum-verified downloads |
| `catalog.py` | The Process catalog seeded from `model-manifest.json`, `call1.model_catalog.LOCAL_PACKS` and the search embedder (`nemotron-3-embed-1b`); `FrozenSelection`s; the published `CatalogSnapshot` |
| `graph.py` | `GraphPlanner`: ingest and reanalysis graphs, and the follow-on jobs added at completion (summary segments, escalations) |
| `worker.py` | Resource slots, claims, heartbeats, execution, output upload, completion, failure, release, the spool, graceful stop |
| `handlers/` | The handler interface (`base.py`), registry (`__init__.py`), code handlers (`code.py`), the `embeddings` handlers (`embeddings.py`), fake handlers (`fake.py`), the Contact Signals v2 stage runners (`signal_stages.py`) and their rules engine (`signals_rules.py`, contract 1.4.0), and the real handlers (`real/`: `media`, `analysis`, `qa`, `summary`, `signals`, `signals_v2`, with `llm`, `masking`, `convert` and `paths` helpers) |
| `ingest.py` | Recording in: registration, source upload, rubric snapshot, signal taxonomy snapshot (v2 only), graph |
| `vocabulary.py` | Dual transcription (decision 33): reading Store's ASR vocabulary, freezing it into the `asr` job, and the transcript's `vocabulary_correction` provenance (see "Dual transcription") |
| `signals_input.py` | Contact Signals v2 wiring: the taxonomy read and snapshot mint at ingest; the snapshot, previous stage artifacts and rerun plan of a reanalysis request |
| `reanalysis.py` | The reanalysis-request consumer (qa, summary, contact_signals, contact_signals_preview, speaker_correction, full, qa_draft_test, embeddings); claims may name `kinds`, and a host with no usable v2 classifier leaves `contact_signals_preview` out of its claims so preview and compare requests stay pending |
| `runtime.py` | Start-up handshake (contract, admin state, hardware profile, catalog snapshot, spool replay), background loops |
| `app.py`, `console.py` | The loopback operator API, the console credential, the static console |
| `training/` | On-device training of the customer LoRA (decision 28): settings, the scheduler and claim pause, label paging, replayed source jobs, the example builders, the dataset, the pluggable trainer, evaluation, the adapter registry (see "On-device training") |
| `transcripts.py`, `audio.py`, `scratch.py`, `ledger.py`, `hardware.py` | Helpers |

## The job graph

At ingest the planner builds, from the registered conversation and its pinned `source_audio`:

- `vad` (validation/VAD);
- `asr` (with dual transcription inside it when the ASR vocabulary is active; see "Dual transcription");
- `speaker` (mono recordings only);
- `tone`, `sentiment`, `enrichment`;
- `embeddings`, a model stage with the search embedder (contract 1.2.0; see "Search embeddings");
- one `qa-<n>-<criterion>` per semantic criterion;
- `qa-det`, when the rubric has deterministic checks;
- `scorecard`;
- `summary`, the assembly;
- `cs-lifecycle`, `cs-resolution` and `cs-merge` (Contact Signals v1), or, when the signal settings
  select v2 and this host can serve it, `cs-categorize`, `cs-subcategorize`, `cs-extract` and
  `cs-merge` (see "Contact Signals v2").

The merge uses `after` edges and optional pass inputs. Every model-backed job freezes its
selection from the catalog. Every QA job pins the Store-minted rubric snapshot under input role
`rubric`. Idempotency keys are `ingest.<conversation>.<ref>`, so repeating an ingest returns the
same conversation and graph.

Two kinds of job are added at completion, as contract follow-ons:

- **Summary segments and synthesis.** Their turn windows need the transcript, so the ASR completion
  adds the segment jobs. The windows follow the pre-split summarizer's chunking: `batch_turns` and
  3,500 bytes. When more than two segments remain, synthesis runs pairwise; a final synthesis
  follows. The completion binds them to the `summary` assembly (`segment:<n>`, `synthesis`).
- **Escalations.** When a `qa_criterion` assessment's trigger is in the criterion's
  `escalation_when`, and an escalation model was frozen at graph time, its completion adds a
  `qa_escalation` with the same pinned inputs. It binds the escalation's `assessment` to the
  scorecard as `escalation:<criterion>`. The core sets the assessment's `escalation_requested`.

Invalid answers complete FLAGGED with usage outcome `validation_rejected`. A provider failure
(`PROVIDER_FAILURE_CODES`) goes to `/fail` while attempts remain. On the claim's final attempt the
core completes it as a FLAGGED assessment (`trigger: provider_error`, outcome `failed`).

## The worker

| Pool | Size | Offers | Job types |
|---|---|---|---|
| `mlx` | 1 | `local_memory` | every type (the MLX and Ollama entries use this slot) |
| `torch` | `slots.torch` | `cpu` | `acoustic_tone`, `text_sentiment`, `contact_signals_categorize`, `contact_signals_subcategorize` (`worker.TORCH_TYPES`). On Gemma (decision 24) the two classifier stages freeze the `local_memory` slot, so the `mlx` pool claims them; the `torch` pool gets only the no-model re-derive, and a later torch "system one" engine |
| `cpu_io` | `slots.cpu_io` | `cpu`, `io` | every other type |
| `outbound:<ref>` | `slots.outbound` | `outbound` | one pool per admin-state connection (none in Stage 2) |

Each pool reserves its free slots, claims at most that many jobs and gives unused slots back at
once. The claim lists this installation's qualified catalog entries and the appliance route class.
If Store says the key is not the primary host, the worker stops offering primary-host jobs.
Store hands out the oldest call's ready jobs first (call by call, within the request's priority
band; `call1/store/queue/README.md`), so Process sends no `conversation_id` affinity hint.

A pool that claimed nothing waits `poll_interval_seconds` (2 s) before asking again, but a
completion or failure whose receipt released dependents or created follow-on jobs wakes every idle
pool at once (`Worker.notify_work`), so the next stage of a call starts without waiting for the
poll. The poll stays as the fallback, for example for jobs another host released.

For every claim, the worker takes these steps:

1. It checks that it has a handler and that the frozen entry is installed and qualified. Otherwise
   it releases the claim with `reject` and `model_unavailable` or `model_unqualified`, and no
   attempt is used.
2. It fetches and checksum-verifies the inputs. When a fetch fails it releases with `requeue` and
   `input_unavailable`.
3. It runs the handler while a heartbeat thread renews the lease. A heartbeat that returns
   `cancel_requested` sets the handler's cancel event.
4. It uploads the outputs. JSON at or under `inline_artifact_max_bytes` goes inline; anything
   larger goes through an upload grant. If Store cannot be reached here (or while it plans the
   follow-ons), the finished result is spooled instead of thrown away (below).
5. It completes the job with the usage row, provenance, `ResultPublication` and follow-ons.
   `job_cancelling` becomes a `cancelled` failure. `claim_token_stale` attaches late usage and
   publishes nothing.

Completion, failure and release bodies are spooled before they are sent. If Store cannot be reached,
the body stays in the spool and the attempt's claim stays alive: the heartbeat thread keeps renewing
the lease of every finished job whose completion or failure is waiting for delivery (the console
lists them under `awaiting_delivery`). A spool thread replays the spool with the same key and body
every 5 seconds and at once when a pool reaches Store again, so an outage shorter than the lease
redoes no work. A completion Store now refuses with `job_cancelling` is turned into a `cancelled`
failure. After a crash, `recover()` replays the spool at start-up; a claim that expired meanwhile
is refused as stale and its measurements become late usage.

A job that finishes while Store is away usually cannot even upload its outputs, so the spool also
holds **deferred publications**: `<job>.<attempt>.publish.json` (the claim, measurements, outcome
and output descriptors) with the output files in `spool/outputs/<job>.<attempt>/` (mode 0600, this
appliance only). The slot is freed at once; the claim is listed under `awaiting_delivery` as
`publish` and heartbeated. The replay first rechecks the claim with a heartbeat: a stale claim
becomes late usage and nothing is published; a requested cancel becomes a `cancelled` failure and
nothing is uploaded; otherwise it plans the follow-ons, uploads the outputs (a job output is
naturally idempotent by job, attempt, kind, slot and checksum, so a half-finished upload is safe
to repeat) and completes the job. After a restart the replay does the same and resumes
heartbeating the inherited claim while Store stays away. The output folder is removed when the
publication is delivered or dropped; an orphaned folder older than an hour is swept.

On stop, the worker stops claiming and waits `shutdown_grace_seconds`, then reports each job still
running as `worker_crashed`, which is transient, so Store retries it. A job claimed while stop()
runs, or still queued in the executor when it shuts down, is released with `requeue` and its slot
returned, so Store does not hold its lease until expiry.

## Handler interface (for `handlers/real/`)

A handler turns one claimed job and its resolved input artifacts into the canonical output
artifacts of its job type, plus usage. It never talks to Store and never picks slots, keys, result
states or follow-ons. `handlers/base.py` is the source of truth.

```python
from call1.process.handlers import (Handler, HandlerJob, HandlerResult, Output, Usage,
                                    HandlerError, ReleaseJob, JobCancelled, HandlerRegistry, RegistryContext)

class Handler:                       # subclass it
    job_type: ClassVar[JobType]      # one handler per job type
    adapter_id: ClassVar[str]        # recorded in AttemptProvenance.adapter_id
    adapter_version: ClassVar[str]
    def ready(self, job: HandlerJob) -> None: ...        # optional; raise ReleaseJob to refuse before inputs are fetched
    def run(self, job: HandlerJob) -> HandlerResult: ...
```

**Input: `HandlerJob`**

- `claimed` is the contract `ClaimedJob`. The job's fields are exposed directly: `job`,
  `job_type`, `parameters`, `selection` (the `FrozenSelection` or `None`), `attempt_number`,
  `final_attempt` and `upstream` (`List[UpstreamOutcome]`).
- `catalog_entry` is the Process `CatalogEntry` for the frozen selection. It carries
  `model_directory`, `runtime`, `provider_model_id` and `model_revision`. Weights live under
  `CALL1_MODELS_DIR` (default `data/models`) in `/<model_directory>`.
- `input(role)` returns an `InputArtifact`, or `None` for an absent optional input.
  `require(role)` returns the input or fails the attempt with `input_unavailable`.
  `inputs_with_prefix("assessment:")` returns every input whose role starts with that prefix.
- An `InputArtifact` holds `.artifact`, the contract `Artifact`, and `.ref`, its `ArtifactRef`.
  `.read_bytes()` and `.path()` return the content, downloaded lazily into this attempt's scratch
  directory and checksum-verified. `.json()` parses it. `.content()` returns the kind's contract
  content model: `TranscriptContent`, `RubricSnapshotContent` and so on.
- `transcript()` returns the `transcript` input with the `speaker_attribution` input applied.
  `attribution()` returns the attribution alone.
- `rubric()` returns the `RubricDefinition` from the pinned snapshot. `criterion()` returns this
  job's `RubricCriterion`. `scorecard_rubric_ref()` returns the `ScorecardRubricRef` the scorecard
  must name.
- `check_cancelled()` raises `JobCancelled`; call it between expensive steps. `progress(fraction,
  note)` is reported with the next heartbeat, as safe text only. `scratch_dir` is this attempt's
  directory and is removed afterwards.

**Output: `HandlerResult(outputs, usage=Usage(), model_revision=None, partial_reason=None)`**

- `outputs` maps each output role of `JOB_TYPE_RULES[job_type].outputs` to exactly one
  `Output(content, content_type=None, sensitivity=None, slot=None, labels={})`, plus at most one per
  role of `optional_outputs` (1.3.0: the `asr` job's `base_transcript` and `vocabulary_pass`,
  docs/DualAsr.md). Any other role fails the attempt.
  - For a JSON kind, `content` is the contract content model. The core dumps it canonically,
    checksums it and uploads it inline or through a grant.
  - For opaque kinds, `content` is bytes or a `Path`.
  - The slot and sensitivity default per job. The defaults are the criterion ID,
    `escalation:<id>`, `segment:<n>`, `synthesis:final` or `synthesis:<a>-<b>`, `<pass>:0`, or
    `""`. The core wraps each slot in `draft:<request_id>:` for draft tests.
- `Usage` holds `tokens_input` and `tokens_output` (`TokenCount`, or `None` for unavailable, never
  estimated silently), `model_load_seconds`, `inference_seconds` (default: the wall time of
  `run`), `audio_seconds_processed`, `peak_memory_bytes`, `provider_reported_model_id` and
  `billing_units`.
- `model_revision` is the exact revision, when it differs from the frozen one.
- `partial_reason` applies to publishing types only; it publishes the result as `partial`.

**Errors**

- `HandlerError(JobErrorCode, detail, outputs=None, usage=None)` fails the attempt. The code comes
  from the contract; `detail` is safe text. For QA types, a `PROVIDER_FAILURE_CODES` code on the
  final attempt becomes a FLAGGED assessment. Pass the attempt's `prompt_input` in `outputs` when
  one was built.
- `ReleaseJob("requeue" | "reject", code, detail, not_before_seconds)` must be raised before
  inference starts, and no attempt is consumed. `requeue` takes `resource_unavailable` or
  `input_unavailable`; `reject` takes configuration codes such as `model_unavailable`,
  `context_limit_exceeded` or `credential_missing`.
- `JobCancelled` comes from `check_cancelled()`. Any other exception becomes `worker_crashed`,
  which is transient.
- For a QA answer that fails schema or quote validation, return a FLAGGED `QaAssessmentContent`
  with `trigger: invalid_answer`; do not raise.

**Registration hook.** In real mode `build_registry` imports `call1.process.handlers.real`. If that
module defines `register(registry: HandlerRegistry, context: RegistryContext)`, the registry calls
it. `context.config` is the `ProcessConfig` and `context.catalog` the `ProcessCatalog`. Call
`registry.register(handler)` once per job type; a later registration replaces the code handler for
that type. You may also set `registry.entry_status = fn(entry) -> (CatalogEntryStatus,
[qualified purposes])` to report installation and qualification. The default counts an entry
installed when `models_root/<model_directory>` exists. A package that fails to import is reported
in the console's notes and never stops Process.

The code handlers are already registered in every mode: `qa_scorecard` (the pre-split
`RubricEvaluator` scoring), `summary_assembly` and `contact_signals_merge`. `embeddings` is a model
stage since contract 1.2.0 (`handlers/embeddings.py`, registered by the fake registry and by
`real/__init__.py`).
`transcripts.py` holds the shared helpers: attribution, speaker correction, segment planning and
the transcript fingerprint. Real handlers import runtimes such as MLX and torch lazily, inside
`run`.

**The import boundary.** `call1/pipeline/__init__.py` used to import `queue_manager`, which imports
`call1.db.repository`, so any `call1.pipeline.<module>` import loaded `call1.db`. Its exports are
now lazy (a PEP 562 module `__getattr__`; `from call1.pipeline import QueueManager` still works),
and the audio validator moved from `call1.ingest.validator` to `call1.pipeline.audio_validation`
(the old path re-exports it). The boundary test imports every module under `handlers/real/`.

## Real handlers (`handlers/real/`)

`python -m call1.process serve` in real mode (the default) runs the pre-split pipeline modules
behind the handler interface. With `CALL1_BACKEND=mlx` (Apple Silicon) it is the appliance stack:

| Job type | Module | Wraps |
|---|---|---|
| `validation_vad` | `media` | `AudioValidator` (ffprobe, codec list, 5 s to 90 min) and `VoiceActivityDetector` |
| `asr` | `media` | `MLXAdapter.transcribe` (Parakeet TDT 0.6B v3 via mlx-audio) with `CALL1_BACKEND=mlx`, else `LocalTranscriber` (faster-whisper, `CALL1_ASR_MODEL`) |
| `speaker_attribution` | `media` | `diarize` (Nemotron-3-Diarization, up to 8 speakers) and the pre-split cluster rule, per whole turn; reviewer corrections |
| `acoustic_tone` | `analysis` | `analyze_tone_blocks`: MERaLiON SER seven-second speaker blocks |
| `text_sentiment` | `analysis` | `analyze_text_sentiment`: Cardiff RoBERTa per turn |
| `enrichment` | `analysis` | `extract_numeric_references`, plus the `pii_findings` output (contract 1.2.0): `openai/privacy-filter` spans through `masking.pii_findings` |
| `qa_deterministic` | `qa` | `RubricEvaluator._evaluate_criterion` for every non-semantic check, then the grounding guardrail |
| `qa_criterion`, `qa_escalation` | `qa` | The `QuestionRouter` path for one model: policy and speaker gates, `_semantic_prompt`, strict parsing, `verify_quoted_evidence` |
| `summary_segment`, `summary_synthesis` | `summary` | `call1.summarizer` chunk and synthesis prompts, one structural retry, the per-segment citation check |
| `summary_assembly` | `summary` | The whole-call citation check and the source-point fallback (replaces the code stage in real mode) |
| `contact_signals_lifecycle`, `_resolution` | `signals` | One `extract_contact_signals` pass each, with its quote and speaker provenance |

`qa_scorecard` and `contact_signals_merge` stay the code stages. The scorecard now also names a
deterministic check's guardrail rejection, as `evaluate_deterministic` did.

### Search embeddings (contract 1.2.0)

The `embeddings` job embeds every turn with `call1/embedding.py`, the same module Store embeds
search queries with (decision 18):

| Handler | When | Scheme |
|---|---|---|
| `NemotronEmbeddingsHandler` (`call1.torch.nemotron_embed`) | real mode | `nemotron-3-embed-1b@c0c9fea`: `nvidia/Nemotron-3-Embed-1B-BF16`, turns as `passage: <text>`, mean pooling, L2, 2048 dims |
| `FakeEmbeddingsHandler` (`call1.fake.embeddings`) | fake mode, or `CALL1_EMBEDDING_BACKEND=fake` | `fake-embedding-v1`, deterministic token hashing, no model |

- **Weights**: `python -m call1.embedding download` fetches the pinned revision into
  `data/models/nemotron-3-embed-1b` (`CALL1_EMBEDDING_PATH` overrides). Without them the claim is
  refused with `model_unavailable` before inference, and a job that froze another revision is
  refused with `model_unqualified`.
- **Runtime**: transformers `AutoModel`/`AutoTokenizer` on torch, no remote code. The checkpoint is a
  bidirectional encoder (`is_causal: false`, honored by transformers 5.x's mask builder; the opt-in
  test checks it). CPU in bfloat16 by default, so Process and Store compute identical vectors.
  Batches of 16 turns, with a cancellation check between batches. Process drops the model after
  each job; Store keeps its own copy resident.
- **Speed and memory** (M3 Pro, 100 turns): bfloat16 on CPU (default) about 26 s and 2.3 GB while
  it runs; `CALL1_EMBEDDING_DTYPE=float32` about 5 s and up to 4.8 GB; `CALL1_EMBEDDING_DEVICE=mps`
  about 2 s. Vectors from these settings agree to cosine 0.9999+ with the default's, so rankings do
  not change, but set the same values on Store.
- **Backend agreement**: `CALL1_EMBEDDING_BACKEND` (`nemotron` or `fake`) wins; else fake-handler
  mode means the fake embedder. Store resolves the same variable (and `CALL1_PROCESS_HANDLERS`), so
  launch both apps with the same environment, as `python -m call1.launch` does.
- **Re-embedding**: calls embedded before 1.2.0 (`hashing-projection-v1`) are not searched until a
  reanalysis of kind `embeddings` runs, which plans one `embeddings` job on the linked transcript.

**Models.** Weights live at `CALL1_MODELS_DIR/<model_directory>` (default `data/models`). The
pre-split variables still win when set: `CALL1_MLX_ASR_PATH`, `CALL1_MLX_DIARIZATION_PATH`,
`CALL1_MLX_TEXT_PATH`, `CALL1_TONE_PATH`, `CALL1_SENTIMENT_PATH`, and `CALL1_OPTIONAL_MODELS_DIR`
for local packs. `registry.entry_status` (`paths.entry_status`) reports each entry as available only
when its runtime imports and its `config.json` and weights are present, so the worker releases a
claim it cannot serve before inference starts. Without `CALL1_BACKEND=mlx`, ASR is faster-whisper
(recorded as `model_revision: faster-whisper:<model>`), the included LLM is the loopback Ollama the
legacy app used, and diarization does not run: mono turns stay unattributed, as they did before the
split (`model_revision: not-run:needs-mlx`).

**LLM calls.** QA, summaries and contact signals all go through `call1.question_models.generate_text`,
so the Stage 0 Pro1 closure and the external-model gate apply unchanged. The frozen entry maps back
to the legacy `QuestionModel` (`call1-bundled`, or a local pack); `generate_text` takes an optional
`text_model_path` so the MLX weights come from the catalog. Only the appliance route is served:
`call1_confidential` is released with `route_policy_rejected`, and any other route with
`route_disabled`, before inference. Errors map to contract codes: transport failures are
`provider_error` (the core retries, then records a FLAGGED assessment on a QA job's final attempt),
the context budget is `context_limit_exceeded`, a missing model is `model_unavailable`, and an answer
that is not the expected JSON is `validation_rejected` for summaries and signal passes. For QA both
are outcomes instead: an invalid answer is a FLAGGED assessment with `trigger: invalid_answer`, and
a prompt over the context budget is a FLAGGED assessment with `trigger: provider_error` and the
attempt's `error_code: context_limit_exceeded`, completed at once (usage outcome `succeeded`,
because the contract records `failed` only for `PROVIDER_FAILURE_CODES`). That is the pre-split
router's behavior: the scorecard still publishes, and a criterion that escalates on
`provider_error` escalates. Tokens are recorded only when
a provider reports them (in-process MLX does not, so they are `unavailable`). MLX peak memory is
measured per attempt.

**Masking.** Model inputs are masked when the frozen route says `masked` (the pre-split value set:
sensitive numeric entities plus the PII patterns, over the whole call). The planner sets it for
every route class in `MaskingSettings.masked_route_classes` and, on the appliance route, for the
text-model purposes (QA, summaries, contact signals) while Store's text masking is on. That is
`MaskingSettings.mask_reviewer_reads`, the split's `redaction.text`, on by default (admin state is
deferred, so the contract default applies). This keeps the legacy behavior (team decision 3): the
pre-split router and summarizer masked prompts even for the included local model whenever
`redaction.text` was on. `mask_model_text: on|off` in the Process config overrides the appliance
rule. ASR, diarization, tone, sentiment and embeddings selections are never marked masked, so a
transcript stays `raw`. Masked outputs are `masked` sensitivity. Jobs on a masked non-appliance
route pin the `enrichment` artifact; on the appliance the handler runs the same deterministic
extractor in-process, so masking adds no graph edge. QA quotes are verified against the masked
text the model saw, and the summary assembly checks citations against the same masked text (the
planner records the flag in its parameters, since the assembly has no selection).

**PII model (team decision 19).** On a masked route the value set is the union of the number rules
(`call1.redaction.find_pii`: separators, spoken digits and code-context money formats) and
`openai/privacy-filter`'s spans (`call1.pii_model`): account numbers, phones, emails, addresses,
URLs, secrets and person names, never dates and never the agent's own name (`agent_display_name`
from the call metadata, read through `HandlerJob.call_metadata()`, plus agent self-introductions).
Since contract 1.2.0 the model runs once per transcript revision, in the `enrichment` job, which
writes the filtered spans as its `pii_findings` output (bound to the transcript it read, and given
the `speaker_attribution` input on mono calls so agent self-introductions are recognized). Every
masked text-model job (QA criterion and escalation, summary segments and assembly, the
contact-signal passes) pins that output as input role `pii_findings` and so waits for
`enrichment`. It uses the findings when they were made from its own transcript input; otherwise,
for example in a graph planned before 1.2.0 or on a pinned artifact from another revision, it runs
the model itself. The masking helper (`handlers/real/masking.py`) loads the model inside the job,
under the inference lock, and releases it after (MPS when available: about 2.4-6 s to load, 1-3 s
per call, ~3 GiB on MPS, all of it freed on release). The spans are cached per transcript
in-process, so other jobs of the call do not load it again. `enrichment` stays in the `cpu` pool,
and the inference lock serializes its model load with other local inference. Weights live at `CALL1_PII_MODEL_PATH`
(default `$CALL1_MODELS_DIR/openai-privacy-filter`, `python -m call1.pii_model download`). A real
install without them fails the job with `model_unavailable`; fake-handler mode and the tests use the
labelled stub (`CALL1_PII_MODEL_BACKEND=stub`). The catalog lists the model as
`openai-privacy-filter` with no purpose (the contract has no masking purpose). Store masks reviewer
reads and mutes audio with the same `pii_findings`, and withholds text while a revision has none.
It runs no model for this (see `call1/store/results/README.md`, "Masking"). In fake-handler mode,
`FakeEnrichment` writes the findings with the stub, and the fake ASR's scripted action
`caller_name` (`CALL1_FAKE_BEHAVIOR={"asr": ["caller_name"]}`) produces a transcript in which the
caller gives a name, so the path can be exercised end to end.

**One MLX slot.** Every MLX-runtime job needs the `local_memory` slot, so the worker's single `mlx`
pool runs them one at a time, and the legacy `inference_lock` still serializes local inference in the
process (tone and sentiment take it too, as `enrich_sentiment` did). Since 2026-09-27 acoustic tone runs
MERaLiON on MPS by default on Apple Silicon (`CALL1_SENTIMENT_DEVICE` unset; CPU where torch has no
Metal; setting the variable pins both tone and text sentiment, which otherwise stays on CPU). Tone's
ffmpeg decode and seven-second block plan run before the lock; the model loads, scores and is released
(MPS cache emptied) inside it. CPU and MPS matched on the six benchmark calls: max |dV/A/D| 3.4e-6,
every emotion argmax equal, 0.37 s a block against 1.40 s (`benchmarks/2026-09-27-throughput.md`).
The PII model's raw spans are cached per call and transcript text, so `speaker_attribution` and
`enrichment` share one detection; Contact Signals stages 1 and 2 load Gemma once per job
(`mlx.keep_text_model_loaded`); the outlines JSON index is compiled before the lock.

## Dual transcription (contract 1.3.0, decision 33; docs/DualAsr.md)

Parakeet stays the transcript; Whisper Small, prompted with the customer's vocabulary in every 30 s
window, only finds candidates; a deterministic rule merge writes a vocabulary term over the Parakeet
words it overlaps in time when they sound and spell alike. It never inserts.

- **Planning** (`vocabulary.py`, `graph.py`). Ingest and full reanalysis read `getAsrVocabulary`
  (a Store without the route means no vocabulary). When the record is `active`, the `asr` job
  freezes `parameters.asr_vocabulary`: the effective terms, their digest and the catalog's
  `asr_vocabulary` entry `whisper-small-vocab`, frozen even when it is not installed. The graph's
  refs and edges do not change; the `asr` estimate gets `estimated_runtime_seconds` at 3.5x
  Parakeet's measured real-time factor. A repeated ingest after the vocabulary changed re-plans
  with the vocabulary the first graph froze, so it returns that graph.
- **The job** (`handlers/real/media.py`, `handlers/real/vocabulary.py`). Parakeet runs as before,
  then the vocabulary pass (`call1/pipeline/vocabulary_asr.py`) and the merge
  (`call1/pipeline/vocabulary_merge.py`). Outputs: `transcript` (merged, `vocabulary_correction`
  set), `base_transcript` (Parakeet's own, always) and `vocabulary_pass` (the raw Whisper pass, only
  when it ran). The worker accepts the job type's `optional_outputs` (`output_roles_ok`).
- **Failure never fails the call.** Missing catalog entry or install (weights, or
  `multilingual.tiktoken` missing or with the wrong checksum) and an import error are
  `model_unavailable`; no MLX backend and no word timings are `configuration_error`; a decode error
  is `validation_rejected`; memory exhaustion is `resource_unavailable`; any other runtime error is
  `provider_error`. Each keeps Parakeet's transcript as `base_only` with a safe note, links
  `base_transcript` and no `vocabulary_pass`, and logs the job ID and exception type only.
  Cancellation during the pass (checked before every Whisper window) cancels the job.
- **Catalog.** `whisper-small-vocab` (MLX, `CALL1_WHISPER_VOCAB_PATH` or `<models>/whisper-small`)
  is installed only with `config.json`, the weights and the pinned `multilingual.tiktoken`
  (`CatalogEntry.required_files`); the console's Models view says which file is missing (`detail`).
  `scripts/provision_models.py --models asr_vocabulary` downloads the pinned weights and bundles the
  tokenizer (`--tokenizer-from <local copy>` for no network, `--tokenizer-only` to add it to an
  existing directory).
- **Merge.** A port of `scripts/research/asr_merge/merge.py` with its exact numbers (vendored Double
  Metaphone, pure-Python Levenshtein). Candidates are per channel; a span crossing a turn, or whose
  words cannot be found in the turn text, is rejected; the turn's `text` and `word_timestamps` are
  rewritten, keeping punctuation outside the first and last word. The glossary shortlist ranks
  exactly as the research does, with pruning bounds (about 1 to 1.6 s per 10-minute call for 97
  terms); above 250 terms it prefilters windows by first letter or first phonetic letter (about
  6 s for 2000 terms).
- **Fake mode.** With `asr_vocabulary`, `FakeAsr` gives its script word timings, plays Parakeet
  (the script) and Whisper (the script with each `FAKE_MISHEARINGS` phrase written as its term, when
  the term is in the vocabulary) and runs the real shortlist and merge. The `vocabulary` script
  (`CALL1_FAKE_BEHAVIOR={"asr": ["script:vocabulary"]}`) yields three replacements with the retail
  seed ("standy cup", "Chad Stone", "after pay"); `vocabulary_fail[:<code>]` yields `base_only`;
  actions combine with commas.

## Contact Signals v2 (contract 1.3.0)

The design is `docs/ContactSignalsV2.md`; team decisions 21 to 24 are authoritative, and decision 24
(the latest) wins where they differ. v2 replaces the two whole-transcript Gemma passes with a
three-stage cascade over ~7 s segments. **All three stages run on the included model, Gemma 4 E2B
(`call1-bundled`)** (decision 24): the real catalog lists it for `signal_category`,
`signal_subcategory` and `signal_extraction` and makes it the default for each, so a real install
plans v2 on Gemma whenever Store's setting says `pipeline: v2`. Laya and Needle are deferred to a
later "system one" experiment; `CLASSIFIER_ENGINES` keeps the hook for one.

**Planning** (`graph.py`, `signals_input.py`).

- *Ingest.* After the rubric snapshot, Process reads `GET /signals/taxonomy`. When the settings say
  `pipeline: v2`, it mints a snapshot of the current version (`mintSignalTaxonomySnapshot`) and plans
  from the snapshot's taxonomy and settings. `v1` and `shadow` build today's passes with no snapshot;
  in shadow mode Store adds the compare companion request itself.
- *The branch.* `add_contact_signals` builds v2 only when both `signal_category` and
  `signal_subcategory` have a usable catalog entry. Gemma counts: it serves both purposes (decision
  24). With only one of the two usable, it builds v1 with `pipeline_note` "v2 selected; no qualified
  classifier on this host" when `v1_fallback` is on, else a lone merge that fails with
  `configuration_error`. A preview or compare
  request never falls back: it is rejected. The planner never substitutes a model within a purpose.
- *The graph.* `cs-categorize` (stage 1) requires the transcript, the attribution (mono) and
  `pii_findings`. `cs-subcategorize` (stage 2) waits on it by an `after` edge. `cs-extract` (stage 3)
  waits on both, and is planned only when an active node has fields or `narrow_quote`. `cs-merge`
  waits on all three by `after` edges and requires `pii_findings`. Every v2 job pins the snapshot
  under input role `taxonomy` and freezes its digest in `parameters.signals`. Stage 2 takes the tone
  and sentiment `after` edges only when its adapter version enables those factors; both are off.
- *Always masked.* The three v2 purposes are in `catalog.ALWAYS_MASKED_PURPOSES`: their selections
  freeze `route.masked = true` even with `mask_model_text` off (decision 22, Q12), and every v2 job
  requires `pii_findings`. So a failed enrichment dead-blocks every v2 job and the merge, the group
  reads `failed`, and retrying enrichment releases the chain.
- *Stage-3 fallback.* The extract job freezes `fallback_entry_id`: the settings' entry, else the
  `signal_extraction` default (Gemma), only when it differs from the primary, is usable here and is
  on the same route class.
- *Reanalysis.* Store resolves `signal_pipeline` and `signal_taxonomy_version` on the request. A v2
  `contact_signals` request pins the call's `signal_categories`, `signal_subcategories` and
  `signal_extraction` and runs only the stages the section 7.5 table names
  (`signals_v2.plan_rerun`): a scope's changed stage-1 digest reruns stage 1 for that scope; a
  threshold-only edit re-derives spans from the stored scores with no model (`stage1_mode:
  rederive`, a code stage). The re-derive is pure, so the planner computes its spans: stages 2 and 3
  run only on spans that are new or whose windows changed (stage 3 only where the node has fields
  or narrowing), and when there are none they are pinned, so the graph is the re-derive and the
  merge. A changed stage-2 digest reruns stage 2 for that category's spans; a
  changed stage-3 path reruns stage 3 for those spans; a `subcategory_threshold` edit is re-derived
  in the merge. A stage it does not rerun is pinned from the previous artifact. `rescore_signals`,
  a new transcript or attribution, `full` and `speaker_correction` rerun everything. A
  `contact_signals_preview` request runs every stage from the snapshot Store minted for it, into
  `draft:<request_id>:` slots, with preview hit IDs. Every job takes the request's `priority` on top
  of its own (+5 previews, -10 backfills and compare requests).

**Stages** (`handlers/signal_stages.py`, shared by the fakes and the real handlers).

| Stage | What it does |
|---|---|
| Segmenter | `call1/pipeline/signal_segments.py`: each turn cut into `max(1, ceil(d / 7))` windows at word gaps (sentence ends first), never inside a masked value; fewer than 3 words merge; offsets mapped into the masked turn; interpolated timing without word timestamps |
| `contact_signals_categorize` | One choice row per scorable segment over the speaker's active categories plus "none", with up to half the engine's context of preceding segments (decision 23). Every option at or above its threshold fires, at most two per segment; windows of one turn and block form a span, isolated with +/-1 segment. Unchanged scopes' scores are carried forward. A re-derive rebuilds spans from stored scores |
| `contact_signals_subcategorize` | One row per span: the active subcategories, "Other" and "Not". "Not" at or above tau_reject rejects the span. Decisions with an unchanged stage-2 digest carry forward. No categories input (stage 1 failed): an empty result, no model |
| `contact_signals_extract` | The path's fields on each span (at most 24 per call), grounded in the core span's masked text; `withheld_pii` for leaked values; the narrowed quote. Extractions with an unchanged stage-3 digest carry forward. Spans that fail on the primary rerun on the declared fallback in the same job (`source: fallback`) |
| `contact_signals_merge` (v2 branch) | Hits from the stage artifacts: section 6.3 hit IDs, the v1 fields filled, quotes and field text re-verified against the masked turn at their offsets (failures are dropped), then one speaker's consecutive hits with the same category and stage-2 outcome merged into one multi-segment hit (decision 25, `signals_v2.merge_multi_segment`: no same-speaker turn between, gap at most `MERGE_GAP_SECONDS` = 20 s; the first keeps its hit ID, later spans become `parts`). A missing categorize output fails it; a missing stage 2 or 3, or spans past the extraction cap, make it `partial` |

Every stage re-masks the call the same way (the number rules plus the pinned `pii_findings`), so the
merge rebuilds exactly the masked turns the stages saw. Taxonomy text (glosses, names, descriptions,
examples, enum values) is masked with the call's values before any engine sees it.

**Engines.**

- *Fake mode.* `fake-signal-classifier` (both classifier purposes) and `fake-signal-extractor` register
  and are the defaults, labelled "(fake)". The classifier scores 0.9 on a segment that contains a
  category's keyword or example (fixed keywords for the built-ins), and picks the first subcategory
  whose example is in the span ("Not" on "not really"). The extractor takes the first enum value in
  the span, the first sentence for strings and the first number; dates and booleans stay absent.
  Grounding, masking and the merge run for real. `FakeBehavior` keys `contact_signals_categorize`,
  `_subcategorize` and `_extract` take the usual actions plus `invalid_answer`, `provider_error` and
  `low_confidence` (scores at the threshold minus 0.05).
- *Fake scripts.* `FakeAsr` emits `SCRIPT` by default; the `asr` action `script:<name>` (`cancel`,
  `competitor`, `caller_name`, `returns`: one multi-segment `fix_proposed` signal) or `FAKE_SCRIPTS_BY_SOURCE` (the source audio's checksum, extended by
  `CALL1_FAKE_SCRIPTS`, a JSON object) picks another. Both new scripts keep `SCRIPT`'s speaker order
  and put the key line in the first three turns.
- *Real mode, stages 1 and 2* (`handlers/real/signals_v2.py`). `RealSignalsCategorize` and
  `RealSignalsSubcategorize` run `GemmaSegmentClassifier` on `call1-bundled`. Stage 1 packs
  consecutive segment rows into token-budgeted prompts (the option legend once, the first row's
  preceding context once) with a constrained `{segment: {labels}}` answer of at most two picks;
  stage 2 sends span rows with a constrained `{assessment, fits, choice}` answer, where `fits: no`
  is "Not". Picks map to fixed scores (0.9 and 0.7; none 0.05 when something was picked, else
  0.95), and the engine's thresholds (0.5, `signal_stages.ENGINE_DEFAULTS`) sit between them, so a
  pick fires and an admin threshold above 0.7 drops second picks. An answer cut off at its bound is
  retried once with twice the bound; invalid JSON or a skipped row is `validation_rejected`. The
  prompts go through `LlmTransport` like QA's, so `is_call1_operated` and `check_route` apply, and
  the attempt records the transport's usage. Any other entry refuses the claim (`model_unavailable`,
  before inference) unless a later engine registers in `CLASSIFIER_ENGINES`; a re-derive always
  runs. Classifier calls run inside `inference_lock`. The prompts, schemas, scores and thresholds
  were tuned on real calls (`benchmarks/2026-09-26-gemma-signals-v2.md`).
- *Real mode, stage 3.* `RealSignalsExtract` runs the included model:
  the token-budgeted batcher packs spans while input plus the summed output bound stays within 7,500
  tokens (measured with the pinned Gemma tokenizer when installed, else a conservative estimate),
  sets `max_tokens` per batch, and sends one constrained-JSON prompt per batch through
  `LlmTransport` (so `is_call1_operated` applies). A span that does not fit alone is `over_budget`.
  The whole stage, fallback included, holds `inference_lock`. `check_route` refuses
  `call1_confidential` and every non-appliance route before inference.

### The rules engine (contract 1.4.0; docs/SignalsEmbeddings.md "Build status")

A category whose recipe engine is `rules` is decided by the rules engine inside the existing jobs.
The legacy `SignalSettings.detection` field no longer overrides category recipes:

- *Categorize.* `run_categorize` wraps its engine with `handlers/signals_rules.maybe_rules_engine`
  (fake and real mode alike). The `RulesClassifier` embeds the job's masked segments with
  `call1.embedding` (`passage:` prefix; Nemotron on the Apple GPU in real mode, the fake embedder on
  fake handlers) inside `run_classifier`'s `inference_lock` hold, then drops the embedder and frees the
  MPS cache before any Gemma work. The bank is the pack the taxonomy pins
  (`data/signal-banks/<bank_id>.json` or `CALL1_SIGNAL_BANK_DIR`, digest-checked) plus entries written
  from the taxonomy's own text; its vectors are cached under `<data>/signal-bank/`
  (`CALL1_SIGNAL_BANK_CACHE` overrides) keyed by the scheme and texts. `call1/pipeline/signal_rules.py`
  evaluates the recipes (kNN share, lexicon with negation veto, speaker, call position; at most two
  fires per segment). Rules categories answer at Gemma's pick scores; categories without a rules
  recipe go to the wrapped engine on the same rows with only their options, and Gemma is loaded only
  when such rows exist. The artifact gains `rules` (provenance, funnel counts, timings) and one
  `rule_decisions` entry per rules span; `catalog_entry_id` is `call1-signal-rules`.
- *Subcategorize.* A rule-decided span passes its kNN subcategory through (`source: rules`, no
  model); a span whose recipe has `check: gemma` goes to today's stage-2 prompt, which confirms it
  (`checked: true`) or rejects it ("Not"). No model loads when no span needs one.
- *Merge.* Every hit of a result in which the rules ran carries `why` (category and subcategory
  source, the check, the rule decision).
- *Fail closed.* A missing embedder fails the categorize job (`model_unavailable`; real mode also
  refuses the claim at `ready` when the weights are missing), and a missing or altered bank pack
  fails it with `input_unavailable`. Never "no signals".

## On-device training (decision 28; docs/OnDeviceTraining.md)

The customer's LoRA is trained **on this host** from the customer's own reviewer labels, and it
improves the included model (`data/models/gemma-4-e2b-it`). It is open core and free: nothing in
`training/` reads a Pro1 connection or entitlement. The labels, the datasets and the adapter never
leave the host; Store sees only the label reads (audited) and, later, the adapter version in each
attempt's `model_revision`.

**Demo model stack.** Settings shows a labelled Gemma → Call1 industry LoRA → private LoRA
simulation when `CALL1_STORE_DEMO=1`. Both adapters are represented as active together; private
training requires an industry edition and pausing private preserves Call1. This preview uses
browser localStorage, not model weights or training APIs. The live MLX runtime still resolves one
adapter; actual composition and private training against it remain to be implemented. Live tools
remain available in Settings under “Live model and training tools”. Models presents the processing
stages as an interactive graph with catalog details on selection.

**Schedule (Settings → On-device training).** Off by default. When on, the `call1-training` thread
wakes every 30 s; a due occurrence (daily, or weekly on a weekday, at `HH:MM` in the host's time
zone) opens a 4-hour start window. It counts new labels once (`listTrainingLabels?after=<cursor>&limit=0`,
not audited): below `min_new_labels` it records a *check* (`state.json`), otherwise a run waits for
the start rule until the window closes (`skipped: busy`). **Train now** skips the schedule and the
threshold. Training is unavailable (409 `training_unavailable`) on a host that is not the primary
host, without MLX (unless the trainer is `fake`), or without a Store connection.

**Start rule and claim pause.** A run starts only when the `mlx` and `torch` pools are empty and
Store lists no `QUEUED`/`RUNNING` `local_memory` job (`listJobs` with `memory_slot`, limit 1);
scheduled runs with `only_when_idle` also need every pool empty, no queued or running job of any
slot and no ingest in the last 15 minutes. Then `worker.pause_claims(run_id)` stops every pool and
the reanalysis consumer from claiming (heartbeats and the spool go on), running jobs get 10 minutes
to finish, the rule is rechecked, and the run proceeds with `inference_lock` held through training
and evaluation. Claims always resume in a `finally`. `worker.describe()` and the overview report
`claims_paused: {run_id, since, until}`.

**A run** (`training/runner.py`): collect (page the label log from `seq` 0; the newest label per
subject wins, a withdrawal removes it) → build (replay each label's source job from its `sources`
with `getJob` and `getArtifactContent`; rebuild the prompt with the engine's own builder,
`signals_v2.render_batch`/`pack_rows`, `qa.qa_prompt` or `media.roles_prompt`, always masked with
the pinned `pii_findings`; a label without findings is skipped, never trained unmasked; answers must
parse back through the engine's parser) → split by call (`sha256(installation:call) % 100`: 20%
held out, 10% validation) → minimums (`skipped`, no GPU) → train (`mlx_lm lora`, thinking-off template, via `call1.process.training.mlx_lora` in a subprocess
with a minimal offline environment; `fake_trainer` in fake mode and every test) → evaluate the
active adapter and the candidate on the held-out prompts production sends → promote only when not
worse. `work/<run_id>/` (0700) is deleted at the end of every run; the raw inputs go as soon as the
masked examples are written.

**Adapters** (`training/registry.py`, `data/process/adapters/`, 0700): versions `ft-<UTC>`, an
atomic `active.json` pointer with `previous`, rollback to any kept version or to the base,
retention of `keep_versions`, and a base fingerprint that sets an adapter aside when the included
model changes. `LlmTransport` resolves the adapter **once per job** for the job's task
(`signal_stage1`, `signal_stage2`, `qa_verdict`, `speaker_roles`; summaries and stage 3 always run
on the base) and wraps each generation in `call1.adapters.mlx.use_text_adapter`;
`CALL1_TEXT_ADAPTER` stays the manual override (`+lora.env`). The attempt's `model_revision` and
`SignalStageProvenance.model_revision` become `<revision>+lora.<version>`, and stage-1/2 answers
from another adapter are never carried forward.

**Ownership.** One process owns `data/process/training/`: the one holding the `flock` on its
`lock` file. `serve` claims it when its scheduler starts. Only the owner recovers an interrupted run
(records it `interrupted` and deletes `work/`), queues runs or ticks the schedule, so a CLI command
run beside a `serve` (`status`, `drain`, `training status`) never touches that serve's run.

CLI: `python -m call1.process training status` (the console's view as JSON; a run owned by a
running serve shows from `state.json`) and `training run` (train now in this process). `training
run` refuses (exit 2) while a `serve` owns training: use Train now in that serve's console, whose
claim pause covers the run. Without a serve it runs in-process; do not start a `serve` or `drain`
beside it until it ends. Local-only qualification (a real `mlx_lm` run, template parity,
memory per host, `it_per_s`) is docs/OnDeviceTraining.md section 7.4.

## Operator API (for the console)

The API is loopback only: the client address and the `Host` header must both be loopback. Reads
need nothing more. Writes need `X-Call1-Console-Token` (or `Authorization: Bearer`), and a
cross-origin `Origin` is refused. Errors are `{code, message, details}`.

| Route | What it returns |
|---|---|
| `GET /process/api/health` | `{ok, state}` |
| `GET /process/api/session` | `{console_credential_configured, token_valid, write_header}` |
| `GET /process/api/overview` | Store URL, dev mode, contract version and compatibility, parameters, runtime state; handlers (mode, notes, missing types); worker (state, stats, pools with sizes and job types, running jobs, spooled count); reanalysis stats; catalog version and publication; scratch bytes; Evaluate URL |
| `GET /process/api/conversations?limit=` | Ledger entries with `progress` (Store's `JobGroupProgress`), `progress_line` ("Transcript ready · QA 3/4 · Summary analyzing") and `evaluate_url`. Progress reads make one short try each (3 s, no retry backoff); after the first unreachable answer the rest of the listing is served from the last progress seen, with `progress_error: "store_unavailable"`, `progress_stale: true` and a top-level `store_unavailable: true`, so the Pipeline view answers at once during an outage |
| `GET /process/api/conversations/{id}` | The conversation, graphs, progress, and every job: status, attempts, waiting reason, blocking, error code, entry, route class and destination |
| `GET /process/api/jobs/{id}` | The contract `Job` and its `Attempt`s (provenance, safe error codes; never content) |
| `POST /process/api/jobs/{id}/retry` `{reason}` | Store retry through Process's key; needs `jobs:control`. A 403 `insufficient_scope` explains how to add it |
| `POST /process/api/jobs/{id}/cancel` `{reason, cascade}` | Store cancel; same scope rule |
| `GET /process/api/catalog` | Entries (status, qualified purposes, runtime, route), defaults, handlers, masking |
| `POST /process/api/recordings` | Multipart `file` plus optional `agent_id`, `agent_display_name`, `agent_extension`, `agent_channel` and `external_call_ref` → `{conversation_id, call_id, graph_id, conversation_created, graph_created, jobs, evaluate_url, metadata_updated, updated_fields, agent_label}`. Metadata the contract rejects answers 422 `validation_failed` before anything is registered |

| `GET /process/api/training` | On-device training: `{available, unavailable_reason, settings, trainer, timezone, next_run_at, labels: {total, new_since_last_run, error}, status: {phase, run_id, trigger, detail, progress, claims_paused}, last_check, active, versions, runs, notices}` (the last 20 runs) |
| `PUT /process/api/training/settings` | `{enabled, schedule: {frequency, weekday, time}, min_new_labels, max_duration_minutes, only_when_idle}` → `{settings}`. 422 `validation_failed` with `details.field`; 409 `training_unavailable` when enabling on a host that cannot train |
| `POST /process/api/training/runs` | Train now: 202 `{run}`; 409 `training_busy` or `training_unavailable` |
| `POST /process/api/training/runs/{run_id}/cancel` | Idempotent cancel → `{run}`; 404 for an unknown run |
| `GET /process/api/training/runs?limit=50` | `{items}`, the run in progress first, then finished runs newest first |
| `POST /process/api/training/active` | `{version: "<id>" \| null}`: rollback or reactivation (no evaluation) → `{active}`; 404 unknown version, 409 `training_busy` or `stale_base` |

Evaluate deep links are `store_url + "/#/calls/<call_id>"`.

**Console credential.** The first `serve` issues it, or `console-token --rotate` replaces it.
Process prints it once as `http://127.0.0.1:8020/#console_token=c1con_…` and stores only its
SHA-256 hash. The console should read the token from the URL fragment, clear the fragment, keep
the token in memory or `sessionStorage`, and send it on writes.

The console itself is served from `call1/process/static/`: `index.html` or `process.html`, plus
assets. When the console is not built, `/` shows a "Not built yet" page instead.

## Tests

```bash
.venv-local/bin/python -m pytest tests/process tests/test_split_boundaries.py -q -p no:warnings
```

`tests/process/conftest.py` runs Store's `create_app` in-process with a `ManualClock`. The Process
Store client reaches it through a `TestClient` used as its httpx client, so no ports are bound. The
tests cover:

- **End to end**: ingest of `sample_audio/call_01_compliant.wav` runs every stage, and Store shows
  the transcript, scorecard, summary, contact signals and search. Mono attribution runs.
- **Idempotency**: a repeated ingest returns the same conversation and graph.
- **Claims**: two workers racing never share a claim.
- **Retry**: a console retry works; a key without `jobs:control` gets the 403 explanation.
- **Recovery**: a completion spooled during a Store outage keeps its claim alive through
  heartbeats and is delivered while the worker runs, without a restart and without redoing the
  work; a cancel that arrives meanwhile becomes a `cancelled` failure; a spooled completion is
  replayed after a restart; a crashed worker's lease expires so another worker finishes the job;
  jobs claimed while the worker stops are released, not stranded.
- **Deferred publications**: outputs that meet a Store outage (upload or follow-on planning) are
  spooled, heartbeated and delivered once without redoing the work, after a restart too (where
  the restarted Process heartbeats the inherited claim); a cancel requested meanwhile becomes a
  `cancelled` failure with nothing uploaded; an expired claim becomes late usage. End to end
  against real servers: `tests/e2e/test_process_control.py::test_p7_*`.
- **Agent identity**: upload fields and CLI flags reach the call, a changed re-upload updates it
  (`metadata_updated`) without a new graph, blanks never reset stored values, bad values are a 422.
- **Worker control**: cancel through the heartbeat, release without an attempt, graceful stop.
- **Large outputs**: output over the inline limit goes through upload grants.
- **QA**: escalation follow-ons, invalid answers, and final-attempt provider failures.
- **Summaries and reanalysis**: multi-segment summaries; qa, speaker-correction, draft-test and
  embeddings reanalysis (a call indexed under `hashing-projection-v1` becomes searchable again);
  rejection of a request that cannot be planned.
- **API**: the operator API and console rules.
- **Dual transcription** (`test_vocabulary_merge.py`, `test_dual_asr.py`): the research unit tests
  ported; the research's TEST decisions on committed word lists (`fixtures/dual_asr_test_calls.json`:
  19 hits, 6 replacements, "standing cup" rejected at phonetic 0.62); Double Metaphone parity with
  the Metaphone package; the merge's text and word rebuild, turns, channels and never inserting; the
  planner, catalog and tokenizer install; every failure row of docs/DualAsr.md section 7 with fakes;
  the fake ASR; the worker's optional outputs; and three runs through the in-process Store.
- **Contact Signals v2** (`test_signals_*.py`): the segmenter (no segment crosses a turn, cuts
  never inside a masked value, exact masked substrings, interpolated timing, UNKNOWN and SYSTEM
  counts, and the section 2 AppTek counts: 988 telephony and 3,802 retail windows, the two 2-block
  retail turns); the graph (shape and `after` edges, forced masking, factor edges, the v1 fallback
  and `configuration_error`, stage-3 fallback freezing, previews, priority); every section 7.5 row
  and the consumer's graphs; the merge (hit IDs, masked-quote re-verification, absence, partial
  results, the extraction cap, the v1 branch); stage 3 (grounding per type, `withheld_pii`,
  narrowing, the batch schema, the batcher at the caps, the in-job fallback, route refusals); the
  fakes, scripts and masking; the Gemma stage-1/2 engine (`test_signals_gemma.py`: packing,
  schemas, pick scores, validation, cancellation, the retry); three runs through the in-process
  Store on fakes (ingest on v2, an enrichment failure dead-blocking the chain until retried, a
  reanalysis that reruns only stage 3); and `test_signals_real_gemma.py`, the real registry and the
  real catalog through the in-process Store with only `generate_text` scripted: `pipeline: v2`
  plans and runs every stage on `call1-bundled` with masked prompts (also with `mask_model_text:
  off`), a field edit reruns only stage 3 on Gemma, a threshold edit is a re-derive and the merge
  with no model call, no raw value planted in the taxonomy reaches a Gemma prompt, and the
  classifier stages refuse confidential and non-appliance routes.
- **Units**: the Store client, config, planner (including appliance text masking), the search embedder's backend selection and fake,
  and the launcher's first run.
- **Launcher demo mode** (`test_launch.py`): the sample metadata, the handler choice, the isolated
  environment, `--reset`'s guard and the browser URLs; then `python -m call1.launch --demo` itself
  with fake handlers on spare ports and a temporary root under `/private/tmp/call1-e2e/`: all five
  calls seeded with their agents and settled, the console token valid, Store's demo sign-in on, the
  log free of credentials, the retail seed and pipeline v2 in Store with v2 hits on every call and
  the cancel script on call 05, a second start that keeps the seed, and `--reset`. The seed is
  applied once per demo root, and a seed that no longer applies falls back to setting the pipeline.
  A taken port is refused.
- **Real handlers, fakes for the runtimes** (`test_real_handlers.py`): each handler's contract
  output and error mapping, parity with the pre-split functions (tone blocks, text sentiment,
  deterministic verdicts), quote verification, masking, the Pro1 closure, summary retries and
  citation checks, and signal provenance, a QA context overflow as a FLAGGED assessment, and
  in-process entity extraction for appliance masking. `test_real_handlers_store.py` runs the real
  registry through the in-process Store with only the model runtimes faked (stereo sample and a
  mono call; masked prompts with a PII script, on by default and off by config; a QA context
  overflow that still publishes the scorecard).
- **On-device training** (`test_training_*.py`, shared builders in `training_support.py`; fake
  trainer and fake generator only, never MLX, torch or `CALL1_REAL_MODELS`): settings and their
  0600 persistence, the schedule (daily, weekly, Phoenix and a DST zone, the missed window, the
  threshold check), the start rule and claim pause against a fake Store client and fake pools
  (claims resume in every outcome), golden examples for every builder on the fake-script
  transcripts (`golden/training/*.jsonl`, regenerate with `CALL1_REGEN_GOLDEN=1`), masking, parser
  round trips, QA digest parity with `prompt_input.prompt_digest`, supersession, dedup, exclusions,
  the split and budgeting, the runner end to end for every outcome, the registry (pointer,
  rollback, pruning, base fingerprint), `text_adapter_path` precedence, `+lora.<version>`
  provenance and one adapter per job, the console routes, no GPU modules in a fake run, and no
  Store writes from the runner.
- **Real models, opt-in** (`tests/process/real/`, marker `real_models`): skipped unless
  `CALL1_REAL_MODELS=1`. It ingests `sample_audio/call_01_compliant.wav` with `CALL1_BACKEND=mlx`
  and the weights in `data/models`, then runs Nemotron-3-Diarization on a mono downmix. Run it with
  `CALL1_REAL_MODELS=1 .venv-local/bin/python -m pytest tests/process/real -m real_models -s -p no:warnings`.
  On an M3 Pro (18 GB), 2026-09-25, the 44 s stereo call settled in 66 s wall clock with every job
  succeeding on its first attempt: tone 14.9 s, the two signal passes 14.5 s and 13.9 s, four QA
  jobs 9.5 s in total (two call the model; SEC-01 and COMP-01 are flagged by the policy gate), ASR
  4.7 s, the summary segment 3.9 s, text sentiment 3.8 s, and the rest under 0.2 s. The LLM reloads
  its weights for every request, as the pre-split adapter does. On the mono downmix, mlx-whisper
  took 2 to 3 s and Sortformer 1 to 3 s, and every one of the 11 turns got the right cluster.
  After decision 17 (2026-09-25, same machine): the stereo call settled in 72 s with ASR (Parakeet
  TDT 0.6B v3) at 3.9 s; on the mono downmix Parakeet took 1.4 s, Nemotron-3-Diarization 0.3 s, and
  every one of the 12 turns got the right cluster.
  `test_real_embedding.py` (CPU, needs only `data/models/nemotron-3-embed-1b`) checks that the
  encoder is bidirectional, reproduces the model card's ranking, ranks the right fake-script turn
  first for four contact-center queries, and runs one call's `embeddings` job and a Store query on
  the real model.

## Deferred and interpretations (raised, not patched)

- **Stage 2 appliance only.** Store refuses other route classes and admin state is 501, so the
  contract defaults apply: masking follows `MaskingSettings()` and there are no outbound pools.
  Pro1 stays closed (`is_call1_operated`); Process never offers the `call1_confidential` route.
- **`ProviderType` has no in-process non-MLX value.** The torch tone, sentiment and embedding
  entries are recorded as `mlx` with destination `in-process`; the Process entry's
  `runtime` says what really runs. A contract minor could add `in_process`/`code`.
- **`embeddings` freezes a selection.** Its type rule has purpose `embeddings`; since 1.2.0 the
  selection is the `nemotron-3-embed-1b` entry at the pinned revision. A fake-mode Process still
  freezes that entry but writes `fake-embedding-v1` vectors; the artifact's `scheme` is what search
  matches on.
- **Summary segments are follow-ons of the ASR completion**, because their windows need the
  transcript. Reanalysis summary graphs plan them up front from the linked transcript.
- **Contact Signals v2 on this branch.**
  - *Fake scripts by source are keyed by checksum.* CustomFlags section 4.9 mapped a recording's file
    name; the file name is not in the ASR job's parameters or the source artifact's labels, and adding
    it would change the ingest graph a re-upload replays. `FAKE_SCRIPTS_BY_SOURCE` and
    `CALL1_FAKE_SCRIPTS` key on the source audio's checksum instead.
  - *Stage 2 and 3 jobs also take the attribution input.* It is not in section 8.1's list for them;
    it keeps every v2 job's masked text identical to stage 1's.
  - *The merge takes no `previous` input.* Nothing in the merge reads the previous result: reruns
    are decided when the graph is planned, from the previous stage artifacts.
  - *Engine thresholds.* The per-engine thresholds live in `signal_stages.ENGINE_DEFAULTS`
    (fake: stage 1 0.3, stage 2 0.5, reject 0.5; Gemma: 0.5 each, between its pick scores). A later
    "system one" engine registers its fitted values there.
- **Contact-signal passes are not split into windows yet.** Each kind runs one pass over the whole
  transcript. Windows need the transcript, and a merge's `after` edges cannot be added through
  `NewDependency`, which creates success edges only. A pass that overflows its context rejects or
  fails, and the merge publishes `partial`.
- **`summary.rubric_highlights` is empty**, which keeps the summary independent of QA.
- **No Store route lists conversations for Process.** The console lists what this installation
  ingested or reanalysed (`conversations.jsonl`); everything shown about them is read from Store.
- **Claims are not retried after a lost response.** The leased jobs return when the lease expires.
- **Real handlers: where they differ from the pre-split pipeline.**
  - *Speaker attribution labels whole turns.* Turn IDs are fixed at ASR, so a mono turn that holds
    two voices gets no cluster (the abstain rule), where `attach_clusters` split it at the word.
  - *Contact-signal prompts are masked too.* Appliance text masking (above) covers the signal
    passes as well as QA and summaries; the legacy pipeline sent signal extraction the raw
    transcript. Quotes are checked against the masked text.
  - *Signal context overflow is a failure.* A contact-signal pass over the MLX 8,192-token budget
    fails with `context_limit_exceeded` and the merge publishes `partial`. (A QA prompt over the
    budget is a FLAGGED assessment, as before the split; see "LLM calls".)
  - *Contact signals run without MLX.* On a non-MLX host they use the loopback Ollama; the legacy
    pipeline skipped extraction and marked it partial.
  - *Summaries use constrained decoding.* They pass `SUMMARY_SCHEMA` to the model, as the
    registry path did; the legacy default path (`summary.model_id` unset) sent no schema on MLX.
  - *Validation does not gate the other stages.* `validation_vad` runs beside ASR, not before it.
    ASR applies the same gates itself, but tone and the other stages of a rejected recording are
    dead-blocked rather than never created.
  - *The summary assembly reads the masking flag from its parameters.* The planner records the
    summary route's `masked` there (the assembly has no selection). A graph planned before that
    field existed falls back to the old rule: masked only off the appliance route.
  - *Attempts for gated QA.* A policy- or speaker-gated QA assessment still carries a model
    attempt record, with no tokens and zero latency, because `QaAssessmentContent.attempt` is
    required.
