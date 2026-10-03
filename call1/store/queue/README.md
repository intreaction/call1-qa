# Store queue area

The queue area of Call1 Store (`call1/store/queue/`, migrations `020`–`029`). It owns conversations,
artifacts and uploads, the processing-job queue, reanalysis requests and draft tests, usage rows and
their report, hardware profiles and catalog snapshots. It serves 44 contract operations; the table in
`call1/store/routing.py` (`_QUEUE`) lists them. This is the processing queue. The human review queue
belongs to the results area, and the two share no table, type or module.

## Modules

| Module | What it does |
|---|---|
| `routes.py` | One thin handler per contract operation, registered on `router` |
| `records.py` | Row-to-model conversion, and `ConversationJobs`, the in-memory dependency view of one conversation (dead-blocking, blocking reasons, publisher state, failure codes, settled) |
| `artifacts.py` | Inline JSON artifacts, upload grants and verified commits, Store-minted rubric snapshots (published and draft) and signal taxonomy snapshots (published `signals:v<N>`, reused while the settings are unchanged; preview `draft:<rid>:signals:preview`), linking and per-slot versions, listings, content grants, orphan cleanup |
| `graphs.py` | Graph validation (`prepare_jobs`, with `_check_rubric_input` and, since 1.3.0, `_check_signals_input`), insertion, initial status, dependent release (`try_release`, `release_dependents`), input resolution and pinning |
| `lifecycle.py` | Graph creation, admission, atomic claims, heartbeat, completion, failure, release, lease expiry, manual retry, cancel, and the maintenance sweep |
| `reanalysis.py` | Reanalysis requests (create, claim by priority then age with the `kinds` filter, expire, reject, fulfil by graph), draft tests and draft results, `contact_signals` widening, the signal taxonomy version and pipeline resolved at creation, preview results |
| `signals.py` | Contact Signals v2 (1.3.0): taxonomy previews, compare previews, rescore and compare backfills, shadow-mode compare companions, `getSignalPreview` (masked results and diffs through `results/api.py`) |
| `usage.py` | Usage rows (written, synthesized `abandoned`, late), the report, the CSV export, medians, hardware profiles |
| `catalog.py` | Catalog snapshots, one per Process installation |
| `progress.py` | Group stakes for `derive_result_state`, pending work, group progress, failure codes |
| `changes.py` | `ChangeLog`: collects job, request, group and catalog events and appends them once per transaction |
| `reads.py` | Registration and the conversation, graph, job and attempt reads |
| `api.py` | What other areas call (below) |

Tables are prefixed `q_` (`020_queue.sql`; `021_signal_requests.sql` adds `q_signal_previews`, `q_signal_backfills` and the 1.3.0 request columns). Other areas never read them; they call `api.py`.

## The job lifecycle

`jobs.JOB_TRANSITIONS` is the state machine. Every write runs in one `BEGIN IMMEDIATE`
transaction, and SQLite serializes writers:

- **Graphs** are idempotent per `(installation, conversation, idempotency_key)`. A job key that
  already names a job of the conversation returns that job when the definition is identical, and
  gets 409 `idempotency_key_reused` when it differs. Store checks upstream job IDs (the same
  conversation only, `graph_invalid`), output roles, pinned checksums (`checksum_mismatch`) and QA
  rubric snapshots (`graph_invalid`). It also checks routes: Stage 2 permits the appliance route
  only (`route_not_permitted`). Jobs start BLOCKED and become QUEUED when their edges are satisfied.
- **Claims** take the write lock, sweep expired leases, apply admission, then walk the ready jobs
  call by call (`lifecycle.CLAIM_CANDIDATES_SQL`): the worker's `conversation_id` affinity hint
  first, then the band (the priority of the reanalysis request the graph fulfils: +5 previews, 0
  ingest and ordinary reanalysis, -10 backfills and compares), then the graph that entered the
  queue first (`created_at`, then insertion order), and only then the job's own `priority desc,
  created_at asc, id asc`. The job's own priority is Process's stage priority plus the request
  offset, so ordering by it alone advanced a batch stage by stage and every call finished at the
  end; now the oldest call's ready jobs go first and a call finishes before the next one starts.
  The order is a preference, never a block: a job the worker cannot take is skipped, so a newer
  call's (or a lower band's) ready jobs still fill slots the older call cannot use. A lower band
  waits only while higher-band work fills every offered slot (backfills are background work). A
  retried job keeps its graph's place. A job is claimed
  when the worker covers its type, execution class (primary host), route class and frozen catalog
  entry, and when an offer with the job's memory slot and outbound connection has room left. The
  `UPDATE ... WHERE status = 'QUEUED'` is also guarded. Two workers never share a claim; the tests
  race 2 HTTP clients and 8 connections.
- **Completion** runs `jobs.COMPLETION_TRANSACTION_STEPS` in one transaction. It looks up the
  completion key before it checks the token, so a replay returns the stored receipt with
  `replayed: true`. It then checks the claim and `job_cancelling`, and verifies the outputs (one per
  declared role, produced under this claim, draft slots), the result, the usage outcome and the
  frozen route. It links the outputs, writes the usage row and marks the job SUCCEEDED. It calls
  `results.projections.apply_completion` next, then inserts follow-on jobs with their dependencies
  and input bindings, releases the dependents and appends the change events. The receipt carries the
  last event's cursor. If the projection raises, nothing commits.
- **Failure** applies `JOB_ERROR_CLASSES`. A transient code with attempts left goes back to QUEUED
  with backoff `min(30 s × 2^(n−1), 3600 s)`; anything else ends FAILED, and a failure after cancel
  ends CANCELLED. After-edge dependents are released in the same transaction. **Release** refunds
  the attempt: `requeue` returns the job to QUEUED (honouring `not_before`) and `reject` fails it.
  A released claim has no usage row.
- **Lease expiry** happens past `expires_at + LEASE_GRACE`. Every claim sweeps; heartbeat,
  complete, fail and release first expire their own job in a separate transaction, so the expiry
  sticks even when the request is then refused as stale. Expiry synthesizes the attempt's
  `abandoned` usage row, which the late worker may fill in once (`attachLateUsage`).
- **Manual retry** (FAILED only) adds `RETRY_ATTEMPT_GRANT` and bumps `retry_generation`; the job
  is QUEUED when its upstreams are satisfied, BLOCKED otherwise. **Cancel** cancels BLOCKED and
  QUEUED jobs at once and marks RUNNING jobs `cancel_requested`. With `cascade`, every transitive
  dependent is cancelled; without it, after-edge dependents are released.
- **Dead-blocked** means BLOCKED behind a FAILED or CANCELLED `requires` upstream, or behind any
  dead-blocked upstream. It is computed on read and never stored, so retrying the upstream clears it.

The results hooks are called inside these transactions: `on_conversation_registered` on
registration, `on_call_metadata_updated` when a re-registration changes call metadata (contract
1.1.0: `reads.register_conversation` merges it with `calls.merge_call_metadata`, saves it and writes
the `call_metadata_updated` audit event in the same transaction), `apply_completion` at step 7, and `on_job_failed` for every failure, lease expiry,
reject release, admission rejection and cancelled job, including each job a cascade cancels.

## What the other areas call (`api.py`)

The agreed eight are `get_conversation`, `get_conversation_by_call`, `get_artifact`,
`list_linked_artifacts`, `group_stakes`, `reanalysis_pending_groups`, `pending_work` and
`create_reanalysis_request`. The area also offers these additions: `failure_codes` (the newest live
publisher's code, or the first ended upstream's), `reanalysis_request_for_group`, `graph_created_at`
and `conversation_settled`. Draft-test graphs never count toward groups, pending work or settled.
Contract 1.3.0 adds `get_job` (any job as `getJob` returns it, or None): the results area's
on-device training label reads resolve each label's source job and its pinned inputs and outputs
through it (docs/OnDeviceTraining.md section 2.3).

Dual transcription (1.3.0, decision 33, docs/DualAsr.md section 4): a completion links every role
of `JobTypeRule.outputs` once plus any role of `optional_outputs` at most once (the `asr` job's
`base_transcript` and `vocabulary_pass`; required roles are linked first), and artifact creation
accepts the optional kinds as declared outputs. Graph creation refuses an `asr` job whose
`parameters.asr_vocabulary` holds more than `max_vocabulary_terms + max_vocabulary_pack_terms` terms
(`graph_invalid`, `asr_vocabulary_cap`); the frozen digest need not be the current vocabulary's.

`listJobs` also filters on `memory_slot` (1.3.0, `q_jobs.memory_slot`). Process's training start
check asks for `local_memory` jobs that are QUEUED or RUNNING with `limit=1`.

## Maintenance

`lifecycle.sweep(store)` expires leases, applies admission, returns expired reanalysis claims to
`pending`, deletes orphan artifacts, and forgets expired upload sessions. An orphan is a job output
that no completion linked. It is deleted after `ORPHAN_ARTIFACT_RETENTION`, once its attempt has
ended, and its bytes go too when no other row shares the checksum. Claims run the sweep at most every
10 minutes (`maybe_sweep`).

## Stage 2 interpretations (raised, not patched)

- Admin state is deferred, so the contract defaults apply: only the `appliance` route class is
  enabled. Other routes get `route_not_permitted` at graph creation, and admission would fail
  anything else as `route_disabled`. No Pro1 job can exist, so `FailureReceipt.pro1_connection_blocked`
  is always false.
- `failJob` is marked audited in the route table. The contract ties that audit to the Pro1
  connection block, which cannot happen in Stage 2, so failures write no audit event (the contract
  has no `job_failed` action).
- A cycle inside one graph request is refused by the contract model itself: FastAPI answers 422
  `validation_failed` before Store sees the graph. Store's own checks (cross-conversation edges,
  output roles, follow-on cycles) answer 422 `graph_invalid`.
- `requestReanalysis` refuses `speaker_correction` (use `POST /calls/{call_id}/speaker-corrections`,
  which checks the review version). It also answers 409 `conflict` while a request of the same kind
  is already pending or claimed for the call.
- The usage price table is deferred: `include_estimates` returns a list of nulls aligned with the
  rows.
- Header idempotency is scoped to the session and call. Requests the results area creates through
  `create_reanalysis_request` are scoped to the account and call.

## Tests

`tests/store/test_queue_*.py` (`test_queue_harness.py` holds the shared builders and fixtures):

```bash
.venv-local/bin/python -m pytest tests/store/test_queue_*.py -q -p no:warnings
```
