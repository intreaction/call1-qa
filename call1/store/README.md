# Call1 Store

Store is the only owner of call, queue and review data. It serves the frozen contract
([`call1/contracts/`](../contracts/README.md), v1.4.0) at `/store/v1`, the **Evaluate** browser app
at `/` and the Store console at `/console/`. Process and Evaluate reach it only over HTTP
([docs/SplitBuild.md](../../docs/Architecture.md)). Store runs exactly one model, the local search
embedder that embeds semantic-search queries (decision 18; see "Contract 1.2.0 in Store"), and no
other model runtime.

## Run it

```bash
python -m call1.store serve                 # http://localhost:8010 (dev mode), data in data/store/
python -m call1.store migrate               # apply pending migrations and exit
python -m call1.store setup-code --email admin@example.com --display-name "Admin"
python -m call1.store issue-service-key --installation mac-mini   # writes data/process/config.json (0600)
python -m call1.embedding download          # the search embedder's weights, into data/models/nemotron-3-embed-1b
python -m call1.store request-reembedding   # re-embed calls indexed under an older search scheme
python -m call1.store apply-signals-seed call1/store/seeds/signals_retail_v1.json --pipeline v2   # demo seed (1.3.0)
python -m call1.store apply-vocabulary-seed call1/store/seeds/asr_vocabulary_retail_v1.json   # demo ASR vocabulary pack (1.3.0)
python -m call1.store signals-pipeline v2      # set the Contact Signals pipeline (v1, shadow or v2)
python -m call1.store project-signals          # project contact signals published before 1.3.0
```

| Variable | Default | Meaning |
|---|---|---|
| `CALL1_STORE_DATA` | `data/store` | `store.db` (SQLite, WAL) and `objects/` (content-addressed by SHA-256) |
| `CALL1_STORE_HOSTNAME` | `localhost` | The fixed Store hostname and WebAuthn RP ID; `localhost` means dev mode |
| `CALL1_STORE_PORT` / `CALL1_STORE_BIND` | `8010` / `127.0.0.1` | Where `serve` listens |
| `CALL1_STORE_DEV` | on for `localhost` | Dev mode (below) |
| `CALL1_STORE_DEV_ORIGINS` | none | Extra `http://localhost:<port>` origins in dev mode (the Vite dev server) |
| `CALL1_STORE_PUBLIC_URL` | derived | Client-facing URL when it differs from `https://<hostname>:<port>` |
| `CALL1_STORE_TLS_CERT` / `CALL1_STORE_TLS_KEY` | none | Required by `serve` outside dev mode |
| `CALL1_STORE_DEMO` | off | **Demo mode** (below): persona sign-in without a passkey. Dev mode only; Store refuses to start with it on otherwise |
| `CALL1_STORE_PARAMETERS` | contract defaults | JSON overrides of `ContractParameters` (reported at `/store/v1/contract`) |
| `CALL1_STORE_MAINTENANCE_SECONDS` | `60` | How often the running app runs `maintenance.run_once`: expire leases, reanalysis claims and upload sessions, apply admission, drop old orphans, prune the change feed past `change_feed_retention_seconds`. `0` turns it off (`StoreConfig.for_tests` does) |
| `CALL1_EMBEDDING_BACKEND` | `fake` when `CALL1_PROCESS_HANDLERS=fake`, else `nemotron` | The search embedder (contract 1.2.0); must match Process's |
| `CALL1_EMBEDDING_PATH` | `$CALL1_MODELS_DIR/nemotron-3-embed-1b` (`data/models/...`) | The embedder's weights |
| `CALL1_EMBEDDING_DTYPE` / `CALL1_EMBEDDING_DEVICE` | `bfloat16` / `cpu` | About 2.3 GB resident; `float32` is faster but about 4.8 GB. Keep equal to Process's |

`issue-service-key` merges these keys into Process's config (`--config`, else
`CALL1_PROCESS_CONFIG`, else `data/process/config.json`, mode 0600): `store_url`,
`installation_id`, `service_key_id`, `service_key` (the one-time token). Without `--scope` the key
holds only what a Stage 2 Process worker calls (`__main__.PROCESS_DEFAULT_SCOPES`: calls:write,
artifacts:read/write, jobs:write, jobs:claim, reanalysis:claim, changes:read, hardware:write,
catalog:publish, usage:read, admin-state:read, and since 1.3.0 training:read, which pages the
reviewer-label log for on-device training); `jobs:control`, `key-release:write` and
`release-trust:write` need an explicit, repeatable `--scope`. With no
`--primary-host`/`--no-primary-host`, the installation is the primary host when no active one is.
`setup-code` refuses a first-admin code while an active admin exists (use `--purpose break_glass`,
optionally `--target-account-id`), and `issue-service-key --primary-host` refuses while another
active installation is primary; both exit 2 with the reason. Auth details: [auth/README.md](auth/README.md).

## Dev mode and the session cookie (the documented deviation)

Stage 2 ships passkeys on `localhost`, a WebAuthn secure context over plain HTTP. The contract has
no dev fallback, and three of its rules cannot hold there, so dev mode deviates in exactly these
places and nowhere else:

1. **Cookie.** `__Host-` cookies require `Secure`, which a plain-HTTP origin cannot set reliably
   in every browser (or in TestClient). Dev mode names the cookie **`call1_session`** without
   `Secure`; it keeps `HttpOnly`, `SameSite=Strict`, `Path=/`, no `Domain`, the SHA-256-only storage
   and the `X-Call1-CSRF` header. Outside dev mode the cookie is exactly `auth.SESSION_COOKIE`
   (`__Host-call1_session`, Secure). Use `principals.set_session_cookie` / `clear_session_cookie`.
2. **Relying party.** `rp_id` is `localhost` and the allowed origin `http://localhost:8010` (plus
   `CALL1_STORE_DEV_ORIGINS`), which `validate_store_hostname` and `WebAuthnRelyingParty` reject.
3. **Grant URLs.** `UploadGrant.url` and `ContentGrant.url` are `http://localhost:...` although the
   contract pattern is `^https://`.

`devmode.respond(Model, payload, config)` is the one place these values cross a contract model:
it validates everything else and emits the dev values. Dev mode binds loopback only
(`StoreConfig` refuses anything else). Outside dev mode the hostname must pass
`validate_store_hostname` and every response validates as the contract.

## Demo mode (class demos, localhost only)

`CALL1_STORE_DEMO=1 python -m call1.store serve` adds two routes **outside** `/store/v1` (they are
not contract routes) so a presenter can sign in without a security key
([auth/demo.py](auth/demo.py)):

| Route | Body | Answer |
|---|---|---|
| `GET /demo/status` | none | `{"demo": true, "label": "...", "personas": [{"persona", "display_name", "role", "email"}]}` |
| `POST /demo/sign-in` | `{"persona": "admin" \| "supervisor" \| "reviewer"}` | the passkey sign-in body `{"session": SessionInfo}` plus `"demo": true, "persona"`, and the session cookie |

- **Off by default.** With `CALL1_STORE_DEMO` unset both routes are 404 `not_found` (they stay
  registered, so the Evaluate page never answers for `/demo/...`). A client detects demo mode by
  `GET /demo/status` returning 200. `/store/v1/status` is unchanged either way.
- **Localhost only.** `StoreConfig` refuses demo mode outside dev mode (`serve` exits 2 with
  `configuration: CALL1_STORE_DEMO works only in dev mode`), and dev mode already binds loopback
  only. The sign-in also requires a loopback `Host` (`localhost`, `127.0.0.1`, `::1`; defeats DNS
  rebinding) and, when a browser sends `Origin`, a Store origin (403 `origin_not_allowed`).
- **A normal session.** Sign-in creates the persona's account on first use (Demo Admin, Demo
  Supervisor, Demo Reviewer at `demo.<persona>@call1-demo.example`), with one placeholder
  authenticator record that holds no key (a session row must name an authenticator), then issues the
  same server-side session, cookie and CSRF token as a passkey sign-in. Every `/store/v1` route then
  enforces role, CSRF and origin as usual. Re-signing in reuses the account, restores its persona
  role, and refuses a disabled persona (403 `account_disabled`).
- **Audited.** Account creation (`account_created`), the placeholder (`authenticator_added`), a
  role/status restore and every sign-in (`account_updated`, `details.event = "demo_sign_in"`, actor
  the persona's session, display "Demo Admin (demo mode)") carry `details.demo_mode = true` and
  `details.demo_persona`. The contract has no sign-in audit action, hence `account_updated` (a sign-in
  sets `last_sign_in_at`).
- **Side effect to know.** Demo Admin is an active admin, so after it exists `setup-code` refuses a
  first-admin code (use `--purpose break_glass`), and it counts for the last-admin rule. Use a
  separate `CALL1_STORE_DATA` directory for demos.
- The passkey ceremonies are untouched and keep working beside demo mode.

## Contract 1.1.0 in Store

- **Agent identity.** `agent_display_name` and `agent_extension` are stored with the call record
  and each review-queue item (migration `041_agent_identity.sql`; rows from before 1.1.0 get NULL,
  which is exact, because no 1.0.x registration could carry them). The call list, call detail,
  review-queue items and escalations always send both, as `null` when unknown. `listCalls` `text`
  matches agent ID, display name, extension and external call reference. The `agent_id` filter
  stays exact on `agent_id`.
- **Re-registration** (`queue/reads.py` `register_conversation`). A known source identity whose
  `call_metadata` changes the stored metadata (`calls.merge_call_metadata`; with nothing stored,
  the comparison is against `CallMetadata()` defaults, so re-sending the defaults is a replay)
  updates it in one transaction: `q_conversations.call_metadata_json`, the results projections
  (`results/projections.py` `on_call_metadata_updated`), a `call` change event with status
  `metadata_updated`, and a `call_metadata_updated` audit event (actor: the Process installation;
  target `call`, or `conversation` when it has no call; details `conversation_id` and the
  comma-separated `updated_fields`, never values). The response has `metadata_updated: true` and
  `updated_fields`. Nothing is reprocessed. A conversation with no call gets the audit event but no
  change event, because there is no call to announce. Unchanged metadata is a pure replay.
- **Hours audited.** `total_hours_audited` keeps 6 decimals (0.0036 s), so a short call is never
  rounded to 0.01 h or 0.0 h before it reaches a client (`results/metrics.py`
  `HOURS_AUDITED_DECIMALS`).

## Contract 1.2.0 in Store

- **The search embedder.** `semanticSearch` embeds the query with `call1/embedding.py`
  (`nvidia/Nemotron-3-Embed-1B-BF16`, the model Process embeds turns with) and ranks only vectors of
  its scheme; see [results/README.md](results/README.md) "Semantic search". Install the weights with
  `python -m call1.embedding download` (into `data/models/nemotron-3-embed-1b`, or
  `CALL1_EMBEDDING_PATH`). Store starts without them; search then answers 503 `search_unavailable`
  and `serve` prints that the embedder is not installed.
- **Memory.** About 2.3 GB resident once the first search has loaded the model (bfloat16 on CPU).
  `CALL1_EMBEDDING_DTYPE=float32` is faster but about 4.8 GB; keep Store and Process on one setting.
- **Status.** `StoreStatus.search_embedder` and the console's "Search embedder" panel (admin) show
  the state without loading the model.
- **Re-embedding.** `requestReanalysis` accepts kind `embeddings` (affects no result group), which
  Process fulfils with a single `embeddings` job; Evaluate's Workbench offers it as "Search index
  only (re-embed)". To migrate every call indexed before 1.2.0 at once, run
  `python -m call1.store request-reembedding` on the Store host: one audited, idempotent request per
  call whose vectors are of another scheme.

## Contract 1.3.0 in Store (Contact Signals v2)

Design: [docs/ContactSignalsV2.md](../../docs/Architecture.md) (sections 7, 9 and 14; decisions
21-23). Process and Store upgrade together (every `ReanalysisRequest` now carries `priority`).

- **14 operations**, all Stage 2. The results area owns the taxonomy, settings, redaction, alert
  rules, hit feedback and metrics (`results/signal_store.py`, `results/signals.py`,
  `results/http_signals.py`, migration `042_signals.sql`); the queue area owns taxonomy snapshots,
  previews, compares and backfills (`queue/signals.py`, `queue/artifacts.py`, migration
  `021_signal_requests.sql`). The queue reads the taxonomy only through `results/api.py`.
- **Snapshots.** `mintSignalTaxonomySnapshot` links the published version with the current settings
  in slot `signals:v<N>`, reusing it while it is current; a redacted version is 409. Graph creation
  checks every v2 job's pinned snapshot against `parameters.signals.taxonomy_digest`
  (`graphs._check_signals_input`, `graph_invalid`), and a preview snapshot only runs in its own
  preview graph.
- **Requests.** Claims order by `priority DESC, requested_at, id` and honour `kinds`. Store resolves
  `signal_taxonomy_version` and `signal_pipeline` at creation. A new `contact_signals` request widens a
  pending, unclaimed one instead of 409. `contact_signals_preview` is a draft-test kind: its graph
  writes draft slots only, and its merge output is recorded as `preview_result_artifact_id`.
- **Shadow mode.** In the transaction that publishes a v1 `contact_signals` result from a non-draft
  graph, Store creates one compare companion (`contact_signals_preview`, priority -10, a `compare`
  preview with no account).
- **Backfills.** `rescore` creates `contact_signals` requests (priority -10) only for calls a
  digest-driven update would change (all with `rescore_signals`); under pipeline `v1` or `shadow` a v1
  result never needs one, because v1 ignores the taxonomy. `compare` runs v2 beside v1 results.
- **Seeds.** `call1/store/seeds/signals_retail_v1.json` is applied only by `apply-signals-seed`
  (`--demo`); installs start with the built-ins. Details: [results/README.md](results/README.md).

## Contract 1.3.0 in Store (dual transcription vocabulary)

Design: [docs/DualAsr.md](../../docs/Architecture.md) (decision 33). Store keeps the vocabulary, checks the
`asr` job's frozen copy, links its optional outputs and masks the corrections reviewers see. It runs
no model.

- **2 operations**, results area: `getAsrVocabulary` (admin `manage_vocabulary`, or Process
  `jobs:write`) and `saveAsrVocabulary` (admin) in `results/vocabulary.py` and
  `results/http_vocabulary.py`, on the singleton row of migration `044_asr_vocabulary.sql`
  (`record_version` 0 and empty until the first save or pack install). Effective terms, digest and
  `active` are derived on every read with the contract functions.
- **Save rules.** The contract models refuse a bad term (no digits) or a repeated one; Store adds
  `max_vocabulary_terms`, pack-only `disabled_pack_terms` (`unknown_pack_term`) and the rule-based PII
  detectors (`pii_detected`). Every refusal is `validation_failed` with `details.field`,
  `details.reason` and `details.index`, never the term (`errors.vocabulary_term_details` maps the
  model's refusals the same way). A stale version is 409 `conflict` with `details.current_version`.
  Audit `asr_vocabulary_saved` and change event `asr_vocabulary` carry versions, digests and counts.
- **Pack install.** `apply-vocabulary-seed <pack.json>` (installer actor, idempotent; `--demo` runs it
  at every start with `seeds/asr_vocabulary_retail_v1.json`) checks `max_vocabulary_pack_terms` and
  the detectors, and drops disabled terms the new pack lacks.
- **Queue.** Graph creation caps `parameters.asr_vocabulary` at `max_vocabulary_terms +
  max_vocabulary_pack_terms` (`graph_invalid`, reason `asr_vocabulary_cap`); the digest need not be
  current. Completion accepts each role of `JobTypeRule.optional_outputs` at most once beside the
  required ones (the `asr` job's `base_transcript` and `vocabulary_pass`), and artifact creation
  accepts the optional kinds. Both are raw and never projected.
- **View.** `TranscriptView.vocabulary_correction` (`records.vocabulary_correction_view`): while the
  text is withheld every replacement is withheld; otherwise a replacement over a masked word or span
  is withheld, `heard` passes the call's Masker and is null when that changed it or it holds a
  digit, and the offsets are mapped into the masked text (null unless they land exactly).
  Whisper's `candidate_text` is never sent.

## Contract 1.4.0 in Store (Contact Signals rules engine)

- **Recipes are taxonomy content.** `SignalCategory.recipe` and `SignalTaxonomy.rules` are saved,
  versioned, previewed, activated, snapshotted and redacted with the taxonomy (`signal_store`). The
  save validator applies the two new caps (`max_signal_recipe_rules`, `max_signal_lexicon_phrases`)
  and runs the definition-text detectors over every lexicon phrase (paths like
  `categories[0].recipe.lexicon.phrases[3]`); redaction tombstones them. `taxonomy_digest` leaves the
  new fields out while null, so stored versions and snapshots keep their digests.
- **Detection is a setting.** `SignalSettings.detection` (`model` default, or `rules`) is saved with
  the other settings (audited with `old_detection`/`new_detection`); a change mints a new snapshot for
  the next graph, as a pipeline change does.
- **Host commands.** `python -m call1.store signals-detection model|rules` sets it;
  `python -m call1.store apply-signals-recipes <seed> [--detection rules]` copies a seed's recipes and
  bank pin into the current taxonomy by category ID (keeping every other on-stage edit) as an audited
  admin save. The retail seed carries the R2 recipes; `--demo` still leaves detection on `model`.
- **Not in Store yet.** The example bank is a Process-side pack file pinned by digest, not a
  Store-owned bank (docs/SignalsEmbeddings.md "Build status").

## Known contract gaps (raised, not patched)

- `StoreStatus.tls` is required, but TLS state is Stage 4: `/status/detail` returns `"tls": null`,
  and `StoreHealth.tls_health` is `untrusted` (no certificate Store has verified).
- Admin state (`getAdminState`, `changeAdminState`) is in neither the brief's in-scope list nor its
  deferred list; it is deferred (501) here, `admin_state_version` is 0, and Store applies the
  contract defaults (appliance route only, masking on). Process sees 501 on `GET /admin/state`.
- Release trust is deferred, so `running_build.manifest_digest` is a fingerprint of the running
  Store and contract sources and `approved` is false.
- The contract's WebAuthn and URL validators reject every dev-mode value above.
- `failJob` is marked `audited=True`, but `AuditAction` has no job-failure action, so Store writes
  no audit event for it (the job, attempt and usage rows record the failure). Open question for
  John: a contract commit adds an action (at least for the Pro1-block case), or the route is
  marked not audited.

## Module map and ownership

Core (this skeleton; changes need the orchestrator):

| File | What it gives the areas |
|---|---|
| `config.py` | `StoreConfig` (data dir, hostname, port, dev mode, demo mode, origins, cookie name, parameters) |
| `context.py` | `Store` (config, `db`, `objects`, `clock`, `auth`) on `app.state.store` |
| `deps.py` | `get_store`, `get_conn` (one connection per request), `current_principal`, `guard_for(route)` |
| `db.py` | `transaction(conn)` (BEGIN IMMEDIATE; nests as SAVEPOINT), `read_snapshot(conn)`, `ts`/`parse_ts`, `dumps`/`loads`, migrations, `store_meta` |
| `objects.py` | `store.objects`: `put_bytes`, `read_bytes`, `open`, `file_response` (Range), `create_upload`, `commit_upload(validate=)`, `content_grant` |
| `principals.py` | Principal types, `AuthBackend` protocol, `hash_secret`, token/secret generators, `require_own_installation`, `require_permission`, session-cookie helpers |
| `errors.py` | `StoreError(code, message, details=...)` and the envelope for every non-2xx |
| `feed.py` | `feed.append(conn, kind, resource_id, version, status, conversation_id=, call_id=)` inside the writing transaction; cursor reads |
| `audit.py` | `audit.append(conn, actor=, action=, target_kind=, target_id=, details=)` (hash-chained); `actor_for(principal)` |
| `hooks.py` | Payload types of the queue→results projection hooks |
| `routing.py` | `OWNER_BY_OPERATION`, `DEFERRED`, `AreaRouter` |
| `devmode.py` | `respond`/`validate` for the dev-mode fields above |
| `pagination.py`, `ids.py`, `clock.py` | Page tokens, `new_id(prefix)`, `SystemClock`/`ManualClock` (`conn.now()`) |
| `app.py` | `create_app()`: registers every contract route (handler, pending 501 or deferred 501), transfer routes, static apps |
| `routes/` | Core handlers: `getStatus`, `getStatusDetail`, `getContract`, `listChanges`, `listAuditEvents`; `transfer.py` (grant PUT/GET) |
| `static.py` | Evaluate at `/`, console at `/console/`, placeholders until built |
| `migrations/010_core.sql` | `store_meta`, `change_events`, `audit_events`, `object_uploads` |

Areas (each owner edits only its package and its migration range):

| Area | Package | Migrations | Operations | Must implement for others |
|---|---|---|---|---|
| queue | `queue/` | `020_queue.sql`, `021_signal_requests.sql` (020-029) | 44: conversations, artifacts and uploads, jobs, reanalysis and draft tests, signal snapshots, previews and backfills, usage report, hardware, catalog | `queue/api.py`; calls `results/projections.py` inside its transactions |
| auth | `auth/` | `030_auth.sql` (030-039) | 31: passkeys, sessions, accounts, invitations, setup codes, break-glass, installations, service keys | `auth/backend.py` (`AuthBackend`), `auth/cli.py`, `auth/api.py` |
| results | `results/` | `040_results.sql`, `041_agent_identity.sql`, `042_signals.sql`, `043_training_labels.sql`, `044_asr_vocabulary.sql` (040-049) | 54: calls and projections, audio, search, reviews, review queue and rules, reviewer profiles, rubrics, metrics, signal taxonomy, alert rules and feedback, training labels, the ASR vocabulary | `results/projections.py` (hooks), `results/api.py`; see [results/README.md](results/README.md) |

The full per-operation table is `routing.py`. 33 operations are deferred (501): custody, release
trust and `admin/pro1`, Stage 4 (TLS, backup), Stage 5 (updater, egress), the price table and admin
state.

### Rules for area code

- Register handlers with `@router.operation("<operationId>")` in `<area>/routes.py` (or modules it
  imports). `create_app` adds the method, path, status code, response model and principal guard,
  and refuses to start when path parameters, the body model, the query model
  (`Annotated[Model, Query()]`) or the `Idempotency-Key` header differ from the contract route.
- Object rules are the handler's: `require_own_installation`, the scope that created an upload
  (`UploadRecord.required_scope`), assignees, `require_permission(P.RESOLVE_ANY_REVIEW)`.
- Every write runs in `with db.transaction(conn):`; audited routes call `audit.append` and followed
  changes call `feed.append` in the same transaction.
- Call another area only through its `api.py` (or the projection hooks). Never read another area's
  tables and never declare a FOREIGN KEY into them; reference rows by ID.
- The processing queue (`queue/`) and the human review queue (`results/`) share no table, type or
  module.

## Tests

`tests/store/conftest.py` gives every test a fresh dev-mode Store in `tmp_path` with a
`ManualClock`, a `TestClient` (no ports), and TEST-ONLY credentials: `mint_service_key(scopes)`,
`service_key_headers`, `mint_session(role)`, `reviewer_session`, `supervisor_session`,
`admin_session` (`.read_headers` for GET, `.headers` with CSRF for writes). The auth backend mints
them directly; no HTTP route does.

Most area tests fake the neighbouring areas (the queue tests' `hooks`/`rubrics` fixtures, the
results tests' `FakeQueue`/`FakeAccounts`). `test_store_integration.py` runs the real stack with no
fakes: a rubric draft test end to end, a speaker correction through its reanalysis graph, and
review-queue distribution across real accounts. `test_store_boundaries.py` checks, in a fresh
interpreter, that Store loads no model runtime (torch and transformers load only with the search
embedder, on the first query) and no Process or legacy module.
`test_store_maintenance.py` covers the background sweep and change-feed pruning.

```bash
.venv-local/bin/python -m pytest tests/store -q -p no:warnings
CALL1_STORE_REQUIRE_COMPLETE=1 .venv-local/bin/python -m pytest tests/store -q -p no:warnings   # fail on pending routes
```

## Synthetic demo history

`python -m call1.store seed-demo-history --count 560` adds fictional call history through
Store-owned registration, artifact validation and result projections. It requires
`CALL1_STORE_DEMO=1`, uses `CALL1_STORE_DATA`, accepts targets from 1 to 5,000 and is idempotent:
raising the target adds the missing sessions; lowering it never deletes sessions. The standard
four-criterion rubric must be present. `python -m call1.launch --demo` runs it automatically
after applying the demo policy, including on existing demo roots.

The default spans 56 days and 12 explicitly named demo agents. Every session has a synthetic
external reference, scripted transcript, summary, QA evidence and an explicit synthetic PII
fixture. These imported snapshots have no source audio, worker jobs, real model attempts or
human-review labels. They populate Calls, QA metrics and review queues; they do not populate
Contact Signals or establish model performance. Use the original five audio examples for
playback and processing. Demo Store aggregates include both kinds of sessions.
