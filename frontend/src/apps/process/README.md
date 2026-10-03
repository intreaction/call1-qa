# Call1 Process — operator console

The Process operator console: an installation's Store connection, worker and job graphs, in the
browser, on loopback only. It talks to `/process/api/*` (documented in
[`call1/process/README.md`](../../../../call1/process/README.md) under "Operator API") and never
to Store directly and never to `/api/v1` — Process's own boundary rule, the same discipline
Evaluate keeps toward `/store/v1` (docs/SplitBuild.md "Architecture rules").

## Run it

```bash
npm --prefix frontend run build:process   # -> call1/process/static/
python -m call1.process serve             # http://127.0.0.1:8020/
# or, Store and Process together:
python -m call1.launch
```

`build:process` builds flat into `call1/process/static/` (`process.html` at the root, plus
`assets/`), and `copy:process-static` (run automatically) copies the shared fonts and icons in
first. The outDir is shared with those copies, so Vite does not empty it; instead
`scripts/clean-app-assets.mjs` removes the previous `process-*.js/.css` bundles before each build. `call1/process/app.py` serves `process.html` (falling back to a "Not built yet" page when
it hasn't been built) and the Process API side by side.

### Dev server

```bash
npm --prefix frontend run build:process
python -m call1.process serve
```

The general `dev` command opens the legacy app; `dev:evaluate` opens Evaluate. Neither starts
a Process development server.

There's no `dev:process` script yet (only `build:process`); `vite.process.config.ts`'s dev
server proxies `/process` to `127.0.0.1:8020` for when one is added. In the meantime, run
`python -m call1.process serve` and reload after each `build:process`.

## What was built

- **`api.ts`** — a hand-typed client for `/process/api/*` (there is no generated contract for this
  surface; it is Process's own loopback API, not part of the frozen Store contract). Every shape
  is typed from `call1/process/app.py`, `runtime.py`, `worker.py` and `catalog.py`. `ProcessApiError`
  carries the safe `{code, message, details}` envelope; `ProcessUnreachableError` is a fetch that
  never got an HTTP answer (Process isn't running).
- **`useConsoleToken.ts`** — the console credential (`call1/process/console.py`) arrives once as a
  URL fragment (`#console_token=...`), which this reads on load *and* on `hashchange` (a
  freshly-printed link opened in a tab that's already showing the app is a same-document
  navigation, not a reload), stores in `sessionStorage` (never `localStorage` — it's a secret),
  and clears from the URL immediately. A tab that never saw the fragment falls back to a manual
  paste field (`components/ConsoleTokenNotice.tsx`).
- **App shell** (`App.tsx`, `components/Header.tsx`) — "Call1 Process" header, tab nav (Overview,
  Import, Pipeline, Models, Settings), theme toggle, and a connection pill reading
  `GET /process/api/session` (Console connected / Read only / No token issued / Can't reach
  Process). Each tab renders inside `components/ViewBoundary.tsx`, an error boundary keyed by tab,
  so a render error in one tab shows a notice with **Try again** instead of blanking the console.
- **Overview** (`views/OverviewView.tsx`) — Store connection (URL, dev mode, contract version and
  compatibility, lease/heartbeat parameters), worker state and stats, an honest **"Worker
  unavailable"** empty state (with the specific reason: not configured, connecting, Store
  unreachable/refused, contract mismatch) rather than implying a worker that hasn't started,
  resource slots per pool, handlers (mode, missing job types), catalog/reanalysis, and an "Open
  Evaluate" link.
- **Import** (`views/ImportView.tsx`) — drag-and-drop or file-picker upload to
  `POST /process/api/recordings` with a progress bar (XHR upload events — `fetch` has no portable
  one), optional agent ID / channel / external ref. Each upload becomes a card: receipt (created vs.
  reused, jobs planned) → transcription state read off the `transcript` result group → "Open in
  Evaluate" → overall analysis progress, polling `GET /process/api/conversations/{id}` until
  `progress.settled`.
  The form carries agent ID, **agent name** and **agent extension** (contract 1.1.0; blanks are
  not sent, so a re-upload never clears stored values), agent channel and **Call reference**
  (`external_call_ref`: the recorder's call ID or a short title; Evaluate's call search matches it,
  but Evaluate does not display it and the Pipeline row stays labelled by filename). The
  receipt shows the call's agent label ("Name (ext)") and, when the same recording came again
  with new details, a **Metadata updated** pill naming the changed fields ("nothing was
  reprocessed"). The file input is visually hidden but stays in the Tab order (`sr-only peer`);
  its "Choose file" label shows the focus ring, and Space or Enter opens the picker.
- **Pipeline** (`views/PipelineView.tsx`) — one row per conversation from
  `GET /process/api/conversations`, using the server's own `progress_line`
  ("Transcript ready · QA 7/10 · Summary 2/4 · Signals waiting"); expands to fetch
  `GET /process/api/conversations/{id}` and list every child job in pipeline order (a
  topological sort on `requires_job_ids`/`after_job_ids`, ties broken by the stage order in
  `labels.ts`): plain-language step name (`labels.ts` maps every `JobType`; "Asr" is
  "Transcription"), status ("Stopped" for a `BLOCKED` job whose upstream failed, `dead_blocked`),
  attempts ("Not run yet" at 0), how long it ran (`started_at`/`ended_at` of the latest counted
  attempt, which Process caches per finished job), model and "On this appliance" for the
  appliance route, and the waiting reason in words. Retry (`FAILED`/`CANCELLED`)
  and Cancel (`QUEUED`/`BLOCKED`/`RUNNING`, with a "also cancel dependents" checkbox for `cascade`)
  use `components/ReasonAction.tsx` — an inline confirm that requires a reason and surfaces
  Process's own explanation on failure (a `jobs:control`-less key gets the server's plain-English
  fix-it message, not a raw 403). Its confirm and dismiss buttons always have distinct names:
  "Retry job" / "Close" and "Cancel job" / "Keep job" (there used to be two buttons called
  "Cancel"). Escape closes the form and returns focus to the button that opened it. Each row's
  "Evaluate" link sits beside its expand button (not nested in it), so it is its own Tab stop.
- **Overview worker counters** are one grid cell per label/value pair, so each value sits on its
  label's row at every width (known issue d: bare `dt`/`dd` grid cells drifted apart at 640px and
  up). A **Waiting for Store** list shows finished jobs whose result is in the spool
  (`awaiting_delivery`: outputs not uploaded yet, or a completion or failure not delivered yet).
- **Models** (`views/ModelsView.tsx`) — an interactive pipeline graph from recording through
  transcription, speaker diarization, transcript protection, analysis, and Evaluate. Select a
  stage to inspect its catalog entries, status, runtime, license, and installed adapter scope.
  **Fine-tuning settings** opens Settings. In demo mode, applicable text stages also show the
  simulated Call1 + private stack, labelled separately from the actual installed model.
- **Settings** (`views/SettingsView.tsx`) — model choice and on-device training. Outside demo mode,
  **Processing model** (`components/ModelPicker.tsx`) selects the Gemma base or any kept installed
  or trained adapter through `POST /process/api/training/active`; requires the console token,
  blocks during training, and persists across reloads. Existing results require reanalysis.
  With `CALL1_STORE_DEMO=1`, `FineTuneExperience.tsx` presents a labelled simulation:
  Gemma → Call1 industry LoRA → private LoRA, with both adapters contributing. Choose a Call1
  edition first, then train the private layer. Pausing private keeps Call1 active; changing the
  edition clears private lineage. Downloads and training are mocked and persist in browser
  localStorage only. No weights, calls, training runs, or active runtime models are changed.
  Real MLX currently loads one adapter; composing both at inference and training private weights
  against that composition require runtime integration. Subscription rubrics and peer comparison
  delivery are described but are not implemented by this simulation.
  Existing live model selection, schedule, history, and rollback controls remain under
  **Live model and training tools** in demo mode, or **Training schedule and history** otherwise.
  Live training (decision 28, docs/OnDeviceTraining.md §6) uses `GET/PUT/POST /process/api/training*`.
  The schedule
  form (the "Scheduled training" switch, daily or weekly, the weekday — indexes 0 = Monday … 6 =
  Sunday, as the backend's `date.weekday()` — the time in the host's zone, minimum new labels,
  maximum duration, only when idle; the weekday is always sent, and a daily schedule ignores it),
  "Next run: Mon 02:00 (zone)" read from `next_run_at`'s own wall-clock time, and the last check.
  **Status**: the phase and detail, a progress bar once `iterations` is known, the claims-paused
  line, **Train now** (a confirm that says processing pauses for up to the maximum duration and
  warns when there are fewer new labels than the scheduled minimum; disabled, with the reason as its
  title, when no label has ever been logged; errors such as 409 `training_busy` shown inline) and
  **Cancel run** while a
  run is in any active phase (`waiting`, `pausing`, `collecting`, `building`, `training`,
  `evaluating`, `deciding`; polling every 5 s then, else 30 s). **Active model**: the version, its
  tasks and held-out accuracy (`eval.overall.candidate` of the evaluation that promoted it), then
  its manifest's details: "Trained on this appliance" with per-task scores and the decision, or,
  for `provenance.kind == "installed"` (`scripts/install_demo_adapter.py`), "Installed" with the
  offline evaluation (`eval.metric`, `unit`, `extra`, `source`, `note`), training data and recipe,
  **Roll back to ft-…** (the `previous` version when still kept, else the newest other kept
  version) and **Use base model**, then every other kept version with **Activate**; these are
  disabled while a run is active. **Run history** rows expand to labels, examples, skips, trainer
  numbers and per-task accuracy. The backend's `notices` (`training_unavailable`,
  `insufficient_scope` with the `issue-service-key` command, `qa_too_long`, `base_changed`) render
  as notices; `labels.error` is an object `{code, message}` and is never rendered raw. The
  confirms use `components/ConfirmAction.tsx` (distinct confirm and dismiss names, no reason field).
- **Overview claims banner** — while on-device training has claims paused
  (`overview.training.claims_paused`), the Overview shows which run, since when and until when at
  most.
- **`components/ui.tsx`** — the same building blocks as Evaluate and the Store console (Button,
  StatusPill with a text label as well as color, Card, PageHeader, EmptyState, Notice, form
  fields, `ProgressBar`, time/byte formatters), so the three UIs read as one system.

## Deferred / open issues

1. No `dev:process` Vite script exists yet (see "Dev server" above) — the other two apps have
   `dev:evaluate` / implicitly `build:store-console`'s dev proxy; Process only got `build:process`
   from the Store-phase agent. Added here only as a note; out of scope for this task to add a new
   script without seeing whether the team wants one.
2. Retry/Cancel were verified against the live operator API (`POST recordings`, job listing,
   catalog, worker/session state) with `CALL1_PROCESS_HANDLERS=fake` against a real Store — the
   fake worker settles jobs too quickly to also catch one `RUNNING` to click Cancel on live in a
   browser; the button logic (status gating, `cascade`, the reason requirement, and the server's
   own error message) is exercised by `tests/process/test_process_api.py`'s retry/cancel/scope
   tests, which pass.
3. `GET /process/api/conversations` has no pagination past `limit` (matches the API: it takes
   `limit` only). "Load more" just raises the limit and refetches; fine at operator-console scale.

## Browser tests

`frontend/e2e/process-console.spec.ts` (dark and light): p1 Import (receipt, reuse, unsupported
file), p1/q16 agent name and extension plus the "Metadata updated" re-upload, c4 worker counters
aligned at 360 to 1600 px with no horizontal scroll, p2 Pipeline, p6 retry and cancel (and the
read-only state), and **c8 keyboard-only operation**: Tab order through the header, a visible focus
ring on every stop, Enter/Space on tabs, the theme toggle, the file picker, every Import field
(the channel select by type-ahead), Upload, the receipt's links, Pipeline rows, Refresh, and the
retry and cancel forms (Escape, distinct button names, the cascade checkbox by Space).

`frontend/e2e/process-training.spec.ts` (dark and light): t1–t7 of docs/OnDeviceTraining.md
§7.3 plus a second-promotion rollback and a label-count error, each on a private stack with fake
handlers, the fake trainer and generator (`processEnv`: `CALL1_FAKE_TRAINING_OUTCOMES`,
`CALL1_FAKE_TRAINER_SECONDS`) and the §1.7 minimums lowered. Runs that train start with
`seedTrainingLabels`, which ingests calls and overrides their QA verdicts as a signed-in reviewer
through Store's review API (`tests/e2e/training_seed.py`) until both training splits have a call.

```bash
cd frontend && npx playwright test e2e/process-console.spec.ts --output /private/tmp/call1-e2e/<unique-dir>
cd frontend && npx playwright test e2e/process-training.spec.ts --output /private/tmp/call1-e2e/<unique-dir>
```

## Verification

```bash
npm --prefix frontend run typecheck        # clean
npm --prefix frontend run build:process    # -> call1/process/static/
.venv-local/bin/python -m pytest tests -q -p no:warnings   # 1705 passed, 13 skipped, 17 subtests passed (2026-09-27)
```

Also run live: `python -m call1.launch` on spare ports with fake handlers, a real Store. Console
credential handshake (fragment → sessionStorage → manual-paste fallback), Overview (Store
connection, worker, slots, handlers, catalog), Import (`POST /process/api/recordings` via `curl`,
matching the client's field names and response shape), Pipeline (row summary line, expand, per-job
status/attempts/route/timings), Models (interactive pipeline graph and stage inspection), and both themes at
desktop and mobile widths.


## Live processing demo

In demo mode, Import offers an attributed 15.5-second AppTek retail excerpt (shown rounded to
16 seconds). **Process demo call** calls `POST /process/api/demo/recordings`, then opens the new
call in Pipeline. No flowchart appears in Import. The endpoint needs the console credential and
`CALL1_STORE_DEMO=1`; each take adds a standard WAV INFO comment so its content identity is new,
while every PCM sample stays unchanged. It uses the regular ingestor and configured handlers.

Expanded Pipeline calls use `LiveProcessingGraph.tsx`, a bounded vertical stage list with a slim progress rail.
It shows conceptual stages with real job counts, failure states, running indicators, elapsed
time from job attempts, and collapsible steps revealing matching jobs inline. **Show all details** at the top expands every
step; **Hide all details** collapses them. Open jobs show recorded attempt events, model/adapter
provenance, error details, and input/output artifact references through the existing job API,
with live polling while work is active. Raw model console output is not captured per job.
Retry/cancel controls remain within their jobs in the bounded scroll region. Reduced-motion preferences
stop animations. The list remains legible at phone widths.

The source excerpt and CC-BY-SA-4.0 attribution are in `call1/process/static/demo/`. On this Mac,
all 18 real jobs completed in 58.9 seconds after restarting Process (first model loads included).
A longer 24-second rehearsal measured 47.6 seconds with loaded models, but about 66 seconds on
its first run. Timing depends on host load and model configuration; these are measured rehearsals,
not a guaranteed SLA. `e2e/process-demo-call.spec.ts` uses fake handlers to verify navigation,
fresh takes, terminal states, stage inspection, responsive bounds, and credential gating.
