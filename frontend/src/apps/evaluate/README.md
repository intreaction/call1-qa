# Call1 Evaluate

The reviewer browser app. Store serves it at `/`; it talks only to `/store/v1` through a typed
client built on `frontend/src/contracts/store-v1.ts` (docs/SplitBuild.md rule 3). It never calls
`/api/v1` (legacy) or Process. The client refuses any path outside `/store/v1`, and the path types
only admit contract routes.

**The one documented exception: `/demo/*`.** Demo mode (`call1/store/auth/demo.py`, off unless
Store was started with `CALL1_STORE_DEMO=1` in dev mode) is not part of the frozen contract, so it
has no generated types. `api/demo.ts` calls `GET /demo/status` and `POST /demo/sign-in` with a
plain `fetch()` instead of the typed client — the only other file under `apps/evaluate` allowed to
call `fetch()` directly (`tests/test_split_boundaries.py` checks this). It lets a reviewer
"Continue as Demo Admin / Supervisor / Reviewer" from the sign-in screen or switch persona from the
account page, without a passkey ceremony; `POST /demo/sign-in` sets the same session cookie a real
sign-in does, so every other `/store/v1` route treats it as an ordinary session (roles, CSRF,
expiry, "your sessions") once issued. The real passkey flow (`api/passkeys.ts`) is untouched and
stays visible on the sign-in screen below the demo buttons. A "Demo mode" badge in the header shows
whenever the server has demo mode on, whether or not the current session used it. For a demo
persona (`useIsDemoPersona()`), the "You have one authenticator" prompts are hidden: a persona's
only authenticator is Store's keyless placeholder, so the nag would be noise.

## Run it

```bash
npm --prefix frontend run build:evaluate        # -> call1/store/static/evaluate/ (plus shared fonts/icons)
python -m call1.store serve                     # http://localhost:8010/
python -m call1.store setup-code --email you@example.com --display-name "You"
# open http://localhost:8010/#/enroll, enter the code, register a passkey or security key
```

Dev server with hot reload (port 5175, proxies `/store/v1` to Store on `127.0.0.1:8010`):

```bash
CALL1_STORE_DEV_ORIGINS=http://localhost:5175 python -m call1.store serve
npm --prefix frontend run dev:evaluate
# open http://localhost:5175 — not 127.0.0.1: the WebAuthn relying-party ID is `localhost`
```

`CALL1_STORE_DEV_ORIGINS` is required for the dev server: Store checks the `Origin` of every
state-changing request and the WebAuthn `clientDataJSON` origin, and without it the Vite origin
gets 403 `origin_not_allowed`. The Vite config sets `strictPort`, so the origin cannot drift.

Store's dev mode deviates from the contract in three documented places (call1/store/README.md):
the cookie is `call1_session` without `Secure`, the relying party is `localhost` with `http://`
origins, and upload/content grant URLs are `http://localhost:…`. Evaluate never inspects the
cookie, passes WebAuthn options to the browser unchanged, and **views must not reject `http://`
grant URLs**.

## Layout

| Path | What it holds |
|---|---|
| `main.tsx`, `App.tsx` | Entry; boot (contract check), public and signed-in layouts, route table |
| `api/client.ts` | `StoreClient`: typed `get/post/put/patch/delete(path, {path, query, headers, body})`, `url()` for `<audio src>`, `raw()` for binary/CSV, `newIdempotencyKey()` |
| `api/errors.ts` | `StoreError` (the contract envelope: `code`, `details`, `retryable`, `requestId`, `retryAfterSeconds`; `err.is('review_version_conflict')` narrows `details`), `StoreUnreachableError`, `ContractMismatchError`, `describeError()`, `isNotImplemented()` |
| `api/contract.ts` | `checkContract()`: refuses a different contract major; `BUILT_FOR_CONTRACT` |
| `api/changes.ts` | `startChangePoller()`: `GET /changes` polling, resync on 410 `cursor_expired`/`cursor_unknown`, back-off and offline reporting |
| `api/passkeys.ts` | `@simplewebauthn/browser` ceremonies: sign-in, enrollment, step-up add-authenticator; browser error copy |
| `api/demo.ts`, `state/demo.ts` | Demo mode (`/demo/*`, the one exception to "only `/store/v1`" — above): `fetchDemoStatus()`, `demoSignIn()`, `useDemoStatus()` |
| `api/derive.ts` | Display of Store's derived result states (text label + tone); `callQaBadge()`; `agentLabel()` |
| `api/queryKeys.ts` | TanStack Query keys and `keysForChange(event)` |
| `api/types.ts` | Named contract types (`CallListItem`, `SessionInfo`, `SignalTaxonomy`, …) |
| `api/signalRules.ts` | Contact Signals rules engine (1.4.0): the recipe checks (regex-safe subset, caps), the editor's simple form over a recipe filter, pack origin text, and the Workbench "why" words |
| `api/signals.ts` | Contact Signals v2 display rules and client-side checks: built-in names, families (lifecycle, resolution, custom), hit and field chip text, alert condition text, the outdated-taxonomy label, `taxonomyProblems()` (the §9.6 caps read from `ContractParameters`, reserved IDs, forbidden PII classes) |
| `state/app.tsx` | `useStore()`, `useSession()`, `useChangeEvents()`, `usePollChanges()` |
| `state/router.ts` | Hash routes, `href()`, `navigate()`, invitation-link normalization |
| `state/connectivity.ts` | `useConnectivity()` for the offline banner |
| `components/` | `Header`, `OfflineBanner`, `ui.tsx` (Button, StatusPill, Chip, Card, PageHeader, EmptyState, NotBuiltYet, Loading, Notice, ErrorNotice, Field, TextInput, TextArea, SelectInput, Checkbox, CharCount, Dialog, OneTimeSecret, date/duration formatters), `ThreadWaveform.tsx` + `threadWaveformGl.ts` (the Workbench's three.js thread waveform, below) |
| `auth/` | Sign-in, enrollment, your account (authenticators and sessions) |
| `admin/` | Accounts, invitations, Process installations and service keys, ASR vocabulary (`admin/VocabularyPanel.tsx`, decision 33) |
| `views/` | One file per product view (below); `views/signals/` (taxonomy tab, category editor, preview, activation dialog, alert rules, versions), `views/workbench/ContactSignalsSection.tsx` and `views/metrics/SignalsMetricsCard.tsx` are their parts |

## Routes

| Hash | Screen | Who |
|---|---|---|
| `#/calls` (default), `#/calls?signal_category=…&signal_subcategory=…&signal_alert=…` | `views/CallsView` — the call list, optionally filtered by signals | every role |
| `#/calls/:callId`, `#/calls/:callId?turn=N` | `views/WorkbenchView` (`turn` seeks to that transcript turn once it loads) | every role |
| `#/rubrics`, `#/rubrics/:rubricId` | `views/RubricsView` (Rubric Studio) | every role reads; supervisor edits |
| `#/signals`, `#/signals/:categoryId`, `#/signals/alerts`, `#/signals/versions` | `views/SignalsView` — Contact Signals v2: taxonomy editor, alert rules, versions | every role reads; admin (`manage_signals`) edits |
| `#/queue` | `views/QueueView` (human review queue) | every role |
| `#/escalations` | `views/EscalationsView` | every role reads; supervisor resolves |
| `#/metrics` | `views/MetricsView` | every role; review agreement is supervisor |
| `#/admin/accounts` · `invitations` · `installations` · `vocabulary` | Admin area (`vocabulary` is the ASR vocabulary panel for dual transcription, decision 33, [docs/DualAsr.md](../../../../docs/Architecture.md)) | admin (nav hidden otherwise) |
| `#/admin/state` · `audit` · `release-trust` · `pro1` · `usage` · `price-table` | "Not built yet" state for the deferred admin screens (absent from the admin nav) | admin |
| `#/account` | Your authenticators and sessions | every role |
| `#/enroll?token=…`, `#/enroll` | Enrollment from an invitation link, or a setup-code form | anonymous |
| `#/sign-in` | Sign-in (any route shows it while signed out) | anonymous |

Store's invitation links are `<store>/enroll#<token>`; the app rewrites that to
`/#/enroll?token=<token>` before the first render and removes the token from the address bar once
enrollment succeeds.

## View contract (for the views agent)

Each file in `views/` default-exports one component. The shell renders exactly one per route and
passes `ViewProps` (`views/types.ts`):

```ts
interface ViewProps {
  client: StoreClient;           // typed Store client — the only way to reach Store
  contract: ContractInfo;        // contract.parameters: Store's effective timings; never hardcode
  session: SignedInSession;      // session (SessionInfo), role, can(permission), atLeast(role), signOut(), refresh()
  navigate(route: Route, options?: { replace?: boolean }): void;
}
WorkbenchView: ViewProps & { callId: string; turn?: number }        // #/calls/:callId[?turn=N] (remounted per callId)
RubricsView:   ViewProps & { rubricId?: string }                    // #/rubrics[/:rubricId]
SignalsView:   ViewProps & { tab: SignalsTab; categoryId?: string } // #/signals[/:categoryId|alerts|versions]
CallsView:     ViewProps & { filters?: CallsFilters }               // #/calls[?signal_category=…]
QueueView, EscalationsView, MetricsView: ViewProps
```

All seven views are implemented: `CallsView`, `WorkbenchView`, `RubricsView`, `SignalsView`,
`QueueView`, `EscalationsView` and `MetricsView`. Views render inside `<main>` (the shell provides the header,
nav, offline banner and padding); each starts with `PageHeader` and a `max-w-* mx-auto` container.

**Workbench** (`views/WorkbenchView.tsx`) reads `GET /store/v1/calls/{call_id}` (`CallDetail`), then
gates the transcript/summary/contact-signals fetches on that call's `results` (`ResultGroup.state`)
so a still-analyzing section never 404s. Verdict override, escalation resolution and "retain
review" all read `expected_version` from `GET /store/v1/calls/{call_id}/review`
(`CallReviewState`) and the machine version from `CallDetail.evaluation`, and are hidden — not
merely disabled — until an evaluation exists or the reviewer lacks the permission. A hidden
`<audio>` element drives playback, with Play/Pause, the clock and a Speed select (1–2x) above the
waveform. Each transcript line's Play button (accessible name "Play from m:ss") seeks to
that turn and plays; the seek position is applied even before the audio metadata has loaded. A
`play()` that a pause interrupted (`AbortError`) or that waits for a user gesture
(`NotAllowedError`) does not mark the audio unavailable; only a load or format failure does.

**Thread waveform** (`components/ThreadWaveform.tsx`, WebGL half in `components/threadWaveformGl.ts`)
is the legacy Workbench's three.js "thread" waveform
(`frontend/src/components/workbench/WaveformDeck.tsx`), copied rather than imported so the legacy
app stays untouched and Evaluate never reaches `@/services/api`, `@/types` or `/api/v1`. It is
data-agnostic: plain props (`sourceKey`, `loadAudio`, `audioRef`, `currentTime`/`duration`/
`playing`, `onSeek`, `onTogglePlay`, optional `turns` and `markers`). It never calls `fetch()`
itself (`tests/test_split_boundaries.py`); the Workbench's `loadAudio` reads
`GET /store/v1/calls/{call_id}/audio` through `client.raw()`, so the session cookie applies.

- **Ported as-is:** decode with `AudioContext` behind an `AbortController` and a generation guard;
  2048 RMS windows, 0.65 gamma, triangular smoothing into a float `DataTexture`; two
  `ShaderMaterial` ribbons (teal Agent, amber Caller, mapped from each turn's `channel`); hover
  wake, click ripple, playhead, hover line and time tip; the detail drawer with the turn's text;
  palette sync on theme change (`call1:theme-change` and the `data-theme` attribute);
  IntersectionObserver and `document.hidden` pause the loop; context-loss handling; full disposal
  (meshes, materials, texture, renderer, AudioContext). The `fabric-*` chrome comes from the shared
  `index.css`.
- **Changed:** markers — contact signals (Flag) and verdict evidence (Quote, labelled with the
  verdict status, "(overridden)" when a reviewer overrode it) — sit in a lane above the stage rather
  than inside the slider; a speaker lane under the ribbon shows transcript turns. The stage is the
  "Seek audio" slider: ←/→ (and j/l) seek 5 s, Shift 15 s, Home/End jump to the ends, **Space plays
  or pauses** (legacy used Space to open the detail), Enter opens the detail, Escape closes it.
- **Muted audio:** Store mutes PII by zeroing samples, so windows of digital silence are pinned to
  zero after smoothing (a flat thread, never filled in by neighbours) and hatched, with a legend
  "Flat = silent or muted (n)". Nothing is inferred beyond the decoded samples.
- **Reduced motion:** the thread is drawn once per change with no drift, wake or ripple; the deck
  reads "Motion reduced" and `data-motion="reduced"`.
- **Fallbacks, never a blank box:** without WebGL (or after a lost context) or when the recording
  cannot be decoded, the stage gives way to a text state and a native range input still named
  "Seek audio"; when the `<audio>` element itself fails, a text state (the masking-aware message
  below) replaces it. `data-state` on the deck is `loading`, `ready`, `no-webgl`, `no-waveform` or
  `audio-error`.
- **Known cost:** three.js is in the Evaluate bundle (about 900 kB minified); code-splitting it is
  a possible follow-up.

**Withheld text (contract 1.2.0).** Store serves no transcript text or audio for a call until the
model PII findings for its current transcript revision exist. The transcript group then reads
`partial`, with a `partial_reason`, and `TranscriptView.text_withheld` is true. The Workbench shows
"Transcript text withheld" with that reason instead of empty turns. A `pending` transcript group
shows "Analyzing", and a `failed` one shows "Transcript unavailable" with its failure code. A failed
audio load reads "Audio is withheld until PII masking finishes for this call." When the text
becomes available, a change-feed job event refetches the transcript and the Workbench retries the
audio once.

**Rubric Studio** (`views/RubricsView.tsx`) lists rubrics, and per rubric offers a draft editor
(`draft_revision` optimistic concurrency), publish, retire, version history and a draft test
(`POST .../draft/tests`, polling `GET .../draft-result`). The criterion editor covers the common
`RubricCheck` fields (check type, phrases, threshold, speaker, model ids, pass/fail/N-A
expressions, escalation triggers); a field this editor doesn't expose is preserved as-is because
the whole `RubricCheck` object round-trips through local state.

**Review queue** (`views/QueueView.tsx`) covers claim-next, assign, start, release and resolve, plus
(`manage_queue_rules`) creating, enabling/disabling and deleting queue rules.

**Escalations** (`views/EscalationsView.tsx`) lists and (`resolve_escalation`) resolves.

**Metrics** (`views/MetricsView.tsx`) covers executive, per-rubric (with a rubric picker) and
(supervisor) review-agreement, all against `ContractParameters`-free date-range query params.
`MetricsQuery.start`/`end` are aware timestamps, so the From/To dates are sent as RFC 3339 with the
browser's own offset: From is the start of that local day, and To is the start of the next local
day, because `end` is exclusive and To should cover the whole chosen day. "Audio audited" shows
minutes under one hour (one 44 s call reads "1 min", not "0.0") and hours with one decimal above.
The lazy-loaded Metrics view uses Recharts 3 through shadcn chart components adapted to Call1's
existing theme tokens (`components/charts/chart.tsx`; MIT attribution beside the source). Your
center shows daily score/volume area charts, criterion outcome stacks and accessible values.
Daily buckets are evaluation dates in UTC; date filters select call creation time.

**Peer comparison** is a separate, demo-only explorer: two fictional cohorts, three measures,
weekly trends, center ranking, interpolated median/quartiles and a separate retail coaching
example. All comparison data, including Demo center, is in `views/metrics/benchmarkDemo.ts` and
is explicitly fictional, never joined to Store results or sent elsewhere. Real installs show
an explanation instead. The shared benchmark service remains planned. `--demo` also adds 560
synthetic Store sessions for the Your center charts (see the class demo script); these are
scripted snapshots, not evidence of model accuracy.

**Agent identity (contract 1.1.0).** Wherever an agent appears (calls list, Workbench header, queue
items, escalations), Evaluate shows `agentLabel(agent_id, agent_display_name, agent_extension)`
from `api/derive.ts`. It follows `call1.contracts.calls.agent_label()` exactly: "Name (ext)", "Name",
or `agent_id` (plus " (ext)") when there is no display name. `agent_id` stays the key for the
agent filter and queue rules, and Evaluate never parses it. A re-upload that changes the metadata
emits a `call` change event (status `metadata_updated`), which invalidates the call list and that
call's detail like any other `call` event, so open screens pick up the new label.

What a view can use:

- **Requests.** `client.get('/store/v1/calls/{call_id}', { path: { call_id } })` — paths, params,
  headers and bodies are type-checked against the contract. CSRF is automatic on writes. Routes
  with header idempotency (`requestReanalysis`, `testRubricDraft`) require
  `headers: { 'Idempotency-Key': key }`. Store digests the body under the key, so one key carries
  one body: use `useIdempotencyKey()` with `sendIdempotent(keys, body, send)` (`api/idempotency.ts`),
  which reuses the key for a retry of the same body, makes a new one when the body changes or after
  a success, and on `idempotency_key_reused` resends once under a fresh key. Expected-version
  writes put the version in the body; on 409 `isVersionConflict(err)` covers
  `review_version_conflict` (`details.current_version`), `conflict`
  (`details.current_evaluation_version`) and `rubric_version_conflict`: invalidate the resource's
  query so the latest state shows, keep the error visible (`describeError` says what changed) and
  let the user submit again — never overwrite or retry on their behalf. The queue's resolve reads
  the call's `review_version` from `GET /calls/{id}/review` (a queue item's `item_version` is a
  different counter). Deferred routes answer 501: check
  `isNotImplemented(err)` and render `NotBuiltYet`.
- **Caching.** TanStack Query, with keys from `queryKeys` (`queryKeys.call(id)`,
  `queryKeys.reviewQueue`, `queryKeys.escalations`, `queryKeys.rubrics`, `queryKeys.metrics`,
  `queryKeys.reanalysis`). Keep the family as the first element so change-feed invalidation
  reaches the query (e.g. `[...queryKeys.call(id), 'transcript']`).
- **Queue rules.** A stored `ReviewQueueRuleRecord` carries `rule_version`, `updated_at` and
  `updated_by_account_id`, and Store forbids extra fields in `ReviewQueueRuleSave.rule`. Build the
  body with `ruleForSave(record)` (`views/QueueView.tsx`), which copies exactly the
  `ReviewQueueRule` fields and fails the typecheck if the contract's rule fields change.
- **Live updates.** The shell polls `GET /store/v1/changes` while signed in and invalidates the
  keys `keysForChange()` names (call/result/job_group/job → `calls`, `call:<id>`, `metrics`;
  review → also `escalations`; review_queue → `review-queue`, `escalations`; reanalysis_request →
  `reanalysis`; rubric → `rubrics`). For anything else use
  `useChangeEvents(events => …, { kinds: ['result'], callId })`. After a write, call
  `usePollChanges()()` to pick the change up at once. On a lost cursor every query is
  invalidated, so views re-snapshot. `CallDetail.change_cursor` is the snapshot position if a
  view needs its own poller (`startChangePoller(client, { after, … })`).
- **Permissions.** Gate controls with `session.can('override_verdict')`,
  `session.can('resolve_escalation')`, `session.can('manage_rubrics')`,
  `session.can('manage_queue_rules')`, `session.can('assign_review')`, `session.can('play_audio')`;
  Store still enforces them. Hide what a role cannot do rather than showing it disabled.
- **States.** `resultStateDisplay(state)` gives the text label and tone for every `ResultState`
  (Analyzing / Ready / Partial / Stale / Needs attention / Not run / Unknown); `callQaBadge()`
  adds the score. Always show text with color (`StatusPill`). Unknown enum values from a newer
  minor render as "Unknown", never an error.
- **Media.** `client.url('/store/v1/calls/{call_id}/audio', { path: { call_id } })` for
  `<audio src>` (same origin; the cookie goes along). Grant URLs are used as Store returns them.
- **Errors and offline.** `ErrorNotice` renders any thrown error; `useConnectivity().offline`
  says whether Store is reachable (the shell already shows a banner). A signed-out answer anywhere
  returns the whole app to sign-in; views do not handle it.

## Contact Signals v2 (contract 1.3.0; rules engine 1.4.0)

The Evaluate side of `docs/ContactSignalsV2.md` §10 (build step F4). Engines are pluggable and may
be fakes; the UI never assumes a model ran, and never shows a number Store did not send. Signals
are **unscored**: every surface says so, and nothing here touches a score or the review version.

**Signals page** (`views/SignalsView.tsx`, `views/signals/`). A "Signals" nav item (lucide `Radar`)
after Rubrics. Reviewers and supervisors read it; only admins (`manage_signals`, decision 22 Q1)
edit, and without the permission every control is hidden or disabled with a "Read-only" note.

- **Pipeline selector:** admins can switch between v1, shadow and v2 (`PUT /signals/settings`,
  `expected_record_version`). Shadow and v2 require a qualified Process classifier. The explanatory
  pipeline status banner is omitted.
- **Detection:** each category's recipe selects Rules + examples or Model (Gemma). There is no
  taxonomy-wide detection switch or status banner. The legacy settings field remains accepted for
  compatibility and does not override recipes.
- **Tree:** the 8 built-ins (lock icon and the text "Built-in"), then custom categories; each node
  shows its gloss, speaker scope, active subcategory and field counts, and calls with a hit in the
  last 7 days (`GET /metrics/signals?start=<7 days ago>&include_inactive=true`).
- **Category editor:** name and gloss with counters (built-ins: disabled, fixed by Call1),
  description and speaker (custom only), examples, stage-1 and stage-2 thresholds (empty = "Engine
  default"; the hint calls it a calibrated model score), quote narrowing, subcategories (add,
  edit, deactivate, reorder; "Other" and "Not <category>" shown as fixed rows) and fields. The
  caller-detail hint remains under example inputs; the repeated warning at the top is omitted.
- **How it's detected** (`views/signals/RecipeEditor.tsx`, `data-testid="signals-recipe-editor"`;
  built-ins too, since `recipe` is a built-in editable field): the engine (Model (Gemma) or Rules +
  examples; switching to Model keeps the recipe with `engine: gemma`), the pack origin ("From the
  Retail pack v1, tuned on 25 public calls", plus ", edited here"), and for Rules: "Score needed"
  (threshold, slider plus number, 0.05–0.95), "Must also pass" (nothing, a phrase, similar to
  examples with its minimum share, either, or both), the speaker (read-only, the category's; a
  custom category's speaker change moves the recipe's speaker rule with it), "Where in the call"
  (anywhere, quarter and half presets, or a custom start window), the phrase list (words or
  patterns, add and remove, "N of 24", phrase weight, negation window 0–6) and "Gemma
  double-check" (`check: gemma`). A pack filter richer than the form (nested or `not` groups, a
  phrase rule with its own phrases) is shown as written and kept. The new-phrase input checks as
  you type (`api/signalRules.ts`): the contract's regex-safe subset (`lexicon_phrase_problem`: no
  look-arounds, named groups, inline flags, back-references, or unbounded repeat around a repeat;
  a letter required), duplicates, the phrase cap, and a digit run that Store's PII rules would
  refuse (a warning; Store is the authority). `taxonomyProblems()` adds the recipe checks and the
  caps `max_signal_recipe_rules` and `max_signal_lexicon_phrases`; a refused recipe path reads
  "Proposed fix › how it's detected › phrase 3". When a save changed a recipe, the activation
  dialog's update reruns every signal stage (`rescore_signals: true`).
- **Field editor:** name, type, description ("This is the only instruction the extractor gets for
  this field"), enum values (enum only, at most 12), and PII class with the forbidden classes
  disabled and labelled "masked before any model sees it".
- **Extraction destinations (§11.1):** above the category fields, admins see each Process host's
  stage-3 extraction entry (its `signal_extraction` default, plus the taxonomy's
  `fallback_extraction_entry_id` when that host has it) with its destination host and route class,
  read from `GET /catalog-snapshots`. Today only the appliance route is permitted, so every entry
  names the appliance; the line matters once BYOK opens after opt-in (decision 22, Q13).
- **IDs** are made from the name when a node is added and never change (there is no delete;
  `active: false` retires). A node added in this session can be removed until it is saved.
  Evaluate never generates `alerts` or `versions` as a category ID, because they are route words
  (a category with such an ID made elsewhere is still valid, but its tree link opens that tab).
- **Caps and checks** (`api/signals.ts` `taxonomyProblems()`): the §9.6 caps come from
  `contract.parameters` (`max_custom_signal_categories`, `max_active_subcategories`,
  `max_fields_per_path`, `max_option_gloss_chars`, `max_signal_alert_rules`,
  `signal_preview_max_calls`, `signal_backfill_max_calls`), never hardcoded. "Add" buttons disable
  at a cap with the reason, counters turn red with "too long", and Save stays disabled with a
  list of what to fix. Store stays the authority: a `validation_failed` refusal names the path
  (`details.field`), which the page turns into "Caller objective › Cancel account › option text".
- **Save** is `PUT /signals/taxonomy` with `expected_record_version`. A no-op save says so. A
  conflict (`signal_taxonomy_conflict`) re-reads, loads the latest saved version and says so; it
  never overwrites another admin's save. If someone else saves while the draft is dirty, a notice
  offers to discard the edits. Success shows "Saved as taxonomy vN", then the activation dialog.
- **Activation dialog** (§9.1): "Update calls from the last [7] days (up to 200)" (a `rescore`
  backfill, digest-driven, `Idempotency-Key`), "Send matching calls to the review queue" (a SIGNAL
  queue rule; shown with `manage_queue_rules`) and "Alert on this" (an alert rule on the last
  edited node). They run in the spec's order — alert rule, queue rule, backfill — and each
  step's outcome is listed; the first failure stops the rest.
- **Test on recent calls:** the five most recent settled calls with published signals are ticked
  (up to `signal_preview_max_calls`). "Run preview" sends the unsaved taxonomy
  (`POST /signals/previews`, `useIdempotencyKey()` + `sendIdempotent()`), then reads
  `GET /signals/previews/{id}` (change feed plus a 3 s backstop while any call is pending). Per
  call: "Analyzing", "Failed (code)", "No change", or the diff scoped to the edited category —
  added, removed, relabelled, fields filled and segments changed (`segments_changed`: a
  multi-segment merge formed, split or grew), each with its masked quote and a Workbench link at
  the turn. A hit that gains its first real subcategory reads "Added: <category> › <subcategory>";
  a move between two subcategories reads "Relabelled from …". When the rules engine ran (hits carry
  `why`), each call also lists its signals by category with how many the rules decided and how
  many Gemma double-checked (`data-testid="signals-preview-counts"`). `builtin_changed` shows "This change also moves built-in signals"; `options_trimmed`
  shows "Some option text was too long for the classifier: shorten <name>".
- **Alert rules tab:** name, condition ("Caller objective › Cancel account › reason = price"),
  status (a rule on an inactive node says it matches nothing) and calls matched in 7 days. The
  editor offers "equals" only for enum and boolean fields.
- **Versions tab:** version, who, when, notes and a 12-character digest; admins can redact the text
  of a non-current version (`POST …/redaction` with the digest and a reason).

**Workbench** (`views/workbench/ContactSignalsSection.tsx`). The card's subtitle says
"Unscored"; the header shows "v1 signals" or "v2 signals" and the group state. Each hit shows a
category chip (family tone plus its name), "›" and a subcategory chip, field chips (`reason:
price`; string values show their masked surface text), alert chips, `start–end`, the speaker, the
family name, "narrowed" when stage 3 narrowed the quote, and the quote. Clicking a hit jumps to
the turn. A multi-segment hit (decision 25, `parts` and `span_end`) adds a "×N segments" chip, a
time range to `span_end`, and a "Show N more segments" disclosure (`aria-expanded`) listing each
part's quote as a button that jumps to its turn. Feedback saved on a part while it was a
separate hit shows as "A segment was confirmed/dismissed before the segments merged" until the
merged hit gets its own verdict. A hit that carries `why` (contract 1.4.0) gets a "Why"
disclosure (`data-testid="signal-why"`, `aria-expanded`): "Found by rules" or "Found by Gemma",
a "Gemma double-checked ✓" chip when `check` is `confirmed`, and when opened "Found by rules ·
score 0.62 ≥ 0.38 · similar examples 0.55 · phrase 'refund' (matched, +0.50)", who decided the
category and subcategory, each rule's outcome, and the up-to-3 nearest example IDs with their
cosines (never text). A hit without `why` shows nothing new. A call with more than 8 signals (real Gemma output runs to 40-odd)
gets a "Show:" row of toggle buttons (`aria-pressed`, `data-testid="signals-filter"`): "All N" and
one per category present with its count, which filters the list (`evaluate-signals` cs10 checks a
42-span call for overflow in both schemes). Reviewers with `override_verdict` get Confirm / Dismiss on the category and Confirm /
Correct (a subcategory select, "Other" included) on the subcategory (`PUT
/calls/{id}/signal-hits/{hit_id}/feedback`, `expected_feedback_version`; a conflict re-reads). A
subcategory verdict given for an earlier subcategory reads "Judged an earlier subcategory".
States (§10.2): not run, "Analyzing", v2 "No signals found" with the categories checked and "N
segments scored" (plus unattributed segments skipped), v1 list, partial by stage ("Subcategories
unavailable (engine error)", "Fields unavailable", Process's `partial_reason`), "Refreshing" with
the previous result, "Needs attention" with the failure code, "Needs attention: signals wait for
PII masking to be retried." when the group failed while the transcript text is withheld, the
outdated label "Scored with taxonomy v3 (current v5): new subcategories not applied" with
**Update signals** (`requestReanalysis`, kind `contact_signals`, `rescore_signals: false`,
idempotency key), and "Text withheld until PII masking finishes". Admins see **Compare with v2**
when `comparison_preview_id` is set (shadow mode), with the same diff as the preview.

**Waveform.** `WaveformMarker` gains `end`, `detail`, `chips`, `family` and `hitId`. A marker
with `end` draws a range band in the marker lane under its pin (`data-signal-band`,
`data-testid="waveform-span-band-<hitId>"`); clustering still uses `time`. The cluster popover
shows category › subcategory, the family name, the time range, field chips and the quote. A
multi-segment signal's marker carries `segments`: its band runs to `span_end`, and the band title
and popover say "×N segments".

**Calls.** "Caller need" (intent subcategory chips from `caller_needs`) and "Signals" (category
chips, "+N", alert chips) columns, each showing "Analyzing" or "Refreshing" while the group is
pending or stale. Category, Subcategory (after a category) and Alert filters map to
`signal_category`, `signal_subcategory` and `signal_alert`, and stay in the address bar.

**Queue.** `RULE_FIELDS` includes `target_signal_alerts`; rule saves always send it. The rule
editor offers the SIGNAL stream (which needs at least one alert) and an "Alerts" checkbox group,
also available on other streams behind "Only calls with a signal alert". Items show trigger-alert
chips and, when a trigger alert no longer matches the call's current signals, "Signals changed
since this item was created".

**Metrics** (`views/metrics/SignalsMetricsCard.tsx`): a Signals card with calls scored, calls
with a signal and calls by pipeline; **Top caller needs** (the Caller objective's subcategories
ranked by calls, with share, each linking to the filtered call list); a category table (hit rate,
hits, category and subcategory precision) with a drill-down (subcategories, enum and boolean
field values, by day, top agents); and an alert table. Precision reads "—" under five judged hits.

**Live updates** (`api/queryKeys.ts`): new keys `signalTaxonomy`, `signalTaxonomyVersions`,
`signalAlertRules`, `signalMetrics(range)` (under `metrics`) and `signalPreview(id)`.
`keysForChange` maps `signal_taxonomy` → the taxonomy, versions, alert rules, calls, metrics and
every call's queries (the outdated label is read-time); `signal_alert_rule` → rules, calls,
metrics and every call (alerts are read-time); `signal_alert` → calls, metrics and that call;
`reanalysis_request` → also the open preview; `catalog` → catalog snapshots. `result` and
`review` events already refresh the call, the lists and metrics.

## Auth rules the shell follows

- Passkey-only: no password field, reset or fallback anywhere. Sign-in is account-first (email,
  then `allowCredentials`), so non-discoverable security keys work. Lost every authenticator? The
  copy sends the reviewer to an admin for a re-invite, which replaces the old authenticators.
- `SessionInfo.csrf_token` is kept in memory only (the client), recovered from
  `GET /store/v1/auth/session` after a reload, and re-read once on 403 `csrf_failed`.
- One-time secrets (invitation links, service-key tokens) live only in component state, are shown
  once with a copy button, and disappear when dismissed.
- A successful enrollment ceremony signs the reviewer in at once (`EnrollPage` calls `signedIn()`
  with the session Store opened), so every route renders signed in right away. The confirmation
  screen stays up because the shell renders the enroll route whatever the session state.
- Enrollment prompts for a second authenticator until one exists (`prompt_second_authenticator`),
  on the account page and in a dismissible banner.

## Not built yet / deferred

- Contact Signals v2: field-level reviewer feedback is later (decision 22 Q10: category and
  subcategory verdicts only), and there is no outbound alert delivery (Q6). The pipeline switch
  stays disabled until a host qualifies a stage-1 engine (F6).

- Speaker correction (`POST .../speaker-corrections`) has no Workbench UI yet; the contract client
  call is trivial to add but was left out to keep the initial views focused on the explicitly
  scoped flows (playback, transcript, scorecard, verdict override, escalation resolution, summary,
  contact signals).
- Invitation delivery through the SMTP relay needs admin state, which Store defers (501); the
  invitation form issues out-of-band links only.
- The admin area covers identity, invitations, installations and keys. Admin state, the audit
  log, release trust, Pro1 clear-block and key releases, usage and the price table are not in this
  shell (Store answers 501 for the deferred ones). They are absent from the admin nav, the admin
  page lists them as not built, and a deep link (`#/admin/state`, `audit`, `release-trust`,
  `pro1`, `usage`, `price-table`) renders `NotBuiltYet`, never another panel.
- The frontend has no unit-test runner yet. The browser suite (`frontend/e2e/evaluate-*.spec.ts`,
  `auth.spec.ts`) drives the built app against a real Store and Process, with Chrome's virtual
  authenticator for the WebAuthn ceremonies. It covers keyboard-only operation of the main flows
  (`evaluate-keyboard.spec.ts`), a lost change-feed cursor (`evaluate-live-updates.spec.ts`), audio
  seeking from the transcript (`evaluate-workbench.spec.ts`), the thread waveform
  (`evaluate-waveform.spec.ts`: canvas, click and keyboard seek, reduced motion, no-WebGL) and the deferred admin screens
  (`evaluate-not-built.spec.ts`).
