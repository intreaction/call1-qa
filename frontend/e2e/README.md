# Browser end-to-end tests (Playwright)

These tests drive the **built** Evaluate that a real Store serves, with a real Process behind it,
in system Chrome locally or Playwright Chromium in CI (`CALL1_E2E_BROWSER` selects the channel). Passkeys come from a **CDP
virtual authenticator** in each page. The Python harness,
[`tests/e2e/`](../../tests/e2e/README.md), starts the servers.

```bash
npm --prefix frontend run test:e2e                          # both projects: dark and light
npm --prefix frontend run test:e2e -- --project=dark        # one color scheme
npm --prefix frontend run test:e2e -- e2e/smoke.spec.ts -g "invited"
CALL1_E2E_KEEP=1 npm --prefix frontend run test:e2e         # keep the stack dir and run dir
npm --prefix frontend run typecheck:e2e                     # tsc over e2e/ and playwright.config.ts
```

The temporary directory shown below is `CALL1_E2E_ROOT` when set, otherwise the system temporary directory plus `call1-e2e`. The Python interpreter is `CALL1_E2E_PYTHON`, then `.venv/bin/python` when available, then `python3`.

Before you start:

- **The build is what gets tested.** Store serves `call1/store/static/evaluate/`. The global setup
  warns when `frontend/src/apps/evaluate` is newer than that build. Run
  `npm --prefix frontend run build:evaluate` only when you mean to test newer source.
- **Artifacts** go to `<temporary-directory>/call1-e2e/playwright/`: `test-results/<run-id>/` (traces and
  screenshots on failure — `outputDir` is unique per invocation by default, so overlapping runs
  never empty each other's in-flight files; set `CALL1_E2E_RUN_ID` to pin one, or pass `--output`
  to override it outright) and `report/`. Open the report with
  `npx playwright show-report <temporary-directory>/call1-e2e/playwright/report`. The Store and Process
  logs are in the stack dir printed at start-up, under `logs/`. Set `CALL1_E2E_KEEP=1` to keep them.
- **Settings:** 3 workers (`CALL1_E2E_WORKERS=n` changes it), files run in parallel and tests
  within a file in order, test timeout 90 s, `expect` timeout 15 s.
- **Do not change application code.** A test that fails because the feature is broken is the
  result. Keep it failing and make the assertion say what is wrong.

## How the stack starts

`e2e/global-setup.ts` runs `.venv/bin/python tests/e2e/serve_stack.py --info-file
<run>/stack.json`. That starts one Store and Process pair for the whole run, with fake handlers,
free ports (never 8000, 8010 or 8020) and data in `<temporary-directory>/call1-e2e/playwright-…/`. It
exports `CALL1_E2E_STACK_FILE` and `CALL1_E2E_RUN_DIR`, a `<temporary-directory>/call1-e2e/pw-run-…/`
directory that holds shared credentials, locks and state. Teardown closes the helper's stdin; it
stops both servers and both directories are deleted.

To use real handlers, set `CALL1_E2E_HANDLERS=real`. Add `CALL1_REAL_MODELS=1` for
`CALL1_BACKEND=mlx` with the repo's `data/models`. `CALL1_E2E_PYTHON` overrides the interpreter.

**The stack is shared by every worker and both projects.** Other tests' calls, accounts,
invitations and rubrics are visible. Create what you assert on, find it by the ID or a unique label
you chose (`agentId: \`pw-${testInfo.project.name}-${Date.now()}\``), and never assert totals or
row positions.

## Fixtures

Import from the fixtures module, not from `@playwright/test`:

```ts
import { test, expect } from './fixtures';

test('reviewer sees a call', async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');           // own context, signed in
  const agent = `pw-${testInfo.project.name}-${Date.now()}`;
  const call = await ingestSample('call_03_dispute_escalation', { agentId: agent });
  await waitUntilSettled(call.call_id);
  await reviewer.page.goto('/#/calls');
  await expect(reviewer.page.getByRole('row').filter({ hasText: agent })).toBeVisible();
});
```

| Fixture | Type | What it is |
|---|---|---|
| `stack` | `StackInfo` | The running stack: `store_url`, `store_port`, `process_url`, `process_port`, `console_token`, `process_console_url`, `service_key`, `installation_id`, `dir`, `logs_dir`, `store_data`, `process_config`, `python`, `repo`, `handlers`, `real_models`, `run_dir` |
| `storeURL` | `string` | `http://localhost:<port>`: Evaluate at `/` and the API at `/store/v1`. **Also the default `baseURL`**, so `page.goto('/#/calls')` works. Use `localhost`, never `127.0.0.1`, because the WebAuthn RP ID is `localhost` |
| `processURL` | `string` | `http://127.0.0.1:<port>`: the Process console and `/process/api` |
| `consoleToken` | `string` | The Process console credential, sent as `X-Call1-Console-Token` on writes |
| `processConsoleURL` | `string` | `processURL + '/#console_token=…'`. Opening it signs the console in. The console reads the fragment, clears it and keeps the token in `sessionStorage` |
| `page` | `Page` | The standard page, plus a **virtual authenticator** and the client-address spread |
| `authenticator` | `VirtualAuthenticator` | The virtual authenticator of `page` |
| `newAuthPage(options?)` | `({ colorScheme? }) => Promise<{ context, page, authenticator }>` | A fresh browser context, standing for a second person or device. It gets the project's color scheme and `baseURL` and is closed after the test |
| `enrollAdmin(page?, options?)` | `(page = the test's page, { email?, displayName?, nickname? }) => Promise<Identity & { purpose }>` | A **new admin through the Evaluate UI** with a CLI setup code (`python -m call1.store setup-code`). Ends signed in on `#/calls`. The first call in a run uses `first_admin`; later calls use `break_glass` without a target, which creates a new admin account and writes a `break_glass_used` audit event, because `first_admin` is refused once an admin exists |
| `enrollWithSetupCode(page, options?)` | `(page, { email?, displayName?, purpose?: 'first_admin' \| 'break_glass', targetAccountId?, nickname? }) => Promise<Identity>` | Any setup-code enrollment through the UI, such as break-glass re-enrollment of an existing account |
| `signIn(page, email)` | `=> Promise<Identity>` | Signs in through Evaluate's sign-in screen, account-first: the email, then the passkey. The account must have enrolled earlier **in this run**; its credential is copied into `page`'s authenticator and the updated signature counter is saved back. Signs out a different account first |
| `createInvitedUser(role?, options?)` | `('reviewer' \| 'supervisor' \| 'admin' = 'reviewer', { email?, displayName?, nickname? }) => Promise<InvitedUser>` | The **stack admin** issues an invitation through `POST /store/v1/admin/invitations`. A **second browser context** opens the `…/enroll#<token>` link, clicks "Create passkey", then "Go to calls". Returns `{ email, displayName, role, accountId, invitationId, invitationUrl, context, page, authenticator }`, with the page signed in. The context is closed after the test |
| `adminApi` (worker) | `StoreApi` | `/store/v1` as the **stack admin**, `stack-admin@e2e.test`: one account per run, enrolled through the UI on first use. Its signed-in storage state is reused by every worker, so there are no repeated sign-ins |
| `apiAs(page)` | `(page) => Promise<StoreApi>` | `/store/v1` as whoever is signed in on `page`, through `page.request` (shares the cookie) with CSRF |
| `ingestSample(name?, options?)` | `(name = 'call_01_compliant', { unique = true, agentId?, agentChannel?, externalCallRef?, filename?, contentType?, expectStatus = 201 }) => Promise<IngestReceipt>` | `POST /process/api/recordings` with the console token, as the console's Import view does. `name` is a file in `sample_audio/`, with or without `.wav`, or an absolute path. With `unique` it uploads a copy whose first 4 samples are nudged. Process identifies a recording by its SHA-256, so an identical upload returns the **same** conversation. Returns `{ conversation_id, call_id, graph_id, conversation_created, graph_created, jobs, evaluate_url }`. `expectStatus: null` returns any status's body |
| `waitUntilSettled(callId, timeoutMs?)` | `=> Promise<JobGroupProgress>` | Polls Store's `GET /conversations/{id}/progress` with the service key until `settled`. Timeout is 60 s with fake handlers and 900 s with real ones |
| `processApi` | `{ get(path, { token? }), post(path, json?, { token? }) }` | The Process loopback API (`/process/api` may be left off the path). It sends the console token unless `token: false`; a string sends that token. Returns a fetch `Response` |
| `spreadClientAddress` | option, default `true` | See "Auth rate limits" below |

**Projects:** `dark` and `light` run every test with `colorScheme: 'dark'` or `'light'`. Evaluate
follows `prefers-color-scheme` until the user toggles, and sets `<html data-theme="dark|light">`.
Contexts from `newAuthPage` and `createInvitedUser` inherit the project's scheme.

### `StoreApi` (from `adminApi` and `apiAs`)

- `get(path, opts?)`, `post(path, data?, opts?)`, `put`, `patch` and `delete` return a Playwright
  `APIResponse`. `/store/v1` may be left off the path.
- `opts` is `{ params?, headers?, idempotencyKey?: true | string, csrf?: boolean }`.
  `idempotencyKey: true` sends a fresh `Idempotency-Key`, which `requestReanalysis` and
  `testRubricDraft` need. Pass a string to repeat a key.
- Writes send `X-Call1-CSRF` and `Origin` unless `csrf: false`.
- `json<T>(method, path, opts?)` parses the body and throws with it on a non-2xx answer.
- `session` is the `SessionInfo`, with `account_id`, `role`, `permissions` and `csrf_token`.
  `refresh()` re-reads it.

### `VirtualAuthenticator`

`credentials()` calls `WebAuthn.getCredentials`. The other members are
`addCredential(c)`, `removeCredential(id)`, `clearCredentials()`,
`setUserVerified(false)` (Store requires UV, so ceremonies should fail) and
`setAutomaticPresence(false)` (the authenticator stops answering, so the prompt waits and the
browser eventually times out). `addVirtualAuthenticator(page, options)` adds one to any page. It
takes `{ protocol, transport, hasResidentKey, hasUserVerification, isUserVerified,
automaticPresenceSimulation }`; the default is ctap2, internal, resident, with UV.

### Helpers (`import { … } from './fixtures'`)

These re-export `harness.ts`:

- `stackInfo()` and `setupCode(email, displayName, { purpose, targetAccountId })`.
- `storeCli(args)`, which runs any `python -m call1.store` command against the stack.
- `serviceRequest(method, path, body?)`: Store with Process's service key (fetch).
- `processRequest(method, path, { json, token })`, `conversationOf(callId)`, `uniqueEmail(prefix)`.
- `readCredentials(email)` and `saveCredentials(email, creds)`.
- `withLock(name, fn)`, a lock across workers. `readState`, `writeState` and `statePath` read and
  write shared run state.
- `enrollViaUI(page, { setupCode | invitationUrl, email, nickname?, enter? })`,
  `enrollAdminWithUI`, `signInWithUI`, `installClientSpread(context)` and
  `addVirtualAuthenticator(page)`, for building your own flows.

## Auth rate limits

Store rate-limits the anonymous auth steps per client address: enrollment begin 10 a minute,
sign-in begin 30 a minute. Every browser here connects from 127.0.0.1. The fixtures therefore give
each browser context its own synthetic address, sent as `X-Forwarded-For` on
`/store/v1/auth/{enroll,sign-in}/begin` only (a `context.route`). Uvicorn trusts that header from
127.0.0.1 by default, so Store keys its limiter on it. Sessions are still created from 127.0.0.1.
Tests of the limits themselves use `test.use({ spreadClientAddress: false })`. They share the
127.0.0.1 bucket with any other test that does the same, so put them in one file.

Sign-in is also limited per email, to 10 per 5 minutes. `signIn` the same account sparingly;
prefer a fresh `createInvitedUser` or `enrollAdmin` per test.
