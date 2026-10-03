# End-to-end tests (Python)

These tests run the three apps the way an operator does. Store and Process are **real server
processes** (`python -m call1.store serve` and `python -m call1.process serve`) on free loopback
ports. Reviewer sessions come from the **real passkey ceremonies**, signed by a software
authenticator. Nothing is minted and nothing runs in-process. The browser suite in
[`frontend/e2e/`](../../frontend/e2e/README.md) uses the same stack through `serve_stack.py`.

```bash
.venv/bin/python -m pytest tests/e2e -q -p no:warnings            # fake handlers, a few seconds
.venv/bin/python -m pytest tests -m "not e2e" -q -p no:warnings   # everything else
CALL1_REAL_MODELS=1 .venv/bin/python -m pytest tests/e2e -m real_models -s -p no:warnings   # Apple Silicon
CALL1_E2E_KEEP=1 .venv/bin/python -m pytest tests/e2e -q            # keep data and logs for a post-mortem
```

Set `CALL1_E2E_ROOT` to override the default system temporary directory plus `call1-e2e`.

## Ground rules

- **Temp data lives under `<temporary-directory>/call1-e2e/<name>-<time>-<pid>-<hex>/`**, never in the
  repository. That covers Store data, Process config and data, logs and
  uploaded copies. The directory is deleted at teardown unless `CALL1_E2E_KEEP=1` is set.
- **Ports are chosen at runtime** and are never 8000, 8010 or 8020, because the user's live apps
  run there.
- **The stack is shared.** The `stack` fixture is one Store and Process pair per pytest session,
  so other tests' calls, accounts and rubrics are visible. Assert on the IDs your own test
  created, and never on totals or "the first row". For an empty Store, a scripted failure or an
  outage (`stop_store()`/`stop_process()`), use `stack_factory`, not the shared stack. As a guard,
  the `stack` fixture calls `ensure_running()` before each test, which restarts a server an earlier
  test stopped.
- **Do not change application code.** A test that fails because a feature is broken is a correct
  result. Keep it failing (no `skip`/`xfail`) and say what is wrong in the assertion message.
- Every test here is marked `e2e` automatically. Add `pytestmark = pytest.mark.e2e` anyway, for
  readers.
- **If pytest is killed with SIGTERM (or SIGKILL) mid-run, its stack's Store and Process
  subprocesses are not stopped** — `Stack.close()`, which sends them SIGTERM, never runs, so they
  keep listening on whatever loopback ports they picked. A run interrupted this way (Ctrl-C twice,
  a CI timeout, `kill` from another shell) can leave a Store/Process pair orphaned. To find and
  clean one up:
  ```bash
  ps aux | grep -E "call1\.(store|process) serve" | grep -v grep   # find the PIDs
  kill <store-pid> <process-pid>                                    # SIGTERM, same as a clean stop
  ```
  An orphaned pair's data lives under `<temporary-directory>/call1-e2e/<name>-<time>-<pid>-<hex>/`, so it is
  safe to remove that directory too once the processes are gone. This does not affect the user's
  live apps on 8000/8010/8020: stack ports are always chosen at runtime, never those three.

## What a stack is

`Stack.start()` runs these steps:

1. `python -m call1.store migrate` on a fresh data dir.
2. `python -m call1.store issue-service-key --installation e2e-<name> --primary-host --scope …`
   with the 12 scopes `call1.launch` uses: Store's default Process scopes plus `jobs:control`. The
   key is written to Process's config (mode 0600).
3. It merges these keys into the config: `port`, `data_dir`, `handlers`,
   `poll_interval_seconds: 0.2` and `shutdown_grace_seconds: 2`, plus anything you pass as
   `process_config`.
4. `python -m call1.process console-token`; the token is kept on `stack.console_token`.
5. `store serve`, then a wait for `GET /store/v1/status` to answer 200.
6. `process serve`, then a wait for `GET /process/api/health` to report `state == "running"`.

Both servers run with the stack dir as their working directory and `PYTHONPATH=<repo>`. Every
inherited `CALL1_*` variable is dropped except `CALL1_REAL_MODELS`, so a shell pointed at live data
cannot leak in. `<stackdir>/data/models` is a symlink to the repo's `data/models`, which the
pre-split code resolves relative to the working directory. Logs are in `<stackdir>/logs/`:
`store.log`, `process.log` and `cli.log` (service keys redacted). When a test fails, the tail of
each stack's logs is attached to the report.

Handlers:

- **Fake handlers** are the default: `CALL1_PROCESS_HANDLERS=fake`. A call settles in about a
  second.
- **Real models:** mark a test `@pytest.mark.real_models` and run it with `CALL1_REAL_MODELS=1`.
  The `stack` fixture then gives it a separate session stack with `CALL1_PROCESS_HANDLERS=real` and
  `CALL1_BACKEND=mlx`. Without `CALL1_REAL_MODELS=1`, those tests are skipped.

## Fixtures (`conftest.py`)

| Fixture | Scope | What you get |
|---|---|---|
| `stack` | function | The `Stack` for this test. That is `e2e_stack`, or `real_stack` when the test is marked `real_models` |
| `e2e_stack` | session | The shared fake-handler stack |
| `real_stack` | session | The shared real-model stack. It skips unless `CALL1_REAL_MODELS=1` |
| `stack_factory` | function | `stack_factory(**Stack kwargs) -> Stack`: a private, started stack, closed after the test |
| `admin_session` | function | `stack.admin()`: the stack's first admin, enrolled with a CLI `first_admin` setup code through the real ceremony (one per stack) |
| `reviewer_session` / `supervisor_session` | function | `stack.user(role)`: invited by the admin, enrolled from the invitation (one per role per stack) |
| `new_user` | function | `new_user(role="reviewer", email=None, display_name=None) -> StoreSession`: a **new** enrolled, signed-in account |
| `ingest` | function | `stack.ingest` (below) |
| `wait_until_settled` | function | `stack.wait_until_settled` |
| `store_get` / `store_post` | function | `stack.store_get` / `stack.store_post` |
| `process_get` / `process_post` | function | `stack.process_get` / `stack.process_post` |

`from .stack import Stack, StoreSession, StackError, unique_email, sample_path` gives you the
classes and helpers. The module has no pytest dependency.

## `Stack` API (`stack.py`)

Constructor (all keyword arguments; `stack_factory` passes them through):

| Argument | Default | Meaning |
|---|---|---|
| `name` | `"stack"` | Directory prefix and installation label |
| `handlers` | `"fake"` | `"fake"` or `"real"` |
| `real_models` | `False` | Real handlers plus `CALL1_BACKEND=mlx` (Apple Silicon) |
| `fake_behavior` | none | `CALL1_FAKE_BEHAVIOR`: `{"asr": ["fail:<code>"], "qa_criterion:REG-01": ["needs_review"], "acoustic_tone": ["hold:5"]}`, keyed by job type (an unknown key such as `tone` is silently ignored). Actions are listed in `call1/process/handlers/fake.py` |
| `store_parameters` | none | `CALL1_STORE_PARAMETERS`: `ContractParameters` overrides such as lease seconds |
| `process_config` | none | Extra keys for Process's config, such as `{"slots": {"cpu_io": 1}}` or `{"stages": {"summary": false}}` |
| `store_env` / `process_env` | none | Extra environment for one server, such as `{"CALL1_STORE_MAINTENANCE_SECONDS": "1"}` |
| `with_process` | `True` | `False`: Store only (no console token, no worker) |
| `spread_clients` | on | Per-session client address for the auth rate limiter (below). `CALL1_E2E_SPREAD_CLIENTS=0` turns it off |
| `keep` | `CALL1_E2E_KEEP=1` | Keep the directory after `close()` |

Attributes: `store_url` (`http://localhost:<port>`, the WebAuthn origin; use `localhost`, not
`127.0.0.1`), `process_url` (`http://127.0.0.1:<port>`), `evaluate_url`, `process_console_url`
(with `#console_token=`), `store_port`, `process_port`, `console_token`, `service_key`,
`service_key_id`, `installation_id`, `dir`, `store_data`, `process_config_path`, `logs_dir`,
`handlers`, `real_models`.

| Method | What it does |
|---|---|
| `ingest(source="call_01_compliant", *, unique=True, agent_id=None, agent_channel=None, external_call_ref=None, filename=None, content_type="audio/wav", expect_status=201) -> dict` | `POST /process/api/recordings` (multipart, console token). `source` is a sample name from `sample_audio/`, with or without `.wav`, or a path. `unique=True` uploads a copy whose first four samples are nudged, which gives it a new SHA-256. Process identifies a recording by its digest, so without that a repeat returns the **same** conversation. Returns Process's receipt `{conversation_id, call_id, graph_id, conversation_created, graph_created, jobs, evaluate_url}`. `expect_status=None` returns the body of any status, such as a 415 |
| `wait_until_settled(call_or_conversation_id, *, timeout=None, interval=0.25) -> dict` | Polls Store's `GET /conversations/{id}/progress` (service key) until `settled`, and returns the `JobGroupProgress`. Timeout is 60 s with fake handlers and 900 s with real ones. On timeout it raises `StackError` with the Process log tail |
| `progress(id)`, `conversation_id(call_id)` | One progress read; call ID to conversation ID (from the ingest receipt, else Process's ledger) |
| `store_get(path, *, session=None, **httpx_kw)`, `store_post(path, json=None, *, session=None, idempotency_key=None, **kw)`, `store_request(method, path, …)` | `/store/v1`. `session` is a `StoreSession` (cookie, plus CSRF on writes), `"service"` (Process's key) or `None` (anonymous). The path may omit `/store/v1` |
| `process_get(path, *, token=True, **kw)`, `process_post(path, json=None, *, token=True, **kw)`, `process_request(...)` | `/process/api` with `X-Call1-Console-Token`. `token=False` sends none; a string sends that token. The path may omit `/process/api` |
| `service_headers()` | `{"Authorization": "Bearer c1sk_…"}` |
| `admin()` | The first admin `StoreSession`, cached |
| `user(role="reviewer", *, email=None, display_name=None, cached=True)` | Invited and enrolled. Cached per role unless `cached=False` or an `email` is given |
| `invite(email, role="reviewer", *, display_name=None, **extra) -> token` | Admin `createInvitation`. Returns the token from the `…/enroll#<token>` URL |
| `new_session(email=None, *, authenticator=None) -> StoreSession` | A fresh browser-like client that is **not** enrolled, for hand-driven ceremonies |
| `setup_code(email, display_name="E2E Admin", *, purpose="first_admin", target_account_id=None) -> str` | `python -m call1.store setup-code`. For `break_glass`, pass `target_account_id` to re-enroll |
| `store_cli(*args, check=True)`, `process_cli(*args, check=True)` | Any host command against this stack, such as `store_cli("issue-service-key", "--installation", "x", "--print-token")`. Returns `CompletedProcess` |
| `stop_store()` / `start_store()` | Stop Store (Process keeps running and sees it unreachable), then restart it on the same port and data |
| `stop_process()` / `start_process(wait_running=True)` | The same for Process (its spool and ledger persist) |
| `ensure_running()` | Restart whichever server is not running |
| `log_tail(name, lines=80)` | `store`, `process` or `cli` |
| `wait_for(predicate, *, timeout=20, interval=0.2, what="…")` | Polls until the predicate returns a truthy value, and returns it |
| `info()` | A JSON-able dict of all of the above. This is what `serve_stack.py` prints |

## `StoreSession` (one reviewer's browser, over HTTP)

Each session has its own `httpx.Client` (cookie jar) and its own `SoftAuthenticator`, the Store
tests' software authenticator from `tests/store/auth_softauthn.py`, reused as is. It signs P-256
credentials with `none` attestation.

| Member | What it does |
|---|---|
| `enroll(*, setup_code=None, invitation_token=None, nickname=None, check=True, **register_kw)` | `/auth/enroll/begin` → authenticator → `/auth/enroll/finish`. It is signed in afterwards. `register_kw` is passed to `SoftAuthenticator.register`, which accepts `origin=`, `rp_id=`, `user_verified=False` and `challenge=` for misbehaving on purpose. With `check=False` the failing response is returned instead of raising |
| `sign_in(*, check=True, email=None, **assert_kw)` | Account-first sign-in. `assert_kw` accepts `origin=`, `tamper=True`, `sign_count=` and `user_verified=False` |
| `sign_out()`, `refresh()` | `POST /auth/sign-out`; re-read `GET /auth/session` |
| `get/post/put/patch/delete(path, json=None, *, idempotency_key=None, csrf=True, headers=None, **httpx_kw)` | Writes send `X-Call1-CSRF` and `Origin: <store_url>`. `idempotency_key=True` sends a fresh `Idempotency-Key`; a string repeats one. `csrf=False` leaves out the header, for CSRF tests |
| `email`, `session` (the `SessionInfo` dict), `account` (from enrollment), `account_id`, `role`, `permissions`, `csrf_token`, `cookies`, `authenticator`, `http`, `client_address` | State |

**Auth rate limits.** Store limits enrollment begin to 10 a minute and sign-in begin to 30 a minute
per client address, and sign-in to 10 per 5 minutes per email. Every test connects from
127.0.0.1. So that unrelated tests on the shared stack don't starve each other, each session sends
its own synthetic `X-Forwarded-For` (`client_address`, such as `10.145.0.3`) on the two `begin`
calls only. Uvicorn trusts that header from 127.0.0.1 by default, so Store keys its limiter on it.
A 429 on a begin call is waited out once, using `Retry-After`. For tests of the limits
themselves, use `stack.new_session(...)` and set `session.client_address = None` and
`session.retry_rate_limited = False`, or use `stack_factory(spread_clients=False)`. The per-email
limit still applies, so don't sign the cached `admin_session` in more than about 10 times in 5
minutes.

## `serve_stack.py`

```bash
.venv/bin/python tests/e2e/serve_stack.py [--handlers fake|real] [--real-models] \
    [--fake-behavior JSON] [--store-parameters JSON] [--process-config JSON] [--name N] [--info-file PATH] [--keep]
```

The script prints `Stack.info()` plus `"ready": true` as one JSON line, then keeps the stack alive
until stdin closes, SIGTERM or Ctrl-C. On a start-up failure it prints `{"ready": false, "error":
…}` and exits 1. The Playwright global setup runs it. It is also a quick way to get a throwaway
stack by hand.

## Known facts worth a test

- An `X-Forwarded-For` sent from loopback changes the address Store's auth rate limiter sees. This
  is uvicorn's default `forwarded_allow_ips=127.0.0.1`. The harness relies on it (above).
