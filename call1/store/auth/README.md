# Store auth area

Passkey-only reviewer sign-in (`webauthn` 3.x), server-side sessions, admin identity, and Process
installations and service keys. This area implements the 31 auth operations in
`call1/store/routing.py` (`_AUTH`), the `AuthBackend` the principal guard calls on every request,
the two host commands and `api.py` for the other areas. Tables are in `migrations/030_auth.sql`.
There is no password endpoint, no reset and no fallback. The one exception is **demo mode**
(`demo.py`), off unless `CALL1_STORE_DEMO=1` and refused outside localhost dev mode, which signs in
three fixed demo personas for class presentations (Store README, "Demo mode").

## First run

```bash
python -m call1.store setup-code --email it@example.com --display-name "IT Admin"   # prints a one-time code
python -m call1.store serve                                                          # open http://localhost:8010/ and enroll with the code
python -m call1.store issue-service-key --installation mac-mini                      # writes data/process/config.json (0600)
```

After that, an admin invites reviewers from Evaluate (`POST /admin/invitations`). The response
carries `invitation_url` (`<store>/enroll#<token>`) once. The admin delivers it out of band.

## Files

| File | What it does |
|---|---|
| `routes.py`, `routes_self.py`, `routes_admin.py` | Handlers: ceremonies and own sessions/authenticators; admin identity |
| `backend.py` | `SqliteAuthBackend`: service-key and session lookups, CSRF check, TEST-ONLY minting |
| `passkeys.py` | WebAuthn options, single-use ceremonies, verification (origin, RP ID, UV, signature, counter) |
| `sessions.py` | Session creation, the guard lookup (idle slide, absolute cap), `SessionInfo`, listings |
| `identity.py` | Operations shared by routes and host commands (setup codes, installations, keys, last-admin rule) |
| `crypto.py` | Area keys, CSRF derivation, decoys, setup-code format |
| `ratelimit.py` | In-process sliding windows for sign-in and enrollment begin |
| `records.py` | Row mapping and queries; only this area imports it |
| `demo.py` | DEMO MODE only (`CALL1_STORE_DEMO`, dev mode only): `/demo/status`, `/demo/sign-in` persona sessions; see the Store README |
| `cli.py`, `api.py` | Host commands; the functions other areas call |

## Rules this area enforces

- **Ceremonies.** Challenges are 32 random bytes, single-use and expire after
  `webauthn_challenge_lifetime_seconds`. Finish consumes the ceremony in its own committed
  transaction before verifying, so a failed or replayed finish cannot reuse it. User verification
  is required. The RP ID is the Store hostname (`localhost` in dev mode). A foreign `Origin` header
  or `clientDataJSON` origin, or a cross-origin iframe, gets 403 `origin_not_allowed`. Every other
  failure is 401 `webauthn_verification_failed` with a `details.reason`.
- **Sign-in is account-first.** For an unknown, disabled or re-invite-pending email, begin returns one
  or two decoy descriptors, HMAC(area key, email), that stay stable per email. Finish accepts only a
  live credential of the account bound at begin. An account disabled between begin and finish gets
  403 `account_disabled`.
- **Sessions.** The cookie value and CSRF token are stored as SHA-256 only. The CSRF token is
  HMAC(area key, cookie), so `GET /auth/session` can return it again. Idle expiry slides (at most
  one write a minute) up to the absolute cap. Role and status are re-read on every request.
- **Accounts.** An account is created when its invitation or setup code is redeemed.
  `pending_enrollment` is therefore unused. Disabling an account revokes its sessions and pending
  re-invites. A re-invite revokes every credential and session when it is issued. Break-glass on a
  named account revokes them when the code is redeemed. The last active admin who can still sign
  in cannot be demoted, disabled or re-invited (409 `last_admin`). An admin can always revoke a
  lost key. Revoking an account's last key makes it `reinvite_required`.
- **Keys.** Store keeps `sha256(token)` only. At most one active installation is the primary host,
  enforced by the route and the host command. The TEST-ONLY mint does not enforce it. A rotation
  copies the installation, scopes, label and expiry, so it never widens access. `grace_seconds: 0`
  revokes the old key at once. Retiring an installation revokes all its keys.
- **Rate limits** (429 `rate_limited`, `Retry-After`). Sign-in begin: 30 per minute per client
  address, and 10 per 5 minutes per email, applied the same way whether or not the account exists.
  Enrollment begin: 10 per minute per client address.

## Deviations and gaps (raised, contract not patched)

- **Dev-mode cookie** `call1_session` without `Secure` (see the Store README). The RP ID
  `localhost` and the `http://localhost` origins appear in ceremony options. `RelyingParty.id` and
  `rpId` have no validator, so no devmode shim is needed.
- **`smtp_relay` invitations** need the customer's relay in admin state, and admin state is
  deferred. They get 409 `conflict` with `reason: smtp_relay_not_configured`.
- **`BreakGlassRecord.audit_event_id`** is the `break_glass_used` event once the code is
  redeemed. Before that it is the `setup_code_issued` event.
- **Reviewer profiles** (`/admin/reviewer-profiles`) belong to the results area in `routing.py`,
  not to this area.
- **Host-command refusals** raise `cli.HostCommandRefused`. It is a `StoreError` and also a
  `ConfigError`, so `__main__` prints `configuration: <reason>` and exits 2. A cleaner message
  needs `__main__` to catch `StoreError`, which is a core change.
- **Rate limits are in memory**, so a restart resets them. They are per Store process, which is
  fine for the single-process `serve`.

## Tests

`tests/store/test_auth_ceremonies.py`, `test_auth_admin.py` and `test_auth_cli.py` run full
register and sign-in ceremonies with `tests/store/auth_softauthn.py`, a software P-256
authenticator (attestation `none`) that emits the same JSON as `@simplewebauthn/browser`.
`tests/store/auth_flows.py` has the bootstrap, invite and sign-in helpers.
