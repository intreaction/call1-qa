# Call1 Store — operations console

Read-only panels on Store's own state, served by Store itself at `/console/`
(`call1/store/static.py`'s `console_site`). Never writes anything — "the plan says the console
can display these but not change them" (docs/SplitBuild.md).

**Demo mode badge.** The header shows a "Demo mode" pill whenever `GET /demo/status` answers
`demo: true` (`demo.ts`; localhost only, off by default — `CALL1_STORE_DEMO=1`,
`call1/store/auth/demo.py`). Badge only — persona sign-in and the account-menu persona switch are
Evaluate's (`frontend/src/apps/evaluate/api/demo.ts`); this console's own admin sign-in stays
passkey-only.

## Run it

```bash
npm --prefix frontend run build:store-console   # -> call1/store/static/console/
python -m call1.store serve                     # http://localhost:8010/console/
```

`build:store-console` also runs `copy:store-static` first, which copies the shared, hand-hosted
IBM Plex Sans files and the Call1 icon set from `call1/ui/static/` into `call1/store/static/`
(see `frontend/scripts/copy-shared-static.mjs`). Those live at the **static root**, not inside
`console/` — `frontend/src/index.css`'s `@font-face` rules and the header logo both reference
them by the same root-absolute path (`/fonts/...`, `/call1-dark-32.png`, ...) that the legacy app
uses, and `call1/store/static.py`'s Evaluate catch-all route (`GET /{path}`) resolves any
unprefixed path against the static root, so the same files serve every app without duplication.
Icons referenced from `store-console.html` itself use plain root-absolute `href`s for the same
reason — **not** `%BASE_URL%` — because `console_site`'s own roots are scoped to `static/console/`
only and would 404 on a bare `/console/favicon.ico` otherwise. Hashed JS/CSS bundles *do* need
`base: '/console/'` (set in `vite.store-console.config.ts`) since those live under
`static/console/assets/` and nowhere else.

For local iteration against a running Store: `npm --prefix frontend run dev` still serves the
legacy app; there's no `dev` script wired for this app yet (`vite --config
vite.store-console.config.ts` works directly if needed — proxy is already set to
`http://127.0.0.1:8010`).

## Panels

| Panel | Route(s) | Auth |
| --- | --- | --- |
| Health | `GET /store/v1/status` | anonymous |
| Contract coverage | none — reads the committed `call1/contracts/openapi.json` at build time (`coverage.ts`), grouped by the contract's own `tags` and `x-call1-stage` | anonymous |
| Process installations & service keys | `GET /store/v1/admin/installations`, `GET /store/v1/admin/service-keys` | admin session |
| Change feed | `GET /store/v1/changes` | admin session |
| Search embedder | `GET /store/v1/status/detail` (`search_embedder`, contract 1.2.0): the local model Store embeds search queries with, and whether it is not installed, installed, loaded, failed or the fake | admin session |

The admin-gated panels call `GET /store/v1/auth/session` once (`useSession.ts`). A 401/403 there
is treated as ordinary "signed out" data, not an error to surface — each panel shows
`SignedOutState` (a link to `/`, where Evaluate's passkey sign-in lives) or, for a signed-in
non-admin session, `InsufficientRoleState`. No panel ever fabricates data for an unreachable or
unauthorized state.

**Contract coverage is deliberately conservative.** It answers "what does the contract say is in
scope, by stage" from the generated `openapi.json` — it does not probe every route to see which
already has a real handler versus a `501 pending`. That distinction is Store's own routing table
(`call1/store/routing.py`: `OWNER_BY_OPERATION`, `DEFERRED`), which isn't exposed over HTTP yet
and this app doesn't add an endpoint for it. See the module doc in `coverage.ts`.

## What's not here yet

- No dark/light-aware favicon swap beyond the header logo (the browser tab icon is always the
  dark mark — a cosmetic gap, not a functional one).
- No live probe of which stage-2 routes have handlers registered vs. still 501 — see above.
- Admin panels don't paginate `page_token`-style; the change feed's "Load more" is the only
  pagination control, using `next_cursor`.

## Design system

Shares `frontend/src/index.css` and `tailwind.config.js` with Evaluate and the Process console
(docs/SplitBuild.md "Design principles"): the same Ink & Signal / Frost & Ink theme tokens, IBM
Plex Sans everywhere, and the lucide-react icon set. A bare `<code>`/`<kbd>`/`<pre>`/`<samp>` (this
app's Coverage panel) inherits IBM Plex Sans too, but that rule is scoped to `.call1-app` — this
app's `<body>` (`store-console.html`), Evaluate's and Process's, not the legacy app's — so a plain
legacy `npm run build` is not silently affected (judge condition 5, 2026-09-25). `font-mono` is
deliberately plain Tailwind (real monospace, not remapped) for the same reason: this app does not
use it. `frontend/e2e/design.spec.ts` checks all three apps for token and icon identity across
themes, the font rule, status-pill text labels, and — for this app specifically — that every
interactive control is reachable via `Tab` in DOM order and operable with `Enter`/`Space` alone.
