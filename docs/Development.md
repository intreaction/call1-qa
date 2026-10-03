# Development

Use Python 3.12, Node.js 22.12+ for frontend work, and FFmpeg on PATH. Install `requirements.txt` (or `requirements-mlx.txt` for Apple Silicon inference) and `requirements-dev.txt` into a virtual environment.

## Frontend

```sh
cd frontend
npm ci
npm run typecheck
npm run build
npm run typecheck:e2e
```

The default build produces all three current apps. `npm run dev` starts Evaluate's development server; follow the origin configuration in `vite.evaluate.config.ts` when connecting it to Store. Shared fonts and icons come from `frontend/public/`, not a legacy application's output.

## Automated tests

From the repository root:

```sh
python scripts/generate_test_audio.py
python -m pytest tests --ignore=tests/e2e -q
python -m pytest tests/e2e -q
```

The generator creates five deterministic non-speech PCM fixtures with the durations/channels expected by fake-handler tests. It refuses to overwrite pre-existing audio. Generated WAVs and their marker are ignored; no Apple System Voices recordings or external speech service is needed. Fake handlers provide the scripted transcripts. These tests check workflow behavior, not ASR or QA model accuracy.

Browser checks use a separate Store/Process stack on spare ports:

```sh
cd frontend
CALL1_E2E_PYTHON="$(pwd)/../.venv/bin/python" npm run test:e2e
```

The browser suite uses installed Google Chrome locally. To use Playwright Chromium instead:

```sh
npx playwright install chromium
CALL1_E2E_BROWSER=chromium CALL1_E2E_PYTHON="$(pwd)/../.venv/bin/python" npm run test:e2e
```

Generated fixtures must exist before starting the suite. `CALL1_E2E_ROOT` sets the temporary data/report directory; by default it is `call1-e2e` under the operating system's temporary directory. No repository data is used.

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests, and manual dispatch. Its two Linux jobs run the complete Python suite (real-model checks stay opt-in), TypeScript checks, all three frontend builds, and the browser smoke, metrics, demo-call, and model-pipeline specs in both themes. Actions are pinned to commits and the workflow has read-only repository permissions. CPU dependencies and scripted handlers require no model weights or external inference credentials.

The browser job is a selected smoke suite, not every Playwright spec. Use `npm run test:e2e` locally for the full browser suite. Test reports are retained for seven days; runtime credentials and uploaded recordings must not be committed.

Real-model tests are opt-in with `CALL1_REAL_MODELS=1`, require appropriate weights and real recordings matching their expected transcript fixtures, and must not be run on generated tones. The removed speech recordings are not included in this source snapshot; real-model qualification needs a separately cleared fixture set.

## Contracts and credentials

`npm run contracts:types` in `frontend/` regenerates Store client types from the checked-in OpenAPI contract. Keep API changes synchronized with the Python contracts and client types.

Run Gitleaks with `.gitleaks.toml` against the exact source tree and full publication history. Its Call1-specific rules supplement the default secret rules. Review findings individually; do not allowlist whole test directories. Public source history must not contain excluded private files or recordings.

## Dependency audit scope

CI rejects moderate or higher advisories in frontend runtime dependencies (`npm audit --omit=dev`). The October 2, 2026 sweep found none after updating Vite and its React plugin. A full development-dependency audit still reports the Tailwind 3 glob/watcher chain (braces stack-exhaustion advisory); the registry offers no compatible patched braces release. Those packages run during development/builds, not in the served application. Moving to Tailwind 4 is a separate stylesheet migration; do not supply untrusted glob patterns to build tooling.
