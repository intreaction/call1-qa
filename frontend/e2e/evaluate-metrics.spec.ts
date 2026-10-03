// evaluate_ui e7: MetricsView (#/metrics) — executive, per-rubric (with picker), review-agreement
// (supervisor). Date-range query params filter results (sent as aware timestamps). Known issue (c):
// "Audio audited" (formerly "Hours audited") must be nonzero for a nonzero total, not display "0.0" (short totals show minutes).
import { test, expect } from './fixtures';

test('e7: executive metrics reflect an ingested call; per-rubric and review-agreement (supervisor) render', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const supervisor = await createInvitedUser('supervisor');
  const before = await (await supervisor.page.request.get('/store/v1/metrics/executive')).json();

  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-e7-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);

  await supervisor.page.goto('/#/metrics');
  await expect(supervisor.page.getByText('Executive summary')).toBeVisible();
  await expect(supervisor.page.getByText('Calls audited')).toBeVisible();

  // Per-rubric: pick the seeded default rubric (its <option value> is the rubric_id) and confirm
  // the table renders without crashing.
  await supervisor.page.getByLabel('Rubric').selectOption({ value: 'call1_standard_v2' });
  await expect(supervisor.page.getByText('Calls evaluated')).toBeVisible();

  // Review agreement is supervisor-only and must not show the "your role" gate for a supervisor.
  await expect(supervisor.page.getByText('Your role does not include review-agreement metrics.')).toHaveCount(0);
  await expect(supervisor.page.getByText('Calls reviewed')).toBeVisible();

  // Known issue (c): "Hours audited" reflects this run's audited call duration and is not the
  // literal string "0.0" once at least one call has been QA'd (via the API, immune to display
  // rounding, so this checks the acceptance criterion literally).
  const after = await (await supervisor.page.request.get('/store/v1/metrics/executive')).json();
  expect(after.total_hours_audited).toBeGreaterThan(before.total_hours_audited);
  // The DISPLAYED value is checked deterministically in the known-issue (c) test below: on this
  // shared stack other tests' calls accumulate, so whether the tile rounds to "0.0" here depends on
  // run order and is not a stable check.
});

test('e7: date-range query params filter results (the date picker sends aware timestamps, not a bare date)', async ({
  createInvitedUser,
}) => {
  const supervisor = await createInvitedUser('supervisor');
  await supervisor.page.goto('/#/metrics');
  await expect(supervisor.page.getByText('Executive summary')).toBeVisible();
  await expect(supervisor.page.getByRole('alert')).toHaveCount(0);

  const requestPromise = supervisor.page.waitForRequest((r) => r.url().includes('/store/v1/metrics/executive?') && r.url().includes('start='));
  await supervisor.page.getByLabel('From', { exact: true }).fill('2024-01-01');
  const request = await requestPromise;
  const response = await request.response();

  // Acceptance: the date-range filter narrows the result. Root cause when this fails: Store's
  // MetricsQuery.start/end are AwareDatetime (call1/contracts/common.py Timestamp — RFC 3339 with
  // an explicit offset), but MetricsView's <input type="date"> sends a bare "YYYY-MM-DD" straight
  // through as the query value (frontend/src/apps/evaluate/views/MetricsView.tsx `range = { start:
  // start || null, ... }`), which Store's contract validation rejects.
  expect(response?.status(), 'Store answered the date-filtered request').toBeLessThan(300);
  // Start of that day in the browser's zone, as RFC 3339 with an explicit offset.
  expect(new URL(request.url()).searchParams.get('start')).toMatch(/^2024-01-01T00:00:00(Z|[+-]\d{2}:\d{2})$/);
  await expect(supervisor.page.getByRole('alert').first()).toHaveCount(0);
  await expect(supervisor.page.getByText('Calls audited')).toBeVisible();

  // To covers the whole chosen day: Store's `end` is exclusive, so it is the start of the next day.
  const endRequest = supervisor.page.waitForRequest((r) => r.url().includes('/store/v1/metrics/executive?') && r.url().includes('end='));
  await supervisor.page.getByLabel('To', { exact: true }).fill('2024-01-31');
  const endReq = await endRequest;
  expect((await endReq.response())?.status()).toBeLessThan(300);
  expect(new URL(endReq.url()).searchParams.get('end')).toMatch(/^2024-02-01T00:00:00(Z|[+-]\d{2}:\d{2})$/);
  await expect(supervisor.page.getByRole('alert')).toHaveCount(0);

  // The filter narrows the result: nothing in the stack was created in January 2024.
  const tile = supervisor.page.getByText('Calls audited', { exact: true }).locator('xpath=..');
  await expect(tile.locator('div').first()).toHaveText('0');
});

test('e7 (known issue c): one short audited call (0.01 h from Store) is not displayed as "Hours audited 0.0"', async ({
  createInvitedUser,
}) => {
  const supervisor = await createInvitedUser('supervisor');
  // The shared stack holds many tests' calls, so its real total is well over 3 minutes. To show the
  // single-call case deterministically, take Store's REAL executive response and set only the two
  // totals to what Store reports for exactly one call_01_compliant (44.01 s):
  // round(44.01 / 3600, 2) = 0.01 h (call1/store/results/metrics.py; the API side is covered by
  // tests/e2e/test_store_api_results.py::test_q15_*). Everything else is Store's own answer.
  await supervisor.page.route('**/store/v1/metrics/executive**', async (route) => {
    // The page may close while a stubbed request is in flight (test teardown): drop it then.
    const response = await route.fetch().catch(() => null);
    const body = response ? await response.json().catch(() => null) : null;
    if (!response || body === null) return;
    body.total_audited_calls = 1;
    body.total_hours_audited = 0.01;
    await route.fulfill({ response, json: body });
  });
  await supervisor.page.goto('/#/metrics');
  const tile = supervisor.page.getByText('Audio audited', { exact: true }).locator('xpath=..');
  await expect(tile).toBeVisible();
  // Root cause when this fails: MetricsView.tsx renders total_hours_audited.toFixed(1), so any
  // total under 0.05 h (3 minutes of audio) reads "0.0" although Store reports a nonzero value.
  await expect(tile.locator('div').first(), 'Audio audited shows a nonzero value for 0.01 h audited').not.toHaveText(/^0(\.0+)?$/);
  // 0.01 h is 36 s: under an hour the tile shows minutes, rounded.
  await expect(tile.locator('div').first()).toHaveText('1 min');
});

// --- Contact Signals v2 addition (docs/ContactSignalsV2.md section 10.3 "Metrics" and section 17
// "evaluate-metrics": "Top caller needs ranking updates after a backfill; precision after
// feedback"). Written independently from the F4 implementation, from the design doc and the
// contract (metrics.py SignalMetrics/SignalCount, "precision is null under 5 judged hits").
test('e7 (signals): the Signals card renders a category table and an alerts table, and "Top caller needs" updates after a rescore backfill', async ({
  createInvitedUser,
  adminApi,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const supervisor = await createInvitedUser('supervisor');
  const before = await (await supervisor.page.request.get('/store/v1/metrics/signals')).json();

  // F5: the backfill window is just this test's call (the shared stack's other calls keep their
  // published signals, so parallel specs never see them go stale).
  const windowStart = new Date(Date.now() - 1_000).toISOString();
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-e7-signals-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  await adminApi.json('POST', '/signals/backfills', {
    data: { mode: 'rescore', rescore_signals: true, created_after: windowStart, created_before: new Date().toISOString(), max_calls: 5 },
    idempotencyKey: true,
  });

  await supervisor.page.goto('/#/metrics');
  await expect(supervisor.page.getByRole('heading', { name: 'Signals', exact: true })).toBeVisible();
  await expect(supervisor.page.getByText('Top caller needs', { exact: true })).toBeVisible();
  const categories = supervisor.page.getByRole('table', { name: 'Signal categories' });
  await expect(categories).toBeVisible();
  await expect(categories.getByRole('columnheader', { name: 'Category precision', exact: true })).toBeVisible();
  await expect(categories.getByRole('columnheader', { name: 'Subcategory precision', exact: true })).toBeVisible();
  await expect(supervisor.page.getByText('Alerts', { exact: true })).toBeVisible();

  await expect
    .poll(async () => {
      const after = await (await supervisor.page.request.get('/store/v1/metrics/signals')).json();
      return after.calls_scored;
    }, { timeout: 20_000 })
    .toBeGreaterThan(before.calls_scored);
});

test('e7 (signals): category precision shows "Needs reviewer feedback" under five judged hits and a number at five or more', async ({ createInvitedUser }) => {
  const supervisor = await createInvitedUser('supervisor');

  // Under 5: Store's own rule (metrics.py _precision_needs_judgements — precision_pct is null).
  await supervisor.page.route('**/store/v1/metrics/signals**', async (route) => {
    // The page may close while a stubbed request is in flight (test teardown): drop it then.
    const response = await route.fetch().catch(() => null);
    const body = response ? await response.json().catch(() => null) : null;
    if (!response || body === null) return;
    body.categories = [{ category_id: 'intent', name: 'Caller objective', hit_rate_pct: 40, hits: 4, confirmed: 3, dismissed: 1, precision_pct: null, subcategory_precision_pct: null }];
    await route.fulfill({ response, json: body });
  });
  await supervisor.page.goto('/#/metrics');
  // The category row, by its drill-down button: alert rules other specs name "Caller objective …"
  // have rows of their own in the alerts table.
  const categoryRow = () => supervisor.page.getByRole('row').filter({ has: supervisor.page.getByRole('button', { name: 'Caller objective', exact: true }) });
  const row = categoryRow();
  // Under five judged hits the cell says what it needs instead of a bare dash.
  await expect(row).toContainText('Needs reviewer feedback');

  // At or above 5: a real number.
  await supervisor.page.unrouteAll({ behavior: 'wait' });
  await supervisor.page.route('**/store/v1/metrics/signals**', async (route) => {
    // The page may close while a stubbed request is in flight (test teardown): drop it then.
    const response = await route.fetch().catch(() => null);
    const body = response ? await response.json().catch(() => null) : null;
    if (!response || body === null) return;
    body.categories = [{ category_id: 'intent', name: 'Caller objective', hit_rate_pct: 40, hits: 6, confirmed: 5, dismissed: 1, precision_pct: 83.3, subcategory_precision_pct: 80 }];
    await route.fulfill({ response, json: body });
  });
  await supervisor.page.reload();
  const row2 = categoryRow();
  await expect(row2).toContainText('83');
});
