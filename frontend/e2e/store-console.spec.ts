/**
 * Tests for the "consoles" inventory area (Store console + Process console UIs) that the existing
 * Python API tests do not already cover end to end: the actual built pages, in a real browser,
 * against the real servers. One test (or tight group) per feature id from the computed task's
 * inventory (`consoles.features[].id`), named with the id in the test title.
 *
 * c5 and c6 need a real Store outage / a Process service key without `jobs:control`, which the
 * shared stack (one Store+Process pair for the whole run, in every other test in this suite)
 * cannot do without breaking every other test. They use a private, disposable stack started with
 * `tests/e2e/outage_stack.py` via `./outageStack.ts` — new test infrastructure alongside the
 * shared harness, not a change to it.
 */
import path from 'node:path';
import { test, expect } from './fixtures';
import { OutageStack, ingestOn, waitUntilSettledOn } from './outageStack';
import { uniqueWav } from './harness';
import fs from 'node:fs';

// ---------------------------------------------------------------------------------------------
// c1 — Store console served at /console/
// ---------------------------------------------------------------------------------------------

test('c1: Store console serves real panels at /console/, never a placeholder', async ({ page, storeURL, request }) => {
  const liveStatus = await (await request.get(`${storeURL}/store/v1/status`)).json();

  await page.goto(`${storeURL}/console/`);
  await expect(page.getByText('Call1 Store', { exact: true })).toBeVisible();
  await expect(page.getByText('operations console')).toBeVisible();

  // The build-honesty rule (c9) applies here too: this must be the real console, not the
  // static.py placeholder page (`_placeholder()` renders the literal heading "Not built yet").
  await expect(page.getByRole('heading', { name: 'Not built yet' })).toHaveCount(0);

  // Health panel: real, live data from GET /store/v1/status — not fabricated.
  await expect(page.getByRole('heading', { name: 'Health' })).toBeVisible();
  await expect(page.getByText('Reachable', { exact: true })).toBeVisible();
  const healthPanel = page.locator('section', { has: page.getByRole('heading', { name: 'Health' }) });
  await expect(healthPanel.getByText('Contract version')).toBeVisible();
  const shownContractVersion = (await healthPanel.locator('dt:text-is("Contract version") + dd').innerText()).trim();
  expect(shownContractVersion.length).toBeGreaterThan(0);
  const liveContractVersion = liveStatus.contract_version ?? liveStatus.contract?.contract_version;
  if (liveContractVersion) expect(shownContractVersion).toBe(String(liveContractVersion));

  // Known contract gap (Store README "Known contract gaps"): TLS is Stage 4, reported honestly as
  // untrusted rather than a fabricated "OK".
  await expect(healthPanel.getByText(/Untrusted \(Stage 4 not built\)/)).toBeVisible();

  // Contract coverage panel: also real (reads the committed openapi.json at build time).
  const coveragePanel = page.locator('section', { has: page.getByRole('heading', { name: 'Contract coverage' }) });
  await expect(coveragePanel.getByText(/^\d+ routes$/)).toBeVisible();
});

// ---------------------------------------------------------------------------------------------
// c2 — Process console served at / on port 8020 (loopback)
// ---------------------------------------------------------------------------------------------

test('c2: Process console serves the real operator app on its loopback port', async ({ page, processConsoleURL, processURL }) => {
  expect(processURL.startsWith('http://127.0.0.1:')).toBe(true);

  await page.goto(processConsoleURL);
  await expect(page.getByText('Call1 Process', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Not built yet' })).toHaveCount(0);

  await expect(page.getByRole('button', { name: 'Overview' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Import' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Pipeline' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Models' })).toBeVisible();

  // The console token in the URL fragment was accepted and the app is actually talking to
  // Process's real /process/api/session.
  await expect(page.getByText('Console connected', { exact: true })).toBeVisible();
});

// ---------------------------------------------------------------------------------------------
// c3 — Overview screen surfaces every documented field, correctly labeled, with real values
// ---------------------------------------------------------------------------------------------

test('c3: Overview surfaces every GET /process/api/overview field, correctly labeled', async (
  { page, processConsoleURL, processApi, ingestSample, waitUntilSettled },
  testInfo,
) => {
  const agent = `pw-c3-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);

  const overview = await (await processApi.get('/overview')).json();

  await page.goto(processConsoleURL);
  await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();

  // Store connection card
  const storeCard = page.locator('section', { has: page.getByRole('heading', { name: 'Store connection' }) });
  await expect(storeCard.getByText(overview.store.url)).toBeVisible();
  await expect(storeCard.locator('dt:text-is("Installation") + dd')).toHaveText(overview.installation_id ?? '—');
  await expect(storeCard.locator('dt:text-is("Bind") + dd')).toHaveText(overview.bind);
  await expect(storeCard.locator('dt:text-is("Contract") + dd')).toContainText(overview.store.contract_version);

  // Worker card — real counters, not zeros, since a call actually settled above. The shared stack
  // keeps processing other tests' calls (parallel workers), so these monotonic counters can move
  // between any single API read and the console's 5 s refresh. The page loaded after `overview`
  // was read, so each shown value is bounded below by that snapshot and above by an API read taken
  // after the UI read, instead of being pinned to one snapshot.
  const workerCard = page.locator('section', { has: page.getByRole('heading', { name: 'Worker', exact: true }) });
  const expectLiveCounter = async (card: typeof workerCard, label: string, read: (o: any) => unknown) => {
    await expect(async () => {
      const shown = Number((await card.locator(`dt:text-is("${label}") + dd`).textContent())?.trim());
      const after = Number(read(await (await processApi.get('/overview')).json()) ?? 0);
      expect(Number.isInteger(shown), `${label} shows a whole number`).toBe(true);
      expect(shown, `${label} is at least the API value read before the page loaded`).toBeGreaterThanOrEqual(Number(read(overview) ?? 0));
      expect(shown, `${label} is no more than the API value read after it`).toBeLessThanOrEqual(after);
    }).toPass({ timeout: 20_000, intervals: [500, 1_000, 2_000] });
  };
  for (const [label, key] of [
    ['Claimed', 'claimed'],
    ['Succeeded', 'succeeded'],
    ['Failed', 'failed'],
    ['Released', 'released'],
    ['Lost', 'lost'],
  ] as const) {
    await expectLiveCounter(workerCard, label, (o) => o.worker.stats[key]);
  }
  // Spooled is a gauge (completions waiting to reach Store), not a counter: it can rise and fall
  // between reads, so the shown value must match a fresh API read, retried until the two agree.
  await expect(async () => {
    const shown = (await workerCard.locator('dt:text-is("Spooled") + dd').textContent())?.trim();
    const now = (await (await processApi.get('/overview')).json()).worker.spooled;
    expect(shown).toBe(String(now));
  }).toPass({ timeout: 20_000, intervals: [500, 1_000, 2_000] });
  expect(Number(overview.worker.stats.succeeded)).toBeGreaterThan(0);

  // Resource slots table — every pool from the JSON has a row.
  const slotsCard = page.locator('section', { has: page.getByRole('heading', { name: 'Resource slots' }) });
  const pools: Array<{ pool: string }> = overview.worker?.pools ?? overview.slots;
  for (const p of pools) await expect(slotsCard.getByRole('cell', { name: p.pool })).toBeVisible();

  // Handlers card
  const handlersCard = page.locator('section', { has: page.getByRole('heading', { name: 'Handlers' }) });
  await expect(handlersCard.getByText(`Mode: ${overview.handlers.mode}`)).toBeVisible();

  // Catalog & reanalysis card
  const catalogCard = page.locator('section', { has: page.getByRole('heading', { name: 'Catalog & reanalysis' }) });
  await expect(catalogCard.locator('dt:text-is("Catalog version") + dd')).toHaveText(overview.catalog.version);
  await expectLiveCounter(catalogCard, 'Reanalysis handled', (o) => o.reanalysis?.handled);

  // This installation card
  const installCard = page.locator('section', { has: page.getByRole('heading', { name: 'This installation' }) });
  await expectLiveCounter(installCard, 'Conversations ingested', (o) => o.conversations);
  expect(Number(overview.conversations)).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------------------------
// c4 — KNOWN ISSUE (d): worker counters render as aligned label/value pairs
// ---------------------------------------------------------------------------------------------

test('c4: Overview worker counters render as aligned label/value pairs, not misaligned text', async (
  { page, processConsoleURL, ingestSample, waitUntilSettled },
  testInfo,
) => {
  await page.setViewportSize({ width: 1280, height: 900 }); // >= the `sm` breakpoint (640px) this bug is specific to
  const agent = `pw-c4-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);

  await page.goto(processConsoleURL);
  const workerCard = page.locator('section', { has: page.getByRole('heading', { name: 'Worker', exact: true }) });
  await expect(workerCard.locator('dt:text-is("Claimed")')).toBeVisible();

  const dts = workerCard.locator('dl dt');
  const dds = workerCard.locator('dl dd');
  const n = await dts.count();
  expect(n).toBe(await dds.count());
  expect(n).toBeGreaterThanOrEqual(6); // Claimed, Succeeded, Failed, Released, Lost, Spooled

  const misaligned: string[] = [];
  for (let i = 0; i < n; i++) {
    const label = (await dts.nth(i).innerText()).trim();
    const dtBox = await dts.nth(i).boundingBox();
    const ddBox = await dds.nth(i).boundingBox();
    if (!dtBox || !ddBox) {
      misaligned.push(`${label}: not visible`);
      continue;
    }
    // A label and its own value must sit on the same visual row (this is what "aligned
    // label/value pairs" means) and the value must sit to the label's right.
    if (Math.abs(dtBox.y - ddBox.y) > 2 || ddBox.x <= dtBox.x) {
      misaligned.push(`${label}: dt@(${dtBox.x.toFixed(0)},${dtBox.y.toFixed(0)}) dd@(${ddBox.x.toFixed(0)},${ddBox.y.toFixed(0)})`);
    }
  }
  expect(misaligned, `misaligned dt/dd pairs at >=640px width:\n${misaligned.join('\n')}`).toEqual([]);
});

// ---------------------------------------------------------------------------------------------
// c5 / c6 — need to break Store, or downgrade Process's service key scope: a private stack
// ---------------------------------------------------------------------------------------------

test.describe('c5 & c6: process console against a private, controllable stack', () => {
  test.describe.configure({ mode: 'serial' });
  let outage: OutageStack;

  test.beforeAll(async () => {
    outage = await OutageStack.start('consoles');
  });

  test.afterAll(async () => {
    await outage?.close();
  });

  test('c5: Pipeline shows the store_unavailable degraded state during a real Store outage', async ({ page, stack }) => {
    const wavPath = path.join(stack.repo, 'sample_audio', 'call_01_compliant.wav');
    const bytes = uniqueWav(fs.readFileSync(wavPath));
    const tmp = path.join(outage.info.dir, 'uploads', `c5-${Date.now()}.wav`);
    fs.mkdirSync(path.dirname(tmp), { recursive: true });
    fs.writeFileSync(tmp, bytes);
    const receipt = await ingestOn(outage, tmp, `pw-c5-${Date.now()}`);
    await waitUntilSettledOn(outage, receipt.conversation_id);

    await page.goto(outage.info.process_console_url);
    await page.getByRole('button', { name: 'Pipeline' }).click();
    const row = page.locator('section', { has: page.locator(`a[href*="${receipt.call_id}"]`) });
    await expect(row).toBeVisible({ timeout: 20_000 });
    await expect(page.getByText('Store is unreachable right now')).toHaveCount(0); // healthy baseline

    await outage.stopStore();
    // The next conversations poll degrades: progress_stale + store_unavailable (README "Behaviour
    // both suites share"). Force it rather than waiting out the 8s refetchInterval.
    await page.getByRole('button', { name: 'Refresh' }).click();
    await expect(page.getByText('Store is unreachable right now, so each row shows the last progress seen.')).toBeVisible({ timeout: 20_000 });
    await expect(row.getByText(/\(last known; Store unreachable\)/)).toBeVisible();

    await outage.startStore();
  });

  test('c6: retry/cancel surface the friendly insufficient_scope explanation, not a raw 403', async ({ page, stack }) => {
    await outage.downgradeScope(); // re-issues Process's service key WITHOUT jobs:control

    const wavPath = path.join(stack.repo, 'sample_audio', 'call_01_compliant.wav');
    const bytes = uniqueWav(fs.readFileSync(wavPath));
    const tmp = path.join(outage.info.dir, 'uploads', `c6-${Date.now()}.wav`);
    fs.mkdirSync(path.dirname(tmp), { recursive: true });
    fs.writeFileSync(tmp, bytes);
    const receipt = await ingestOn(outage, tmp, `pw-c6-${Date.now()}`);

    await page.goto(outage.info.process_console_url);
    await page.getByRole('button', { name: 'Pipeline' }).click();
    const row = page.locator('section', { has: page.locator(`a[href*="${receipt.call_id}"]`) });
    await expect(row).toBeVisible({ timeout: 20_000 });
    await row.getByRole('button', { expanded: false }).first().click();
    await row.getByRole('button', { name: 'Show all details', exact: true }).click();
    const cancelButton = row.getByRole('button', { name: 'Cancel' }).first();
    await expect(cancelButton).toBeVisible({ timeout: 20_000 });
    await cancelButton.click();
    await row.getByLabel('Reason').fill('c6 e2e scope test');
    // Submit the cancellation, preserving the separate Close action and the server-side scope check.
    await row.locator('form button[type="submit"]').click();

    await expect(row.getByText(/lacks the jobs:control scope/)).toBeVisible({ timeout: 20_000 });
    await expect(row.getByText(/^403/)).toHaveCount(0); // never a raw status code dumped at the user
  });
});

// ---------------------------------------------------------------------------------------------
// c7 — Console-token bootstrap: fragment → sessionStorage → cleared URL → sent on writes
// ---------------------------------------------------------------------------------------------

test('c7: console token bootstraps from the URL fragment, clears the URL, and is sent on writes', async ({
  page,
  processConsoleURL,
  consoleToken,
  stack,
}) => {
  expect(processConsoleURL).toContain(`#console_token=${consoleToken}`);

  await page.goto(processConsoleURL);
  await expect(page.getByText('Console connected', { exact: true })).toBeVisible();

  // The fragment is stripped from the visible address bar immediately.
  await expect.poll(() => page.evaluate(() => window.location.hash)).toBe('');
  expect(page.url()).not.toContain('console_token=');

  // It is kept in sessionStorage (never localStorage — it's a secret), not lost.
  const stored = await page.evaluate(() => window.sessionStorage.getItem('call1-process-console-token'));
  expect(stored).toBe(consoleToken);
  const inLocalStorage = await page.evaluate(() => window.localStorage.getItem('call1-process-console-token'));
  expect(inLocalStorage).toBeNull();

  // And it is actually sent on a write: a real upload through the Import tab.
  await page.getByRole('button', { name: 'Import' }).click();
  const wavPath = path.join(stack.repo, 'sample_audio', 'call_01_compliant.wav');
  await page.locator('#process-import-file').setInputFiles(wavPath);
  const uploadRequest = page.waitForRequest((req) => req.url().includes('/process/api/recordings') && req.method() === 'POST');
  await page.getByRole('button', { name: 'Upload and ingest' }).click();
  const req = await uploadRequest;
  expect(req.headers()['x-call1-console-token']).toBe(consoleToken);
});

// ---------------------------------------------------------------------------------------------
// c9 — Honest empty / not-built-yet / signed-out states, never fake data
// ---------------------------------------------------------------------------------------------

test('c9: Store console admin panels are honestly signed-out or role-gated, never fabricated', async ({
  page,
  storeURL,
  enrollAdmin,
  createInvitedUser,
}) => {
  // Signed out entirely: both admin-gated panels say so, and list no data.
  await page.goto(`${storeURL}/console/`);
  const installationsPanel = page.locator('section', { has: page.getByRole('heading', { name: 'Process installations & service keys' }) });
  const changeFeedPanel = page.locator('section', { has: page.getByRole('heading', { name: 'Change feed' }) });
  await expect(installationsPanel.getByText('Signed out', { exact: true })).toBeVisible();
  await expect(changeFeedPanel.getByText('Signed out', { exact: true })).toBeVisible();
  await expect(installationsPanel.getByText(/No Process installation/)).toHaveCount(0);

  // Signed in, but a reviewer (not admin): role-gated, still no fabricated data.
  const reviewer = await createInvitedUser('reviewer');
  await reviewer.page.goto(`${storeURL}/console/`);
  const reviewerInstallPanel = reviewer.page.locator('section', { has: reviewer.page.getByRole('heading', { name: 'Process installations & service keys' }) });
  await expect(reviewerInstallPanel.getByText('Admin role required')).toBeVisible();
  await expect(reviewerInstallPanel.getByText(/reviewer/)).toBeVisible();

  // Signed in as an actual admin: real data, not a placeholder — this Process installation shows up.
  const admin = await enrollAdmin(page);
  expect(admin.role).toBe('admin');
  await page.goto(`${storeURL}/console/`);
  const adminInstallPanel = page.locator('section', { has: page.getByRole('heading', { name: 'Process installations & service keys' }) });
  await expect(adminInstallPanel.getByText('Signed out')).toHaveCount(0);
  await expect(adminInstallPanel.getByText('Installations', { exact: true })).toBeVisible();
});

// ---------------------------------------------------------------------------------------------
// c10 — Static asset build separation: no cross-app contamination
// ---------------------------------------------------------------------------------------------

test('c10: each app serves only its own hashed bundle, from its own asset root', async ({ page, storeURL, processConsoleURL, request }) => {
  const bundleRequests: Record<string, string[]> = { evaluate: [], console: [], process: [] };

  page.on('request', (req) => {
    const url = req.url();
    if (!/\.(js|css)(\?|$)/.test(url)) return;
    if (url.startsWith(storeURL) && url.includes('/console/assets/')) bundleRequests.console.push(url);
    else if (url.startsWith(storeURL) && url.includes('/assets/')) bundleRequests.evaluate.push(url);
    else if (url.includes('/assets/') && !url.startsWith(storeURL)) bundleRequests.process.push(url);
  });

  await page.goto(`${storeURL}/#/`);
  await page.goto(`${storeURL}/console/`);
  await page.goto(processConsoleURL);

  expect(bundleRequests.evaluate.length, 'Evaluate should load a bundle').toBeGreaterThan(0);
  expect(bundleRequests.console.length, 'Store console should load a bundle').toBeGreaterThan(0);
  expect(bundleRequests.process.length, 'Process console should load a bundle').toBeGreaterThan(0);

  for (const url of bundleRequests.evaluate) expect(path.basename(url)).toMatch(/^evaluate-/);
  for (const url of bundleRequests.console) expect(path.basename(url)).toMatch(/^store-console-/);
  for (const url of bundleRequests.process) expect(path.basename(url)).toMatch(/^process-/);

  // No 404s for any asset any of the three pages requested (a sign of cross-app path confusion).
  for (const url of [...bundleRequests.evaluate, ...bundleRequests.console, ...bundleRequests.process]) {
    const res = await request.get(url);
    expect(res.status(), url).toBe(200);
  }
});
