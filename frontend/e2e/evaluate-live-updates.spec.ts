// evaluate_ui e11: live updates via change-feed polling and query invalidation. A write elsewhere
// reflects in the UI within one poll interval without a manual refresh.
import { test, expect } from './fixtures';
import { ingestWithMetadata } from './evaluate-helpers';

test('e11: a call ingested elsewhere appears in an already-open calls list within one poll interval', async ({
  createInvitedUser,
  ingestSample,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  await reviewer.page.goto('/#/calls');
  await expect(reviewer.page.getByRole('heading', { name: 'Calls' })).toBeVisible();

  const agent = `pw-e11-${testInfo.project.name}-${Date.now()}`;
  // Ingest "elsewhere" (Process's API, not through this page) after the list is already open.
  await ingestSample('call_01_compliant', { agentId: agent });

  // The change poller (api/changes.ts) runs every 5s by default; the calls query invalidates on a
  // `call` change event (keysForChange). No reload, no manual refetch.
  await expect(reviewer.page.getByText(agent)).toBeVisible({ timeout: 12_000 });
});

// e11 coverage gap: the UI side of a lost change-feed cursor. Store answers 410 `cursor_expired`
// (retention passed) or `cursor_unknown` (a restore started a new feed epoch); Evaluate must take a
// full re-snapshot (every query re-read) and carry on polling from a fresh cursor, with no error
// shown and no reload.
for (const code of ['cursor_expired', 'cursor_unknown'] as const) {
  test(`e11: a 410 ${code} from the change feed makes Evaluate re-snapshot and keep polling`, async ({
    createInvitedUser,
    ingestSample,
  }, testInfo) => {
    const reviewer = await createInvitedUser('reviewer');
    const page = reviewer.page;
    let injected = false;
    const feedCalls: { after: string | null; at: number }[] = [];
    await page.route(
      (url) => url.pathname === '/store/v1/changes',
      async (route) => {
        const after = new URL(route.request().url()).searchParams.get('after');
        feedCalls.push({ after, at: Date.now() });
        if (after && !injected) {
          injected = true;
          await route.fulfill({
            status: 410,
            json: {
              code,
              message: code === 'cursor_expired' ? 'The cursor is older than the retention window.' : 'The cursor belongs to another feed epoch.',
              details: code === 'cursor_expired' ? { oldest_cursor: 'e2e-oldest' } : { feed_epoch: 'e2e-new-epoch' },
              retryable: false,
              request_id: 'e2e-injected',
            },
          });
          return;
        }
        await route.fallback();
      },
    );

    await page.goto('/#/calls');
    await expect(page.getByRole('heading', { name: 'Calls' })).toBeVisible();

    // The first resumable poll (with `after`) gets the 410; the re-snapshot is a fresh read of the
    // calls list that starts after it.
    const resnapshot = page.waitForRequest(
      (r) => injected && new URL(r.url()).pathname === '/store/v1/calls' && r.method() === 'GET',
      { timeout: 20_000 },
    );
    await expect.poll(() => injected, { timeout: 20_000 }).toBe(true);
    await resnapshot;
    // Polling restarts from the feed's latest cursor (a poll without `after`), then resumes with one.
    const afterInjection = () => feedCalls.slice(feedCalls.findIndex((c) => c.after !== null) + 1);
    await expect
      .poll(() => afterInjection().some((c) => c.after === null), { message: 'a fresh-cursor poll after the 410' })
      .toBe(true);
    await expect
      .poll(
        () => {
          const later = afterInjection();
          const fresh = later.findIndex((c) => c.after === null);
          return fresh >= 0 && later.slice(fresh + 1).some((c) => c.after !== null);
        },
        { message: 'polling resumes from the new cursor', timeout: 20_000 },
      )
      .toBe(true);
    await expect(page.getByRole('alert')).toHaveCount(0);
    await expect(page.getByText('Offline', { exact: true })).toHaveCount(0);

    // Live updates still work after the resync: a new call appears without a reload.
    const agent = `pw-e11-${code}-${testInfo.project.name}-${Date.now()}`;
    await ingestSample('call_01_compliant', { agentId: agent });
    await expect(page.getByText(agent)).toBeVisible({ timeout: 15_000 });
  });
}

// Team decision 2 (2026-09-25): re-uploading the same recording with different call metadata
// updates the call's metadata (audited, a `call` change event with status "metadata_updated"),
// without reprocessing. An open calls list and Workbench pick the new label up from the feed.
test('e11/q16: re-uploading the same recording with a new agent identity updates the open list and Workbench without a reload', async ({
  createInvitedUser,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const agentId = `pw-q16-${testInfo.project.name}-${Date.now()}`;
  const first = await ingestWithMetadata({ agent_id: agentId });
  await waitUntilSettled(first.receipt.call_id);

  await reviewer.page.goto('/#/calls');
  const link = reviewer.page.locator(`a[href="#/calls/${first.receipt.call_id}"]`);
  await expect(link.getByTestId('call-agent')).toHaveText(agentId);

  // Same bytes, so the same conversation; only the metadata differs.
  const again = await ingestWithMetadata({ agent_display_name: 'Bob', agent_extension: '202' }, { bytes: first.bytes });
  expect(again.receipt.call_id).toBe(first.receipt.call_id);
  expect(again.receipt.conversation_created).toBe(false);
  await expect(link.getByTestId('call-agent')).toHaveText('Bob (202)', { timeout: 15_000 });

  await link.click();
  await expect(reviewer.page.getByTestId('workbench-header').getByTestId('call-agent')).toHaveText('Bob (202)');
});

// --- Contact Signals v2 addition (docs/ContactSignalsV2.md section 10.3 "Live updates" and section
// 17 "evaluate-live-updates": "a second browser sees chips and rankings appear with no reload").
// Written independently from the F4 implementation, from the design doc's keysForChange table
// (signal_taxonomy, signal_alert_rule, signal_alert, result with a contact-signals: status, review
// with signal_feedback, reanalysis_request).
test("e11 (signals): a `result` change event with a contact-signals status refreshes an open calls list's chips, with no reload", async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const agent = `pw-e11-signals-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);

  let injected = false;
  await reviewer.page.route('**/store/v1/changes*', async (route) => {
    const url = new URL(route.request().url());
    const after = url.searchParams.get('after');
    if (after && !injected) {
      injected = true;
      // The page may close while a stubbed request is in flight (test teardown): drop it then.
      const response = await route.fetch().catch(() => null);
      const body = response ? await response.json().catch(() => null) : null;
      if (!response || body === null) return;
      body.events = [
        ...body.events,
        { cursor: 'e2e-signal-cursor-1', occurred_at: new Date().toISOString(), kind: 'result', resource_id: receipt.call_id, call_id: receipt.call_id, version: 2, status: 'contact_signals:available' },
      ];
      body.next_cursor = 'e2e-signal-cursor-1';
      return route.fulfill({ response, json: body });
    }
    await route.fallback();
  });
  // The calls list itself is patched once, standing in for the fresh projection a real
  // contact_signals publish would have written by the time the invalidated query refetches.
  await reviewer.page.route('**/store/v1/calls?*', async (route) => {
    // The page may close while a stubbed request is in flight (test teardown): drop it then.
    const response = await route.fetch().catch(() => null);
    const body = response ? await response.json().catch(() => null) : null;
    if (!response || body === null) return;
    if (injected) {
      body.items = body.items.map((item: { call_id: string }) =>
        item.call_id === receipt.call_id ? { ...item, contact_signals_state: 'available', signal_categories: ['intent'], caller_needs: ['billing_question'] } : item,
      );
    }
    await route.fulfill({ response, json: body });
  });

  await reviewer.page.goto('/#/calls');
  const row = reviewer.page.getByRole('row').filter({ hasText: agent });
  await expect(row).toBeVisible();

  await expect.poll(() => injected, { timeout: 20_000 }).toBe(true);
  // keysForChange (section 10.3): `result` with a `contact_signals:` status invalidates the call
  // and the lists — the new chip appears without any reload.
  // .first(): once the names load, the subcategory may also title the row.
  await expect(row.getByText('Billing question', { exact: false }).or(row.getByText('billing_question', { exact: false })).first()).toBeVisible({ timeout: 15_000 });
});
