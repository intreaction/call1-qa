// evaluate_ui e2: CallsView (#/calls) — list renders every call with resultStateDisplay text+color
// and callQaBadge score; agent name/extension shown per row (known issue b, settled by the
// 2026-09-25 team decision 1: contract 1.1.0 adds CallMetadata.agent_display_name and
// agent_extension, and Evaluate shows "Name (ext)" with the agent_id fallback of `agent_label()`).
import type { Page, Route } from '@playwright/test';
import { test, expect } from './fixtures';
import { ingestWithMetadata } from './evaluate-helpers';

test('e2: calls list shows a row with status text, QA badge and score for an ingested call', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const agent = `pw-e2-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);

  await reviewer.page.goto('/#/calls');
  await expect(reviewer.page.getByRole('heading', { name: 'Calls' })).toBeVisible();

  const row = reviewer.page.getByRole('row').filter({ hasText: agent });
  await expect(row).toBeVisible();

  // Every derived state cell carries a text label, not color alone (StatusPill always renders text).
  const cells = row.locator('td');
  await expect(cells.nth(2)).not.toHaveText('');
  await expect(cells.nth(3)).not.toHaveText('');
  await expect(cells.nth(4)).not.toHaveText('');

  // callQaBadge: once QA settles PASS on the default rubric, the QA cell shows a score alongside
  // the badge text (formatScore(overall_score), an integer 0-100 out of 100).
  const qaCell = cells.nth(2);
  await expect(qaCell).toContainText(/\d{1,3}/);
});

test('e2 (known issue b): a call ingested with a display name and extension shows "Name (ext)" in the list and the Workbench header', async ({
  createInvitedUser,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  // A recorder or PBX integration supplies the identity as separate fields (contract 1.1.0). The
  // agent_id stays the stable key for filters and rules and is never shown in place of the name.
  const agentId = `pw-e2-agent-${testInfo.project.name}-${Date.now()}`;
  const name = `Samantha ${testInfo.project.name} ${Date.now() % 100000}`;
  const { receipt } = await ingestWithMetadata({ agent_id: agentId, agent_display_name: name, agent_extension: '104' });
  await waitUntilSettled(receipt.call_id);

  await reviewer.page.goto('/#/calls');
  const link = reviewer.page.locator(`a[href="#/calls/${receipt.call_id}"]`);
  await expect(link).toContainText(`${name} (104)`);
  await expect(link).not.toContainText('Unknown');

  // The agent_id filter keeps working on the stable id.
  await reviewer.page.getByLabel('Filter by agent or call reference').fill(agentId);
  await expect(link).toBeVisible();

  await link.click();
  // The Workbench header names the agent (as the title, or under the caller objective when one is known).
  await expect(reviewer.page.getByTestId('workbench-header').getByTestId('call-agent')).toHaveText(`${name} (104)`);
});

test('e2 (known issue b): with no agent identity supplied the row shows the agent_id fallback, never an invented name', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { unique: true });
  await waitUntilSettled(receipt.call_id);
  await reviewer.page.goto('/#/calls');
  const link = reviewer.page.locator(`a[href="#/calls/${receipt.call_id}"]`);
  // CallMetadata.agent_id defaults to "Unknown"; agent_label() falls back to it. Nothing is derived
  // from the recording itself.
  await expect(link.getByTestId('call-agent')).toHaveText('Unknown');
});

/** Rewrite one call's agent fields in Store's real answer (list items or one call record). */
async function withAgentFields(page: Page, callId: string, fields: { agent_display_name: string | null; agent_extension: string | null }) {
  const patchItem = (item: Record<string, unknown>) => (item.call_id === callId ? { ...item, ...fields } : item);
  await page.route(
    (url) => url.pathname === '/store/v1/calls' || url.pathname === `/store/v1/calls/${callId}`,
    async (route: Route) => {
      // The page may close while a stubbed request is in flight (test teardown): drop it then.
      const response = await route.fetch().catch(() => null);
      const body = response ? await response.json().catch(() => null) : null;
      if (!response || body === null) return;
      if (Array.isArray(body.items)) body.items = body.items.map(patchItem);
      if (body.call && typeof body.call === 'object') body.call = patchItem(body.call);
      await route.fulfill({ response, json: body });
    },
  );
}

test('e2: the agent label follows the contract rule for every combination of name and extension', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const agentId = `pw-e2-label-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId });
  await waitUntilSettled(receipt.call_id);
  const link = reviewer.page.locator(`a[href="#/calls/${receipt.call_id}"]`).getByTestId('call-agent');

  // call1.contracts.calls.agent_label(): name+ext, name only, ext only (agent_id + ext), neither.
  const cases: [string | null, string | null, string][] = [
    ['Bob', '202', 'Bob (202)'],
    ['Bob', null, 'Bob'],
    [null, '202', `${agentId} (202)`],
    [null, null, agentId],
  ];
  for (const [display, extension, expected] of cases) {
    await reviewer.page.unrouteAll({ behavior: 'wait' });
    await withAgentFields(reviewer.page, receipt.call_id, { agent_display_name: display, agent_extension: extension });
    await reviewer.page.goto('/#/calls');
    await reviewer.page.reload();
    await expect(link, `display=${display} extension=${extension}`).toHaveText(expected);
    await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
    await expect(reviewer.page.getByTestId('workbench-header').getByTestId('call-agent'), `Workbench header, display=${display} extension=${extension}`).toHaveText(expected);
  }
});

// --- Contact Signals v2 addition (docs/ContactSignalsV2.md section 10.3 "Calls" and section 17
// "evaluate-calls": "Caller need column; category, subcategory and alert filters"; the column is
// headed "Caller objective" to match the Workbench and the taxonomy). Written
// independently from the F4 implementation, from the design doc and the already-committed contract
// (calls.py CallListItem.signal_categories/caller_needs/signal_alerts, CallListQuery.signal_category
// /signal_subcategory/signal_alert). Route-stubs the calls list so the row's chips are deterministic
// without depending on a live classifier.
test('e2 (signals): "Caller objective" and "Signals" columns show subcategory and category chips with a "+N" overflow', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const agent = `pw-e2-signals-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);

  await reviewer.page.route('**/store/v1/calls?*', async (route) => {
    // The page may close while a stubbed request is in flight (test teardown): drop it then.
    const response = await route.fetch().catch(() => null);
    const body = response ? await response.json().catch(() => null) : null;
    if (!response || body === null) return;
    body.items = body.items.map((item: { call_id: string }) =>
      item.call_id === receipt.call_id
        ? { ...item, contact_signals_state: 'available', signal_categories: ['intent', 'issue', 'friction', 'deferred'], caller_needs: ['billing_question'], signal_alerts: ['pw_alert'] }
        : item,
    );
    await route.fulfill({ response, json: body });
  });

  await reviewer.page.goto('/#/calls');
  const row = reviewer.page.getByRole('row').filter({ hasText: agent });
  await expect(row).toBeVisible();
  // The subcategory shows in the Caller objective column (and may also title the row).
  await expect(row.locator('[data-column="caller-need"]').getByText('Billing question', { exact: false }).or(row.locator('[data-column="caller-need"]').getByText('billing_question', { exact: false }))).toBeVisible();
  // F5 reconciliation: the Signals column shows two category chips, then "+N" for the rest. The
  // caller objective (intent) has its own column, so the Signals column does not repeat it.
  const signalsCell = row.locator('[data-column="signals"]');
  await expect(signalsCell.getByText('Caller objective', { exact: true })).toHaveCount(0);
  await expect(signalsCell.getByText('Reported issue', { exact: true })).toBeVisible();
  await expect(signalsCell.getByText('Friction point', { exact: true })).toBeVisible();
  await expect(signalsCell.getByText('+1', { exact: true })).toBeVisible(); // 3 non-intent categories: two chips plus "+1"
});

test('e2 (signals): category, subcategory and alert filters narrow listCalls', async ({ createInvitedUser, adminApi }, testInfo) => {
  // F5 reconciliation: the filters are labelled "Filter by signal category/subcategory/alert"; the
  // subcategory options come from the taxonomy (plus the fixed "Other"), and the alert options from
  // the alert rules, so this test makes one rule rather than assuming one exists. Definition text
  // is checked for caller details (section 9.4), so the rule name carries no digits.
  const letters = Array.from({ length: 6 }, () => String.fromCharCode(97 + Math.floor(Math.random() * 26))).join('');
  const alertRuleId = `pw_e2_alert_${testInfo.project.name}_${letters}`;
  const rule = { rule_id: alertRuleId, name: `Filter check ${testInfo.project.name} ${letters}`, condition: { category_id: 'intent' }, enabled: true };
  const created = await adminApi.json<{ record_version: number }>('PUT', `/signals/alert-rules/${alertRuleId}`, {
    data: { rule, expected_record_version: 0 },
  });
  try {

    const reviewer = await createInvitedUser('reviewer');
    await reviewer.page.goto('/#/calls');
    await expect(reviewer.page.getByRole('heading', { name: 'Calls' })).toBeVisible();

    const categoryRequest = reviewer.page.waitForRequest((r) => r.url().includes('/store/v1/calls?') && r.url().includes('signal_category='));
    await reviewer.page.getByLabel('Filter by signal category').selectOption('intent');
    const categoryReq = await categoryRequest;
    expect(new URL(categoryReq.url()).searchParams.get('signal_category')).toBe('intent');

    const subRequest = reviewer.page.waitForRequest((r) => r.url().includes('/store/v1/calls?') && r.url().includes('signal_subcategory='));
    await reviewer.page.getByLabel('Filter by signal subcategory').selectOption('other');
    const subReq = await subRequest;
    expect(new URL(subReq.url()).searchParams.get('signal_subcategory')).toBe('other');

    const alertRequest = reviewer.page.waitForRequest((r) => r.url().includes('/store/v1/calls?') && r.url().includes('signal_alert='));
    await reviewer.page.getByLabel('Filter by signal alert').selectOption(alertRuleId);
    const alertReq = await alertRequest;
    expect(new URL(alertReq.url()).searchParams.get('signal_alert')).toBe(alertRuleId);
    // The filtered list is shareable: the address bar carries the filters.
    await expect(reviewer.page).toHaveURL(/signal_category=intent/);
  } finally {
    // Disable the rule: an enabled rule on a built-in matches every other spec's calls on the shared
    // stack (alert chips in their Workbench, rows in their metrics).
    await adminApi.json('PUT', `/signals/alert-rules/${alertRuleId}`, {
      data: { rule: { ...rule, enabled: false }, expected_record_version: created.record_version },
    });
  }
});
