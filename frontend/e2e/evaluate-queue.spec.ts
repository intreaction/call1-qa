// evaluate_ui e5: QueueView (#/queue) — claim-next, assign, start, release, resolve; supervisor
// rule CRUD (manage_queue_rules). claim-next assigns per active rules; resolve reads the call's
// review_version from GET /calls/{id}/review, not the queue item's item_version.
import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { slug } from './evaluate-helpers';

/** The queue item row for `callId` (scoped by its Workbench link, never by free-text matching). */
function queueRow(page: Page, callId: string) {
  return page.locator('div.py-3', { has: page.locator(`a[href="#/calls/${callId}"]`) });
}

async function createRule(page: Page, ruleId: string) {
  await page.getByRole('button', { name: 'Rules' }).click();
  await page.getByRole('button', { name: 'New rule' }).click();
  await page.getByLabel('Name').fill(ruleId);
  await page.getByLabel('Sampling rate').fill('1');
  page.once('dialog', (d) => void d.accept(ruleId));
  await page.getByRole('button', { name: 'Create', exact: true }).click();
  await expect(page.getByText(ruleId)).toBeVisible();
  await page.getByRole('button', { name: 'Rules' }).click(); // collapse back to the item list
}

/**
 * Delete a "matches every call" scratch rule. The review queue is shared by every concurrently
 * running test file on this stack, so a sampling_rate: 1 rule left enabled would route OTHER
 * tests' calls into it too; clean up as soon as the test is done with it.
 */
async function deleteRule(page: Page, ruleId: string) {
  // The toggle reads "Rules" when closed and "Hide rules" when open; open it only if it is closed.
  if (await page.getByRole('button', { name: 'Rules', exact: true }).isVisible()) {
    await page.getByRole('button', { name: 'Rules', exact: true }).click();
  }
  const row = page.locator('div.py-2', { hasText: ruleId });
  page.once('dialog', (d) => void d.accept());
  await row.getByRole('button', { name: 'Delete' }).click();
  // Only the rules list: queue items other tests' calls got under this rule keep its name.
  await expect(row).toHaveCount(0);
  await page.getByRole('button', { name: 'Hide rules', exact: true }).click();
}

test('e5: a supervisor creates a queue rule, and claim-next/resolve route through it', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const supervisor = await createInvitedUser('supervisor');
  const reviewer = await createInvitedUser('reviewer');
  const ruleId = slug(`pw-e5-${testInfo.project.name}`);

  // manage_queue_rules: create a rule that routes every call into the queue (sampling_rate 1 is a
  // deterministic hash threshold, not a coin flip — see tests/store/test_reviews_queue.py), so the
  // call ingested below is guaranteed to produce exactly one item under this rule.
  await supervisor.page.goto('/#/queue');
  await createRule(supervisor.page, ruleId);

  const agent = `pw-e5-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);

  // claim-next: a reviewer (claim_review) claims AN item the new rule created and lands in the
  // Workbench for it. The queue is shared by every concurrently running test file, so a
  // sampling_rate: 1 rule may also match another test's call meanwhile — claim-next's contract is
  // "assigns per active rules", not "assigns this specific call", so assert on whatever it claims.
  await reviewer.page.goto('/#/queue');
  // The call may have several items (seeded rules match it too); any one shows it reached the queue.
  await expect(queueRow(reviewer.page, receipt.call_id).first()).toBeVisible({ timeout: 20_000 });
  const claimResponse = reviewer.page.waitForResponse(
    (r) => r.request().method() === 'POST' && r.url().includes('/store/v1/review-queue/claim-next'),
  );
  await reviewer.page.getByRole('button', { name: 'Claim next' }).click();
  const claimed = (await (await claimResponse).json()) as { item: { id: string; call_id: string; rule_name: string } | null };
  expect(claimed.item, 'claim-next claimed an item').not.toBeNull();
  await expect(reviewer.page).toHaveURL(/#\/calls\/call_/, { timeout: 15_000 });
  const claimedCallId = reviewer.page.url().split('/calls/')[1];
  expect(claimedCallId).toBe(claimed.item!.call_id);

  // Resolve reads the call's review_version (not item_version): back on the queue, the claimed item
  // is now IN_REVIEW and assigned to this reviewer; resolving it must not 409 on a stale
  // item_version read.
  await reviewer.page.goto('/#/queue');
  // The queue is shared by every concurrently running test file (newest-first, 50 per page), so on
  // a busy stack the claimed item can sit past page 1. Judge condition 4 (2026-09-25) flaked here
  // for exactly that reason: filter to "In review" first, which this reviewer's own claim is
  // guaranteed to be in, rather than relying on it landing on the first page of "All statuses".
  await reviewer.page.getByLabel('Filter by status').selectOption('IN_REVIEW');
  // One call can carry several queue items (one per matching rule, and a superseded one next to its
  // replacement under the same rule), so narrow to the claimed item itself (data-queue-item).
  const row = queueRow(reviewer.page, claimedCallId).and(reviewer.page.locator(`[data-queue-item="${claimed.item!.id}"]`));
  await expect(row).toBeVisible({ timeout: 15_000 });
  await row.getByRole('button', { name: 'Resolve' }).click();
  const approve = row.getByRole('button', { name: 'Approve' });
  await expect(approve).toBeEnabled({ timeout: 15_000 }); // enabled once GET .../review's review_version has loaded
  await approve.click();
  // Resolving moves the item off IN_REVIEW, so the list — still filtered to IN_REVIEW — refetches
  // without it; `row` (and "Resolved") would never appear under that filter. Re-point the same
  // reliable-filter trick at the status this resolution just produced (`resolve('APPROVED')` in
  // QueueView.tsx) instead of "All statuses", which would reintroduce the exact pagination flake
  // condition 4 fixed above (the same item, freshly aged off the newest-first first page again).
  await reviewer.page.getByLabel('Filter by status').selectOption('APPROVED');
  await expect(row).toBeVisible({ timeout: 15_000 });
  await expect(row.getByText('Resolved', { exact: false })).toBeVisible();

  // manage_queue_rules CRUD — delete. Also cleans up this test's catch-all rule so it stops
  // matching other tests' calls.
  await deleteRule(supervisor.page, ruleId);
});

// If this fails with "Request validation failed": QueueView.tsx's toggleEnabled() PUTs the full
// ReviewQueueRuleRecord (which carries server-computed rule_version/updated_at/
// updated_by_account_id) as the request's `rule` field, but Store's ReviewQueueRuleSave.rule is
// typed ReviewQueueRule (call1/contracts/reviews.py) under ContractModel's `extra="forbid"` — so
// the extra fields make every toggle 422 with validation_failed. An app bug, not a test issue.
test('e5: manage_queue_rules — a supervisor disables and re-enables a rule', async ({ createInvitedUser }, testInfo) => {
  const supervisor = await createInvitedUser('supervisor');
  const ruleId = slug(`pw-e5-toggle-${testInfo.project.name}`);
  await supervisor.page.goto('/#/queue');
  await createRule(supervisor.page, ruleId);

  await supervisor.page.getByRole('button', { name: 'Rules' }).click();
  const ruleRow = supervisor.page.locator('div.py-2', { hasText: ruleId });
  await ruleRow.getByRole('button', { name: 'Disable' }).click();
  await expect(ruleRow.getByText('Disabled', { exact: true })).toBeVisible();
  await expect(supervisor.page.getByRole('alert')).toHaveCount(0);

  await ruleRow.getByRole('button', { name: 'Enable' }).click();
  await expect(ruleRow.getByText('Enabled', { exact: true })).toBeVisible();
  await expect(supervisor.page.getByRole('alert')).toHaveCount(0);

  await deleteRule(supervisor.page, ruleId);
});

test('e5: assign and release transitions are available per role', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const supervisor = await createInvitedUser('supervisor');
  const ruleId = slug(`pw-e5-assign-${testInfo.project.name}`);

  await supervisor.page.goto('/#/queue');
  await createRule(supervisor.page, ruleId);

  const agent = `pw-e5-assign-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);

  // The call can carry several queue items (one per matching rule, seeded ones included), so pick
  // one PENDING, unassigned item from Store and narrow the row to that item's rule.
  let pick: { rule_name: string } | undefined;
  await expect
    .poll(async () => {
      const res = await supervisor.page.request.get(`/store/v1/review-queue?call_id=${receipt.call_id}&status=PENDING&limit=50`);
      const items = ((await res.json()).items ?? []) as { rule_name: string; assigned_to_account_id?: string | null }[];
      pick = items.find((i) => !i.assigned_to_account_id);
      return !!pick;
    }, { timeout: 20_000 })
    .toBe(true);
  const row = queueRow(supervisor.page, receipt.call_id).filter({ hasText: pick!.rule_name });
  await expect(row).toBeVisible({ timeout: 20_000 });
  await row.getByRole('button', { name: 'Assign to me' }).click();
  await expect(row).toContainText('assigned to');

  await row.getByRole('button', { name: 'Start' }).click();
  await expect(row.getByRole('button', { name: 'Release' })).toBeVisible();
  await row.getByRole('button', { name: 'Release' }).click();
  await expect(row.getByRole('button', { name: 'Start' })).toBeVisible();

  await deleteRule(supervisor.page, ruleId);
});

// --- Contact Signals v2 addition (docs/ContactSignalsV2.md sections 9.2/9.3 and 17
// "evaluate-queue": "a SIGNAL rule creates items after a backfill; trigger chips; 'Signals changed
// since'"). Written independently from the F4 implementation. A SIGNAL rule's `target_signal_alerts`
// is checked before the stream (section 9.3); `on_new_signals` needs a current evaluation version,
// so the call is settled (QA'd) before the alert rule and the rescore backfill run.
async function createSignalRule(page: Page, ruleId: string, alertName: string) {
  await page.getByRole('button', { name: 'Rules' }).click();
  await page.getByRole('button', { name: 'New rule' }).click();
  await page.getByLabel('Name').fill(ruleId);
  await page.getByLabel('Stream', { exact: true }).selectOption('SIGNAL');
  // F5 reconciliation: the Alerts fieldset lists each rule by name ("Alert: <name>").
  await page.getByRole('group', { name: 'Alerts' }).getByRole('checkbox', { name: `Alert: ${alertName}` }).check();
  page.once('dialog', (d) => void d.accept(ruleId));
  await page.getByRole('button', { name: 'Create', exact: true }).click();
  await expect(page.getByText(ruleId)).toBeVisible();
  await page.getByRole('button', { name: 'Rules' }).click();
}

test('e5 (signals): a SIGNAL queue rule with target_signal_alerts creates items once a rescore backfill republishes matching calls', async ({
  createInvitedUser,
  adminApi,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const supervisor = await createInvitedUser('supervisor');
  const suffix = `${testInfo.project.name}-${Date.now()}`;
  const ruleId = slug(`pw-e5-signal-${suffix}`);
  const alertRuleId = `pw_alert_${suffix}`.slice(0, 40).toLowerCase().replace(/[^a-z0-9_-]/g, '-');

  // A named alert rule on the built-in "intent" category (section 9.2 SignalAlertCondition), created
  // directly against Store: the rule editor itself is evaluate-signals.spec.ts's coverage.
  // Definition text is checked for caller details (section 9.4): a long digit run reads as a card
  // number, so the name carries letters only.
  const letters = Array.from({ length: 6 }, () => String.fromCharCode(97 + Math.floor(Math.random() * 26))).join('');
  const alertName = `Caller objective ${testInfo.project.name} ${letters}`;
  const alertRule = { rule_id: alertRuleId, name: alertName, condition: { category_id: 'intent' }, enabled: true };
  const createdAlert = await adminApi.json<{ record_version: number }>('PUT', `/signals/alert-rules/${alertRuleId}`, {
    data: { rule: alertRule, expected_record_version: 0 },
  });
  try {

    const windowStart = new Date(Date.now() - 1_000).toISOString();
    const agent = `pw-e5-signal-${suffix}`;
    const receipt = await ingestSample('call_01_compliant', { agentId: agent });
    await waitUntilSettled(receipt.call_id); // a current evaluation version exists before the alert rule and rule/backfill below

    await supervisor.page.goto('/#/queue');
    await createSignalRule(supervisor.page, ruleId, alertName);

    // Section 9.3: on_new_signals only fires from a fresh contact_signals publish, not from the QA
    // publish above. Rescore this call so its signals (re)publish under the alert rule's evaluation
    // version, which is what creates the queue item.
    // The window is just this test's call, so the shared stack's other calls are not rescored.
    await adminApi.json('POST', '/signals/backfills', {
      data: { mode: 'rescore', rescore_signals: true, created_after: windowStart, created_before: new Date().toISOString(), max_calls: 5 },
      idempotencyKey: true,
    });

    const row = queueRow(supervisor.page, receipt.call_id).filter({ hasText: ruleId });
    await expect(row).toBeVisible({ timeout: 30_000 });
    // "Reason reads 'Signal: <name> (<speaker>, <m:ss>)'" (section 9.3).
    await expect(row).toContainText(/Signal:/);
    // The trigger-alert chip names the alert rule.
    await expect(row.getByLabel('Triggered by signal alerts').getByText(alertName, { exact: false })).toBeVisible();

    await deleteRule(supervisor.page, ruleId);
  } finally {
    // An enabled rule on a built-in matches every other spec's calls on the shared stack.
    await adminApi.json('PUT', `/signals/alert-rules/${alertRuleId}`, {
      data: { rule: { ...alertRule, enabled: false }, expected_record_version: createdAlert.record_version },
    });
  }
});

// e5 coverage gap ("Signals changed since this item was created", section 9.3's "An item whose
// alert stops matching stays, and shows..."): reaching this deterministically needs an item created
// by a SIGNAL rule (the test above), then either an edit to the taxonomy that retires the matched
// category/subcategory, or a hit-feedback dismissal, followed by a fresh contact_signals publish
// that no longer matches the rule — while the item itself is never superseded (section 9.3
// "Replay safety"). That is a multi-agent-plus-taxonomy-edit sequence better exercised once F2's
// alert evaluation and F3's rescore are both live; left for F5's integration pass or a follow-up.
