// evaluate_ui e6: EscalationsView (#/escalations) — list and (supervisor, resolve_escalation)
// resolve. A non-supervisor sees the list but no resolve control; a supervisor can resolve and it
// updates immediately via change-feed invalidation.
import { test, expect } from './fixtures';
import { publishGuaranteedFailRubric, reanalyzeAgainst, slug } from './evaluate-helpers';

test('e6: a reviewer sees escalations but not the resolve control; a supervisor resolves one', async ({
  createInvitedUser,
  adminApi,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const id = testInfo.project.name;
  const reviewer = await createInvitedUser('reviewer');
  const supervisor = await createInvitedUser('supervisor');

  const rubric = await publishGuaranteedFailRubric(adminApi, slug(`pw-e6-${id}`));
  const agent = `pw-e6-${id}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);
  await reanalyzeAgainst(adminApi, receipt.call_id, rubric);

  const rowFor = (page: typeof reviewer.page) => page.locator('div.px-4.py-3', { has: page.locator(`a[href="#/calls/${receipt.call_id}"]`) });

  await reviewer.page.goto('/#/escalations');
  const reviewerRow = rowFor(reviewer.page);
  await expect(reviewerRow).toBeVisible({ timeout: 20_000 });
  await expect(reviewerRow).toContainText(agent);
  await expect(reviewerRow.getByRole('button', { name: 'Resolve' })).toHaveCount(0);

  await supervisor.page.goto('/#/escalations');
  const supervisorRow = rowFor(supervisor.page);
  await expect(supervisorRow).toBeVisible({ timeout: 20_000 });
  await supervisorRow.getByRole('button', { name: 'Resolve' }).click();
  await supervisorRow.getByRole('button', { name: 'Approve' }).click();

  // Live update via change-feed invalidation: the resolved item drops out of the default
  // (status=PENDING) filter without a manual reload.
  await expect(rowFor(supervisor.page)).toHaveCount(0, { timeout: 15_000 });
});
