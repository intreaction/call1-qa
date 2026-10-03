// evaluate_ui e4: RubricsView (#/rubrics, #/rubrics/:rubricId) — Rubric Studio: list, draft editor
// with optimistic concurrency (409 rubric_version_conflict), publish, retire, version history, and
// a draft test that polls to a terminal result.
import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { slug } from './evaluate-helpers';

/** Both the rubric and a criterion have a "Name" field; scope to the "Criteria" card's section. */
function criteriaCard(page: Page) {
  return page.locator('section', { has: page.getByRole('heading', { name: 'Criteria' }) });
}

test('e4: draft editor saves and publishes a new rubric', async ({ createInvitedUser }, testInfo) => {
  const supervisor = await createInvitedUser('supervisor');
  const rubricId = slug(`pw-e4-${testInfo.project.name}`);

  await supervisor.page.goto('/#/rubrics');
  await supervisor.page.getByPlaceholder('new-rubric-id').fill(rubricId);
  await supervisor.page.getByRole('button', { name: 'New rubric' }).click();
  await expect(supervisor.page).toHaveURL(new RegExp(`#/rubrics/${rubricId}$`));

  await supervisor.page.getByLabel('Name', { exact: true }).fill('Greeting rubric');
  await supervisor.page.getByRole('button', { name: 'Add criterion' }).click();
  const criteria = criteriaCard(supervisor.page);
  await criteria.getByLabel('Criterion id').fill('C1');
  await criteria.getByLabel('Name', { exact: true }).fill('Greeting');

  await supervisor.page.getByRole('button', { name: 'Save draft' }).click();
  await expect(supervisor.page.getByText('Draft saved.')).toBeVisible();

  const publish = supervisor.page.getByRole('button', { name: 'Publish', exact: true });
  await expect(publish).toBeEnabled();
  await publish.click();
  await expect(supervisor.page.getByText('Published.')).toBeVisible();

  await supervisor.page.getByRole('button', { name: 'published', exact: true }).click();
  await expect(supervisor.page.getByText('Greeting', { exact: true })).toBeVisible();
  await expect(supervisor.page.getByText('Pass threshold: 80')).toBeVisible();
});

test('e4: a stale draft save is rejected with rubric_version_conflict, per the version-conflict UX contract', async ({
  createInvitedUser,
  newAuthPage,
  signIn,
}, testInfo) => {
  const supervisor = await createInvitedUser('supervisor');
  const rubricId = slug(`pw-e4-conflict-${testInfo.project.name}`);

  await supervisor.page.goto('/#/rubrics');
  await supervisor.page.getByPlaceholder('new-rubric-id').fill(rubricId);
  await supervisor.page.getByRole('button', { name: 'New rubric' }).click();
  const nameField = supervisor.page.getByLabel('Name', { exact: true });
  await nameField.fill('v1 from page A');
  await supervisor.page.getByRole('button', { name: 'Save draft' }).click();
  await expect(supervisor.page.getByText('Draft saved.')).toBeVisible();

  // A second browser context, same supervisor account, opens the SAME draft before page A's next
  // edit — both now hold draft_revision 1 locally.
  const second = await newAuthPage();
  await signIn(second.page, supervisor.email);
  await second.page.goto(`/#/rubrics/${rubricId}`);
  const secondName = second.page.getByLabel('Name', { exact: true });
  await expect(secondName).toHaveValue('v1 from page A');

  // Page A saves again, advancing the revision Store holds.
  await nameField.fill('v2 from page A');
  await supervisor.page.getByRole('button', { name: 'Save draft' }).click();
  await expect(supervisor.page.getByText('Draft saved.')).toBeVisible();

  // Page B still holds the stale revision; its save is rejected with the conflict, the error stays
  // visible (not silently overwritten or auto-retried), and the query invalidation re-reads the
  // latest draft so a resubmit would succeed.
  await secondName.fill('page B never should have won');
  await second.page.getByRole('button', { name: 'Save draft' }).click();
  // ErrorNotice renders red notices with role="alert".
  await expect(second.page.getByRole('alert')).toBeVisible();
  await expect(secondName).toHaveValue('v2 from page A', { timeout: 15_000 });
});

test('e4: a draft test runs against a real call and polls to a terminal result', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const supervisor = await createInvitedUser('supervisor');
  const rubricId = slug(`pw-e4-drafttest-${testInfo.project.name}`);
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-e4-drafttest-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);

  await supervisor.page.goto('/#/rubrics');
  await supervisor.page.getByPlaceholder('new-rubric-id').fill(rubricId);
  await supervisor.page.getByRole('button', { name: 'New rubric' }).click();
  await supervisor.page.getByLabel('Name', { exact: true }).fill('Draft test rubric');
  await supervisor.page.getByRole('button', { name: 'Add criterion' }).click();
  const criteria = criteriaCard(supervisor.page);
  await criteria.getByLabel('Criterion id').fill('DT-1');
  await criteria.getByLabel('Name', { exact: true }).fill('Thanks for calling');
  // Open the check-type editor and set a phrase the fixed fake-handler script always contains, so
  // the draft test's outcome does not depend on the (unavailable) LLM path.
  await criteria.getByRole('button', { name: 'Thanks for calling' }).click();
  await criteria.getByLabel('Phrases').fill('Thank you for calling');
  await supervisor.page.getByRole('button', { name: 'Save draft' }).click();
  await expect(supervisor.page.getByText('Draft saved.')).toBeVisible();

  await supervisor.page.getByLabel('Call id').fill(receipt.call_id);
  await supervisor.page.getByRole('button', { name: 'Run test' }).click();

  // Terminal: either a scored result, a failure notice, or (deferred) NotBuiltYet — never stuck on
  // "Scoring…" forever.
  await expect(supervisor.page.getByText('Scoring…')).toHaveCount(0, { timeout: 30_000 });
  const terminal = supervisor.page
    .getByText(/^Score: \d+$/)
    .or(supervisor.page.getByText(/Draft test failed/))
    .or(supervisor.page.getByText(/not built yet/i));
  await expect(terminal).toBeVisible();
});
