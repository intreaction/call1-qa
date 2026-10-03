// Smoke test for the browser harness: servers up, an admin enrolled through Evaluate's UI with a
// CLI setup code, a recording ingested through Process (fake handlers) appearing in Evaluate's
// call list, and an invited reviewer enrolling from the link in a second browser context.
import { test, expect } from './fixtures';

test('the stack is up on private ports', async ({ stack, storeURL, processApi, request }) => {
  for (const port of [stack.store_port, stack.process_port]) expect([8000, 8010, 8020]).not.toContain(port);
  expect(stack.dir.startsWith('/private/tmp/call1-e2e/')).toBe(true);

  const status = await request.get(`${storeURL}/store/v1/status`);
  expect(status.ok()).toBe(true);

  const health = await (await processApi.get('/health', { token: false })).json();
  expect(health.state).toBe('running');
  const session = await (await processApi.get('/session')).json();
  expect(session.token_valid).toBe(true);
});

test('an admin enrolls in Evaluate and sees an ingested call in the list', async ({ page, enrollAdmin, ingestSample, waitUntilSettled, colorScheme }, testInfo) => {
  const admin = await enrollAdmin(page);
  expect(admin.role).toBe('admin');
  await expect(page.locator('html')).toHaveAttribute('data-theme', colorScheme === 'light' ? 'light' : 'dark');

  const agent = `pw-smoke-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  expect(receipt.conversation_created).toBe(true);
  await waitUntilSettled(receipt.call_id);

  await page.goto('/#/calls');
  await expect(page.getByRole('heading', { name: 'Calls' })).toBeVisible();
  const row = page.getByRole('row').filter({ hasText: agent });
  await expect(row).toBeVisible();
  await row.getByRole('link').click();
  await expect(page).toHaveURL(new RegExp(`#/calls/${receipt.call_id}$`));
});

test('an invited reviewer enrolls from the link, then signs in elsewhere', async ({ createInvitedUser, newAuthPage, signIn }) => {
  const reviewer = await createInvitedUser('reviewer');
  expect(reviewer.role).toBe('reviewer');
  await expect(reviewer.page.getByRole('button', { name: 'Sign out' })).toBeVisible();
  await expect(reviewer.page.getByRole('link', { name: 'Admin' })).toHaveCount(0);

  const { page } = await newAuthPage();
  const again = await signIn(page, reviewer.email);
  expect(again.accountId).toBe(reviewer.accountId);
  await expect(page.getByRole('heading', { name: 'Calls' })).toBeVisible();
});
