// evaluate_ui e8: Admin area (#/admin/accounts, invitations, installations) is hidden from nav for
// non-admins; an admin can view/manage accounts, invitations, installations.
import { test, expect } from './fixtures';

test('e8: admin nav is hidden from a reviewer and a supervisor, and shown for an admin', async ({
  createInvitedUser,
}) => {
  const reviewer = await createInvitedUser('reviewer');
  await reviewer.page.goto('/#/calls');
  await expect(reviewer.page.getByRole('link', { name: 'Admin' })).toHaveCount(0);
  // Server is the source of truth: direct navigation to the admin route must not leak admin UI.
  await reviewer.page.goto('/#/admin/accounts');
  await expect(reviewer.page.getByText('Admin role required')).toBeVisible();

  const supervisor = await createInvitedUser('supervisor');
  await supervisor.page.goto('/#/calls');
  await expect(supervisor.page.getByRole('link', { name: 'Admin' })).toHaveCount(0);
});

test('e8: an admin manages accounts, invitations and installations', async ({ createInvitedUser }, testInfo) => {
  const admin = await createInvitedUser('admin');
  await admin.page.goto('/#/calls');
  await admin.page.getByRole('link', { name: 'Admin', exact: true }).click();
  await expect(admin.page).toHaveURL(/#\/admin\/accounts$/);
  // The list pages 100 accounts at a time, and a full run on the shared stack enrolls more than
  // that: load pages until this admin's row shows.
  const mine = admin.page.getByText(admin.email);
  const loadMore = admin.page.getByRole('button', { name: 'Load more' });
  await expect(async () => {
    if (!(await mine.isVisible()) && (await loadMore.isVisible())) await loadMore.click();
    await expect(mine).toBeVisible({ timeout: 1_000 });
  }).toPass({ timeout: 30_000 });

  await admin.page.getByRole('link', { name: 'Invitations' }).click();
  await expect(admin.page).toHaveURL(/#\/admin\/invitations$/);
  const email = `pw-e8-${testInfo.project.name}-${Date.now()}@e2e.test`;
  await admin.page.getByLabel('Email').fill(email);
  await admin.page.getByLabel('Display name').fill('E8 Invitee');
  await admin.page.getByRole('button', { name: 'Issue link' }).click();
  await expect(admin.page.getByText('Invitation link')).toBeVisible();
  await expect(admin.page.getByText(email, { exact: true })).toBeVisible();

  await admin.page.getByRole('link', { name: 'Installations and keys' }).click();
  await expect(admin.page).toHaveURL(/#\/admin\/installations$/);
  // The e2e stack's own installation (issue-service-key at startup) must be listed.
  await expect(admin.page.getByText(/installation|service key/i).first()).toBeVisible();
});
