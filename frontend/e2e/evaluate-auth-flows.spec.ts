// evaluate_ui e1: sign-in shown on any route while signed out; a signed-out API answer anywhere
// returns the whole app to sign-in.
// evaluate_ui e10: enrollment (#/enroll?token=, #/enroll) — Store's `<store>/enroll#<token>` link
// is rewritten to `/#/enroll?token=<token>` before first render, and the token is removed from the
// address bar once enrollment succeeds.
// evaluate_ui e13 (known issue a): a reviewer is treated as signed-in on ANY route right after a
// successful enrollment ceremony, not only #/calls after clicking "Go to calls".
import { test, expect } from './fixtures';
import * as h from './harness';

test('e1: a deep link while signed out renders sign-in, not the linked view', async ({ newAuthPage }) => {
  const { page } = await newAuthPage();
  await page.goto('/#/queue');
  await expect(page.getByRole('heading', { name: 'Sign in' })).toBeVisible();
  await expect(page.getByLabel('Account email')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Review queue' })).toHaveCount(0);
});

test('e1: a signed-out API answer mid-use returns the whole app to sign-in', async ({ createInvitedUser }) => {
  const reviewer = await createInvitedUser('reviewer');
  await reviewer.page.goto('/#/calls');
  await expect(reviewer.page.getByRole('button', { name: 'Sign out' })).toBeVisible();

  // Simulate the session vanishing server-side (expiry, revocation) without the client knowing yet:
  // drop the cookie, then trigger any request. The client must notice the 401 and fall back to
  // sign-in from wherever the reviewer was — not just from an explicit "Sign out" click.
  await reviewer.context.clearCookies();
  await reviewer.page.getByRole('link', { name: 'Rubrics' }).click();
  await expect(reviewer.page.getByRole('heading', { name: 'Sign in' })).toBeVisible({ timeout: 15_000 });
});

test('e10: an invitation link (<store>/enroll#token) is rewritten to /#/enroll?token=, and the token leaves the address bar once enrolled', async ({
  adminApi,
  newAuthPage,
}, testInfo) => {
  const email = h.uniqueEmail(`pw-e10-${testInfo.project.name}`);
  const issued = await adminApi.json<{ invitation_url: string }>('POST', '/admin/invitations', {
    data: { email, display_name: 'E10 Invitee', role: 'reviewer' },
  });

  const { page } = await newAuthPage();
  await page.goto(issued.invitation_url); // Store's own <store>/enroll#<token> form
  await expect(page).toHaveURL(/#\/enroll\?token=/);
  await expect(page.getByRole('heading', { name: 'Accept your invitation' })).toBeVisible();

  await page.getByRole('button', { name: 'Create passkey' }).click();
  await expect(page.getByRole('heading', { name: 'Authenticator enrolled' })).toBeVisible({ timeout: 15_000 });
  // scrubEnrollmentToken(): the one-time secret must not linger in the visible address bar.
  await expect(page).not.toHaveURL(/token=/);
});

test('e10: the setup-code form is offered at #/enroll with no token', async ({ page, storeURL }) => {
  await page.goto(`${storeURL}/#/enroll`);
  await expect(page.getByRole('heading', { name: 'Enroll with a setup code' })).toBeVisible();
  await expect(page.getByLabel('Setup code')).toBeVisible();
});

test('e13 (known issue a): navigating to another route right after enrollment shows that route signed in, not sign-in', async ({
  adminApi,
  newAuthPage,
}, testInfo) => {
  const email = h.uniqueEmail(`pw-e13-${testInfo.project.name}`);
  const issued = await adminApi.json<{ invitation_url: string }>('POST', '/admin/invitations', {
    data: { email, display_name: 'E13 Invitee', role: 'reviewer' },
  });

  const { page } = await newAuthPage();
  await page.goto(issued.invitation_url);
  await page.getByRole('button', { name: 'Create passkey' }).click();
  await expect(page.getByRole('heading', { name: 'Authenticator enrolled' })).toBeVisible({ timeout: 15_000 });

  // Do NOT click "Go to calls" or "Add a second authenticator" (the only two paths that currently
  // call signedIn()). Instead, go straight to a different route, the way a reviewer following a
  // bookmarked link or a second tab would. The ceremony already succeeded and (per WorkbenchView/
  // EnrollPage's own comment: "the session starts when the reviewer leaves this screen") Store
  // already holds an active session for this account at this point.
  await page.evaluate(() => {
    window.location.hash = '#/queue';
  });

  // Acceptance: this route renders signed in (Review queue), not bounced to sign-in.
  await expect(page.getByRole('heading', { name: 'Review queue' })).toBeVisible({ timeout: 10_000 });
});
