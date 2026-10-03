// evaluate_ui e16: deferred/not-built areas correctly show NotBuiltYet or are simply absent, never
// fake data or a silently broken control — speaker-correction UI,
// SMTP invitation delivery, and the admin state/release-trust/Pro1/usage/price-table screens.
import { test, expect } from './fixtures';

test('e16: Workbench has no speaker-correction control; audio is the ported thread waveform over a real <audio>', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-e16-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  await expect(reviewer.page.getByRole('heading', { name: 'Scorecard' })).toBeVisible();

  // No speaker-correction control anywhere on the Workbench (the contract call exists; the UI does
  // not, per the README's "Not built yet" list) — absent, not a disabled/broken button.
  await expect(reviewer.page.getByText(/speaker correction/i)).toHaveCount(0);
  await expect(reviewer.page.getByRole('button', { name: /correct speaker/i })).toHaveCount(0);

  // The ported thread waveform (see evaluate-waveform.spec.ts) drives a real <audio> element.
  await expect(reviewer.page.getByTestId('thread-waveform')).toBeVisible();
  await expect(reviewer.page.locator('audio')).toBeAttached();
});

test('e16: invitations offer only an out-of-band link, never an "email sent" claim (SMTP delivery is deferred)', async ({
  createInvitedUser,
}, testInfo) => {
  const admin = await createInvitedUser('admin');
  await admin.page.goto('/#/admin/invitations');
  const email = `pw-e16-${testInfo.project.name}-${Date.now()}@e2e.test`;
  await admin.page.getByLabel('Email').fill(email);
  await admin.page.getByLabel('Display name').fill('E16 Invitee');
  await admin.page.getByRole('button', { name: 'Issue link' }).click();

  await expect(admin.page.getByText('Invitation link')).toBeVisible();
  await expect(admin.page.getByText(/does not email invitations yet/i)).toBeVisible(); // honest deferral copy
  await expect(admin.page.getByText(/email sent|sent an email|we emailed/i)).toHaveCount(0);
});

test('e16: metrics review-agreement is honest about its state (data or NotBuiltYet), never fabricated', async ({
  createInvitedUser,
}) => {
  const supervisor = await createInvitedUser('supervisor');
  await supervisor.page.goto('/#/metrics');
  const section = supervisor.page.locator('section', { has: supervisor.page.getByText('Review agreement') });
  await expect(section).toBeVisible();
  const real = section.getByText('Calls reviewed');
  const notBuilt = section.getByText(/not built yet/i);
  await expect(real.or(notBuilt)).toBeVisible({ timeout: 15_000 });
});

// e16 coverage gap: the deferred admin screens (admin state, audit, release trust, Pro1, usage, the
// price table) are absent from the admin nav, the admin page names them as not built, and a deep
// link to one renders a "not built yet" state, never another screen or fake data.
test('e16: deferred admin screens are absent from the admin nav and a deep link says "not built yet"', async ({
  createInvitedUser,
}) => {
  const admin = await createInvitedUser('admin');
  await admin.page.goto('/#/admin/accounts');
  const sections = admin.page.getByRole('navigation', { name: 'Admin sections' });
  await expect(sections.getByRole('link')).toHaveText(['Accounts', 'Invitations', 'Installations and keys', 'Vocabulary']);
  for (const name of [/state/i, /audit/i, /release/i, /pro1/i, /usage/i, /price/i]) {
    await expect(sections.getByRole('link', { name })).toHaveCount(0);
  }
  await expect(admin.page.getByText(/^Coming later: admin settings, the audit log, release trust, Pro1/)).toBeVisible();

  const deferred: [string, RegExp][] = [
    ['state', /^Admin state is not built yet/],
    ['audit', /^The audit log is not built yet/],
    ['release-trust', /^Release trust is not built yet/],
    ['pro1', /^Pro1 key release is not built yet/],
    ['usage', /^Usage reporting is not built yet/],
    ['price-table', /^The usage price table is not built yet/],
  ];
  for (const [slug, title] of deferred) {
    await admin.page.goto(`/#/admin/${slug}`);
    await expect(admin.page.getByText(title), `#/admin/${slug}`).toBeVisible();
    // Not a real panel under another name: no accounts table, no forms.
    await expect(admin.page.getByRole('table')).toHaveCount(0);
    await expect(admin.page.getByRole('textbox')).toHaveCount(0);
    await expect(sections.locator('[aria-current="page"]')).toHaveCount(0);
  }
});
