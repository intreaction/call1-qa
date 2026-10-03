// c8/evaluate coverage gap: Evaluate's main flows by keyboard alone. Tab reaches every control on
// the path, each focused control shows a visible focus indicator, and Enter/Space operate it:
// navigation, opening a call from the list, a verdict override in the Workbench, the theme toggle,
// sign-out and an account-first passkey sign-in. No mouse click is used after the setup.
import { test, expect } from './fixtures';
import { hasVisibleFocus, tabTo } from './evaluate-helpers';

test('keyboard only: navigate, open a call, override a verdict, toggle the theme, sign out and sign back in', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const page = reviewer.page;
  const agent = `pw-kbd-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);

  await page.goto('/#/calls');
  await expect(page.getByRole('heading', { name: 'Calls' })).toBeVisible();
  await page.locator('body').focus();

  // Header navigation.
  const nav = page.getByRole('navigation', { name: 'Evaluate' });
  const queueLink = nav.getByRole('link', { name: 'Queue' });
  await tabTo(page, queueLink);
  expect(await hasVisibleFocus(queueLink), 'the Queue link shows a focus indicator').toBe(true);
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#\/queue$/);
  await expect(page.getByRole('heading', { name: 'Review queue' })).toBeVisible();

  const callsLink = nav.getByRole('link', { name: 'Calls' });
  await tabTo(page, callsLink, { shift: true });
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#\/calls$/);
  await expect(page.getByRole('heading', { name: 'Calls', exact: true })).toBeVisible();

  // Open the call from the list (the calls table, not a queue item that links to the same call).
  const row = page.getByRole('table').locator(`a[href="#/calls/${receipt.call_id}"]`);
  await expect(row).toBeVisible();
  await tabTo(page, row, { max: 200 });
  expect(await hasVisibleFocus(row), 'the call row link shows a focus indicator').toBe(true);
  await page.keyboard.press('Enter');
  await expect(page.getByTestId('workbench-header').getByTestId('call-agent')).toHaveText(agent);

  // Verdict override: Override opens the choices, Fail saves it.
  await expect(page.getByText('Call Recording Disclosure', { exact: true })).toBeVisible();
  const override = page.getByRole('button', { name: 'Override' }).first();
  await tabTo(page, override, { max: 300 });
  expect(await hasVisibleFocus(override), 'Override shows a focus indicator').toBe(true);
  await page.keyboard.press('Enter');
  const fail = page.getByRole('button', { name: 'Fail', exact: true });
  await tabTo(page, fail);
  await page.keyboard.press('Space');
  await expect(page.getByText('Override saved.')).toBeVisible();

  // Theme toggle with Space.
  const toggle = page.getByRole('button', { name: /^Switch to (light|dark) theme$/ });
  const before = await toggle.getAttribute('aria-label');
  await tabTo(page, toggle, { max: 300, shift: true });
  expect(await hasVisibleFocus(toggle), 'the theme toggle shows a focus indicator').toBe(true);
  await page.keyboard.press('Space');
  await expect(toggle).not.toHaveAttribute('aria-label', before ?? '');

  // Sign out, then sign in again account-first: type the email and press Enter.
  const signOut = page.getByRole('banner').getByRole('button', { name: 'Sign out' });
  await tabTo(page, signOut, { max: 300 });
  await page.keyboard.press('Enter');
  const email = page.getByLabel('Account email');
  await expect(email).toBeVisible();
  await tabTo(page, email);
  await page.keyboard.type(reviewer.email);
  await page.keyboard.press('Enter');
  await expect(page.getByRole('banner').getByRole('button', { name: 'Sign out' })).toBeVisible({ timeout: 20_000 });
});

// --- Contact Signals v2 addition (docs/ContactSignalsV2.md section 17 "evaluate-keyboard": "tree,
// chips and feedback controls reachable"). Written independently from the F4 implementation.
test('keyboard only: the Signals tree, hit chips and feedback controls are reachable and operable by keyboard', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const admin = await createInvitedUser('admin');
  const page = admin.page;

  // The Signals tree (admin nav item, section 10.1).
  await page.goto('/#/calls');
  await page.locator('body').focus();
  const nav = page.getByRole('navigation', { name: 'Evaluate' });
  const signalsLink = nav.getByRole('link', { name: 'Signals' });
  await tabTo(page, signalsLink, { max: 40 });
  expect(await hasVisibleFocus(signalsLink), 'the Signals nav link shows a focus indicator').toBe(true);
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#\/signals$/);
  await expect(page.getByRole('heading', { name: 'Signals', exact: true })).toBeVisible();

  // F5 reconciliation: each tree row is one link (name, "Built-in", gloss and counts inside it).
  const firstBuiltIn = page.getByRole('navigation', { name: 'Signal categories' }).getByRole('link', { name: /^Caller objective Built-in/ });
  await tabTo(page, firstBuiltIn, { max: 60 });
  expect(await hasVisibleFocus(firstBuiltIn), 'a tree row shows a focus indicator').toBe(true);
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/#\/signals\/intent$/);

  // A hit's chip and feedback controls, in the Workbench.
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-kbd-signals-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  await page.route(`**/store/v1/calls/${receipt.call_id}`, async (route) => {
    // The page may close while a stubbed request is in flight (test teardown): drop it then.
    const response = await route.fetch().catch(() => null);
    const body = response ? await response.json().catch(() => null) : null;
    if (!response || body === null) return;
    body.results = (body.results ?? []).map((g: { kind: string }) => (g.kind === 'contact_signals' ? { ...g, state: 'available', version: 1 } : g));
    await route.fulfill({ response, json: body });
  });
  // The section re-reads the signals after a feedback write, so the stub serves what was stored.
  let feedback: Record<string, unknown>[] = [];
  await page.route(`**/store/v1/calls/${receipt.call_id}/contact-signals`, async (route) => {
    await route.fulfill({
      json: {
        call_id: receipt.call_id, artifact_id: 'art_e2e', version: 1, completeness: 'complete', passes: [],
        transcript_fingerprint: 'a'.repeat(64), generated_at: new Date().toISOString(), pipeline: 'v2',
        taxonomy: { version: 1, digest: 'a'.repeat(64) },
        stages: [{ stage: 'categorize', included: true }, { stage: 'subcategorize', included: true }],
        segmentation: { segmenter_version: 'seg-v1', window_seconds: 7, segments: 10, scored_segments: 10, skipped_unattributed: 0, skipped_system: 0, interpolated_turns: 0 },
        stage1_digests: {}, feedback, alerts: [],
        signals: [{
          id: 'intent.abcdef012345.12345678.t0b0', kind: 'intent', label: 'Caller objective', start: 0, end: 5, speaker: 'CALLER',
          quote: 'Hi, I have a question.', turn_id: 0, char_start: 0, char_end: 20, confidence: 0.9, review_status: 'unreviewed',
          category_id: 'intent', category_digest: 'abcdef012345', category_confidence: 0.9,
          subcategory_id: 'other', subcategory_label: 'Other', subcategory_digest: '000000000000', subcategory_confidence: 0.4,
          span: { block: 0, first_window: 0, last_window: 0, timing: 'words', context_start: 0, context_end: 5 },
          fields: [], quote_narrowed: false,
        }],
      },
    });
  });
  await page.route(`**/store/v1/calls/${receipt.call_id}/signal-hits/*/feedback`, async (route) => {
    const stored = {
      call_id: receipt.call_id, hit_id: 'intent.abcdef012345.12345678.t0b0', category_verdict: 'confirmed', subcategory_id: null,
      subcategory_digest: null, subcategory_verdict: null, corrected_subcategory_id: null, note: null,
      account_id: admin.accountId, feedback_version: 1, updated_at: new Date().toISOString(),
    };
    feedback = [stored];
    await route.fulfill({ json: stored });
  });
  await page.goto(`/#/calls/${receipt.call_id}`);
  const section = page.locator('section', { has: page.getByRole('heading', { name: 'Contact signals' }) });
  // The hit row is one button (its chips are inside it) that jumps to the turn.
  const chip = section.getByRole('button', { name: /^Caller objective › Other at \d+:\d\d: jump to the turn$/ });
  await tabTo(page, chip, { max: 300 });
  expect(await hasVisibleFocus(chip), 'a signal chip shows a focus indicator').toBe(true);

  const confirm = section.getByRole('button', { name: 'Confirm', exact: true });
  await tabTo(page, confirm, { max: 30 });
  expect(await hasVisibleFocus(confirm), 'the Confirm control shows a focus indicator').toBe(true);
  await page.keyboard.press('Enter');
  await expect(section.getByText('Confirmed', { exact: true })).toBeVisible();
  // The subcategory verdict control is reachable too.
  const confirmSub = section.getByRole('button', { name: 'Confirm subcategory Other' });
  await tabTo(page, confirmSub, { max: 300 });
  expect(await hasVisibleFocus(confirmSub), 'the subcategory Confirm control shows a focus indicator').toBe(true);
});
