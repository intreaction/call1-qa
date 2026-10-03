// Auth area, in the browser: Evaluate (the built app Store serves) against real Store and Process,
// with Chrome's CDP virtual authenticators. Test names carry the auth inventory's feature IDs.
//
//   a1  enrollment with a setup code and from an invitation link
//   a2  (known issue a) signed in on any route right after enrollment (also evaluate e13)
//   a3  account-first sign-in with a non-discoverable USB security key
//   a4  CSRF: recovered from GET /auth/session after a reload, re-read once on 403 csrf_failed
//   a5  roles: a reviewer sees no admin controls, and Store refuses them anyway
//   a8  your account: authenticators and sessions, adding a second authenticator (also evaluate e9)
//   a9  the admin area: invitations, accounts, installations and keys
import type { Page, Request } from '@playwright/test';
import {
  test,
  expect,
  addVirtualAuthenticator,
  enrollViaUI,
  installClientSpread,
  serviceRequest,
  stackInfo,
  uniqueEmail,
  type StoreApi,
} from './fixtures';

// The header's Sign out (the account page has its own "Sign out" for other sessions).
const signOutButton = (page: Page) => page.getByRole('banner').getByRole('button', { name: 'Sign out' });
const signInEmail = (page: Page) => page.getByLabel('Account email');

async function invite(adminApi: StoreApi, email: string, role: 'reviewer' | 'supervisor' | 'admin' = 'reviewer', displayName = 'E2E Invitee') {
  const response = await adminApi.post('/admin/invitations', { email, display_name: displayName, role });
  expect(response.status(), await response.text()).toBe(200);
  return (await response.json()) as { invitation: { id: string }; invitation_url: string };
}

// --- a1 ------------------------------------------------------------------------------------------

test('a1: an admin enrolls with a setup code from the Store host', async ({ page, enrollAdmin }) => {
  const admin = await enrollAdmin(page, { nickname: 'a1 admin key' });
  expect(admin.role).toBe('admin');
  await expect(signOutButton(page)).toBeVisible();
  await expect(page.getByRole('link', { name: 'Admin', exact: true })).toBeVisible();
});

test('a1: an invitation link enrolls a reviewer bound to the new passkey, and the token leaves the address bar', async ({
  page,
  adminApi,
  authenticator,
  newAuthPage,
  storeURL,
}) => {
  const email = uniqueEmail('a1-link');
  const issued = await invite(adminApi, email, 'reviewer', 'A1 Link');
  const token = issued.invitation_url.split('#')[1];
  expect(issued.invitation_url).toBe(`${storeURL}/enroll#${token}`);

  await page.goto(issued.invitation_url);
  // Store's /enroll#<token> form is rewritten to /#/enroll?token=<token> before the first render.
  await expect(page).toHaveURL(`${storeURL}/#/enroll?token=${token}`);
  await expect(page.getByRole('heading', { name: 'Accept your invitation' })).toBeVisible();
  await page.getByLabel('Name this authenticator (optional)').fill('a1 laptop');
  await page.getByRole('button', { name: 'Create passkey' }).click();
  await expect(page.getByRole('heading', { name: 'Authenticator enrolled' })).toBeVisible();
  await expect(page.getByText(`(${email}) is enrolled as`)).toBeVisible();
  // The one-time token is removed from the address bar once enrollment succeeds.
  expect(page.url()).not.toContain(token);
  expect(new URL(page.url()).hash).toBe('#/enroll');

  const credentials = await authenticator.credentials();
  expect(credentials).toHaveLength(1);
  await page.getByRole('button', { name: 'Go to calls' }).click();
  await expect(signOutButton(page)).toBeVisible();

  const own = await page.request.get(`${storeURL}/store/v1/auth/authenticators`);
  const items = ((await own.json()) as { items: { credential_id: string; nickname: string }[] }).items;
  const toB64url = (b64: string) => b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  expect(items.map((c) => c.credential_id)).toEqual([toB64url(credentials[0].credentialId)]);
  expect(items[0].nickname).toBe('a1 laptop');

  // The link is single-use: opening it again is refused with readable copy.
  const { page: second } = await newAuthPage();
  await second.goto(issued.invitation_url);
  await second.getByRole('button', { name: 'Create passkey' }).click();
  await expect(second.getByText('This invitation link is invalid, expired, revoked or already used.')).toBeVisible();
});

// --- a2 / e13 (known issue a) -------------------------------------------------------------------

test('a2/e13: right after enrollment the reviewer is signed in on any route, not only after "Go to calls"', async ({
  page,
  adminApi,
  storeURL,
}) => {
  await installClientSpread(page.context());
  const email = uniqueEmail('a2-reviewer');
  const issued = await invite(adminApi, email, 'reviewer', 'A2 Reviewer');
  await enrollViaUI(page, { invitationUrl: issued.invitation_url, email, enter: false });
  await expect(page.getByRole('heading', { name: 'Authenticator enrolled' })).toBeVisible();

  // Store already holds the session: the page's own cookie reads it.
  const session = await page.request.get(`${storeURL}/store/v1/auth/session`);
  expect(session.status()).toBe(200);

  // Following any in-app link (here: typing a route) must land signed in, not on the sign-in form.
  for (const route of ['#/queue', '#/account', '#/metrics', '#/calls']) {
    await page.goto(`${storeURL}/${route}`);
    await expect
      .soft(signOutButton(page), `after enrollment, ${route} should render signed in; it rendered the sign-in form instead`)
      .toBeVisible({ timeout: 5_000 });
    await expect.soft(signInEmail(page), `${route} must not ask the just-enrolled reviewer to sign in`).toHaveCount(0, { timeout: 2_000 });
  }

  // A full reload does pick the session up (GET /auth/session), which isolates the defect to the SPA's state.
  await page.reload();
  await expect(signOutButton(page)).toBeVisible();
});

// --- a3 ------------------------------------------------------------------------------------------

test('a3: a non-discoverable USB security key enrolls and signs in account-first (allowCredentials)', async ({ browser, adminApi, storeURL }) => {
  const securityKey = { transport: 'usb' as const, hasResidentKey: false };
  const email = uniqueEmail('a3-key');
  const issued = await invite(adminApi, email, 'reviewer', 'A3 Security Key');

  const enrollContext = await browser.newContext({ baseURL: storeURL });
  await installClientSpread(enrollContext);
  const enrollPage = await enrollContext.newPage();
  const enrollKey = await addVirtualAuthenticator(enrollPage, securityKey);
  const identity = await enrollViaUI(enrollPage, { invitationUrl: issued.invitation_url, email, nickname: 'a3 yubikey' });
  const [credential] = await enrollKey.credentials();
  expect(credential.isResidentCredential).toBe(false);
  await enrollContext.close();

  // Another browser with the same (non-resident) key: email first, then the key answers allowCredentials.
  const context = await browser.newContext({ baseURL: storeURL });
  await installClientSpread(context);
  const page = await context.newPage();
  const key = await addVirtualAuthenticator(page, securityKey);
  await key.addCredential(credential);
  const beginRequest = page.waitForRequest((r) => r.url().endsWith('/store/v1/auth/sign-in/begin'));
  const beginResponse = page.waitForResponse((r) => r.url().endsWith('/store/v1/auth/sign-in/begin'));
  await page.goto(`${storeURL}/#/calls`);
  await signInEmail(page).fill(email);
  await page.getByRole('button', { name: 'Continue' }).click();
  expect(((await beginRequest).postDataJSON() as { email: string }).email).toBe(email);
  const options = ((await (await beginResponse).json()) as { options: { allowCredentials: { id: string }[] } }).options;
  expect(options.allowCredentials.length).toBeGreaterThan(0);
  await expect(signOutButton(page)).toBeVisible();
  const session = await (await page.request.get(`${storeURL}/store/v1/auth/session`)).json();
  expect(session.account_id).toBe(identity.accountId);
  await context.close();
});

// --- a4 ------------------------------------------------------------------------------------------

test('a4: after a reload the CSRF token comes back from GET /auth/session, and a rejected token is re-read once', async ({
  createInvitedUser,
  storeURL,
}) => {
  const reviewer = await createInvitedUser('reviewer', { nickname: 'a4 key' });
  const page = reviewer.page;
  await page.goto(`${storeURL}/#/account`);
  await page.reload(); // the in-memory token is gone; only the cookie survives
  await expect(page.getByRole('heading', { name: 'Your account' })).toBeVisible();

  const seen: { method: string; path: string; csrf?: string; status?: number }[] = [];
  const track = (request: Request) => {
    const url = new URL(request.url());
    if (!url.pathname.startsWith('/store/v1/auth/')) return;
    seen.push({ method: request.method(), path: url.pathname, csrf: request.headers()['x-call1-csrf'] });
  };
  page.on('request', track);

  // Make Store reject the first rename's token, as it would after a session-key change.
  let tampered = false;
  await page.route('**/store/v1/auth/authenticators/*', async (route) => {
    if (route.request().method() === 'PATCH' && !tampered) {
      tampered = true;
      await route.continue({ headers: { ...route.request().headers(), 'x-call1-csrf': 'A'.repeat(43) } });
      return;
    }
    await route.continue();
  });

  const row = page.getByRole('listitem').filter({ hasText: 'a4 key' });
  await row.getByRole('button', { name: 'Rename' }).click();
  await page.getByLabel('Authenticator name').fill('a4 renamed');
  await page.getByRole('button', { name: 'Save' }).click();
  await expect(page.getByText('a4 renamed')).toBeVisible();

  const patches = seen.filter((r) => r.method === 'PATCH');
  expect(patches, JSON.stringify(seen)).toHaveLength(2);
  expect(patches[1].csrf, 'the retry carries the token re-read from GET /auth/session').toBeTruthy();
  const firstPatch = seen.findIndex((r) => r.method === 'PATCH');
  const reread = seen.findIndex((r, i) => i > firstPatch && r.method === 'GET' && r.path === '/store/v1/auth/session');
  expect(reread, JSON.stringify(seen)).toBeGreaterThan(firstPatch);
  expect(seen.filter((r) => r.method === 'PATCH')).toHaveLength(2); // exactly one retry
});

// --- a5 ------------------------------------------------------------------------------------------

test('a5: a reviewer sees no admin controls, and Store refuses admin and supervisor writes anyway', async ({
  createInvitedUser,
  apiAs,
  storeURL,
}) => {
  const reviewer = await createInvitedUser('reviewer');
  const page = reviewer.page;
  await expect(page.getByRole('link', { name: 'Admin', exact: true })).toHaveCount(0);
  await page.goto(`${storeURL}/#/admin/accounts`);
  await expect(page.getByText('Admin role required')).toBeVisible();

  const api = await apiAs(page);
  for (const [method, path, data] of [
    ['GET', '/admin/accounts', undefined],
    ['POST', '/admin/invitations', { email: uniqueEmail('a5-sneak'), display_name: 'Sneak', role: 'admin' }],
    ['POST', '/admin/installations', { label: 'a5-sneak', primary_host: false }],
    ['PUT', '/review-queue/rules/rule_a5', { expected_version: null }],
    ['GET', '/metrics/review-agreement', undefined],
  ] as const) {
    const response = await api.fetch(method, path, { data });
    expect(response.status(), `${method} ${path}`).toBe(403);
    expect(((await response.json()) as { code: string }).code).toBe('insufficient_role');
  }
});

// --- a8 / e9 -------------------------------------------------------------------------------------

test('a8/e9: the account page lists authenticators and sessions, and adding a second authenticator clears the prompt', async ({
  createInvitedUser,
  storeURL,
}) => {
  const reviewer = await createInvitedUser('reviewer', { nickname: 'a8 first key' });
  const { page, authenticator } = reviewer;

  // The dismissible banner prompts for a second authenticator on every other screen.
  await page.goto(`${storeURL}/#/calls`);
  await expect(page.getByText('You have one authenticator. If you lose it')).toBeVisible();

  await page.goto(`${storeURL}/#/account`);
  await expect(page.getByRole('heading', { name: 'Your account' })).toBeVisible();
  const firstRow = page.getByRole('listitem').filter({ hasText: 'a8 first key' });
  await expect(firstRow.getByText('Used for this session')).toBeVisible();
  await expect(page.getByText('This browser')).toBeVisible();
  await expect(page.getByText('You have one authenticator. Add a second one')).toBeVisible();

  // Step-up: confirm with the existing passkey, then register a new one. The existing authenticator
  // refuses to register again (it holds an excluded credential), so plug in a backup key for step 2.
  await page.getByRole('button', { name: 'Add another authenticator' }).click();
  await page.getByLabel('Name the new authenticator (optional)').fill('a8 backup key');
  await page.getByRole('button', { name: 'Start' }).click();
  await expect(page.getByText('1. Confirm with an existing authenticator — done')).toBeVisible();
  const registerButton = page.getByRole('button', { name: 'Register the new authenticator' });
  await expect(registerButton).toBeVisible();
  await authenticator.setAutomaticPresence(false);
  await authenticator.cdp.send('WebAuthn.addVirtualAuthenticator', {
    options: {
      protocol: 'ctap2',
      ctap2Version: 'ctap2_1',
      transport: 'usb',
      hasResidentKey: true,
      hasUserVerification: true,
      isUserVerified: true,
      automaticPresenceSimulation: true,
    },
  });
  await registerButton.click();
  await expect(page.getByText('Added “a8 backup key”. You can now sign in with it.')).toBeVisible();
  await authenticator.setAutomaticPresence(true);

  await expect(page.getByRole('listitem').filter({ hasText: 'a8 backup key' })).toBeVisible();
  await expect(page.getByText('You have one authenticator. Add a second one')).toHaveCount(0);
  const session = await (await page.request.get(`${storeURL}/store/v1/auth/session`)).json();
  expect(session.prompt_second_authenticator).toBe(false);
  await page.goto(`${storeURL}/#/calls`);
  await expect(page.getByRole('heading', { name: 'Calls' })).toBeVisible();
  await expect(page.getByText('You have one authenticator. If you lose it')).toHaveCount(0);
});

test('a8/e9: revoking another session from the account page signs that browser out', async ({ createInvitedUser, newAuthPage, signIn, storeURL }) => {
  const reviewer = await createInvitedUser('reviewer');
  const other = await newAuthPage();
  await signIn(other.page, reviewer.email);

  const page = reviewer.page;
  await page.goto(`${storeURL}/#/account`);
  const sessions = page.getByRole('listitem').filter({ hasText: 'Signed in' });
  await expect(sessions).toHaveCount(2);
  const otherRow = sessions.filter({ hasNotText: 'This browser' });
  await otherRow.getByRole('button', { name: 'Revoke' }).click();
  await expect(sessions).toHaveCount(1);

  // The other browser's next request finds its session gone and returns to sign-in.
  await other.page.goto(`${storeURL}/#/queue`);
  await other.page.reload();
  await expect(signInEmail(other.page)).toBeVisible();
});

// --- a9 ------------------------------------------------------------------------------------------

test('a9: the admin area issues an out-of-band invitation, lists the account, and manages installations and keys', async ({
  page,
  enrollAdmin,
  browser,
  storeURL,
}) => {
  await enrollAdmin(page, { nickname: 'a9 admin key' });

  // Invitations: the link is shown once, the invitation is listed as pending, then redeemed.
  await page.goto(`${storeURL}/#/admin/invitations`);
  await expect(page.getByRole('heading', { name: 'Admin' })).toBeVisible();
  const email = uniqueEmail('a9-invitee');
  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Display name').fill('A9 Invitee');
  await page.getByLabel('Role').selectOption('supervisor');
  await page.getByRole('button', { name: 'Issue link' }).click();
  const link = page.getByLabel('Invitation link');
  await expect(page.getByText('Invitation link — shown once')).toBeVisible();
  const invitationUrl = await link.inputValue();
  expect(invitationUrl.startsWith(`${storeURL}/enroll#`)).toBe(true);
  await expect(page.getByRole('listitem').filter({ hasText: email }).getByText('Pending', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'I have saved it — hide' }).click();
  await expect(page.getByText('Invitation link — shown once')).toHaveCount(0);

  const inviteeContext = await browser.newContext({ baseURL: storeURL });
  await installClientSpread(inviteeContext);
  const inviteePage = await inviteeContext.newPage();
  const invitee = await enrollViaUI(inviteePage, { invitationUrl, email });
  expect(invitee.role).toBe('supervisor');
  await inviteeContext.close();

  await page.getByLabel('Filter by status').selectOption('redeemed');
  await expect(page.getByRole('listitem').filter({ hasText: email }).getByText('Redeemed', { exact: true })).toBeVisible();

  // Accounts: the new account is active with one authenticator.
  await page.goto(`${storeURL}/#/admin/accounts`);
  const accountRow = page.getByRole('button').filter({ hasText: email });
  await expect(accountRow).toContainText('Active');
  await expect(accountRow).toContainText('1 authenticator');
  await expect(accountRow).toContainText('Supervisor'); // roles display capitalised

  // Installations and keys: the stack's installation is the primary host; register another,
  // issue it a key (token shown once, and it works), then revoke it (it stops working).
  await page.goto(`${storeURL}/#/admin/installations`);
  await expect(page.getByText('Primary host', { exact: true }).first()).toBeVisible();
  const label = `a9-inst-${Date.now().toString(36)}`;
  await page.getByRole('button', { name: 'Register installation' }).click();
  await page.getByLabel('Label').fill(label);
  await expect(page.getByRole('checkbox')).toBeDisabled(); // a primary host already exists
  await page.getByRole('button', { name: 'Register', exact: true }).click();
  const block = page.locator('div.rounded-md.border').filter({ has: page.getByText(label, { exact: true }) }).first();
  await expect(block).toBeVisible();
  await block.getByRole('button', { name: 'Issue key' }).click();
  await page.getByLabel('Key label').fill(`${label} key`);
  await page.getByRole('button', { name: 'Issue key' }).last().click();
  const secretLabel = `Service key for ${label}`;
  await expect(page.getByText(`${secretLabel} — shown once`)).toBeVisible();
  const token = await page.getByLabel(secretLabel).inputValue();
  expect(token).toMatch(/^c1sk_/);
  const works = await fetch(`http://127.0.0.1:${stackInfo().store_port}/store/v1/status/detail`, { headers: { Authorization: `Bearer ${token}` } });
  expect(works.status).toBe(200);

  const keyRow = block.locator('div').filter({ hasText: `${label} key` }).filter({ has: page.getByRole('button', { name: 'Revoke' }) }).last();
  await keyRow.getByRole('button', { name: 'Revoke' }).click();
  await page.getByLabel('Reason (recorded in the audit log)').fill('e2e a9');
  await page.getByRole('button', { name: 'Revoke' }).last().click();
  await expect.poll(async () => (await fetch(`http://127.0.0.1:${stackInfo().store_port}/store/v1/status/detail`, { headers: { Authorization: `Bearer ${token}` } })).status).toBe(401);
  // The stack's own key is untouched.
  expect((await serviceRequest('GET', '/status/detail')).status).toBe(200);
});
