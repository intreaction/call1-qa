/**
 * Playwright fixtures for the Call1 e2e suite. Import `test` and `expect` from here, not from
 * `@playwright/test`:
 *
 *   import { test, expect } from './fixtures';
 *
 * See e2e/README.md for the full fixture reference.
 */
import { test as base, expect, type Browser, type BrowserContext, type CDPSession, type Page } from '@playwright/test';
import fs from 'node:fs';
import * as h from './harness';

export { expect };
export * from './harness';

export type Role = 'reviewer' | 'supervisor' | 'admin';

/** An account enrolled (or signed in) through Evaluate in this run. */
export interface Identity {
  email: string;
  displayName: string;
  role: Role;
  accountId: string;
}

// --- the CDP virtual authenticator ---------------------------------------------------------

export interface VirtualAuthenticatorOptions {
  protocol?: 'ctap2' | 'u2f';
  transport?: 'usb' | 'nfc' | 'ble' | 'cable' | 'internal';
  hasResidentKey?: boolean;
  hasUserVerification?: boolean;
  isUserVerified?: boolean;
  automaticPresenceSimulation?: boolean;
}

export interface VirtualAuthenticator {
  cdp: CDPSession;
  authenticatorId: string;
  /** `WebAuthn.getCredentials`: every credential this authenticator holds, with its signCount. */
  credentials(): Promise<h.StoredCredential[]>;
  addCredential(credential: h.StoredCredential): Promise<void>;
  removeCredential(credentialId: string): Promise<void>;
  clearCredentials(): Promise<void>;
  /** false: the next ceremonies report no user verification (Store requires it). */
  setUserVerified(verified: boolean): Promise<void>;
  /** false: the authenticator stops answering (the ceremony waits, as if nobody touched the key). */
  setAutomaticPresence(enabled: boolean): Promise<void>;
}

const authenticators = new WeakMap<Page, Promise<VirtualAuthenticator>>();

/**
 * Give `page` a CDP virtual authenticator (WebAuthn.enable + addVirtualAuthenticator: ctap2,
 * internal transport, resident keys, user verification, automatic presence). One per page; a
 * second call returns the same one. The `page` fixture already has one.
 */
export function addVirtualAuthenticator(page: Page, options: VirtualAuthenticatorOptions = {}): Promise<VirtualAuthenticator> {
  const existing = authenticators.get(page);
  if (existing) return existing;
  const created = (async () => {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('WebAuthn.enable', { enableUI: false });
    const { authenticatorId } = await cdp.send('WebAuthn.addVirtualAuthenticator', {
      options: {
        protocol: 'ctap2',
        ctap2Version: 'ctap2_1',
        transport: 'internal',
        hasResidentKey: true,
        hasUserVerification: true,
        isUserVerified: true,
        automaticPresenceSimulation: true,
        ...options,
      },
    });
    const va: VirtualAuthenticator = {
      cdp,
      authenticatorId,
      credentials: async () => (await cdp.send('WebAuthn.getCredentials', { authenticatorId })).credentials as h.StoredCredential[],
      addCredential: async (credential) => {
        await cdp.send('WebAuthn.addCredential', { authenticatorId, credential: credential as never });
      },
      removeCredential: async (credentialId) => {
        await cdp.send('WebAuthn.removeCredential', { authenticatorId, credentialId });
      },
      clearCredentials: async () => {
        await cdp.send('WebAuthn.clearCredentials', { authenticatorId });
      },
      setUserVerified: async (isUserVerified) => {
        await cdp.send('WebAuthn.setUserVerified', { authenticatorId, isUserVerified });
      },
      setAutomaticPresence: async (enabled) => {
        await cdp.send('WebAuthn.setAutomaticPresenceSimulation', { authenticatorId, enabled });
      },
    };
    return va;
  })();
  authenticators.set(page, created);
  return created;
}

// --- per-context client address for the anonymous auth steps ------------------------------

const spread = new WeakMap<BrowserContext, string>();

/**
 * Store rate-limits enrollment begin (10/min) and sign-in begin (30/min) per client address, and
 * every browser here connects from 127.0.0.1. This gives the context its own synthetic address on
 * those two calls only (`X-Forwarded-For`, which uvicorn trusts from 127.0.0.1), so tests on the
 * shared stack do not trip each other's limits. Sessions are still created from 127.0.0.1.
 * Disable per test with `test.use({ spreadClientAddress: false })` (rate-limit tests).
 */
export async function installClientSpread(context: BrowserContext): Promise<string> {
  const known = spread.get(context);
  if (known) return known;
  const address = h.syntheticClientAddress();
  spread.set(context, address);
  await context.route(/\/store\/v1\/auth\/(enroll|sign-in)\/begin(\?|$)/, (route) =>
    route.continue({ headers: { ...route.request().headers(), 'x-forwarded-for': address } }),
  );
  return address;
}

// --- UI flows (module-level so worker fixtures can use them) --------------------------------

async function sessionIdentity(page: Page): Promise<Identity> {
  const api = await h.StoreApi.forRequest(page.request, h.stackInfo().store_url);
  return { email: api.session.email, displayName: api.session.display_name, role: api.session.role, accountId: api.session.account_id };
}

export interface EnrollViaUIOptions {
  /** Enroll with a setup code on `#/enroll` ... */
  setupCode?: string;
  /** ... or open an invitation link (`<store>/enroll#<token>`). */
  invitationUrl?: string;
  /** The account's email, for the credential store (so `signIn` can use it later). */
  email: string;
  nickname?: string;
  /** Default true: click "Go to calls" and wait for the signed-in shell. */
  enter?: boolean;
}

/** Drive Evaluate's enrollment screen with the page's virtual authenticator. */
export async function enrollViaUI(page: Page, options: EnrollViaUIOptions): Promise<Identity> {
  const stack = h.stackInfo();
  const auth = await addVirtualAuthenticator(page);
  const before = new Set((await auth.credentials()).map((c) => c.credentialId));
  if (options.setupCode) {
    await page.goto(`${stack.store_url}/#/enroll`);
    await expect(page.getByRole('heading', { name: 'Enroll with a setup code' })).toBeVisible();
    await page.getByLabel('Setup code').fill(options.setupCode);
  } else if (options.invitationUrl) {
    await page.goto(options.invitationUrl);
    await expect(page.getByRole('heading', { name: 'Accept your invitation' })).toBeVisible();
  } else {
    throw new Error('enrollViaUI needs a setupCode or an invitationUrl');
  }
  if (options.nickname) await page.getByLabel('Name this authenticator (optional)').fill(options.nickname);
  await page.getByRole('button', { name: 'Create passkey' }).click();
  await expect(page.getByRole('heading', { name: 'Authenticator enrolled' })).toBeVisible();
  const created = (await auth.credentials()).filter((c) => !before.has(c.credentialId));
  if (created.length === 0) throw new Error('enrollment finished but the virtual authenticator holds no new credential');
  h.saveCredentials(options.email, created);
  if (options.enter !== false) {
    await page.getByRole('button', { name: 'Go to calls' }).click();
    await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible();
    return sessionIdentity(page);
  }
  return { email: options.email, displayName: '', role: 'admin', accountId: '' };
}

export interface SetupCodeEnrollment {
  email?: string;
  displayName?: string;
  purpose?: h.SetupCodePurpose;
  targetAccountId?: string;
  nickname?: string;
}

/** `python -m call1.store setup-code` then Evaluate's enrollment screen. Always a real UI enrollment. */
export async function enrollWithSetupCode(page: Page, options: SetupCodeEnrollment = {}): Promise<Identity> {
  await installClientSpread(page.context());
  const email = options.email ?? h.uniqueEmail('admin');
  const displayName = options.displayName ?? 'E2E Admin';
  const code = await h.setupCode(email, displayName, { purpose: options.purpose ?? 'first_admin', targetAccountId: options.targetAccountId });
  return enrollViaUI(page, { setupCode: code, email, nickname: options.nickname });
}

/**
 * A new admin enrolled through Evaluate with a CLI setup code. The first call on the stack uses a
 * `first_admin` code; later calls use `break_glass` codes without a target (each creates a new
 * admin account and writes a `break_glass_used` audit event), because `first_admin` is refused once
 * an admin exists.
 */
export async function enrollAdminWithUI(page: Page, options: Omit<SetupCodeEnrollment, 'purpose' | 'targetAccountId'> = {}): Promise<Identity & { purpose: h.SetupCodePurpose }> {
  if (!h.readState('first-admin')) {
    const first = await h.withLock('first-admin', async () => {
      if (h.readState('first-admin')) return null;
      const identity = await enrollWithSetupCode(page, { ...options, purpose: 'first_admin' });
      h.writeState('first-admin', identity);
      return { ...identity, purpose: 'first_admin' as const };
    });
    if (first) return first;
  }
  const identity = await enrollWithSetupCode(page, { ...options, purpose: 'break_glass' });
  return { ...identity, purpose: 'break_glass' };
}

/**
 * Sign `page` in as `email` through Evaluate's sign-in screen. The account's credential (saved when
 * it enrolled in this run) is copied into the page's virtual authenticator first; the updated
 * signature counter is saved back. Signs out a different signed-in account first.
 */
export async function signInWithUI(page: Page, email: string): Promise<Identity> {
  const stack = h.stackInfo();
  const auth = await addVirtualAuthenticator(page);
  await installClientSpread(page.context());
  return h.withLock(`credential-${email.toLowerCase()}`, async () => {
    const stored = h.readCredentials(email);
    if (stored.length === 0) throw new Error(`no credential enrolled for ${email} in this run`);
    const present = new Map((await auth.credentials()).map((c) => [c.credentialId, c]));
    for (const credential of stored) {
      const have = present.get(credential.credentialId);
      if (!have) {
        await auth.addCredential(credential);
      } else if (have.signCount < credential.signCount) {
        await auth.removeCredential(credential.credentialId);
        await auth.addCredential(credential);
      }
    }
    await page.goto(`${stack.store_url}/#/calls`);
    const emailField = page.getByLabel('Account email');
    const signOut = page.getByRole('button', { name: 'Sign out' });
    await expect(emailField.or(signOut)).toBeVisible();
    if (await signOut.isVisible()) {
      const current = await sessionIdentity(page);
      if (current.email.toLowerCase() === email.toLowerCase()) return current;
      await signOut.click();
      await expect(emailField).toBeVisible();
    }
    await emailField.fill(email);
    await page.getByRole('button', { name: 'Continue' }).click();
    await expect(signOut).toBeVisible();
    const ids = new Set(stored.map((c) => c.credentialId));
    h.saveCredentials(email, (await auth.credentials()).filter((c) => ids.has(c.credentialId)));
    return sessionIdentity(page);
  });
}

export interface AuthPage {
  context: BrowserContext;
  page: Page;
  authenticator: VirtualAuthenticator;
}

async function openAuthPage(
  browser: Browser,
  options: { colorScheme?: 'light' | 'dark' | 'no-preference' | null; spreadClientAddress?: boolean } = {},
): Promise<AuthPage> {
  const context = await browser.newContext({ baseURL: h.stackInfo().store_url, colorScheme: options.colorScheme ?? undefined });
  if (options.spreadClientAddress !== false) await installClientSpread(context);
  const page = await context.newPage();
  const authenticator = await addVirtualAuthenticator(page);
  return { context, page, authenticator };
}

const STACK_ADMIN_EMAIL = 'stack-admin@e2e.test';

export interface InvitedUser extends Identity, AuthPage {
  invitationId: string;
  invitationUrl: string;
}

// --- fixtures ----------------------------------------------------------------------------

type TestFixtures = {
  /** The running stack (ports, URLs, tokens, data dirs). */
  stack: h.StackInfo;
  /** `http://localhost:<port>`: Evaluate at `/`, the API at `/store/v1`. Also the default baseURL. */
  storeURL: string;
  /** `http://127.0.0.1:<port>`: the Process console and `/process/api`. */
  processURL: string;
  /** The Process console credential (send as `X-Call1-Console-Token`). */
  consoleToken: string;
  /** `processURL` + `/#console_token=…`: the console, signed in. */
  processConsoleURL: string;
  /** Option (default true): see `installClientSpread`. */
  spreadClientAddress: boolean;
  /** The default `page`'s virtual authenticator. */
  authenticator: VirtualAuthenticator;
  /** A fresh browser context + page with its own virtual authenticator (closed after the test). */
  newAuthPage: (options?: { colorScheme?: 'light' | 'dark' }) => Promise<AuthPage>;
  /** A new admin enrolled through Evaluate with a CLI setup code (default page: the test's `page`). */
  enrollAdmin: (page?: Page, options?: Omit<SetupCodeEnrollment, 'purpose' | 'targetAccountId'>) => Promise<Identity & { purpose: h.SetupCodePurpose }>;
  /** Any setup-code enrollment through the UI (`purpose`, `targetAccountId` for break-glass re-enrollment). */
  enrollWithSetupCode: (page: Page, options?: SetupCodeEnrollment) => Promise<Identity>;
  /** Sign `page` in as an account enrolled earlier in this run. */
  signIn: (page: Page, email: string) => Promise<Identity>;
  /** The stack admin invites `role` over the API; a second browser context enrolls from the link. */
  createInvitedUser: (role?: Role, options?: { email?: string; displayName?: string; nickname?: string }) => Promise<InvitedUser>;
  /** Upload a sample (or any file) through Process's API. */
  ingestSample: typeof h.ingestSample;
  /** Wait for a call's conversation to settle (Store's JobGroupProgress, service key). */
  waitUntilSettled: typeof h.waitUntilSettled;
  /** `/store/v1` as whoever is signed in on `page` (cookie + CSRF). */
  apiAs: (page: Page) => Promise<h.StoreApi>;
  /** Process's loopback API with the console token. */
  processApi: {
    get: (path: string, options?: { token?: boolean | string }) => Promise<Response>;
    post: (path: string, json?: unknown, options?: { token?: boolean | string }) => Promise<Response>;
  };
};

type WorkerFixtures = {
  /** `/store/v1` as the stack admin (one account for the whole run; storage state reused). */
  adminApi: h.StoreApi;
};

export const test = base.extend<TestFixtures, WorkerFixtures>({
  stack: async ({}, use) => use(h.stackInfo()),
  storeURL: async ({ stack }, use) => use(stack.store_url),
  processURL: async ({ stack }, use) => use(stack.process_url),
  consoleToken: async ({ stack }, use) => use(stack.console_token),
  processConsoleURL: async ({ stack }, use) => use(stack.process_console_url),
  baseURL: async ({}, use) => use(h.stackInfo().store_url),
  spreadClientAddress: [true, { option: true }],

  page: async ({ page, spreadClientAddress }, use) => {
    await addVirtualAuthenticator(page);
    if (spreadClientAddress) await installClientSpread(page.context());
    await use(page);
  },
  authenticator: async ({ page }, use) => use(await addVirtualAuthenticator(page)),

  newAuthPage: async ({ browser, colorScheme, spreadClientAddress }, use) => {
    const opened: BrowserContext[] = [];
    await use(async (options = {}) => {
      const made = await openAuthPage(browser, { colorScheme: options.colorScheme ?? colorScheme, spreadClientAddress });
      opened.push(made.context);
      return made;
    });
    for (const context of opened) await context.close();
  },

  enrollAdmin: async ({ page }, use) => use((target, options) => enrollAdminWithUI(target ?? page, options)),
  enrollWithSetupCode: async ({}, use) => use((page, options) => enrollWithSetupCode(page, options)),
  signIn: async ({}, use) => use((page, email) => signInWithUI(page, email)),

  adminApi: [
    async ({ browser, playwright }, use) => {
      const stack = h.stackInfo();
      const api = await h.withLock('stack-admin', async () => {
        const statePath = h.statePath('stack-admin.storage.json');
        if (fs.existsSync(statePath)) {
          const request = await playwright.request.newContext({ storageState: statePath });
          try {
            return await h.StoreApi.forRequest(request, stack.store_url);
          } catch {
            await request.dispose(); // the session ended: sign in again below
          }
        }
        const { context, page } = await openAuthPage(browser);
        try {
          if (h.readState('stack-admin') && h.readCredentials(STACK_ADMIN_EMAIL).length > 0) {
            await signInWithUI(page, STACK_ADMIN_EMAIL);
          } else {
            const identity = await enrollAdminWithUI(page, { email: STACK_ADMIN_EMAIL, displayName: 'E2E Stack Admin', nickname: 'stack admin key' });
            h.writeState('stack-admin', identity);
          }
          await context.storageState({ path: statePath });
        } finally {
          await context.close();
        }
        const request = await playwright.request.newContext({ storageState: statePath });
        return h.StoreApi.forRequest(request, stack.store_url);
      });
      await use(api);
      await api.request.dispose();
    },
    { scope: 'worker' },
  ],

  createInvitedUser: async ({ adminApi, browser, colorScheme, spreadClientAddress }, use) => {
    const opened: BrowserContext[] = [];
    await use(async (role = 'reviewer', options = {}) => {
      const email = options.email ?? h.uniqueEmail(role);
      const displayName = options.displayName ?? `E2E ${role}`;
      const response = await adminApi.post('/admin/invitations', { email, display_name: displayName, role });
      if (!response.ok()) throw new Error(`createInvitation answered ${response.status()}: ${await response.text()}`);
      const issued = (await response.json()) as { invitation: { id: string }; invitation_url: string };
      const made = await openAuthPage(browser, { colorScheme, spreadClientAddress });
      opened.push(made.context);
      const identity = await enrollViaUI(made.page, { invitationUrl: issued.invitation_url, email, nickname: options.nickname });
      return { ...identity, ...made, invitationId: issued.invitation.id, invitationUrl: issued.invitation_url };
    });
    for (const context of opened) await context.close();
  },

  ingestSample: async ({}, use) => use(h.ingestSample),
  waitUntilSettled: async ({}, use) => use(h.waitUntilSettled),
  apiAs: async ({ storeURL }, use) => use((page) => h.StoreApi.forRequest(page.request, storeURL)),
  processApi: async ({}, use) =>
    use({
      get: (p, options = {}) => h.processRequest('GET', p, { token: options.token }),
      post: (p, json, options = {}) => h.processRequest('POST', p, { json, token: options.token }),
    }),
});
