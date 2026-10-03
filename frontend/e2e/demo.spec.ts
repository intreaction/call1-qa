// Demo mode (task item 1 and 4, 2026-09-25 — "demo-ready for a class presentation; auth does not
// need to be fully functional. Keep the real passkey code intact; add a clearly-labelled,
// localhost-only demo mode on top."): `GET /demo/status`, the sign-in screen's "Continue as Demo
// …" buttons above the untouched passkey form, the header's "Demo mode" badge, and the account
// page's persona switch (`call1/store/auth/demo.py`, `frontend/src/apps/evaluate/api/demo.ts`).
//
// Demo mode is off by default (CALL1_STORE_DEMO=1 turns it on) and the shared Playwright stack
// runs with it off, so every other spec's real passkey flows are unaffected — this file starts its
// own private, Store-only stack (no Process needed: demo sign-in creates its own accounts and the
// nav/badge checks need no call data) with demo mode on, and a second, ordinary check against the
// shared stack confirms demo mode is genuinely off there.
import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { startPrivateStack, type PrivateStack } from './private-stack';

const PERSONAS = [
  { key: 'admin', label: 'Demo Admin', role: 'admin' },
  { key: 'supervisor', label: 'Demo Supervisor', role: 'supervisor' },
  { key: 'reviewer', label: 'Demo Reviewer', role: 'reviewer' },
] as const;

const demoBadge = (page: Page) => page.getByRole('banner').getByText('Demo mode', { exact: true });
const signOutButton = (page: Page) => page.getByRole('banner').getByRole('button', { name: 'Sign out' });
// exact: true — a non-exact "Admin" also matches the account link's "Demo Admin admin" name.
const adminNav = (page: Page) => page.getByRole('banner').getByRole('link', { name: 'Admin', exact: true });

test.describe('demo mode', () => {
  let demo: PrivateStack;

  test.beforeAll(async () => {
    demo = await startPrivateStack({ name: 'pw-demo', storeEnv: { CALL1_STORE_DEMO: '1' }, noProcess: true });
  });
  test.afterAll(async () => {
    await demo.close();
  });

  test('demo: GET /demo/status reports demo mode on with the three personas', async () => {
    const res = await fetch(`${demo.info.store_url}/demo/status`);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { demo: boolean; label: string; personas: { persona: string; role: string }[] };
    expect(body.demo).toBe(true);
    expect(body.label.length).toBeGreaterThan(0);
    expect(new Set(body.personas.map((p) => p.persona))).toEqual(new Set(['admin', 'supervisor', 'reviewer']));
  });

  test('demo: the sign-in screen shows "Continue as Demo …" buttons above the untouched passkey form, and the badge', async ({
    browser,
  }) => {
    const context = await browser.newContext({ baseURL: demo.info.store_url });
    const page = await context.newPage();
    try {
      await page.goto('/#/sign-in');

      // The demo buttons, prominent, above the form.
      for (const persona of PERSONAS) {
        await expect(page.getByRole('button', { name: `Continue as ${persona.label}`, exact: true })).toBeVisible();
      }

      // The real passkey form is still there, unchanged: account-first email field, then Continue.
      await expect(page.getByLabel('Account email')).toBeVisible();
      const passkeyContinue = page.getByRole('button', { name: 'Continue', exact: true });
      await expect(passkeyContinue).toBeVisible();

      // The demo buttons come first in the DOM (above the form), per task item 1.
      const demoButtonBox = await page.getByRole('button', { name: `Continue as ${PERSONAS[0].label}`, exact: true }).boundingBox();
      const passkeyBox = await passkeyContinue.boundingBox();
      expect(demoButtonBox).not.toBeNull();
      expect(passkeyBox).not.toBeNull();
      expect(demoButtonBox!.y).toBeLessThan(passkeyBox!.y);

      // The header badge, visible even signed out.
      await expect(demoBadge(page)).toBeVisible();
    } finally {
      await context.close();
    }
  });

  for (const persona of PERSONAS) {
    test(`demo: sign in as ${persona.label} — role-appropriate nav, badge visible`, async ({ browser }) => {
      const context = await browser.newContext({ baseURL: demo.info.store_url });
      const page = await context.newPage();
      try {
        await page.goto('/#/sign-in');
        await page.getByRole('button', { name: `Continue as ${persona.label}`, exact: true }).click();

        await expect(signOutButton(page)).toBeVisible({ timeout: 15_000 });
        await expect(page).toHaveURL(/#\/calls/);

        // The badge persists once signed in.
        await expect(demoBadge(page)).toBeVisible();
        // A persona's only authenticator is Store's keyless placeholder, so the one-authenticator
        // nag (App.tsx SecondAuthenticatorBanner) is hidden for it.
        await expect(page.getByText('You have one authenticator.')).toHaveCount(0);

        // The role pill next to the account link shows the persona's real role, and Admin nav is
        // present only for the admin persona (Header.tsx: can('manage_accounts')).
        await expect(page.getByRole('banner').getByText(persona.role, { exact: true })).toBeVisible();
        if (persona.role === 'admin') {
          await expect(adminNav(page)).toBeVisible();
        } else {
          await expect(adminNav(page)).toHaveCount(0);
        }

        // Every role reaches the shared views (Header.NAV has no role gating below admin).
        for (const label of ['Calls', 'Rubrics', 'Queue', 'Escalations', 'Metrics']) {
          await expect(page.getByRole('banner').getByRole('link', { name: label })).toBeVisible();
        }
      } finally {
        await context.close();
      }
    });
  }

  test('demo: the account page switches persona without a passkey, and marks the current one', async ({ browser }) => {
    const context = await browser.newContext({ baseURL: demo.info.store_url });
    const page = await context.newPage();
    try {
      await page.goto('/#/sign-in');
      await page.getByRole('button', { name: 'Continue as Demo Reviewer', exact: true }).click();
      await expect(signOutButton(page)).toBeVisible({ timeout: 15_000 });

      await page.goto('/#/account');
      await expect(page.getByRole('heading', { name: 'Demo mode' })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Signed in as Demo Reviewer' })).toBeVisible();

      await page.getByRole('button', { name: 'Continue as Demo Supervisor', exact: true }).click();
      await expect(page.getByRole('button', { name: 'Signed in as Demo Supervisor' })).toBeVisible({ timeout: 15_000 });
      // Switching persona is a fresh session: the profile card reflects it at once.
      await expect(page.getByText('demo.supervisor@call1-demo.example')).toBeVisible();
    } finally {
      await context.close();
    }
  });

  test('demo: /demo/sign-in refuses an unknown persona', async () => {
    const res = await fetch(`${demo.info.store_url}/demo/sign-in`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ persona: 'ceo' }),
    });
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);
  });
});

test.describe('demo mode is off by default', () => {
  // The shared Playwright stack (every other spec file) runs without CALL1_STORE_DEMO, so this
  // confirms real passkey sign-in is genuinely unaffected: no demo buttons, no badge, 404.
  test('demo: off on the shared stack — no buttons, no badge, /demo/status is 404', async ({ page, storeURL }) => {
    await page.goto('/#/sign-in');
    await expect(page.getByLabel('Account email')).toBeVisible();
    for (const persona of PERSONAS) {
      await expect(page.getByRole('button', { name: `Continue as ${persona.label}`, exact: true })).toHaveCount(0);
    }
    await expect(page.getByRole('banner').getByText('Demo mode', { exact: true })).toHaveCount(0);

    const res = await page.request.get(`${storeURL}/demo/status`);
    expect(res.status()).toBe(404);
  });
});
