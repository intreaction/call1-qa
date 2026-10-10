/**
 * c8 — the shared design system (docs/SplitBuild.md "Design principles"): Ink & Signal / Frost &
 * Ink theme tokens, IBM Plex Sans only, a sun/moon toggle persisted in localStorage `call1-theme`
 * and applied before first paint, and every status indicator carrying a text label as well as
 * color — checked identically across all three apps (Evaluate, Store console, Process console).
 *
 * Scope note: "no monospace" is checked on each app's landing/chrome view (sign-in or calls list
 * for Evaluate; the console pages as-is for the other two), which is what c8's "shared design
 * system" claim is actually about. Evaluate's deeper Workbench/Rubric Studio/Escalations views use
 * Tailwind's default `font-mono` utility (undefined in tailwind.config.js's `fontFamily.mono`, so
 * it falls back to the browser's monospace stack, not IBM Plex Sans) for tabular numbers and IDs;
 * that is a real gap against "IBM Plex Sans only" but belongs to the `evaluate_ui` inventory area,
 * not `consoles`, so it is not asserted here — see the run report.
 */
import { test, expect } from './fixtures';
import type { Page } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

type AppKey = 'evaluate' | 'store-console' | 'process-console';
const APPS: AppKey[] = ['evaluate', 'store-console', 'process-console'];

async function urlFor(app: AppKey, storeURL: string, processConsoleURL: string): Promise<string> {
  if (app === 'evaluate') return `${storeURL}/#/`;
  if (app === 'store-console') return `${storeURL}/console/`;
  return processConsoleURL;
}

/** Every app's theme toggle button has this exact accessible-name pattern (Header.tsx / ConsoleHeader.tsx). */
function themeToggle(page: Page) {
  return page.getByRole('button', { name: /Switch to (dark|light) theme/ });
}

for (const app of APPS) {
  test.describe(`c8 design system — ${app}`, () => {
    test(`c8: ${app} theme toggle persists to localStorage call1-theme and applies before paint`, async ({
      page,
      storeURL,
      processConsoleURL,
    }) => {
      const url = await urlFor(app, storeURL, processConsoleURL);
      await page.goto(url);

      const html = page.locator('html');
      const before = await html.getAttribute('data-theme');
      expect(before === 'dark' || before === 'light').toBe(true);

      await themeToggle(page).click();
      const after = await html.getAttribute('data-theme');
      expect(after).not.toBe(before);

      // Persisted under the documented key.
      const stored = await page.evaluate(() => window.localStorage.getItem('call1-theme'));
      expect(stored).toBe(after);

      // "Applies before paint": reload, and the boot script (a synchronous, blocking <script> in
      // <head> — see call1/store/static/*/index.html / process.html) must have already set
      // data-theme from localStorage by the time the DOM is parsed, before the React app (a
      // deferred `type="module"` script) runs. Checking it at `domcontentloaded`, before waiting
      // for the SPA to hydrate, is what distinguishes this from the (later) React effect in
      // useTheme.ts, which would otherwise cause a flash of the wrong theme.
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      const atParseTime = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
      expect(atParseTime).toBe(after);
      const colorScheme = await page.evaluate(() => document.documentElement.style.colorScheme);
      expect(colorScheme).toBe(after);

      // Following the OS scheme still works for a *different* browser context that never chose.
    });

    test(`c8: ${app} uses IBM Plex Sans with no monospace fallback, in both themes`, async ({ page, storeURL, processConsoleURL }) => {
      const url = await urlFor(app, storeURL, processConsoleURL);
      await page.goto(url);
      await page.waitForLoadState('networkidle');

      for (const theme of ['dark', 'light'] as const) {
        const current = await page.locator('html').getAttribute('data-theme');
        if (current !== theme) await themeToggle(page).click();
        await expect(page.locator('html')).toHaveAttribute('data-theme', theme);

        const bodyFont = await page.evaluate(() => getComputedStyle(document.body).fontFamily);
        expect(bodyFont, `${app} body font-family in ${theme}`).toMatch(/IBM Plex Sans/i);

        const offenders = await page.evaluate(() => {
          const bad: string[] = [];
          document.querySelectorAll('body *').forEach((el) => {
            const text = (el.textContent || '').trim();
            if (!text) return;
            // Only leaf-ish nodes: skip containers whose own font we'd double-report via children.
            if (el.children.length > 0) return;
            const ff = getComputedStyle(el as Element).fontFamily || '';
            if (/(^|[\s,])(monospace|ui-monospace|menlo|consolas|courier|sfmono)/i.test(ff)) {
              bad.push(`${el.tagName.toLowerCase()}.${(el as HTMLElement).className || ''} "${text.slice(0, 30)}": ${ff}`);
            }
          });
          return bad;
        });
        expect(offenders, `${app} (${theme}) elements not on IBM Plex Sans:\n${offenders.join('\n')}`).toEqual([]);
      }
    });

    test(`c8: ${app} status indicators carry a text label, not color alone`, async ({
      page,
      storeURL,
      processConsoleURL,
      enrollAdmin,
      ingestSample,
      waitUntilSettled,
    }, testInfo) => {
      if (app === 'evaluate') {
        // Needs at least one call with a status badge: sign in and ingest one.
        await enrollAdmin(page);
        const agent = `pw-c8-${testInfo.project.name}-${Date.now()}`;
        const receipt = await ingestSample('call_01_compliant', { agentId: agent });
        await waitUntilSettled(receipt.call_id);
        await page.goto(`${storeURL}/#/calls`);
        await expect(page.getByRole('heading', { name: 'Calls' })).toBeVisible();
      } else {
        const url = await urlFor(app, storeURL, processConsoleURL);
        await page.goto(url);
        // Wait for the pill-bearing async data (GET /process/api/session, GET /store/v1/status)
        // to land before scanning, instead of racing the first render.
        if (app === 'process-console') await expect(page.getByText('Console connected', { exact: true })).toBeVisible();
        else await expect(page.getByText('Reachable', { exact: true })).toBeVisible();
      }

      // StatusPill's exact anatomy, identical in every app's ui.tsx: a `span[aria-hidden="true"]`
      // color dot as the first child of a pill whose own text is the label.
      const pills = await page.evaluate(() => {
        const dots = Array.from(document.querySelectorAll('span[aria-hidden="true"]')).filter((el) =>
          (el.getAttribute('class') || '').includes('rounded-full'),
        );
        return dots.map((dot) => {
          const pill = dot.parentElement;
          const text = (pill?.textContent || '').trim();
          return { text, hasLabel: text.length > 0 };
        });
      });

      expect(pills.length, `${app}: expected at least one status pill on screen`).toBeGreaterThan(0);
      const unlabelled = pills.filter((p) => !p.hasLabel);
      expect(unlabelled, `${app}: status pills with color but no text label`).toEqual([]);
    });
  });
}

// -------------------------------------------------------------------------------------------
// Coverage gap: keyboard-only operation of the Store console (Tab order, focus-visible,
// Enter/Space activation). Signed out and un-ingested is the one state every run starts in and
// stays reachable in, so it's the deterministic baseline: HealthPanel and CoveragePanel never add
// a focusable control of their own (see HealthPanel.tsx / CoveragePanel.tsx), and the three
// admin-gated panels always render `SignedOutState`'s single link when there is no session
// (InstallationsPanel.tsx / ChangeFeedPanel.tsx / SearchEmbedderPanel.tsx, contract 1.2.0). That
// makes exactly five controls, in DOM order: the header's theme toggle, the Coverage panel's "By
// area" disclosure, and one "Sign in with a passkey on Evaluate" link per admin-gated panel.
// -------------------------------------------------------------------------------------------

/**
 * The admin-gated panels (InstallationsPanel, ChangeFeedPanel, SearchEmbedderPanel) share one
 * `useSession()` query (`['console-session']`, GET /store/v1/auth/session) and render their
 * `SignedOutState` link only once it resolves — so the last of these links becoming visible is the signal that every
 * control on the page (which all mount synchronously before that) is present and stable. Waiting
 * on it, rather than on elapsed time, is what keeps a Tab-count script deterministic under load.
 */
function signInLinks(page: Page) {
  return page.getByRole('link', { name: 'Sign in with a passkey on Evaluate' });
}

test.describe('c8 design system — store-console keyboard operation', () => {
  test('c8: every interactive control is reachable via Tab, in order, with nothing skipped or extra', async ({ page, storeURL }) => {
    await page.goto(`${storeURL}/console/`);
    await expect(signInLinks(page).nth(2)).toBeVisible(); // wait out hydration + the session check

    const focused = () =>
      page.evaluate(() => {
        const el = document.activeElement as HTMLElement | null;
        if (!el || el === document.body) return null;
        return { tag: el.tagName.toLowerCase(), ariaLabel: el.getAttribute('aria-label'), text: (el.textContent || '').trim().slice(0, 60) };
      });

    const steps: Array<Awaited<ReturnType<typeof focused>>> = [];
    for (let i = 0; i < 6; i++) {
      await page.keyboard.press('Tab');
      steps.push(await focused());
    }

    expect(steps[0]?.tag, 'control 1').toBe('button');
    expect(steps[0]?.ariaLabel, 'control 1 (theme toggle)').toMatch(/Switch to (dark|light) theme/);
    expect(steps[1]?.tag, 'control 2').toBe('button');
    expect(steps[1]?.text, 'control 2 (coverage disclosure)').toMatch(/^By area/);
    expect(steps[2]?.tag, 'control 3').toBe('a');
    expect(steps[2]?.text, 'control 3 (installations sign-in link)').toBe('Sign in with a passkey on Evaluate');
    expect(steps[3]?.tag, 'control 4').toBe('a');
    expect(steps[3]?.text, 'control 4 (change feed sign-in link)').toBe('Sign in with a passkey on Evaluate');
    expect(steps[4]?.tag, 'control 5').toBe('a');
    expect(steps[4]?.text, 'control 5 (search embedder sign-in link)').toBe('Sign in with a passkey on Evaluate');
    // A 6th Tab leaves the page's own content — there is nothing else on screen to reach.
    expect(steps[5], 'no 6th focusable control').toBeNull();
  });

  test('c8: Tab reaches every control with a visible focus indicator (focus-visible)', async ({ page, storeURL }) => {
    await page.goto(`${storeURL}/console/`);
    await expect(signInLinks(page).nth(2)).toBeVisible(); // wait out hydration + the session check

    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Tab');
      const outline = await page.evaluate(() => {
        const el = document.activeElement as HTMLElement | null;
        if (!el) return null;
        const style = getComputedStyle(el);
        return { style: style.outlineStyle, width: style.outlineWidth };
      });
      expect(outline?.style, `control ${i + 1} focus-visible outline style`).not.toBe('none');
      expect(outline?.width, `control ${i + 1} focus-visible outline width`).not.toBe('0px');
    }
  });

  test('c8: Enter activates the theme toggle from the keyboard', async ({ page, storeURL }) => {
    await page.goto(`${storeURL}/console/`);
    await expect(signInLinks(page).nth(2)).toBeVisible(); // wait out hydration + the session check
    await page.keyboard.press('Tab'); // control 1: theme toggle
    await expect(themeToggle(page)).toBeFocused();

    const html = page.locator('html');
    const before = await html.getAttribute('data-theme');
    await page.keyboard.press('Enter');

    await expect(html).not.toHaveAttribute('data-theme', before ?? '');
    const after = await html.getAttribute('data-theme');
    const stored = await page.evaluate(() => window.localStorage.getItem('call1-theme'));
    expect(stored, 'theme toggled via Enter persists like a click').toBe(after);
  });

  test('c8: Space activates the Coverage panel disclosure from the keyboard', async ({ page, storeURL }) => {
    await page.goto(`${storeURL}/console/`);
    await expect(signInLinks(page).nth(2)).toBeVisible(); // wait out hydration + the session check
    await page.keyboard.press('Tab'); // control 1: theme toggle
    await page.keyboard.press('Tab'); // control 2: coverage disclosure

    const disclosure = page.getByRole('button', { name: /^By area/ });
    await expect(disclosure).toBeFocused();
    await expect(disclosure).toHaveAttribute('aria-expanded', 'false');

    await page.keyboard.press(' ');
    await expect(disclosure).toHaveAttribute('aria-expanded', 'true');

    // Toggles back with a second press, like any disclosure button should.
    await page.keyboard.press(' ');
    await expect(disclosure).toHaveAttribute('aria-expanded', 'false');
  });

  test('c8: Enter follows the sign-in link from the keyboard, a real anchor rather than a click-only handler', async ({ page, storeURL }) => {
    await page.goto(`${storeURL}/console/`);
    await expect(signInLinks(page).nth(2)).toBeVisible(); // wait out hydration + the session check
    await page.keyboard.press('Tab'); // control 1: theme toggle
    await page.keyboard.press('Tab'); // control 2: coverage disclosure
    await page.keyboard.press('Tab'); // control 3: installations panel sign-in link

    const link = signInLinks(page).first();
    await expect(link).toBeFocused();
    await expect(link).toHaveAttribute('href', '/');

    await page.keyboard.press('Enter');
    await page.waitForLoadState('networkidle');
    await expect(page.getByRole('heading', { name: 'Sign in', exact: true })).toBeVisible();
  });
});

// -------------------------------------------------------------------------------------------
// Coverage gap: no cross-app check that the three UIs share identical theme tokens and icon set
// (the per-app c8 tests above check the theme toggle, font and status labels per app, not token
// identity across apps). All three read the same frontend/src/index.css at build time, so this
// derives the token list from that file rather than hand-maintaining a second copy that could
// drift from it.
// -------------------------------------------------------------------------------------------

/** Every `--custom-property` declared in index.css's `:root`/`[data-theme=...]` token blocks. */
function themeTokenNames(): string[] {
  const source = fs.readFileSync(path.join(HERE, '../src/index.css'), 'utf8');
  const blocks = [
    source.match(/:root,\s*\[data-theme="dark"\]\s*{([\s\S]*?)\n}/)?.[1] ?? '',
    source.match(/\[data-theme="light"\]\s*{([\s\S]*?)\n}/)?.[1] ?? '',
  ];
  const names = new Set<string>();
  for (const block of blocks) {
    for (const m of block.matchAll(/^\s*(--[a-zA-Z0-9-]+):/gm)) names.add(m[1]);
  }
  return [...names].sort();
}

const THEME_TOKENS = themeTokenNames();

test('c8: theme tokens and icon set are identical across Evaluate, Store console and Process console', async ({
  page,
  storeURL,
  processConsoleURL,
}) => {
  expect(THEME_TOKENS.length, 'index.css should declare theme tokens to check').toBeGreaterThan(10);

  for (const theme of ['dark', 'light'] as const) {
    const tokenValues: Record<AppKey, Record<string, string>> = { evaluate: {}, 'store-console': {}, 'process-console': {} };
    const icon: Record<AppKey, { kind: string; viewBox: string; strokeWidth: string; fill: string } | null> = {
      evaluate: null,
      'store-console': null,
      'process-console': null,
    };

    for (const app of APPS) {
      const url = await urlFor(app, storeURL, processConsoleURL);
      await page.goto(url);
      await page.waitForLoadState('networkidle');

      const current = await page.locator('html').getAttribute('data-theme');
      if (current !== theme) await themeToggle(page).click();
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);

      tokenValues[app] = await page.evaluate((names: string[]) => {
        const style = getComputedStyle(document.documentElement);
        const out: Record<string, string> = {};
        for (const n of names) out[n] = style.getPropertyValue(n).trim();
        return out;
      }, THEME_TOKENS);

      // Icon set identity: the theme toggle's own lucide-react icon (present, and identical
      // markup, in every app's header — ConsoleHeader.tsx / evaluate's Header.tsx / process's
      // Header.tsx). Same icon library, same version, same rendered geometry.
      const svg = themeToggle(page).locator('svg');
      icon[app] = await svg.evaluate((el) => {
        const cls = el.getAttribute('class') || '';
        const kind = cls.includes('lucide-moon') ? 'moon' : cls.includes('lucide-sun') ? 'sun' : cls;
        return {
          kind,
          viewBox: el.getAttribute('viewBox') || '',
          strokeWidth: el.getAttribute('stroke-width') || '',
          fill: el.getAttribute('fill') || '',
        };
      });
    }

    const mismatches: string[] = [];
    for (const tokenName of THEME_TOKENS) {
      const [a, b, c] = APPS.map((app) => tokenValues[app][tokenName]);
      if (!(a === b && b === c)) {
        mismatches.push(`${tokenName}: evaluate=${a ?? '(missing)'} store-console=${b ?? '(missing)'} process-console=${c ?? '(missing)'}`);
      }
    }
    expect(mismatches, `${theme} theme: token values differ across apps:\n${mismatches.join('\n')}`).toEqual([]);

    for (const app of APPS) {
      expect(icon[app], `${app} (${theme}): theme toggle should render a lucide icon`).not.toBeNull();
      expect(icon[app]!.kind, `${app} (${theme}) icon`).toBe(theme === 'light' ? 'sun' : 'moon');
    }
    const [e, sc, pc] = APPS.map((app) => icon[app]!);
    expect(sc, `${theme}: store-console icon should match evaluate's`).toEqual(e);
    expect(pc, `${theme}: process-console icon should match evaluate's`).toEqual(e);
  }
});

// -------------------------------------------------------------------------------------------
// Contact Signals v2 addition (docs/ContactSignalsV2.md section 17 "design": "the Signals page and
// the Workbench section in dark and light"). Written independently from the F4 implementation:
// the same three checks already run per-app above (theme toggle applies, IBM Plex Sans with no
// monospace fallback, status pills carry a text label), scoped to these two new views instead of
// every app's landing view.
// -------------------------------------------------------------------------------------------

test.describe('c8 design system — Signals page and Workbench contact-signals section', () => {
  test('c8 (signals): the Signals page and the Workbench section follow the theme toggle in both color schemes', async ({
    createInvitedUser,
    ingestSample,
    waitUntilSettled,
  }, testInfo) => {
    const admin = await createInvitedUser('admin');
    const receipt = await ingestSample('call_01_compliant', { agentId: `pw-c8-signals-${testInfo.project.name}-${Date.now()}` });
    await waitUntilSettled(receipt.call_id);

    for (const view of ['/#/signals', `/#/calls/${receipt.call_id}`]) {
      await admin.page.goto(view);
      const heading = view.startsWith('/#/signals') ? admin.page.getByRole('heading', { name: 'Signals', exact: true }) : admin.page.getByRole('heading', { name: 'Contact signals' });
      await expect(heading).toBeVisible();

      for (const theme of ['dark', 'light'] as const) {
        const html = admin.page.locator('html');
        if ((await html.getAttribute('data-theme')) !== theme) await themeToggle(admin.page).click();
        await expect(html).toHaveAttribute('data-theme', theme);
        await expect(heading).toBeVisible();

        const bodyFont = await admin.page.evaluate(() => getComputedStyle(document.body).fontFamily);
        expect(bodyFont, `Signals/Workbench (${view}) body font-family in ${theme}`).toMatch(/IBM Plex Sans/i);
      }
    }
  });
  test('c8 (signals): the current process is explained in text', async ({ createInvitedUser }) => {
    const admin = await createInvitedUser('admin');
    await admin.page.goto('/#/signals');
    const pipeline = admin.page.getByTestId('signals-pipeline-banner');
    await expect(pipeline).toContainText('Semantic similarity → Laya → Gemma');
    await expect(pipeline).toContainText('uncertain candidates go to Gemma');
    await expect(pipeline.getByRole('combobox')).toHaveCount(0);
  });

});
