// Evaluate semantic call search (⌘K / "Search calls") and the mobile header (BUG D4).
//
// The shared e2e stack may run without the local search embedder, so the search route is answered
// by `page.route` with hits on a real ingested call: the spec covers Evaluate's dialog (ranked hits
// labelled with the call's agent, the matched snippet and timestamp, keyboard navigation and the
// Workbench deep link at `?turn=<n>`), not Store's ranking, which tests/store covers. One test runs
// the real route and accepts either answer (hits, or 503 search_unavailable shown as a message).
import { test, expect } from './fixtures';
import type { Page } from '@playwright/test';

const SEARCH_ROUTE = '**/store/v1/search/semantic';

async function mockSearch(page: Page, callId: string) {
  await page.route(SEARCH_ROUTE, async (route) => {
    const body = route.request().postDataJSON() as { query: string };
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        query: body.query,
        count: 2,
        embedding_scheme: 'e2e',
        calls_needing_reembedding: 0,
        results: [
          { call_id: callId, turn_id: 2, speaker: 'AGENT', start_time: 11.46, end_time: 18.11, text: 'Could you please verify your account number and date of birth?', similarity_score: 0.81 },
          { call_id: callId, turn_id: 4, speaker: 'AGENT', start_time: 28.14, end_time: 33.99, text: 'Thank you for verifying. Your deposit cleared this morning.', similarity_score: 0.66 },
        ],
      }),
    });
  });
}

test('search calls: ⌘K opens it, hits show the call, snippet and time, and Enter opens the Workbench at that turn', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const page = reviewer.page;
  const agent = `pw-search-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);
  await mockSearch(page, receipt.call_id);

  await page.goto('/#/calls');
  await expect(page.getByRole('heading', { name: 'Calls', exact: true })).toBeVisible();

  // The header button opens it; Escape closes it and returns focus to the button.
  const button = page.getByRole('button', { name: 'Search calls' });
  await button.click();
  const dialog = page.getByRole('dialog', { name: 'Search calls' });
  await expect(dialog).toBeVisible();
  const input = dialog.getByRole('combobox', { name: 'Search call transcripts' });
  await expect(input).toBeFocused();
  await expect(dialog.getByRole('button', { name: 'Stanley cup in stock' })).toBeVisible(); // examples before typing
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(button).toBeFocused();

  // The shortcut opens it from anywhere.
  await page.locator('body').click();
  await page.keyboard.press('ControlOrMeta+k');
  await expect(dialog).toBeVisible();
  await input.fill('verify identity');
  const options = dialog.getByRole('option');
  await expect(options).toHaveCount(2);
  await expect(options.first().getByTestId('search-hit-call')).toHaveText(agent);
  await expect(options.first()).toContainText('0:11');
  await expect(options.first()).toContainText('Agent');
  await expect(options.first().locator('mark').first()).toHaveText(/verify/i);
  await expect(options.first()).toHaveAttribute('aria-selected', 'true');
  await expect(options.first()).toHaveAttribute('href', `#/calls/${receipt.call_id}?turn=2`);

  await input.press('ArrowDown');
  await expect(options.nth(1)).toHaveAttribute('aria-selected', 'true');
  await expect(input).toHaveAttribute('aria-activedescendant', (await options.nth(1).getAttribute('id')) ?? '');
  await input.press('Enter');
  await expect(dialog).toBeHidden();
  await expect(page).toHaveURL(new RegExp(`#/calls/${receipt.call_id}\\?turn=4$`));
  await expect(page.getByTestId('workbench-header').getByTestId('call-agent')).toHaveText(agent);
});

test('search calls: empty results and an unavailable search model are explained', async ({ createInvitedUser }) => {
  const reviewer = await createInvitedUser('reviewer');
  const page = reviewer.page;
  let unavailable = false;
  await page.route(SEARCH_ROUTE, async (route) => {
    const { query } = route.request().postDataJSON() as { query: string };
    if (unavailable) {
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ code: 'search_unavailable', message: 'Search is unavailable', details: { reason: 'not_installed' }, retryable: false, request_id: 'e2e' }),
      });
      return;
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ query, count: 0, results: [], embedding_scheme: 'e2e', calls_needing_reembedding: 0 }) });
  });
  await page.goto('/#/calls');
  await page.getByRole('button', { name: 'Search calls' }).click();
  const dialog = page.getByRole('dialog', { name: 'Search calls' });
  const input = dialog.getByRole('combobox', { name: 'Search call transcripts' });
  await input.fill('nothing like this was said');
  await expect(dialog.getByText(/No matching moments for/)).toBeVisible();
  unavailable = true;
  await input.fill('something else entirely');
  await expect(dialog.getByRole('alert')).toContainText('Search is not available on this Store');
});

test('search calls: the real route answers with hits or an explained 503', async ({ createInvitedUser }) => {
  const reviewer = await createInvitedUser('reviewer');
  const page = reviewer.page;
  await page.goto('/#/calls');
  await page.getByRole('button', { name: 'Search calls' }).click();
  const dialog = page.getByRole('dialog', { name: 'Search calls' });
  await dialog.getByRole('combobox', { name: 'Search call transcripts' }).fill('verify the account');
  await expect(dialog.getByRole('option').first().or(dialog.getByRole('alert')).or(dialog.getByText(/No matching moments/))).toBeVisible({ timeout: 20_000 });
});

test('mobile header at 390px: no horizontal page scroll, nav scrolls on its own row, controls stay on screen', async ({ createInvitedUser }) => {
  const reviewer = await createInvitedUser('reviewer');
  const page = reviewer.page;
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/#/calls');
  await expect(page.getByRole('heading', { name: 'Calls', exact: true })).toBeVisible();

  const widths = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, viewport: window.innerWidth }));
  expect(widths.scroll, 'no horizontal page scroll').toBeLessThanOrEqual(widths.viewport);

  for (const name of ['Sign out', 'Search calls']) {
    const box = await page.getByRole('button', { name }).boundingBox();
    expect(box, `${name} is laid out`).not.toBeNull();
    expect(box!.x + box!.width, `${name} is on screen`).toBeLessThanOrEqual(390);
  }
  const toggle = await page.getByRole('button', { name: /^Switch to (light|dark) theme$/ }).boundingBox();
  expect(toggle!.x + toggle!.width).toBeLessThanOrEqual(390);

  // The nav has real width on its own row and its links scroll into view.
  const nav = page.getByRole('navigation', { name: 'Evaluate' });
  const navBox = await nav.boundingBox();
  expect(navBox!.width).toBeGreaterThan(300);
  const metrics = nav.getByRole('link', { name: 'Metrics' });
  await metrics.scrollIntoViewIfNeeded();
  await metrics.click();
  await expect(page).toHaveURL(/#\/metrics$/);

  // The search dialog fits the phone too.
  await page.getByRole('button', { name: 'Search calls' }).click();
  const dialogBox = await page.getByRole('dialog', { name: 'Search calls' }).boundingBox();
  expect(dialogBox!.x).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.x + dialogBox!.width).toBeLessThanOrEqual(390);
});
