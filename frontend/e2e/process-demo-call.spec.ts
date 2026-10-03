import { test, expect } from './fixtures';
import { startPrivateStack } from './private-stack';

test('live demo graph: runs fresh calls, shows real terminal status, fits mobile and respects credentials', async ({ page }) => {
  const priv = await startPrivateStack({ name: 'pw-live-demo', processEnv: { CALL1_STORE_DEMO: '1' } });
  try {
    await page.goto(priv.info.process_console_url);
    await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Import' }).click();
    const studio = page.getByTestId('demo-call-studio');
    await expect(studio.getByTestId('live-processing-graph')).toHaveCount(0);
    await studio.getByRole('button', { name: 'Process demo call', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Pipeline', exact: true })).toBeVisible();
    const graph = page.getByTestId('live-processing-graph');
    await expect(graph.getByTestId('processing-stage')).toHaveCount(8);
    await expect(graph.getByRole('heading', { name: 'Your call is ready.' })).toBeVisible({ timeout: 45_000 });
    const first = await (await priv.process('GET', '/conversations')).json();
    const row = page.getByRole('button', { name: /AppTek · stock inquiry/ }).first();
    const card = page.locator('section').filter({ has: row });
    await graph.getByRole('button', { name: 'Inspect Score the call jobs' }).click();
    await expect(card.getByRole('region', { name: 'Score the call details' })).toContainText('QA criterion');
    await expect(graph.getByRole('button', { name: 'Inspect Score the call jobs' })).toHaveAttribute('aria-expanded', 'true');
    await expect(card.getByRole('region', { name: 'Score the call details' }).getByTestId('job-diagnostics').first()).toContainText('Attempt 1 started');
    await graph.getByRole('button', { name: 'Inspect Score the call jobs' }).click();
    await expect(card.getByRole('region', { name: 'Score the call details' })).toHaveCount(0);
    await graph.getByRole('button', { name: 'Show all details' }).click();
    await expect(graph.getByRole('region')).toHaveCount(8);
    await graph.getByRole('button', { name: 'Hide all details' }).click();
    await expect(graph.getByRole('region')).toHaveCount(0);
    await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Import' }).click();
    await studio.getByRole('button', { name: 'Process demo call', exact: true }).click();
    await expect(graph.getByRole('heading', { name: 'Your call is ready.' })).toBeVisible({ timeout: 45_000 });
    const second = await (await priv.process('GET', '/conversations')).json();
    expect(second.items[0].conversation_id).not.toEqual(first.items[0].conversation_id);
    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
    await page.evaluate(() => sessionStorage.clear());
    await page.reload();
    await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Import' }).click();
    await expect(studio.getByRole('button', { name: 'Process demo call', exact: true })).toBeDisabled();
  } finally { await priv.close(); }
});
