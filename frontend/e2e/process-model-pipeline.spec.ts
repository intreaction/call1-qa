import { test, expect } from './fixtures';
import { startPrivateStack } from './private-stack';

test('model graph: inspect stages, open fine-tuning settings, and fit a phone', async ({ page }) => {
  // The shared fixture's URL is Store; use a private Process so the test has its own session.
  const priv = await startPrivateStack({ name: 'pw-model-graph' });
  try {
    await page.goto(priv.info.process_console_url);
    await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Models' }).click();
    const graph = page.getByTestId('model-pipeline');
    await expect(graph).toContainText('Call recording');
    await expect(graph.getByRole('button')).toHaveCount(10);
    await expect(graph).toContainText('Ready to review in Evaluate');
    await graph.getByRole('button', { name: 'Inspect Rubric QA models' }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('pipeline-model-details')).toContainText('Call1 Included');
    await expect(graph.getByRole('button', { name: 'Inspect Rubric QA models' })).toHaveAttribute('aria-expanded', 'true');
    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
    await page.getByRole('button', { name: 'Fine-tuning settings' }).click();
    await expect(page.getByRole('heading', { name: 'Settings', exact: true })).toBeVisible();
  } finally { await priv.close(); }
});

test('demo stack: pull editions, add and pause a private layer, reset lineage, and persist without changing live models', async ({ page }) => {
  const priv = await startPrivateStack({ name: 'pw-demo-model-stack', processEnv: { CALL1_STORE_DEMO: '1' } });
  try {
    await page.goto(priv.info.process_console_url);
    await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Settings' }).click();
    const demo = page.getByTestId('fine-tune-experience');
    await expect(demo).toContainText('Demo preview');
    const before = await (await priv.process('GET', '/training')).json();
    await expect(demo.getByRole('button', { name: 'Train private fine-tune' })).toBeDisabled();
    await demo.getByRole('button', { name: 'Get fine-tune', exact: true }).click();
    await expect(demo.getByRole('status')).toContainText('Retail & e-commerce is ready');
    await demo.getByRole('button', { name: 'Train private fine-tune' }).click();
    await expect(demo.getByRole('status')).toContainText('Your private fine-tune is ready');
    await expect(demo.getByRole('list', { name: 'Model layers' })).toContainText('Private LoRA · active alongside Call1');
    await demo.getByRole('switch', { name: 'Use private fine-tune' }).click();
    await expect(demo.getByRole('list', { name: 'Model layers' })).toContainText('Paused · Call1 layer stays active');
    await page.reload();
    await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Settings' }).click();
    await expect(demo.getByRole('list', { name: 'Model layers' })).toContainText('Retail & e-commerce');
    await expect(demo.getByRole('switch', { name: 'Use private fine-tune' })).toHaveAttribute('aria-checked', 'false');
    await demo.getByLabel('Industry edition').selectOption('banking-v1');
    await demo.getByRole('button', { name: 'Get fine-tune', exact: true }).click();
    await expect(demo.getByRole('status')).toContainText('Banking & finance is ready');
    await expect(demo.getByRole('switch', { name: 'Use private fine-tune' })).toHaveCount(0);
    await expect(demo.getByRole('list', { name: 'Model layers' })).toContainText('Add your team’s knowledge');
    const after = await (await priv.process('GET', '/training')).json();
    expect(after.active).toEqual(before.active);
    expect(after.versions).toEqual(before.versions);
    expect(after.runs).toEqual(before.runs);
    await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Models' }).click();
    await expect(page.getByTestId('model-pipeline')).toContainText('Demo stack: Call1 Banking & finance');
  } finally { await priv.close(); }
});
