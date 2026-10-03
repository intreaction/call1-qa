// Process console Settings → On-device training (docs/OnDeviceTraining.md §6, feature IDs t1–t7 in
// §7.3), in a real browser against the BUILT console that `python -m call1.process serve` serves.
// Every run here uses `startPrivateStack` with fake handlers, the fake trainer and the fake
// generator (`CALL1_FAKE_TRAINING_OUTCOMES` picks each run's decision, `CALL1_FAKE_TRAINER_SECONDS`
// slows the trainer), and the §1.7 minimums lowered in the Process config. Runs that train get real
// reviewer labels first: `seedTrainingLabels` ingests calls and overrides their QA verdicts through
// Store's review API as a signed-in reviewer (tests/e2e/training_seed.py). No MLX, no torch and no
// `CALL1_REAL_MODELS` anywhere in this file.
import type { Locator, Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { startPrivateStack } from './private-stack';

async function openConsole(page: Page, url: string): Promise<void> {
  await page.goto(url);
  await expect(page.getByText('Console connected')).toBeVisible();
}

async function openSettings(page: Page): Promise<void> {
  await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Settings' }).click();
  await expect(page.getByRole('heading', { name: 'Settings', level: 1 })).toBeVisible();
  const tools = page.getByTestId('live-training-tools');
  if (await tools.getAttribute('open') === null) await tools.locator('summary').click();
}

function card(page: Page, heading: string): Locator {
  return page.locator('section').filter({ has: page.getByRole('heading', { name: heading }) });
}

/** "Train now" opens a confirm that says what the run does (pauses processing, for up to the
 * maximum duration); "Start training" starts it. */
async function trainNow(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Train now' }).click();
  const confirm = page.getByRole('group', { name: 'Train now' });
  await expect(confirm).toContainText('pauses call processing');
  await confirm.getByRole('button', { name: 'Start training', exact: true }).click();
}

/**
 * A stack with the §1.7 minimums low enough for a few seeded calls to clear them. `outcomes` is
 * `CALL1_FAKE_TRAINING_OUTCOMES` (one consumed per run that reaches evaluation: promote, reject,
 * invalid, crash); `trainerSeconds` is `CALL1_FAKE_TRAINER_SECONDS` (the cancel flow).
 */
function trainingStack(name: string, options: { labels?: boolean; outcomes?: string; trainerSeconds?: number } = {}) {
  const processEnv: Record<string, string> = {};
  if (options.outcomes) processEnv.CALL1_FAKE_TRAINING_OUTCOMES = options.outcomes;
  if (options.trainerSeconds) processEnv.CALL1_FAKE_TRAINER_SECONDS = String(options.trainerSeconds);
  return startPrivateStack({
    name,
    processConfig: {
      training: { min_new_labels: 1, min_labeled_calls: 1, min_train_examples: 1, min_eval_items: 1 },
    },
    processEnv,
    seedTrainingLabels: options.labels ?? false,
  });
}

test('t1 Settings: enable the default daily schedule, then weekly on Monday; it persists across a reload', async ({ page }) => {
  const priv = await trainingStack('pw-t1');
  try {
    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);

    const toggle = page.getByRole('switch', { name: 'Scheduled training' });
    await expect(toggle).toHaveAttribute('aria-checked', 'false');
    await expect(page.getByLabel('Frequency')).toHaveValue('daily');
    await toggle.click();
    await expect(toggle).toHaveAttribute('aria-checked', 'true');
    await page.getByRole('button', { name: 'Save' }).click(); // the default schedule: daily
    await expect(page.getByText('Saved.')).toBeVisible();
    await expect(page.getByText(/^Next run: \w{3} 02:00 \(/)).toBeVisible();

    await page.getByLabel('Frequency').selectOption('weekly');
    await expect(page.getByLabel('Weekday')).toHaveValue('6'); // the backend default: 6 = Sunday
    await expect(page.getByLabel('Weekday').locator('option:checked')).toHaveText('Sunday');
    await page.getByLabel('Weekday').selectOption({ label: 'Monday' });
    await page.getByLabel('Time').fill('02:00');
    await page.getByRole('button', { name: 'Save' }).click();
    await expect(page.getByText('Saved.')).toBeVisible();

    const saved = (await (await priv.process('GET', '/training', { token: false })).json()) as {
      settings: { enabled: boolean; schedule: { frequency: string; weekday: number; time: string } };
    };
    expect(saved.settings.enabled).toBe(true);
    expect(saved.settings.schedule).toEqual({ frequency: 'weekly', weekday: 0, time: '02:00' }); // 0 = Monday

    await page.reload();
    await expect(page.getByText('Console connected')).toBeVisible();
    await openSettings(page);
    await expect(page.getByRole('switch', { name: 'Scheduled training' })).toHaveAttribute('aria-checked', 'true');
    await expect(page.getByLabel('Frequency')).toHaveValue('weekly');
    await expect(page.getByLabel('Weekday')).toHaveValue('0');
    await expect(page.getByLabel('Weekday').locator('option:checked')).toHaveText('Monday');
    await expect(page.getByLabel('Time')).toHaveValue('02:00');
    await expect(page.getByText(/^Next run: Mon 02:00 \(/)).toBeVisible();
  } finally {
    await priv.close();
  }
});

test('t2 Settings: Train now leads to a promoted run, and the active version shows', async ({ page }) => {
  test.setTimeout(150_000);
  const priv = await trainingStack('pw-t2', { labels: true, outcomes: 'promote' });
  try {
    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);
    await expect(card(page, 'On-device training')).toContainText(/\d+ label\(s\) logged/);
    await trainNow(page);
    const activeCard = card(page, 'Active model');
    await expect(activeCard).not.toContainText('Base model (no adapter)', { timeout: 60_000 });
    await expect(activeCard).toContainText(/ft-\d{8}T\d{6}Z/);
    await expect(activeCard).toContainText('held-out accuracy');
    const history = card(page, 'Run history');
    await expect(history.getByText('Promoted', { exact: true }).first()).toBeVisible();
    await expect(page.getByRole('button', { name: 'Train now' })).toBeEnabled();
  } finally {
    await priv.close();
  }
});

test('t3 Settings: a rejected run shows its reason, and the active version is unchanged', async ({ page }) => {
  test.setTimeout(150_000);
  const priv = await trainingStack('pw-t3', { labels: true, outcomes: 'reject' });
  try {
    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);
    const activeCard = card(page, 'Active model');
    await expect(activeCard).toContainText('Base model (no adapter)');
    await trainNow(page);
    const history = card(page, 'Run history');
    const row = history.getByRole('button', { name: /Rejected/ }).first();
    await expect(row).toBeVisible({ timeout: 60_000 });
    await expect(row).toContainText('rejected:');
    await row.click();
    await expect(row).toHaveAttribute('aria-expanded', 'true');
    await expect(history.getByText('Per task')).toBeVisible();
    await expect(activeCard).toContainText('Base model (no adapter)');
  } finally {
    await priv.close();
  }
});

test('t4 Settings: roll back to the base, and reactivate', async ({ page }) => {
  test.setTimeout(150_000);
  const priv = await trainingStack('pw-t4', { labels: true, outcomes: 'promote' });
  try {
    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);
    await trainNow(page);
    const activeCard = card(page, 'Active model');
    await expect(activeCard).not.toContainText('Base model (no adapter)', { timeout: 60_000 });
    const version = (await activeCard.getByText(/^ft-\d{8}T\d{6}Z$/).first().innerText()).trim();

    await activeCard.getByRole('button', { name: 'Use base model' }).click();
    await activeCard.getByRole('group', { name: 'Use base model' }).getByRole('button', { name: 'Use base model', exact: true }).click();
    await expect(activeCard).toContainText('Base model (no adapter)');

    await activeCard.getByRole('button', { name: 'Activate' }).click();
    await activeCard.getByRole('group', { name: 'Activate' }).getByRole('button', { name: 'Activate', exact: true }).click();
    await expect(activeCard.getByText(version, { exact: true }).first()).toBeVisible();
    await expect(activeCard).not.toContainText('Base model (no adapter)');
  } finally {
    await priv.close();
  }
});

test('t4 Settings: after a second promotion, roll back to the previous version', async ({ page }) => {
  test.setTimeout(150_000);
  const priv = await trainingStack('pw-t4b', { labels: true, outcomes: 'promote,promote' });
  try {
    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);
    const activeCard = card(page, 'Active model');
    const history = card(page, 'Run history');
    await trainNow(page);
    await expect(history.getByText('Promoted', { exact: true })).toHaveCount(1, { timeout: 60_000 });
    const first = (await activeCard.getByText(/^ft-\d{8}T\d{6}Z$/).first().innerText()).trim();
    await expect(page.getByRole('button', { name: 'Train now' })).toBeEnabled();
    await trainNow(page);
    await expect(history.getByText('Promoted', { exact: true })).toHaveCount(2, { timeout: 60_000 });
    await expect(activeCard.getByText(/^ft-\d{8}T\d{6}Z$/).first()).not.toHaveText(first);

    const rollback = activeCard.getByRole('button', { name: `Roll back to ${first}` });
    await expect(rollback).toBeVisible();
    await rollback.click();
    await activeCard.getByRole('group', { name: `Roll back to ${first}` }).getByRole('button', { name: `Roll back to ${first}`, exact: true }).click();
    await expect(activeCard.getByText(/^ft-\d{8}T\d{6}Z$/).first()).toHaveText(first);
  } finally {
    await priv.close();
  }
});

test('t5 Settings: cancel a slow fake run', async ({ page }) => {
  test.setTimeout(150_000);
  const priv = await trainingStack('pw-t5', { labels: true, trainerSeconds: 60 });
  try {
    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);
    await trainNow(page);
    await expect(page.getByRole('button', { name: 'Cancel run' })).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole('button', { name: 'Train now' })).toBeDisabled();
    await expect(page.getByText(/^Training: iteration \d+ of \d+/).first()).toBeVisible({ timeout: 30_000 });
    await page.getByRole('button', { name: 'Cancel run' }).click();
    await page.getByRole('group', { name: 'Cancel run' }).getByRole('button', { name: 'Cancel run', exact: true }).click();
    const history = card(page, 'Run history');
    await expect(history.getByText('Cancelled', { exact: true }).first()).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole('button', { name: 'Train now' })).toBeEnabled();
    await expect(card(page, 'Active model')).toContainText('Base model (no adapter)');
  } finally {
    await priv.close();
  }
});

test('t6 Settings: without a token, the controls are disabled and say why', async ({ page }) => {
  const priv = await trainingStack('pw-t6');
  try {
    await page.goto(`${priv.info.process_url}/`); // no fragment: no token in this tab
    await openSettings(page);
    const trainNowButton = page.getByRole('button', { name: 'Train now' });
    await expect(trainNowButton).toBeDisabled();
    await expect(trainNowButton).toHaveAttribute('title', 'Connect the console token to start a run');
    const save = page.getByRole('button', { name: 'Save' });
    await expect(save).toBeDisabled();
  } finally {
    await priv.close();
  }
});

test('t6 Settings: with no reviewer labels, Train now is disabled and says why', async ({ page }) => {
  const priv = await trainingStack('pw-t6c');
  try {
    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);
    await expect(card(page, 'On-device training')).toContainText('0 label(s) logged');
    const trainNowButton = page.getByRole('button', { name: 'Train now' });
    await expect(trainNowButton).toBeDisabled();
    await expect(trainNowButton).toHaveAttribute('title', /nothing to train on/);
    await expect(card(page, 'Run history')).toContainText('No training runs on this appliance yet.');
  } finally {
    await priv.close();
  }
});

test('t7 Settings: the whole flow works by keyboard alone', async ({ page }) => {
  test.setTimeout(150_000);
  const priv = await trainingStack('pw-t7', { labels: true, outcomes: 'promote' });
  try {
    await page.setViewportSize({ width: 1280, height: 900 });
    await openConsole(page, priv.info.process_console_url);
    await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Settings' }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByRole('heading', { name: 'Settings', level: 1 })).toBeVisible();
  const tools = page.getByTestId('live-training-tools');
  if (await tools.getAttribute('open') === null) await tools.locator('summary').click();

    const toggle = page.getByRole('switch', { name: 'Scheduled training' });
    await toggle.focus();
    await page.keyboard.press('Space');
    await expect(toggle).toHaveAttribute('aria-checked', 'true');
    await page.keyboard.press('Space');
    await expect(toggle).toHaveAttribute('aria-checked', 'false');

    await page.getByRole('button', { name: 'Save' }).focus();
    await page.keyboard.press('Enter');
    await expect(page.getByText('Saved.')).toBeVisible();

    await page.getByRole('button', { name: 'Train now' }).focus();
    await page.keyboard.press('Enter');
    // The confirm opens with "Start training" focused; Enter starts the run.
    await expect(page.getByRole('button', { name: 'Start training', exact: true })).toBeFocused();
    await page.keyboard.press('Enter');

    const history = card(page, 'Run history');
    const row = history.getByRole('button', { name: /Promoted/ }).first();
    await expect(row).toBeVisible({ timeout: 60_000 });
    await row.focus();
    await page.keyboard.press('Enter');
    await expect(row).toHaveAttribute('aria-expanded', 'true');
  } finally {
    await priv.close();
  }
});

test('t6 Settings: a label-count error shows the fix and the page still renders', async ({ page }) => {
  const priv = await trainingStack('pw-t6b');
  try {
    // What an older Process key without training:read gets (403 insufficient_scope), as the backend
    // reports it: an error object, plus a notice with the command that fixes it.
    await page.route('**/process/api/training', async (route) => {
      const response = await route.fetch();
      const body = await response.json();
      body.labels = { total: null, new_since_last_run: null, error: { code: 'insufficient_scope', message: 'This Process service key lacks the training:read scope' } };
      body.notices = [
        ...(body.notices ?? []),
        { code: 'insufficient_scope', message: 'On the Store host, issue a key that holds it: python -m call1.store issue-service-key' },
      ];
      await route.fulfill({ response, json: body });
    });
    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);
    await expect(page.getByText(/python -m call1\.store issue-service-key/)).toBeVisible();
    await expect(page.getByRole('switch', { name: 'Scheduled training' })).toBeVisible();
    await expect(card(page, 'Run history')).toBeVisible();
    await expect(page.getByRole('navigation', { name: 'Process' })).toBeVisible();
  } finally {
    await priv.close();
  }
});

// The model library is an operator action in Models; Settings continues to own training.
test('t8 Settings: choose between multiple fine-tunes and the base, with selection persisted', async ({ page }) => {
  test.setTimeout(180_000);
  const priv = await trainingStack('pw-t8-models', { labels: true, outcomes: 'promote,promote' });
  try {
    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);
    const history = card(page, 'Run history');
    await trainNow(page);
    await expect(history.getByText('Promoted', { exact: true })).toHaveCount(1, { timeout: 60_000 });
    await expect(page.getByRole('button', { name: 'Train now' })).toBeEnabled();
    await trainNow(page);
    await expect(history.getByText('Promoted', { exact: true })).toHaveCount(2, { timeout: 60_000 });
    const state = await (await priv.process('GET', '/training', { token: false })).json() as { versions: { version: string }[] };
    expect(state.versions).toHaveLength(2);

    await openSettings(page);
    const picker = page.getByTestId('processing-model-picker');
    const select = picker.getByLabel('Model to use');
    await expect(select.locator('option')).toHaveCount(3);
    for (const id of ['base', ...state.versions.map((v) => v.version)]) {
      await select.selectOption(id);
      await picker.getByRole('button', { name: 'Use selected model' }).click();
      await expect(picker.getByRole('status')).toHaveText('Model selection saved.');
      const persisted = await (await priv.process('GET', '/training', { token: false })).json() as { active: { version: string } | null };
      expect(persisted.active?.version ?? 'base').toBe(id);
      await page.reload();
      await openSettings(page);
      await expect(select).toHaveValue(id);
      await expect(picker.getByTestId('active-processing-model')).toContainText(id === 'base' ? 'Gemma 4 E2B · Base' : id);
    }
    await expect(picker).toContainText('Existing call results change only after reanalysis');
  } finally {
    await priv.close();
  }
});

test('t9 Settings: base-only library, read-only credential gate, and a training run blocks switching', async ({ page }) => {
  test.setTimeout(150_000);
  const priv = await trainingStack('pw-t9-models', { labels: true, trainerSeconds: 60 });
  try {
    await page.goto(priv.info.process_url);
    await openSettings(page);
    const picker = page.getByTestId('processing-model-picker');
    await expect(picker.getByLabel('Model to use')).toBeDisabled();
    await expect(picker).toContainText('No fine-tuned models installed yet');
    await expect(picker.getByRole('button', { name: 'Use selected model' })).toBeDisabled();

    await openConsole(page, priv.info.process_console_url);
    await openSettings(page);
    await trainNow(page);
    await expect(card(page, 'Status').getByRole('button', { name: 'Cancel run' })).toBeVisible();
    await openSettings(page);
    await expect(picker).toContainText('Model selection is paused while a training run is queued or running');
    await expect(picker.getByLabel('Model to use')).toBeDisabled();
    await expect(picker.getByRole('button', { name: 'Use selected model' })).toBeDisabled();
  } finally {
    await priv.close();
  }
});
