// Process console Import and Pipeline flows (inventory area "process": p1 ingest, p2 job graph,
// p6 retry/cancel, p11 operator API through the console), in a real browser against the BUILT
// console that `python -m call1.process serve` serves. Tests are named after the feature IDs.
import type { Page } from '@playwright/test';
import { test, expect, uniqueWav, samplePath, processRequest, waitUntilSettled, ingestSample } from './fixtures';
import { startPrivateStack } from './private-stack';
import fs from 'node:fs';

interface ConversationItem {
  conversation_id: string;
  call_id: string;
  label: string;
  progress_line: string | null;
  evaluate_url: string;
}
interface ConversationDetail {
  conversation: { call_id: string; call_metadata: { agent_id: string } };
  jobs: { id: string; job_type: string; status: string }[];
}

/** The Pipeline view's plain-language job names (src/apps/process/labels.ts) for the job types a
 * fake-handler graph plans; anything else is checked by count only. */
const JOB_LABEL: Record<string, string> = {
  validation_vad: 'Audio validation',
  asr: 'Transcription',
  speaker_attribution: 'Speaker attribution',
  enrichment: 'Transcript enrichment',
  acoustic_tone: 'Acoustic tone',
  text_sentiment: 'Text sentiment',
  embeddings: 'Search embeddings',
  qa_criterion: 'QA criterion',
  qa_scorecard: 'QA scorecard',
  summary_assembly: 'Summary: assembly',
  contact_signals_lifecycle: 'Contact signals: lifecycle',
  contact_signals_merge: 'Contact signals: merge',
};

async function openConsole(page: Page, url: string): Promise<void> {
  await page.goto(url);
  await expect(page.getByText('Console connected')).toBeVisible();
  await expect(page).not.toHaveURL(/console_token=/); // the credential leaves the address bar at once
}

async function tab(page: Page, name: 'Overview' | 'Import' | 'Pipeline' | 'Models'): Promise<void> {
  await page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name }).click();
  await expect(page.getByRole('heading', { name, level: 1 }).or(page.getByRole('heading', { name }).first())).toBeVisible();
}

test('p1 Import: upload a recording, follow it to settled, open Evaluate; the same bytes again are reused', async ({ page, processConsoleURL, storeURL }, testInfo) => {
  await openConsole(page, processConsoleURL);
  await tab(page, 'Import');
  const filename = `p1-import-${testInfo.project.name}-${Date.now()}.wav`;
  const agent = `agent-p1-ui-${testInfo.project.name}-${Date.now()}`;
  const buffer = uniqueWav(fs.readFileSync(samplePath('call_01_compliant')));

  await page.locator('#process-import-file').setInputFiles({ name: filename, mimeType: 'audio/wav', buffer });
  await expect(page.getByText(filename)).toBeVisible();
  await page.getByLabel('Agent ID').fill(agent);
  await page.getByRole('button', { name: 'Upload and ingest' }).click();

  const card = page.locator('section').filter({ has: page.getByRole('heading', { name: filename }) });
  await expect(card).toHaveCount(1, { timeout: 30_000 });
  await expect(card.getByText('Receipt', { exact: true })).toBeVisible();
  await expect(card.getByText(/conversation created · graph created · \d+ jobs planned/)).toBeVisible();
  await expect(card.getByText('Transcript ready', { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(card.getByText('Analysis settled', { exact: true })).toBeVisible({ timeout: 60_000 });
  const href = await card.getByRole('link', { name: /Open in Evaluate/ }).getAttribute('href');
  expect(href).toMatch(new RegExp(`^${storeURL.replace(/[.]/g, '\\.')}/#/calls/call_[A-Za-z0-9]+$`));

  const callId = href!.split('/#/calls/')[1];
  const list = (await (await processRequest('GET', '/conversations?limit=200', { token: false })).json()) as { items: ConversationItem[] };
  const item = list.items.find((i) => i.call_id === callId);
  expect(item, 'the upload is in Process ledger').toBeTruthy();
  const detail = (await (await processRequest('GET', `/conversations/${item!.conversation_id}`, { token: false })).json()) as ConversationDetail;
  expect(detail.conversation.call_metadata.agent_id).toBe(agent);

  // The same bytes again: Store returns the same conversation and graph, and the card says so.
  await page.locator('#process-import-file').setInputFiles({ name: filename, mimeType: 'audio/wav', buffer });
  await page.getByRole('button', { name: 'Upload and ingest' }).click();
  await expect(card).toHaveCount(2, { timeout: 30_000 });
  await expect(page.getByText(/conversation reused · graph reused · \d+ jobs planned/)).toBeVisible();
});

test('p1 Import: an unsupported file shows the server explanation, not a receipt', async ({ page, processConsoleURL }) => {
  await openConsole(page, processConsoleURL);
  await tab(page, 'Import');
  await page.locator('#process-import-file').setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('not audio') });
  await page.getByRole('button', { name: 'Upload and ingest' }).click();
  const card = page.locator('section').filter({ has: page.getByRole('heading', { name: 'notes.txt' }) });
  await expect(card).toBeVisible();
  await expect(card.getByText(/unsupported|not a supported|audio/i).first()).toBeVisible();
  await expect(card.getByText('Receipt')).toHaveCount(0);
});

test('p2 Pipeline: a conversation row shows its progress line and expands to every job of its graph', async ({ page, processConsoleURL }, testInfo) => {
  const filename = `p2-pipeline-${testInfo.project.name}-${Date.now()}.wav`;
  const receipt = await ingestSample('call_01_compliant', { filename, agentId: `agent-p2-ui-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  const detail = (await (await processRequest('GET', `/conversations/${receipt.conversation_id}`, { token: false })).json()) as ConversationDetail;
  const list = (await (await processRequest('GET', '/conversations?limit=200', { token: false })).json()) as { items: ConversationItem[] };
  const item = list.items.find((i) => i.conversation_id === receipt.conversation_id)!;
  expect(item.progress_line).toContain('Transcript ready');

  await openConsole(page, processConsoleURL);
  await tab(page, 'Pipeline');
  const row = page.getByRole('button', { name: new RegExp(filename.replace(/[.]/g, '\\.')) });
  await expect(row).toBeVisible();
  await expect(row).toContainText('Complete');
  await expect(row).toContainText(item.progress_line!);
  await expect(page.getByRole('link', { name: /Evaluate/ }).first()).toBeVisible();

  await row.click();
  await expect(row).toHaveAttribute('aria-expanded', 'true');
  const card = page.locator('section').filter({ has: row });
  await expect(card.getByTestId('live-processing-graph')).toBeVisible();
  await expect(card.getByTestId('process-job-row')).toHaveCount(0);
  await card.getByRole('button', { name: 'Show all details' }).click();
  const jobRows = card.getByTestId('process-job-row');
  await expect(jobRows).toHaveCount(detail.jobs.length);
  for (const type of new Set(detail.jobs.map((j) => j.job_type))) {
    if (JOB_LABEL[type]) await expect(jobRows.filter({ hasText: JOB_LABEL[type] }).first()).toBeVisible();
  }
  await expect(jobRows.filter({ hasText: 'Succeeded' })).toHaveCount(detail.jobs.length);
  await expect(jobRows.first()).toContainText(/Attempt 1\/\d+/);
  // Pipeline order: audio validation and transcription come first, each finished job shows how long it ran.
  await expect(jobRows.first()).toContainText(/Audio validation|Transcription/);
  await expect(jobRows.filter({ hasText: /ran \d+ (ms|s)|ran \d+ min/ })).toHaveCount(detail.jobs.length);
  await expect(card).not.toContainText('Appliance ·');
});

test('p6 Pipeline: retry a failed job and cancel a running one, each with a reason', async ({ page }) => {
  test.setTimeout(150_000);
  const priv = await startPrivateStack({
    name: 'pw-p6',
    // The held job runs in the mlx pool, so the torch slot stays free for the retried tone job.
    fakeBehavior: { acoustic_tone: ['fail:validation_rejected'], contact_signals_lifecycle: ['hold:120'] },
    storeParameters: { heartbeat_interval_seconds: 5, lease_duration_seconds: 30 },
    // The held job is v1's contact_signals_lifecycle, so pin v1 even in a CALL1_SIGNALS_PIPELINE=v2 run.
    signalsPipeline: 'v1',
  });
  try {
    const receipt = await priv.ingest('call_01_compliant', { filename: 'p6-controls.wav', agentId: 'agent-p6-ui' });
    // Wait until tone has failed and sentiment is held RUNNING.
    await expect
      .poll(async () => {
        const d = (await (await priv.process('GET', `/conversations/${receipt.conversation_id}`)).json()) as ConversationDetail;
        const status = (t: string) => d.jobs.find((j) => j.job_type === t)?.status;
        return `${status('acoustic_tone')}/${status('contact_signals_lifecycle')}`;
      }, { timeout: 30_000 })
      .toBe('FAILED/RUNNING');

    await openConsole(page, priv.info.process_console_url);
    await tab(page, 'Pipeline');
    const row = page.getByRole('button', { name: /p6-controls\.wav/ });
    await row.click();
    const card = page.locator('section').filter({ has: row });
    await card.getByRole('button', { name: 'Show all details' }).click();

    const tone = card.locator('[data-testid=process-job-row]').filter({ hasText: 'Acoustic tone' });
    await expect(tone).toContainText('Failed');
    await expect(tone).toContainText('Validation Rejected');
    await tone.getByRole('button', { name: 'Retry' }).click();
    const reason = tone.getByRole('textbox', { name: 'Reason' });
    await expect(reason).toBeFocused();
    await expect(tone.locator('form button[type=submit]')).toBeDisabled(); // a reason is required
    await reason.fill('e2e: retry after the scripted failure');
    await tone.locator('form button[type=submit]').click();
    await expect(tone).toContainText('Succeeded', { timeout: 30_000 });
    await expect(tone).toContainText('Attempt 2/');

    const sentiment = card.locator('[data-testid=process-job-row]').filter({ hasText: 'Contact signals: lifecycle' });
    await expect(sentiment).toContainText('Running');
    await expect(sentiment.getByRole('button', { name: 'Retry' })).toHaveCount(0);
    await sentiment.getByRole('button', { name: 'Cancel' }).click();
    await expect(sentiment.getByRole('checkbox', { name: 'Also cancel jobs that depend on this one' })).toBeChecked();
    await sentiment.getByRole('textbox', { name: 'Reason' }).fill('e2e: cancel the held job');
    await sentiment.locator('form button[type=submit]').click();
    await expect(sentiment).toContainText(/Cancel requested|Cancelled/, { timeout: 15_000 });
    await expect(sentiment).toContainText('Cancelled', { timeout: 30_000 });
    await expect(sentiment.getByRole('button', { name: 'Retry' })).toBeVisible();

    const job = (await (await priv.service('GET', `/jobs/${(await (await priv.process('GET', `/conversations/${receipt.conversation_id}`)).json() as ConversationDetail).jobs.find((j) => j.job_type === 'contact_signals_lifecycle')!.id}`)).json()) as { status: string; retry_generation: number };
    expect(job.status).toBe('CANCELLED');
  } finally {
    await priv.close();
  }
});

test('p6 Pipeline: without the console token, retry and cancel are disabled and say why', async ({ page }) => {
  test.setTimeout(120_000);
  const priv = await startPrivateStack({ name: 'pw-p6-ro', fakeBehavior: { acoustic_tone: ['fail:validation_rejected'] } });
  try {
    await priv.ingest('call_01_compliant', { filename: 'p6-readonly.wav' });
    await expect
      .poll(async () => ((await (await priv.process('GET', '/conversations')).json()) as { items: { progress: { settled: boolean } | null }[] }).items[0]?.progress?.settled, { timeout: 30_000 })
      .toBe(true);
    await page.goto(`${priv.info.process_url}/`); // no fragment: no token in this tab
    await tab(page, 'Pipeline');
    const row = page.getByRole('button', { name: /p6-readonly\.wav/ });
    await row.click();
    await page.locator('section').filter({ has: row }).getByRole('button', { name: 'Show all details' }).click();
    const tone = page.locator('section').filter({ has: row }).locator('[data-testid=process-job-row]').filter({ hasText: 'Acoustic tone' });
    const retry = tone.getByRole('button', { name: 'Retry' });
    await expect(retry).toBeDisabled();
    await expect(retry).toHaveAttribute('title', 'Connect the console token to retry jobs');
  } finally {
    await priv.close();
  }
});

// ---------------------------------------------------------------------------------------------
// Agent identity at ingest (contract 1.1.0, team decisions 1 and 2): the Import form carries the
// agent's display name and extension, and the same recording again with different details
// updates the call's metadata, audited by Store, without reprocessing.
// ---------------------------------------------------------------------------------------------

interface UploadedCard {
  filename: string;
  buffer: Buffer;
}

async function importWithMouse(page: Page, upload: UploadedCard, fields: { name?: string; ext?: string; agentId?: string }) {
  await page.locator('#process-import-file').setInputFiles({ name: upload.filename, mimeType: 'audio/wav', buffer: upload.buffer });
  if (fields.agentId !== undefined) await page.getByLabel('Agent ID').fill(fields.agentId);
  if (fields.name !== undefined) await page.getByLabel('Agent name').fill(fields.name);
  if (fields.ext !== undefined) await page.getByLabel('Agent extension').fill(fields.ext);
  await page.getByRole('button', { name: 'Upload and ingest' }).click();
}

test('p1/q16 Import: agent name and extension reach the call, and a re-upload with new details updates it without reprocessing', async ({ page, processConsoleURL }, testInfo) => {
  await openConsole(page, processConsoleURL);
  await tab(page, 'Import');
  const upload = {
    filename: `p1-agent-${testInfo.project.name}-${Date.now()}.wav`,
    buffer: uniqueWav(fs.readFileSync(samplePath('call_01_compliant'))),
  };
  const agentId = `agent-p1-id-${testInfo.project.name}-${Date.now()}`;
  await importWithMouse(page, upload, { agentId, name: 'Samantha', ext: '104' });
  const cards = page.locator('section').filter({ has: page.getByRole('heading', { name: upload.filename }) });
  await expect(cards).toHaveCount(1, { timeout: 30_000 });
  const first = cards.first();
  await expect(first.getByText(/conversation created · graph created/)).toBeVisible();
  await expect(first.getByText('Samantha (104)', { exact: true })).toBeVisible();
  await expect(first.getByText('Metadata updated')).toHaveCount(0);

  // The same bytes with a new name and extension: the newest card is on top.
  await importWithMouse(page, upload, { name: 'Bob', ext: '202' });
  await expect(cards).toHaveCount(2, { timeout: 30_000 });
  const second = cards.first();
  await expect(second.getByText(/conversation reused · graph reused/)).toBeVisible();
  await expect(second.getByText('Metadata updated', { exact: true })).toBeVisible();
  await expect(second.getByText('agent name, agent extension · nothing was reprocessed')).toBeVisible();
  await expect(second.getByText('Bob (202)', { exact: true })).toBeVisible();

  const href = await second.getByRole('link', { name: /Open in Evaluate/ }).getAttribute('href');
  const callId = href!.split('/#/calls/')[1];
  const list = (await (await processRequest('GET', '/conversations?limit=200', { token: false })).json()) as { items: ConversationItem[] };
  const item = list.items.find((i) => i.call_id === callId)!;
  const detail = (await (await processRequest('GET', `/conversations/${item.conversation_id}`, { token: false })).json()) as {
    conversation: { call_metadata: { agent_id: string; agent_display_name: string | null; agent_extension: string | null } };
    graphs?: unknown[];
    jobs: { graph_id?: string }[];
  };
  // The name and extension changed; the agent ID the first upload named was kept (not reset to "Unknown").
  expect(detail.conversation.call_metadata).toMatchObject({ agent_id: agentId, agent_display_name: 'Bob', agent_extension: '202' });
  expect(new Set(detail.jobs.map((j) => j.graph_id)).size).toBe(1); // nothing was reprocessed

  // Invalid details are refused before anything is registered, with the server's explanation.
  await importWithMouse(page, { ...upload, filename: `bad-${upload.filename}` }, { ext: 'ext 104' });
  const bad = page.locator('section').filter({ has: page.getByRole('heading', { name: `bad-${upload.filename}` }) });
  await expect(bad.getByText(/agent_extension/)).toBeVisible();
  await expect(bad.getByText('Receipt')).toHaveCount(0);
});

// ---------------------------------------------------------------------------------------------
// Known issue (d), Process console side: the Overview worker counters are label/value pairs on
// one row at every width, not only at the desktop width store-console.spec.ts c4 checks.
// ---------------------------------------------------------------------------------------------

test('c4 Overview: worker counters stay aligned label/value pairs at phone, tablet and desktop widths', async ({ page, processConsoleURL }) => {
  await openConsole(page, processConsoleURL);
  const workerCard = page.locator('section', { has: page.getByRole('heading', { name: 'Worker', exact: true }) });
  for (const width of [360, 639, 640, 768, 1024, 1280, 1600]) {
    await page.setViewportSize({ width, height: 900 });
    await expect(workerCard.locator('dt:text-is("Claimed")')).toBeVisible();
    const dts = workerCard.locator('dl dt');
    const dds = workerCard.locator('dl dd');
    const n = await dts.count();
    expect(n).toBe(await dds.count());
    expect(n).toBeGreaterThanOrEqual(6);
    const bad: string[] = [];
    for (let i = 0; i < n; i++) {
      const label = (await dts.nth(i).innerText()).trim();
      const dt = await dts.nth(i).boundingBox();
      const dd = await dds.nth(i).boundingBox();
      if (!dt || !dd) bad.push(`${label}: not visible`);
      else if (Math.abs(dt.y - dd.y) > 2 || dd.x <= dt.x) bad.push(`${label}: dt@(${dt.x.toFixed(0)},${dt.y.toFixed(0)}) dd@(${dd.x.toFixed(0)},${dd.y.toFixed(0)})`);
    }
    expect(bad, `misaligned worker counters at ${width}px`).toEqual([]);
    // No horizontal page scroll at any width.
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  }
});

// ---------------------------------------------------------------------------------------------
// c8 coverage gap: every Process console control can be reached and operated by keyboard alone
// (Tab order, a visible focus indicator, Enter/Space activation, Escape to close a form).
// ---------------------------------------------------------------------------------------------

interface Focused {
  tag: string;
  id: string;
  type: string;
  name: string;
  expanded: string | null;
  ring: boolean;
  /** The text of the job row (`li`) the element sits in, if any. */
  row: string;
}

async function focused(page: Page): Promise<Focused> {
  return page.evaluate(() => {
    const el = document.activeElement as HTMLElement | null;
    if (!el || el === document.body) return { tag: 'body', id: '', type: '', name: '', expanded: null, ring: false, row: '' };
    const label = el.getAttribute('aria-label') || (el.id && document.querySelector(`label[for="${el.id}"]`)?.textContent) || el.innerText || '';
    // A visible focus indicator: a ring (box-shadow) or outline on the element, or, for the
    // visually hidden file input, on the label that stands in for it.
    const shows = (node: Element | null) => {
      if (!node) return false;
      const style = getComputedStyle(node);
      return (style.boxShadow && style.boxShadow !== 'none') || (style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0);
    };
    const ring = shows(el) || (el.classList.contains('sr-only') && shows(el.nextElementSibling));
    return {
      tag: el.tagName.toLowerCase(),
      id: el.id,
      type: (el as HTMLInputElement).type ?? '',
      name: label.replace(/\s+/g, ' ').trim(),
      expanded: el.getAttribute('aria-expanded'),
      ring: Boolean(ring),
      row: (el.closest('li')?.textContent ?? '').replace(/\s+/g, ' '),
    };
  });
}

/** Press Tab (or Shift+Tab) until the focused element matches, and return it. Fails after `max` presses. */
async function tabTo(page: Page, match: (f: Focused) => boolean, what: string, { back = false, max = 80 } = {}): Promise<Focused> {
  const seen: string[] = [];
  for (let i = 0; i < max; i++) {
    await page.keyboard.press(back ? 'Shift+Tab' : 'Tab');
    const f = await focused(page);
    seen.push(`${f.tag}${f.id ? '#' + f.id : ''}:${f.name.slice(0, 30)}`);
    if (match(f)) {
      expect(f.ring, `${what} has no visible focus indicator`).toBe(true);
      return f;
    }
  }
  throw new Error(`Tab never reached ${what}; focus went through: ${seen.join(' | ')}`);
}

const isButton = (name: string | RegExp, row?: string) => (f: Focused) =>
  f.tag === 'button' && (typeof name === 'string' ? f.name === name : name.test(f.name)) && (row === undefined || f.row.includes(row));

test('c8 keyboard: the Process console header, Import, Overview and Pipeline work with the keyboard alone', async ({ page, processConsoleURL }, testInfo) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await openConsole(page, processConsoleURL);

  // Tab order through the header: the four tabs in order, then the theme toggle.
  for (const name of ['Overview', 'Import', 'Pipeline', 'Models']) await tabTo(page, isButton(name), `the ${name} tab`);
  const toggle = await tabTo(page, isButton(/Switch to (dark|light) theme/), 'the theme toggle');
  await page.keyboard.press('Enter');
  const toggled = await focused(page);
  expect(toggled.name).not.toBe(toggle.name); // Enter switched the theme, focus stayed put
  await page.keyboard.press('Space');
  expect((await focused(page)).name).toBe(toggle.name); // and Space switches it back

  // Enter on a tab opens it.
  await tabTo(page, isButton('Import'), 'the Import tab', { back: true });
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Import', level: 1 })).toBeVisible();
  await expect(page.getByRole('navigation', { name: 'Process' }).getByRole('button', { name: 'Import' })).toHaveAttribute('aria-current', 'page');

  // The file input is in the Tab order and Space opens the picker.
  await tabTo(page, (f) => f.id === 'process-import-file', 'the file input');
  const chooser = page.waitForEvent('filechooser');
  await page.keyboard.press('Space');
  const filename = `c8-keys-${testInfo.project.name}-${Date.now()}.wav`;
  await (await chooser).setFiles({ name: filename, mimeType: 'audio/wav', buffer: uniqueWav(fs.readFileSync(samplePath('call_01_compliant'))) });
  await expect(page.getByText(filename)).toBeVisible();

  // Every field by Tab, typed into; the channel select by type-ahead (arrow keys open the popup on macOS).
  await tabTo(page, (f) => f.tag === 'input' && f.name === 'Agent ID', 'Agent ID');
  await page.keyboard.type(`agent-c8-${Date.now()}`);
  await tabTo(page, (f) => f.tag === 'input' && f.name === 'Agent name', 'Agent name');
  await page.keyboard.type('Kim');
  await tabTo(page, (f) => f.tag === 'input' && f.name === 'Agent extension', 'Agent extension');
  await page.keyboard.type('311');
  await tabTo(page, (f) => f.tag === 'select', 'Agent channel');
  await page.keyboard.type('Channel 0');
  await expect(page.getByLabel('Agent channel')).toHaveValue('0');
  await tabTo(page, (f) => f.tag === 'input' && f.name === 'Call reference', 'Call reference');
  await page.keyboard.type('kbd-1');
  await tabTo(page, isButton('Upload and ingest'), 'Upload and ingest');
  await page.keyboard.press('Enter');
  const card = page.locator('section').filter({ has: page.getByRole('heading', { name: filename }) });
  await expect(card.getByText('Receipt', { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(card.getByText('Kim (311)', { exact: true })).toBeVisible();
  // The card's Dismiss and Open in Evaluate controls are reachable too.
  await tabTo(page, (f) => f.tag === 'a' && /Open in Evaluate/.test(f.name), 'Open in Evaluate');
  await tabTo(page, isButton('Dismiss'), 'Dismiss', { back: true });

  // Pipeline: open the tab, reach the new row, expand and collapse it with Enter and Space.
  await tabTo(page, isButton('Pipeline'), 'the Pipeline tab', { back: true });
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Pipeline', level: 1 })).toBeVisible();
  await expect(page.getByRole('button', { name: new RegExp(filename.replace(/[.]/g, '\\.')) })).toBeVisible();
  const row = await tabTo(page, (f) => f.tag === 'button' && f.name.includes(filename), 'the conversation row');
  expect(row.expanded).toBe('false');
  await page.keyboard.press('Enter');
  expect((await focused(page)).expanded).toBe('true');
  await expect(page.locator('section').filter({ has: page.getByRole('button', { name: new RegExp(filename.replace(/[.]/g, '\\.')) }) }).getByTestId('live-processing-graph')).toBeVisible();
  await page.keyboard.press('Space');
  expect((await focused(page)).expanded).toBe('false');
  // The row's Evaluate link is its own Tab stop, right after the row.
  await page.keyboard.press('Tab');
  const link = await focused(page);
  expect(link.tag).toBe('a');
  expect(link.name).toContain('Evaluate');

  // Overview and Models by keyboard.
  for (const name of ['Overview', 'Models'] as const) {
    await tabTo(page, isButton(name), `the ${name} tab`, { back: true });
    await page.keyboard.press('Enter');
    await expect(page.getByRole('heading', { name, level: 1 })).toBeVisible();
    await tabTo(page, isButton('Refresh'), `${name} Refresh`);
    await page.keyboard.press('Enter');
  }
});

test('c8/p6 keyboard: retry and cancel forms work by keyboard, with distinctly named confirm and dismiss buttons', async ({ page }) => {
  test.setTimeout(150_000);
  const priv = await startPrivateStack({
    name: 'pw-c8-keys',
    fakeBehavior: { acoustic_tone: ['fail:validation_rejected'], contact_signals_lifecycle: ['hold:120'] },
    storeParameters: { heartbeat_interval_seconds: 5, lease_duration_seconds: 30 },
    // The held job is v1's contact_signals_lifecycle, so pin v1 even in a CALL1_SIGNALS_PIPELINE=v2 run.
    signalsPipeline: 'v1',
  });
  try {
    const receipt = await priv.ingest('call_01_compliant', { filename: 'c8-keys.wav', agentId: 'agent-c8-keys' });
    await expect
      .poll(async () => {
        const d = (await (await priv.process('GET', `/conversations/${receipt.conversation_id}`)).json()) as ConversationDetail;
        const status = (t: string) => d.jobs.find((j) => j.job_type === t)?.status;
        return `${status('acoustic_tone')}/${status('contact_signals_lifecycle')}`;
      }, { timeout: 30_000 })
      .toBe('FAILED/RUNNING');

    await page.setViewportSize({ width: 1280, height: 900 });
    await openConsole(page, priv.info.process_console_url);
    await tabTo(page, isButton('Pipeline'), 'the Pipeline tab');
    await page.keyboard.press('Enter');
    await expect(page.getByRole('button', { name: /c8-keys\.wav/ })).toBeVisible();
    await tabTo(page, (f) => f.tag === 'button' && f.name.includes('c8-keys.wav'), 'the conversation row');
    await page.keyboard.press('Enter');
    const card = page.locator('section').filter({ has: page.getByRole('button', { name: /c8-keys\.wav/ }) });
    await tabTo(page, isButton('Show all details'), 'Show all details');
    await page.keyboard.press('Enter');
    const tone = card.locator('[data-testid=process-job-row]').filter({ hasText: 'Acoustic tone' });
    await expect(tone).toContainText('Failed');

    // Retry: Enter opens the form with the reason focused; Escape closes it and returns focus.
    await tabTo(page, isButton('Retry', 'Acoustic tone'), 'Retry');
    await page.keyboard.press('Enter');
    await expect(tone.getByRole('textbox', { name: 'Reason' })).toBeFocused();
    await expect(tone.getByRole('button', { name: 'Retry job', exact: true })).toBeVisible();
    await expect(tone.getByRole('button', { name: 'Close', exact: true })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(tone.getByRole('textbox', { name: 'Reason' })).toHaveCount(0);
    expect((await focused(page)).name).toBe('Retry');
    await page.keyboard.press('Enter');
    await page.keyboard.type('keyboard: retry after the scripted failure');
    await tabTo(page, isButton('Retry job'), 'Retry job');
    await page.keyboard.press('Enter');
    await expect(tone).toContainText('Succeeded', { timeout: 30_000 });

    // Cancel: the confirm and the dismiss have different names; no button in the form is just "Cancel".
    const held = card.locator('[data-testid=process-job-row]').filter({ hasText: 'Contact signals: lifecycle' });
    await expect(held).toContainText('Running');
    await tabTo(page, isButton('Cancel', 'Contact signals: lifecycle'), 'Cancel');
    await page.keyboard.press('Enter');
    const form = held.getByRole('form', { name: 'Cancel' });
    await expect(form).toBeVisible();
    const names = (await form.getByRole('button').allInnerTexts()).map((t) => t.trim());
    expect(names).toEqual(['Cancel job', 'Keep job']);
    expect(new Set(names).size).toBe(names.length);
    await expect(form.getByRole('button', { name: 'Cancel', exact: true })).toHaveCount(0);
    // "Keep job" dismisses without cancelling.
    await tabTo(page, isButton('Keep job'), 'Keep job');
    await page.keyboard.press('Enter');
    await expect(form).toHaveCount(0);
    await expect(held).toContainText('Running');
    expect((await focused(page)).name).toBe('Cancel');
    // Then cancel for real, by keyboard: reason, the cascade checkbox by Space, then "Cancel job".
    await page.keyboard.press('Enter');
    await page.keyboard.type('keyboard: cancel the held job');
    await tabTo(page, (f) => f.tag === 'input' && f.type === 'checkbox', 'the cascade checkbox');
    await page.keyboard.press('Space');
    await expect(held.getByRole('checkbox', { name: 'Also cancel jobs that depend on this one' })).not.toBeChecked();
    await page.keyboard.press('Space');
    await expect(held.getByRole('checkbox', { name: 'Also cancel jobs that depend on this one' })).toBeChecked();
    await tabTo(page, isButton('Cancel job'), 'Cancel job');
    await page.keyboard.press('Enter');
    await expect(held).toContainText('Cancelled', { timeout: 30_000 });
  } finally {
    await priv.close();
  }
});
