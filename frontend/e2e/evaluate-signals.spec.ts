// Contact Signals v2 (docs/ContactSignalsV2.md section 10.1, section 17 "Playwright" list,
// "evaluate-signals"): SignalsView (#/signals, #/signals/:categoryId) — the taxonomy tree and
// editor, the field editor, caps, "Test on recent calls", the activation dialog and the pipeline
// banner — and, in cs6, the class-demo moment of section 15 end to end.
//
// The taxonomy is one Store-wide record, and saves, alert rules and backfills change what every
// other spec sees on a shared stack. So the taxonomy-editor tests run on a PRIVATE stack per
// project (dark, light), started the way `python -m call1.launch --demo --handlers fake` starts
// the class demo: Store demo mode (the labelled "Continue as Demo …" sign-in), the retail seed
// taxonomy (call1/store/seeds/signals_retail_v1.json, decision 22), pipeline v2, and the first
// ingested call playing the `cancel` fake script (section 8.6). Tests in the describe run
// in order on that stack: read-only checks first, drafts that are discarded next, then the saves.
//
// Selectors follow the built UI (frontend/src/apps/evaluate/views/signals/): the tree is the
// "Signal categories" navigation, the editor is `signals-category-editor`, a new subcategory opens
// in `signals-subcategory-editor`, and the save bar is `signals-save-bar`.
import type { Browser, Locator, Page } from '@playwright/test';
import type { components } from '../src/contracts/store-v1';
type SignalTaxonomyRecord = components['schemas']['SignalTaxonomyRecord'];
import { test, expect } from './fixtures';
import { startPrivateStack, type PrivateStack } from './private-stack';

const BUILT_INS = ['Caller objective', 'Reported issue', 'Friction point', 'Proposed fix', 'Agent completed', 'Caller confirmed', 'Still unresolved', 'Deferred'];
// FORBIDDEN_FIELD_PII_CLASSES (call1/contracts/signals.py): masked before any model sees them.
const FORBIDDEN_PII = ['caller_name', 'account_number', 'card_number', 'phone', 'email', 'address', 'url', 'secret', 'government_id'];

// The retail seed's custom categories (call1/store/seeds/signals_retail_v1.md section 1).
const SEED_CUSTOM_CATEGORIES = 2;

type Persona = 'Admin' | 'Supervisor' | 'Reviewer';

async function signInDemo(browser: Browser, stack: PrivateStack, persona: Persona, colorScheme: 'light' | 'dark' | 'no-preference' | null | undefined): Promise<Page> {
  const context = await browser.newContext({ baseURL: stack.info.store_url, colorScheme: colorScheme ?? undefined });
  const page = await context.newPage();
  await page.goto('/#/sign-in');
  await page.getByRole('button', { name: `Continue as Demo ${persona}`, exact: true }).click();
  await expect(page.getByRole('banner').getByRole('button', { name: 'Sign out' })).toBeVisible({ timeout: 15_000 });
  return page;
}

async function settle(stack: PrivateStack, conversationId: string, timeoutMs = 90_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let last = '';
  while (Date.now() < deadline) {
    const response = await stack.service('GET', `/conversations/${conversationId}/progress`);
    last = await response.text();
    if (response.ok && (JSON.parse(last) as { settled: boolean }).settled) return;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`conversation ${conversationId} did not settle: ${last}`);
}

const tree = (page: Page) => page.getByRole('navigation', { name: 'Signal categories' });
const editor = (page: Page) => page.getByTestId('signals-category-editor');
const saveBar = (page: Page) => page.getByTestId('signals-save-bar');
const saveButton = (page: Page) => saveBar(page).getByRole('button', { name: 'Save', exact: true });
const threshold = (scope: Page | Locator) => scope.getByLabel('Threshold', { exact: true });

async function openCategory(page: Page, id: string, name: string): Promise<void> {
  await page.goto(`/#/signals/${id}`);
  await expect(editor(page).getByRole('heading', { name, exact: true })).toBeVisible();
}

test.describe('Signals: the taxonomy editor on a demo-mode stack (section 10.1)', () => {
  let stack: PrivateStack;
  let cancelCall: { call_id: string; conversation_id: string };
  let plainCall: { call_id: string; conversation_id: string };

  test.beforeAll(async ({ browser, colorScheme }, testInfo) => {
    testInfo.setTimeout(240_000);
    stack = await startPrivateStack({
      name: `pw-signals-${testInfo.project.name}`,
      storeEnv: { CALL1_STORE_DEMO: '1' },
      signalsPipeline: 'v2',
      signalsSeed: 'call1/store/seeds/signals_retail_v1.json',
      // The first ASR attempt plays the cancel script, as call_05 does in the fake-handler demo.
      fakeBehavior: { asr: ['script:cancel'] },
    });
    // This suite exercises model-based subcategory previews. Select that engine on each
    // recipe explicitly; the rules suite separately covers recipe-driven rule detection.
    const setup = await signInDemo(browser, stack, 'Admin', colorScheme);
    const current = await (await setup.request.get('/store/v1/signals/taxonomy')).json() as SignalTaxonomyRecord;
    const session = await (await setup.request.get('/store/v1/auth/session')).json();
    for (const category of current.current.taxonomy.categories) {
      if (category.recipe) category.recipe.engine = 'gemma';
    }
    const configured = await setup.request.put('/store/v1/signals/taxonomy', {
      headers: { Origin: stack.info.store_url, 'X-Call1-CSRF': session.csrf_token },
      data: { taxonomy: current.current.taxonomy, expected_record_version: current.record_version },
    });
    expect(configured.ok()).toBeTruthy();
    await setup.context().close();
    cancelCall = await stack.ingest('call_05_pii_heavy', { agentId: 'pw-signals-cancel' });
    await settle(stack, cancelCall.conversation_id);
    plainCall = await stack.ingest('call_01_compliant', { agentId: 'pw-signals-plain' });
    await settle(stack, plainCall.conversation_id);
  });
  test.afterAll(async () => {
    await stack?.close();
  });

  test('cs1: built-in categories show as locked, with name, gloss and speaker read-only', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await admin.goto('/#/signals');
    await expect(admin.getByRole('heading', { name: 'Signals', exact: true })).toBeVisible();

    for (const name of BUILT_INS) {
      const row = tree(admin).getByRole('link', { name: new RegExp(`^${name}\\b`) });
      await expect(row).toBeVisible();
      await expect(row).toContainText('Built-in');
    }

    await tree(admin).getByRole('link', { name: /^Caller objective\b/ }).click();
    await expect(admin).toHaveURL(/#\/signals\/intent$/);
    const ed = editor(admin);
    await expect(ed.getByRole('heading', { name: 'Caller objective', exact: true })).toBeVisible();
    await expect(ed.getByText('Built-in', { exact: true })).toBeVisible();

    // Built-ins keep their name, gloss and speaker fixed (section 7.2 BUILTIN_EDITABLE_FIELDS).
    await expect(ed.getByLabel('Name', { exact: true })).toBeDisabled();
    await expect(ed.getByLabel('Gloss', { exact: true })).toBeDisabled();
    await expect(ed.getByText('Caller', { exact: true })).toBeVisible(); // the fixed speaker, as text
    await expect(ed.getByLabel('Speaker', { exact: true })).toHaveCount(0); // no speaker control

    // But a built-in still takes subcategories, fields and thresholds.
    await expect(ed.getByRole('button', { name: 'Add subcategory' })).toBeEnabled();
    await expect(ed.getByRole('button', { name: 'Add category field' })).toBeEnabled();
    await expect(threshold(ed)).toBeEnabled();
    await expect(ed.getByLabel('Subcategory threshold', { exact: true })).toBeEnabled();
    // The fixed "Other" and "Not" options close every subcategory list.
    await expect(ed.getByText('Not Caller objective', { exact: true })).toBeVisible();
  });

  test('cs4: a reviewer sees the taxonomy read-only, with no Save control', async ({ browser, colorScheme }) => {
    const reviewer = await signInDemo(browser, stack, 'Reviewer', colorScheme);
    await reviewer.goto('/#/signals');
    await expect(reviewer.getByRole('heading', { name: 'Signals', exact: true })).toBeVisible();
    await expect(reviewer.getByText('Read-only', { exact: true })).toBeVisible();
    await tree(reviewer).getByRole('link', { name: /^Caller objective\b/ }).click();
    await expect(editor(reviewer).getByRole('heading', { name: 'Caller objective', exact: true })).toBeVisible();
    await expect(reviewer.getByText('Read-only: only admins change the taxonomy', { exact: false })).toBeVisible();
    await expect(saveBar(reviewer)).toHaveCount(0);
    await expect(reviewer.getByRole('button', { name: 'Save', exact: true })).toHaveCount(0);
    await expect(reviewer.getByRole('button', { name: 'Add subcategory' })).toHaveCount(0);
    await expect(reviewer.getByRole('button', { name: 'Add category' })).toHaveCount(0);
    await expect(threshold(editor(reviewer))).toBeDisabled();
    // The pipeline switch is admin-only.
    await expect(reviewer.getByRole('button', { name: 'Switch pipeline' })).toHaveCount(0);
  });

  test('cs7: forbidden PII classes are disabled in the field editor, with the masking reason shown', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await openCategory(admin, 'intent', 'Caller objective');
    const ed = editor(admin);
    await ed.getByRole('button', { name: 'Add category field' }).click();
    await ed.getByLabel('Field name', { exact: true }).fill('Plan tier');
    await ed.getByRole('button', { name: 'Add', exact: true }).click();

    const field = ed.getByTestId('signals-field-editor');
    await expect(field).toBeVisible();
    const pii = field.getByLabel('PII class', { exact: true });
    for (const forbidden of FORBIDDEN_PII) {
      await expect(pii.locator(`option[value="${forbidden}"]`)).toBeDisabled();
    }
    await expect(pii.locator('option[value="organization"]')).toBeEnabled();
    await expect(pii.locator('option[value="none"]')).toBeEnabled();
    // "masked before any model sees it" (section 10.1 "Field editor").
    await expect(field.locator('p').filter({ hasText: 'are masked before any model sees it' })).toBeVisible(); // the field hint
    await expect(pii.locator('option[value="caller_name"]')).toHaveText(/masked before any model sees it/);

    // The draft is marked unsaved, and Discard drops it.
    await expect(saveBar(admin).getByText('Unsaved changes', { exact: true })).toBeVisible();
    await saveBar(admin).getByRole('button', { name: 'Discard' }).click();
    await expect(field).toHaveCount(0);
    await expect(saveBar(admin).getByText(/^Saved as v\d+$/)).toBeVisible();
  });

  test('cs5: caps — an over-length custom gloss and a 9th custom category are refused with a cap message', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await admin.goto('/#/signals');
    const addCategory = tree(admin).getByRole('button', { name: 'Add category' });

    // max_option_gloss_chars is 40 (section 9.6): a 60-character gloss on a new custom category.
    await addCategory.click();
    await tree(admin).getByLabel('New category name', { exact: true }).fill('Overlong gloss');
    await tree(admin).getByRole('button', { name: 'Add', exact: true }).click();
    await expect(editor(admin).getByRole('heading', { name: 'Overlong gloss', exact: true })).toBeVisible();
    await editor(admin).getByLabel('Gloss', { exact: true }).fill('x'.repeat(60));
    await expect(saveBar(admin).getByText(/option text is 60 characters; keep it to 40/)).toBeVisible();
    await expect(saveButton(admin)).toBeDisabled();

    // max_custom_signal_categories is 8: with 8 custom categories in the draft, a 9th can't be added.
    // The retail seed already has two (upsell_attempt, agent_conduct_concern), plus "Overlong gloss".
    for (let i = 2 + SEED_CUSTOM_CATEGORIES; i <= 8; i += 1) {
      await addCategory.click();
      await tree(admin).getByLabel('New category name', { exact: true }).fill(`Cap filler ${String.fromCharCode(96 + i)}`);
      await tree(admin).getByRole('button', { name: 'Add', exact: true }).click();
    }
    await expect(addCategory).toBeDisabled();
    await expect(tree(admin).getByText(/At most 8 active custom categories/)).toBeVisible();

    // Nothing reached Store: discarding restores the saved taxonomy.
    await saveBar(admin).getByRole('button', { name: 'Discard' }).click();
    await expect(addCategory).toBeEnabled();
    await expect(tree(admin).getByRole('link', { name: /^Cap filler/ })).toHaveCount(0);
  });

  test('cs6: the section-15 demo — add "Cancel account" with a reason field, test it on recent calls, save with backfill and alert', async ({
    browser,
    colorScheme,
  }) => {
    test.setTimeout(180_000);
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await openCategory(admin, 'intent', 'Caller objective');
    const ed = editor(admin);

    // Step 1: the subcategory, then the example the fake stage 2 matches.
    await ed.getByRole('button', { name: 'Add subcategory' }).click();
    await ed.getByLabel('Name', { exact: true }).last().fill('Cancel account');
    await ed.getByLabel('Gloss', { exact: true }).last().fill('Caller wants to cancel their account');
    await ed.getByRole('button', { name: 'Add', exact: true }).click();
    const sub = ed.getByTestId('signals-subcategory-editor');
    await expect(sub).toBeVisible();
    await sub.getByLabel('New subcategory example', { exact: true }).fill('cancel');
    await sub.getByRole('button', { name: 'Add example' }).click();
    await expect(sub.getByText('“cancel”', { exact: true })).toBeVisible();

    // Step 2: an enum field, reason.
    await sub.getByRole('button', { name: 'Add subcategory field' }).click();
    await sub.getByLabel('Field name', { exact: true }).fill('reason');
    await sub.getByRole('button', { name: 'Add', exact: true }).click();
    const field = sub.getByTestId('signals-field-editor');
    await field.getByLabel('Field type', { exact: true }).selectOption('enum');
    for (const value of ['price', 'service quality', 'moving', 'other']) {
      await field.getByLabel('New value', { exact: true }).fill(value);
      await field.getByRole('button', { name: 'Add value' }).click();
    }
    await field.getByLabel('Field description', { exact: true }).fill('Why the caller wants to cancel.');
    await expect(saveBar(admin).getByText('Fix these before saving:')).toHaveCount(0);

    // Step 3: Test on recent calls. The cancellation call gains the subcategory and its reason;
    // the other call's objective stays unclassified ('Other' reads the same as no subcategory).
    await saveBar(admin).getByRole('button', { name: 'Test on recent calls' }).click();
    const preview = admin.getByTestId('signals-preview');
    await expect(preview.getByRole('checkbox')).toHaveCount(2);
    await preview.getByRole('button', { name: 'Run preview' }).click();
    const cancelRow = preview.locator(`[data-preview-call="${cancelCall.call_id}"]`);
    const plainRow = preview.locator(`[data-preview-call="${plainCall.call_id}"]`);
    await expect(cancelRow.getByText('Changed', { exact: true })).toBeVisible({ timeout: 60_000 });
    await expect(cancelRow.getByText('Added: Caller objective › Cancel account', { exact: true })).toBeVisible();
    await expect(cancelRow.getByText('reason: price', { exact: true })).toBeVisible();
    await expect(cancelRow).toContainText('I want to cancel my account');
    await expect(plainRow.getByText('No change', { exact: true })).toBeVisible({ timeout: 60_000 });

    // Step 4: Save, then the activation dialog: update the last 7 days and alert on the new node.
    await saveButton(admin).click();
    const dialog = admin.getByRole('dialog', { name: /^Saved as taxonomy v\d+$/ });
    await expect(dialog).toBeVisible();
    await dialog.getByRole('checkbox', { name: /^Update calls from the last 7 days/ }).check();
    await dialog.getByRole('checkbox', { name: /^Alert on this: Caller objective › Cancel account/ }).check();
    await dialog.getByRole('button', { name: 'Confirm' }).click();
    await expect(dialog.getByText('Alert rule created: "Caller objective › Cancel account".', { exact: true })).toBeVisible();
    await expect(dialog.getByText(/^Updating [12] calls? to taxonomy v\d+/)).toBeVisible();
    await dialog.getByRole('button', { name: 'Close', exact: true }).last().click();

    // Step 5: the Calls list lights up the cancellation call's Caller need once the backfill lands
    // (the change feed drives it; no reload).
    await admin.goto('/#/calls');
    const callRow = admin.getByRole('row').filter({ hasText: 'pw-signals-cancel' });
    await expect(callRow.getByText('Cancel account', { exact: false }).first()).toBeVisible({ timeout: 60_000 });
    await expect(admin.getByRole('row').filter({ hasText: 'pw-signals-plain' }).getByText('Cancel account', { exact: false })).toHaveCount(0);

    // Step 6: the Workbench shows the subcategory, the field and the alert on the hit.
    await admin.goto(`/#/calls/${cancelCall.call_id}`);
    const card = admin.getByTestId('contact-signals');
    const hit = card.locator('[data-signal-hit^="intent."]').filter({ hasText: 'Cancel account' });
    await expect(hit).toBeVisible({ timeout: 30_000 });
    await hit.getByRole('button', { name: /^Details for/ }).click();
    await expect(hit.getByText('reason: price', { exact: true })).toBeVisible();
    await expect(hit.getByTitle('An alert rule matches this signal')).toContainText('Caller objective › Cancel account');
  });

  test('cs3: a stale save from a second admin page re-reads instead of silently overwriting', async ({ browser, colorScheme }) => {
    const first = await signInDemo(browser, stack, 'Admin', colorScheme);
    const second = await signInDemo(browser, stack, 'Admin', colorScheme);
    await openCategory(first, 'intent', 'Caller objective');
    await openCategory(second, 'intent', 'Caller objective');

    // The second page starts an edit from the same version.
    await threshold(editor(second)).fill('0.6');
    await expect(saveBar(second).getByText('Unsaved changes', { exact: true })).toBeVisible();

    // The first page saves a different threshold.
    await threshold(editor(first)).fill('0.45');
    await saveButton(first).click();
    const dialog = first.getByRole('dialog', { name: /^Saved as taxonomy v\d+$/ });
    await expect(dialog).toBeVisible();
    await dialog.getByRole('button', { name: 'Not now' }).click();

    // The second page never overwrites it: either the live change feed has already marked its draft
    // stale (Save disabled, with a notice), or its save is refused with a conflict and the editor
    // re-reads. Either way it ends showing the first admin's value.
    const staleNotice = second.getByText('This taxonomy changed elsewhere', { exact: false });
    if (await saveButton(second).isEnabled()) await saveButton(second).click();
    await expect(staleNotice.or(saveBar(second).getByText(/changed since you read it/))).toBeVisible();
    const discard = second.getByRole('button', { name: /^Discard my edits and load v\d+$/ });
    await expect(async () => {
      if (await discard.isVisible()) await discard.click();
      await expect(threshold(editor(second))).toHaveValue('0.45', { timeout: 1_000 });
    }).toPass({ timeout: 20_000 });
  });

  test('cs8: the current process has no legacy mode selector', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await admin.goto('/#/signals');
    const banner = admin.getByTestId('signals-pipeline-banner');
    await expect(banner).toContainText('Semantic similarity → Laya → Gemma');
    await expect(banner.getByRole('combobox')).toHaveCount(0);
    await expect(banner.getByRole('button')).toHaveCount(0);
  });
});

// --- Multi-segment signals (decision 25; docs/ContactSignalsV2.md section 6.5) -----------------
// The v2 merge folds one speaker's consecutive hits with the same category and subcategory into one
// hit: the anchor keeps its hit ID and quote, later spans arrive as `parts`, and `span_end` is the
// last part's end. The fake e2e stack's default script has no such run (the `returns` fake script
// does, but ASR scripts are fixed per Process start), so this route-stubs the signals read with a
// merged hit, the way evaluate-waveform's e3 does, and checks the Workbench and the waveform.
test.describe('Signals: multi-segment signals in the Workbench (decision 25)', () => {
  test('cs9: a merged hit shows "×2 segments", bands to span_end, and lists its part quotes from the keyboard', async ({
    createInvitedUser,
    ingestSample,
    waitUntilSettled,
  }, testInfo) => {
    const reviewer = await createInvitedUser('reviewer');
    const receipt = await ingestSample('call_01_compliant', { agentId: `pw-cs9-merged-${testInfo.project.name}-${Date.now()}` });
    await waitUntilSettled(receipt.call_id);

    const transcript = (await (await reviewer.page.request.get(`/store/v1/calls/${receipt.call_id}/transcript`)).json()) as {
      turns: { turn_id: number; speaker: string; start_time: number; end_time: number }[];
    };
    const agentTurns = transcript.turns.filter((t) => t.speaker === 'AGENT');
    expect(agentTurns.length).toBeGreaterThanOrEqual(2);
    const [first, second] = agentTurns;
    const hitId = `fix_proposed.abcdef012345.12345678.t${first.turn_id}b0`;
    const anchorQuote = 'I can offer you a prepaid return label by email today.';
    const partQuote = 'Once the carrier scans it, we can offer a replacement or a refund.';

    await reviewer.page.route(`**/store/v1/calls/${receipt.call_id}`, async (route) => {
      const response = await route.fetch().catch(() => null);
      const body = response ? await response.json().catch(() => null) : null;
      if (!response || body === null) return;
      body.results = (body.results ?? []).map((g: { kind: string }) => (g.kind === 'contact_signals' ? { ...g, state: 'available', version: 1 } : g));
      await route.fulfill({ response, json: body });
    });
    await reviewer.page.route(`**/store/v1/calls/${receipt.call_id}/contact-signals`, async (route) => {
      await route.fulfill({
        json: {
          call_id: receipt.call_id, artifact_id: 'art_e2e', version: 1, completeness: 'complete', passes: [],
          transcript_fingerprint: 'a'.repeat(64), generated_at: new Date().toISOString(), pipeline: 'v2',
          taxonomy: { version: 1, digest: 'a'.repeat(64) },
          stages: [{ stage: 'categorize', included: true }, { stage: 'subcategorize', included: true }],
          segmentation: { segmenter_version: 'seg-v1', window_seconds: 7, segments: 10, scored_segments: 10, skipped_unattributed: 0, skipped_system: 0, interpolated_turns: 0 },
          stage1_digests: {}, feedback: [], alerts: [], taxonomy_status: null, text_withheld: false, comparison_preview_id: null,
          signals: [{
            id: hitId, kind: 'fix_proposed', label: 'Proposed fix', start: first.start_time, end: first.end_time,
            speaker: 'AGENT', quote: anchorQuote, turn_id: first.turn_id, char_start: 0, char_end: anchorQuote.length, confidence: 0.9,
            review_status: 'unreviewed', category_id: 'fix_proposed', category_digest: 'abcdef012345', category_confidence: 0.9,
            subcategory_id: null, subcategory_label: null, subcategory_digest: null, subcategory_confidence: null,
            span: { block: 0, first_window: 0, last_window: 0, timing: 'words', context_start: first.start_time, context_end: second.end_time },
            fields: [], quote_narrowed: false,
            parts: [{ turn_id: second.turn_id, block: 0, start: second.start_time, end: second.end_time, quote: partQuote, char_start: 0, char_end: partQuote.length }],
            span_end: second.end_time,
          }],
        },
      });
    });

    await reviewer.page.goto(`/#/calls/${receipt.call_id}`);

    const card = reviewer.page.getByTestId('contact-signals');
    const row = card.locator(`[data-signal-hit="${hitId}"]`);
    await expect(row).toBeVisible({ timeout: 30_000 });
    await expect(row).toHaveAttribute('data-segments', '2');

    // The chip says how many segments the one signal spans; the time range runs to span_end.
    await expect(row.getByTestId('signal-segments')).toHaveText('×2 segments');
    const clock = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
    await expect(row).toContainText(`${clock(first.start_time)}–${clock(second.end_time)}`);
    await expect(row).toContainText(anchorQuote);
    await expect(row.getByText(partQuote)).toHaveCount(0); // collapsed until asked

    // Keyboard: the disclosure is a real button, reachable and operable without a pointer.
    await row.getByRole('button', { name: /^Details for/ }).click();
    const toggle = row.getByRole('button', { name: 'Show 1 more segment' });
    await toggle.focus();
    await expect(toggle).toBeFocused();
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await reviewer.page.keyboard.press('Enter');
    const hide = row.getByRole('button', { name: 'Hide the other segment' });
    await expect(hide).toHaveAttribute('aria-expanded', 'true');
    const parts = row.getByTestId('signal-parts');
    await expect(parts).toBeVisible();
    await expect(parts).toContainText(partQuote);
    await expect(parts.getByRole('button', { name: new RegExp(`^Segment 2 of 2 at ${clock(second.start_time)}: jump to the turn$`) })).toBeVisible();

    // The waveform band covers the whole signal (start → span_end) and says so in its title.
    const deck = reviewer.page.getByTestId('thread-waveform');
    await expect(deck).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    const band = deck.getByTestId(`waveform-span-band-${hitId}`);
    await expect(band).toHaveAttribute('data-segments', '2');
    await expect(band).toHaveAttribute('title', new RegExp(`${clock(first.start_time)}–${clock(second.end_time)} \\(×2 segments\\)$`));
  });
});

// --- Dense calls (benchmarks/2026-09-26-gemma-signals-v2.md: 14–42 kept spans per real call) ------
// Real Gemma output on a retail call can carry 40-odd spans. This route-stubs 42 hits (every
// category, long subcategory names, field chips, a merged hit, an alert) onto a settled call and
// checks that the Workbench stays inside the viewport in both schemes: no horizontal page scroll,
// every row inside the card, bands and pins inside the waveform deck.
test.describe('Signals: a dense call in the Workbench (30–45 spans)', () => {
  test('cs10: 42 spans on one call keep the Workbench inside the viewport', async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
    test.setTimeout(120_000);
    const reviewer = await createInvitedUser('reviewer');
    const receipt = await ingestSample('call_01_compliant', { agentId: `pw-cs10-dense-${testInfo.project.name}-${Date.now()}` });
    await waitUntilSettled(receipt.call_id);
    const transcript = (await (await reviewer.page.request.get(`/store/v1/calls/${receipt.call_id}/transcript`)).json()) as {
      turns: { turn_id: number; speaker: string; start_time: number; end_time: number }[];
    };
    const turns = transcript.turns;
    expect(turns.length).toBeGreaterThan(2);
    const kinds: [string, string, string][] = [
      ['intent', 'Caller objective', 'CALLER'], ['issue', 'Reported issue', 'CALLER'], ['friction', 'Friction point', 'CALLER'],
      ['fix_proposed', 'Proposed fix', 'AGENT'], ['agent_reports_completed', 'Agent completed', 'AGENT'],
      ['caller_confirms_resolved', 'Caller confirmed', 'CALLER'], ['deferred', 'Deferred', 'AGENT'],
      ['upsell_attempt', 'Upsell / cross-sell attempt', 'AGENT'],
    ];
    const longSub = 'Retention discount offer (cancel-save) with a deliberately long subcategory name';
    const signals = Array.from({ length: 42 }, (_, i) => {
      const turn = turns[i % turns.length];
      const block = Math.floor(i / turns.length);
      const [kind, label, speaker] = kinds[i % kinds.length];
      const quote = `Span ${i + 1}: ${'a fairly long quoted sentence from the masked transcript that must wrap, '.repeat(1 + (i % 3))}`.trim();
      const merged = i === 5 && turns.length > 3;
      return {
        id: `${kind}.abcdef012345.12345678.t${turn.turn_id}b${block}`, kind, label, start: turn.start_time, end: turn.end_time, speaker, quote,
        turn_id: turn.turn_id, char_start: 0, char_end: quote.length, confidence: 0.6 + (i % 4) / 10, review_status: 'unreviewed',
        category_id: kind, category_digest: 'abcdef012345', category_confidence: 0.9,
        subcategory_id: i % 3 === 0 ? 'other' : `sub_${i}`, subcategory_label: i % 3 === 0 ? 'Other' : i % 5 === 1 ? longSub : `Subcategory ${i}`,
        subcategory_digest: i % 3 === 0 ? null : 'abcdef012345', subcategory_confidence: i % 3 === 0 ? null : 0.8,
        span: { block, first_window: 0, last_window: 0, timing: 'words', context_start: turn.start_time, context_end: turn.end_time },
        fields: i % 4 === 1
          ? [{ field_id: 'reason', name: 'reason', value: i % 8 === 1 ? 'service quality and a very long free-text value that should wrap cleanly' : 'price',
               status: 'extracted', type: 'enum', evidence: null, surface: null, char_start: 0, char_end: 5, turn_id: null }]
          : [],
        quote_narrowed: false,
        parts: merged ? [{ turn_id: turns[3].turn_id, block: 0, start: turns[3].start_time, end: turns[3].end_time, quote: 'The second segment.', char_start: 0, char_end: 19 }] : [],
        span_end: merged ? turns[3].end_time : null,
      };
    });
    await reviewer.page.route(`**/store/v1/calls/${receipt.call_id}`, async (route) => {
      const response = await route.fetch().catch(() => null);
      const body = response ? await response.json().catch(() => null) : null;
      if (!response || body === null) return;
      body.results = (body.results ?? []).map((g: { kind: string }) => (g.kind === 'contact_signals' ? { ...g, state: 'available', version: 1 } : g));
      await route.fulfill({ response, json: body });
    });
    await reviewer.page.route(`**/store/v1/calls/${receipt.call_id}/contact-signals`, async (route) => {
      await route.fulfill({
        json: {
          call_id: receipt.call_id, artifact_id: 'art_e2e', version: 1, completeness: 'complete', passes: [],
          transcript_fingerprint: 'a'.repeat(64), generated_at: new Date().toISOString(), pipeline: 'v2',
          taxonomy: { version: 1, digest: 'a'.repeat(64) },
          stages: [{ stage: 'categorize', included: true }, { stage: 'subcategorize', included: true }, { stage: 'extract', included: true }],
          segmentation: { segmenter_version: 'seg-v1', window_seconds: 7, segments: 60, scored_segments: 60, skipped_unattributed: 0, skipped_system: 0, interpolated_turns: 0 },
          stage1_digests: {}, feedback: [],
          alerts: [{ rule_id: 'rule_dense', name: 'Caller objective › Cancel account', hit_ids: [signals[0].id, signals[8].id] }],
          taxonomy_status: null, text_withheld: false, comparison_preview_id: null, signals,
        },
      });
    });

    const page = reviewer.page;
    await page.goto(`/#/calls/${receipt.call_id}`);
    const card = page.getByTestId('contact-signals');
    await expect(card.locator('[data-signal-hit]')).toHaveCount(42, { timeout: 30_000 });
    const deck = page.getByTestId('thread-waveform');
    await expect(deck).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
    await expect(deck.locator('[data-signal-band]')).toHaveCount(42);

    const layout = await page.evaluate(() => {
      const doc = document.documentElement;
      const card = document.querySelector('[data-testid="contact-signals"]')!.getBoundingClientRect();
      const deck = document.querySelector('[data-testid="thread-waveform"]')!.getBoundingClientRect();
      const rows = Array.from(document.querySelectorAll('[data-signal-hit]')).map((r) => r.getBoundingClientRect());
      const bands = Array.from(document.querySelectorAll('[data-signal-band]')).map((b) => b.getBoundingClientRect());
      const pins = Array.from(document.querySelectorAll('[aria-label="Waveform markers"] button')).map((b) => b.getBoundingClientRect());
      const inside = (r: DOMRect, box: DOMRect) => r.left >= box.left - 1 && r.right <= box.right + 1;
      return {
        pageOverflow: doc.scrollWidth - doc.clientWidth,
        rowsOutside: rows.filter((r) => !inside(r, card)).length,
        bandsOutside: bands.filter((r) => !inside(r, deck)).length,
        pinsOutside: pins.filter((r) => !inside(r, deck)).length,
        pins: pins.length,
      };
    });
    expect(layout.pageOverflow).toBeLessThanOrEqual(0);
    expect(layout.rowsOutside).toBe(0);
    expect(layout.bandsOutside).toBe(0);
    expect(layout.pinsOutside).toBe(0);
    expect(layout.pins).toBeGreaterThan(0);
    expect(layout.pins).toBeLessThan(42); // nearby markers cluster into one pin
    await expect(card.getByText('Caller objective › Cancel account', { exact: true }).first()).toBeVisible();

    // A dense call gets a category filter: one pressed button per category present, with counts.
    const filter = card.getByTestId('signals-filter');
    const all = filter.getByRole('button', { name: 'All 42', exact: true });
    await expect(all).toHaveAttribute('aria-pressed', 'true');
    await filter.getByRole('button', { name: 'Caller objective 6', exact: true }).click();
    await expect(card.locator('[data-signal-hit]')).toHaveCount(6);
    await expect(card.locator('[data-signal-hit^="intent."]')).toHaveCount(6);
    await expect(filter.getByRole('button', { name: 'Caller objective 6', exact: true })).toHaveAttribute('aria-pressed', 'true');
    await filter.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath('dense-signals-filtered.png'), animations: 'disabled' });
    await all.click();
    await expect(card.locator('[data-signal-hit]')).toHaveCount(42);
    await page.screenshot({ path: testInfo.outputPath('dense-workbench.png'), animations: 'disabled' });
  });
});

// Notes for spec authors: the category, subcategory and field editors label their inputs "Name",
// "Gloss", "Field name", "Field type", "Field description" and "PII class"; "Threshold" and
// "Subcategory threshold" are both on a category, so match "Threshold" with exact: true. A new
// subcategory's inputs are the LAST "Name"/"Gloss" in the category editor while its add form is
// open. "Test on recent calls" lists settled calls by agent and time (not by call ID); each result
// row carries data-preview-call="<call_id>".
