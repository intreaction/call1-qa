// Contact Signals rules engine in Evaluate (contract 1.4.0, docs/SignalsEmbeddings.md §10): the
// per-category detection configuration, a category's "How it's detected" recipe editor (the
// retail pack's recipes, live phrase checks, save), the "Why" disclosure on Workbench hits, and the
// rules counts in "Test on recent calls".
//
// The taxonomy and its settings are one Store-wide record, so the editor tests run on PRIVATE
// stacks per project (dark, light), signed in through Store demo mode as evaluate-signals.spec.ts
// does:
//   - "retail": the retail seed (call1/store/seeds/signals_retail_v1.json), whose categories carry
//     the pack's R2 recipes. Nothing runs detection there, so its pinned example bank is never read.
//   - "fake rules": the built-in taxonomy with no recipes. The test writes intent's and
//     caller_confirms_resolved's recipes through the editor, ingests
//     call_01 (the fake script's caller says "a question about a fee" and "that makes sense") and
//     reads the hit's why in the Workbench, as tests/e2e/test_signal_rules_e2e.py does on the API.
// The detailed why rendering (neighbours, outcomes, a Gemma hit, a hit without why) route-stubs
// the signals read on the shared stack, the way evaluate-signals cs9 does.
import type { Browser, Locator, Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { startPrivateStack, type PrivateStack } from './private-stack';

type Persona = 'Admin' | 'Supervisor' | 'Reviewer';
type Scheme = 'light' | 'dark' | 'no-preference' | null | undefined;

async function signInDemo(browser: Browser, stack: PrivateStack, persona: Persona, colorScheme: Scheme, viewport?: { width: number; height: number }): Promise<Page> {
  const context = await browser.newContext({ baseURL: stack.info.store_url, colorScheme: colorScheme ?? undefined, ...(viewport ? { viewport } : {}) });
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

const editor = (page: Page) => page.getByTestId('signals-category-editor');
const recipe = (page: Page) => page.getByTestId('signals-recipe-editor');
const lexicon = (page: Page) => page.getByTestId('signals-lexicon-editor');
const detection = (page: Page) => page.getByTestId('signals-detection');
const saveBar = (page: Page) => page.getByTestId('signals-save-bar');
const saveButton = (page: Page) => saveBar(page).getByRole('button', { name: 'Save', exact: true });

async function openCategory(page: Page, id: string, name: string): Promise<void> {
  await page.goto(`/#/signals/${id}`);
  await expect(editor(page).getByRole('heading', { name, exact: true })).toBeVisible();
}

async function addPhrase(scope: Locator, phrase: string): Promise<void> {
  await scope.getByLabel('New phrase', { exact: true }).fill(phrase);
  await scope.getByRole('button', { name: 'Add phrase' }).click();
}

async function saveAndDismiss(page: Page): Promise<void> {
  await expect(saveButton(page)).toBeEnabled();
  await saveButton(page).click();
  const dialog = page.getByRole('dialog', { name: /^Saved as taxonomy v\d+$/ });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: 'Not now' }).click();
  await expect(dialog).toHaveCount(0);
}

test.describe('Signal rules: the recipe editor on the retail pack', () => {
  let stack: PrivateStack;

  test.beforeAll(async ({}, testInfo) => {
    testInfo.setTimeout(180_000);
    stack = await startPrivateStack({
      name: `pw-signal-rules-retail-${testInfo.project.name}`,
      storeEnv: { CALL1_STORE_DEMO: '1' },
      signalsPipeline: 'v2',
      signalsSeed: 'call1/store/seeds/signals_retail_v1.json',
    });
  });
  test.afterAll(async () => {
    await stack?.close();
  });

  test('sr1: a pack recipe reads in words — origin, engine, score, gate, call position and phrases', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await openCategory(admin, 'issue', 'Reported issue');
    const r = recipe(admin);
    await expect(r.getByRole('heading', { name: "How it's detected" })).toBeVisible();
    await expect(r.getByTestId('signals-recipe-origin')).toContainText('From the Retail pack v1, tuned on 25 public calls');
    await expect(r.getByRole('radio', { name: 'Rules + examples' })).toBeChecked();
    await expect(r.getByLabel('Score needed', { exact: true })).toHaveValue('0.4');
    await expect(r.getByLabel('Must also pass', { exact: true })).toHaveValue('phrase');
    await expect(r.getByLabel('Where in the call', { exact: true })).toHaveValue('q1');
    await expect(r.getByText('Caller turns only', { exact: false })).toBeVisible();
    const lx = lexicon(admin);
    await expect(lx.getByText('Phrases (9 of 24)', { exact: true })).toBeVisible();
    await expect(lx.getByRole('radio', { name: 'Patterns (regular expressions)' })).toBeChecked();
    await expect(lx.getByLabel('Phrase weight', { exact: true })).toHaveValue('0.5');
    await expect(r.getByRole('checkbox', { name: /^Gemma double-check/ })).not.toBeChecked();

    // caller_confirms_resolved is the one pack recipe with the Gemma check; friction's "either" gate.
    await openCategory(admin, 'caller_confirms_resolved', 'Caller confirmed');
    await expect(recipe(admin).getByRole('checkbox', { name: /^Gemma double-check/ })).toBeChecked();
    await expect(recipe(admin).getByLabel('Where in the call', { exact: true })).toHaveValue('h2');
    await openCategory(admin, 'friction', 'Friction point');
    await expect(recipe(admin).getByLabel('Must also pass', { exact: true })).toHaveValue('phrase_or_similar');
    await expect(recipe(admin).getByLabel('Similar-examples share at least', { exact: true })).toHaveValue('0.1');

    await expect(recipe(admin)).not.toContainText('saved but not used');
  });

  test('sr2: phrase checks mirror Store — regex safety, a letter, duplicates and the PII warning', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await openCategory(admin, 'issue', 'Reported issue');
    const lx = lexicon(admin);
    const input = lx.getByLabel('New phrase', { exact: true });
    const add = lx.getByRole('button', { name: 'Add phrase' });
    const check = lx.getByTestId('signals-phrase-check');

    await input.fill('(?=refund)');
    await expect(check).toContainText('no look-arounds, named groups or inline flags');
    await expect(add).toBeDisabled();
    await expect(input).toHaveAttribute('aria-invalid', 'true');

    await input.fill('(\\w+ )*refund');
    await expect(check).toContainText('an unbounded repeat around another repeat is not allowed');
    await expect(add).toBeDisabled();

    await input.fill('(refund)\\1');
    await expect(check).toContainText('back-references, look-arounds and conditionals are not allowed');

    await input.fill('[0-9]+');
    await expect(check).toContainText('a phrase contains at least one letter');
    await expect(add).toBeDisabled();

    await input.fill('card ending 4111 1111');
    await expect(check).toContainText('looks like caller details');
    await expect(add).toBeEnabled(); // a warning: Store is the authority on its PII rules

    await input.fill('\\brefund(ed|s)?\\b');
    await expect(check).toHaveText('');
    await add.click();
    await expect(lx.getByText('Phrases (10 of 24)', { exact: true })).toBeVisible();
    await expect(lx.getByText('\\brefund(ed|s)?\\b', { exact: true })).toBeVisible();
    await expect(recipe(admin).getByText('Edited', { exact: true })).toBeVisible();
    await expect(recipe(admin).getByTestId('signals-recipe-origin')).toContainText(', edited here');

    // The same phrase again is a duplicate; the keyboard adds with Enter.
    await input.fill('\\brefund(ed|s)?\\b');
    await expect(check).toContainText('already in the list');
    await expect(add).toBeDisabled();
    await input.fill('\\bexchange\\b');
    await input.press('Enter');
    await expect(lx.getByText('Phrases (11 of 24)', { exact: true })).toBeVisible();

    // Removing is a labelled button per phrase.
    await lx.getByRole('button', { name: 'Remove phrase 11' }).click();
    await expect(lx.getByText('Phrases (10 of 24)', { exact: true })).toBeVisible();

    // Nothing reached Store.
    await saveBar(admin).getByRole('button', { name: 'Discard' }).click();
    await expect(lx.getByText('Phrases (9 of 24)', { exact: true })).toBeVisible();
  });

  test('sr3: a recipe that needs phrases and has none blocks Save; the fix saves and persists', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await openCategory(admin, 'issue', 'Reported issue');
    const r = recipe(admin);
    const lx = lexicon(admin);

    // Dropping the list takes the "a phrase matches" requirement with it.
    await lx.getByRole('button', { name: 'Remove phrase list' }).click();
    await expect(lx.getByText('Phrases (0 of 24)', { exact: true })).toBeVisible();
    await expect(r.getByLabel('Must also pass', { exact: true })).toHaveValue('none');

    // Asking for a phrase with no list is refused before Store sees it.
    await r.getByLabel('Must also pass', { exact: true }).selectOption('phrase');
    await expect(r.getByText(/"a phrase matches" tests the phrase list, so add at least one phrase/)).toBeVisible();
    await expect(saveBar(admin).getByText('Fix these before saving:')).toBeVisible();
    await expect(saveButton(admin)).toBeDisabled();

    // An out-of-range score is not taken into the draft.
    const score = r.getByLabel('Score needed', { exact: true });
    await score.fill('0.99');
    await expect(r.getByText('Use a number from 0.05 to 0.95.')).toBeVisible();

    // Fix: plain words, one phrase, a weight, a new score and a custom window.
    await lx.getByRole('radio', { name: 'Words' }).check();
    await addPhrase(lx, 'not working');
    await expect(saveBar(admin).getByText('Fix these before saving:')).toHaveCount(0);
    await lx.getByLabel('Phrase weight', { exact: true }).fill('0.45');
    await lx.getByLabel('Negation window (words)', { exact: true }).fill('2');
    await score.fill('0.42');
    await r.getByLabel('Where in the call', { exact: true }).selectOption('custom');
    await r.getByLabel('Starts from', { exact: true }).fill('0.1');
    await r.getByLabel('Starts by', { exact: true }).fill('0.6');
    await r.getByRole('checkbox', { name: /^Gemma double-check/ }).check();
    await saveAndDismiss(admin);

    // Reload: every part came back from Store.
    await admin.reload();
    await expect(editor(admin).getByRole('heading', { name: 'Reported issue', exact: true })).toBeVisible();
    const r2 = recipe(admin);
    const lx2 = lexicon(admin);
    await expect(lx2.getByText('Phrases (1 of 24)', { exact: true })).toBeVisible();
    await expect(lx2.getByText('not working', { exact: true })).toBeVisible();
    await expect(lx2.getByRole('radio', { name: 'Words' })).toBeChecked();
    await expect(lx2.getByLabel('Phrase weight', { exact: true })).toHaveValue('0.45');
    await expect(lx2.getByLabel('Negation window (words)', { exact: true })).toHaveValue('2');
    await expect(r2.getByLabel('Score needed', { exact: true })).toHaveValue('0.42');
    await expect(r2.getByLabel('Must also pass', { exact: true })).toHaveValue('phrase');
    await expect(r2.getByLabel('Where in the call', { exact: true })).toHaveValue('custom');
    await expect(r2.getByLabel('Starts from', { exact: true })).toHaveValue('0.1');
    await expect(r2.getByLabel('Starts by', { exact: true })).toHaveValue('0.6');
    await expect(r2.getByRole('checkbox', { name: /^Gemma double-check/ })).toBeChecked();
    await expect(r2.getByTestId('signals-recipe-origin')).toContainText('Retail pack v1');

    // Model engine keeps the recipe (engine: gemma) and says so.
    await r2.getByRole('radio', { name: 'Model (Gemma)' }).check();
    await expect(r2).toContainText("Its rules recipe is kept but not used.");
    await expect(r2.getByLabel('Score needed', { exact: true })).toHaveCount(0);
    await r2.getByRole('radio', { name: 'Rules + examples' }).check();
    await expect(r2.getByLabel('Score needed', { exact: true })).toHaveValue('0.42');
    await expect(saveBar(admin).getByText('Unsaved changes')).toHaveCount(0);
  });

  test('sr4: categories control detection without a global switch', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await openCategory(admin, 'fix_proposed', 'Proposed fix');
    await expect(detection(admin)).toHaveCount(0);
    await expect(recipe(admin).getByRole('radio', { name: 'Rules + examples' })).toBeChecked();
    await expect(recipe(admin)).not.toContainText('saved but not used');
    await expect(editor(admin).getByRole('status').filter({ hasText: "Don't paste caller details" })).toHaveCount(0);
    const reviewer = await signInDemo(browser, stack, 'Reviewer', colorScheme);
    await openCategory(reviewer, 'fix_proposed', 'Proposed fix');
    await expect(detection(reviewer)).toHaveCount(0);
    await expect(recipe(reviewer).getByLabel('Score needed', { exact: true })).toBeDisabled();
    await expect(lexicon(reviewer).getByLabel('New phrase', { exact: true })).toHaveCount(0);
  });

  test('sr5: at 390 px the recipe editor fits the viewport', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme, { width: 390, height: 844 });
    await openCategory(admin, 'friction', 'Friction point');
    await expect(recipe(admin)).toBeVisible();
    await expect(detection(admin)).toHaveCount(0);
    const overflow = await admin.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, 'horizontal page scroll at 390 px').toBeLessThanOrEqual(0);
    const box = await recipe(admin).boundingBox();
    expect(box && box.x >= 0 && box.x + box.width <= 390, 'recipe editor inside the viewport').toBeTruthy();
  });
});

test.describe('Signal rules: detection by rules on fake handlers, end to end', () => {
  let stack: PrivateStack;

  test.beforeAll(async ({}, testInfo) => {
    testInfo.setTimeout(180_000);
    stack = await startPrivateStack({
      name: `pw-signal-rules-fake-${testInfo.project.name}`,
      storeEnv: { CALL1_STORE_DEMO: '1' },
      signalsPipeline: 'v2',
    });
  });
  test.afterAll(async () => {
    await stack?.close();
  });

  test('sr6: recipes written in the editor determine detection, and a new call explains its hits', async ({ browser, colorScheme }) => {
    test.setTimeout(180_000);
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);

    // Caller objective: rules, a phrase must match, "question about", weight 0.5, score 0.4.
    await openCategory(admin, 'intent', 'Caller objective');
    let r = recipe(admin);
    await expect(r.getByRole('radio', { name: 'Model (Gemma)' })).toBeChecked();
    await expect(r.getByTestId('signals-recipe-origin')).toHaveCount(0);
    await r.getByRole('radio', { name: 'Rules + examples' }).check();
    await expect(r.getByLabel('Score needed', { exact: true })).toHaveValue('0.375');
    await r.getByLabel('Score needed', { exact: true }).fill('0.4');
    await addPhrase(lexicon(admin), 'question about');
    await r.getByLabel('Must also pass', { exact: true }).selectOption('phrase');
    await lexicon(admin).getByLabel('Phrase weight', { exact: true }).fill('0.5');

    // Caller confirmed: the same shape with the Gemma double-check.
    await openCategory(admin, 'caller_confirms_resolved', 'Caller confirmed');
    r = recipe(admin);
    await r.getByRole('radio', { name: 'Rules + examples' }).check();
    await r.getByLabel('Score needed', { exact: true }).fill('0.4');
    await addPhrase(lexicon(admin), 'makes sense');
    await r.getByLabel('Must also pass', { exact: true }).selectOption('phrase');
    await lexicon(admin).getByLabel('Phrase weight', { exact: true }).fill('0.5');
    await r.getByRole('checkbox', { name: /^Gemma double-check/ }).check();
    await saveAndDismiss(admin);

    await admin.goto('/#/signals');
    await expect(detection(admin)).toHaveCount(0);

    const receipt = await stack.ingest('call_01_compliant', { agentId: `pw-rules-${Date.now()}` });
    await settle(stack, receipt.conversation_id);

    const reviewer = await signInDemo(browser, stack, 'Reviewer', colorScheme);
    await reviewer.goto(`/#/calls/${receipt.call_id}`);
    const card = reviewer.getByTestId('contact-signals');
    await expect(card).toHaveAttribute('data-pipeline', 'v2', { timeout: 30_000 });
    const intent = card.locator('[data-signal-hit^="intent."]').first();
    await expect(intent).toBeVisible({ timeout: 30_000 });
    const why = intent.getByTestId('signal-why');
    await expect(why).toHaveAttribute('data-source', 'rules');
    await expect(why).toContainText('Found by rules');
    const toggle = why.getByRole('button', { name: 'Why' });
    await toggle.focus();
    await reviewer.keyboard.press('Enter');
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');
    const summary = why.getByTestId('signal-why-summary');
    await expect(summary).toContainText(/^Found by rules · score \d\.\d\d ≥ 0\.40 · similar examples \d\.\d\d · phrase 'question about' \(matched, \+0\.50\)/);
    await expect(why).toContainText('Category decided by rules · subcategory decided by rules');
    await expect(why.getByRole('list', { name: 'Rule outcomes' })).toContainText('Phrase ✓');

    // Categories without a recipe are still Gemma's (the fake classifier), and say so.
    const gemmaHit = card.locator('[data-signal-hit]').filter({ has: reviewer.locator('[data-testid="signal-why"][data-source="gemma"]') });
    if (await gemmaHit.count()) await expect(gemmaHit.first().getByTestId('signal-why')).toContainText('Found by Gemma');

    // The fake stage 2 may confirm or reject the checked category; a confirmed hit says so.
    const confirmed = card.locator('[data-signal-hit^="caller_confirms_resolved."]');
    if (await confirmed.count()) await expect(confirmed.first().getByTestId('signal-why-checked')).toHaveText('Gemma double-checked ✓');

    // "Test on recent calls" counts the rules hits per category.
    await admin.goto('/#/signals/intent');
    await saveBar(admin).getByRole('button', { name: 'Test on recent calls' }).click();
    const preview = admin.getByTestId('signals-preview');
    await preview.getByRole('button', { name: 'Run preview' }).click();
    const row = preview.locator(`[data-preview-call="${receipt.call_id}"]`);
    const counts = row.getByTestId('signals-preview-counts');
    await expect(counts).toBeVisible({ timeout: 60_000 });
    await expect(counts.locator('[data-category="intent"]')).toContainText(/Caller objective \d+ · (all|\d+) by rules/);
  });
});

// --- The why disclosure in detail (route-stubbed, shared stack) ---------------------------------
test.describe('Signal rules: the "Why" disclosure on Workbench hits', () => {
  test('sr7: rules hit with check, neighbours and outcomes; a Gemma hit; a hit without why', async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
    const reviewer = await createInvitedUser('reviewer');
    const receipt = await ingestSample('call_01_compliant', { agentId: `pw-sr7-why-${testInfo.project.name}-${Date.now()}` });
    await waitUntilSettled(receipt.call_id);
    const transcript = (await (await reviewer.page.request.get(`/store/v1/calls/${receipt.call_id}/transcript`)).json()) as {
      turns: { turn_id: number; speaker: string; start_time: number; end_time: number }[];
    };
    const caller = transcript.turns.filter((t) => t.speaker === 'CALLER');
    const agent = transcript.turns.filter((t) => t.speaker === 'AGENT');
    expect(caller.length).toBeGreaterThanOrEqual(2);
    expect(agent.length).toBeGreaterThanOrEqual(1);

    const base = (id: string, category: string, label: string, t: { turn_id: number; start_time: number; end_time: number }, speaker: string, quote: string) => ({
      id, kind: category, label, start: t.start_time, end: t.end_time, speaker, quote, turn_id: t.turn_id, char_start: 0, char_end: quote.length, confidence: 0.9,
      review_status: 'unreviewed', category_id: category, category_digest: 'abcdef012345', category_confidence: 0.9,
      subcategory_id: null, subcategory_label: null, subcategory_digest: null, subcategory_confidence: null,
      span: { block: 0, first_window: 0, last_window: 0, timing: 'words', context_start: t.start_time, context_end: t.end_time },
      fields: [], quote_narrowed: false, parts: [], span_end: null,
    });
    const rulesId = `caller_confirms_resolved.abcdef012345.12345678.t${caller[1].turn_id}b0`;
    const gemmaId = `fix_proposed.abcdef012345.12345678.t${agent[0].turn_id}b0`;
    const plainId = `intent.abcdef012345.12345678.t${caller[0].turn_id}b0`;
    const signals = [
      {
        ...base(rulesId, 'caller_confirms_resolved', 'Caller confirmed', caller[1], 'CALLER', 'Okay, that makes sense.'),
        why: {
          category_source: 'rules', subcategory_source: 'gemma', check: 'confirmed',
          rule: {
            span_key: `caller_confirms_resolved.t${caller[1].turn_id}b0`, category_id: 'caller_confirms_resolved', segment_index: 3, recipe_digest: 'abcdef012345',
            score: 0.6234, threshold: 0.375, knn_share: 0.5512, lexicon_weight: 0.2, lexicon_match: true, lexicon_phrase: 0,
            outcomes: [
              { rule_id: 'speaker', type: 'speaker', result: 'pass', value: null, phrase_index: null, vetoed: false },
              { rule_id: 'position', type: 'call_position', result: 'pass', value: 0.82, phrase_index: null, vetoed: false },
              { rule_id: 'lexicon', type: 'phrase', result: 'pass', value: null, phrase_index: 0, vetoed: false },
            ],
            neighbours: [
              { entry_id: 'retail-v1:1042', cosine: 0.912, carries_category: true, subcategory_id: null },
              { entry_id: 'taxonomy:caller_confirms_resolved#0', cosine: 0.871, carries_category: true, subcategory_id: 'other' },
              { entry_id: 'retail-v1:77', cosine: 0.803, carries_category: false, subcategory_id: null },
            ],
            subcategory_id: null, subcategory_share: 0.4, check: true,
          },
        },
      },
      { ...base(gemmaId, 'fix_proposed', 'Proposed fix', agent[0], 'AGENT', 'I can waive that fee for you.'), why: { category_source: 'gemma', subcategory_source: 'gemma', check: null, rule: null } },
      { ...base(plainId, 'intent', 'Caller objective', caller[0], 'CALLER', 'I have a question about a fee.'), why: null },
    ];

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
          signals,
        },
      });
    });

    await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
    const card = reviewer.page.getByTestId('contact-signals');
    const rulesRow = card.locator(`[data-signal-hit="${rulesId}"]`);
    await expect(rulesRow).toBeVisible({ timeout: 30_000 });

    const why = rulesRow.getByTestId('signal-why');
    await expect(why).toHaveAttribute('data-source', 'rules');
    await expect(why.getByTestId('signal-why-checked')).toHaveText('Gemma double-checked ✓');
    await expect(why.getByTestId('signal-why-panel')).toHaveCount(0); // collapsed

    const toggle = why.getByRole('button', { name: 'Why' });
    await toggle.focus();
    await expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await reviewer.page.keyboard.press('Enter');
    await expect(toggle).toHaveAttribute('aria-expanded', 'true');
    // The phrase text comes from the current taxonomy's recipe; the shared stack's has none, so it is numbered.
    await expect(why.getByTestId('signal-why-summary')).toHaveText(
      /^Found by rules · score 0\.62 ≥ 0\.38 · similar examples 0\.55 · phrase( '.+'| 1) \(matched, \+0\.20\) · Gemma double-checked ✓$/,
    );
    await expect(why).toContainText('Category decided by rules · subcategory decided by Gemma');
    const outcomes = why.getByRole('list', { name: 'Rule outcomes' });
    await expect(outcomes).toContainText('Speaker ✓');
    await expect(outcomes).toContainText('Call position ✓ (starts at 82% of the call)');
    await expect(outcomes).toContainText('Phrase ✓ (phrase 1)');
    const neighbours = why.getByRole('list', { name: 'Nearest examples' }).getByRole('listitem');
    await expect(neighbours).toHaveCount(3);
    await expect(neighbours.nth(0)).toContainText('retail-v1:1042');
    await expect(neighbours.nth(0)).toContainText('cosine 0.91');
    await expect(neighbours.nth(0)).toContainText('labelled Caller confirmed');
    await expect(neighbours.nth(2)).toContainText('labelled otherwise');
    // Ids and numbers only: no example text.
    await expect(why).not.toContainText('makes sense');

    const gemma = card.locator(`[data-signal-hit="${gemmaId}"]`).getByTestId('signal-why');
    await expect(gemma).toHaveAttribute('data-source', 'gemma');
    await expect(gemma).toContainText('Found by Gemma');
    await expect(gemma.getByTestId('signal-why-checked')).toHaveCount(0);

    await expect(card.locator(`[data-signal-hit="${plainId}"]`)).toBeVisible();
    await expect(card.locator(`[data-signal-hit="${plainId}"]`).getByTestId('signal-why')).toHaveCount(0);
  });
});
