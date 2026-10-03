// Dual transcription merged on the customer's vocabulary (docs/DualAsr.md, decision 33, section 8
// "Evaluate", section 9 "Tests": "admin edits and saves, a digit term is refused inline, a pack
// term is switched off, a conflict; the Workbench underline, its tooltip on hover and keyboard
// focus, the masked-original wording, and the base_only notice"). Dark and light run automatically
// (playwright.config.ts projects).
//
// The vocabulary is one Store-wide singleton record, and once it is on every ingest plans dual
// transcription. So, like evaluate-signals.spec.ts, the admin editor and the real-merge Workbench
// tests run on a PRIVATE stack per project (dark, light), started the way `python -m call1.launch
// --demo --handlers fake` starts the class demo: Store demo mode (the "Continue as Demo …" sign-in)
// and the retail vocabulary seed installed as the industry pack. Its first ingest plays the fake
// ASR handler's `vocabulary` script (call1/process/handlers/fake.py VOCABULARY_SCRIPT): "Parakeet"
// mishears three pack terms, the fake "Whisper" pass hears them, and the real rule merge writes
// them over the base words, so Store computes the view's char offsets and `heard` itself. The
// second ingest adds `vocabulary_fail`, for a real base_only transcript. The shared stack never
// gets a vocabulary, so no other spec's ingests depend on this file's run order.
//
// Only the masked-original wording stays a response stub on the shared stack: no fake script
// masks a replaced phrase, and Store's own masking of `heard` is covered in pytest
// (tests/store/test_asr_vocabulary.py).
import type { Browser, BrowserContext, Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { startPrivateStack, type PrivateStack } from './private-stack';

/** A pack term of call1/store/seeds/asr_vocabulary_retail_v1.json (docs/DualAsr.md section 10). */
const PACK_TERM = 'Converse';

/** The fake `vocabulary` script's corrections (fake.py FAKE_MISHEARINGS): term -> what "Parakeet" heard. */
const SCRIPTED: [string, string][] = [
  ['Stanley cup', 'standy cup'],
  ['Chadstone', 'Chad Stone'],
  ['Afterpay', 'after pay'],
];

type Persona = 'Admin' | 'Reviewer';
type ColorScheme = 'light' | 'dark' | 'no-preference' | null | undefined;

/** Contexts opened on the private stack, closed after each test: left open, their pages outlive the
 * stack and keep retrying it in the worker's browser for the rest of the run. */
const openContexts: BrowserContext[] = [];

async function signInDemo(browser: Browser, stack: PrivateStack, persona: Persona, colorScheme: ColorScheme): Promise<Page> {
  const context = await browser.newContext({ baseURL: stack.info.store_url, colorScheme: colorScheme ?? undefined });
  openContexts.push(context);
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

test.describe('Vocabulary on a demo-mode stack with the retail pack installed (docs/DualAsr.md section 8)', () => {
  let stack: PrivateStack;
  let merged: { call_id: string; conversation_id: string };
  let baseOnly: { call_id: string; conversation_id: string };

  test.beforeAll(async ({}, testInfo) => {
    testInfo.setTimeout(240_000);
    stack = await startPrivateStack({
      name: `pw-vocabulary-${testInfo.project.name}`,
      storeEnv: { CALL1_STORE_DEMO: '1' },
      vocabularySeed: 'call1/store/seeds/asr_vocabulary_retail_v1.json',
      // First ASR attempt: the scripted mishearings, merged. Second: the same script, but the
      // vocabulary pass fails, so the call keeps Parakeet's words (base_only).
      fakeBehavior: { asr: ['script:vocabulary', 'script:vocabulary,vocabulary_fail'] },
    });
    merged = await stack.ingest('call_01_compliant', { agentId: 'pw-vocab-merged' });
    await settle(stack, merged.conversation_id);
    baseOnly = await stack.ingest('call_03_dispute_escalation', { agentId: 'pw-vocab-base-only' });
    await settle(stack, baseOnly.conversation_id);
  });
  test.afterEach(async () => {
    await Promise.all(openContexts.splice(0).map((context) => context.close()));
  });
  test.afterAll(async () => {
    await stack?.close();
  });

  test('Workbench: each merged correction is a mark whose tooltip names what Parakeet heard, on hover and keyboard focus', async ({
    browser,
    colorScheme,
  }) => {
    const reviewer = await signInDemo(browser, stack, 'Reviewer', colorScheme);
    const transcript = reviewer.waitForResponse((r) => r.url().includes(`/store/v1/calls/${merged.call_id}/transcript`) && r.ok());
    await reviewer.goto(`/#/calls/${merged.call_id}`);
    await expect(reviewer.getByRole('heading', { name: 'Transcript' })).toBeVisible();

    // What Store sent: the three real replacements, with Store-computed offsets that land on the terms.
    const view = (await (await transcript).json()) as {
      turns: { turn_id: number; text: string }[];
      vocabulary_correction: { status: string; replacements: { turn_id: number; char_start: number; char_end: number; term: string; heard: string | null }[] };
    };
    expect(view.vocabulary_correction.status).toBe('applied');
    const replacements = view.vocabulary_correction.replacements;
    expect(replacements.map((r) => r.term)).toEqual(SCRIPTED.map(([term]) => term));
    for (const r of replacements) {
      const turn = view.turns.find((t) => t.turn_id === r.turn_id);
      expect(turn?.text.slice(r.char_start, r.char_end)).toBe(r.term);
    }
    await expect(reviewer.getByText(`${SCRIPTED.length} words corrected from the vocabulary.`)).toBeVisible();

    for (const [term, heard] of SCRIPTED) {
      const r = replacements.find((x) => x.term === term)!;
      expect(r.heard).toBe(heard);
      const label = `Parakeet heard "${heard}"`;
      const mark = reviewer.getByText(term, { exact: true });
      await expect(mark).toBeVisible();

      await mark.hover();
      await expect(reviewer.getByRole('tooltip', { name: label })).toBeVisible();
      await reviewer.mouse.move(0, 0);
      await expect(reviewer.getByRole('tooltip')).toHaveCount(0);

      // Keyboard focus shows the same tooltip; Escape closes it without losing focus.
      await mark.focus();
      await expect(reviewer.getByRole('tooltip', { name: label })).toBeVisible();
      await reviewer.keyboard.press('Escape');
      await expect(reviewer.getByRole('tooltip')).toHaveCount(0);
      await expect(mark).toBeFocused();
    }
    // Parakeet's words never reach the page as transcript text.
    for (const [, heard] of SCRIPTED) await expect(reviewer.getByText(heard, { exact: false })).toHaveCount(0);
  });

  test('Workbench: a failed vocabulary pass shows the base_only notice and no correction', async ({ browser, colorScheme }) => {
    const reviewer = await signInDemo(browser, stack, 'Reviewer', colorScheme);
    await reviewer.goto(`/#/calls/${baseOnly.call_id}`);
    await expect(reviewer.getByRole('heading', { name: 'Transcript' })).toBeVisible();
    await expect(reviewer.getByText(/^Vocabulary correction didn't run for this call:/)).toBeVisible();
    await expect(reviewer.getByText(/word.*corrected from the vocabulary/)).toHaveCount(0);
    await expect(reviewer.getByRole('tooltip')).toHaveCount(0);
  });

  test('admin: a digit term is refused inline before any save reaches Store', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    await admin.goto('/#/admin/vocabulary');
    await expect(admin.getByRole('heading', { name: 'Dual transcription' })).toBeVisible();

    const input = admin.getByLabel('Add a term');
    await input.fill('Aisle 42');
    await expect(admin.getByText("Terms can't contain numbers.")).toBeVisible();
    // Disabled at the DOM level: the browser itself refuses the click, so the term is never added
    // and no PUT ever reaches Store for it.
    await expect(admin.getByRole('button', { name: 'Add', exact: true })).toBeDisabled();
  });

  test('admin: an admin adds a term, switches a pack term off, and saves', async ({ browser, colorScheme }) => {
    const admin = await signInDemo(browser, stack, 'Admin', colorScheme);
    const term = 'Loyalty rewards card';

    await admin.goto('/#/admin/vocabulary');
    await expect(admin.getByText('Retail starter vocabulary')).toBeVisible();
    const packChip = admin.getByLabel(`${PACK_TERM}: on`);
    await expect(packChip).toBeVisible();
    await packChip.uncheck();

    await admin.getByLabel('Add a term').fill(term);
    await admin.getByRole('button', { name: 'Add', exact: true }).click();
    await expect(admin.getByText(term, { exact: true })).toBeVisible();

    await admin.getByTestId('vocabulary-save-bar').getByRole('button', { name: 'Save' }).click();
    await expect(admin.getByText(/^Saved as v\d+\.$/)).toBeVisible();

    // Persisted: a fresh load shows the same state.
    await admin.reload();
    await expect(admin.getByLabel(`${PACK_TERM}: off`)).toBeVisible();
    await expect(admin.getByText(term, { exact: true })).toBeVisible();
  });

  test('admin: a stale save is rejected with a conflict, and the latest saved settings load', async ({ browser, colorScheme }) => {
    const termA = 'Conflict term A';
    const termB = 'Conflict term B';

    // Two contexts signed in as the same demo admin open the page before either saves — both hold
    // the same record_version locally.
    const first = await signInDemo(browser, stack, 'Admin', colorScheme);
    await first.goto('/#/admin/vocabulary');
    await expect(first.getByRole('heading', { name: 'Dual transcription' })).toBeVisible();
    const second = await signInDemo(browser, stack, 'Admin', colorScheme);
    await second.goto('/#/admin/vocabulary');
    await expect(second.getByRole('heading', { name: 'Dual transcription' })).toBeVisible();

    // The first page adds a term and saves, advancing the version Store holds.
    await first.getByLabel('Add a term').fill(termA);
    await first.getByRole('button', { name: 'Add', exact: true }).click();
    await first.getByTestId('vocabulary-save-bar').getByRole('button', { name: 'Save' }).click();
    await expect(first.getByText(/^Saved as v\d+\.$/)).toBeVisible();

    // The second page still holds the stale version; its save is rejected with a conflict (not
    // silently overwritten or auto-retried), and the latest saved settings (the first page's term,
    // not the second's) load.
    await second.getByLabel('Add a term').fill(termB);
    await second.getByRole('button', { name: 'Add', exact: true }).click();
    await second.getByTestId('vocabulary-save-bar').getByRole('button', { name: 'Save' }).click();
    await expect(second.getByRole('alert')).toBeVisible();
    await expect(second.getByText(termA, { exact: true })).toBeVisible({ timeout: 15_000 });
    await expect(second.getByText(termB, { exact: true })).toHaveCount(0);
  });
});

test.describe('Vocabulary: Workbench on the shared stack, which has no vocabulary (docs/DualAsr.md section 8)', () => {
  test('a masked-original correction says the original words are masked, never the raw text', async ({
    createInvitedUser,
    ingestSample,
    waitUntilSettled,
  }, testInfo) => {
    const reviewer = await createInvitedUser('reviewer');
    const receipt = await ingestSample('call_01_compliant', { agentId: `pw-vocab-masked-${testInfo.project.name}-${Date.now()}` });
    await waitUntilSettled(receipt.call_id);

    const CORRECTED = 'Nordbank';
    await reviewer.page.route(`**/store/v1/calls/${receipt.call_id}/transcript`, async (route) => {
      const response = await route.fetch().catch(() => null);
      const body = response ? await response.json().catch(() => null) : null;
      if (!response || body === null) return;
      const turn = (body.turns ?? [])[0];
      if (turn) {
        turn.text = `Your card through ${CORRECTED} is on file.`;
        const start = turn.text.indexOf(CORRECTED);
        body.vocabulary_correction = {
          status: 'applied',
          note: null,
          replacement_count: 1,
          withheld_count: 0,
          replacements: [
            {
              turn_id: turn.turn_id,
              char_start: start,
              char_end: start + CORRECTED.length,
              word_start: 0,
              word_end: 1,
              start_time: turn.start_time,
              end_time: turn.end_time,
              term: 'Nordbank',
              source: 'industry_pack',
              heard: null, // masking changed it, or it held a digit (docs/DualAsr.md section 8)
            },
          ],
        };
      }
      await route.fulfill({ response, json: body });
    });

    await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
    await expect(reviewer.page.getByRole('heading', { name: 'Transcript' })).toBeVisible();
    const mark = reviewer.page.getByText(CORRECTED, { exact: true });
    await mark.hover();
    await expect(reviewer.page.getByRole('tooltip', { name: 'Corrected from the vocabulary. The original words are masked.' })).toBeVisible();
  });

  test('a call with no vocabulary correction shows neither a mark nor a notice', async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
    const reviewer = await createInvitedUser('reviewer');
    const receipt = await ingestSample('call_01_compliant', { agentId: `pw-vocab-none-${testInfo.project.name}-${Date.now()}` });
    await waitUntilSettled(receipt.call_id);

    await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
    await expect(reviewer.page.getByRole('heading', { name: 'Transcript' })).toBeVisible();
    await expect(reviewer.page.getByText(/word.*corrected from the vocabulary/)).toHaveCount(0);
    await expect(reviewer.page.getByText(/Vocabulary correction didn't run/)).toHaveCount(0);
  });
});
