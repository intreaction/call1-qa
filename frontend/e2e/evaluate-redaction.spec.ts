// evaluate_ui e15 (known issue e): summaries must not restate sensitive values (SSN, 16-digit card
// number) in prose beyond the masking policy. `call1/store/results/masking.py` masks transcript
// text, verdict evidence/reasoning, summary text and contact-signal quotes by the same PII regexes
// (SSN, card, phone, account number, PIN — call1/redaction.py PII_PATTERNS) whenever
// `mask_reviewer_reads` is true (the contract default while admin state is deferred).
//
// This is NOT exercisable against the shared fake-handler e2e stack: the fake ASR always returns
// the fixed SCRIPT (call1/process/handlers/fake.py), which contains no SSN- or card-shaped digit
// runs, and the fake enrichment handler never emits ACCOUNT_NUMBER/PHONE_NUMBER numeric entities —
// so there is nothing in a fake-handler call for the masking pass to ever redact, and a "not
// verbatim" assertion there would be vacuously true rather than a real check. Exercising the actual
// policy needs real ASR output on `sample_audio/call_05_pii_heavy.wav` (its name suggests it is the
// fixture meant for this), which needs a real-handler stack (`CALL1_E2E_HANDLERS=real
// CALL1_REAL_MODELS=1` — see frontend/e2e/README.md and call1-qa/CLAUDE.md "Local-only work").
// This spec is written to run against such a stack when one is available and is skipped otherwise,
// matching the existing `real_stack`/`CALL1_REAL_MODELS` opt-in convention (tests/e2e/README.md).
import { test, expect } from './fixtures';

const SSN_PATTERN = /\b\d{3}-\d{2}-\d{4}\b/;
const CARD_PATTERN = /\b(?:\d[ -]?){13,19}\d\b/;

test('e15: a call transcribed from PII-heavy audio does not restate a raw SSN or card number in its summary', async ({
  stack,
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  test.skip(stack.handlers !== 'real', 'needs a real-handler stack: CALL1_E2E_HANDLERS=real CALL1_REAL_MODELS=1 (not exercisable with fake handlers — see file header)');
  // Real models take minutes per call (tests/e2e/test_process_real.py runs ~95 s alone), longer
  // than the suite's 90 s default; waitUntilSettled below already allows 15 minutes.
  test.setTimeout(16 * 60_000);

  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_05_pii_heavy', { agentId: `pw-e15-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id, 900_000);

  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  await expect(reviewer.page.getByRole('heading', { name: 'Summary' })).toBeVisible();
  const summaryText = (await reviewer.page.locator('main').innerText()).replace(/\s+/g, ' ');

  expect(summaryText).not.toMatch(SSN_PATTERN);
  expect(summaryText).not.toMatch(CARD_PATTERN);
  // Masking replaces sensitive spans with a marker rather than deleting context; if the call really
  // contains PII, we expect to see that the redaction ran, not merely an absence of raw digits.
  expect(summaryText).toMatch(/█|REDACTED|\*{3,}/);
});

// --- Contact Signals v2 addition (docs/ContactSignalsV2.md section 11.2/10.2 and section 17
// "evaluate-redaction": "quotes and field string values masked, and withheld while findings are
// pending"). Written independently from the F4 implementation.
//
// "Withheld while findings pending" is deterministic on fakes (it is a read-time gate, not a
// masking-quality claim): stub getContactSignals to return `text_withheld: true` with `[REDACTED]`
// quote/field text, matching section 10.2's "Text withheld (1.2.0)" row ("Categories, times and
// chips only. Quotes and field text arrive as [REDACTED]").
test('e15 (signals): while PII findings are pending, quotes and field text read [REDACTED] and the section says so', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-e15-signals-withheld-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);

  await reviewer.page.route(`**/store/v1/calls/${receipt.call_id}`, async (route) => {
    // The page may close while a stubbed request is in flight (test teardown): drop it then.
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
        segmentation: { segmenter_version: 'seg-v1', window_seconds: 7, segments: 10, scored_segments: 10, skipped_unattributed: 0, skipped_system: 0, interpolated_turns: 0 },
        stage1_digests: {}, feedback: [], alerts: [], text_withheld: true,
        signals: [{
          id: 'intent.abcdef012345.12345678.t0b0', kind: 'intent', label: 'Caller objective', start: 0, end: 5, speaker: 'CALLER',
          quote: '[REDACTED]', turn_id: 0, char_start: 0, char_end: 11, confidence: 0.9, review_status: 'unreviewed',
          category_id: 'intent', category_digest: 'abcdef012345', category_confidence: 0.9,
          subcategory_id: 'billing_question', subcategory_label: 'Billing question', subcategory_digest: 'fedcba543210', subcategory_confidence: 0.8,
          span: { block: 0, first_window: 0, last_window: 0, timing: 'words', context_start: 0, context_end: 5 },
          fields: [{ field_id: 'reason', name: 'Reason', type: 'string', value: '[REDACTED]', surface: '[REDACTED]', evidence: null, evidence_grounded: null, source: 'stage3' }],
          quote_narrowed: false,
        }],
      },
    });
  });

  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  const section = reviewer.page.locator('section', { has: reviewer.page.getByRole('heading', { name: 'Contact signals' }) });
  await expect(section).toContainText('Text withheld until PII masking finishes');
  await expect(section).toContainText('[REDACTED]');
  // Category/subcategory chips and times still show (section 10.2: "Categories, times and chips
  // only").
  await expect(section.getByText('Caller objective', { exact: true })).toBeVisible();
  await expect(section.getByText('Billing question', { exact: true })).toBeVisible();
});
