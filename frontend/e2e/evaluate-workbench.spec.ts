// evaluate_ui e3: WorkbenchView (#/calls/:callId) — playback, transcript, scorecard, verdict
// override, escalation resolution, summary, contact signals; sections gate on ResultGroup.state;
// override/escalation/retain-review controls are hidden (not disabled) until an evaluation exists
// or permission is present; a hidden <audio> drives playback and the ThreadWaveform stage (or its
// native-range fallback) is the "Seek audio" slider — evaluate-waveform.spec.ts covers it.
import { test, expect } from './fixtures';
import { publishGuaranteedFailRubric, reanalyzeAgainst, slug } from './evaluate-helpers';

test('e3: Workbench renders playback, transcript, summary, contact signals and scorecard for a settled call', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const agent = `pw-e3-${testInfo.project.name}-${Date.now()}`;
  const receipt = await ingestSample('call_01_compliant', { agentId: agent });
  await waitUntilSettled(receipt.call_id);

  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  await expect(reviewer.page.getByTestId('workbench-header').getByTestId('call-agent')).toHaveText(agent);

  // Playback: a hidden <audio> element; the "Seek audio" slider is the ThreadWaveform stage (or a
  // native range input when WebGL is unavailable).
  await expect(reviewer.page.locator('audio')).toBeAttached();
  const seekBar = reviewer.page.getByRole('slider', { name: 'Seek audio' });
  await expect(seekBar).toBeVisible();
  // The transport Play button lives in the same card as the seek bar (as opposed to each
  // transcript turn's own "Play" jump-to-turn link, which is also named exactly "Play").
  const transportCard = reviewer.page.locator('section', { has: seekBar });
  await expect(transportCard.getByRole('button', { name: 'Play' })).toBeVisible();

  // Sections gate on ResultGroup.state and never 404 while analyzing/disabled; each renders content
  // once its group settles to `available` on the fake pipeline.
  await expect(reviewer.page.getByRole('heading', { name: 'Transcript' })).toBeVisible();
  await expect(reviewer.page.locator('#workbench-turn-0')).toContainText('Thank you for calling');
  await expect(reviewer.page.getByRole('heading', { name: 'Summary' })).toBeVisible();
  await expect(reviewer.page.getByRole('heading', { name: 'Contact signals' })).toBeVisible();
  await expect(reviewer.page.getByRole('heading', { name: 'Scorecard' })).toBeVisible();
  await reviewer.page.getByRole('button', { name: 'Review scorecard', exact: true }).click();
  await expect(reviewer.page.getByRole('dialog', { name: 'Review scorecard' }).getByText('out of 100')).toBeVisible();
  const explanation = reviewer.page.getByTestId('score-explanation');
  await expect(explanation).toContainText('How this score is calculated');
  await expect(explanation).toContainText('assessed weight × 100');
  await expect(explanation).toContainText('Checks needing review and not-applicable checks are excluded');
  await expect(explanation).toContainText('To pass: at least');
  // Every seeded rubric criterion appears with a quoted-evidence or "no supporting quote" line.
  await expect(reviewer.page.getByRole('dialog').getByText('Call Recording Disclosure', { exact: true })).toBeVisible();
});

test('e3: verdict override reads expected_version and is offered to a reviewer (override_verdict)', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-e3-override-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);

  await reviewer.page.getByRole('button', { name: 'Review Call Recording Disclosure', exact: true }).click();

  // REG-01 "Call Recording Disclosure" is the seeded rubric's first criterion, so it is the first
  // verdict card (and its "Override" button) in document order.
  await expect(reviewer.page.getByRole('dialog').getByText('Call Recording Disclosure', { exact: true })).toBeVisible();
  await reviewer.page.getByRole('button', { name: 'Override' }).first().click();
  await reviewer.page.getByRole('button', { name: 'Fail' }).click();

  await expect(reviewer.page.getByText('Override saved.')).toBeVisible();
  await expect(reviewer.page.getByText('(overridden)')).toBeVisible();
  await reviewer.page.keyboard.press('Escape');
  await reviewer.page.getByRole('button', { name: 'Review scorecard', exact: true }).click();
  await expect(reviewer.page.getByTestId('score-explanation')).toContainText('This score includes current reviewer decisions');
});

test('e3: escalation approve/override controls are hidden for a reviewer and shown for a supervisor (resolve_escalation)', async ({
  createInvitedUser,
  adminApi,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const supervisor = await createInvitedUser('supervisor');
  const id = testInfo.project.name;

  // Deterministically escalate one call (see evaluate-helpers.ts): the fixed fake-handler script
  // always PASSes the default rubric, so force a critical failure via a throwaway rubric + reanalysis.
  const rubric = await publishGuaranteedFailRubric(adminApi, slug(`pw-e3-escalate-${id}`));
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-e3-escalate-${id}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  await reanalyzeAgainst(adminApi, receipt.call_id, rubric);

  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);

  await reviewer.page.getByRole('button', { name: 'Review scorecard', exact: true }).click();
  await expect(reviewer.page.getByRole('dialog').getByText('Critical fail', { exact: true })).toBeVisible();
  await expect(reviewer.page.getByTestId('score-explanation')).toContainText('A critical failure prevents a pass, regardless of the numeric score');
  await expect(reviewer.page.getByTestId('score-explanation')).toContainText('Failed critical check:');
  // The escalation banner (yellow, distinct from the per-criterion override controls every
  // reviewer sees): its own Approve/Override are hidden, not merely disabled, without
  // resolve_escalation (a reviewer permission set does not include it).
  // By test ID: other yellow chips on the page (signal alert chips from rules other specs made on
  // the shared stack) share the banner's colour classes.
  const reviewerBanner = reviewer.page.getByTestId('escalation-banner');
  await expect(reviewerBanner).toContainText('waiting for a supervisor');
  await expect(reviewerBanner.getByRole('button', { name: 'Approve' })).toHaveCount(0);
  await expect(reviewerBanner.getByRole('button', { name: 'Override' })).toHaveCount(0);

  await supervisor.page.goto(`/#/calls/${receipt.call_id}`);

  await supervisor.page.getByRole('button', { name: 'Review scorecard', exact: true }).click();
  const supervisorBanner = supervisor.page.getByTestId('escalation-banner');
  const approve = supervisorBanner.getByRole('button', { name: 'Approve' });
  await expect(approve).toBeVisible();
  await approve.click();
  await expect(supervisor.page.getByText(/Escalation approved\./)).toBeVisible();
});

// e3 coverage gap: the native <audio> seek from the transcript. A transcript line's Play moves
// playback to that turn's start time, the clock and the seek slider (the waveform stage) follow it,
// and the slider itself moves playback.
test('e3: Play on a transcript line seeks the audio to that turn, and the seek bar moves playback', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-e3-seek-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  const page = reviewer.page;

  const transcript = await (await page.request.get(`/store/v1/calls/${receipt.call_id}/transcript`)).json();
  const turns = transcript.turns as { turn_id: number; start_time: number; end_time: number }[];
  const turn = turns.find((t) => t.start_time >= 2) ?? turns[turns.length - 1];
  expect(turn, 'a transcript turn that starts after 0 s').toBeTruthy();

  await page.goto(`/#/calls/${receipt.call_id}`);
  const audio = page.locator('audio');
  await expect(audio).toBeAttached();
  await expect.poll(() => audio.evaluate((a: HTMLAudioElement) => a.readyState), { message: 'audio metadata loads' }).toBeGreaterThanOrEqual(1);

  const line = page.locator(`#workbench-turn-${turn.turn_id}`);
  await line.getByRole('button', { name: /^Play from / }).click();
  await expect
    .poll(() => audio.evaluate((a: HTMLAudioElement) => a.currentTime), { message: `playback moved to turn ${turn.turn_id}` })
    .toBeGreaterThanOrEqual(turn.start_time - 0.05);
  const afterJump = await audio.evaluate((a: HTMLAudioElement) => a.currentTime);
  expect(afterJump).toBeLessThan(turn.end_time + 5);
  // The seek slider is the ThreadWaveform stage (aria-valuenow in whole seconds).
  const seekBar = page.getByRole('slider', { name: 'Seek audio' });
  await expect.poll(async () => Number(await seekBar.getAttribute('aria-valuenow'))).toBeGreaterThanOrEqual(Math.floor(turn.start_time));

  // Pause, then drive playback from the seek slider alone (Home, then one 5 s step).
  await audio.evaluate((a: HTMLAudioElement) => a.pause());
  await seekBar.focus();
  await page.keyboard.press('Home');
  await page.keyboard.press('ArrowRight');
  await expect.poll(() => audio.evaluate((a: HTMLAudioElement) => a.currentTime)).toBeCloseTo(5, 1);
  await expect(page.getByText(/^0:05 \//)).toBeVisible();

  // Full review opens without replacing the dashboard or resetting playback. Escape restores focus.
  const reviewButton = page.getByRole('button', { name: 'Review scorecard', exact: true });
  await reviewButton.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('dialog', { name: 'Review scorecard' })).toBeVisible();
  await expect.poll(() => audio.evaluate((a: HTMLAudioElement) => a.currentTime)).toBeCloseTo(5, 1);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
  await expect(reviewButton).toBeFocused();
  for (const name of ['Summary', 'Transcript', 'Contact signals', 'Scorecard']) {
    await expect(page.getByRole('heading', { name, exact: true })).toBeVisible();
  }
  // A criterion opens a contextual popout; navigation changes the inspected check.
  // Escape and outside click dismiss it; quoted evidence focuses the transcript.
  await page.getByRole('button', { name: 'Review Call Recording Disclosure', exact: true }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toHaveAttribute('aria-modal', 'false');
  await expect(dialog.getByRole('heading', { name: 'Call Recording Disclosure' })).toBeVisible();
  await dialog.getByRole('button', { name: 'Next criterion' }).click();
  await expect(dialog).not.toContainText('Call Recording Disclosure');
  await dialog.getByRole('button', { name: 'Previous criterion' }).click();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Review Call Recording Disclosure', exact: true })).toBeFocused();
  await page.getByRole('button', { name: 'Review Call Recording Disclosure', exact: true }).click();
  await page.getByRole('heading', { name: 'Summary', exact: true }).click();
  await expect(dialog).toHaveCount(0);
  await page.getByRole('button', { name: 'Review Call Recording Disclosure', exact: true }).click();
  const evidence = dialog.locator('button').filter({ hasText: 'Thank you for calling' }).first();
  await evidence.click();
  await expect(dialog).toHaveCount(0);
  await expect(page.locator('#workbench-turn-0')).toBeFocused();
  // On a narrow viewport the inspection card stays inside the screen and remains dismissible.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Review Call Recording Disclosure', exact: true }).click();
  const bounds = await dialog.boundingBox();
  expect(bounds).not.toBeNull();
  expect(bounds!.x).toBeGreaterThanOrEqual(11);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(379);
  expect(bounds!.y).toBeGreaterThanOrEqual(11);
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(833);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Review Call Recording Disclosure', exact: true })).toBeFocused();
});

// --- Contact Signals v2 additions (docs/ContactSignalsV2.md sections 10.2 and 17's Playwright
// list, "evaluate-workbench"). Written from the design doc and the already-committed contract
// (call1/contracts/calls.py ContactSignalsView/ResultGroup, call1/contracts/contents.py
// ContactSignalView), independently of the concurrent F4 implementation agent. Route-stubs the
// call detail (`/store/v1/calls/{id}`, whose `results` list carries each group's ResultState) and
// the signals content (`/store/v1/calls/{id}/contact-signals`, getContactSignals) to reach states a
// fake-handler run cannot otherwise force deterministically (failed/stale/outdated/etc).
import type { Route } from '@playwright/test';

/** The Contact signals section: the closest ancestor container of its heading. */
function signalsSection(page: import('@playwright/test').Page) {
  return page.locator('section', { has: page.getByRole('heading', { name: 'Contact signals' }) });
}

interface StubGroup {
  kind: string;
  state: string;
  version?: number | null;
  partial_reason?: string | null;
  failure_code?: string | null;
}

/** Patch one result group (by kind) in the call-detail response, and optionally stub the signals
 * content endpoint's body for the same call. */
async function stubContactSignals(
  page: import('@playwright/test').Page,
  callId: string,
  group: Partial<StubGroup>,
  content?: Record<string, unknown> | null,
) {
  await page.route(`**/store/v1/calls/${callId}`, async (route: Route) => {
    // The page may close while a stubbed request is in flight (test teardown): drop it then.
    const response = await route.fetch().catch(() => null);
    const body = response ? await response.json().catch(() => null) : null;
    if (!response || body === null) return;
    body.results = (body.results ?? []).map((g: StubGroup) => (g.kind === 'contact_signals' ? { ...g, ...group } : g));
    if (!body.results.some((g: StubGroup) => g.kind === 'contact_signals')) body.results.push({ kind: 'contact_signals', state: 'disabled', ...group });
    await route.fulfill({ response, json: body });
  });
  if (content !== undefined) {
    await page.route(`**/store/v1/calls/${callId}/contact-signals`, async (route: Route) => {
      if (content === null) return route.fulfill({ status: 404, json: { code: 'not_found', message: 'no contact signals', retryable: false, request_id: 'e2e' } });
      await route.fulfill({ json: content });
    });
  }
}

const STATE_CASES: { name: string; group: Partial<StubGroup>; content: Record<string, unknown> | null; expect: RegExp | string }[] = [
  { name: 'disabled', group: { state: 'disabled' }, content: null, expect: 'Contact signals were not run for this call' },
  { name: 'pending', group: { state: 'pending' }, content: null, expect: 'Analyzing' },
  {
    name: 'available, v2, no hits',
    group: { state: 'available', version: 1 },
    content: {
      call_id: '__CALL__', artifact_id: 'art_e2e', version: 1, completeness: 'complete', signals: [], passes: [],
      transcript_fingerprint: 'a'.repeat(64), generated_at: new Date().toISOString(), pipeline: 'v2',
      taxonomy: { version: 1, digest: 'a'.repeat(64) },
      stages: [{ stage: 'categorize', included: true }],
      segmentation: { segmenter_version: 'seg-v1', window_seconds: 7, segments: 42, scored_segments: 40, skipped_unattributed: 2, skipped_system: 0, interpolated_turns: 0 },
      stage1_digests: {}, feedback: [], alerts: [],
    },
    expect: /No signals found.*Scored 40 transcript segments/is,
  },
  {
    name: 'partial: subcategories unavailable',
    group: { state: 'partial', version: 1, partial_reason: 'subcategorize_failed' },
    content: {
      call_id: '__CALL__', artifact_id: 'art_e2e', version: 1, completeness: 'partial', partial_reason: 'subcategorize_failed', signals: [], passes: [],
      transcript_fingerprint: 'a'.repeat(64), generated_at: new Date().toISOString(), pipeline: 'v2',
      taxonomy: { version: 1, digest: 'a'.repeat(64) },
      stages: [{ stage: 'categorize', included: true }, { stage: 'subcategorize', included: false, failure_code: 'engine_error' }],
      segmentation: { segmenter_version: 'seg-v1', window_seconds: 7, segments: 10, scored_segments: 10, skipped_unattributed: 0, skipped_system: 0, interpolated_turns: 0 },
      stage1_digests: {}, feedback: [], alerts: [],
    },
    expect: /Subcategories unavailable.*engine error/is,
  },
  { name: 'stale', group: { state: 'stale', version: 1, reanalysis_request_id: 'rq_e2e' } as Partial<StubGroup>, content: null, expect: 'Refreshing' },
  { name: 'failed', group: { state: 'failed', failure_code: 'engine_error' }, content: null, expect: /Needs attention.*engine error/is },
];

for (const c of STATE_CASES) {
  test(`e3 signals (state): ${c.name} shows its section-10.2 state text`, async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
    const reviewer = await createInvitedUser('reviewer');
    const receipt = await ingestSample('call_01_compliant', { agentId: `pw-cs-state-${c.name.replace(/\W+/g, '-')}-${testInfo.project.name}-${Date.now()}` });
    await waitUntilSettled(receipt.call_id);
    const content = c.content === null ? c.content : { ...c.content, call_id: receipt.call_id };
    await stubContactSignals(reviewer.page, receipt.call_id, c.group, content);
    await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
    await expect(signalsSection(reviewer.page)).toContainText(c.expect);
  });
}

test('e3 signals: PII-masking failure names the specific reason (section 10.2)', async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
  // F5 reconciliation: there is no `pii_findings_missing` job error code. The §10.2 case is "every
  // v2 job dead-blocked on pii_findings": the group reads `failed` with the upstream's code, and the
  // transcript stays withheld (contract 1.2.0 `text_withheld`) because the findings never arrived.
  // The Workbench keys the specific text on exactly that pair (signalsWaitOnMasking).
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-cs-pii-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  await stubContactSignals(reviewer.page, receipt.call_id, { state: 'failed', failure_code: 'provider_error' }, null);
  await reviewer.page.route(`**/store/v1/calls/${receipt.call_id}/transcript`, async (route: Route) => {
    // The page may close while a stubbed request is in flight (test teardown): drop it then.
    const response = await route.fetch().catch(() => null);
    const body = response ? await response.json().catch(() => null) : null;
    if (!response || body === null) return;
    await route.fulfill({ response, json: { ...body, text_withheld: true } });
  });
  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  await expect(signalsSection(reviewer.page)).toContainText('signals wait for PII masking to be retried');
});

test('e3 signals: an outdated taxonomy label offers "Update signals" (requestReanalysis, kind contact_signals)', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-cs-outdated-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  await stubContactSignals(
    reviewer.page,
    receipt.call_id,
    { state: 'available', version: 1 },
    {
      call_id: receipt.call_id, artifact_id: 'art_e2e', version: 1, completeness: 'complete', signals: [], passes: [],
      transcript_fingerprint: 'a'.repeat(64), generated_at: new Date().toISOString(), pipeline: 'v2',
      taxonomy: { version: 3, digest: 'a'.repeat(64) },
      stages: [{ stage: 'categorize', included: true }, { stage: 'subcategorize', included: true }],
      segmentation: { segmenter_version: 'seg-v1', window_seconds: 7, segments: 10, scored_segments: 10, skipped_unattributed: 0, skipped_system: 0, interpolated_turns: 0 },
      stage1_digests: {}, feedback: [], alerts: [],
      taxonomy_status: { scored_version: 3, current_version: 5, outdated_stages: ['subcategorize'], thresholds_changed: false },
    },
  );
  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  await expect(signalsSection(reviewer.page)).toContainText(/Scored with taxonomy v3.*current v5/is);
  const updateButton = signalsSection(reviewer.page).getByRole('button', { name: 'Update signals' });
  await expect(updateButton).toBeVisible();
  const reanalysisRequest = reviewer.page.waitForRequest((r) => r.method() === 'POST' && r.url().includes(`/calls/${receipt.call_id}/reanalysis-requests`));
  await updateButton.click();
  const sent = await reanalysisRequest;
  expect((sent.postDataJSON() as { kind: string }).kind).toBe('contact_signals');
  expect(sent.headers()['idempotency-key'], 'Update signals sends an idempotency key').toBeTruthy();
  await expect(signalsSection(reviewer.page).getByRole('button', { name: 'Update requested' })).toBeVisible();
});

test('e3 signals: category/subcategory and field chips, quote, times and jump-to-turn on a v2 hit', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-cs-chips-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);

  const transcript = await (await reviewer.page.request.get(`/store/v1/calls/${receipt.call_id}/transcript`)).json();
  const turn = transcript.turns[0] as { turn_id: number; start_time: number; end_time: number };

  await stubContactSignals(
    reviewer.page,
    receipt.call_id,
    { state: 'available', version: 1 },
    {
      call_id: receipt.call_id, artifact_id: 'art_e2e', version: 1, completeness: 'complete', passes: [],
      transcript_fingerprint: 'a'.repeat(64), generated_at: new Date().toISOString(), pipeline: 'v2',
      taxonomy: { version: 1, digest: 'a'.repeat(64) },
      stages: [{ stage: 'categorize', included: true }, { stage: 'subcategorize', included: true }, { stage: 'extract', included: true }],
      segmentation: { segmenter_version: 'seg-v1', window_seconds: 7, segments: 10, scored_segments: 10, skipped_unattributed: 0, skipped_system: 0, interpolated_turns: 0 },
      stage1_digests: {}, feedback: [], alerts: [],
      signals: [
        {
          id: `intent.abcdef012345.12345678.t${turn.turn_id}b0`, kind: 'intent', label: 'Caller objective', start: turn.start_time, end: turn.end_time,
          speaker: 'CALLER', quote: 'I have a question about a fee on my account.', turn_id: turn.turn_id,
          char_start: 0, char_end: 45, confidence: 0.91, review_status: 'unreviewed',
          category_id: 'intent', category_digest: 'abcdef012345', category_confidence: 0.91,
          subcategory_id: 'billing_question', subcategory_label: 'Billing question', subcategory_digest: 'fedcba543210', subcategory_confidence: 0.83,
          span: { block: 0, first_window: 0, last_window: 0, timing: 'words', context_start: Math.max(0, turn.start_time - 7), context_end: turn.end_time + 7 },
          fields: [{ field_id: 'reason', name: 'Reason', type: 'string', value: 'a fee', surface: 'a fee', evidence: null, evidence_grounded: null, source: 'stage3' }],
          quote_narrowed: false,
        },
      ],
    },
  );
  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);

  const section = signalsSection(reviewer.page);
  await expect(section.getByText('Caller objective', { exact: true })).toBeVisible();
  await expect(section.getByText('Billing question', { exact: true })).toBeVisible();
  await section.getByRole('button', { name: /^Details for/ }).click();
  await expect(section.getByText('reason: a fee', { exact: false })).toBeVisible();
  await expect(section.getByText('I have a question about a fee on my account.', { exact: false })).toBeVisible();

  await section.getByText('Caller objective', { exact: true }).click();
  await expect(reviewer.page.locator(`#workbench-turn-${turn.turn_id}`)).toBeFocused();
  const seekBar = reviewer.page.getByRole('slider', { name: 'Seek audio' });
  await expect.poll(async () => Number(await seekBar.getAttribute('aria-valuenow'))).toBeGreaterThanOrEqual(Math.floor(turn.start_time));
});

test('e3 signals: confirm/dismiss a category, correct a subcategory, with a version conflict re-read', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-cs-feedback-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  const hitId = 'intent.abcdef012345.12345678.t0b0';

  // F5 reconciliation: the section re-reads the signals after every feedback write, so the stub
  // serves the stored feedback (the verdict text comes from that re-read, not from the PUT answer).
  // The hit carries a subcategory the current taxonomy does not list, so "Other" is a correction.
  let feedback: Record<string, unknown>[] = [];
  await stubContactSignals(reviewer.page, receipt.call_id, { state: 'available', version: 1 });
  await reviewer.page.route(`**/store/v1/calls/${receipt.call_id}/contact-signals`, async (route: Route) => {
    await route.fulfill({
      json: {
        call_id: receipt.call_id, artifact_id: 'art_e2e', version: 1, completeness: 'complete', passes: [],
        transcript_fingerprint: 'a'.repeat(64), generated_at: new Date().toISOString(), pipeline: 'v2',
        taxonomy: { version: 1, digest: 'a'.repeat(64) },
        stages: [{ stage: 'categorize', included: true }, { stage: 'subcategorize', included: true }],
        segmentation: { segmenter_version: 'seg-v1', window_seconds: 7, segments: 10, scored_segments: 10, skipped_unattributed: 0, skipped_system: 0, interpolated_turns: 0 },
        stage1_digests: {}, feedback, alerts: [],
        signals: [{
          id: hitId, kind: 'intent', label: 'Caller objective', start: 0, end: 5, speaker: 'CALLER', quote: 'Hi, I have a question.',
          turn_id: 0, char_start: 0, char_end: 20, confidence: 0.9, review_status: 'unreviewed',
          category_id: 'intent', category_digest: 'abcdef012345', category_confidence: 0.9,
          subcategory_id: 'billing_question', subcategory_label: 'Billing question', subcategory_digest: 'fedcba543210', subcategory_confidence: 0.8,
          span: { block: 0, first_window: 0, last_window: 0, timing: 'words', context_start: 0, context_end: 5 },
          fields: [], quote_narrowed: false,
        }],
      },
    });
  });
  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  const section = signalsSection(reviewer.page);
  await expect(section.getByText('Caller objective', { exact: true })).toBeVisible();
  await expect(section.getByText('Billing question', { exact: true })).toBeVisible();

  await section.getByRole('button', { name: /^Details for/ }).click();

  // Confirm the category verdict (expected_feedback_version 0 on the first write).
  const feedbackUrl = `**/store/v1/calls/${receipt.call_id}/signal-hits/${hitId}/feedback`;
  await reviewer.page.route(feedbackUrl, async (route: Route) => {
    const body = route.request().postDataJSON() as { expected_feedback_version: number; category_verdict: string | null };
    if (body.expected_feedback_version !== 0) {
      return route.fulfill({ status: 409, json: { code: 'conflict', message: 'stale feedback version', details: { current_version: 1 }, retryable: false, request_id: 'e2e' } });
    }
    const stored = {
      call_id: receipt.call_id, hit_id: hitId, category_verdict: body.category_verdict, subcategory_id: null, subcategory_digest: null, subcategory_verdict: null,
      corrected_subcategory_id: null, note: null, account_id: reviewer.accountId, feedback_version: 1, updated_at: new Date().toISOString(),
    };
    feedback = [stored];
    await route.fulfill({ json: stored });
  });
  await section.getByRole('button', { name: 'Confirm', exact: true }).click();
  await expect(section.getByText('Confirmed', { exact: true })).toBeVisible();

  // A second writer got there first: the correction answers 409, and the section says so and
  // re-reads instead of failing silently.
  await reviewer.page.unroute(feedbackUrl);
  await reviewer.page.route(feedbackUrl, async (route: Route) => {
    await route.fulfill({ status: 409, json: { code: 'conflict', message: 'stale feedback version', details: { current_version: 2 }, retryable: false, request_id: 'e2e' } });
  });
  const reread = reviewer.page.waitForRequest((r) => r.method() === 'GET' && r.url().endsWith(`/calls/${receipt.call_id}/contact-signals`));
  await section.getByRole('button', { name: 'Correct', exact: true }).click();
  await section.getByLabel('Correct subcategory to').selectOption({ label: 'Other' });
  await section.getByRole('button', { name: 'Save correction' }).click();
  await expect(section.getByText(/changed this since you opened it/i)).toBeVisible();
  await reread;
});

test('scorecard carousel shows distinct applied rubrics and keeps earlier assessments read-only', async ({
  createInvitedUser, adminApi, ingestSample, waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-rubric-carousel-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  const rubric = await publishGuaranteedFailRubric(adminApi, slug(`pw-carousel-${testInfo.project.name}`));
  // Wait for the request itself: the original graph can be settled before Process claims it.
  for (let application = 0; application < 2; application++) {
    const request = await adminApi.json<{ id: string }>('POST', `/calls/${receipt.call_id}/reanalysis-requests`, {
      data: { kind: 'qa', rubric: { rubric_id: rubric.rubric_id, version: rubric.version, digest: rubric.digest }, note: null },
      idempotencyKey: true,
    });
    await expect.poll(async () => (await adminApi.json<{ status: string }>('GET', `/reanalysis-requests/${request.id}`)).status, { timeout: 30_000 }).toBe('fulfilled');
  }
  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  const carousel = reviewer.page.getByRole('region', { name: 'Applied rubrics' });
  await expect(carousel).toContainText('1 / 2');
  await expect(carousel).toContainText('Current assessment');
  await expect(carousel.getByRole('group', { name: 'Criterion results' }).getByRole('button')).toHaveCount(1);
  await reviewer.page.getByRole('button', { name: 'Next rubric' }).click();
  await expect(carousel).toContainText('2 / 2');
  await expect(carousel).toContainText('Earlier application');
  await reviewer.page.getByRole('button', { name: 'Review scorecard', exact: true }).click();
  const dialog = reviewer.page.getByRole('dialog', { name: 'Review scorecard' });
  await expect(dialog).toContainText('Earlier rubric application');
  await expect(dialog.getByRole('button', { name: 'Override', exact: true })).toHaveCount(0);
  await expect(dialog.getByRole('button', { name: 'Approve', exact: true })).toHaveCount(0);
  await reviewer.page.keyboard.press('Escape');
  await reviewer.page.getByRole('button', { name: 'Review Call Recording Disclosure', exact: true }).click();
  const popout = reviewer.page.getByRole('dialog');
  await expect(popout).toHaveAttribute('aria-modal', 'false');
  await expect(popout).toContainText('read-only assessment');
  await expect(popout.getByRole('button', { name: 'Override', exact: true })).toHaveCount(0);
  await reviewer.page.keyboard.press('Escape');
  await reviewer.page.getByRole('button', { name: 'Previous rubric' }).click();
  await reviewer.page.getByRole('button', { name: 'Review scorecard', exact: true }).click();
  await expect(dialog.getByRole('button', { name: 'Override', exact: true })).toBeVisible();
});


test('e3: corner reanalysis menu supports keyboard selection, notes and safe retries', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-menu-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);
  const attempts: { body: unknown; key: string | undefined }[] = [];
  await reviewer.page.route(`**/store/v1/calls/${receipt.call_id}/reanalysis-requests`, async (route) => {
    const request = route.request();
    attempts.push({ body: request.postDataJSON(), key: request.headers()['idempotency-key'] });
    // Simulate an uncertain server response: retry must send the same body and key.
    await route.fulfill({ status: attempts.length === 1 ? 503 : 201, json: attempts.length === 1 ? { message: 'Temporarily unavailable' } : { id: 'req_menu' } });
  });
  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  const trigger = reviewer.page.getByTestId('workbench-header').getByRole('button', { name: 'Reanalysis', exact: true });
  await expect(trigger).toBeVisible();
  await expect(reviewer.page.getByRole('heading', { name: 'Request reanalysis' })).toHaveCount(0);
  await trigger.focus();
  await trigger.press('ArrowDown');
  const menu = reviewer.page.getByRole('menu');
  await expect(menu.getByRole('menuitem')).toHaveCount(5);
  await expect(menu.getByRole('menuitem', { name: 'Full (everything)', exact: true })).toBeFocused();
  await reviewer.page.keyboard.press('ArrowDown');
  await expect(menu.getByRole('menuitem', { name: 'QA only', exact: true })).toBeFocused();
  await reviewer.page.keyboard.press('Enter');
  const dialog = reviewer.page.getByRole('dialog', { name: 'Request reanalysis' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByLabel('Kind', { exact: true })).toHaveValue('qa');
  await dialog.getByLabel('Note', { exact: true }).fill('Review the updated rubric.');
  await dialog.getByRole('button', { name: 'Request', exact: true }).click();
  await expect(dialog.getByRole('button', { name: 'Request', exact: true })).toBeEnabled();
  expect(attempts).toHaveLength(1);
  await dialog.getByRole('button', { name: 'Request', exact: true }).click();
  await expect(dialog).toContainText('Reanalysis requested.');
  expect(attempts).toHaveLength(2);
  expect(attempts[0].body).toEqual({ kind: 'qa', note: 'Review the updated rubric.', rescore_signals: false });
  expect(attempts[0].key).toBeTruthy();
  expect(attempts[1]).toEqual(attempts[0]);
  await reviewer.page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await reviewer.page.setViewportSize({ width: 390, height: 844 });
  const triggerBounds = await trigger.boundingBox();
  expect(triggerBounds!.x + triggerBounds!.width).toBeLessThanOrEqual(390);
  await trigger.click();
  await expect(menu).toBeVisible();
  const bounds = await menu.boundingBox();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390);
  await reviewer.page.keyboard.press('Escape');
  await expect(menu).toHaveCount(0);
  await expect(trigger).toBeFocused();
});
