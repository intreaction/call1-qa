// evaluate_ui e3 (waveform): the Workbench's ThreadWaveform — the legacy three.js thread waveform
// ported to Evaluate (src/apps/evaluate/components/ThreadWaveform.tsx). The stage is the "Seek
// audio" slider: the canvas renders once the Store audio decodes, a click seeks the <audio>
// element, arrow keys/Home/End seek and Space plays or pauses. With prefers-reduced-motion the
// thread stays still (data-motion="reduced", "Motion reduced"); without WebGL the stage gives way
// to a text state and a native seek bar, never a blank box.
//
// Set CALL1_E2E_SCREENSHOT_DIR to also save a Workbench screenshot per color scheme
// (`workbench-waveform-<project>.png`).
import fs from 'node:fs';
import path from 'node:path';
import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { COUNT, speakerTimeline } from '../src/apps/evaluate/components/threadWaveformGl';

test('speaker coloring preserves gaps, overlaps and half-open turn boundaries', () => {
  const data = speakerTimeline([
    { start: 1, end: 4, speaker: 'AGENT' },
    { start: 3, end: 6, speaker: 'CALLER' },
    { start: 7, end: 8, speaker: 'UNKNOWN' },
    { start: 9, end: 9, speaker: 'AGENT' },
  ], COUNT, 1);
  const at = (i: number) => Array.from(data.slice(i * 4, i * 4 + 2));
  expect(at(0)).toEqual([0, 0]);
  expect(at(1)).toEqual([85, 85]);
  expect(at(3)).toEqual([255, 255]);
  expect(at(4)).toEqual([170, 170]);
  expect(at(6)).toEqual([0, 0]);
  expect(at(7)).toEqual([0, 0]);
  expect(at(9)).toEqual([0, 0]);
});

test('speaker coloring respects stereo channel evidence and clears missing or invalid timing', () => {
  const data = speakerTimeline([
    { start: -1, end: 2, speaker: 'CALLER', channel: 0 },
    { start: 0, end: 2, speaker: 'AGENT', channel: 1 },
    { start: 3, end: 6, speaker: 'AGENT' }, // No channel evidence: do not guess.
    { start: NaN, end: 6, speaker: 'CALLER', channel: 0 },
    { start: 8, end: 7, speaker: 'AGENT', channel: 1 },
  ], COUNT, 2);
  expect(Array.from(data.slice(0, 2))).toEqual([170, 85]);
  expect(Array.from(data.slice(2 * 4, 6 * 4))).toEqual(new Array(16).fill(0));
  expect(speakerTimeline([], COUNT, 1).every((v) => v === 0)).toBe(true);
  expect(speakerTimeline([{ start: 0, end: 9, speaker: 'AGENT' }], NaN, 1).every((v) => v === 0)).toBe(true);
});

const audioState = (page: Page) =>
  page.evaluate(() => {
    const a = document.querySelector('audio');
    return { duration: a?.duration ?? NaN, currentTime: a?.currentTime ?? NaN, paused: a?.paused ?? true };
  });

async function waitForDuration(page: Page): Promise<number> {
  await expect.poll(async () => (await audioState(page)).duration, { timeout: 20_000 }).toBeGreaterThan(0);
  return (await audioState(page)).duration;
}

async function openSettledCall(
  fixtures: { createInvitedUser: (role: 'reviewer') => Promise<{ page: Page }>; ingestSample: typeof import('./harness').ingestSample; waitUntilSettled: typeof import('./harness').waitUntilSettled },
  label: string,
  prepare?: (page: Page) => Promise<unknown>,
) {
  const reviewer = await fixtures.createInvitedUser('reviewer');
  const receipt = await fixtures.ingestSample('call_05_pii_heavy', { agentId: label });
  await fixtures.waitUntilSettled(receipt.call_id);
  if (prepare) await prepare(reviewer.page);
  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  // The page is already on Evaluate, so the goto is only a hash change: reload so init scripts
  // (the no-WebGL shim) run in a fresh document.
  if (prepare) await reviewer.page.reload();
  await expect(reviewer.page.getByTestId('workbench-header').getByTestId('call-agent')).toHaveText(label);
  return reviewer.page;
}

test('e3 waveform: the thread canvas renders from Store audio and a click seeks', async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
  const page = await openSettledCall(
    { createInvitedUser, ingestSample, waitUntilSettled },
    `pw-wave-${testInfo.project.name}-${Date.now()}`,
    (p) => p.setViewportSize({ width: 1440, height: 1000 }),
  );
  const deck = page.getByTestId('thread-waveform');
  await expect(deck).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
  await expect(deck).toHaveAttribute('data-motion', 'full');

  const stage = page.getByRole('slider', { name: 'Seek audio' });
  await expect(stage).toBeVisible();
  const canvas = stage.locator('canvas');
  await expect(canvas).toBeVisible();
  const drawn = await canvas.evaluate((c: HTMLCanvasElement) => ({
    width: c.width,
    height: c.height,
    webgl: !!(c.getContext('webgl2') ?? c.getContext('webgl')),
  }));
  expect(drawn.width).toBeGreaterThan(0);
  expect(drawn.height).toBeGreaterThan(0);
  expect(drawn.webgl).toBe(true);
  // A legend names the speakers from the transcript; the status is a text label, not color alone.
  await expect(page.getByTestId('thread-waveform-status')).toContainText(/Mono|Stereo/);

  // Click at 75% of the ribbon (the ribbon spans 4%..96% of the stage).
  const duration = await waitForDuration(page);
  const box = (await stage.boundingBox())!;
  await stage.click({ position: { x: box.width * (0.04 + 0.92 * 0.75), y: box.height / 2 } });
  await expect.poll(async () => (await audioState(page)).currentTime).toBeGreaterThan(duration * 0.75 - 1);
  expect((await audioState(page)).currentTime).toBeLessThan(duration * 0.75 + 1);
  // The detail drawer names the moment.
  await expect(page.getByRole('button', { name: 'Close waveform detail' })).toBeVisible();

  const dir = process.env.CALL1_E2E_SCREENSHOT_DIR;
  if (dir) {
    fs.mkdirSync(dir, { recursive: true });
    await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.45); // show the hover wake
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(dir, `workbench-waveform-${testInfo.project.name}.png`) });
  }
});

test('e3 waveform: arrow keys, Home/End seek and Space plays or pauses on the focused stage', async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
  const page = await openSettledCall({ createInvitedUser, ingestSample, waitUntilSettled }, `pw-wave-keys-${testInfo.project.name}-${Date.now()}`);
  await expect(page.getByTestId('thread-waveform')).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
  const duration = await waitForDuration(page);
  const stage = page.getByRole('slider', { name: 'Seek audio' });
  await stage.focus();
  await expect(stage).toBeFocused();

  const time = async () => (await audioState(page)).currentTime;
  await page.keyboard.press('Home');
  await expect.poll(time).toBeLessThan(0.5);
  await page.keyboard.press('ArrowRight');
  await expect.poll(time).toBeCloseTo(Math.min(5, duration), 0);
  await page.keyboard.press('Shift+ArrowRight');
  await expect.poll(time).toBeCloseTo(Math.min(20, duration), 0);
  await page.keyboard.press('ArrowLeft');
  await expect.poll(time).toBeCloseTo(Math.min(15, duration), 0);
  await page.keyboard.press('End');
  await expect.poll(time).toBeGreaterThan(duration - 0.5);
  await expect(stage).toHaveAttribute('aria-valuetext', / of /);

  await page.keyboard.press('Home');
  await page.keyboard.press(' ');
  await expect.poll(async () => (await audioState(page)).paused).toBe(false);
  await expect(page.getByRole('button', { name: 'Pause' })).toBeVisible();
  await page.keyboard.press(' ');
  await expect.poll(async () => (await audioState(page)).paused).toBe(true);
});

test('e3 waveform: prefers-reduced-motion keeps the thread still and still seeks', async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
  const page = await openSettledCall(
    { createInvitedUser, ingestSample, waitUntilSettled },
    `pw-wave-still-${testInfo.project.name}-${Date.now()}`,
    (p) => p.emulateMedia({ reducedMotion: 'reduce' }),
  );
  const deck = page.getByTestId('thread-waveform');
  await expect(deck).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });
  await expect(deck).toHaveAttribute('data-motion', 'reduced');
  await expect(deck.getByText('Motion reduced')).toBeVisible();

  const duration = await waitForDuration(page);
  const stage = page.getByRole('slider', { name: 'Seek audio' });
  const box = (await stage.boundingBox())!;
  await stage.click({ position: { x: box.width * (0.04 + 0.92 * 0.5), y: box.height / 2 } });
  await expect.poll(async () => (await audioState(page)).currentTime).toBeGreaterThan(duration * 0.5 - 1);
});

test('e3 waveform: without WebGL a text state and a native seek bar replace the stage', async ({ createInvitedUser, ingestSample, waitUntilSettled }, testInfo) => {
  const page = await openSettledCall(
    { createInvitedUser, ingestSample, waitUntilSettled },
    `pw-wave-nogl-${testInfo.project.name}-${Date.now()}`,
    (p) =>
      p.addInitScript(() => {
        const original = HTMLCanvasElement.prototype.getContext;
        // Test shim: refuse every WebGL context, keep 2D.
        (HTMLCanvasElement.prototype as unknown as { getContext: (type: string, ...rest: unknown[]) => unknown }).getContext = function (
          this: HTMLCanvasElement,
          type: string,
          ...rest: unknown[]
        ) {
          if (/webgl/i.test(type)) return null;
          return (original as (...args: unknown[]) => unknown).call(this, type, ...rest);
        };
      }),
  );
  const deck = page.getByTestId('thread-waveform');
  await expect(deck).toHaveAttribute('data-state', 'no-webgl');
  await expect(page.getByTestId('thread-waveform-fallback')).toContainText('needs WebGL');
  await expect(deck.locator('canvas')).toBeHidden();

  const duration = await waitForDuration(page);
  const seekBar = page.getByRole('slider', { name: 'Seek audio' });
  await expect(seekBar).toBeVisible();
  expect(await seekBar.evaluate((el) => el.tagName)).toBe('INPUT');
  const target = Math.min(10, Math.floor(duration / 2));
  await seekBar.fill(String(target));
  await expect.poll(async () => (await audioState(page)).currentTime).toBeCloseTo(target, 0);
});

// --- Contact Signals v2 addition (docs/ContactSignalsV2.md section 10.2 "Waveform" and section 17
// "evaluate-waveform": span bands, cluster popover with subcategory and fields). WaveformMarker
// gains an optional `end`; a marker with `end` draws a range band under its pin. Written
// independently from the F4 implementation, from the design doc plus the contract
// (ContactSignalView / SignalSpanView, call1/contracts/contents.py). Route-stubs the signals
// content endpoint so the band's time range is deterministic, since fake stage-1/2/3 engines are
// not guaranteed to land a hit at a chosen time on this branch.
test('e3 waveform (signals): a contact-signal hit with an end time draws a range band, and its marker opens a popover with subcategory and fields', async ({
  createInvitedUser,
  ingestSample,
  waitUntilSettled,
}, testInfo) => {
  const reviewer = await createInvitedUser('reviewer');
  const receipt = await ingestSample('call_01_compliant', { agentId: `pw-wave-signals-${testInfo.project.name}-${Date.now()}` });
  await waitUntilSettled(receipt.call_id);

  const transcript = await (await reviewer.page.request.get(`/store/v1/calls/${receipt.call_id}/transcript`)).json();
  const turn = transcript.turns[0] as { turn_id: number; start_time: number; end_time: number };
  const spanEnd = Math.min(turn.end_time, turn.start_time + 4);

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
        stage1_digests: {}, feedback: [], alerts: [],
        signals: [{
          id: `intent.abcdef012345.12345678.t${turn.turn_id}b0`, kind: 'intent', label: 'Caller objective', start: turn.start_time, end: spanEnd,
          speaker: 'CALLER', quote: 'I have a question about a fee on my account.', turn_id: turn.turn_id,
          char_start: 0, char_end: 45, confidence: 0.91, review_status: 'unreviewed',
          category_id: 'intent', category_digest: 'abcdef012345', category_confidence: 0.91,
          subcategory_id: 'billing_question', subcategory_label: 'Billing question', subcategory_digest: 'fedcba543210', subcategory_confidence: 0.83,
          span: { block: 0, first_window: 0, last_window: 0, timing: 'words', context_start: turn.start_time, context_end: spanEnd },
          fields: [{ field_id: 'reason', name: 'Reason', type: 'string', value: 'a fee', surface: 'a fee', evidence: null, evidence_grounded: null, source: 'stage3' }],
          quote_narrowed: false,
        }],
      },
    });
  });

  await reviewer.page.goto(`/#/calls/${receipt.call_id}`);
  const deck = reviewer.page.getByTestId('thread-waveform');
  await expect(deck).toHaveAttribute('data-state', 'ready', { timeout: 30_000 });

  // A range band under the marker's pin (assumed test id — see the run report's selector list).
  const band = deck.getByTestId(`waveform-span-band-intent.abcdef012345.12345678.t${turn.turn_id}b0`).or(deck.locator('[data-signal-band]').first());
  await expect(band).toBeVisible();

  // Clicking the marker pin (the band itself is not interactive) opens the cluster popover with
  // category › subcategory, field chips and the quote (section 10.2 "Waveform"). F5 reconciliation:
  // the popover hangs off the pin in the marker lane; the stage's own detail drawer is a different
  // surface (seek position and speaker only).
  const lane = reviewer.page.getByRole('group', { name: 'Waveform markers' });
  // The pin may share a cluster with verdict evidence near the same time ("2 markers").
  await lane.getByRole('button', { name: /^(Caller objective|\d+ markers) at \d+:\d\d$/ }).first().click();
  const popover = lane.getByRole('button', { name: 'Close markers' }).locator('xpath=../..');
  await expect(popover).toContainText('Caller objective');
  await expect(popover).toContainText('Billing question');
  await expect(popover).toContainText('reason', { ignoreCase: true });
  await expect(popover).toContainText('I have a question about a fee on my account.');
});
