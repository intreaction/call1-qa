/**
 * Helpers specific to the `evaluate-*.spec.ts` files (feature-inventory coverage for the
 * evaluate_ui area). Not part of the shared harness (`harness.ts`/`fixtures.ts`) — kept here per
 * e2e/README.md's "add a small helper in your own files" note.
 *
 * The fixed fake-handler script (`call1/process/handlers/fake.py` SCRIPT) always PASSes every
 * criterion of the seeded default rubric (`call1_standard_v2`), and the shared e2e stack has no
 * way to override `CALL1_FAKE_BEHAVIOR` per test. To get a deterministic critical-failure /
 * escalated call without touching the shared default rubric (other tests depend on its baseline
 * PASS behaviour), we publish a throwaway rubric with one critical `phrase_any` criterion whose
 * phrase never occurs in the fixed script (so it always FAILs — `call1/process/handlers/code.py`
 * `score()`: a critical FAIL sets `critical_failure=True` and therefore `requires_human_review`),
 * then reanalyze one call against it (`POST /calls/{id}/reanalysis-requests`, `kind: "qa"`,
 * `rubric: {...}`) — a real, documented feature (`call1/process/reanalysis.py`), not a hack. That
 * reanalysis is scoped to the one call; it does not affect any other test's calls.
 */
import fs from 'node:fs';
import path from 'node:path';
import type { Locator, Page } from '@playwright/test';
import * as h from './harness';

export interface RubricRef {
  rubric_id: string;
  version: number;
  digest: string;
}

function blankCheck(overrides: Record<string, unknown>) {
  return {
    aggregation: 'mean',
    check_type: 'phrase_any',
    comparison: 'gte',
    escalation_model_id: null,
    escalation_when: ['needs_review'],
    fail_when: null,
    metric: 'text_polarity',
    metric_threshold: 0,
    min_coverage: 0.8,
    min_samples: 2,
    not_applicable_when: null,
    pass_when: null,
    pattern: null,
    phrases: [] as string[],
    policy_context: null,
    primary_model_id: null,
    requires_policy: false,
    response_phrases: [] as string[],
    speaker: 'AGENT',
    threshold: 80,
    trigger_phrases: [] as string[],
    window_seconds: null,
    ...overrides,
  };
}

/** Publish (via the API, as `api`) a one-criterion rubric under `rubricId` that always FAILs. */
export async function publishGuaranteedFailRubric(api: h.StoreApi, rubricId: string): Promise<RubricRef> {
  const definition = {
    rubric_id: rubricId,
    name: rubricId,
    category: 'GENERAL',
    pass_threshold: 80,
    description: 'e2e: deterministic critical failure, for escalation/queue coverage.',
    criteria: [
      {
        criterion_id: 'FORCE-FAIL',
        name: 'Forced failure',
        category: 'COMPLIANCE',
        description: 'A phrase that never occurs in the fixed fake-handler transcript.',
        weight: 100,
        critical: true,
        check: blankCheck({ check_type: 'phrase_any', phrases: ['xyzzy-never-said-in-the-fake-script'] }),
      },
    ],
  };
  const draft = await api.json<{ draft_revision: number }>('PUT', `/rubrics/${rubricId}/draft`, {
    data: { definition, expected_draft_revision: null },
  });
  const published = await api.json<{ ref: RubricRef }>('POST', `/rubrics/${rubricId}/publish`, {
    data: { expected_current_version: 0, expected_draft_revision: draft.draft_revision },
  });
  return published.ref;
}

/** Publish a rubric with one criterion guaranteed to PASS (phrase present in every fake script). */
export async function publishGuaranteedPassRubric(api: h.StoreApi, rubricId: string): Promise<RubricRef> {
  const definition = {
    rubric_id: rubricId,
    name: rubricId,
    category: 'GENERAL',
    pass_threshold: 50,
    description: 'e2e: deterministic pass, for draft-test coverage.',
    criteria: [
      {
        criterion_id: 'ALWAYS-PASS',
        name: 'Always passes',
        category: 'GENERAL',
        description: 'A phrase every fake-handler transcript opens with.',
        weight: 100,
        critical: false,
        check: blankCheck({ check_type: 'phrase_any', phrases: ['Thank you for calling'] }),
      },
    ],
  };
  const draft = await api.json<{ draft_revision: number }>('PUT', `/rubrics/${rubricId}/draft`, {
    data: { definition, expected_draft_revision: null },
  });
  const published = await api.json<{ ref: RubricRef }>('POST', `/rubrics/${rubricId}/publish`, {
    data: { expected_current_version: 0, expected_draft_revision: draft.draft_revision },
  });
  return published.ref;
}

/** Reanalyze `callId`'s QA against `rubric` and wait for the new evaluation to settle. */
export async function reanalyzeAgainst(api: h.StoreApi, callId: string, rubric: RubricRef): Promise<void> {
  await api.json('POST', `/calls/${callId}/reanalysis-requests`, {
    data: { kind: 'qa', rubric: { rubric_id: rubric.rubric_id, version: rubric.version, digest: rubric.digest }, note: null },
    idempotencyKey: true,
  });
  await h.waitUntilSettled(callId);
}

/** A short, collision-resistant id fragment for rubric ids / queue rule ids owned by one test. */
export function slug(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Upload a recording through Process's API (`POST /process/api/recordings`) with arbitrary call
 * metadata form fields, including the contract 1.1.0 agent identity (`agent_display_name`,
 * `agent_extension`). Returns the receipt and the exact bytes sent, so a test can re-upload the
 * same recording (same SHA-256, same conversation) with different metadata.
 */
export async function ingestWithMetadata(
  fields: Record<string, string>,
  options: { name?: string; bytes?: Buffer; expectStatus?: number } = {},
): Promise<{ receipt: h.IngestReceipt & Record<string, unknown>; bytes: Buffer }> {
  const stack = h.stackInfo();
  const file = h.samplePath(options.name ?? 'call_01_compliant');
  const bytes = options.bytes ?? h.uniqueWav(fs.readFileSync(file));
  const form = new FormData();
  form.append('file', new Blob([new Uint8Array(bytes)], { type: 'audio/wav' }), path.basename(file));
  for (const [key, value] of Object.entries(fields)) form.append(key, value);
  const response = await fetch(`${stack.process_url}/process/api/recordings`, {
    method: 'POST',
    headers: { 'X-Call1-Console-Token': stack.console_token },
    body: form,
  });
  const receipt = (await response.json()) as h.IngestReceipt & Record<string, unknown>;
  const ok = options.expectStatus !== undefined ? response.status === options.expectStatus : response.status < 300;
  if (!ok) throw new Error(`ingest answered ${response.status}: ${JSON.stringify(receipt)}`);
  return { receipt, bytes };
}

/** Press Tab (at most `max` times) until `target` has focus; fails the test if it never does. */
export async function tabTo(page: Page, target: Locator, { max = 80, shift = false } = {}): Promise<void> {
  for (let i = 0; i < max; i += 1) {
    if (await target.evaluate((el) => el === document.activeElement).catch(() => false)) return;
    await page.keyboard.press(shift ? 'Shift+Tab' : 'Tab');
  }
  const active = await page.evaluate(() => {
    const el = document.activeElement as HTMLElement | null;
    return el ? `${el.tagName.toLowerCase()} "${(el.getAttribute('aria-label') ?? el.textContent ?? '').trim().slice(0, 60)}"` : 'nothing';
  });
  throw new Error(`Tab never reached ${target}; focus is on ${active}`);
}

/** The focused element shows a visible focus indicator (an outline or a focus ring box-shadow). */
export async function hasVisibleFocus(target: Locator): Promise<boolean> {
  return target.evaluate((el) => {
    const style = getComputedStyle(el);
    const outline = style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0;
    const ring = style.boxShadow !== 'none' && style.boxShadow !== '';
    return el === document.activeElement && (outline || ring);
  });
}
