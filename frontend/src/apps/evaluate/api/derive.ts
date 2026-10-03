// Display rules for the derived result states Store computes (`calls.derive_result_state`).
// Evaluate only renders them; it never re-derives a state from job data. Every status gets a text
// label as well as a tone (docs/SplitBuild.md design principles). Unknown values from a newer
// contract minor render as "Unknown", never as an error.

import type { CallListItem, EscalationStatus, ResultState, ReviewQueueStatus, VerdictStatus } from './types';

export type Tone = 'green' | 'yellow' | 'red' | 'blue' | 'magenta' | 'neutral';

export interface StateDisplay {
  label: string;
  tone: Tone;
  /** One sentence for a tooltip or screen reader. */
  description: string;
}

const RESULT_STATE_DISPLAY: Record<ResultState, StateDisplay> = {
  pending: { label: 'Analyzing', tone: 'blue', description: 'Processing has not published this result yet.' },
  available: { label: 'Ready', tone: 'green', description: 'The result is published and current.' },
  partial: { label: 'Partial', tone: 'yellow', description: 'Published, but some parts could not be produced.' },
  stale: { label: 'Stale', tone: 'magenta', description: 'A reanalysis is under way; the shown result is the previous one.' },
  failed: { label: 'Needs attention', tone: 'red', description: 'Processing stopped without a result. An admin can retry it.' },
  disabled: { label: 'Not run', tone: 'neutral', description: 'Nothing was requested for this result.' },
};

export function resultStateDisplay(state: ResultState | string | null | undefined): StateDisplay {
  if (state && state in RESULT_STATE_DISPLAY) return RESULT_STATE_DISPLAY[state as ResultState];
  return { label: 'Unknown', tone: 'neutral', description: `Store reported a state this build does not know (${state ?? 'none'}).` };
}

/** "87" for 87.4; scores are 0–100 (`Scorecard.overall_score`). */
export function formatScore(score: number): string {
  return `${Math.round(score)}`;
}

export interface CallBadge extends StateDisplay {
  /** The score to show beside the label, when the QA result carries one. */
  score: number | null;
}

/**
 * The one badge the calls list shows for a call's QA: Analyzing / Needs attention / Partial /
 * Stale / score. A stale or partial result still shows its (old or partial) score next to the
 * label; an available one shows the score with Pass / Fail / Critical fail.
 */
export function callQaBadge(call: Pick<CallListItem, 'qa_state' | 'overall_score' | 'passed' | 'critical_failure'>): CallBadge {
  const base = resultStateDisplay(call.qa_state);
  const score = call.overall_score ?? null;
  if (call.qa_state === 'available') {
    if (score === null) return { ...resultStateDisplay('pending'), score: null };
    if (call.critical_failure) return { label: 'Critical fail', tone: 'red', description: 'A critical criterion failed.', score };
    if (call.passed === false) return { label: 'Fail', tone: 'red', description: 'The scorecard did not pass.', score };
    if (call.passed === true) return { label: 'Pass', tone: 'green', description: 'The scorecard passed.', score };
    return { label: 'Scored', tone: 'green', description: base.description, score };
  }
  if (call.qa_state === 'stale' || call.qa_state === 'partial') return { ...base, score };
  return { ...base, score: null };
}

const VERDICT_STATUS_DISPLAY: Record<VerdictStatus, StateDisplay> = {
  PASS: { label: 'Pass', tone: 'green', description: 'This criterion passed.' },
  FAIL: { label: 'Fail', tone: 'red', description: 'This criterion did not pass.' },
  FLAGGED: { label: 'Flagged', tone: 'yellow', description: 'The model could not decide confidently; a human should review it.' },
  NOT_APPLICABLE: { label: 'Not applicable', tone: 'neutral', description: 'This criterion does not apply to this call.' },
};

export function verdictStatusDisplay(status: VerdictStatus | string | null | undefined): StateDisplay {
  if (status && status in VERDICT_STATUS_DISPLAY) return VERDICT_STATUS_DISPLAY[status as VerdictStatus];
  return { label: 'Unknown', tone: 'neutral', description: `Store reported a status this build does not know (${status ?? 'none'}).` };
}

const ESCALATION_STATUS_DISPLAY: Record<EscalationStatus, StateDisplay> = {
  NONE: { label: 'No escalation', tone: 'neutral', description: 'Nothing on this call needed supervisor attention.' },
  PENDING: { label: 'Pending', tone: 'yellow', description: 'Waiting for a supervisor to resolve it.' },
  APPROVED: { label: 'Approved', tone: 'green', description: 'A supervisor approved the automated result.' },
  OVERRIDDEN: { label: 'Overridden', tone: 'magenta', description: 'A supervisor overrode the automated result.' },
};

export function escalationStatusDisplay(status: EscalationStatus | string | null | undefined): StateDisplay {
  if (status && status in ESCALATION_STATUS_DISPLAY) return ESCALATION_STATUS_DISPLAY[status as EscalationStatus];
  return { label: 'Unknown', tone: 'neutral', description: `Store reported a status this build does not know (${status ?? 'none'}).` };
}

const REVIEW_QUEUE_STATUS_DISPLAY: Record<ReviewQueueStatus, StateDisplay> = {
  PENDING: { label: 'Pending', tone: 'blue', description: 'Waiting to be claimed.' },
  IN_REVIEW: { label: 'In review', tone: 'yellow', description: 'A reviewer is working on it.' },
  APPROVED: { label: 'Approved', tone: 'green', description: 'Resolved: the automated result was approved.' },
  OVERRIDDEN: { label: 'Overridden', tone: 'magenta', description: 'Resolved: the automated result was overridden.' },
  SUPERSEDED: { label: 'Superseded', tone: 'neutral', description: 'Replaced by a newer queue item for this call.' },
};

export function reviewQueueStatusDisplay(status: ReviewQueueStatus | string | null | undefined): StateDisplay {
  if (status && status in REVIEW_QUEUE_STATUS_DISPLAY) return REVIEW_QUEUE_STATUS_DISPLAY[status as ReviewQueueStatus];
  return { label: 'Unknown', tone: 'neutral', description: `Store reported a status this build does not know (${status ?? 'none'}).` };
}

/**
 * The normative agent label (contract 1.1.0, `call1.contracts.calls.agent_label`): "Name (ext)"
 * with both, "Name" with only a display name, and `agent_id` (plus " (ext)" when there is an
 * extension) without one. Empty strings count as absent, as in the Python rule. `agent_id` stays
 * the key for filters and rules; this is display only and never parses it.
 */
export function agentLabel(agentId: string, displayName?: string | null, extension?: string | null): string;
export function agentLabel(agentId: string | null | undefined, displayName?: string | null, extension?: string | null): string | null;
export function agentLabel(agentId: string | null | undefined, displayName?: string | null, extension?: string | null): string | null {
  const base = displayName || agentId;
  if (!base) return null;
  return extension ? `${base} (${extension})` : base;
}
