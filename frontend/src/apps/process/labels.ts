// Plain-language names for the identifiers the Process API returns: job types (the JobType enum
// in call1/contracts/jobs.py), job statuses, waiting reasons, adapter tasks and route classes.
// Anything not listed falls back to `humanize`, so a new enum value still renders.

import { humanize } from './components/ui';

/** Every JobType, in pipeline order: a job's upstreams come before it. Also the tie-break when
 * two jobs have no dependency between them. */
export const JOB_TYPE_ORDER: string[] = [
  'validation_vad',
  'asr',
  'speaker_attribution',
  'enrichment',
  'acoustic_tone',
  'text_sentiment',
  'embeddings',
  'qa_deterministic',
  'qa_criterion',
  'qa_escalation',
  'qa_scorecard',
  'summary_segment',
  'summary_synthesis',
  'summary_assembly',
  'contact_signals_categorize',
  'contact_signals_subcategorize',
  'contact_signals_extract',
  'contact_signals_lifecycle',
  'contact_signals_resolution',
  'contact_signals_merge',
];

const JOB_TYPE_LABEL: Record<string, string> = {
  validation_vad: 'Audio validation',
  asr: 'Transcription',
  speaker_attribution: 'Speaker attribution',
  enrichment: 'Transcript enrichment',
  acoustic_tone: 'Acoustic tone',
  text_sentiment: 'Text sentiment',
  embeddings: 'Search embeddings',
  qa_deterministic: 'QA rule checks',
  qa_criterion: 'QA criterion',
  qa_escalation: 'QA escalation (second opinion)',
  qa_scorecard: 'QA scorecard',
  summary_segment: 'Summary: segment',
  summary_synthesis: 'Summary: synthesis',
  summary_assembly: 'Summary: assembly',
  contact_signals_categorize: 'Contact signals: category',
  contact_signals_subcategorize: 'Contact signals: subcategory',
  contact_signals_extract: 'Contact signals: field extraction',
  contact_signals_lifecycle: 'Contact signals: lifecycle',
  contact_signals_resolution: 'Contact signals: resolution',
  contact_signals_merge: 'Contact signals: merge',
};

export function jobTypeLabel(jobType: string): string {
  return JOB_TYPE_LABEL[jobType] ?? humanize(jobType);
}

const JOB_STATUS_LABEL: Record<string, string> = {
  BLOCKED: 'Waiting',
  QUEUED: 'Queued',
  RUNNING: 'Running',
  SUCCEEDED: 'Succeeded',
  FAILED: 'Failed',
  CANCELLED: 'Cancelled',
  WAITING_PROVIDER: 'Waiting for provider',
};

/** A BLOCKED job whose upstream failed (`dead_blocked`) reads "Stopped": it will not start on
 * its own, unlike one that is only waiting for earlier steps to finish. */
export function jobStatusLabel(status: string, waitingReason?: string | null): string {
  if (status === 'BLOCKED' && waitingReason === 'dead_blocked') return 'Stopped';
  return JOB_STATUS_LABEL[status] ?? humanize(status);
}

/** WaitingReason values (call1/contracts/jobs.py). `dead_blocked` means an upstream job failed or
 * was cancelled, so this one cannot start until that one is retried. */
const WAITING_REASON_LABEL: Record<string, string> = {
  waiting_for_dependencies: 'Waiting on earlier steps',
  dead_blocked: 'An earlier step failed; retry that step to continue',
  retry_backoff: 'Retrying after a short delay',
  deferred_by_worker: 'Deferred by the worker; will run shortly',
  waiting_for_worker: 'Waiting for a worker that can run it',
  pro1_pending_verification: 'Waiting for Pro1 verification',
};

export function waitingReasonLabel(reason: string): string {
  return WAITING_REASON_LABEL[reason] ?? humanize(reason);
}

/** Adapter tasks (call1/process/training/registry.py `task_for`). */
const TASK_LABEL: Record<string, string> = {
  signal_stage1: 'Contact signals: category',
  signal_stage2: 'Contact signals: subcategory',
  qa_verdict: 'QA verdicts',
  speaker_roles: 'Speaker roles',
};

export function taskLabel(task: string): string {
  return TASK_LABEL[task] ?? humanize(task);
}

/** Where a job's model runs, from its frozen route. `appliance` is always this machine, whatever
 * runtime the model uses (MLX or PyTorch), so the provider protocol is not shown for it. */
export function routeLabel(routeClass: string | null | undefined, destinationHost?: string | null): string | null {
  if (!routeClass) return null;
  if (routeClass === 'appliance') return 'On this appliance';
  const names: Record<string, string> = {
    customer_lan: 'Customer LAN server',
    call1_confidential: 'Pro1 Confidential',
    customer_directed: 'Customer-directed provider (BYOK)',
  };
  const base = names[routeClass] ?? humanize(routeClass);
  return destinationHost && destinationHost !== 'in-process' ? `${base} → ${destinationHost}` : base;
}

/** Runtime names as operators know them. */
export function runtimeLabel(runtime: string): string {
  return { mlx: 'MLX (Apple GPU)', torch: 'PyTorch', code: 'Rules (no model)', ollama: 'Ollama' }[runtime] ?? runtime;
}

/** Handler modes (call1/process/handlers): real models, or fakes for tests and dry runs. */
export function handlerModeLabel(mode: string): string {
  if (mode === 'real') return 'Real models';
  if (mode === 'fake') return 'Fake handlers (tests and dry runs; no models load)';
  return humanize(mode);
}
