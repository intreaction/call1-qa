// TanStack Query keys, shared by the shell and the views so change-feed invalidation reaches
// every cached read. Keys are arrays whose first element is the resource family.

import type { ChangeEvent } from './types';

export const queryKeys = {
  contract: ['contract'] as const,
  session: ['session'] as const,
  calls: (filters?: Record<string, unknown>) => (filters ? (['calls', filters] as const) : (['calls'] as const)),
  call: (callId: string) => ['call', callId] as const,
  reviewQueue: ['review-queue'] as const,
  escalations: ['escalations'] as const,
  rubrics: ['rubrics'] as const,
  rubric: (rubricId: string) => ['rubrics', rubricId] as const,
  metrics: ['metrics'] as const,
  reanalysis: ['reanalysis'] as const,
  ownAuthenticators: ['own-authenticators'] as const,
  ownSessions: ['own-sessions'] as const,
  // Contact signals v2 (contract 1.3.0). The taxonomy, its versions, alert rules and previews share
  // the `signals` family; signal metrics sit under `metrics` so every metrics invalidation reaches
  // them. A call's signals stay at `[...call(id), 'contact-signals']`.
  signals: ['signals'] as const,
  signalTaxonomy: ['signals', 'taxonomy'] as const,
  signalTaxonomyVersions: ['signals', 'versions'] as const,
  signalAlertRules: ['signals', 'alert-rules'] as const,
  signalPreviews: ['signals', 'preview'] as const,
  signalPreview: (previewId: string) => ['signals', 'preview', previewId] as const,
  signalMetrics: (range?: Record<string, unknown>) =>
    range ? (['metrics', 'signals', range] as const) : (['metrics', 'signals'] as const),
  catalogSnapshots: ['catalog-snapshots'] as const,
  // ASR vocabulary (contract 1.3.0, decision 33, docs/DualAsr.md): the singleton settings record.
  asrVocabulary: ['asr-vocabulary'] as const,
  admin: {
    accounts: ['admin', 'accounts'] as const,
    accountAuthenticators: (accountId: string) => ['admin', 'accounts', accountId, 'authenticators'] as const,
    invitations: ['admin', 'invitations'] as const,
    installations: ['admin', 'installations'] as const,
    serviceKeys: ['admin', 'service-keys'] as const,
  },
};

/** The query-key prefixes a change event makes stale. Events carry IDs, never content. */
export function keysForChange(event: ChangeEvent): ReadonlyArray<readonly unknown[]> {
  const keys: (readonly unknown[])[] = [];
  const call = event.call_id ? queryKeys.call(event.call_id) : null;
  switch (event.kind) {
    case 'call':
    case 'result':
    case 'job_group':
    case 'job':
      keys.push(queryKeys.calls(), queryKeys.metrics);
      if (call) keys.push(call);
      break;
    case 'review':
      keys.push(queryKeys.calls(), queryKeys.escalations, queryKeys.metrics);
      if (call) keys.push(call);
      break;
    case 'review_queue':
      keys.push(queryKeys.reviewQueue, queryKeys.escalations);
      if (call) keys.push(call);
      break;
    case 'reanalysis_request':
      // A signal preview's calls are reanalysis requests: the open preview re-reads.
      keys.push(queryKeys.reanalysis, queryKeys.calls(), queryKeys.signalPreviews);
      if (call) keys.push(call);
      break;
    case 'rubric':
      keys.push(queryKeys.rubrics);
      break;
    case 'signal_taxonomy':
      // saved:v<N> or settings. Every call's contact signals carry a read-time taxonomy status (the
      // "Scored with taxonomy vN (current vM)" label), so the whole `call` family re-reads, and the
      // alert rules' `node_active` and the list fields (active nodes only) may change too.
      keys.push(queryKeys.signalTaxonomy, queryKeys.signalTaxonomyVersions, queryKeys.signalAlertRules, queryKeys.calls(), queryKeys.metrics, ['call']);
      break;
    case 'signal_alert_rule':
      // Alerts are evaluated at read time: rule edits change lists, metrics and every call's `alerts`.
      keys.push(queryKeys.signalAlertRules, queryKeys.calls(), queryKeys.metrics, ['call']);
      break;
    case 'signal_alert':
      keys.push(queryKeys.calls(), queryKeys.metrics);
      if (call) keys.push(call);
      break;
    default:
      // admin_state, pro1_connection: nothing the shell caches today.
      break;
    case 'catalog':
      keys.push(queryKeys.catalogSnapshots);
      break;
    case 'asr_vocabulary':
      // Settings only; existing transcripts aren't re-run (docs/DualAsr.md section 4).
      keys.push(queryKeys.asrVocabulary);
      break;
  }
  return keys;
}
