// `#/queue` — the human review queue (`/store/v1/review-queue/*`), never the processing queue.
// Claim-next, assign, start, release and resolve, plus (supervisor) queue rules.
//
// Contact signals (contract 1.3.0, docs/ContactSignalsV2.md §9.3, §10.3): the SIGNAL stream, an
// "Alerts" condition on any rule (`target_signal_alerts`), trigger-alert chips on items, and
// "Signals changed since this item was created" when an item's alert no longer matches.

import { useState } from 'react';
import { useInfiniteQuery, useQuery, useQueryClient } from '@tanstack/react-query';
import { Bell, CheckCircle2, ClipboardCheck, Settings2, Zap } from 'lucide-react';
import {
  agentLabel,
  alertRuleName,
  isVersionConflict,
  queryKeys,
  reviewQueueStatusDisplay,
  type DistributionStrategy,
  type ReviewQueueItem,
  type ReviewQueueRule,
  type ReviewQueueRuleRecord,
  type ReviewQueueStatus,
  type ReviewStream,
  type SignalAlertRuleRecord,
} from '../api';
import {
  Button,
  Card,
  Checkbox,
  Chip,
  EmptyState,
  ErrorNotice,
  Field,
  Loading,
  PageHeader,
  SelectInput,
  StatusPill,
  TextInput,
  formatDateTime,
} from '../components/ui';
import { href } from '../state/router';
import type { QueueViewProps } from './types';

const STREAMS: ReviewStream[] = ['TRIAGE', 'AUDIT_SAMPLE', 'MANDATE', 'CALIBRATION', 'SIGNAL'];
const DISTRIBUTIONS: DistributionStrategy[] = ['ROUND_ROBIN', 'LEAST_OUTSTANDING', 'SKILL_MATCHED', 'UNASSIGNED_CLAIM'];

/** Plain-language names for the contract enums (display only; the values sent to Store are unchanged). */
const STREAM_LABELS: Record<ReviewStream, string> = {
  TRIAGE: 'Triage',
  AUDIT_SAMPLE: 'Audit sample',
  MANDATE: 'Mandate',
  CALIBRATION: 'Calibration',
  SIGNAL: 'Signal alert',
};
const DISTRIBUTION_LABELS: Record<DistributionStrategy, string> = {
  ROUND_ROBIN: 'Round robin',
  LEAST_OUTSTANDING: 'Fewest open items',
  SKILL_MATCHED: 'Skill matched',
  UNASSIGNED_CLAIM: 'Reviewers claim',
};
const streamLabel = (s: string) => STREAM_LABELS[s as ReviewStream] ?? s;
const distributionLabel = (s: string) => DISTRIBUTION_LABELS[s as DistributionStrategy] ?? s;

// Exactly the fields of `ReviewQueueRule` (the body of `ReviewQueueRuleSave.rule`). The `satisfies`
// check fails the build if the contract adds or drops a rule field, so an extra record-only field
// (`rule_version`, `updated_at`, `updated_by_account_id`) can never reach Store, which forbids
// extra fields and answers 422.
const RULE_FIELDS = {
  critical_failure_only: true,
  description: true,
  distribution_strategy: true,
  enabled: true,
  id: true,
  low_confidence_only: true,
  name: true,
  rank: true,
  sampling_rate: true,
  stream: true,
  target_agents: true,
  target_domains: true,
  target_signal_alerts: true,
  target_skills: true,
} as const satisfies Record<keyof ReviewQueueRule, true>;

/** The saveable rule inside a stored rule record: only `ReviewQueueRule` fields. */
export function ruleForSave(record: ReviewQueueRuleRecord): ReviewQueueRule {
  const rule: Record<string, unknown> = {};
  for (const key of Object.keys(RULE_FIELDS) as (keyof ReviewQueueRule)[]) {
    if (record[key] !== undefined) rule[key] = record[key];
  }
  return rule as ReviewQueueRule;
}

export default function QueueView({ client, session, navigate }: QueueViewProps) {
  const qc = useQueryClient();
  const [status, setStatus] = useState<ReviewQueueStatus | ''>('');
  const [unassignedOnly, setUnassignedOnly] = useState(false);
  const [showRules, setShowRules] = useState(false);
  const [claimError, setClaimError] = useState<unknown>(null);
  const [claiming, setClaiming] = useState(false);

  const statsQuery = useQuery({
    queryKey: [...queryKeys.reviewQueue, 'stats'],
    queryFn: ({ signal }) => client.get('/store/v1/review-queue/stats', { signal }),
  });

  const listQuery = useInfiniteQuery({
    queryKey: [...queryKeys.reviewQueue, { status: status || null, unassignedOnly }],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      client.get('/store/v1/review-queue', {
        query: { limit: 50, page_token: pageParam, status: status || null, unassigned_only: unassignedOnly },
        signal,
      }),
    getNextPageParam: (last) => last.next_page_token,
  });
  const items = listQuery.data?.pages.flatMap((p) => p.items) ?? [];

  // Alert-rule names for trigger chips and the rule editor (every role reads them: read_calls).
  const alertRulesQuery = useQuery({
    queryKey: queryKeys.signalAlertRules,
    queryFn: ({ signal }) => client.get('/store/v1/signals/alert-rules', { signal }),
    retry: false,
    // The rule editor's Alerts list must include a rule made moments ago elsewhere (another tab,
    // another admin), even inside the default 15 s staleTime and before the change feed's next poll.
    refetchOnMount: 'always',
  });
  const alertRules = alertRulesQuery.data?.items;

  const invalidate = () => {
    void qc.invalidateQueries({ queryKey: queryKeys.reviewQueue });
  };

  async function claimNext() {
    setClaiming(true);
    setClaimError(null);
    try {
      const res = await client.post('/store/v1/review-queue/claim-next');
      if (!res.item) {
        setClaimError(new Error('Nothing is claimable right now.'));
      } else {
        invalidate();
        navigate({ name: 'workbench', callId: res.item.call_id });
      }
    } catch (err) {
      setClaimError(err);
    } finally {
      setClaiming(false);
    }
  }

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      <PageHeader
        title="Review queue"
        description="Calls waiting for a human decision."
        right={
          <>
            <Button icon={Zap} variant="primary" busy={claiming} disabled={!session.can('claim_review')} onClick={() => void claimNext()}>
              Claim next
            </Button>
            {session.can('manage_queue_rules') && (
              <Button icon={Settings2} onClick={() => setShowRules((s) => !s)}>
                {showRules ? 'Hide rules' : 'Rules'}
              </Button>
            )}
          </>
        }
      />

      <ErrorNotice error={claimError} />

      {statsQuery.data && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          <Stat label="Pending" value={statsQuery.data.pending} />
          <Stat label="In review" value={statsQuery.data.in_review} />
          <Stat label="Unassigned" value={statsQuery.data.unassigned} />
          <Stat label="Stale" value={statsQuery.data.stale} />
          <Stat label="Total" value={statsQuery.data.total} />
        </div>
      )}

      {showRules && session.can('manage_queue_rules') && <QueueRules client={client} alertRules={alertRules} />}

      <Card
        title="Items"
        icon={ClipboardCheck}
        right={
          <>
            <SelectInput aria-label="Filter by status" value={status} onChange={(e) => setStatus(e.target.value as ReviewQueueStatus | '')} className="w-40">
              <option value="">All statuses</option>
              <option value="PENDING">Pending</option>
              <option value="IN_REVIEW">In review</option>
              <option value="APPROVED">Approved</option>
              <option value="OVERRIDDEN">Overridden</option>
              <option value="SUPERSEDED">Superseded</option>
            </SelectInput>
            <label className="flex items-center gap-2 text-sm text-fg whitespace-nowrap">
              <input type="checkbox" checked={unassignedOnly} onChange={(e) => setUnassignedOnly(e.target.checked)} />
              Unassigned only
            </label>
          </>
        }
      >
        {listQuery.isLoading && <Loading label="Loading queue…" />}
        <ErrorNotice error={listQuery.error} />
        {listQuery.isSuccess && items.length === 0 && <EmptyState icon={ClipboardCheck} title="Nothing in the queue" />}
        {items.length > 0 && (
          <div className="divide-y divide-border-muted">
            {items.map((item) => (
              <QueueRow key={item.id} item={item} client={client} session={session} navigate={navigate} onChanged={invalidate} alertRules={alertRules} />
            ))}
          </div>
        )}
        {listQuery.hasNextPage && (
          <div className="mt-3 flex justify-center">
            <Button busy={listQuery.isFetchingNextPage} onClick={() => void listQuery.fetchNextPage()}>
              Load more
            </Button>
          </div>
        )}
      </Card>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-border-muted bg-canvas-subtle px-3 py-2 text-center">
      <div className="text-lg font-semibold text-fg tabular-nums">{value}</div>
      <div className="text-xs text-fg-muted">{label}</div>
    </div>
  );
}

function QueueRow({
  item,
  client,
  session,
  navigate,
  onChanged,
  alertRules,
}: {
  item: ReviewQueueItem;
  client: QueueViewProps['client'];
  session: QueueViewProps['session'];
  navigate: QueueViewProps['navigate'];
  onChanged: () => void;
  alertRules: SignalAlertRuleRecord[] | undefined;
}) {
  const triggers = item.trigger_alert_rule_ids ?? [];
  const open = item.status === 'PENDING' || item.status === 'IN_REVIEW';
  // An item stays when its alert stops matching (§9.3); say so. The call's current alerts come from
  // its contact signals (shared cache key with the Workbench, so the change feed refreshes it).
  const signalsQuery = useQuery({
    queryKey: [...queryKeys.call(item.call_id), 'contact-signals'],
    queryFn: ({ signal }) => client.get('/store/v1/calls/{call_id}/contact-signals', { path: { call_id: item.call_id }, signal }),
    enabled: triggers.length > 0 && open,
    retry: false,
  });
  const currentAlerts = signalsQuery.data ? new Set(signalsQuery.data.alerts.map((a) => a.rule_id)) : null;
  const signalsChanged = currentAlerts !== null && triggers.some((id) => !currentAlerts.has(id));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [notes, setNotes] = useState('');
  const [resolving, setResolving] = useState(false);
  const d = reviewQueueStatusDisplay(item.status);
  const isMine = item.assigned_to_account_id === session.session.account_id;
  const qc = useQueryClient();

  // Resolving writes a review decision, which Store checks against the call's review_version
  // (not the queue item's item_version), so read the call's review state when the form opens.
  const reviewQuery = useQuery({
    queryKey: [...queryKeys.call(item.call_id), 'review'],
    queryFn: ({ signal }) => client.get('/store/v1/calls/{call_id}/review', { path: { call_id: item.call_id }, signal }),
    enabled: resolving,
  });
  const reviewVersion = reviewQuery.data?.review_version;

  async function act<T>(fn: () => Promise<T>) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChanged();
      void qc.invalidateQueries({ queryKey: queryKeys.call(item.call_id) });
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) {
        // Re-read the item and the call's review state; the next submit uses the new versions.
        onChanged();
        void qc.invalidateQueries({ queryKey: queryKeys.call(item.call_id) });
      }
    } finally {
      setBusy(false);
    }
  }

  const resolve = (status: 'APPROVED' | 'OVERRIDDEN') => {
    if (reviewVersion === undefined) return;
    void act(() =>
      client.post('/store/v1/review-queue/items/{item_id}/resolve', {
        path: { item_id: item.id },
        body: {
          status,
          evaluation_version: item.evaluation_version,
          expected_item_version: item.item_version,
          expected_review_version: reviewVersion,
          reviewer_notes: notes || null,
        },
      }),
    );
  };

  return (
    <div className="py-3 space-y-1.5" data-queue-item={item.id}>
      <div className="flex items-center justify-between gap-2">
        <a href={href({ name: 'workbench', callId: item.call_id })} className="min-w-0">
          <span className="font-medium text-fg">{agentLabel(item.agent_id, item.agent_display_name, item.agent_extension) ?? item.call_id}</span>
          <span className="text-xs text-fg-muted ml-2">{item.rule_name}</span>
        </a>
        <div className="flex items-center gap-2 shrink-0">
          {item.stale && (
            <StatusPill tone="magenta" title="A newer machine result exists than this item was created for.">
              Stale
            </StatusPill>
          )}
          {item.critical_failure && <StatusPill tone="red">Critical</StatusPill>}
          <StatusPill tone={d.tone}>{d.label}</StatusPill>
        </div>
      </div>
      {triggers.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5" aria-label="Triggered by signal alerts">
          {triggers.map((id) => (
            <Chip key={id} tone="yellow" icon={Bell} title="The signal alert that created this item">
              {alertRuleName(alertRules, id)}
            </Chip>
          ))}
          {signalsChanged && (
            <StatusPill tone="magenta" title="The call's signals were updated and this alert no longer matches; the item stays.">
              Signals changed since this item was created
            </StatusPill>
          )}
        </div>
      )}
      <div className="text-xs text-fg-muted flex flex-wrap gap-x-3">
        <span>{item.reason}</span>
        <span>{streamLabel(item.stream)}</span>
        <span>score {item.overall_score ?? '—'}</span>
        <span>created {formatDateTime(item.created_at)}</span>
        {item.assigned_display_name && <span>assigned to {item.assigned_display_name}</span>}
      </div>
      <ErrorNotice error={error} />

      {item.status === 'PENDING' && (
        <div className="flex gap-1.5">
          {session.can('assign_review') && !item.assigned_to_account_id && (
            <Button
              size="sm"
              busy={busy}
              onClick={() => void act(() => client.post('/store/v1/review-queue/items/{item_id}/assign', { path: { item_id: item.id }, body: { account_id: session.session.account_id, expected_item_version: item.item_version } }))}
            >
              Assign to me
            </Button>
          )}
          {(isMine || session.can('claim_review')) && (
            <Button
              size="sm"
              variant="primary"
              busy={busy}
              onClick={() => void act(() => client.post('/store/v1/review-queue/items/{item_id}/start', { path: { item_id: item.id }, body: { expected_item_version: item.item_version } }))}
            >
              Start
            </Button>
          )}
        </div>
      )}

      {item.status === 'IN_REVIEW' && (isMine || session.atLeast('supervisor')) && (
        <div className="space-y-1.5">
          {resolving ? (
            <div className="flex flex-wrap items-end gap-2">
              <Field label="Reviewer notes">{(id) => <TextInput id={id} value={notes} maxLength={2000} className="w-64" onChange={(e) => setNotes(e.target.value)} />}</Field>
              <Button size="sm" variant="primary" busy={busy} disabled={reviewVersion === undefined} onClick={() => resolve('APPROVED')}>
                Approve
              </Button>
              <Button size="sm" variant="danger" busy={busy} disabled={reviewVersion === undefined} onClick={() => resolve('OVERRIDDEN')}>
                Override
              </Button>
              <Button size="sm" variant="ghost" disabled={busy} onClick={() => setResolving(false)}>
                Cancel
              </Button>
              {reviewQuery.isLoading && <Loading label="Reading the call's review state…" />}
              <ErrorNotice error={reviewQuery.error} />
            </div>
          ) : (
            <div className="flex gap-1.5">
              <Button size="sm" variant="primary" onClick={() => setResolving(true)}>
                Resolve
              </Button>
              <Button
                size="sm"
                busy={busy}
                onClick={() => void act(() => client.post('/store/v1/review-queue/items/{item_id}/release', { path: { item_id: item.id }, body: { expected_item_version: item.item_version } }))}
              >
                Release
              </Button>
              <Button size="sm" variant="ghost" onClick={() => navigate({ name: 'workbench', callId: item.call_id })}>
                Open in Workbench
              </Button>
            </div>
          )}
        </div>
      )}

      {(item.status === 'APPROVED' || item.status === 'OVERRIDDEN' || item.status === 'SUPERSEDED') && (
        <div className="flex items-center gap-1.5 text-xs text-fg-muted">
          <CheckCircle2 className="w-3.5 h-3.5" aria-hidden="true" />
          Resolved {item.resolved_at ? formatDateTime(item.resolved_at) : ''}
        </div>
      )}
    </div>
  );
}

function QueueRules({ client, alertRules }: { client: QueueViewProps['client']; alertRules: SignalAlertRuleRecord[] | undefined }) {
  const qc = useQueryClient();
  const rulesQuery = useQuery({
    queryKey: [...queryKeys.reviewQueue, 'rules'],
    queryFn: ({ signal }) => client.get('/store/v1/review-queue/rules', { signal }),
  });
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState({
    name: '',
    description: '',
    stream: 'TRIAGE' as ReviewStream,
    distribution_strategy: 'UNASSIGNED_CLAIM' as DistributionStrategy,
    sampling_rate: 0,
    rank: 100,
    critical_failure_only: false,
    low_confidence_only: false,
    enabled: true,
    target_agents: [] as string[],
    target_domains: [] as string[],
    target_skills: [] as string[],
    target_signal_alerts: [] as string[],
  });
  // Any stream may require signal alerts ("20% of cancellations" is AUDIT_SAMPLE plus an alert);
  // the SIGNAL stream always does, so its Alerts group is always open.
  const [alertsOnly, setAlertsOnly] = useState(false);
  const showAlerts = draft.stream === 'SIGNAL' || alertsOnly;
  const signalNeedsAlert = draft.stream === 'SIGNAL' && draft.target_signal_alerts.length === 0;

  async function createRule() {
    setBusy(true);
    setError(null);
    try {
      const id = window.prompt('Rule id?')?.trim();
      if (!id) return;
      await client.put('/store/v1/review-queue/rules/{rule_id}', {
        path: { rule_id: id },
        body: { expected_rule_version: 0, rule: { ...draft, target_signal_alerts: showAlerts ? draft.target_signal_alerts : [], id } },
      });
      setCreating(false);
      void qc.invalidateQueries({ queryKey: [...queryKeys.reviewQueue, 'rules'] });
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) void qc.invalidateQueries({ queryKey: [...queryKeys.reviewQueue, 'rules'] });
    } finally {
      setBusy(false);
    }
  }

  async function toggleEnabled(rule: ReviewQueueRuleRecord) {
    setBusy(true);
    setError(null);
    try {
      await client.put('/store/v1/review-queue/rules/{rule_id}', {
        path: { rule_id: rule.id },
        body: { expected_rule_version: rule.rule_version, rule: { ...ruleForSave(rule), enabled: !rule.enabled } },
      });
      void qc.invalidateQueries({ queryKey: [...queryKeys.reviewQueue, 'rules'] });
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) void qc.invalidateQueries({ queryKey: [...queryKeys.reviewQueue, 'rules'] });
    } finally {
      setBusy(false);
    }
  }

  async function remove(rule: ReviewQueueRuleRecord) {
    if (!window.confirm(`Delete rule "${rule.name}"?`)) return;
    setBusy(true);
    setError(null);
    try {
      await client.delete('/store/v1/review-queue/rules/{rule_id}', { path: { rule_id: rule.id } });
      void qc.invalidateQueries({ queryKey: [...queryKeys.reviewQueue, 'rules'] });
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) void qc.invalidateQueries({ queryKey: [...queryKeys.reviewQueue, 'rules'] });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card
      title="Queue rules"
      subtitle="Which calls route into the review queue, and how they're distributed."
      right={
        <Button size="sm" onClick={() => setCreating((c) => !c)}>
          {creating ? 'Cancel' : 'New rule'}
        </Button>
      }
    >
      {rulesQuery.isLoading && <Loading label="Loading rules…" />}
      <ErrorNotice error={rulesQuery.error} />
      <ErrorNotice error={error} />

      {creating && (
        <div className="mb-3 p-3 rounded-md border border-border-muted bg-canvas space-y-2">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <Field label="Name">{(id) => <TextInput id={id} value={draft.name} onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))} />}</Field>
            <Field label="Stream">
              {(id) => (
                <SelectInput id={id} value={draft.stream} onChange={(e) => setDraft((d) => ({ ...d, stream: e.target.value as ReviewStream }))}>
                  {STREAMS.map((s) => (
                    <option key={s} value={s}>
                      {STREAM_LABELS[s]}
                    </option>
                  ))}
                </SelectInput>
              )}
            </Field>
            <Field label="Distribution">
              {(id) => (
                <SelectInput id={id} value={draft.distribution_strategy} onChange={(e) => setDraft((d) => ({ ...d, distribution_strategy: e.target.value as DistributionStrategy }))}>
                  {DISTRIBUTIONS.map((s) => (
                    <option key={s} value={s}>
                      {DISTRIBUTION_LABELS[s]}
                    </option>
                  ))}
                </SelectInput>
              )}
            </Field>
            <Field label="Sampling rate" hint="0–1">
              {(id) => (
                <TextInput
                  id={id}
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={draft.sampling_rate}
                  onChange={(e) => setDraft((d) => ({ ...d, sampling_rate: Number(e.target.value) }))}
                />
              )}
            </Field>
          </div>
          <label className="flex items-center gap-2 text-sm text-fg">
            <input type="checkbox" checked={draft.critical_failure_only} onChange={(e) => setDraft((d) => ({ ...d, critical_failure_only: e.target.checked }))} />
            Critical failures only
          </label>
          <label className="flex items-center gap-2 text-sm text-fg">
            <input type="checkbox" checked={draft.low_confidence_only} onChange={(e) => setDraft((d) => ({ ...d, low_confidence_only: e.target.checked }))} />
            Low-confidence verdicts only
          </label>
          {draft.stream !== 'SIGNAL' && (
            <Checkbox
              checked={alertsOnly}
              onChange={(e) => setAlertsOnly(e.target.checked)}
              label="Only calls with a signal alert"
              hint="The call must match at least one chosen alert, checked before the stream."
            />
          )}
          {showAlerts && (
            <fieldset className="space-y-1 rounded-md border border-border-muted p-2">
              <legend className="text-xs font-medium text-fg-muted px-1">Alerts</legend>
              {(alertRules ?? []).length === 0 ? (
                <p className="text-xs text-fg-muted">No signal alert rules yet. An admin adds them under Signals → Alert rules.</p>
              ) : (
                (alertRules ?? []).map((r) => (
                  <Checkbox
                    key={r.rule_id}
                    checked={draft.target_signal_alerts.includes(r.rule_id)}
                    onChange={(e) =>
                      setDraft((d) => ({
                        ...d,
                        target_signal_alerts: e.target.checked ? [...d.target_signal_alerts, r.rule_id] : d.target_signal_alerts.filter((x) => x !== r.rule_id),
                      }))
                    }
                    label={`Alert: ${r.name}`}
                    hint={r.enabled ? undefined : 'Disabled: matches nothing until enabled.'}
                  />
                ))
              )}
              {signalNeedsAlert && <p className="text-xs text-primer-yellowFg">A SIGNAL rule needs at least one alert.</p>}
            </fieldset>
          )}
          <Button busy={busy} disabled={signalNeedsAlert} onClick={() => void createRule()}>
            Create
          </Button>
        </div>
      )}

      {rulesQuery.data && rulesQuery.data.items.length === 0 && <EmptyState title="No rules yet" />}
      {rulesQuery.data && rulesQuery.data.items.length > 0 && (
        <div className="divide-y divide-border-muted">
          {rulesQuery.data.items.map((rule) => (
            <div key={rule.id} className="py-2 flex flex-wrap items-center justify-between gap-2 text-sm">
              <div className="min-w-0">
                <span className="font-medium text-fg">{rule.name}</span>
                <span className="text-xs text-fg-muted ml-2">
                  {streamLabel(rule.stream)} · {distributionLabel(rule.distribution_strategy)} · rank {rule.rank}
                  {(rule.target_signal_alerts ?? []).length > 0 && ` · alerts: ${(rule.target_signal_alerts ?? []).map((id) => alertRuleName(alertRules, id)).join(', ')}`}
                </span>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                <StatusPill tone={rule.enabled ? 'green' : 'neutral'}>{rule.enabled ? 'Enabled' : 'Disabled'}</StatusPill>
                <Button size="sm" busy={busy} onClick={() => void toggleEnabled(rule)}>
                  {rule.enabled ? 'Disable' : 'Enable'}
                </Button>
                <Button size="sm" variant="danger" busy={busy} onClick={() => void remove(rule)}>
                  Delete
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
