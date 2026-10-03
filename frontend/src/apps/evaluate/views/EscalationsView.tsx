// `#/escalations` — `GET /store/v1/escalations`; supervisors resolve with
// `POST /store/v1/calls/{call_id}/escalation`.

import { useState } from 'react';
import { useInfiniteQuery, useQueryClient } from '@tanstack/react-query';
import { AlertTriangle } from 'lucide-react';
import { agentLabel, escalationStatusDisplay, isVersionConflict, queryKeys, type EscalationListItem, type EscalationStatus } from '../api';
import { Button, EmptyState, ErrorNotice, Field, Loading, PageHeader, SelectInput, StatusPill, TextInput, formatDateTime } from '../components/ui';
import { href } from '../state/router';
import type { EscalationsViewProps } from './types';

export default function EscalationsView({ client, session }: EscalationsViewProps) {
  const qc = useQueryClient();
  const [status, setStatus] = useState<EscalationStatus | ''>('PENDING');

  const q = useInfiniteQuery({
    queryKey: [...queryKeys.escalations, status],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) => client.get('/store/v1/escalations', { query: { limit: 50, page_token: pageParam, status: status || null }, signal }),
    getNextPageParam: (last) => last.next_page_token,
  });
  const items = q.data?.pages.flatMap((p) => p.items) ?? [];

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      <PageHeader
        title="Escalations"
        description="Criteria the machine flagged for a supervisor."
        right={
          <SelectInput aria-label="Filter by status" value={status} onChange={(e) => setStatus(e.target.value as EscalationStatus | '')} className="w-44 max-w-full">
            <option value="">All statuses</option>
            <option value="PENDING">Pending</option>
            <option value="APPROVED">Approved</option>
            <option value="OVERRIDDEN">Overridden</option>
          </SelectInput>
        }
      />
      {q.isLoading && <Loading label="Loading escalations…" />}
      <ErrorNotice error={q.error} />
      {q.isSuccess && items.length === 0 && (
        <EmptyState icon={AlertTriangle} title={status === 'PENDING' ? 'No escalations waiting' : 'No escalations match this filter'}>
          {status === 'PENDING' ? 'Calls the machine flags for a supervisor appear here.' : undefined}
        </EmptyState>
      )}
      {items.length > 0 && (
        <div className="rounded-lg border border-border bg-canvas-subtle divide-y divide-border-muted">
          {items.map((item) => (
            <EscalationRow
              key={item.call_id}
              item={item}
              client={client}
              session={session}
              onResolved={() => {
                void qc.invalidateQueries({ queryKey: queryKeys.escalations });
                void qc.invalidateQueries({ queryKey: queryKeys.call(item.call_id) });
              }}
            />
          ))}
        </div>
      )}
      {q.hasNextPage && (
        <div className="flex justify-center">
          <Button busy={q.isFetchingNextPage} onClick={() => void q.fetchNextPage()}>
            Load more
          </Button>
        </div>
      )}
    </div>
  );
}

function EscalationRow({
  item,
  client,
  session,
  onResolved,
}: {
  item: EscalationListItem;
  client: EscalationsViewProps['client'];
  session: EscalationsViewProps['session'];
  onResolved: () => void;
}) {
  const d = escalationStatusDisplay(item.escalation_status);
  const [notes, setNotes] = useState('');
  const [resolving, setResolving] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function resolve(status: 'APPROVED' | 'OVERRIDDEN') {
    setBusy(true);
    setError(null);
    try {
      await client.post('/store/v1/calls/{call_id}/escalation', {
        path: { call_id: item.call_id },
        body: { escalation_status: status, evaluation_version: item.evaluation_version, expected_version: item.review_version, reviewer_notes: notes || null },
      });
      setResolving(false);
      onResolved();
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) onResolved(); // re-read the list: the row shows the call's current state
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="px-4 py-3 space-y-1.5">
      <div className="flex items-center justify-between gap-2">
        <a href={href({ name: 'workbench', callId: item.call_id })} className="min-w-0">
          <span className="font-medium text-fg">{agentLabel(item.agent_id, item.agent_display_name, item.agent_extension)}</span>
          <span className="text-xs text-fg-muted ml-2 break-all">{item.call_id}</span>
        </a>
        <div className="flex items-center gap-2 shrink-0">
          {item.critical_failure && <StatusPill tone="red">Critical</StatusPill>}
          <StatusPill tone={d.tone}>{d.label}</StatusPill>
        </div>
      </div>
      <div className="text-xs text-fg-muted flex flex-wrap gap-x-3">
        <span>score {item.overall_score ?? '—'}</span>
        <span>{formatDateTime(item.created_at)}</span>
      </div>
      <ErrorNotice error={error} />
      {item.escalation_status === 'PENDING' && session.can('resolve_escalation') && (
        <div>
          {resolving ? (
            <div className="flex flex-wrap items-end gap-2">
              <Field label="Reviewer notes">{(id) => <TextInput id={id} value={notes} maxLength={2000} className="w-64 max-w-full" onChange={(e) => setNotes(e.target.value)} />}</Field>
              <Button size="sm" variant="primary" busy={busy} onClick={() => void resolve('APPROVED')}>
                Approve
              </Button>
              <Button size="sm" variant="danger" busy={busy} onClick={() => void resolve('OVERRIDDEN')}>
                Override
              </Button>
              <Button size="sm" variant="ghost" disabled={busy} onClick={() => setResolving(false)}>
                Cancel
              </Button>
            </div>
          ) : (
            <Button size="sm" onClick={() => setResolving(true)}>
              Resolve
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
