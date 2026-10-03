import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ChevronDown, ChevronRight, ExternalLink, RefreshCw, Workflow } from 'lucide-react';
import {
  cancelJob,
  describeError,
  getConversation,
  listConversations,
  queryKeys,
  retryJob,
  type ConversationJob,
  type ConversationListItem,
} from '../api';
import { ConsoleTokenNotice } from '../components/ConsoleTokenNotice';
import { JobDiagnostics } from '../components/JobDiagnostics';
import { LiveProcessingGraph } from '../components/LiveProcessingGraph';
import { ReasonAction } from '../components/ReasonAction';
import {
  Button,
  Card,
  EmptyState,
  ErrorNotice,
  formatDateTime,
  formatRelative,
  humanize,
  Loading,
  Notice,
  PageHeader,
  StatusPill,
  type Tone,
} from '../components/ui';
import { JOB_TYPE_ORDER, jobStatusLabel, jobTypeLabel, routeLabel, waitingReasonLabel } from '../labels';

const JOB_STATUS_TONE: Record<string, Tone> = {
  SUCCEEDED: 'green',
  RUNNING: 'blue',
  QUEUED: 'neutral',
  BLOCKED: 'yellow',
  FAILED: 'red',
  CANCELLED: 'neutral',
  WAITING_PROVIDER: 'yellow',
};

/** Jobs in pipeline order: every job after the jobs it depends on (its `requires` and `after`
 * upstreams), ties broken by the usual stage order, then criterion, then creation time. */
function orderJobs(jobs: ConversationJob[]): ConversationJob[] {
  const rank = (j: ConversationJob) => {
    const i = JOB_TYPE_ORDER.indexOf(j.job_type);
    return i === -1 ? JOB_TYPE_ORDER.length : i;
  };
  const before = (a: ConversationJob, b: ConversationJob) =>
    rank(a) - rank(b) || (a.criterion_id ?? '').localeCompare(b.criterion_id ?? '') || a.created_at.localeCompare(b.created_at);
  const byId = new Map(jobs.map((j) => [j.id, j]));
  const pending = new Map<string, number>();
  const downstream = new Map<string, string[]>();
  for (const j of jobs) {
    const ups = [...new Set([...(j.requires_job_ids ?? []), ...(j.after_job_ids ?? [])])].filter((id) => byId.has(id) && id !== j.id);
    pending.set(j.id, ups.length);
    for (const u of ups) downstream.set(u, [...(downstream.get(u) ?? []), j.id]);
  }
  const ready = jobs.filter((j) => pending.get(j.id) === 0);
  const out: ConversationJob[] = [];
  while (ready.length > 0) {
    ready.sort(before);
    const next = ready.shift()!;
    out.push(next);
    for (const d of downstream.get(next.id) ?? []) {
      const left = (pending.get(d) ?? 0) - 1;
      pending.set(d, left);
      if (left === 0) ready.push(byId.get(d)!);
    }
  }
  // A cycle cannot happen in a valid graph; if one did, keep the leftovers rather than hide them.
  if (out.length < jobs.length) out.push(...jobs.filter((j) => !out.includes(j)).sort(before));
  return out;
}

/** "850 ms", "42 s", "3 min 05 s". */
function shortDuration(ms: number): string {
  if (ms < 1000) return `${Math.max(0, Math.round(ms))} ms`;
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s} s`;
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, '0')} s`;
}

function runTime(job: ConversationJob): string | null {
  if (!job.started_at) return null;
  const start = Date.parse(job.started_at);
  if (Number.isNaN(start)) return null;
  if (job.ended_at) {
    const end = Date.parse(job.ended_at);
    return Number.isNaN(end) ? null : `ran ${shortDuration(end - start)}`;
  }
  return job.status === 'RUNNING' ? `running for ${shortDuration(Date.now() - start)}` : null;
}

/** The overall state for the row's summary pill, worst-group-wins: a conversation with any
 * failed group needs attention even if others finished. */
function overallState(item: ConversationListItem): { tone: Tone; label: string } {
  if (item.progress_error && !item.progress) return { tone: 'red', label: 'Progress unavailable' };
  if (item.progress_error) return { tone: 'yellow', label: 'Last known' };
  const groups = item.progress?.groups ?? [];
  if (groups.some((g) => g.state === 'failed')) return { tone: 'red', label: 'Needs attention' };
  if (groups.length > 0 && groups.every((g) => g.state === 'available' || g.state === 'disabled')) {
    return { tone: 'green', label: 'Complete' };
  }
  if (groups.some((g) => g.state === 'stale')) return { tone: 'yellow', label: 'Stale' };
  return { tone: 'blue', label: 'Analyzing' };
}

function JobRow({
  job,
  conversationId,
  token,
  canWrite,
}: {
  job: ConversationJob;
  conversationId: string;
  token: string | null;
  canWrite: boolean;
}) {
  const queryClient = useQueryClient();
  const [cascade, setCascade] = useState(true);
  const invalidate = () => queryClient.invalidateQueries({ queryKey: queryKeys.conversation(conversationId) });

  const retryable = job.status === 'FAILED' || job.status === 'CANCELLED';
  const cancellable = job.status === 'QUEUED' || job.status === 'BLOCKED' || job.status === 'RUNNING';
  const route = [job.catalog_entry, routeLabel(job.route_class, job.destination_host)].filter(Boolean).join(' · ');
  const label = jobTypeLabel(job.job_type);
  const ran = runTime(job);

  return (
    <li data-testid="process-job-row" className="py-2.5 border-t border-border-muted first:border-t-0">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-fg">{label}</span>
            {job.criterion_id && <span className="text-xs text-fg-subtle">{job.criterion_id}</span>}
            <StatusPill tone={job.waiting_reason === 'dead_blocked' ? 'red' : (JOB_STATUS_TONE[job.status] ?? 'neutral')}>
              {jobStatusLabel(job.status, job.waiting_reason)}
            </StatusPill>
            {job.cancel_requested && <StatusPill tone="yellow">Cancel requested</StatusPill>}
          </div>
          <div className="text-xs text-fg-muted mt-0.5 flex flex-wrap gap-x-3">
            <span>{job.attempt_count === 0 ? 'Not run yet' : `Attempt ${job.attempt_count}/${job.max_attempts}`}</span>
            {ran && <span className="tabular-nums">{ran}</span>}
            {route && <span>{route}</span>}
            {job.waiting_reason && <span>{waitingReasonLabel(job.waiting_reason)}</span>}
            {job.error_code && <span className="text-primer-redFg">{humanize(job.error_code)}</span>}
            {job.completed_at ? (
              <span title={formatDateTime(job.completed_at)}>finished {formatRelative(job.completed_at)}</span>
            ) : (
              <span title={formatDateTime(job.updated_at)}>updated {formatRelative(job.updated_at)}</span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {retryable && (
            <ReasonAction
              label="Retry"
              confirmLabel="Retry job"
              dismissLabel="Close"
              variant="secondary"
              disabled={!canWrite}
              disabledReason="Connect the console token to retry jobs"
              warning={`Retry ${label}? This starts a new attempt.`}
              onConfirm={async (reason) => {
                await retryJob(job.id, reason, token);
                invalidate();
              }}
            />
          )}
          {cancellable && (
            <ReasonAction
              label="Cancel"
              confirmLabel="Cancel job"
              dismissLabel="Keep job"
              variant="danger"
              disabled={!canWrite}
              disabledReason="Connect the console token to cancel jobs"
              warning={`Cancel ${label}? This stops it before it produces a result.`}
              extraFields={
                <label className="flex items-center gap-2 text-sm text-fg-muted">
                  <input
                    type="checkbox"
                    checked={cascade}
                    onChange={(e) => setCascade(e.target.checked)}
                    className="rounded border-border-control focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                  />
                  Also cancel jobs that depend on this one
                </label>
              }
              onConfirm={async (reason) => {
                await cancelJob(job.id, reason, cascade, token);
                invalidate();
              }}
            />
          )}
        </div>
      </div>
      <JobDiagnostics jobId={job.id} />
    </li>
  );
}

function ConversationRow({ item, token, canWrite, autoOpen }: { item: ConversationListItem; token: string | null; canWrite: boolean; autoOpen: boolean }) {
  const [open, setOpen] = useState(autoOpen);
  const detail = useQuery({
    queryKey: queryKeys.conversation(item.conversation_id),
    queryFn: () => getConversation(item.conversation_id),
    enabled: open,
    refetchInterval: (query) => (open && !query.state.data?.progress?.settled ? 750 : false),
  });
  const overall = overallState(item);

  return (
    <Card className="p-0">
      {/* The Evaluate link sits beside the row's toggle button, not inside it: a link nested in a
          button is invalid and muddles keyboard focus and the button's accessible name. */}
      <div className="flex items-center gap-2 rounded-lg hover:bg-canvas-inset">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex-1 min-w-0 flex items-center gap-3 px-4 py-3 text-left rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
          aria-expanded={open}
        >
          {open ? (
            <ChevronDown className="w-4 h-4 text-fg-subtle shrink-0" aria-hidden="true" />
          ) : (
            <ChevronRight className="w-4 h-4 text-fg-subtle shrink-0" aria-hidden="true" />
          )}
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium text-fg truncate">{item.label ?? item.call_id ?? item.conversation_id}</span>
              <StatusPill tone={overall.tone}>{overall.label}</StatusPill>
            </div>
            <p className="text-xs text-fg-muted truncate mt-0.5">
              {item.progress_line
                ? item.progress_stale
                  ? `${item.progress_line} (last known; Store unreachable)`
                  : item.progress_line
                : item.progress_error
                  ? `Progress unavailable: ${item.progress_error === 'store_unavailable' ? 'Store unreachable' : item.progress_error}`
                  : 'No progress yet'}
            </p>
          </div>
        </button>
        {item.evaluate_url && (
          <a
            href={item.evaluate_url}
            target="_blank"
            rel="noreferrer"
            className="mr-4 inline-flex items-center gap-1 text-xs text-primer-blueFg hover:underline shrink-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue rounded"
          >
            Evaluate <ExternalLink className="w-3 h-3" aria-hidden="true" />
          </a>
        )}
      </div>
      {open && (
        <div className="px-4 pb-4">
          {detail.isLoading && <Loading label="Loading jobs…" />}
          <ErrorNotice error={detail.error} />
          {detail.data && (
            <div className="space-y-3">
              <LiveProcessingGraph detail={detail.data} compact active startedAt={(() => {
                const starts = detail.data.jobs.map((j) => Date.parse(j.started_at ?? '')).filter(Number.isFinite);
                return starts.length ? Math.min(...starts) : null;
              })()} renderJobs={(jobs) => (
                <ul aria-label="Job details" className="rounded-md border border-border-muted bg-canvas-subtle p-3">
                  {orderJobs(jobs).map((job) => <JobRow key={job.id} job={job} conversationId={item.conversation_id} token={token} canWrite={canWrite} />)}
                </ul>
              )} />
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

export default function PipelineView({
  focusConversation,
  token,
  tokenConfigured,
  onToken,
}: {
  focusConversation: string | null;
  token: string | null;
  tokenConfigured: boolean | undefined;
  onToken(token: string): void;
}) {
  const [limit, setLimit] = useState(50);
  const query = useQuery({ queryKey: queryKeys.conversations(limit), queryFn: () => listConversations(limit), refetchInterval: 8_000 });
  const canWrite = Boolean(token);

  return (
    <div className="max-w-4xl mx-auto space-y-3">
      <PageHeader
        title="Pipeline"
        description="One row per call this installation processed. Expand a call to follow its live processing flow. Select a stage for job details."
        right={
          <Button icon={RefreshCw} size="sm" busy={query.isFetching} onClick={() => void query.refetch()}>
            Refresh
          </Button>
        }
      />
      {!canWrite && <ConsoleTokenNotice configured={tokenConfigured} onToken={onToken} />}
      {query.isLoading && <Loading label="Loading conversations…" />}
      <ErrorNotice error={query.error} />
      {query.data?.store_unavailable && (
        <Notice tone="yellow">Store is unreachable right now, so each row shows the last progress seen. Rows refresh once Store is back.</Notice>
      )}
      {query.data && query.data.items.length === 0 && (
        <EmptyState icon={Workflow} title="No conversations yet">
          Upload a recording from the Import tab to see its processing steps here.
        </EmptyState>
      )}
      {query.data && query.data.items.length > 0 && (
        <div className="space-y-2">
          {query.data.items.map((item) => (
            <ConversationRow key={item.conversation_id} item={item} token={token} canWrite={canWrite} autoOpen={item.conversation_id === focusConversation} />
          ))}
        </div>
      )}
      {query.data && query.data.items.length < query.data.total && (
        <div className="flex justify-center pt-2">
          <Button variant="secondary" onClick={() => setLimit((l) => l + 50)}>
            Load more ({query.data.items.length} of {query.data.total})
          </Button>
        </div>
      )}
      {query.isError && <p className="text-xs text-fg-subtle">{describeError(query.error)}</p>}
    </div>
  );
}
