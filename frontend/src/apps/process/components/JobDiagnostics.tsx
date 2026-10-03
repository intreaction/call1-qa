import { useQuery } from '@tanstack/react-query';
import { getJob, queryKeys } from '../api';
import { ErrorNotice, Loading, humanize } from './ui';

function objects(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object') : [];
}
function string(value: unknown): string { return typeof value === 'string' ? value : ''; }
function eventTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export function JobDiagnostics({ jobId }: { jobId: string }) {
  const query = useQuery({ queryKey: queryKeys.job(jobId), queryFn: () => getJob(jobId), refetchInterval: (q) => ['RUNNING', 'QUEUED', 'BLOCKED', 'WAITING_PROVIDER'].includes(q.state.data?.job.status ?? '') ? 2_000 : false });
  if (query.isLoading) return <Loading label="Loading details…" />;
  const data = query.data;
  const attempts = [...(data?.attempts ?? [])].sort((a, b) => a.attempt_number - b.attempt_number || a.started_at.localeCompare(b.started_at));
  const provenance = attempts[attempts.length - 1]?.provenance;
  const inputs = objects(data?.job.resolved_inputs);
  const outputs = objects(data?.job.outputs);
  const fields = [
    ['Model revision', provenance?.model_revision], ['Runtime adapter', provenance?.adapter_id ? `${provenance.adapter_id}${provenance.adapter_version ? ` · ${provenance.adapter_version}` : ''}` : null],
    ['Worker', attempts[attempts.length - 1]?.worker_id],
  ].filter((field) => field[1]);
  return <div className="mt-2 space-y-3 text-xs" data-testid="job-diagnostics">
    <ErrorNotice error={query.error} />
    {data && <>
      {fields.length > 0 && <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-fg-muted">{fields.map(([label, value]) => <div key={label} className="contents"><dt className="text-fg-subtle">{label}</dt><dd className="break-all">{value}</dd></div>)}</dl>}
      <div className="rounded-md border border-border-muted bg-canvas-inset px-3 py-2 font-mono text-[11px] leading-5 text-fg-muted overflow-x-auto" aria-label="Recorded execution events">
        {string(data.job.created_at) && <p><time>{eventTime(string(data.job.created_at))}</time> · Job created</p>}
        {attempts.map((attempt, index) => <div key={`${attempt.attempt_number}-${attempt.started_at}-${index}`}>
          <p><time>{eventTime(attempt.started_at)}</time> · Attempt {attempt.attempt_number} started{attempt.counts_as_attempt ? '' : ' (released claim)'}</p>
          {attempt.ended_at ? <p><time>{eventTime(attempt.ended_at)}</time> · {humanize(attempt.status)}</p> : <p>Attempt {attempt.attempt_number} · {humanize(attempt.status)}</p>}
          {attempt.error_code && <p className="text-primer-redFg">{attempt.error_code}{attempt.error_detail ? `: ${attempt.error_detail}` : ''}</p>}
        </div>)}
        {!attempts.length && <p>No execution attempts yet.</p>}
        {string(data.job.error_detail) && <p className="text-primer-redFg whitespace-pre-wrap break-words">{string(data.job.error_detail)}</p>}
      </div>
      {(inputs.length > 0 || outputs.length > 0) && <dl className="space-y-1 text-fg-subtle">
        {inputs.map((input, i) => <div key={`in-${i}`} className="flex gap-2"><dt className="shrink-0">Input · {string(input.role)}</dt><dd className="font-mono break-all">{string(input.artifact_id)}</dd></div>)}
        {outputs.map((output, i) => <div key={`out-${i}`} className="flex gap-2"><dt className="shrink-0">Output · {string(output.role) || string(output.slot)}</dt><dd className="font-mono break-all">{string(output.artifact_id)}</dd></div>)}
      </dl>}
    </>}
  </div>;
}
