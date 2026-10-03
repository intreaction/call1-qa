import { useQuery } from '@tanstack/react-query';
import { Activity, AlertOctagon, Cpu, ExternalLink, Gauge, Layers, RefreshCw, Server } from 'lucide-react';
import { getOverview, queryKeys, type Overview, type WorkerDescribe } from '../api';
import { Button, Card, ErrorNotice, formatBytes, formatDateTime, formatDuration, formatRelative, humanize, Loading, Notice, PageHeader, StatusPill, type Tone } from '../components/ui';
import { handlerModeLabel, jobTypeLabel } from '../labels';

const STORE_STATE_TONE: Record<string, Tone> = {
  connected: 'green',
  running: 'green',
  connecting: 'blue',
  not_configured: 'yellow',
  stopped: 'neutral',
  store_unreachable: 'red',
  store_refused: 'red',
  contract_mismatch: 'red',
  error: 'red',
};

function toneFor(state: string): Tone {
  return STORE_STATE_TONE[state] ?? 'neutral';
}

/** Honest "Worker unavailable" wording per docs/SplitBuild.md — never implies work is happening
 * when the worker hasn't started. */
function WorkerUnavailable({ overview }: { overview: Overview }) {
  const reasons: Record<string, string> = {
    not_configured: 'Process has no Store connection configured yet. Issue a service key on the Store host and write it to this installation’s config.',
    connecting: 'Process is connecting to Store. The worker starts once the handshake finishes.',
    store_unreachable: "Process can't reach Store, so the worker never started.",
    store_refused: 'Store refused the connection, so the worker never started.',
    contract_mismatch: 'Store speaks a different contract major, so Process refused to start the worker.',
    error: 'Process hit an unexpected error during start-up.',
    stopped: 'Process is shutting down or has stopped; no worker is running.',
  };
  const detail = reasons[overview.state] ?? 'The worker is not running.';
  return (
    <div className="flex flex-col items-center text-center gap-2 py-8 px-4">
      <div className="w-9 h-9 rounded-full bg-primer-redSubtle border border-primer-redBorder flex items-center justify-center">
        <AlertOctagon className="w-4 h-4 text-primer-redFg" aria-hidden="true" />
      </div>
      <p className="text-sm font-medium text-fg">Worker unavailable</p>
      <p className="text-sm text-fg-muted max-w-sm">{detail}</p>
      {overview.store.error && <p className="text-xs text-fg-subtle">{overview.store.error.code}: {overview.store.error.message}</p>}
    </div>
  );
}

const DELIVERY_LABEL: Record<string, string> = {
  publish: 'outputs not uploaded yet',
  complete: 'completion not delivered yet',
  fail: 'failure not delivered yet',
};

function WorkerPanel({ worker }: { worker: WorkerDescribe }) {
  const counters: Array<[string, number]> = [
    ['Claimed', worker.stats.claimed],
    ['Succeeded', worker.stats.succeeded],
    ['Failed', worker.stats.failed],
    ['Released', worker.stats.released],
    ['Lost', worker.stats.lost],
    ['Spooled', worker.spooled],
  ];
  const waiting = worker.awaiting_delivery ?? [];
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <StatusPill tone={worker.state === 'running' ? 'green' : worker.state === 'stopped' ? 'neutral' : 'yellow'}>{humanize(worker.state)}</StatusPill>
        <span className="text-xs text-fg-muted">{worker.worker_id}</span>
        {worker.primary_host && <StatusPill tone="blue">Primary host</StatusPill>}
      </div>
      {/* Each label/value pair is one grid cell, so a value always sits on its label's row, to its
          right, at every width (known issue d: bare dt/dd cells drifted apart at >= 640px). */}
      <dl className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-1.5 text-sm">
        {counters.map(([label, value]) => (
          <div key={label} className="flex items-baseline justify-between gap-3 min-w-0">
            <dt className="text-fg-muted truncate">{label}</dt>
            <dd className="text-fg tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
      {worker.stats.last_error && <Notice tone="yellow">Last error: {worker.stats.last_error}</Notice>}
      {worker.running.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-fg-muted uppercase tracking-wide mb-1.5">Running now</h3>
          <ul className="space-y-1.5">
            {worker.running.map((r) => (
              <li key={r.job_id} className="flex items-center justify-between gap-2 text-sm">
                <span className="text-fg truncate">{jobTypeLabel(r.job_type)} <span className="text-fg-subtle text-xs">{r.pool}</span></span>
                <span className="text-fg-muted text-xs shrink-0">{formatDuration(r.running_seconds)}{r.cancel_requested ? ' · cancelling' : ''}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {waiting.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold text-fg-muted uppercase tracking-wide mb-1.5">Waiting for Store</h3>
          <p className="text-xs text-fg-subtle mb-1.5">
            Finished work kept in the spool. Process keeps each claim alive and delivers it when Store answers again.
          </p>
          <ul className="space-y-1.5">
            {waiting.map((r) => (
              <li key={`${r.job_id}-${r.attempt_number}`} className="flex items-center justify-between gap-2 text-sm">
                <span className="text-fg truncate">{jobTypeLabel(r.job_type)}</span>
                <StatusPill tone="yellow">{DELIVERY_LABEL[r.awaiting_delivery ?? ''] ?? 'waiting'}</StatusPill>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function SlotsPanel({ overview }: { overview: Overview }) {
  const pools = overview.worker?.pools;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-fg-muted uppercase tracking-wide">
            <th className="py-1.5 pr-3 font-medium">Pool</th>
            <th className="py-1.5 pr-3 font-medium">In use</th>
            <th className="py-1.5 pr-3 font-medium">Size</th>
            <th className="py-1.5 font-medium">Step types</th>
          </tr>
        </thead>
        <tbody>
          {pools
            ? pools.map((p) => (
                <tr key={p.pool} className="border-t border-border-muted">
                  <td className="py-1.5 pr-3 text-fg font-medium">{p.pool}</td>
                  <td className="py-1.5 pr-3 text-fg">
                    {p.in_use} / {p.size}
                  </td>
                  <td className="py-1.5 pr-3 text-fg-muted">{p.size}</td>
                  <td className="py-1.5 text-fg-muted">{p.job_types.length}</td>
                </tr>
              ))
            : overview.slots.map((p) => (
                <tr key={p.pool} className="border-t border-border-muted">
                  <td className="py-1.5 pr-3 text-fg font-medium">{p.pool}</td>
                  <td className="py-1.5 pr-3 text-fg-subtle">not started</td>
                  <td className="py-1.5 pr-3 text-fg-muted">{p.size}</td>
                  <td className="py-1.5 text-fg-subtle">—</td>
                </tr>
              ))}
        </tbody>
      </table>
    </div>
  );
}

export default function OverviewView() {
  const query = useQuery({ queryKey: queryKeys.overview, queryFn: getOverview, refetchInterval: 5_000 });

  return (
    <div className="max-w-5xl mx-auto">
      <PageHeader
        title="Overview"
        description="Is this Process connected to Store, is its worker running, and what is it working on right now."
        right={
          <Button icon={RefreshCw} size="sm" busy={query.isFetching} onClick={() => void query.refetch()}>
            Refresh
          </Button>
        }
      />
      {query.isLoading && <Loading label="Loading overview…" />}
      <ErrorNotice error={query.error} />
      {query.data?.training?.claims_paused && (
        <div className="mb-4" aria-live="polite">
          <Notice tone="yellow">
            Processing paused for on-device training (run {query.data.training.claims_paused.run_id}, since{' '}
            {formatRelative(query.data.training.claims_paused.since)}
            {query.data.training.claims_paused.until ? `, at most until ${formatDateTime(query.data.training.claims_paused.until)}` : ''}). New
            jobs wait until the run ends; see Settings.
          </Notice>
        </div>
      )}
      {query.data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card
            title="Store connection"
            icon={Server}
            subtitle="Where calls and results are kept"
            right={<StatusPill tone={toneFor(query.data.state)}>{humanize(query.data.state)}</StatusPill>}
          >
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-sm">
              <dt className="text-fg-muted">Store URL</dt>
              <dd className="text-fg break-all">
                {query.data.store.url ?? 'not configured'}
                {query.data.store.dev_mode && <span className="text-primer-yellowFg"> (dev mode)</span>}
              </dd>
              <dt className="text-fg-muted">Contract</dt>
              <dd className="text-fg">
                {query.data.store.contract_version ?? '—'}
                <span className="text-fg-subtle"> (Process built for {query.data.store.process_contract_version})</span>
                {query.data.store.compatible === true && <StatusPill tone="green">Compatible</StatusPill>}
                {query.data.store.compatible === false && <StatusPill tone="red">Mismatch</StatusPill>}
              </dd>
              {query.data.store.parameters && (
                <>
                  <dt className="text-fg-muted">Lease / heartbeat</dt>
                  <dd className="text-fg">
                    {query.data.store.parameters.lease_duration_seconds}s / {query.data.store.parameters.heartbeat_interval_seconds}s
                  </dd>
                  <dt className="text-fg-muted">Max claim batch</dt>
                  <dd className="text-fg">{query.data.store.parameters.max_claim_batch}</dd>
                  <dt className="text-fg-muted">Inline artifact limit</dt>
                  <dd className="text-fg">{formatBytes(query.data.store.parameters.inline_artifact_max_bytes)}</dd>
                </>
              )}
              <dt className="text-fg-muted">Installation</dt>
              <dd className="text-fg break-all">{query.data.installation_id ?? '—'}</dd>
              <dt className="text-fg-muted">Bind</dt>
              <dd className="text-fg">{query.data.bind}</dd>
              <dt className="text-fg-muted">Started</dt>
              <dd className="text-fg">{formatRelative(query.data.started_at)}</dd>
            </dl>
            {query.data.store.error && (
              <div className="mt-3">
                <Notice tone="red">{query.data.store.error.message}</Notice>
              </div>
            )}
          </Card>

          <Card title="Worker" icon={Cpu} subtitle="Runs processing steps on this computer">
            {query.data.worker ? <WorkerPanel worker={query.data.worker} /> : <WorkerUnavailable overview={query.data} />}
          </Card>

          <Card title="Resource slots" icon={Gauge} subtitle="How many steps of each kind run at once">
            <SlotsPanel overview={query.data} />
            <p className="text-xs text-fg-subtle mt-2">
              mlx is the Apple GPU and runs one step at a time. torch (PyTorch models) and cpu_io (file and network work) are sized in the
              Process config.
            </p>
          </Card>

          <Card title="Handlers" icon={Activity} subtitle={`Mode: ${query.data.handlers.mode} · ${handlerModeLabel(query.data.handlers.mode)}`}>
            {query.data.handlers.missing_job_types.length === 0 ? (
              <StatusPill tone="green">Every processing step has code to run it</StatusPill>
            ) : (
              <div className="space-y-1.5">
                <StatusPill tone="yellow">{query.data.handlers.missing_job_types.length} step(s) with nothing to run them</StatusPill>
                <ul className="text-sm text-fg-muted list-disc list-inside">
                  {query.data.handlers.missing_job_types.map((t) => (
                    <li key={t}>{jobTypeLabel(t)}</li>
                  ))}
                </ul>
              </div>
            )}
            {query.data.handlers.notes.length > 0 && (
              <ul className="mt-2 space-y-1">
                {query.data.handlers.notes.map((n, i) => (
                  <li key={i} className="text-xs text-fg-subtle">
                    {n}
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="Catalog & reanalysis" icon={Layers}>
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-sm">
              <dt className="text-fg-muted">Catalog version</dt>
              <dd className="text-fg text-xs">{query.data.catalog.version}</dd>
              <dt className="text-fg-muted">Published</dt>
              <dd className="text-fg">{query.data.catalog.published ? formatRelative(query.data.catalog.published.published_at) : 'not yet'}</dd>
              <dt className="text-fg-muted">Route settings from</dt>
              <dd className="text-fg">{query.data.admin_state}</dd>
              <dt className="text-fg-muted">Reanalysis handled</dt>
              <dd className="text-fg">{query.data.reanalysis?.handled ?? 0}</dd>
              <dt className="text-fg-muted">Reanalysis rejected</dt>
              <dd className="text-fg">{query.data.reanalysis?.rejected ?? 0}</dd>
            </dl>
            {query.data.reanalysis?.last_error && (
              <div className="mt-2">
                <Notice tone="yellow">{query.data.reanalysis.last_error}</Notice>
              </div>
            )}
          </Card>

          <Card title="This installation" icon={Server}>
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-sm">
              <dt className="text-fg-muted">Conversations ingested</dt>
              <dd className="text-fg">{query.data.conversations}</dd>
              <dt className="text-fg-muted">Scratch in use</dt>
              <dd className="text-fg">{formatBytes(query.data.scratch_bytes)}</dd>
              <dt className="text-fg-muted">Hardware profile</dt>
              <dd className="text-fg break-all text-xs">{query.data.hardware_profile_id ?? '—'}</dd>
            </dl>
            {query.data.evaluate_url && (
              <div className="mt-3">
                <a
                  href={query.data.evaluate_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 text-sm text-primer-blueFg hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue rounded"
                >
                  Open Evaluate <ExternalLink className="w-3.5 h-3.5" aria-hidden="true" />
                </a>
              </div>
            )}
          </Card>
        </div>
      )}
      {query.data && <p className="text-xs text-fg-subtle mt-3">Refreshes every 5s.</p>}
    </div>
  );
}
