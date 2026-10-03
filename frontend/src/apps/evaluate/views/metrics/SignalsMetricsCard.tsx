// The Metrics page's Signals card (docs/ContactSignalsV2.md §10.3, §9.5): "Top caller needs" (the
// Caller objective's subcategories ranked by calls, each linking to the filtered call list), a
// category table with hit rate and precision, a drill-down per category (subcategories, enum and
// boolean field values, by day, top agents), and the alert table. Precision reads "—" until five
// hits are judged. Store computes every number from its projection tables; nothing is derived here.

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Radar } from 'lucide-react';
import {
  FAMILY_DISPLAY,
  agentLabel,
  formatPct,
  isNotImplemented,
  queryKeys,
  signalFamily,
  type SignalCategoryMetric,
  type SignalCount,
  type StoreClient,
} from '../../api';
import { Card, Chip, EmptyState, ErrorNotice, Loading, NotBuiltYet } from '../../components/ui';
import type { SignedInSession } from '../../state/app';
import { href } from '../../state/router';

const PRECISION_HINT = 'Shown once at least 5 hits are judged by reviewers';

export function SignalsMetricsCard({ client, session, start, end }: { client: StoreClient; session: SignedInSession; start: string | null; end: string | null }) {
  const range = { start, end };
  const q = useQuery({
    queryKey: queryKeys.signalMetrics(range),
    queryFn: ({ signal }) => client.get('/store/v1/metrics/signals', { query: range, signal }),
    enabled: session.can('read_metrics'),
  });
  const [drill, setDrill] = useState<string | null>(null);
  const data = q.data;
  const drilled = data?.categories.find((c) => c.category_id === drill);

  return (
    <Card title="Signals" icon={Radar} subtitle="What callers need and what happens on calls. Unscored: signals never change a score.">
      {!session.can('read_metrics') ? (
        <p className="text-sm text-fg-muted">Your role does not include metrics.</p>
      ) : q.isLoading ? (
        <Loading label="Loading signal metrics…" />
      ) : q.error ? (
        isNotImplemented(q.error) ? <NotBuiltYet what="Signal metrics" /> : <ErrorNotice error={q.error} />
      ) : data ? (
        <div className="space-y-5" data-testid="signals-metrics">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            <Tile label="Calls with signals scored" value={data.calls_scored} />
            <Tile label="Calls with a signal" value={data.calls_with_signal} />
            <Tile
              label="Pipeline"
              value={
                Object.entries(data.calls_by_pipeline)
                  .map(([k, n]) => `${k} · ${n} call${n === 1 ? '' : 's'}`)
                  .join(', ') || '—'
              }
            />
          </div>

          <section aria-labelledby="top-caller-needs" className="space-y-2">
            <h3 id="top-caller-needs" className="text-sm font-semibold text-fg">
              Top caller needs
            </h3>
            {data.top_caller_needs.length === 0 ? (
              <EmptyState title="No caller needs yet">
                Caller objective has no subcategory hits in this range. Add subcategories under Signals → Caller objective, then update recent calls.
              </EmptyState>
            ) : (
              <ol className="space-y-1.5" aria-label="Top caller needs">
                {data.top_caller_needs.map((n, i) => (
                  <NeedRow key={n.id} rank={i + 1} need={n} />
                ))}
              </ol>
            )}
          </section>

          <section aria-labelledby="signal-categories" className="space-y-2">
            <h3 id="signal-categories" className="text-sm font-semibold text-fg">
              Categories
            </h3>
            {data.categories.length === 0 ? (
              <EmptyState title="No signal results in this range" />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm" aria-label="Signal categories">
                  <thead>
                    <tr className="text-left text-xs text-fg-muted border-b border-border">
                      <th scope="col" className="font-medium py-1.5 pr-2">Category</th>
                      <th scope="col" className="font-medium py-1.5 pr-2">Hit rate</th>
                      <th scope="col" className="font-medium py-1.5 pr-2">Hits</th>
                      <th scope="col" className="font-medium py-1.5 pr-2" title={PRECISION_HINT}>Category precision</th>
                      <th scope="col" className="font-medium py-1.5 pr-2" title={PRECISION_HINT}>Subcategory precision</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border-muted">
                    {data.categories.map((c) => {
                      const fam = FAMILY_DISPLAY[signalFamily(c.category_id)];
                      return (
                        <tr key={c.category_id} className={drill === c.category_id ? 'bg-canvas' : undefined}>
                          <td className="py-1.5 pr-2">
                            <button
                              type="button"
                              aria-expanded={drill === c.category_id}
                              onClick={() => setDrill(drill === c.category_id ? null : c.category_id)}
                              className="text-left text-fg hover:underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                            >
                              {c.name}
                            </button>{' '}
                            <Chip tone={fam.tone}>{c.builtin ? fam.label : 'Custom'}</Chip>
                            {!c.active && <span className="text-xs text-fg-subtle ml-1">retired</span>}
                          </td>
                          <td className="py-1.5 pr-2 tabular-nums text-fg-muted">
                            {formatPct(c.hit_rate_pct)} <span className="text-xs">({c.calls_with_hit}/{c.calls_scored})</span>
                          </td>
                          <td className="py-1.5 pr-2 tabular-nums text-fg-muted">{c.hits_total}</td>
                          <td className="py-1.5 pr-2 tabular-nums text-fg-muted" title={c.precision_pct === null ? PRECISION_HINT : `${c.confirmed} confirmed, ${c.dismissed} dismissed`}>
                            <Precision pct={c.precision_pct} />
                          </td>
                          <td className="py-1.5 pr-2 tabular-nums text-fg-muted" title={c.subcategory_precision_pct === null ? PRECISION_HINT : undefined}>
                            <Precision pct={c.subcategory_precision_pct} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            {drilled && <CategoryDrillDown metric={drilled} />}
          </section>

          <section aria-labelledby="signal-alerts" className="space-y-2">
            <h3 id="signal-alerts" className="text-sm font-semibold text-fg">
              Alerts
            </h3>
            {data.alerts.length === 0 ? (
              <p className="text-sm text-fg-muted">No alert rules. An admin adds them under Signals → Alert rules.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm" aria-label="Signal alerts">
                  <thead>
                    <tr className="text-left text-xs text-fg-muted border-b border-border">
                      <th scope="col" className="font-medium py-1.5 pr-2">Alert</th>
                      <th scope="col" className="font-medium py-1.5 pr-2">Status</th>
                      <th scope="col" className="font-medium py-1.5 pr-2">Calls matched</th>
                      <th scope="col" className="font-medium py-1.5 pr-2">Match rate</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border-muted">
                    {data.alerts.map((a) => (
                      <tr key={a.rule_id}>
                        <td className="py-1.5 pr-2">
                          <a className="text-fg hover:underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue" href={href({ name: 'calls', filters: { signal_alert: a.rule_id } })}>
                            {a.name}
                          </a>
                        </td>
                        <td className="py-1.5 pr-2 text-fg-muted">{a.enabled ? 'Enabled' : 'Disabled'}</td>
                        <td className="py-1.5 pr-2 tabular-nums text-fg-muted">{a.calls_matched}</td>
                        <td className="py-1.5 pr-2 tabular-nums text-fg-muted">{formatPct(a.match_rate_pct)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </div>
      ) : null}
    </Card>
  );
}

function NeedRow({ rank, need }: { rank: number; need: SignalCount }) {
  const share = Math.max(0, Math.min(100, need.share_pct));
  return (
    <li data-need={need.id}>
      <a
        href={href({ name: 'calls', filters: { signal_category: 'intent', signal_subcategory: need.id } })}
        className="grid grid-cols-[1.5rem_minmax(0,1fr)_auto] items-center gap-2 rounded-md px-2 py-1.5 hover:bg-canvas focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
      >
        <span className="text-xs text-fg-subtle tabular-nums">{rank}.</span>
        <span className="min-w-0">
          <span className="block text-sm text-fg truncate">{need.name}</span>
          <span className="block h-1.5 rounded-full bg-canvas-inset overflow-hidden mt-1" aria-hidden="true">
            <span className="block h-full bg-primer-blue" style={{ width: `${share}%` }} />
          </span>
        </span>
        <span className="text-xs text-fg-muted tabular-nums whitespace-nowrap" title={need.precision_pct === null ? PRECISION_HINT : `Precision ${formatPct(need.precision_pct)}`}>
          {need.calls_with_hit} call{need.calls_with_hit === 1 ? '' : 's'} · {formatPct(need.share_pct)} of hits
        </span>
      </a>
    </li>
  );
}

function CategoryDrillDown({ metric: c }: { metric: SignalCategoryMetric }) {
  return (
    <div className="rounded-md border border-border-muted bg-canvas p-3 space-y-3" aria-label={`${c.name} details`}>
      <p className="text-sm font-medium text-fg">{c.name}</p>
      {c.subcategories.length > 0 ? (
        <table className="w-full text-sm" aria-label={`${c.name} subcategories`}>
          <thead>
            <tr className="text-left text-xs text-fg-muted border-b border-border">
              <th scope="col" className="font-medium py-1 pr-2">Subcategory</th>
              <th scope="col" className="font-medium py-1 pr-2">Calls</th>
              <th scope="col" className="font-medium py-1 pr-2">Share of hits</th>
              <th scope="col" className="font-medium py-1 pr-2" title={PRECISION_HINT}>Precision</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-muted">
            {c.subcategories.map((s) => (
              <tr key={s.id}>
                <td className="py-1 pr-2">
                  <a className="text-fg hover:underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue" href={href({ name: 'calls', filters: { signal_category: c.category_id, signal_subcategory: s.id } })}>
                    {s.name}
                  </a>
                </td>
                <td className="py-1 pr-2 tabular-nums text-fg-muted">{s.calls_with_hit}</td>
                <td className="py-1 pr-2 tabular-nums text-fg-muted">{formatPct(s.share_pct)}</td>
                <td className="py-1 pr-2 tabular-nums text-fg-muted" title={s.precision_pct === null ? PRECISION_HINT : undefined}>
                  <Precision pct={s.precision_pct} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="text-xs text-fg-muted">No subcategory results for this category in this range.</p>
      )}
      {c.fields.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-fg-muted">Field values</p>
          {c.fields.map((f) => (
            <div key={`${f.node_id}.${f.field_id}`} className="text-xs text-fg-muted">
              <span className="text-fg">{f.name}</span>:{' '}
              {f.values.length === 0 ? 'no values yet' : f.values.map((v) => `${typeof v.value === 'boolean' ? (v.value ? 'yes' : 'no') : v.value} (${v.count})`).join(', ')}
            </div>
          ))}
        </div>
      )}
      {c.by_day.length > 0 && (
        <div>
          <p className="text-xs font-medium text-fg-muted mb-1">By day (calls with a hit / calls scored)</p>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
            {c.by_day.map((d) => (
              <span key={d.day}>
                {d.day}: {d.calls_with_hit}/{d.calls_scored}
              </span>
            ))}
          </div>
        </div>
      )}
      {c.top_agents.length > 0 && (
        <div>
          <p className="text-xs font-medium text-fg-muted mb-1">Top agents</p>
          <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
            {c.top_agents.map((a) => (
              <li key={a.agent_id}>
                {agentLabel(a.agent_id, a.agent_display_name, a.agent_extension)}: {a.calls_with_hit}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** Precision, or a short "needs reviewer feedback" hint while fewer than 5 hits are judged. */
function Precision({ pct }: { pct: number | null }) {
  if (pct !== null) return <>{formatPct(pct)}</>;
  return <span className="text-xs text-fg-subtle">Needs reviewer feedback</span>;
}

function Tile({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border border-border-muted bg-canvas-subtle px-3 py-2 text-center">
      <div className="text-lg font-semibold text-fg tabular-nums">{value}</div>
      <div className="text-xs text-fg-muted">{label}</div>
    </div>
  );
}
