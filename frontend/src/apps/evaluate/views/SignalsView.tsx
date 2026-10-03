// `#/signals` — Contact Signals v2 admin (docs/ContactSignalsV2.md §9 and §10.1): the taxonomy
// editor (built-in and custom categories, subcategories, extraction fields), "Test on recent calls",
// the activation dialog, alert rules, versions and the pipeline selector.
//
// Everyone with `read_calls` reads it; only `manage_signals` (admin, decision 22 Q1) edits. Store is
// the authority for every rule shown here: the client-side checks only put the reason next to the
// field before a save, and a refusal from Store is shown with the path it names.

import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Bell, History, Radar } from 'lucide-react';
import {
  PIPELINE_LABEL,
  cloneTaxonomy,
  daysAgoIso,
  queryKeys,
  sameTaxonomy,
  type SignalPipeline,
  type SignalTaxonomy,
  type SignalTaxonomyRecord,
} from '../api';
import { Button, EmptyState, ErrorNotice, Loading, PageHeader, SelectInput, StatusPill } from '../components/ui';
import { href, type SignalsTab } from '../state/router';
import { usePollChanges } from '../state/app';
import { AlertRulesTab } from './signals/AlertRulesTab';
import { TaxonomyTab } from './signals/TaxonomyTab';
import { VersionsTab } from './signals/VersionsTab';
import type { SignalsViewProps } from './types';

const TABS: { tab: SignalsTab; label: string; icon: typeof Radar }[] = [
  { tab: 'taxonomy', label: 'Taxonomy', icon: Radar },
  { tab: 'alerts', label: 'Alert rules', icon: Bell },
  { tab: 'versions', label: 'Versions', icon: History },
];

export default function SignalsView(props: SignalsViewProps) {
  const { client, session, tab } = props;
  const canManage = session.can('manage_signals');

  const recordQuery = useQuery({
    queryKey: queryKeys.signalTaxonomy,
    queryFn: ({ signal }) => client.get('/store/v1/signals/taxonomy', { signal }),
  });
  const record = recordQuery.data;

  const rulesQuery = useQuery({
    queryKey: queryKeys.signalAlertRules,
    queryFn: ({ signal }) => client.get('/store/v1/signals/alert-rules', { signal }),
  });

  // "Calls with a hit in the last 7 days" per node and per alert (§10.1). The window start is fixed
  // for the page's lifetime so the query key stays stable.
  const [weekStart] = useState(() => daysAgoIso(7));
  const weekRange = useMemo(() => ({ start: weekStart, include_inactive: true, window: '7d' }), [weekStart]);
  const weekQuery = useQuery({
    queryKey: queryKeys.signalMetrics(weekRange),
    queryFn: ({ signal }) => client.get('/store/v1/metrics/signals', { query: { start: weekStart, include_inactive: true }, signal }),
    enabled: session.can('read_metrics'),
    retry: false,
  });

  // The editor's working copy. It follows the published version until the admin edits, and is kept
  // across tab changes (the view stays mounted for every #/signals route). Dirtiness is measured
  // against the version the draft was taken from, so another admin's save never makes an untouched
  // draft look edited; a settings-only change (same taxonomy version) never makes a draft stale.
  const [draft, setDraft] = useState<SignalTaxonomy | null>(null);
  const [base, setBase] = useState<{ version: number; taxonomy: SignalTaxonomy } | null>(null);
  const dirty = draft !== null && base !== null && !sameTaxonomy(draft, base.taxonomy);
  useEffect(() => {
    if (!record) return;
    if (draft === null || base === null || (!dirty && base.version !== record.current.version)) {
      setDraft(cloneTaxonomy(record.current.taxonomy));
      setBase({ version: record.current.version, taxonomy: cloneTaxonomy(record.current.taxonomy) });
    }
  }, [record, draft, base, dirty]);
  const staleDraft = dirty && record !== undefined && base !== null && base.version !== record.current.version;

  const resetDraft = (to?: SignalTaxonomyRecord) => {
    const r = to ?? record;
    if (!r) return;
    setDraft(cloneTaxonomy(r.current.taxonomy));
    setBase({ version: r.current.version, taxonomy: cloneTaxonomy(r.current.taxonomy) });
  };

  return (
    <div className="max-w-6xl mx-auto space-y-4">
      <PageHeader
        title="Signals"
        description="What Call1 listens for in every call: categories, subcategories, fields to extract, and the alerts they raise. Signals are unscored: they never change a call's score."
        right={
          canManage ? (
            <StatusPill tone="magenta" title="You hold manage_signals">
              Admin: can edit
            </StatusPill>
          ) : (
            <StatusPill tone="neutral" title="Only admins change the taxonomy (manage_signals)">
              Read-only
            </StatusPill>
          )
        }
      />

      {recordQuery.isLoading && <Loading label="Loading the signal taxonomy…" />}
      <ErrorNotice error={recordQuery.error} />

      {record && <PipelineSelector {...props} record={record} />}

      <nav aria-label="Signals sections" className="flex flex-wrap gap-1 border-b border-border">
        {TABS.map(({ tab: t, label, icon: Icon }) => {
          const active = tab === t;
          return (
            <a
              key={t}
              href={href({ name: 'signals', tab: t })}
              aria-current={active ? 'page' : undefined}
              className={`flex items-center gap-1.5 px-3 h-9 -mb-px text-sm border-b-2 rounded-t focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue ${
                active ? 'border-primer-blue text-fg font-medium' : 'border-transparent text-fg-muted hover:text-fg'
              }`}
            >
              <Icon className="w-3.5 h-3.5" aria-hidden="true" />
              {label}
              {t === 'taxonomy' && dirty && <span className="text-xs text-primer-yellowFg">(unsaved)</span>}
            </a>
          );
        })}
      </nav>

      {record && draft && tab === 'taxonomy' && (
        <TaxonomyTab
          {...props}
          record={record}
          draft={draft}
          dirty={dirty}
          staleDraft={staleDraft}
          setDraft={setDraft}
          resetDraft={resetDraft}
          alertRules={rulesQuery.data?.items ?? []}
          week={weekQuery.data}
        />
      )}
      {record && tab === 'alerts' && (
        <AlertRulesTab {...props} record={record} rulesQuery={rulesQuery} week={weekQuery.data} weekError={weekQuery.error} />
      )}
      {record && tab === 'versions' && <VersionsTab {...props} record={record} />}
      {!record && !recordQuery.isLoading && !recordQuery.error && <EmptyState title="No taxonomy" />}
    </div>
  );
}

/**
 * The pipeline selector (§10.1): admins choose which pipeline runs; shadow and v2
 * stay disabled until a Process host reports a qualified stage-1 classifier (`signal_category`) in
 * its catalog snapshot. Switching back to v1 is always allowed (§14 rollback).
 */
function PipelineSelector({ client, session, record }: SignalsViewProps & { record: SignalTaxonomyRecord }) {
  const qc = useQueryClient();
  const pollNow = usePollChanges();
  const canManage = session.can('manage_signals');
  const pipeline = record.settings.pipeline;
  const [choice, setChoice] = useState<SignalPipeline>(pipeline);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => setChoice(pipeline), [pipeline]);

  const catalogQuery = useQuery({
    queryKey: queryKeys.catalogSnapshots,
    queryFn: ({ signal }) => client.get('/store/v1/catalog-snapshots', { signal }),
    enabled: canManage,
    retry: false,
  });
  const qualified = (catalogQuery.data?.items ?? []).some((snap) =>
    snap.entries.some((e) => e.status === 'available' && e.qualified_for.includes('signal_category')),
  );

  async function apply() {
    setBusy(true);
    setError(null);
    try {
      const next = await client.put('/store/v1/signals/settings', {
        body: { settings: { ...record.settings, pipeline: choice }, expected_record_version: record.record_version },
      });
      qc.setQueryData(queryKeys.signalTaxonomy, next);
      pollNow();
    } catch (err) {
      setError(err);
      void qc.invalidateQueries({ queryKey: queryKeys.signalTaxonomy });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div data-testid="signals-pipeline-banner" className="space-y-2">
      {canManage && (
        <div className="flex flex-wrap items-end gap-2 text-sm">
          <label className="flex flex-col gap-1 text-xs font-medium text-fg-muted">
            Pipeline
            <SelectInput value={choice} onChange={(e) => setChoice(e.target.value as SignalPipeline)} className="w-56">
              <option value="v1">{PIPELINE_LABEL.v1}</option>
              <option value="shadow" disabled={!qualified}>
                {PIPELINE_LABEL.shadow}
                {!qualified ? ' — needs a qualified classifier' : ''}
              </option>
              <option value="v2" disabled={!qualified}>
                {PIPELINE_LABEL.v2}
                {!qualified ? ' — needs a qualified classifier' : ''}
              </option>
            </SelectInput>
          </label>
          <Button busy={busy} disabled={choice === pipeline} onClick={() => void apply()}>
            Switch pipeline
          </Button>
          {!qualified && (
            <p className="text-xs text-fg-muted basis-full">
              {catalogQuery.isLoading
                ? 'Checking the Process hosts for a qualified classifier…'
                : catalogQuery.error
                  ? 'Could not read the Process catalogs, so shadow and v2 stay off.'
                  : 'Shadow and v2 stay off until a Process host reports a qualified stage-1 classifier (engine not installed).'}
            </p>
          )}
          <div className="basis-full">
            <ErrorNotice error={error} />
          </div>
        </div>
      )}
    </div>
  );
}
