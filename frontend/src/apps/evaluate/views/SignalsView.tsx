// `#/signals` — Contact Signals v2 admin (docs/ContactSignalsV2.md §9 and §10.1): the taxonomy
// editor (built-in and custom categories, subcategories, extraction fields), "Test on recent calls",
// the activation dialog, alert rules, versions and the current pipeline status.
//
// Everyone with `read_calls` reads it; only `manage_signals` (admin, decision 22 Q1) edits. Store is
// the authority for every rule shown here: the client-side checks only put the reason next to the
// field before a save, and a refusal from Store is shown with the path it names.

import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Bell, History, Radar } from 'lucide-react';
import {
  cloneTaxonomy,
  daysAgoIso,
  queryKeys,
  sameTaxonomy,
  type SignalTaxonomy,
  type SignalTaxonomyRecord,
} from '../api';
import { EmptyState, ErrorNotice, Loading, PageHeader, StatusPill } from '../components/ui';
import { href, type SignalsTab } from '../state/router';
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

      {record && <PipelineStatus />}

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

function PipelineStatus() {
  return (
    <div data-testid="signals-pipeline-banner" className="rounded-lg border border-border bg-canvas-subtle p-3 text-sm">
      <p className="font-medium">Semantic similarity → Laya → Gemma</p>
      <p className="text-fg-muted mt-1">Semantic matches propose signals. Strong Laya decisions keep them; uncertain candidates go to Gemma for confirmation.</p>
    </div>
  );
}
