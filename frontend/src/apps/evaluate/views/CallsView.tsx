import { useEffect, useMemo, useState } from 'react';
import { useInfiniteQuery, useQuery } from '@tanstack/react-query';
import { Flag, Phone, Search } from 'lucide-react';
import {
  FAMILY_DISPLAY,
  agentLabel,
  alertRuleName,
  callQaBadge,
  categoryName,
  escalationStatusDisplay,
  formatScore,
  queryKeys,
  resultStateDisplay,
  signalFamily,
  subcategoryName,
  type CallListItem,
  type ResultState,
  type SignalAlertRuleRecord,
  type SignalTaxonomy,
} from '../api';
import { Button, Chip, EmptyState, ErrorNotice, Loading, PageHeader, SelectInput, StatusPill, TextInput, formatDateTime, formatDuration } from '../components/ui';
import { href, type CallsFilters } from '../state/router';
import type { CallsViewProps } from './types';

/** Category chips shown before "+N" in the Signals column. */
const SIGNAL_CHIPS = 2;

/**
 * `#/calls` — the paged call list (`GET /store/v1/calls`, newest first) with the derived result
 * states Store computes. Every state has a text label as well as a color. The change feed
 * invalidates `['calls']`, so the list follows Processing without a reload.
 *
 * Contact signals (contract 1.3.0, docs/ContactSignalsV2.md §10.3): a "Caller objective" column
 * (intent subcategories), a "Signals" column (the other categories, "+N"; the caller objective has
 * its own column so it is not repeated there), and Category, Subcategory and Alert
 * filters. `#/calls?signal_category=…&signal_subcategory=…&signal_alert=…` opens the list filtered,
 * which is how a metrics row links here.
 */
export default function CallsView({ client, filters: routeFilters }: CallsViewProps) {
  const [text, setText] = useState('');
  const [debounced, setDebounced] = useState('');
  const [needsReview, setNeedsReview] = useState(false);
  const [category, setCategory] = useState(routeFilters?.signal_category ?? '');
  const [subcategory, setSubcategory] = useState(routeFilters?.signal_subcategory ?? '');
  const [alert, setAlert] = useState(routeFilters?.signal_alert ?? '');

  useEffect(() => {
    const t = setTimeout(() => setDebounced(text.trim()), 300);
    return () => clearTimeout(t);
  }, [text]);

  // Names for the chips and filters. Reviewers read both (read_calls); a failure leaves IDs shown.
  const taxonomyQuery = useQuery({
    queryKey: queryKeys.signalTaxonomy,
    queryFn: ({ signal }) => client.get('/store/v1/signals/taxonomy', { signal }),
    retry: false,
  });
  const rulesQuery = useQuery({
    queryKey: queryKeys.signalAlertRules,
    queryFn: ({ signal }) => client.get('/store/v1/signals/alert-rules', { signal }),
    retry: false,
  });
  const taxonomy = taxonomyQuery.data?.current.taxonomy;
  const rules = rulesQuery.data?.items;

  const filters = {
    text: debounced || null,
    needs_review: needsReview ? true : null,
    signal_category: category || null,
    signal_subcategory: category && subcategory ? subcategory : null,
    signal_alert: alert || null,
  };
  const q = useInfiniteQuery({
    queryKey: queryKeys.calls(filters),
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      client.get('/store/v1/calls', { query: { limit: 50, page_token: pageParam, ...filters }, signal }),
    getNextPageParam: (last) => last.next_page_token,
  });
  const calls = q.data?.pages.flatMap((p) => p.items) ?? [];
  const filtered = Boolean(debounced || needsReview || category || alert);

  // Keep the address bar in step with the signal filters, so the filtered list can be shared.
  useEffect(() => {
    const f: CallsFilters = {};
    if (category) f.signal_category = category;
    if (category && subcategory) f.signal_subcategory = subcategory;
    if (alert) f.signal_alert = alert;
    const target = href({ name: 'calls', filters: f });
    if (window.location.hash !== target && window.location.hash.startsWith('#/calls')) {
      window.history.replaceState(null, '', `${window.location.pathname}${target}`);
    }
  }, [category, subcategory, alert]);

  const selectedCategory = taxonomy?.categories.find((c) => c.category_id === category);
  const categories = useMemo(() => taxonomy?.categories.filter((c) => c.active || c.category_id === category) ?? [], [taxonomy, category]);

  return (
    <div className="max-w-6xl mx-auto">
      <PageHeader
        title="Calls"
        description="Newest first. Open a call to review it in the Workbench."
        right={
          <>
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-fg-subtle absolute left-2.5 top-1/2 -translate-y-1/2" aria-hidden="true" />
              <TextInput
                type="search"
                aria-label="Filter by agent or call reference"
                placeholder="Agent or call ref"
                value={text}
                maxLength={200}
                onChange={(e) => setText(e.target.value)}
                className="pl-8 w-48 sm:w-60"
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-fg whitespace-nowrap">
              <input type="checkbox" checked={needsReview} onChange={(e) => setNeedsReview(e.target.checked)} />
              Needs review
            </label>
          </>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-[repeat(3,minmax(0,14rem))_auto] items-center gap-2 mb-3" role="group" aria-label="Signal filters">
        <SelectInput
          aria-label="Filter by signal category"
          value={category}
         
          onChange={(e) => {
            setCategory(e.target.value);
            setSubcategory('');
          }}
        >
          <option value="">Any signal category</option>
          {categories.map((c) => (
            <option key={c.category_id} value={c.category_id}>
              {c.name}
            </option>
          ))}
          {category && !categories.some((c) => c.category_id === category) && <option value={category}>{categoryName(taxonomy, category)}</option>}
        </SelectInput>
        <SelectInput
          aria-label="Filter by signal subcategory"
          value={subcategory}
         
          disabled={!category}
          title={category ? undefined : 'Pick a category first'}
          onChange={(e) => setSubcategory(e.target.value)}
        >
          <option value="">{category ? 'Any subcategory' : 'Subcategory'}</option>
          {(selectedCategory?.subcategories ?? []).map((s) => (
            <option key={s.subcategory_id} value={s.subcategory_id}>
              {s.name}
              {s.active ? '' : ' (inactive)'}
            </option>
          ))}
          {category && <option value="other">Other</option>}
          {subcategory && subcategory !== 'other' && !selectedCategory?.subcategories.some((s) => s.subcategory_id === subcategory) && (
            <option value={subcategory}>{subcategory}</option>
          )}
        </SelectInput>
        <SelectInput aria-label="Filter by signal alert" value={alert} onChange={(e) => setAlert(e.target.value)}>
          <option value="">Any alert</option>
          {(rules ?? []).map((r) => (
            <option key={r.rule_id} value={r.rule_id}>
              {r.name}
              {r.enabled ? '' : ' (disabled)'}
            </option>
          ))}
          {alert && !(rules ?? []).some((r) => r.rule_id === alert) && <option value={alert}>{alert}</option>}
        </SelectInput>
        {(category || alert) && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              setCategory('');
              setSubcategory('');
              setAlert('');
            }}
          >
            Clear signal filters
          </Button>
        )}
      </div>

      {q.isLoading && <Loading label="Loading calls…" />}
      <ErrorNotice error={q.error} />
      {q.isSuccess && calls.length === 0 && (
        <EmptyState icon={Phone} title={filtered ? 'No calls match' : 'No calls yet'}>
          {filtered
            ? 'Try a different filter.'
            : 'Calls appear here as soon as Process registers them with Store. Upload or import audio in the Process console.'}
        </EmptyState>
      )}

      {calls.length > 0 && (
        <div className="rounded-lg border border-border bg-canvas-subtle overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-fg-muted border-b border-border">
                <th scope="col" className="font-medium px-3 py-2">Call</th>
                <th scope="col" className="font-medium px-3 py-2">Length</th>
                <th scope="col" className="font-medium px-3 py-2">QA</th>
                <th scope="col" className="font-medium px-3 py-2">Transcript</th>
                <th scope="col" className="font-medium px-3 py-2">Summary</th>
                <th scope="col" className="font-medium px-3 py-2">Caller objective</th>
                <th scope="col" className="font-medium px-3 py-2">Signals</th>
                <th scope="col" className="font-medium px-3 py-2">Review</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-muted">
              {calls.map((c) => (
                <CallRow key={c.call_id} call={c} taxonomy={taxonomy} rules={rules} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {q.hasNextPage && (
        <div className="mt-3 flex justify-center">
          <Button busy={q.isFetchingNextPage} onClick={() => void q.fetchNextPage()}>
            Load more
          </Button>
        </div>
      )}
    </div>
  );
}

function StatePill({ state }: { state: ResultState }) {
  const d = resultStateDisplay(state);
  return (
    <StatusPill tone={d.tone} title={d.description}>
      {d.label}
    </StatusPill>
  );
}

/** "Analyzing" while the group is pending, "Refreshing" while stale (§10.3); null otherwise. */
function SignalsStatePill({ state }: { state: ResultState | undefined }) {
  if (state === 'pending') return <StatusPill tone="blue" title="Contact signals are being analyzed">Analyzing</StatusPill>;
  if (state === 'stale') return <StatusPill tone="magenta" title="An update is under way; the previous signals are shown">Refreshing</StatusPill>;
  if (state === 'failed') return <StatusPill tone="red" title="Contact signals stopped without a result">Needs attention</StatusPill>;
  return null;
}

function CallRow({ call: c, taxonomy, rules }: { call: CallListItem; taxonomy: SignalTaxonomy | undefined; rules: SignalAlertRuleRecord[] | undefined }) {
  const badge = callQaBadge(c);
  const link = href({ name: 'workbench', callId: c.call_id });
  const signalsState = c.contact_signals_state;
  const needs = c.caller_needs ?? [];
  // The caller objective (intent) has its own column; the Signals column shows the rest.
  const categories = (c.signal_categories ?? []).filter((id) => id !== 'intent');
  // The row's title: what the caller wanted (the first caller-objective subcategory, once the names
  // have loaded), which tells one agent's calls apart; the agent label is the fallback and
  // otherwise the secondary line.
  const firstNeed = needs.find((id) => id !== 'other');
  const objective = taxonomy && firstNeed ? subcategoryName(taxonomy, 'intent', firstNeed) : null;
  const agent = agentLabel(c.agent_id, c.agent_display_name, c.agent_extension);
  const review = c.review_status ? escalationStatusDisplay(c.review_status) : null;
  const alerts = c.signal_alerts ?? [];
  return (
    <tr className="hover:bg-canvas transition-colors" data-call-row={c.call_id}>
      <td className="px-3 py-2">
        <a href={link} className="flex flex-col min-w-[10rem] rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue">
          {objective ? (
            <>
              <span className="text-fg font-medium">{objective}</span>
              <span className="text-xs text-fg" data-testid="call-agent">
                {agent}
              </span>
            </>
          ) : (
            <span className="text-fg font-medium" data-testid="call-agent">
              {agent}
            </span>
          )}
          <span className="text-xs text-fg-muted">
            {formatDateTime(c.created_at)}
            {c.rubric_id && ` · ${c.rubric_id}`}
          </span>
        </a>
      </td>
      <td className="px-3 py-2 text-fg-muted tabular-nums">{formatDuration(c.duration_seconds)}</td>
      <td className="px-3 py-2">
        <span className="flex items-center gap-2">
          <StatusPill tone={badge.tone} title={badge.description}>
            {badge.label}
          </StatusPill>
          {badge.score !== null && (
            <span className="tabular-nums text-fg" aria-label={`${c.requires_human_review && !c.critical_failure ? 'Provisional score' : 'Score'} ${formatScore(badge.score)} out of 100`}>
              {formatScore(badge.score)}
            </span>
          )}
        </span>
      </td>
      <td className="px-3 py-2">
        <StatePill state={c.transcript_state} />
      </td>
      <td className="px-3 py-2">
        <StatePill state={c.summary_state} />
      </td>
      <td className="px-3 py-2" data-column="caller-need">
        <span className="flex flex-wrap items-center gap-1 max-w-[14rem]">
          <SignalsStatePill state={signalsState} />
          {needs.map((id) => (
            <Chip key={id} tone="blue" title="Caller objective subcategory">
              {subcategoryName(taxonomy, 'intent', id)}
            </Chip>
          ))}
          {needs.length === 0 && signalsState !== 'pending' && signalsState !== 'failed' && <span className="text-xs text-fg-subtle">—</span>}
        </span>
      </td>
      <td className="px-3 py-2" data-column="signals">
        <span className="flex flex-wrap items-center gap-1 max-w-[16rem]">
          <SignalsStatePill state={signalsState} />
          {categories.slice(0, SIGNAL_CHIPS).map((id) => (
            <Chip key={id} tone={FAMILY_DISPLAY[signalFamily(id)].tone} title="Signal category">
              {categoryName(taxonomy, id)}
            </Chip>
          ))}
          {categories.length > SIGNAL_CHIPS && (
            <span className="text-xs text-fg-muted" title={categories.slice(SIGNAL_CHIPS).map((id) => categoryName(taxonomy, id)).join(', ')}>
              +{categories.length - SIGNAL_CHIPS}
            </span>
          )}
          {alerts.map((id) => (
            <Chip key={`alert-${id}`} tone="yellow" icon={Flag} title="Alert rule matching this call">
              {alertRuleName(rules, id)}
            </Chip>
          ))}
          {categories.length === 0 && alerts.length === 0 && signalsState !== 'pending' && signalsState !== 'stale' && signalsState !== 'failed' && (
            <span className="text-xs text-fg-subtle">{signalsState === 'disabled' ? 'Not run' : '—'}</span>
          )}
        </span>
      </td>
      <td className="px-3 py-2">
        {c.requires_human_review && (!c.review_status || c.review_status === 'PENDING') ? (
          <StatusPill tone="yellow" title="Escalated: waiting for a supervisor">
            <Flag className="w-3 h-3" aria-hidden="true" />
            Needs review
          </StatusPill>
        ) : review && c.review_status !== 'NONE' ? (
          <StatusPill tone={review.tone} title={review.description}>
            {review.label === 'Unknown' ? c.review_status : review.label}
          </StatusPill>
        ) : (
          <span className="text-xs text-fg-subtle">—</span>
        )}
      </td>
    </tr>
  );
}
