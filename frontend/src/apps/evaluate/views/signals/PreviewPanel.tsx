// "Test on recent calls" (docs/ContactSignalsV2.md §10.1): run the unsaved taxonomy on a few
// settled calls into draft slots and show, per call, a diff scoped to the edited category. Nothing
// reaches the calls' results, queue or metrics. `PreviewCallResult` is shared with the Workbench's
// "Compare with v2" (§14 shadow mode).

import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { FlaskConical, MapPin } from 'lucide-react';
import {
  agentLabel,
  categoryById,
  fieldChipText,
  hitCategoryId,
  hitCategoryName,
  hitPathLabel,
  hitWhy,
  queryKeys,
  sendIdempotent,
  signalFamily,
  FAMILY_DISPLAY,
  useIdempotencyKey,
  type ContactSignalView,
  type ContractInfo,
  type SignalPreview,
  type SignalPreviewCall,
  type SignalTaxonomy,
  type StoreClient,
} from '../../api';
import { Button, Checkbox, Chip, ErrorNotice, Loading, Notice, StatusPill, formatDateTime } from '../../components/ui';
import { href } from '../../state/router';

/** The span key a hit ID ends with (`t<turn>b<block>`, §6.3): preview IDs and published hit IDs
 * of the same span share it, so a diff can show both sides. */
function spanSuffix(id: string): string | null {
  return /\.(t\d+b\d+)$/.exec(id)?.[1] ?? null;
}

function categoryOfHitId(id: string): string {
  return id.split('.')[0] ?? id;
}

export function findHit(signals: ContactSignalView[] | undefined, id: string): ContactSignalView | undefined {
  if (!signals) return undefined;
  const exact = signals.find((s) => s.id === id);
  if (exact) return exact;
  const suffix = spanSuffix(id);
  const cat = categoryOfHitId(id);
  return suffix ? signals.find((s) => hitCategoryId(s) === cat && spanSuffix(s.id) === suffix) : undefined;
}

/** "Some option text was too long for the classifier: shorten <name>" names (§10.1). The contract
 * lists category or subcategory IDs; a dotted `category.subcategory` is read as a path. */
function trimmedNames(ids: string[], taxonomy: SignalTaxonomy | undefined): string[] {
  return ids.map((id) => {
    const [a, b] = id.split('.');
    if (b) {
      const c = categoryById(taxonomy, a);
      return c?.subcategories.find((s) => s.subcategory_id === b)?.name ?? id;
    }
    const c = categoryById(taxonomy, a);
    if (c) return c.name;
    for (const cat of taxonomy?.categories ?? []) {
      const s = cat.subcategories.find((x) => x.subcategory_id === a);
      if (s) return `${cat.name} › ${s.name}`;
    }
    return id;
  });
}

export function PreviewPanel({
  client,
  contract,
  draft,
  scopeCategoryId,
  onClose,
}: {
  client: StoreClient;
  contract: ContractInfo;
  draft: SignalTaxonomy;
  scopeCategoryId: string | undefined;
  onClose(): void;
}) {
  const max = contract.parameters.signal_preview_max_calls;
  const recentQuery = useQuery({
    queryKey: queryKeys.calls({ purpose: 'signal-preview-candidates' }),
    queryFn: ({ signal }) => client.get('/store/v1/calls', { query: { limit: 50 }, signal }),
  });
  // Settled calls whose signals are published: a preview diffs against the published result.
  const candidates = useMemo(
    () => (recentQuery.data?.items ?? []).filter((c) => c.contact_signals_state === 'available' || c.contact_signals_state === 'partial').slice(0, 20),
    [recentQuery.data],
  );
  const [picked, setPicked] = useState<string[] | null>(null);
  const selected = picked ?? candidates.slice(0, Math.min(5, max)).map((c) => c.call_id);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const idempotency = useIdempotencyKey();

  const previewQuery = useQuery({
    queryKey: queryKeys.signalPreview(previewId ?? 'none'),
    queryFn: ({ signal }) => client.get('/store/v1/signals/previews/{preview_id}', { path: { preview_id: previewId! }, signal }),
    enabled: previewId !== null,
    // The change feed re-reads on reanalysis_request events; this is the backstop while any call is pending.
    refetchInterval: (q) => ((q.state.data as SignalPreview | undefined)?.calls.some((c) => c.state === 'pending') ? 3000 : false),
  });

  async function run() {
    setBusy(true);
    setError(null);
    const body = { taxonomy: draft, call_ids: selected };
    try {
      const preview = await sendIdempotent(idempotency, body, (key) =>
        client.post('/store/v1/signals/previews', { headers: { 'Idempotency-Key': key }, body }),
      );
      setPreviewId(preview.id);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  const toggle = (callId: string, on: boolean) => {
    const next = on ? [...selected, callId] : selected.filter((id) => id !== callId);
    setPicked(next);
  };

  const preview = previewQuery.data;
  const scopeName = scopeCategoryId ? (categoryById(draft, scopeCategoryId)?.name ?? scopeCategoryId) : null;

  return (
    <section className="rounded-lg border border-primer-blueBorder bg-canvas-subtle p-4 space-y-3" aria-label="Test on recent calls" data-testid="signals-preview">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-fg flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-fg-subtle" aria-hidden="true" />
          Test on recent calls
        </h3>
        <Button size="sm" variant="ghost" onClick={onClose}>
          Close test
        </Button>
      </div>
      <p className="text-xs text-fg-muted">
        Runs your unsaved taxonomy on up to {max} settled calls, beside their published signals. Nothing changes on the calls.
        {scopeName && ` Changes are shown for ${scopeName}.`}
      </p>

      {recentQuery.isLoading && <Loading label="Finding recent settled calls…" />}
      <ErrorNotice error={recentQuery.error} />
      {recentQuery.isSuccess && candidates.length === 0 && (
        <Notice>No settled calls with published signals yet. Test again once calls finish processing.</Notice>
      )}
      {candidates.length > 0 && (
        <fieldset className="space-y-1">
          <legend className="text-xs font-medium text-fg-muted mb-1">
            Calls ({selected.length} of at most {max})
          </legend>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
            {candidates.map((c) => {
              const on = selected.includes(c.call_id);
              return (
                <Checkbox
                  key={c.call_id}
                  checked={on}
                  disabled={!on && selected.length >= max}
                  onChange={(e) => toggle(c.call_id, e.target.checked)}
                  label={`${agentLabel(c.agent_id, c.agent_display_name, c.agent_extension)} · ${formatDateTime(c.created_at)}`}
                />
              );
            })}
          </div>
        </fieldset>
      )}
      <div className="flex items-center gap-2">
        <Button variant="primary" icon={FlaskConical} busy={busy} disabled={selected.length === 0} onClick={() => void run()}>
          Run preview
        </Button>
        {preview && <span className="text-xs text-fg-muted">Test started {formatDateTime(preview.created_at)}</span>}
      </div>
      <ErrorNotice error={error} />
      <ErrorNotice error={previewQuery.error} />

      {preview && (
        <PreviewResults
          client={client}
          preview={preview}
          taxonomy={draft}
          scopeCategoryId={scopeCategoryId}
          callName={(id) => {
            const c = recentQuery.data?.items.find((x) => x.call_id === id);
            return c ? `${agentLabel(c.agent_id, c.agent_display_name, c.agent_extension)} · ${formatDateTime(c.created_at)}` : id;
          }}
        />
      )}
    </section>
  );
}

export function PreviewResults({
  client,
  preview,
  taxonomy,
  scopeCategoryId,
  callName,
  onlyCallId,
}: {
  client: StoreClient;
  preview: SignalPreview;
  taxonomy: SignalTaxonomy | undefined;
  scopeCategoryId?: string;
  callName(id: string): string;
  onlyCallId?: string;
}) {
  const trimmed = trimmedNames(preview.options_trimmed, taxonomy);
  const calls = onlyCallId ? preview.calls.filter((c) => c.call_id === onlyCallId) : preview.calls;
  return (
    <div className="space-y-2">
      {trimmed.length > 0 && (
        <Notice tone="yellow">Some option text was too long for the classifier: shorten {trimmed.join(', ')}.</Notice>
      )}
      {calls.map((c) => (
        <PreviewCallResult key={c.request_id} client={client} call={c} scopeCategoryId={scopeCategoryId} name={callName(c.call_id)} />
      ))}
    </div>
  );
}

function HitLine({ hit, callId, verb }: { hit: ContactSignalView | undefined; callId: string; verb: string }) {
  if (!hit) return <li className="text-xs text-fg-muted">{verb}: a signal this page can't show</li>;
  const fam = FAMILY_DISPLAY[signalFamily(hitCategoryId(hit))];
  const chips = hit.fields.map(fieldChipText).filter((t): t is string => t !== null);
  return (
    <li className="rounded border border-border-muted bg-canvas p-2 space-y-1">
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="font-medium text-fg">
          {verb}: {hitPathLabel(hit)}
        </span>
        <Chip tone={fam.tone}>{fam.label}</Chip>
        {chips.map((t) => (
          <Chip key={t}>{t}</Chip>
        ))}
      </div>
      <p className="text-xs italic text-fg-muted">&ldquo;{hit.quote}&rdquo;</p>
      <a
        href={href({ name: 'workbench', callId, turn: hit.turn_id ?? undefined })}
        className="inline-flex items-center gap-1 text-xs text-primer-blueFg hover:underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
      >
        <MapPin className="w-3 h-3" aria-hidden="true" />
        Open in Workbench at this turn
      </a>
    </li>
  );
}

export function PreviewCallResult({
  client,
  call,
  scopeCategoryId,
  name,
}: {
  client: StoreClient;
  call: SignalPreviewCall;
  scopeCategoryId?: string;
  name: string;
}) {
  const diff = call.diff;
  const inScope = (id: string) => !scopeCategoryId || categoryOfHitId(id) === scopeCategoryId;
  const added = diff?.added.filter(inScope) ?? [];
  const removed = diff?.removed.filter(inScope) ?? [];
  const relabelled = diff?.relabelled.filter(inScope) ?? [];
  const fieldsChanged = diff?.fields_changed.filter(inScope) ?? [];
  const segmentsChanged = diff?.segments_changed?.filter(inScope) ?? [];
  const builtinChanged = diff?.builtin_changed ?? [];
  const needsPublished = removed.length > 0 || relabelled.length > 0 || builtinChanged.length > 0;
  const publishedQuery = useQuery({
    queryKey: [...queryKeys.call(call.call_id), 'contact-signals'],
    queryFn: ({ signal }) => client.get('/store/v1/calls/{call_id}/contact-signals', { path: { call_id: call.call_id }, signal }),
    enabled: call.state === 'available' && needsPublished,
    retry: false,
  });
  const newSignals = call.result?.signals;
  const oldSignals = publishedQuery.data?.signals;
  const nothing = added.length + removed.length + relabelled.length + fieldsChanged.length + segmentsChanged.length + builtinChanged.length === 0;

  return (
    <div className="rounded-md border border-border-muted bg-canvas-subtle p-3 space-y-2" data-preview-call={call.call_id}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <a href={href({ name: 'workbench', callId: call.call_id })} className="text-sm font-medium text-fg hover:underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue">
          {name}
        </a>
        {call.state === 'pending' ? (
          <StatusPill tone="blue">Analyzing</StatusPill>
        ) : call.state === 'failed' ? (
          <StatusPill tone="red" title="The test run for this call stopped without a result">
            Failed{call.failure_code ? ` (${call.failure_code})` : ''}
          </StatusPill>
        ) : nothing ? (
          <StatusPill tone="neutral">No change</StatusPill>
        ) : (
          <StatusPill tone="magenta">Changed</StatusPill>
        )}
      </div>
      {call.state === 'available' && newSignals && <RulesCounts signals={newSignals} scopeCategoryId={scopeCategoryId} />}
      {call.state === 'available' && !nothing && (
        <ul className="space-y-1.5">
          {added.map((id) => (
            <HitLine key={`a-${id}`} verb="Added" hit={findHit(newSignals, id)} callId={call.call_id} />
          ))}
          {removed.map((id) => (
            <HitLine key={`r-${id}`} verb="Removed" hit={findHit(oldSignals, id)} callId={call.call_id} />
          ))}
          {relabelled.map((id) => {
            const before = findHit(oldSignals, id);
            const after = findHit(newSignals, id);
            // A hit with no subcategory (or 'Other') that now gets one reads as the subcategory
            // being added (§15 step 3); a move between two subcategories is a relabel.
            const hadSubcategory = before !== undefined && before.subcategory_id != null && before.subcategory_id !== 'other';
            const verb = before === undefined ? 'Relabelled' : hadSubcategory ? `Relabelled from ${hitPathLabel(before)}` : 'Added';
            return <HitLine key={`l-${id}`} verb={verb} hit={after} callId={call.call_id} />;
          })}
          {fieldsChanged.map((id) => (
            <HitLine key={`f-${id}`} verb="Fields filled" hit={findHit(newSignals, id)} callId={call.call_id} />
          ))}
          {segmentsChanged.map((id) => (
            <HitLine key={`s-${id}`} verb="Segments changed" hit={findHit(newSignals, id)} callId={call.call_id} />
          ))}
          {builtinChanged.length > 0 && (
            <li className="space-y-1">
              <Notice tone="yellow">This change also moves built-in signals: new options share the classifier's scores with the built-ins.</Notice>
              <ul className="space-y-1">
                {builtinChanged.map((id) => {
                  const inNew = findHit(newSignals, id);
                  return <HitLine key={`b-${id}`} verb={inNew ? 'Built-in added' : 'Built-in removed'} hit={inNew ?? findHit(oldSignals, id)} callId={call.call_id} />;
                })}
              </ul>
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

/**
 * Per-category counts of a test run in which the rules engine ran (contract 1.4.0: every hit then
 * carries `why`): how many signals each category found, how many the rules decided and how many
 * Gemma double-checked. Shown beside the diff, which only lists what changed.
 */
function RulesCounts({ signals, scopeCategoryId }: { signals: ContactSignalView[]; scopeCategoryId?: string }) {
  if (!signals.some((s) => hitWhy(s) !== null)) return null;
  const by = new Map<string, { name: string; total: number; rules: number; checked: number }>();
  for (const sig of signals) {
    const id = hitCategoryId(sig);
    const entry = by.get(id) ?? { name: hitCategoryName(sig), total: 0, rules: 0, checked: 0 };
    const why = hitWhy(sig);
    entry.total += 1;
    if (why?.category_source === 'rules') entry.rules += 1;
    if (why?.check === 'confirmed') entry.checked += 1;
    by.set(id, entry);
  }
  const rows = [...by.entries()].sort(([a], [b]) => (a === scopeCategoryId ? -1 : b === scopeCategoryId ? 1 : 0));
  return (
    <div className="space-y-1" data-testid="signals-preview-counts">
      <p className="text-xs font-medium text-fg-muted">Signals found in this test, by category</p>
      <ul className="flex flex-wrap gap-1.5">
        {rows.map(([id, r]) => (
          <li key={id} data-category={id}>
            <Chip tone={id === scopeCategoryId ? 'blue' : 'neutral'}>
              {r.name} {r.total}
              {r.rules > 0 ? ` · ${r.rules === r.total ? 'all' : r.rules} by rules` : ' · by Gemma'}
              {r.checked > 0 ? ` · ${r.checked} double-checked` : ''}
            </Chip>
          </li>
        ))}
      </ul>
    </div>
  );
}
