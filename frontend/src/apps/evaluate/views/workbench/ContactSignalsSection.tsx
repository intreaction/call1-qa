// The Workbench's contact-signals card (docs/ContactSignalsV2.md §10.2): spans with category and
// subcategory chips, extracted fields, reviewer feedback (confirm or dismiss the category; confirm
// or correct the subcategory), multi-segment signals (decision 25: "×N segments" and the part quotes,
// §6.5), a category filter once a call has more than FILTER_MIN signals (real Gemma output runs to
// 40-odd spans a call), the honest per-state texts, "Update signals" when the taxonomy moved
// on, and "Compare with v2" for admins in shadow mode. Signals are unscored: nothing here changes a
// score or the review version.

import { useId, useState } from 'react';
import { useQuery, useQueryClient, type UseQueryResult } from '@tanstack/react-query';
import { Bell, CheckCircle2, ChevronDown, ChevronRight, GitCompare, Layers, RefreshCw, ShieldAlert, Sparkles } from 'lucide-react';
import {
  FAMILY_DISPLAY,
  FIELD_STATUS_TEXT,
  SPEAKER_LABEL,
  RULE_TYPE_LABEL,
  categoryById,
  decisionPhrase,
  describeError,
  fmt2,
  hitWhy,
  fieldChipText,
  hitCategoryId,
  hitCategoryName,
  hitPartIds,
  hitParts,
  hitSegmentCount,
  hitSegmentsText,
  hitSpanEnd,
  hitSubcategoryName,
  isVersionConflict,
  partialStageTexts,
  queryKeys,
  resultStateDisplay,
  sendIdempotent,
  signalFamily,
  signalsWaitOnMasking,
  sourceLabel,
  subcategoryName,
  taxonomyOutdatedText,
  whySummary,
  useIdempotencyKey,
  type ContactSignalView,
  type ContactSignalsView,
  type ResultGroup,
  type SignalHitFeedback,
  type SignalHitWhy,
  type SignalTaxonomy,
  type StoreClient,
} from '../../api';
import { Button, Card, Chip, EmptyState, ErrorNotice, Loading, Notice, SelectInput, StatusPill } from '../../components/ui';
import { failureReason } from '../../components/failureText';
import type { SignedInSession } from '../../state/app';
import { PreviewResults } from '../signals/PreviewPanel';

/** Above this many signals the list gets a category filter ("All 42", "Caller objective 6", …). */
const FILTER_MIN = 8;

/** "Scored 77 transcript segments" (plus the unattributed ones skipped, when any). */
function segmentsText(seg: { scored_segments: number; skipped_unattributed: number }): string {
  const plural = (n: number) => (n === 1 ? '' : 's');
  const skipped =
    seg.skipped_unattributed > 0
      ? ` · skipped ${seg.skipped_unattributed} where the speaker was unclear`
      : '';
  return `Scored ${seg.scored_segments} transcript segment${plural(seg.scored_segments)}${skipped}`;
}

function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '0:00';
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

export function ContactSignalsSection({
  client,
  session,
  callId,
  group,
  query,
  transcriptWithheld,
  transcriptFailed = false,
  onJump,
  onWrote,
}: {
  client: StoreClient;
  session: SignedInSession;
  callId: string;
  group: ResultGroup | undefined;
  query: UseQueryResult<ContactSignalsView>;
  transcriptWithheld: boolean;
  /** The transcript group failed (e.g. the recording was rejected): there is nothing to read. */
  transcriptFailed?: boolean;
  onJump: (turnId: number | null | undefined, timeHint?: number | null) => void;
  onWrote: () => void;
}) {
  const view = query.data;
  const state = group?.state ?? 'disabled';
  const d = resultStateDisplay(state);
  const pill = (
    <>
      {view && <StatusPill tone={view.pipeline === 'v2' ? 'blue' : 'neutral'}>{view.pipeline === 'v2' ? 'v2 signals' : 'v1 signals'}</StatusPill>}
      <StatusPill tone={d.tone} title={`${d.description}${group?.failure_code ? ` (${failureReason(group.failure_code)})` : group?.partial_reason ? ` — ${group.partial_reason}` : ''}`}>
        {state === 'stale' ? 'Refreshing' : d.label}
      </StatusPill>
    </>
  );

  // Names for correction options and "categories checked": the taxonomy version the result was
  // scored with (v2), else the current one.
  const scoredVersion = view?.taxonomy?.version ?? null;
  const currentQuery = useQuery({
    queryKey: queryKeys.signalTaxonomy,
    queryFn: ({ signal }) => client.get('/store/v1/signals/taxonomy', { signal }),
    enabled: view !== undefined,
    retry: false,
  });
  const scoredQuery = useQuery({
    queryKey: [...queryKeys.signalTaxonomyVersions, scoredVersion],
    queryFn: ({ signal }) => client.get('/store/v1/signals/taxonomy/versions/{version}', { path: { version: scoredVersion! }, signal }),
    enabled: scoredVersion !== null && scoredVersion !== currentQuery.data?.current.version,
    retry: false,
  });
  const scoredTaxonomy: SignalTaxonomy | undefined =
    scoredVersion !== null && scoredVersion !== currentQuery.data?.current.version ? scoredQuery.data?.taxonomy : currentQuery.data?.current.taxonomy;

  let body: React.ReactNode;
  if (state === 'disabled') {
    body = <EmptyState title="Contact signals were not run for this call" />;
  } else if (state === 'pending') {
    body = <EmptyState title="Analyzing">Signals appear here when the category and subcategory stages finish.</EmptyState>;
  } else if (state === 'failed') {
    body = signalsWaitOnMasking(group, transcriptWithheld) ? (
      <EmptyState icon={ShieldAlert} title="Needs attention: signals wait for PII masking to be retried.">
        No signal stage runs on unmasked text. An admin can retry PII masking for this call.
      </EmptyState>
    ) : (
      <EmptyState title="Needs attention">
        {transcriptFailed
          ? 'There is no transcript to find contact signals in.'
          : `Contact signals stopped without a result${group?.failure_code ? ` (${failureReason(group.failure_code)})` : ''}. An admin can retry it.`}
      </EmptyState>
    );
  } else if (query.isLoading) {
    body = <Loading label="Loading contact signals…" />;
  } else if (query.isError) {
    body = <ErrorNotice error={query.error} />;
  } else if (!view) {
    body = <EmptyState title="No contact signals yet" />;
  } else {
    body = (
      <SignalsBody
        client={client}
        session={session}
        callId={callId}
        view={view}
        stale={state === 'stale'}
        scoredTaxonomy={scoredTaxonomy}
        onJump={onJump}
        onWrote={onWrote}
      />
    );
  }

  return (
    <Card title="Contact signals" subtitle="Unscored: signals never change the score" right={pill}>
      <div data-testid="contact-signals" data-state={state} data-pipeline={view?.pipeline}>
        {body}
      </div>
    </Card>
  );
}

function SignalsBody({
  client,
  session,
  callId,
  view,
  stale,
  scoredTaxonomy,
  onJump,
  onWrote,
}: {
  client: StoreClient;
  session: SignedInSession;
  callId: string;
  view: ContactSignalsView;
  stale: boolean;
  scoredTaxonomy: SignalTaxonomy | undefined;
  onJump: (turnId: number | null | undefined, timeHint?: number | null) => void;
  onWrote: () => void;
}) {
  const outdated = taxonomyOutdatedText(view.taxonomy_status);
  const partial = view.completeness === 'partial' ? partialStageTexts(view) : [];
  const alertsByHit = new Map<string, string[]>();
  for (const a of view.alerts) for (const id of a.hit_ids) alertsByHit.set(id, [...(alertsByHit.get(id) ?? []), a.name]);
  const [compare, setCompare] = useState(false);
  const [only, setOnly] = useState<string | null>(null);

  const seg = view.segmentation;
  // Category filter (a dense call): each category present, in order of first appearance.
  const present = new Map<string, { name: string; count: number }>();
  for (const sig of view.signals) {
    const id = hitCategoryId(sig);
    const entry = present.get(id);
    if (entry) entry.count += 1;
    else present.set(id, { name: hitCategoryName(sig), count: 1 });
  }
  const filtering = view.signals.length > FILTER_MIN && present.size > 1;
  const active = filtering && only !== null && present.has(only) ? only : null;
  const shown = active === null ? view.signals : view.signals.filter((sig) => hitCategoryId(sig) === active);
  const checked = scoredTaxonomy?.categories.filter((c) => c.active).map((c) => c.name) ?? [];

  return (
    <div className="space-y-2">
      {stale && <Notice tone="magenta">Refreshing: an update is under way. The previous result is shown until it finishes.</Notice>}
      {view.text_withheld && (
        <Notice tone="yellow" icon={ShieldAlert}>
          Text withheld until PII masking finishes. Categories, times and chips are shown; quotes and field text are not.
        </Notice>
      )}
      {outdated && <UpdateSignals client={client} session={session} callId={callId} text={outdated} onWrote={onWrote} />}
      {partial.length > 0 && (
        <Notice tone="yellow">
          <ul className="space-y-0.5">
            {partial.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </Notice>
      )}
      {view.pipeline_note && <Notice>{view.pipeline_note}</Notice>}
      {view.alerts.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5" aria-label="Alerts on this call">
          <span className="text-xs text-fg-muted">Alerts:</span>
          {view.alerts.map((a) => (
            <Chip key={a.rule_id} tone="yellow" icon={Bell}>
              {a.name}
            </Chip>
          ))}
        </div>
      )}
      {view.pipeline === 'v1' && view.comparison_preview_id && session.can('manage_signals') && (
        <div className="space-y-2">
          <Button size="sm" icon={GitCompare} aria-expanded={compare} onClick={() => setCompare((c) => !c)}>
            {compare ? 'Hide v2 comparison' : 'Compare with v2'}
          </Button>
          {compare && <CompareWithV2 client={client} previewId={view.comparison_preview_id} callId={callId} taxonomy={scoredTaxonomy} />}
        </div>
      )}

      {view.signals.length === 0 ? (
        view.pipeline === 'v2' ? (
          <EmptyState title="No signals found">
            {checked.length > 0 && <span className="block">Checked: {checked.join(', ')}.</span>}
            {seg && (
              <span className="block">{segmentsText(seg)}</span>
            )}
          </EmptyState>
        ) : (
          <EmptyState title="No signals found">v1 signals: none of the built-in categories were detected.</EmptyState>
        )
      ) : (
        <>
          {view.pipeline === 'v2' && seg && (
            <p className="text-xs text-fg-subtle">{segmentsText(seg)}</p>
          )}
          {filtering && (
            <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Show signals by category" data-testid="signals-filter">
              <span className="text-xs text-fg-muted">Show:</span>
              <Button size="sm" variant={active === null ? 'primary' : 'secondary'} aria-pressed={active === null} onClick={() => setOnly(null)}>
                All {view.signals.length}
              </Button>
              {[...present].map(([id, { name, count }]) => (
                <Button key={id} size="sm" variant={active === id ? 'primary' : 'secondary'} aria-pressed={active === id} onClick={() => setOnly(active === id ? null : id)}>
                  {name} {count}
                </Button>
              ))}
            </div>
          )}
          <ul className="space-y-2" aria-label="Signals">
            {shown.map((sig) => (
              <SignalRow
                key={sig.id}
                client={client}
                session={session}
                callId={callId}
                sig={sig}
                feedback={view.feedback.find((f) => f.hit_id === sig.id)}
                earlierFeedback={hitPartIds(sig)
                  .map((id) => view.feedback.find((f) => f.hit_id === id))
                  .find((f): f is SignalHitFeedback => f !== undefined && f.category_verdict !== null)}
                alerts={alertsByHit.get(sig.id) ?? []}
                taxonomy={scoredTaxonomy}
                withheld={view.text_withheld}
                onJump={onJump}
                onWrote={onWrote}
              />
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function SignalRow({
  client,
  session,
  callId,
  sig,
  feedback,
  earlierFeedback,
  alerts,
  taxonomy,
  withheld,
  onJump,
  onWrote,
}: {
  client: StoreClient;
  session: SignedInSession;
  callId: string;
  sig: ContactSignalView;
  feedback: SignalHitFeedback | undefined;
  /** A category verdict saved on one of this merged hit's parts while it was a separate hit. */
  earlierFeedback?: SignalHitFeedback;
  alerts: string[];
  taxonomy: SignalTaxonomy | undefined;
  withheld: boolean;
  onJump: (turnId: number | null | undefined, timeHint?: number | null) => void;
  onWrote: () => void;
}) {
  const qc = useQueryClient();
  const fam = FAMILY_DISPLAY[signalFamily(hitCategoryId(sig))];
  const sub = hitSubcategoryName(sig);
  const chips = sig.fields.map((f) => ({ f, text: fieldChipText(f) })).filter((x): x is { f: typeof x.f; text: string } => x.text !== null);
  const canJudge = session.can('override_verdict');
  // The subcategory verdict counts only for the subcategory it judged (§6.3).
  const subVerdictCurrent =
    feedback?.subcategory_verdict != null && feedback.subcategory_id === sig.subcategory_id && feedback.subcategory_digest === sig.subcategory_digest;
  const subVerdictDetached = feedback?.subcategory_verdict != null && !subVerdictCurrent;
  const [correcting, setCorrecting] = useState(false);
  const [correction, setCorrection] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [showParts, setShowParts] = useState(false);
  const partsId = useId();
  const parts = hitParts(sig);
  const segments = hitSegmentsText(sig);
  const spanEnd = hitSpanEnd(sig);

  const category = categoryById(taxonomy, hitCategoryId(sig));
  const options = [
    ...(category?.subcategories.filter((s) => s.active && s.subcategory_id !== sig.subcategory_id).map((s) => ({ id: s.subcategory_id, name: s.name })) ?? []),
    ...(sig.subcategory_id !== 'other' ? [{ id: 'other', name: 'Other' }] : []),
  ];

  async function send(update: { category_verdict?: 'confirmed' | 'dismissed' | null; subcategory_verdict?: 'confirmed' | 'corrected' | null; corrected_subcategory_id?: string | null }) {
    setBusy(true);
    setError(null);
    const keepSub = subVerdictCurrent ? { subcategory_verdict: feedback?.subcategory_verdict ?? null, corrected_subcategory_id: feedback?.corrected_subcategory_id ?? null } : {};
    try {
      await client.put('/store/v1/calls/{call_id}/signal-hits/{hit_id}/feedback', {
        path: { call_id: callId, hit_id: sig.id },
        body: {
          category_verdict: feedback?.category_verdict ?? null,
          ...keepSub,
          ...update,
          expected_feedback_version: feedback?.feedback_version ?? 0,
        },
      });
      setCorrecting(false);
      void qc.invalidateQueries({ queryKey: [...queryKeys.call(callId), 'contact-signals'] });
      onWrote();
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) void qc.invalidateQueries({ queryKey: [...queryKeys.call(callId), 'contact-signals'] });
    } finally {
      setBusy(false);
    }
  }

  const categoryVerdict = feedback?.category_verdict ?? null;
  const correctedName = feedback?.corrected_subcategory_id
    ? feedback.corrected_subcategory_id === 'other'
      ? 'Other'
      : (category?.subcategories.find((s) => s.subcategory_id === feedback.corrected_subcategory_id)?.name ?? feedback.corrected_subcategory_id)
    : null;

  return (
    <li
      className="rounded-md border border-border-muted bg-canvas text-sm"
      data-signal-hit={sig.id}
      data-family={signalFamily(hitCategoryId(sig))}
      data-segments={hitSegmentCount(sig)}
    >
      <button
        type="button"
        onClick={() => onJump(sig.turn_id, sig.start)}
        aria-label={`${hitCategoryName(sig)}${sub ? ` › ${sub}` : ''} at ${clock(sig.start)}${segments ? `, across ${hitSegmentCount(sig)} segments` : ''}: jump to the turn`}
        className="w-full text-left p-2.5 rounded-t-md hover:bg-canvas-inset transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primer-blue"
      >
        <span className="flex flex-wrap items-center gap-1.5 mb-1">
          <Chip tone={fam.tone} title="Signal category">
            {hitCategoryName(sig)}
          </Chip>
          {sub && (
            <>
              <span className="text-fg-subtle text-xs" aria-hidden="true">
                ›
              </span>
              <Chip tone="neutral" title="Subcategory">
                {sub}
              </Chip>
            </>
          )}
          {segments && (
            <span data-testid="signal-segments">
              <Chip tone="neutral" icon={Layers} title={`One signal: the same speaker, category and subcategory across ${hitSegmentCount(sig)} consecutive segments`}>
                {segments}
              </Chip>
            </span>
          )}
          {chips.map(({ f, text }) => (
            <Chip key={f.field_id} title={FIELD_STATUS_TEXT[f.status]}>
              {text}
            </Chip>
          ))}
          {alerts.map((a) => (
            <Chip key={a} tone="yellow" icon={Bell} title="An alert rule matches this signal">
              {a}
            </Chip>
          ))}
        </span>
        <span className="flex flex-wrap items-center gap-x-2 text-xs text-fg-muted mb-1">
          <span className="tabular-nums">
            {clock(sig.start)}–{clock(spanEnd)}
          </span>
          <span>{SPEAKER_LABEL[sig.speaker] ?? sig.speaker}</span>
          {sig.quote_narrowed && <span className="text-primer-blueFg">narrowed</span>}
        </span>
        <span className="block text-fg-muted italic">{withheld && sig.quote === '[REDACTED]' ? '[REDACTED]' : <>&ldquo;{sig.quote}&rdquo;</>}</span>
      </button>

      {hitWhy(sig) && <WhyDisclosure why={hitWhy(sig)!} categoryId={hitCategoryId(sig)} taxonomy={taxonomy} />}

      {parts.length > 0 && (
        <div className="border-t border-border-muted px-2.5 py-1.5">
          <button
            type="button"
            aria-expanded={showParts}
            aria-controls={partsId}
            onClick={() => setShowParts((v) => !v)}
            className="inline-flex items-center gap-1 text-xs font-semibold text-primer-blueFg hover:underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
          >
            {showParts ? <ChevronDown className="w-3 h-3" aria-hidden="true" /> : <ChevronRight className="w-3 h-3" aria-hidden="true" />}
            {showParts ? `Hide the other segment${parts.length === 1 ? '' : 's'}` : `Show ${parts.length} more segment${parts.length === 1 ? '' : 's'}`}
          </button>
          {showParts && (
            <ol id={partsId} className="mt-1.5 space-y-1" aria-label="Segments of this signal" data-testid="signal-parts">
              {parts.map((part, i) => (
                <li key={`${part.turn_id}-${part.block}`}>
                  <button
                    type="button"
                    onClick={() => onJump(part.turn_id, part.start)}
                    aria-label={`Segment ${i + 2} of ${parts.length + 1} at ${clock(part.start)}: jump to the turn`}
                    className="w-full text-left rounded px-2 py-1 border-l-2 border-border hover:bg-canvas-inset transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primer-blue"
                  >
                    <span className="block text-[11px] text-fg-subtle tabular-nums">
                      Segment {i + 2} · {clock(part.start)}–{clock(part.end)}
                    </span>
                    <span className="block text-fg-muted italic" data-part-quote>
                      {withheld && part.quote === '[REDACTED]' ? '[REDACTED]' : <>&ldquo;{part.quote}&rdquo;</>}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}

      {canJudge && (
        <div className="border-t border-border-muted px-2.5 py-1.5 space-y-1.5">
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            <span className="text-fg-muted">Category:</span>
            <Button size="sm" variant={categoryVerdict === 'confirmed' ? 'primary' : 'ghost'} aria-pressed={categoryVerdict === 'confirmed'} busy={busy} onClick={() => void send({ category_verdict: 'confirmed' })}>
              Confirm
            </Button>
            <Button size="sm" variant={categoryVerdict === 'dismissed' ? 'danger' : 'ghost'} aria-pressed={categoryVerdict === 'dismissed'} busy={busy} onClick={() => void send({ category_verdict: 'dismissed' })}>
              Dismiss
            </Button>
            {categoryVerdict && <span className="text-fg-subtle">{categoryVerdict === 'confirmed' ? 'Confirmed' : 'Dismissed'}</span>}
            {!categoryVerdict && earlierFeedback && (
              <span className="text-fg-subtle" data-testid="signal-earlier-feedback">
                A segment was {earlierFeedback.category_verdict === 'confirmed' ? 'confirmed' : 'dismissed'} before the segments merged
              </span>
            )}
            {sig.subcategory_id && (
              <>
                <span className="text-fg-muted ml-2">Subcategory:</span>
                <Button
                  size="sm"
                  variant={subVerdictCurrent && feedback?.subcategory_verdict === 'confirmed' ? 'primary' : 'ghost'}
                  aria-pressed={subVerdictCurrent && feedback?.subcategory_verdict === 'confirmed'}
                  aria-label={`Confirm subcategory ${sub ?? ''}`.trim()}
                  busy={busy}
                  onClick={() => void send({ subcategory_verdict: 'confirmed', corrected_subcategory_id: null })}
                >
                  Confirm
                </Button>
                <Button
                  size="sm"
                  variant={subVerdictCurrent && feedback?.subcategory_verdict === 'corrected' ? 'primary' : 'ghost'}
                  aria-expanded={correcting}
                  disabled={busy || options.length === 0}
                  onClick={() => setCorrecting((c) => !c)}
                >
                  Correct
                </Button>
                {subVerdictCurrent && feedback?.subcategory_verdict === 'corrected' && correctedName && <span className="text-fg-subtle">Corrected to {correctedName}</span>}
                {subVerdictCurrent && feedback?.subcategory_verdict === 'confirmed' && <span className="text-fg-subtle">Confirmed</span>}
                {subVerdictDetached && <span className="text-fg-subtle">Judged an earlier subcategory</span>}
              </>
            )}
          </div>
          {correcting && (
            <div className="flex flex-wrap items-center gap-1.5">
              <div className="w-full sm:w-72">
                <SelectInput aria-label="Correct subcategory to" value={correction} onChange={(e) => setCorrection(e.target.value)}>
                  <option value="">Choose the right subcategory…</option>
                  {options.map((o) => (
                    <option key={o.id} value={o.id}>
                      {o.name}
                    </option>
                  ))}
                </SelectInput>
              </div>
              <Button size="sm" variant="primary" busy={busy} disabled={!correction} onClick={() => void send({ subcategory_verdict: 'corrected', corrected_subcategory_id: correction })}>
                Save correction
              </Button>
              <Button size="sm" variant="ghost" disabled={busy} onClick={() => setCorrecting(false)}>
                Cancel
              </Button>
            </div>
          )}
          <ErrorNotice error={error} />
        </div>
      )}
    </li>
  );
}

/**
 * Why a hit was found (`ContactSignalView.why`, contract 1.4.0; docs/SignalsEmbeddings.md §2.5):
 * whether the rules engine or Gemma decided the category and subcategory, Gemma's double-check,
 * and for a rules hit the score against the threshold, the similar-examples share, the phrase
 * that matched, each rule's outcome and the nearest example IDs (never their text). A hit without
 * `why` (a result with no rules run) shows nothing.
 */
function WhyDisclosure({ why, categoryId, taxonomy }: { why: SignalHitWhy; categoryId: string; taxonomy: SignalTaxonomy | undefined }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const rule = why.rule;
  const phrase = rule ? decisionPhrase(taxonomy, categoryId, rule.lexicon_phrase) : null;
  const summary = whySummary(why, phrase);
  const subName = rule ? (rule.subcategory_id ? subcategoryName(taxonomy, categoryId, rule.subcategory_id) : 'Other') : null;
  return (
    <div className="border-t border-border-muted px-2.5 py-1.5 text-xs" data-testid="signal-why" data-source={why.category_source}>
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen((v) => !v)}
          className="inline-flex items-center gap-1 font-semibold text-primer-blueFg hover:underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
        >
          {open ? <ChevronDown className="w-3 h-3" aria-hidden="true" /> : <ChevronRight className="w-3 h-3" aria-hidden="true" />}
          Why
        </button>
        <span className="text-fg-muted">{why.category_source === 'rules' ? 'Found by rules' : 'Found by Gemma'}</span>
        {why.check === 'confirmed' && (
          <span data-testid="signal-why-checked">
            <Chip tone="green" icon={CheckCircle2} title="Gemma read this rule-found span and confirmed it">
              Gemma double-checked ✓
            </Chip>
          </span>
        )}
      </div>
      {open && (
        <div id={panelId} className="mt-1.5 space-y-1.5" data-testid="signal-why-panel">
          <p className="text-fg break-words" data-testid="signal-why-summary">
            {summary}
            {why.check === 'confirmed' ? ' · Gemma double-checked ✓' : ''}
          </p>
          <p className="text-fg-muted">
            Category decided by {sourceLabel(why.category_source)} · subcategory decided by {sourceLabel(why.subcategory_source)}
            {rule && why.subcategory_source === 'rules' && subName && ` (${subName}, ${fmt2(rule.subcategory_share)} of the example vote)`}
          </p>
          {rule && rule.outcomes.length > 0 && (
            <div>
              <p className="font-medium text-fg-muted">Rules</p>
              <ul className="flex flex-wrap gap-1" aria-label="Rule outcomes">
                {rule.outcomes.map((o) => {
                  const detail =
                    o.type === 'call_position' && o.value !== null
                      ? ` (starts at ${Math.round(o.value * 100)}% of the call)`
                      : o.type === 'similar_to_examples' && o.value !== null
                        ? ` (${fmt2(o.value)})`
                        : o.type === 'phrase' && o.vetoed
                          ? ' (vetoed by a negation)'
                          : o.type === 'phrase' && o.phrase_index !== null
                            ? ` (phrase ${o.phrase_index + 1})`
                            : '';
                  return (
                    <li key={o.rule_id}>
                      <Chip tone={o.result === 'pass' ? 'green' : 'red'} title={`Rule ${o.rule_id}`}>
                        {RULE_TYPE_LABEL[o.type]} {o.result === 'pass' ? '✓' : '✗'}
                        {detail}
                      </Chip>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
          {rule && rule.neighbours.length > 0 && (
            <div>
              <p className="font-medium text-fg-muted">Nearest examples</p>
              <ol className="space-y-0.5" aria-label="Nearest examples">
                {rule.neighbours.map((n) => (
                  <li key={n.entry_id} className="flex flex-wrap items-center gap-x-2 text-fg-muted">
                    <code className="text-fg break-all">{n.entry_id}</code>
                    <span className="tabular-nums">cosine {fmt2(n.cosine)}</span>
                    <span>
                      {n.carries_category
                        ? `labelled ${categoryById(taxonomy, categoryId)?.name ?? categoryId}${n.subcategory_id && n.subcategory_id !== 'other' ? ` › ${subcategoryName(taxonomy, categoryId, n.subcategory_id)}` : ''}`
                        : 'labelled otherwise'}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function UpdateSignals({ client, session, callId, text, onWrote }: { client: StoreClient; session: SignedInSession; callId: string; text: string; onWrote(): void }) {
  const idempotency = useIdempotencyKey();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [done, setDone] = useState(false);
  async function update() {
    setBusy(true);
    setError(null);
    const body = { kind: 'contact_signals' as const, note: null, rescore_signals: false };
    try {
      await sendIdempotent(idempotency, { callId, ...body }, (key) =>
        client.post('/store/v1/calls/{call_id}/reanalysis-requests', { path: { call_id: callId }, headers: { 'Idempotency-Key': key }, body }),
      );
      setDone(true);
      onWrote();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Notice tone="yellow" icon={Sparkles}>
      <span className="block">{text}</span>
      {session.can('request_reanalysis') && (
        <span className="flex flex-wrap items-center gap-2 mt-1.5">
          <Button size="sm" icon={done ? CheckCircle2 : RefreshCw} busy={busy} disabled={done} onClick={() => void update()}>
            {done ? 'Update requested' : 'Update signals'}
          </Button>
          <span className="text-xs">Reruns only the stages the taxonomy change outdated.</span>
        </span>
      )}
      {error !== null && <span className="block mt-1 text-primer-redFg">{describeError(error)}</span>}
    </Notice>
  );
}

function CompareWithV2({ client, previewId, callId, taxonomy }: { client: StoreClient; previewId: string; callId: string; taxonomy: SignalTaxonomy | undefined }) {
  const q = useQuery({
    queryKey: queryKeys.signalPreview(previewId),
    queryFn: ({ signal }) => client.get('/store/v1/signals/previews/{preview_id}', { path: { preview_id: previewId }, signal }),
  });
  if (q.isLoading) return <Loading label="Loading the v2 comparison…" />;
  if (q.error) return <ErrorNotice error={q.error} />;
  if (!q.data) return null;
  return <PreviewResults client={client} preview={q.data} taxonomy={taxonomy} onlyCallId={callId} callName={() => 'v2 beside v1 on this call'} />;
}
