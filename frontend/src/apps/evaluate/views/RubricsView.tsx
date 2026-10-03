// `#/rubrics` and `#/rubrics/:rubricId` — Rubric Studio: list, draft editor with optimistic
// concurrency (`draft_revision`), publish, version history, and draft tests run through a
// reanalysis request against a real call.

import { useEffect, useState } from 'react';
import { useInfiniteQuery, useQuery, useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, ClipboardList, Plus, Trash2 } from 'lucide-react';
import {
  StoreError,
  describeError,
  isNotImplemented,
  isVersionConflict,
  sendIdempotent,
  useIdempotencyKey,
  queryKeys,
  verdictStatusDisplay,
  type CheckType,
  type EscalationTrigger,
  type ReanalysisRequest,
  type RubricCategory,
  type RubricCheck,
  type RubricCriterion,
  type RubricDefinition,
  type RubricDraft,
  type RubricSummary,
  type RubricVersion,
  type SpeakerRole,
} from '../api';
import {
  Button,
  Card,
  EmptyState,
  ErrorNotice,
  Field,
  Loading,
  NotBuiltYet,
  Notice,
  PageHeader,
  CharCount,
  SelectInput,
  StatusPill,
  TextArea,
  TextInput,
  formatDateTime,
} from '../components/ui';
import { href } from '../state/router';
import type { RubricsViewProps } from './types';

const CATEGORIES: RubricCategory[] = ['GENERAL', 'BANKING', 'CUSTOMER_CARE', 'COLLECTIONS', 'HEALTHCARE', 'TECH_SUPPORT'];
const CHECK_TYPES: CheckType[] = ['phrase_any', 'phrase_all', 'phrase_none', 'conditional_response', 'sentiment_metric', 'semantic_judgement', 'custom_regex'];
// RubricCheck.policy_context max_length (call1/contracts/rubrics.py).
const POLICY_CONTEXT_MAX = 12000;
const ESCALATION_TRIGGERS: EscalationTrigger[] = ['needs_review', 'invalid_answer', 'provider_error', 'always'];

function blankCheck(): RubricCheck {
  return {
    aggregation: 'mean',
    check_type: 'phrase_any',
    comparison: 'gte',
    escalation_model_id: null,
    escalation_when: ['needs_review'],
    fail_when: null,
    metric: 'text_polarity',
    metric_threshold: 0,
    min_coverage: 0.8,
    min_samples: 2,
    not_applicable_when: null,
    pass_when: null,
    pattern: null,
    phrases: [],
    policy_context: null,
    primary_model_id: null,
    requires_policy: false,
    response_phrases: [],
    speaker: 'AGENT',
    threshold: 80,
    trigger_phrases: [],
    window_seconds: null,
  };
}

function blankCriterion(): RubricCriterion {
  return {
    category: 'COMPLIANCE',
    check: blankCheck(),
    criterion_id: '',
    critical: false,
    description: '',
    name: '',
    weight: 25,
  };
}

function blankDefinition(rubricId: string): RubricDefinition {
  return { category: 'GENERAL', criteria: [], description: '', name: '', pass_threshold: 80, rubric_id: rubricId };
}

/**
 * `#/rubrics` and `#/rubrics/:rubricId` — Rubric Studio.
 */
export default function RubricsView({ rubricId, client, session, navigate }: RubricsViewProps) {
  if (!rubricId) return <RubricList client={client} navigate={navigate} />;
  return <RubricDetail rubricId={rubricId} client={client} session={session} navigate={navigate} />;
}

// --- list -----------------------------------------------------------------------------------------

function RubricList({ client, navigate }: Pick<RubricsViewProps, 'client' | 'navigate'>) {
  const [includeRetired, setIncludeRetired] = useState(false);
  const [newId, setNewId] = useState('');
  const q = useInfiniteQuery({
    queryKey: [...queryKeys.rubrics, includeRetired],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) => client.get('/store/v1/rubrics', { query: { limit: 100, page_token: pageParam, include_retired: includeRetired }, signal }),
    getNextPageParam: (last) => last.next_page_token,
  });
  const rubrics = q.data?.pages.flatMap((p) => p.items) ?? [];

  return (
    <div className="max-w-5xl mx-auto">
      <PageHeader
        title="Rubrics"
        description="What Evaluate scores calls against."
        right={
          <>
            <label className="flex items-center gap-2 text-sm text-fg">
              <input type="checkbox" checked={includeRetired} onChange={(e) => setIncludeRetired(e.target.checked)} />
              Include retired
            </label>
            <TextInput placeholder="new-rubric-id" value={newId} maxLength={80} className="w-40" onChange={(e) => setNewId(e.target.value)} />
            <Button
              icon={Plus}
              disabled={!newId.trim()}
              onClick={() => navigate({ name: 'rubrics', rubricId: newId.trim() })}
            >
              New rubric
            </Button>
          </>
        }
      />
      {q.isLoading && <Loading label="Loading rubrics…" />}
      <ErrorNotice error={q.error} />
      {q.isSuccess && rubrics.length === 0 && <EmptyState icon={ClipboardList} title="No rubrics yet">Create one with an id above.</EmptyState>}
      {rubrics.length > 0 && (
        <div className="rounded-lg border border-border bg-canvas-subtle divide-y divide-border-muted">
          {rubrics.map((r) => (
            <RubricRow key={r.rubric_id} rubric={r} />
          ))}
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

function RubricRow({ rubric }: { rubric: RubricSummary }) {
  return (
    <a href={href({ name: 'rubrics', rubricId: rubric.rubric_id })} className="flex items-center justify-between gap-3 px-4 py-3 hover:bg-canvas transition-colors">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-fg">{rubric.name || rubric.rubric_id}</span>
          <span className="text-xs text-fg-subtle">{rubric.category}</span>
          {rubric.has_draft && <StatusPill tone="blue">Draft</StatusPill>}
        </div>
        <p className="text-xs text-fg-muted mt-0.5">
          {rubric.criteria_count} criteria · pass ≥ {rubric.pass_threshold} · updated {formatDateTime(rubric.updated_at)}
        </p>
      </div>
      <span className="text-xs text-fg-muted shrink-0">{rubric.current_version ? `v${rubric.current_version}` : 'unpublished'}</span>
    </a>
  );
}

// --- detail -----------------------------------------------------------------------------------

type Tab = 'draft' | 'published' | 'versions';

function RubricDetail({ rubricId, client, session, navigate }: { rubricId: string } & Pick<RubricsViewProps, 'client' | 'session' | 'navigate'>) {
  const qc = useQueryClient();
  const [tab, setTab] = useState<Tab>('draft');
  const canManage = session.can('manage_rubrics');

  const publishedQuery = useQuery({
    queryKey: [...queryKeys.rubric(rubricId), 'published'],
    queryFn: ({ signal }) => client.get('/store/v1/rubrics/{rubric_id}', { path: { rubric_id: rubricId }, signal }),
    retry: false,
  });
  const publishedMissing = publishedQuery.error instanceof StoreError && publishedQuery.error.status === 404;

  const draftQuery = useQuery({
    queryKey: [...queryKeys.rubric(rubricId), 'draft'],
    queryFn: ({ signal }) => client.get('/store/v1/rubrics/{rubric_id}/draft', { path: { rubric_id: rubricId }, signal }),
    retry: false,
    enabled: canManage,
  });
  const draftMissing = draftQuery.error instanceof StoreError && draftQuery.error.status === 404;

  const versionsQuery = useQuery({
    queryKey: [...queryKeys.rubric(rubricId), 'versions'],
    queryFn: ({ signal }) => client.get('/store/v1/rubrics/{rubric_id}/versions', { path: { rubric_id: rubricId }, signal }),
    enabled: tab === 'versions',
  });

  // The editor waits for the first settle of both queries, then stays mounted. A refetch of a
  // query that settled as a 404 (a brand-new rubric has no draft and no published version) goes
  // back to pending, so gating on isLoading alone would unmount the editor after every save and
  // drop its notice and any in-flight edits.
  const settled = !publishedQuery.isLoading && !(canManage && draftQuery.isLoading);
  const [readyFor, setReadyFor] = useState<string | null>(null);
  useEffect(() => {
    if (settled) setReadyFor(rubricId);
  }, [settled, rubricId]);
  const editorReady = settled || readyFor === rubricId;

  const invalidateAll = () => {
    void qc.invalidateQueries({ queryKey: queryKeys.rubric(rubricId) });
    void qc.invalidateQueries({ queryKey: queryKeys.rubrics });
  };

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      <PageHeader
        title={publishedQuery.data?.definition.name || rubricId}
        description={<span className="break-all">{rubricId}</span>}
        right={
          <Button onClick={() => navigate({ name: 'rubrics' })} icon={ClipboardList}>
            All rubrics
          </Button>
        }
      />

      <div className="flex gap-1.5 border-b border-border-muted">
        {(['draft', 'published', 'versions'] as Tab[]).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`px-3 py-1.5 text-sm font-medium capitalize border-b-2 -mb-px transition-colors ${
              tab === t ? 'border-primer-blue text-fg' : 'border-transparent text-fg-muted hover:text-fg'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'draft' &&
        (!canManage ? (
          <Card>
            <p className="text-sm text-fg-muted">Your role can read rubrics but not edit drafts.</p>
          </Card>
        ) : !editorReady ? (
          <Loading label="Loading draft…" />
        ) : (
          // Mount only once both the draft and the published version have settled, so the editor
          // seeds from whichever exists (a draft, else a copy of the published version) instead of
          // mounting blank and missing the published version that arrives later. Keyed on the
          // rubric so navigating between rubrics starts a fresh editor.
          <DraftEditor
            key={rubricId}
            rubricId={rubricId}
            client={client}
            draft={draftMissing ? null : draftQuery.data}
            draftError={draftMissing ? null : draftQuery.error}
            published={publishedMissing ? null : publishedQuery.data}
            session={session}
            onChanged={invalidateAll}
          />
        ))}

      {tab === 'published' && (
        <Card title="Published version">
          {publishedQuery.isLoading ? (
            <Loading label="Loading…" />
          ) : publishedMissing ? (
            <EmptyState title="Nothing published yet">Publish a draft to make it live.</EmptyState>
          ) : publishedQuery.error ? (
            <ErrorNotice error={publishedQuery.error} />
          ) : (
            <RubricDefinitionView definition={publishedQuery.data!.definition} />
          )}
        </Card>
      )}

      {tab === 'versions' && (
        <Card title="Version history">
          {versionsQuery.isLoading ? (
            <Loading label="Loading…" />
          ) : versionsQuery.error ? (
            <ErrorNotice error={versionsQuery.error} />
          ) : !versionsQuery.data || versionsQuery.data.items.length === 0 ? (
            <EmptyState title="No published versions" />
          ) : (
            <div className="divide-y divide-border-muted">
              {versionsQuery.data.items.map((v) => (
                <div key={v.ref.version} className="py-2.5 flex items-center justify-between gap-2 text-sm">
                  <div>
                    <span className="font-medium text-fg">v{v.ref.version}</span>
                    <span className="text-xs text-fg-muted ml-2">
                      {v.status} · published {formatDateTime(v.published_at)}
                    </span>
                    {v.notes && <p className="text-xs text-fg-muted mt-0.5">{v.notes}</p>}
                  </div>
                  <span className="text-xs text-fg-subtle truncate max-w-[10rem]" title={v.ref.digest}>
                    {v.ref.digest.slice(0, 12)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

function RubricDefinitionView({ definition }: { definition: RubricDefinition }) {
  return (
    <div className="space-y-3 text-sm">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-fg-muted">
        <span>Category: {definition.category}</span>
        <span>Pass threshold: {definition.pass_threshold}</span>
        <span>{definition.criteria?.length ?? 0} criteria</span>
      </div>
      {definition.description && <p className="text-fg-muted">{definition.description}</p>}
      <div className="space-y-2">
        {(definition.criteria ?? []).map((c) => (
          <div key={c.criterion_id} className="rounded-md border border-border-muted bg-canvas p-2.5">
            <div className="flex items-center gap-2">
              <span className="font-medium text-fg">{c.name}</span>
              <span className="text-xs text-fg-subtle">{c.criterion_id}</span>
              {c.critical && <StatusPill tone="red">Critical</StatusPill>}
              <span className="text-xs text-fg-muted ml-auto">weight {c.weight}</span>
            </div>
            {c.description && <p className="text-xs text-fg-muted mt-1">{c.description}</p>}
            <p className="text-xs text-fg-subtle mt-1">check: {c.check.check_type}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// --- draft editor -----------------------------------------------------------------------------

function DraftEditor({
  rubricId,
  client,
  draft,
  draftError,
  published,
  session,
  onChanged,
}: {
  rubricId: string;
  client: RubricsViewProps['client'];
  draft: RubricDraft | null | undefined;
  draftError: unknown;
  published: RubricVersion | null | undefined;
  session: RubricsViewProps['session'];
  onChanged: () => void;
}) {
  const [definition, setDefinition] = useState<RubricDefinition>(() => draft?.definition ?? published?.definition ?? blankDefinition(rubricId));
  const [expectedRevision, setExpectedRevision] = useState<number | null>(draft?.draft_revision ?? null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Re-sync local editor state when a save, discard or publish changes the draft revision or the
  // published version, without clobbering in-flight edits (neither changes while the user types).
  // The first seed happens in useState: RubricDetail mounts this only once both queries settle.
  useEffect(() => {
    if (draft) {
      setDefinition(draft.definition);
      setExpectedRevision(draft.draft_revision);
    } else if (published) {
      setDefinition(published.definition);
      setExpectedRevision(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft?.draft_revision, published?.ref.version]);

  const criteriaCount = definition.criteria?.length ?? 0;
  // Guards against an empty rubric going live: never publish one with no criteria, and never save
  // an empty draft over a published version that has criteria. A brand-new rubric may still save
  // its name and description before any criterion exists.
  const publishedHasCriteria = (published?.definition.criteria?.length ?? 0) > 0;
  const saveBlocked = criteriaCount === 0 && publishedHasCriteria;
  const publishBlocked = criteriaCount === 0;

  async function save() {
    if (saveBlocked) return;
    setBusy(true);
    setError(null);
    try {
      const saved = await client.put('/store/v1/rubrics/{rubric_id}/draft', {
        path: { rubric_id: rubricId },
        body: { definition, expected_draft_revision: expectedRevision },
      });
      setExpectedRevision(saved.draft_revision);
      setNotice('Draft saved.');
      onChanged();
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) onChanged(); // re-read: the latest saved draft or version loads
    } finally {
      setBusy(false);
    }
  }

  async function discard() {
    if (!window.confirm('Discard this draft? This cannot be undone.')) return;
    setBusy(true);
    setError(null);
    try {
      await client.delete('/store/v1/rubrics/{rubric_id}/draft', { path: { rubric_id: rubricId } });
      onChanged();
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) onChanged(); // re-read: the latest saved draft or version loads
    } finally {
      setBusy(false);
    }
  }

  async function publish() {
    if (publishBlocked) return;
    if (expectedRevision === null) {
      setError(new Error('Save the draft before publishing.'));
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await client.post('/store/v1/rubrics/{rubric_id}/publish', {
        path: { rubric_id: rubricId },
        body: { expected_current_version: published?.ref.version ?? 0, expected_draft_revision: expectedRevision },
      });
      setNotice('Published.');
      onChanged();
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) onChanged(); // re-read: the latest saved draft or version loads
    } finally {
      setBusy(false);
    }
  }

  function updateCriterion(index: number, patch: Partial<RubricCriterion>) {
    setDefinition((d) => ({ ...d, criteria: (d.criteria ?? []).map((c, i) => (i === index ? { ...c, ...patch } : c)) }));
  }
  function updateCheck(index: number, patch: Partial<RubricCheck>) {
    setDefinition((d) => ({ ...d, criteria: (d.criteria ?? []).map((c, i) => (i === index ? { ...c, check: { ...c.check, ...patch } } : c)) }));
  }
  function removeCriterion(index: number) {
    setDefinition((d) => ({ ...d, criteria: (d.criteria ?? []).filter((_, i) => i !== index) }));
  }
  function addCriterion() {
    setDefinition((d) => ({ ...d, criteria: [...(d.criteria ?? []), blankCriterion()] }));
  }

  return (
    <div className="space-y-4">
      <ErrorNotice error={draftError} />
      {!draft && published && <Notice tone="blue">No draft yet — editing a copy of the published version (v{published.ref.version}).</Notice>}
      {!draft && !published && <Notice tone="blue">No draft and nothing published yet. Fill this in and save.</Notice>}

      <Card title="Rubric">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Field label="Name">{(id) => <TextInput id={id} value={definition.name} maxLength={200} onChange={(e) => setDefinition((d) => ({ ...d, name: e.target.value }))} />}</Field>
          <Field label="Category">
            {(id) => (
              <SelectInput id={id} value={definition.category} onChange={(e) => setDefinition((d) => ({ ...d, category: e.target.value as RubricCategory }))}>
                {CATEGORIES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </SelectInput>
            )}
          </Field>
          <Field label="Pass threshold">
            {(id) => (
              <TextInput
                id={id}
                type="number"
                min={0}
                max={100}
                value={definition.pass_threshold}
                onChange={(e) => setDefinition((d) => ({ ...d, pass_threshold: Number(e.target.value) }))}
              />
            )}
          </Field>
          <Field label="Description">
            {(id) => <TextInput id={id} value={definition.description} maxLength={2000} onChange={(e) => setDefinition((d) => ({ ...d, description: e.target.value }))} />}
          </Field>
        </div>
      </Card>

      <Card
        title="Criteria"
        subtitle={`${criteriaCount} criteria`}
        right={
          <Button size="sm" icon={Plus} onClick={addCriterion}>
            Add criterion
          </Button>
        }
      >
        <div className="space-y-3">
          {(definition.criteria ?? []).map((c, i) => (
            <CriterionEditor key={i} criterion={c} onChange={(patch) => updateCriterion(i, patch)} onChangeCheck={(patch) => updateCheck(i, patch)} onRemove={() => removeCriterion(i)} />
          ))}
          {(definition.criteria ?? []).length === 0 && <EmptyState title="No criteria yet">Add one above.</EmptyState>}
        </div>
      </Card>

      <ErrorNotice error={error} />
      {notice && (
        <Notice tone="green" icon={CheckCircle2}>
          {notice}
        </Notice>
      )}

      {criteriaCount === 0 && (
        <p className="text-xs text-fg-muted">{saveBlocked ? 'Add at least one criterion before saving or publishing.' : 'Add at least one criterion before publishing.'}</p>
      )}
      <div className="flex flex-wrap gap-2">
        <Button variant="primary" busy={busy} disabled={saveBlocked} onClick={() => void save()}>
          Save draft
        </Button>
        <Button variant="danger" busy={busy} disabled={!draft} onClick={() => void discard()}>
          Discard draft
        </Button>
        <Button busy={busy} disabled={expectedRevision === null || publishBlocked} onClick={() => void publish()}>
          Publish
        </Button>
        {published && <RetireControl rubricId={rubricId} client={client} currentVersion={published.ref.version} onChanged={onChanged} />}
      </div>

      <DraftTest rubricId={rubricId} client={client} expectedRevision={expectedRevision} session={session} onChanged={onChanged} />
    </div>
  );
}

function RetireControl({ rubricId, client, currentVersion, onChanged }: { rubricId: string; client: RubricsViewProps['client']; currentVersion: number; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  async function retire() {
    const reason = window.prompt('Reason for retiring this rubric?');
    if (!reason) return;
    setBusy(true);
    setError(null);
    try {
      await client.post('/store/v1/rubrics/{rubric_id}/retire', { path: { rubric_id: rubricId }, body: { expected_current_version: currentVersion, reason } });
      onChanged();
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) onChanged(); // re-read: the latest saved draft or version loads
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Button variant="danger" busy={busy} onClick={() => void retire()}>
        Retire
      </Button>
      {error !== null && <ErrorNotice error={error} />}
    </>
  );
}

function CriterionEditor({
  criterion,
  onChange,
  onChangeCheck,
  onRemove,
}: {
  criterion: RubricCriterion;
  onChange(patch: Partial<RubricCriterion>): void;
  onChangeCheck(patch: Partial<RubricCheck>): void;
  onRemove(): void;
}) {
  const [open, setOpen] = useState(false);
  const check = criterion.check;

  return (
    <div className="rounded-md border border-border-muted bg-canvas p-3 space-y-2.5">
      <div className="flex items-center gap-2">
        <button type="button" className="text-sm font-medium text-fg truncate flex-1 text-left" onClick={() => setOpen((o) => !o)}>
          {criterion.name || criterion.criterion_id || 'Untitled criterion'}
        </button>
        <Button size="sm" variant="ghost" icon={Trash2} aria-label="Remove criterion" onClick={onRemove} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        <Field label="Criterion id" hint="Slot-safe id; becomes the qa_assessment slot.">
          {(id) => <TextInput id={id} value={criterion.criterion_id} maxLength={120} onChange={(e) => onChange({ criterion_id: e.target.value })} />}
        </Field>
        <Field label="Name">{(id) => <TextInput id={id} value={criterion.name} maxLength={200} onChange={(e) => onChange({ name: e.target.value })} />}</Field>
        <Field label="Category">{(id) => <TextInput id={id} value={criterion.category} maxLength={80} onChange={(e) => onChange({ category: e.target.value })} />}</Field>
        <Field label="Weight">{(id) => <TextInput id={id} type="number" value={criterion.weight} onChange={(e) => onChange({ weight: Number(e.target.value) })} />}</Field>
        <label className="flex items-center gap-2 text-sm text-fg sm:col-span-2">
          <input type="checkbox" checked={criterion.critical} onChange={(e) => onChange({ critical: e.target.checked })} />
          Critical (an automatic fail if this fails)
        </label>
        <Field label="Description">{(id) => <TextInput id={id} value={criterion.description} maxLength={2000} onChange={(e) => onChange({ description: e.target.value })} />}</Field>
        <label className="flex items-center gap-2 text-sm text-fg sm:col-span-2">
          <input type="checkbox" checked={check.requires_policy} onChange={(e) => onChangeCheck({ requires_policy: e.target.checked })} />
          Requires policy context
        </label>
        {check.requires_policy && (
          <div className="sm:col-span-2">
            <Field
              label="Policy text"
              hint={
                <>
                  The policy the model judges this criterion against, e.g. your refund or verification rules. Without it, this criterion is always flagged for
                  review. <CharCount value={check.policy_context} max={POLICY_CONTEXT_MAX} />
                </>
              }
            >
              {(id) => (
                <TextArea
                  id={id}
                  rows={4}
                  maxLength={POLICY_CONTEXT_MAX}
                  placeholder="Paste the policy wording this criterion should be checked against."
                  value={check.policy_context ?? ''}
                  onChange={(e) => onChangeCheck({ policy_context: e.target.value || null })}
                />
              )}
            </Field>
          </div>
        )}
      </div>

      {open && (
        <div className="pt-2.5 border-t border-border-muted grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          <Field label="Check type">
            {(id) => (
              <SelectInput id={id} value={check.check_type} onChange={(e) => onChangeCheck({ check_type: e.target.value as CheckType })}>
                {CHECK_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t.replace(/_/g, ' ')}
                  </option>
                ))}
              </SelectInput>
            )}
          </Field>
          <Field label="Speaker">
            {(id) => (
              <SelectInput id={id} value={check.speaker ?? ''} onChange={(e) => onChangeCheck({ speaker: (e.target.value || null) as SpeakerRole | null })}>
                <option value="AGENT">Agent</option>
                <option value="CALLER">Caller</option>
                <option value="SYSTEM">System</option>
                <option value="UNKNOWN">Unknown</option>
                <option value="">Either</option>
              </SelectInput>
            )}
          </Field>
          <Field label="Phrases" hint="Comma-separated.">
            {(id) => (
              <TextInput
                id={id}
                value={(check.phrases ?? []).join(', ')}
                onChange={(e) => onChangeCheck({ phrases: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })}
              />
            )}
          </Field>
          <Field label="Threshold" hint="Phrase similarity, 0–100.">
            {(id) => <TextInput id={id} type="number" min={0} max={100} value={check.threshold} onChange={(e) => onChangeCheck({ threshold: Number(e.target.value) })} />}
          </Field>
          <Field label="Pattern" hint="For custom_regex checks.">
            {(id) => <TextInput id={id} value={check.pattern ?? ''} onChange={(e) => onChangeCheck({ pattern: e.target.value || null })} />}
          </Field>
          <Field label="Primary model id" hint="Catalog entry; blank inherits the purpose default.">
            {(id) => <TextInput id={id} value={check.primary_model_id ?? ''} onChange={(e) => onChangeCheck({ primary_model_id: e.target.value || null })} />}
          </Field>
          <Field label="Window seconds" hint=">0 first N seconds, <0 last N seconds, blank = whole call.">
            {(id) => (
              <TextInput
                id={id}
                type="number"
                value={check.window_seconds ?? ''}
                onChange={(e) => onChangeCheck({ window_seconds: e.target.value === '' ? null : Number(e.target.value) })}
              />
            )}
          </Field>
          <Field label="Metric" hint="For sentiment_metric checks.">
            {(id) => (
              <SelectInput id={id} value={check.metric} onChange={(e) => onChangeCheck({ metric: e.target.value as RubricCheck['metric'] })}>
                <option value="text_polarity">text_polarity</option>
                <option value="valence">valence</option>
                <option value="arousal">arousal</option>
                <option value="dominance">dominance</option>
              </SelectInput>
            )}
          </Field>
          <Field label="Metric threshold">
            {(id) => <TextInput id={id} type="number" value={check.metric_threshold} onChange={(e) => onChangeCheck({ metric_threshold: Number(e.target.value) })} />}
          </Field>
          <Field label="Comparison">
            {(id) => (
              <SelectInput id={id} value={check.comparison} onChange={(e) => onChangeCheck({ comparison: e.target.value as RubricCheck['comparison'] })}>
                <option value="gte">≥</option>
                <option value="lte">≤</option>
              </SelectInput>
            )}
          </Field>
          <Field label="Pass when" hint="Expression evaluated against the model's answer.">
            {(id) => <TextInput id={id} value={check.pass_when ?? ''} onChange={(e) => onChangeCheck({ pass_when: e.target.value || null })} />}
          </Field>
          <Field label="Fail when">{(id) => <TextInput id={id} value={check.fail_when ?? ''} onChange={(e) => onChangeCheck({ fail_when: e.target.value || null })} />}</Field>
          <Field label="Not applicable when">
            {(id) => <TextInput id={id} value={check.not_applicable_when ?? ''} onChange={(e) => onChangeCheck({ not_applicable_when: e.target.value || null })} />}
          </Field>
          <Field label="Escalation model id" hint="Blank inherits; 'none' disables.">
            {(id) => <TextInput id={id} value={check.escalation_model_id ?? ''} onChange={(e) => onChangeCheck({ escalation_model_id: e.target.value || null })} />}
          </Field>
          <div className="sm:col-span-2 flex flex-wrap gap-3">
            {ESCALATION_TRIGGERS.map((t) => (
              <label key={t} className="flex items-center gap-1.5 text-xs text-fg">
                <input
                  type="checkbox"
                  checked={(check.escalation_when ?? []).includes(t)}
                  onChange={(e) => {
                    const set = new Set(check.escalation_when ?? []);
                    if (e.target.checked) set.add(t);
                    else set.delete(t);
                    onChangeCheck({ escalation_when: Array.from(set) });
                  }}
                />
                escalate on {t.replace(/_/g, ' ')}
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// --- draft test -----------------------------------------------------------------------------------

function DraftTest({
  rubricId,
  client,
  expectedRevision,
  session,
  onChanged,
}: {
  rubricId: string;
  client: RubricsViewProps['client'];
  expectedRevision: number | null;
  session: RubricsViewProps['session'];
  onChanged: () => void;
}) {
  const [callId, setCallId] = useState('');
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [request, setRequest] = useState<ReanalysisRequest | null>(null);
  // One key per logical test: reused for a retry of the same body, replaced when the call, the
  // note or the draft revision changes (Store digests the body under the key) and after a success.
  const idempotency = useIdempotencyKey();

  const resultQuery = useQuery({
    queryKey: [...queryKeys.reanalysis, 'draft-result', request?.id],
    queryFn: ({ signal }) => client.get('/store/v1/reanalysis-requests/{request_id}/draft-result', { path: { request_id: request!.id }, signal }),
    enabled: !!request,
    refetchInterval: (q) => (q.state.data?.state === 'pending' ? 2000 : false),
  });

  if (!session.can('request_reanalysis')) return null;

  async function submit() {
    if (expectedRevision === null || !callId.trim()) return;
    setBusy(true);
    setError(null);
    setRequest(null);
    const body = { call_id: callId.trim(), expected_draft_revision: expectedRevision, note: note || null };
    try {
      const req = await sendIdempotent(idempotency, { rubricId, ...body }, (key) =>
        client.post('/store/v1/rubrics/{rubric_id}/draft/tests', {
          path: { rubric_id: rubricId },
          headers: { 'Idempotency-Key': key },
          body,
        }),
      );
      setRequest(req);
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) onChanged(); // re-read the draft: the test must name its current revision
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Draft test" subtitle="Score a real call with the unpublished draft, without touching its live scorecard.">
      {expectedRevision === null && <Notice tone="blue">Save the draft first.</Notice>}
      <div className="flex flex-wrap items-end gap-2 mt-2">
        <Field label="Call id">{(id) => <TextInput id={id} value={callId} maxLength={200} className="w-64" onChange={(e) => setCallId(e.target.value)} />}</Field>
        <Field label="Note">{(id) => <TextInput id={id} value={note} maxLength={500} className="w-64" onChange={(e) => setNote(e.target.value)} />}</Field>
        <Button busy={busy} disabled={expectedRevision === null || !callId.trim()} onClick={() => void submit()}>
          Run test
        </Button>
      </div>
      <ErrorNotice error={error} />
      {request && (
        <div className="mt-3">
          {resultQuery.isLoading || resultQuery.data?.state === 'pending' ? (
            <Loading label="Scoring…" />
          ) : resultQuery.error ? (
            isNotImplemented(resultQuery.error) ? (
              <NotBuiltYet what="Draft results" />
            ) : (
              <ErrorNotice error={resultQuery.error} />
            )
          ) : resultQuery.data?.state === 'failed' ? (
            <Notice tone="red">Draft test failed{resultQuery.data.failure_code ? `: ${resultQuery.data.failure_code}` : '.'}</Notice>
          ) : resultQuery.data?.scorecard ? (
            <div className="space-y-2">
              <div className="text-sm font-medium text-fg">Score: {Math.round(resultQuery.data.scorecard.overall_score)}</div>
              {resultQuery.data.scorecard.verdicts.map((v) => {
                const d = verdictStatusDisplay(v.status);
                return (
                  <div key={v.criterion_id} className="flex items-center gap-2 text-xs">
                    <StatusPill tone={d.tone}>{d.label}</StatusPill>
                    <span className="text-fg">{v.criterion_name}</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-xs text-fg-muted">{describeError(resultQuery.error) || 'Waiting for a result…'}</p>
          )}
        </div>
      )}
    </Card>
  );
}
