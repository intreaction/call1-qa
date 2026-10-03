// The Signals page's taxonomy tab (docs/ContactSignalsV2.md §10.1): the category tree, the editor,
// the save bar, "Test on recent calls" and the activation dialog.

import { useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { FlaskConical, Lock, Plus, RotateCcw, Save } from 'lucide-react';
import {
  NEW_CATEGORY_RESERVED_IDS,
  StoreError,
  cloneTaxonomy,
  describeTaxonomyPath,
  isVersionConflict,
  queryKeys,
  recipesChanged,
  slugId,
  speakerScopeLabel,
  taxonomyProblems,
  type SignalAlertRuleRecord,
  type SignalCategory,
  type SignalMetrics,
  type SignalTaxonomy,
  type SignalTaxonomyRecord,
} from '../../api';
import { Button, CharCount, EmptyState, ErrorNotice, Field, Notice, StatusPill, TextInput } from '../../components/ui';
import { usePollChanges } from '../../state/app';
import { href } from '../../state/router';
import type { SignalsViewProps } from '../types';
import { ActivationDialog, type EditedNode } from './ActivationDialog';
import { CategoryEditor, ProblemList, extractionDestinations } from './CategoryEditor';
import { PreviewPanel } from './PreviewPanel';

interface Props extends SignalsViewProps {
  record: SignalTaxonomyRecord;
  draft: SignalTaxonomy;
  dirty: boolean;
  staleDraft: boolean;
  setDraft(t: SignalTaxonomy): void;
  resetDraft(to?: SignalTaxonomyRecord): void;
  alertRules: SignalAlertRuleRecord[];
  week: SignalMetrics | undefined;
}

export function TaxonomyTab({ client, contract, session, navigate, categoryId, record, draft, dirty, staleDraft, setDraft, resetDraft, alertRules, week }: Props) {
  const qc = useQueryClient();
  // §11.1: the editor shows each extraction entry's destination host. The catalogs are admin-readable.
  const canReadCatalogs = session.can('manage_signals');
  const catalogQuery = useQuery({
    queryKey: queryKeys.catalogSnapshots,
    queryFn: ({ signal }) => client.get('/store/v1/catalog-snapshots', { signal }),
    enabled: canReadCatalogs,
    retry: false,
  });
  const fallbackEntryId = record.settings.fallback_extraction_entry_id;
  const extraction = useMemo(
    () => (catalogQuery.data ? extractionDestinations(catalogQuery.data.items, fallbackEntryId) : undefined),
    [catalogQuery.data, fallbackEntryId],
  );
  const pollNow = usePollChanges();
  const readOnly = !session.can('manage_signals');
  const caps = contract.parameters;
  const saved = record.current.taxonomy;
  const selectedId = categoryId ?? draft.categories[0]?.category_id;
  const category = draft.categories.find((c) => c.category_id === selectedId);
  const problems = useMemo(() => taxonomyProblems(draft, caps), [draft, caps]);
  const [lastEdited, setLastEdited] = useState<EditedNode | null>(null);
  const [notes, setNotes] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [activation, setActivation] = useState<{ version: number; taxonomy: SignalTaxonomy; node: EditedNode | null; recipeChanged: boolean } | null>(null);
  const [testing, setTesting] = useState(false);
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState('');

  const activeCustom = draft.categories.filter((c) => !c.builtin && c.active).length;
  const customCap = activeCustom >= caps.max_custom_signal_categories || draft.categories.filter((c) => !c.builtin).length >= 16;

  const week7 = useMemo(() => {
    const byCategory = new Map<string, number>();
    const bySub = new Map<string, Map<string, number>>();
    for (const m of week?.categories ?? []) {
      byCategory.set(m.category_id, m.calls_with_hit);
      bySub.set(m.category_id, new Map(m.subcategories.map((s) => [s.id, s.calls_with_hit])));
    }
    return { byCategory, bySub };
  }, [week]);

  function edit(cid: string, update: (c: SignalCategory) => void, subcategoryId?: string) {
    const next = cloneTaxonomy(draft);
    const target = next.categories.find((c) => c.category_id === cid);
    if (!target) return;
    update(target);
    setDraft(next);
    setLastEdited({ categoryId: cid, subcategoryId });
    setNotice(null);
  }

  function addCategory() {
    const name = newName.trim();
    if (!name) return;
    const id = slugId(name, new Set(draft.categories.map((c) => c.category_id)), NEW_CATEGORY_RESERVED_IDS);
    const next = cloneTaxonomy(draft);
    next.categories.push({
      category_id: id,
      builtin: false,
      name,
      gloss: '',
      description: null,
      speaker: null,
      examples: [],
      threshold: null,
      subcategory_threshold: null,
      subcategories: [],
      fields: [],
      narrow_quote: false,
      active: true,
      recipe: null,
    });
    setDraft(next);
    setLastEdited({ categoryId: id });
    setAdding(false);
    setNewName('');
    navigate({ name: 'signals', tab: 'taxonomy', categoryId: id });
  }

  function removeCategory(cid: string) {
    const next = cloneTaxonomy(draft);
    next.categories = next.categories.filter((c) => c.category_id !== cid);
    setDraft(next);
    setLastEdited(null);
    navigate({ name: 'signals', tab: 'taxonomy' }, { replace: true });
  }

  async function save() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const before = record.current.version;
      const next = await client.put('/store/v1/signals/taxonomy', {
        body: { taxonomy: draft, expected_record_version: record.record_version, notes: notes.trim() || null },
      });
      qc.setQueryData(queryKeys.signalTaxonomy, next);
      void qc.invalidateQueries({ queryKey: queryKeys.signalTaxonomyVersions });
      resetDraft(next);
      setNotes('');
      pollNow();
      if (next.current.version === before) {
        setNotice('Nothing changed, so no new version was published.');
      } else {
        setNotice(`Saved as taxonomy v${next.current.version}`);
        setActivation({ version: next.current.version, taxonomy: next.current.taxonomy, node: lastEdited, recipeChanged: recipesChanged(saved, next.current.taxonomy) });
      }
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) {
        // Re-read and load the latest saved version: never overwrite another admin's save.
        const latest = await qc.fetchQuery({
          queryKey: queryKeys.signalTaxonomy,
          queryFn: ({ signal }) => client.get('/store/v1/signals/taxonomy', { signal }),
          staleTime: 0,
        });
        resetDraft(latest);
      }
    } finally {
      setBusy(false);
    }
  }

  const refusedPath = error instanceof StoreError && error.code === 'validation_failed' && typeof error.details.field === 'string' ? error.details.field : null;

  return (
    <div className="space-y-4">
      {staleDraft && (
        <Notice tone="yellow">
          This taxonomy changed elsewhere: v{record.current.version} was saved while you were editing, so your draft can't be saved over it.{' '}
          <button type="button" className="underline rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue" onClick={() => resetDraft()}>
            Discard my edits and load v{record.current.version}
          </button>
        </Notice>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)] gap-4 items-start">
        <nav aria-label="Signal categories" className="rounded-lg border border-border bg-canvas-subtle">
          <ul className="divide-y divide-border-muted">
            {draft.categories.map((c) => {
              const active = c.category_id === selectedId;
              const savedCat = saved.categories.find((x) => x.category_id === c.category_id);
              const hits = week7.byCategory.get(c.category_id);
              const catProblems = problems.filter((p) => p.categoryId === c.category_id).length;
              const changed = !savedCat || JSON.stringify(savedCat) !== JSON.stringify(c);
              return (
                <li key={c.category_id}>
                  <a
                    href={href({ name: 'signals', tab: 'taxonomy', categoryId: c.category_id })}
                    aria-current={active ? 'page' : undefined}
                    className={`block px-3 py-2 text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primer-blue ${
                      active ? 'bg-canvas' : 'hover:bg-canvas'
                    } ${c.active ? '' : 'opacity-70'}`}
                  >
                    <span className="flex items-center gap-1.5">
                      <span className={`font-medium truncate ${active ? 'text-primer-blueFg' : 'text-fg'}`}>{c.name || 'New category'}</span>
                      {c.builtin && (
                        <span className="inline-flex items-center gap-0.5 text-[11px] text-fg-subtle shrink-0" title="Call1's fixed category">
                          <Lock className="w-3 h-3" aria-hidden="true" />
                          Built-in
                        </span>
                      )}
                      {!c.active && <span className="text-[11px] text-fg-subtle shrink-0">Retired</span>}
                      {changed && <span className="text-[11px] text-primer-yellowFg shrink-0">Edited</span>}
                      {catProblems > 0 && <span className="text-[11px] text-primer-redFg shrink-0">{catProblems} to fix</span>}
                    </span>
                    <span className="block text-xs text-fg-muted truncate">{c.gloss || 'No gloss yet'}</span>
                    <span className="block text-[11px] text-fg-subtle">
                      {speakerScopeLabel(c.speaker)} · {c.subcategories.filter((s) => s.active).length} subcategories · {c.fields.length + c.subcategories.reduce((n, s) => n + s.fields.length, 0)} fields
                      {hits !== undefined && ` · ${hits} call${hits === 1 ? '' : 's'} in 7 days`}
                    </span>
                  </a>
                </li>
              );
            })}
          </ul>
          {!readOnly && (
            <div className="p-3 border-t border-border-muted space-y-2">
              {adding ? (
                <>
                  <Field label="New category name" hint={<CharCount value={newName} max={40} />}>
                    {(id) => (
                      <TextInput
                        id={id}
                        value={newName}
                        maxLength={60}
                        autoFocus
                        onChange={(e) => setNewName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') addCategory();
                        }}
                      />
                    )}
                  </Field>
                  <div className="flex gap-2">
                    <Button size="sm" variant="primary" disabled={!newName.trim()} onClick={addCategory}>
                      Add
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => setAdding(false)}>
                      Cancel
                    </Button>
                  </div>
                </>
              ) : (
                <Button size="sm" icon={Plus} disabled={customCap} onClick={() => setAdding(true)}>
                  Add category
                </Button>
              )}
              {customCap && (
                <p className="text-xs text-primer-yellowFg" role="status">
                  At most {caps.max_custom_signal_categories} active custom categories: each one is another option the classifier weighs on every segment. Retire one to add another.
                </p>
              )}
            </div>
          )}
        </nav>

        <div className="rounded-lg border border-border bg-canvas-subtle p-4 min-w-0 space-y-4">
          {readOnly && (
            <Notice>
              Read-only: only admins change the taxonomy. You can read every category, subcategory and field here.
            </Notice>
          )}
          {category ? (
            <CategoryEditor
              key={category.category_id}
              category={category}
              saved={saved.categories.find((c) => c.category_id === category.category_id)}
              readOnly={readOnly}
              caps={caps}
              problems={problems.filter((p) => p.categoryId === category.category_id)}
              weekBySubcategory={week7.bySub.get(category.category_id) ?? new Map()}
              onChange={(update, sub) => edit(category.category_id, update, sub)}
              onRemove={() => removeCategory(category.category_id)}
              extraction={extraction}
            />
          ) : (
            <EmptyState title="No such category">
              There is no category <span className="break-all">{selectedId}</span> in this taxonomy.{' '}
              <a className="text-primer-blueFg hover:underline" href={href({ name: 'signals', tab: 'taxonomy' })}>
                Show the first one
              </a>
            </EmptyState>
          )}
        </div>
      </div>

      {!readOnly && (
        <div className={`${dirty ? 'sticky bottom-0 z-10 shadow-lg' : ''} rounded-lg border border-border bg-canvas p-3 space-y-2`} data-testid="signals-save-bar">
          {problems.length > 0 && dirty && (
            <div className="space-y-1">
              <p className="text-xs font-medium text-primer-redFg">Fix these before saving:</p>
              <ProblemList problems={problems.slice(0, 6)} />
              {problems.length > 6 && <p className="text-xs text-fg-muted">…and {problems.length - 6} more.</p>}
            </div>
          )}
          <div className="flex flex-wrap items-end gap-2">
            {dirty ? <StatusPill tone="yellow">Unsaved changes</StatusPill> : <StatusPill tone="green">Saved as v{record.current.version}</StatusPill>}
            <div className="flex-1 min-w-[12rem]">
              <Field label="Version notes (optional)">
                {(id) => <TextInput id={id} value={notes} maxLength={500} disabled={!dirty} placeholder="What changed and why" onChange={(e) => setNotes(e.target.value)} />}
              </Field>
            </div>
            <Button icon={FlaskConical} onClick={() => setTesting((t) => !t)} aria-expanded={testing}>
              Test on recent calls
            </Button>
            <Button icon={RotateCcw} variant="ghost" disabled={!dirty || busy} onClick={() => resetDraft()}>
              Discard
            </Button>
            <Button icon={Save} variant="primary" busy={busy} disabled={!dirty || staleDraft || problems.length > 0} onClick={() => void save()}>
              Save
            </Button>
          </div>
          {notice && (
            <Notice tone="green" icon={Save}>
              {notice}
            </Notice>
          )}
          <ErrorNotice error={error}>
            {refusedPath && <span className="block mt-1">Store refused {describeTaxonomyPath(draft, refusedPath)}. Change that text and save again.</span>}
          </ErrorNotice>
        </div>
      )}

      {testing && !readOnly && (
        <PreviewPanel client={client} contract={contract} draft={draft} scopeCategoryId={lastEdited?.categoryId ?? selectedId} onClose={() => setTesting(false)} />
      )}

      {activation && (
        <ActivationDialog
          client={client}
          contract={contract}
          session={session}
          version={activation.version}
          taxonomy={activation.taxonomy}
          node={activation.node}
          recipeChanged={activation.recipeChanged}
          alertRules={alertRules}
          onClose={() => setActivation(null)}
          onDone={() => {
            void qc.invalidateQueries({ queryKey: queryKeys.signalAlertRules });
            void qc.invalidateQueries({ queryKey: queryKeys.reviewQueue });
            pollNow();
          }}
        />
      )}
    </div>
  );
}
