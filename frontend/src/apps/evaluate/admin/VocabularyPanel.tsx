// Admin -> Vocabulary (docs/DualAsr.md section 8, "manage_vocabulary"): the switch, the installed
// industry pack (read-only terms, each with a switch to turn it off) and the customer's own terms,
// saved together as one `AsrVocabularySettings` document with optimistic-concurrency (`record_version`).
//
// Dual transcription (Whisper Small prompted with this vocabulary, merged onto the Parakeet
// transcript) runs only when the switch is on AND the effective vocabulary is non-empty.

import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { BookMarked, Plus, Save, Tag, X } from 'lucide-react';
import {
  StoreError,
  effectiveVocabularyPreview,
  isNotImplemented,
  isVersionConflict,
  normalizeVocabularyTerm,
  queryKeys,
  vocabularySaveReasonText,
  vocabularyTermKey,
  vocabularyTermProblem,
  vocabularyTermProblemText,
  type AsrVocabularyRecord,
} from '../api';
import { Button, Card, Checkbox, ErrorNotice, Loading, Notice, NotBuiltYet, StatusPill, TextInput } from '../components/ui';
import { useSession, useStore } from '../state/app';

// Slack for punctuation the term rule allows beyond the raw character cap, so the field never
// clips a term the rule would still accept.
const VOCABULARY_TERM_INPUT_MAX = 80;

export function VocabularyPanel() {
  const { client, contract } = useStore();
  const session = useSession();
  const qc = useQueryClient();
  const readOnly = !session.can('manage_vocabulary');
  const caps = contract.parameters;

  const query = useQuery({
    queryKey: queryKeys.asrVocabulary,
    queryFn: ({ signal }) => client.get('/store/v1/vocabulary', { signal }),
    retry: false,
  });

  if (query.isLoading) return <Loading label="Loading the vocabulary…" />;
  if (isNotImplemented(query.error)) {
    return (
      <NotBuiltYet what="The vocabulary API">
        Store has not built <code>getAsrVocabulary</code>/<code>saveAsrVocabulary</code> yet
        (docs/DualAsr.md). Nothing here is hidden or simulated.
      </NotBuiltYet>
    );
  }
  if (query.isError) return <ErrorNotice error={query.error} />;
  if (!query.data) return null;

  return <VocabularyEditor client={client} caps={caps} record={query.data} readOnly={readOnly} onSaved={(next) => {
    qc.setQueryData(queryKeys.asrVocabulary, next);
  }} />;
}

function VocabularyEditor({
  client,
  caps,
  record,
  readOnly,
  onSaved,
}: {
  client: ReturnType<typeof useStore>['client'];
  caps: ReturnType<typeof useStore>['contract']['parameters'];
  record: AsrVocabularyRecord;
  readOnly: boolean;
  onSaved(next: AsrVocabularyRecord): void;
}) {
  const qc = useQueryClient();
  const [enabled, setEnabled] = useState(record.settings.enabled);
  const [customerTerms, setCustomerTerms] = useState<string[]>(record.settings.customer_terms);
  const [disabledPackTerms, setDisabledPackTerms] = useState<string[]>(record.settings.disabled_pack_terms);
  const [newTerm, setNewTerm] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // A save elsewhere reset the record: reload the draft onto it rather than fighting the version.
  useEffect(() => {
    setEnabled(record.settings.enabled);
    setCustomerTerms(record.settings.customer_terms);
    setDisabledPackTerms(record.settings.disabled_pack_terms);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record.record_version]);

  const packTerms = useMemo(() => record.pack?.terms ?? [], [record.pack]);
  const effective = useMemo(() => effectiveVocabularyPreview(packTerms, disabledPackTerms, customerTerms), [packTerms, disabledPackTerms, customerTerms]);
  const active = enabled && effective.length > 0;
  const disabledSet = useMemo(() => new Set(disabledPackTerms.map(vocabularyTermKey)), [disabledPackTerms]);

  const dirty =
    enabled !== record.settings.enabled ||
    JSON.stringify(customerTerms) !== JSON.stringify(record.settings.customer_terms) ||
    JSON.stringify(disabledPackTerms) !== JSON.stringify(record.settings.disabled_pack_terms);

  const newTermNormalized = normalizeVocabularyTerm(newTerm);
  const newTermProblem = newTerm.trim() ? vocabularyTermProblem(newTermNormalized) : null;
  const newTermDuplicate = !newTermProblem && customerTerms.some((t) => vocabularyTermKey(t) === vocabularyTermKey(newTermNormalized));
  const atCap = customerTerms.length >= caps.max_vocabulary_terms;

  function addTerm() {
    if (readOnly || !newTerm.trim() || newTermProblem || newTermDuplicate || atCap) return;
    setCustomerTerms((terms) => [...terms, newTermNormalized]);
    setNewTerm('');
    setNotice(null);
  }

  function removeTerm(term: string) {
    setCustomerTerms((terms) => terms.filter((t) => t !== term));
    setNotice(null);
  }

  function togglePackTerm(term: string, on: boolean) {
    const key = vocabularyTermKey(term);
    setDisabledPackTerms((disabled) => (on ? disabled.filter((t) => vocabularyTermKey(t) !== key) : [...disabled, term]));
    setNotice(null);
  }

  function resetDraft() {
    setEnabled(record.settings.enabled);
    setCustomerTerms(record.settings.customer_terms);
    setDisabledPackTerms(record.settings.disabled_pack_terms);
    setError(null);
    setNotice(null);
  }

  async function save() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const before = record.record_version;
      const next = await client.put('/store/v1/vocabulary', {
        body: {
          expected_record_version: record.record_version,
          settings: { enabled, customer_terms: customerTerms, disabled_pack_terms: disabledPackTerms },
        },
      });
      onSaved(next);
      setNotice(next.record_version === before ? 'Nothing changed, so no new version was saved.' : `Saved as v${next.record_version}.`);
    } catch (err) {
      setError(err);
      if (isVersionConflict(err)) {
        const latest = await qc.fetchQuery({
          queryKey: queryKeys.asrVocabulary,
          queryFn: ({ signal }) => client.get('/store/v1/vocabulary', { signal }),
          staleTime: 0,
        });
        onSaved(latest);
      }
    } finally {
      setBusy(false);
    }
  }

  const refusedReason = error instanceof StoreError && error.code === 'validation_failed' ? error.details.reason : undefined;

  return (
    <div className="space-y-4">
      <Card
        title="Dual transcription"
        icon={BookMarked}
        subtitle="A second transcription pass listens for these terms and fixes words the main transcript misheard. It only replaces a word that matches by sound and spelling, and never adds words."
        right={active ? <StatusPill tone="green">On</StatusPill> : <StatusPill tone="neutral">{enabled ? 'Off: the vocabulary is empty' : 'Off'}</StatusPill>}
      >
        <Checkbox
          label="Correct transcripts with this vocabulary"
          hint="Runs only while this is on and at least one term is active below."
          checked={enabled}
          disabled={readOnly}
          onChange={(e) => {
            setEnabled(e.target.checked);
            setNotice(null);
          }}
        />
      </Card>

      {record.pack && (
        <Card
          title={record.pack.title}
          icon={Tag}
          subtitle={`${record.pack.industry} pack, v${record.pack.version} — ${record.pack.terms.length} terms. Read-only; switch individual terms off if one causes false corrections.`}
        >
          <ul className="flex flex-wrap gap-1.5">
            {packTerms.map((term) => {
              const on = !disabledSet.has(vocabularyTermKey(term));
              return (
                <li key={term}>
                  <label
                    className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full border text-xs font-medium cursor-pointer ${
                      on ? 'bg-primer-blueSubtle text-primer-blueFg border-primer-blueBorder' : 'bg-canvas-inset text-fg-subtle border-border line-through'
                    } ${readOnly ? 'cursor-default' : ''}`}
                  >
                    <input
                      type="checkbox"
                      checked={on}
                      disabled={readOnly}
                      onChange={(e) => togglePackTerm(term, e.target.checked)}
                      className="accent-primer-blue focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                      aria-label={`${term}: ${on ? 'on' : 'off'}`}
                    />
                    {term}
                  </label>
                </li>
              );
            })}
          </ul>
        </Card>
      )}

      <Card
        title="Your terms"
        icon={Tag}
        subtitle="Business terms only: products, brands, stores, services. No people's names and no numbers."
        right={
          <span className={`text-xs tabular-nums ${atCap ? 'text-primer-redFg font-medium' : 'text-fg-subtle'}`}>
            {customerTerms.length}/{caps.max_vocabulary_terms}
          </span>
        }
      >
        {!readOnly && (
          <div className="flex items-start gap-2 mb-3">
            <div className="flex-1">
              <TextInput
                aria-label="Add a term"
                placeholder="e.g. Stanley cup"
                value={newTerm}
                maxLength={VOCABULARY_TERM_INPUT_MAX}
                disabled={atCap}
                onChange={(e) => setNewTerm(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addTerm();
                  }
                }}
              />
              {newTerm.trim() && newTermProblem && <p className="text-xs text-primer-redFg mt-1">{vocabularyTermProblemText(newTermProblem)}</p>}
              {newTerm.trim() && !newTermProblem && newTermDuplicate && <p className="text-xs text-primer-yellowFg mt-1">Already in your terms.</p>}
            </div>
            <Button icon={Plus} disabled={!newTerm.trim() || !!newTermProblem || newTermDuplicate || atCap} onClick={addTerm}>
              Add
            </Button>
          </div>
        )}
        {atCap && !readOnly && (
          <p className="text-xs text-primer-yellowFg mb-2" role="status">
            At most {caps.max_vocabulary_terms} of your own terms. Remove one to add another.
          </p>
        )}
        {customerTerms.length === 0 ? (
          <p className="text-sm text-fg-muted">No terms yet.</p>
        ) : (
          <ul className="flex flex-wrap gap-1.5">
            {customerTerms.map((term) => (
              <li key={term}>
                <span className="inline-flex items-center gap-1 pl-2 pr-1 py-1 rounded-full border border-border bg-canvas text-xs font-medium text-fg">
                  {term}
                  {!readOnly && (
                    <button
                      type="button"
                      onClick={() => removeTerm(term)}
                      aria-label={`Remove ${term}`}
                      className="w-4 h-4 rounded-full hover:bg-canvas-inset flex items-center justify-center focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                    >
                      <X className="w-3 h-3" aria-hidden="true" />
                    </button>
                  )}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      {!readOnly && (
        <div className={`${dirty ? 'sticky bottom-0 z-10 shadow-lg' : ''} rounded-lg border border-border bg-canvas p-3 space-y-2`} data-testid="vocabulary-save-bar">
          <div className="flex flex-wrap items-center gap-2">
            {dirty ? <StatusPill tone="yellow">Unsaved changes</StatusPill> : <StatusPill tone="green">Saved as v{record.record_version}</StatusPill>}
            <span className="text-xs text-fg-muted">{effective.length} effective term{effective.length === 1 ? '' : 's'}</span>
            <span className="flex-1" />
            <Button variant="ghost" disabled={!dirty || busy} onClick={resetDraft}>
              Discard
            </Button>
            <Button icon={Save} variant="primary" busy={busy} disabled={!dirty} onClick={() => void save()}>
              Save
            </Button>
          </div>
          {notice && (
            <Notice tone="green" icon={Save}>
              {notice}
            </Notice>
          )}
          <ErrorNotice error={error}>
            {refusedReason !== undefined && <span className="block mt-1">{vocabularySaveReasonText(refusedReason, caps.max_vocabulary_terms)}</span>}
            {isVersionConflict(error) && <span className="block mt-1">The latest saved settings are loaded above; check them and save again.</span>}
          </ErrorNotice>
        </div>
      )}
    </div>
  );
}
