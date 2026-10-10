// The category editor of the Signals page (docs/ContactSignalsV2.md §10.1): a category's own text
// and settings, its subcategories (with the fixed "Other" and "Not <category>" options), and the
// extraction fields on the category and on each subcategory.
//
// Built-in categories keep their Call1 name, gloss and speaker; an admin changes only their
// examples, thresholds, narrow_quote, subcategories and fields (§7.2). IDs are fixed when a node is
// added and never change: there is no delete, `active: false` retires a node, and a node added in
// this session can be removed again until it is saved.

import { useState } from 'react';
import { ArrowDown, ArrowUp, ChevronDown, ChevronRight, Lock, Plus, Trash2 } from 'lucide-react';
import {
  FIELD_TYPES,
  FIELD_TYPE_LABEL,
  FORBIDDEN_PII_CLASSES,
  PII_CLASSES,
  PII_CLASS_LABEL,
  RESERVED_FIELD_IDS,
  RESERVED_SUBCATEGORY_IDS,
  slugId,
  speakerScopeLabel,
  syncRecipeSpeaker,
  type ContractParameters,
  type FieldPiiClass,
  type CatalogSnapshot,
  type SignalCategory,
  type SignalField,
  type SignalFieldType,
  type SignalSubcategory,
  type SpeakerRole,
  type TaxonomyProblem,
} from '../../api';
import { Button, CharCount, Checkbox, Chip, Field, Notice, SelectInput, StatusPill, TextArea, TextInput } from '../../components/ui';
import { RecipeEditor } from './RecipeEditor';

export type CapParams = Pick<
  ContractParameters,
  'max_custom_signal_categories' | 'max_active_subcategories' | 'max_fields_per_path' | 'max_option_gloss_chars' | 'max_signal_recipe_rules' | 'max_signal_lexicon_phrases'
>;

export const CALLER_DETAILS_WARNING = "Don't paste caller details. Describe the behavior.";

/** One stage-3 extraction entry and where its (masked) span text goes (§11.1). */
export interface ExtractionDestination {
  installationId: string;
  entryId: string;
  displayName: string;
  routeClass: string;
  destinationHost: string;
  role: 'default' | 'fallback';
}

const ROUTE_CLASS_LABEL: Record<string, string> = {
  appliance: 'on the appliance',
  customer_lan: 'customer LAN',
  customer_directed: 'customer-directed provider (BYOK)',
};

/** The stage-3 extraction entries each Process host would use: its `signal_extraction` default and
 * the taxonomy's fallback entry, when that host has it (§11.1: the editor shows each extraction
 * entry's destination host). */
export function extractionDestinations(snapshots: CatalogSnapshot[], fallbackEntryId: string | null | undefined): ExtractionDestination[] {
  const out: ExtractionDestination[] = [];
  for (const snap of snapshots) {
    const picks: Array<[string | undefined, 'default' | 'fallback']> = [
      [snap.defaults?.signal_extraction?.entry_id, 'default'],
      [fallbackEntryId ?? undefined, 'fallback'],
    ];
    for (const [entryId, role] of picks) {
      if (!entryId) continue;
      const entry = snap.entries.find((e) => e.entry.entry_id === entryId);
      if (!entry || out.some((d) => d.installationId === snap.installation_id && d.entryId === entryId)) continue;
      out.push({
        installationId: snap.installation_id,
        entryId,
        displayName: entry.display_name,
        routeClass: entry.route_class,
        destinationHost: entry.destination_host,
        role,
      });
    }
  }
  return out;
}

function ExtractionDestinations({ destinations }: { destinations: ExtractionDestination[] | undefined }) {
  if (destinations === undefined) return null;
  return (
    <div data-testid="signals-extraction-destinations" className="text-xs text-fg-muted space-y-1">
      <p>Fields are filled by the stage-3 extractor. Span text, field descriptions and taxonomy text reach it masked.</p>
      {destinations.length === 0 ? (
        <p>No Process host reports a signal extraction entry yet.</p>
      ) : (
        <ul className="space-y-0.5">
          {destinations.map((d) => (
            <li key={`${d.installationId}:${d.entryId}`}>
              <span className="font-medium text-fg">{d.displayName}</span>
              {d.role === 'fallback' ? ' (fallback)' : ''}: sends to{' '}
              <span className="font-mono break-all">{d.destinationHost}</span>, {ROUTE_CLASS_LABEL[d.routeClass] ?? d.routeClass}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface EditorProps {
  category: SignalCategory;
  saved: SignalCategory | undefined;
  readOnly: boolean;
  caps: CapParams;
  problems: TaxonomyProblem[];
  /** Calls with a hit in the last 7 days, per subcategory ID (from signal metrics). */
  weekBySubcategory: Map<string, number>;
  onChange(update: (c: SignalCategory) => void, subcategoryId?: string): void;
  /** Remove a category added in this session (never a saved one: there is no delete). */
  onRemove(): void;
  /** Stage-3 extraction entries and their destination hosts; undefined when the catalogs are not readable. */
  extraction?: ExtractionDestination[];
}

/** One category's editor. `onChange` mutates a copy of the draft; `subcategoryId` names the edited
 * subcategory, so the activation dialog knows which node to alert on. */
export function CategoryEditor({ category: c, saved, readOnly, caps, problems, weekBySubcategory, onChange, onRemove, extraction }: EditorProps) {
  const builtin = c.builtin;
  const isNew = saved === undefined;
  const glossMax = caps.max_option_gloss_chars;
  const activeSubs = c.subcategories.filter((s) => s.active).length;
  const [openSub, setOpenSub] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState('');
  const [newGloss, setNewGloss] = useState('');

  const subCapReached = activeSubs >= caps.max_active_subcategories;
  const totalSubCapReached = c.subcategories.length >= 24;

  function addSubcategory() {
    const name = newName.trim();
    if (!name) return;
    const id = slugId(name, new Set(c.subcategories.map((s) => s.subcategory_id)), RESERVED_SUBCATEGORY_IDS);
    onChange((cat) => {
      cat.subcategories.push({ subcategory_id: id, name, gloss: newGloss.trim(), description: null, examples: [], fields: [], narrow_quote: null, active: true });
    }, id);
    setOpenSub(id);
    setAdding(false);
    setNewName('');
    setNewGloss('');
  }

  const categoryProblems = problems.filter((p) => !/\.subcategories\[/.test(p.path) && !/\.recipe\b/.test(p.path));
  const recipeProblemList = problems.filter((p) => /^categories\[\d+\]\.recipe\b/.test(p.path));

  return (
    <div className="space-y-4" data-testid="signals-category-editor">
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="text-base font-semibold text-fg">{c.name || 'New category'}</h2>
        {builtin ? (
          <StatusPill tone="neutral" title="Call1's fixed category: its name, gloss and speaker cannot change">
            <Lock className="w-3 h-3" aria-hidden="true" />
            Built-in
          </StatusPill>
        ) : (
          <StatusPill tone="magenta">Custom</StatusPill>
        )}
        {!c.active && <StatusPill tone="neutral">Retired</StatusPill>}
        {isNew && <StatusPill tone="yellow">New, not saved</StatusPill>}
        <span className="text-xs text-fg-muted">
          ID <span className="text-fg">{c.category_id}</span> · {speakerScopeLabel(c.speaker)}
        </span>
      </div>


      {categoryProblems.length > 0 && <ProblemList problems={categoryProblems} />}

      <section className="grid grid-cols-1 sm:grid-cols-2 gap-3" aria-label="Category">
        {builtin ? (
          <>
            <Field label="Name" hint="Fixed by Call1.">
              {(id) => <TextInput id={id} value={c.name} disabled readOnly />}
            </Field>
            <ReadOnlyText label="Speaker" value={speakerScopeLabel(c.speaker)} />
            <div className="sm:col-span-2">
              <Field label="Gloss" hint="The option text the classifier reads for this category. Fixed by Call1.">
                {(id) => <TextInput id={id} value={c.gloss} disabled readOnly />}
              </Field>
            </div>
          </>
        ) : (
          <>
            <Field label="Name" hint={<CharCount value={c.name} max={40} />}>
              {(id) => (
                <TextInput id={id} value={c.name} disabled={readOnly} maxLength={60} onChange={(e) => onChange((cat) => void (cat.name = e.target.value))} />
              )}
            </Field>
            <Field label="Speaker" hint="Whose turns this category is checked on.">
              {(id) => (
                <SelectInput
                  id={id}
                  value={c.speaker ?? ''}
                  disabled={readOnly}
                  onChange={(e) =>
                    onChange((cat) => {
                      cat.speaker = (e.target.value || null) as SpeakerRole | null;
                      syncRecipeSpeaker(cat);
                    })
                  }
                >
                  <option value="">Either</option>
                  <option value="AGENT">Agent</option>
                  <option value="CALLER">Caller</option>
                </SelectInput>
              )}
            </Field>
            <div className="sm:col-span-2">
              <Field
                label="Gloss"
                hint={
                  <span className="flex flex-wrap justify-between gap-2">
                    <span>The option text the classifier reads for every segment. Keep it short and concrete.</span>
                    <CharCount value={c.gloss} max={glossMax} />
                  </span>
                }
              >
                {(id) => (
                  <TextInput id={id} value={c.gloss} disabled={readOnly} maxLength={80} onChange={(e) => onChange((cat) => void (cat.gloss = e.target.value))} />
                )}
              </Field>
            </div>
            <div className="sm:col-span-2">
              <Field label="Description" hint={<span className="flex justify-between gap-2"><span>For reviewers and the field extractor. Optional.</span><CharCount value={c.description} max={240} /></span>}>
                {(id) => (
                  <TextArea
                    id={id}
                    value={c.description ?? ''}
                    disabled={readOnly}
                    maxLength={300}
                    onChange={(e) => onChange((cat) => void (cat.description = e.target.value.replace(/[\r\n]+/g, ' ') || null))}
                  />
                )}
              </Field>
            </div>
          </>
        )}
        <ThresholdInput
          label="Threshold"
          value={c.threshold}
          readOnly={readOnly}
          onChange={(v) => onChange((cat) => void (cat.threshold = v))}
        />
        <ThresholdInput
          label="Subcategory threshold"
          value={c.subcategory_threshold}
          readOnly={readOnly}
          onChange={(v) => onChange((cat) => void (cat.subcategory_threshold = v))}
        />
        <div className="sm:col-span-2">
          <Checkbox
            label="Narrow the quote"
            hint="The extractor trims the quote to the words that carry the signal. Runs the field stage on every span of this category."
            checked={c.narrow_quote}
            disabled={readOnly}
            onChange={(e) => onChange((cat) => void (cat.narrow_quote = e.target.checked))}
          />
        </div>
        <div className="sm:col-span-2">
          <ExamplesEditor label="Category examples" examples={c.examples} readOnly={readOnly} onChange={(next) => onChange((cat) => void (cat.examples = next))} />
        </div>
        {!builtin && !readOnly && (
          <div className="sm:col-span-2 flex flex-wrap items-center gap-2">
            {isNew ? (
              <Button size="sm" variant="danger" icon={Trash2} onClick={onRemove}>
                Remove new category
              </Button>
            ) : c.active ? (
              <Button size="sm" onClick={() => onChange((cat) => void (cat.active = false))}>
                Retire category
              </Button>
            ) : (
              <Button size="sm" onClick={() => onChange((cat) => void (cat.active = true))}>
                Reactivate category
              </Button>
            )}
            <span className="text-xs text-fg-subtle">
              {isNew ? 'Not saved yet, so it can still be removed.' : 'There is no delete: a retired category keeps its results and stops running.'}
            </span>
          </div>
        )}
      </section>

      <RecipeEditor
        category={c}
        saved={saved}
        readOnly={readOnly}
        caps={caps}
        problems={recipeProblemList}
        onChange={(update) => onChange(update)}
      />

      <section aria-label="Fields" className="space-y-2">
        <h3 className="text-sm font-semibold text-fg">Category fields</h3>
        <ExtractionDestinations destinations={extraction} />
        <FieldList
          fields={c.fields}
          savedFields={saved?.fields}
          readOnly={readOnly}
          takenIds={[]}
          pathCount={c.fields.length}
          caps={caps}
          scope="category"
          onChange={(next) => onChange((cat) => void (cat.fields = next))}
        />
      </section>

      <section aria-label="Subcategories" className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-fg">
            Subcategories <span className="text-xs font-normal text-fg-muted">({activeSubs} active of {caps.max_active_subcategories})</span>
          </h3>
          {!readOnly && (
            <Button size="sm" icon={Plus} disabled={subCapReached || totalSubCapReached} onClick={() => setAdding((a) => !a)}>
              Add subcategory
            </Button>
          )}
        </div>
        {!readOnly && subCapReached && (
          <p className="text-xs text-primer-yellowFg" role="status">
            At most {caps.max_active_subcategories} active subcategories per category: the classifier compares them all at once. Deactivate one to add another.
          </p>
        )}
        {!readOnly && !subCapReached && totalSubCapReached && (
          <p className="text-xs text-primer-yellowFg" role="status">
            This category already holds 24 subcategories, active or retired, the most it can hold.
          </p>
        )}
        {adding && (
          <div className="rounded-md border border-border-muted bg-canvas p-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
            <Field label="Name" hint={<CharCount value={newName} max={40} />}>
              {(id) => <TextInput id={id} value={newName} maxLength={60} autoFocus onChange={(e) => setNewName(e.target.value)} />}
            </Field>
            <Field label="Gloss" hint={<CharCount value={newGloss} max={glossMax} />}>
              {(id) => (
                <TextInput
                  id={id}
                  value={newGloss}
                  maxLength={80}
                  onChange={(e) => setNewGloss(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') addSubcategory();
                  }}
                />
              )}
            </Field>
            <p className="text-xs text-fg-subtle sm:col-span-2">
              The gloss is the option text the classifier picks from, e.g. "Caller wants to cancel their account". The ID is set from the name and never changes.
            </p>
            <div className="flex gap-2 sm:col-span-2">
              <Button size="sm" variant="primary" disabled={!newName.trim()} onClick={addSubcategory}>
                Add
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setAdding(false)}>
                Cancel
              </Button>
            </div>
          </div>
        )}

        <ul className="space-y-1.5">
          {c.subcategories.map((s, j) => {
            const open = openSub === s.subcategory_id;
            const savedSub = saved?.subcategories.find((x) => x.subcategory_id === s.subcategory_id);
            const subProblems = problems.filter((p) => p.path.includes(`.subcategories[${j}]`));
            const week = weekBySubcategory.get(s.subcategory_id);
            return (
              <li key={s.subcategory_id} className="rounded-md border border-border-muted bg-canvas" data-subcategory={s.subcategory_id}>
                <div className="flex flex-wrap items-center gap-2 px-2.5 py-2">
                  <button
                    type="button"
                    aria-expanded={open}
                    onClick={() => setOpenSub(open ? null : s.subcategory_id)}
                    className="flex items-center gap-1.5 min-w-0 flex-1 text-left rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                  >
                    {open ? <ChevronDown className="w-3.5 h-3.5 shrink-0" aria-hidden="true" /> : <ChevronRight className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />}
                    <span className="font-medium text-sm text-fg truncate">{s.name || s.subcategory_id}</span>
                    <span className="text-xs text-fg-muted truncate">{s.gloss}</span>
                  </button>
                  {!s.active && <StatusPill tone="neutral">Inactive</StatusPill>}
                  {!savedSub && <StatusPill tone="yellow">New</StatusPill>}
                  {subProblems.length > 0 && <StatusPill tone="red">{subProblems.length === 1 ? '1 problem' : `${subProblems.length} problems`}</StatusPill>}
                  <span className="text-xs text-fg-subtle whitespace-nowrap">
                    {s.fields.length} field{s.fields.length === 1 ? '' : 's'}
                    {week !== undefined && ` · ${week} call${week === 1 ? '' : 's'} in 7 days`}
                  </span>
                  {!readOnly && (
                    <span className="flex items-center gap-1">
                      <IconButton label={`Move ${s.name} up`} icon={ArrowUp} disabled={j === 0} onClick={() => onChange((cat) => move(cat.subcategories, j, -1), s.subcategory_id)} />
                      <IconButton
                        label={`Move ${s.name} down`}
                        icon={ArrowDown}
                        disabled={j === c.subcategories.length - 1}
                        onClick={() => onChange((cat) => move(cat.subcategories, j, 1), s.subcategory_id)}
                      />
                    </span>
                  )}
                </div>
                {open && (
                  <SubcategoryEditor
                    sub={s}
                    saved={savedSub}
                    categoryFields={c.fields}
                    readOnly={readOnly}
                    caps={caps}
                    problems={subProblems}
                    subCapReached={subCapReached}
                    onChange={(update) =>
                      onChange((cat) => {
                        const target = cat.subcategories.find((x) => x.subcategory_id === s.subcategory_id);
                        if (target) update(target);
                      }, s.subcategory_id)
                    }
                    onRemove={() => onChange((cat) => void (cat.subcategories = cat.subcategories.filter((x) => x.subcategory_id !== s.subcategory_id)))}
                  />
                )}
              </li>
            );
          })}
          <li className="rounded-md border border-dashed border-border-muted px-2.5 py-2 flex flex-wrap items-center gap-2 text-sm">
            <Lock className="w-3 h-3 text-fg-subtle" aria-hidden="true" />
            <span className="font-medium text-fg">Other</span>
            <span className="text-xs text-fg-muted">Fixed: a real {c.name || 'signal'} that fits none of the subcategories.</span>
          </li>
          <li className="rounded-md border border-dashed border-border-muted px-2.5 py-2 flex flex-wrap items-center gap-2 text-sm">
            <Lock className="w-3 h-3 text-fg-subtle" aria-hidden="true" />
            <span className="font-medium text-fg">Not {c.name || 'this category'}</span>
            <span className="text-xs text-fg-muted">Fixed: rejects a span that only looked like one. This is the precision check.</span>
          </li>
        </ul>
      </section>
    </div>
  );
}

function move<T>(list: T[], index: number, delta: number) {
  const to = index + delta;
  if (to < 0 || to >= list.length) return;
  const [item] = list.splice(index, 1);
  list.splice(to, 0, item);
}

function SubcategoryEditor({
  sub: s,
  saved,
  categoryFields,
  readOnly,
  caps,
  problems,
  subCapReached,
  onChange,
  onRemove,
}: {
  sub: SignalSubcategory;
  saved: SignalSubcategory | undefined;
  categoryFields: SignalField[];
  readOnly: boolean;
  caps: CapParams;
  problems: TaxonomyProblem[];
  subCapReached: boolean;
  onChange(update: (s: SignalSubcategory) => void): void;
  onRemove(): void;
}) {
  const glossMax = caps.max_option_gloss_chars;
  return (
    <div className="border-t border-border-muted p-3 space-y-3" data-testid="signals-subcategory-editor">
      {problems.length > 0 && <ProblemList problems={problems} />}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Field label="Name" hint={<CharCount value={s.name} max={40} />}>
          {(id) => <TextInput id={id} value={s.name} disabled={readOnly} maxLength={60} onChange={(e) => onChange((x) => void (x.name = e.target.value))} />}
        </Field>
        <Field label="Quote narrowing" hint="Inherit follows the category's setting.">
          {(id) => (
            <SelectInput
              id={id}
              value={s.narrow_quote === null ? '' : s.narrow_quote ? 'on' : 'off'}
              disabled={readOnly}
              onChange={(e) => onChange((x) => void (x.narrow_quote = e.target.value === '' ? null : e.target.value === 'on'))}
            >
              <option value="">Inherit from category</option>
              <option value="on">Narrow the quote</option>
              <option value="off">Keep the whole span</option>
            </SelectInput>
          )}
        </Field>
        <div className="sm:col-span-2">
          <Field
            label="Gloss"
            hint={
              <span className="flex flex-wrap justify-between gap-2">
                <span>The option text the classifier picks from on each span.</span>
                <CharCount value={s.gloss} max={glossMax} />
              </span>
            }
          >
            {(id) => <TextInput id={id} value={s.gloss} disabled={readOnly} maxLength={80} onChange={(e) => onChange((x) => void (x.gloss = e.target.value))} />}
          </Field>
        </div>
        <div className="sm:col-span-2">
          <Field label="Description" hint={<span className="flex justify-between gap-2"><span>For reviewers and the field extractor. Optional.</span><CharCount value={s.description} max={240} /></span>}>
            {(id) => (
              <TextArea
                id={id}
                value={s.description ?? ''}
                disabled={readOnly}
                maxLength={300}
                onChange={(e) => onChange((x) => void (x.description = e.target.value.replace(/[\r\n]+/g, ' ') || null))}
              />
            )}
          </Field>
        </div>
        <div className="sm:col-span-2">
          <ExamplesEditor label="Subcategory examples" examples={s.examples} readOnly={readOnly} onChange={(next) => onChange((x) => void (x.examples = next))} />
        </div>
      </div>
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-fg-muted uppercase tracking-wide">Subcategory fields</h4>
        <FieldList
          fields={s.fields}
          savedFields={saved?.fields}
          readOnly={readOnly}
          takenIds={categoryFields.map((f) => f.field_id)}
          pathCount={categoryFields.length + s.fields.length}
          caps={caps}
          scope="subcategory"
          onChange={(next) => onChange((x) => void (x.fields = next))}
        />
      </div>
      {!readOnly && (
        <div className="flex flex-wrap items-center gap-2">
          {!saved ? (
            <Button size="sm" variant="danger" icon={Trash2} onClick={onRemove}>
              Remove new subcategory
            </Button>
          ) : s.active ? (
            <Button size="sm" onClick={() => onChange((x) => void (x.active = false))}>
              Deactivate subcategory
            </Button>
          ) : (
            <Button size="sm" disabled={subCapReached} onClick={() => onChange((x) => void (x.active = true))}>
              Reactivate subcategory
            </Button>
          )}
          <span className="text-xs text-fg-subtle">ID {s.subcategory_id} (permanent)</span>
        </div>
      )}
    </div>
  );
}

// --- fields --------------------------------------------------------------------------------------

function FieldList({
  fields,
  savedFields,
  readOnly,
  takenIds,
  pathCount,
  caps,
  scope,
  onChange,
}: {
  fields: SignalField[];
  savedFields: SignalField[] | undefined;
  readOnly: boolean;
  takenIds: string[];
  pathCount: number;
  caps: CapParams;
  scope: 'category' | 'subcategory';
  onChange(next: SignalField[]): void;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState('');
  const pathCap = pathCount >= caps.max_fields_per_path;
  const nodeCap = fields.length >= 8;

  function add() {
    const n = name.trim();
    if (!n) return;
    const id = slugId(n, new Set([...takenIds, ...fields.map((f) => f.field_id)]), RESERVED_FIELD_IDS);
    onChange([...fields, { field_id: id, name: n, type: 'string', description: '', enum_values: [], pii_class: 'none' }]);
    setOpen(id);
    setAdding(false);
    setName('');
  }

  return (
    <div className="space-y-1.5">
      {fields.length === 0 && (
        <p className="text-xs text-fg-subtle">
          No fields{scope === 'subcategory' ? ' of its own' : ''}. Fields are optional: without them (and without quote narrowing) the extractor never runs.
        </p>
      )}
      {fields.map((f, k) => {
        const isOpen = open === f.field_id;
        const isNew = !savedFields?.some((x) => x.field_id === f.field_id);
        return (
          <div key={f.field_id} className="rounded-md border border-border-muted bg-canvas-subtle" data-field={f.field_id}>
            <button
              type="button"
              aria-expanded={isOpen}
              onClick={() => setOpen(isOpen ? null : f.field_id)}
              className="w-full flex flex-wrap items-center gap-2 px-2.5 py-1.5 text-left rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
            >
              {isOpen ? <ChevronDown className="w-3.5 h-3.5" aria-hidden="true" /> : <ChevronRight className="w-3.5 h-3.5" aria-hidden="true" />}
              <span className="text-sm font-medium text-fg">{f.name || f.field_id}</span>
              <Chip>{FIELD_TYPE_LABEL[f.type]}</Chip>
              {f.type === 'enum' && f.enum_values.length > 0 && <span className="text-xs text-fg-muted truncate">{f.enum_values.join(', ')}</span>}
              <span className="text-xs text-fg-subtle">PII: {PII_CLASS_LABEL[f.pii_class]}</span>
              {isNew && <StatusPill tone="yellow">New</StatusPill>}
            </button>
            {isOpen && (
              <FieldEditor
                field={f}
                isNew={isNew}
                readOnly={readOnly}
                onChange={(update) => {
                  const next = fields.map((x) => ({ ...x, enum_values: [...x.enum_values] }));
                  update(next[k]);
                  onChange(next);
                }}
                onRemove={() => onChange(fields.filter((x) => x.field_id !== f.field_id))}
              />
            )}
          </div>
        );
      })}
      {!readOnly && (
        <div className="space-y-1.5">
          {adding ? (
            <div className="flex flex-wrap items-end gap-2">
              <Field label="Field name" hint={<CharCount value={name} max={40} />}>
                {(id) => (
                  <TextInput
                    id={id}
                    value={name}
                    maxLength={60}
                    autoFocus
                    className="w-56"
                    onChange={(e) => setName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') add();
                    }}
                  />
                )}
              </Field>
              <Button size="sm" variant="primary" disabled={!name.trim()} onClick={add}>
                Add
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setAdding(false)}>
                Cancel
              </Button>
            </div>
          ) : (
            <Button size="sm" icon={Plus} disabled={pathCap || nodeCap} onClick={() => setAdding(true)}>
              Add {scope} field
            </Button>
          )}
          {(pathCap || nodeCap) && (
            <p className="text-xs text-primer-yellowFg" role="status">
              {nodeCap
                ? `A ${scope} holds at most 8 fields of its own.`
                : `At most ${caps.max_fields_per_path} fields on a category and subcategory path: the extractor fills them all in one pass.`}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function FieldEditor({
  field: f,
  isNew,
  readOnly,
  onChange,
  onRemove,
}: {
  field: SignalField;
  isNew: boolean;
  readOnly: boolean;
  onChange(update: (f: SignalField) => void): void;
  onRemove(): void;
}) {
  const [value, setValue] = useState('');
  const forbidden = FORBIDDEN_PII_CLASSES.has(f.pii_class);
  function addValue() {
    const v = value.trim();
    if (!v) return;
    onChange((x) => void (x.enum_values = [...x.enum_values, v]));
    setValue('');
  }
  return (
    <div className="border-t border-border-muted p-3 grid grid-cols-1 sm:grid-cols-2 gap-3" data-testid="signals-field-editor">
      <Field label="Field name" hint={<CharCount value={f.name} max={40} />}>
        {(id) => <TextInput id={id} value={f.name} disabled={readOnly} maxLength={60} onChange={(e) => onChange((x) => void (x.name = e.target.value))} />}
      </Field>
      <Field label="Field type" hint={isNew ? undefined : 'Changing the type of a saved field reruns the field stage on update.'}>
        {(id) => (
          <SelectInput
            id={id}
            value={f.type}
            disabled={readOnly}
            onChange={(e) =>
              onChange((x) => {
                x.type = e.target.value as SignalFieldType;
                if (x.type !== 'enum') x.enum_values = [];
              })
            }
          >
            {FIELD_TYPES.map((t) => (
              <option key={t} value={t}>
                {FIELD_TYPE_LABEL[t]}
              </option>
            ))}
          </SelectInput>
        )}
      </Field>
      <div className="sm:col-span-2">
        <Field
          label="Field description"
          hint={
            <span className="flex flex-wrap justify-between gap-2">
              <span>This is the only instruction the extractor gets for this field.</span>
              <CharCount value={f.description} max={200} />
            </span>
          }
        >
          {(id) => (
            <TextInput
              id={id}
              value={f.description}
              disabled={readOnly}
              maxLength={240}
              onChange={(e) => onChange((x) => void (x.description = e.target.value.replace(/[\r\n]+/g, ' ')))}
            />
          )}
        </Field>
      </div>
      {f.type === 'enum' && (
        <div className="sm:col-span-2 space-y-1.5">
          <p className="text-xs font-medium text-fg-muted">Values ({f.enum_values.length} of 12)</p>
          <div className="flex flex-wrap gap-1.5">
            {f.enum_values.length === 0 && <span className="text-xs text-primer-yellowFg">Add at least one value.</span>}
            {f.enum_values.map((v, m) => (
              <span key={`${v}-${m}`} className="inline-flex items-center gap-1 rounded border border-border bg-canvas px-1.5 py-0.5 text-xs text-fg">
                {v}
                {!readOnly && (
                  <button
                    type="button"
                    aria-label={`Remove value ${v}`}
                    onClick={() => onChange((x) => void (x.enum_values = x.enum_values.filter((_, i) => i !== m)))}
                    className="text-fg-muted hover:text-fg rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                  >
                    <Trash2 className="w-3 h-3" aria-hidden="true" />
                  </button>
                )}
              </span>
            ))}
          </div>
          {!readOnly && f.enum_values.length < 12 && (
            <div className="flex items-end gap-2">
              <Field label="New value">
                {(id) => (
                  <TextInput
                    id={id}
                    value={value}
                    maxLength={40}
                    className="w-48"
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') addValue();
                    }}
                  />
                )}
              </Field>
              <Button size="sm" disabled={!value.trim()} onClick={addValue}>
                Add value
              </Button>
            </div>
          )}
        </div>
      )}
      <div className="sm:col-span-2">
        <Field
          label="PII class"
          hint="Caller name, account and card numbers, phone, email, address, URL, secrets and government IDs are masked before any model sees it, so they can't be fields."
        >
          {(id) => (
            <SelectInput id={id} value={f.pii_class} disabled={readOnly} onChange={(e) => onChange((x) => void (x.pii_class = e.target.value as FieldPiiClass))}>
              {PII_CLASSES.map((p) => {
                const blocked = FORBIDDEN_PII_CLASSES.has(p);
                return (
                  <option key={p} value={p} disabled={blocked && p !== f.pii_class}>
                    {PII_CLASS_LABEL[p]}
                    {blocked ? ' — masked before any model sees it' : ''}
                  </option>
                );
              })}
            </SelectInput>
          )}
        </Field>
        {forbidden && <p className="text-xs text-primer-redFg mt-1">{PII_CLASS_LABEL[f.pii_class]} is masked before any model sees it. Pick another class.</p>}
      </div>
      {!readOnly && (
        <div className="sm:col-span-2 flex items-center gap-2">
          {isNew ? (
            <Button size="sm" variant="danger" icon={Trash2} onClick={onRemove}>
              Remove new field
            </Button>
          ) : (
            <span className="text-xs text-fg-subtle">
              ID {f.field_id} (permanent). A saved field stays: results and alerts refer to it.
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// --- small pieces --------------------------------------------------------------------------------

function ExamplesEditor({ label, examples, readOnly, onChange }: { label: string; examples: string[]; readOnly: boolean; onChange(next: string[]): void }) {
  const [value, setValue] = useState('');
  const add = () => {
    const v = value.trim().replace(/\s+/g, ' ');
    if (!v) return;
    onChange([...examples, v]);
    setValue('');
  };
  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-fg-muted">
        {label} ({examples.length} of 5)
      </p>
      {examples.length === 0 ? (
        <p className="text-xs text-fg-subtle">None. Examples help people read the category and guide the extractor; the classifier reads only the gloss.</p>
      ) : (
        <ul className="space-y-1">
          {examples.map((ex, k) => (
            <li key={`${ex}-${k}`} className="flex items-center gap-2 text-sm">
              <span className="italic text-fg-muted flex-1 min-w-0 break-words">&ldquo;{ex}&rdquo;</span>
              {!readOnly && (
                <button
                  type="button"
                  aria-label={`Remove example ${k + 1}`}
                  onClick={() => onChange(examples.filter((_, i) => i !== k))}
                  className="text-fg-muted hover:text-fg rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                >
                  <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
      {!readOnly && examples.length < 5 && (
        <div className="flex items-end gap-2">
          <div className="flex-1">
            <Field label={`New ${label.toLowerCase().replace(/s$/, '')}`} hint={CALLER_DETAILS_WARNING}>
              {(id) => (
                <TextInput
                  id={id}
                  value={value}
                  maxLength={120}
                  onChange={(e) => setValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') add();
                  }}
                />
              )}
            </Field>
          </div>
          <Button size="sm" disabled={!value.trim()} onClick={add} className="mb-5">
            Add example
          </Button>
        </div>
      )}
    </div>
  );
}

function ThresholdInput({ label, value, readOnly, onChange }: { label: string; value: number | null; readOnly: boolean; onChange(v: number | null): void }) {
  const [text, setText] = useState(value === null ? '' : String(value));
  const [lastValue, setLastValue] = useState(value);
  if (value !== lastValue) {
    setLastValue(value);
    setText(value === null ? '' : String(value));
  }
  const n = Number(text);
  const invalid = text !== '' && (!Number.isFinite(n) || n < 0.05 || n > 0.95);
  return (
    <Field
      label={label}
      hint={
        <span>
          {value === null ? <span className="font-medium text-fg-muted">Engine default. </span> : null}A decision score from 0.05 to 0.95, not an accuracy probability or a share of calls. Leave empty for the engine default; tune it with the test.
          {invalid && <span className="block text-primer-redFg">Use a number from 0.05 to 0.95.</span>}
        </span>
      }
    >
      {(id) => (
        <TextInput
          id={id}
          type="number"
          min={0.05}
          max={0.95}
          step={0.05}
          placeholder="Engine default"
          value={text}
          disabled={readOnly}
          onChange={(e) => {
            setText(e.target.value);
            const v = e.target.value === '' ? null : Number(e.target.value);
            if (v === null) onChange(null);
            else if (Number.isFinite(v) && v >= 0.05 && v <= 0.95) onChange(Math.round(v * 1000) / 1000);
          }}
        />
      )}
    </Field>
  );
}

function ReadOnlyText({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className="text-xs font-medium text-fg-muted flex items-center gap-1">
        <Lock className="w-3 h-3" aria-hidden="true" />
        {label}
      </span>
      <span className="text-sm text-fg">{value}</span>
      {hint && <span className="text-xs text-fg-subtle">{hint}</span>}
    </div>
  );
}

function IconButton({ label, icon: Icon, disabled, onClick }: { label: string; icon: typeof ArrowUp; disabled?: boolean; onClick(): void }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="w-6 h-6 rounded border border-transparent hover:border-border text-fg-muted hover:text-fg flex items-center justify-center disabled:opacity-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
    >
      <Icon className="w-3.5 h-3.5" aria-hidden="true" />
    </button>
  );
}

export function ProblemList({ problems }: { problems: TaxonomyProblem[] }) {
  return (
    <Notice tone="red">
      <ul className="list-disc pl-4 space-y-0.5">
        {problems.map((p, i) => (
          <li key={`${p.path}-${i}`}>{p.message}</li>
        ))}
      </ul>
    </Notice>
  );
}
