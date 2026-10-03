// "How it's detected" in the category editor (contract 1.4.0, docs/SignalsEmbeddings.md §10): the
// category's rules-engine recipe. Engine (Model or Rules); for Rules the score needed, the
// similar-examples vote, the phrase list (words or patterns, the negation veto, live checks that
// mirror Store's regex-safety and PII rules), the speaker (from the category), an optional call
// position window, and the optional Gemma double-check. A recipe richer than the form (a pack's
// nested filter) is shown as written and kept; its numbers, phrases and check stay editable.
//
// Built-in categories take a recipe too (`BUILTIN_EDITABLE_FIELDS` includes `recipe`).

import { useId, useState } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import {
  GATE_LABEL,
  NEGATION_VETO_MAX_WORDS,
  PII_TEXT_MESSAGE,
  POSITION_PRESETS,
  RECIPE_THRESHOLD_MAX,
  RECIPE_THRESHOLD_MIN,
  buildRecipeFilter,
  filterRules,
  filterText,
  gateUsesPhrase,
  gateUsesSimilar,
  lexiconPhraseProblem,
  looksLikeCallerNumber,
  newRulesRecipe,
  parseRecipeForm,
  positionPresetId,
  recipeOriginText,
  speakerScopeLabel,
  type LexiconSyntax,
  type RecipeCapParams,
  type RecipeForm,
  type RecipeGate,
  type SignalCategory,
  type SignalRecipe,
  type TaxonomyProblem,
} from '../../api';
import { Button, Field, Notice, SelectInput, StatusPill, TextInput } from '../../components/ui';

interface Props {
  category: SignalCategory;
  saved: SignalCategory | undefined;
  readOnly: boolean;
  caps: RecipeCapParams;
  /** Problems on this category's recipe paths. */
  problems: TaxonomyProblem[];
  onChange(update: (c: SignalCategory) => void): void;
}

const round3 = (n: number) => Math.round(n * 1000) / 1000;

export function RecipeEditor({ category: c, saved, readOnly, caps, problems, onChange }: Props) {
  const recipe = c.recipe ?? null;
  const rules = recipe?.engine === 'rules';
  const form = recipe ? parseRecipeForm(recipe.filter) : null;
  const edited = saved !== undefined && JSON.stringify(saved.recipe ?? null) !== JSON.stringify(recipe);
  const origin = recipeOriginText(recipe?.origin);
  const engineName = useId();

  function setRecipe(update: (r: SignalRecipe) => void) {
    onChange((cat) => {
      if (!cat.recipe) return;
      update(cat.recipe);
    });
  }

  function setForm(update: (f: RecipeForm) => void) {
    onChange((cat) => {
      if (!cat.recipe) return;
      const f = parseRecipeForm(cat.recipe.filter);
      if (!f) return;
      update(f);
      cat.recipe.filter = buildRecipeFilter(f);
    });
  }

  function setEngine(engine: 'rules' | 'gemma') {
    onChange((cat) => {
      if (engine === 'rules') {
        if (!cat.recipe) cat.recipe = newRulesRecipe();
        else cat.recipe.engine = 'rules';
      } else if (cat.recipe) {
        // Keep the recipe (its phrases and numbers) so switching back loses nothing.
        cat.recipe.engine = 'gemma';
      }
    });
  }

  return (
    <section aria-label="How it's detected" className="space-y-3" data-testid="signals-recipe-editor">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold text-fg">How it's detected</h3>
        {recipe && rules && <StatusPill tone="blue">Rules + examples</StatusPill>}
        {(!recipe || !rules) && <StatusPill tone="neutral">Model (Gemma)</StatusPill>}
        {edited && <StatusPill tone="yellow">Edited</StatusPill>}
      </div>
      {origin && (
        <p className="text-xs text-fg-muted" data-testid="signals-recipe-origin">
          From the {origin}
          {edited ? ', edited here' : ''}.
        </p>
      )}
      {problems.length > 0 && (
        <Notice tone="red">
          <ul className="list-disc pl-4 space-y-0.5">
            {problems.map((p, i) => (
              <li key={`${p.path}-${i}`}>{p.message}</li>
            ))}
          </ul>
        </Notice>
      )}

      <fieldset className="space-y-1.5">
        <legend className="text-xs font-medium text-fg-muted mb-1">Engine</legend>
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          <Radio name={engineName} label="Model (Gemma)" checked={!rules} disabled={readOnly} onChange={() => setEngine('gemma')} />
          <Radio name={engineName} label="Rules + examples" checked={rules} disabled={readOnly} onChange={() => setEngine('rules')} />
        </div>
        <p className="text-xs text-fg-subtle">
          {rules
            ? 'Found by phrases and by how similar each segment is to labelled examples, with no model. Gemma still fills the fields.'
            : recipe
              ? "Gemma decides this category. Its rules recipe is kept but not used."
              : 'Gemma reads every segment and decides this category, as before.'}
        </p>
      </fieldset>

      {recipe && rules && (
        <div className="space-y-4 rounded-md border border-border-muted bg-canvas p-3">
          <RangeNumber
            label="Score needed"
            value={recipe.threshold}
            min={RECIPE_THRESHOLD_MIN}
            max={RECIPE_THRESHOLD_MAX}
            step={0.005}
            readOnly={readOnly}
            hint="Score = the similar-examples share, plus the phrase weight when a phrase matches. A segment fires when its score reaches this."
            onChange={(v) => setRecipe((r) => void (r.threshold = v))}
          />

          {form ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <Field label="Must also pass" hint="Checked before the score counts.">
                {(id) => (
                  <SelectInput
                    id={id}
                    value={form.gate}
                    disabled={readOnly}
                    onChange={(e) => setForm((f) => void (f.gate = e.target.value as RecipeGate))}
                  >
                    {(Object.keys(GATE_LABEL) as RecipeGate[]).map((g) => (
                      <option key={g} value={g}>
                        {GATE_LABEL[g]}
                      </option>
                    ))}
                  </SelectInput>
                )}
              </Field>
              {gateUsesSimilar(form.gate) ? (
                <RangeNumber
                  label="Similar-examples share at least"
                  value={form.minShare}
                  min={0}
                  max={1}
                  step={0.01}
                  readOnly={readOnly}
                  hint="The share of the nearest labelled examples that carry this category."
                  onChange={(v) => setForm((f) => void (f.minShare = v))}
                />
              ) : (
                <div />
              )}
              <ReadOnlyLine
                label="Speaker"
                value={c.speaker === 'AGENT' || c.speaker === 'CALLER' ? `${speakerScopeLabel(c.speaker)} turns only (the category's speaker)` : 'Either speaker (the category’s speaker)'}
              />
              <PositionEditor form={form} readOnly={readOnly} onChange={setForm} />
            </div>
          ) : (
            <div className="space-y-1" data-testid="signals-recipe-filter">
              <p className="text-xs font-medium text-fg-muted">Rules</p>
              <p className="text-sm text-fg break-words">{filterText(recipe.filter)}</p>
              <p className="text-xs text-fg-subtle">
                These rules came from a pack and use more than this form can edit, so they are kept as written. The score, phrases and double-check below stay editable.
              </p>
            </div>
          )}

          <LexiconEditor recipe={recipe} form={form} readOnly={readOnly} caps={caps} onChange={onChange} />

          <div className="space-y-1">
            <label className="flex items-start gap-2 text-sm text-fg">
              <input
                type="checkbox"
                className="mt-0.5 accent-primer-blue focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                checked={recipe.check === 'gemma'}
                disabled={readOnly}
                onChange={(e) => setRecipe((r) => void (r.check = e.target.checked ? 'gemma' : 'none'))}
              />
              <span className={readOnly ? 'opacity-60' : ''}>
                Gemma double-check
                <span className="block text-xs text-fg-subtle">
                  Gemma reads each span the rules found and confirms it (with its subcategory) or rejects it. Slower; turn it on only where it has been measured to help.
                </span>
              </span>
            </label>
          </div>
        </div>
      )}
    </section>
  );
}

function PositionEditor({ form, readOnly, onChange }: { form: RecipeForm; readOnly: boolean; onChange(update: (f: RecipeForm) => void): void }) {
  const preset = positionPresetId(form.position);
  return (
    <div className="space-y-2">
      <Field label="Where in the call" hint="Where the segment starts, as a share of the call's length.">
        {(id) => (
          <SelectInput
            id={id}
            value={preset}
            disabled={readOnly}
            onChange={(e) =>
              onChange((f) => {
                const v = e.target.value;
                if (v === 'anywhere') f.position = null;
                else if (v === 'custom') f.position = { ruleId: f.position?.ruleId ?? 'position', from: f.position?.from ?? 0, to: f.position?.to ?? 1 };
                else {
                  const p = POSITION_PRESETS.find((x) => x.id === v);
                  if (p) f.position = { ruleId: f.position?.ruleId ?? 'position', from: p.from, to: p.to };
                }
              })
            }
          >
            <option value="anywhere">Anywhere in the call</option>
            {POSITION_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
            <option value="custom">Custom window</option>
          </SelectInput>
        )}
      </Field>
      {form.position && (
        <div className="grid grid-cols-2 gap-2">
          <FractionInput label="Starts from" value={form.position.from} readOnly={readOnly} onChange={(v) => onChange((f) => void (f.position && (f.position.from = v)))} />
          <FractionInput label="Starts by" value={form.position.to} readOnly={readOnly} onChange={(v) => onChange((f) => void (f.position && (f.position.to = v)))} />
          {form.position.to <= form.position.from && <p className="col-span-2 text-xs text-primer-redFg">The window must end after it starts.</p>}
        </div>
      )}
    </div>
  );
}

function LexiconEditor({
  recipe,
  form,
  readOnly,
  caps,
  onChange,
}: {
  recipe: SignalRecipe;
  form: RecipeForm | null;
  readOnly: boolean;
  caps: RecipeCapParams;
  onChange(update: (c: SignalCategory) => void): void;
}) {
  const lexicon = recipe.lexicon;
  // With no list yet, the syntax is only chosen here; the list is created with its first phrase.
  const [pendingSyntax, setPendingSyntax] = useState<LexiconSyntax>('words');
  const syntax: LexiconSyntax = lexicon?.syntax ?? pendingSyntax;
  const checkId = useId();
  const phrases = lexicon?.phrases ?? [];
  const cap = caps.max_signal_lexicon_phrases;
  const [value, setValue] = useState('');
  const syntaxName = useId();
  const trimmed = value.trim();
  const problem = trimmed ? lexiconPhraseProblem(trimmed, syntax) : null;
  const duplicate = trimmed !== '' && phrases.some((p) => p.toLowerCase() === trimmed.toLowerCase());
  const pii = trimmed !== '' && looksLikeCallerNumber(trimmed);
  const atCap = phrases.length >= cap;
  const usesLexiconRule = filterRules(recipe.filter).some((r) => r.params.type === 'phrase' && !r.params.phrases);

  function edit(update: (r: SignalRecipe) => void) {
    onChange((cat) => {
      if (cat.recipe) update(cat.recipe);
    });
  }

  function add() {
    if (!trimmed || problem || duplicate || atCap) return;
    edit((r) => {
      if (!r.lexicon) r.lexicon = { syntax, phrases: [], negation_veto_words: 0 };
      r.lexicon.phrases = [...r.lexicon.phrases, trimmed];
    });
    setValue('');
  }

  function removeList() {
    onChange((cat) => {
      const r = cat.recipe;
      if (!r) return;
      r.lexicon = null;
      r.lexicon_weight = 0;
      // A "phrase matches" requirement tests the list, so it goes with it.
      const f = parseRecipeForm(r.filter);
      if (f && gateUsesPhrase(f.gate)) {
        f.gate = gateUsesSimilar(f.gate) ? 'similar' : 'none';
        r.filter = buildRecipeFilter(f);
      }
    });
  }

  return (
    <div className="space-y-2" data-testid="signals-lexicon-editor">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium text-fg-muted">
          Phrases ({phrases.length} of {cap})
        </p>
        {!readOnly && lexicon && (
          <Button size="sm" variant="ghost" icon={Trash2} onClick={removeList}>
            Remove phrase list
          </Button>
        )}
      </div>
      <fieldset className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <legend className="sr-only">Phrase syntax</legend>
        {(['words', 'regex'] as const).map((s) => (
          <Radio
            key={s}
            name={syntaxName}
            label={s === 'words' ? 'Words' : 'Patterns (regular expressions)'}
            checked={syntax === s}
            disabled={readOnly}
            onChange={() => (lexicon ? edit((r) => r.lexicon && void (r.lexicon.syntax = s)) : setPendingSyntax(s))}
          />
        ))}
      </fieldset>
      <p className="text-xs text-fg-subtle">
        {syntax === 'words'
          ? 'Plain words, matched as whole words and ignoring case, e.g. "refund" or "speak to a manager".'
          : 'Patterns ignore case. Allowed: words, [classes], \\b, (…) and (?:…) groups, | and repeats. Not allowed: look-arounds, named groups, back-references, or a repeat around a repeat.'}
      </p>
      {phrases.length === 0 ? (
        <p className="text-xs text-fg-subtle">
          No phrases. {usesLexiconRule || (form && gateUsesPhrase(form.gate)) ? 'Add at least one: this recipe requires a phrase match.' : 'Phrases are optional; without them the similar-examples share decides.'}
        </p>
      ) : (
        <ul className="space-y-1" aria-label="Phrases">
          {phrases.map((ph, k) => {
            const bad = lexiconPhraseProblem(ph, syntax);
            return (
              <li key={`${ph}-${k}`} className="rounded border border-border-muted bg-canvas-subtle px-2 py-1">
                <div className="flex items-start gap-2">
                  <code className="flex-1 min-w-0 break-all text-xs text-fg">{ph}</code>
                  {!readOnly && (
                    <button
                      type="button"
                      aria-label={`Remove phrase ${k + 1}`}
                      onClick={() => edit((r) => r.lexicon && void (r.lexicon.phrases = r.lexicon.phrases.filter((_, i) => i !== k)))}
                      className="text-fg-muted hover:text-fg rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
                    >
                      <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
                    </button>
                  )}
                </div>
                {bad && <p className="text-xs text-primer-redFg">Phrase {k + 1}: {bad}.</p>}
              </li>
            );
          })}
        </ul>
      )}
      {!readOnly && (
        <div className="space-y-1">
          <div className="flex flex-wrap items-end gap-2">
            <div className="flex-1 min-w-[12rem]">
              <Field label="New phrase">
                {(id) => (
                  <TextInput
                    id={id}
                    value={value}
                    maxLength={320}
                    disabled={atCap}
                    aria-invalid={problem !== null || duplicate}
                    aria-describedby={checkId}
                    className={syntax === 'regex' ? 'font-mono' : undefined}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        add();
                      }
                    }}
                  />
                )}
              </Field>
            </div>
            <Button size="sm" icon={Plus} disabled={!trimmed || problem !== null || duplicate || atCap} onClick={add}>
              Add phrase
            </Button>
          </div>
          <PhraseCheck id={checkId} problem={problem} duplicate={duplicate} pii={pii} atCap={atCap} cap={cap} />
        </div>
      )}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <RangeNumber
          label="Phrase weight"
          value={recipe.lexicon_weight}
          min={0}
          max={1}
          step={0.05}
          readOnly={readOnly || !lexicon}
          hint={lexicon ? 'Added to the score when a phrase matches. 0 keeps phrases out of the score.' : 'Add a phrase first.'}
          onChange={(v) => edit((r) => void (r.lexicon_weight = v))}
        />
        <Field label="Negation window (words)" hint={`Ignore a match within this many words after "not", "can't", "never"… 0 turns it off (at most ${NEGATION_VETO_MAX_WORDS}).`}>
          {(id) => (
            <TextInput
              id={id}
              type="number"
              min={0}
              max={NEGATION_VETO_MAX_WORDS}
              step={1}
              value={lexicon?.negation_veto_words ?? 0}
              disabled={readOnly || !lexicon}
              onChange={(e) => {
                const n = Math.max(0, Math.min(NEGATION_VETO_MAX_WORDS, Math.round(Number(e.target.value) || 0)));
                edit((r) => r.lexicon && void (r.lexicon.negation_veto_words = n));
              }}
            />
          )}
        </Field>
      </div>
    </div>
  );
}

function PhraseCheck({ id, problem, duplicate, pii, atCap, cap }: { id: string; problem: string | null; duplicate: boolean; pii: boolean; atCap: boolean; cap: number }) {
  let text: React.ReactNode = null;
  if (atCap) text = <span className="text-primer-yellowFg">At most {cap} phrases in a list. Remove one to add another.</span>;
  else if (problem) text = <span className="text-primer-redFg">Not allowed: {problem}.</span>;
  else if (duplicate) text = <span className="text-primer-yellowFg">That phrase is already in the list.</span>;
  else if (pii) text = <span className="text-primer-yellowFg">{PII_TEXT_MESSAGE} Store refuses a phrase its PII rules match.</span>;
  return (
    <p id={id} className="text-xs min-h-[1rem]" role="status" data-testid="signals-phrase-check">
      {text}
    </p>
  );
}

function Radio({ name, label, checked, disabled, onChange }: { name: string; label: string; checked: boolean; disabled?: boolean; onChange(): void }) {
  return (
    <label className={`inline-flex items-center gap-1.5 text-sm text-fg ${disabled ? 'opacity-60' : ''}`}>
      <input
        type="radio"
        name={name}
        checked={checked}
        disabled={disabled}
        onChange={onChange}
        className="accent-primer-blue focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
      />
      {label}
    </label>
  );
}

/** A number with a slider beside it. The number field carries the label; the slider is a second
 * way to set the same value. */
function RangeNumber({
  label,
  value,
  min,
  max,
  step,
  readOnly,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  readOnly: boolean;
  hint?: string;
  onChange(v: number): void;
}) {
  const [text, setText] = useState(String(value));
  const [last, setLast] = useState(value);
  if (value !== last) {
    setLast(value);
    setText(String(value));
  }
  const n = Number(text);
  const invalid = text === '' || !Number.isFinite(n) || n < min || n > max;
  const commit = (raw: string) => {
    setText(raw);
    const v = Number(raw);
    if (raw !== '' && Number.isFinite(v) && v >= min && v <= max) onChange(round3(v));
  };
  return (
    <Field
      label={label}
      hint={
        <span>
          {hint}
          {invalid && (
            <span className="block text-primer-redFg">
              Use a number from {min} to {max}.
            </span>
          )}
        </span>
      }
    >
      {(id) => (
        <div className="flex items-center gap-2">
          <input
            type="range"
            aria-label={`${label} slider`}
            min={min}
            max={max}
            step={step}
            value={Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : value}
            disabled={readOnly}
            onChange={(e) => commit(e.target.value)}
            className="flex-1 min-w-0 accent-primer-blue focus:outline-none focus-visible:ring-2 focus-visible:ring-primer-blue"
          />
          <div className="w-24 shrink-0">
            <TextInput
              id={id}
              type="number"
              min={min}
              max={max}
              step={step}
              value={text}
              disabled={readOnly}
              aria-invalid={invalid}
              onChange={(e) => commit(e.target.value)}
            />
          </div>
        </div>
      )}
    </Field>
  );
}

function FractionInput({ label, value, readOnly, onChange }: { label: string; value: number; readOnly: boolean; onChange(v: number): void }) {
  const [text, setText] = useState(String(value));
  const [last, setLast] = useState(value);
  if (value !== last) {
    setLast(value);
    setText(String(value));
  }
  return (
    <Field label={label} hint={`${Math.round(value * 100)}% of the call`}>
      {(id) => (
        <TextInput
          id={id}
          type="number"
          min={0}
          max={1}
          step={0.05}
          value={text}
          disabled={readOnly}
          onChange={(e) => {
            setText(e.target.value);
            const v = Number(e.target.value);
            if (e.target.value !== '' && Number.isFinite(v) && v >= 0 && v <= 1) onChange(round3(v));
          }}
        />
      )}
    </Field>
  );
}

function ReadOnlyLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className="text-xs font-medium text-fg-muted">{label}</span>
      <span className="text-sm text-fg">{value}</span>
    </div>
  );
}

