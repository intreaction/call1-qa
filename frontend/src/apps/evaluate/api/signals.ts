// Display rules and client-side checks for Contact Signals v2 (contract 1.3.0,
// docs/ContactSignalsV2.md §7, §9 and §10). Store stays the authority: its save validator applies
// the same caps from its effective `ContractParameters` and refuses by path. These checks exist so
// an admin sees the reason next to the field before saving, never to replace Store's.
//
// Nothing here invents data. A label Evaluate cannot resolve (an unknown category, subcategory or
// alert rule) falls back to its ID, never to a guess.

import type { Tone } from './derive';
import type {
  ContactSignalKind,
  ContactSignalView,
  ContactSignalsView,
  ContractParameters,
  ExtractedFieldView,
  FieldPiiClass,
  ResultGroup,
  SignalAlertCondition,
  SignalAlertConditionInput,
  SignalAlertRuleRecord,
  SignalCategory,
  SignalField,
  SignalFieldType,
  SignalHitPart,
  SignalPipeline,
  SignalSubcategory,
  SignalTaxonomy,
  SignalTaxonomyStatus,
  SpeakerRole,
} from './types';
import { recipeProblems, type RecipeCapParams } from './signalRules';

// --- fixed vocabulary (call1/contracts/signals.py) ----------------------------------------------

/** `BUILTIN_SIGNAL_CATEGORIES` names: Call1-authored and fixed. */
export const BUILTIN_CATEGORY_NAMES: Record<Exclude<ContactSignalKind, 'custom'>, string> = {
  intent: 'Caller objective',
  issue: 'Reported issue',
  friction: 'Friction point',
  fix_proposed: 'Proposed fix',
  agent_reports_completed: 'Agent completed',
  caller_confirms_resolved: 'Caller confirmed',
  caller_reports_unresolved: 'Still unresolved',
  deferred: 'Deferred',
};

export const BUILTIN_CATEGORY_IDS = Object.keys(BUILTIN_CATEGORY_NAMES) as Exclude<ContactSignalKind, 'custom'>[];

export function isBuiltinCategoryId(id: string): id is Exclude<ContactSignalKind, 'custom'> {
  return Object.prototype.hasOwnProperty.call(BUILTIN_CATEGORY_NAMES, id);
}

/** `SIGNAL_NODE_ID_PATTERN`: immutable IDs of categories, subcategories, fields and alert rules. */
export const SIGNAL_NODE_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,39}$/;
/** `RESERVED_CATEGORY_IDS` plus every built-in ID: a custom category never uses one. */
export const RESERVED_CATEGORY_IDS: ReadonlySet<string> = new Set(['none', 'custom', ...BUILTIN_CATEGORY_IDS]);
/** IDs Evaluate never generates for a new category: the contract's reserved IDs and the route words
 * `alerts` and `versions`, because `#/signals/:categoryId` shares its path with `#/signals/alerts`
 * and `#/signals/versions` (docs/ContactSignalsV2.md §10.1). */
export const NEW_CATEGORY_RESERVED_IDS: ReadonlySet<string> = new Set([...RESERVED_CATEGORY_IDS, 'alerts', 'versions']);
/** `RESERVED_SUBCATEGORY_IDS`: stage 2's fixed "Other" and "Not <category>" options. */
export const RESERVED_SUBCATEGORY_IDS: ReadonlySet<string> = new Set(['other', 'not']);
/** `RESERVED_FIELD_IDS`: `quote` is the narrowed quote (§5.4). */
export const RESERVED_FIELD_IDS: ReadonlySet<string> = new Set(['quote']);

/** `FORBIDDEN_FIELD_PII_CLASSES`: masked before any model sees them, so they can never be a field. */
export const FORBIDDEN_PII_CLASSES: ReadonlySet<FieldPiiClass> = new Set<FieldPiiClass>([
  'caller_name',
  'account_number',
  'card_number',
  'phone',
  'email',
  'address',
  'url',
  'secret',
  'government_id',
]);

export const PII_CLASS_LABEL: Record<FieldPiiClass, string> = {
  none: 'None (business data)',
  organization: 'Organization',
  product: 'Product',
  amount: 'Amount',
  date: 'Date',
  agent_name: 'Agent name',
  caller_name: 'Caller name',
  account_number: 'Account number',
  card_number: 'Card number',
  phone: 'Phone number',
  email: 'Email address',
  address: 'Postal address',
  url: 'URL',
  secret: 'Secret (PIN, password)',
  government_id: 'Government ID',
};

export const PII_CLASSES = Object.keys(PII_CLASS_LABEL) as FieldPiiClass[];

export const FIELD_TYPE_LABEL: Record<SignalFieldType, string> = {
  string: 'Text',
  enum: 'One of a list',
  boolean: 'Yes / no',
  number: 'Number',
  amount: 'Amount',
  date: 'Date',
};

export const FIELD_TYPES = Object.keys(FIELD_TYPE_LABEL) as SignalFieldType[];

/** A category's speaker scope: built-ins are fixed; a custom category is Agent, Caller or either. */
export function speakerScopeLabel(speaker: SpeakerRole | null | undefined): string {
  if (speaker === 'AGENT') return 'Agent';
  if (speaker === 'CALLER') return 'Caller';
  return 'Either speaker';
}

export const SPEAKER_LABEL: Record<SpeakerRole, string> = { AGENT: 'Agent', CALLER: 'Caller', SYSTEM: 'System', UNKNOWN: 'Unknown' };

// --- families and names -----------------------------------------------------------------------

/** The waveform color family (§10.2): v1's lifecycle pass kinds, resolution pass kinds, or custom. */
export type SignalFamily = 'lifecycle' | 'resolution' | 'custom';

const LIFECYCLE: ReadonlySet<string> = new Set(['intent', 'issue', 'friction']);

export function signalFamily(categoryIdOrKind: string | null | undefined): SignalFamily {
  if (!categoryIdOrKind || !isBuiltinCategoryId(categoryIdOrKind)) return 'custom';
  return LIFECYCLE.has(categoryIdOrKind) ? 'lifecycle' : 'resolution';
}

export const FAMILY_DISPLAY: Record<SignalFamily, { label: string; tone: Tone }> = {
  lifecycle: { label: 'Lifecycle', tone: 'blue' },
  resolution: { label: 'Resolution', tone: 'green' },
  custom: { label: 'Custom', tone: 'magenta' },
};

/** The category a hit belongs to: `category_id` on v2 hits, the kind on v1 hits. */
export function hitCategoryId(sig: Pick<ContactSignalView, 'category_id' | 'kind'>): string {
  return sig.category_id ?? sig.kind;
}

/** Display name of a hit's category: the fixed name for a built-in (v1 or v2), otherwise `label`,
 * which carries a custom category's name (and a kind this build does not know). */
export function hitCategoryName(sig: Pick<ContactSignalView, 'kind' | 'label' | 'category_id'>): string {
  const id = sig.category_id ?? sig.kind;
  if (sig.kind !== 'custom' && isBuiltinCategoryId(id)) return BUILTIN_CATEGORY_NAMES[id];
  return sig.label;
}

/** Display name of a hit's subcategory, or null when stage 2 assigned none. */
export function hitSubcategoryName(sig: Pick<ContactSignalView, 'subcategory_id' | 'subcategory_label'>): string | null {
  if (!sig.subcategory_id) return null;
  if (sig.subcategory_label) return sig.subcategory_label;
  return sig.subcategory_id === 'other' ? 'Other' : sig.subcategory_id;
}

/** A multi-segment hit's later spans (decision 25, ContactSignalsV2 §6.5), in call order. Tolerates
 * a payload without the field (a pre-merge result or a test stub). */
export function hitParts(sig: { parts?: SignalHitPart[] | null }): SignalHitPart[] {
  return sig.parts ?? [];
}

/** The IDs a merged hit's parts had as separate hits (§6.3: the anchor ID with the part's
 * `t<turn>b<block>`). Store lists feedback saved on them before the merge (decision 25). */
export function hitPartIds(sig: { id: string; parts?: SignalHitPart[] | null }): string[] {
  if (!/\.t\d+b\d+$/.test(sig.id)) return [];
  return hitParts(sig).map((p) => sig.id.replace(/\.t\d+b\d+$/, `.t${p.turn_id}b${p.block}`));
}

/** How many segments a hit spans: 1, or 1 + its parts. */
export function hitSegmentCount(sig: { parts?: SignalHitPart[] | null }): number {
  return 1 + hitParts(sig).length;
}

/** Where a hit's time range ends: the last part's end on a multi-segment hit, else its own end. */
export function hitSpanEnd(sig: { end: number; span_end?: number | null }): number {
  return sig.span_end != null && sig.span_end > sig.end ? sig.span_end : sig.end;
}

/** "×3 segments" for a multi-segment hit, null for a single span. */
export function hitSegmentsText(sig: { parts?: SignalHitPart[] | null }): string | null {
  const n = hitSegmentCount(sig);
  return n > 1 ? `×${n} segments` : null;
}

/** "Caller objective › Cancel account" */
export function hitPathLabel(sig: Pick<ContactSignalView, 'kind' | 'label' | 'category_id' | 'subcategory_id' | 'subcategory_label'>): string {
  const sub = hitSubcategoryName(sig);
  return sub ? `${hitCategoryName(sig)} › ${sub}` : hitCategoryName(sig);
}

export function categoryById(taxonomy: SignalTaxonomy | undefined, id: string): SignalCategory | undefined {
  return taxonomy?.categories.find((c) => c.category_id === id);
}

export function categoryName(taxonomy: SignalTaxonomy | undefined, id: string): string {
  return categoryById(taxonomy, id)?.name ?? (isBuiltinCategoryId(id) ? BUILTIN_CATEGORY_NAMES[id] : id);
}

export function subcategoryName(taxonomy: SignalTaxonomy | undefined, categoryId: string, subcategoryId: string): string {
  if (subcategoryId === 'other') return 'Other';
  return categoryById(taxonomy, categoryId)?.subcategories.find((s) => s.subcategory_id === subcategoryId)?.name ?? subcategoryId;
}

// --- fields -----------------------------------------------------------------------------------

/** The text a field chip shows, or null when the field has nothing to show (absent: absence stays
 * absence, §6.2). String-like values show their masked surface text (§10.2). */
export function fieldChipText(field: ExtractedFieldView): string | null {
  // `status` is required by the contract; a reader built for a newer minor that omits it still
  // shows a value it was sent rather than hiding it.
  const status = field.status ?? (field.value !== null && field.value !== undefined ? 'extracted' : 'absent');
  switch (status) {
    case 'extracted': {
      let value: string;
      if (field.type === 'boolean') value = field.value === true ? 'yes' : field.value === false ? 'no' : String(field.value);
      else if (field.surface && field.type !== 'enum') value = field.surface;
      else value = String(field.value);
      return `${field.name}: ${value}`;
    }
    case 'withheld_pii':
      return `${field.name}: withheld (PII)`;
    default:
      return null;
  }
}

export const FIELD_STATUS_TEXT: Record<ExtractedFieldView['status'], string> = {
  extracted: 'Extracted from the span',
  absent: 'Not in the span',
  ungrounded: 'The extractor answered, but the words were not in the span, so it was dropped',
  withheld_pii: 'Withheld: the value was personal data',
  invalid: 'The extractor answered with a value of the wrong type',
};

// --- alert conditions ---------------------------------------------------------------------------

/** "Caller objective › Cancel account › reason = price" (§10.1). */
export function conditionText(condition: SignalAlertCondition | SignalAlertConditionInput, taxonomy: SignalTaxonomy | undefined): string {
  const parts = [categoryName(taxonomy, condition.category_id)];
  if (condition.subcategory_id) parts.push(subcategoryName(taxonomy, condition.category_id, condition.subcategory_id));
  if (condition.field_id) {
    const field = fieldsForCondition(taxonomy, condition.category_id, condition.subcategory_id ?? null).find((f) => f.field_id === condition.field_id);
    const name = field?.name ?? condition.field_id;
    const eq = condition.field_equals;
    if (eq === null || eq === undefined) parts.push(`${name} extracted`);
    else parts.push(`${name} = ${typeof eq === 'boolean' ? (eq ? 'yes' : 'no') : eq}`);
  }
  let text = parts.join(' › ');
  if (condition.min_confidence !== null && condition.min_confidence !== undefined) text += ` (confidence ≥ ${condition.min_confidence.toFixed(2)})`;
  return text;
}

/** The fields an alert condition may name (`signal_alert_condition_problem`): with no subcategory,
 * the category's fields and every subcategory's; with one, that path's. */
export function fieldsForCondition(taxonomy: SignalTaxonomy | undefined, categoryId: string, subcategoryId: string | null): SignalField[] {
  const c = categoryById(taxonomy, categoryId);
  if (!c) return [];
  if (!subcategoryId) return [...c.fields, ...c.subcategories.flatMap((s) => s.fields)];
  if (subcategoryId === 'other') return [...c.fields];
  const s = c.subcategories.find((x) => x.subcategory_id === subcategoryId);
  return [...c.fields, ...(s?.fields ?? [])];
}

export function alertRuleName(rules: SignalAlertRuleRecord[] | undefined, ruleId: string): string {
  return rules?.find((r) => r.rule_id === ruleId)?.name ?? ruleId;
}

// --- pipeline and staleness ---------------------------------------------------------------------

export const PIPELINE_TEXT: Record<SignalPipeline, string> = {
  v1: 'Built-in categories run on v1. Subcategories, custom categories and fields apply when Contact Signals v2 is on',
  shadow: 'v2 runs beside v1 for comparison on every call v1 scores',
  v2: 'Contact Signals v2 is on: every new call and update runs the category, subcategory and field stages',
};

export const PIPELINE_LABEL: Record<SignalPipeline, string> = { v1: 'v1', shadow: 'Shadow (v1 + v2 compare)', v2: 'v2' };

const STAGE_NOUN: Record<SignalTaxonomyStatus['outdated_stages'][number], string> = {
  categorize: 'categories',
  subcategorize: 'subcategories',
  extract: 'fields',
};

/** "Scored with taxonomy v3 (current v5): new subcategories not applied" (§10.2), or null when the
 * result is current. v1 results carry no scored version and are never "outdated" by the editor. */
export function taxonomyOutdatedText(status: SignalTaxonomyStatus | null | undefined): string | null {
  if (!status || status.scored_version === null) return null;
  const outdated = status.outdated_stages.length > 0;
  if (!outdated && !status.thresholds_changed && status.scored_version === status.current_version) return null;
  const head = `Scored with taxonomy v${status.scored_version} (current v${status.current_version})`;
  if (outdated) return `${head}: new ${status.outdated_stages.map((s) => STAGE_NOUN[s]).join(', ')} not applied`;
  if (status.thresholds_changed) return `${head}: new thresholds not applied`;
  return head;
}

/** A job or stage error code in words (`engine_error` → "engine error"), for the §10.2 texts. */
export function failureCodeText(code: string | null | undefined): string {
  return (code ?? '').replace(/_/g, ' ').trim();
}

const STAGE_TEXT_PREFIX: Record<string, string> = {
  categorize: 'Categories unavailable',
  subcategorize: 'Subcategories unavailable',
  extract: 'Fields unavailable',
};

/** What a partial v2 result is missing, named by stage (§10.2 "partial": "Subcategories unavailable
 * (engine error)"). Process names anything else (for example "N spans over the extraction cap") in
 * `partial_reason`, "; "-separated; each piece is shown once, in words, and a piece that repeats a
 * stage already named (or a bare `<stage>_failed` code) is dropped. */
export function partialStageTexts(view: Pick<ContactSignalsView, 'stages' | 'partial_reason'>): string[] {
  const out: string[] = [];
  const failedStages = new Set<string>();
  for (const stage of view.stages) {
    if (stage.included) continue;
    const code = stage.failure_code;
    const why = !code || code === 'model_unavailable' || code === 'provider_error' || code === 'worker_crashed' ? 'engine error' : failureCodeText(code);
    const prefix = STAGE_TEXT_PREFIX[stage.stage];
    if (prefix) {
      out.push(`${prefix} (${why})`);
      failedStages.add(stage.stage);
    }
  }
  for (const raw of (view.partial_reason ?? '').split(';')) {
    let piece = raw.trim();
    if (!piece) continue;
    const bare = /^(categorize|subcategorize|extract)_failed$/.exec(piece);
    if (bare && failedStages.has(bare[1])) continue;
    piece = piece.replace(/^[a-z]+(?:_[a-z]+)+:\s*/, ''); // a machine tag such as "extraction_cap: "
    piece = piece.replace(/\(([a-z]+(?:_[a-z]+)+)\)/g, (_m, code: string) => `(${failureCodeText(code)})`);
    if (/^[a-z]+(?:_[a-z]+)+$/.test(piece)) piece = failureCodeText(piece);
    const covered = Object.entries(STAGE_TEXT_PREFIX).some(([stage, prefix]) => failedStages.has(stage) && piece.startsWith(`${prefix} (`));
    if (!covered && !out.includes(piece)) out.push(piece);
  }
  return out;
}

/** The PII-masking dead-block (§10.2): the group failed and the transcript text is still withheld. */
export function signalsWaitOnMasking(group: ResultGroup | undefined, transcriptWithheld: boolean): boolean {
  return group?.state === 'failed' && transcriptWithheld;
}

// --- taxonomy editing ---------------------------------------------------------------------------

/** A node ID from a display name: lower-case, `_` for spaces, unique among `taken`, never reserved. */
export function slugId(name: string, taken: ReadonlySet<string>, reserved: ReadonlySet<string> = new Set()): string {
  let base = name
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 36);
  if (!base || !/^[a-z0-9]/.test(base)) base = `n${base}`.slice(0, 36);
  let id = base;
  for (let n = 2; taken.has(id) || reserved.has(id); n += 1) id = `${base}_${n}`;
  return id;
}

export interface TaxonomyProblem {
  /** The contract path, e.g. `categories[9].subcategories[2].gloss` (as Store's `details.field`). */
  path: string;
  /** The node the problem belongs to, for jumping to it in the tree. */
  categoryId: string;
  message: string;
}

const ONE_LINE = /^\S(?:.*\S)?$/;

function textProblems(path: string, categoryId: string, label: string, value: string | null | undefined, max: number, required: boolean): TaxonomyProblem[] {
  if (value === null || value === undefined || value === '') return required ? [{ path, categoryId, message: `${label} is required.` }] : [];
  const out: TaxonomyProblem[] = [];
  if (value.length > max) out.push({ path, categoryId, message: `${label} is ${value.length} characters; the limit is ${max}.` });
  if (/[\r\n]/.test(value)) out.push({ path, categoryId, message: `${label} must be one line.` });
  return out;
}

function fieldProblems(base: string, categoryId: string, fields: SignalField[], taken: string[]): TaxonomyProblem[] {
  const out: TaxonomyProblem[] = [];
  const seen = new Set(taken);
  fields.forEach((f, k) => {
    const p = `${base}.fields[${k}]`;
    const who = `Field "${f.name || f.field_id}"`;
    if (!SIGNAL_NODE_ID_PATTERN.test(f.field_id)) out.push({ path: `${p}.field_id`, categoryId, message: `${who} needs an ID of lower-case letters, digits, _ or -.` });
    if (RESERVED_FIELD_IDS.has(f.field_id)) out.push({ path: `${p}.field_id`, categoryId, message: `${who}: the ID "quote" is reserved for the narrowed quote.` });
    if (seen.has(f.field_id)) out.push({ path: `${p}.field_id`, categoryId, message: `${who}: field IDs must be unique on a category and subcategory path.` });
    seen.add(f.field_id);
    out.push(...textProblems(`${p}.name`, categoryId, `${who} name`, f.name, 40, true));
    out.push(...textProblems(`${p}.description`, categoryId, `${who} description`, f.description, 200, true));
    if (FORBIDDEN_PII_CLASSES.has(f.pii_class)) out.push({ path: `${p}.pii_class`, categoryId, message: `${who}: ${PII_CLASS_LABEL[f.pii_class]} is masked before any model sees it, so it cannot be a field.` });
    if (f.type === 'enum') {
      if (!f.enum_values.length) out.push({ path: `${p}.enum_values`, categoryId, message: `${who} lists at least one value.` });
      if (f.enum_values.length > 12) out.push({ path: `${p}.enum_values`, categoryId, message: `${who} has ${f.enum_values.length} values; the limit is 12.` });
      const folded = f.enum_values.map((v) => v.toLowerCase());
      if (new Set(folded).size !== folded.length) out.push({ path: `${p}.enum_values`, categoryId, message: `${who}: values must be unique, ignoring case.` });
      f.enum_values.forEach((v, m) => {
        if (!ONE_LINE.test(v) || v.length > 40) out.push({ path: `${p}.enum_values[${m}]`, categoryId, message: `${who}: value "${v}" must be 1–40 characters with no leading or trailing space.` });
      });
    } else if (f.enum_values.length) {
      out.push({ path: `${p}.enum_values`, categoryId, message: `${who}: only a "one of a list" field has values.` });
    }
  });
  return out;
}

function exampleProblems(base: string, categoryId: string, examples: string[]): TaxonomyProblem[] {
  const out: TaxonomyProblem[] = [];
  if (examples.length > 5) out.push({ path: `${base}.examples`, categoryId, message: `At most 5 examples (${examples.length} listed).` });
  examples.forEach((e, k) => {
    if (!ONE_LINE.test(e) || e.length > 120) out.push({ path: `${base}.examples[${k}]`, categoryId, message: `Example ${k + 1} must be one line of 1–120 characters, with no leading or trailing space.` });
  });
  return out;
}

/**
 * Every problem Store's save would refuse that Evaluate can see locally: the §9.6 caps from the
 * contract parameters Store reported (`signal_taxonomy_cap_violations`), the Pydantic outer bounds,
 * reserved and duplicate IDs, and forbidden PII classes. Store's number-rule PII detectors run
 * only on Store; their refusals arrive as `validation_failed` with `details.field`.
 */
export function taxonomyProblems(
  taxonomy: SignalTaxonomy,
  params: Pick<ContractParameters, 'max_custom_signal_categories' | 'max_active_subcategories' | 'max_fields_per_path' | 'max_option_gloss_chars'> & Partial<RecipeCapParams>,
): TaxonomyProblem[] {
  const out: TaxonomyProblem[] = [];
  const glossMax = params.max_option_gloss_chars;
  const activeCustom = taxonomy.categories.filter((c) => !c.builtin && c.active);
  if (activeCustom.length > params.max_custom_signal_categories) {
    const extra = activeCustom[params.max_custom_signal_categories];
    out.push({ path: 'categories', categoryId: extra.category_id, message: `At most ${params.max_custom_signal_categories} active custom categories (${activeCustom.length} active). Retire one first.` });
  }
  if (taxonomy.categories.filter((c) => !c.builtin).length > 16) out.push({ path: 'categories', categoryId: '', message: 'At most 16 custom categories in total, active or retired.' });
  const activeNames = new Map<string, string>();
  taxonomy.categories.forEach((c, i) => {
    const base = `categories[${i}]`;
    const cid = c.category_id;
    if (!c.builtin) {
      if (!SIGNAL_NODE_ID_PATTERN.test(cid) || RESERVED_CATEGORY_IDS.has(cid)) out.push({ path: `${base}.category_id`, categoryId: cid, message: `The category ID "${cid}" is reserved or not allowed.` });
      out.push(...textProblems(`${base}.name`, cid, 'Category name', c.name, 40, true));
      out.push(...textProblems(`${base}.gloss`, cid, 'Category option text', c.gloss, 80, true));
      if (c.active && c.gloss.length > glossMax) {
        out.push({ path: `${base}.gloss`, categoryId: cid, message: `"${c.name}" option text is ${c.gloss.length} characters; keep it to ${glossMax} so the classifier reads all of it.` });
      }
      out.push(...textProblems(`${base}.description`, cid, 'Category description', c.description, 240, false));
    }
    if (c.active) {
      const key = c.name.trim().toLowerCase();
      if (activeNames.has(key)) out.push({ path: `${base}.name`, categoryId: cid, message: `Two active categories are named "${c.name}".` });
      activeNames.set(key, cid);
    }
    out.push(...exampleProblems(base, cid, c.examples));
    if (c.fields.length > 8) out.push({ path: `${base}.fields`, categoryId: cid, message: `"${c.name}" has ${c.fields.length} fields; a category holds at most 8.` });
    if (c.active && c.fields.length > params.max_fields_per_path) out.push({ path: `${base}.fields`, categoryId: cid, message: `"${c.name}" has ${c.fields.length} fields; the limit per path is ${params.max_fields_per_path}.` });
    out.push(...fieldProblems(base, cid, c.fields, []));
    // 1.4.0: the rules-engine recipe (docs/SignalsEmbeddings.md). Caps default to the contract's.
    const recipeCaps = { max_signal_recipe_rules: params.max_signal_recipe_rules ?? 8, max_signal_lexicon_phrases: params.max_signal_lexicon_phrases ?? 24 };
    for (const rp of recipeProblems(c, base, recipeCaps)) out.push({ path: rp.path, categoryId: cid, message: rp.message });
    if (c.subcategories.length > 24) out.push({ path: `${base}.subcategories`, categoryId: cid, message: `"${c.name}" has ${c.subcategories.length} subcategories, active or retired; the limit is 24.` });
    const active = c.subcategories.filter((s) => s.active);
    if (c.active && active.length > params.max_active_subcategories) {
      out.push({ path: `${base}.subcategories`, categoryId: cid, message: `"${c.name}" has ${active.length} active subcategories; the limit is ${params.max_active_subcategories}. Deactivate one first.` });
    }
    const subIds = new Set<string>();
    const subNames = new Set<string>();
    c.subcategories.forEach((s, j) => {
      const sb = `${base}.subcategories[${j}]`;
      const who = `Subcategory "${s.name || s.subcategory_id}"`;
      if (!SIGNAL_NODE_ID_PATTERN.test(s.subcategory_id) || RESERVED_SUBCATEGORY_IDS.has(s.subcategory_id)) out.push({ path: `${sb}.subcategory_id`, categoryId: cid, message: `${who}: the ID "${s.subcategory_id}" is reserved or not allowed.` });
      if (subIds.has(s.subcategory_id)) out.push({ path: `${sb}.subcategory_id`, categoryId: cid, message: `${who}: subcategory IDs must be unique in a category.` });
      subIds.add(s.subcategory_id);
      out.push(...textProblems(`${sb}.name`, cid, `${who} name`, s.name, 40, true));
      out.push(...textProblems(`${sb}.gloss`, cid, `${who} option text`, s.gloss, 80, true));
      out.push(...textProblems(`${sb}.description`, cid, `${who} description`, s.description, 240, false));
      out.push(...exampleProblems(sb, cid, s.examples));
      if (s.fields.length > 8) out.push({ path: `${sb}.fields`, categoryId: cid, message: `${who} has ${s.fields.length} fields; a subcategory holds at most 8.` });
      out.push(...fieldProblems(sb, cid, s.fields, c.fields.map((f) => f.field_id)));
      if (s.active && c.active) {
        const key = s.name.trim().toLowerCase();
        if (subNames.has(key)) out.push({ path: `${sb}.name`, categoryId: cid, message: `Two active subcategories of "${c.name}" are named "${s.name}".` });
        subNames.add(key);
        if (s.gloss.length > glossMax) out.push({ path: `${sb}.gloss`, categoryId: cid, message: `${who} option text is ${s.gloss.length} characters; keep it to ${glossMax} so the classifier reads all of it.` });
        const pathFields = c.fields.length + s.fields.length;
        if (pathFields > params.max_fields_per_path) out.push({ path: `${sb}.fields`, categoryId: cid, message: `${who} has ${pathFields} fields with its category's; the limit per path is ${params.max_fields_per_path}.` });
      }
    });
  });
  return out;
}

/** A readable name for a contract path Store refused (`details.field`), e.g.
 * `categories[0].subcategories[2].gloss` → `Caller objective › Cancel account › option text`. */
export function describeTaxonomyPath(taxonomy: SignalTaxonomy | undefined, path: string): string {
  const recipe = /^categories\[(\d+)\]\.recipe(?:\.(lexicon|filter))?(?:.*?\.phrases\[(\d+)\])?/.exec(path);
  if (recipe && taxonomy) {
    const c = taxonomy.categories[Number(recipe[1])];
    if (c) {
      const where = recipe[3] !== undefined ? `phrase ${Number(recipe[3]) + 1}` : recipe[2] === 'filter' ? 'rules' : recipe[2] === 'lexicon' ? 'phrase list' : 'settings';
      return `${c.name} › how it's detected › ${where}`;
    }
  }
  const m = /^categories\[(\d+)\](?:\.subcategories\[(\d+)\])?(?:\.fields\[(\d+)\])?(?:\.(\w+))?(?:\[(\d+)\])?/.exec(path);
  if (!m || !taxonomy) return path;
  const c = taxonomy.categories[Number(m[1])];
  if (!c) return path;
  const parts = [c.name];
  let node: SignalCategory | SignalSubcategory = c;
  if (m[2] !== undefined) {
    const s = c.subcategories[Number(m[2])];
    if (s) {
      parts.push(s.name);
      node = s;
    }
  }
  if (m[3] !== undefined) {
    const f = node.fields[Number(m[3])];
    if (f) parts.push(`field ${f.name}`);
  }
  const attr = m[4];
  const labels: Record<string, string> = { gloss: 'option text', name: 'name', description: 'description', examples: 'example', enum_values: 'value' };
  if (attr) parts.push(`${labels[attr] ?? attr}${m[5] !== undefined ? ` ${Number(m[5]) + 1}` : ''}`);
  return parts.join(' › ');
}

/** A deep copy the editor can mutate freely. */
export function cloneTaxonomy(t: SignalTaxonomy): SignalTaxonomy {
  return JSON.parse(JSON.stringify(t)) as SignalTaxonomy;
}

export function sameTaxonomy(a: SignalTaxonomy | undefined, b: SignalTaxonomy | undefined): boolean {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

/** "7 days ago" as an RFC 3339 instant (a backfill's `created_after`, the 7-day metrics window). */
export function daysAgoIso(days: number, now = Date.now()): string {
  return new Date(now - days * 86_400_000).toISOString();
}

export function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return `${value === 0 || value >= 10 ? Math.round(value) : value.toFixed(1)}%`;
}
