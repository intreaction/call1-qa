// Contact Signals rules engine (contract 1.4.0, docs/SignalsEmbeddings.md): client-side checks
// for a category's detection recipe, the editor's simple form over a recipe filter, and the words
// the Workbench uses to say why a hit was found. Store stays the authority: its save validator
// applies the same rules (call1/contracts/signals.py `lexicon_phrase_problem`, the recipe
// validators and `signal_taxonomy_cap_violations`) and refuses by path. These checks exist so an
// admin sees the reason next to the field before saving.

import type {
  ContactSignalView,
  ContractParameters,
  SignalCategory,
  SignalHitWhy,
  SignalLexicon,
  SignalRecipe,
  SignalRule,
  SignalRuleExpr,
  SignalTaxonomy,
  SpeakerRole,
} from './types';

// --- contract constants (call1/contracts/signals.py) ------------------------------------------

export const LEXICON_PHRASE_MAX_CHARS = 300;
export const MAX_LEXICON_PHRASES_CEILING = 64;
export const MAX_RECIPE_RULES_CEILING = 16;
export const MAX_RULE_DEPTH = 3;
export const NEGATION_VETO_MAX_WORDS = 6;
export const RECIPE_THRESHOLD_MIN = 0.05;
export const RECIPE_THRESHOLD_MAX = 0.95;
export const RECIPE_DEFAULT_THRESHOLD = 0.375;

export type RecipeCapParams = Pick<ContractParameters, 'max_signal_recipe_rules' | 'max_signal_lexicon_phrases'>;
export type Detection = 'model' | 'rules';
export type LexiconSyntax = SignalLexicon['syntax'];

export const DETECTION_LABEL: Record<Detection, string> = { model: 'Model (Gemma)', rules: 'Rules + examples' };
export const DETECTION_TEXT: Record<Detection, string> = {
  model: 'Gemma decides every category and subcategory. Rules recipes are kept but not used',
  rules: 'Categories with a rules recipe are found by phrases and similar examples, with no model; Gemma fills fields, checks where asked, and still decides categories without a recipe',
};

/** Store's §9.4 refusal text for definition text a PII detector matches (signal_store.check_text). */
export const PII_TEXT_MESSAGE = 'Definition text looks like caller details (a number the PII rules mask). Describe the behavior instead.';

// --- lexicon phrases ----------------------------------------------------------------------------

const ONE_LINE = /^\S(?:.*\S)?$/;
const LETTER = /\p{L}/u;

/**
 * Why `pattern` is outside the safe regex subset, or null: mirrors `_regex_problem`. Allowed:
 * literals, classes, `\b` and the like, `(...)`/`(?:...)` groups, alternation and repeats.
 * Refused: look-arounds, named groups, inline flags, back-references, conditionals, and an
 * unbounded repeat around a repeat (catastrophic backtracking).
 */
export function regexProblem(pattern: string): string | null {
  const unescaped = pattern.replace(/\\[\s\S]/g, '');
  if (/\(\?(?!:)/.test(unescaped)) return 'only (...) and (?:...) groups are allowed: no look-arounds, named groups or inline flags';
  try {
    new RegExp(pattern, 'i');
  } catch (err) {
    const why = err instanceof Error ? err.message.replace(/^Invalid regular expression: /, '') : String(err);
    return `not a valid pattern (${why.slice(0, 80)})`;
  }
  // Walk the pattern: a frame per group records whether it holds a repeat anywhere inside.
  const frames: { hasRepeat: boolean }[] = [{ hasRepeat: false }];
  let i = 0;
  // The atom a following quantifier applies to: a group's `hasRepeat`, or false for a single item.
  let lastAtom: { hasRepeat: boolean } | null = null;
  while (i < pattern.length) {
    const ch = pattern[i];
    if (ch === '\\') {
      const next = pattern[i + 1] ?? '';
      if (/[1-9]/.test(next)) return 'back-references, look-arounds and conditionals are not allowed';
      i += 2;
      lastAtom = { hasRepeat: false };
      continue;
    }
    if (ch === '[') {
      i += 1;
      if (pattern[i] === '^') i += 1;
      if (pattern[i] === ']') i += 1;
      while (i < pattern.length && pattern[i] !== ']') i += pattern[i] === '\\' ? 2 : 1;
      i += 1;
      lastAtom = { hasRepeat: false };
      continue;
    }
    if (ch === '(') {
      frames.push({ hasRepeat: false });
      i += pattern.startsWith('(?:', i) ? 3 : 1;
      lastAtom = null;
      continue;
    }
    if (ch === ')') {
      const closed = frames.length > 1 ? frames.pop()! : { hasRepeat: false };
      if (closed.hasRepeat) frames[frames.length - 1].hasRepeat = true;
      i += 1;
      lastAtom = closed;
      continue;
    }
    if (ch === '|') {
      i += 1;
      lastAtom = null;
      continue;
    }
    const quant = /^(?:[*+?]|\{(\d*)(,?)(\d*)\})/.exec(pattern.slice(i));
    const braces = quant && quant[0].startsWith('{');
    const validBraces = braces && (quant[1] !== '' || quant[3] !== '');
    if (quant && (!braces || validBraces)) {
      const unbounded = quant[0] === '*' || quant[0] === '+' || (braces && quant[2] === ',' && quant[3] === '');
      if (unbounded && lastAtom?.hasRepeat) return 'an unbounded repeat around another repeat is not allowed';
      frames[frames.length - 1].hasRepeat = true;
      i += quant[0].length;
      if (pattern[i] === '?' || pattern[i] === '+') i += 1; // lazy or possessive
      lastAtom = null;
      continue;
    }
    i += 1;
    lastAtom = { hasRepeat: false };
  }
  return null;
}

/** Why a lexicon phrase is refused, or null: mirrors `lexicon_phrase_problem` plus the
 * `LexiconPhrase` bounds (one line, no leading or trailing space, at most 300 characters). */
export function lexiconPhraseProblem(phrase: string, syntax: LexiconSyntax): string | null {
  if (!phrase) return 'a phrase is not empty';
  if (!ONE_LINE.test(phrase)) return 'a phrase is one line, with no leading or trailing space';
  if (phrase.length > LEXICON_PHRASE_MAX_CHARS) return `a phrase is at most ${LEXICON_PHRASE_MAX_CHARS} characters (this has ${phrase.length})`;
  if (!LETTER.test(phrase)) return 'a phrase contains at least one letter';
  if (syntax === 'regex') return regexProblem(phrase);
  return null;
}

/** A rough, local stand-in for Store's number-rule PII detectors (`call1.redaction.find_pii`): a
 * run of four or more digits. Store may still refuse text this passes, and names the path. */
export function looksLikeCallerNumber(text: string): boolean {
  return /\d(?:[\s().-]?\d){3,}/.test(text);
}

// --- the editor's simple form over a recipe filter ---------------------------------------------

/** What else a segment must pass before its score counts (the recipe filter, beside speaker and
 * call position). */
export type RecipeGate = 'none' | 'phrase' | 'similar' | 'phrase_or_similar' | 'phrase_and_similar';

export const GATE_LABEL: Record<RecipeGate, string> = {
  none: 'Nothing else: the score decides',
  phrase: 'A phrase matches',
  similar: 'It is similar to examples',
  phrase_or_similar: 'A phrase matches, or it is similar to examples',
  phrase_and_similar: 'A phrase matches and it is similar to examples',
};

export interface RecipeForm {
  speaker: { ruleId: string; speaker: SpeakerRole } | null;
  position: { ruleId: string; from: number; to: number } | null;
  gate: RecipeGate;
  minShare: number;
  phraseRuleId: string;
  similarRuleId: string;
}

export const DEFAULT_MIN_SHARE = 0.1;

/** Call position presets: where the segment starts, as a fraction of the call (quartile chips). */
export const POSITION_PRESETS: { id: string; label: string; from: number; to: number }[] = [
  { id: 'q1', label: 'Only in the first quarter', from: 0, to: 0.25 },
  { id: 'h1', label: 'Only in the first half', from: 0, to: 0.5 },
  { id: 'after_q1', label: 'After the first quarter', from: 0.25, to: 1 },
  { id: 'h2', label: 'Only in the second half', from: 0.5, to: 1 },
  { id: 'q4', label: 'Only in the last quarter', from: 0.75, to: 1 },
];

export function positionPresetId(position: { from: number; to: number } | null): string {
  if (!position) return 'anywhere';
  return POSITION_PRESETS.find((p) => p.from === position.from && p.to === position.to)?.id ?? 'custom';
}

export function positionText(from: number, to: number): string {
  const preset = POSITION_PRESETS.find((p) => p.from === from && p.to === to);
  if (preset) return preset.label.replace(/^Only in/, 'in').replace(/^After/, 'after');
  return `starting between ${Math.round(from * 100)}% and ${Math.round(to * 100)}% of the call`;
}

type Leaf = SignalRule['params'];

function leafOf(expr: SignalRuleExpr): SignalRule | null {
  return expr.op === 'rule' && expr.rule ? expr.rule : null;
}

function isLexiconPhrase(params: Leaf): boolean {
  return params.type === 'phrase' && (params.phrases === null || params.phrases === undefined);
}

/** The simple form of a recipe filter, or null when the filter is richer than the form (a pack's
 * `not`, nested groups, or a phrase rule with its own phrases): the editor then shows it as it is. */
export function parseRecipeForm(filter: SignalRuleExpr | null | undefined): RecipeForm | null {
  const form: RecipeForm = { speaker: null, position: null, gate: 'none', minShare: DEFAULT_MIN_SHARE, phraseRuleId: 'lexicon', similarRuleId: 'similar' };
  if (!filter) return form;
  const children = filter.op === 'rule' ? [filter] : filter.op === 'all' ? filter.children : null;
  if (!children) return null;
  let phrase = false;
  let similar = false;
  let either = false;
  for (const child of children) {
    const single = leafOf(child);
    if (single) {
      const p = single.params;
      if (p.type === 'speaker' && !form.speaker) form.speaker = { ruleId: single.rule_id, speaker: p.speaker };
      else if (p.type === 'call_position' && !form.position) form.position = { ruleId: single.rule_id, from: p.start_from, to: p.start_to };
      else if (isLexiconPhrase(p) && !phrase && !either) {
        phrase = true;
        form.phraseRuleId = single.rule_id;
      } else if (p.type === 'similar_to_examples' && !similar && !either) {
        similar = true;
        form.similarRuleId = single.rule_id;
        form.minShare = p.min_share;
      } else return null;
      continue;
    }
    if (child.op === 'any' && child.children.length === 2 && !phrase && !similar && !either) {
      const [a, b] = child.children.map(leafOf);
      const pr = [a, b].find((r) => r && isLexiconPhrase(r.params));
      const sr = [a, b].find((r) => r && r.params.type === 'similar_to_examples');
      if (!pr || !sr || sr.params.type !== 'similar_to_examples') return null;
      either = true;
      form.phraseRuleId = pr.rule_id;
      form.similarRuleId = sr.rule_id;
      form.minShare = sr.params.min_share;
      continue;
    }
    return null;
  }
  form.gate = either ? 'phrase_or_similar' : phrase && similar ? 'phrase_and_similar' : phrase ? 'phrase' : similar ? 'similar' : 'none';
  return form;
}

function leaf(ruleId: string, params: Leaf): SignalRuleExpr {
  return { op: 'rule', rule: { rule_id: ruleId, params }, children: [] };
}

/** The recipe filter for a simple form: `all` over speaker, call position and the gate (null when
 * nothing is required). Rule IDs are kept from the parsed filter. */
export function buildRecipeFilter(form: RecipeForm): SignalRuleExpr | null {
  const children: SignalRuleExpr[] = [];
  if (form.speaker) children.push(leaf(form.speaker.ruleId, { type: 'speaker', speaker: form.speaker.speaker }));
  if (form.position) children.push(leaf(form.position.ruleId, { type: 'call_position', start_from: form.position.from, start_to: form.position.to }));
  const phrase = leaf(form.phraseRuleId, { type: 'phrase', syntax: 'words', phrases: null, negation_veto_words: 0 });
  const similar = leaf(form.similarRuleId, { type: 'similar_to_examples', min_share: form.minShare });
  if (form.gate === 'phrase') children.push(phrase);
  if (form.gate === 'similar') children.push(similar);
  if (form.gate === 'phrase_and_similar') children.push(phrase, similar);
  if (form.gate === 'phrase_or_similar') children.push({ op: 'any', rule: null, children: [phrase, similar] });
  return children.length ? { op: 'all', rule: null, children } : null;
}

export function gateUsesPhrase(gate: RecipeGate): boolean {
  return gate === 'phrase' || gate === 'phrase_or_similar' || gate === 'phrase_and_similar';
}

export function gateUsesSimilar(gate: RecipeGate): boolean {
  return gate === 'similar' || gate === 'phrase_or_similar' || gate === 'phrase_and_similar';
}

/** A new rules recipe for a category that had none: kNN share only, the default threshold. */
export function newRulesRecipe(): SignalRecipe {
  return { engine: 'rules', filter: null, lexicon: null, lexicon_weight: 0, threshold: RECIPE_DEFAULT_THRESHOLD, check: 'none', origin: null };
}

export function filterRules(expr: SignalRuleExpr | null | undefined): SignalRule[] {
  if (!expr) return [];
  if (expr.rule) return [expr.rule];
  return expr.children.flatMap(filterRules);
}

function filterDepth(expr: SignalRuleExpr): number {
  return expr.op === 'rule' ? 0 : 1 + Math.max(0, ...expr.children.map(filterDepth));
}

/** Keep a recipe's speaker rules in step with the category's speaker (a speaker rule must agree
 * with it). A category open to either speaker leaves them as they are. */
export function syncRecipeSpeaker(c: SignalCategory): void {
  if (!c.recipe?.filter || (c.speaker !== 'AGENT' && c.speaker !== 'CALLER')) return;
  const speaker = c.speaker;
  for (const rule of filterRules(c.recipe.filter)) if (rule.params.type === 'speaker') rule.params.speaker = speaker;
}

/** "Retail pack v1, tuned on 25 public calls" for a pack origin (`pack:retail@1 (…)`), else the
 * origin as written. Null when the recipe has none (written here). */
export function recipeOriginText(origin: string | null | undefined): string | null {
  if (!origin) return null;
  const m = /^pack:([a-z0-9_-]+)@(\d+)\b\s*(?:\((.*)\))?/i.exec(origin);
  if (!m) return origin;
  const name = m[1].charAt(0).toUpperCase() + m[1].slice(1).replace(/[_-]/g, ' ');
  // The retail pack's recipes were tuned on the 25 public training calls of its example bank.
  const tuned = m[1] === 'retail' && m[2] === '1' ? ', tuned on 25 public calls' : '';
  return `${name} pack v${m[2]}${tuned}${m[3] ? ` (${m[3]})` : ''}`;
}

/** A one-line reading of a filter the simple form cannot edit, e.g.
 * "all of: speaker Caller, any of: (phrase, similar ≥ 0.10)". */
export function filterText(expr: SignalRuleExpr | null | undefined): string {
  if (!expr) return 'no filter';
  if (expr.rule) {
    const p = expr.rule.params;
    switch (p.type) {
      case 'speaker':
        return `speaker ${p.speaker === 'AGENT' ? 'Agent' : 'Caller'}`;
      case 'call_position':
        return positionText(p.start_from, p.start_to);
      case 'similar_to_examples':
        return `similar to examples ≥ ${p.min_share.toFixed(2)}`;
      case 'phrase':
        return p.phrases ? `one of ${p.phrases.length} own phrase${p.phrases.length === 1 ? '' : 's'}` : 'a lexicon phrase';
    }
  }
  const inner = expr.children.map(filterText).join(', ');
  if (expr.op === 'not') return `not (${inner})`;
  return `${expr.op === 'all' ? 'all of' : 'any of'}: (${inner})`;
}

// --- recipe problems ----------------------------------------------------------------------------

export interface RecipeProblem {
  path: string;
  message: string;
}

/** Every problem Store's save would refuse in one category's recipe that Evaluate can see locally
 * (the contract validators and the 1.4.0 caps). `base` is the category's path. */
export function recipeProblems(c: SignalCategory, base: string, caps: RecipeCapParams): RecipeProblem[] {
  const r = c.recipe;
  if (!r) return [];
  const out: RecipeProblem[] = [];
  const who = `"${c.name || c.category_id}" detection`;
  const p = `${base}.recipe`;
  if (!(r.threshold >= RECIPE_THRESHOLD_MIN && r.threshold <= RECIPE_THRESHOLD_MAX)) out.push({ path: `${p}.threshold`, message: `${who}: the score needed is from ${RECIPE_THRESHOLD_MIN} to ${RECIPE_THRESHOLD_MAX}.` });
  if (!(r.lexicon_weight >= 0 && r.lexicon_weight <= 1)) out.push({ path: `${p}.lexicon_weight`, message: `${who}: the phrase weight is from 0 to 1.` });
  if (r.lexicon_weight > 0 && !r.lexicon) out.push({ path: `${p}.lexicon_weight`, message: `${who}: a phrase weight needs a phrase list.` });
  const phraseCap = Math.min(caps.max_signal_lexicon_phrases, MAX_LEXICON_PHRASES_CEILING);
  const checkPhrases = (path: string, phrases: string[], syntax: LexiconSyntax, what: string) => {
    if (phrases.length === 0) out.push({ path, message: `${who}: ${what} needs at least one phrase.` });
    if (phrases.length > phraseCap) out.push({ path, message: `${who}: ${what} has ${phrases.length} phrases; the limit is ${phraseCap}.` });
    phrases.forEach((ph, k) => {
      const problem = lexiconPhraseProblem(ph, syntax);
      if (problem) out.push({ path: `${path}[${k}]`, message: `${who}: phrase ${k + 1}: ${problem}.` });
    });
  };
  if (r.lexicon) {
    checkPhrases(`${p}.lexicon.phrases`, r.lexicon.phrases, r.lexicon.syntax, 'the phrase list');
    const n = r.lexicon.negation_veto_words;
    if (!(Number.isInteger(n) && n >= 0 && n <= NEGATION_VETO_MAX_WORDS)) out.push({ path: `${p}.lexicon.negation_veto_words`, message: `${who}: the negation window is 0 to ${NEGATION_VETO_MAX_WORDS} words.` });
  }
  if (r.filter) {
    const rules = filterRules(r.filter);
    const ruleCap = Math.min(caps.max_signal_recipe_rules, MAX_RECIPE_RULES_CEILING);
    if (rules.length > ruleCap) out.push({ path: `${p}.filter`, message: `${who}: ${rules.length} rules; the limit is ${ruleCap}.` });
    if (filterDepth(r.filter) > MAX_RULE_DEPTH) out.push({ path: `${p}.filter`, message: `${who}: the filter nests more than ${MAX_RULE_DEPTH} deep.` });
    const ids = rules.map((x) => x.rule_id);
    if (new Set(ids).size !== ids.length) out.push({ path: `${p}.filter`, message: `${who}: rule IDs must be unique.` });
    const walk = (e: SignalRuleExpr, path: string) => {
      if ((e.op === 'all' || e.op === 'any') && e.children.length === 0) out.push({ path, message: `${who}: an "${e.op}" group needs at least one rule.` });
      if (e.op === 'not' && e.children.length !== 1) out.push({ path, message: `${who}: "not" takes exactly one rule.` });
      if (e.rule) {
        const prm = e.rule.params;
        if (prm.type === 'phrase') {
          if (prm.phrases) checkPhrases(`${path}.rule.params.phrases`, prm.phrases, prm.syntax, `rule ${e.rule.rule_id}`);
          else if (!r.lexicon) out.push({ path: `${p}.lexicon`, message: `${who}: "a phrase matches" tests the phrase list, so add at least one phrase.` });
        }
        if (prm.type === 'speaker' && c.speaker && (c.speaker === 'AGENT' || c.speaker === 'CALLER') && prm.speaker !== c.speaker) {
          out.push({ path: `${path}.rule.params.speaker`, message: `${who}: the speaker rule must match the category's speaker.` });
        }
        if (prm.type === 'call_position' && !(prm.start_from >= 0 && prm.start_to <= 1 && prm.start_to > prm.start_from)) {
          out.push({ path: `${path}.rule.params`, message: `${who}: the call position ends after it starts, between 0 and 1.` });
        }
        if (prm.type === 'similar_to_examples' && !(prm.min_share >= 0 && prm.min_share <= 1)) {
          out.push({ path: `${path}.rule.params.min_share`, message: `${who}: the similar-examples share is from 0 to 1.` });
        }
      }
      e.children.forEach((child, k) => walk(child, `${path}.children[${k}]`));
    };
    walk(r.filter, `${p}.filter`);
  }
  return out;
}

/** The categories whose recipe would be decided by rules under rules detection. */
export function rulesCategories(taxonomy: SignalTaxonomy | undefined): SignalCategory[] {
  return (taxonomy?.categories ?? []).filter((c) => c.active && c.recipe?.engine === 'rules');
}

/** Did any category's recipe (or the taxonomy's rules settings) change between two versions? */
export function recipesChanged(before: SignalTaxonomy | undefined, after: SignalTaxonomy | undefined): boolean {
  if (!before || !after) return false;
  if (JSON.stringify(before.rules ?? null) !== JSON.stringify(after.rules ?? null)) return true;
  const old = new Map(before.categories.map((c) => [c.category_id, JSON.stringify(c.recipe ?? null)]));
  return after.categories.some((c) => (old.get(c.category_id) ?? 'null') !== JSON.stringify(c.recipe ?? null));
}

// --- why a hit was found (ContactSignalView.why) -----------------------------------------------

export function hitWhy(sig: Pick<ContactSignalView, 'why'> | { why?: SignalHitWhy | null }): SignalHitWhy | null {
  return sig.why ?? null;
}

export function sourceLabel(source: 'rules' | 'gemma' | null | undefined): string {
  return source === 'rules' ? 'rules' : source === 'gemma' ? 'Gemma' : 'not decided';
}

/** The lexicon phrase a rule decision names, from the taxonomy the result was scored with. */
export function decisionPhrase(taxonomy: SignalTaxonomy | undefined, categoryId: string, index: number | null | undefined): string | null {
  if (index === null || index === undefined) return null;
  const lexicon = taxonomy?.categories.find((c) => c.category_id === categoryId)?.recipe?.lexicon;
  return lexicon?.phrases[index] ?? null;
}

export function fmt2(n: number): string {
  return (Math.round(n * 100) / 100).toFixed(2);
}

/** System One metadata distinguishes the semantic cascade from historical rule decisions. */
export function isSemanticDecision(why: SignalHitWhy): boolean {
  const rule = why.rule;
  return why.category_source === 'rules' && rule != null &&
    (rule.system_one_score != null || rule.system_one_fallback != null || rule.system_one_kept === true);
}

/** The compact "why" line: "Found by rules · score 0.62 ≥ 0.38 · similar examples 0.55 · phrase
 * 'refund' (matched)", or "Found by Gemma". */
export function whySummary(why: SignalHitWhy, phrase: string | null): string {
  if (why.category_source !== 'rules' || !why.rule) return 'Found by Gemma';
  const r = why.rule;
  const semantic = isSemanticDecision(why);
  const parts = [semantic ? 'Found by semantic similarity' : 'Found by rules', `score ${fmt2(r.score)} ≥ ${fmt2(r.threshold)}`, `similar examples ${fmt2(r.knn_share)}`];
  if (r.lexicon_match) {
    const shown = phrase ? ` '${phrase.length > 40 ? `${phrase.slice(0, 39)}…` : phrase}'` : r.lexicon_phrase !== null ? ` ${r.lexicon_phrase + 1}` : '';
    parts.push(`phrase${shown} (matched${r.lexicon_weight > 0 ? `, +${fmt2(r.lexicon_weight)}` : ''})`);
  } else if (r.lexicon_weight > 0) {
    parts.push('no phrase matched');
  }
  return parts.join(' · ');
}

export const RULE_TYPE_LABEL: Record<SignalRule['params']['type'], string> = {
  similar_to_examples: 'Similar to examples',
  phrase: 'Phrase',
  speaker: 'Speaker',
  call_position: 'Call position',
};
