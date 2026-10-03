// Client-side mirror of call1/contracts/vocabulary.py's term rule (docs/DualAsr.md section 8: "live
// validation that mirrors vocabulary_term_problem"). Store's save is authoritative and re-checks
// every term; this exists only so the admin editor can say why a term is refused before a round
// trip, and to preview the effective vocabulary while a draft is unsaved.

export const VOCABULARY_TERM_MAX_CHARS = 60;
export const VOCABULARY_TERM_MAX_WORDS = 6;

const PUNCTUATION = new Set([' ', "'", '’', '&', '.', '-']);

export type VocabularyTermProblem =
  | 'not_normalized'
  | 'empty'
  | 'too_long'
  | 'too_many_words'
  | 'digit'
  | 'character'
  | 'must_start_with_letter'
  | 'too_few_letters';

/** NFC, trimmed, internal whitespace collapsed to single spaces (normalize_vocabulary_term). */
export function normalizeVocabularyTerm(term: string): string {
  return term.normalize('NFC').split(/\s+/).filter(Boolean).join(' ');
}

const IS_LETTER = /\p{L}/u;
const IS_MARK = /\p{M}/u;
const IS_DIGIT = /[\p{Nd}\p{Nl}\p{No}]/u;

/** Why `term` is not a valid vocabulary term, or null (vocabulary_term_problem). */
export function vocabularyTermProblem(term: string): VocabularyTermProblem | null {
  if (term !== normalizeVocabularyTerm(term)) return 'not_normalized';
  if (!term) return 'empty';
  if (term.length > VOCABULARY_TERM_MAX_CHARS) return 'too_long';
  if (term.split(' ').length > VOCABULARY_TERM_MAX_WORDS) return 'too_many_words';
  let letters = 0;
  for (const ch of term) {
    if (IS_DIGIT.test(ch)) return 'digit';
    if (IS_LETTER.test(ch)) {
      letters += 1;
      continue;
    }
    if (IS_MARK.test(ch) || PUNCTUATION.has(ch)) continue;
    return 'character';
  }
  if (!IS_LETTER.test(term[0])) return 'must_start_with_letter';
  if (letters < 2) return 'too_few_letters';
  return null;
}

/** The identity of a term: accents stripped, case folded, letters only ("Wi-Fi" -> "wifi"). Two
 * terms with one key are the same term (vocabulary_term_key). */
export function vocabularyTermKey(term: string): string {
  const decomposed = term.normalize('NFKD').toLowerCase();
  let out = '';
  for (const ch of decomposed) if (IS_LETTER.test(ch)) out += ch;
  return out;
}

const PROBLEM_TEXT: Record<VocabularyTermProblem, string> = {
  not_normalized: 'Extra or unusual spacing — retype the term.',
  empty: 'Enter a term.',
  too_long: `Terms are at most ${VOCABULARY_TERM_MAX_CHARS} characters.`,
  too_many_words: `Terms are at most ${VOCABULARY_TERM_MAX_WORDS} words.`,
  digit: "Terms can't contain numbers.",
  character: "Only letters, spaces and ' ’ & . - are allowed.",
  must_start_with_letter: 'Terms must start with a letter.',
  too_few_letters: 'Terms need at least two letters.',
};

export function vocabularyTermProblemText(problem: VocabularyTermProblem): string {
  return PROBLEM_TEXT[problem];
}

/** A safe sentence for a `validation_failed` refusal on `saveAsrVocabulary`: `details.reason` is
 * either a `vocabulary_term_problem` code, or `too_many_terms` / `unknown_pack_term` / `pii_detected`
 * (docs/DualAsr.md section 4). Store never echoes the term. */
export function vocabularySaveReasonText(reason: unknown, maxTerms: number): string {
  if (reason === 'too_many_terms') return `At most ${maxTerms} of your own terms.`;
  if (reason === 'unknown_pack_term') return "That isn't a term of the installed pack.";
  if (reason === 'pii_detected') return "That looks like a person's name or other personal detail, not a business term.";
  if (typeof reason === 'string' && reason in PROBLEM_TEXT) return PROBLEM_TEXT[reason as VocabularyTermProblem];
  return 'Store refused this term.';
}

export type VocabularyTermPreview = { term: string; source: 'industry_pack' | 'customer' };

/** Local preview of the effective vocabulary while editing, before a save round trip: pack terms
 * not disabled, then customer terms not already present by key (effective_vocabulary). */
export function effectiveVocabularyPreview(packTerms: string[], disabledPackTerms: string[], customerTerms: string[]): VocabularyTermPreview[] {
  const disabled = new Set(disabledPackTerms.map(vocabularyTermKey));
  const seen = new Set<string>();
  const out: VocabularyTermPreview[] = [];
  for (const term of packTerms) {
    const key = vocabularyTermKey(term);
    if (disabled.has(key) || seen.has(key)) continue;
    seen.add(key);
    out.push({ term, source: 'industry_pack' });
  }
  for (const term of customerTerms) {
    const key = vocabularyTermKey(term);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ term, source: 'customer' });
  }
  return out;
}

/** Whether dual transcription would run with this draft (vocabulary_active): enabled and non-empty. */
export function vocabularyActivePreview(enabled: boolean, effective: VocabularyTermPreview[]): boolean {
  return enabled && effective.length > 0;
}
