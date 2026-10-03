"""Dual transcription merge: the base transcript (Parakeet) corrected on the customer's vocabulary
from a vocabulary-prompted Whisper pass (team decision 33, docs/DualAsr.md sections 1 and 5).

Pure functions: no model, file or network access. Ported from the research
(``scripts/research/asr_merge/merge.py`` and ``common.py``, benchmarks/2026-09-26-dual-asr-merge.md)
with the same normalisation, similarity measures and candidate rule, so the frozen thresholds keep
their meaning:

* **Candidates.** Each Whisper hit on a vocabulary term (``vocab_hits``: normalised joined letters,
  longest term first) is paired with the base words that overlap it in time (``time_slack_seconds``
  each side), narrowed to the contiguous sub-span of at most the term's length +
  ``max_span_extra_tokens`` tokens most similar to the term. A hit with no overlapping base word is
  dropped: **the merge never inserts**. A span whose tokens already equal the term's is a no-op.
* **The rule** (``rule_accepts``): Double Metaphone similarity >= ``phonetic_min``, character
  similarity >= ``character_min``, word counts differing by <= ``max_word_delta``.
* **Applying** writes the term over the base span, keeping the base timings.

Product differences from the research (docs/DualAsr.md section 5):

* the thresholds come from the frozen ``VocabularyMergeRule``;
* candidates are generated per channel, and a span that crosses a turn is rejected;
* applying rewrites the merged turn's ``text`` over the character range of the replaced words,
  keeping punctuation outside the first and last word ("cup." becomes "Stanley cup."), and its
  ``word_timestamps`` (the term's words share the base span evenly, with the base words' mean
  probability); a candidate whose words cannot be located in the turn text is rejected;
* when accepted spans overlap, the earlier-listed hit wins (the research's score is 0 under the rule);
* no ``rapidfuzz``: the pure-Python Levenshtein the research falls back to gives the same numbers;
* Double Metaphone is vendored (``_vendor/doublemetaphone.py``, Metaphone 0.6, BSD 3-clause);
* the glossary shortlist (``rank_terms``) is the research's ranking with exact pruning bounds; above
  ``EXACT_SHORTLIST_TERMS`` terms it also prefilters windows by first letter or first phonetic code
  before scoring, so a catalogue of 2000 terms stays cheap.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from call1.contracts.contents import AsrPassWord, TranscriptReplacement, TranscriptTurnContent, WordTimestampView
from call1.contracts.vocabulary import AsrVocabularyTerm, VocabularyMergeRule, VocabularyTermSource

from ._vendor.doublemetaphone import doublemetaphone

DEFAULT_RULE = VocabularyMergeRule()
SLACK_SECONDS = DEFAULT_RULE.time_slack_seconds
RULE_PHONETIC = DEFAULT_RULE.phonetic_min
RULE_CHARACTER = DEFAULT_RULE.character_min
RULE_MAX_WORD_DELTA = DEFAULT_RULE.max_word_delta

TEXT_LIMIT = 400
"""``TranscriptReplacement.heard`` and ``candidate_text`` are at most this many characters."""


# ---------------------------------------------------------------------------------- normalisation
FILLERS = {"um", "uh", "ah", "hm", "hmm", "mhm", "mm", "er", "erm", "eh", "oh", "ohh", "huh", "uhm", "mmm"}
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> List[str]:
    """Integer to English words (0 <= n < 10**9), the way the research normaliser spells numbers."""
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (number_words(n % 100) if n % 100 else [])
    for size, name in ((10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            return number_words(n // size) + [name] + (number_words(n % size) if n % size else [])
    return [str(n)]


def _digits_to_words(token: str) -> List[str]:
    t = token
    out: List[str] = []
    suffix: List[str] = []
    if t.startswith("$"):
        t, suffix = t[1:], ["dollars"]
    if t.endswith("%"):
        t, suffix = t[:-1], ["percent"]
    m = re.fullmatch(r"(\d+)(st|nd|rd|th|s|k|gb)?", t)
    if not m:
        parts = re.split(r"[:.,/]", t)
        if all(p.isdigit() for p in parts if p) and any(parts):
            for p in parts:
                if p:
                    out += number_words(int(p)) if len(p) < 10 else list(p)
            return out + suffix
        return [token]
    digits, tail = m.group(1), m.group(2)
    if len(digits) > 4 or (len(digits) > 1 and digits.startswith("0")):
        out = [_ONES[int(d)] for d in digits]  # a read-out code: digit by digit
    elif len(digits) == 4 and 1900 <= int(digits) <= 2099:
        out = number_words(int(digits[:2])) + (number_words(int(digits[2:])) if digits[2:] != "00" else ["hundred"])
    else:
        out = number_words(int(digits))
    if tail == "gb":
        out.append("gigabytes")
    elif tail == "k":
        out.append("thousand")
    return out + suffix


def normalize_text(text: str, drop_fillers: bool = True) -> List[str]:
    """Lower-case word tokens for term matching (the research's ``common.normalize_text``): drops
    parenthesised marks and partial words ("f~"), strips accents, apostrophes and punctuation, splits
    hyphens, spells digits out, and folds "ok" into "okay"."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\S*~\S*", " ", text)
    text = text.lower().replace("’", "'").replace("&", " and ")
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)          # don't -> dont, curry's -> currys
    text = re.sub(r"[-–—_/]", " ", text)
    tokens: List[str] = []
    for raw in text.split():
        raw = raw.strip(".,?!;:\"'()[]{}…")
        if not raw:
            continue
        if re.search(r"\d", raw):
            tokens += [w for w in _digits_to_words(raw)]
            continue
        raw = re.sub(r"[^a-z0-9]", "", raw)
        if not raw:
            continue
        if raw == "ok":
            raw = "okay"
        if drop_fillers and raw in FILLERS:
            continue
        tokens.append(raw)
    return tokens


def term_tokens(term: str) -> List[str]:
    return normalize_text(term, drop_fillers=False)


# ------------------------------------------------------------------------------------- similarity
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def string_similarity(a: str, b: str) -> float:
    """1 - normalised Levenshtein distance, in [0, 1] (equal to rapidfuzz's normalized_similarity)."""
    if not a and not b:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max(len(a), len(b))


@lru_cache(maxsize=200_000)
def word_codes(token: str) -> Tuple[str, str]:
    """(primary, secondary) Double Metaphone codes of one normalised token (the secondary falls back
    to the primary, and an empty code to the token's first letter)."""
    primary, secondary = doublemetaphone(token)
    return primary or token[:1].upper(), secondary or primary or token[:1].upper()


def phonetic_similarity(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    """Best similarity between the concatenated Double Metaphone codes of two token sequences
    (primary with primary or secondary, and secondary with either), so "still organ" ~ "Stillorgan"."""
    if not a_tokens or not b_tokens:
        return 0.0
    a = [word_codes(t) for t in a_tokens]
    b = [word_codes(t) for t in b_tokens]
    best = 0.0
    for ia in (0, 1):
        for ib in (0, 1):
            best = max(best, string_similarity("".join(c[ia] for c in a), "".join(c[ib] for c in b)))
    return best


def character_similarity(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    return string_similarity("".join(a_tokens), "".join(b_tokens))


# ------------------------------------------------------------------------------------- vocab hits
def flatten(words: Sequence[Dict]) -> Tuple[List[str], List[int]]:
    """Normalised tokens of a word list, and for each token the index of the word it came from."""
    tokens, owner = [], []
    for i, w in enumerate(words):
        for t in normalize_text(w["word"], drop_fillers=False):
            tokens.append(t)
            owner.append(i)
    return tokens, owner


def _match_at(tokens: Sequence[str], i: int, target: str, max_size: int) -> Optional[int]:
    joined = ""
    for size in range(1, max_size + 1):
        if i + size > len(tokens):
            return None
        joined += tokens[i + size - 1]
        if joined == target:
            return i + size
        if not target.startswith(joined):
            return None
    return None


def vocab_hits(tokens: Sequence[str], vocab: Sequence[str]) -> List[Tuple[int, int, int]]:
    """Non-overlapping (start, end, term index) token spans, scanning left to right and preferring
    the longest term at each position ("Ticketek VIP" over "Ticketek"). Terms are matched on joined
    letters, so spacing and hyphenation differences do not matter. Targets are indexed by first
    letter (the research's scan, without visiting every term at every token)."""
    targets = []
    for k, term in enumerate(vocab):
        tt = term_tokens(term)
        targets.append(("".join(tt), len(tt), k))
    targets.sort(key=lambda x: (-len(x[0]), x[2]))  # longest first, then vocabulary order (the research's stable sort)
    by_first: Dict[str, List[Tuple[str, int, int]]] = {}
    for target, n, k in targets:
        if target:
            by_first.setdefault(target[0], []).append((target, n, k))
    hits = []
    i = 0
    while i < len(tokens):
        for target, n, k in by_first.get(tokens[i][:1], ()):
            end = _match_at(tokens, i, target, n + 1)
            if end:
                hits.append((i, end, k))
                i = end
                break
        else:
            i += 1
    return hits


def term_relevance(term: str, words: Sequence[Dict]) -> float:
    """How strongly a transcript suggests ``term`` was said: the best 0.5*phonetic + 0.5*character
    similarity over windows of the term's length +-1 (the research's exact score)."""
    tt = term_tokens(term)
    tokens, _ = flatten(words)
    best = 0.0
    for size in {max(1, len(tt) - 1), len(tt), len(tt) + 1}:
        for i in range(0, max(0, len(tokens) - size) + 1):
            window = tokens[i:i + size]
            if not window:
                continue
            c = character_similarity(tt, window)
            if c < 0.4:
                continue
            best = max(best, 0.5 * c + 0.5 * phonetic_similarity(tt, window))
    return best


EXACT_SHORTLIST_TERMS = 250
"""Up to this many terms, ``rank_terms`` scores every window (the research's exact ranking); above it,
only windows sharing a prefilter key with the term."""


def _keys(tokens: Sequence[str]) -> set:
    """Prefilter keys of a token sequence: its first letter and the first letter of its phonetic codes."""
    if not tokens or not tokens[0]:
        return set()
    primary, secondary = word_codes(tokens[0])
    return {"c:" + tokens[0][0], "p:" + primary[:1], "p:" + secondary[:1]}


def rank_terms(terms: Sequence[str], words: Sequence[Dict], *, exact_limit: int = EXACT_SHORTLIST_TERMS) -> List[Tuple[str, float]]:
    """``terms`` ordered by relevance to ``words`` (highest first, ties in vocabulary order), for the
    glossary shortlist. Each term's score is ``term_relevance``; windows are deduplicated and skipped
    only when a bound proves they cannot reach the 0.4 character floor or beat the term's best so far
    (length difference, shared letters), so the ranking is exact. Above ``exact_limit`` terms a window
    is also scored only when it shares a prefilter key with the term (first letter, or first Double
    Metaphone letter; docs/DualAsr.md section 5), which keeps a 2000-term catalogue cheap."""
    from collections import Counter

    tokens, _ = flatten(words)
    prefilter = len(terms) > exact_limit
    windows: Dict[int, List[Tuple[Tuple[str, ...], str, "Counter[str]"]]] = {}
    keyed: Dict[int, Dict[str, List[int]]] = {}

    def table(size: int):
        if size not in windows:
            distinct: Dict[Tuple[str, ...], None] = {}
            for i in range(0, max(0, len(tokens) - size) + 1):
                window = tuple(tokens[i:i + size])
                if window:
                    distinct[window] = None
            rows = [(w, "".join(w), Counter("".join(w))) for w in distinct]
            windows[size] = rows
            index: Dict[str, List[int]] = {}
            if prefilter:
                for n, (w, _, _) in enumerate(rows):
                    for key in _keys(w):
                        index.setdefault(key, []).append(n)
            keyed[size] = index
        return windows[size], keyed[size]

    scored = []
    for position, term in enumerate(terms):
        tt = term_tokens(term)
        best = 0.0
        if tt:
            joined = "".join(tt)
            letters = Counter(joined)
            keys = _keys(tt)
            for size in {max(1, len(tt) - 1), len(tt), len(tt) + 1}:
                rows, index = table(size)
                chosen = sorted({n for key in keys for n in index.get(key, ())}) if prefilter else range(len(rows))
                for n in chosen:
                    window, other, other_letters = rows[n]
                    longest = max(len(joined), len(other))
                    if not longest:
                        continue
                    if 1.0 - abs(len(joined) - len(other)) / longest < 0.4:
                        continue  # the character similarity cannot reach 0.4
                    bound = sum((letters & other_letters).values()) / longest
                    if bound < 0.4 or 0.5 * bound + 0.5 <= best:
                        continue  # cannot reach the floor, or cannot beat the best window so far
                    c = string_similarity(joined, other)
                    if c < 0.4 or 0.5 * c + 0.5 <= best:
                        continue
                    best = max(best, 0.5 * c + 0.5 * phonetic_similarity(tt, window))
        scored.append((position, term, best))
    scored.sort(key=lambda x: (-x[2], x[0]))
    return [(term, score) for _, term, score in scored]


def pack_glossary(ranked: Sequence[str], count_tokens: Callable[[str], int], limit: int) -> Tuple[str, List[str]]:
    """The research's glossary packing: walk the ranked terms and keep each one whose addition keeps
    ``" Glossary: <terms>."`` within ``limit`` tokens. Returns (glossary text, chosen terms)."""
    chosen: List[str] = []
    for term in ranked:
        text = ", ".join(chosen + [term])
        if count_tokens(" Glossary: " + text + ".") > limit:
            continue
        chosen.append(term)
    return "Glossary: " + ", ".join(chosen) + ".", chosen


# ------------------------------------------------------------------------------------- candidates
@dataclass
class Candidate:
    term: str
    term_index: int
    w_start: float
    w_end: float
    w_text: str
    p_first: int = -1           # base word span [p_first, p_last), or -1 when nothing overlaps
    p_last: int = -1
    p_text: str = ""
    p_start: float = 0.0
    p_end: float = 0.0
    no_overlap: bool = False
    already_correct: bool = False
    features: Dict[str, float] = field(default_factory=dict)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def generate_candidates(base: Sequence[Dict], whisper: Sequence[Dict], vocab: Sequence[str],
                        rule: VocabularyMergeRule = DEFAULT_RULE) -> List[Candidate]:
    """One candidate per Whisper vocabulary hit, in hit order. The base span is the contiguous run of
    words overlapping the hit's time (+-slack) whose sub-span (at most the term's length +
    ``max_span_extra_tokens`` tokens) is most similar to the term. Words are dicts with ``word``,
    ``start`` and ``end``, each list in time order."""
    slack = rule.time_slack_seconds
    w_tokens, w_owner = flatten(whisper)
    p_norm = [normalize_text(w["word"], drop_fillers=False) for w in base]
    out = []
    for start, end, k in vocab_hits(w_tokens, vocab):
        term = vocab[k]
        tt = term_tokens(term)
        wi0, wi1 = w_owner[start], w_owner[end - 1] + 1
        ws, we = whisper[wi0]["start"], max(whisper[wi0]["end"], whisper[wi1 - 1]["end"])
        cand = Candidate(term=term, term_index=k, w_start=ws, w_end=we, w_text=" ".join(w["word"] for w in whisper[wi0:wi1]))
        overlapping = [i for i, w in enumerate(base)
                       if _overlap(ws - slack, we + slack, w["start"], w["end"]) > 0 or ws - slack <= w["start"] <= we + slack]
        if not overlapping:
            cand.no_overlap = True
            out.append(cand)
            continue
        lo, hi = overlapping[0], overlapping[-1] + 1
        best, best_key = None, None
        for i in range(lo, hi):
            for j in range(i + 1, hi + 1):
                toks = [t for x in p_norm[i:j] for t in x]
                if len(toks) > len(tt) + rule.max_span_extra_tokens:
                    break
                if not toks:
                    continue
                ph, ch = phonetic_similarity(tt, toks), character_similarity(tt, toks)
                t_ov = _overlap(ws, we, base[i]["start"], base[j - 1]["end"])
                key = (0.5 * ph + 0.5 * ch, t_ov)
                if best_key is None or key > best_key:
                    best_key, best = key, (i, j, toks, ph, ch)
        if best is None:  # only empty-normalising words (fillers, punctuation) overlap
            cand.no_overlap = True
            out.append(cand)
            continue
        i, j, toks, ph, ch = best
        ps, pe = base[i]["start"], base[j - 1]["end"]
        union = max(we, pe) - min(ws, ps)
        cand.p_first, cand.p_last = i, j
        cand.p_text = " ".join(w["word"] for w in base[i:j])
        cand.p_start, cand.p_end = ps, pe
        cand.already_correct = list(toks) == list(tt)  # exact tokens; "chad stone" is still rewritten
        cand.features = {
            "phonetic_sim": ph,
            "char_sim": ch,
            "time_overlap": _overlap(ws, we, ps, pe) / union if union > 0 else 1.0,
            "word_delta": abs(len(toks) - len(tt)),
        }
        out.append(cand)
    return out


def rule_accepts(c: Candidate, rule: VocabularyMergeRule = DEFAULT_RULE) -> bool:
    if c.no_overlap or c.already_correct:
        return False
    f = c.features
    return f["phonetic_sim"] >= rule.phonetic_min and f["char_sim"] >= rule.character_min and f["word_delta"] <= rule.max_word_delta


def apply_replacements(base: Sequence[Dict], candidates: Iterable[Candidate],
                       accept: Callable[[Candidate], bool]) -> Tuple[List[Dict], List[Dict]]:
    """The research's word-level apply, on flat word dicts: each accepted candidate's term written over
    its base span (the term's words share the span's time evenly). When accepted spans overlap, the
    earlier-listed candidate wins. Returns (words, replacement records). The product applies per turn
    with ``merge_transcript``; this stays for parity with the research tests."""
    taken: set = set()
    chosen = []
    for c in candidates:
        if c.no_overlap or c.already_correct or not accept(c):
            continue
        span = set(range(c.p_first, c.p_last))
        if span & taken:
            continue
        taken |= span
        chosen.append(c)
    chosen.sort(key=lambda c: c.p_first)
    words: List[Dict] = []
    records: List[Dict] = []
    at = 0
    for c in chosen:
        words.extend(dict(w) for w in base[at:c.p_first])
        parts = c.term.split()
        dur = (c.p_end - c.p_start) / len(parts)
        for n, part in enumerate(parts):
            words.append({"word": part, "start": c.p_start + n * dur, "end": c.p_start + (n + 1) * dur, "replaced": True})
        records.append({"term": c.term, "parakeet": c.p_text, "whisper": c.w_text, "start": c.p_start, "end": c.p_end})
        at = c.p_last
    words.extend(dict(w) for w in base[at:])
    return words, records


# ------------------------------------------------------------------------ the transcript (product)
def locate_words(text: str, words: Sequence[str]) -> List[Optional[Tuple[int, int]]]:
    """Each word's (start, end) character range in ``text``, found in order; None for a word that
    cannot be found after the previous one."""
    out: List[Optional[Tuple[int, int]]] = []
    cursor = 0
    for word in words:
        token = word.strip()
        if not token:
            out.append(None)
            continue
        at = text.find(token, cursor)
        if at < 0:
            out.append(None)
            continue
        out.append((at, at + len(token)))
        cursor = at + len(token)
    return out


_LEADING = re.compile(r"^\W*")
_TRAILING = re.compile(r"\W*$")


@dataclass
class _Accepted:
    candidate: Candidate
    turn: int          # index into the turns list
    first: int         # word index in the base turn
    last: int          # exclusive
    source: VocabularyTermSource


@dataclass
class MergeOutcome:
    """``merge_transcript``'s result: the merged turns, the replacements (transcript order) and the
    number of vocabulary hits (candidates before the rule)."""

    turns: List[TranscriptTurnContent]
    replacements: List[TranscriptReplacement]
    candidates: int
    rejected: Dict[str, int] = field(default_factory=dict)
    """Why accepted-by-rule candidates were still not applied: ``crosses_turn``, ``unlocated``, ``overlap``."""


def _group_key(channels: int, channel: Optional[int]) -> Optional[int]:
    return channel if channels >= 2 else None


def merge_transcript(turns: Sequence[TranscriptTurnContent], pass_words: Sequence[AsrPassWord], terms: Sequence[AsrVocabularyTerm],
                     rule: VocabularyMergeRule = DEFAULT_RULE, *, channels: int = 1) -> MergeOutcome:
    """The rule merge over a whole transcript. Candidates are generated per channel (stereo: the base
    turns and the pass words of one channel; mono: everything together), accepted by the rule,
    rejected when their base span crosses a turn or cannot be located in the turn text, and applied
    turn by turn. Turns without word timings take no replacement."""
    vocab = [t.term for t in terms]
    sources = [t.source for t in terms]
    groups: Dict[Optional[int], List[Tuple[int, int, WordTimestampView]]] = {}
    for ti, turn in enumerate(turns):
        for wi, word in enumerate(turn.word_timestamps or []):
            groups.setdefault(_group_key(channels, turn.channel), []).append((ti, wi, word))
    pass_groups: Dict[Optional[int], List[AsrPassWord]] = {}
    for word in pass_words:
        pass_groups.setdefault(_group_key(channels, word.channel), []).append(word)

    located: Dict[int, List[Optional[Tuple[int, int]]]] = {}

    def locations(ti: int) -> List[Optional[Tuple[int, int]]]:
        if ti not in located:
            located[ti] = locate_words(turns[ti].text, [w.word for w in (turns[ti].word_timestamps or [])])
        return located[ti]

    total = 0
    rejected: Dict[str, int] = {}
    accepted: List[_Accepted] = []
    for key in sorted(set(groups) | set(pass_groups), key=lambda k: -1 if k is None else k):
        base = sorted(groups.get(key, []), key=lambda x: (x[2].start_time, x[0], x[1]))
        whisper = sorted(pass_groups.get(key, []), key=lambda w: (w.start_time, w.end_time))
        if not whisper:
            continue
        base_dicts = [{"word": w.word, "start": w.start_time, "end": w.end_time} for _, _, w in base]
        whisper_dicts = [{"word": w.word, "start": w.start_time, "end": w.end_time} for w in whisper]
        candidates = generate_candidates(base_dicts, whisper_dicts, vocab, rule)
        total += len(candidates)
        taken: set = set()
        for c in candidates:
            if not rule_accepts(c, rule):
                continue
            span = base[c.p_first:c.p_last]
            turn_ids = {ti for ti, _, _ in span}
            indexes = [wi for _, wi, _ in span]
            if len(turn_ids) != 1 or indexes != list(range(indexes[0], indexes[0] + len(indexes))):
                rejected["crosses_turn"] = rejected.get("crosses_turn", 0) + 1
                continue
            ti = next(iter(turn_ids))
            first, last = indexes[0], indexes[-1] + 1
            spots = locations(ti)[first:last]
            text = turns[ti].text
            if any(s is None for s in spots) or any(text[a[1]:b[0]].strip() for a, b in zip(spots, spots[1:])):  # type: ignore[index]
                rejected["unlocated"] = rejected.get("unlocated", 0) + 1
                continue
            cells = {(ti, wi) for wi in range(first, last)}
            if cells & taken:
                rejected["overlap"] = rejected.get("overlap", 0) + 1
                continue
            taken |= cells
            accepted.append(_Accepted(c, ti, first, last, sources[c.term_index]))

    by_turn: Dict[int, List[_Accepted]] = {}
    for item in accepted:
        by_turn.setdefault(item.turn, []).append(item)
    merged: List[TranscriptTurnContent] = []
    replacements: List[TranscriptReplacement] = []
    for ti, turn in enumerate(turns):
        items = sorted(by_turn.get(ti, []), key=lambda x: x.first)
        if not items:
            merged.append(turn)
            continue
        base_words = list(turn.word_timestamps or [])
        spots = locations(ti)
        text = turn.text
        new_text: List[str] = []
        new_words: List[WordTimestampView] = []
        text_at = 0
        word_at = 0
        out_len = 0
        for item in items:
            c = item.candidate
            s0 = spots[item.first][0]  # type: ignore[index]
            e1 = spots[item.last - 1][1]  # type: ignore[index]
            first_token = base_words[item.first].word.strip()
            last_token = base_words[item.last - 1].word.strip()
            leading = _LEADING.match(first_token).group()  # type: ignore[union-attr]
            trailing = _TRAILING.search(last_token).group()  # type: ignore[union-attr]
            if item.first == item.last - 1 and len(leading) + len(trailing) > len(first_token):
                trailing = ""  # an all-punctuation single word: keep it once
            before = text[text_at:s0]
            new_text.append(before)
            out_len += len(before)
            written = leading + c.term + trailing
            char_start = out_len + len(leading)
            new_text.append(written)
            out_len += len(written)
            new_words.extend(base_words[word_at:item.first])
            parts = c.term.split(" ")
            span_start = base_words[item.first].start_time
            span_end = max(span_start, base_words[item.last - 1].end_time)
            probs = [w.probability for w in base_words[item.first:item.last]]
            probability = min(1.0, max(0.0, sum(probs) / len(probs))) if probs else 0.0
            step = (span_end - span_start) / len(parts)
            word_start = len(new_words)
            for n, part in enumerate(parts):
                w_start = span_start + n * step
                w_end = span_end if n == len(parts) - 1 else span_start + (n + 1) * step
                label = (leading if n == 0 else "") + part + (trailing if n == len(parts) - 1 else "")
                new_words.append(WordTimestampView(word=label, start_time=w_start, end_time=max(w_start, w_end), probability=probability))
            heard = text[s0:e1] or c.p_text
            replacements.append(TranscriptReplacement(
                turn_id=turn.turn_id, word_start=word_start, word_end=word_start + len(parts), char_start=char_start,
                char_end=char_start + len(c.term), term=c.term, source=item.source, heard=heard[:TEXT_LIMIT],
                candidate_text=(c.w_text or c.term)[:TEXT_LIMIT], start_time=span_start, end_time=span_end,
                candidate_start_time=max(0.0, c.w_start), candidate_end_time=max(max(0.0, c.w_start), c.w_end),
                phonetic_similarity=min(1.0, max(0.0, c.features["phonetic_sim"])),
                character_similarity=min(1.0, max(0.0, c.features["char_sim"])),
            ))
            text_at = e1
            word_at = item.last
        new_text.append(text[text_at:])
        new_words.extend(base_words[word_at:])
        merged.append(turn.model_copy(update={"text": "".join(new_text), "word_timestamps": new_words}))
    replacements.sort(key=lambda r: (r.turn_id, r.word_start))
    return MergeOutcome(turns=merged, replacements=replacements, candidates=total, rejected=rejected)


def base_words_of(turns: Sequence[TranscriptTurnContent], channel: Optional[int] = None) -> List[Dict]:
    """The base transcript's words as research word dicts (time order), optionally one channel's."""
    words = [{"word": w.word, "start": w.start_time, "end": w.end_time}
             for t in turns if channel is None or t.channel == channel for w in (t.word_timestamps or [])]
    words.sort(key=lambda w: w["start"])
    return words


__all__ = [
    "Candidate", "DEFAULT_RULE", "MergeOutcome", "apply_replacements", "base_words_of", "character_similarity", "flatten",
    "generate_candidates", "locate_words", "merge_transcript", "normalize_text", "pack_glossary", "phonetic_similarity",
    "rank_terms", "rule_accepts", "string_similarity", "term_relevance", "term_tokens", "vocab_hits", "word_codes",
]
