"""The Contact Signals rules engine (docs/SignalsEmbeddings.md sections 3 and 4; contract 1.4.0).

Pure code, like ``signals_v2``: it imports neither Store nor ``call1.db`` and loads no model. The
Process stage (``call1.process.handlers.signals_rules``) embeds the masked segments and the example
bank with the search embedder and hands the vectors in; everything else happens here:

* **Lexicons** (section 3.2): a category's phrases, ``words`` (plain words on word boundaries) or
  ``regex`` (the contract's safe subset), matched case-insensitively on the normalized masked text,
  with the negation veto (a cue within ``negation_veto_words`` words before a match).
* **The kNN share** (section 3.1): the top ``k`` same-speaker neighbours of a segment in the bank,
  weighted ``exp((cos - top) / T)`` times the entry's weight; a category's share is the weighted share
  of neighbours that carry it, and the subcategory vote is the same over its subcategories.
* **Recipes** (section 4): the filter tree over typed rules, the score ``share + b * lexicon``, the
  threshold, at most two fires per segment by margin, and per span the strongest fire (the "why").

Ported from the research engine (``scripts/research/signals_embed/rules_engine.py`` and
``run_rules_hybrid.Bank``) with the v1 rule types only. Deliberate differences, recorded in the doc's
status notes: a fire whose neighbours carry no subcategory of its category takes Other rather than the
first subcategory, and the sentiment bonus is not a v1 rule (section 3.5). The negation cues are the
research engine's exactly, including its quirk that any word ending in "nt" ("want", "replacement",
"discount") counts as a cue: the tuned R2 recipes (``fix_proposed``'s 3-word veto) were measured with
it, so fixing it needs a re-tune (``NEGATION_CUE``).
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from call1.contracts.common import canonical_digest
from call1.contracts.contents import (
    SignalNeighbour,
    SignalRuleCounts,
    SignalRuleDecision,
    SignalRuleOutcome,
    SignalRuleType,
    SpeakerRole,
    short_digest,
)
from call1.contracts.signals import (
    SignalCategory,
    SignalKnnSettings,
    SignalLexicon,
    SignalPhraseParams,
    SignalPositionParams,
    SignalRecipe,
    SignalRuleExpr,
    SignalSimilarParams,
    SignalSpeakerParams,
    SignalTaxonomy,
    recipe_digest,
)

ENGINE_VERSION = "signal-rules-v1"
"""Recorded in ``SignalRulesProvenance.engine_version`` and folded into the calibration ID, so a
change to the engine's rules is never carried forward as current."""

MAX_FIRES = 2
PICK_SCORES = (0.9, 0.7)
"""The stage-1 probability of the first and second fire, as Gemma's constrained pick scores them, so
the engine thresholds (0.5) fire exactly the fires and an admin threshold behaves as it does for Gemma."""
FIRED_NONE = 0.05
UNFIRED_NONE = 0.95
MATCH_TEXT_LIMIT = 2000
"""Characters of a segment a lexicon is matched against (a ~7 s segment is far shorter)."""
OTHER = "other"
ANY_SPEAKER = "any"

# --- text -------------------------------------------------------------------------------------------

_PARENS = re.compile(r"\([^)]*\)")
_SPACES = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9'\[\]]+")
NEGATION_CUE = re.compile(r"^(not|no|never|cannot|unable|unfortunately|nothing|nope)$|n'?t$")
"""A word that negates what follows ("not", "can't", "never"...): the research engine's ``NEG``,
unchanged for parity with the tuned recipes. Known quirk: ``n'?t$`` also matches words ending in
"nt" ("want", "replacement", "discount"), so a 3-word veto also drops such matches; a corrected cue
list is a recipe change that needs a re-tune (docs/SignalsEmbeddings.md status)."""


def normalize(text: str) -> str:
    """Lower-case; drop disfluency markers "(um)", "~" cut-offs and curly apostrophes (the research
    engine's normalization, so the tuned lexicons match the same way)."""
    t = _PARENS.sub(" ", text or "").replace("~", "").replace("´", "'").replace("’", "'")
    return _SPACES.sub(" ", t).strip().lower()


def words_of(norm: str) -> List[Tuple[int, str]]:
    return [(m.start(), m.group(0)) for m in _WORD.finditer(norm)]


def speaker_key(role: Any) -> str:
    """'agent', 'caller' or 'unknown' for a ``SpeakerRole`` (or its value)."""
    value = role.value if isinstance(role, SpeakerRole) else str(role or "")
    return {"AGENT": "agent", "CALLER": "caller"}.get(value.upper(), "unknown")


# --- lexicons -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LexiconHit:
    matched: bool
    phrase_index: Optional[int] = None
    vetoed: bool = False

    @property
    def passes(self) -> bool:
        return self.matched and not self.vetoed


def _words_pattern(phrase: str) -> str:
    body = r"\s+".join(re.escape(w) for w in normalize(phrase).split())
    return r"(?<![\w'])" + body + r"(?![\w'])"


@functools.lru_cache(maxsize=512)
def compile_lexicon(syntax: str, phrases: Tuple[str, ...]) -> "re.Pattern[str]":
    """One alternation of the phrases, each in a named group ``p<i>``, so a match says which phrase
    matched (``Match.lastgroup``) and ``finditer`` walks matches exactly as one hand-written
    alternation would."""
    parts = [(p if syntax == "regex" else _words_pattern(p)) for p in phrases]
    return re.compile("|".join(f"(?P<p{i}>{part})" for i, part in enumerate(parts)), re.IGNORECASE)


def match_lexicon(norm: str, syntax: str, phrases: Sequence[str], veto_words: int = 0) -> LexiconHit:
    """Whether the lexicon matches the normalized text; the first matching phrase; and whether a
    negation cue stood within ``veto_words`` words before any match (or is its first word)."""
    if not phrases:
        return LexiconHit(False)
    text = norm[:MATCH_TEXT_LIMIT]
    matches = list(compile_lexicon(syntax, tuple(phrases)).finditer(text))
    if not matches:
        return LexiconHit(False)
    first = matches[0].lastgroup
    index = int(first[1:]) if first and first.startswith("p") else 0
    vetoed = False
    if veto_words > 0:
        words = words_of(text)
        negations = [j for j, (_, w) in enumerate(words) if NEGATION_CUE.search(w)]
        if negations:
            starts = [c for c, _ in words]
            for m in matches:
                wi = sum(1 for c in starts if c < m.start())
                if any(wi - veto_words <= j < wi + 1 for j in negations):
                    vetoed = True
                    break
    return LexiconHit(True, index, vetoed)


# --- the example bank ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class BankEntry:
    """One example: masked text, the speaker it was said by ('agent', 'caller', or 'any' for text
    written from the taxonomy), its labels as (category, subcategory or None) pairs (none: a
    background segment), and its weight in the vote."""

    entry_id: str
    text: str
    speaker: str
    labels: Tuple[Tuple[str, Optional[str]], ...] = ()
    weight: float = 1.0


def bank_pack_digest(pack: Mapping[str, Any]) -> str:
    """The digest a taxonomy pins (``SignalExampleBankRef.digest``): ``canonical_digest`` of the pack's
    ID and entries (text, speaker, labels), so a changed example changes the digest."""
    return canonical_digest({"bank_id": pack["bank_id"], "entries": list(pack["entries"])})


def pack_entries(pack: Mapping[str, Any]) -> List[BankEntry]:
    out = []
    for e in pack["entries"]:
        labels = tuple((str(lab["category_id"]), lab.get("subcategory_id")) for lab in e.get("labels") or [])
        out.append(BankEntry(entry_id=str(e["entry_id"]), text=str(e["text"]), speaker=str(e["speaker"]), labels=labels,
                             weight=float(e.get("weight", 1.0))))
    return out


_WHO = {"caller": "Caller", "agent": "Agent"}


def taxonomy_entries(taxonomy: SignalTaxonomy, weight: float, mask=lambda t: t) -> List[BankEntry]:
    """Entries written from the taxonomy's own (masked) text, as the research bank's seed prototypes
    and examples (``embed_cache.prototype_texts``, ``passage:`` variant): the subcategories' examples
    one by one, then one text per category and per subcategory (speaker word, name, gloss, examples).
    Every active category and subcategory; speaker ``any`` for an either-speaker category."""
    if weight <= 0:
        return []
    examples: List[BankEntry] = []
    protos: List[BankEntry] = []
    for c in taxonomy.categories:
        if not c.active:
            continue
        spk = speaker_key(c.speaker) if c.speaker is not None else ANY_SPEAKER
        who = _WHO.get(spk, "Speaker")
        head = f"{who}: {mask(c.name)}. {mask(c.gloss or '')}".strip()
        cex = "; ".join(mask(e) for e in c.examples)
        protos.append(BankEntry(f"taxonomy:{c.category_id}", head + (f" For example: {cex}" if cex else ""), spk, ((c.category_id, None),), weight))
        for s in c.subcategories:
            if not s.active:
                continue
            ex = "; ".join(mask(e) for e in s.examples)
            text = f"{who}: {mask(c.name)}, {mask(s.name)}. {mask(s.gloss or '')}" + (f" For example: \"{ex}\"" if ex else "")
            protos.append(BankEntry(f"taxonomy:{c.category_id}/{s.subcategory_id}", text, spk, ((c.category_id, s.subcategory_id),), weight))
            for k, e in enumerate(s.examples):
                examples.append(BankEntry(f"taxonomy:{c.category_id}/{s.subcategory_id}#{k}", mask(e), spk,
                                          ((c.category_id, s.subcategory_id),), weight))
    return examples + protos


@dataclass
class KnnResult:
    share: np.ndarray  # [n, C] per category (``KnnIndex.categories`` order)
    vote: np.ndarray  # [n, S] per (category, subcategory or 'other') (``KnnIndex.subkeys`` order)
    top: np.ndarray  # [n, <=3] bank row indexes of the nearest same-speaker entries
    top_cos: np.ndarray  # [n, <=3]


class KnnIndex:
    """The bank as matrices: vectors (unit-norm rows), speakers, weights, category labels (0/1) and
    subcategory labels (a category's subcategories share 1). Labels of categories outside
    ``categories`` are ignored; an unknown subcategory counts as Other."""

    def __init__(self, entries: Sequence[BankEntry], vectors: np.ndarray, taxonomy: SignalTaxonomy) -> None:
        if len(entries) != len(vectors):
            raise ValueError("one vector per bank entry")
        self.entries = list(entries)
        self.V = np.asarray(vectors, dtype=np.float32)
        self.categories = [c.category_id for c in taxonomy.categories if c.active]
        self.cat_ix = {c: i for i, c in enumerate(self.categories)}
        self.subkeys: List[Tuple[str, str]] = []
        for c in taxonomy.categories:
            if not c.active:
                continue
            for s in c.subcategories:
                if s.active:
                    self.subkeys.append((c.category_id, s.subcategory_id))
            self.subkeys.append((c.category_id, OTHER))
        self.sub_ix = {k: i for i, k in enumerate(self.subkeys)}
        n = len(self.entries)
        self.Yc = np.zeros((n, len(self.categories)), dtype=np.float32)
        self.Ys = np.zeros((n, len(self.subkeys)), dtype=np.float32)
        for i, e in enumerate(self.entries):
            per_cat: Dict[str, List[str]] = {}
            for cat, sub in e.labels:
                if cat in self.cat_ix:
                    per_cat.setdefault(cat, []).append(sub if sub and (cat, sub) in self.sub_ix else OTHER)
            for cat, subs in per_cat.items():
                self.Yc[i, self.cat_ix[cat]] = 1.0
                for sub in subs:
                    self.Ys[i, self.sub_ix[(cat, sub)]] += 1.0 / len(subs)
        self.speakers = np.array([e.speaker for e in self.entries], dtype=object)
        self.weights = np.array([e.weight for e in self.entries], dtype=np.float32)

    def scores(self, Q: np.ndarray, speakers: Sequence[str], knn: SignalKnnSettings, neighbours: int = 3) -> KnnResult:
        """The kNN share per category and the subcategory vote for each query row (unit-norm vectors),
        over same-speaker entries (and entries of speaker 'any')."""
        n = len(Q)
        C, S = len(self.categories), len(self.subkeys)
        if n == 0 or len(self.entries) == 0:
            return KnnResult(np.zeros((n, C), np.float32), np.zeros((n, S), np.float32), np.zeros((n, 0), int), np.zeros((n, 0), np.float32))
        sims = np.asarray(Q, dtype=np.float32) @ self.V.T
        spk = np.asarray(list(speakers), dtype=object)
        allowed = (spk[:, None] == self.speakers[None, :]) | (self.speakers[None, :] == ANY_SPEAKER)
        sims = np.where(allowed, sims, -np.inf)
        k = min(knn.k, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        tsim = np.take_along_axis(sims, top, axis=1)
        best = np.max(tsim, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", over="ignore"):
            w = np.exp((tsim - np.where(np.isfinite(best), best, 0.0)) / knn.temperature) * self.weights[top]
        w = np.where(np.isfinite(tsim), w, 0.0)
        wsum = w.sum(axis=1, keepdims=True).clip(min=1e-9)
        share = np.einsum("nk,nkc->nc", w, self.Yc[top]) / wsum
        vote = np.einsum("nk,nks->ns", w, self.Ys[top]) / wsum
        m = min(neighbours, k)
        order = np.argsort(-tsim, axis=1, kind="stable")[:, :m]
        near = np.take_along_axis(top, order, axis=1)
        near_cos = np.take_along_axis(tsim, order, axis=1)
        return KnnResult(share.astype(np.float32), vote.astype(np.float32), near, near_cos.astype(np.float32))


# --- recipes --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleUnit:
    """One masked segment the engine scores (a ``signal_segments.Segment``'s fields)."""

    index: int
    turn_id: int
    window: int
    block: int
    speaker: str
    text: str
    start: float
    end: float


@dataclass
class Fire:
    """One category's recipe on one unit, when it passed."""

    category_id: str
    unit_index: int
    row: int
    score: float
    threshold: float
    share: float
    lexicon: LexiconHit
    lexicon_weight: float
    outcomes: List[SignalRuleOutcome]

    @property
    def margin(self) -> float:
        return self.score - self.threshold


@dataclass
class CategoryPlan:
    category: SignalCategory
    recipe: SignalRecipe
    digest12: str
    speaker: Optional[str]  # 'agent'/'caller', or None for either

    @property
    def category_id(self) -> str:
        return self.category.category_id


def plan_categories(categories: Iterable[SignalCategory]) -> List[CategoryPlan]:
    out = []
    for c in categories:
        if c.recipe is None:
            continue
        out.append(CategoryPlan(c, c.recipe, short_digest(recipe_digest(c) or ""), speaker_key(c.speaker) if c.speaker is not None else None))
    return out


def _in_scope(plan: CategoryPlan, speaker: str) -> bool:
    if plan.speaker is None:
        return speaker in ("agent", "caller", "unknown")
    return speaker == plan.speaker


def _eval_rule(rule, unit: RuleUnit, norm: str, share: float, position: float, recipe_lexicon: Optional[SignalLexicon],
               lexicon_hit: LexiconHit) -> SignalRuleOutcome:
    p = rule.params
    if isinstance(p, SignalSimilarParams):
        return SignalRuleOutcome(rule_id=rule.rule_id, type=SignalRuleType.SIMILAR_TO_EXAMPLES, result="pass" if share >= p.min_share else "fail",
                                 value=round(float(share), 6))
    if isinstance(p, SignalPhraseParams):
        hit = lexicon_hit if p.phrases is None else match_lexicon(norm, p.syntax, p.phrases, p.negation_veto_words)
        return SignalRuleOutcome(rule_id=rule.rule_id, type=SignalRuleType.PHRASE, result="pass" if hit.passes else "fail",
                                 phrase_index=hit.phrase_index, vetoed=hit.vetoed)
    if isinstance(p, SignalSpeakerParams):
        return SignalRuleOutcome(rule_id=rule.rule_id, type=SignalRuleType.SPEAKER,
                                 result="pass" if unit.speaker == speaker_key(p.speaker) else "fail")
    if isinstance(p, SignalPositionParams):
        return SignalRuleOutcome(rule_id=rule.rule_id, type=SignalRuleType.CALL_POSITION,
                                 result="pass" if p.start_from <= position <= p.start_to else "fail", value=round(position, 6))
    raise ValueError(f"unknown rule type {type(p).__name__}")  # pragma: no cover - the contract's enum is closed


def _eval_expr(expr: SignalRuleExpr, outcomes: Mapping[str, SignalRuleOutcome]) -> bool:
    if expr.op == "rule":
        return outcomes[expr.rule.rule_id].result == "pass"  # type: ignore[union-attr]
    if expr.op == "not":
        return not _eval_expr(expr.children[0], outcomes)
    parts = (_eval_expr(child, outcomes) for child in expr.children)
    return all(parts) if expr.op == "all" else any(parts)


@dataclass
class CallRules:
    """The engine's answer for one call: per unit row, the fires kept (at most two, largest margin
    first); every fire that passed its recipe; the kNN result; and per-category counts."""

    kept: Dict[int, List[Fire]]
    fires: List[Fire]
    knn: KnnResult
    counts: List[SignalRuleCounts]


def evaluate_call(units: Sequence[RuleUnit], duration_seconds: float, plans: Sequence[CategoryPlan], knn: KnnResult,
                  index: KnnIndex, *, max_fires: int = MAX_FIRES) -> CallRules:
    """Every recipe on every in-scope unit (row ``i`` of ``knn`` is ``units[i]``). Rules are cheap, so
    all of a recipe's rules are evaluated and recorded; the filter's result does not depend on the
    order it is written in. A unit fires a category when the filter passes and the score reaches the
    threshold; at most ``max_fires`` fire per unit, largest margin first (ties by taxonomy order)."""
    order = {p.category_id: k for k, p in enumerate(plans)}
    norms = [normalize(u.text) for u in units]
    fires: List[Fire] = []
    counts: List[SignalRuleCounts] = []
    for plan in plans:
        recipe = plan.recipe
        col = index.cat_ix.get(plan.category_id)
        in_scope = passed = 0
        for i, unit in enumerate(units):
            if not _in_scope(plan, unit.speaker):
                continue
            in_scope += 1
            share = float(knn.share[i, col]) if col is not None else 0.0
            position = min(1.0, max(0.0, unit.start / duration_seconds)) if duration_seconds > 0 else 0.0
            lexicon = recipe.lexicon
            hit = match_lexicon(norms[i], lexicon.syntax, lexicon.phrases, lexicon.negation_veto_words) if lexicon is not None else LexiconHit(False)
            rules = recipe.filter.rules() if recipe.filter is not None else []
            outcomes = {r.rule_id: _eval_rule(r, unit, norms[i], share, position, lexicon, hit) for r in rules}
            ok = _eval_expr(recipe.filter, outcomes) if recipe.filter is not None else True
            if not ok:
                continue
            passed += 1
            score = share + recipe.lexicon_weight * (1.0 if hit.passes else 0.0)
            if score >= recipe.threshold:
                fires.append(Fire(plan.category_id, unit.index, i, score, recipe.threshold, share, hit, recipe.lexicon_weight,
                                  [outcomes[r.rule_id] for r in rules]))
        counts.append(SignalRuleCounts(category_id=plan.category_id, segments=in_scope, filter_passed=passed, fired=0))
    by_row: Dict[int, List[Fire]] = {}
    for f in fires:
        by_row.setdefault(f.row, []).append(f)
    kept: Dict[int, List[Fire]] = {}
    fired_count: Dict[str, int] = {}
    for row, found in by_row.items():
        found.sort(key=lambda f: (-f.margin, order.get(f.category_id, 0)))
        kept[row] = found[:max_fires]
        for f in kept[row]:
            fired_count[f.category_id] = fired_count.get(f.category_id, 0) + 1
    counts = [c.model_copy(update={"fired": fired_count.get(c.category_id, 0)}) for c in counts]
    return CallRules(kept=kept, fires=fires, knn=knn, counts=counts)


def stage1_answer(options: Sequence[str], kept: Sequence[Fire], none_option: str = "none") -> Dict[str, float]:
    """A unit's stage-1 row as the categorize stage stores it: the kept fires at the pick scores (0.9,
    then 0.7), every other rules option 0, and 'none' 0.05 when something fired, else 0.95."""
    answer = {o: 0.0 for o in options}
    picked = [f.category_id for f in kept if f.category_id in answer][:MAX_FIRES]
    for cid, score in zip(picked, PICK_SCORES):
        answer[cid] = score
    answer[none_option] = FIRED_NONE if picked else UNFIRED_NONE
    return answer


def subcategory_vote(index: KnnIndex, knn: KnnResult, row: int, category_id: str) -> Tuple[Optional[str], float]:
    """The kNN vote's subcategory of ``category_id`` for a unit (None = Other) and its share of the
    category's vote. No vote for the category at all gives Other with share 0."""
    cols = [j for j, (c, _) in enumerate(index.subkeys) if c == category_id]
    if not cols:
        return None, 0.0
    votes = knn.vote[row, cols]
    total = float(votes.sum())
    if total <= 0:
        return None, 0.0
    j = int(np.argmax(votes))
    sub = index.subkeys[cols[j]][1]
    return (None if sub == OTHER else sub), round(min(1.0, float(votes[j]) / total), 6)


def neighbours_for(index: KnnIndex, knn: KnnResult, row: int, category_id: str) -> List[SignalNeighbour]:
    out = []
    col = index.cat_ix.get(category_id)
    for j, cos in zip(knn.top[row], knn.top_cos[row]):
        if not math.isfinite(float(cos)):
            continue
        entry = index.entries[int(j)]
        carries = col is not None and bool(index.Yc[int(j), col])
        sub = next((s or OTHER for c, s in entry.labels if c == category_id), None) if carries else None
        if sub is not None and sub != OTHER and (category_id, sub) not in index.sub_ix:
            sub = OTHER
        out.append(SignalNeighbour(entry_id=entry.entry_id[:200], cosine=round(max(-1.0, min(1.0, float(cos))), 6), carries_category=carries,
                                   subcategory_id=sub))
    return out


def span_decision(span, fires_by_unit: Mapping[Tuple[int, str], Fire], units_by_turn_window: Mapping[Tuple[int, int], RuleUnit],
                  plan: CategoryPlan, index: KnnIndex, knn: KnnResult) -> Optional[SignalRuleDecision]:
    """The "why" of one span of a rules category: its strongest fire (largest margin) among the span's
    windows, with that unit's outcomes, nearest entries and kNN subcategory."""
    best: Optional[Fire] = None
    for window in range(span.first_window, span.last_window + 1):
        unit = units_by_turn_window.get((span.turn_id, window))
        if unit is None:
            continue
        fire = fires_by_unit.get((unit.index, span.category_id))
        if fire is not None and (best is None or fire.margin > best.margin):
            best = fire
    if best is None:
        return None
    sub, sub_share = subcategory_vote(index, knn, best.row, span.category_id)
    return SignalRuleDecision(
        span_key=span.span_key, category_id=span.category_id, segment_index=best.unit_index, recipe_digest=plan.digest12,
        score=round(best.score, 6), threshold=best.threshold, knn_share=round(min(1.0, max(0.0, best.share)), 6),
        lexicon_weight=best.lexicon_weight, lexicon_match=best.lexicon.passes, lexicon_phrase=best.lexicon.phrase_index,
        outcomes=best.outcomes[:16], neighbours=neighbours_for(index, knn, best.row, span.category_id), subcategory_id=sub,
        subcategory_share=sub_share, check=plan.recipe.check == "gemma")


def texts_digest(entries: Sequence[BankEntry], scheme: str) -> str:
    """The cache key of a list of entries' vectors: the embedder scheme and every text in order."""
    return canonical_digest({"scheme": scheme, "texts": [e.text for e in entries]})


__all__ = ["ANY_SPEAKER", "BankEntry", "CallRules", "CategoryPlan", "ENGINE_VERSION", "Fire", "KnnIndex", "KnnResult", "LexiconHit",
           "MAX_FIRES", "NEGATION_CUE", "PICK_SCORES", "RuleUnit", "bank_pack_digest", "compile_lexicon", "evaluate_call", "match_lexicon",
           "neighbours_for", "normalize", "pack_entries", "plan_categories", "span_decision", "speaker_key", "stage1_answer",
           "subcategory_vote", "taxonomy_entries", "texts_digest"]
