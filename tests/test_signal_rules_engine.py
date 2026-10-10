"""The pure rules engine (``call1/pipeline/signal_rules.py``, docs/SignalsEmbeddings.md sections 3-4):
lexicons and the negation veto, the kNN share vote, recipe evaluation, the two-fires cap, span
decisions, and parity with the research engine (``scripts/research/signals_embed/rules_engine.py``)
on a small fixture with the seed's R2 recipes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from call1.contracts.signals import SignalKnnSettings, SignalTaxonomy, SignalTaxonomySave
from call1.pipeline import signal_rules as sr

REPO = Path(__file__).resolve().parents[1]
SEED = REPO / "call1" / "store" / "seeds" / "signals_retail_v1.json"
RESEARCH = REPO / "scripts" / "research" / "signals_embed"


def seed_taxonomy() -> SignalTaxonomy:
    return SignalTaxonomySave.model_validate(json.loads(SEED.read_text())).taxonomy


# --- lexicons ------------------------------------------------------------------------------------------


def test_words_phrases_match_on_word_boundaries_ignoring_case_and_curly_apostrophes():
    assert sr.match_lexicon(sr.normalize("That’s PERFECT, thanks"), "words", ["that's perfect"]).matched
    assert not sr.match_lexicon(sr.normalize("imperfect"), "words", ["perfect"]).matched
    hit = sr.match_lexicon(sr.normalize("hello? can you hear me"), "words", ["zzz", "can you hear", "hello?"])
    assert hit.matched and hit.phrase_index == 2  # the first match in the text ("hello?") names its phrase


def test_regex_phrases_name_the_phrase_that_matched_even_with_inner_groups():
    phrases = [r"\b(we|i)('ll| will| can)( \w+){0,3} (refund|replace)\b", r"\bprice ?match\b"]
    assert sr.match_lexicon("we can price match that", "regex", phrases).phrase_index == 1
    assert sr.match_lexicon("i will happily refund it", "regex", phrases).phrase_index == 0


def test_the_negation_veto_looks_back_n_words_with_the_research_cues():
    offer = ["refund"]
    assert sr.match_lexicon("i can't refund that", "words", offer, 3).vetoed
    assert sr.match_lexicon("i didnt say we would refund", "words", offer, 3).vetoed is False  # "didnt" is 4 words back
    assert sr.match_lexicon("i didnt say refund", "words", offer, 3).vetoed
    # the research cue list's quirk, kept for parity with the tuned recipes: "...nt" words count
    assert sr.match_lexicon("if you want a refund", "words", offer, 3).vetoed
    assert not sr.match_lexicon("i can't refund that", "words", offer, 0).vetoed  # 0 = off
    assert not sr.match_lexicon("i can't refund that", "words", offer, 3).passes


def test_disfluency_markers_are_dropped_before_matching():
    assert sr.normalize("I'd (um) like to~ order") == "i'd like to order"


# --- the kNN share -----------------------------------------------------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_the_knn_share_is_the_weighted_share_of_same_speaker_neighbours():
    tax = seed_taxonomy()
    entries = [
        sr.BankEntry("b:0", "a", "caller", (("intent", "check_stock_availability"),)),
        sr.BankEntry("b:1", "b", "caller", ()),
        sr.BankEntry("b:2", "c", "agent", (("fix_proposed", None),)),
        sr.BankEntry("b:3", "d", "any", (("intent", "loyalty_points_question"),), weight=2.0),
    ]
    V = np.stack([_unit([1, 0, 0]), _unit([0.9, 0.1, 0]), _unit([1, 0, 0]), _unit([0, 1, 0])])
    index = sr.KnnIndex(entries, V, tax)
    knn = SignalKnnSettings(k=3, temperature=0.1, taxonomy_example_weight=2.0)
    res = index.scores(np.stack([_unit([1, 0, 0])]), ["caller"], knn)
    sims = np.array([1.0, float(V[1] @ V[0]), 0.0])
    w = np.exp((sims - sims.max()) / 0.1) * np.array([1.0, 1.0, 2.0])
    share = (w[0] + w[2]) / w.sum()
    assert res.share[0, index.cat_ix["intent"]] == pytest.approx(share, rel=1e-5)
    assert res.share[0, index.cat_ix["fix_proposed"]] == 0  # the agent entry is never a caller's neighbour
    sub, sub_share = sr.subcategory_vote(index, res, 0, "intent")
    assert sub == "check_stock_availability" and sub_share == pytest.approx(w[0] / (w[0] + w[2]), rel=1e-5)
    near = sr.neighbours_for(index, res, 0, "intent")
    assert [n.entry_id for n in near][:1] == ["b:0"] and near[0].carries_category and near[0].subcategory_id == "check_stock_availability"
    # an entry with an unknown subcategory counts as Other
    odd = sr.KnnIndex([sr.BankEntry("x", "t", "caller", (("intent", "retired_sub"),))], V[:1], tax)
    assert sr.subcategory_vote(odd, odd.scores(V[:1], ["caller"], knn), 0, "intent") == (None, 1.0)


def test_knn_matches_the_research_vote_formula_on_random_vectors():
    rng = np.random.default_rng(7)
    tax = seed_taxonomy()
    cats = [c.category_id for c in tax.categories]
    n_bank, d = 60, 16
    V = rng.normal(size=(n_bank, d)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    speakers = rng.choice(["agent", "caller", "any"], size=n_bank, p=[0.45, 0.45, 0.1])
    entries = [sr.BankEntry(f"b:{i}", "t", str(speakers[i]), ((cats[i % len(cats)], None),) if i % 3 else (), 2.0 if speakers[i] == "any" else 1.0)
               for i in range(n_bank)]
    index = sr.KnnIndex(entries, V, tax)
    Q = rng.normal(size=(5, d)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    qspk = ["agent", "caller", "caller", "agent", "caller"]
    knn = SignalKnnSettings(k=10, temperature=0.1)
    got = index.scores(Q, qspk, knn)
    # run_rules_hybrid.Bank.scores, verbatim in substance
    S = Q @ V.T
    spk = np.array(qspk)
    S = np.where((spk[:, None] == speakers[None, :]) | (speakers[None, :] == "any"), S, -np.inf)
    top = np.argpartition(-S, 9, axis=1)[:, :10]
    ts = np.take_along_axis(S, top, axis=1)
    w = np.exp((ts - ts.max(axis=1, keepdims=True)) / 0.1) * index.weights[top]
    w = np.where(np.isfinite(ts), w, 0.0)
    ws = w.sum(axis=1, keepdims=True).clip(min=1e-9)
    want = np.einsum("nk,nkc->nc", w, index.Yc[top]) / ws
    assert np.allclose(got.share, want, atol=1e-6)


def test_taxonomy_entries_follow_the_research_prototype_texts():
    tax = seed_taxonomy()
    entries = sr.taxonomy_entries(tax, 2.0)
    intent = next(e for e in entries if e.entry_id == "taxonomy:intent")
    assert intent.text == "Caller: Caller objective. Caller says what they want or why they called" and intent.speaker == "caller"
    stock = next(e for e in entries if e.entry_id == "taxonomy:intent/check_stock_availability")
    assert stock.text.startswith("Caller: Caller objective, Check stock / availability. Asks if an item is in stock For example: \"")
    assert any(e.entry_id.startswith("taxonomy:intent/check_stock_availability#") and e.text == "do you have this in stock" for e in entries)
    assert all(e.weight == 2.0 for e in entries) and sr.taxonomy_entries(tax, 0) == []


def test_the_pack_digest_covers_entries_and_id():
    pack = {"bank_id": "b", "entries": [{"entry_id": "b:0", "speaker": "caller", "text": "hi", "labels": []}]}
    other = {"bank_id": "b", "entries": [{"entry_id": "b:0", "speaker": "caller", "text": "hi!", "labels": []}]}
    assert sr.bank_pack_digest(pack) != sr.bank_pack_digest(other) and sr.bank_pack_digest(pack).startswith("sha256:")
    assert sr.pack_entries(pack)[0] == sr.BankEntry("b:0", "hi", "caller", (), 1.0)


# --- recipes -------------------------------------------------------------------------------------------------


FIXTURE = [  # (speaker, text), one ~7 s unit per turn, 12 s apart
    ("agent", "Thank you for calling, how can I help you today?"),
    ("caller", "Hi, I'm looking for a pair of boots, do you have them in stock?"),
    ("caller", "The last ones I bought don't fit, they're too tight, so I need to return them."),
    ("agent", "We can send you a replacement pair today, or I can offer a store credit."),
    ("agent", "I can't refund that one over the phone, unfortunately."),
    ("caller", "Can you hear me? Hello? The line keeps breaking up."),
    ("agent", "I've processed that return for you and it's gone through."),
    ("agent", "Would you like to add a protection plan while I have you?"),
    ("agent", "You'll receive a confirmation email within two business days."),
    ("caller", "That's perfect, thank you so much, that's all I needed."),
    ("agent", "Just kidding honey, I'm the number one rep here."),
    ("caller", "I'll take this further, I'll contact the ombudsman, this is not acceptable."),
    ("caller", "Yeah, okay."),
]


def _fixture_units():
    units = []
    for i, (spk, text) in enumerate(FIXTURE):
        units.append(sr.RuleUnit(index=i, turn_id=i, window=0, block=0, speaker=spk, text=text, start=12.0 * i, end=12.0 * i + 7))
    return units, 12.0 * len(FIXTURE)


def test_every_fire_passes_its_filter_and_reaches_its_threshold_and_at_most_two_are_kept():
    tax = seed_taxonomy()
    units, duration = _fixture_units()
    rng = np.random.default_rng(3)
    index = sr.KnnIndex([], np.zeros((0, 4), np.float32), tax)
    share = rng.uniform(0, 0.5, size=(len(units), len(index.categories))).astype(np.float32)
    knn = sr.KnnResult(share, np.zeros((len(units), len(index.subkeys)), np.float32), np.zeros((len(units), 0), int), np.zeros((len(units), 0)))
    plans = sr.plan_categories(tax.categories)
    out = sr.evaluate_call(units, duration, plans, knn, index)
    assert out.fires and all(f.score >= f.threshold for f in out.fires)
    assert all(len(v) <= 2 for v in out.kept.values())
    for row, kept in out.kept.items():
        assert [f.margin for f in kept] == sorted((f.margin for f in kept), reverse=True)
    # speaker scopes: no agent category on a caller unit
    scope = {c.category_id: sr.speaker_key(c.speaker) for c in tax.categories}
    assert all(scope[f.category_id] == units[f.row].speaker for f in out.fires)
    # the vetoed offer never fires fix_proposed; the unvetoed one can
    assert not any(f.category_id == "fix_proposed" and f.row == 4 for f in out.fires)
    counts = {c.category_id: c for c in out.counts}
    assert counts["intent"].segments == sum(1 for s, _ in FIXTURE if s == "caller")
    assert sum(c.fired for c in out.counts) == sum(len(v) for v in out.kept.values())
    answer = sr.stage1_answer(["intent", "issue", "none"], out.kept.get(1, []))
    assert answer["none"] in (0.05, 0.95)


def _research_engine():
    if not (RESEARCH / "rules_engine.py").exists():
        pytest.skip("research engine not present")
    sys.path.insert(0, str(RESEARCH))
    try:
        import rules_engine as re_  # noqa: WPS433
    finally:
        sys.path.remove(str(RESEARCH))
    return re_


R2 = {  # recipes-tuned.json -> R2 (training-LOO tuning), with the sentiment bonus off (not a v1 rule)
    "intent": {"pos_lo": 0.0, "pos_hi": 1.0, "len_min": 0.0, "b_phrase": 0.2, "theta": 0.375},
    "issue": {"pos_lo": 0.0, "pos_hi": 0.25, "e_min": 99.0, "s_min": 99.0, "w_min": 0, "w_max": 999, "b_phrase": 0.5, "b_sent": 0.0, "theta": 0.4},
    "friction": {"len_min": 0.0, "e_min": 0.1, "s_min": 99.0, "rep_min": 0.0, "gap_min": 99.0, "tone_min": 99.0, "pos_lo": 0.0, "pos_hi": 1.0,
                 "b_phrase": 0.5, "b_sent": 0.0, "theta": 0.4},
    "fix_proposed": {"tense_on": 0, "e_min": 99.0, "neg_near": 3, "pos_lo": 0.0, "pos_hi": 1.0, "fol_w": 0, "fol_min": 0.3, "num_max": 999,
                     "b_phrase": 0.3, "theta": 0.4},
    "agent_reports_completed": {"tense_on": 0, "e_min": 99.0, "neg_near": 0, "pos_lo": 0.0, "pos_hi": 1.0, "gap_min": 0.0, "fol_w": 0,
                                "fol_min": 0.3, "b_phrase": 0.5, "theta": 0.4},
    "caller_confirms_resolved": {"e_min": 99.0, "s_min": 0.0, "pos_lo": 0.5, "pos_hi": 1.0, "neg_near": 0, "fol_w": 0, "fol_min": 0.3, "w_max": 999,
                                 "b_phrase": 0.2, "b_sent": 0.0, "theta": 0.4},
    "caller_reports_unresolved": {"e_min": 99.0, "s_min": 99.0, "pos_lo": 0.0, "pos_hi": 1.0, "b_phrase": 0.5, "b_sent": 0.0, "theta": 0.4},
    "deferred": {"tense_on": 0, "e_min": 99.0, "noq_on": 0, "pos_lo": 0.25, "pos_hi": 1.0, "b_phrase": 0.2, "theta": 0.35},
    "upsell_attempt": {"e_min": 0.1, "pos_lo": 0.0, "pos_hi": 1.0, "tense_on": 0, "b_phrase": 0.3, "theta": 0.35},
    "agent_conduct_concern": {"len_min": 0.0, "e_min": 99.0, "sn_min": 99.0, "s_min": 99.0, "tone_min": 99.0, "b_phrase": 0.5, "b_sent": 0.0,
                              "theta": 0.4},
}


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_parity_with_the_research_engine_on_the_seed_recipes(seed):
    """The seed's recipes (built from R2 by scripts/build_signal_rules_seed.py) fire exactly what the
    research engine fires with the R2 parameters, on the same units and kNN shares."""
    re_ = _research_engine()
    tax = seed_taxonomy()
    units, duration = _fixture_units()
    rng = np.random.default_rng(seed)
    index = sr.KnnIndex([], np.zeros((0, 4), np.float32), tax)
    assert index.categories == re_.CATS
    share = rng.uniform(0, 0.6, size=(len(units), len(re_.CATS))).astype(np.float32)
    knn = sr.KnnResult(share, np.zeros((len(units), len(index.subkeys)), np.float32), np.zeros((len(units), 0), int), np.zeros((len(units), 0)))
    mine = sr.evaluate_call(units, duration, sr.plan_categories(tax.categories), knn, index)
    call = {"call_id": "fixture", "duration_seconds": duration,
            "turns": [{"turn_id": u.turn_id, "speaker": u.speaker, "start_time": u.start, "end_time": u.end, "text": u.text} for u in units]}
    segs = [{"turn_id": u.turn_id, "window": 0, "speaker": u.speaker, "text": u.text, "start": u.start, "end": u.end} for u in units]
    U = re_.build_units([call], [segs], [np.zeros((len(units), 3), np.float32)], [share])
    fires = {c: re_.fire_category(U, c, R2[c]) for c in re_.CATS}
    theirs = re_.combine(U, fires)
    assert {r: [f.category_id for f in kept] for r, kept in mine.kept.items()} == theirs


def test_lexicon_parity_with_the_research_regexes():
    re_ = _research_engine()
    tax = seed_taxonomy()
    texts = [t for _, t in FIXTURE] + ["do you carry the blue ones?", "is there a way to get it sooner?", "I'd love to order two",
                                       "that'd be brilliant", "sorry, hello?", "the supplier restocks next week", "whatever you say"]
    for c in tax.categories:
        lex = c.recipe.lexicon
        for t in texts:
            norm = sr.normalize(t)
            assert sr.match_lexicon(norm, lex.syntax, lex.phrases).matched == bool(re_._LEX_RE[c.category_id].search(norm)), (c.category_id, t)


def test_span_decision_takes_the_strongest_window_and_the_knn_subcategory():
    tax = seed_taxonomy()
    units = [sr.RuleUnit(index=i, turn_id=1, window=i, block=0, speaker="caller", text="do you have them in stock?", start=float(i), end=i + 1.0)
             for i in range(3)]
    index = sr.KnnIndex([], np.zeros((0, 4), np.float32), tax)
    share = np.zeros((3, len(index.categories)), np.float32)
    share[:, index.cat_ix["intent"]] = [0.3, 0.6, 0.4]
    vote = np.zeros((3, len(index.subkeys)), np.float32)
    vote[1, index.sub_ix[("intent", "check_stock_availability")]] = 0.5
    vote[1, index.sub_ix[("intent", "other")]] = 0.1
    knn = sr.KnnResult(share, vote, np.zeros((3, 0), int), np.zeros((3, 0)))
    plans = [p for p in sr.plan_categories(tax.categories) if p.category_id == "intent"]
    out = sr.evaluate_call(units, 10.0, plans, knn, index)
    from call1.contracts.contents import SignalSpanRef

    span = SignalSpanRef(span_key="intent.t1b0", category_id="intent", turn_id=1, block=0, first_window=0, last_window=2, peak_window=0,
                         peak_probability=0.9, context_first=0, context_last=2)
    fires = {(f.unit_index, f.category_id): f for kept in out.kept.values() for f in kept}
    decision = sr.span_decision(span, fires, {(u.turn_id, u.window): u for u in units}, plans[0], index, knn)
    assert decision.segment_index == 1 and decision.subcategory_id == "check_stock_availability"
    assert decision.subcategory_share == pytest.approx(0.5 / 0.6, rel=1e-5) and decision.recipe_digest == plans[0].digest12
    assert decision.knn_share == pytest.approx(0.6, rel=1e-5) and decision.lexicon_match and decision.check
