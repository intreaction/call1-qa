"""The Contact Signals rules engine inside the Process stages (docs/SignalsEmbeddings.md, contract
1.4.0), on the fake handlers and the fake embedder: rules detection decides its categories with no
classifier call, mixed taxonomies still send the other categories to the (fake) classifier, a
recipe's check sends its spans to stage 2 as confirm-or-reject, every hit records its "why", the
default (``detection: model``) is today's pipeline unchanged, and a missing embedder or bank fails
closed."""

from __future__ import annotations

import json

import pytest

from call1.contracts.contents import SIGNAL_RULES_ENTRY_ID, SignalCategoriesContent, SpeakerRole
from call1.contracts.errors import JobErrorCode
from call1.contracts.signals import SignalExampleBankRef, SignalRecipe, SignalRulesConfig, SignalSettings, SignalTaxonomy
from call1.pipeline import signal_rules
from call1.process.handlers import build_registry
from call1.process.handlers.base import HandlerError
from call1.process.handlers.fake import FakeBehavior, FakeSignalClassifier

from .test_signals_support import cancel_taxonomy, run_v2

RULES = SignalSettings(pipeline="v2", detection="rules")


def recipe(phrases, *, weight=0.5, threshold=0.4, check="none", engine="rules", phrase_filter=True, veto=0) -> SignalRecipe:
    children = [{"op": "rule", "rule": {"rule_id": "lexicon", "params": {"type": "phrase"}}}] if phrase_filter else []
    return SignalRecipe.model_validate({
        "engine": engine, "filter": {"op": "all", "children": children} if children else None,
        "lexicon": {"syntax": "words", "phrases": list(phrases), "negation_veto_words": veto},
        "lexicon_weight": weight, "threshold": threshold, "check": check})


def taxonomy_with(recipes, base=None, rules=None) -> SignalTaxonomy:
    base = base or cancel_taxonomy(fields=False, narrow=False)
    data = base.model_dump()
    for c in data["categories"]:
        if c["category_id"] in recipes:
            c["recipe"] = recipes[c["category_id"]].model_dump()
    if rules is not None:
        data["rules"] = rules.model_dump()
    return SignalTaxonomy.model_validate(data)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CALL1_SIGNAL_BANK_CACHE", str(tmp_path / "bank-cache"))
    monkeypatch.setenv("CALL1_SIGNAL_BANK_DIR", str(tmp_path / "banks"))


class CountingBehavior(FakeBehavior):
    pass


def _counting_registry(monkeypatch):
    """The fake registry, counting the fake classifier's rows per stage."""
    seen = {1: [], 2: []}
    original = FakeSignalClassifier.choose

    def choose(self, rows):
        seen[self.stage].extend(rows)
        return original(self, rows)

    monkeypatch.setattr(FakeSignalClassifier, "choose", choose)
    return build_registry("fake", fake_behavior=FakeBehavior()), seen


# --- all rules -----------------------------------------------------------------------------------------


def _all_rules_taxonomy():
    # Every active category gets a rules recipe; only intent's lexicon can match the script.
    base = cancel_taxonomy(fields=False, narrow=False)
    recipes = {c.category_id: recipe(["zzz never said"]) for c in base.categories}
    recipes["intent"] = recipe(["question about"], weight=0.5, threshold=0.4)
    return taxonomy_with(recipes, base)


def test_all_rules_taxonomy_decides_with_no_classifier_rows(tmp_path, monkeypatch):
    registry, seen = _counting_registry(monkeypatch)
    run = run_v2(tmp_path, taxonomy=_all_rules_taxonomy(), settings=RULES, registry=registry)
    cats = run.categories
    assert cats.provenance.catalog_entry_id == SIGNAL_RULES_ENTRY_ID
    assert cats.rules is not None and set(cats.rules.rules_categories) == {c.category_id for c in _all_rules_taxonomy().categories if c.active}
    assert cats.rules.embedder_scheme == "fake-embedding-v1" and cats.rules.gemma_categories == []
    assert [s.category_id for s in cats.spans] == ["intent"] and cats.spans[0].turn_id == 1
    assert seen[1] == [] and seen[2] == []  # no classifier row in stage 1 or 2
    decision = cats.rule_decisions[0]
    assert decision.span_key == cats.spans[0].span_key and decision.lexicon_match and decision.lexicon_phrase == 0
    assert decision.score >= decision.threshold == 0.4 and [o.rule_id for o in decision.outcomes] == ["lexicon"]
    sub = run.subcategories.decisions[0]
    assert sub.source == "rules" and not sub.checked and sub.factors == [] and run.subcategories.provenance.rows == 0
    hit = run.result.signals[0]
    assert hit.category_id == "intent" and hit.why is not None and hit.why.category_source == "rules"
    assert hit.why.subcategory_source == "rules" and hit.why.check is None and hit.why.rule == decision
    counts = {c.category_id: c for c in cats.rules.counts}
    assert counts["intent"].fired == 1 and counts["intent"].segments >= 1 and counts["deferred"].fired == 0


def test_the_knn_vote_picks_the_subcategory_from_the_taxonomy_examples(tmp_path):
    # With no bank, the taxonomy's own examples are the bank: "question about a fee" is the fee_question example.
    run = run_v2(tmp_path, taxonomy=_all_rules_taxonomy(), settings=RULES)
    decision = run.categories.rule_decisions[0]
    assert decision.subcategory_id == "fee_question" and decision.knn_share > 0 and decision.neighbours
    assert decision.neighbours[0].entry_id.startswith("taxonomy:intent") and decision.neighbours[0].carries_category
    hit = run.result.signals[0]
    assert hit.subcategory_id == "fee_question" and hit.subcategory_label == "Fee question"


def test_rules_recipe_runs_with_legacy_default_detection(tmp_path, monkeypatch):
    registry, seen = _counting_registry(monkeypatch)
    run = run_v2(tmp_path, taxonomy=_all_rules_taxonomy(), settings=SignalSettings(pipeline="v2"), registry=registry)
    assert run.categories.rules is not None and run.categories.rule_decisions
    assert run.categories.provenance.catalog_entry_id == SIGNAL_RULES_ENTRY_ID
    assert not seen[1]  # all categories use rules, so no model categorization is needed



def test_engine_gemma_recipe_keeps_the_classifier_for_that_category(tmp_path, monkeypatch):
    registry, seen = _counting_registry(monkeypatch)
    base = cancel_taxonomy(fields=False, narrow=False)
    tax = taxonomy_with({"intent": recipe(["question about"]), "issue": recipe(["fee"], engine="gemma")}, base)
    run = run_v2(tmp_path, taxonomy=tax, settings=RULES, registry=registry)
    assert run.categories.rules.rules_categories == ["intent"] and "issue" in run.categories.rules.gemma_categories
    # the classifier saw the rows without the rules category's option
    assert seen[1] and all("intent" not in [o for o, _ in r.options] for r in seen[1])


# --- mixed taxonomies ------------------------------------------------------------------------------------


def test_mixed_taxonomy_rules_and_classifier_in_one_call(tmp_path, monkeypatch):
    registry, seen = _counting_registry(monkeypatch)
    tax = taxonomy_with({"intent": recipe(["question about"])})
    run = run_v2(tmp_path, taxonomy=tax, settings=RULES, registry=registry)
    by_cat = {h.category_id: h for h in run.result.signals}
    # intent by the rules; issue ("fee") and caller_confirms_resolved ("makes sense") by the fake classifier
    assert {"intent", "issue", "caller_confirms_resolved"} <= set(by_cat)
    assert by_cat["intent"].why.category_source == "rules" and by_cat["issue"].why.category_source == "gemma"
    assert by_cat["issue"].why.rule is None and by_cat["issue"].why.subcategory_source == "gemma"
    assert run.categories.rules.gemma_categories and "intent" not in run.categories.rules.gemma_categories
    # stage 1 rows for the classifier never offer the rules category; stage 2 saw only the classifier's spans
    assert seen[1] and all("intent" not in [o for o, _ in r.options] for r in seen[1])
    assert seen[2] and all(not r.key.startswith("intent.") for r in seen[2])
    subs = {d.span_key: d for d in run.subcategories.decisions}
    assert all(d.source == ("rules" if k.startswith("intent.") else "engine") for k, d in subs.items())


def test_at_most_two_fires_per_segment_across_rules_and_classifier(tmp_path):
    base = cancel_taxonomy(fields=False, narrow=False)
    tax = taxonomy_with({"intent": recipe(["question about"]), "friction": recipe(["a fee"])}, base)
    run = run_v2(tmp_path, taxonomy=tax, settings=RULES)
    per_segment = {}
    for s in run.categories.scores:
        fired = [o for o, p in s.probabilities.items() if o != "none" and p >= 0.5]
        per_segment[s.index] = fired
    assert max(len(v) for v in per_segment.values()) <= 2
    turn1 = {sp.category_id for sp in run.categories.spans if sp.turn_id == 1}
    assert {"intent", "friction"} <= turn1 or len(turn1) == 2


# --- the check -----------------------------------------------------------------------------------------------


def test_check_on_sends_the_rules_spans_to_stage_two_and_records_confirmed(tmp_path, monkeypatch):
    registry, seen = _counting_registry(monkeypatch)
    tax = taxonomy_with({"intent": recipe(["question about"], check="gemma")})
    run = run_v2(tmp_path, taxonomy=tax, settings=RULES, registry=registry)
    intent_key = next(d.span_key for d in run.categories.rule_decisions)
    assert run.categories.rule_decisions[0].check and run.categories.rules.checked_categories == ["intent"]
    assert intent_key in [r.key for r in seen[2]]
    decision = next(d for d in run.subcategories.decisions if d.span_key == intent_key)
    assert decision.checked and decision.source == "engine"
    hit = next(h for h in run.result.signals if h.category_id == "intent")
    assert hit.why.check == "confirmed" and hit.why.subcategory_source == "gemma"
    # the fake stage 2 picks by the subcategory examples: "question about a fee" -> fee_question
    assert hit.subcategory_id == "fee_question"


def test_check_rejects_a_rule_span_that_stage_two_calls_not(tmp_path):
    from call1.process.handlers.fake import SCRIPT

    script = list(SCRIPT)
    script[1] = (SpeakerRole.CALLER, "Not really a question about a fee, not really.")
    tax = taxonomy_with({"intent": recipe(["question about"], check="gemma")})
    run = run_v2(tmp_path, script=script, taxonomy=tax, settings=RULES)
    key = run.categories.rule_decisions[0].span_key
    assert next(d for d in run.subcategories.decisions if d.span_key == key).decision == "rejected"
    assert all(h.category_id != "intent" for h in run.result.signals)


def test_check_off_passes_through_and_loads_nothing_in_stage_two(tmp_path, monkeypatch):
    loads = []
    monkeypatch.setattr(FakeSignalClassifier, "load", lambda self: loads.append(self.stage))
    tax = _all_rules_taxonomy()
    run_v2(tmp_path, taxonomy=tax, settings=RULES)
    assert loads == []  # neither stage loaded the classifier


# --- the bank ---------------------------------------------------------------------------------------------------


def _write_pack(folder, entries):
    pack = {"bank_id": "test-v1", "entries": entries}
    pack["digest"] = signal_rules.bank_pack_digest(pack)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "test-v1.json").write_text(json.dumps(pack))
    return pack


def test_a_bank_pack_joins_the_vote_and_its_vectors_are_cached(tmp_path, monkeypatch):
    pack = _write_pack(tmp_path / "banks", [
        {"entry_id": "test-v1:0", "speaker": "caller", "text": "I have a question about a fee on my bill",
         "labels": [{"category_id": "intent", "subcategory_id": "fee_question"}]},
        {"entry_id": "test-v1:1", "speaker": "caller", "text": "okay thanks", "labels": []},
    ])
    rules = SignalRulesConfig(bank=SignalExampleBankRef(bank_id="test-v1", digest=pack["digest"]))
    tax = taxonomy_with({"intent": recipe(["zzz"], phrase_filter=False, weight=0.0, threshold=0.3)}, rules=rules)
    run = run_v2(tmp_path, taxonomy=tax, settings=RULES)
    assert run.categories.rules.bank == rules.bank and run.categories.rules.bank_entries == 2
    decision = run.categories.rule_decisions[0]
    assert decision.category_id == "intent" and not decision.lexicon_match and decision.knn_share >= 0.3
    assert any(n.entry_id == "test-v1:0" for n in decision.neighbours)
    cached = list((tmp_path / "bank-cache").glob("*.npy"))
    assert len(cached) == 2  # the pack's vectors and the taxonomy entries'


def test_a_missing_or_altered_bank_fails_closed(tmp_path):
    rules = SignalRulesConfig(bank=SignalExampleBankRef(bank_id="test-v1", digest="sha256:" + "0" * 64))
    tax = taxonomy_with({"intent": recipe(["question about"])}, rules=rules)
    with pytest.raises(HandlerError) as missing:
        run_v2(tmp_path / "a", taxonomy=tax, settings=RULES)
    assert missing.value.code is JobErrorCode.INPUT_UNAVAILABLE
    _write_pack(tmp_path / "banks", [{"entry_id": "test-v1:0", "speaker": "caller", "text": "hello there", "labels": []}])
    with pytest.raises(HandlerError) as altered:
        run_v2(tmp_path / "b", taxonomy=tax, settings=RULES)
    assert altered.value.code is JobErrorCode.INPUT_UNAVAILABLE and "digest" in str(altered.value)


def test_a_missing_embedder_fails_closed_not_no_signals(tmp_path, monkeypatch):
    from call1 import embedding
    from call1.process.handlers import signals_rules

    class Missing(embedding.FakeEmbedder):
        def embed_documents(self, texts, **kw):
            raise embedding.EmbedderUnavailable("not_installed", "the search embedding model is not installed")

    monkeypatch.setattr(signals_rules, "embedder_for", lambda device: Missing())
    with pytest.raises(HandlerError) as err:
        run_v2(tmp_path, taxonomy=_all_rules_taxonomy(), settings=RULES)
    assert err.value.code is JobErrorCode.MODEL_UNAVAILABLE


def test_real_categorize_refuses_the_claim_without_embedder_weights(tmp_path, monkeypatch):
    from call1 import embedding
    from call1.contracts.artifacts import ArtifactKind
    from call1.contracts.jobs import JobType
    from call1.process.handlers.base import ReleaseJob
    from call1.process.handlers.signals_rules import rules_ready

    from .test_signals_support import make_job, snapshot

    monkeypatch.setenv("CALL1_EMBEDDING_BACKEND", "nemotron")
    monkeypatch.setattr(embedding, "weights_installed", lambda path=None: False)
    snap = snapshot(_all_rules_taxonomy(), RULES)
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, {"taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap)})
    with pytest.raises(ReleaseJob) as refused:
        rules_ready(job)
    assert refused.value.disposition == "reject" and refused.value.code is JobErrorCode.MODEL_UNAVAILABLE
    # A rules recipe still needs the embedder with the legacy default setting.
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE,
                   {"taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snapshot(_all_rules_taxonomy(), SignalSettings(pipeline="v2")))})
    with pytest.raises(ReleaseJob):
        rules_ready(job)


# --- re-derive and provenance ------------------------------------------------------------------------------------


def test_a_threshold_rederive_keeps_the_rule_decisions_of_surviving_spans(tmp_path):
    from call1.contracts.artifacts import ArtifactKind
    from call1.contracts.jobs import JobType

    from .test_signals_support import findings_for, make_job, script_transcript, signal_params, snapshot

    tax = taxonomy_with({"intent": recipe(["question about"])})
    run = run_v2(tmp_path, taxonomy=tax, settings=RULES)
    snap = snapshot(tax, RULES)
    transcript = script_transcript()
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE,
                   {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript)),
                    "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap),
                    "previous_categories": (ArtifactKind.SIGNAL_CATEGORIES, run.categories)},
                   parameters=signal_params(snap, stage1_mode="rederive"))
    again: SignalCategoriesContent = build_registry("fake").get(JobType.CONTACT_SIGNALS_CATEGORIZE).run(job).outputs["categories"].content
    assert again.mode == "rederive" and again.rules == run.categories.rules
    assert {d.span_key for d in again.rule_decisions} <= {s.span_key for s in again.spans}
    assert again.rule_decisions == [d for d in run.categories.rule_decisions if d.span_key in {s.span_key for s in again.spans}]


def test_the_calibration_id_changes_with_the_recipe(tmp_path):
    a = run_v2(tmp_path / "a", taxonomy=taxonomy_with({"intent": recipe(["question about"], threshold=0.4)}), settings=RULES)
    b = run_v2(tmp_path / "b", taxonomy=taxonomy_with({"intent": recipe(["question about"], threshold=0.45)}), settings=RULES)
    assert a.categories.provenance.calibration_id != b.categories.provenance.calibration_id
    assert a.categories.provenance.calibration_id.startswith(signal_rules.ENGINE_VERSION + ":")
