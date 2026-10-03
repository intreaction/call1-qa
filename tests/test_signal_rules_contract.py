"""Contract 1.4.0: the rules-engine recipe models, their validation and limits, digests that keep
every earlier taxonomy's identity, text paths and redaction of lexicon phrases, and the provenance
("why") on stage artifacts and hits (docs/SignalsEmbeddings.md section 9)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from call1.contracts import CONTRACT_VERSION
from call1.contracts.common import ContractParameters, canonical_digest
from call1.contracts.contents import SignalHitWhy, SignalRuleDecision, SpanSubcategoryDecision
from call1.contracts.signals import (
    SignalCategory,
    SignalLexicon,
    SignalRecipe,
    SignalSettings,
    SignalTaxonomy,
    builtin_signal_taxonomy,
    lexicon_phrase_problem,
    recipe_digest,
    redact_signal_taxonomy_text,
    rules_categories,
    signal_taxonomy_cap_violations,
    signal_taxonomy_text_paths,
    taxonomy_digest,
)

BUILTIN_V1_DIGEST = "sha256:1e4c75cca6816b99dd2a684089381b0ef5b07debde6f2085f5a2b76403bf5090"
"""What Store's migration 042 seeded before 1.4.0 (call1/store/migrations/042_signals.sql)."""


def recipe(**kw) -> SignalRecipe:
    base = {"engine": "rules", "filter": {"op": "all", "children": [
        {"op": "rule", "rule": {"rule_id": "speaker", "params": {"type": "speaker", "speaker": "CALLER"}}},
        {"op": "rule", "rule": {"rule_id": "lexicon", "params": {"type": "phrase"}}}]},
        "lexicon": {"syntax": "words", "phrases": ["calling about"]}, "lexicon_weight": 0.2, "threshold": 0.4}
    return SignalRecipe.model_validate({**base, **kw})


def with_intent_recipe(r: SignalRecipe) -> SignalTaxonomy:
    data = builtin_signal_taxonomy().model_dump()
    data["categories"][0]["recipe"] = r.model_dump()
    return SignalTaxonomy.model_validate(data)


def test_the_version_is_1_4_0_and_minor():
    assert CONTRACT_VERSION == "1.4.0"


def test_earlier_taxonomies_keep_their_digests():
    assert taxonomy_digest(builtin_signal_taxonomy()) == BUILTIN_V1_DIGEST
    legacy = builtin_signal_taxonomy().model_dump(mode="json")
    legacy.pop("rules")
    for c in legacy["categories"]:
        c.pop("recipe")
    assert canonical_digest(legacy) == BUILTIN_V1_DIGEST
    ruled = with_intent_recipe(recipe())
    assert taxonomy_digest(ruled) != BUILTIN_V1_DIGEST
    assert recipe_digest(ruled.categories[0]) != recipe_digest(with_intent_recipe(recipe(threshold=0.45)).categories[0])
    assert recipe_digest(ruled.categories[0]) == recipe_digest(with_intent_recipe(recipe(origin="pack:retail@1")).categories[0])
    assert recipe_digest(ruled.categories[1]) is None


@pytest.mark.parametrize("pattern,ok", [
    (r"\b(we|i)('ll| will)( \w+){0,3} (send|refund)\b", True),
    (r"\bcan you hear\b|hello\s*\?", True),
    (r"why.*\?", True),
    (r"(a+)+", False),  # unbounded repeat around a repeat
    (r"(?=x)y", False), (r"(?P<n>x)", False), (r"(?i)x", False), (r"(x)\1", False), (r"(x", False), ("1234", False),
])
def test_regex_phrases_are_limited_to_a_safe_subset(pattern, ok):
    assert (lexicon_phrase_problem(pattern, "regex") is None) is ok


def test_recipe_validation():
    with pytest.raises(ValidationError):  # a lexicon weight needs a lexicon
        SignalRecipe(lexicon_weight=0.3)
    with pytest.raises(ValidationError):  # a phrase rule on the recipe's lexicon needs one
        SignalRecipe.model_validate({"filter": {"op": "rule", "rule": {"rule_id": "p", "params": {"type": "phrase"}}}})
    with pytest.raises(ValidationError):  # rule IDs unique
        recipe(filter={"op": "all", "children": [{"op": "rule", "rule": {"rule_id": "a", "params": {"type": "similar_to_examples", "min_share": 0.1}}},
                                                 {"op": "rule", "rule": {"rule_id": "a", "params": {"type": "similar_to_examples", "min_share": 0.2}}}]})
    deep = {"op": "rule", "rule": {"rule_id": "s", "params": {"type": "similar_to_examples", "min_share": 0.1}}}
    for _ in range(4):
        deep = {"op": "not", "children": [deep]}
    with pytest.raises(ValidationError):
        recipe(filter=deep)
    with pytest.raises(ValidationError):  # not takes exactly one child
        recipe(filter={"op": "not", "children": []})
    with pytest.raises(ValidationError):
        recipe(threshold=0.99)
    with pytest.raises(ValidationError):
        SignalLexicon(phrases=["x"], negation_veto_words=9)
    with pytest.raises(ValidationError):  # a speaker rule agrees with the category
        with_intent_recipe(recipe(filter={"op": "rule", "rule": {"rule_id": "s", "params": {"type": "speaker", "speaker": "AGENT"}}}))
    with pytest.raises(ValidationError):  # call position is ordered
        recipe(filter={"op": "rule", "rule": {"rule_id": "q", "params": {"type": "call_position", "start_from": 0.5, "start_to": 0.25}}})
    assert SignalRecipe().engine == "rules" and SignalRecipe().check == "none" and SignalSettings().detection == "model"


def test_caps_text_paths_and_redaction_cover_lexicon_phrases():
    r = recipe(filter={"op": "any", "children": [
        {"op": "rule", "rule": {"rule_id": "lexicon", "params": {"type": "phrase"}}},
        {"op": "rule", "rule": {"rule_id": "own", "params": {"type": "phrase", "syntax": "regex", "phrases": [r"\bmanager\b", "supervisor"]}}}]})
    t = with_intent_recipe(r)
    paths = dict(signal_taxonomy_text_paths(t))
    assert paths["categories[0].recipe.lexicon.phrases[0]"] == "calling about"
    assert paths["categories[0].recipe.filter.children[1].rule.params.phrases[1]"] == "supervisor"
    tight = ContractParameters(max_signal_lexicon_phrases=1, max_signal_recipe_rules=1)
    caps = {(v.field, v.cap) for v in signal_taxonomy_cap_violations(t, tight)}
    assert ("categories[0].recipe.filter.children[1].rule.params.phrases", "max_signal_lexicon_phrases") in caps
    assert ("categories[0].recipe.filter", "max_signal_recipe_rules") in caps
    assert signal_taxonomy_cap_violations(t) == []
    redacted = redact_signal_taxonomy_text(t)
    rr = redacted.categories[0].recipe
    assert rr.lexicon.phrases[0].startswith("[REDACTED ") and all(p.startswith("[REDACTED ") for p in rr.filter.children[1].rule.params.phrases)
    assert redact_signal_taxonomy_text(redacted) == redacted
    assert rr.threshold == r.threshold and rr.lexicon_weight == r.lexicon_weight


def test_rules_categories_follow_recipe_regardless_of_legacy_detection():
    t = with_intent_recipe(recipe())
    assert [c.category_id for c in rules_categories(t, SignalSettings())] == ["intent"]
    assert [c.category_id for c in rules_categories(t, SignalSettings(detection="rules"))] == ["intent"]
    gemma = with_intent_recipe(recipe(engine="gemma"))
    assert rules_categories(gemma, SignalSettings(detection="rules")) == []


def test_builtins_may_carry_a_recipe_but_not_change_their_constants():
    c = builtin_signal_taxonomy().categories[0].model_copy(update={"recipe": recipe()})
    assert SignalCategory.model_validate(c.model_dump()).recipe is not None


def test_provenance_models_validate():
    d = SignalRuleDecision(span_key="intent.t1b0", category_id="intent", segment_index=3, recipe_digest="0123456789ab", score=0.5,
                           threshold=0.4, knn_share=0.3, lexicon_weight=0.2, lexicon_match=True, subcategory_id="x")
    with pytest.raises(ValidationError):
        d.model_copy(update={"span_key": "issue.t1b0"}).model_validate(d.model_copy(update={"span_key": "issue.t1b0"}).model_dump())
    with pytest.raises(ValidationError):
        SignalRuleDecision.model_validate({**d.model_dump(), "subcategory_id": "other"})
    assert SignalHitWhy(category_source="rules", subcategory_source="rules", rule=d).rule == d
    with pytest.raises(ValidationError):
        SignalHitWhy(category_source="gemma", rule=d)
    with pytest.raises(ValidationError):
        SignalHitWhy(category_source="gemma", check="confirmed")
    with pytest.raises(ValidationError):
        SpanSubcategoryDecision(span_key="intent.t1b0", stage2_digest="sha256:" + "0" * 64, probabilities={"other": 0.9, "not": 0.05},
                                decision="other", confidence=0.95, factors=[], status="decided", source="rules", checked=True)
