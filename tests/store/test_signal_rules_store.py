"""Rules-engine recipes in Store (contract 1.4.0, docs/SignalsEmbeddings.md sections 9.2-9.3): recipes
are part of a taxonomy version (saved, validated, capped, text-checked, redacted, snapshotted), the
detection switch is an audited settings save, and ``apply-signals-recipes`` installs a seed's recipes
into an edited taxonomy."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from call1.contracts.common import ContractParameters
from call1.contracts.signals import SignalRecipe, SignalTaxonomy, SignalTaxonomySave, taxonomy_digest
from call1.store import __main__ as store_cli
from call1.store.app import create_app
from call1.store.config import StoreConfig
from call1.store.results import signal_store

from .conftest import STORE_BASE_URL
from .test_signals_harness import TAXONOMY, V, put_taxonomy, save_body

SEED = Path(__file__).resolve().parents[2] / "call1" / "store" / "seeds" / "signals_retail_v1.json"

INTENT_RECIPE = SignalRecipe.model_validate({
    "engine": "rules", "filter": {"op": "all", "children": [
        {"op": "rule", "rule": {"rule_id": "speaker", "params": {"type": "speaker", "speaker": "CALLER"}}},
        {"op": "any", "children": [{"op": "rule", "rule": {"rule_id": "lexicon", "params": {"type": "phrase"}}},
                                   {"op": "rule", "rule": {"rule_id": "similar", "params": {"type": "similar_to_examples", "min_share": 0.1}}}]}]},
    "lexicon": {"syntax": "regex", "phrases": [r"\bi('d| would) like to\b", "calling about"], "negation_veto_words": 0},
    "lexicon_weight": 0.2, "threshold": 0.375, "check": "gemma"})


def with_recipe(taxonomy: SignalTaxonomy = TAXONOMY, recipe: SignalRecipe = INTENT_RECIPE, category_id: str = "intent") -> SignalTaxonomy:
    data = taxonomy.model_dump(mode="json")
    for c in data["categories"]:
        if c["category_id"] == category_id:
            c["recipe"] = recipe.model_dump(mode="json")
    return SignalTaxonomy.model_validate(data)


def test_a_recipe_is_saved_as_part_of_the_next_version(client, admin_session, reviewer_session):
    put_taxonomy(client, admin_session, TAXONOMY)
    saved = put_taxonomy(client, admin_session, with_recipe())
    assert saved["current"]["version"] == 3
    got = client.get(f"{V}/signals/taxonomy/versions/3", headers=reviewer_session.read_headers).json()
    intent = next(c for c in got["taxonomy"]["categories"] if c["category_id"] == "intent")
    assert SignalRecipe.model_validate(intent["recipe"]) == INTENT_RECIPE
    assert got["digest"] == taxonomy_digest(with_recipe()) != taxonomy_digest(TAXONOMY)
    # the version without the recipe keeps its digest, and its categories carry no recipe
    v2 = client.get(f"{V}/signals/taxonomy/versions/2", headers=reviewer_session.read_headers).json()
    assert v2["digest"] == taxonomy_digest(TAXONOMY) and all(c.get("recipe") is None for c in v2["taxonomy"]["categories"])
    replay = put_taxonomy(client, admin_session, with_recipe())
    assert replay["current"]["version"] == 3  # a no-op replay


def test_lexicon_phrases_go_through_the_definition_text_detectors(client, admin_session):
    risky = INTENT_RECIPE.model_copy(update={"lexicon": INTENT_RECIPE.lexicon.model_copy(update={"phrases": ["my card is 4111 1111 1111 1111"]})})
    refused = client.put(f"{V}/signals/taxonomy", json=save_body(with_recipe(recipe=risky), 1), headers=admin_session.headers)
    assert refused.status_code == 422, refused.text
    details = refused.json()["details"]
    assert details["field"] == "categories[0].recipe.lexicon.phrases[0]" and details["reason"] == "sensitive_text"
    assert "4111" not in refused.text


def test_unsafe_regex_and_malformed_recipes_are_refused(client, admin_session):
    body = save_body(with_recipe(), 1)
    intent = next(c for c in body["taxonomy"]["categories"] if c["category_id"] == "intent")
    for bad in ([r"(a+)+b"], [r"(?=look)ahead"], [r"(\w)\1 repeat"], ["123"]):
        intent["recipe"]["lexicon"]["phrases"] = bad
        assert client.put(f"{V}/signals/taxonomy", json=body, headers=admin_session.headers).status_code == 422, bad
    intent["recipe"]["lexicon"]["phrases"] = ["calling about"]
    intent["recipe"]["filter"]["children"][0]["rule"]["params"]["speaker"] = "AGENT"  # disagrees with intent's speaker
    assert client.put(f"{V}/signals/taxonomy", json=body, headers=admin_session.headers).status_code == 422


def test_recipe_caps_come_from_the_effective_contract_parameters(tmp_path, clock):
    tight = StoreConfig.for_tests(tmp_path / "tight").with_overrides(parameters=ContractParameters(max_signal_lexicon_phrases=1,
                                                                                                    max_signal_recipe_rules=8))
    app = create_app(tight, clock=clock)
    with TestClient(app, base_url=STORE_BASE_URL) as client, app.state.store.connection() as conn:
        admin = app.state.store.auth.mint_session_for_tests(conn, role="admin")
        refused = client.put(f"{V}/signals/taxonomy", json=save_body(with_recipe(), 1), headers=admin.headers)
        assert refused.status_code == 422
        assert refused.json()["details"]["cap"] == "max_signal_lexicon_phrases"
        assert refused.json()["details"]["field"] == "categories[0].recipe.lexicon.phrases"


def test_redaction_tombstones_lexicon_phrases_and_keeps_the_digest(client, admin_session, reviewer_session):
    put_taxonomy(client, admin_session, with_recipe())
    put_taxonomy(client, admin_session, TAXONOMY)
    v2 = client.get(f"{V}/signals/taxonomy/versions/2", headers=reviewer_session.read_headers).json()
    out = client.post(f"{V}/signals/taxonomy/versions/2/redaction", json={"digest": v2["digest"], "reason": "a phrase quoted a caller"},
                      headers=admin_session.headers)
    assert out.status_code == 200, out.text
    intent = next(c for c in out.json()["taxonomy"]["categories"] if c["category_id"] == "intent")
    assert all(p.startswith("[REDACTED ") for p in intent["recipe"]["lexicon"]["phrases"]) and "calling about" not in out.text
    assert out.json()["digest"] == v2["digest"]


def test_detection_is_an_audited_settings_save_and_reaches_the_snapshot(client, admin_session, store):
    record = client.get(f"{V}/signals/taxonomy", headers=admin_session.read_headers).json()
    assert record["settings"]["detection"] == "model"
    saved = client.put(f"{V}/signals/settings", json={"settings": {**record["settings"], "detection": "rules"}, "expected_record_version": 1},
                       headers=admin_session.headers)
    assert saved.status_code == 200 and saved.json()["settings"]["detection"] == "rules"
    [entry] = client.get(f"{V}/admin/audit", params={"action": "signal_settings_changed"}, headers=admin_session.read_headers).json()["items"]
    assert entry["details"]["old_detection"] == "model" and entry["details"]["new_detection"] == "rules"
    from call1.contracts.signals import signal_taxonomy_snapshot_current

    with store.connection() as conn:
        current = signal_store.record(conn)
    assert current.settings.detection == "rules"
    # a snapshot taken under model detection is not current any more, so the next mint takes the new settings
    from call1.contracts.signals import SignalSettings, SignalTaxonomySnapshotContent

    old = SignalTaxonomySnapshotContent(source="published", taxonomy_ref=current.current.ref, taxonomy=current.current.taxonomy,
                                        settings=SignalSettings())
    assert not signal_taxonomy_snapshot_current(old, current.current.ref, current.settings)


def test_apply_signals_recipes_installs_the_seed_recipes_into_an_edited_taxonomy(tmp_path, monkeypatch, capsys):
    data = tmp_path / "store"
    monkeypatch.setenv("CALL1_STORE_DATA", str(data))
    monkeypatch.setenv("CALL1_STORE_MAINTENANCE_SECONDS", "0")
    seed = SignalTaxonomySave.model_validate(json.loads(SEED.read_text(encoding="utf-8")))
    assert all(c.recipe is not None and c.recipe.engine == "rules" for c in seed.taxonomy.categories)
    assert seed.taxonomy.rules is not None and seed.taxonomy.rules.bank.bank_id == "retail-v1"
    # a demo root whose taxonomy was edited after the seed (a renamed subcategory), still on model detection
    assert store_cli.main(["apply-signals-seed", str(SEED), "--pipeline", "v2"]) == 0
    from call1.store.db import Database

    edited = seed.taxonomy.model_dump(mode="json")
    for c in edited["categories"]:
        c["recipe"] = None
    edited["rules"] = None
    edited["categories"][0]["subcategories"][0]["name"] = "Stock check"
    with Database(data / "store.db").connection() as conn:
        from call1.store import audit, db

        with db.transaction(conn):
            signal_store.save_taxonomy(conn, SignalTaxonomySave(taxonomy=SignalTaxonomy.model_validate(edited), expected_record_version=3),
                                       actor=audit.installer_actor("test"), account_id=None, parameters=ContractParameters())
    capsys.readouterr()
    assert store_cli.main(["apply-signals-recipes", str(SEED), "--detection", "rules"]) == 0
    assert "published as taxonomy v4" in capsys.readouterr().err
    with Database(data / "store.db").connection() as conn:
        record = signal_store.record(conn)
    assert record.settings.detection == "rules" and record.settings.pipeline == "v2"
    assert record.current.taxonomy.categories[0].subcategories[0].name == "Stock check"  # the edit is kept
    assert [c.recipe for c in record.current.taxonomy.categories] == [c.recipe for c in seed.taxonomy.categories]
    assert record.current.taxonomy.rules == seed.taxonomy.rules
    assert store_cli.main(["apply-signals-recipes", str(SEED)]) == 0
    assert "already current" in capsys.readouterr().err
    assert store_cli.main(["signals-detection", "model"]) == 0
    assert "signal detection is model" in capsys.readouterr().err


def test_the_seed_recipes_pass_the_save_validator_and_select_rules():
    body = SignalTaxonomySave.model_validate(json.loads(SEED.read_text(encoding="utf-8")))
    signal_store.check_taxonomy(body.taxonomy, ContractParameters())
    checked = [c.category_id for c in body.taxonomy.categories if c.recipe.check == "gemma"]
    assert checked == ["caller_confirms_resolved"]
    from call1.contracts.signals import SignalSettings, rules_categories

    assert len(rules_categories(body.taxonomy, SignalSettings())) == 10
    assert len(rules_categories(body.taxonomy, SignalSettings(detection="rules"))) == 10
