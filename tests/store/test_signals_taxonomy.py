"""Contact Signals v2 taxonomy, settings, alert rules and redaction in Store (contract 1.3.0;
docs/ContactSignalsV2.md sections 7.2, 9.1, 9.2, 9.4 and 9.6; F2 acceptance "Taxonomy" and "Caps")."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from call1.contracts.common import ContractParameters
from call1.contracts.signals import (
    SignalSubcategory,
    SignalTaxonomySave,
    builtin_signal_taxonomy,
    signal_taxonomy_cap_violations,
    taxonomy_digest,
)
from call1.store import __main__ as store_cli
from call1.store.app import create_app
from call1.store.config import StoreConfig
from call1.store.results import signal_store

from .conftest import STORE_BASE_URL
from .test_signals_harness import BILLING, CANCEL, TAXONOMY, UPSELL, V, put_alert, put_taxonomy, save_body, taxonomy_with

SEED = Path(__file__).resolve().parents[2] / "call1" / "store" / "seeds" / "signals_retail_v1.json"


def _changes(client, session, kind):
    return [e for e in client.get(f"{V}/changes", headers=session.read_headers).json()["events"] if e["kind"] == kind]


def _audit(client, session, action):
    return client.get(f"{V}/admin/audit", params={"action": action}, headers=session.read_headers).json()["items"]


# --- install-time version 1 ------------------------------------------------------------------------


def test_install_seeds_version_1_with_the_builtins_only(client, reviewer_session, service_key_headers):
    record = client.get(f"{V}/signals/taxonomy", headers=reviewer_session.read_headers)
    assert record.status_code == 200, record.text
    body = record.json()
    assert body["record_version"] == 1 and body["current"]["version"] == 1
    assert body["current"]["published_by_account_id"] is None
    assert body["current"]["taxonomy"] == builtin_signal_taxonomy().model_dump(mode="json")
    assert body["current"]["digest"] == taxonomy_digest(builtin_signal_taxonomy())
    assert body["settings"] == {"pipeline": "v1", "v1_fallback": True, "fallback_extraction_entry_id": None, "detection": "model"}
    # Process reads it too (jobs:write), to mint snapshots at ingest.
    assert client.get(f"{V}/signals/taxonomy", headers=service_key_headers).status_code == 200


# --- save, version, no-op replay, conflict --------------------------------------------------------


def test_save_publishes_the_next_version_and_a_no_op_replay_changes_nothing(client, admin_session, reviewer_session):
    saved = put_taxonomy(client, admin_session, TAXONOMY)
    assert saved["current"]["version"] == 2 and saved["record_version"] == 2
    assert saved["current"]["digest"] == taxonomy_digest(TAXONOMY)
    assert saved["current"]["published_by_account_id"] == admin_session.account_id
    again = client.put(f"{V}/signals/taxonomy", json=save_body(TAXONOMY, 2), headers=admin_session.headers)
    assert again.status_code == 200 and again.json() == saved  # same digest: the record, unchanged
    versions = client.get(f"{V}/signals/taxonomy/versions", headers=reviewer_session.read_headers).json()["items"]
    assert [v["version"] for v in versions] == [2, 1]
    one = client.get(f"{V}/signals/taxonomy/versions/1", headers=reviewer_session.read_headers).json()
    assert one["taxonomy"] == builtin_signal_taxonomy().model_dump(mode="json")
    assert client.get(f"{V}/signals/taxonomy/versions/9", headers=reviewer_session.read_headers).status_code == 404
    [event] = _changes(client, admin_session, "signal_taxonomy")
    assert event["status"] == "saved:v2" and event["resource_id"] == "signal_taxonomy"
    [entry] = _audit(client, admin_session, "signal_taxonomy_saved")
    assert entry["details"]["version"] == 2 and entry["details"]["digest"] == taxonomy_digest(TAXONOMY)
    assert "categories[0]" in entry["details"]["changed_paths"] and "Cancel" not in json.dumps(entry["details"])


def test_a_stale_record_version_is_409_with_the_current_versions(client, admin_session):
    put_taxonomy(client, admin_session, TAXONOMY)
    stale = client.put(f"{V}/signals/taxonomy", json=save_body(taxonomy_with(intent_threshold=0.3), 1), headers=admin_session.headers)
    assert stale.status_code == 409
    body = stale.json()
    assert body["code"] == "signal_taxonomy_conflict" and body["details"] == {"current_version": 2, "record_version": 2}


def test_only_admins_manage_the_taxonomy_and_reviewers_read_it(client, reviewer_session, supervisor_session):
    for session in (reviewer_session, supervisor_session):
        refused = client.put(f"{V}/signals/taxonomy", json=save_body(TAXONOMY, 1), headers=session.headers)
        assert refused.status_code == 403, refused.text
    assert client.get(f"{V}/signals/alert-rules", headers=reviewer_session.read_headers).status_code == 200


def test_builtin_edits_are_refused(client, admin_session):
    data = TAXONOMY.model_dump(mode="json")
    data["categories"][0]["name"] = "Why they called"
    refused = client.put(f"{V}/signals/taxonomy", json={"taxonomy": data, "expected_record_version": 1}, headers=admin_session.headers)
    assert refused.status_code == 422 and refused.json()["code"] == "validation_failed"
    data = TAXONOMY.model_dump(mode="json")
    data["categories"][1]["active"] = False
    assert client.put(f"{V}/signals/taxonomy", json={"taxonomy": data, "expected_record_version": 1}, headers=admin_session.headers).status_code == 422
    # What an admin may change on a built-in: thresholds, subcategories, fields, narrow_quote, examples.
    data = TAXONOMY.model_dump(mode="json")
    data["categories"][1].update(threshold=0.4, examples=["disputed fee"], narrow_quote=True)
    ok = client.put(f"{V}/signals/taxonomy", json={"taxonomy": data, "expected_record_version": 1}, headers=admin_session.headers)
    assert ok.status_code == 200, ok.text


def test_number_rules_refuse_by_path_never_by_value(client, admin_session):
    card = "4111 1111 1111 1111"
    sub = CANCEL.model_copy(update={"examples": ["cancel", f"card {card} please"]})
    refused = client.put(f"{V}/signals/taxonomy", json=save_body(taxonomy_with(intent_subs=[sub]), 1), headers=admin_session.headers)
    assert refused.status_code == 422, refused.text
    body = refused.json()
    assert body["code"] == "validation_failed" and body["details"]["field"] == "categories[0].subcategories[0].examples[1]"
    assert card not in refused.text and "4111" not in refused.text
    upsell = UPSELL.model_copy(update={"description": "Offers the plan to 555-123-4567 callers"})
    refused = client.put(f"{V}/signals/taxonomy", json=save_body(taxonomy_with(custom=[upsell]), 1), headers=admin_session.headers)
    assert refused.json()["details"]["field"] == "categories[8].description" and "555" not in refused.text


def test_redaction_tombstones_a_non_current_version_and_keeps_its_digest(client, admin_session, reviewer_session):
    put_taxonomy(client, admin_session, TAXONOMY)
    v2 = client.get(f"{V}/signals/taxonomy/versions/2", headers=reviewer_session.read_headers).json()
    body = {"digest": v2["digest"], "reason": "an example quoted a caller"}
    current = client.post(f"{V}/signals/taxonomy/versions/2/redaction", json=body, headers=admin_session.headers)
    assert current.status_code == 409 and current.json()["details"]["reason"] == "redact_current"
    put_taxonomy(client, admin_session, taxonomy_with(custom=[UPSELL.model_copy(update={"active": False})]))
    wrong = client.post(f"{V}/signals/taxonomy/versions/2/redaction", json={**body, "digest": "sha256:" + "0" * 64}, headers=admin_session.headers)
    assert wrong.status_code == 409 and wrong.json()["details"]["reason"] == "digest_mismatch"
    redacted = client.post(f"{V}/signals/taxonomy/versions/2/redaction", json=body, headers=admin_session.headers)
    assert redacted.status_code == 200, redacted.text
    out = redacted.json()
    assert out["text_redacted"] and out["digest"] == v2["digest"] and out["redacted_by_account_id"] == admin_session.account_id
    text = json.dumps(out["taxonomy"])
    assert "Cancel account" not in text and "Upsell" not in text and "[REDACTED 1]" in text
    assert "Caller objective" in text  # built-in constants stay
    replay = client.post(f"{V}/signals/taxonomy/versions/2/redaction", json=body, headers=admin_session.headers)
    assert replay.status_code == 200 and replay.json() == out
    [entry] = _audit(client, admin_session, "signal_taxonomy_redacted")
    assert entry["details"] == {"version": 2, "digest": v2["digest"]}
    assert client.get(f"{V}/signals/taxonomy/versions/2", headers=reviewer_session.read_headers).json()["text_redacted"] is True


# --- settings -------------------------------------------------------------------------------------


def test_settings_share_the_record_version_and_are_audited(client, admin_session):
    saved = client.put(f"{V}/signals/settings", json={"settings": {"pipeline": "shadow"}, "expected_record_version": 1}, headers=admin_session.headers)
    assert saved.status_code == 200, saved.text
    assert saved.json()["settings"]["pipeline"] == "shadow" and saved.json()["record_version"] == 2
    stale = client.put(f"{V}/signals/taxonomy", json=save_body(TAXONOMY, 1), headers=admin_session.headers)
    assert stale.status_code == 409
    [entry] = _audit(client, admin_session, "signal_settings_changed")
    assert entry["details"]["old_pipeline"] == "v1" and entry["details"]["new_pipeline"] == "shadow"
    assert [e["status"] for e in _changes(client, admin_session, "signal_taxonomy")] == ["settings"]


# --- caps from ContractParameters ------------------------------------------------------------------


def test_caps_come_from_the_effective_contract_parameters(tmp_path, clock, mint_session):
    tight = StoreConfig.for_tests(tmp_path / "tight").with_overrides(parameters=ContractParameters(max_option_gloss_chars=30,
                                                                                                    max_active_subcategories=1))
    app = create_app(tight, clock=clock)
    with TestClient(app, base_url=STORE_BASE_URL) as client, app.state.store.connection() as conn:
        admin = app.state.store.auth.mint_session_for_tests(conn, role="admin")
        # "Caller wants to cancel their account" is 36 characters: fine at 40, refused at 30.
        refused = client.put(f"{V}/signals/taxonomy", json=save_body(taxonomy_with(intent_subs=[CANCEL], custom=()), 1), headers=admin.headers)
        assert refused.status_code == 422
        assert refused.json()["details"] == {**refused.json()["details"], "field": "categories[0].subcategories[0].gloss", "cap": "max_option_gloss_chars",
                                             "limit": 30}
        short = CANCEL.model_copy(update={"gloss": "Caller wants to cancel"})
        refused = client.put(f"{V}/signals/taxonomy", json=save_body(taxonomy_with(intent_subs=[short, short.model_copy(update={
            "subcategory_id": "close", "name": "Close"})], custom=()), 1), headers=admin.headers)
        assert refused.json()["details"]["cap"] == "max_active_subcategories"
        assert refused.json()["details"]["field"] == "categories[0].subcategories[1]"
        ok = client.put(f"{V}/signals/taxonomy", json=save_body(taxonomy_with(intent_subs=[short], custom=()), 1), headers=admin.headers)
        assert ok.status_code == 200, ok.text


def test_the_default_caps_accept_the_same_taxonomy(client, admin_session):
    put_taxonomy(client, admin_session, taxonomy_with(intent_subs=[CANCEL], custom=()))


# --- alert rules -----------------------------------------------------------------------------------


def test_alert_rules_are_validated_against_the_taxonomy_and_report_node_active(client, admin_session, reviewer_session):
    put_taxonomy(client, admin_session, TAXONOMY)
    rule = put_alert(client, admin_session, "cancel-price", {"category_id": "intent", "subcategory_id": "cancel_account", "field_id": "reason",
                                                             "field_equals": "price"}, name="Cancel for price")
    assert rule["record_version"] == 1 and rule["node_active"] is True
    for bad, reason in (({"category_id": "nope"}, "unknown_category"), ({"category_id": "intent", "subcategory_id": "x"}, "unknown_subcategory"),
                        ({"category_id": "intent", "field_id": "reason", "field_equals": "free"}, "field_equals_type")):
        refused = put_alert(client, admin_session, "bad", bad, expect=422)
        assert refused["details"]["reason"] == reason
    stale = client.put(f"{V}/signals/alert-rules/cancel-price", json={"rule": {"rule_id": "cancel-price", "name": "x", "condition": {"category_id": "intent"}},
                                                                     "expected_record_version": 0}, headers=admin_session.headers)
    assert stale.status_code == 409 and stale.json()["details"]["current_version"] == 1
    assert put_alert(client, admin_session, "cancel-price", {"category_id": "intent"}, name="Account 12345678 callers", expect=422)["details"]["field"] == "rule.name"
    disabled = put_alert(client, admin_session, "cancel-price", {"category_id": "intent", "subcategory_id": "cancel_account"}, enabled=False)
    assert disabled["record_version"] == 2
    assert [e["status"] for e in _changes(client, admin_session, "signal_alert_rule")] == ["saved", "disabled"]
    # Retire the subcategory: the rule reads node_active false.
    put_taxonomy(client, admin_session, taxonomy_with(intent_subs=[CANCEL.model_copy(update={"active": False}), BILLING]))
    [listed] = client.get(f"{V}/signals/alert-rules", headers=reviewer_session.read_headers).json()["items"]
    assert listed["node_active"] is False
    assert client.put(f"{V}/signals/alert-rules/x", json={"rule": {"rule_id": "x", "name": "x", "condition": {"category_id": "intent"}},
                                                          "expected_record_version": 0}, headers=reviewer_session.headers).status_code == 403


def test_alert_rule_count_is_capped(tmp_path, clock):
    app = create_app(StoreConfig.for_tests(tmp_path / "s").with_overrides(parameters=ContractParameters(max_signal_alert_rules=1)), clock=clock)
    with TestClient(app, base_url=STORE_BASE_URL) as client, app.state.store.connection() as conn:
        admin = app.state.store.auth.mint_session_for_tests(conn, role="admin")
        put_alert(client, admin, "one", {"category_id": "intent"})
        assert put_alert(client, admin, "two", {"category_id": "issue"}, expect=422)["details"]["cap"] == "max_signal_alert_rules"
        put_alert(client, admin, "one", {"category_id": "issue"})  # replacing an existing rule is not a new one


# --- the retail seed ---------------------------------------------------------------------------------


def test_the_retail_seed_passes_the_save_validator():
    body = SignalTaxonomySave.model_validate(json.loads(SEED.read_text(encoding="utf-8")))
    assert signal_taxonomy_cap_violations(body.taxonomy, ContractParameters()) == []
    signal_store.check_taxonomy(body.taxonomy, ContractParameters())  # caps and the definition-text detectors
    assert len([c for c in body.taxonomy.categories if not c.builtin]) == 2


def test_apply_signals_seed_is_an_audited_admin_save_and_idempotent(tmp_path, monkeypatch, capsys):
    data = tmp_path / "store"
    monkeypatch.setenv("CALL1_STORE_DATA", str(data))
    monkeypatch.setenv("CALL1_STORE_MAINTENANCE_SECONDS", "0")
    assert store_cli.main(["apply-signals-seed", str(SEED), "--pipeline", "v2"]) == 0
    assert "published as taxonomy v2" in capsys.readouterr().err
    assert store_cli.main(["apply-signals-seed", str(SEED)]) == 0
    assert "already current" in capsys.readouterr().err
    from call1.store.db import Database

    with Database(data / "store.db").connection() as conn:
        record = signal_store.record(conn)
        assert record.current.version == 2 and record.settings.pipeline == "v2"
        assert record.current.digest == taxonomy_digest(SignalTaxonomySave.model_validate(json.loads(SEED.read_text())).taxonomy)
        actions = [r["action"] for r in conn.execute("SELECT action, actor_kind FROM audit_events ORDER BY sequence")]
        assert actions == ["signal_taxonomy_saved", "signal_settings_changed"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"taxonomy": {"categories": []}, "expected_record_version": 1}))
    assert store_cli.main(["apply-signals-seed", str(bad)]) == 2
    assert store_cli.main(["signals-pipeline", "v1"]) == 0
    assert "signal pipeline is v1" in capsys.readouterr().err


def test_published_nodes_are_retired_never_deleted(client, admin_session):
    put_taxonomy(client, admin_session, TAXONOMY)
    dropped = client.put(f"{V}/signals/taxonomy", json=save_body(taxonomy_with(custom=()), 2), headers=admin_session.headers)
    assert dropped.status_code == 422 and dropped.json()["details"]["reason"] == "node_removed"
    assert dropped.json()["details"]["node_id"] == "upsell"
    dropped = client.put(f"{V}/signals/taxonomy", json=save_body(taxonomy_with(intent_subs=[CANCEL]), 2), headers=admin_session.headers)
    assert dropped.json()["details"] == {**dropped.json()["details"], "field": "categories[0].subcategories", "node_id": "billing_question"}
    retired = UPSELL.model_copy(update={"active": False})
    put_taxonomy(client, admin_session, taxonomy_with(custom=[retired]))


def test_subcategory_ids_other_and_not_are_reserved(client, admin_session):
    data = TAXONOMY.model_dump(mode="json")
    data["categories"][0]["subcategories"].append(SignalSubcategory(subcategory_id="retention", name="Retention", gloss="Asks to stay").model_dump(mode="json"))
    data["categories"][0]["subcategories"][-1]["subcategory_id"] = "other"
    refused = client.put(f"{V}/signals/taxonomy", json={"taxonomy": data, "expected_record_version": 1}, headers=admin_session.headers)
    assert refused.status_code == 422


@pytest.fixture(autouse=True)
def _no_pipeline_env(monkeypatch):
    monkeypatch.delenv("CALL1_SIGNALS_PIPELINE", raising=False)
