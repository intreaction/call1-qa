"""``call1.demo_setup``: the demo's retail rubric policy, alert rule and queue rule over the Store
API (BUG D1: without a policy SEC-01 and COMP-01 always need review, so no demo call can pass)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from call1.contracts.signals import SignalTaxonomySave
from call1.demo_setup import DEMO_ALERT_RULE, DEMO_QUEUE_RULE, DEMO_RUBRIC_ID, DemoSetupError, apply_demo_setup
from call1.models.schemas import RubricCriterion
from call1.pipeline.contextual_rubrics import RETAIL_DEMO_POLICY, with_policy
from call1.pipeline.evaluator import DEFAULT_RUBRIC, RubricEvaluator
from call1.store import audit
from call1.store.app import create_app
from call1.store.config import StoreConfig
from call1.store.results import signal_store

from .conftest import STORE_BASE_URL

V = "/store/v1"
SEED = Path(__file__).resolve().parents[2] / "call1" / "store" / "seeds" / "signals_retail_v1.json"


@pytest.fixture
def demo_app(tmp_path, clock):
    return create_app(StoreConfig.for_tests(tmp_path / "demo-store", demo_mode=True), clock=clock)


def _seed_taxonomy(app) -> None:
    body = SignalTaxonomySave.model_validate(json.loads(SEED.read_text(encoding="utf-8")))
    store = app.state.store
    with store.connection() as conn:
        signal_store.apply_seed(conn, body, actor=audit.installer_actor("test"), parameters=store.config.parameters, pipeline="v2")


def test_the_shipped_default_rubric_keeps_policy_empty():
    """Non-demo installs state their own policy: the default flags SEC-01/COMP-01 until they do."""
    checks = {c.criterion_id: c.check for c in DEFAULT_RUBRIC.criteria}
    assert checks["SEC-01"].requires_policy and not checks["SEC-01"].policy_context
    assert checks["COMP-01"].requires_policy and not checks["COMP-01"].policy_context
    assert set(RETAIL_DEMO_POLICY) == {"SEC-01", "COMP-01"}


def test_with_policy_fills_only_the_named_criteria_and_is_idempotent():
    definition = DEFAULT_RUBRIC.model_dump(mode="json")
    filled, changed = with_policy(definition)
    assert changed and definition["criteria"] != filled["criteria"]  # the input is not mutated
    by_id = {c["criterion_id"]: c["check"] for c in filled["criteria"]}
    assert by_id["SEC-01"]["policy_context"] == RETAIL_DEMO_POLICY["SEC-01"]
    assert by_id["COMP-01"]["policy_context"] == RETAIL_DEMO_POLICY["COMP-01"]
    assert not by_id["REG-01"].get("policy_context") and not by_id["ETIQ-01"].get("policy_context")
    again, changed_again = with_policy(filled)
    assert again == filled and changed_again is False


def test_the_policy_reaches_the_semantic_prompt_and_lifts_the_policy_gate():
    filled, _ = with_policy(DEFAULT_RUBRIC.model_dump(mode="json"))
    sec = RubricCriterion.model_validate(next(c for c in filled["criteria"] if c["criterion_id"] == "SEC-01"))
    evaluator = RubricEvaluator(DEFAULT_RUBRIC)
    prompt = evaluator._semantic_prompt(sec.check, [])
    assert "Identity verification policy" in prompt
    # the requires_policy gate flags only when policy_context is empty
    assert sec.check.requires_policy and sec.check.policy_context.strip()


def test_apply_demo_setup_publishes_the_policy_and_creates_the_alert_and_queue_rules(demo_app):
    _seed_taxonomy(demo_app)
    lines = []
    with TestClient(demo_app, base_url=STORE_BASE_URL) as client:
        summary = apply_demo_setup(client, say=lines.append)
        assert summary["rubric_version"] == 2 and summary["alert_rule"] == 1 and summary["queue_rule"] == 1
        assert "alert_error" not in summary

        from call1.demo_setup import sign_in

        headers = sign_in(client, "reviewer")
        rubric = client.get(f"{V}/rubrics/{DEMO_RUBRIC_ID}", headers=headers).json()
        assert rubric["ref"]["version"] == 2
        checks = {c["criterion_id"]: c["check"] for c in rubric["definition"]["criteria"]}
        assert checks["SEC-01"]["policy_context"] == RETAIL_DEMO_POLICY["SEC-01"]
        assert checks["COMP-01"]["policy_context"] == RETAIL_DEMO_POLICY["COMP-01"]
        rules = client.get(f"{V}/signals/alert-rules", headers=headers).json()["items"]
        assert [(r["rule_id"], r["node_active"], r["condition"]["subcategory_id"]) for r in rules] == [
            (DEMO_ALERT_RULE["rule_id"], True, "check_stock_availability")]
        queue_rules = {r["id"]: r for r in client.get(f"{V}/review-queue/rules", headers=headers).json()["items"]}
        assert queue_rules[DEMO_QUEUE_RULE["id"]]["stream"] == "SIGNAL"
        assert queue_rules[DEMO_QUEUE_RULE["id"]]["target_signal_alerts"] == [DEMO_ALERT_RULE["rule_id"]]

        # idempotent: a second run changes nothing
        again = apply_demo_setup(client, say=lines.append)
        assert again == {"rubric_version": None, "alert_rule": None, "queue_rule": None}
        assert client.get(f"{V}/rubrics/{DEMO_RUBRIC_ID}", headers=headers).json()["ref"]["version"] == 2


def test_apply_demo_setup_reanalyze_with_no_calls_requests_nothing(demo_app):
    _seed_taxonomy(demo_app)
    with TestClient(demo_app, base_url=STORE_BASE_URL) as client:
        summary = apply_demo_setup(client, reanalyze=True, say=lambda _: None)
    assert summary["reanalysis_requests"] == 0


def test_without_the_retail_taxonomy_the_policy_still_applies(demo_app):
    lines = []
    with TestClient(demo_app, base_url=STORE_BASE_URL) as client:
        summary = apply_demo_setup(client, say=lines.append)
    assert summary["rubric_version"] == 2 and "alert_error" in summary
    assert any("Alert rule not applied" in line for line in lines)


def test_apply_demo_setup_refuses_a_store_without_demo_mode(tmp_path, clock):
    app = create_app(StoreConfig.for_tests(tmp_path / "plain-store"), clock=clock)
    with TestClient(app, base_url=STORE_BASE_URL) as client:
        with pytest.raises(DemoSetupError, match="not in demo mode"):
            apply_demo_setup(client, say=lambda _: None)
