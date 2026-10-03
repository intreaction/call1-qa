"""Store status and contract endpoints through the real Store server (inventory q1, q2)."""

from __future__ import annotations

import pytest

from call1.contracts.admin import StoreHealth, StoreStatus
from call1.contracts.common import CONTRACT_VERSION, ContractParameters

from .store_api_support import worker_harness

pytestmark = pytest.mark.e2e


def test_q1_status_and_status_detail(stack, admin_session, reviewer_session):
    """GET /status (anonymous) and /status/detail (admin or admin-state:read key) answer 200 with the
    contract version and the StoreHealth/StoreStatus fields; TLS is the documented Stage 4 gap:
    ``tls`` is null and ``tls_health`` is ``untrusted``."""
    health = stack.store_get("/status")
    assert health.status_code == 200, health.text
    body = health.json()
    assert set(body) == set(StoreHealth.model_fields), f"StoreHealth fields differ: {sorted(set(body) ^ set(StoreHealth.model_fields))}"
    assert body["contract"]["contract_version"] == CONTRACT_VERSION
    assert body["tls_health"] == "untrusted", body["tls_health"]
    assert body["store_hostname"] == "localhost"
    assert body["relying_party"]["rp_id"] == "localhost"
    assert stack.store_url in body["relying_party"]["allowed_origins"]

    # /status/detail is not anonymous, and a plain reviewer lacks manage_admin_state.
    assert stack.store_get("/status/detail").status_code == 401
    denied = reviewer_session.get("/status/detail")
    assert denied.status_code == 403, denied.text

    for who, response in (("admin", admin_session.get("/status/detail")), ("service key", stack.store_get("/status/detail", session="service"))):
        assert response.status_code == 200, f"{who}: {response.status_code} {response.text}"
        detail = response.json()
        assert set(detail) == set(StoreStatus.model_fields), f"{who}: StoreStatus fields differ: {sorted(set(detail) ^ set(StoreStatus.model_fields))}"
        assert detail["contract"]["contract_version"] == CONTRACT_VERSION
        assert detail["tls"] is None, f"{who}: tls should be the null Stage 4 placeholder, got {detail['tls']!r}"
        assert detail["running_build"]["approved"] is False
        assert detail["admin_state_version"] == 0
        assert detail["latest_change_cursor"].startswith(detail["feed_epoch"] + "-")


def test_q2_contract_reports_effective_parameter_overrides(stack_factory):
    """GET /store/v1/contract returns the effective ContractParameters, honouring
    CALL1_STORE_PARAMETERS, and Store applies them (the claim lease)."""
    overrides = {"lease_duration_seconds": 120, "heartbeat_interval_seconds": 40, "max_claim_batch": 4}
    private = stack_factory(name="params", with_process=False, store_parameters=overrides)

    response = private.store_get("/contract")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["contract_version"] == CONTRACT_VERSION
    defaults = ContractParameters().model_dump(mode="json")
    expected = {**defaults, **overrides}
    assert body["parameters"] == expected, {k: (body["parameters"].get(k), v) for k, v in expected.items() if body["parameters"].get(k) != v}
    # /status reports the same effective values.
    assert private.store_get("/status").json()["contract"]["parameters"] == expected

    # Store applies them: a claim leases for the overridden duration, and returns at most
    # max_claim_batch jobs ("most jobs one claim call may return") even when more are ready.
    q = worker_harness(private)
    for _ in range(3):
        q.ingest()  # 2 ready jobs each (validation_vad, asr): 6 ready
    claimed = q.claim(max_jobs=16)
    assert claimed["lease_duration_seconds"] == 120 and claimed["heartbeat_interval_seconds"] == 40, claimed
    assert len(claimed["jobs"]) == 4, f"max_claim_batch=4 should cap the claim, got {len(claimed['jobs'])} jobs"
    lease = claimed["jobs"][0]["job"]["lease"]
    assert lease is not None, claimed["jobs"][0]["job"]
