"""GET /status, /status/detail and /contract, and the dev/production configuration rules."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from call1.contracts.admin import StoreHealth
from call1.contracts.common import CONTRACT_PARAMETERS, CONTRACT_VERSION, ContractInfo, ServiceScope
from call1.store.app import create_app
from call1.store.config import ConfigError, StoreConfig


def test_status_is_anonymous_and_reports_the_contract_and_dev_relying_party(client, store):
    response = client.get("/store/v1/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(StoreHealth.model_fields)
    assert body["contract"] == {"contract_version": CONTRACT_VERSION, "parameters": CONTRACT_PARAMETERS.model_dump(mode="json")}
    assert body["store_hostname"] == "localhost"
    assert body["relying_party"] == {"rp_id": "localhost", "rp_name": "Call1 Store", "allowed_origins": ["http://localhost:8010"]}
    assert body["tls_health"] == "untrusted"  # TLS state is Stage 4
    assert body["server_time"] == "2026-09-25T12:00:00Z"
    assert response.headers["Cache-Control"] == "no-store"


def test_contract_reports_the_effective_parameters(tmp_path, clock):
    params = CONTRACT_PARAMETERS.model_copy(update={"lease_duration_seconds": 60, "heartbeat_interval_seconds": 20})
    app = create_app(StoreConfig.for_tests(tmp_path / "s", parameters=params), clock=clock)
    body = TestClient(app).get("/store/v1/contract").json()
    info = ContractInfo.model_validate(body)
    assert info.contract_version == CONTRACT_VERSION
    assert info.parameters.lease_duration_seconds == 60 and info.parameters.heartbeat_interval_seconds == 20


def test_status_detail_needs_an_admin_session_or_admin_state_read(client, mint_session, mint_service_key):
    from call1.contracts.common import ReviewerRole

    assert client.get("/store/v1/status/detail").json()["code"] == "unauthenticated"
    for role in (ReviewerRole.REVIEWER, ReviewerRole.SUPERVISOR):
        denied = client.get("/store/v1/status/detail", headers=mint_session(role).read_headers)
        assert denied.status_code == 403 and denied.json()["code"] == "insufficient_role"
    no_scope = client.get("/store/v1/status/detail", headers=mint_service_key([ServiceScope.JOBS_CLAIM]).headers)
    assert no_scope.status_code == 403 and no_scope.json()["code"] == "insufficient_scope"
    assert no_scope.json()["details"]["required_scope"] == "admin-state:read"
    admin = client.get("/store/v1/status/detail", headers=mint_session(ReviewerRole.ADMIN).read_headers)
    process = client.get("/store/v1/status/detail", headers=mint_service_key([ServiceScope.ADMIN_STATE_READ]).headers)
    assert admin.status_code == 200 and process.status_code == 200


def test_status_detail_reports_schema_feed_and_build(client, admin_session):
    body = client.get("/store/v1/status/detail", headers=admin_session.read_headers).json()
    changes = client.get("/store/v1/changes", headers=admin_session.read_headers).json()
    assert body["feed_epoch"] == changes["feed_epoch"]
    assert body["latest_change_cursor"] == changes["latest_cursor"]
    assert body["schema_version"] >= 10
    assert body["object_store_kind"] == "local_directory"
    assert body["admin_state_version"] == 0
    assert body["tls"] is None  # Stage 4; contract gap raised with the orchestrator
    build = body["running_build"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", build["manifest_digest"]) and build["approved"] is False


def test_production_mode_status_is_exactly_the_contract(tmp_path, clock):
    config = StoreConfig.for_tests(tmp_path / "s", hostname="qa.example.com")
    assert not config.dev_mode
    assert config.public_base_url == "https://qa.example.com:8010"
    assert config.cookie_name == "__Host-call1_session" and config.cookie_secure
    app = create_app(config, clock=clock)
    response = TestClient(app, base_url="https://qa.example.com:8010").get("/store/v1/status")
    health = StoreHealth.model_validate(response.json())
    assert health.relying_party.rp_id == "qa.example.com"
    assert health.relying_party.allowed_origins == ["https://qa.example.com:8010"]


def test_dev_mode_config_rules(tmp_path):
    dev = StoreConfig.for_tests(tmp_path)
    assert dev.dev_mode and dev.cookie_name == "call1_session" and not dev.cookie_secure
    assert dev.public_base_url == "http://localhost:8010"
    assert StoreConfig.for_tests(tmp_path, dev_origins=("http://localhost:5173",)).allowed_origins == ("http://localhost:8010", "http://localhost:5173")
    with pytest.raises(ConfigError):
        StoreConfig.for_tests(tmp_path, hostname="qa.example.com", dev_mode=True)
    with pytest.raises(ConfigError):
        StoreConfig.for_tests(tmp_path, dev_mode=False)  # localhost is never a production hostname
    with pytest.raises(ConfigError):
        StoreConfig.for_tests(tmp_path, bind_host="0.0.0.0")  # plain HTTP stays on loopback
    with pytest.raises(ConfigError):
        StoreConfig.for_tests(tmp_path, dev_origins=("https://evil.example.com",))
    with pytest.raises(ConfigError):
        StoreConfig.for_tests(tmp_path, hostname="qa.example.com", public_url="https://other.example.com")


def test_config_from_env(tmp_path):
    env = {"CALL1_STORE_DATA": str(tmp_path / "d"), "CALL1_STORE_PORT": "9001", "CALL1_STORE_PARAMETERS": '{"lease_duration_seconds": 90}'}
    config = StoreConfig.from_env(env)
    assert config.data_dir == (tmp_path / "d").resolve() and config.port == 9001 and config.dev_mode
    assert config.parameters.lease_duration_seconds == 90
    assert config.db_path.name == "store.db" and config.objects_dir.name == "objects"
