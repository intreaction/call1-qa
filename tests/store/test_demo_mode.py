"""DEMO MODE (``call1/store/auth/demo.py``): persona sign-in without a passkey, localhost only.

Off: ``/demo/*`` is 404 and the Evaluate catch-all never answers for it. On: ``/demo/sign-in``
issues the same server-side session cookie and CSRF token a passkey sign-in does, and the
``/store/v1`` routes enforce roles, CSRF and origin against it as usual. Outside dev mode the
configuration refuses demo mode, so ``serve`` never starts.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from call1.contracts.auth import SessionInfo
from call1.contracts.common import ReviewerRole
from call1.store import audit
from call1.store.__main__ import main as store_main
from call1.store.app import create_app
from call1.store.auth import records
from call1.store.config import ConfigError, StoreConfig
from call1.contracts.events import AuditAction, AuditQuery

from .conftest import STORE_BASE_URL

API = "/store/v1"
PERSONAS = {"admin": ("Demo Admin", "admin"), "supervisor": ("Demo Supervisor", "supervisor"), "reviewer": ("Demo Reviewer", "reviewer")}


@pytest.fixture
def demo_app(tmp_path, clock):
    return create_app(StoreConfig.for_tests(tmp_path / "demo-store", demo_mode=True), clock=clock)


def _client(app, base_url: str = STORE_BASE_URL) -> TestClient:
    return TestClient(app, base_url=base_url)


def _sign_in(client: TestClient, persona: str, **kwargs):
    return client.post("/demo/sign-in", json={"persona": persona}, **kwargs)


# --- demo off ---------------------------------------------------------------------------------


def test_demo_routes_are_404_when_demo_mode_is_off(client, store):
    assert store.config.demo_mode is False
    status = client.get("/demo/status")
    assert status.status_code == 404 and status.json()["code"] == "not_found"
    assert "text/html" not in status.headers.get("content-type", "")  # not the Evaluate page
    for body in ({"persona": "admin"}, {"persona": "nobody"}, None):
        signed = client.post("/demo/sign-in", json=body)
        assert signed.status_code == 404 and signed.json()["code"] == "not_found"
        assert client.cookies.get(store.config.cookie_name) is None
    with store.connection() as conn:
        assert records.account_row_by_email(conn, "demo.admin@call1-demo.example") is None


def test_demo_mode_defaults_off_and_reads_call1_store_demo(tmp_path):
    base = {"CALL1_STORE_DATA": str(tmp_path / "s")}
    assert StoreConfig.from_env(base).demo_mode is False
    assert StoreConfig.from_env({**base, "CALL1_STORE_DEMO": "0"}).demo_mode is False
    config = StoreConfig.from_env({**base, "CALL1_STORE_DEMO": "1"})
    assert config.demo_mode is True and config.dev_mode is True


# --- refuses non-localhost --------------------------------------------------------------------


def test_demo_mode_refuses_to_start_outside_dev_mode(tmp_path, capsys):
    with pytest.raises(ConfigError, match="CALL1_STORE_DEMO works only in dev mode"):
        StoreConfig(data_dir=tmp_path / "s", hostname="store.example.com", demo_mode=True)
    with pytest.raises(ConfigError, match="CALL1_STORE_DEMO works only in dev mode"):
        StoreConfig(data_dir=tmp_path / "s", hostname="localhost", dev_mode=False, demo_mode=True)
    with pytest.raises(ConfigError, match="CALL1_STORE_DEMO works only in dev mode"):
        StoreConfig.from_env({"CALL1_STORE_DATA": str(tmp_path / "s"), "CALL1_STORE_HOSTNAME": "store.example.com",
                              "CALL1_STORE_DEMO": "1"})
    # dev mode itself refuses a non-loopback bind, so demo mode never listens off-host
    with pytest.raises(ConfigError, match="loopback"):
        StoreConfig(data_dir=tmp_path / "s", bind_host="0.0.0.0", demo_mode=True)


def test_serve_exits_2_before_listening_when_demo_is_on_outside_dev_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CALL1_STORE_DATA", str(tmp_path / "s"))
    monkeypatch.setenv("CALL1_STORE_HOSTNAME", "store.example.com")
    monkeypatch.setenv("CALL1_STORE_DEMO", "1")
    monkeypatch.setenv("CALL1_STORE_PORT", "59999")  # never reached: config is refused first
    assert store_main(["serve"]) == 2
    assert "CALL1_STORE_DEMO works only in dev mode" in capsys.readouterr().err


def test_demo_sign_in_refuses_a_non_loopback_host_and_a_foreign_origin(demo_app):
    rebound = _client(demo_app, "http://evil.example:8010")
    refused = _sign_in(rebound, "admin")
    assert refused.status_code == 403 and refused.json()["details"]["reason"] == "demo_host_not_loopback"
    assert rebound.cookies.get("call1_session") is None
    with _client(demo_app) as client:
        foreign = _sign_in(client, "admin", headers={"Origin": "http://evil.example"})
        assert foreign.status_code == 403 and foreign.json()["code"] == "origin_not_allowed"
        assert client.cookies.get("call1_session") is None
    for base in ("http://127.0.0.1:8010", "http://localhost:8010"):
        assert _sign_in(_client(demo_app, base), "reviewer").status_code == 200


# --- demo on ----------------------------------------------------------------------------------


def test_demo_status_lists_the_three_personas(demo_app):
    with _client(demo_app) as client:
        response = client.get("/demo/status")
    assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store"
    body = response.json()
    assert body["demo"] is True and "Demo mode" in body["label"]
    assert [(p["persona"], p["display_name"], p["role"]) for p in body["personas"]] == [
        ("admin", "Demo Admin", "admin"), ("supervisor", "Demo Supervisor", "supervisor"), ("reviewer", "Demo Reviewer", "reviewer")]
    # the contract status is unchanged by demo mode
    with _client(demo_app) as client:
        status = client.get(f"{API}/status").json()
    assert "demo" not in status


@pytest.mark.parametrize("persona", sorted(PERSONAS))
def test_demo_sign_in_issues_a_normal_session_cookie_and_csrf(demo_app, persona):
    name, role = PERSONAS[persona]
    with _client(demo_app) as client:
        response = _sign_in(client, persona, headers={"Origin": STORE_BASE_URL})
        assert response.status_code == 200
        body = response.json()
        assert body["demo"] is True and body["persona"] == persona
        info = SessionInfo.model_validate(body["session"])
        assert info.display_name == name and info.role.value == role
        cookie = response.headers["set-cookie"]
        assert cookie.startswith("call1_session=") and "HttpOnly" in cookie and "SameSite=strict" in cookie and "Path=/" in cookie
        assert response.headers["Cache-Control"] == "no-store"
        session = client.get(f"{API}/auth/session")
        assert session.status_code == 200
        assert session.json()["account_id"] == info.account_id and session.json()["csrf_token"] == info.csrf_token
        assert session.json()["role"] == role


def test_demo_session_enforces_roles_and_csrf_on_store_v1_routes(demo_app):
    clients = {}
    for persona in PERSONAS:
        client = _client(demo_app)
        csrf = _sign_in(client, persona).json()["session"]["csrf_token"]
        clients[persona] = (client, {"X-Call1-CSRF": csrf})

    # admin-only read
    for persona, expected in (("reviewer", 403), ("supervisor", 403), ("admin", 200)):
        client, _ = clients[persona]
        response = client.get(f"{API}/status/detail")
        assert response.status_code == expected, persona
        if expected == 403:
            assert response.json()["code"] == "insufficient_role"
    client, _ = clients["admin"]
    assert client.get(f"{API}/admin/accounts").status_code == 200
    reviewer, _ = clients["reviewer"]
    assert reviewer.get(f"{API}/admin/accounts").json()["code"] == "insufficient_role"

    # a state-changing route needs the CSRF token, exactly as after a passkey sign-in
    reviewer, csrf = clients["reviewer"]
    missing = reviewer.post(f"{API}/auth/sign-out")
    assert missing.status_code == 403 and missing.json()["code"] == "csrf_failed"
    assert reviewer.post(f"{API}/auth/sign-out", headers=csrf).status_code == 200
    assert reviewer.get(f"{API}/auth/session").status_code == 401


def test_demo_sign_in_is_idempotent_and_audited_with_the_persona(demo_app):
    store = demo_app.state.store
    with _client(demo_app) as client:
        first = _sign_in(client, "supervisor").json()["session"]
        second = _sign_in(client, "supervisor").json()["session"]
    assert first["account_id"] == second["account_id"] and first["session_id"] != second["session_id"]
    with store.connection() as conn:
        rows = conn.execute("SELECT * FROM auth_accounts WHERE email_key LIKE 'demo.%'").fetchall()
        assert len(rows) == 1 and rows[0]["display_name"] == "Demo Supervisor"
        assert len(records.live_credential_rows(conn, first["account_id"])) == 1
        events = audit.list_events(conn, AuditQuery(limit=50)).items
        assert audit.verify_chain(conn)
    demo_events = [e for e in events if e.details.get("demo_mode") is True]
    assert all(e.details["demo_persona"] == "supervisor" for e in demo_events)
    actions = [e.action for e in reversed(demo_events)]
    assert actions == [AuditAction.ACCOUNT_CREATED, AuditAction.AUTHENTICATOR_ADDED, AuditAction.ACCOUNT_UPDATED, AuditAction.ACCOUNT_UPDATED]
    sign_ins = [e for e in demo_events if e.details.get("event") == "demo_sign_in"]
    assert [e.actor.session_id for e in reversed(sign_ins)] == [first["session_id"], second["session_id"]]
    assert all(e.actor.kind.value == "reviewer" and e.actor.display == "Demo Supervisor (demo mode)" for e in sign_ins)


def test_demo_sign_in_restores_a_changed_role_and_refuses_a_disabled_persona(demo_app):
    store = demo_app.state.store
    with _client(demo_app) as client:
        reviewer = _sign_in(client, "reviewer").json()["session"]
    with store.connection() as conn:
        records.update_account(conn, reviewer["account_id"], store.clock.now(), role=ReviewerRole.SUPERVISOR)
        conn.commit()
    with _client(demo_app) as client:
        assert _sign_in(client, "reviewer").json()["session"]["role"] == "reviewer"
    with store.connection() as conn:
        records.update_account(conn, reviewer["account_id"], store.clock.now(), status="disabled")
        conn.commit()
    with _client(demo_app) as client:
        refused = _sign_in(client, "reviewer")
    assert refused.status_code == 403 and refused.json()["code"] == "account_disabled"


def test_demo_sign_in_rejects_an_unknown_persona(demo_app):
    with _client(demo_app) as client:
        response = client.post("/demo/sign-in", json={"persona": "root"})
        assert response.status_code == 422
        assert client.cookies.get("call1_session") is None


def test_passkey_sign_in_still_works_beside_demo_mode(demo_app):
    """The real ceremonies are untouched: a passkey admin enrolls, a demo persona signs in, and the
    passkey admin signs in again."""
    from .auth_flows import bootstrap_admin

    admin = bootstrap_admin(demo_app)
    with _client(demo_app) as client:
        assert _sign_in(client, "admin").status_code == 200
    again = admin.sign_in()
    assert again.status_code == 200, again.text
    assert admin.get("/auth/session").json()["account_id"] == admin.account_id
    assert admin.get("/auth/session").json()["display_name"] == "First Admin"
