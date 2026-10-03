"""The principal guard: service keys, reviewer sessions, roles, CSRF and origin."""

from __future__ import annotations

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from call1.contracts.api import ANONYMOUS, Route, process, session
from call1.contracts.auth import Permission
from call1.contracts.common import ReviewerRole, ServiceScope
from call1.store.deps import current_principal, guard_for
from call1.store.errors import StoreError
from call1.store.principals import (
    Principal,
    ServiceKeyPrincipal,
    SessionPrincipal,
    generate_service_key_token,
    hash_secret,
    parse_bearer,
    require_own_installation,
    require_permission,
)

R, P, S = ReviewerRole, Permission, ServiceScope

TEST_ROUTES = [
    Route("GET", "/guarded/reviewer", "tReviewer", "", "t", [session(R.REVIEWER, P.READ_CALLS)]),
    Route("POST", "/guarded/reviewer-write", "tReviewerWrite", "", "t", [session(R.REVIEWER, P.OVERRIDE_VERDICT)]),
    Route("GET", "/guarded/admin-or-process", "tMixed", "", "t", [session(R.ADMIN, P.MANAGE_ADMIN_STATE), process(S.ADMIN_STATE_READ)]),
    Route("POST", "/guarded/process", "tProcess", "", "t", [process(S.JOBS_WRITE)]),
    Route("GET", "/guarded/anonymous", "tAnonymous", "", "t", [ANONYMOUS]),
]


def _describe(principal: Principal = Depends(current_principal)) -> dict:
    if isinstance(principal, ServiceKeyPrincipal):
        return {"kind": "key", "installation_id": principal.installation_id}
    if isinstance(principal, SessionPrincipal):
        return {"kind": "session", "role": principal.role.value, "account_id": principal.account_id}
    return {"kind": "anonymous"}


@pytest.fixture
def guarded(app):
    for route in TEST_ROUTES:
        app.add_api_route(route.path, _describe, methods=[route.method], dependencies=[Depends(guard_for(route))])
        app.router.routes.insert(0, app.router.routes.pop())  # ahead of the Evaluate catch-all
    with TestClient(app, base_url="http://localhost:8010") as client:
        yield client


def _code(response) -> str:
    return response.json()["code"]


def test_no_credential_is_unauthenticated(guarded):
    response = guarded.get("/guarded/reviewer")
    assert response.status_code == 401 and _code(response) == "unauthenticated"
    assert guarded.get("/guarded/anonymous").json() == {"kind": "anonymous"}


def test_service_keys(guarded, mint_service_key):
    key = mint_service_key([S.JOBS_WRITE])
    ok = guarded.post("/guarded/process", headers=key.headers)
    assert ok.status_code == 200 and ok.json() == {"kind": "key", "installation_id": key.installation_id}
    assert _code(guarded.post("/guarded/process", headers={"Authorization": "Bearer nonsense"})) == "unauthenticated"
    unknown, _ = generate_service_key_token()
    assert _code(guarded.post("/guarded/process", headers={"Authorization": f"Bearer {unknown}"})) == "unauthenticated"
    lacking = guarded.post("/guarded/process", headers=mint_service_key([S.CHANGES_READ]).headers)
    assert lacking.status_code == 403 and _code(lacking) == "insufficient_scope"
    wrong_principal = guarded.get("/guarded/reviewer", headers=key.headers)
    assert wrong_principal.status_code == 403 and _code(wrong_principal) == "forbidden"
    assert guarded.get("/guarded/admin-or-process", headers=mint_service_key([S.ADMIN_STATE_READ]).headers).status_code == 200


def test_sessions_and_roles(guarded, reviewer_session, admin_session):
    assert guarded.get("/guarded/reviewer", headers=reviewer_session.read_headers).json()["role"] == "reviewer"
    denied = guarded.get("/guarded/admin-or-process", headers=reviewer_session.read_headers)
    assert denied.status_code == 403 and _code(denied) == "insufficient_role"
    assert denied.json()["details"] == {"required_role": "admin", "required_permission": "manage_admin_state"}
    assert guarded.get("/guarded/admin-or-process", headers=admin_session.read_headers).json()["role"] == "admin"
    on_process_route = guarded.post("/guarded/process", headers=admin_session.headers)
    assert on_process_route.status_code == 403 and _code(on_process_route) == "forbidden"
    stranger = guarded.get("/guarded/reviewer", headers={"Cookie": "call1_session=not-a-session"})
    assert stranger.status_code == 401 and _code(stranger) == "unauthenticated"


def test_writes_need_csrf_and_an_allowed_origin(guarded, reviewer_session):
    missing = guarded.post("/guarded/reviewer-write", headers=reviewer_session.read_headers)
    assert missing.status_code == 403 and _code(missing) == "csrf_failed"
    wrong = guarded.post("/guarded/reviewer-write", headers={**reviewer_session.read_headers, "X-Call1-CSRF": "x" * 43})
    assert _code(wrong) == "csrf_failed"
    assert guarded.post("/guarded/reviewer-write", headers=reviewer_session.headers).status_code == 200
    foreign = guarded.post("/guarded/reviewer-write", headers={**reviewer_session.headers, "Origin": "http://evil.example"})
    assert foreign.status_code == 403 and _code(foreign) == "origin_not_allowed"
    same = guarded.post("/guarded/reviewer-write", headers={**reviewer_session.headers, "Origin": "http://localhost:8010"})
    assert same.status_code == 200
    assert guarded.get("/guarded/reviewer", headers=reviewer_session.read_headers).status_code == 200  # reads need no CSRF


def test_sessions_expire_on_the_store_clock(guarded, reviewer_session, clock, store):
    assert guarded.get("/guarded/reviewer", headers=reviewer_session.read_headers).status_code == 200
    clock.advance(seconds=store.config.parameters.session_idle_lifetime_seconds + 1)
    expired = guarded.get("/guarded/reviewer", headers=reviewer_session.read_headers)
    assert expired.status_code == 401 and _code(expired) == "session_expired"


def test_object_rule_helpers():
    key = ServiceKeyPrincipal(key_id="key_1", installation_id="inst_a", scopes=frozenset({S.JOBS_CLAIM}))
    require_own_installation(key, "inst_a")
    require_own_installation(key, None)
    with pytest.raises(StoreError) as other:
        require_own_installation(key, "inst_b")
    assert other.value.code.value == "forbidden"
    reviewer = SessionPrincipal(session_id="s", account_id="a", email="r@example.com", display_name="R", role=R.REVIEWER)
    with pytest.raises(StoreError) as weak:
        require_permission(reviewer, P.RESOLVE_ANY_REVIEW)
    assert weak.value.code.value == "insufficient_role"
    supervisor = SessionPrincipal(session_id="s", account_id="a", email="r@example.com", display_name="R", role=R.SUPERVISOR)
    assert require_permission(supervisor, P.RESOLVE_ANY_REVIEW) is supervisor


def test_token_format_and_hashing():
    token, prefix = generate_service_key_token()
    assert token.startswith(prefix + "_") and len(prefix) == 11
    assert parse_bearer(f"Bearer {token}") == token and parse_bearer(f"bearer  {token} ") == token
    assert parse_bearer("Basic abc") is None and parse_bearer(None) is None and parse_bearer("Bearer c1sk_short") is None
    assert hash_secret(token).startswith("sha256:") and len(hash_secret(token)) == 71
