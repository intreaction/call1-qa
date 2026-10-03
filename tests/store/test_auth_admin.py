"""Admin identity over HTTP: accounts and roles, re-invites and lost keys, invitations, break-glass,
installations and service keys (issue, scope, rotate, revoke, retire), with role, CSRF and audit
checks. People enroll through real ceremonies (``auth_flows``); only the probes for other roles use
the TEST-ONLY minted sessions."""

from __future__ import annotations

from datetime import timedelta

import pytest

from call1.contracts.auth import SetupCodeIssueRequest
from call1.contracts.common import ServiceScope
from call1.store import audit
from call1.store.auth import api as auth_api
from call1.store.auth import cli as auth_cli

from .auth_flows import API, Person, bootstrap_admin, enroll, invite, invite_and_enroll, new_client, sign_in
from .auth_softauthn import SoftAuthenticator


def _code(response) -> str:
    return response.json()["code"]


def _reason(response) -> str:
    return response.json()["details"].get("reason")


def _audit(store, action: str) -> list:
    with store.connection() as conn:
        return [audit._from_row(r) for r in conn.execute("SELECT * FROM audit_events WHERE action = ? ORDER BY sequence", (action,))]


@pytest.fixture
def admin(app) -> Person:
    return bootstrap_admin(app)


# --- who may call ----------------------------------------------------------------------------


ADMIN_READS = ["/admin/accounts", "/admin/invitations", "/admin/setup-codes", "/admin/break-glass", "/admin/installations", "/admin/service-keys"]


@pytest.mark.parametrize("path", ADMIN_READS)
def test_admin_identity_routes_need_the_admin_role(client, path, reviewer_session, supervisor_session, admin_session, service_key_headers):
    for session in (reviewer_session, supervisor_session):
        denied = client.get(API + path, headers=session.read_headers)
        assert denied.status_code == 403 and _code(denied) == "insufficient_role"
    assert client.get(API + path, headers=admin_session.read_headers).status_code == 200
    by_key = client.get(API + path, headers=service_key_headers)
    assert by_key.status_code == 403 and _code(by_key) == "forbidden"
    assert _code(client.get(API + path)) == "unauthenticated"


def test_admin_writes_need_csrf(client, admin_session):
    body = {"label": "mac-mini"}
    missing = client.post(f"{API}/admin/installations", json=body, headers=admin_session.read_headers)
    assert missing.status_code == 403 and _code(missing) == "csrf_failed"
    foreign = client.post(f"{API}/admin/installations", json=body, headers={**admin_session.headers, "Origin": "http://evil.example"})
    assert foreign.status_code == 403 and _code(foreign) == "origin_not_allowed"
    assert client.post(f"{API}/admin/installations", json=body, headers=admin_session.headers).status_code == 201


# --- accounts ---------------------------------------------------------------------------------


def test_accounts_list_filter_page_and_get(app, admin, mint_session, conn):
    for role in ("reviewer", "reviewer", "supervisor"):
        mint_session(role)
    page = admin.get("/admin/accounts", params={"limit": 2}).json()
    assert len(page["items"]) == 2 and page["next_page_token"]
    rest = admin.get("/admin/accounts", params={"limit": 2, "page_token": page["next_page_token"]}).json()
    assert len(rest["items"]) == 2 and rest["next_page_token"] is None
    ids = [a["id"] for a in page["items"] + rest["items"]]
    assert len(set(ids)) == 4 and admin.account_id in ids
    reviewers = admin.get("/admin/accounts", params={"role": "reviewer"}).json()["items"]
    assert len(reviewers) == 2 and {a["role"] for a in reviewers} == {"reviewer"}
    one = admin.get(f"/admin/accounts/{admin.account_id}").json()
    assert one["email"] == "admin@example.com" and one["authenticator_count"] == 1 and one["last_sign_in_at"]
    assert admin.get("/admin/accounts/acct_missing").status_code == 404
    assert _code(admin.get("/admin/accounts", params={"page_token": "!!"})) == "validation_failed"
    # the api other areas use sees the same rows
    assert auth_api.get_account(conn, admin.account_id).email == "admin@example.com"
    assert len(auth_api.list_accounts(conn, role="reviewer")) == 2 and auth_api.get_account(conn, "acct_missing") is None


def test_a_role_change_applies_on_the_next_request(app, admin, store):
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    assert _code(reviewer.get("/admin/accounts")) == "insufficient_role"
    promoted = admin.write("PATCH", f"/admin/accounts/{reviewer.account_id}", json={"role": "admin", "reason": "team lead"})
    assert promoted.status_code == 200 and promoted.json()["role"] == "admin"
    info = reviewer.get("/auth/session").json()
    assert info["role"] == "admin" and "manage_accounts" in info["permissions"]
    assert reviewer.get("/admin/accounts").status_code == 200  # same session, new role
    renamed = admin.write("PATCH", f"/admin/accounts/{reviewer.account_id}", json={"display_name": "Rev Lead", "role": "reviewer", "reason": "rotation"})
    assert renamed.json()["display_name"] == "Rev Lead" and reviewer.get("/auth/session").json()["role"] == "reviewer"
    assert _code(reviewer.get("/admin/accounts")) == "insufficient_role"
    events = _audit(store, "account_updated")
    assert len(events) == 2 and events[0].actor.account_id == admin.account_id and events[0].details["role"] == "admin"
    assert events[0].details["reason"] == "team lead" and events[1].details["previous_role"] == "admin"


def test_disabling_revokes_sessions_and_reenabling_restores_sign_in(app, admin, store):
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    begun = new_client(app)
    ceremony = begun.post(f"{API}/auth/sign-in/begin", json={"email": reviewer.email}).json()  # begun while still active
    disabled = admin.write("PATCH", f"/admin/accounts/{reviewer.account_id}", json={"status": "disabled", "reason": "left the team"})
    assert disabled.status_code == 200 and disabled.json()["status"] == "disabled"
    assert _code(reviewer.get("/auth/session")) == "account_disabled"
    late = begun.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": ceremony["ceremony_id"], "credential": reviewer.authn.assert_(ceremony["options"])})
    assert late.status_code == 403 and _code(late) == "account_disabled"
    event = _audit(store, "account_disabled")[0]
    assert event.target.id == reviewer.account_id and event.details["revoked_session_count"] == 1
    enabled = admin.write("PATCH", f"/admin/accounts/{reviewer.account_id}", json={"status": "active", "reason": "back"})
    assert enabled.status_code == 200 and enabled.json()["status"] == "active"
    assert reviewer.get("/auth/session").status_code == 401  # the old session stays revoked
    assert reviewer.sign_in().status_code == 200


def test_the_last_admin_cannot_be_demoted_disabled_or_reinvited(app, admin):
    for body in ({"role": "supervisor", "reason": "x"}, {"status": "disabled", "reason": "x"}):
        refused = admin.write("PATCH", f"/admin/accounts/{admin.account_id}", json=body)
        assert refused.status_code == 409 and _reason(refused) == "last_admin"
    reinvite = admin.write("POST", "/admin/invitations", json={"email": admin.email, "display_name": "A", "role": "admin",
                                                              "reinvite_of_account_id": admin.account_id})
    assert reinvite.status_code == 409 and _reason(reinvite) == "last_admin"
    second = invite_and_enroll(app, admin, "admin2@example.com", role="admin")
    assert admin.write("PATCH", f"/admin/accounts/{second.account_id}", json={"role": "reviewer", "reason": "x"}).status_code == 200
    assert admin.write("PATCH", f"/admin/accounts/{second.account_id}", json={"role": "admin", "reason": "x"}).status_code == 200
    assert second.write("PATCH", f"/admin/accounts/{admin.account_id}", json={"role": "reviewer", "reason": "handover"}).status_code == 200


# --- lost keys and re-invites -----------------------------------------------------------------


def test_revoking_the_last_authenticator_requires_a_reinvite(app, admin, store):
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    keys = admin.get(f"/admin/accounts/{reviewer.account_id}/authenticators").json()["items"]
    assert len(keys) == 1
    assert admin.write("DELETE", f"/admin/accounts/{reviewer.account_id}/authenticators/cred_missing").status_code == 404
    assert admin.write("DELETE", f"/admin/accounts/{reviewer.account_id}/authenticators/{keys[0]['id']}").status_code == 204
    account = admin.get(f"/admin/accounts/{reviewer.account_id}").json()
    assert account["status"] == "reinvite_required" and account["authenticator_count"] == 0
    assert reviewer.get("/auth/session").status_code == 401
    refused = admin.write("PATCH", f"/admin/accounts/{reviewer.account_id}", json={"status": "active", "reason": "x"})
    assert refused.status_code == 409
    event = _audit(store, "authenticator_removed")[0]
    assert event.target.id == keys[0]["id"] and event.details["status"] == "reinvite_required"


def test_a_reinvite_revokes_credentials_and_sessions_then_reenrolls_the_same_account(app, admin, store):
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    old_key = next(iter(reviewer.authn.credentials.values()))
    wrong_email = admin.write("POST", "/admin/invitations", json={"email": "other@example.com", "display_name": "R", "role": "reviewer",
                                                                 "reinvite_of_account_id": reviewer.account_id})
    assert wrong_email.status_code == 422 and _reason(wrong_email) == "reinvite_email_mismatch"
    token = invite(admin, "rev@example.com", "supervisor", "Rev Again", reinvite_of_account_id=reviewer.account_id)
    assert reviewer.get("/auth/session").status_code == 401
    assert admin.get(f"/admin/accounts/{reviewer.account_id}").json()["status"] == "reinvite_required"
    assert sign_in(new_client(app), reviewer.authn, reviewer.email, credential=old_key).status_code == 401
    fresh = Person(new_client(app), SoftAuthenticator(), reviewer.email)
    enrolled = enroll(fresh.client, fresh.authn, invitation_token=token)
    assert enrolled.status_code == 200, enrolled.text
    account = enrolled.json()["account"]
    assert account["id"] == reviewer.account_id and account["status"] == "active" and account["role"] == "supervisor"
    assert account["authenticator_count"] == 1 and account["display_name"] == "Rev Again"
    assert sign_in(new_client(app), reviewer.authn, reviewer.email, credential=old_key).status_code == 401
    assert fresh.sign_in().status_code == 200
    reinvite = _audit(store, "reinvite_issued")[0]
    assert reinvite.details["revoked_credential_count"] == 1 and reinvite.details["revoked_session_count"] == 1


def test_admins_revoke_every_session_of_an_account(app, admin, store):
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    phone = Person(new_client(app), reviewer.authn, reviewer.email)
    phone.sign_in()
    assert admin.write("DELETE", f"/admin/accounts/{reviewer.account_id}/sessions").status_code == 204
    assert reviewer.get("/auth/session").status_code == 401 and phone.get("/auth/session").status_code == 401
    assert admin.write("DELETE", "/admin/accounts/acct_missing/sessions").status_code == 404
    assert _audit(store, "session_revoked")[0].details["revoked_session_count"] == 2


# --- invitations ------------------------------------------------------------------------------


def test_invitations_are_single_pending_revocable_and_expire(app, admin, store, clock):
    issued = admin.write("POST", "/admin/invitations", json={"email": "new@example.com", "display_name": "New", "role": "reviewer"})
    assert issued.status_code == 200
    body = issued.json()
    assert body["invitation_url"].startswith("http://localhost:8010/enroll#")
    token = body["invitation_url"].split("#", 1)[1]
    assert body["invitation"]["token_hash"].startswith("sha256:") and token not in issued.text.replace(body["invitation_url"], "")
    assert body["invitation"]["issued_by_account_id"] == admin.account_id and body["invitation"]["status"] == "pending"
    duplicate = admin.write("POST", "/admin/invitations", json={"email": "NEW@example.com", "display_name": "New", "role": "reviewer"})
    assert duplicate.status_code == 409 and _reason(duplicate) == "invitation_pending"
    existing = admin.write("POST", "/admin/invitations", json={"email": admin.email, "display_name": "A", "role": "reviewer"})
    assert existing.status_code == 409 and _reason(existing) == "account_exists"
    relay = admin.write("POST", "/admin/invitations", json={"email": "x@example.com", "display_name": "X", "role": "reviewer", "delivery": "smtp_relay"})
    assert relay.status_code == 409 and _reason(relay) == "smtp_relay_not_configured"
    listed = admin.get("/admin/invitations", params={"status": "pending"}).json()["items"]
    assert [i["id"] for i in listed] == [body["invitation"]["id"]] and "invitation_url" not in listed[0]
    assert admin.write("DELETE", f"/admin/invitations/{body['invitation']['id']}").status_code == 204
    revoked_use = new_client(app).post(f"{API}/auth/enroll/begin", json={"invitation_token": token})
    assert revoked_use.status_code == 410 and _reason(revoked_use) == "revoked"
    again = admin.write("DELETE", f"/admin/invitations/{body['invitation']['id']}")
    assert again.status_code == 410 and _code(again) == "invitation_invalid"
    assert admin.write("DELETE", "/admin/invitations/inv_missing").status_code == 404
    late = invite(admin, "late@example.com")
    clock.advance(seconds=store.config.parameters.invitation_lifetime_seconds + 1)
    assert admin.sign_in().status_code == 200  # the admin's own session ran out meanwhile
    expired = new_client(app).post(f"{API}/auth/enroll/begin", json={"invitation_token": late})
    assert expired.status_code == 410 and _reason(expired) == "expired"
    assert [i["email"] for i in admin.get("/admin/invitations", params={"status": "expired"}).json()["items"]] == ["late@example.com"]
    assert len(admin.get("/admin/invitations").json()["items"]) == 2
    assert [e.action.value for e in _audit(store, "invitation_revoked")] == ["invitation_revoked"]


# --- setup codes and break-glass --------------------------------------------------------------


def test_break_glass_reenrolls_a_named_account_and_is_listed(app, admin, store):
    demoted = invite_and_enroll(app, admin, "lead@example.com", role="supervisor")
    old_key = next(iter(demoted.authn.credentials.values()))
    request = SetupCodeIssueRequest(purpose="break_glass", email="lead@example.com", display_name="Lead (recovered)",
                                    target_account_id=demoted.account_id, os_user="root")
    code, record = auth_cli.issue_setup_code(store, request)
    assert record.target_account_id == demoted.account_id and record.used_at is None
    with pytest.raises(auth_cli.HostCommandRefused) as mismatch:
        auth_cli.issue_setup_code(store, request.model_copy(update={"email": "someone@example.com"}))
    assert mismatch.value.details["reason"] == "target_email_mismatch"
    assert demoted.get("/auth/session").status_code == 200  # nothing is revoked at issue
    fresh = Person(new_client(app), SoftAuthenticator(), "lead@example.com")
    enrolled = enroll(fresh.client, fresh.authn, setup_code=code)
    assert enrolled.status_code == 200, enrolled.text
    account = enrolled.json()["account"]
    assert account["id"] == demoted.account_id and account["role"] == "admin" and account["authenticator_count"] == 1
    assert demoted.get("/auth/session").status_code == 401
    assert sign_in(new_client(app), demoted.authn, demoted.email, credential=old_key).status_code == 401
    used = _audit(store, "break_glass_used")
    assert len(used) == 1 and used[0].actor.kind.value == "break_glass" and used[0].details["revoked_credential_count"] == 1
    listing = admin.get("/admin/break-glass").json()["items"]
    assert len(listing) == 1 and listing[0]["revoked_credential_count"] == 1 and listing[0]["audit_event_id"] == used[0].id
    assert listing[0]["enrolled_account_id"] == demoted.account_id and listing[0]["os_user"] == "root"
    codes = admin.get("/admin/setup-codes").json()["items"]
    assert [c["purpose"] for c in codes] == ["break_glass", "first_admin"] and all(c["used_at"] for c in codes)
    assert all(c["code_hash"].startswith("sha256:") for c in codes) and code.replace("-", "") not in str(codes)
    issued = _audit(store, "setup_code_issued")
    assert {e.actor.kind.value for e in issued} == {"installer", "break_glass"}
    assert codes[0]["audit_event_id"] == issued[-1].id


def test_break_glass_without_a_target_creates_a_new_admin(app, admin, store):
    code, _ = auth_cli.issue_setup_code(store, SetupCodeIssueRequest(purpose="break_glass", email="it@example.com", display_name="IT", os_user="root"))
    enrolled = enroll(new_client(app), SoftAuthenticator(), setup_code=code)
    assert enrolled.status_code == 200 and enrolled.json()["account"]["role"] == "admin"
    assert enrolled.json()["account"]["id"] != admin.account_id
    with pytest.raises(auth_cli.HostCommandRefused):
        auth_cli.issue_setup_code(store, SetupCodeIssueRequest(purpose="break_glass", email="it@example.com", display_name="IT", os_user="root"))


# --- installations and service keys -----------------------------------------------------------


def test_installations_have_at_most_one_primary_host(app, admin, store):
    first = admin.write("POST", "/admin/installations", json={"label": "mac-mini", "primary_host": True})
    assert first.status_code == 201 and first.json()["primary_host"] is True and first.json()["created_by_account_id"] == admin.account_id
    second = admin.write("POST", "/admin/installations", json={"label": "studio", "primary_host": True})
    assert second.status_code == 409 and _reason(second) == "primary_host_taken"
    same_label = admin.write("POST", "/admin/installations", json={"label": "mac-mini"})
    assert same_label.status_code == 409 and _reason(same_label) == "label_in_use"
    helper = admin.write("POST", "/admin/installations", json={"label": "studio"}).json()
    assert helper["primary_host"] is False
    retired = admin.write("POST", f"/admin/installations/{first.json()['id']}/retire", json={"reason": "replaced"})
    assert retired.status_code == 200 and retired.json()["retired_at"]
    twice = admin.write("POST", f"/admin/installations/{first.json()['id']}/retire", json={"reason": "again"})
    assert twice.status_code == 409 and _code(twice) == "invalid_transition"
    assert admin.write("POST", "/admin/installations/inst_missing/retire", json={"reason": "x"}).status_code == 404
    assert admin.write("POST", "/admin/installations", json={"label": "new-primary", "primary_host": True}).status_code == 201
    assert sorted(i["label"] for i in admin.get("/admin/installations").json()["items"]) == ["mac-mini", "new-primary", "studio"]
    with store.connection() as conn:
        assert auth_api.get_installation(conn, helper["id"]).label == "studio"


def _key(admin, installation_id, scopes, **extra):
    response = admin.write("POST", "/admin/service-keys", json={"installation_id": installation_id, "label": "worker", "scopes": scopes, **extra})
    assert response.status_code == 201, response.text
    return response.json()


def test_service_keys_are_shown_once_scoped_and_bound_to_their_installation(app, admin, client, store):
    installation = admin.write("POST", "/admin/installations", json={"label": "mac-mini", "primary_host": True}).json()
    issued = _key(admin, installation["id"], ["admin-state:read", "changes:read"])
    token = issued["token"]
    assert token.startswith(issued["key"]["key_prefix"] + "_") and issued["recommended_env_name"] == "CALL1_STORE_SERVICE_KEY"
    listed = admin.get("/admin/service-keys").json()["items"]
    assert len(listed) == 1 and token not in str(listed) and listed[0]["key_hash"].startswith("sha256:")
    headers = {"Authorization": f"Bearer {token}"}
    assert client.get(f"{API}/status/detail", headers=headers).status_code == 200
    narrow = _key(admin, installation["id"], ["jobs:claim"])
    lacking = client.get(f"{API}/status/detail", headers={"Authorization": f"Bearer {narrow['token']}"})
    assert lacking.status_code == 403 and _code(lacking) == "insufficient_scope"
    on_admin_route = client.get(f"{API}/admin/accounts", headers=headers)
    assert on_admin_route.status_code == 403 and _code(on_admin_route) == "forbidden"
    with store.connection() as conn:
        principal = store.auth.authenticate_service_key(conn, token, store.clock.now())
    assert principal.installation_id == installation["id"] and principal.primary_host is True
    assert principal.scopes == frozenset({ServiceScope.ADMIN_STATE_READ, ServiceScope.CHANGES_READ})
    assert admin.write("POST", "/admin/service-keys", json={"installation_id": "inst_missing", "label": "x", "scopes": ["jobs:claim"]}).status_code == 404
    past = admin.write("POST", "/admin/service-keys", json={"installation_id": installation["id"], "label": "x", "scopes": ["jobs:claim"],
                                                          "expires_at": "2020-01-01T00:00:00Z"})
    assert past.status_code == 422
    assert [e.details["key_prefix"] for e in _audit(store, "service_key_issued")] == [issued["key"]["key_prefix"], narrow["key"]["key_prefix"]]


def test_rotation_keeps_the_old_key_until_grace_then_rejects_it(app, admin, client, store, clock):
    installation = admin.write("POST", "/admin/installations", json={"label": "mac-mini"}).json()
    old = _key(admin, installation["id"], ["changes:read"])
    rotated = admin.write("POST", f"/admin/service-keys/{old['key']['id']}/rotate", json={"grace_seconds": 600})
    assert rotated.status_code == 200, rotated.text
    new = rotated.json()
    assert new["key"]["rotated_from_key_id"] == old["key"]["id"] and new["key"]["scopes"] == ["changes:read"]
    assert new["key"]["installation_id"] == installation["id"] and new["token"] != old["token"]
    record = {k["id"]: k for k in admin.get("/admin/service-keys").json()["items"]}[old["key"]["id"]]
    assert record["superseded_by_key_id"] == new["key"]["id"] and record["grace_until"]
    for token in (old["token"], new["token"]):
        assert client.get(f"{API}/changes", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    clock.advance(seconds=601)
    assert _code(client.get(f"{API}/changes", headers={"Authorization": f"Bearer {old['token']}"})) == "unauthenticated"
    assert client.get(f"{API}/changes", headers={"Authorization": f"Bearer {new['token']}"}).status_code == 200
    again = admin.write("POST", f"/admin/service-keys/{old['key']['id']}/rotate", json={})
    assert again.status_code == 409 and _reason(again) == "already_rotated"
    immediate = admin.write("POST", f"/admin/service-keys/{new['key']['id']}/rotate", json={"grace_seconds": 0}).json()
    assert _code(client.get(f"{API}/changes", headers={"Authorization": f"Bearer {new['token']}"})) == "unauthenticated"
    assert client.get(f"{API}/changes", headers={"Authorization": f"Bearer {immediate['token']}"}).status_code == 200
    assert len(_audit(store, "service_key_rotated")) == 2
    assert admin.write("POST", "/admin/service-keys/key_missing/rotate", json={}).status_code == 404


def test_revoked_expired_and_retired_keys_are_rejected(app, admin, client, store, clock):
    installation = admin.write("POST", "/admin/installations", json={"label": "mac-mini"}).json()
    revoked = _key(admin, installation["id"], ["changes:read"])
    response = admin.write("POST", f"/admin/service-keys/{revoked['key']['id']}/revoke", json={"reason": "leaked"})
    assert response.status_code == 200 and response.json()["revoked_at"]
    rejected = client.get(f"{API}/changes", headers={"Authorization": f"Bearer {revoked['token']}"})
    assert rejected.status_code == 401 and _code(rejected) == "unauthenticated" and rejected.headers["WWW-Authenticate"] == "Bearer"
    assert admin.write("POST", f"/admin/service-keys/{revoked['key']['id']}/revoke", json={"reason": "again"}).status_code == 200
    assert len(_audit(store, "service_key_revoked")) == 1
    cannot_rotate = admin.write("POST", f"/admin/service-keys/{revoked['key']['id']}/rotate", json={})
    assert cannot_rotate.status_code == 409 and _reason(cannot_rotate) == "key_not_active"
    expires = (store.clock.now() + timedelta(hours=1)).isoformat()
    expiring = _key(admin, installation["id"], ["changes:read"], expires_at=expires)
    assert client.get(f"{API}/changes", headers={"Authorization": f"Bearer {expiring['token']}"}).status_code == 200
    clock.advance(hours=1)
    assert _code(client.get(f"{API}/changes", headers={"Authorization": f"Bearer {expiring['token']}"})) == "unauthenticated"
    live = _key(admin, installation["id"], ["changes:read"])
    admin.write("POST", f"/admin/installations/{installation['id']}/retire", json={"reason": "decommissioned"})
    assert _code(client.get(f"{API}/changes", headers={"Authorization": f"Bearer {live['token']}"})) == "unauthenticated"
    keys = {k["id"]: k for k in admin.get("/admin/service-keys").json()["items"]}
    assert keys[live["key"]["id"]]["revoked_at"]
    assert admin.write("POST", "/admin/service-keys", json={"installation_id": installation["id"], "label": "x", "scopes": ["jobs:claim"]}).status_code == 404
    assert _audit(store, "installation_retired")[0].details["revoked_key_count"] == 2


def test_minted_test_keys_use_the_real_lookup(mint_service_key, store, conn):
    key = mint_service_key([ServiceScope.JOBS_CLAIM], label="fixture")
    principal = store.auth.authenticate_service_key(conn, key.token, store.clock.now())
    assert principal == key.principal
    reused = mint_service_key([ServiceScope.JOBS_WRITE], installation_id=key.installation_id, primary_host=False)
    assert reused.installation_id == key.installation_id and reused.principal.primary_host is True


def test_every_identity_action_is_in_one_hash_chain(app, admin, store):
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    admin.write("PATCH", f"/admin/accounts/{reviewer.account_id}", json={"role": "supervisor", "reason": "x"})
    installation = admin.write("POST", "/admin/installations", json={"label": "m"}).json()
    key = _key(admin, installation["id"], ["jobs:claim"])
    admin.write("POST", f"/admin/service-keys/{key['key']['id']}/rotate", json={})
    feed = admin.get("/admin/audit", params={"limit": 200}).json()["items"]
    actions = [e["action"] for e in reversed(feed)]
    assert actions == ["setup_code_issued", "account_created", "setup_code_redeemed", "authenticator_added", "invitation_issued",
                       "account_created", "invitation_redeemed", "authenticator_added", "account_updated", "installation_registered",
                       "service_key_issued", "service_key_rotated"]
    with store.connection() as conn:
        assert audit.verify_chain(conn)
    assert all(key["token"] not in str(e) for e in feed)
