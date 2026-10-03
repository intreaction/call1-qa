"""Auth area, admin identity and Process keys, end to end against real Store and Process processes.

Feature IDs from the auth inventory:
    a6  setup codes and break-glass admin recovery (host commands, auditable listing)
    a7  installations and service keys: issue-service-key scopes, primary-host exclusivity,
        rotation, revocation, and the jobs:control scope (Store and Process console)
    a9  the admin API behind Evaluate's admin area: accounts, invitations, installations, keys
"""

from __future__ import annotations

import httpx
import pytest

from .auth_support import audit_events, code_of, describe, details_of, only_credential, sign_in_with
from .stack import unique_email

pytestmark = pytest.mark.e2e


def _key_client(stack, token: str) -> httpx.Client:
    return httpx.Client(base_url=stack.store_url, timeout=30.0, headers={"Authorization": f"Bearer {token}"})


# --- a6 ---------------------------------------------------------------------------------------


def test_a6_first_admin_code_is_refused_while_an_admin_exists(stack, admin_session):
    email = unique_email("a6-second-first")
    proc = stack.store_cli("setup-code", "--email", email, "--display-name", "Should Not", "--purpose", "first_admin", check=False)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "configuration:" in proc.stderr and "admin" in proc.stderr.lower(), proc.stderr
    assert "Setup code for" not in proc.stdout, proc.stdout
    codes = admin_session.get("/admin/setup-codes").json()["items"]
    assert not any(c["email"] == email for c in codes), "a refused setup-code must not leave a record"


def test_a6_break_glass_without_target_creates_a_new_admin_and_is_audited(stack, admin_session):
    email = unique_email("a6-glass")
    code = stack.setup_code(email, "A6 Break Glass", purpose="break_glass")
    person = stack.new_session(email)
    body = person.enroll(setup_code=code).json()
    assert body["account"]["role"] == "admin" and body["account"]["id"] != admin_session.account_id, body

    records = admin_session.get("/admin/break-glass").json()["items"]
    record = next((r for r in records if r["enrolled_account_id"] == body["account"]["id"]), None)
    assert record is not None, records
    assert record["used_at"] and record["os_user"] and record["target_account_id"] is None and record["revoked_credential_count"] == 0

    events = audit_events(stack, admin_session, action="break_glass_used")
    event = next((e for e in events if e["id"] == record["audit_event_id"]), None)
    assert event is not None, f"break-glass record's audit_event_id {record['audit_event_id']} is not a break_glass_used event"
    assert event["actor"]["kind"] == "break_glass", event
    assert event["target"]["id"] == record["setup_code_id"], event["target"]

    # Only admins read the listing.
    reviewer = stack.user("reviewer")
    refused = reviewer.get("/admin/break-glass")
    assert refused.status_code == 403 and code_of(refused) == "insufficient_role", describe(refused)


def test_a6_break_glass_on_an_account_reenrolls_it_and_revokes_old_credentials(stack, admin_session):
    target = stack.user("reviewer", cached=False)
    old_authenticator = target.authenticator
    assert target.get("/calls").status_code == 200

    code = stack.setup_code(target.email, "A6 Recovered", purpose="break_glass", target_account_id=target.account_id)
    # Redeeming the code, not issuing it, revokes the old credentials.
    assert target.get("/calls").status_code == 200

    rescued = stack.new_session(target.email)
    body = rescued.enroll(setup_code=code).json()
    assert body["account"]["id"] == target.account_id and body["account"]["role"] == "admin", body
    assert body["account"]["authenticator_count"] == 1

    assert target.get("/calls").status_code == 401, "the account's old sessions end at redemption"
    stale = sign_in_with(stack, target.email, old_authenticator, credential=only_credential(old_authenticator))
    assert stale.status_code == 401 and code_of(stale) == "webauthn_verification_failed", describe(stale)

    records = admin_session.get("/admin/break-glass").json()["items"]
    record = next(r for r in records if r["target_account_id"] == target.account_id)
    assert record["revoked_credential_count"] == 1 and record["enrolled_account_id"] == target.account_id, record

    # The code is single-use.
    again = stack.new_session(target.email).enroll(setup_code=code, check=False)
    assert again.status_code == 410 and code_of(again) == "setup_code_invalid", describe(again)


# --- a7 ---------------------------------------------------------------------------------------


def test_a7_issue_service_key_refuses_a_second_primary_host(stack, admin_session):
    label = "a7-second-primary"
    proc = stack.store_cli("issue-service-key", "--installation", label, "--primary-host", "--print-token", check=False)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "primary" in proc.stderr.lower(), proc.stderr
    assert "c1sk_" not in proc.stdout, "no token may be printed on refusal"
    installations = admin_session.get("/admin/installations").json()["items"]
    assert not any(i["label"] == label for i in installations), "the refused command must not register an installation"

    # The API refuses the same thing.
    api = admin_session.post("/admin/installations", json={"label": "a7-api-primary", "primary_host": True})
    assert api.status_code == 409 and code_of(api) == "conflict", describe(api)


def test_a7_default_scope_key_cannot_retry_or_cancel_jobs(stack, admin_session):
    proc = stack.store_cli("issue-service-key", "--installation", "a7-default-scopes", "--no-primary-host", "--print-token")
    token = proc.stdout.strip().splitlines()[-1]
    assert token.startswith("c1sk_"), proc.stdout

    keys = admin_session.get("/admin/service-keys").json()["items"]
    record = next(k for k in keys if token.startswith(k["key_prefix"]))
    assert "jobs:control" not in record["scopes"] and "jobs:claim" in record["scopes"], record["scopes"]
    assert token not in str(keys), "the token is never listed"

    receipt = stack.ingest("call_01_compliant", agent_id="a7-scope")
    stack.wait_until_settled(receipt["call_id"])
    jobs = stack.store_get("/jobs", session="service", params={"conversation_id": receipt["conversation_id"], "limit": 5}).json()["items"]
    job_id = jobs[0]["id"]

    default_key = _key_client(stack, token)
    assert default_key.get(f"/store/v1/conversations/{receipt['conversation_id']}/progress").status_code == 200
    for action, body in (("retry", {"reason": "e2e a7"}), ("cancel", {"reason": "e2e a7", "cascade": False})):
        refused = default_key.post(f"/store/v1/jobs/{job_id}/{action}", json=body)
        assert refused.status_code == 403 and code_of(refused) == "insufficient_scope", describe(refused)
        assert "jobs:control" in details_of(refused).get("required_scope", ""), refused.text

    # The launcher's key (with jobs:control) gets past the scope check: a settled job is not retryable.
    allowed = stack.store_post(f"/jobs/{job_id}/retry", json={"reason": "e2e a7"}, session="service")
    assert allowed.status_code != 403, describe(allowed)


def test_a7_process_console_explains_a_key_without_jobs_control(stack_factory):
    private = stack_factory(name="a7-console-scope")
    receipt = private.ingest("call_01_compliant", agent_id="a7-console")
    private.wait_until_settled(receipt["call_id"])
    job_id = private.store_get("/jobs", session="service", params={"conversation_id": receipt["conversation_id"], "limit": 1}).json()["items"][0]["id"]

    admin = private.admin()
    default_scopes = ["calls:write", "artifacts:read", "artifacts:write", "jobs:write", "jobs:claim", "reanalysis:claim", "changes:read",
                      "hardware:write", "catalog:publish", "usage:read", "admin-state:read"]
    issued = admin.post("/admin/service-keys", json={"installation_id": private.installation_id, "label": "a7 default scopes",
                                                     "scopes": default_scopes})
    assert issued.status_code == 201, describe(issued)
    private.stop_process()
    private._merge_process_config({"service_key": issued.json()["token"], "service_key_id": issued.json()["key"]["id"]})
    private.start_process()

    for action, body in (("retry", {"reason": "e2e a7"}), ("cancel", {"reason": "e2e a7", "cascade": False})):
        response = private.process_post(f"/jobs/{job_id}/{action}", json=body)
        assert response.status_code == 403 and code_of(response) == "insufficient_scope", describe(response)
        message = response.json()["message"]
        assert "jobs:control" in message and "issue-service-key" in message, message
    assert private.process_get("/health", token=False).json()["state"] == "running"


def test_a7_rotation_revocation_and_retirement(stack, admin_session):
    installation = admin_session.post("/admin/installations", json={"label": unique_email("a7-inst").split("@")[0], "primary_host": False})
    assert installation.status_code == 201, describe(installation)
    installation_id = installation.json()["id"]
    assert installation.json()["primary_host"] is False

    created = admin_session.post("/admin/service-keys", json={"installation_id": installation_id, "label": "a7 key",
                                                              "scopes": ["changes:read", "admin-state:read"]})
    assert created.status_code == 201, describe(created)
    first = created.json()
    assert _key_client(stack, first["token"]).get("/store/v1/status/detail").status_code == 200
    # Scopes are exact: this key cannot claim jobs.
    claim = _key_client(stack, first["token"]).post("/store/v1/jobs/claim", json={})
    assert claim.status_code == 403 and code_of(claim) == "insufficient_scope", describe(claim)

    # Rotation with the default grace keeps both keys working, with the same scopes.
    rotated = admin_session.post(f"/admin/service-keys/{first['key']['id']}/rotate", json={})
    assert rotated.status_code == 200, describe(rotated)
    second = rotated.json()
    assert sorted(second["key"]["scopes"]) == sorted(first["key"]["scopes"])
    assert second["key"]["installation_id"] == installation_id and second["key"]["rotated_from_key_id"] == first["key"]["id"]
    assert _key_client(stack, first["token"]).get("/store/v1/status/detail").status_code == 200
    assert _key_client(stack, second["token"]).get("/store/v1/status/detail").status_code == 200

    # grace_seconds 0 revokes the old key at once.
    third = admin_session.post(f"/admin/service-keys/{second['key']['id']}/rotate", json={"grace_seconds": 0}).json()
    assert _key_client(stack, second["token"]).get("/store/v1/status/detail").status_code == 401
    assert _key_client(stack, third["token"]).get("/store/v1/status/detail").status_code == 200

    revoked = admin_session.post(f"/admin/service-keys/{third['key']['id']}/revoke", json={"reason": "e2e a7"})
    assert revoked.status_code == 200 and revoked.json()["revoked_at"], describe(revoked)
    assert _key_client(stack, third["token"]).get("/store/v1/status/detail").status_code == 401

    # Retiring the installation revokes every key it still holds (the first key is in its grace period).
    retired = admin_session.post(f"/admin/installations/{installation_id}/retire", json={"reason": "e2e a7"})
    assert retired.status_code == 200 and retired.json()["retired_at"], describe(retired)
    assert _key_client(stack, first["token"]).get("/store/v1/status/detail").status_code == 401

    # The stack's own key was never touched.
    assert stack.store_get("/status/detail", session="service").status_code == 200


# --- a9 ---------------------------------------------------------------------------------------


def test_a9_admin_invitations_out_of_band_list_revoke(stack, admin_session):
    email = unique_email("a9-invite")
    smtp = admin_session.post("/admin/invitations", json={"email": email, "display_name": "A9", "role": "reviewer", "delivery": "smtp_relay"})
    assert smtp.status_code == 409 and details_of(smtp).get("reason") == "smtp_relay_not_configured", describe(smtp)

    issued = admin_session.post("/admin/invitations", json={"email": email, "display_name": "A9", "role": "supervisor"})
    assert issued.status_code == 200, describe(issued)
    invitation = issued.json()["invitation"]
    assert invitation["delivery"] == "out_of_band" and invitation["status"] == "pending" and invitation["role"] == "supervisor"
    assert invitation["issued_by_account_id"] == admin_session.account_id
    token = issued.json()["invitation_url"].split("#", 1)[1]
    assert token not in str(invitation), "the token is never stored or listed, only its hash"

    duplicate = admin_session.post("/admin/invitations", json={"email": email, "display_name": "A9", "role": "reviewer"})
    assert duplicate.status_code == 409 and details_of(duplicate).get("reason") == "invitation_pending", describe(duplicate)

    pending = admin_session.get("/admin/invitations", params={"status": "pending", "limit": 200}).json()["items"]
    assert any(i["id"] == invitation["id"] for i in pending)

    revoke = admin_session.delete(f"/admin/invitations/{invitation['id']}")
    assert revoke.status_code == 204, describe(revoke)
    revoked = admin_session.get("/admin/invitations", params={"status": "revoked", "limit": 200}).json()["items"]
    assert any(i["id"] == invitation["id"] for i in revoked)
    dead = stack.new_session(email).enroll(invitation_token=token, check=False)
    assert dead.status_code == 410 and code_of(dead) == "invitation_invalid", describe(dead)


def test_a9_admin_accounts_authenticators_sessions_and_reinvite(stack, admin_session):
    person = stack.user("reviewer", cached=False)
    accounts = admin_session.get("/admin/accounts", params={"limit": 200, "role": "reviewer"}).json()["items"]
    row = next((a for a in accounts if a["id"] == person.account_id), None)
    assert row is not None and row["status"] == "active" and row["authenticator_count"] == 1 and row["email"] == person.email, accounts[:2]
    assert admin_session.get(f"/admin/accounts/{person.account_id}").json()["email"] == person.email

    creds = admin_session.get(f"/admin/accounts/{person.account_id}/authenticators").json()["items"]
    assert [c["credential_id"] for c in creds] == [only_credential(person.authenticator).id]

    # Revoke every session of the account.
    revoke_sessions = admin_session.delete(f"/admin/accounts/{person.account_id}/sessions")
    assert revoke_sessions.status_code == 204, describe(revoke_sessions)
    assert person.get("/calls").status_code == 401
    person.sign_in()

    # Revoking the last authenticator makes the account reinvite_required and signs it out.
    revoke_key = admin_session.delete(f"/admin/accounts/{person.account_id}/authenticators/{creds[0]['id']}")
    assert revoke_key.status_code == 204, describe(revoke_key)
    assert admin_session.get(f"/admin/accounts/{person.account_id}").json()["status"] == "reinvite_required"
    assert person.get("/calls").status_code == 401

    # A re-invite restores it under the same account ID with a new authenticator.
    reinvite = admin_session.post("/admin/invitations", json={"email": person.email, "display_name": "A9 Reinvited", "role": "reviewer",
                                                              "reinvite_of_account_id": person.account_id})
    assert reinvite.status_code == 200, describe(reinvite)
    fresh = stack.new_session(person.email)
    body = fresh.enroll(invitation_token=reinvite.json()["invitation_url"].split("#", 1)[1]).json()
    assert body["account"]["id"] == person.account_id and body["account"]["status"] == "active"
    assert fresh.get("/calls").status_code == 200

    actions = {e["action"] for e in audit_events(stack, admin_session, target_id=person.account_id)}
    assert "session_revoked" in actions, actions


def test_a9_admin_installations_and_keys_listing(stack, admin_session):
    installations = admin_session.get("/admin/installations").json()["items"]
    mine = next((i for i in installations if i["id"] == stack.installation_id), None)
    assert mine is not None and mine["primary_host"] is True and not mine.get("retired_at"), installations
    keys = admin_session.get("/admin/service-keys").json()["items"]
    key = next((k for k in keys if k["id"] == stack.service_key_id), None)
    assert key is not None and key["installation_id"] == stack.installation_id and "jobs:control" in key["scopes"], keys
    assert stack.service_key.startswith(key["key_prefix"])
    assert stack.service_key not in str(keys), "service-key tokens are never listed"
