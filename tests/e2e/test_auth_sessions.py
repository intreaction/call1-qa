"""Auth area, sessions and roles, end to end against real Store server processes.

Feature IDs from the auth inventory:
    a4  server-side sessions, the dev-mode cookie, CSRF on writes (and its recovery from GET /auth/session)
    a5  roles and permissions enforced by Store whatever the UI shows
    a8  your own authenticators (step-up add, rename, remove) and sessions (list, revoke)
"""

from __future__ import annotations

import pytest

from .auth_support import add_authenticator, clone_cookie_client, code_of, describe, details_of, only_credential, sign_in_with
from .softauthn import SoftAuthenticator
from .stack import unique_email

pytestmark = pytest.mark.e2e


# --- a4 ---------------------------------------------------------------------------------------


def test_a4_dev_mode_session_cookie_attributes(stack, admin_session):
    email = unique_email("a4-cookie")
    person = stack.new_session(email)
    finish = person.enroll(invitation_token=stack.invite(email, "reviewer"))
    set_cookie = finish.headers.get("set-cookie", "")
    assert set_cookie.startswith("call1_session="), f"dev mode names the cookie call1_session: {set_cookie!r}"
    assert "__Host-" not in set_cookie
    attributes = [part.strip().lower() for part in set_cookie.split(";")[1:]]
    assert "httponly" in attributes, set_cookie
    assert "samesite=strict" in attributes, set_cookie
    assert "path=/" in attributes, set_cookie
    assert not any(a.startswith("domain=") for a in attributes), set_cookie
    assert "secure" not in attributes, "dev mode (plain-HTTP localhost) drops Secure; see call1/store/README.md"

    cookie = person.cookies["call1_session"]
    session = finish.json()["signed_in"]["session"]
    assert cookie not in (session["session_id"], session["csrf_token"]), "the handle and CSRF token are never the cookie value"


def test_a4_writes_require_csrf_and_the_token_is_recoverable(stack, admin_session):
    person = stack.user("reviewer", cached=False)
    authenticator_id = person.get("/auth/authenticators").json()["items"][0]["id"]
    path = f"/auth/authenticators/{authenticator_id}"

    missing = person.patch(path, json={"nickname": "no csrf"}, csrf=False)
    assert missing.status_code == 403 and code_of(missing) == "csrf_failed", describe(missing)

    wrong = person.patch(path, json={"nickname": "wrong csrf"}, csrf=False, headers={"X-Call1-CSRF": "A" * 43})
    assert wrong.status_code == 403 and code_of(wrong) == "csrf_failed", describe(wrong)

    foreign = person.patch(path, json={"nickname": "foreign origin"}, headers={"Origin": "http://evil.example:9999"})
    assert foreign.status_code == 403 and code_of(foreign) == "origin_not_allowed", describe(foreign)

    # A reload (a new tab with only the cookie) recovers the same token from GET /auth/session.
    tab = clone_cookie_client(stack, person)
    recovered = tab.get("/store/v1/auth/session")
    assert recovered.status_code == 200, describe(recovered)
    token = recovered.json()["csrf_token"]
    assert token == person.csrf_token, "the CSRF token is session-bound and re-readable"
    renamed = tab.patch(f"/store/v1{path}", json={"nickname": "renamed after reload"},
                        headers={"X-Call1-CSRF": token, "Origin": stack.store_url})
    assert renamed.status_code == 200 and renamed.json()["nickname"] == "renamed after reload", describe(renamed)

    # Reads never need it.
    assert person.get("/auth/authenticators").status_code == 200


def test_a4_sign_out_ends_the_server_side_session(stack, admin_session):
    person = stack.user("reviewer", cached=False)
    tab = clone_cookie_client(stack, person)
    assert tab.get("/store/v1/calls").status_code == 200

    assert person.sign_out().status_code == 200
    replay = tab.get("/store/v1/calls")
    assert replay.status_code == 401, "a copied cookie must stop working once its session is signed out: " + describe(replay)
    assert person.get("/auth/session").status_code == 401

    # Signing in again makes a new session (a new handle and CSRF token).
    old_session_id = person.session.get("session_id")
    person.sign_in()
    assert person.session["session_id"] != old_session_id


# --- a5 ---------------------------------------------------------------------------------------

_SUPERVISOR_ONLY = [
    ("PUT", "/rubrics/rub_e2e_a5/draft", {"expected_draft_revision": None, "rubric": {}}),
    ("PUT", "/review-queue/rules/rule_e2e_a5", {"expected_version": None}),
    ("POST", "/calls/call_e2e_a5/escalation", {"expected_version": 0, "resolution": "dismissed", "note": "x"}),
    ("POST", "/review-queue/items/rqi_e2e_a5/assign", {"expected_version": 0, "assignee_account_id": None}),
    ("GET", "/metrics/review-agreement", None),
    ("GET", "/jobs", None),
]
_ADMIN_ONLY = [
    ("GET", "/admin/accounts", None),
    ("POST", "/admin/invitations", {"email": "a5-nobody@e2e.test", "display_name": "Nobody", "role": "admin"}),
    ("GET", "/admin/invitations", None),
    ("GET", "/admin/audit", None),
    ("GET", "/admin/setup-codes", None),
    ("GET", "/admin/break-glass", None),
    ("GET", "/admin/installations", None),
    ("POST", "/admin/installations", {"label": "a5-rogue", "primary_host": False}),
    ("GET", "/admin/service-keys", None),
    ("DELETE", "/admin/accounts/acct_e2e_a5/sessions", None),
]


def _call(person, method, path, body):
    if method == "GET":
        return person.get(path)
    return person.request(method, path, json=body)


def test_a5_reviewer_is_refused_supervisor_and_admin_actions(stack, admin_session):
    reviewer = stack.user("reviewer", cached=False)
    for method, path, body in _SUPERVISOR_ONLY + _ADMIN_ONLY:
        response = _call(reviewer, method, path, body)
        assert response.status_code == 403 and code_of(response) == "insufficient_role", f"reviewer: {describe(response)}"
    # And nothing was created by the refused admin write.
    invites = admin_session.get("/admin/invitations", params={"limit": 200}).json()["items"]
    assert not any(i["email"] == "a5-nobody@e2e.test" for i in invites)


def test_a5_supervisor_is_refused_admin_actions_but_not_supervisor_ones(stack, admin_session):
    supervisor = stack.user("supervisor", cached=False)
    for method, path, body in _ADMIN_ONLY:
        response = _call(supervisor, method, path, body)
        assert response.status_code == 403 and code_of(response) == "insufficient_role", f"supervisor: {describe(response)}"
    for path in ("/metrics/review-agreement", "/jobs", "/admin/reviewer-profiles"):
        response = supervisor.get(path)
        assert response.status_code == 200, f"supervisor: {describe(response)}"


def test_a5_role_change_applies_from_the_next_request(stack, admin_session):
    person = stack.user("supervisor", cached=False)
    assert person.get("/metrics/review-agreement").status_code == 200

    demote = admin_session.patch(f"/admin/accounts/{person.account_id}", json={"role": "reviewer", "reason": "e2e a5 demotion"})
    assert demote.status_code == 200 and demote.json()["role"] == "reviewer", describe(demote)

    refused = person.get("/metrics/review-agreement")
    assert refused.status_code == 403 and code_of(refused) == "insufficient_role", describe(refused)
    session = person.refresh()
    assert session["role"] == "reviewer" and "resolve_escalation" not in session["permissions"], session

    promote = admin_session.patch(f"/admin/accounts/{person.account_id}", json={"role": "supervisor", "reason": "e2e a5 promotion"})
    assert promote.status_code == 200
    assert person.get("/metrics/review-agreement").status_code == 200


def test_a5_disabled_account_loses_its_sessions_and_cannot_sign_in(stack, admin_session):
    person = stack.user("reviewer", cached=False)
    disable = admin_session.patch(f"/admin/accounts/{person.account_id}", json={"status": "disabled", "reason": "e2e a5 disable"})
    assert disable.status_code == 200 and disable.json()["status"] == "disabled", describe(disable)

    after = person.get("/calls")
    assert after.status_code in (401, 403) and code_of(after) in ("unauthenticated", "session_expired", "account_disabled"), describe(after)

    response = sign_in_with(stack, person.email, person.authenticator, credential=only_credential(person.authenticator))
    assert response.status_code in (401, 403), describe(response)

    enable = admin_session.patch(f"/admin/accounts/{person.account_id}", json={"status": "active", "reason": "e2e a5 enable"})
    assert enable.status_code == 200 and enable.json()["status"] == "active", describe(enable)
    person.sign_in()
    assert person.get("/calls").status_code == 200


def test_a5_account_disabled_between_begin_and_finish_gets_account_disabled(stack, admin_session):
    import httpx

    from .auth_support import begin_sign_in, finish_sign_in

    person = stack.user("reviewer", cached=False)
    client = httpx.Client(base_url=stack.store_url, timeout=30.0)
    begin = begin_sign_in(stack, person.email, client=client, forwarded_for="10.240.0.5")
    assert begin.status_code == 200
    disable = admin_session.patch(f"/admin/accounts/{person.account_id}", json={"status": "disabled", "reason": "e2e a5 race"})
    assert disable.status_code == 200
    assertion = person.authenticator.assert_(begin.json()["options"], origin=stack.store_url)
    finish = finish_sign_in(stack, client, begin.json()["ceremony_id"], assertion)
    assert finish.status_code == 403 and code_of(finish) == "account_disabled", describe(finish)


def test_a5_last_admin_cannot_demote_or_disable_themselves(stack_factory):
    private = stack_factory(name="a5-last-admin", with_process=False)
    admin = private.admin()
    demote = admin.patch(f"/admin/accounts/{admin.account_id}", json={"role": "reviewer", "reason": "e2e"})
    assert demote.status_code == 409 and details_of(demote).get("reason") == "last_admin", describe(demote)
    disable = admin.patch(f"/admin/accounts/{admin.account_id}", json={"status": "disabled", "reason": "e2e"})
    assert disable.status_code == 409 and details_of(disable).get("reason") == "last_admin", describe(disable)
    assert admin.refresh()["role"] == "admin"


# --- a8 ---------------------------------------------------------------------------------------


def test_a8_own_authenticators_step_up_add_and_second_authenticator_prompt(stack, admin_session):
    person = stack.user("reviewer", cached=False)
    assert person.refresh()["prompt_second_authenticator"] is True

    listed = person.get("/auth/authenticators").json()["items"]
    assert len(listed) == 1 and listed[0]["credential_id"] == only_credential(person.authenticator).id

    # The last authenticator cannot be removed.
    last = person.delete(f"/auth/authenticators/{listed[0]['id']}")
    assert last.status_code == 409 and details_of(last).get("reason") == "last_authenticator", describe(last)

    # Step-up needs a user-verified assertion from an existing authenticator.
    refused = add_authenticator(person, reauth_kw={"user_verified": False}, check=False)
    assert refused.status_code == 401 and code_of(refused) == "webauthn_verification_failed", describe(refused)

    added = add_authenticator(person, nickname="a8 backup key")
    backup = added.extensions["e2e_authenticator"]
    assert added.json()["credential"]["nickname"] == "a8 backup key"
    assert added.json()["signed_in"] is None, "adding an authenticator does not start a new session"

    assert person.refresh()["prompt_second_authenticator"] is False
    listed = person.get("/auth/authenticators").json()["items"]
    assert len(listed) == 2 and {c["nickname"] for c in listed} >= {"a8 backup key"}

    # The backup key alone signs in.
    via_backup = sign_in_with(stack, person.email, backup)
    assert via_backup.status_code == 200, describe(via_backup)
    backup_session = via_backup.json()["session"]
    assert backup_session["authenticator_id_used"] == added.json()["credential"]["id"]

    # Rename, then remove the backup: the session it signed in ends; this one does not.
    renamed = person.patch(f"/auth/authenticators/{added.json()['credential']['id']}", json={"nickname": "a8 spare"})
    assert renamed.status_code == 200 and renamed.json()["nickname"] == "a8 spare"
    backup_client = via_backup.extensions["e2e_client"]
    assert backup_client.get("/store/v1/calls").status_code == 200
    removed = person.delete(f"/auth/authenticators/{added.json()['credential']['id']}")
    assert removed.status_code == 204, describe(removed)
    assert backup_client.get("/store/v1/calls").status_code == 401, "removing an authenticator ends the sessions it signed in"
    assert person.get("/calls").status_code == 200
    assert person.refresh()["prompt_second_authenticator"] is True


def test_a8_own_sessions_list_and_revoke(stack, admin_session):
    person = stack.user("reviewer", cached=False)
    other = sign_in_with(stack, person.email, person.authenticator)
    assert other.status_code == 200, describe(other)
    other_id = other.json()["session"]["session_id"]
    other_client = other.extensions["e2e_client"]

    listed = person.get("/auth/sessions").json()["items"]
    by_id = {s["session_id"]: s for s in listed}
    assert by_id[person.session["session_id"]]["current"] is True
    assert by_id[other_id]["current"] is False
    assert all(s["account_id"] == person.account_id for s in listed)
    assert not any(person.cookies["call1_session"] in str(s) for s in listed), "cookie values are never listed"

    revoke = person.delete(f"/auth/sessions/{other_id}")
    assert revoke.status_code == 204, describe(revoke)
    assert other_client.get("/store/v1/calls").status_code == 401
    assert other_id not in {s["session_id"] for s in person.get("/auth/sessions").json()["items"]}

    # Another account's session is not yours to revoke.
    stranger = stack.user("reviewer", cached=False)
    foreign = person.delete(f"/auth/sessions/{stranger.session['session_id']}")
    assert foreign.status_code == 404, describe(foreign)
    assert stranger.get("/calls").status_code == 200


def test_a8_step_up_cannot_use_another_accounts_credential(stack, admin_session):
    person = stack.user("reviewer", cached=False)
    stranger = stack.user("reviewer", cached=False)
    begin = person.post("/auth/authenticators/begin", json={"nickname": "hijack"})
    assert begin.status_code == 200
    body = begin.json()
    reauth = stranger.authenticator.assert_(body["reauthentication"], origin=stack.store_url,
                                            credential=only_credential(stranger.authenticator))
    credential = SoftAuthenticator().register(body["options"], origin=stack.store_url)
    finish = person.post("/auth/authenticators/finish",
                         json={"ceremony_id": body["ceremony_id"], "credential": credential, "reauthentication": reauth})
    assert finish.status_code == 401 and code_of(finish) == "webauthn_verification_failed", describe(finish)
    assert len(person.get("/auth/authenticators").json()["items"]) == 1
