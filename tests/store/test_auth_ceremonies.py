"""Passkey ceremonies end to end: enrollment, account-first sign-in with decoys, step-up
add-authenticator, sessions, CSRF and rate limits, against a software authenticator and the real
``webauthn`` verification. No ceremony is minted; every request goes through the HTTP routes."""

from __future__ import annotations

import pytest

from call1.contracts.auth import AccountStatus
from call1.contracts.errors import ErrorCode
from call1.store import audit
from call1.store.app import create_app
from call1.store.auth import cli as auth_cli
from call1.store.auth.ratelimit import SIGN_IN_PER_CLIENT, SIGN_IN_PER_EMAIL
from call1.contracts.auth import SetupCodeIssueRequest

from .auth_flows import API, Person, bootstrap_admin, enroll, invite, invite_and_enroll, new_client, setup_code, sign_in
from .auth_softauthn import ORIGIN, SoftAuthenticator


def _code(response) -> str:
    return response.json()["code"]


def _reason(response) -> str:
    return response.json()["details"].get("reason")


def _actions(store) -> list:
    with store.connection() as conn:
        return [row["action"] for row in conn.execute("SELECT action FROM audit_events ORDER BY sequence")]


# --- enrollment -------------------------------------------------------------------------------


def test_first_admin_enrolls_with_a_setup_code_and_is_signed_in(app, store):
    client, authn = new_client(app), SoftAuthenticator()
    code = setup_code(store, "Admin@Example.com", "First Admin")
    begin = client.post(f"{API}/auth/enroll/begin", json={"setup_code": code.lower().replace("-", " ")})  # typed loosely
    assert begin.status_code == 200, begin.text
    options = begin.json()["options"]
    assert options["rp"] == {"id": "localhost", "name": "Call1 Store"}
    assert options["user"]["name"] == "Admin@Example.com" and options["user"]["id"] != "Admin@Example.com"
    assert options["authenticatorSelection"]["userVerification"] == "required" and options["attestation"] == "none"
    assert [p["alg"] for p in options["pubKeyCredParams"]] == [-7, -257] and options["timeout"] == 120_000
    finish = client.post(f"{API}/auth/enroll/finish", json={"ceremony_id": begin.json()["ceremony_id"],
                                                            "credential": authn.register(options), "nickname": "Blue key"})
    assert finish.status_code == 200, finish.text
    body = finish.json()
    assert body["account"]["role"] == "admin" and body["account"]["status"] == "active" and body["account"]["authenticator_count"] == 1
    credential = body["credential"]
    assert credential["attestation_format"] == "none" and credential["nickname"] == "Blue key"
    assert credential["transports"] == ["usb", "nfc"]  # the unknown browser transport is dropped
    assert credential["aaguid"] == "00000000-0000-0000-0000-000000000000"
    session = body["signed_in"]["session"]
    assert session["role"] == "admin" and session["prompt_second_authenticator"] is True
    assert session["authenticator_id_used"] == credential["id"]
    cookie = finish.headers["set-cookie"]
    assert cookie.startswith("call1_session=") and "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert "Path=/" in cookie and "Secure" not in cookie and "Domain" not in cookie  # dev-mode cookie (Store README)
    again = client.get(f"{API}/auth/session")
    assert again.status_code == 200 and again.json()["csrf_token"] == session["csrf_token"]
    reuse = enroll(new_client(app), SoftAuthenticator(), setup_code=code)
    assert reuse.status_code == 410 and _code(reuse) == "setup_code_invalid" and _reason(reuse) == "used"
    assert _actions(store) == ["setup_code_issued", "account_created", "setup_code_redeemed", "authenticator_added"]
    with store.connection() as conn:
        assert audit.verify_chain(conn)
        stored = conn.execute("SELECT cookie_hash, csrf_hash FROM auth_sessions").fetchone()
    assert stored["cookie_hash"].startswith("sha256:") and session["csrf_token"] not in stored["csrf_hash"]


def test_first_admin_codes_stop_once_an_admin_exists(app, store):
    early = setup_code(store, "second@example.com", "Second")  # issued before anyone enrolled
    bootstrap_admin(app)
    with pytest.raises(auth_cli.HostCommandRefused) as refused:
        setup_code(store, "third@example.com", "Third")
    assert refused.value.code is ErrorCode.CONFLICT and refused.value.details["reason"] == "active_admin_exists"
    late = enroll(new_client(app), SoftAuthenticator(), setup_code=early)
    assert late.status_code == 410 and _reason(late) == "active_admin_exists"


def test_setup_codes_expire(app, store, clock):
    code = setup_code(store, "admin@example.com")
    clock.advance(seconds=store.config.parameters.setup_code_lifetime_seconds + 1)
    response = enroll(new_client(app), SoftAuthenticator(), setup_code=code)
    assert response.status_code == 410 and _code(response) == "setup_code_invalid" and _reason(response) == "expired"
    unknown = new_client(app).post(f"{API}/auth/enroll/begin", json={"setup_code": "NOPE-NOPE-NOPE"})
    assert unknown.status_code == 410 and _reason(unknown) == "unknown"
    both = new_client(app).post(f"{API}/auth/enroll/begin", json={"setup_code": code, "invitation_token": "x" * 20})
    assert both.status_code == 422 and _code(both) == "validation_failed"


def test_invited_reviewer_enrolls_once_and_signs_in_again(app, store):
    admin = bootstrap_admin(app)
    token = invite(admin, "rev@example.com", "reviewer", "Rev One")
    reviewer = Person(new_client(app), SoftAuthenticator(), "rev@example.com")
    enrolled = enroll(reviewer.client, reviewer.authn, invitation_token=token)
    assert enrolled.status_code == 200 and enrolled.json()["account"]["display_name"] == "Rev One"
    assert enrolled.json()["signed_in"]["session"]["role"] == "reviewer"
    again = enroll(new_client(app), SoftAuthenticator(), invitation_token=token)
    assert again.status_code == 410 and _code(again) == "invitation_invalid" and _reason(again) == "redeemed"
    laptop = Person(new_client(app), reviewer.authn, "REV@example.com")  # a second browser, same key, email in any case
    signed = laptop.sign_in()
    assert signed.status_code == 200, signed.text
    assert signed.json()["session"]["account_id"] == enrolled.json()["account"]["id"]
    listed = admin.get("/admin/invitations", params={"status": "redeemed"}).json()["items"]
    assert [i["account_id"] for i in listed] == [enrolled.json()["account"]["id"]]
    with store.connection() as conn:
        count = conn.execute("SELECT sign_count, last_used_at FROM auth_credentials WHERE account_id = ?", (laptop.account_id,)).fetchone()
    assert count["sign_count"] == 1 and count["last_used_at"] is not None
    assert "invitation_redeemed" in _actions(store)


# --- sign-in: account-first, decoys, binding ---------------------------------------------------


def test_sign_in_never_reveals_whether_an_account_exists(app, store):
    admin = bootstrap_admin(app)
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    client = new_client(app)

    def begin(email):
        response = client.post(f"{API}/auth/sign-in/begin", json={"email": email})
        assert response.status_code == 200, response.text
        return response.json()

    real = begin("rev@example.com")
    assert [c["id"] for c in real["options"]["allowCredentials"]] == list(reviewer.authn.credentials)
    unknown, unknown_again = begin("nobody@example.com"), begin("Nobody@Example.com")
    assert unknown["options"]["allowCredentials"] == unknown_again["options"]["allowCredentials"]  # deterministic decoys
    assert 1 <= len(unknown["options"]["allowCredentials"]) <= 2
    assert set(unknown) == set(real) and set(unknown["options"]) == set(real["options"])
    for decoy in unknown["options"]["allowCredentials"]:
        assert set(decoy) == {"type", "id", "transports"} and decoy["transports"]
    other = begin("someone-else@example.com")["options"]["allowCredentials"]
    assert other != unknown["options"]["allowCredentials"]
    # a decoy ceremony never signs anyone in, whatever credential comes back
    decoy_begin = begin("nobody@example.com")
    forged = reviewer.authn.assert_(decoy_begin["options"], credential=next(iter(reviewer.authn.credentials.values())))
    finish = client.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": decoy_begin["ceremony_id"], "credential": forged})
    assert finish.status_code == 401 and _code(finish) == "webauthn_verification_failed"
    # a disabled account gets decoys too
    admin.write("PATCH", f"/admin/accounts/{reviewer.account_id}", json={"status": "disabled", "reason": "left"})
    disabled = begin("rev@example.com")["options"]["allowCredentials"]
    assert [c["id"] for c in disabled] != list(reviewer.authn.credentials)


def test_finish_accepts_only_a_credential_of_the_bound_account(app):
    admin = bootstrap_admin(app)
    alice = invite_and_enroll(app, admin, "alice@example.com")
    bob = invite_and_enroll(app, admin, "bob@example.com")
    client = new_client(app)
    begin = client.post(f"{API}/auth/sign-in/begin", json={"email": "alice@example.com"}).json()
    bobs = next(iter(bob.authn.credentials.values()))
    wrong = bob.authn.assert_(begin["options"], credential=bobs)
    response = client.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": begin["ceremony_id"], "credential": wrong})
    assert response.status_code == 401 and _reason(response) == "credential_not_registered"
    assert alice.sign_in().status_code == 200


# --- verification failures -------------------------------------------------------------------


@pytest.mark.parametrize("origin", ["http://evil.example", "https://localhost:8010", "http://localhost:9999"])
def test_a_foreign_origin_is_rejected(app, store, origin):
    code = setup_code(store, "admin@example.com")
    response = enroll(new_client(app), SoftAuthenticator(), setup_code=code, origin=origin)
    assert response.status_code == 403 and _code(response) == "origin_not_allowed"
    admin = bootstrap_admin(app, email="other@example.com")
    signed = sign_in(new_client(app), admin.authn, "other@example.com", origin=origin)
    assert signed.status_code == 403 and _code(signed) == "origin_not_allowed"


def test_a_foreign_origin_header_is_rejected(app):
    admin = bootstrap_admin(app)
    client = new_client(app)
    begin = client.post(f"{API}/auth/sign-in/begin", json={"email": admin.email}).json()
    credential = admin.authn.assert_(begin["options"])
    response = client.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": begin["ceremony_id"], "credential": credential},
                           headers={"Origin": "http://evil.example"})
    assert response.status_code == 403 and _code(response) == "origin_not_allowed"


def test_a_wrong_relying_party_id_is_rejected(app, store):
    code = setup_code(store, "admin@example.com")
    response = enroll(new_client(app), SoftAuthenticator(), setup_code=code, rp_id="evil.example")
    assert response.status_code == 401 and _code(response) == "webauthn_verification_failed"
    admin = bootstrap_admin(app, email="other@example.com")
    signed = sign_in(new_client(app), admin.authn, admin.email, rp_id="evil.example")
    assert signed.status_code == 401 and _code(signed) == "webauthn_verification_failed"


def test_challenges_are_single_use_and_expire(app, store, clock):
    admin = bootstrap_admin(app)
    client = new_client(app)
    begin = client.post(f"{API}/auth/sign-in/begin", json={"email": admin.email}).json()
    body = {"ceremony_id": begin["ceremony_id"], "credential": admin.authn.assert_(begin["options"])}
    assert client.post(f"{API}/auth/sign-in/finish", json=body).status_code == 200
    replay = client.post(f"{API}/auth/sign-in/finish", json=body)
    assert replay.status_code == 401 and _reason(replay) == "challenge_already_used"
    # a failed finish burns the ceremony too
    begin = client.post(f"{API}/auth/sign-in/begin", json={"email": admin.email}).json()
    bad = client.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": begin["ceremony_id"],
                                                           "credential": admin.authn.assert_(begin["options"], tamper=True)})
    assert bad.status_code == 401 and _reason(bad) == "assertion_invalid"
    retry = client.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": begin["ceremony_id"], "credential": admin.authn.assert_(begin["options"])})
    assert retry.status_code == 401 and _reason(retry) == "challenge_already_used"
    # an assertion over another ceremony's challenge fails
    first = client.post(f"{API}/auth/sign-in/begin", json={"email": admin.email}).json()
    second = client.post(f"{API}/auth/sign-in/begin", json={"email": admin.email}).json()
    swapped = admin.authn.assert_(first["options"], challenge=second["options"]["challenge"])
    assert client.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": first["ceremony_id"], "credential": swapped}).status_code == 401
    # expiry
    begin = client.post(f"{API}/auth/sign-in/begin", json={"email": admin.email}).json()
    clock.advance(seconds=store.config.parameters.webauthn_challenge_lifetime_seconds + 1)
    late = client.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": begin["ceremony_id"], "credential": admin.authn.assert_(begin["options"])})
    assert late.status_code == 401 and _reason(late) == "challenge_expired"
    unknown = client.post(f"{API}/auth/sign-in/finish", json={"ceremony_id": "cer_nope", "credential": body["credential"]})
    assert unknown.status_code == 401 and _reason(unknown) == "unknown_ceremony"


def test_user_verification_is_required(app, store):
    code = setup_code(store, "admin@example.com")
    response = enroll(new_client(app), SoftAuthenticator(), setup_code=code, user_verified=False)
    assert response.status_code == 401 and _reason(response) == "registration_invalid"
    admin = bootstrap_admin(app, email="other@example.com")
    signed = sign_in(new_client(app), admin.authn, admin.email, user_verified=False)
    assert signed.status_code == 401 and _reason(signed) == "assertion_invalid"


def test_signature_counter_regressions_are_rejected(app):
    admin = bootstrap_admin(app)
    assert admin.sign_in().status_code == 200
    assert admin.sign_in().status_code == 200  # counter now 2
    cloned = admin.sign_in(sign_count=2)
    assert cloned.status_code == 401 and _code(cloned) == "webauthn_verification_failed"
    passkey = SoftAuthenticator(counter=False)  # synced passkeys report 0 every time
    other = bootstrap_admin_with(app, "passkey@example.com", passkey)
    assert other.sign_in().status_code == 200 and other.sign_in().status_code == 200


def bootstrap_admin_with(app, email, authn) -> Person:
    store = app.state.store
    code, _ = auth_cli.issue_setup_code(store, SetupCodeIssueRequest(purpose="break_glass", email=email, display_name="BG", os_user="root"))
    person = Person(new_client(app), authn, email)
    response = enroll(person.client, authn, setup_code=code)
    assert response.status_code == 200, response.text
    person.session = response.json()["signed_in"]["session"]
    return person


# --- step-up add authenticator ---------------------------------------------------------------


def test_add_authenticator_is_a_step_up_ceremony(app, store):
    admin = bootstrap_admin(app)
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    first_id = next(iter(reviewer.authn.credentials))
    begin = reviewer.write("POST", "/auth/authenticators/begin", json={"nickname": "Backup key"})
    assert begin.status_code == 200, begin.text
    body = begin.json()
    assert [c["id"] for c in body["reauthentication"]["allowCredentials"]] == [first_id]
    assert body["reauthentication"]["userVerification"] == "required"
    assert [c["id"] for c in body["options"]["excludeCredentials"]] == [first_id]
    assert body["options"]["user"]["id"] == reviewer.authn.credentials[first_id].user_handle
    backup = SoftAuthenticator(transports=["usb"])
    finish = reviewer.write("POST", "/auth/authenticators/finish", json={
        "ceremony_id": body["ceremony_id"],
        "reauthentication": reviewer.authn.assert_(body["reauthentication"]),
        "credential": backup.register(body["options"]),
    })
    assert finish.status_code == 200, finish.text
    assert finish.json()["signed_in"] is None and finish.json()["credential"]["nickname"] == "Backup key"
    assert finish.json()["account"]["authenticator_count"] == 2
    assert reviewer.get("/auth/session").json()["prompt_second_authenticator"] is False
    listed = reviewer.get("/auth/authenticators").json()["items"]
    assert len(listed) == 2
    assert Person(new_client(app), backup, reviewer.email).sign_in().status_code == 200
    assert _actions(store).count("authenticator_added") == 3


def test_add_authenticator_needs_a_fresh_verified_assertion_from_this_account(app):
    admin = bootstrap_admin(app)
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    other = invite_and_enroll(app, admin, "other@example.com")

    def attempt(**reauth_kw):
        body = reviewer.write("POST", "/auth/authenticators/begin", json={}).json()
        credential = reauth_kw.pop("credential", None)
        reauth = (other.authn if credential is not None else reviewer.authn).assert_(body["reauthentication"], credential=credential, **reauth_kw)
        return reviewer.write("POST", "/auth/authenticators/finish", json={
            "ceremony_id": body["ceremony_id"], "reauthentication": reauth, "credential": SoftAuthenticator().register(body["options"])})

    no_uv = attempt(user_verified=False)
    assert no_uv.status_code == 401 and _reason(no_uv) == "assertion_invalid"
    foreign = attempt(credential=next(iter(other.authn.credentials.values())))
    assert foreign.status_code == 401 and _reason(foreign) == "reauthentication_credential_not_registered"
    # a ceremony begun by one session cannot be finished by another
    body = reviewer.write("POST", "/auth/authenticators/begin", json={}).json()
    other_session = Person(new_client(app), reviewer.authn, reviewer.email)
    other_session.sign_in()
    stolen = other_session.write("POST", "/auth/authenticators/finish", json={
        "ceremony_id": body["ceremony_id"], "reauthentication": reviewer.authn.assert_(body["reauthentication"]),
        "credential": SoftAuthenticator().register(body["options"])})
    assert stolen.status_code == 401 and _reason(stolen) == "ceremony_not_this_session"
    # CSRF is required on both steps
    no_csrf = reviewer.client.post(f"{API}/auth/authenticators/begin", json={})
    assert no_csrf.status_code == 403 and _code(no_csrf) == "csrf_failed"
    assert len(reviewer.get("/auth/authenticators").json()["items"]) == 1


# --- own authenticators -----------------------------------------------------------------------


def test_own_authenticators_rename_and_remove_but_never_the_last(app, store):
    admin = bootstrap_admin(app)
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    own = reviewer.get("/auth/authenticators").json()["items"][0]
    renamed = reviewer.write("PATCH", f"/auth/authenticators/{own['id']}", json={"nickname": "Desk key"})
    assert renamed.status_code == 200 and renamed.json()["nickname"] == "Desk key"
    last = reviewer.write("DELETE", f"/auth/authenticators/{own['id']}")
    assert last.status_code == 409 and _reason(last) == "last_authenticator"
    admins = admin.get("/auth/authenticators").json()["items"][0]
    assert reviewer.write("DELETE", f"/auth/authenticators/{admins['id']}").status_code == 404
    # add a second key, sign in elsewhere with the first, then remove the first: that other session ends
    body = reviewer.write("POST", "/auth/authenticators/begin", json={}).json()
    backup = SoftAuthenticator()
    reviewer.write("POST", "/auth/authenticators/finish", json={"ceremony_id": body["ceremony_id"],
                                                                "reauthentication": reviewer.authn.assert_(body["reauthentication"]),
                                                                "credential": backup.register(body["options"])})
    elsewhere = Person(new_client(app), reviewer.authn, reviewer.email)
    assert elsewhere.sign_in().status_code == 200
    removed = reviewer.write("DELETE", f"/auth/authenticators/{own['id']}")
    assert removed.status_code == 204
    assert elsewhere.get("/auth/session").status_code == 401
    assert reviewer.get("/auth/session").status_code == 200  # the session doing the removal stays
    removed_key = reviewer.authn.credentials[own["credential_id"]]
    assert sign_in(new_client(app), reviewer.authn, reviewer.email, credential=removed_key).status_code == 401
    assert "authenticator_removed" in _actions(store)


# --- sessions ---------------------------------------------------------------------------------


def test_the_csrf_token_is_recoverable_and_required_on_writes(app):
    admin = bootstrap_admin(app)
    tab = {"Cookie": f"call1_session={admin.client.cookies.get('call1_session')}"}  # a new tab: the cookie only
    reloaded = new_client(app)
    info = reloaded.get(f"{API}/auth/session", headers=tab)
    assert info.status_code == 200 and info.json()["csrf_token"] == admin.session["csrf_token"]
    missing = admin.client.post(f"{API}/auth/sign-out")
    assert missing.status_code == 403 and _code(missing) == "csrf_failed"
    wrong = admin.client.post(f"{API}/auth/sign-out", headers={"X-Call1-CSRF": "A" * 43})
    assert wrong.status_code == 403 and _code(wrong) == "csrf_failed"
    foreign = admin.write("POST", "/auth/sign-out", headers={"Origin": "http://evil.example"})
    assert foreign.status_code == 403 and _code(foreign) == "origin_not_allowed"
    other = invite_and_enroll(app, admin, "rev@example.com")
    crossed = admin.client.post(f"{API}/auth/sign-out", headers=other.csrf)  # another session's token
    assert crossed.status_code == 403 and _code(crossed) == "csrf_failed"
    out = admin.write("POST", "/auth/sign-out", headers={"Origin": ORIGIN})
    assert out.status_code == 200 and out.json() == {"signed_out": True}
    assert admin.get("/auth/session").status_code == 401
    assert reloaded.get(f"{API}/auth/session", headers=tab).status_code == 401


def test_sessions_are_listed_and_revocable_by_their_owner(app, store):
    admin = bootstrap_admin(app)
    phone = Person(new_client(app), admin.authn, admin.email)
    phone.sign_in()
    listed = admin.get("/auth/sessions").json()["items"]
    assert {s["session_id"] for s in listed} == {admin.session["session_id"], phone.session["session_id"]}
    assert [s["current"] for s in listed if s["session_id"] == admin.session["session_id"]] == [True]
    assert all("csrf_token" not in s for s in listed)
    assert admin.write("DELETE", f"/auth/sessions/{phone.session['session_id']}").status_code == 204
    assert phone.get("/auth/session").status_code == 401
    assert admin.write("DELETE", f"/auth/sessions/{phone.session['session_id']}").status_code == 404
    other = invite_and_enroll(app, admin, "rev@example.com")
    assert admin.write("DELETE", f"/auth/sessions/{other.session['session_id']}").status_code == 404  # not yours
    own = admin.write("DELETE", f"/auth/sessions/{admin.session['session_id']}")
    assert own.status_code == 204 and "call1_session=" in own.headers.get("set-cookie", "")
    assert _actions(store).count("session_revoked") == 2


def test_sessions_expire_idle_and_absolute_and_slide_on_use(app, store, clock):
    params = store.config.parameters
    admin = bootstrap_admin(app)  # t = 0: idle expiry 8 h, absolute 12 h
    clock.advance(seconds=params.session_idle_lifetime_seconds - 120)
    assert admin.get("/auth/session").status_code == 200  # t = 7:58, slides the idle expiry (capped at 12 h)
    clock.advance(hours=2)
    info = admin.get("/auth/session")  # t = 9:58: alive only because it slid
    assert info.status_code == 200 and info.json()["idle_expires_at"] == info.json()["absolute_expires_at"]
    clock.advance(hours=2, minutes=5)  # t = 12:03: past the hard cap whatever the use
    expired = admin.get("/auth/session")
    assert expired.status_code == 401 and _code(expired) == "session_expired"
    idle = Person(new_client(app), admin.authn, admin.email)
    idle.sign_in()
    clock.advance(seconds=params.session_idle_lifetime_seconds)
    assert _code(idle.get("/auth/session")) == "session_expired"


def test_sessions_survive_a_store_restart(app, store, store_config, clock):
    admin = bootstrap_admin(app)
    cookie = admin.client.cookies.get("call1_session")
    restarted = create_app(store_config, clock=clock)
    info = new_client(restarted).get(f"{API}/auth/session", headers={"Cookie": f"call1_session={cookie}"})
    assert info.status_code == 200 and info.json()["csrf_token"] == admin.session["csrf_token"]


def test_a_disabled_account_is_refused_on_its_next_request(app):
    admin = bootstrap_admin(app)
    reviewer = invite_and_enroll(app, admin, "rev@example.com")
    assert admin.write("PATCH", f"/admin/accounts/{reviewer.account_id}", json={"status": "disabled", "reason": "left"}).status_code == 200
    refused = reviewer.get("/auth/session")
    assert refused.status_code == 403 and _code(refused) == "account_disabled"
    key = next(iter(reviewer.authn.credentials.values()))
    retry = sign_in(new_client(app), reviewer.authn, reviewer.email, credential=key)  # begin offers decoys; the real key fails
    assert retry.status_code == 401 and _code(retry) == "webauthn_verification_failed"


# --- rate limits ------------------------------------------------------------------------------


def test_sign_in_begin_is_rate_limited_per_email_and_per_client(app, clock):
    client = new_client(app)
    for _ in range(SIGN_IN_PER_EMAIL.max_hits):
        assert client.post(f"{API}/auth/sign-in/begin", json={"email": "target@example.com"}).status_code == 200
    limited = client.post(f"{API}/auth/sign-in/begin", json={"email": "Target@example.com"})
    assert limited.status_code == 429 and _code(limited) == "rate_limited" and limited.json()["retryable"] is True
    assert int(limited.headers["Retry-After"]) >= 1
    clock.advance(seconds=SIGN_IN_PER_EMAIL.window_seconds + 1)
    assert client.post(f"{API}/auth/sign-in/begin", json={"email": "target@example.com"}).status_code == 200
    clock.advance(seconds=SIGN_IN_PER_CLIENT.window_seconds + 1)
    statuses = [client.post(f"{API}/auth/sign-in/begin", json={"email": f"u{i}@example.com"}).status_code
                for i in range(SIGN_IN_PER_CLIENT.max_hits + 1)]
    assert statuses[:-1] == [200] * SIGN_IN_PER_CLIENT.max_hits and statuses[-1] == 429


def test_enrollment_begin_is_rate_limited(app):
    client = new_client(app)
    statuses = [client.post(f"{API}/auth/enroll/begin", json={"setup_code": f"GUESS-{i:05d}"}).status_code for i in range(11)]
    assert statuses[:10] == [410] * 10 and statuses[10] == 429


def test_no_password_or_minting_route_exists(app):
    paths = {route.path for route in app.routes if getattr(route, "path", "").startswith(f"{API}/auth")}
    assert not any("password" in p or "mint" in p or "reset" in p for p in paths)
    assert AccountStatus.REINVITE_REQUIRED.value == "reinvite_required"
