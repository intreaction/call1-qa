"""Auth area, enrollment and sign-in, end to end against real Store (and Process) server processes.

Feature IDs from the auth inventory:
    a1  passkey enrollment via setup code and invitation link
    a2  (known issue a) the session exists the moment enrollment finishes: server side of the check;
        the UI side is frontend/e2e/auth.spec.ts
    a3  account-first sign-in with allowCredentials, including non-discoverable security keys, and
        the sign-in rate limits
"""

from __future__ import annotations

import httpx
import pytest

from .auth_support import begin_sign_in, code_of, describe, details_of, finish_sign_in, only_credential, sign_in_with
from .stack import unique_email

pytestmark = pytest.mark.e2e


# --- a1 ---------------------------------------------------------------------------------------


def test_a1_setup_code_enrolls_first_admin(stack_factory):
    private = stack_factory(name="a1-first-admin", with_process=False)
    email = "first-admin@e2e.test"
    code = private.setup_code(email, "First Admin")

    person = private.new_session(email)
    finish = person.enroll(setup_code=code, nickname="first key")
    body = finish.json()
    assert body["account"]["email"] == email and body["account"]["role"] == "admin", body
    assert body["account"]["status"] == "active" and body["account"]["authenticator_count"] == 1, body
    assert body["credential"]["credential_id"] == only_credential(person.authenticator).id
    assert body["credential"]["nickname"] == "first key"
    assert body["signed_in"]["session"]["role"] == "admin"
    assert "manage_accounts" in body["signed_in"]["session"]["permissions"]

    session = person.get("/auth/session")
    assert session.status_code == 200 and session.json()["account_id"] == body["account"]["id"], describe(session)

    # The code is single-use.
    again = private.new_session("someone-else@e2e.test").enroll(setup_code=code, check=False)
    assert again.status_code == 410 and code_of(again) == "setup_code_invalid", describe(again)

    # The admin sees the redeemed code (issuance is host-only; the listing is audit).
    codes = person.get("/admin/setup-codes").json()["items"]
    record = next(c for c in codes if c["email"] == email)
    assert record["purpose"] == "first_admin" and record["used_at"] and record["enrolled_account_id"] == body["account"]["id"], record

    # A malformed or unknown code is refused at begin, before any account exists.
    bogus = private.new_session("bogus@e2e.test").enroll(setup_code="ABCD-EFGH-IJKL-MNOP", check=False)
    assert bogus.status_code in (410, 422) and code_of(bogus) in ("setup_code_invalid", "validation_failed"), describe(bogus)


def test_a1_invitation_link_enrolls_reviewer_bound_to_passkey(stack, admin_session):
    email = unique_email("a1-invitee")
    issued = admin_session.post("/admin/invitations", json={"email": email, "display_name": "A1 Invitee", "role": "reviewer"})
    assert issued.status_code == 200, describe(issued)
    url = issued.json()["invitation_url"]
    assert url.startswith(f"{stack.store_url}/enroll#"), url
    token = url.split("#", 1)[1]
    invitation_id = issued.json()["invitation"]["id"]

    person = stack.new_session(email)
    body = person.enroll(invitation_token=token, nickname="a1 key").json()
    assert body["account"]["role"] == "reviewer" and body["account"]["email"] == email, body
    assert body["signed_in"]["session"]["role"] == "reviewer"

    # Bound to exactly the passkey that enrolled.
    own = person.get("/auth/authenticators").json()["items"]
    assert [c["credential_id"] for c in own] == [only_credential(person.authenticator).id], own

    # The invitation is redeemed by that account, and single-use.
    listed = admin_session.get("/admin/invitations", params={"status": "redeemed", "limit": 200}).json()["items"]
    record = next((i for i in listed if i["id"] == invitation_id), None)
    assert record is not None and record["account_id"] == body["account"]["id"], listed[:3]
    again = stack.new_session(email).enroll(invitation_token=token, check=False)
    assert again.status_code == 410 and code_of(again) == "invitation_invalid", describe(again)

    # Sign out, then back in with the same passkey.
    assert person.sign_out().status_code == 200
    assert person.get("/auth/session").status_code == 401
    person.sign_in()
    assert person.get("/auth/session").json()["account_id"] == body["account"]["id"]

    # Another enrolled account's passkey cannot sign in as this one.
    other = stack.user("reviewer", cached=False)
    intruder = sign_in_with(stack, email, other.authenticator, credential=only_credential(other.authenticator))
    assert intruder.status_code == 401 and code_of(intruder) == "webauthn_verification_failed", describe(intruder)
    assert details_of(intruder).get("reason") == "credential_not_registered", intruder.text


def test_a1_enrollment_refuses_foreign_origin_and_missing_user_verification(stack, admin_session):
    email = unique_email("a1-misbehave")
    token = stack.invite(email, "reviewer")

    foreign = stack.new_session(email).enroll(invitation_token=token, origin="http://evil.example:9999", check=False)
    assert foreign.status_code == 403 and code_of(foreign) == "origin_not_allowed", describe(foreign)

    # The failed finish consumed only its ceremony; a new ceremony still redeems the invitation...
    no_uv = stack.new_session(email).enroll(invitation_token=token, user_verified=False, check=False)
    assert no_uv.status_code == 401 and code_of(no_uv) == "webauthn_verification_failed", describe(no_uv)

    # ...and a correct one does.
    ok = stack.new_session(email)
    ok.enroll(invitation_token=token)
    assert ok.role == "reviewer"


# --- a2 (server side of known issue a) --------------------------------------------------------


def test_a2_enrollment_finish_establishes_the_session_immediately(stack, admin_session):
    """The server half of known issue (a): enroll/finish sets the cookie and every signed-in read
    works on the very next request, with no sign-in step. (Evaluate's UI half is in auth.spec.ts.)"""
    email = unique_email("a2-reviewer")
    person = stack.new_session(email)
    finish = person.enroll(invitation_token=stack.invite(email, "reviewer"))
    assert finish.status_code == 200
    assert person.cookies.get("call1_session"), person.cookies
    for path in ("/auth/session", "/calls", "/review-queue", "/escalations", "/rubrics", "/auth/authenticators", "/auth/sessions"):
        response = person.get(path)
        assert response.status_code == 200, describe(response)
    assert person.get("/auth/session").json()["session_id"] == finish.json()["signed_in"]["session"]["session_id"]


# --- a3 ---------------------------------------------------------------------------------------


def test_a3_sign_in_is_account_first_with_allow_credentials(stack, admin_session):
    person = stack.user("reviewer", cached=False)
    live = [c["credential_id"] for c in person.get("/auth/authenticators").json()["items"]]

    known = begin_sign_in(stack, person.email, forwarded_for="10.251.0.1")
    assert known.status_code == 200, describe(known)
    options = known.json()["options"]
    assert sorted(d["id"] for d in options["allowCredentials"]) == sorted(live), options
    assert all(d["type"] == "public-key" for d in options["allowCredentials"])
    assert options["rpId"] == "localhost"

    # An unknown email gets the same shape with one or two decoys, stable per email.
    ghost = unique_email("a3-ghost")
    first = begin_sign_in(stack, ghost, forwarded_for="10.251.0.2")
    second = begin_sign_in(stack, ghost, forwarded_for="10.251.0.3")
    assert first.status_code == second.status_code == 200
    decoys = [d["id"] for d in first.json()["options"]["allowCredentials"]]
    assert 1 <= len(decoys) <= 2, decoys
    assert decoys == [d["id"] for d in second.json()["options"]["allowCredentials"]], "decoys must be stable per email"
    assert set(first.json()) == set(known.json()) and set(first.json()["options"]) == set(options)
    assert not set(decoys) & set(live)


def test_a3_non_discoverable_security_key_signs_in_without_user_handle(stack, admin_session):
    """A security key with a non-resident credential answers from allowCredentials and returns no
    userHandle. Account-first sign-in must accept that."""
    person = stack.user("reviewer", cached=False)

    def drop_user_handle(assertion):
        assertion["response"]["userHandle"] = None

    response = sign_in_with(stack, person.email, person.authenticator, mutate=drop_user_handle)
    assert response.status_code == 200, describe(response)
    assert response.json()["session"]["account_id"] == person.account_id

    def omit_user_handle(assertion):
        assertion["response"].pop("userHandle", None)

    response = sign_in_with(stack, person.email, person.authenticator, mutate=omit_user_handle)
    assert response.status_code == 200, describe(response)


def test_a3_sign_in_refuses_tampered_replayed_and_unverified_assertions(stack, admin_session):
    person = stack.user("reviewer", cached=False)
    credential = only_credential(person.authenticator)

    baseline = _sign_in_custom(stack, person)
    assert baseline.status_code == 200, describe(baseline)

    bad_signature = _sign_in_custom(stack, person, tamper=True)
    assert bad_signature.status_code == 401 and code_of(bad_signature) == "webauthn_verification_failed", describe(bad_signature)

    stale = _sign_in_custom(stack, person, sign_count=max(credential.sign_count - 1, 0))
    assert stale.status_code == 401 and code_of(stale) == "webauthn_verification_failed", describe(stale)

    unverified = _sign_in_custom(stack, person, user_verified=False)
    assert unverified.status_code == 401 and code_of(unverified) == "webauthn_verification_failed", describe(unverified)

    foreign = _sign_in_custom(stack, person, origin="http://evil.example:9999")
    assert foreign.status_code == 403 and code_of(foreign) == "origin_not_allowed", describe(foreign)


def _sign_in_custom(stack, person, **assert_kw):
    client = httpx.Client(base_url=stack.store_url, timeout=30.0)
    begin = begin_sign_in(stack, person.email, client=client, forwarded_for="10.252.1.1")
    assert begin.status_code == 200, describe(begin)
    body = begin.json()
    assert_kw.setdefault("origin", stack.store_url)
    assertion = person.authenticator.assert_(body["options"], **assert_kw)
    return finish_sign_in(stack, client, body["ceremony_id"], assertion)


def test_a3_sign_in_rate_limits_per_email_and_per_client(stack_factory):
    """Sign-in begin: 10 per 5 minutes per email however the client presents itself, and 30 per
    minute per client address. A loopback client's X-Forwarded-For is taken as its address
    (uvicorn's default forwarded_allow_ips=127.0.0.1): that is the harness's own assumption, and it
    must never lift the per-email limit."""
    private = stack_factory(name="a3-ratelimit", with_process=False, spread_clients=False)
    victim = unique_email("a3-victim")

    for i in range(10):
        ok = begin_sign_in(private, victim, forwarded_for=f"10.253.{i}.1")
        assert ok.status_code == 200, describe(ok)
    limited = begin_sign_in(private, victim, forwarded_for="10.253.99.1")
    assert limited.status_code == 429 and code_of(limited) == "rate_limited", describe(limited)
    assert int(limited.headers.get("Retry-After", "0")) > 0, limited.headers

    # Per client address: 30 a minute, whatever the email.
    address = "10.254.0.7"
    for i in range(30):
        ok = begin_sign_in(private, unique_email(f"a3-spray-{i}"), forwarded_for=address)
        assert ok.status_code == 200, f"begin {i}: {describe(ok)}"
    limited = begin_sign_in(private, unique_email("a3-spray-x"), forwarded_for=address)
    assert limited.status_code == 429 and code_of(limited) == "rate_limited", describe(limited)
    # Documented harness assumption: a loopback client naming another address is keyed on it.
    elsewhere = begin_sign_in(private, unique_email("a3-spray-y"), forwarded_for="10.254.0.8")
    assert elsewhere.status_code == 200, describe(elsewhere)


def test_a3_enrollment_begin_rate_limit_per_client(stack_factory):
    private = stack_factory(name="a3-enroll-limit", with_process=False, spread_clients=False)
    headers = {"Origin": private.store_url, "X-Forwarded-For": "10.255.0.9"}
    statuses = []
    for _ in range(11):
        response = private.http.post("/store/v1/auth/enroll/begin", json={"setup_code": "AAAA-BBBB-CCCC-DDDD"}, headers=headers)
        statuses.append(response.status_code)
    assert all(s != 429 for s in statuses[:10]), statuses
    assert statuses[10] == 429, statuses
