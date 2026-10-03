"""Smoke test for the e2e harness: real servers up, passkey enrollment through the real ceremony,
and a recording ingested through Process (fake handlers) listed by Store's calls API, which is
what Evaluate's call list reads."""

from __future__ import annotations

import pytest

from .stack import E2E_ROOT

pytestmark = pytest.mark.e2e


def test_servers_are_up_on_private_ports(stack):
    assert stack.store_port not in (8000, 8010, 8020) and stack.process_port not in (8000, 8010, 8020)
    assert stack.dir.resolve().is_relative_to(E2E_ROOT)

    status = stack.store_get("/status")
    assert status.status_code == 200, status.text

    health = stack.process_get("/health", token=False)
    assert health.status_code == 200 and health.json()["state"] == "running", health.text

    console = stack.process_get("/session")
    assert console.json()["token_valid"] is True, console.text
    assert stack.process_get("/session", token="c1con_wrong").json()["token_valid"] is False

    overview = stack.process_get("/overview").json()
    assert overview["handlers"]["mode"] == "fake", overview["handlers"]


def test_enrolled_admin_sees_an_ingested_call(stack, admin_session, reviewer_session):
    assert admin_session.role == "admin"
    assert reviewer_session.role == "reviewer"
    assert admin_session.cookies, "the session cookie was set by enroll/finish"

    receipt = stack.ingest("call_01_compliant", agent_id="agent-smoke")
    assert receipt["conversation_created"] is True and receipt["call_id"], receipt

    progress = stack.wait_until_settled(receipt["call_id"])
    assert progress["settled"] is True

    listing = admin_session.get("/calls", params={"limit": 100})
    assert listing.status_code == 200, listing.text
    row = next((c for c in listing.json()["items"] if c["call_id"] == receipt["call_id"]), None)
    assert row is not None, listing.json()
    assert row["agent_id"] == "agent-smoke"

    detail = reviewer_session.get(f"/calls/{receipt['call_id']}")
    assert detail.status_code == 200, detail.text

    # A write without the CSRF header is refused; the same write with it passes the CSRF check.
    no_csrf = admin_session.post("/admin/invitations", json={"email": "csrf-check@e2e.test", "display_name": "x", "role": "reviewer"},
                                 csrf=False)
    assert no_csrf.status_code == 403, no_csrf.text

    # Sign out, then sign back in with the same authenticator.
    assert admin_session.sign_out().status_code < 300
    assert admin_session.get("/calls").status_code == 401
    admin_session.sign_in()
    assert admin_session.get("/calls").status_code == 200


def test_private_stack_starts_empty(stack_factory):
    private = stack_factory(name="empty", with_process=False)
    admin = private.admin()
    listing = admin.get("/calls")
    assert listing.status_code == 200 and listing.json()["items"] == []


@pytest.mark.real_models
def test_real_models_settle_a_call(stack, admin_session):
    """Opt-in (CALL1_REAL_MODELS=1, Apple Silicon, weights in data/models): the real stack."""
    assert stack.process_get("/overview").json()["handlers"]["mode"] == "real"
    receipt = stack.ingest("call_01_compliant", agent_id="agent-real")
    progress = stack.wait_until_settled(receipt["call_id"])
    states = {g["kind"]: g["state"] for g in progress["groups"]}
    assert states.get("transcript") == "available", states
    transcript = admin_session.get(f"/calls/{receipt['call_id']}/transcript")
    assert transcript.status_code == 200, transcript.text
