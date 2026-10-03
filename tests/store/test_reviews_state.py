"""Call-level review writes: expected versions, the stale-write rule, escalation, retention,
speaker corrections, history and audit."""

from __future__ import annotations

from call1.contracts.contents import ResultKind, VerdictStatus
from call1.contracts.jobs import JobType

from .test_results_harness import accounts, fq, transcript  # noqa: F401


def _override(client, session, call_id, criterion="REG-01", *, expected=0, evaluation=1, status="PASS", **extra):
    return client.post(f"/store/v1/calls/{call_id}/verdicts/{criterion}",
                       json={"status": status, "expected_version": expected, "evaluation_version": evaluation, **extra}, headers=session.headers)


def test_two_reviewers_cannot_overwrite_each_other(fq, client, mint_session):
    a, b = mint_session("reviewer"), mint_session("reviewer")
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.9), ("SEC-01", VerdictStatus.PASS, 0.9)])
    first = _override(client, a, conv.call_id, expected=0, reviewer_notes="disclosure was given", reason_code="model_misread_evidence")
    assert first.status_code == 200 and first.json()["review_version"] == 1 and first.json()["change_cursor"]
    second = _override(client, b, conv.call_id, expected=0, status="FAIL")
    assert second.status_code == 409 and second.json()["code"] == "review_version_conflict" and second.json()["details"] == {"current_version": 1}
    retry = _override(client, b, conv.call_id, "SEC-01", expected=1, status="FAIL")
    assert retry.json()["review_version"] == 2
    state = client.get(f"/store/v1/calls/{conv.call_id}/review", headers=a.read_headers).json()
    assert state["review_version"] == 2 and state["reviewed_evaluation_version"] == 1 and state["staleness"] == "current"
    assert [(o["criterion_id"], o["original_status"], o["status"], o["account_id"]) for o in state["overrides"]] == [
        ("REG-01", "FAIL", "PASS", a.account_id), ("SEC-01", "PASS", "FAIL", b.account_id)]
    assert client.get(f"/store/v1/calls/{conv.call_id}", headers=a.read_headers).json()["review_version"] == 2
    missing = _override(client, a, conv.call_id, "NOPE-9", expected=2)
    assert missing.status_code == 404


def test_stale_review_writes_are_rejected_after_a_new_machine_version(fq, client, reviewer_session, supervisor_session):
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.9)])
    assert _override(client, reviewer_session, conv.call_id, expected=0, evaluation=1).status_code == 200
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.PASS, 0.9)])
    stale = _override(client, reviewer_session, conv.call_id, expected=1, evaluation=1, status="FAIL")
    assert stale.status_code == 409 and stale.json()["code"] == "conflict"
    assert stale.json()["details"]["current_evaluation_version"] == 2
    state = client.get(f"/store/v1/calls/{conv.call_id}/review", headers=reviewer_session.read_headers).json()
    assert state["staleness"] == "stale" and state["current_evaluation_version"] == 2 and state["reviewed_evaluation_version"] == 1
    assert state["review_version"] == 1  # Store's staleness change is not a human write

    wrong = client.post(f"/store/v1/calls/{conv.call_id}/review/retain", json={"expected_version": 0}, headers=supervisor_session.headers)
    assert wrong.json()["code"] == "review_version_conflict"
    denied = client.post(f"/store/v1/calls/{conv.call_id}/review/retain", json={"expected_version": 1}, headers=reviewer_session.headers)
    assert denied.json()["code"] == "insufficient_role"
    kept = client.post(f"/store/v1/calls/{conv.call_id}/review/retain", json={"expected_version": 1, "note": "still right"},
                       headers=supervisor_session.headers)
    assert kept.status_code == 200 and kept.json()["review_version"] == 2
    state = client.get(f"/store/v1/calls/{conv.call_id}/review", headers=reviewer_session.read_headers).json()
    assert state["staleness"] == "retained" and state["retained_by_account_id"] == supervisor_session.account_id
    again = client.post(f"/store/v1/calls/{conv.call_id}/review/retain", json={"expected_version": 2}, headers=supervisor_session.headers)
    assert again.json()["code"] == "invalid_transition"

    fresh = _override(client, reviewer_session, conv.call_id, expected=2, evaluation=2, status="FAIL")
    assert fresh.status_code == 200
    state = client.get(f"/store/v1/calls/{conv.call_id}/review", headers=reviewer_session.read_headers).json()
    assert state["staleness"] == "current" and state["reviewed_evaluation_version"] == 2
    assert [o["evaluation_version"] for o in state["overrides"]] == [2]


def test_escalation_resolution_is_a_supervisor_decision(fq, client, reviewer_session, supervisor_session, admin_session):
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FLAGGED, 0.4)], requires_human_review=True)
    url = f"/store/v1/calls/{conv.call_id}/escalation"
    body = {"escalation_status": "APPROVED", "expected_version": 0, "evaluation_version": 1, "reviewer_notes": "checked"}
    assert client.post(url, json=body, headers=reviewer_session.headers).json()["code"] == "insufficient_role"
    assert client.post(url, json={**body, "evaluation_version": 2}, headers=supervisor_session.headers).json()["code"] == "conflict"
    done = client.post(url, json=body, headers=supervisor_session.headers)
    assert done.status_code == 200 and done.json()["review_version"] == 1
    state = client.get(f"/store/v1/calls/{conv.call_id}/review", headers=reviewer_session.read_headers).json()
    assert state["escalation_status"] == "APPROVED" and state["escalation_resolved_by_account_id"] == supervisor_session.account_id
    escalations = client.get("/store/v1/escalations", params={"status": "APPROVED"}, headers=reviewer_session.read_headers).json()["items"]
    assert [(e["call_id"], e["escalation_status"], e["review_version"]) for e in escalations] == [(conv.call_id, "APPROVED", 1)]
    assert client.get("/store/v1/escalations", headers=reviewer_session.read_headers).json()["items"] == []  # default: PENDING

    calm = fq.register()
    fq.ingest_qa(calm, [("REG-01", VerdictStatus.PASS, 0.99)])
    none = client.post(f"/store/v1/calls/{calm.call_id}/escalation", json=body, headers=supervisor_session.headers)
    assert none.status_code == 409 and none.json()["code"] == "invalid_transition"

    audit = client.get("/store/v1/admin/audit", params={"target_id": conv.call_id}, headers=admin_session.read_headers).json()["items"]
    assert [e["action"] for e in audit] == ["escalation_resolved"]


def test_new_version_reopens_escalation_and_marks_resolution_stale(fq, client, reviewer_session, supervisor_session):
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FLAGGED, 0.4)], requires_human_review=True)
    client.post(f"/store/v1/calls/{conv.call_id}/escalation", json={"escalation_status": "OVERRIDDEN", "expected_version": 0, "evaluation_version": 1},
                headers=supervisor_session.headers)
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FLAGGED, 0.3)], requires_human_review=True)
    state = client.get(f"/store/v1/calls/{conv.call_id}/review", headers=reviewer_session.read_headers).json()
    assert state["staleness"] == "stale" and state["escalation_status"] == "OVERRIDDEN"
    again = client.post(f"/store/v1/calls/{conv.call_id}/escalation",
                        json={"escalation_status": "APPROVED", "expected_version": 1, "evaluation_version": 2}, headers=supervisor_session.headers)
    assert again.status_code == 200
    history = client.get(f"/store/v1/calls/{conv.call_id}/review/history", headers=reviewer_session.read_headers).json()["items"]
    assert [h["kind"] for h in history] == ["escalation_resolution", "marked_stale", "escalation_resolution"]


def test_speaker_correction_creates_a_reanalysis_request(fq, client, reviewer_session, admin_session):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.QA_SCORECARD])
    fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("AGENT", "hello"), ("AGENT", "I am the caller actually")])})
    url = f"/store/v1/calls/{conv.call_id}/speaker-corrections"
    bad_turn = client.post(url, json={"correction": {"turn_id": 7, "speaker": "CALLER"}, "expected_version": 0}, headers=reviewer_session.headers)
    assert bad_turn.status_code == 422
    stale = client.post(url, json={"correction": {"turn_id": 1, "speaker": "CALLER"}, "expected_version": 5}, headers=reviewer_session.headers)
    assert stale.json()["code"] == "review_version_conflict"
    created = client.post(url, json={"correction": {"turn_id": 1, "speaker": "CALLER", "notes": "wrong channel"}, "expected_version": 0},
                          headers=reviewer_session.headers)
    assert created.status_code == 201
    request = created.json()
    assert request["kind"] == "speaker_correction" and request["speaker_correction"]["turn_id"] == 1 and request["status"] == "pending"
    assert request["requested_by_account_id"] == reviewer_session.account_id and conv.requests[0].id == request["id"]
    detail = client.get(f"/store/v1/calls/{conv.call_id}", headers=reviewer_session.read_headers).json()
    assert detail["review_version"] == 1
    groups = {g["kind"]: g for g in detail["results"]}
    assert groups["transcript"]["state"] == "available" and groups["transcript"]["reanalysis_request_id"] is None
    assert groups["qa"]["state"] == "pending" and groups["qa"]["reanalysis_request_id"] == request["id"]  # the correction affects QA
    assert conv.pending_reanalysis >= {ResultKind.QA, ResultKind.TONE}
    history = client.get(f"/store/v1/calls/{conv.call_id}/review/history", headers=reviewer_session.read_headers).json()["items"]
    assert history[0]["kind"] == "speaker_correction" and history[0]["payload"]["reanalysis_request_id"] == request["id"]
    audit = client.get("/store/v1/admin/audit", params={"action": "reanalysis_requested"}, headers=admin_session.read_headers).json()["items"]
    assert audit[0]["target"]["id"] == conv.call_id


def test_review_writes_need_csrf_and_the_right_session(fq, client, reviewer_session, service_key_headers):
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.9)])
    url = f"/store/v1/calls/{conv.call_id}/verdicts/REG-01"
    body = {"status": "PASS", "expected_version": 0, "evaluation_version": 1}
    assert client.post(url, json=body, headers=reviewer_session.read_headers).json()["code"] == "csrf_failed"
    assert client.post(url, json=body, headers=service_key_headers).json()["code"] == "forbidden"
    assert client.post(url, json=body, headers={**reviewer_session.headers, "Origin": "https://evil.example"}).json()["code"] == "origin_not_allowed"
