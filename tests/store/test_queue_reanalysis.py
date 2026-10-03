"""Reanalysis requests (create, claim, expire, reject, fulfil by graph) and draft tests."""

from __future__ import annotations

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.contents import ResultKind
from call1.contracts.events import Actor, ActorKind
from call1.contracts.jobs import JobType, ReanalysisKind, SpeakerCorrection
from call1.store.queue import api as queue_api

from .test_queue_harness import CRITERION, RUBRIC_ID, RUBRIC_REF, hooks, job, pinned, q, rubrics, upstream_input  # noqa: F401


def _request(q, session, call_id, body=None, *, key="click-000001", expect=201):
    headers = {**session.headers, "Idempotency-Key": key} if key else session.headers
    response = q.client.post(f"/store/v1/calls/{call_id}/reanalysis-requests", json=body or {"kind": "qa"}, headers=headers)
    assert response.status_code == expect, response.text
    return response.json()


def _claim(q, max_requests=4):
    return q.post("/reanalysis-requests/claim", {"worker_id": "w1", "max_requests": max_requests}).json()["requests"]


def test_reanalysis_requests_are_durable_and_idempotent(q, store, reviewer_session):
    conversation = q.register()
    call_id = conversation["call_id"]
    request = _request(q, reviewer_session, call_id, {"kind": "qa", "rubric": RUBRIC_REF.model_dump(mode="json")})
    assert request["status"] == "pending" and request["kind"] == "qa" and request["requested_by_account_id"] == reviewer_session.account_id
    assert request["idempotency_key"] == "click-000001" and request["call_id"] == call_id and request["conversation_id"] == conversation["id"]
    assert _request(q, reviewer_session, call_id, {"kind": "qa", "rubric": RUBRIC_REF.model_dump(mode="json")}) == request
    reused = _request(q, reviewer_session, call_id, {"kind": "summary"}, expect=409)
    assert reused["code"] == "idempotency_key_reused" and reused["details"]["original_id"] == request["id"]
    duplicate = _request(q, reviewer_session, call_id, {"kind": "qa"}, key="click-000002", expect=409)
    assert duplicate["code"] == "conflict" and duplicate["details"]["existing_request_id"] == request["id"]
    assert _request(q, reviewer_session, call_id, key=None, expect=422)["code"] == "validation_failed"
    assert _request(q, reviewer_session, call_id, key="short", expect=422)["details"]["reason"] == "invalid_idempotency_key"
    assert _request(q, reviewer_session, "call_missing", key="click-000003", expect=404)["code"] == "not_found"
    correction = {"kind": "speaker_correction", "speaker_correction": {"turn_id": 0, "speaker": "CALLER"}}
    assert _request(q, reviewer_session, call_id, correction, key="click-000004", expect=422)["details"]["reason"] == "use_speaker_corrections"
    wrong_digest = dict(RUBRIC_REF.model_dump(mode="json"), digest="sha256:" + "3" * 64)
    assert _request(q, reviewer_session, call_id, {"kind": "full", "rubric": wrong_digest}, key="click-000005", expect=409)["details"]["reason"] == "rubric_digest_mismatch"
    with store.connection() as conn:
        assert queue_api.reanalysis_pending_groups(conn, conversation["id"]) == frozenset({ResultKind.QA})
        assert queue_api.reanalysis_request_for_group(conn, conversation["id"], ResultKind.QA) == request["id"]
    listed = q.get(f"/calls/{call_id}/reanalysis-requests", headers=reviewer_session.read_headers).json()["items"]
    assert [r["id"] for r in listed] == [request["id"]]
    assert q.get(f"/reanalysis-requests/{request['id']}", headers=reviewer_session.read_headers).json() == request
    audit = q.get("/admin/audit", headers=q.mint_session("admin").read_headers, params={"action": "reanalysis_requested"}).json()["items"]
    assert audit[0]["target"] == {"kind": "call", "id": call_id} and audit[0]["details"]["request_id"] == request["id"]
    assert q.post(f"/calls/{call_id}/reanalysis-requests", {"kind": "qa"}, headers={**q.headers, "Idempotency-Key": "click-000009"}, expect=403).json()["code"] == "forbidden"


def test_claims_expire_back_to_pending_and_a_rejection_needs_the_active_claim(q, reviewer_session):
    conversation = q.register()
    request = _request(q, reviewer_session, conversation["call_id"])
    [claimed] = _claim(q)
    assert claimed["request"]["status"] == "claimed" and claimed["request"]["claimed_by_installation_id"] == q.installation_id
    assert claimed["claim_expires_at"] == "2026-09-25T12:05:00Z" and _claim(q) == []
    q.clock.advance(300)
    assert q.get(f"/reanalysis-requests/{request['id']}").json()["status"] == "pending"  # claimed -> pending on expiry
    [again] = _claim(q)
    stale = q.post(f"/reanalysis-requests/{request['id']}/reject", {"claim_token": claimed["claim_token"], "reason": "no"}, expect=409).json()
    assert stale["code"] == "claim_token_stale"
    rejected = q.post(f"/reanalysis-requests/{request['id']}/reject", {"claim_token": again["claim_token"], "reason": "call too short"}).json()
    assert rejected["status"] == "rejected" and rejected["rejected_reason"] == "call too short"
    twice = q.post(f"/reanalysis-requests/{request['id']}/reject", {"claim_token": again["claim_token"], "reason": "x"}, expect=409).json()
    assert twice["code"] == "invalid_transition"


def test_the_graph_that_carries_the_claim_token_fulfils_the_request(q, reviewer_session, store):
    conversation = q.register()
    snapshot = q.rubric_snapshot(conversation["id"])
    request = _request(q, reviewer_session, conversation["call_id"])
    [claimed] = _claim(q)
    jobs = [job("score", JobType.QA_SCORECARD, key="reanalysis-score-1", inputs=[pinned("rubric", snapshot)])]
    stale = q.graph(conversation["id"], jobs, key="reanalysis-graph-0", reason="reanalysis", request_id=request["id"], claim_token="t" * 43, expect=409).json()
    assert stale["code"] == "claim_token_stale"
    graph = q.graph(conversation["id"], jobs, key="reanalysis-graph-1", reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"])
    assert graph["reanalysis_request_id"] == request["id"]
    fulfilled = q.get(f"/reanalysis-requests/{request['id']}").json()
    assert fulfilled["status"] == "fulfilled" and fulfilled["graph_id"] == graph["graph_id"]
    replay = q.graph(conversation["id"], jobs, key="reanalysis-graph-1", reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"])
    assert replay["created"] is False and replay["graph_id"] == graph["graph_id"]
    second = q.graph(conversation["id"], jobs, key="reanalysis-graph-2", reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"], expect=409).json()
    assert second["code"] == "invalid_transition"
    with store.connection() as conn:
        assert queue_api.reanalysis_pending_groups(conn, conversation["id"]) == frozenset()


def test_draft_tests_snapshot_the_draft_and_never_touch_the_calls_results(q, supervisor_session, reviewer_session, hooks, store):
    conversation = q.register()
    call_id = conversation["call_id"]
    headers = {**supervisor_session.headers, "Idempotency-Key": "draft-test-01"}
    body = {"call_id": call_id, "expected_draft_revision": 1}
    created = q.client.post(f"/store/v1/rubrics/{RUBRIC_ID}/draft/tests", json=body, headers=headers)
    assert created.status_code == 201, created.text
    request = created.json()
    assert request["kind"] == "qa_draft_test" and request["status"] == "pending"
    draft = request["draft_rubric"]
    snapshot = q.get(f"/artifacts/{draft['snapshot_artifact_id']}").json()
    assert snapshot["slot"] == f"draft:{request['id']}:rubric:{RUBRIC_ID}:r1" and snapshot["linked"]
    assert q.client.post(f"/store/v1/rubrics/{RUBRIC_ID}/draft/tests", json=body, headers=headers).json() == request
    stale_draft = q.client.post(f"/store/v1/rubrics/{RUBRIC_ID}/draft/tests", json=dict(body, expected_draft_revision=2),
                                headers={**supervisor_session.headers, "Idempotency-Key": "draft-test-02"})
    assert stale_draft.status_code == 409 and stale_draft.json()["code"] == "rubric_version_conflict"
    denied = q.client.post(f"/store/v1/rubrics/{RUBRIC_ID}/draft/tests", json=body, headers={**reviewer_session.headers, "Idempotency-Key": "draft-test-03"})
    assert denied.status_code == 403
    # draft snapshots stay out of default listings
    assert all(not a["slot"].startswith("draft:") for a in q.get(f"/conversations/{conversation['id']}/artifacts").json()["items"])

    [claimed] = _claim(q)
    rubric_in = pinned("rubric", snapshot)
    draft_params = {"draft_rubric": draft}
    live = q.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, key="draft-score-0", inputs=[rubric_in])], key="draft-graph-0",
                   reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"], expect=422).json()
    assert live["details"]["reason"] == "draft_rubric_mismatch"
    graph = q.graph(conversation["id"], [
        job("crit", JobType.QA_CRITERION, key="draft-crit-1", inputs=[rubric_in], parameters=draft_params),
        job("score", JobType.QA_SCORECARD, key="draft-score-1", requires=["crit"], inputs=[rubric_in, upstream_input("assessment:greeting", "crit", "assessment")],
            parameters=draft_params),
    ], key="draft-graph-1", reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"])
    ids = q.ids(graph)
    crit = q.claim_one(ids["crit"])
    live_slot = q.inline(conversation["id"], ArtifactKind.QA_ASSESSMENT, slot=CRITERION, job_id=ids["crit"], token=crit["claim_token"], expect=422).json()
    assert live_slot["details"]["reason"] == "draft_slot_required"
    q.complete(crit, result=None)
    pending = q.get(f"/reanalysis-requests/{request['id']}/draft-result", headers=supervisor_session.read_headers).json()
    assert pending["state"] == "pending" and pending["scorecard"] is None
    score = q.claim_one(ids["score"])
    outputs = q.outputs_for(score)
    published = q.post(f"/jobs/{ids['score']}/complete", q.completion(score, outputs, result={"kind": "qa", "state": "available"}), expect=422).json()
    assert published["details"]["reason"] == "result_in_draft_test"
    receipt = q.complete(score, outputs=outputs, result=None)
    assert receipt["result_version"] is None
    assert hooks.completions[-1].job.draft_test_request_id == request["id"]
    result = q.get(f"/reanalysis-requests/{request['id']}/draft-result", headers=supervisor_session.read_headers).json()
    assert result["state"] == "available" and result["scorecard"]["overall_score"] == 90 and result["draft_revision"] == 1
    assert q.get(f"/reanalysis-requests/{request['id']}").json()["draft_result_artifact_id"] == receipt["linked_artifact_ids"][0]
    with store.connection() as conn:
        work = queue_api.pending_work(conn, conversation["id"])
        stakes = queue_api.group_stakes(conn, conversation["id"])
    assert work.jobs_total == 0 and work.settled is True  # draft-test jobs never count
    assert [s.draft_test for s in stakes[ResultKind.QA]] == [True]
    assert q.get(f"/conversations/{conversation['id']}/progress").json()["groups"][3] == {
        "kind": "qa", "state": "disabled", "total": 0, "succeeded": 0, "running": 0, "queued": 0, "blocked": 0, "dead_blocked": 0,
        "failed": 0, "cancelled": 0, "waiting_reason": None}


def test_a_failed_draft_test_reports_failed(q, supervisor_session):
    conversation = q.register()
    created = q.client.post(f"/store/v1/rubrics/{RUBRIC_ID}/draft/tests", json={"call_id": conversation["call_id"], "expected_draft_revision": 1},
                            headers={**supervisor_session.headers, "Idempotency-Key": "draft-test-11"}).json()
    [claimed] = _claim(q)
    snapshot = q.get(f"/artifacts/{created['draft_rubric']['snapshot_artifact_id']}").json()
    graph = q.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, key="draft-score-9", inputs=[pinned("rubric", snapshot)],
                                             parameters={"draft_rubric": created["draft_rubric"]})],
                    key="draft-graph-9", reason="reanalysis", request_id=created["id"], claim_token=claimed["claim_token"])
    q.fail(q.claim_one(q.ids(graph)["score"]), "configuration_error")
    result = q.get(f"/reanalysis-requests/{created['id']}/draft-result", headers=supervisor_session.read_headers).json()
    assert result["state"] == "failed" and result["failure_code"] == "configuration_error"
    not_draft = _request(q, q.mint_session("reviewer"), conversation["call_id"], key="click-000100")
    assert q.get(f"/reanalysis-requests/{not_draft['id']}/draft-result", headers=supervisor_session.read_headers, expect=404).json()["code"] == "not_found"


def test_the_results_area_creates_speaker_correction_requests(q, store):
    conversation = q.register()
    correction = SpeakerCorrection(turn_id=0, speaker="CALLER")
    actor = Actor(kind=ActorKind.REVIEWER, account_id="acct_reviewer")
    with store.connection() as conn:
        first = queue_api.create_reanalysis_request(conn, conversation_id=conversation["id"], kind=ReanalysisKind.SPEAKER_CORRECTION,
                                                    requested_by=actor, idempotency_key="speaker-fix-1", speaker_correction=correction, reason="agent label")
        again = queue_api.create_reanalysis_request(conn, conversation_id=conversation["id"], kind=ReanalysisKind.SPEAKER_CORRECTION,
                                                    requested_by=actor, idempotency_key="speaker-fix-1", speaker_correction=correction, reason="agent label")
        plain = queue_api.create_reanalysis_request(conn, conversation_id=conversation["id"], kind=ReanalysisKind.SPEAKER_CORRECTION,
                                                    requested_by=actor, speaker_correction=correction)
        groups = queue_api.reanalysis_pending_groups(conn, conversation["id"])
    assert again == first and plain.id != first.id and first.speaker_correction == correction and first.requested_by_account_id == "acct_reviewer"
    assert ResultKind.TRANSCRIPT not in groups and ResultKind.QA in groups
