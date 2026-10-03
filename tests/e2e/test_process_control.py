"""Process features p6-p8 (inventory area "process"), end to end against real servers: retry and
cancel through the operator API and the jobs:control scope, the completion spool across a real
Store outage and a Process restart, and the reanalysis consumer for every kind.

Each test is named after its inventory feature ID.
"""

from __future__ import annotations

import time

import pytest

from .process_helpers import attempts_of, by_type, graph, job, jobs_of, mono_copy, one, wait_job, wait_job_of_type

pytestmark = pytest.mark.e2e

FAST_LEASES = {"heartbeat_interval_seconds": 5, "lease_duration_seconds": 30}


# --- p6: retry and cancel ---------------------------------------------------------------------------


def test_p6_retry_and_cancel_through_the_operator_api_then_403_without_jobs_control(stack_factory):
    private = stack_factory(name="p6", store_parameters=FAST_LEASES,
                            fake_behavior={"acoustic_tone": ["fail:validation_rejected"], "asr": ["ok", "hold:60"]})

    # A: tone fails terminally (validation_rejected), the rest of the call settles.
    a = private.ingest("call_01_compliant", agent_id="agent-p6-a")
    private.wait_until_settled(a["call_id"])
    tone = one(jobs_of(private, a["conversation_id"]), "acoustic_tone")
    assert tone["status"] == "FAILED" and tone["error_code"] == "validation_rejected", tone

    # Writes need the console credential.
    anonymous = private.process_post(f"/jobs/{tone['id']}/retry", {"reason": "no token"}, token=False)
    assert anonymous.status_code == 401 and anonymous.json()["code"] == "console_credential_invalid", anonymous.text

    retried = private.process_post(f"/jobs/{tone['id']}/retry", {"reason": "e2e: retry the failed tone job"})
    assert retried.status_code == 200, retried.text
    assert retried.json()["job"]["retry_generation"] == 1, retried.json()["job"]
    done = wait_job(private, tone["id"], lambda j: j["status"] == "SUCCEEDED", timeout=30, what="retried tone succeeds")
    assert done["attempt_count"] == 2
    assert [x["status"] for x in attempts_of(private, tone["id"])] == ["failed", "succeeded"]

    # B: ASR holds; cancelling it with cascade cancels everything downstream.
    b = private.ingest("call_01_compliant", agent_id="agent-p6-b")
    asr = wait_job_of_type(private, b["conversation_id"], "asr", lambda j: j["status"] == "RUNNING")
    cancelled = private.process_post(f"/jobs/{asr['id']}/cancel", {"reason": "e2e: cancel with dependents", "cascade": True})
    assert cancelled.status_code == 200, cancelled.text
    body = cancelled.json()
    downstream = {j["id"] for j in jobs_of(private, b["conversation_id"]) if j["id"] != asr["id"] and j["job_type"] != "validation_vad"}
    assert downstream <= set(body["cancelled_job_ids"]), f"cascade left {sorted(downstream - set(body['cancelled_job_ids']))}"
    wait_job(private, asr["id"], lambda j: j["status"] == "CANCELLED", timeout=15, what="held ASR cancelled")
    progress = private.wait_until_settled(b["call_id"])
    assert progress["settled"] is True

    # The same key without jobs:control: re-issue Process's key with the default scopes.
    private.stop_process()
    private.store_cli("issue-service-key", "--installation", f"e2e-{private.name}", "--config", str(private.process_config_path))
    private.start_process()
    for path, payload in ((f"/jobs/{tone['id']}/retry", {"reason": "no scope"}), (f"/jobs/{asr['id']}/cancel", {"reason": "no scope"})):
        refused = private.process_post(path, payload)
        assert refused.status_code == 403, refused.text
        error = refused.json()
        assert error["code"] == "insufficient_scope", error
        assert "jobs:control" in error["message"] and "issue-service-key" in error["message"], error["message"]


# --- p7: the completion spool -----------------------------------------------------------------------


def _wait_spooled(stack, job_id, timeout=20.0):
    """The finished job's completion is in Process's spool and its claim is listed as awaiting delivery."""
    deadline = time.monotonic() + timeout
    worker = {}
    while time.monotonic() < deadline:
        worker = stack.process_get("/overview").json()["worker"]
        if worker["spooled"] >= 1 and any(r["job_id"] == job_id for r in worker["awaiting_delivery"]):
            return worker
        time.sleep(0.2)
    raise AssertionError(
        f"the held job {job_id} finished during the Store outage but no completion was spooled "
        f"(spooled={worker.get('spooled')}, awaiting_delivery={worker.get('awaiting_delivery')}, running={worker.get('running')}); "
        f"Process log: {[l for l in stack.log_tail('process', 200).splitlines() if job_id in l and 'upload' in l][-1:]}")


def test_p7_completion_spooled_during_a_store_outage_is_delivered_without_redoing_work(stack_factory):
    """Store goes away while a job runs and stays away past the job's end (and past a heartbeat
    interval), then comes back well inside the lease: the finished work is delivered once, not redone."""
    private = stack_factory(name="p7-outage", store_parameters=dict(FAST_LEASES, lease_grace_seconds=0),
                            fake_behavior={"acoustic_tone": ["hold:8"]})
    receipt = private.ingest("call_01_compliant", agent_id="agent-p7")
    tone = wait_job_of_type(private, receipt["conversation_id"], "acoustic_tone", lambda j: j["status"] == "RUNNING")
    private.stop_store()
    still_running = [r["job_id"] for r in private.process_get("/overview").json()["worker"]["running"]]
    assert tone["id"] in still_running, "setup: the held job must still be running when Store goes away"
    # The handler finishes ~8 s in; the client's upload retries give up ~6 s later. Store returns ~21 s in:
    # still inside the 30 s lease the last heartbeat (at most 5 s in) renewed.
    time.sleep(20)
    during = private.process_get("/overview").json()["worker"]
    private.start_store()
    done = wait_job(private, tone["id"], lambda j: j["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"), timeout=90, what="tone delivered")
    assert done["status"] == "SUCCEEDED", done
    attempts = [x["status"] for x in attempts_of(private, tone["id"])]
    assert attempts == ["succeeded"], (
        f"the work was redone after an outage shorter than the lease: attempts {attempts}; during the outage Process had "
        f"spooled={during['spooled']} awaiting_delivery={during['awaiting_delivery']} running={[r['job_id'] for r in during['running']]}")
    private.wait_for(lambda: private.process_get("/overview").json()["worker"]["spooled"] == 0, timeout=20, what="spool drained")
    private.wait_until_settled(receipt["call_id"])


def test_p7_spool_is_replayed_after_a_process_restart(stack_factory):
    private = stack_factory(name="p7-restart", store_parameters=FAST_LEASES, fake_behavior={"acoustic_tone": ["hold:8"]})
    receipt = private.ingest("call_01_compliant", agent_id="agent-p7r")
    tone = wait_job_of_type(private, receipt["conversation_id"], "acoustic_tone", lambda j: j["status"] == "RUNNING")
    private.stop_store()
    assert tone["id"] in [r["job_id"] for r in private.process_get("/overview").json()["worker"]["running"]], "setup: held job still running"
    _wait_spooled(private, tone["id"])
    private.stop_process()
    spool = list((private.process_dir / "data" / "spool").rglob("*"))
    assert any(p.is_file() for p in spool), f"nothing on disk in the spool: {spool}"
    private.start_store()
    private.start_process()
    done = wait_job(private, tone["id"], lambda j: j["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"), timeout=30, what="replayed tone")
    assert done["status"] == "SUCCEEDED", done
    assert [x["status"] for x in attempts_of(private, tone["id"])] == ["succeeded"]
    private.wait_until_settled(receipt["call_id"])


def test_p7_a_job_cancelled_while_its_completion_waits_becomes_a_cancelled_failure(stack_factory):
    private = stack_factory(name="p7-cancel", store_parameters=FAST_LEASES, fake_behavior={"acoustic_tone": ["hold:8"]})
    receipt = private.ingest("call_01_compliant", agent_id="agent-p7c")
    tone = wait_job_of_type(private, receipt["conversation_id"], "acoustic_tone", lambda j: j["status"] == "RUNNING")
    private.stop_store()
    assert tone["id"] in [r["job_id"] for r in private.process_get("/overview").json()["worker"]["running"]], "setup: held job still running"
    _wait_spooled(private, tone["id"])
    private.stop_process()
    private.start_store()
    response = private.store_post(f"/jobs/{tone['id']}/cancel", {"reason": "e2e: cancelled while spooled", "cascade": False},
                                  session="service", idempotency_key=True)
    assert response.status_code == 200, response.text
    private.start_process()
    done = wait_job(private, tone["id"], lambda j: j["status"] in ("SUCCEEDED", "FAILED", "CANCELLED"), timeout=30, what="cancelled tone")
    assert done["status"] == "CANCELLED", done
    attempts = attempts_of(private, tone["id"])
    assert attempts[-1]["status"] == "cancelled" and attempts[-1]["error_code"] == "cancelled", attempts


# --- p8: the reanalysis consumer ----------------------------------------------------------------


def _request(session, call_id, kind, **extra):
    response = session.post(f"/calls/{call_id}/reanalysis-requests", json={"kind": kind, **extra}, idempotency_key=True)
    assert response.status_code == 201, response.text
    return response.json()


def _fulfilled(stack, session, request_id, timeout=30.0):
    def probe():
        body = session.get(f"/reanalysis-requests/{request_id}").json()
        return body if body["status"] in ("fulfilled", "rejected") else None
    return stack.wait_for(probe, timeout=timeout, what=f"reanalysis {request_id} handled")


def _graph_types(stack, graph_id, conversation_id):
    g = graph(stack, graph_id)
    assert g["reason"] == "reanalysis", g["reason"]
    ids = {j["job_id"] for j in g["jobs"]}
    return sorted(j["job_type"] for j in g["jobs"]), ids


def _settle_graph(stack, conversation_id, ids, timeout=60.0):
    def probe():
        jobs = [j for j in jobs_of(stack, conversation_id) if j["id"] in ids]
        return jobs if all(j["status"] in ("SUCCEEDED", "FAILED", "CANCELLED") for j in jobs) else None
    jobs = stack.wait_for(probe, timeout=timeout, what="reanalysis graph settled")
    bad = [(j["job_type"], j["status"], j.get("error_code")) for j in jobs if j["status"] != "SUCCEEDED"]
    assert not bad, bad
    return jobs


def test_p8_every_reanalysis_kind_plans_its_job_subset(stack_factory):
    private = stack_factory(name="p8", process_config={"summary_batch_turns": 60})
    admin = private.admin()
    receipt = private.ingest(mono_copy(private), unique=False, agent_id="agent-p8")
    private.wait_until_settled(receipt["call_id"])
    call, conv = receipt["call_id"], receipt["conversation_id"]
    criteria = private.store_get("/rubrics/call1_standard_v2", session="service").json()["definition"]["criteria"]
    semantic = sum(1 for c in criteria if c["check"]["check_type"] == "semantic_judgement")
    det = 1 if any(c["check"]["check_type"] != "semantic_judgement" for c in criteria) else 0
    qa = ["qa_criterion"] * semantic + ["qa_deterministic"] * det + ["qa_scorecard"]
    cs = ["contact_signals_lifecycle", "contact_signals_merge", "contact_signals_resolution"]
    expected = {
        "qa": sorted(qa),
        "summary": sorted(["summary_segment", "summary_assembly"]),
        "contact_signals": sorted(cs),
    }
    for kind, types in expected.items():
        request = _request(admin, call, kind, note=f"e2e {kind}")
        handled = _fulfilled(private, admin, request["id"])
        assert handled["status"] == "fulfilled" and handled["graph_id"], handled
        planned, ids = _graph_types(private, handled["graph_id"], conv)
        assert planned == types, f"{kind}: planned {planned}, expected {types}"
        _settle_graph(private, conv, ids)
        detail = admin.get(f"/calls/{call}").json()
        assert all(g["state"] == "available" for g in detail["results"] if g["kind"] in ("qa", "summary", "contact_signals")), detail["results"]

    # speaker_correction: a code-stage attribution job, then everything downstream of it.
    review = admin.get(f"/calls/{call}/review").json()
    turn = admin.get(f"/calls/{call}/transcript").json()["turns"][0]["turn_id"]
    corrected = admin.post(f"/calls/{call}/speaker-corrections",
                           json={"correction": {"turn_id": turn, "speaker": "CALLER"}, "expected_version": review["review_version"]})
    assert corrected.status_code in (200, 201), corrected.text
    handled = _fulfilled(private, admin, corrected.json()["id"])
    assert handled["status"] == "fulfilled", handled
    planned, ids = _graph_types(private, handled["graph_id"], conv)
    assert planned == sorted(["speaker_attribution", "acoustic_tone", "text_sentiment", *qa, "summary_segment", "summary_assembly", *cs]), planned
    jobs = _settle_graph(private, conv, ids)
    assert one(jobs, "speaker_attribution")["parameters"]["speaker_correction"]["turn_id"] == turn

    # full: the ingest graph again, from the source audio.
    request = _request(admin, call, "full")
    handled = _fulfilled(private, admin, request["id"])
    assert handled["status"] == "fulfilled", handled
    planned, ids = _graph_types(private, handled["graph_id"], conv)
    ingest_types = sorted(j["job_type"] for j in graph(private, receipt["graph_id"])["jobs"] if not j["ref"].startswith("sum-"))
    assert sorted(t for t in planned if t != "summary_segment") == ingest_types, (planned, ingest_types)
    _settle_graph(private, conv, ids)

    # qa_draft_test: the rubric's draft, scored into draft:<request>: slots, readable on the request.
    published = private.store_get("/rubrics/call1_standard_v2", session="service").json()
    definition = dict(published["definition"], description="e2e draft for a draft test")
    saved = admin.put("/rubrics/call1_standard_v2/draft", json={"definition": definition, "expected_draft_revision": 0})
    assert saved.status_code in (200, 201), saved.text
    draft = saved.json()
    test = admin.post("/rubrics/call1_standard_v2/draft/tests", json={"call_id": call, "expected_draft_revision": draft["draft_revision"]},
                      idempotency_key=True)
    assert test.status_code in (200, 201), test.text
    handled = _fulfilled(private, admin, test.json()["id"])
    assert handled["status"] == "fulfilled", handled
    planned, ids = _graph_types(private, handled["graph_id"], conv)
    assert planned == sorted(qa), planned
    result = private.wait_for(lambda: (lambda r: r if r["state"] != "pending" else None)(admin.get(f"/reanalysis-requests/{handled['id']}/draft-result").json()),
                              timeout=30, what="draft result")
    assert result["state"] == "available" and result["scorecard"], result
    overview = private.process_get("/overview").json()["reanalysis"]
    assert overview["handled"] == 6 and overview["rejected"] == 0, overview


def test_p8_a_request_that_cannot_be_planned_is_rejected(stack_factory):
    private = stack_factory(name="p8-reject", fake_behavior={"asr": ["fail:validation_rejected"]})
    admin = private.admin()
    receipt = private.ingest("call_01_compliant", agent_id="agent-p8r")
    private.wait_until_settled(receipt["call_id"])
    assert one(jobs_of(private, receipt["conversation_id"]), "asr")["status"] == "FAILED"
    request = _request(admin, receipt["call_id"], "summary")
    handled = _fulfilled(private, admin, request["id"])
    assert handled["status"] == "rejected", handled
    assert "transcript" in (handled["rejected_reason"] or ""), handled["rejected_reason"]
    assert handled["graph_id"] is None
    overview = private.process_get("/overview").json()["reanalysis"]
    assert overview["rejected"] == 1 and "rejected" in (overview["last_error"] or ""), overview
