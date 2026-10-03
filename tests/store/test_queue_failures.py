"""Failure classes, backoff and attempt limits, lease expiry, late usage, release, retry, cancel."""

from __future__ import annotations

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.common import ServiceScope
from call1.contracts.jobs import JobType

from .test_queue_harness import V, hooks, job, pinned, q, rubrics, upstream_input, usage  # noqa: F401


def test_transient_failures_back_off_then_exhaust_the_attempts(q, hooks):
    setup = q.ingest()
    asr = setup["ids"]["asr"]
    first = q.claim_one(asr)
    receipt = q.fail(first, "provider_timeout")
    assert receipt["status"] == "QUEUED" and receipt["error_class"] == "transient" and receipt["attempts_remaining"] == 2
    assert receipt["next_run_at"] == "2026-09-25T12:00:30Z" and receipt["released_job_ids"] == []
    queued = q.job(asr)
    assert queued["waiting_reason"] == "retry_backoff" and queued["error_code"] == "provider_timeout" and queued["attempt_count"] == 1
    assert q.claim(job_types=[JobType.ASR])["jobs"] == []  # still backing off
    q.clock.advance(30)
    second = q.claim_one(asr)
    assert second["attempt_number"] == 2 and second["final_attempt"] is False
    assert q.fail(second, "provider_error")["next_run_at"] == "2026-09-25T12:01:30Z"  # 30 s x 2
    q.clock.advance(60)
    third = q.claim_one(asr)
    assert third["attempt_number"] == 3 and third["final_attempt"] is True
    last = q.fail(third, "provider_error")
    assert last["status"] == "FAILED" and last["attempts_remaining"] == 0 and last["next_run_at"] is None
    failed = q.job(asr)
    assert failed["status"] == "FAILED" and failed["attempt_count"] == 3 and failed["claim_count"] == 3
    assert [f.status.value for f in hooks.failures] == ["QUEUED", "QUEUED", "FAILED"]
    assert [f.terminal for f in hooks.failures] == [False, False, True]
    assert len(q.get(f"/conversations/{setup['conversation_id']}/usage").json()["items"]) == 3


def test_failure_replays_and_key_reuse(q):
    setup = q.ingest()
    claimed = q.claim_one(setup["ids"]["asr"])
    body = q.failure(claimed, "provider_error", key="fail-key-0001")
    first = q.post(f"/jobs/{setup['ids']['asr']}/fail", body).json()
    assert q.post(f"/jobs/{setup['ids']['asr']}/fail", body).json() == dict(first, replayed=True)
    assert q.post(f"/jobs/{setup['ids']['asr']}/fail", dict(body, error_detail="other"), expect=409).json()["code"] == "completion_key_reused"
    completion = q.completion(claimed, [], key="fail-key-0001")
    completion["outputs"] = [{"role": "transcript", "artifact_id": setup["audio"]["id"], "checksum": setup["audio"]["checksum"]}]
    assert q.post(f"/jobs/{setup['ids']['asr']}/complete", completion, expect=409).json()["code"] == "completion_key_reused"


def test_configuration_errors_fail_at_once_and_dead_block_dependents(q):
    setup = q.ingest()
    ids = setup["ids"]
    receipt = q.fail(q.claim_one(ids["asr"]), "model_unavailable")
    assert receipt["status"] == "FAILED" and receipt["error_class"] == "configuration" and receipt["attempts_remaining"] == 2
    enrich = q.job(ids["enrich"])
    assert enrich["status"] == "BLOCKED" and enrich["waiting_reason"] == "dead_blocked"
    assert enrich["blocking"] == [{"job_id": ids["asr"], "job_type": "asr", "status": "FAILED", "edge": "requires", "dead": True}]


def test_after_edges_release_merges_when_an_upstream_fails(q):
    conversation = q.register()
    transcript = q.inline(conversation["id"], ArtifactKind.TRANSCRIPT).json()
    graph = q.graph(conversation["id"], [
        job("life", JobType.CONTACT_SIGNALS_LIFECYCLE, inputs=[pinned("transcript", transcript)], priority=2),
        job("res", JobType.CONTACT_SIGNALS_RESOLUTION, inputs=[pinned("transcript", transcript)], priority=1),
        job("merge", JobType.CONTACT_SIGNALS_MERGE, after=["life", "res"], inputs=[
            upstream_input("lifecycle", "life", "pass", optional=True), upstream_input("resolution", "res", "pass", optional=True)]),
    ])
    ids = q.ids(graph)
    q.run(ids["life"], result=None)
    assert q.status(ids["merge"]) == "BLOCKED"
    failed = q.fail(q.claim_one(ids["res"]), "context_limit_exceeded")
    assert failed["status"] == "FAILED" and failed["released_job_ids"] == [ids["merge"]]
    merge = q.claim_one(ids["merge"])
    assert [(i["role"], i["artifact"] is None) for i in merge["inputs"]] == [("lifecycle", False), ("resolution", True)]
    assert {(u["job_id"], u["edge"], u["status"], u["error_code"]) for u in merge["upstream"]} == {
        (ids["life"], "after", "SUCCEEDED", None), (ids["res"], "after", "FAILED", "context_limit_exceeded")}
    assert [r["role"] for r in merge["job"]["resolved_inputs"]] == ["lifecycle"]


def test_lease_expiry_requeues_synthesizes_usage_and_makes_the_token_stale(q, hooks):
    setup = q.ingest()
    asr = setup["ids"]["asr"]
    claimed = q.claim_one(asr)
    q.clock.advance(300 + 29)
    assert q.heartbeat(claimed)["lease_expires_at"] == "2026-09-25T12:10:29Z"  # still inside the grace: renewed
    q.clock.advance(300 + 31)
    assert q.claim(job_types=[JobType.VALIDATION_VAD], max_jobs=1)["jobs"]  # any claim call sweeps expired leases
    expired = q.job(asr)
    assert expired["status"] == "QUEUED" and expired["error_code"] == "lease_expired" and expired["waiting_reason"] == "retry_backoff"
    stale = q.heartbeat(claimed, expect=409).json()
    assert stale["code"] == "claim_token_stale" and stale["details"] == {"current_attempt_number": 1, "status": "QUEUED", "your_attempt_outcome": "lease_expired"}
    completion = q.complete(claimed, outputs=[{"role": "transcript", "artifact_id": setup["audio"]["id"], "checksum": setup["audio"]["checksum"]}], expect=409)
    assert completion.json()["details"]["your_attempt_outcome"] == "lease_expired"
    [row] = q.get(f"/conversations/{setup['conversation_id']}/usage").json()["items"]
    assert row["outcome"] == "abandoned" and row["recorded_by"] == "store_synthesized" and row["error_code"] == "lease_expired"
    assert row["tokens_input"] == {"count": None, "source": "unavailable"} and row["hardware_profile_id"] == "hw_test" and row["total_seconds"] == 629
    [attempt] = q.get(f"/jobs/{asr}/attempts").json()["items"]
    assert attempt["status"] == "lease_expired" and attempt["usage_record_id"] == row["id"]
    assert hooks.failures[-1].error_code.value == "lease_expired" and hooks.failures[-1].terminal is False


def test_lease_expiry_on_the_last_attempt_fails_the_job(q):
    conversation = q.register()
    audio = q.upload_audio(conversation["id"])
    graph = q.graph(conversation["id"], [job("asr", JobType.ASR, max_attempts=1, inputs=[pinned("audio", audio)])])
    asr = q.ids(graph)["asr"]
    claimed = q.claim_one(asr)
    assert claimed["final_attempt"] is True
    q.clock.advance(331)
    assert q.heartbeat(claimed, expect=409).json()["details"]["your_attempt_outcome"] == "lease_expired"  # the heartbeat itself expired it
    failed = q.job(asr)
    assert failed["status"] == "FAILED" and failed["error_code"] == "lease_expired"


def test_lease_expiry_after_cancel_cancels(q):
    setup = q.ingest()
    q.claim_one(setup["ids"]["asr"])
    q.cancel(setup["ids"]["asr"], cascade=False)
    q.clock.advance(331)
    q.claim(max_jobs=1)
    assert q.status(setup["ids"]["asr"]) == "CANCELLED"
    assert q.get(f"/jobs/{setup['ids']['asr']}/attempts").json()["items"][0]["status"] == "lease_expired"


def test_a_late_worker_attaches_its_usage_once(q):
    setup = q.ingest()
    asr = setup["ids"]["asr"]
    claimed = q.claim_one(asr)
    q.clock.advance(331)
    q.claim(max_jobs=1, job_types=[JobType.VALIDATION_VAD])
    late = {"claim_token": claimed["claim_token"], "usage": usage(inference_seconds=42.0)}
    record = q.post(f"/jobs/{asr}/attempts/1/usage", late).json()
    assert record["recorded_by"] == "process_late" and record["outcome"] == "abandoned" and record["inference_seconds"] == 42.0
    assert record["tokens_input"] == {"count": 100, "source": "local_tokenizer"} and record["error_code"] == "lease_expired"
    assert q.post(f"/jobs/{asr}/attempts/1/usage", late).json() == record  # a replay
    again = q.post(f"/jobs/{asr}/attempts/1/usage", dict(late, usage=usage(inference_seconds=1.0)), expect=409).json()
    assert again["code"] == "conflict" and again["details"]["reason"] == "already_attached"
    wrong = q.post(f"/jobs/{asr}/attempts/1/usage", dict(late, claim_token="q" * 43), expect=409).json()
    assert wrong["code"] == "claim_token_stale"
    q.clock.advance(30)
    running = q.claim_one(asr)
    not_expired = q.post(f"/jobs/{asr}/attempts/2/usage", {"claim_token": running["claim_token"], "usage": usage()}, expect=409).json()
    assert not_expired["details"]["reason"] == "not_lease_expired"
    assert q.post(f"/jobs/{asr}/attempts/9/usage", late, expect=404).json()["code"] == "not_found"


def test_release_requeue_refunds_the_attempt(q):
    setup = q.ingest()
    asr = setup["ids"]["asr"]
    claimed = q.claim_one(asr)
    receipt = q.release(claimed, not_before="2026-09-25T12:05:00Z", key="release-key-1")
    assert receipt["status"] == "QUEUED" and receipt["attempt_count"] == 0 and receipt["next_run_at"] == "2026-09-25T12:05:00Z"
    replay = q.release(claimed, not_before="2026-09-25T12:05:00Z", key="release-key-1")
    assert replay == dict(receipt, replayed=True)
    job_ = q.job(asr)
    assert job_["attempt_count"] == 0 and job_["claim_count"] == 1 and job_["waiting_reason"] == "deferred_by_worker"
    [attempt] = q.get(f"/jobs/{asr}/attempts").json()["items"]
    assert attempt["status"] == "released" and attempt["counts_as_attempt"] is False and attempt["usage_record_id"] is None
    assert q.get(f"/conversations/{setup['conversation_id']}/usage").json()["items"] == []
    assert q.claim(job_types=[JobType.ASR])["jobs"] == []
    q.clock.advance(300)
    again = q.claim_one(asr)
    assert again["attempt_number"] == 2 and again["job"]["attempt_count"] == 1  # attempt numbers are never reused


def test_release_reject_fails_without_consuming_an_attempt(q, hooks):
    setup = q.ingest()
    claimed = q.claim_one(setup["ids"]["asr"])
    receipt = q.release(claimed, "reject", "model_unavailable")
    assert receipt["status"] == "FAILED" and receipt["attempt_count"] == 0
    failed = q.job(setup["ids"]["asr"])
    assert failed["status"] == "FAILED" and failed["error_code"] == "model_unavailable" and failed["attempt_count"] == 0
    assert hooks.failures[-1].status.value == "FAILED"
    wrong = q.release(q.claim_one(setup["ids"]["vad"]), "reject", "resource_unavailable", expect=422).json()
    assert wrong["code"] == "validation_failed"


def test_release_after_cancel_is_an_invalid_transition(q):
    setup = q.ingest()
    claimed = q.claim_one(setup["ids"]["asr"])
    q.cancel(setup["ids"]["asr"])
    refused = q.release(claimed, expect=409).json()
    assert refused["code"] == "invalid_transition" and refused["details"]["reason"] == "cancel_requested"


def test_manual_retry_requeues_one_failed_job(q, store):
    setup = q.ingest()
    ids = setup["ids"]
    q.fail(q.claim_one(ids["asr"]), "model_unavailable")
    retried = q.retry(ids["asr"])
    assert retried["status"] == "QUEUED" and retried["max_attempts"] == 4 and retried["retry_generation"] == 1 and retried["error_code"] is None
    assert q.job(ids["enrich"])["waiting_reason"] == "waiting_for_dependencies"  # no longer dead-blocked
    assert q.retry(ids["asr"], expect=409).json()["code"] == "invalid_transition"
    events = q.get("/admin/audit", headers=q.mint_session("admin").read_headers, params={"action": "job_retried"}).json()["items"]
    assert events[0]["target"] == {"kind": "job", "id": ids["asr"]} and events[0]["actor"]["kind"] == "process_service"
    controller = q.mint_service_key([ServiceScope.JOBS_WRITE])
    assert q.post(f"/jobs/{ids['asr']}/retry", {"reason": "x"}, headers=controller.headers, expect=403).json()["code"] == "insufficient_scope"


def test_retrying_a_merge_waits_for_its_retried_upstream(q):
    conversation = q.register()
    transcript = q.inline(conversation["id"], ArtifactKind.TRANSCRIPT).json()
    graph = q.graph(conversation["id"], [
        job("life", JobType.CONTACT_SIGNALS_LIFECYCLE, inputs=[pinned("transcript", transcript)]),
        job("merge", JobType.CONTACT_SIGNALS_MERGE, after=["life"], inputs=[upstream_input("lifecycle", "life", "pass", optional=True)]),
    ])
    ids = q.ids(graph)
    q.fail(q.claim_one(ids["life"]), "credential_missing")
    q.release(q.claim_one(ids["merge"]), "reject", "configuration_error")
    q.retry(ids["life"])
    assert q.retry(ids["merge"])["status"] == "BLOCKED"
    q.run(ids["life"], result=None)
    assert q.status(ids["merge"]) == "QUEUED"


def test_cancel_cascades_or_releases_after_edges(q, hooks):
    setup = q.ingest()
    ids = setup["ids"]
    response = q.cancel(ids["asr"])
    assert response["job"]["status"] == "CANCELLED" and set(response["cancelled_job_ids"]) == {ids["asr"], ids["enrich"], ids["sentiment"]}
    assert {f.job.job_id for f in hooks.failures} == {ids["asr"], ids["enrich"], ids["sentiment"]}
    assert q.cancel(ids["asr"], expect=409).json()["code"] == "invalid_transition"
    assert q.retry(ids["asr"], expect=409).json()["code"] == "invalid_transition"  # CANCELLED is final

    conversation = q.register()
    transcript = q.inline(conversation["id"], ArtifactKind.TRANSCRIPT).json()
    graph = q.graph(conversation["id"], [
        job("life", JobType.CONTACT_SIGNALS_LIFECYCLE, inputs=[pinned("transcript", transcript)]),
        job("merge", JobType.CONTACT_SIGNALS_MERGE, after=["life"], inputs=[upstream_input("lifecycle", "life", "pass", optional=True)]),
    ])
    merge_ids = q.ids(graph)
    kept = q.cancel(merge_ids["life"], cascade=False)
    assert kept["cancelled_job_ids"] == [merge_ids["life"]] and q.status(merge_ids["merge"]) == "QUEUED"
    audit = q.get("/admin/audit", headers=q.mint_session("admin").read_headers, params={"action": "job_cancelled"}).json()["items"]
    assert {e["target"]["id"] for e in audit} == {ids["asr"], merge_ids["life"]}


def test_cancel_of_a_running_job_cascades_to_its_dependents_now(q):
    setup = q.ingest()
    ids = setup["ids"]
    claimed = q.claim_one(ids["asr"])
    response = q.cancel(ids["asr"])
    assert response["job"]["status"] == "RUNNING" and response["job"]["cancel_requested"] is True
    assert set(response["cancelled_job_ids"]) == {ids["enrich"], ids["sentiment"]}
    assert q.fail(claimed, "provider_error")["status"] == "CANCELLED"  # a failure after cancel acknowledges it
