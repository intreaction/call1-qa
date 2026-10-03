"""Job graphs, atomic claims, heartbeats and idempotent completion (one transaction)."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.catalog import ModelPurpose
from call1.contracts.common import ServiceScope
from call1.contracts.events import ChangeKind
from call1.contracts.jobs import JobType
from call1.store import db, feed
from call1.store.queue import lifecycle

from .test_queue_harness import (  # noqa: F401
    CRITERION,
    RUBRIC_REF,
    V,
    content,
    hooks,
    job,
    pinned,
    q,
    resource,
    rubrics,
    selection,
    upstream_input,
)


# --- graphs -----------------------------------------------------------------------------------


def test_graph_starts_ready_jobs_queued_and_dependents_blocked(q):
    setup = q.ingest()
    graph, ids = setup["graph"], setup["ids"]
    assert graph["created"] is True and graph["reason"] == "ingest"
    statuses = {j["ref"]: j["status"] for j in graph["jobs"]}
    assert statuses == {"vad": "QUEUED", "asr": "QUEUED", "enrich": "BLOCKED", "sentiment": "BLOCKED"}
    assert {(e["job_id"], e["upstream_job_id"], e["kind"]) for e in graph["edges"]} == {
        (ids["enrich"], ids["asr"], "requires"), (ids["sentiment"], ids["asr"], "requires")}
    enrich = q.job(ids["enrich"])
    assert enrich["requires_job_ids"] == [ids["asr"]] and enrich["waiting_reason"] == "waiting_for_dependencies"
    assert enrich["blocking"] == [{"job_id": ids["asr"], "job_type": "asr", "status": "QUEUED", "edge": "requires", "dead": False}]
    assert enrich["inputs"] == [{"role": "transcript", "artifact": None, "upstream": {"ref": None, "job_id": ids["asr"], "output_role": "transcript"}, "optional": False}]
    asr = q.job(ids["asr"])
    assert asr["resolved_inputs"] == [{"role": "audio", "artifact_id": setup["audio"]["id"], "checksum": setup["audio"]["checksum"]}]
    assert asr["max_attempts"] == 3 and asr["execution_class"] == "primary_host" and asr["waiting_reason"] == "waiting_for_worker"
    assert q.get(f"/job-graphs/{graph['graph_id']}").json()["jobs"] == graph["jobs"]


def test_graph_creation_is_idempotent_by_key(q):
    conversation = q.register()
    audio = q.upload_audio(conversation["id"])
    jobs = [job("vad", JobType.VALIDATION_VAD, inputs=[pinned("audio", audio)])]
    first = q.graph(conversation["id"], jobs, key="ingest-graph-1")
    again = q.graph(conversation["id"], jobs, key="ingest-graph-1")
    assert again["created"] is False and again["graph_id"] == first["graph_id"] and again["jobs"] == first["jobs"]
    changed = [job("vad", JobType.VALIDATION_VAD, inputs=[pinned("audio", audio)], priority=9)]
    reused = q.graph(conversation["id"], changed, key="ingest-graph-1", expect=409).json()
    assert reused["code"] == "idempotency_key_reused" and reused["details"]["original_id"] == first["graph_id"]
    # the same job key in a new graph: an identical definition is the existing job, a different one is refused
    replayed = q.graph(conversation["id"], jobs, key="ingest-graph-2")
    assert replayed["created"] is True and replayed["jobs"][0]["job_id"] == first["jobs"][0]["job_id"]
    clash = q.graph(conversation["id"], changed, key="ingest-graph-3", expect=409).json()
    assert clash["code"] == "idempotency_key_reused" and clash["details"]["original_id"] == first["jobs"][0]["job_id"]


def test_graph_validation(q):
    conversation, other = q.register(), q.register()
    audio = q.upload_audio(conversation["id"])
    other_graph = q.graph(other["id"], [job("vad", JobType.VALIDATION_VAD, inputs=[pinned("audio", q.upload_audio(other["id"]))])])
    foreign = other_graph["jobs"][0]["job_id"]
    cross = q.graph(conversation["id"], [job("enrich", JobType.ENRICHMENT, requires_ids=[foreign])], expect=422).json()
    assert cross["code"] == "graph_invalid" and cross["details"]["reason"] == "cross_conversation"
    unknown = q.graph(conversation["id"], [job("enrich", JobType.ENRICHMENT, requires_ids=["job_nope"])], expect=422).json()
    assert unknown["details"]["reason"] == "unknown_job"
    cycle = q.graph(conversation["id"], [job("a", JobType.ENRICHMENT, requires=["b"]), job("b", JobType.ENRICHMENT, requires=["a"])], expect=422).json()
    assert cycle["code"] == "validation_failed"  # the contract model refuses cycles before Store sees the graph
    mine = q.graph(conversation["id"], [job("asr", JobType.ASR, inputs=[pinned("audio", audio)])])
    bad_role = q.graph(conversation["id"], [job("enrich", JobType.ENRICHMENT, requires_ids=[mine["jobs"][0]["job_id"]],
                                                inputs=[{"role": "t", "upstream": {"job_id": mine["jobs"][0]["job_id"], "output_role": "nope"}}])], expect=422).json()
    assert bad_role["details"]["reason"] == "unknown_output_role"
    wrong_checksum = q.graph(conversation["id"], [job("vad2", JobType.VALIDATION_VAD, inputs=[{"role": "audio", "artifact": {
        "artifact_id": audio["id"], "checksum": "sha256:" + "1" * 64}}])], expect=422).json()
    assert wrong_checksum["code"] == "checksum_mismatch"
    lan = selection(ModelPurpose.SEMANTIC_QA)
    lan["route"] = {"route_class": "customer_lan", "provider_type": "ollama", "destination_host": "llm.example.com", "masked": True,
                    "provider_connection_ref": "lan-1"}
    snapshot = q.rubric_snapshot(conversation["id"])
    refused = q.graph(conversation["id"], [job("qa", JobType.QA_CRITERION, sel=lan, inputs=[pinned("rubric", snapshot)])], expect=403).json()
    assert refused["code"] == "route_not_permitted"
    duplicate = q.graph(conversation["id"], [job("x", JobType.ENRICHMENT, key="same-key-1"), job("y", JobType.ENRICHMENT, key="same-key-1")], expect=422).json()
    assert duplicate["details"]["reason"] == "duplicate_idempotency_key"


def test_qa_jobs_pin_a_matching_rubric_snapshot(q):
    conversation = q.register()
    missing = q.graph(conversation["id"], [job("score", JobType.QA_SCORECARD)], expect=422).json()
    assert missing["details"]["reason"] == "rubric_input_missing"
    transcript = q.inline(conversation["id"], ArtifactKind.TRANSCRIPT).json()
    wrong_kind = q.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, inputs=[pinned("rubric", transcript)])], expect=422).json()
    assert wrong_kind["details"]["reason"] == "rubric_input_kind"
    snapshot = q.rubric_snapshot(conversation["id"])
    other_version = dict(RUBRIC_REF.model_dump(mode="json"), version=2)
    mismatch = q.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, inputs=[pinned("rubric", snapshot)], parameters={"rubric": other_version})], expect=422).json()
    assert mismatch["details"]["reason"] == "rubric_snapshot_mismatch"
    ok = q.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, inputs=[pinned("rubric", snapshot)])])
    assert ok["jobs"][0]["status"] == "QUEUED"


# --- claims -----------------------------------------------------------------------------------


def test_claim_orders_by_priority_and_fills_the_offered_slots(q):
    setup = q.ingest()
    ids = setup["ids"]
    response = q.claim(max_jobs=1)
    assert [c["job"]["id"] for c in response["jobs"]] == [ids["asr"]]  # priority 5 first
    claimed = response["jobs"][0]
    assert claimed["attempt_number"] == 1 and claimed["final_attempt"] is False and claimed["slot_offer_index"] == 0
    assert claimed["job"]["status"] == "RUNNING" and claimed["job"]["lease"]["attempt_number"] == 1
    assert claimed["job"]["lease"]["installation_id"] == q.installation_id and claimed["job"]["attempt_count"] == 1
    assert claimed["inputs"] == [{"role": "audio", "artifact": q.get(f"/artifacts/{setup['audio']['id']}").json()}]
    assert response["lease_duration_seconds"] == 300 and response["heartbeat_interval_seconds"] == 100
    cpu_only = q.claim(max_jobs=4, slots=[{"memory_slot": "cpu", "count": 4}])
    assert cpu_only["jobs"] == [] and cpu_only["no_eligible_reason"] == "ready jobs need a slot you did not offer"
    wrong_types = q.claim(job_types=[JobType.SUMMARY_SEGMENT])
    assert wrong_types["jobs"] == [] and wrong_types["no_eligible_reason"] == "ready jobs need capabilities this worker lacks"
    assert [c["job"]["id"] for c in q.claim()["jobs"]] == [ids["vad"]]
    assert q.claim()["no_eligible_reason"] == "no ready jobs"


def test_claims_respect_primary_host_catalog_entries_and_installation(q, mint_service_key):
    setup = q.ingest()
    not_primary = q.claim(primary_host=False)
    assert not_primary["jobs"] == []  # both ready jobs are primary_host
    assert q.claim(entries=[])["jobs"][0]["job"]["job_type"] == "validation_vad"  # asr's frozen entry is not qualified here
    other = mint_service_key([ServiceScope.JOBS_CLAIM], primary_host=False)
    impostor = q.claim(headers=other.headers, installation_id=q.installation_id, expect=403).json()
    assert impostor["code"] == "forbidden" and impostor["details"]["reason"] == "installation_mismatch"
    claim_as_primary = q.claim(headers=other.headers, installation_id=other.installation_id, expect=403).json()
    assert claim_as_primary["details"]["reason"] == "not_primary_host"
    assert q.claim(headers=other.headers, installation_id=other.installation_id, primary_host=False)["jobs"] == []
    assert setup["ids"]["asr"] in [c["job"]["id"] for c in q.claim()["jobs"]]


def test_two_concurrent_claimers_see_one_winner(q):
    setup = q.ingest()
    target = setup["ids"]["asr"]
    barrier = threading.Barrier(2)
    results = []

    def claimer(worker_id):
        barrier.wait()
        results.append(q.claim(max_jobs=1, worker_id=worker_id, job_types=[JobType.ASR]))

    threads = [threading.Thread(target=claimer, args=(f"w{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    winners = [r for r in results if r["jobs"]]
    assert len(winners) == 1 and winners[0]["jobs"][0]["job"]["id"] == target
    assert len([r for r in results if not r["jobs"]]) == 1


def test_many_concurrent_claimers_never_share_a_claim(q, store):
    """Service-level stress: eight connections race for twenty jobs; each job is claimed once."""
    conversation = q.register()
    audio = q.upload_audio(conversation["id"])
    q.graph(conversation["id"], [job(f"v{i}", JobType.VALIDATION_VAD, inputs=[pinned("audio", audio)]) for i in range(20)])
    from call1.contracts.jobs import ClaimRequest

    request = ClaimRequest.model_validate({"worker": q.worker(slots=[{"memory_slot": "local_memory", "count": 3}]), "max_jobs": 3})
    barrier = threading.Barrier(8)
    claimed, errors = [], []

    def claimer():
        barrier.wait()
        with store.connection() as conn:
            try:
                while True:
                    got = lifecycle.claim(conn, store, q.key.principal, request).jobs
                    if not got:
                        return
                    claimed.extend((c.job.id, c.claim_token) for c in got)
            except Exception as exc:  # pragma: no cover - the assertion below reports it
                errors.append(exc)

    threads = [threading.Thread(target=claimer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    job_ids = [job_id for job_id, _ in claimed]
    assert len(job_ids) == 20 and len(set(job_ids)) == 20 and len({t for _, t in claimed}) == 20


# --- heartbeat --------------------------------------------------------------------------------


def test_heartbeat_renews_the_lease_and_rejects_a_stale_token(q):
    setup = q.ingest()
    claimed = q.claim_one(setup["ids"]["asr"])
    q.clock.advance(200)
    beat = q.heartbeat(claimed)
    assert beat["cancel_requested"] is False and beat["lease_expires_at"] == "2026-09-25T12:08:20Z"
    stale = q.heartbeat(dict(claimed, claim_token="z" * 43), expect=409).json()
    assert stale["code"] == "claim_token_stale" and stale["details"] == {"current_attempt_number": 1, "status": "RUNNING", "your_attempt_outcome": None}
    assert q.post("/jobs/job_missing/heartbeat", {"claim_token": "x" * 43}, expect=404).json()["code"] == "not_found"


# --- completion -------------------------------------------------------------------------------


def test_completion_links_outputs_writes_usage_and_releases_dependents(q, hooks):
    setup = q.ingest()
    ids = setup["ids"]
    claimed = q.claim_one(ids["asr"])
    receipt = q.complete(claimed)
    assert receipt["status"] == "SUCCEEDED" and receipt["replayed"] is False and receipt["attempt_number"] == 1
    assert sorted(receipt["released_job_ids"]) == sorted([ids["enrich"], ids["sentiment"]]) and receipt["created_job_ids"] == []
    assert receipt["result_version"] == 1  # what the projection hook returned
    asr = q.job(ids["asr"])
    assert asr["status"] == "SUCCEEDED" and asr["lease"] is None and asr["result_version"] == 1 and asr["completed_at"]
    assert [o["artifact_id"] for o in asr["outputs"]] == receipt["linked_artifact_ids"]
    transcript = q.get(f"/artifacts/{receipt['linked_artifact_ids'][0]}").json()
    enrich = q.job(ids["enrich"])
    assert enrich["status"] == "QUEUED" and enrich["resolved_inputs"] == [{"role": "transcript", "artifact_id": transcript["id"], "checksum": transcript["checksum"]}]
    usage_rows = q.get(f"/conversations/{setup['conversation_id']}/usage").json()["items"]
    assert [u["id"] for u in usage_rows] == [receipt["usage_record_id"]]
    row = usage_rows[0]
    assert row["outcome"] == "succeeded" and row["purpose"] == "asr" and row["route_class"] == "appliance" and row["recorded_by"] == "process"
    assert row["catalog_entry"] == {"entry_id": "mlx-asr", "entry_version": 1} and row["queue_wait_seconds"] == 0
    attempts = q.get(f"/jobs/{ids['asr']}/attempts").json()["items"]
    assert attempts[0]["status"] == "succeeded" and attempts[0]["usage_record_id"] == receipt["usage_record_id"] and attempts[0]["provenance"]["adapter_id"] == "adapter"
    [completion] = hooks.completions
    assert completion.job.job_id == ids["asr"] and completion.job.status.value == "SUCCEEDED" and completion.receipt_id == receipt["receipt_id"]
    assert completion.result.kind.value == "transcript" and [o.role for o in completion.outputs] == ["transcript"]
    assert completion.outputs[0].artifact.linked and completion.outputs[0].artifact.version == 1


def test_dependent_is_released_only_after_every_upstream_succeeds(q):
    conversation = q.register()
    audio = q.upload_audio(conversation["id"])
    graph = q.graph(conversation["id"], [
        job("vad", JobType.VALIDATION_VAD, inputs=[pinned("audio", audio)]),
        job("asr", JobType.ASR, inputs=[pinned("audio", audio)]),
        job("enrich", JobType.ENRICHMENT, requires=["asr", "vad"], inputs=[upstream_input("transcript", "asr", "transcript"),
                                                                          upstream_input("vad", "vad", "vad_metrics")]),
    ])
    ids = q.ids(graph)
    first = q.run(ids["asr"])
    assert first["released_job_ids"] == [] and q.status(ids["enrich"]) == "BLOCKED"
    assert q.job(ids["enrich"])["blocking"] == [{"job_id": ids["vad"], "job_type": "validation_vad", "status": "QUEUED", "edge": "requires", "dead": False}]
    second = q.run(ids["vad"])
    assert second["released_job_ids"] == [ids["enrich"]]
    enrich = q.job(ids["enrich"])
    assert enrich["status"] == "QUEUED" and {r["role"] for r in enrich["resolved_inputs"]} == {"transcript", "vad"}


def test_replayed_completion_returns_the_original_receipt(q):
    setup = q.ingest()
    claimed = q.claim_one(setup["ids"]["asr"])
    outputs = q.outputs_for(claimed)
    body = q.completion(claimed, outputs, key="complete-key-1")
    first = q.post(f"/jobs/{claimed['job']['id']}/complete", body).json()
    q.clock.advance(3600)  # the lease is long gone
    again = q.post(f"/jobs/{claimed['job']['id']}/complete", body).json()
    assert again == dict(first, replayed=True)
    changed = dict(body, usage=dict(body["usage"], inference_seconds=9.0))
    reused = q.post(f"/jobs/{claimed['job']['id']}/complete", changed, expect=409).json()
    assert reused["code"] == "completion_key_reused" and reused["details"]["original_receipt_id"] == first["receipt_id"]
    another = q.post(f"/jobs/{claimed['job']['id']}/complete", dict(body, completion_key="complete-key-2"), expect=409).json()
    assert another["code"] == "claim_token_stale" and another["details"]["your_attempt_outcome"] == "succeeded"
    assert len(q.get(f"/conversations/{setup['conversation_id']}/usage").json()["items"]) == 1


def test_completion_verifies_outputs_result_usage_and_route(q):
    setup = q.ingest()
    claimed = q.claim_one(setup["ids"]["asr"])
    outputs = q.outputs_for(claimed)
    jid = claimed["job"]["id"]

    def attempt(expect, **kw):
        return q.post(f"/jobs/{jid}/complete", q.completion(claimed, kw.pop("outputs", outputs), **kw), expect=expect).json()

    wrong_checksum = [dict(outputs[0], checksum="sha256:" + "2" * 64)]
    assert attempt(422, outputs=wrong_checksum)["code"] == "checksum_mismatch"
    source = setup["audio"]
    assert attempt(422, outputs=[{"role": "transcript", "artifact_id": source["id"], "checksum": source["checksum"]}])["details"]["reason"] == "output_kind"
    assert attempt(422, result=None)["details"]["reason"] == "result_required"
    assert attempt(422, result={"kind": "qa", "state": "available"})["details"]["reason"] == "result_required"
    failed = attempt(422, usage_body={**q.completion(claimed, outputs)["usage"], "outcome": "failed", "error_code": "provider_error"})
    assert failed["details"]["reason"] == "outcome_not_allowed"
    assert attempt(403, provenance={"worker_id": "w1", "installation_id": q.installation_id, "adapter_id": "a", "adapter_version": "1"})["code"] == "route_not_permitted"
    assert attempt(403, provenance={"worker_id": "w1", "installation_id": "inst_other", "adapter_id": "a", "adapter_version": "1"})["code"] == "forbidden"
    assert q.status(jid) == "RUNNING"
    assert q.complete(claimed, outputs=outputs)["status"] == "SUCCEEDED"


def test_outputs_of_another_attempt_are_refused(q):
    setup = q.ingest()
    first = q.claim_one(setup["ids"]["asr"])
    old_outputs = q.outputs_for(first)
    q.release(first)
    second = q.claim_one(setup["ids"]["asr"])
    refused = q.post(f"/jobs/{setup['ids']['asr']}/complete", q.completion(second, old_outputs), expect=422).json()
    assert refused["details"]["reason"] == "not_produced_under_this_claim"


def test_non_publishers_complete_without_a_result(q):
    setup = q.ingest()
    q.run(setup["ids"]["asr"])
    claimed = q.claim_one(setup["ids"]["enrich"])
    outputs = q.outputs_for(claimed)
    refused = q.post(f"/jobs/{claimed['job']['id']}/complete", q.completion(claimed, outputs, result={"kind": "transcript", "state": "available"}), expect=422).json()
    assert refused["details"]["reason"] == "result_not_published_by_type"
    assert q.complete(claimed, outputs=outputs, result=None)["result_version"] is None


def test_completion_is_one_transaction_with_the_projection(q, hooks, store, client):
    setup = q.ingest()
    ids = setup["ids"]
    claimed = q.claim_one(ids["asr"])
    outputs = q.outputs_for(claimed)
    body = q.completion(claimed, outputs, key="complete-atomic-1")
    hooks.raise_on_completion = RuntimeError("projection failed")
    quiet = TestClient(q.client.app, base_url="http://localhost:8010", raise_server_exceptions=False)
    response = quiet.post(V + f"/jobs/{ids['asr']}/complete", json=body, headers=q.headers)
    assert response.status_code == 503 and response.json()["code"] == "store_unavailable"
    asr = q.job(ids["asr"])
    assert asr["status"] == "RUNNING" and asr["outputs"] == [] and asr["lease"]["attempt_number"] == 1
    assert q.get(f"/artifacts/{outputs[0]['artifact_id']}").json()["linked"] is False
    assert q.status(ids["enrich"]) == "BLOCKED"
    assert q.get(f"/conversations/{setup['conversation_id']}/usage").json()["items"] == []

    def project(conn, completion):  # a projection write commits with the receipt
        feed.append(conn, ChangeKind.RESULT, completion.job.conversation_id, 1, "available", conversation_id=completion.job.conversation_id)

    hooks.raise_on_completion = None
    hooks.on_completion = project
    receipt = q.post(f"/jobs/{ids['asr']}/complete", body).json()
    assert receipt["replayed"] is False and receipt["released_job_ids"]
    with store.connection() as conn:
        events = feed.read(conn, audience="admin", after=None, limit=1000, kinds=None, retention_seconds=3600).events
    result_events = [e for e in events if e.kind is ChangeKind.RESULT]
    job_events = [e for e in events if e.kind is ChangeKind.JOB and e.resource_id == ids["asr"] and e.status == "SUCCEEDED"]
    assert len(result_events) == 1 and len(job_events) == 1
    assert result_events[0].cursor <= receipt["change_cursor"] and job_events[0].cursor <= receipt["change_cursor"]


def test_follow_on_escalation_binds_its_assessment_to_the_scorecard(q, hooks):
    conversation = q.register()
    snapshot = q.rubric_snapshot(conversation["id"])
    rubric_in = pinned("rubric", snapshot)
    graph = q.graph(conversation["id"], [
        job("crit", JobType.QA_CRITERION, inputs=[rubric_in]),
        job("score", JobType.QA_SCORECARD, requires=["crit"], inputs=[rubric_in, upstream_input(f"assessment:{CRITERION}", "crit", "assessment")]),
    ])
    ids = q.ids(graph)
    claimed = q.claim_one(ids["crit"])
    follow_on = {"jobs": [job("esc", JobType.QA_ESCALATION, key="escalation-greeting-1", requires_ids=[ids["crit"]], inputs=[rubric_in],
                              parameters={"escalation_trigger": "needs_review"})],
                 "add_dependencies": [{"dependent_job_id": ids["score"], "requires_ref": "esc", "input_role": f"escalation:{CRITERION}", "output_role": "assessment"}]}
    receipt = q.complete(claimed, follow_on=follow_on, result=None)
    [esc_id] = receipt["created_job_ids"]
    assert receipt["released_job_ids"] == []  # the scorecard now also waits for the escalation
    escalation = q.job(esc_id)
    assert escalation["status"] == "QUEUED" and escalation["graph_id"] == graph["graph_id"] and escalation["requires_job_ids"] == [ids["crit"]]
    score = q.job(ids["score"])
    assert score["status"] == "BLOCKED" and set(score["requires_job_ids"]) == {ids["crit"], esc_id}
    assert score["inputs"][-1] == {"role": f"escalation:{CRITERION}", "artifact": None, "upstream": {"ref": None, "job_id": esc_id, "output_role": "assessment"}, "optional": False}
    assert esc_id in [j["job_id"] for j in q.get(f"/job-graphs/{graph['graph_id']}").json()["jobs"]]
    escalated = q.complete(q.claim_one(esc_id), result=None)
    assert escalated["released_job_ids"] == [ids["score"]]
    roles = {r["role"] for r in q.job(ids["score"])["resolved_inputs"]}
    assert roles == {"rubric", f"assessment:{CRITERION}", f"escalation:{CRITERION}"}


def test_follow_on_dependencies_are_checked(q):
    conversation = q.register()
    snapshot = q.rubric_snapshot(conversation["id"])
    rubric_in = pinned("rubric", snapshot)
    graph = q.graph(conversation["id"], [
        job("crit", JobType.QA_CRITERION, inputs=[rubric_in], priority=1),
        job("other", JobType.QA_CRITERION, key="other-criterion", inputs=[rubric_in]),
        job("score", JobType.QA_SCORECARD, requires=["crit"], inputs=[rubric_in, upstream_input("assessment:greeting", "crit", "assessment")]),
    ])
    ids = q.ids(graph)
    claimed = q.claim_one(ids["crit"])
    outputs = q.outputs_for(claimed)
    esc = job("esc", JobType.QA_ESCALATION, key="escalation-key-1", requires_ids=[ids["crit"]], inputs=[rubric_in])

    def complete_with(add, jobs=None):
        follow_on = {"jobs": jobs or [esc], "add_dependencies": add}
        return q.post(f"/jobs/{ids['crit']}/complete", q.completion(claimed, outputs, result=None, follow_on=follow_on), expect=422).json()

    assert complete_with([{"dependent_job_id": ids["other"], "requires_ref": "esc"}])["details"]["reason"] == "dependent_not_blocked"
    taken = complete_with([{"dependent_job_id": ids["score"], "requires_ref": "esc", "input_role": "rubric", "output_role": "assessment"}])
    assert taken["details"]["reason"] == "input_role_exists"
    looping = job("esc", JobType.QA_ESCALATION, key="escalation-key-2", requires_ids=[ids["crit"], ids["score"]], inputs=[rubric_in])
    assert complete_with([{"dependent_job_id": ids["score"], "requires_ref": "esc"}], jobs=[looping])["details"]["reason"] == "dependency_cycle"
    assert q.status(ids["crit"]) == "RUNNING"


def test_a_completion_after_cancel_is_refused(q, hooks):
    setup = q.ingest()
    claimed = q.claim_one(setup["ids"]["asr"])
    q.cancel(setup["ids"]["asr"], cascade=False)
    assert q.heartbeat(claimed)["cancel_requested"] is True
    assert q.complete(claimed, expect=409).json()["code"] == "job_cancelling"
    receipt = q.fail(claimed, "cancelled")
    assert receipt["status"] == "CANCELLED" and receipt["error_class"] == "terminal"
    assert q.status(setup["ids"]["asr"]) == "CANCELLED"
    assert q.job(setup["ids"]["enrich"])["blocking"][0]["dead"] is True


def test_claims_prefer_the_affinity_conversation_and_cap_the_batch(q, store):
    first, second = q.ingest(), q.ingest()
    preferred = q.claim(max_jobs=1, job_types=[JobType.VALIDATION_VAD], conversation_id=second["conversation_id"])["jobs"]
    assert [c["job"]["id"] for c in preferred] == [second["ids"]["vad"]]
    conversation = q.register()
    audio = q.upload_audio(conversation["id"])
    q.graph(conversation["id"], [job(f"v{i}", JobType.VALIDATION_VAD, inputs=[pinned("audio", audio)]) for i in range(20)])
    batch = q.claim(max_jobs=64, slots=[{"memory_slot": "local_memory", "count": 64}], job_types=[JobType.VALIDATION_VAD])["jobs"]
    assert len(batch) == 16  # MAX_CLAIM_BATCH
    assert first["ids"]["vad"] in [c["job"]["id"] for c in batch]


def _claimed_ids(response):
    return [c["job"]["id"] for c in response["jobs"]]


def _reanalysis_graph(q, store, reviewer_session, *, band: int, key: str):
    """A qa reanalysis graph (one scorecard job, job priority 0) whose request has priority ``band``
    (+5 like a taxonomy preview, -10 like a backfill); the band is set directly on the request."""
    conversation = q.register()
    snapshot = q.rubric_snapshot(conversation["id"])
    headers = {**reviewer_session.headers, "Idempotency-Key": f"click-{key}"}
    created = q.client.post(f"{V}/calls/{conversation['call_id']}/reanalysis-requests", json={"kind": "qa"}, headers=headers)
    assert created.status_code == 201, created.text
    request = created.json()
    with store.connection() as conn, db.transaction(conn):
        conn.execute("UPDATE q_reanalysis_requests SET priority = ? WHERE id = ?", (band, request["id"]))
    [claimed] = q.post("/reanalysis-requests/claim", {"worker_id": "w1", "max_requests": 4}).json()["requests"]
    graph = q.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, key=f"band-score-{key}", inputs=[pinned("rubric", snapshot)])],
                    key=f"band-graph-{key}", reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"])
    return q.ids(graph)["score"]


def test_claims_finish_the_oldest_call_before_starting_the_next(q):
    first, second = q.ingest(), q.ingest()  # same priorities; the first call entered the queue first
    a, b = first["ids"], second["ids"]
    # stage priority orders jobs within a call; the older call goes first even though b's asr outranks a's vad
    assert _claimed_ids(q.claim(max_jobs=1)) == [a["asr"]]
    assert _claimed_ids(q.claim(max_jobs=1)) == [a["vad"]]
    assert _claimed_ids(q.claim(max_jobs=1)) == [b["asr"]]
    assert _claimed_ids(q.claim(max_jobs=1)) == [b["vad"]]  # the old order (priority first) gave a asr, b asr, a vad, b vad


def test_released_dependents_of_the_oldest_call_go_before_a_newer_calls_ready_jobs(q):
    first, second = q.ingest(), q.ingest()
    a, b = first["ids"], second["ids"]
    q.run(a["asr"])  # releases a's enrich and sentiment (priority 0)
    order = _claimed_ids(q.claim(max_jobs=16))
    assert set(order[:3]) == {a["vad"], a["enrich"], a["sentiment"]}  # all of the first call's ready jobs ...
    assert order[3:] == [b["asr"], b["vad"]]  # ... before the second call's, although b's asr has priority 5


def test_a_higher_band_jumps_ahead_and_a_lower_band_waits_but_is_not_starved(q, store, reviewer_session):
    fresh = q.ingest()["ids"]
    preview = _reanalysis_graph(q, store, reviewer_session, band=5, key="000001")
    backfill = _reanalysis_graph(q, store, reviewer_session, band=-10, key="000002")
    q.clock.advance(1)
    later = q.ingest()["ids"]
    order = _claimed_ids(q.claim(max_jobs=16))
    # +5 first (created after the fresh call, job priority 0 < asr's 5), then band 0 call by call, -10 last
    assert order == [preview, fresh["asr"], fresh["vad"], later["asr"], later["vad"], backfill]


def test_a_lower_band_job_takes_slots_the_higher_band_cannot_use(q, store, reviewer_session):
    fresh = q.ingest()["ids"]
    backfill = _reanalysis_graph(q, store, reviewer_session, band=-10, key="000003")
    # a worker that cannot run the fresh call's jobs still gets the backfill job: bands order, never block
    assert _claimed_ids(q.claim(job_types=[JobType.QA_SCORECARD])) == [backfill]
    assert _claimed_ids(q.claim(max_jobs=16)) == [fresh["asr"], fresh["vad"]]


def test_the_affinity_hint_still_comes_before_the_call_order(q):
    first, second = q.ingest(), q.ingest()
    assert _claimed_ids(q.claim(max_jobs=2, conversation_id=second["conversation_id"])) == [second["ids"]["asr"], second["ids"]["vad"]]
    assert _claimed_ids(q.claim(max_jobs=2)) == [first["ids"]["asr"], first["ids"]["vad"]]


def test_admission_fails_queued_jobs_whose_route_is_no_longer_permitted(q, store, hooks):
    """Graph creation already refuses non-appliance routes (Stage 2 admin-state defaults); a job
    whose route became unpermitted after creation fails at admission without consuming an attempt."""
    setup = q.ingest()
    with store.connection() as conn:
        conn.execute("UPDATE q_jobs SET route_class = 'customer_lan' WHERE id = ?", (setup["ids"]["asr"],))
    q.claim(max_jobs=1, job_types=[JobType.VALIDATION_VAD])
    rejected = q.job(setup["ids"]["asr"])
    assert rejected["status"] == "FAILED" and rejected["error_code"] == "route_disabled" and rejected["attempt_count"] == 0
    assert hooks.failures[-1].job.job_id == setup["ids"]["asr"] and hooks.failures[-1].attempt_number is None


def test_large_outputs_upload_under_the_claim_and_link_at_completion(q):
    setup = q.ingest()
    claimed = q.claim_one(setup["ids"]["asr"])
    payload = content(ArtifactKind.TRANSCRIPT)
    from call1.contracts.common import canonical_json
    from call1.store.objects import sha256_checksum

    data = canonical_json(payload)
    body = {"kind": "transcript", "slot": "", "content_type": "application/json", "size_bytes": len(data), "checksum": sha256_checksum(data),
            "content_contract": "transcript.v1", "sensitivity": "masked", "producing_job_id": setup["ids"]["asr"], "claim_token": claimed["claim_token"]}
    stale = dict(body, claim_token="k" * 43)
    assert q.post(f"/conversations/{setup['conversation_id']}/artifacts/uploads", stale, expect=409).json()["code"] == "claim_token_stale"
    grant = q.post(f"/conversations/{setup['conversation_id']}/artifacts/uploads", body, expect=201).json()
    from .test_queue_harness import path

    assert q.client.put(path(grant["url"]), content=data).status_code == 200
    art = q.post(f"/artifact-uploads/{grant['upload_id']}/commit", {"checksum": body["checksum"], "size_bytes": len(data)}, expect=201).json()
    assert art["id"] == grant["artifact_id"] and art["linked"] is False and art["storage"] == "object"
    receipt = q.complete(claimed, outputs=[{"role": "transcript", "artifact_id": art["id"], "checksum": art["checksum"]}])
    assert receipt["linked_artifact_ids"] == [art["id"]] and q.get(f"/artifacts/{art['id']}").json()["version"] == 1
