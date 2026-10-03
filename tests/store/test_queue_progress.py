"""Group progress, pending work, stakes for derived states, the change feed, and one run through
the real results projection hooks."""

from __future__ import annotations

import pytest

from call1.contracts.calls import PublisherState
from call1.contracts.contents import ResultKind
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType
from call1.store.queue import api as queue_api
from call1.store.queue import lifecycle

from .test_queue_harness import QueueHarness, hooks, job, pinned, q, rubrics, snapshot_content, upstream_input  # noqa: F401


def _groups(progress):
    return {g["kind"]: g for g in progress["groups"]}


def test_group_progress_counts_jobs_and_derives_states(q, reviewer_session):
    setup = q.ingest()
    ids, conversation_id = setup["ids"], setup["conversation_id"]
    progress = q.get(f"/conversations/{conversation_id}/progress", headers=reviewer_session.read_headers).json()
    groups = _groups(progress)
    assert groups["transcript"] == {"kind": "transcript", "state": "pending", "total": 3, "succeeded": 0, "running": 0, "queued": 2, "blocked": 1,
                                    "dead_blocked": 0, "failed": 0, "cancelled": 0, "waiting_reason": "waiting_for_worker"}
    assert groups["text_sentiment"]["blocked"] == 1 and groups["text_sentiment"]["waiting_reason"] == "waiting_for_dependencies"
    assert groups["qa"]["state"] == "disabled" and groups["qa"]["total"] == 0
    assert progress["supporting_jobs_total"] == 0 and progress["settled"] is False
    q.fail(q.claim_one(ids["asr"]), "model_unavailable")
    groups = _groups(q.get(f"/conversations/{conversation_id}/progress").json())
    assert groups["transcript"]["state"] == "failed" and groups["transcript"]["failed"] == 1
    assert groups["text_sentiment"] == dict(groups["text_sentiment"], state="failed", blocked=1, dead_blocked=1, waiting_reason="dead_blocked")
    q.complete(q.claim_one(ids["vad"]))
    progress = q.get(f"/conversations/{conversation_id}/progress").json()
    assert progress["settled"] is True  # every job terminal or dead-blocked
    with q.store.connection() as conn:
        work = queue_api.pending_work(conn, conversation_id)
        codes = queue_api.failure_codes(conn, conversation_id)
    assert (work.jobs_total, work.jobs_failed, work.jobs_blocked, work.jobs_succeeded, work.settled) == (4, 1, 2, 1, True)
    assert codes[ResultKind.TRANSCRIPT] is JobErrorCode.MODEL_UNAVAILABLE and codes[ResultKind.TEXT_SENTIMENT] is JobErrorCode.MODEL_UNAVAILABLE
    assert codes[ResultKind.QA] is None
    assert q.get("/conversations/conv_missing/progress", expect=404).json()["code"] == "not_found"


def test_group_stakes_follow_the_newest_graph(q, store):
    setup = q.ingest()
    conversation_id = setup["conversation_id"]
    q.run(setup["ids"]["asr"])
    q.clock.advance(5)
    newer = q.graph(conversation_id, [job("asr2", JobType.ASR, key="rerun-asr-000", inputs=[pinned("audio", setup["audio"])])], key="rerun-graph-1")
    with store.connection() as conn:
        stakes = queue_api.group_stakes(conn, conversation_id)
        created = queue_api.graph_created_at(conn, newer["graph_id"])
    transcript = stakes[ResultKind.TRANSCRIPT]
    assert [(s.graph_id, s.publisher) for s in transcript] == [(setup["graph"]["graph_id"], PublisherState.SUCCEEDED), (newer["graph_id"], PublisherState.IN_PROGRESS)]
    assert all(s.has_group_jobs and not s.draft_test for s in transcript) and created == transcript[1].graph_created_at
    assert stakes[ResultKind.QA] == [] and set(stakes) == set(ResultKind)
    assert _groups(q.get(f"/conversations/{conversation_id}/progress").json())["transcript"]["state"] == "stale"


def test_the_change_feed_carries_job_and_group_events_by_principal(q, reviewer_session, mint_session):
    setup = q.ingest()
    receipt = q.run(setup["ids"]["asr"])
    process_feed = q.get("/changes", params={"limit": 1000}).json()
    kinds = {e["kind"] for e in process_feed["events"]}
    assert {"job", "job_group"} <= kinds
    asr_events = [e for e in process_feed["events"] if e["resource_id"] == setup["ids"]["asr"]]
    assert [e["status"] for e in asr_events] == ["QUEUED", "RUNNING", "SUCCEEDED"]
    assert all(e["call_id"] == q.get(f"/conversations/{setup['conversation_id']}").json()["call_id"] for e in asr_events)
    assert receipt["change_cursor"] == max(e["cursor"] for e in process_feed["events"])
    reviewer_feed = q.get("/changes", headers=reviewer_session.read_headers).json()
    assert {e["kind"] for e in reviewer_feed["events"]} == {"job_group"}  # job events start at supervisor
    supervisor_feed = q.get("/changes", headers=mint_session("supervisor").read_headers).json()
    assert "job" in {e["kind"] for e in supervisor_feed["events"]}
    after = q.get("/changes", params={"after": receipt["change_cursor"]}).json()
    assert after["events"] == [] and after["next_cursor"] == receipt["change_cursor"]


def test_supervisors_list_jobs_in_claim_order_and_page(q, mint_session):
    conversation = q.register()
    audio = q.upload_audio(conversation["id"])
    q.graph(conversation["id"], [job(f"v{i}", JobType.VALIDATION_VAD, priority=i % 3, inputs=[pinned("audio", audio)]) for i in range(7)])
    supervisor = mint_session("supervisor")
    page = q.get("/jobs", headers=supervisor.read_headers, params={"conversation_id": conversation["id"], "limit": 3}).json()
    seen = page["items"]
    while page["next_page_token"]:
        page = q.get("/jobs", headers=supervisor.read_headers, params={"conversation_id": conversation["id"], "limit": 3, "page_token": page["next_page_token"]}).json()
        seen += page["items"]
    assert len(seen) == 7 and [j["priority"] for j in seen] == sorted([j["priority"] for j in seen], reverse=True)
    assert q.get("/jobs", headers=supervisor.read_headers, params={"status": "RUNNING"}).json()["items"] == []
    assert q.get("/jobs", headers=q.mint_session("reviewer").read_headers, expect=403).json()["code"] == "insufficient_role"
    assert q.get("/jobs/job_missing", expect=404).json()["code"] == "not_found"


def test_maintenance_sweep_expires_leases_and_removes_old_orphans(q, store):
    setup = q.ingest()
    claimed = q.claim_one(setup["ids"]["asr"])
    orphan = q.outputs_for(claimed)[0]
    q.fail(claimed, "provider_error")
    q.clock.advance(30)
    running = q.claim_one(setup["ids"]["asr"])
    q.clock.advance(24 * 3600 + 1)
    report = lifecycle.sweep(store)
    assert report["leases_expired"] == 1 and report["orphans_deleted"] == 1
    assert q.get(f"/artifacts/{orphan['artifact_id']}", expect=404).json()["code"] == "not_found"
    assert q.status(running["job"]["id"]) == "QUEUED"


# --- through the real results area --------------------------------------------------------------


@pytest.fixture
def real(client, store, clock, service_key, mint_service_key, mint_session, rubrics, monkeypatch) -> QueueHarness:
    """The harness without the recording hooks: completions run the results area's projections."""
    from call1.store.results import api as results_api
    from call1.store.results import records as results_records

    monkeypatch.setattr(results_api, "result_groups", results_records.result_groups)
    return QueueHarness(client, store, clock, service_key, mint_service_key, mint_session)


def test_completions_publish_through_the_results_projection(real, reviewer_session):
    setup = real.ingest()
    conversation = real.get(f"/conversations/{setup['conversation_id']}").json()
    call = real.get(f"/calls/{conversation['call_id']}", headers=reviewer_session.read_headers).json()
    assert {g["kind"]: g["state"] for g in call["results"]}["transcript"] == "pending"
    receipt = real.run(setup["ids"]["asr"])
    assert receipt["result_version"] == 1
    groups = _groups(real.get(f"/conversations/{setup['conversation_id']}/progress").json())
    # Published, but no PII findings for it yet (contract 1.2.0): 'partial', with the text withheld.
    assert groups["transcript"]["state"] == "partial" and groups["text_sentiment"]["state"] == "pending"
    transcript = real.get(f"/calls/{conversation['call_id']}/transcript", headers=reviewer_session.read_headers).json()
    assert transcript["version"] == 1 and transcript["turns"][0]["turn_id"] == 0
    assert transcript["text_withheld"] is True and all(t["text"] == "" for t in transcript["turns"])
    real.fail(real.claim_one(setup["ids"]["sentiment"]), "model_unavailable")
    groups = _groups(real.get(f"/conversations/{setup['conversation_id']}/progress").json())
    assert groups["text_sentiment"]["state"] == "failed"
