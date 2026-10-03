"""Process against a real, in-process Store with fake handlers: ingest, the full Stage 2 graph,
publication, claim races, retries, cancellation and restart recovery. No ports are bound."""

from __future__ import annotations

import io
import threading
import time
import wave
from collections import Counter

import pytest

from call1.contracts.common import ReviewerRole
from call1.contracts.jobs import JobStatus, JobType
from call1.process.handlers.fake import FakeBehavior
from call1.process.store_client import StoreUnavailable

from .conftest import SAMPLE, STORE_URL, write_headers

V = "/store/v1"


def _states(detail) -> dict:
    return {g["kind"]: g["state"] for g in detail["results"]}


def _jobs(runtime, conversation_id):
    return runtime.client.list_jobs(conversation_id=conversation_id)


def _mono_wav(path, seconds: float = 24.0) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x01" * int(8000 * seconds))


def test_ingest_runs_every_stage_and_store_shows_the_call(make_runtime, store_http, session):
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE, agent_id="agent-7")
    assert result.conversation_created and result.graph_created
    assert result.evaluate_url == f"{STORE_URL}/#/calls/{result.call_id}"

    ran = worker.drain()
    jobs = _jobs(runtime, result.conversation_id)
    assert ran == len(jobs)
    assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
    types = Counter(j.job_type for j in jobs)
    # stereo fixture: no speaker attribution; the four semantic criteria of call1_standard_v2 each get a QA job
    assert JobType.SPEAKER_ATTRIBUTION not in types
    assert types[JobType.QA_CRITERION] == 4 and types[JobType.QA_SCORECARD] == 1
    assert types[JobType.SUMMARY_SEGMENT] >= 1 and types[JobType.SUMMARY_ASSEMBLY] == 1  # segments were added at ASR completion
    assert types[JobType.CONTACT_SIGNALS_MERGE] == 1 and types[JobType.EMBEDDINGS] == 1
    progress = runtime.client.get_progress(result.conversation_id)
    assert progress.settled

    reviewer = session(ReviewerRole.REVIEWER)
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    assert _states(detail) == {"transcript": "available", "tone": "available", "text_sentiment": "available", "qa": "available",
                               "summary": "available", "contact_signals": "available"}
    assert detail["call"]["agent_id"] == "agent-7" and detail["call"]["channels"] == 2 and detail["call"]["duration_seconds"] == pytest.approx(44.009, abs=0.01)
    assert detail["pending_work"]["settled"] is True

    transcript = store_http.get(f"{V}/calls/{result.call_id}/transcript", headers=reviewer.read_headers).json()
    assert transcript["turns"] and {t["speaker"] for t in transcript["turns"]} == {"AGENT", "CALLER"}
    assert transcript["tone_blocks"] and transcript["vad_metrics"] is not None

    evaluation = store_http.get(f"{V}/calls/{result.call_id}/evaluation", headers=reviewer.read_headers).json()
    assert evaluation["rubric"]["rubric_id"] == "call1_standard_v2" and evaluation["rubric"]["rubric_version"] == 1
    assert [v["criterion_id"] for v in evaluation["verdicts"]] == ["REG-01", "SEC-01", "COMP-01", "ETIQ-01"]
    assert evaluation["overall_score"] == 100 and evaluation["passed"] is True
    assert all(v["model_attempts"][0]["catalog_entry_id"] == "call1-bundled" for v in evaluation["verdicts"])

    summary = store_http.get(f"{V}/calls/{result.call_id}/summary", headers=reviewer.read_headers).json()
    assert summary["narrative"] and summary["segments"] >= 1 and summary["catalog_entry_id"] == "call1-bundled"
    signals = store_http.get(f"{V}/calls/{result.call_id}/contact-signals", headers=reviewer.read_headers).json()
    assert signals["completeness"] == "complete" and signals["signals"]

    # the fake search embedder's vectors (fake-handler Process and the suite's Store agree on the backend)
    hits = store_http.post(f"{V}/search/semantic", json={"query": "this call may be recorded for quality"},
                           headers=write_headers(reviewer)).json()
    assert hits["count"] >= 1 and hits["results"][0]["call_id"] == result.call_id

    # the catalog snapshot and hardware profile were published at start-up
    snapshots = store_http.get(f"{V}/catalog-snapshots", headers=reviewer.read_headers).json()["items"]
    mine = [s for s in snapshots if s["installation_id"] == runtime.config.installation_id]
    assert mine and {e["entry"]["entry_id"] for e in mine[0]["entries"]} >= {"parakeet-tdt-0.6b-v3", "call1-bundled", "nemotron-3-embed-1b"}

    # a repeated ingest of the same recording is idempotent
    again = runtime.ingestor.ingest_file(SAMPLE, agent_id="agent-7")
    assert (again.conversation_id, again.graph_id) == (result.conversation_id, result.graph_id)
    assert not again.conversation_created and not again.graph_created


def test_mono_recordings_get_speaker_attribution(make_runtime, store_http, session, tmp_path):
    runtime = make_runtime()
    worker = runtime.connect()
    path = tmp_path / "mono.wav"
    _mono_wav(path)
    result = runtime.ingestor.ingest_file(path)
    worker.drain()
    jobs = _jobs(runtime, result.conversation_id)
    assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
    assert Counter(j.job_type for j in jobs)[JobType.SPEAKER_ATTRIBUTION] == 1
    reviewer = session()
    transcript = store_http.get(f"{V}/calls/{result.call_id}/transcript", headers=reviewer.read_headers).json()
    assert transcript["speaker_attribution_version"] == 1
    assert {t["speaker"] for t in transcript["turns"]} == {"AGENT", "CALLER"}


def test_two_workers_racing_never_share_a_claim(make_runtime, mint_key):
    key = mint_key()
    first = make_runtime("p1", key=key)
    second = make_runtime("p2", key=key)
    w1, w2 = first.connect(), second.connect()
    conversations = []
    for n in range(3):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(bytes([n]) * 4 * 8000 * 12)
        path = first.config.data_dir / f"race-{n}.wav"
        path.write_bytes(buf.getvalue())
        conversations.append(first.ingestor.ingest_file(path).conversation_id)

    # both workers claim the ready cpu_io jobs at the same instant
    barrier = threading.Barrier(2)
    claimed = {}

    def grab(name, worker):
        barrier.wait()
        pool = next(p for p in worker.pools if p.name == "cpu_io")
        claimed[name] = (worker, pool, worker.claim(pool))

    threads = [threading.Thread(target=grab, args=(n, w)) for n, w in (("a", w1), ("b", w2))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    ids_a = {c.job.id for c in claimed["a"][2]}
    ids_b = {c.job.id for c in claimed["b"][2]}
    assert ids_a.isdisjoint(ids_b) and len(ids_a | ids_b) == 3  # three validation_vad jobs, each claimed once
    for worker, pool, jobs in claimed.values():
        for job in jobs:
            worker.execute(job, pool)

    done = {}
    threads = [threading.Thread(target=lambda n=n, w=w: done.__setitem__(n, w.drain())) for n, w in (("a", w1), ("b", w2))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    w1.drain()
    w2.drain()
    all_jobs = [j for c in conversations for j in _jobs(first, c)]
    assert {j.status for j in all_jobs} == {JobStatus.SUCCEEDED}
    for job in all_jobs:
        attempts = first.client.list_attempts(job.id)
        assert [a.status.value for a in attempts] == ["succeeded"], (job.job_type, attempts)
    assert w1.stats.succeeded + w2.stats.succeeded == len(all_jobs)


def test_a_failed_job_is_retried_from_the_console(make_runtime, store_http, session):
    from fastapi.testclient import TestClient

    from call1.process import console
    from call1.process.app import create_app

    behavior = FakeBehavior({"qa_criterion:SEC-01": ["fail:configuration_error"]})
    runtime = make_runtime(behavior=behavior)
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    jobs = {(j.job_type, j.parameters.criterion_id): j for j in _jobs(runtime, result.conversation_id)}
    failed = jobs[(JobType.QA_CRITERION, "SEC-01")]
    assert failed.status is JobStatus.FAILED and failed.error_code.value == "configuration_error"
    scorecard = jobs[(JobType.QA_SCORECARD, None)]
    assert scorecard.status is JobStatus.BLOCKED and scorecard.blocking and scorecard.blocking[0].dead
    reviewer = session()
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    assert _states(detail)["qa"] == "failed" and _states(detail)["transcript"] == "available"

    token, _ = console.issue(runtime.config)
    with TestClient(create_app(runtime, start_background=False, loopback_only=False)) as api:
        refused = api.post(f"/process/api/jobs/{failed.id}/retry", json={"reason": "fixed"})
        assert refused.status_code == 401 and refused.json()["code"] == "console_credential_invalid"
        conv = api.get(f"/process/api/conversations/{result.conversation_id}").json()
        assert any(j["id"] == failed.id and j["status"] == "FAILED" for j in conv["jobs"])
        assert "QA" in (conv["progress_line"] or "")
        retried = api.post(f"/process/api/jobs/{failed.id}/retry", json={"reason": "fixed"}, headers={console.HEADER: token})
        assert retried.status_code == 200, retried.text
        assert retried.json()["job"]["status"] == "QUEUED" and retried.json()["job"]["retry_generation"] == 1

    worker.drain()
    assert {j.status for j in _jobs(runtime, result.conversation_id)} == {JobStatus.SUCCEEDED}
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    assert _states(detail)["qa"] == "available"


def test_retry_without_jobs_control_says_so(make_runtime, mint_key):
    from fastapi.testclient import TestClient

    from call1.process import console
    from call1.process.app import create_app

    runtime = make_runtime(key=mint_key(control=False), behavior=FakeBehavior({"enrichment": ["fail:configuration_error"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    failed = next(j for j in _jobs(runtime, result.conversation_id) if j.status is JobStatus.FAILED)
    token, _ = console.issue(runtime.config)
    with TestClient(create_app(runtime, start_background=False, loopback_only=False)) as api:
        response = api.post(f"/process/api/jobs/{failed.id}/retry", json={"reason": "x"}, headers={console.HEADER: token})
        assert response.status_code == 403
        assert response.json()["code"] == "insufficient_scope" and "jobs:control" in response.json()["message"]
        cancel = api.post(f"/process/api/jobs/{failed.id}/cancel", json={"reason": "x"}, headers={console.HEADER: token})
        assert cancel.status_code == 403 and "jobs:control" in cancel.json()["message"]


def test_restart_replays_a_spooled_completion(make_runtime, mint_key):
    key = mint_key()
    runtime = make_runtime("p1", key=key)
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    real_complete = worker.client.complete_job

    def outage(job_id, body):
        raise StoreUnavailable("simulated Store outage")

    worker.client.complete_job = outage  # the first completion is lost
    pool = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(pool)
    assert asr.job.job_type is JobType.ASR
    assert worker.execute(asr, pool) == "spooled"
    assert runtime.spool.count() == 1
    assert runtime.client.get_job(asr.job.id).status is JobStatus.RUNNING
    worker.client.complete_job = real_complete

    # a new Process on the same data directory replays the spool at start-up
    restarted = make_runtime("p1", key=key)
    restarted.connect()
    assert restarted.spool.count() == 0
    assert restarted.client.get_job(asr.job.id).status is JobStatus.SUCCEEDED
    restarted.worker.drain()
    assert {j.status for j in _jobs(restarted, result.conversation_id)} == {JobStatus.SUCCEEDED}


def test_restart_after_a_crash_recovers_through_lease_expiry(make_runtime, mint_key, clock):
    key = mint_key()
    crashed = make_runtime("p1", key=key)
    worker = crashed.connect()
    result = crashed.ingestor.ingest_file(SAMPLE)
    pool = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(pool)  # claimed, then the process dies without finishing

    restarted = make_runtime("p1", key=key)
    fresh = restarted.connect()
    fresh.drain()
    assert restarted.client.get_job(asr.job.id).status is JobStatus.RUNNING  # still leased to the dead worker
    params = restarted.client.parameters
    clock.advance(params.lease_duration_seconds + params.lease_grace_seconds + 1)
    fresh.drain()  # the claim sweep expires the lease and requeues with backoff
    clock.advance(params.retry_backoff_max_seconds)
    fresh.drain()
    assert {j.status for j in _jobs(restarted, result.conversation_id)} == {JobStatus.SUCCEEDED}
    attempts = restarted.client.list_attempts(asr.job.id)
    assert [a.status.value for a in attempts] == ["lease_expired", "succeeded"]


def test_cancel_reaches_a_running_job_through_its_heartbeat(make_runtime):
    runtime = make_runtime(behavior=FakeBehavior({"asr": ["hold:5"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    pool = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(pool)
    outcome = {}
    thread = threading.Thread(target=lambda: outcome.setdefault("value", worker.execute(asr, pool)))
    thread.start()
    assert runtime.wait_until(lambda: bool(worker.running()), 2)
    runtime.client.cancel_job(asr.job.id, "operator cancel", cascade=True)
    worker.heartbeat_due(force=True)
    thread.join(5)
    assert outcome["value"] == "failed"
    jobs = {j.job_type: j for j in _jobs(runtime, result.conversation_id)}
    assert jobs[JobType.ASR].status is JobStatus.CANCELLED


def test_worker_releases_a_job_it_cannot_run_without_using_an_attempt(make_runtime):
    runtime = make_runtime(behavior=FakeBehavior({"validation_vad": ["release:model_unavailable"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    vad = next(j for j in _jobs(runtime, result.conversation_id) if j.job_type is JobType.VALIDATION_VAD)
    assert vad.status is JobStatus.FAILED and vad.attempt_count == 0 and vad.claim_count == 1
    assert worker.stats.released == 1


def test_graceful_stop_reports_interrupted_jobs(make_runtime):
    runtime = make_runtime(behavior=FakeBehavior({"asr": ["hold:30"]}), shutdown_grace_seconds=0.2)
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.start()
    assert runtime.wait_until(lambda: any(r.claimed.job.job_type is JobType.ASR for r in worker.running()), 5)
    worker.stop(grace=0.2)
    asr = next(j for j in _jobs(runtime, result.conversation_id) if j.job_type is JobType.ASR)
    assert asr.status is JobStatus.QUEUED and asr.error_code.value == "worker_crashed"  # transient: Store retries it
    assert worker.state == "stopped"


def test_outputs_over_the_inline_limit_go_through_upload_grants(tmp_path, clock):
    from fastapi.testclient import TestClient

    from call1.contracts.artifacts import ArtifactStorage
    from call1.contracts.common import ContractParameters, ServiceScope
    from call1.process.config import ProcessConfig
    from call1.process.runtime import ProcessRuntime
    from call1.process.store_client import StoreClient
    from call1.store.app import create_app as create_store_app
    from call1.store.config import StoreConfig

    small = StoreConfig.for_tests(tmp_path / "store-small", parameters=ContractParameters(inline_artifact_max_bytes=1024))
    app = create_store_app(small, clock=clock)
    store = app.state.store
    with store.connection() as conn:
        key = store.auth.mint_service_key_for_tests(conn, scopes=list(ServiceScope), primary_host=True)
    with TestClient(app, base_url=STORE_URL) as http:
        config = ProcessConfig(config_path=tmp_path / "p" / "config.json", store_url=STORE_URL, installation_id=key.installation_id,
                               service_key=key.token, data_dir=tmp_path / "p", handlers="fake")
        runtime = ProcessRuntime(config, client=StoreClient(STORE_URL, key.token, http=http, sleep=lambda _s: None))
        worker = runtime.connect()
        assert runtime.client.parameters.inline_artifact_max_bytes == 1024
        result = runtime.ingestor.ingest_file(SAMPLE)
        worker.drain()
        jobs = _jobs(runtime, result.conversation_id)
        assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
        embeddings = next(j for j in jobs if j.job_type is JobType.EMBEDDINGS)
        artifact = runtime.client.get_artifact(embeddings.outputs[0].artifact_id)
        assert artifact.storage is ArtifactStorage.OBJECT and artifact.size_bytes > 1024
        with store.connection() as conn:
            reviewer = store.auth.mint_session_for_tests(conn, role=ReviewerRole.REVIEWER)
        transcript = http.get(f"{V}/calls/{result.call_id}/transcript", headers=reviewer.read_headers).json()
        assert transcript["turns"]


def test_a_claim_lost_mid_run_publishes_nothing_and_attaches_late_usage(make_runtime, clock):
    runtime = make_runtime(behavior=FakeBehavior({"asr": ["hold:5"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    pool = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(pool)
    outcome = {}
    thread = threading.Thread(target=lambda: outcome.setdefault("value", worker.execute(asr, pool)))
    thread.start()
    assert runtime.wait_until(lambda: bool(worker.running()), 2)
    params = runtime.client.parameters
    clock.advance(params.lease_duration_seconds + params.lease_grace_seconds + 1)
    worker.heartbeat_due(force=True)  # Store expires the lease first, so the token is stale
    thread.join(5)
    assert outcome["value"] == "lost" and worker.stats.lost == 1
    job = runtime.client.get_job(asr.job.id)
    assert job.status is JobStatus.QUEUED and not job.outputs
    usage = runtime.client._json("GET", f"/conversations/{result.conversation_id}/usage")["items"]
    row = next(u for u in usage if u["job_id"] == asr.job.id)
    assert row["outcome"] == "abandoned" and row["recorded_by"] == "process_late"


# --- Store outages while Process runs ---------------------------------------------------------


def _outage(job_id, body):
    raise StoreUnavailable("simulated Store outage")


def test_a_spooled_completion_keeps_its_claim_alive_and_is_delivered_without_a_restart(make_runtime, clock):
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    real_complete = worker.client.complete_job
    mlx = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(mlx)
    # Store is unreachable for ASR's completion (other jobs' completions get through)
    worker.client.complete_job = lambda job_id, body: _outage(job_id, body) if job_id == asr.job.id else real_complete(job_id, body)
    assert worker.execute(asr, mlx) == "spooled"
    assert runtime.spool.count() == 1 and [r.claimed.job.id for r in worker.undelivered()] == [asr.job.id]
    assert worker.describe()["awaiting_delivery"][0]["awaiting_delivery"] == "complete"

    # the outage outlasts the original lease, but the heartbeat keeps renewing the finished job's claim
    params = runtime.client.parameters
    for _ in range(3):
        clock.advance(params.lease_duration_seconds / 2)
        assert worker.heartbeat_due(force=True) == 1
    cpu = next(p for p in worker.pools if p.name == "cpu_io")
    for other in worker.claim(cpu):  # a claim runs Store's lease sweep: the renewed lease survives it
        worker.execute(other, cpu)
    assert runtime.client.get_job(asr.job.id).status is JobStatus.RUNNING

    worker.client.complete_job = real_complete
    assert worker.recover() == 1  # what the spool thread does while Process runs
    assert runtime.spool.count() == 0 and worker.undelivered() == []
    assert [a.status.value for a in runtime.client.list_attempts(asr.job.id)] == ["succeeded"]  # no work redone
    worker.drain()
    assert {j.status for j in _jobs(runtime, result.conversation_id)} == {JobStatus.SUCCEEDED}


def test_a_transient_outage_at_completion_heals_while_the_worker_runs(make_runtime):
    runtime = make_runtime()
    worker = runtime.connect()
    worker.replay_interval = 0.05
    real_complete = worker.client.complete_job
    failures = {"left": 2}
    lock = threading.Lock()

    def flaky(job_id, body):
        with lock:
            if failures["left"] > 0:
                failures["left"] -= 1
                raise StoreUnavailable("simulated Store outage")
        return real_complete(job_id, body)

    worker.client.complete_job = flaky
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.start()
    try:
        assert runtime.wait_until(
            lambda: {j.status for j in _jobs(runtime, result.conversation_id)} == {JobStatus.SUCCEEDED}, 30, interval=0.1)
        assert runtime.wait_until(lambda: runtime.spool.count() == 0 and not worker.undelivered(), 5)
    finally:
        worker.stop(grace=2.0)
    assert failures["left"] == 0  # the outage happened, and no restart was needed
    for job in _jobs(runtime, result.conversation_id):
        assert [a.status.value for a in runtime.client.list_attempts(job.id)] == ["succeeded"], job.job_type
    assert worker.stats.lost == 0


def test_a_cancel_that_arrives_while_a_completion_waits_is_reported_as_cancelled(make_runtime):
    runtime = make_runtime()
    worker = runtime.connect()
    runtime.ingestor.ingest_file(SAMPLE)
    real_complete = worker.client.complete_job
    worker.client.complete_job = _outage
    mlx = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(mlx)
    assert worker.execute(asr, mlx) == "spooled"
    runtime.client.cancel_job(asr.job.id, "operator cancel", cascade=True)
    worker.client.complete_job = real_complete
    worker.recover()  # Store refuses the completion (job_cancelling): a cancelled failure is spooled
    assert runtime.spool.count() == 1 and worker.undelivered()
    assert worker.recover() == 1
    assert runtime.spool.count() == 0 and worker.undelivered() == []
    assert runtime.client.get_job(asr.job.id).status is JobStatus.CANCELLED


# --- Store outages when a job finishes: its outputs cannot be uploaded (p7) ----------------------


def _upload_outage(worker):
    """Make every output upload (inline and granted) meet an unreachable Store; returns a restore."""
    real = (worker.client.create_inline_artifact, worker.client.upload_artifact)
    worker.client.create_inline_artifact = lambda *a, **k: _outage(None, None)
    worker.client.upload_artifact = lambda *a, **k: _outage(None, None)

    def restore():
        worker.client.create_inline_artifact, worker.client.upload_artifact = real
    return restore


def test_outputs_that_meet_a_store_outage_are_spooled_and_delivered_without_redoing_work(make_runtime, clock):
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    mlx = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(mlx)
    restore = _upload_outage(worker)
    assert worker.execute(asr, mlx) == "spooled"
    assert mlx.in_use == 0  # the slot is free again; the result waits in the spool, not in a thread
    assert runtime.spool.has(asr.job.id, asr.attempt_number, "publish") and runtime.spool.count() == 1
    assert any(runtime.spool.outputs_dir(asr.job.id, asr.attempt_number, create=False).iterdir())
    [waiting] = worker.describe()["awaiting_delivery"]
    assert waiting["job_id"] == asr.job.id and waiting["awaiting_delivery"] == "publish"

    # the outage outlasts the original lease; heartbeats keep the finished job's claim alive
    params = runtime.client.parameters
    for _ in range(3):
        clock.advance(params.lease_duration_seconds / 2)
        assert worker.heartbeat_due(force=True) == 1
    assert worker.recover() == 0  # still unreachable: nothing is lost, nothing is sent
    assert runtime.spool.has(asr.job.id, asr.attempt_number, "publish")

    restore()
    assert worker.recover() == 1  # the spool thread's replay: recheck the claim, upload, complete
    assert runtime.spool.count() == 0 and worker.undelivered() == []
    assert not runtime.spool.outputs_dir(asr.job.id, asr.attempt_number, create=False).exists()
    assert [a.status.value for a in runtime.client.list_attempts(asr.job.id)] == ["succeeded"]  # no work redone
    worker.drain()
    jobs = _jobs(runtime, result.conversation_id)
    assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
    assert any(j.job_type is JobType.SUMMARY_SEGMENT for j in jobs)  # the ASR follow-ons were planned on replay


def test_an_outage_while_follow_ons_are_planned_is_spooled_not_failed(make_runtime):
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    mlx = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(mlx)
    real_graph = worker.client.get_job_graph
    worker.client.get_job_graph = lambda graph_id: _outage(None, None)  # ASR's summary follow-on reads the graph
    assert worker.execute(asr, mlx) == "spooled"
    assert runtime.client.get_job(asr.job.id).status is JobStatus.RUNNING  # not a configuration_error failure
    worker.client.get_job_graph = real_graph
    assert worker.recover() == 1
    worker.drain()
    assert {j.status for j in _jobs(runtime, result.conversation_id)} == {JobStatus.SUCCEEDED}
    assert [a.status.value for a in runtime.client.list_attempts(asr.job.id)] == ["succeeded"]


def test_restart_replays_a_spooled_publication(make_runtime, mint_key):
    key = mint_key()
    runtime = make_runtime("p1", key=key)
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    mlx = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(mlx)
    _upload_outage(worker)
    assert worker.execute(asr, mlx) == "spooled"

    restarted = make_runtime("p1", key=key)  # a new Process on the same data directory, Store back
    restarted.connect()  # replays the spool at start-up
    assert restarted.spool.count() == 0
    assert restarted.client.get_job(asr.job.id).status is JobStatus.SUCCEEDED
    assert [a.status.value for a in restarted.client.list_attempts(asr.job.id)] == ["succeeded"]
    restarted.worker.drain()
    assert {j.status for j in _jobs(restarted, result.conversation_id)} == {JobStatus.SUCCEEDED}


def test_a_restarted_process_keeps_a_spooled_publication_alive_until_it_can_upload(make_runtime, mint_key, clock):
    key = mint_key()
    runtime = make_runtime("p1", key=key)
    worker = runtime.connect()
    runtime.ingestor.ingest_file(SAMPLE)
    mlx = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(mlx)
    _upload_outage(worker)
    assert worker.execute(asr, mlx) == "spooled"

    restarted = make_runtime("p1", key=key)
    restore = _upload_outage(restarted)  # Store answers again, but uploads still fail
    fresh = restarted.connect()  # the start-up replay rechecks the claim, then defers again
    assert restarted.spool.has(asr.job.id, asr.attempt_number, "publish")
    assert [r.claimed.job.id for r in fresh.undelivered()] == [asr.job.id]
    params = restarted.client.parameters
    for _ in range(3):  # the restarted Process heartbeats the claim it inherited from the spool
        clock.advance(params.lease_duration_seconds / 2)
        assert fresh.heartbeat_due(force=True) == 1
    restore()
    assert fresh.recover() == 1
    assert restarted.spool.count() == 0 and fresh.undelivered() == []
    assert [a.status.value for a in restarted.client.list_attempts(asr.job.id)] == ["succeeded"]


def test_a_job_cancelled_while_its_outputs_wait_becomes_a_cancelled_failure(make_runtime):
    runtime = make_runtime()
    worker = runtime.connect()
    runtime.ingestor.ingest_file(SAMPLE)
    mlx = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(mlx)
    restore = _upload_outage(worker)
    assert worker.execute(asr, mlx) == "spooled"
    runtime.client.cancel_job(asr.job.id, "operator cancel", cascade=False)
    restore()
    assert worker.recover() == 1  # the claim recheck sees the cancel: nothing is uploaded
    assert runtime.spool.count() == 0 and worker.undelivered() == []
    assert runtime.client.get_job(asr.job.id).status is JobStatus.CANCELLED
    [attempt] = runtime.client.list_attempts(asr.job.id)
    assert attempt.status.value == "cancelled" and attempt.error_code.value == "cancelled"
    artifacts = runtime.client.list_artifacts(asr.job.conversation_id, include_unlinked=True)
    assert not [a for a in artifacts if a.producing_job_id == asr.job.id]


def test_a_spooled_publication_whose_claim_expired_becomes_late_usage(make_runtime, clock):
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    mlx = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(mlx)
    restore = _upload_outage(worker)
    assert worker.execute(asr, mlx) == "spooled"
    params = runtime.client.parameters
    clock.advance(params.lease_duration_seconds + params.lease_grace_seconds + 1)  # no heartbeat got through
    restore()
    assert worker.recover() == 0
    assert runtime.spool.count() == 0 and worker.undelivered() == [] and worker.stats.lost == 1
    usage = runtime.client._json("GET", f"/conversations/{result.conversation_id}/usage")["items"]
    row = next(u for u in usage if u["job_id"] == asr.job.id)
    assert row["outcome"] == "abandoned" and row["recorded_by"] == "process_late"


def test_jobs_claimed_while_the_worker_stops_are_released_not_stranded(make_runtime):
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    mlx = next(p for p in worker.pools if p.name == "mlx")
    [asr] = worker.claim(mlx)
    assert mlx.in_use == 1
    worker._stopping.set()  # stop() ran between the claim and the hand-off: no executor any more
    worker._dispatch([asr], mlx)
    assert mlx.in_use == 0 and worker.stats.released == 1
    job = runtime.client.get_job(asr.job.id)
    assert job.status is JobStatus.QUEUED and job.attempt_count == 0
    assert result.conversation_id == job.conversation_id


def test_a_job_cancelled_in_the_executor_queue_is_released(make_runtime):
    from concurrent.futures import ThreadPoolExecutor

    runtime = make_runtime(behavior=FakeBehavior({"asr": ["hold:5"]}))
    worker = runtime.connect()
    runtime.ingestor.ingest_file(SAMPLE)
    mlx = next(p for p in worker.pools if p.name == "mlx")
    gate = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(gate.wait, 5)  # occupy the only thread so the job's future stays queued
    worker._executor = executor
    [asr] = worker.claim(mlx)
    worker._dispatch([asr], mlx)
    worker._executor = None
    executor.shutdown(wait=False, cancel_futures=True)  # what stop() does
    gate.set()
    assert mlx.in_use == 0 and worker.stats.released == 1
    assert runtime.client.get_job(asr.job.id).status is JobStatus.QUEUED


def test_a_completion_that_releases_dependents_wakes_idle_pools_before_the_poll(make_runtime):
    # A 60 s poll: without the wake, the stages after the first would wait a full poll each.
    runtime = make_runtime(poll_interval_seconds=60.0)
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.start()
    try:
        assert runtime.wait_until(
            lambda: {j.status for j in _jobs(runtime, result.conversation_id)} == {JobStatus.SUCCEEDED}, 20, interval=0.1)
    finally:
        worker.stop(grace=2.0)
    assert worker.state == "stopped"  # stop() woke the idle loops too; it did not wait out the poll


def test_wake_only_on_receipts_that_release_or_create_jobs(make_runtime):
    from types import SimpleNamespace

    worker = make_runtime().connect()
    worker.poll_interval = 60.0
    for receipt, woke in ((SimpleNamespace(released_job_ids=[], created_job_ids=[]), False), (None, False),
                          (SimpleNamespace(released_job_ids=["job_1"]), True), (SimpleNamespace(released_job_ids=[], created_job_ids=["job_2"]), True)):
        seen = worker._work_generation
        worker._wake_if_ready(receipt)
        assert (worker._work_generation != seen) is woke, receipt
    seen = worker._work_generation
    waiter = threading.Thread(target=worker._wait_for_work, args=(seen,))
    started = time.monotonic()
    waiter.start()
    worker.notify_work()
    waiter.join(timeout=5)
    assert not waiter.is_alive() and time.monotonic() - started < 5
    # a wake that lands between reading the generation and waiting is not lost
    seen = worker._work_generation
    worker.notify_work()
    started = time.monotonic()
    worker._wait_for_work(seen)
    assert time.monotonic() - started < 1
