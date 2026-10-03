"""QA outcomes (escalation follow-ons, invalid answers, final-attempt provider failures), summary
segmentation, and the reanalysis-request consumer, all against a real in-process Store."""

from __future__ import annotations

from collections import Counter

from call1.contracts.common import ReviewerRole
from call1.contracts.jobs import JobStatus, JobType
from call1.process.handlers.fake import FakeBehavior

from .conftest import SAMPLE, write_headers

V = "/store/v1"


def _jobs(runtime, conversation_id):
    return runtime.client.list_jobs(conversation_id=conversation_id)


def _assessment(runtime, job):
    out = next(o for o in job.outputs if o.role == "assessment")
    import json

    return json.loads(runtime.client.download(runtime.client.get_artifact(out.artifact_id)))


def test_needs_review_fires_the_escalation_as_a_follow_on_job(make_runtime, store_http, session):
    behavior = FakeBehavior({"qa_criterion:REG-01": ["needs_review"]})
    runtime = make_runtime(behavior=behavior, escalation_entry_id="gemma4-e4b")
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    jobs = _jobs(runtime, result.conversation_id)
    assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
    [escalation] = [j for j in jobs if j.job_type is JobType.QA_ESCALATION]
    assert escalation.parameters.criterion_id == "REG-01" and escalation.parameters.escalation_trigger == "needs_review"
    assert escalation.selection.catalog_entry.entry_id == "gemma4-e4b"
    primary = next(j for j in jobs if j.job_type is JobType.QA_CRITERION and j.parameters.criterion_id == "REG-01")
    assert _assessment(runtime, primary)["escalation_requested"] is True
    scorecard = next(j for j in jobs if j.job_type is JobType.QA_SCORECARD)
    assert "escalation:REG-01" in {i.role for i in scorecard.inputs}

    evaluation = store_http.get(f"{V}/calls/{result.call_id}/evaluation", headers=session().read_headers).json()
    reg = next(v for v in evaluation["verdicts"] if v["criterion_id"] == "REG-01")
    assert reg["status"] == "PASS" and len(reg["model_attempts"]) == 2  # the escalation resolved the flagged primary


def test_without_an_escalation_model_a_flag_needs_a_human(make_runtime, store_http, session):
    runtime = make_runtime(behavior=FakeBehavior({"qa_criterion:REG-01": ["invalid_answer"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    jobs = _jobs(runtime, result.conversation_id)
    assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
    assert not any(j.job_type is JobType.QA_ESCALATION for j in jobs)
    primary = next(j for j in jobs if j.job_type is JobType.QA_CRITERION and j.parameters.criterion_id == "REG-01")
    usage = store_http.get(f"{V}/conversations/{result.conversation_id}/usage", headers=runtime.test_key.headers).json()["items"]
    assert next(u for u in usage if u["job_id"] == primary.id)["outcome"] == "validation_rejected"
    evaluation = store_http.get(f"{V}/calls/{result.call_id}/evaluation", headers=session().read_headers).json()
    assert evaluation["requires_human_review"] is True and evaluation["passed"] is False


def test_a_provider_failure_on_the_final_attempt_completes_flagged(make_runtime, store_http, session, clock):
    behavior = FakeBehavior({"qa_criterion:SEC-01": ["provider_error"] * 3})
    runtime = make_runtime(behavior=behavior)
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    for _ in range(4):
        worker.drain()
        clock.advance(runtime.client.parameters.retry_backoff_max_seconds)
    jobs = _jobs(runtime, result.conversation_id)
    assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
    sec = next(j for j in jobs if j.job_type is JobType.QA_CRITERION and j.parameters.criterion_id == "SEC-01")
    assert sec.attempt_count == 3
    assessment = _assessment(runtime, sec)
    assert assessment["status"] == "FLAGGED" and assessment["trigger"] == "provider_error"
    assert assessment["attempt"]["error_code"] == "provider_error"
    outcomes = [a.status.value for a in runtime.client.list_attempts(sec.id)]
    assert outcomes == ["failed", "failed", "succeeded"]


def test_long_transcripts_get_segments_and_a_pairwise_synthesis(make_runtime, store_http, session):
    runtime = make_runtime(summary_batch_turns=2)
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    jobs = _jobs(runtime, result.conversation_id)
    assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
    types = Counter(j.job_type for j in jobs)
    assert types[JobType.SUMMARY_SEGMENT] == 5  # nine turns, two per segment
    # pairwise while more than two remain, as the pre-split summarizer: (0,1) (2,3) | ((0,1),(2,3)) | final with segment 4
    assert types[JobType.SUMMARY_SYNTHESIS] == 4
    summary = store_http.get(f"{V}/calls/{result.call_id}/summary", headers=session().read_headers).json()
    assert summary["segments"] == 5 and summary["grounding"]["chunked"] is True


def _request_reanalysis(store_http, call_id, reviewer, kind="qa", key="rq-0001-abcdef"):
    response = store_http.post(f"{V}/calls/{call_id}/reanalysis-requests", json={"kind": kind},
                               headers={**write_headers(reviewer), "Idempotency-Key": key})
    assert response.status_code == 201, response.text
    return response.json()


def test_reanalysis_requests_become_graphs(make_runtime, store_http, session, clock):
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    reviewer = session(ReviewerRole.REVIEWER)
    request = _request_reanalysis(store_http, result.call_id, reviewer)
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    assert {g["kind"]: g["state"] for g in detail["results"]}["qa"] == "stale"

    clock.advance(1)
    assert runtime.reanalysis.poll_once() == 1
    fulfilled = store_http.get(f"{V}/reanalysis-requests/{request['id']}", headers=reviewer.read_headers).json()
    assert fulfilled["status"] == "fulfilled" and fulfilled["graph_id"]
    worker.drain()
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    assert {g["kind"]: g["state"] for g in detail["results"]}["qa"] == "available"
    assert detail["evaluation"]["version"] == 2
    ledger = runtime.ledger.get(result.conversation_id)
    assert [g["reason"] for g in ledger["graphs"]] == ["ingest", "reanalysis:qa"]


def test_a_speaker_correction_reruns_the_downstream_publishers(make_runtime, store_http, session, clock, tmp_path):
    import wave

    runtime = make_runtime()
    worker = runtime.connect()
    path = tmp_path / "mono.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x02" * 8000 * 24)
    result = runtime.ingestor.ingest_file(path)
    worker.drain()
    reviewer = session(ReviewerRole.REVIEWER)
    review = store_http.get(f"{V}/calls/{result.call_id}/review", headers=reviewer.read_headers).json()
    response = store_http.post(f"{V}/calls/{result.call_id}/speaker-corrections",
                               json={"correction": {"turn_id": 0, "speaker": "CALLER"}, "expected_version": review["review_version"]},
                               headers=write_headers(reviewer))
    assert response.status_code == 201, response.text
    clock.advance(1)
    assert runtime.reanalysis.poll_once() == 1
    worker.drain()
    transcript = store_http.get(f"{V}/calls/{result.call_id}/transcript", headers=reviewer.read_headers).json()
    assert transcript["speaker_attribution_version"] == 2
    assert next(t for t in transcript["turns"] if t["turn_id"] == 0)["speaker"] == "CALLER"
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    states = {g["kind"]: g["state"] for g in detail["results"]}
    assert states["qa"] == "available" and states["summary"] == "available" and detail["evaluation"]["version"] == 2


def test_a_rubric_draft_test_scores_in_draft_slots(make_runtime, store_http, session, clock):
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    supervisor = session(ReviewerRole.SUPERVISOR)
    current = store_http.get(f"{V}/rubrics/call1_standard_v2", headers=supervisor.read_headers).json()
    definition = dict(current["definition"], name="Standard (draft)", criteria=current["definition"]["criteria"][:2])
    saved = store_http.put(f"{V}/rubrics/call1_standard_v2/draft", json={"definition": definition, "expected_draft_revision": 0},
                           headers=write_headers(supervisor))
    assert saved.status_code == 200, saved.text
    created = store_http.post(f"{V}/rubrics/call1_standard_v2/draft/tests", json={"call_id": result.call_id, "expected_draft_revision": 1},
                              headers={**write_headers(supervisor), "Idempotency-Key": "studio-test-0001"})
    assert created.status_code == 201, created.text
    clock.advance(1)
    assert runtime.reanalysis.poll_once() == 1
    worker.drain()
    draft = store_http.get(f"{V}/reanalysis-requests/{created.json()['id']}/draft-result", headers=supervisor.read_headers).json()
    assert draft["state"] == "available"
    assert [v["criterion_id"] for v in draft["scorecard"]["verdicts"]] == ["REG-01", "SEC-01"]
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=supervisor.read_headers).json()
    assert detail["evaluation"]["version"] == 1  # the call's own QA is untouched


def test_a_reanalysis_that_cannot_be_planned_is_rejected(make_runtime, store_http, session, clock):
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    reviewer = session(ReviewerRole.REVIEWER)
    request = _request_reanalysis(store_http, result.call_id, reviewer, kind="summary", key="rq-0002-abcdef")
    clock.advance(1)
    runtime.reanalysis.poll_once()  # no transcript exists yet
    rejected = store_http.get(f"{V}/reanalysis-requests/{request['id']}", headers=reviewer.read_headers).json()
    assert rejected["status"] == "rejected" and "transcript" in rejected["rejected_reason"]
    worker.drain()


def test_an_embeddings_reanalysis_re_embeds_a_call_indexed_under_an_old_scheme(make_runtime, store_http, session, clock):
    """Contract 1.2.0: a call embedded under hashing-projection-v1 is not searched until reanalysis
    kind ``embeddings`` re-embeds it with the configured embedder. No result group goes stale."""
    from call1.contracts.contents import EmbeddingsContent, TurnEmbedding
    from call1.process.handlers.base import HandlerResult, Output
    from call1.process.handlers.embeddings import FakeEmbeddingsHandler

    class LegacyEmbeddings(FakeEmbeddingsHandler):
        adapter_id = "call1.code.hashing_projection"

        def run(self, job):
            turns = job.transcript().turns
            return HandlerResult(outputs={"embeddings": Output(EmbeddingsContent(
                scheme="hashing-projection-v1", dimensions=4, turn_vectors=[TurnEmbedding(turn_id=t.turn_id, vector=[1.0, 0, 0, 0]) for t in turns]))})

    runtime = make_runtime()
    runtime.registry.register(LegacyEmbeddings())
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    reviewer = session(ReviewerRole.REVIEWER)
    query = {"query": "this call may be recorded for quality", "call_id": result.call_id}
    before = store_http.post(f"{V}/search/semantic", json=query, headers=write_headers(reviewer)).json()
    assert before["count"] == 0 and before["calls_needing_reembedding"] == 1 and before["embedding_scheme"] == "fake-embedding-v1"

    runtime.registry.register(FakeEmbeddingsHandler())
    request = _request_reanalysis(store_http, result.call_id, reviewer, kind="embeddings", key="rq-0003-abcdef")
    detail = store_http.get(f"{V}/calls/{result.call_id}", headers=reviewer.read_headers).json()
    assert "stale" not in {g["state"] for g in detail["results"]}
    clock.advance(1)
    assert runtime.reanalysis.poll_once() == 1
    fulfilled = store_http.get(f"{V}/reanalysis-requests/{request['id']}", headers=reviewer.read_headers).json()
    assert fulfilled["status"] == "fulfilled"
    graph_jobs = [j for j in _jobs(runtime, result.conversation_id) if j.graph_id == fulfilled["graph_id"]]
    assert [j.job_type for j in graph_jobs] == [JobType.EMBEDDINGS]
    worker.drain()
    after = store_http.post(f"{V}/search/semantic", json=query, headers=write_headers(reviewer)).json()
    assert after["count"] >= 1 and after["calls_needing_reembedding"] == 0
    assert runtime.ledger.get(result.conversation_id)["graphs"][-1]["reason"] == "reanalysis:embeddings"


def test_the_store_host_command_requests_re_embedding_for_every_old_scheme_call(make_runtime, store, store_http, session, clock, monkeypatch):
    """``python -m call1.store request-reembedding``: one ``embeddings`` request per call indexed
    under another scheme, idempotent, fulfilled by Process like any reanalysis request."""
    from call1.contracts.contents import EmbeddingsContent, TurnEmbedding
    from call1.process.handlers.base import HandlerResult, Output
    from call1.process.handlers.embeddings import FakeEmbeddingsHandler
    from call1.store import __main__ as store_cli

    class LegacyEmbeddings(FakeEmbeddingsHandler):
        def run(self, job):
            return HandlerResult(outputs={"embeddings": Output(EmbeddingsContent(
                scheme="hashing-projection-v1", dimensions=2, turn_vectors=[TurnEmbedding(turn_id=t.turn_id, vector=[1.0, 0]) for t in job.transcript().turns]))})

    runtime = make_runtime()
    runtime.registry.register(LegacyEmbeddings())
    worker = runtime.connect()
    calls = [runtime.ingestor.ingest_file(SAMPLE, agent_id=f"agent-{n}", external_call_ref=f"ref-{n}") for n in range(2)]
    worker.drain()
    runtime.registry.register(FakeEmbeddingsHandler())
    monkeypatch.setattr(store_cli, "_open_store", lambda args: store)
    assert store_cli.main(["request-reembedding"]) == 0
    assert store_cli.main(["request-reembedding"]) == 0  # idempotent: the same requests
    reviewer = session(ReviewerRole.REVIEWER)
    for call in calls:
        items = store_http.get(f"{V}/calls/{call.call_id}/reanalysis-requests", headers=reviewer.read_headers).json()["items"]
        assert [r["kind"] for r in items] == ["embeddings"]
    clock.advance(1)
    while runtime.reanalysis.poll_once():
        pass
    worker.drain()
    body = store_http.post(f"{V}/search/semantic", json={"query": "this call may be recorded for quality"}, headers=write_headers(reviewer)).json()
    assert body["calls_needing_reembedding"] == 0 and {h["call_id"] for h in body["results"]} >= {c.call_id for c in calls}
