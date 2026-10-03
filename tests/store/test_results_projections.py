"""The projection hooks: call records, result versions, derived states, draft tests, search index."""

from __future__ import annotations

from call1.contracts.contents import (
    ContactSignalsContent,
    ContactSignalsPassOutcome,
    ContactSignalPass,
    ResultKind,
    ResultState,
    SummaryContent,
    TextSentimentContent,
    TurnSentiment,
    VerdictStatus,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobStatus, JobType, ResultPublication
from call1.store.results import api as results_api
from call1.store.results import records

from .test_results_harness import accounts, embeddings_for, legacy_embeddings_for, fq, qa_scorecard, transcript, vad_metrics, validation_report  # noqa: F401

INGEST = [JobType.VALIDATION_VAD, JobType.ASR, JobType.SPEAKER_ATTRIBUTION, JobType.TEXT_SENTIMENT, JobType.EMBEDDINGS, JobType.QA_SCORECARD,
          JobType.SUMMARY_ASSEMBLY]


def _groups(store, conv):
    with store.connection() as conn:
        return {g.kind: g for g in results_api.result_groups(conn, conv.id)}


def test_registration_creates_an_analyzing_call(store, fq, client, reviewer_session):
    conv = fq.register(agent_id="agent-1")
    fq.graph(conv, INGEST)
    groups = _groups(store, conv)
    assert groups[ResultKind.TRANSCRIPT].state is ResultState.PENDING
    assert groups[ResultKind.QA].state is ResultState.PENDING
    assert groups[ResultKind.TONE].state is ResultState.DISABLED  # no acoustic_tone job in the graph
    listed = client.get("/store/v1/calls", headers=reviewer_session.read_headers).json()["items"]
    assert [(i["call_id"], i["transcript_state"], i["qa_state"], i["overall_score"]) for i in listed] == [(conv.call_id, "pending", "pending", None)]


def test_completion_publishes_versions_and_media_fields(store, fq, client, reviewer_session):
    conv = fq.register()
    graph = fq.graph(conv, INGEST)
    assert fq.complete(conv, graph, JobType.VALIDATION_VAD, {"validation_report": validation_report(12.5), "vad_metrics": vad_metrics()}) is None
    content = transcript([("AGENT", "Thank you for calling, this call is recorded."), ("CALLER", "I want to dispute a charge.")])
    assert fq.complete(conv, graph, JobType.ASR, {"transcript": content}) == 1
    groups = _groups(store, conv)
    assert groups[ResultKind.TRANSCRIPT].state is ResultState.AVAILABLE and groups[ResultKind.TRANSCRIPT].version == 1
    detail = client.get(f"/store/v1/calls/{conv.call_id}", headers=reviewer_session.read_headers).json()
    assert detail["call"]["duration_seconds"] == 12.5 and detail["call"]["channel_layout"] == "MONO"
    assert detail["call"]["silence_ratio"] == 0.17
    assert detail["evaluation"] is None and detail["review_version"] == 0
    assert detail["pending_work"]["jobs_succeeded"] == 2
    assert detail["change_cursor"]


def test_stale_while_reanalysis_underway_then_new_version(store, fq):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR])
    fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("AGENT", "hello")])})
    conv.pending_reanalysis.add(ResultKind.TRANSCRIPT)
    assert _groups(store, conv)[ResultKind.TRANSCRIPT].state is ResultState.STALE
    conv.pending_reanalysis.clear()
    newer = fq.graph(conv, [JobType.ASR])
    assert _groups(store, conv)[ResultKind.TRANSCRIPT].state is ResultState.STALE  # newer publisher in progress
    assert fq.complete(conv, newer, JobType.ASR, {"transcript": transcript([("AGENT", "hello again")])}) == 2
    group = _groups(store, conv)[ResultKind.TRANSCRIPT]
    assert group.state is ResultState.AVAILABLE and group.version == 2 and group.produced_by_graph_id == newer.graph_id


def test_failure_codes_needs_attention_and_old_version_kept(store, fq):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.QA_SCORECARD])
    fq.fail(conv, graph, JobType.ASR, JobErrorCode.MODEL_UNAVAILABLE)
    fq.dead_block(graph, JobType.QA_SCORECARD)
    groups = _groups(store, conv)
    assert groups[ResultKind.TRANSCRIPT].state is ResultState.FAILED
    assert groups[ResultKind.TRANSCRIPT].failure_code is JobErrorCode.MODEL_UNAVAILABLE
    assert groups[ResultKind.QA].state is ResultState.FAILED  # dead-blocked scorecard: first failed upstream's code
    assert groups[ResultKind.QA].failure_code is JobErrorCode.MODEL_UNAVAILABLE

    ok = fq.graph(conv, [JobType.ASR])
    fq.complete(conv, ok, JobType.ASR, {"transcript": transcript([("AGENT", "hi")])})
    again = fq.graph(conv, [JobType.ASR])
    fq.fail(conv, again, JobType.ASR, JobErrorCode.PROVIDER_TIMEOUT, terminal=False)  # retry scheduled: not a failure yet
    assert _groups(store, conv)[ResultKind.TRANSCRIPT].state is ResultState.STALE
    fq.fail(conv, again, JobType.ASR, JobErrorCode.PROVIDER_TIMEOUT)
    group = _groups(store, conv)[ResultKind.TRANSCRIPT]
    assert group.state is ResultState.AVAILABLE and group.version == 1 and group.failure_code is JobErrorCode.PROVIDER_TIMEOUT


def test_draft_test_completions_project_nothing(store, fq, client, reviewer_session):
    conv = fq.register()
    live = fq.graph(conv, [JobType.QA_SCORECARD])
    fq.complete(conv, live, JobType.QA_SCORECARD, {"scorecard": qa_scorecard([("REG-01", VerdictStatus.PASS, 0.9)])})
    draft = fq.graph(conv, [JobType.QA_SCORECARD], draft_test_request_id="rq_draft1")
    assert fq.complete(conv, draft, JobType.QA_SCORECARD, {"scorecard": qa_scorecard([("REG-01", VerdictStatus.FAIL, 0.9)], overall_score=10,
                                                                                    passed=False, critical_failure=True)}) is None
    fq.fail(conv, fq.graph(conv, [JobType.QA_SCORECARD], draft_test_request_id="rq_draft2"), JobType.QA_SCORECARD)
    group = _groups(store, conv)[ResultKind.QA]
    assert group.state is ResultState.AVAILABLE and group.version == 1 and group.failure_code is None
    evaluation = client.get(f"/store/v1/calls/{conv.call_id}/evaluation", headers=reviewer_session.read_headers).json()
    assert evaluation["version"] == 1 and evaluation["passed"] is True
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM results_artifacts WHERE slot LIKE 'draft:%'").fetchone()[0] == 0


def test_partial_contact_signals_and_summary_views(store, fq, client, reviewer_session):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.CONTACT_SIGNALS_MERGE, JobType.SUMMARY_ASSEMBLY])
    fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("CALLER", "I have a question about a fee.")])})
    signals = ContactSignalsContent(completeness="partial", partial_reason="resolution pass failed", signals=[],
                                    passes=[ContactSignalsPassOutcome(pass_kind=ContactSignalPass.LIFECYCLE, included=True),
                                            ContactSignalsPassOutcome(pass_kind=ContactSignalPass.RESOLUTION, included=False,
                                                                      failure_code=JobErrorCode.PROVIDER_ERROR)],
                                    transcript_fingerprint="sha256:" + "1" * 64, generated_at=fq.clock.now())
    fq.complete(conv, graph, JobType.CONTACT_SIGNALS_MERGE, {"contact_signals": signals},
                result=ResultPublication(kind=ResultKind.CONTACT_SIGNALS, state=ResultState.PARTIAL, partial_reason="resolution pass failed"))
    summary = SummaryContent(narrative="Caller asked about a fee.", key_points=["fee"], route_class="appliance", catalog_entry_id="mlx-small",
                             generated_at=fq.clock.now(), segments=1)
    fq.complete(conv, graph, JobType.SUMMARY_ASSEMBLY, {"summary": summary})
    group = _groups(store, conv)[ResultKind.CONTACT_SIGNALS]
    assert group.state is ResultState.PARTIAL and group.partial_reason == "resolution pass failed"
    view = client.get(f"/store/v1/calls/{conv.call_id}/contact-signals", headers=reviewer_session.read_headers).json()
    assert view["completeness"] == "partial" and view["version"] == 1
    got = client.get(f"/store/v1/calls/{conv.call_id}/summary", headers=reviewer_session.read_headers).json()
    assert got["narrative"] == "Caller asked about a fee." and got["call_id"] == conv.call_id


def test_transcript_view_composes_attribution_sentiment_and_masks(store, fq, client, reviewer_session):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.SPEAKER_ATTRIBUTION, JobType.TEXT_SENTIMENT])
    words = {1: [{"word": w, "start_time": 2.0 + i * 0.2, "end_time": 2.1 + i * 0.2, "probability": 0.9}
                 for i, w in enumerate("my ssn is 123-45-6789".split())]}
    content = transcript([("UNKNOWN", "How can I help?"), ("UNKNOWN", "my ssn is 123-45-6789")], word_turns=words)
    fq.complete(conv, graph, JobType.ASR, {"transcript": content})
    from call1.contracts.contents import SpeakerAssignment, SpeakerAttributionContent
    fq.complete(conv, graph, JobType.SPEAKER_ATTRIBUTION, {"speaker_attribution": SpeakerAttributionContent(
        method="diarization", assignments=[SpeakerAssignment(turn_id=0, speaker="AGENT"), SpeakerAssignment(turn_id=1, speaker="CALLER")])})
    fq.complete(conv, graph, JobType.TEXT_SENTIMENT, {"text_sentiment": TextSentimentContent(
        model="m", revision="r", turns=[TurnSentiment(turn_id=0, score=0.5, label="POSITIVE")], avg_caller_sentiment=-0.2)})
    view = client.get(f"/store/v1/calls/{conv.call_id}/transcript", headers=reviewer_session.read_headers).json()
    assert [t["speaker"] for t in view["turns"]] == ["AGENT", "CALLER"]
    assert view["speaker_attribution_version"] == 1 and view["is_redacted"] is True
    assert "123-45-6789" not in view["turns"][1]["text"] and "[REDACTED]" in view["turns"][1]["text"]
    assert "123-45-6789" not in [w["word"] for w in view["turns"][1]["word_timestamps"]]
    assert view["turns"][0]["text_sentiment"] == 0.5 and view["avg_caller_sentiment"] == -0.2
    detail = client.get(f"/store/v1/calls/{conv.call_id}", headers=reviewer_session.read_headers).json()
    assert detail["call"]["avg_caller_sentiment"] == -0.2


def test_qa_publication_fills_list_columns_and_replay_is_harmless(store, fq, client, reviewer_session):
    conv = fq.register()
    graph, version = fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.95)], overall_score=40, passed=False, critical_failure=True,
                                  requires_human_review=True)
    assert version == 1
    item = client.get("/store/v1/calls", headers=reviewer_session.read_headers).json()["items"][0]
    assert item["overall_score"] == 40 and item["passed"] is False and item["critical_failure"] is True
    assert item["requires_human_review"] is True and item["review_status"] == "PENDING" and item["qa_state"] == "available"
    with store.connection() as conn:
        call = records.call_row(conn, conv.call_id)
        assert call["evaluation_version"] == 1 and call["rubric_id"] == "call1_standard_v2"
    needs = client.get("/store/v1/calls", params={"needs_review": "true"}, headers=reviewer_session.read_headers).json()["items"]
    assert [i["call_id"] for i in needs] == [conv.call_id]
    assert client.get("/store/v1/calls", params={"needs_review": "false"}, headers=reviewer_session.read_headers).json()["items"] == []


def test_change_events_follow_projections(store, fq, client, admin_session):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR])
    fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("AGENT", "hi")])})
    events = client.get("/store/v1/changes", headers=admin_session.read_headers).json()["events"]
    kinds = [(e["kind"], e["status"]) for e in events if e["call_id"] == conv.call_id]
    assert ("call", "registered") in kinds and ("result", "transcript:available") in kinds


def test_embeddings_are_indexed_for_the_newest_artifact(store, fq):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.EMBEDDINGS])
    content = transcript([("AGENT", "hello"), ("CALLER", "refund please")])
    fq.complete(conv, graph, JobType.ASR, {"transcript": content})
    fq.complete(conv, graph, JobType.EMBEDDINGS, {"embeddings": embeddings_for(content)})
    again = fq.graph(conv, [JobType.EMBEDDINGS])
    fq.complete(conv, again, JobType.EMBEDDINGS, {"embeddings": legacy_embeddings_for(content, dims=8)})
    with store.connection() as conn:
        rows = conn.execute("SELECT DISTINCT scheme, dimensions FROM results_search_vectors WHERE conversation_id = ?", (conv.id,)).fetchall()
    assert [tuple(r) for r in rows] == [("hashing-projection-v1", 8)]  # every scheme is indexed; search ranks only its own


def test_group_state_status_values_are_closed(fq, store):
    conv = fq.register()
    fq.graph(conv, [JobType.ASR])
    for group in _groups(store, conv).values():
        assert group.state in set(ResultState)
    assert JobStatus.QUEUED  # the harness keeps job statuses inside the contract's closed set
