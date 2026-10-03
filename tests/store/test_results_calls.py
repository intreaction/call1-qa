"""Call reads over HTTP: listing, detail, evaluation versions, audio with Range and masking, search."""

from __future__ import annotations

import io
import wave

from call1.contracts.contents import EnrichmentContent, NumericEntityContent, TurnEnrichment, VerdictStatus
from call1.contracts.jobs import JobType
from call1 import embedding

from .test_results_harness import accounts, embeddings_for, legacy_embeddings_for, fq, qa_scorecard, transcript  # noqa: F401


def _wav(seconds: float = 4.0, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x10\x27" * int(seconds * rate))
    return buf.getvalue()


def test_list_calls_filters_and_pages(fq, client, reviewer_session):
    calls = [fq.register(agent_id=f"agent-{i % 2}", external_call_ref=f"PBX-{i}") for i in range(5)]
    h = reviewer_session.read_headers
    first = client.get("/store/v1/calls", params={"limit": 2}, headers=h).json()
    assert [i["call_id"] for i in first["items"]] == [calls[4].call_id, calls[3].call_id]
    second = client.get("/store/v1/calls", params={"limit": 2, "page_token": first["next_page_token"]}, headers=h).json()
    assert [i["call_id"] for i in second["items"]] == [calls[2].call_id, calls[1].call_id]
    third = client.get("/store/v1/calls", params={"limit": 2, "page_token": second["next_page_token"]}, headers=h).json()
    assert [i["call_id"] for i in third["items"]] == [calls[0].call_id] and third.get("next_page_token") is None
    by_agent = client.get("/store/v1/calls", params={"agent_id": "agent-1"}, headers=h).json()["items"]
    assert {i["call_id"] for i in by_agent} == {calls[1].call_id, calls[3].call_id}
    by_text = client.get("/store/v1/calls", params={"text": "PBX-2"}, headers=h).json()["items"]
    assert [i["call_id"] for i in by_text] == [calls[2].call_id]
    bad = client.get("/store/v1/calls", params={"page_token": "not-a-token"}, headers=h)
    assert bad.status_code == 422 and bad.json()["code"] == "validation_failed"


def test_calls_need_a_session_and_unknown_calls_are_404(fq, client, reviewer_session, service_key_headers):
    assert client.get("/store/v1/calls").status_code == 401
    assert client.get("/store/v1/calls", headers=service_key_headers).json()["code"] == "forbidden"
    for path in ("", "/transcript", "/evaluation", "/evaluations/1", "/summary", "/contact-signals", "/audio", "/review"):
        r = client.get(f"/store/v1/calls/call_missing{path}", headers=reviewer_session.read_headers)
        assert r.status_code == 404 and r.json()["code"] == "not_found", path
    conv = fq.register()
    for path in ("/transcript", "/evaluation", "/summary", "/contact-signals"):
        assert client.get(f"/store/v1/calls/{conv.call_id}{path}", headers=reviewer_session.read_headers).status_code == 404


def test_evaluation_versions_stay_readable_and_evidence_is_masked(fq, client, reviewer_session):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR])
    fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("CALLER", "my card is 4111 1111 1111 1111")])})
    fq.ingest_qa(conv, [("SEC-01", VerdictStatus.PASS, 0.9)], evidence="card 4111 1111 1111 1111 read back")
    fq.ingest_qa(conv, [("SEC-01", VerdictStatus.FAIL, 0.9)], overall_score=30, passed=False)
    h = reviewer_session.read_headers
    current = client.get(f"/store/v1/calls/{conv.call_id}/evaluation", headers=h).json()
    assert current["version"] == 2 and current["verdicts"][0]["status"] == "FAIL"
    first = client.get(f"/store/v1/calls/{conv.call_id}/evaluations/1", headers=h).json()
    assert first["version"] == 1 and "4111 1111 1111 1111" not in first["verdicts"][0]["quoted_evidence"]
    assert "[REDACTED]" in first["verdicts"][0]["quoted_evidence"]
    assert client.get(f"/store/v1/calls/{conv.call_id}/evaluations/3", headers=h).status_code == 404


def test_audio_streams_with_range_and_mutes_sensitive_spans(fq, client, reviewer_session):
    from call1.contracts.artifacts import ArtifactKind
    conv = fq.register()
    audio = _wav()
    fq.artifact(conv, ArtifactKind.SOURCE_AUDIO, audio, content_type="audio/wav")
    h = reviewer_session.read_headers
    # Before a transcript and its PII findings exist, the audio is withheld (contract 1.2.0, fail closed).
    withheld = client.get(f"/store/v1/calls/{conv.call_id}/audio", headers=h)
    assert withheld.status_code == 503 and withheld.json()["details"]["reason"] == "pii_findings_pending"
    assert withheld.json()["retryable"] is True

    graph = fq.graph(conv, [JobType.ASR, JobType.ENRICHMENT])
    fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("AGENT", "hello there"), ("CALLER", "nothing sensitive")])})
    whole = client.get(f"/store/v1/calls/{conv.call_id}/audio", headers=h)
    assert whole.status_code == 200 and whole.content == audio and whole.headers["content-type"] == "audio/wav"
    ranged = client.get(f"/store/v1/calls/{conv.call_id}/audio", headers={**h, "Range": "bytes=0-99"})
    assert ranged.status_code == 206 and ranged.content == audio[:100]

    graph = fq.graph(conv, [JobType.ASR, JobType.ENRICHMENT])
    fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("AGENT", "hello there"), ("CALLER", "account 884-210-993 please")])})
    fq.complete(conv, graph, JobType.ENRICHMENT, {"enrichment": EnrichmentContent(turns=[TurnEnrichment(turn_id=1, numeric_entities=[
        NumericEntityContent(raw_text="884-210-993", normalized_value="884210993", entity_type="ACCOUNT_NUMBER", start_time=2.5, end_time=3.0)])])})
    masked = client.get(f"/store/v1/calls/{conv.call_id}/audio", headers=h)
    assert masked.status_code == 200 and masked.content != audio
    with wave.open(io.BytesIO(masked.content)) as w:
        frames = w.readframes(w.getnframes())
    silent = frames[int(2.6 * 8000) * 2: int(2.9 * 8000) * 2]
    assert set(silent) == {0}
    assert frames[:200] == audio[44:244]  # outside the spoken value the audio is unchanged
    again = client.get(f"/store/v1/calls/{conv.call_id}/audio", headers={**h, "Range": "bytes=10-19"})
    assert again.status_code == 206 and again.content == masked.content[10:20]


def test_audio_fails_closed_when_it_cannot_be_masked(fq, client, reviewer_session, monkeypatch):
    from call1.contracts.artifacts import ArtifactKind
    from call1.store.results import audio as audio_module
    monkeypatch.setattr(audio_module.shutil, "which", lambda name: None)
    conv = fq.register()
    fq.artifact(conv, ArtifactKind.SOURCE_AUDIO, b"ID3fake-mp3-bytes", content_type="audio/mpeg")
    graph = fq.graph(conv, [JobType.ASR])
    fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("CALLER", "ssn 123-45-6789")])})
    r = client.get(f"/store/v1/calls/{conv.call_id}/audio", headers=reviewer_session.read_headers)
    assert r.status_code == 503 and r.json()["code"] == "store_unavailable" and r.json()["retryable"] is False


def test_semantic_search_ranks_masks_and_filters(fq, client, reviewer_session):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.EMBEDDINGS])
    content = transcript([("AGENT", "Thank you for calling, how can I help"), ("CALLER", "I want to dispute an unauthorized charge on 123-45-6789"),
                          ("AGENT", "Let me verify your identity first")])
    fq.complete(conv, graph, JobType.ASR, {"transcript": content})
    fq.complete(conv, graph, JobType.EMBEDDINGS, {"embeddings": embeddings_for(content)})
    other = fq.register()
    other_graph = fq.graph(other, [JobType.ASR, JobType.EMBEDDINGS])
    other_content = transcript([("AGENT", "The weather is lovely today")])
    fq.complete(other, other_graph, JobType.ASR, {"transcript": other_content})
    fq.complete(other, other_graph, JobType.EMBEDDINGS, {"embeddings": embeddings_for(other_content)})

    h = reviewer_session.headers
    body = client.post("/store/v1/search/semantic", json={"query": "dispute a charge", "top_k": 2}, headers=h).json()
    assert body["count"] == 2 and body["results"][0]["call_id"] == conv.call_id and body["results"][0]["turn_id"] == 1
    assert "123-45-6789" not in body["results"][0]["text"]
    scores = [r["similarity_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 1 for s in scores)
    agents = client.post("/store/v1/search/semantic", json={"query": "verify identity", "speaker_filter": "AGENT", "call_id": conv.call_id},
                         headers=h).json()
    assert agents["results"] and {r["speaker"] for r in agents["results"]} == {"AGENT"} and {r["call_id"] for r in agents["results"]} == {conv.call_id}
    none = client.post("/store/v1/search/semantic", json={"query": "dispute", "min_score": 1.0}, headers=h).json()
    assert none == {"query": "dispute", "count": 0, "results": [], "embedding_scheme": "fake-embedding-v1", "calls_needing_reembedding": 0}


def test_semantic_search_ranks_only_the_configured_scheme(fq, client, reviewer_session):
    """Contract 1.2.0: vectors of another scheme (hashing-projection-v1) are never compared with the
    query; their calls are counted as needing re-embedding."""
    current = fq.register()
    graph = fq.graph(current, [JobType.ASR, JobType.EMBEDDINGS])
    content = transcript([("CALLER", "I want to dispute this charge")])
    fq.complete(current, graph, JobType.ASR, {"transcript": content})
    fq.complete(current, graph, JobType.EMBEDDINGS, {"embeddings": embeddings_for(content)})
    legacy = fq.register()
    legacy_graph = fq.graph(legacy, [JobType.ASR, JobType.EMBEDDINGS])
    legacy_content = transcript([("CALLER", "I want to dispute this charge")])
    fq.complete(legacy, legacy_graph, JobType.ASR, {"transcript": legacy_content})
    fq.complete(legacy, legacy_graph, JobType.EMBEDDINGS, {"embeddings": legacy_embeddings_for(legacy_content)})

    body = client.post("/store/v1/search/semantic", json={"query": "dispute this charge"}, headers=reviewer_session.headers).json()
    assert {r["call_id"] for r in body["results"]} == {current.call_id}
    assert body["embedding_scheme"] == embedding.FAKE_SCHEME and body["calls_needing_reembedding"] == 1
    scoped = client.post("/store/v1/search/semantic", json={"query": "dispute", "call_id": legacy.call_id}, headers=reviewer_session.headers).json()
    assert scoped["count"] == 0 and scoped["calls_needing_reembedding"] == 1

    # Re-embedding (what reanalysis kind "embeddings" produces) brings the call back into search.
    again = fq.graph(legacy, [JobType.EMBEDDINGS])
    fq.complete(legacy, again, JobType.EMBEDDINGS, {"embeddings": embeddings_for(legacy_content)})
    body = client.post("/store/v1/search/semantic", json={"query": "dispute this charge"}, headers=reviewer_session.headers).json()
    assert {r["call_id"] for r in body["results"]} == {current.call_id, legacy.call_id} and body["calls_needing_reembedding"] == 0


def test_semantic_search_without_the_embedder_is_a_clear_503(fq, client, reviewer_session, admin_session, monkeypatch, tmp_path):
    monkeypatch.setenv("CALL1_EMBEDDING_BACKEND", "nemotron")
    monkeypatch.setenv("CALL1_EMBEDDING_PATH", str(tmp_path / "no-weights"))
    r = client.post("/store/v1/search/semantic", json={"query": "dispute"}, headers=reviewer_session.headers)
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "search_unavailable" and body["retryable"] is False
    assert body["details"] == {"reason": "not_installed", "embedding_scheme": embedding.MODEL_SCHEME}
    status = client.get("/store/v1/status/detail", headers=admin_session.read_headers).json()
    assert status["search_embedder"]["state"] == "not_installed" and status["search_embedder"]["scheme"] == embedding.MODEL_SCHEME
    monkeypatch.setenv("CALL1_EMBEDDING_BACKEND", "fake")
    assert client.get("/store/v1/status/detail", headers=admin_session.read_headers).json()["search_embedder"]["state"] == "fake"


def test_store_serves_no_score_before_the_scorecard(fq, client, reviewer_session):
    conv = fq.register()
    fq.graph(conv, [JobType.QA_SCORECARD])
    item = client.get("/store/v1/calls", headers=reviewer_session.read_headers).json()["items"][0]
    assert item["overall_score"] is None and item["qa_state"] == "pending"
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.PASS, 0.99)])
    assert qa_scorecard  # factory shared with the other modules
