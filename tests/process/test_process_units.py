"""Unit tests for Process pieces that need no Store: the Store client's transport rules, config,
the graph planner, segmentation, the search embedder and the fake handlers' content."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from call1.contracts.admin import MaskingSettings
from call1.contracts.artifacts import ArtifactKind, Sensitivity, UploadGrantRequest
from call1.contracts.catalog import ModelPurpose
from call1.contracts.common import CONTRACT_PARAMETERS, CONTRACT_VERSION, canonical_digest
from call1.contracts.contents import SpeakerRole, TranscriptContent, TranscriptTurnContent
from call1.contracts.custody import RouteClass
from call1.contracts.jobs import JOB_TYPE_RULES, JobDefinition, JobType, MemorySlot, SpeakerCorrection
from call1.contracts.rubrics import CheckType, RubricCheck, RubricCriterion, RubricDefinition, RubricVersionRef
from call1 import embedding
from call1.process.audio import AudioInfo
from call1.process.catalog import CatalogError, seeded_catalog
from call1.process.config import ConfigError, ProcessConfig, SlotSizes
from call1.process.graph import GraphPlanner, PlanError, Src, new_jobs
from call1.process.handlers import build_registry
from call1.process.store_client import ContractMismatch, StoreClient, StoreError, StoreUnavailable
from call1.process.transcripts import apply_speaker_correction, plan_segments

BASE = "http://localhost:8010"


def _client(handler, **kw) -> StoreClient:
    return StoreClient(BASE, "c1sk_test_secret", http=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _s: None, **kw)


def _contract(version: str):
    return {"contract_version": version, "parameters": CONTRACT_PARAMETERS.model_dump(mode="json")}


# --- store client ------------------------------------------------------------------------------


def test_contract_check_refuses_another_major_and_keeps_parameters():
    ok = _client(lambda r: httpx.Response(200, json=_contract(CONTRACT_VERSION)))
    assert ok.check_contract().contract_version == CONTRACT_VERSION
    minor = _client(lambda r: httpx.Response(200, json=_contract("1.9.0")))
    assert minor.check_contract().contract_version == "1.9.0"
    with pytest.raises(ContractMismatch):
        _client(lambda r: httpx.Response(200, json=_contract("2.0.0"))).check_contract()


def test_retries_5xx_and_connection_errors_then_gives_up():
    calls = []

    def flaky(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503, json={"code": "store_unavailable", "message": "busy", "details": {}, "retryable": True})
        return httpx.Response(200, json=_contract(CONTRACT_VERSION))

    assert _client(flaky).check_contract() and len(calls) == 3

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(StoreUnavailable):
        _client(down, retries=2).check_contract()


def test_errors_carry_the_contract_envelope_and_the_key_goes_only_to_store():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(409, json={"code": "claim_token_stale", "message": "stale", "details": {"status": "QUEUED"}, "retryable": False})

    client = _client(handler)
    with pytest.raises(StoreError) as caught:
        client.get_job("job_1")
    assert caught.value.code == "claim_token_stale" and caught.value.status == 409 and caught.value.details == {"status": "QUEUED"}
    assert len(seen) == 1 and seen[0].headers["authorization"] == "Bearer c1sk_test_secret"


def _grant(url):
    return {"upload_id": "upl_1", "artifact_id": "art_1", "method": "PUT", "url": url, "headers": {"Content-Type": "audio/wav"},
            "expires_at": "2026-09-25T12:00:00Z", "max_bytes": 4}


def _artifact():
    return {"id": "art_1", "conversation_id": "conv_1", "kind": "source_audio", "slot": "", "content_type": "audio/wav", "size_bytes": 4,
            "checksum": "sha256:" + "a" * 64, "content_contract": "audio.v1", "sensitivity": "raw", "producing_job_id": None, "labels": {},
            "linked": True, "linked_by_receipt_id": None, "version": 1, "storage": "object", "committed_at": "2026-09-25T12:00:00Z",
            "superseded_by": None}


def test_dev_mode_grant_urls_are_accepted_on_the_store_origin_only():
    puts = []

    def handler(request):
        if request.method == "PUT":
            puts.append(request)
            return httpx.Response(200, json={})
        if request.url.path.endswith("/uploads"):
            return httpx.Response(201, json=_grant(f"{BASE}/store/transfer/uploads/upl_1?token=" + "t" * 20))
        return httpx.Response(201, json=_artifact())

    body = UploadGrantRequest(kind=ArtifactKind.SOURCE_AUDIO, content_type="audio/wav", size_bytes=4, checksum="sha256:" + "a" * 64,
                              content_contract="audio.v1", sensitivity=Sensitivity.RAW)
    assert _client(handler).upload_artifact("conv_1", body, b"abcd").id == "art_1"
    assert len(puts) == 1 and "authorization" not in puts[0].headers  # the grant is its own capability

    def foreign(request):
        if request.url.path.endswith("/uploads"):
            return httpx.Response(201, json=_grant("http://evil.example:8010/steal"))
        return httpx.Response(201, json=_artifact())

    with pytest.raises(StoreError) as caught:
        _client(foreign).upload_artifact("conv_1", body, b"abcd")
    assert caught.value.code == "grant_rejected"


def test_dev_mode_treats_the_loopback_names_as_one_store_host():
    body = UploadGrantRequest(kind=ArtifactKind.SOURCE_AUDIO, content_type="audio/wav", size_bytes=4, checksum="sha256:" + "a" * 64,
                              content_contract="audio.v1", sensitivity=Sensitivity.RAW)

    def handler_for(grant_url):
        def handler(request):
            if request.method == "PUT":
                return httpx.Response(200, json={})
            if request.url.path.endswith("/uploads"):
                return httpx.Response(201, json=_grant(grant_url))
            return httpx.Response(201, json=_artifact())
        return handler

    # Store names itself localhost; Process may be configured with 127.0.0.1 (the Vite proxy's address)
    for base in ("http://127.0.0.1:8010", "http://[::1]:8010", "http://localhost:8010"):
        client = StoreClient(base, "c1sk_test", http=httpx.Client(transport=httpx.MockTransport(
            handler_for("http://localhost:8010/store/transfer/uploads/upl_1?token=" + "t" * 20))), sleep=lambda _s: None)
        assert client.upload_artifact("conv_1", body, b"abcd").id == "art_1", base
    for wrong in ("http://localhost:9999/x", "http://example.com:8010/x", "https://localhost:8010/x"):
        client = StoreClient("http://127.0.0.1:8010", "c1sk_test", http=httpx.Client(transport=httpx.MockTransport(handler_for(wrong))),
                             sleep=lambda _s: None)
        with pytest.raises(StoreError) as caught:
            client.upload_artifact("conv_1", body, b"abcd")
        assert caught.value.code == "grant_rejected", wrong


def test_display_reads_can_skip_retries_and_shorten_the_timeout():
    seen = []

    def handler(request):
        seen.append(request.extensions.get("timeout"))
        raise httpx.ConnectError("refused")

    client = _client(handler)
    with pytest.raises(StoreUnavailable):
        client.get_progress("conv_1", retry=False, timeout=1.5)
    assert len(seen) == 1 and seen[0]["connect"] == 1.5
    seen.clear()
    with pytest.raises(StoreUnavailable):
        client.get_progress("conv_1")
    assert len(seen) == client.retries + 1


def test_downloads_are_checksum_verified():
    client = _client(lambda r: httpx.Response(200, content=b"not the bytes"))
    from call1.contracts.artifacts import Artifact

    with pytest.raises(StoreError) as caught:
        client.download(Artifact.model_validate(_artifact()))
    assert caught.value.code == "checksum_mismatch"


# --- config ------------------------------------------------------------------------------------


def test_registration_sends_only_the_call_metadata_the_caller_supplied():
    from datetime import datetime, timezone

    from call1.contracts.calls import ConversationRegistration, IngestionKind, SourceKind, SourceReference
    from call1.process.ingest import InvalidCallMetadata, build_call_metadata

    sent = {}

    def handler(request):
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json={"conversation": {
            "id": "conv_1", "ingestion_kind": "call_audio", "call_id": "call_1", "created_at": "2026-09-25T00:00:00Z",
            "source": sent["body"]["source"], "call_metadata": {"agent_id": "a-7", "agent_display_name": "Sam"}},
            "created": False, "metadata_updated": True, "updated_fields": ["agent_display_name"]})

    source = SourceReference(kind=SourceKind.API_UPLOAD, content_digest="sha256:" + "0" * 64, received_at=datetime.now(timezone.utc))
    metadata = build_call_metadata(agent_display_name="  Sam ", agent_extension=None, agent_id="")
    registered = _client(handler).register_conversation(
        ConversationRegistration(ingestion_kind=IngestionKind.CALL_AUDIO, source=source, call_metadata=metadata))
    # No agent_id: "Unknown" (the model default) is never sent, so a re-upload cannot reset a known agent.
    assert sent["body"]["call_metadata"] == {"agent_display_name": "Sam"}
    assert registered.metadata_updated and registered.updated_fields == ["agent_display_name"]
    assert build_call_metadata().model_dump(exclude_unset=True) == {}
    full = build_call_metadata(agent_id="a-104", agent_display_name="Samantha", agent_extension="104", agent_channel=1)
    assert full.model_dump(exclude_unset=True) == {"agent_id": "a-104", "agent_display_name": "Samantha", "agent_extension": "104",
                                                   "agent_channel": 1}
    with pytest.raises(InvalidCallMetadata, match="agent_extension"):
        build_call_metadata(agent_extension="ext 104")
    with pytest.raises(InvalidCallMetadata, match="agent_display_name"):
        build_call_metadata(agent_display_name="n" * 101)


def test_config_rules(tmp_path):
    with pytest.raises(ConfigError):
        ProcessConfig(store_url="http://store.example.com")  # plain HTTP only to a loopback dev Store
    with pytest.raises(ConfigError):
        ProcessConfig(bind_host="0.0.0.0")
    with pytest.raises(ConfigError):
        SlotSizes(mlx=2)
    assert ProcessConfig(store_url="https://qa.example.com").store_url == "https://qa.example.com"
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"store_url": "http://localhost:8010/", "service_key": "c1sk_x", "installation_id": "inst_1",
                                "slots": {"cpu_io": 2}, "stages": {"contact_signals": False}}))
    config = ProcessConfig.load(path, env={"CALL1_PROCESS_HANDLERS": "fake"})
    assert config.store_url == "http://localhost:8010" and config.dev_store and config.handlers == "fake"
    assert config.slots.cpu_io == 2 and not config.stages.contact_signals and config.configured
    assert "c1sk_x" not in repr(config)


# --- catalog and planner -----------------------------------------------------------------------


def _rubric(extra_criteria=()):
    criteria = [RubricCriterion(criterion_id="REG-01", name="Recording", check=RubricCheck(check_type=CheckType.SEMANTIC_JUDGEMENT)),
                *extra_criteria]
    definition = RubricDefinition(rubric_id="r1", name="R1", criteria=criteria)
    return definition, RubricVersionRef(rubric_id="r1", version=1, digest=canonical_digest(definition))


def _artifact_model(kind="source_audio", aid="art_src"):
    from call1.contracts.artifacts import Artifact

    data = dict(_artifact(), id=aid, kind=kind, content_type="audio/wav" if kind == "source_audio" else "application/json",
                content_contract="audio.v1" if kind == "source_audio" else f"{kind}.v1", sensitivity="raw" if kind == "source_audio" else "derived")
    return Artifact.model_validate(data)


def test_ingest_graph_follows_the_job_type_rules():
    catalog = seeded_catalog(mode="fake")
    planner = GraphPlanner(catalog, ProcessConfig())
    definition, ref = _rubric([RubricCriterion(criterion_id="PH-1", name="Greeting", check=RubricCheck(phrases=["hello"]))])
    stereo = AudioInfo(container="wav", content_type="audio/wav", size_bytes=10, channels=2, duration_seconds=90)
    graph = planner.ingest(conversation_id="conv_1", source=_artifact_model(), audio=stereo, rubric_ref=ref, rubric=definition,
                           snapshot=_artifact_model("rubric_snapshot", "art_snap"))
    by_ref = {j.ref: j for j in graph.jobs}
    assert "speaker" not in by_ref and {"vad", "asr", "tone", "sentiment", "enrichment", "embeddings", "qa-0-reg-01", "qa-det", "scorecard",
                                        "summary", "cs-lifecycle", "cs-resolution", "cs-merge"} == set(by_ref)
    for job in graph.jobs:
        JobDefinition.model_validate(job.model_dump(mode="json"))  # every definition is contract-valid on its own
        assert (job.selection is not None) == (JOB_TYPE_RULES[job.job_type].purpose is not None)
    assert by_ref["asr"].resource_estimate.memory_slot is MemorySlot.LOCAL_MEMORY
    assert by_ref["tone"].resource_estimate.memory_slot is MemorySlot.CPU
    assert by_ref["embeddings"].selection.catalog_entry.entry_id == "nemotron-3-embed-1b"
    assert by_ref["embeddings"].selection.model_revision == embedding.MODEL_REVISION  # a model stage since contract 1.2.0
    assert {i.role for i in by_ref["scorecard"].inputs} == {"rubric", "assessment:REG-01", "verdicts"}
    assert by_ref["cs-merge"].after_refs == ["cs-lifecycle", "cs-resolution"] and all(
        i.optional for i in by_ref["cs-merge"].inputs if i.role.startswith("pass:"))
    assert by_ref["asr"].parameters.extra["summary_plan"] == "summary"
    mono = AudioInfo(container="wav", content_type="audio/wav", size_bytes=10, channels=1, duration_seconds=90)
    graph = planner.ingest(conversation_id="conv_1", source=_artifact_model(), audio=mono, rubric_ref=ref, rubric=definition,
                           snapshot=_artifact_model("rubric_snapshot", "art_snap"))
    by_ref = {j.ref: j for j in graph.jobs}
    assert by_ref["speaker"].requires_refs == ["asr"] and "speaker" in by_ref["tone"].requires_refs


def _masked_graph(config: ProcessConfig, masking=None):
    planner = GraphPlanner(seeded_catalog(mode="fake"), config, masking)
    definition, ref = _rubric()
    stereo = AudioInfo(container="wav", content_type="audio/wav", size_bytes=10, channels=2, duration_seconds=90)
    graph = planner.ingest(conversation_id="conv_1", source=_artifact_model(), audio=stereo, rubric_ref=ref, rubric=definition,
                           snapshot=_artifact_model("rubric_snapshot", "art_snap"))
    return planner, {j.ref: j for j in graph.jobs}


def test_appliance_text_model_prompts_follow_stores_text_masking():
    """Judge condition 1, team decision 3: masked by default (Store's mask_reviewer_reads, the
    legacy redaction.text), for the text-model purposes only; the config can override it."""
    planner, by_ref = _masked_graph(ProcessConfig())
    text_refs = ("qa-0-reg-01", "cs-lifecycle", "cs-resolution")
    assert all(by_ref[r].selection.route.route_class is RouteClass.APPLIANCE and by_ref[r].selection.route.masked for r in text_refs)
    assert not any(by_ref[r].selection.route.masked for r in ("asr", "tone", "sentiment", "embeddings"))
    # no new graph edges: the handlers extract the numeric entities themselves on the appliance
    assert not any(i.role == "enrichment" for r in (*text_refs, "summary") for i in by_ref[r].inputs)
    assert by_ref["summary"].parameters.extra["masked"] is True
    # the ASR completion's summary segments carry the flag too
    jobs = new_jobs("k.sum")
    segments = planner.add_summary_jobs(jobs, plan_segments(_transcript(10), 60), Src.job("job_asr", "transcript"), None,
                                        planner.pick(ModelPurpose.SUMMARY))
    seg = jobs.defs[0]
    assert segments and seg.selection.route.masked and seg.parameters.extra["masked"] is True

    for config, masking in ((ProcessConfig(mask_model_text="off"), None), (ProcessConfig(), MaskingSettings(mask_reviewer_reads=False))):
        _, by_ref = _masked_graph(config, masking)
        assert not any(by_ref[r].selection.route.masked for r in text_refs) and by_ref["summary"].parameters.extra["masked"] is False
        assert not any(i.role == "enrichment" for r in (*text_refs, "summary") for i in by_ref[r].inputs)
    _, by_ref = _masked_graph(ProcessConfig(mask_model_text="on"), MaskingSettings(mask_reviewer_reads=False))
    assert all(by_ref[r].selection.route.masked for r in text_refs)

    assert ProcessConfig.from_mapping({"mask_model_text": False}).mask_model_text == "off"
    assert ProcessConfig.from_mapping({}, env={"CALL1_PROCESS_MASK_MODEL_TEXT": "ON"}).mask_model_text == "on"
    assert ProcessConfig.from_mapping({}).mask_model_text == "store"
    with pytest.raises(ConfigError):
        ProcessConfig(mask_model_text="sometimes")


def test_planner_refuses_models_the_catalog_cannot_serve():
    catalog = seeded_catalog(mode="fake")
    planner = GraphPlanner(catalog, ProcessConfig())
    bad = RubricCriterion(criterion_id="X-1", name="X", check=RubricCheck(check_type=CheckType.SEMANTIC_JUDGEMENT, primary_model_id="nope"))
    definition, ref = _rubric([bad])
    with pytest.raises(PlanError):
        planner.ingest(conversation_id="conv_1", source=_artifact_model(), audio=AudioInfo("wav", "audio/wav", 1, 2, None, 10.0),
                       rubric_ref=ref, rubric=definition, snapshot=_artifact_model("rubric_snapshot", "art_snap"))
    with pytest.raises(CatalogError):
        seeded_catalog(overrides={"asr": "missing-entry"})
    unsupported = seeded_catalog(mode="fake")
    assert not unsupported.usable(unsupported.get("gemma4-12b"), next(iter(unsupported.get("gemma4-12b").purposes)))


def test_real_mode_marks_missing_weights_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("CALL1_MODELS_DIR", str(tmp_path))
    catalog = seeded_catalog(mode="real")
    status = {e["entry_id"]: e["status"] for e in catalog.describe()}
    assert status["nemotron-3-embed-1b"] == "not_installed" and status["parakeet-tdt-0.6b-v3"] == "not_installed"
    registry = build_registry("real", config=ProcessConfig(), catalog=catalog)
    assert registry.mode == "real" and JobType.QA_SCORECARD in registry.job_types()  # code handlers run in every mode
    if registry.get(JobType.ASR) is None:  # no real-handler package in this build: those types are simply not offered
        assert JobType.ASR in registry.missing() and any("Real handlers" in n for n in registry.notes)
    assert not any(h["adapter_id"].startswith("fake.") for h in registry.describe())  # real mode never falls back to fake content


def _transcript(n: int) -> TranscriptContent:
    return TranscriptContent(duration_seconds=n * 2.0, is_redacted=False, turns=[
        TranscriptTurnContent(turn_id=i, speaker=SpeakerRole.AGENT if i % 2 == 0 else SpeakerRole.CALLER, speaker_cluster=f"c{i % 2}",
                              start_time=i * 2.0, end_time=i * 2.0 + 1.5, text=f"turn {i} " + "word " * 10) for i in range(n)])


def test_segments_follow_the_pre_split_chunking():
    windows = plan_segments(_transcript(130), batch_turns=60, max_bytes=1_000_000)
    assert [(w.turn_start, w.turn_end) for w in windows] == [(0, 59), (60, 119), (120, 129)]
    bounded = plan_segments(_transcript(130), batch_turns=60)  # about 70 bytes a line: the 3500-byte bound decides
    assert all(w.turn_end - w.turn_start + 1 <= 50 for w in bounded) and bounded[-1].turn_end == 129
    small = plan_segments(_transcript(0))
    assert [(w.turn_start, w.turn_end) for w in small] == [(0, 0)]
    tiny_bytes = plan_segments(_transcript(10), batch_turns=60, max_bytes=120)
    assert len(tiny_bytes) > 1 and tiny_bytes[0].turn_start == 0 and tiny_bytes[-1].turn_end == 9


def test_speaker_corrections_relabel_a_turn_or_its_cluster():
    transcript = _transcript(4)
    one = apply_speaker_correction(transcript, None, SpeakerCorrection(turn_id=1, speaker="AGENT"))
    assert one.method == "reviewer_correction" and one.corrected_turn_ids == [1]
    cluster = apply_speaker_correction(transcript, None, SpeakerCorrection(turn_id=1, speaker="AGENT", apply_to_cluster=True))
    assert cluster.corrected_turn_ids == [1, 3] and {a.speaker for a in cluster.assignments} == {SpeakerRole.AGENT}


def test_embedding_backend_selection_and_schemes():
    # Explicit CALL1_EMBEDDING_BACKEND wins; else fake handlers mean the fake embedder; else Nemotron.
    assert embedding.configured_backend(env={}) == "nemotron"
    assert embedding.configured_backend(env={"CALL1_PROCESS_HANDLERS": "fake"}) == "fake"
    assert embedding.configured_backend("fake", env={}) == "fake"
    assert embedding.configured_backend("fake", env={"CALL1_EMBEDDING_BACKEND": "nemotron"}) == "nemotron"
    assert embedding.configured_backend("real", env={"CALL1_EMBEDDING_BACKEND": "fake"}) == "fake"
    with pytest.raises(embedding.EmbeddingConfigError):
        embedding.configured_backend(env={"CALL1_EMBEDDING_BACKEND": "hashing"})
    assert embedding.MODEL_SCHEME == "nemotron-3-embed-1b@c0c9fea" and embedding.MODEL_DIMENSIONS == 2048
    assert embedding.FAKE_SCHEME != embedding.MODEL_SCHEME and not {embedding.FAKE_SCHEME, embedding.MODEL_SCHEME} & embedding.LEGACY_SCHEMES
    assert embedding.weights_path({}) == Path("data/models/nemotron-3-embed-1b")
    assert embedding.weights_path({"CALL1_MODELS_DIR": "/m"}) == Path("/m/nemotron-3-embed-1b")
    assert embedding.weights_path({"CALL1_EMBEDDING_PATH": "/w", "CALL1_MODELS_DIR": "/m"}) == Path("/w")


def test_fake_embedder_is_deterministic_unit_length_and_ranks_by_overlap():
    fake = embedding.get_embedder("fake")
    assert fake is embedding.get_embedder("fake") and fake.scheme == "fake-embedding-v1"
    turns = ["This call may be recorded for quality assurance.", "I want to dispute the fee on my account.", "Please verify your date of birth."]
    vectors = fake.embed_documents(turns)
    assert vectors == fake.embed_documents(turns) and all(len(v) == fake.dimensions for v in vectors)
    assert all(abs(sum(x * x for x in v) - 1.0) < 1e-3 for v in vectors)
    query = fake.embed_query("dispute a fee")
    scores = [sum(a * b for a, b in zip(query, v)) for v in vectors]
    assert scores.index(max(scores)) == 1
    assert fake.embed_documents([""])[0][0] == 1.0  # empty text still yields a unit vector


def test_missing_model_weights_are_unavailable_not_a_fallback(tmp_path):
    missing = embedding.NemotronEmbedder(tmp_path / "absent")
    assert missing.status()["state"] == "not_installed" and missing.status()["scheme"] == embedding.MODEL_SCHEME
    with pytest.raises(embedding.EmbedderUnavailable) as err:
        missing.embed_query("anything")
    assert err.value.reason == "not_installed" and not missing.loaded


def test_every_job_type_has_a_fake_or_code_handler():
    registry = build_registry("fake")
    assert registry.missing() == []
    assert registry.get(JobType.EMBEDDINGS).adapter_id == "call1.fake.embeddings"
    assert registry.get(JobType.ASR).adapter_id == "fake.asr"
