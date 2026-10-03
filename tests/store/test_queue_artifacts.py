"""Conversations and artifacts: natural idempotency, canonical content, uploads, grants, slots."""

from __future__ import annotations

from fastapi.testclient import TestClient

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.common import ContractParameters, ServiceScope, canonical_json
from call1.store.app import create_app
from call1.store.config import StoreConfig
from call1.store.objects import sha256_checksum

from .test_queue_harness import AUDIO, RUBRIC_ID, V, content, descriptor, hooks, job, path, pinned, q, rubrics  # noqa: F401


def test_registration_is_idempotent_by_source_identity(q, hooks):
    first = q.register(etag="abc")
    again = q.post("/conversations", {"ingestion_kind": "call_audio", "call_metadata": {"agent_id": "agent-7"},
                                      "source": {"kind": "s3_event", "bucket": "rec", "object_key": "calls/1.wav", "etag": "abc",
                                                 "received_at": "2026-09-25T12:30:00Z"}}).json()
    assert again["created"] is False and again["conversation"]["id"] == first["id"] and again["conversation"]["call_id"] == first["call_id"]
    other = q.register(etag="def")
    assert other["id"] != first["id"] and other["call_id"] != first["call_id"]
    assert first["registered_by_installation_id"] == q.installation_id
    assert [c.id for c in hooks.registered] == [first["id"], other["id"]]  # the hook runs for created conversations only

    text = q.post("/conversations", {"ingestion_kind": "text_import", "source": {"kind": "local_import", "content_digest": sha256_checksum(b"t"),
                                                                                "received_at": "2026-09-25T12:00:00Z"}}).json()
    assert text["created"] is True and text["conversation"]["call_id"] is None
    assert q.get(f"/conversations/{first['id']}").json() == first


def test_conversation_reads_need_the_right_principal(q, mint_session):
    conversation = q.register()
    supervisor, reviewer = mint_session("supervisor"), mint_session("reviewer")
    assert q.get(f"/conversations/{conversation['id']}", headers=supervisor.read_headers).json()["id"] == conversation["id"]
    assert q.get(f"/conversations/{conversation['id']}", headers=reviewer.read_headers, expect=403).json()["code"] == "insufficient_role"
    assert q.get("/conversations/conv_missing", expect=404).json()["code"] == "not_found"
    reader = q.mint_service_key([ServiceScope.ARTIFACTS_READ])
    assert q.post("/conversations", {}, headers=reader.headers, expect=403).json()["code"] == "insufficient_scope"


def test_inline_artifacts_store_canonical_bytes_and_link_with_versions(q):
    conversation = q.register()
    payload = content(ArtifactKind.TRANSCRIPT)
    first = q.inline(conversation["id"], ArtifactKind.TRANSCRIPT, payload).json()
    assert first["linked"] is True and first["version"] == 1 and first["storage"] == "inline" and first["producing_job_id"] is None
    assert first["checksum"] == sha256_checksum(canonical_json(payload))
    again = q.inline(conversation["id"], ArtifactKind.TRANSCRIPT, payload).json()
    assert again["id"] == first["id"]  # natural idempotency: (conversation, kind, slot, checksum)

    changed = dict(payload, duration_seconds=99.0)
    second = q.inline(conversation["id"], ArtifactKind.TRANSCRIPT, changed).json()
    assert second["version"] == 2 and second["id"] != first["id"]
    assert q.get(f"/artifacts/{first['id']}").json()["superseded_by"] == second["id"]
    content_response = q.get(f"/artifacts/{second['id']}/content")
    assert content_response.content == canonical_json(changed)
    listed = q.get(f"/conversations/{conversation['id']}/artifacts").json()["items"]
    assert [a["id"] for a in listed] == [second["id"]]
    everything = q.get(f"/conversations/{conversation['id']}/artifacts", params={"include_superseded": "true"}).json()["items"]
    assert [a["id"] for a in everything] == [first["id"], second["id"]]


def test_inline_artifacts_reject_wrong_checksums_non_canonical_payloads_and_reserved_kinds(q):
    conversation = q.register()
    body = descriptor(ArtifactKind.TRANSCRIPT, content(ArtifactKind.TRANSCRIPT))
    wrong = dict(body, checksum="sha256:" + "0" * 64)
    assert q.post(f"/conversations/{conversation['id']}/artifacts", wrong, expect=422).json()["code"] == "checksum_mismatch"
    wrong_size = dict(body, size_bytes=body["size_bytes"] + 1)
    assert q.post(f"/conversations/{conversation['id']}/artifacts", wrong_size, expect=422).json()["details"]["reason"] == "size_differs"
    partial = content(ArtifactKind.TRANSCRIPT)
    del partial["language"]  # not the full dump: Store never fills defaults
    bad = descriptor(ArtifactKind.TRANSCRIPT, content(ArtifactKind.TRANSCRIPT))
    bad["payload"] = partial
    assert q.post(f"/conversations/{conversation['id']}/artifacts", bad, expect=422).json()["code"] == "validation_failed"
    snapshot = descriptor(ArtifactKind.TRANSCRIPT, content(ArtifactKind.TRANSCRIPT))
    snapshot.update(kind="rubric_snapshot", content_contract="rubric_snapshot.v1", sensitivity="derived")
    assert q.post(f"/conversations/{conversation['id']}/artifacts", snapshot, expect=422).json()["code"] == "validation_failed"
    draft_slot = q.inline(conversation["id"], ArtifactKind.TRANSCRIPT, slot="draft:rq_x:", expect=422).json()
    assert draft_slot["details"]["reason"] == "draft_slot_reserved"
    assert q.inline("conv_missing", ArtifactKind.TRANSCRIPT, expect=404).json()["code"] == "not_found"


def test_inline_artifacts_have_a_size_limit(tmp_path, clock):
    config = StoreConfig.for_tests(tmp_path / "small", parameters=ContractParameters(inline_artifact_max_bytes=1024))
    app = create_app(config, clock=clock)
    store = app.state.store
    with store.connection() as conn:
        key = store.auth.mint_service_key_for_tests(conn, scopes=list(ServiceScope), primary_host=True)
    client = TestClient(app, base_url="http://localhost:8010")
    conversation = client.post(V + "/conversations", headers=key.headers, json={
        "ingestion_kind": "text_import", "source": {"kind": "local_import", "content_digest": sha256_checksum(b"x"), "received_at": "2026-09-25T12:00:00Z"}}).json()
    payload = content(ArtifactKind.TRANSCRIPT)
    payload["turns"] = [dict(payload["turns"][0], turn_id=i, text="word " * 40) for i in range(10)]
    response = client.post(V + f"/conversations/{conversation['conversation']['id']}/artifacts", headers=key.headers,
                           json=descriptor(ArtifactKind.TRANSCRIPT, payload))
    assert response.status_code == 413 and response.json()["code"] == "payload_too_large"


def test_upload_grant_put_commit_and_download(q):
    conversation = q.register()
    audio = q.upload_audio(conversation["id"])
    assert audio["kind"] == "source_audio" and audio["storage"] == "object" and audio["linked"] and audio["version"] == 1
    assert audio["checksum"] == sha256_checksum(AUDIO) and audio["size_bytes"] == len(AUDIO)
    # a second grant for the same source reserves the same artifact, and its commit is a replay
    body = {"kind": "source_audio", "slot": "", "content_type": "audio/wav", "size_bytes": len(AUDIO), "checksum": sha256_checksum(AUDIO),
            "content_contract": "audio.v1", "sensitivity": "raw"}
    grant = q.post(f"/conversations/{conversation['id']}/artifacts/uploads", body, expect=201).json()
    assert grant["artifact_id"] == audio["id"] and grant["url"].startswith("http://localhost:8010/store/transfer/uploads/")
    committed = q.post(f"/artifact-uploads/{grant['upload_id']}/commit", {"checksum": body["checksum"], "size_bytes": len(AUDIO)}, expect=201).json()
    assert committed["id"] == audio["id"]

    grant_body = q.get(f"/artifacts/{audio['id']}/content-grant").json()
    assert grant_body["checksum"] == audio["checksum"] and grant_body["size_bytes"] == len(AUDIO)
    downloaded = q.client.get(path(grant_body["url"]))
    assert downloaded.status_code == 200 and downloaded.content == AUDIO
    direct = q.get(f"/artifacts/{audio['id']}/content", headers={**q.headers, "Range": "bytes=0-3"}, expect=206)
    assert direct.content == AUDIO[:4]


def test_json_uploads_must_be_canonical_and_commits_need_the_granting_scope(q):
    conversation = q.register()
    data = b'{"turns": []}'  # not the canonical enrichment document
    body = {"kind": "enrichment", "slot": "", "content_type": "application/json", "size_bytes": len(data), "checksum": sha256_checksum(data),
            "content_contract": "enrichment.v1", "sensitivity": "raw"}
    grant = q.post(f"/conversations/{conversation['id']}/artifacts/uploads", body, expect=201).json()
    assert q.client.put(path(grant["url"]), content=data).status_code == 200
    evidence_key = q.mint_service_key([ServiceScope.KEY_RELEASE_WRITE], installation_id=q.installation_id)
    refused = q.post(f"/artifact-uploads/{grant['upload_id']}/commit", {"checksum": body["checksum"], "size_bytes": len(data)},
                     headers=evidence_key.headers, expect=403).json()
    assert refused["code"] == "insufficient_scope" and refused["details"]["required_scope"] == "artifacts:write"
    rejected = q.post(f"/artifact-uploads/{grant['upload_id']}/commit", {"checksum": body["checksum"], "size_bytes": len(data)}, expect=422).json()
    assert rejected["code"] == "validation_failed" and rejected["details"]["reason"] == "not_canonical_content"
    assert q.get(f"/conversations/{conversation['id']}/artifacts").json()["items"] == []
    assert q.post("/artifact-uploads/upl_missing/commit", {"checksum": body["checksum"], "size_bytes": 1}, expect=404).json()["code"] == "not_found"


def test_job_outputs_need_the_active_claim_and_stay_unlinked_until_completion(q):
    setup = q.ingest()
    conversation_id, asr = setup["conversation_id"], setup["ids"]["asr"]
    claimed = q.claim_one(asr)
    stale = q.inline(conversation_id, ArtifactKind.TRANSCRIPT, job_id=asr, token="x" * 43, expect=409).json()
    assert stale["code"] == "claim_token_stale"
    wrong_kind = q.inline(conversation_id, ArtifactKind.SUMMARY_SEGMENT, content(ArtifactKind.ENRICHMENT), job_id=asr, token=claimed["claim_token"], expect=422)
    assert wrong_kind.json()["code"] == "validation_failed"
    output = q.inline(conversation_id, ArtifactKind.TRANSCRIPT, job_id=asr, token=claimed["claim_token"]).json()
    assert output["linked"] is False and output["version"] is None and output["producing_job_id"] == asr
    again = q.inline(conversation_id, ArtifactKind.TRANSCRIPT, job_id=asr, token=claimed["claim_token"]).json()
    assert again["id"] == output["id"]  # (job, attempt, kind, slot, checksum)
    assert all(a["id"] != output["id"] for a in q.get(f"/conversations/{conversation_id}/artifacts").json()["items"])
    unlinked = q.get(f"/conversations/{conversation_id}/artifacts", params={"include_unlinked": "true", "kind": "transcript"}).json()["items"]
    assert [a["id"] for a in unlinked] == [output["id"]]
    receipt = q.complete(claimed, outputs=[{"role": "transcript", "artifact_id": output["id"], "checksum": output["checksum"]}])
    linked = q.get(f"/artifacts/{output['id']}").json()
    assert linked["linked"] and linked["version"] == 1 and linked["linked_by_receipt_id"] == receipt["receipt_id"]
    # the claim ended, so a replay of the same output still returns it, and a new output is refused
    assert q.inline(conversation_id, ArtifactKind.TRANSCRIPT, job_id=asr, token=claimed["claim_token"]).json()["id"] == output["id"]
    fresh = content(ArtifactKind.TRANSCRIPT, duration_seconds=1.0)
    assert q.inline(conversation_id, ArtifactKind.TRANSCRIPT, fresh, job_id=asr, token=claimed["claim_token"], expect=409).json()["code"] == "claim_token_stale"


def test_rubric_snapshots_are_minted_once_per_conversation_and_version(q):
    conversation = q.register()
    snapshot = q.rubric_snapshot(conversation["id"])
    assert snapshot["kind"] == "rubric_snapshot" and snapshot["slot"] == f"rubric:{RUBRIC_ID}:v1" and snapshot["linked"] and snapshot["version"] == 1
    assert snapshot["labels"] == {"rubric_id": RUBRIC_ID, "rubric_version": "1"} and snapshot["sensitivity"] == "derived"
    assert q.rubric_snapshot(conversation["id"])["id"] == snapshot["id"]
    missing = q.post(f"/conversations/{conversation['id']}/rubric-snapshots", {"rubric_id": RUBRIC_ID, "version": 9}, expect=404).json()
    assert missing["code"] == "not_found"
    body = q.get(f"/artifacts/{snapshot['id']}/content").json()
    assert body["source"] == "published" and body["rubric_version"] == 1


def test_artifact_listing_pages_and_filters(q):
    conversation = q.register()
    ids = []
    for index in range(5):
        ids.append(q.inline(conversation["id"], ArtifactKind.PROMPT_INPUT, content(ArtifactKind.PROMPT_INPUT, template_version=str(index)),
                            slot=f"segment:{index}").json()["id"])
        q.clock.advance(1)
    q.inline(conversation["id"], ArtifactKind.TRANSCRIPT)
    page = q.get(f"/conversations/{conversation['id']}/artifacts", params={"kind": "prompt_input", "limit": 2}).json()
    seen = [a["id"] for a in page["items"]]
    while page["next_page_token"]:
        page = q.get(f"/conversations/{conversation['id']}/artifacts", params={"kind": "prompt_input", "limit": 2, "page_token": page["next_page_token"]}).json()
        seen += [a["id"] for a in page["items"]]
    assert seen == ids
    one = q.get(f"/conversations/{conversation['id']}/artifacts", params={"slot": "segment:3"}).json()["items"]
    assert [a["id"] for a in one] == [ids[3]]
    assert q.get(f"/conversations/{conversation['id']}/artifacts", params={"page_token": "!!"}, expect=422).json()["code"] == "validation_failed"


def test_supervisors_read_artifact_metadata_but_not_bytes(q, mint_session):
    conversation = q.register()
    art = q.inline(conversation["id"], ArtifactKind.TRANSCRIPT).json()
    supervisor = mint_session("supervisor")
    assert q.get(f"/artifacts/{art['id']}", headers=supervisor.read_headers).json()["id"] == art["id"]
    assert q.get(f"/artifacts/{art['id']}/content", headers=supervisor.read_headers, expect=403).json()["code"] == "forbidden"
