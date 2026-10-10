"""Contact Signals v2 in the queue area, through the real queue, results and auth areas (contract 1.3.0;
docs/ContactSignalsV2.md sections 7.5, 8.1 and 14; F2 acceptance "Snapshots and graphs" and "Requests").

Process is played by ``QueueHarness`` over HTTP with a minted service key, Evaluate by minted sessions."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.jobs import JobType
from call1.contracts.signals import taxonomy_digest

from .test_queue_harness import QueueHarness, job, pinned
from .test_signals_harness import BILLING, CANCEL, TAXONOMY, V, hit, put_taxonomy, reason_field, taxonomy_with, v1_content, v2_content


@pytest.fixture
def real(client, store, clock, service_key, mint_service_key, mint_session) -> QueueHarness:
    return QueueHarness(client, store, clock, service_key, mint_service_key, mint_session)


def _mint(real, conversation_id: str, version: int, *, expect: int = 201):
    response = real.post(f"/conversations/{conversation_id}/signal-taxonomy-snapshots", {"version": version}, expect=expect)
    return response.json()


def _settings(client, admin, **settings):
    record = client.get(f"{V}/signals/taxonomy", headers=admin.read_headers).json()
    saved = client.put(f"{V}/signals/settings", json={"settings": {**record["settings"], **settings}, "expected_record_version": record["record_version"]},
                       headers=admin.headers)
    assert saved.status_code == 200, saved.text


_PUBLISHES = itertools.count(1)


def _publish(real, conversation, content) -> Dict[str, Any]:
    graph = real.graph(conversation["id"], [job("merge", JobType.CONTACT_SIGNALS_MERGE, key=f"merge-publish-{next(_PUBLISHES)}")])
    merge = real.claim_one(real.ids(graph)["merge"])
    return real.complete(merge, outputs=real.outputs_for(merge, payloads={"contact_signals": content.model_dump(mode="json")}))


def _claim(real, **body) -> List[Dict[str, Any]]:
    return real.post("/reanalysis-requests/claim", {"worker_id": "w1", "max_requests": 16, **body}).json()["requests"]


def _requests(real, call_id) -> List[Dict[str, Any]]:
    reader = real.mint_session("reviewer")
    return real.get(f"/calls/{call_id}/reanalysis-requests", headers=reader.read_headers).json()["items"]


def _v2_job(ref: str, job_type: JobType, snapshot: Dict[str, Any], digest: str, **params):
    return job(ref, job_type, inputs=[pinned("taxonomy", snapshot)], parameters={"signals": {"taxonomy_digest": digest, **params}})


# --- snapshots ------------------------------------------------------------------------------------


def test_the_snapshot_mint_is_idempotent_per_version_and_settings(real, client, admin_session):
    conversation = real.register()
    first = _mint(real, conversation["id"], 1)
    assert first["kind"] == "signal_taxonomy_snapshot" and first["slot"] == "signals:v1" and first["sensitivity"] == "derived"
    assert first["producing_job_id"] is None and first["linked"] and first["version"] == 1
    assert _mint(real, conversation["id"], 1)["id"] == first["id"]
    content = real.get(f"/artifacts/{first['id']}/content").json()
    assert content["source"] == "published" and content["taxonomy_ref"]["version"] == 1 and content["settings"]["pipeline"] == "v2"
    _settings(client, admin_session, pipeline="shadow")
    changed = _mint(real, conversation["id"], 1)
    assert changed["id"] != first["id"] and changed["slot"] == "signals:v1" and changed["version"] == 2
    assert real.get(f"/artifacts/{changed['id']}/content").json()["settings"]["pipeline"] == "shadow"
    assert _mint(real, conversation["id"], 1)["id"] == changed["id"]
    assert _mint(real, conversation["id"], 7, expect=404)["code"] == "not_found"
    # Process may not upload one itself.
    from .test_queue_harness import descriptor

    body = descriptor(ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, content, slot="signals:v9")
    assert real.post(f"/conversations/{conversation['id']}/artifacts", body, expect=None).status_code == 422


def test_a_redacted_version_cannot_be_minted(real, client, admin_session):
    put_taxonomy(client, admin_session, TAXONOMY)
    put_taxonomy(client, admin_session, taxonomy_with(intent_threshold=0.3))
    v2 = client.get(f"{V}/signals/taxonomy/versions/2", headers=admin_session.read_headers).json()
    assert client.post(f"{V}/signals/taxonomy/versions/2/redaction", json={"digest": v2["digest"], "reason": "quoted a caller"},
                       headers=admin_session.headers).status_code == 200
    refused = _mint(real, real.register()["id"], 2, expect=409)
    assert refused["code"] == "conflict" and refused["details"]["reason"] == "redacted"


def test_graph_creation_checks_the_pinned_snapshot_digest(real):
    conversation = real.register()
    snapshot = _mint(real, conversation["id"], 1)
    digest = real.get(f"/artifacts/{snapshot['id']}/content").json()["taxonomy_ref"]["digest"]
    wrong = real.graph(conversation["id"], [_v2_job("cat", JobType.CONTACT_SIGNALS_CATEGORIZE, snapshot, "sha256:" + "0" * 64)], expect=422).json()
    assert wrong["code"] == "graph_invalid" and wrong["details"]["reason"] == "signal_taxonomy_mismatch"
    rubric_like = real.graph(conversation["id"], [job("cat", JobType.CONTACT_SIGNALS_CATEGORIZE, inputs=[pinned("taxonomy", real.upload_audio(conversation["id"]))],
                                                      parameters={"signals": {"taxonomy_digest": digest}})], expect=422).json()
    assert rubric_like["details"]["reason"] == "signal_taxonomy_input_kind"
    graph = real.graph(conversation["id"], [
        _v2_job("cat", JobType.CONTACT_SIGNALS_CATEGORIZE, snapshot, digest),
        _v2_job("rederive", JobType.CONTACT_SIGNALS_CATEGORIZE, snapshot, digest, stage1_mode="rederive") | {"selection": None},
    ])
    assert {j["job_type"] for j in graph["jobs"]} == {"contact_signals_categorize"}
    assert real.job(real.ids(graph)["rederive"])["selection"] is None


# --- previews --------------------------------------------------------------------------------------


def test_a_preview_runs_into_draft_slots_only_and_is_recorded(real, client, admin_session, reviewer_session, store):
    conversation = real.register()
    call_id = conversation["call_id"]
    unsaved = taxonomy_with()
    body = {"taxonomy": unsaved.model_dump(mode="json"), "call_ids": [call_id]}
    headers = {**admin_session.headers, "Idempotency-Key": "preview-0001"}
    created = client.post(f"{V}/signals/previews", json=body, headers=headers)
    assert created.status_code == 201, created.text
    preview = created.json()
    assert preview["source"] == "preview" and preview["taxonomy_ref"] == {"version": None, "digest": taxonomy_digest(unsaved)}
    [pending] = preview["calls"]
    assert pending["state"] == "pending" and pending["call_id"] == call_id
    assert client.post(f"{V}/signals/previews", json=body, headers=headers).json()["id"] == preview["id"]  # replay
    reused = client.post(f"{V}/signals/previews", json={**body, "taxonomy": None}, headers=headers)
    assert reused.status_code == 409 and reused.json()["code"] == "idempotency_key_reused"
    assert client.post(f"{V}/signals/previews", json=body, headers={**reviewer_session.headers, "Idempotency-Key": "preview-0002"}).status_code == 403

    [claimed] = _claim(real)
    request = claimed["request"]
    assert request["kind"] == "contact_signals_preview" and request["priority"] == 5 and request["signal_pipeline"] == "v2"
    assert request["signal_preview_id"] == preview["id"] and request["signal_taxonomy_version"] == 1
    snapshot = real.get(f"/artifacts/{request['signal_taxonomy_snapshot_artifact_id']}").json()
    assert snapshot["slot"] == f"draft:{request['id']}:signals:preview"
    snap = real.get(f"/artifacts/{snapshot['id']}/content").json()
    assert snap["source"] == "preview" and snap["preview_id"] == preview["id"] and snap["taxonomy_ref"]["version"] is None

    # A published snapshot is not the request's: refused in the preview graph; the preview snapshot is refused elsewhere.
    published = _mint(real, conversation["id"], 1)
    pub_digest = real.get(f"/artifacts/{published['id']}/content").json()["taxonomy_ref"]["digest"]
    outside = real.graph(conversation["id"], [_v2_job("cat", JobType.CONTACT_SIGNALS_CATEGORIZE, snapshot, taxonomy_digest(unsaved))], expect=422).json()
    assert outside["details"]["reason"] == "signal_preview_outside_preview"
    mismatch = real.graph(conversation["id"], [_v2_job("cat", JobType.CONTACT_SIGNALS_CATEGORIZE, published, pub_digest)], reason="reanalysis",
                          request_id=request["id"], claim_token=claimed["claim_token"], expect=422).json()
    assert mismatch["details"]["reason"] == "signal_preview_snapshot_mismatch"

    digest = taxonomy_digest(unsaved)
    graph = real.graph(conversation["id"], [
        _v2_job("cat", JobType.CONTACT_SIGNALS_CATEGORIZE, snapshot, digest, preview_id=preview["id"]),
        job("merge", JobType.CONTACT_SIGNALS_MERGE, after=["cat"], inputs=[pinned("taxonomy", snapshot)],
            parameters={"signals": {"taxonomy_digest": digest, "preview_id": preview["id"]}}),
    ], reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"])
    slot = f"draft:{request['id']}:"
    cat = real.claim_one(real.ids(graph)["cat"])
    real.fail(cat, "model_unavailable")  # stage 1 fails here; the merge still runs by its after edge
    merge = real.claim_one(real.ids(graph)["merge"])
    result = v2_content(unsaved, 1, [hit(unsaved, "intent", 1, sub="cancel_account", quote="I want to cancel", preview=True, fields=[reason_field()])])
    result = result.model_copy(update={"taxonomy": result.taxonomy.model_copy(update={"version": None})})
    live = real.inline(conversation["id"], ArtifactKind.CONTACT_SIGNALS, result.model_dump(mode="json"), slot="", job_id=merge["job"]["id"],
                       token=merge["claim_token"], expect=422).json()
    assert live["details"]["reason"] == "draft_slot_required"
    outputs = real.outputs_for(merge, slots={"contact_signals": slot}, payloads={"contact_signals": result.model_dump(mode="json")})
    receipt = real.complete(merge, outputs=outputs, result=None)
    assert receipt["result_version"] is None

    done = client.get(f"{V}/signals/previews/{preview['id']}", headers=admin_session.read_headers).json()
    [call] = done["calls"]
    assert call["state"] == "available" and call["result"]["signals"][0]["subcategory_id"] == "cancel_account"
    assert call["diff"]["added"] == [result.signals[0].id] and call["diff"]["builtin_changed"] == [result.signals[0].id]
    request_after = real.get(f"/reanalysis-requests/{request['id']}").json()
    assert request_after["preview_result_artifact_id"] == outputs[0]["artifact_id"] and request_after["status"] == "fulfilled"
    # Nothing reached the call's group, projections, queue or metrics.
    assert client.get(f"{V}/calls/{call_id}/contact-signals", headers=reviewer_session.read_headers).status_code == 404
    with store.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM results_signal_hits").fetchone()[0] == 0
        assert conn.execute("SELECT signals_version FROM results_calls WHERE call_id = ?", (call_id,)).fetchone()[0] is None
    audit = client.get(f"{V}/admin/audit", params={"action": "signal_preview_requested"}, headers=admin_session.read_headers).json()["items"]
    assert audit[0]["details"] == {"preview_id": preview["id"], "taxonomy_digest": taxonomy_digest(unsaved), "call_count": 1}


def test_a_preview_checks_text_and_caps_like_a_save(real, client, admin_session):
    call_id = real.register()["call_id"]
    bad = taxonomy_with(intent_subs=[CANCEL.model_copy(update={"examples": ["ssn 123-45-6789"]})])
    refused = client.post(f"{V}/signals/previews", json={"taxonomy": bad.model_dump(mode="json"), "call_ids": [call_id]},
                          headers={**admin_session.headers, "Idempotency-Key": "preview-0003"})
    assert refused.status_code == 422 and refused.json()["details"]["field"] == "categories[0].subcategories[0].examples[0]"
    assert "123-45" not in refused.text
    missing = client.post(f"{V}/signals/previews", json={"call_ids": ["call_missing000000000000"]}, headers={**admin_session.headers, "Idempotency-Key": "preview-0004"})
    assert missing.status_code == 404
    current = client.post(f"{V}/signals/previews", json={"call_ids": [call_id]}, headers={**admin_session.headers, "Idempotency-Key": "preview-0005"})
    assert current.status_code == 201 and current.json()["taxonomy_ref"]["version"] == 1


# --- requests: priority, kinds, widening -------------------------------------------------------------


def test_claims_order_by_priority_then_age_and_kinds_filter(real, client, admin_session, reviewer_session, clock):
    a, b = real.register(), real.register()
    plain = client.post(f"{V}/calls/{a['call_id']}/reanalysis-requests", json={"kind": "summary"},
                        headers={**reviewer_session.headers, "Idempotency-Key": "summary-0001"}).json()
    clock.advance(1)
    signals = client.post(f"{V}/calls/{b['call_id']}/reanalysis-requests", json={"kind": "contact_signals"},
                          headers={**reviewer_session.headers, "Idempotency-Key": "signals-0001"}).json()
    assert signals["signal_taxonomy_version"] == 1 and signals["signal_pipeline"] == "v2" and signals["priority"] == 0
    assert plain["signal_taxonomy_version"] is None and plain["signal_pipeline"] is None
    clock.advance(1)
    preview = client.post(f"{V}/signals/previews", json={"call_ids": [a["call_id"]]}, headers={**admin_session.headers, "Idempotency-Key": "preview-0006"}).json()
    only = _claim(real, kinds=["contact_signals"], max_requests=4)
    assert [c["request"]["id"] for c in only] == [signals["id"]]
    rest = _claim(real)
    assert [c["request"]["kind"] for c in rest] == ["contact_signals_preview", "summary"]
    assert rest[0]["request"]["signal_preview_id"] == preview["id"]


def test_a_new_contact_signals_request_widens_a_pending_one(real, client, admin_session, reviewer_session):
    call_id = real.register()["call_id"]
    first = client.post(f"{V}/calls/{call_id}/reanalysis-requests", json={"kind": "contact_signals"},
                        headers={**reviewer_session.headers, "Idempotency-Key": "signals-0002"}).json()
    put_taxonomy(client, admin_session, TAXONOMY)
    _settings(client, admin_session, pipeline="v2")
    second = client.post(f"{V}/calls/{call_id}/reanalysis-requests", json={"kind": "contact_signals", "rescore_signals": True},
                         headers={**reviewer_session.headers, "Idempotency-Key": "signals-0003"})
    assert second.status_code == 201, second.text
    widened = second.json()
    assert widened["id"] == first["id"] and widened["signal_taxonomy_version"] == 2 and widened["signal_pipeline"] == "v2"
    assert widened["rescore_signals"] is True
    assert len(_requests(real, call_id)) == 1
    _claim(real)
    third = client.post(f"{V}/calls/{call_id}/reanalysis-requests", json={"kind": "contact_signals"},
                        headers={**reviewer_session.headers, "Idempotency-Key": "signals-0004"})
    assert third.status_code == 409 and third.json()["details"]["reason"] == "already_pending"
    refused = client.post(f"{V}/calls/{call_id}/reanalysis-requests", json={"kind": "contact_signals_preview"},
                          headers={**reviewer_session.headers, "Idempotency-Key": "signals-0005"})
    assert refused.status_code == 422


# --- backfills ---------------------------------------------------------------------------------------


def test_a_rescore_backfill_is_digest_driven_unless_rescore_signals(real, client, admin_session, clock):
    put_taxonomy(client, admin_session, TAXONOMY)
    v1_call, current_call, outdated_call, pending_call = (real.register() for _ in range(4))
    _publish(real, v1_call, v1_content())
    _publish(real, current_call, v2_content(TAXONOMY, 2, [hit(TAXONOMY, "intent", 1, sub="cancel_account", fields=[reason_field()])]))
    _publish(real, pending_call, v2_content(TAXONOMY, 2, []))
    edited = taxonomy_with(intent_subs=[CANCEL, BILLING, BILLING.model_copy(update={"subcategory_id": "hours", "name": "Hours", "gloss": "Asks about hours"})])
    old = taxonomy_with()
    _publish(real, outdated_call, v2_content(old, 2, []))
    put_taxonomy(client, admin_session, edited)
    _publish(real, current_call, v2_content(edited, 3, [hit(edited, "intent", 1, sub="cancel_account", fields=[reason_field()])]))
    _publish(real, pending_call, v2_content(edited, 3, []))
    body = {"mode": "rescore", "created_after": "2020-01-01T00:00:00Z"}

    def backfill(key, **extra):
        response = client.post(f"{V}/signals/backfills", json={**body, **extra}, headers={**admin_session.headers, "Idempotency-Key": key})
        assert response.status_code == 201, response.text
        return response.json()

    _settings(client, admin_session, pipeline="v1")
    # Under historical v1, a digest-driven backfill has nothing to rerun.
    none = backfill("backfill-0001")
    assert (none["calls_matched"], none["requests_created"], none["calls_skipped"]) == (4, 0, 4)
    _settings(client, admin_session, pipeline="v2")
    client.post(f"{V}/calls/{pending_call['call_id']}/reanalysis-requests", json={"kind": "contact_signals"},
                headers={**admin_session.headers, "Idempotency-Key": "signals-0006"})
    out = backfill("backfill-0002")
    assert (out["calls_matched"], out["requests_created"], out["calls_skipped"]) == (4, 2, 2)
    assert out["mode"] == "rescore" and out["taxonomy_version"] == 3 and out["preview_id"] is None
    for conv in (v1_call, outdated_call):
        [request] = _requests(real, conv["call_id"])
        assert (request["kind"], request["priority"], request["signal_backfill_id"], request["rescore_signals"]) == ("contact_signals", -10, out["id"], False)
    assert _requests(real, current_call["call_id"]) == []
    [widened] = _requests(real, pending_call["call_id"])
    assert widened["priority"] == 0 and widened["signal_backfill_id"] is None  # widened, not duplicated
    assert client.post(f"{V}/signals/backfills", json={**body, "max_calls": 3}, headers={**admin_session.headers, "Idempotency-Key": "backfill-0002"}).status_code == 409
    assert backfill("backfill-0002")["id"] == out["id"]  # replay
    _claim(real)
    everything = backfill("backfill-0003", rescore_signals=True)
    assert everything["requests_created"] == 1  # current_call; the others have claimed requests underway
    [request] = _requests(real, current_call["call_id"])
    assert request["rescore_signals"] is True
    audit = client.get(f"{V}/admin/audit", params={"action": "signal_backfill_requested"}, headers=admin_session.read_headers).json()["items"]
    assert audit[0]["details"]["backfill_id"] == everything["id"] and audit[0]["details"]["mode"] == "rescore"


def test_a_compare_backfill_runs_v2_beside_v1_into_one_preview(real, client, admin_session):
    v1_call, v2_call = real.register(), real.register()
    _publish(real, v1_call, v1_content())
    _publish(real, v2_call, v2_content(TAXONOMY, 1, []))
    out = client.post(f"{V}/signals/backfills", json={"mode": "compare", "created_after": "2020-01-01T00:00:00Z"},
                      headers={**admin_session.headers, "Idempotency-Key": "compare-0001"}).json()
    assert (out["mode"], out["requests_created"], out["calls_skipped"]) == ("compare", 1, 1) and out["preview_id"]
    [request] = _requests(real, v1_call["call_id"])
    assert (request["kind"], request["priority"], request["signal_preview_id"], request["signal_pipeline"]) == (
        "contact_signals_preview", -10, out["preview_id"], "v2")
    snapshot = real.get(f"/artifacts/{request['signal_taxonomy_snapshot_artifact_id']}").json()
    assert snapshot["slot"] == "signals:v1"
    preview = client.get(f"{V}/signals/previews/{out['preview_id']}", headers=admin_session.read_headers).json()
    assert preview["source"] == "compare" and [c["call_id"] for c in preview["calls"]] == [v1_call["call_id"]]
    capped = client.post(f"{V}/signals/backfills", json={"mode": "compare", "created_after": "2020-01-01T00:00:00Z", "max_calls": 500},
                         headers={**admin_session.headers, "Idempotency-Key": "compare-0002"})
    assert capped.status_code == 201


# --- shadow mode --------------------------------------------------------------------------------------


def test_shadow_mode_creates_one_compare_companion_per_v1_publish_from_a_non_draft_graph(real, client, admin_session, store):
    _settings(client, admin_session, pipeline="shadow")
    conversation = real.register()
    call_id = conversation["call_id"]
    _publish(real, conversation, v1_content())
    [companion] = _requests(real, call_id)
    assert companion["kind"] == "contact_signals_preview" and companion["priority"] == -10 and companion["requested_by_account_id"] is None
    preview = client.get(f"{V}/signals/previews/{companion['signal_preview_id']}", headers=admin_session.read_headers).json()
    assert preview["source"] == "compare" and preview["created_by_account_id"] is None
    _publish(real, conversation, v1_content())
    assert len(_requests(real, call_id)) == 2  # one per publish
    _publish(real, conversation, v2_content(TAXONOMY, 1, []))
    assert len(_requests(real, call_id)) == 2  # a v2 publish gets none

    # The companion's own (draft) graph publishes nothing and gets no companion.
    claimed = next(c for c in _claim(real) if c["request"]["id"] == companion["id"])
    snapshot = real.get(f"/artifacts/{companion['signal_taxonomy_snapshot_artifact_id']}").json()
    digest = real.get(f"/artifacts/{snapshot['id']}/content").json()["taxonomy_ref"]["digest"]
    graph = real.graph(conversation["id"], [job("merge", JobType.CONTACT_SIGNALS_MERGE, inputs=[pinned("taxonomy", snapshot)],
                                                parameters={"signals": {"taxonomy_digest": digest}})],
                       reason="reanalysis", request_id=companion["id"], claim_token=claimed["claim_token"])
    merge = real.claim_one(real.ids(graph)["merge"])
    result = v2_content(TAXONOMY, 1, [hit(TAXONOMY, "intent", 1, preview=True)])
    real.complete(merge, outputs=real.outputs_for(merge, slots={"contact_signals": f"draft:{companion['id']}:"},
                                                  payloads={"contact_signals": result.model_dump(mode="json")}), result=None)
    assert len(_requests(real, call_id)) == 2
    view = client.get(f"{V}/calls/{call_id}/contact-signals", headers=admin_session.read_headers).json()
    assert view["comparison_preview_id"] == companion["signal_preview_id"] and view["pipeline"] == "v2"
    done = client.get(f"{V}/signals/previews/{companion['signal_preview_id']}", headers=admin_session.read_headers).json()
    assert done["calls"][0]["state"] == "available" and done["calls"][0]["diff"] is not None


def test_a_failed_preview_merge_reads_failed(real, client, admin_session):
    conversation = real.register()
    preview = client.post(f"{V}/signals/previews", json={"call_ids": [conversation["call_id"]]},
                          headers={**admin_session.headers, "Idempotency-Key": "preview-0009"}).json()
    [claimed] = _claim(real)
    request = claimed["request"]
    snapshot = real.get(f"/artifacts/{request['signal_taxonomy_snapshot_artifact_id']}").json()
    digest = real.get(f"/artifacts/{snapshot['id']}/content").json()["taxonomy_ref"]["digest"]
    graph = real.graph(conversation["id"], [job("merge", JobType.CONTACT_SIGNALS_MERGE, inputs=[pinned("taxonomy", snapshot)], max_attempts=1,
                                                parameters={"signals": {"taxonomy_digest": digest}})],
                       reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"])
    real.fail(real.claim_one(real.ids(graph)["merge"]), "input_unavailable")
    [call] = client.get(f"{V}/signals/previews/{preview['id']}", headers=admin_session.read_headers).json()["calls"]
    assert call["state"] == "failed" and call["failure_code"] == "input_unavailable"
