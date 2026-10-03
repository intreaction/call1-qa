"""The processing queue and its data API through the real Store server (inventory q3-q7).

q3 and q4 drive a Store-only private stack (no Process worker racing for the jobs) with a
hand-rolled worker: the Store tests' ``QueueHarness`` payload builders over a real HTTP client and
the stack's real service key (``store_api_support.worker_harness``). q5-q7 use the shared stack,
where the real Process server plans reanalysis graphs and publishes its catalog and hardware.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from call1.contracts.jobs import REANALYSIS_KIND_AFFECTS, JOB_TYPE_RULES, JobType, ReanalysisKind

from .store_api_support import error_code, graph_job_types, poll, queue_harness_module, settled_call, worker_harness

pytestmark = pytest.mark.e2e


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _grant_path(url: str) -> str:
    return queue_harness_module().path(url)


@pytest.fixture
def store_only(stack_factory):
    return stack_factory(name="queue", with_process=False)


# --- q3: conversations, uploads, grants and checksums -------------------------------------------


def test_q3_conversation_registration_upload_and_content_grants(store_only):
    q = worker_harness(store_only)
    http = q.http

    # Registration is idempotent on the source identity.
    first = q.post("/conversations", {"ingestion_kind": "call_audio", "call_metadata": {"agent_id": "agent-q3"},
                                       "source": {"kind": "s3_event", "bucket": "rec", "object_key": "q3.wav", "etag": "etag-q3",
                                                  "received_at": "2026-09-25T11:59:00Z"}}).json()
    assert first["created"] is True and first["conversation"]["call_id"], first
    again = q.post("/conversations", {"ingestion_kind": "call_audio", "call_metadata": {"agent_id": "agent-q3"},
                                       "source": {"kind": "s3_event", "bucket": "rec", "object_key": "q3.wav", "etag": "etag-q3",
                                                  "received_at": "2026-09-25T11:59:00Z"}}).json()
    assert again["created"] is False and again["conversation"]["id"] == first["conversation"]["id"], again
    conversation = first["conversation"]["id"]

    # Upload grant -> PUT the bytes -> commit: the artifact carries the SHA-256 of what was sent.
    data = b"RIFF" + bytes(range(256)) * 64
    descriptor = {"kind": "source_audio", "slot": "", "content_type": "audio/wav", "size_bytes": len(data), "checksum": _sha(data),
                  "content_contract": "audio.v1", "sensitivity": "raw"}
    grant = q.post(f"/conversations/{conversation}/artifacts/uploads", descriptor, expect=201).json()
    assert grant["method"] == "PUT" and grant["max_bytes"] >= len(data), grant
    put = http.put(_grant_path(grant["url"]), content=data, headers={"Content-Type": "audio/wav"})
    assert put.status_code == 200 and put.json()["checksum"] == _sha(data), put.text
    artifact = q.post(f"/artifact-uploads/{grant['upload_id']}/commit", {"checksum": _sha(data), "size_bytes": len(data)}, expect=201).json()
    assert artifact["id"] == grant["artifact_id"] and artifact["checksum"] == _sha(data) and artifact["size_bytes"] == len(data), artifact

    # Reading back: a content grant URL and the direct content route both return the same bytes.
    content_grant = q.get(f"/artifacts/{artifact['id']}/content-grant").json()
    assert content_grant["checksum"] == _sha(data) and content_grant["size_bytes"] == len(data), content_grant
    downloaded = http.get(_grant_path(content_grant["url"]))
    assert downloaded.status_code == 200 and downloaded.content == data, (downloaded.status_code, len(downloaded.content))
    direct = http.get(f"/store/v1/artifacts/{artifact['id']}/content", headers=q.headers)
    assert direct.status_code == 200 and direct.content == data
    assert http.get(_grant_path(content_grant["url"]).replace("downloads/", "downloads/x")).status_code >= 400, "a forged download token is refused"

    # A second grant for the same content in the same conversation is the same artifact (natural
    # idempotency), so the negative cases below declare content Store has never seen.
    same = q.post(f"/conversations/{conversation}/artifacts/uploads", descriptor, expect=201).json()
    assert same["artifact_id"] == artifact["id"], (same, artifact["id"])

    # Bytes that differ from the grant's checksum are refused at PUT, and the grant cannot be committed.
    declared = b"RIFF" + bytes(reversed(range(256))) * 64
    other = b"RIFF" + bytes((i * 7) % 256 for i in range(256)) * 64
    grant2 = q.post(f"/conversations/{conversation}/artifacts/uploads", {**descriptor, "checksum": _sha(declared)}, expect=201).json()
    wrong = http.put(_grant_path(grant2["url"]), content=other[:len(declared)], headers={"Content-Type": "audio/wav"})
    assert wrong.status_code == 422 and error_code(wrong) == "checksum_mismatch", (wrong.status_code, wrong.text)
    not_committed = q.post(f"/artifact-uploads/{grant2['upload_id']}/commit", {"checksum": _sha(declared), "size_bytes": len(declared)}, expect=None)
    assert not_committed.status_code == 422 and error_code(not_committed) == "checksum_mismatch", (not_committed.status_code, not_committed.text)

    # More bytes than the grant allows: 413 payload_too_large.
    grant3 = q.post(f"/conversations/{conversation}/artifacts/uploads", {**descriptor, "checksum": _sha(other)}, expect=201).json()
    oversize = http.put(_grant_path(grant3["url"]), content=other + b"extra-bytes", headers={"Content-Type": "audio/wav"})
    assert oversize.status_code == 413 and error_code(oversize) == "payload_too_large", (oversize.status_code, oversize.text)

    # A commit whose checksum disagrees with the grant is refused.
    grant4 = q.post(f"/conversations/{conversation}/artifacts/uploads", {**descriptor, "checksum": _sha(other)}, expect=201).json()
    assert http.put(_grant_path(grant4["url"]), content=other, headers={"Content-Type": "audio/wav"}).status_code == 200
    mismatch = q.post(f"/artifact-uploads/{grant4['upload_id']}/commit", {"checksum": _sha(declared), "size_bytes": len(other)}, expect=None)
    assert mismatch.status_code == 422 and error_code(mismatch) == "checksum_mismatch", (mismatch.status_code, mismatch.text)

    # An inline artifact whose declared checksum is wrong is refused.
    module = queue_harness_module()
    from call1.contracts.artifacts import ArtifactKind

    body = module.descriptor(ArtifactKind.VAD_METRICS, module.content(ArtifactKind.VAD_METRICS))
    body["checksum"] = _sha(b"not the payload")
    bad_inline = q.post(f"/conversations/{conversation}/artifacts", body, expect=None)
    assert bad_inline.status_code == 422 and error_code(bad_inline) == "checksum_mismatch", (bad_inline.status_code, bad_inline.text)

    listing = q.get(f"/conversations/{conversation}/artifacts").json()["items"]
    assert [a["id"] for a in listing if a["kind"] == "source_audio"].count(artifact["id"]) == 1
    assert grant2["artifact_id"] not in {a["id"] for a in listing}, "a refused upload must not be listed as an artifact"


def test_q3_inline_artifact_over_the_limit_is_refused(stack_factory):
    private = stack_factory(name="inline", with_process=False, store_parameters={"inline_artifact_max_bytes": 1024})
    q = worker_harness(private)
    conversation = q.register()["id"]
    module = queue_harness_module()
    from call1.contracts.artifacts import ArtifactKind

    payload = module.content(ArtifactKind.TRANSCRIPT)
    payload["turns"] = [dict(payload["turns"][0], turn_id=i, start_time=float(i), end_time=float(i) + 0.5) for i in range(40)]
    body = module.descriptor(ArtifactKind.TRANSCRIPT, payload)
    assert body["size_bytes"] > 1024
    response = q.post(f"/conversations/{conversation}/artifacts", body, expect=None)
    assert response.status_code == 413 and error_code(response) == "payload_too_large", (response.status_code, response.text)


# --- q4: the job queue --------------------------------------------------------------------------


def test_q4_racing_workers_never_share_a_claim(store_only):
    q = worker_harness(store_only)
    for _ in range(6):
        q.ingest()  # 12 ready jobs
    barrier = threading.Barrier(12)

    def claim(worker: int):
        client = httpx.Client(base_url=store_only.store_url, timeout=30.0)
        body = {"worker": q.worker(worker_id=f"racer-{worker}"), "max_jobs": 2}
        barrier.wait()
        response = client.post("/store/v1/jobs/claim", json=body, headers=q.headers)
        assert response.status_code == 200, response.text
        return [(c["job"]["id"], c["claim_token"]) for c in response.json()["jobs"]]

    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        results = list(pool.map(claim, range(12)))
    claimed = [job for batch in results for job in batch]
    ids = [job_id for job_id, _ in claimed]
    assert len(ids) == len(set(ids)), f"a job was claimed twice: {sorted(i for i in ids if ids.count(i) > 1)}"
    assert len(ids) == 12, f"12 ready jobs, {len(ids)} claimed"


def test_q4_claim_heartbeat_complete_commits_everything_together(store_only):
    q = worker_harness(store_only)
    admin = store_only.admin()
    run = q.ingest()
    ids = run["ids"]
    conversation = run["conversation_id"]
    call_id = q.get(f"/conversations/{conversation}").json()["call_id"]

    claimed = q.claim_one(ids["asr"])
    assert claimed["job"]["status"] == "RUNNING" and claimed["claim_token"], claimed

    # Heartbeat renews the lease; a stale or foreign claim token is rejected.
    beat = q.heartbeat(claimed)
    assert beat["cancel_requested"] is False and beat["lease_expires_at"], beat
    stale = q.post(f"/jobs/{ids['asr']}/heartbeat", {"claim_token": "clm_" + "0" * 40}, expect=None)
    assert stale.status_code == 409 and error_code(stale) == "claim_token_stale", (stale.status_code, stale.text)

    before = store_only.store_get("/changes", session="service").json()["latest_cursor"]
    receipt = q.complete(claimed)
    assert receipt["status"] == "SUCCEEDED" and receipt["replayed"] is False, receipt

    # Outputs linked, usage row written, result projected, dependents released and a change event
    # appended: all visible together from the receipt.
    assert len(receipt["linked_artifact_ids"]) == 1
    art = q.get(f"/artifacts/{receipt['linked_artifact_ids'][0]}").json()
    assert art["linked"] is True and art["kind"] == "transcript" and art["linked_by_receipt_id"] == receipt["receipt_id"], art
    usage = q.get(f"/conversations/{conversation}/usage").json()["items"]
    assert receipt["usage_record_id"] in {u["id"] for u in usage}, usage
    assert set(receipt["released_job_ids"]) == {ids["enrich"], ids["sentiment"]}, receipt
    for job_id in receipt["released_job_ids"]:
        assert q.status(job_id) == "QUEUED"
    assert receipt["result_version"] == 1
    detail = admin.get(f"/calls/{call_id}")
    assert detail.status_code == 200, detail.text
    transcript_group = next(g for g in detail.json()["results"] if g["kind"] == "transcript")
    # Contract 1.2.0: published, but its PII findings (the enrichment job, not run yet) do not exist,
    # so the group reads 'partial' and the text is withheld.
    assert transcript_group["state"] == "partial" and transcript_group["version"] == 1, transcript_group
    assert "PII masking" in transcript_group["partial_reason"], transcript_group
    feed = store_only.store_get("/changes", session="service", params={"after": before, "limit": 200}).json()["events"]
    assert any(e["kind"] == "job" and e["resource_id"] == ids["asr"] and e["status"] == "SUCCEEDED" for e in feed), feed
    cursors = [e["cursor"] for e in feed]
    assert receipt["change_cursor"] in cursors, (receipt["change_cursor"], cursors)

    # A replay of the same completion is idempotent; the old claim token is dead afterwards.
    replay = q.complete(claimed, outputs=[{"role": "transcript", "artifact_id": art["id"], "checksum": art["checksum"]}])
    assert replay["replayed"] is True and replay["receipt_id"] == receipt["receipt_id"], replay
    dead = q.post(f"/jobs/{ids['asr']}/heartbeat", {"claim_token": claimed["claim_token"]}, expect=None)
    assert dead.status_code == 409, (dead.status_code, dead.text)

    # Progress reflects the queue: transcript published, the rest still to run.
    progress = q.get(f"/conversations/{conversation}/progress").json()
    groups = {g["kind"]: g for g in progress["groups"]}
    assert groups["transcript"]["state"] == "partial" and progress["settled"] is False, progress


def test_q4_fail_release_retry_and_cancel(store_only):
    q = worker_harness(store_only)
    module = queue_harness_module()
    conversation = q.register()["id"]
    audio = q.upload_audio(conversation)
    graph = q.graph(conversation, [
        module.job("vad", JobType.VALIDATION_VAD, inputs=[module.pinned("audio", audio)], max_attempts=1),
        module.job("asr", JobType.ASR, inputs=[module.pinned("audio", audio)], priority=5),
        module.job("enrich", JobType.ENRICHMENT, requires=["asr"], inputs=[module.upstream_input("transcript", "asr", "transcript")]),
        module.job("sentiment", JobType.TEXT_SENTIMENT, requires=["asr"], inputs=[module.upstream_input("transcript", "asr", "transcript")]),
    ])
    ids = q.ids(graph)

    # Release (requeue) puts the job back; the released claim is then stale.
    claimed = q.claim_one(ids["asr"])
    released = q.release(claimed)
    assert q.status(ids["asr"]) == "QUEUED", released
    stale = q.post(f"/jobs/{ids['asr']}/heartbeat", {"claim_token": claimed["claim_token"]}, expect=None)
    assert stale.status_code == 409 and error_code(stale) in ("claim_token_stale", "job_not_claimable"), stale.text

    # A terminal failure on the last attempt fails the job; retry (jobs:control) requeues it.
    vad = q.claim_one(ids["vad"])
    failed = q.fail(vad, "provider_error")
    assert q.status(ids["vad"]) == "FAILED", failed
    attempts = q.get(f"/jobs/{ids['vad']}/attempts").json()["items"]
    assert len(attempts) == 1, attempts
    retried = q.retry(ids["vad"])
    assert retried["status"] in ("QUEUED", "WAITING_PROVIDER") and retried["max_attempts"] >= 2, retried
    assert q.status(ids["vad"]) == "QUEUED"

    # Cancel with cascade cancels the job and its dependents.
    cancelled = q.cancel(ids["asr"], cascade=True)
    assert cancelled["job"]["status"] == "CANCELLED", cancelled
    assert {ids["enrich"], ids["sentiment"]} <= set(cancelled["cancelled_job_ids"]), cancelled
    for ref in ("enrich", "sentiment"):
        assert q.status(ids[ref]) == "CANCELLED"

    # Without jobs:control a Process key cannot retry or cancel: 403 insufficient_scope.
    issued = store_only.store_cli("issue-service-key", "--installation", "q4-limited", "--no-primary-host", "--print-token")
    token = next(line for line in issued.stdout.split() if line.startswith("c1sk_"))
    limited = {"Authorization": f"Bearer {token}"}
    for verb in ("retry", "cancel"):
        refused = q.post(f"/jobs/{ids['vad']}/{verb}", {"reason": "x"}, expect=None, headers=limited)
        assert refused.status_code == 403 and error_code(refused) == "insufficient_scope", (verb, refused.status_code, refused.text)

    # The failed attempt left a usage row with its outcome (a release is not an attempt).
    usage = q.get(f"/conversations/{conversation}/usage").json()["items"]
    failed_rows = [u for u in usage if u["job_id"] == ids["vad"] and u["attempt_number"] == 1]
    assert len(failed_rows) == 1 and failed_rows[0]["outcome"] == "failed" and failed_rows[0]["error_code"] == "provider_error", usage


def test_q4_progress_reports_settled_after_every_job_finishes(store_only):
    q = worker_harness(store_only)
    run = q.ingest()
    for _ in range(4):
        batch = q.claim(max_jobs=16)["jobs"]
        for claimed in batch:
            q.complete(claimed)
    progress = q.get(f"/conversations/{run['conversation_id']}/progress").json()
    assert progress["settled"] is True, progress
    groups = {g["kind"]: g["state"] for g in progress["groups"]}
    assert groups["transcript"] == "available" and groups["text_sentiment"] == "available", groups


# --- q5: reanalysis requests (shared stack: the real Process plans the graphs) -------------------


def _publishers(types) -> set:
    return {JOB_TYPE_RULES[JobType(t)].publishes.value for t in types if JOB_TYPE_RULES[JobType(t)].publishes is not None}


@pytest.mark.parametrize("kind", ["qa", "summary", "contact_signals", "full"])
def test_q5_reanalysis_plans_the_graph_subset_for_its_kind(stack, reviewer_session, kind):
    receipt = settled_call(stack, "call_01_compliant", agent_id=f"agent-q5-{kind}")
    call_id = receipt["call_id"]
    key = f"q5-{kind}-{call_id}"
    created = reviewer_session.post(f"/calls/{call_id}/reanalysis-requests", json={"kind": kind, "note": "e2e"}, idempotency_key=key)
    assert created.status_code == 201, created.text
    request = created.json()
    assert request["kind"] == kind and request["status"] in ("pending", "claimed", "fulfilled"), request

    # Same key, same body: the same request (no duplicate). Same key, different body: 409.
    again = reviewer_session.post(f"/calls/{call_id}/reanalysis-requests", json={"kind": kind, "note": "e2e"}, idempotency_key=key)
    assert again.status_code in (200, 201) and again.json()["id"] == request["id"], again.text
    other_kind = "summary" if kind != "summary" else "qa"
    reused = reviewer_session.post(f"/calls/{call_id}/reanalysis-requests", json={"kind": other_kind}, idempotency_key=key)
    assert reused.status_code == 409 and error_code(reused) == "idempotency_key_reused", (reused.status_code, reused.text)
    listed = [r for r in reviewer_session.get(f"/calls/{call_id}/reanalysis-requests").json()["items"] if r["idempotency_key"] == key]
    assert len(listed) == 1, listed

    fulfilled = poll(lambda: reviewer_session.get(f"/reanalysis-requests/{request['id']}").json(),
                     lambda r: r["status"] in ("fulfilled", "rejected"), timeout=30, what=f"{kind} reanalysis to be planned")
    assert fulfilled["status"] == "fulfilled" and fulfilled["graph_id"], fulfilled
    stack.wait_until_settled(call_id)
    types = set(graph_job_types(stack, fulfilled["graph_id"]).values())
    expected = {k.value for k in REANALYSIS_KIND_AFFECTS[ReanalysisKind(kind)]}
    assert _publishers(types) == expected, f"{kind}: graph publishes {sorted(_publishers(types))}, contract expects {sorted(expected)} (job types {sorted(types)})"
    if kind != "full":
        assert "asr" not in types and "validation_vad" not in types, f"{kind} reanalysis must not redo transcription: {sorted(types)}"
    detail = reviewer_session.get(f"/calls/{call_id}").json()
    for group in detail["results"]:
        if group["kind"] in expected:
            assert group["state"] == "available" and group["version"] >= 2, group


def test_q5_speaker_correction_and_draft_test_kinds(stack, reviewer_session):
    receipt = settled_call(stack, "call_01_compliant", agent_id="agent-q5-speaker")
    call_id = receipt["call_id"]
    review = reviewer_session.get(f"/calls/{call_id}/review").json()
    created = reviewer_session.post(f"/calls/{call_id}/speaker-corrections",
                                    json={"correction": {"turn_id": 1, "speaker": "AGENT"}, "expected_version": review["review_version"]})
    assert created.status_code == 201, created.text
    request = created.json()
    assert request["kind"] == "speaker_correction", request
    done = poll(lambda: reviewer_session.get(f"/reanalysis-requests/{request['id']}").json(), lambda r: r["status"] in ("fulfilled", "rejected"),
                timeout=30, what="speaker correction to be planned")
    assert done["status"] == "fulfilled", done
    stack.wait_until_settled(call_id)
    types = set(graph_job_types(stack, done["graph_id"]).values())
    expected = {k.value for k in REANALYSIS_KIND_AFFECTS[ReanalysisKind.SPEAKER_CORRECTION]}
    assert _publishers(types) == expected and "speaker_attribution" in types and "asr" not in types, sorted(types)


# --- q6: usage ----------------------------------------------------------------------------------


def test_q6_usage_rows_and_report(stack, admin_session, reviewer_session):
    started = datetime.now(timezone.utc) - timedelta(seconds=5)
    receipt = settled_call(stack, "call_01_compliant", agent_id="agent-q6")
    conversation = receipt["conversation_id"]
    progress = stack.progress(conversation)
    total_jobs = sum(g["total"] for g in progress["groups"]) + progress["supporting_jobs_total"]

    rows = admin_session.get(f"/conversations/{conversation}/usage").json()["items"]
    jobs = stack.store_get("/jobs", session="service", params={"conversation_id": conversation, "limit": 200}).json()["items"]
    assert jobs, "listJobs should list the conversation's jobs"
    attempts = {(j["id"], n) for j in jobs for n in range(1, j["attempt_count"] + 1)}
    recorded = {(r["job_id"], r["attempt_number"]) for r in rows}
    assert attempts <= recorded, f"attempts without a usage row: {sorted(attempts - recorded)}"
    assert len(recorded) >= total_jobs

    # The report: rows grouped by purpose, and totals that include this conversation's attempts.
    end = datetime.now(timezone.utc) + timedelta(seconds=5)
    report = admin_session.post("/admin/usage/report", json={"start": started.isoformat(), "end": end.isoformat(), "group_by": ["outcome"]})
    assert report.status_code == 200, report.text
    body = report.json()
    succeeded = sum(r["succeeded"] for r in body["rows"])
    attempts_total = sum(r["attempts"] for r in body["rows"])
    mine = [r for r in rows if r["outcome"] == "succeeded"]
    assert attempts_total >= len(rows) and succeeded >= len(mine), body
    assert body["per_scored_call"]["scored_calls"] >= 1, body["per_scored_call"]
    inference = sum(r["inference_seconds_total"] for r in body["rows"])
    assert inference >= sum(r["inference_seconds"] or 0 for r in rows) - 1e-6

    # Per-attempt listing and CSV export agree with the rows; reviewers may not read the report.
    listed = list(admin_session.get("/admin/usage/records", params={"limit": 200, "conversation_id": conversation}).json()["items"])
    assert {r["id"] for r in rows} <= {r["id"] for r in listed} or len(listed) == 200, "listUsageRecords misses this conversation's rows"
    csv = admin_session.post("/admin/usage/records.csv", json={"start": started.isoformat(), "end": end.isoformat()})
    assert csv.status_code == 200 and "text/csv" in csv.headers.get("content-type", ""), (csv.status_code, csv.headers)
    assert all(r["id"] in csv.text for r in rows), "the CSV export should contain every usage row of the window"
    assert reviewer_session.post("/admin/usage/report", json={"start": started.isoformat(), "end": end.isoformat()}).status_code == 403


# --- q7: hardware profiles and catalog snapshots -------------------------------------------------


def test_q7_process_publishes_catalog_and_hardware_that_store_serves_unchanged(stack, admin_session):
    overview = stack.process_get("/overview").json()
    process_catalog = stack.process_get("/catalog").json()
    snapshots = [s for s in admin_session.get("/catalog-snapshots").json()["items"] if s["installation_id"] == stack.installation_id]
    assert snapshots, "Store has no catalog snapshot from the stack's Process installation"
    latest = max(snapshots, key=lambda s: s["published_at"])
    assert latest["catalog_version"] == overview["catalog"]["version"] == process_catalog["version"], (latest["catalog_version"], overview["catalog"])
    store_entries = {e["entry"]["entry_id"]: e for e in latest["entries"]}
    process_entries = {e["entry_id"]: e for e in process_catalog["entries"]}
    assert set(store_entries) == set(process_entries), sorted(set(store_entries) ^ set(process_entries))
    for entry_id, entry in process_entries.items():
        stored = store_entries[entry_id]
        for field in ("model_revision", "route_class", "provider_type", "destination_host", "status"):
            assert stored[field] == entry[field], (entry_id, field, stored[field], entry[field])

    hardware_id = overview.get("hardware_profile_id")
    assert hardware_id, overview
    profiles = {p["id"]: p for p in admin_session.get("/hardware-profiles").json()["items"]}
    assert hardware_id in profiles, (hardware_id, sorted(profiles))


def test_q7_hardware_and_catalog_round_trip(stack, stack_factory, admin_session):
    private = stack_factory(name="catalog", with_process=False)
    headers = private.service_headers()
    from call1.contracts.usage import HardwareProfileFields, hardware_fingerprint

    fields = {"kind": "process_host", "source": "measured", "chip": "Test Chip", "accelerator": None, "memory_bytes": 17179869184,
              "os_name": "Darwin", "os_version": "27.2.0", "runtime_versions": {"python": "3.12.1"}}
    profile = {**fields, "fingerprint": hardware_fingerprint(HardwareProfileFields.model_validate(fields))}
    put = private.store_request("PUT", "/hardware-profiles", json=profile, headers=headers)
    assert put.status_code == 200, put.text
    stored = put.json()
    assert {k: stored[k] for k in profile} == profile, stored
    again = private.store_request("PUT", "/hardware-profiles", json=profile, headers=headers).json()
    assert again["id"] == stored["id"], (again, stored)
    listed = private.store_get("/hardware-profiles", session="service").json()["items"]
    assert [p for p in listed if p["id"] == stored["id"]] and {k: listed[0][k] for k in profile} == profile

    # A catalog snapshot (the shared stack's, re-addressed to this installation) is served back unchanged.
    source = next(s for s in admin_session.get("/catalog-snapshots").json()["items"] if s["installation_id"] == stack.installation_id)
    snapshot = dict(source, installation_id=private.installation_id)
    published = private.store_request("PUT", "/catalog-snapshot", json=snapshot, headers=headers)
    assert published.status_code == 200, published.text
    admin = private.admin()
    served = [s for s in admin.get("/catalog-snapshots").json()["items"] if s["catalog_version"] == snapshot["catalog_version"]]
    assert served, "the published snapshot is not listed"
    got = served[0]
    for field in ("installation_id", "catalog_version", "entries", "defaults"):
        assert got[field] == snapshot[field], field
