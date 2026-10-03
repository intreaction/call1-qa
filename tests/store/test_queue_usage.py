"""Usage rows per attempt and their report, CSV export and medians; hardware profiles; catalog snapshots."""

from __future__ import annotations

import csv
import io

from call1.contracts.common import ServiceScope
from call1.contracts.jobs import JobType
from call1.contracts.usage import USAGE_CSV_COLUMNS, HardwareProfileFields, hardware_fingerprint

from .test_queue_harness import hooks, q, rubrics, usage  # noqa: F401

RANGE = {"start": "2026-09-25T00:00:00Z", "end": "2026-09-26T00:00:00Z"}


def _usage_history(q):
    """asr: one failed attempt then a success; vad: one success; sentiment: one abandoned attempt."""
    setup = q.ingest()
    ids = setup["ids"]
    q.fail(q.claim_one(ids["asr"]), "provider_error")
    q.clock.advance(30)
    q.complete(q.claim_one(ids["asr"]), usage_body=usage(billing_units=[{"unit": "requests", "quantity": 1}]))
    q.complete(q.claim_one(ids["vad"]))
    q.claim_one(ids["sentiment"])
    q.clock.advance(400)
    q.claim(max_jobs=1, job_types=[JobType.ENRICHMENT])
    return setup


def test_every_attempt_has_exactly_one_usage_row(q, admin_session):
    setup = _usage_history(q)
    rows = q.get(f"/conversations/{setup['conversation_id']}/usage").json()["items"]
    assert sorted((r["job_type"], r["attempt_number"], r["outcome"]) for r in rows) == [
        ("asr", 1, "failed"), ("asr", 2, "succeeded"), ("text_sentiment", 1, "abandoned"), ("validation_vad", 1, "succeeded")]
    vad = next(r for r in rows if r["job_type"] == "validation_vad")
    assert vad["purpose"] is None and vad["route_class"] is None and vad["catalog_entry"] is None  # a code stage
    assert q.get(f"/conversations/{setup['conversation_id']}/usage", headers=admin_session.read_headers).json()["items"] == rows
    page = q.get("/admin/usage/records", headers=admin_session.read_headers, params={"limit": 3}).json()
    rest = q.get("/admin/usage/records", headers=admin_session.read_headers, params={"limit": 3, "page_token": page["next_page_token"]}).json()
    assert len(page["items"]) == 3 and len(rest["items"]) == 1 and rest["next_page_token"] is None
    failed_only = q.get("/admin/usage/records", headers=admin_session.read_headers, params={"outcome": "failed"}).json()["items"]
    assert [r["error_code"] for r in failed_only] == ["provider_error"]


def test_usage_report_and_csv_agree(q, admin_session):
    _usage_history(q)
    report = q.post("/admin/usage/report", {**RANGE, "group_by": ["purpose", "outcome"], "include_estimates": True}, headers=admin_session.headers).json()
    rows = {(r["keys"]["purpose"], r["keys"]["outcome"]): r for r in report["rows"]}
    assert set(rows) == {("asr", "failed"), ("asr", "succeeded"), ("text_sentiment", "abandoned"), ("none", "succeeded")}
    assert rows[("asr", "succeeded")]["billing_units"] == [{"unit": "requests", "quantity": 1.0}]
    assert rows[("text_sentiment", "abandoned")]["attempts_with_unavailable_tokens"] == 1
    assert sum(r["attempts"] for r in report["rows"]) == 4 and report["estimates_by_row"] == [None] * len(report["rows"])
    assert report["per_scored_call"]["scored_calls"] == 0 and report["per_scored_call"]["tokens_by_route_class"] == {"appliance": 240, "none": 120}
    exported = q.post("/admin/usage/records.csv", RANGE, headers=admin_session.headers)
    assert exported.headers["content-type"].startswith("text/csv")
    lines = list(csv.reader(io.StringIO(exported.text)))
    assert lines[0] == USAGE_CSV_COLUMNS and len(lines) == 5
    by_column = [dict(zip(lines[0], line)) for line in lines[1:]]
    assert sum(int(r["tokens_input"] or 0) for r in by_column) == sum(r["tokens_input_total"] for r in report["rows"])
    assert {r["outcome"] for r in by_column} == {"failed", "succeeded", "abandoned"}
    assert next(r for r in by_column if r["outcome"] == "abandoned")["tokens_input"] == ""
    audit = q.get("/admin/audit", headers=admin_session.read_headers, params={"action": "usage_report_exported"}).json()["items"]
    assert [e["details"]["format"] for e in audit] == ["csv", "json"]
    assert q.post("/admin/usage/report", RANGE, headers=q.mint_session("supervisor").headers, expect=403).json()["code"] == "insufficient_role"


def test_usage_medians_are_aggregates_per_entry_route_and_hardware(q, admin_session):
    _usage_history(q)
    medians = q.get("/usage/medians").json()
    rows = {(r["catalog_entry"]["entry_id"], r["purpose"]): r for r in medians["rows"]}
    assert set(rows) == {("mlx-asr", "asr"), ("mlx-text_sentiment", "text_sentiment")}
    asr = rows[("mlx-asr", "asr")]
    assert asr["attempts"] == 2 and asr["succeeded"] == 1 and asr["median_inference_seconds"] == 2.0 and asr["median_tokens_input"] == 100.0
    assert rows[("mlx-text_sentiment", "text_sentiment")]["median_inference_seconds"] is None
    assert q.get("/usage/medians", params={"purpose": "asr"}).json()["rows"][0]["purpose"] == "asr"
    assert q.get("/usage/medians", headers=admin_session.read_headers).json()["rows"] == medians["rows"]


def test_hardware_profiles_upsert_by_fingerprint(q, admin_session):
    fields = HardwareProfileFields(kind="process_host", source="measured", chip="Apple M4 Pro", memory_bytes=64 * 2**30, os_name="macOS", os_version="26.0")
    body = {**fields.model_dump(mode="json"), "fingerprint": hardware_fingerprint(fields)}
    first = q.client.put("/store/v1/hardware-profiles", json=body, headers=q.headers).json()
    assert first["fingerprint"] == body["fingerprint"] and first["first_seen_at"] == first["last_seen_at"]
    q.clock.advance(60)
    again = q.client.put("/store/v1/hardware-profiles", json=body, headers=q.headers).json()
    assert again["id"] == first["id"] and again["last_seen_at"] == "2026-09-25T12:01:00Z" and again["first_seen_at"] == first["first_seen_at"]
    bad = q.client.put("/store/v1/hardware-profiles", json=dict(body, chip="Apple M5"), headers=q.headers)
    assert bad.status_code == 422
    listed = q.get("/hardware-profiles", headers=admin_session.read_headers).json()["items"]
    assert [p["id"] for p in listed] == [first["id"]]


def _snapshot(installation_id, *, host="in-process", route_class="appliance", version="cat-1"):
    return {"installation_id": installation_id, "catalog_version": version, "published_at": "2026-09-25T12:00:00Z", "defaults": {},
            "entries": [{"entry": {"entry_id": "mlx-asr", "entry_version": 1}, "display_name": "Whisper", "purposes": ["asr"], "provider_type": "mlx",
                         "route_class": route_class, "destination_host": host, "model_family": "whisper", "model_revision": "r1",
                         "status": "available", "qualified_for": ["asr"]}]}


def test_catalog_snapshots_are_published_per_installation(q, reviewer_session, mint_service_key):
    first = q.client.put("/store/v1/catalog-snapshot", json=_snapshot(q.installation_id), headers=q.headers)
    assert first.status_code == 200 and first.json()["catalog_version"] == "cat-1"
    q.client.put("/store/v1/catalog-snapshot", json=_snapshot(q.installation_id), headers=q.headers)  # unchanged: no new audit event
    q.client.put("/store/v1/catalog-snapshot", json=_snapshot(q.installation_id, version="cat-2"), headers=q.headers)
    listed = q.get("/catalog-snapshots", headers=reviewer_session.read_headers).json()["items"]
    assert [s["catalog_version"] for s in listed] == ["cat-2"]
    other = mint_service_key([ServiceScope.CATALOG_PUBLISH])
    foreign = q.client.put("/store/v1/catalog-snapshot", json=_snapshot(q.installation_id), headers=other.headers)
    assert foreign.status_code == 403 and foreign.json()["code"] == "forbidden"
    call1 = q.client.put("/store/v1/catalog-snapshot", json=_snapshot(q.installation_id, host="api.call1.cc", route_class="customer_directed"), headers=q.headers)
    assert call1.status_code == 403 and call1.json()["code"] == "route_not_permitted"
    audit = q.get("/admin/audit", headers=q.mint_session("admin").read_headers, params={"action": "catalog_snapshot_published"}).json()["items"]
    assert [e["details"]["catalog_version"] for e in audit] == ["cat-2", "cat-1"]
    feed = q.get("/changes", params={"kinds": "catalog"}).json()["events"]
    assert [e["resource_id"] for e in feed] == [q.installation_id, q.installation_id]
