"""Rubrics: the seeded default, drafts with draft_revision, publish, versions, retire, snapshot API."""

from __future__ import annotations

import pytest

from call1.contracts.common import ServiceScope, canonical_digest
from call1.contracts.errors import ErrorCode
from call1.contracts.rubrics import RubricDefinition, RubricSnapshotContent
from call1.store import db
from call1.store.errors import StoreError
from call1.store.results import api as results_api

SEED = "call1_standard_v2"


def _definition(rubric_id: str = "acme_v1", *, name: str = "Acme QA", criteria: int = 1) -> dict:
    return {
        "rubric_id": rubric_id, "name": name, "category": "BANKING", "pass_threshold": 75,
        "criteria": [{"criterion_id": f"C-{i}", "name": f"Criterion {i}", "weight": 50,
                      "check": {"check_type": "phrase_any", "phrases": ["thank you"]}} for i in range(criteria)],
    }


def test_seeded_default_rubric_is_the_pre_split_default(client, reviewer_session):
    from call1.pipeline.evaluator import DEFAULT_RUBRIC
    legacy = DEFAULT_RUBRIC.model_dump(mode="json")
    for criterion in legacy["criteria"]:
        criterion["rule_type"] = None
        criterion["parameters"] = None
    expected = RubricDefinition.model_validate(legacy)
    got = client.get(f"/store/v1/rubrics/{SEED}", headers=reviewer_session.read_headers).json()
    assert got["ref"] == {"rubric_id": SEED, "version": 1, "digest": canonical_digest(expected)}
    assert RubricDefinition.model_validate(got["definition"]) == expected
    assert got["status"] == "active" and got["published_by_account_id"] is None
    listed = client.get("/store/v1/rubrics", headers=reviewer_session.read_headers).json()["items"]
    assert [(s["rubric_id"], s["current_version"], s["criteria_count"], s["has_draft"]) for s in listed] == [(SEED, 1, 4, False)]


def test_process_reads_rubrics_with_jobs_write(client, mint_service_key):
    key = mint_service_key([ServiceScope.JOBS_WRITE])
    assert client.get(f"/store/v1/rubrics/{SEED}/versions/1", headers=key.headers).status_code == 200
    other = mint_service_key([ServiceScope.USAGE_READ])
    assert client.get(f"/store/v1/rubrics/{SEED}", headers=other.headers).json()["code"] == "insufficient_scope"


def test_draft_revisions_conflict_and_never_repeat(client, supervisor_session, reviewer_session):
    h = supervisor_session.headers
    url = "/store/v1/rubrics/acme_v1/draft"
    assert client.put(url, json={"definition": _definition(), "expected_draft_revision": 3}, headers=h).json()["code"] == "rubric_version_conflict"
    first = client.put(url, json={"definition": _definition()}, headers=h).json()
    assert first["draft_revision"] == 1 and first["based_on_version"] is None
    stale = client.put(url, json={"definition": _definition(name="B"), "expected_draft_revision": 0}, headers=h)
    assert stale.status_code == 409 and stale.json()["details"] == {"current_version": 0, "draft_revision": 1}
    second = client.put(url, json={"definition": _definition(name="B"), "expected_draft_revision": 1}, headers=h).json()
    assert second["draft_revision"] == 2 and second["definition"]["name"] == "B"
    mismatch = client.put(url, json={"definition": _definition("other"), "expected_draft_revision": 2}, headers=h)
    assert mismatch.status_code == 422
    assert client.get(url, headers=reviewer_session.read_headers).json()["code"] == "insufficient_role"
    listed = client.get("/store/v1/rubrics", headers=supervisor_session.read_headers).json()["items"]
    assert {"acme_v1": (None, True)} == {s["rubric_id"]: (s["current_version"], s["has_draft"]) for s in listed if s["rubric_id"] == "acme_v1"}
    assert client.get("/store/v1/rubrics/acme_v1", headers=supervisor_session.read_headers).status_code == 404
    assert client.delete(url, headers=h).status_code == 204
    assert client.delete(url, headers=h).status_code == 404
    third = client.put(url, json={"definition": _definition()}, headers=h).json()
    assert third["draft_revision"] == 1  # the rubric had no version, so discarding removed it entirely


def test_publish_versions_and_retire(client, supervisor_session, admin_session):
    h = supervisor_session.headers
    client.put("/store/v1/rubrics/acme_v1/draft", json={"definition": _definition()}, headers=h)
    wrong = client.post("/store/v1/rubrics/acme_v1/publish", json={"expected_current_version": 1, "expected_draft_revision": 1}, headers=h)
    assert wrong.status_code == 409 and wrong.json()["code"] == "rubric_version_conflict"
    v1 = client.post("/store/v1/rubrics/acme_v1/publish", json={"expected_current_version": 0, "expected_draft_revision": 1, "notes": "first"},
                     headers=h)
    assert v1.status_code == 201
    body = v1.json()
    assert body["ref"]["version"] == 1 and body["ref"]["digest"] == canonical_digest(RubricDefinition.model_validate(_definition()))
    assert body["published_by_account_id"] == supervisor_session.account_id
    assert client.get("/store/v1/rubrics/acme_v1/draft", headers=h).status_code == 404  # publishing consumes the draft
    assert client.post("/store/v1/rubrics/acme_v1/publish", json={"expected_current_version": 1, "expected_draft_revision": 1},
                       headers=h).status_code == 404

    draft = client.put("/store/v1/rubrics/acme_v1/draft", json={"definition": _definition(criteria=2)}, headers=h).json()
    assert draft["based_on_version"] == 1 and draft["draft_revision"] == 2
    v2 = client.post("/store/v1/rubrics/acme_v1/publish", json={"expected_current_version": 1, "expected_draft_revision": 2}, headers=h).json()
    assert v2["ref"]["version"] == 2
    versions = client.get("/store/v1/rubrics/acme_v1/versions", headers=h).json()["items"]
    assert [v["ref"]["version"] for v in versions] == [2, 1]
    assert client.get("/store/v1/rubrics/acme_v1/versions/1", headers=h).json()["definition"]["criteria"][0]["criterion_id"] == "C-0"

    empty = client.put("/store/v1/rubrics/acme_v1/draft", json={"definition": _definition(criteria=0), "expected_draft_revision": 0}, headers=h).json()
    bad = client.post("/store/v1/rubrics/acme_v1/publish", json={"expected_current_version": 2, "expected_draft_revision": empty["draft_revision"]},
                      headers=h)
    assert bad.status_code == 422 and bad.json()["code"] == "validation_failed"

    stale = client.post("/store/v1/rubrics/acme_v1/retire", json={"expected_current_version": 1, "reason": "old"}, headers=h)
    assert stale.status_code == 409
    retired = client.post("/store/v1/rubrics/acme_v1/retire", json={"expected_current_version": 2, "reason": "replaced"}, headers=h).json()
    assert retired["status"] == "retired"
    visible = {s["rubric_id"] for s in client.get("/store/v1/rubrics", headers=h).json()["items"]}
    assert "acme_v1" not in visible
    everything = {s["rubric_id"] for s in client.get("/store/v1/rubrics", params={"include_retired": "true"}, headers=h).json()["items"]}
    assert "acme_v1" in everything
    audit = client.get("/store/v1/admin/audit", params={"target_kind": "rubric"}, headers=admin_session.read_headers).json()["items"]
    assert [e["action"] for e in audit] == ["rubric_retired", "rubric_published", "rubric_published"]
    assert all(e["actor"]["account_id"] == supervisor_session.account_id for e in audit)


def test_rubric_list_filters_and_pages(client, supervisor_session):
    h = supervisor_session.headers
    for rid in ("a_rubric", "b_rubric"):
        client.put(f"/store/v1/rubrics/{rid}/draft", json={"definition": _definition(rid)}, headers=h)
    page = client.get("/store/v1/rubrics", params={"limit": 2}, headers=h).json()
    assert [s["rubric_id"] for s in page["items"]] == ["a_rubric", "b_rubric"]
    rest = client.get("/store/v1/rubrics", params={"limit": 2, "page_token": page["next_page_token"]}, headers=h).json()
    assert [s["rubric_id"] for s in rest["items"]] == [SEED]
    banking = client.get("/store/v1/rubrics", params={"category": "BANKING"}, headers=h).json()["items"]
    assert {s["rubric_id"] for s in banking} == {"a_rubric", "b_rubric"}


def test_snapshot_api_for_the_queue_area(store, client, supervisor_session):
    with store.connection() as conn:
        snapshot = results_api.rubric_snapshot_content(conn, SEED, 1)
        assert isinstance(snapshot, RubricSnapshotContent) and snapshot.source == "published" and snapshot.rubric_version == 1
        assert snapshot.digest == results_api.rubric_version(conn, SEED, 1).ref.digest
        assert results_api.rubric_snapshot_content(conn, SEED, 9) is None and results_api.rubric_version(conn, "nope", 1) is None
        with pytest.raises(StoreError) as missing:
            results_api.draft_snapshot_content(conn, SEED, expected_draft_revision=1)
        assert missing.value.code is ErrorCode.NOT_FOUND
    client.put(f"/store/v1/rubrics/{SEED}/draft", json={"definition": {**_definition(SEED), "name": "Edited"}}, headers=supervisor_session.headers)
    with store.connection() as conn, db.transaction(conn):
        draft = results_api.draft_snapshot_content(conn, SEED, expected_draft_revision=1)
        assert draft.source == "draft" and draft.draft_revision == 1 and draft.definition.name == "Edited"
        with pytest.raises(StoreError) as moved:
            results_api.draft_snapshot_content(conn, SEED, expected_draft_revision=2)
        assert moved.value.code is ErrorCode.RUBRIC_VERSION_CONFLICT and moved.value.details == {"current_version": 1, "draft_revision": 1}


def test_rubric_changes_reach_the_feed(client, supervisor_session, admin_session):
    client.put("/store/v1/rubrics/acme_v1/draft", json={"definition": _definition()}, headers=supervisor_session.headers)
    client.post("/store/v1/rubrics/acme_v1/publish", json={"expected_current_version": 0, "expected_draft_revision": 1},
                headers=supervisor_session.headers)
    events = client.get("/store/v1/changes", params={"kinds": ["rubric"]}, headers=admin_session.read_headers).json()["events"]
    assert [(e["resource_id"], e["status"], e["version"]) for e in events] == [("acme_v1", "draft_saved", 1), ("acme_v1", "published", 1)]
