"""The human review queue: items created when QA commits, supersession, distribution, the item
state machine, rules and reviewer profiles. None of it touches the processing job queue."""

from __future__ import annotations

from call1.contracts.contents import VerdictStatus
from call1.contracts.jobs import JobType
from call1.contracts.reviews import REVIEW_QUEUE_TRANSITIONS, ReviewQueueStatus

from .test_results_harness import accounts, fq, transcript  # noqa: F401

TRIAGE = "rule-triage-critical-lowconf"
MANDATE = "rule-mandate-dispute-escalation"
AUDIT_SAMPLE = "rule-audit-sample-drift"


def _items(client, session, **params):
    return client.get("/store/v1/review-queue", params=params, headers=session.read_headers).json()["items"]


def _unsampled(items):
    """Drop the seeded 10% drift-audit sample. It draws on sha256(call_id:rule:version), and call IDs
    are random per run, so a confident PASS lands in it about one run in ten."""
    return [i for i in items if i["rule_id"] != AUDIT_SAMPLE]


def test_qa_commit_creates_items_for_matching_rules(fq, client, reviewer_session, accounts):
    conv = fq.register(agent_id="agent-9")
    graph = fq.graph(conv, [JobType.ASR])
    fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("CALLER", "I want to dispute this charge")])})
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.95), ("SEC-01", VerdictStatus.PASS, 0.5)], overall_score=45, passed=False,
                 critical_failure=True)
    items = _items(client, reviewer_session)
    assert {i["rule_id"] for i in items} == {TRIAGE, MANDATE}
    triage = next(i for i in items if i["rule_id"] == TRIAGE)
    assert triage["status"] == "PENDING" and triage["evaluation_version"] == 1 and triage["stale"] is False
    assert triage["reason"] == "Triage: critical compliance breach and low-confidence verdicts."
    assert triage["urgency_score"] == 95.5 and triage["agent_id"] == "agent-9" and triage["critical_failure"] is True
    assert triage["assigned_to_account_id"] is None  # no accounts accept assignments yet

    calm = fq.register()
    fq.ingest_qa(calm, [("REG-01", VerdictStatus.PASS, 0.99)])
    assert all(i["rule_id"] == AUDIT_SAMPLE for i in _items(client, reviewer_session, call_id=calm.call_id))
    stats = client.get("/store/v1/review-queue/stats", headers=reviewer_session.read_headers).json()
    assert stats["pending"] >= 2 and stats["by_stream"]["TRIAGE"] == 1 and stats["unassigned"] == stats["pending"]


def test_new_machine_version_supersedes_unresolved_items(fq, client, reviewer_session):
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.9)], critical_failure=True, passed=False, overall_score=50)
    claimed = client.post("/store/v1/review-queue/claim-next", headers=reviewer_session.headers).json()["item"]
    assert claimed["status"] == "IN_REVIEW" and claimed["assigned_to_account_id"] == reviewer_session.account_id
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.PASS, 0.99)])  # no rule matches the new scorecard
    old = client.get(f"/store/v1/review-queue/items/{claimed['id']}", headers=reviewer_session.read_headers).json()
    assert old["status"] == "SUPERSEDED" and old["stale"] is True and old["superseded_by_item_id"]
    replacement = client.get(f"/store/v1/review-queue/items/{old['superseded_by_item_id']}", headers=reviewer_session.read_headers).json()
    assert replacement["evaluation_version"] == 2 and replacement["status"] == "PENDING" and replacement["rule_id"] == TRIAGE
    assert replacement["assigned_to_account_id"] == reviewer_session.account_id  # carried forward to the same reviewer
    resolve = client.post(f"/store/v1/review-queue/items/{claimed['id']}/resolve",
                          json={"status": "APPROVED", "expected_item_version": old["item_version"], "expected_review_version": 0,
                                "evaluation_version": 1}, headers=reviewer_session.headers)
    assert resolve.status_code == 409 and resolve.json()["code"] == "conflict"
    fresh = _unsampled(_items(client, reviewer_session, include_stale="false"))
    assert [i["id"] for i in fresh] == [replacement["id"]]


def test_assignment_needs_a_supervisor_and_a_known_account(fq, client, reviewer_session, supervisor_session):
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.9)], critical_failure=True, passed=False)
    item = _items(client, reviewer_session)[0]
    base = f"/store/v1/review-queue/items/{item['id']}"
    body = {"account_id": reviewer_session.account_id, "expected_item_version": item["item_version"]}
    assert client.post(f"{base}/assign", json=body, headers=reviewer_session.headers).json()["code"] == "insufficient_role"
    assert client.post(f"{base}/assign", json=body, headers=supervisor_session.headers).status_code == 404  # not an active account
    assert client.get("/store/v1/review-queue/items/rvw_missing", headers=reviewer_session.read_headers).status_code == 404


def test_item_state_machine(fq, client, mint_session, supervisor_session, admin_session, accounts):
    alice, bob = mint_session("reviewer", display_name="Alice"), mint_session("reviewer", display_name="Bob")
    accounts.add(alice)
    accounts.add(bob)
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.9)], critical_failure=True, passed=False)
    item = _items(client, alice)[0]
    iid, base = item["id"], f"/store/v1/review-queue/items/{item['id']}"
    assert item["assigned_to_account_id"] in (alice.account_id, bob.account_id)  # LEAST_OUTSTANDING picked a reviewer
    version = item["item_version"]

    assigned = client.post(f"{base}/assign", json={"account_id": alice.account_id, "expected_item_version": version}, headers=supervisor_session.headers)
    assert assigned.status_code == 200 and assigned.json()["assigned_display_name"] == "Alice"
    version = assigned.json()["item_version"]
    assert client.post(f"{base}/assign", json={"account_id": bob.account_id, "expected_item_version": version - 1},
                       headers=supervisor_session.headers).json()["details"]["current_item_version"] == version
    assert client.post(f"{base}/start", json={"expected_item_version": version}, headers=bob.headers).json()["code"] == "forbidden"
    started = client.post(f"{base}/start", json={"expected_item_version": version}, headers=alice.headers).json()
    assert started["status"] == "IN_REVIEW" and started["started_at"]
    version = started["item_version"]
    assert client.post(f"{base}/start", json={"expected_item_version": version}, headers=alice.headers).json()["code"] == "invalid_transition"
    assert client.post(f"{base}/release", json={"expected_item_version": version}, headers=bob.headers).json()["code"] == "insufficient_role"
    released = client.post(f"{base}/release", json={"expected_item_version": version, "note": "out of office"}, headers=alice.headers).json()
    assert released["status"] == "PENDING" and released["assigned_to_account_id"] is None
    version = released["item_version"]
    started = client.post(f"{base}/start", json={"expected_item_version": version}, headers=bob.headers).json()
    version = started["item_version"]

    resolve = {"status": "OVERRIDDEN", "expected_item_version": version, "expected_review_version": 0, "evaluation_version": 1,
               "reviewer_notes": "the disclosure was there"}
    assert client.post(f"{base}/resolve", json=resolve, headers=alice.headers).json()["code"] == "insufficient_role"
    assert client.post(f"{base}/resolve", json={**resolve, "expected_review_version": 3}, headers=bob.headers).json()["code"] \
        == "review_version_conflict"
    done = client.post(f"{base}/resolve", json=resolve, headers=bob.headers)
    assert done.status_code == 200 and done.json()["review_version"] == 1
    final = client.get(base, headers=bob.read_headers).json()
    assert final["status"] == "OVERRIDDEN" and final["resolved_by_account_id"] == bob.account_id
    assert client.post(f"{base}/release", json={"expected_item_version": final["item_version"]}, headers=bob.headers).json()["code"] \
        == "invalid_transition"
    review = client.get(f"/store/v1/calls/{conv.call_id}/review", headers=bob.read_headers).json()
    assert review["review_version"] == 1 and review["reviewed_evaluation_version"] == 1
    audit = client.get("/store/v1/admin/audit", params={"action": "review_resolved"}, headers=admin_session.read_headers).json()["items"]
    assert audit[0]["target"] == {"kind": "review_queue_item", "id": iid}


def test_supervisor_resolves_for_another_reviewer(fq, client, mint_session, supervisor_session, accounts):
    alice = mint_session("reviewer", display_name="Alice")
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.9)], critical_failure=True, passed=False)
    item = client.post("/store/v1/review-queue/claim-next", headers=alice.headers).json()["item"]
    assert client.post("/store/v1/review-queue/claim-next", headers=alice.headers).json() == {"item": None}
    done = client.post(f"/store/v1/review-queue/items/{item['id']}/resolve",
                       json={"status": "APPROVED", "expected_item_version": item["item_version"], "expected_review_version": 0,
                             "evaluation_version": 1}, headers=supervisor_session.headers)
    assert done.status_code == 200


def test_distribution_strategies(fq, client, mint_session, supervisor_session, accounts):
    ann, ben = mint_session("reviewer", display_name="Ann"), mint_session("reviewer", display_name="Ben")
    accounts.add(ann)
    accounts.add(ben)
    profile = client.patch(f"/store/v1/admin/reviewer-profiles/{ben.account_id}", json={"skills": ["ESCALATIONS"], "capacity_weight": 0.5},
                           headers=supervisor_session.headers)
    assert profile.status_code == 200 and profile.json()["skills"] == ["ESCALATIONS"]
    assert client.patch("/store/v1/admin/reviewer-profiles/acct_nobody", json={"skills": []}, headers=supervisor_session.headers).status_code == 404
    for _ in range(3):
        conv = fq.register()
        graph = fq.graph(conv, [JobType.ASR])
        fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("CALLER", "this is fraud, a refund please")])})
        fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.9)], critical_failure=True, passed=False)
    mandate = _items(client, supervisor_session, stream="MANDATE")
    assert {i["assigned_to_account_id"] for i in mandate} == {ben.account_id}  # SKILL_MATCHED: only Ben holds ESCALATIONS
    profiles = {p["account_id"]: p for p in client.get("/store/v1/admin/reviewer-profiles", headers=supervisor_session.read_headers).json()["items"]}
    assert profiles[ben.account_id]["pending_count"] >= 3 and profiles[ann.account_id]["capacity_weight"] == 1.0
    triage = _items(client, supervisor_session, stream="TRIAGE")
    assert {i["assigned_to_account_id"] for i in triage} == {ann.account_id}  # LEAST_OUTSTANDING: Ben is busier per capacity

    client.patch(f"/store/v1/admin/reviewer-profiles/{ann.account_id}", json={"accepting_assignments": False}, headers=supervisor_session.headers)
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.FAIL, 0.9)], critical_failure=True, passed=False)
    assert _items(client, supervisor_session, call_id=conv.call_id)[0]["assigned_to_account_id"] == ben.account_id


def test_rules_crud_with_expected_versions(fq, client, supervisor_session, reviewer_session, admin_session):
    rules = client.get("/store/v1/review-queue/rules", headers=reviewer_session.read_headers).json()["items"]
    assert [r["id"] for r in rules] == [TRIAGE, MANDATE, AUDIT_SAMPLE]
    rule = {"id": "rule-calibration", "name": "Calibration", "stream": "CALIBRATION", "rank": 5}
    url = "/store/v1/review-queue/rules/rule-calibration"
    assert client.put(url, json={"rule": rule, "expected_rule_version": 0}, headers=reviewer_session.headers).json()["code"] == "insufficient_role"
    created = client.put(url, json={"rule": rule, "expected_rule_version": 0}, headers=supervisor_session.headers).json()
    assert created["rule_version"] == 1 and created["updated_by_account_id"] == supervisor_session.account_id
    conflict = client.put(url, json={"rule": rule, "expected_rule_version": 0}, headers=supervisor_session.headers)
    assert conflict.status_code == 409 and conflict.json()["details"]["current_rule_version"] == 1
    mismatch = client.put(url, json={"rule": {**rule, "id": "other"}, "expected_rule_version": 1}, headers=supervisor_session.headers)
    assert mismatch.status_code == 422
    conv = fq.register()
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.PASS, 0.99)])
    assert [i["rule_id"] for i in _unsampled(_items(client, reviewer_session, call_id=conv.call_id))] == ["rule-calibration"]
    disabled = client.put(url, json={"rule": {**rule, "enabled": False}, "expected_rule_version": 1}, headers=supervisor_session.headers).json()
    assert disabled["rule_version"] == 2 and disabled["enabled"] is False
    assert client.delete(url, headers=supervisor_session.headers).status_code == 204
    assert client.delete(url, headers=supervisor_session.headers).status_code == 404
    audit = client.get("/store/v1/admin/audit", params={"action": "queue_rule_changed"}, headers=admin_session.read_headers).json()["items"]
    assert [e["details"]["change"] for e in audit] == ["deleted", "updated", "created"]


def test_review_queue_is_not_the_job_queue():
    statuses = {s.value for s in ReviewQueueStatus}
    assert not statuses & {"QUEUED", "RUNNING", "BLOCKED", "SUCCEEDED", "FAILED", "CANCELLED"}
    assert {(t.from_status, t.to_status) for t in REVIEW_QUEUE_TRANSITIONS} >= {(ReviewQueueStatus.PENDING, ReviewQueueStatus.IN_REVIEW)}
    import call1.store.results.review_queue as human_queue
    source = open(human_queue.__file__).read()
    assert "results_review_items" in source and "FROM jobs" not in source and "queue_jobs" not in source
