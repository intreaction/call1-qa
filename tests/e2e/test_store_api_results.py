"""Calls, projections, search, the human review queue, rubrics and metrics through the real Store
and Process servers (inventory q8-q12, q15, q16).

Tests that change stack-wide state (queue rules, rubric publication, scripted fake failures, metric
totals) run on a private stack; the rest use the shared stack and find their own calls by ID.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from call1.contracts.contents import ResultKind

from .store_api_support import error_code, poll, settled_call

pytestmark = pytest.mark.e2e

SECTION = {"transcript": "transcript", "qa": "evaluation", "summary": "summary", "contact_signals": "contact-signals"}
RUBRIC = "call1_standard_v2"


def _groups(session, call_id):
    detail = session.get(f"/calls/{call_id}")
    assert detail.status_code == 200, detail.text
    return {g["kind"]: g for g in detail.json()["results"]}, detail.json()


def _assert_sections_follow_state(session, call_id, groups):
    """A group that has a version (available, partial, stale) serves its section; a group with no
    version (pending, failed first run, disabled) answers 404 not_found, which is why Evaluate gates
    each section on ResultGroup.state instead of requesting it."""
    for kind, path in SECTION.items():
        group = groups[kind]
        response = session.get(f"/calls/{call_id}/{path}")
        if group["version"] is not None:
            assert response.status_code == 200, f"{kind} is {group['state']} v{group['version']} but its section answered {response.status_code}: {response.text}"
            body = response.json()
            if "version" in body:
                assert body["version"] == group["version"], (kind, body.get("version"), group)
        else:
            assert response.status_code == 404 and error_code(response) == "not_found", (kind, group, response.status_code, response.text)


# --- q8: calls and result projections -----------------------------------------------------------


def test_q8_result_groups_track_state_through_analysis_failure_partial_and_stale(stack_factory):
    behavior = {
        "summary_segment": ["hold:4", "hold:4"],                      # first run and the reanalysis both run slowly
        "acoustic_tone": ["fail:configuration_error"],                # tone fails outright: 'failed' (Needs attention)
        "contact_signals_lifecycle": ["fail:configuration_error"],   # one pass fails: contact signals 'partial'
    }
    private = stack_factory(name="states", fake_behavior=behavior)
    reviewer = private.user("reviewer")
    receipt = private.ingest("call_01_compliant", agent_id="agent-q8")
    call_id = receipt["call_id"]

    # While the summary is still analyzing the call is readable and the summary group is pending.
    groups, detail = poll(lambda: _groups(reviewer, call_id), lambda gd: gd[0]["transcript"]["state"] == "available",
                          timeout=20, what="transcript to publish")
    assert set(groups) == {k.value for k in ResultKind}, sorted(groups)
    assert groups["summary"]["state"] == "pending" and groups["summary"]["version"] is None, groups["summary"]
    assert detail["pending_work"]["settled"] is False, detail["pending_work"]
    _assert_sections_follow_state(reviewer, call_id, groups)

    private.wait_until_settled(call_id)
    groups, detail = _groups(reviewer, call_id)
    assert groups["tone"]["state"] == "failed" and groups["tone"]["failure_code"] == "configuration_error", groups["tone"]
    assert groups["contact_signals"]["state"] == "partial" and groups["contact_signals"]["partial_reason"], groups["contact_signals"]
    assert groups["summary"]["state"] == "available" and groups["summary"]["version"] == 1, groups["summary"]
    for kind in ("transcript", "qa", "text_sentiment"):
        assert groups[kind]["state"] == "available", groups[kind]
    assert detail["pending_work"]["settled"] is True
    _assert_sections_follow_state(reviewer, call_id, groups)
    assert reviewer.get(f"/calls/{call_id}/contact-signals").json()["completeness"] == "partial"

    # A summary reanalysis makes the summary 'stale' (the old version still readable) until it republishes.
    request = reviewer.post(f"/calls/{call_id}/reanalysis-requests", json={"kind": "summary"}, idempotency_key=True)
    assert request.status_code == 201, request.text
    groups, _ = _groups(reviewer, call_id)
    assert groups["summary"]["state"] == "stale" and groups["summary"]["version"] == 1, groups["summary"]
    assert groups["summary"]["reanalysis_request_id"] == request.json()["id"], groups["summary"]
    _assert_sections_follow_state(reviewer, call_id, groups)
    private.wait_until_settled(call_id)
    groups, _ = poll(lambda: _groups(reviewer, call_id), lambda gd: gd[0]["summary"]["state"] == "available", timeout=20,
                     what="summary reanalysis to publish")
    assert groups["summary"]["version"] == 2, groups["summary"]
    assert reviewer.get(f"/calls/{call_id}/summary").json()["version"] == 2

    # The list row carries the same per-section states.
    row = next(c for c in reviewer.get("/calls", params={"limit": 50}).json()["items"] if c["call_id"] == call_id)
    assert (row["transcript_state"], row["qa_state"], row["summary_state"]) == ("available", "available", "available"), row


def test_q8_disabled_stage_reports_not_run(stack_factory):
    private = stack_factory(name="disabled", process_config={"stages": {"summary": False}})
    reviewer = private.user("reviewer")
    receipt = settled_call(private, "call_01_compliant", agent_id="agent-q8-off")
    groups, _ = _groups(reviewer, receipt["call_id"])
    assert groups["summary"]["state"] == "disabled", groups["summary"]
    _assert_sections_follow_state(reviewer, receipt["call_id"], groups)


# --- q9: semantic search parity -----------------------------------------------------------------


def test_q9_semantic_search_matches_process_embeddings(stack, reviewer_session):
    """Contract 1.2.0: Process and Store embed with the same module (the fake embedder on this
    fake-handler stack), so the artifact's vectors are exactly what Store would compute."""
    from call1 import embedding

    receipt = settled_call(stack, "call_03_dispute_escalation", agent_id="agent-q9")
    call_id, conversation = receipt["call_id"], receipt["conversation_id"]

    # Process's embeddings artifact equals the shared embedder's vectors of the transcript, turn by turn.
    artifacts = stack.store_get(f"/conversations/{conversation}/artifacts", session="service", params={"limit": 200}).json()["items"]
    embeddings = next(a for a in artifacts if a["kind"] == "embeddings" and a["linked"])
    transcript_art = next(a for a in artifacts if a["kind"] == "transcript" and a["linked"])
    content = json.loads(stack.store_get(f"/artifacts/{embeddings['id']}/content", session="service").content)
    transcript = json.loads(stack.store_get(f"/artifacts/{transcript_art['id']}/content", session="service").content)
    fake = embedding.get_embedder("fake")
    assert content["scheme"] == fake.scheme == "fake-embedding-v1" and content["dimensions"] == fake.dimensions
    text = {t["turn_id"]: t["text"] for t in transcript["turns"]}
    for tv in content["turn_vectors"]:
        assert tv["vector"] == fake.embed_documents([text[tv["turn_id"]]])[0], f"turn {tv['turn_id']}: Process vector differs from the shared embedder's"

    # Searching a turn's own text through the API returns that turn first, with similarity ~1.
    view = reviewer_session.get(f"/calls/{call_id}/transcript").json()
    target = max(view["turns"], key=lambda t: len(t["text"]))
    response = reviewer_session.post("/search/semantic", json={"query": target["text"], "call_id": call_id, "top_k": 3})
    assert response.status_code == 200, response.text
    hits = response.json()["results"]
    assert hits and hits[0]["call_id"] == call_id and hits[0]["turn_id"] == target["turn_id"], hits
    assert hits[0]["similarity_score"] >= 0.99, hits[0]
    assert response.json()["count"] == len(hits) and response.json()["embedding_scheme"] == "fake-embedding-v1"
    assert response.json()["calls_needing_reembedding"] == 0

    # Across all calls, the speaker filter is honoured.
    agent_only = reviewer_session.post("/search/semantic", json={"query": "fee on my account", "speaker_filter": "AGENT", "top_k": 20}).json()
    assert all(h["speaker"] == "AGENT" for h in agent_only["results"]), agent_only


# --- q10: reviews, the review queue and its rules -----------------------------------------------


def _rule(rule_id, **over):
    rule = {"id": rule_id, "name": "E2E calibration", "stream": "CALIBRATION", "enabled": True, "rank": 5,
            "distribution_strategy": "UNASSIGNED_CLAIM", "target_skills": [], "critical_failure_only": False, "low_confidence_only": False,
            "sampling_rate": 0.0, "target_domains": [], "target_agents": [], "description": "every call, claim pool"}
    rule.update(over)
    return rule


def test_q10_review_queue_distribution_transitions_and_rules(stack_factory):
    private = stack_factory(name="reviewq")
    supervisor = private.user("supervisor")
    alice = private.user("reviewer", email="alice@e2e.test", display_name="Alice")
    bob = private.user("reviewer", email="bob@e2e.test", display_name="Bob")

    # Only Alice has the ESCALATIONS skill the dispute mandate rule (SKILL_MATCHED) targets.
    profile = supervisor.patch(f"/admin/reviewer-profiles/{alice.account_id}", json={"skills": ["ESCALATIONS"]})
    assert profile.status_code == 200 and profile.json()["skills"] == ["ESCALATIONS"], profile.text
    assert bob.patch(f"/admin/reviewer-profiles/{bob.account_id}", json={"skills": ["ESCALATIONS"]}).status_code == 403

    # Rule CRUD with rule_version optimistic concurrency; reviewers cannot manage rules.
    saved = supervisor.put("/review-queue/rules/rule-e2e-calibration", json={"rule": _rule("rule-e2e-calibration"), "expected_rule_version": 0})
    assert saved.status_code == 200 and saved.json()["rule_version"] == 1, saved.text
    stale = supervisor.put("/review-queue/rules/rule-e2e-calibration", json={"rule": _rule("rule-e2e-calibration"), "expected_rule_version": 0})
    assert stale.status_code == 409 and error_code(stale) == "conflict", (stale.status_code, stale.text)
    denied = bob.put("/review-queue/rules/rule-e2e-other", json={"rule": _rule("rule-e2e-other"), "expected_rule_version": 0})
    assert denied.status_code == 403, denied.text
    assert "rule-e2e-calibration" in {r["id"] for r in bob.get("/review-queue/rules").json()["items"]}

    receipt = settled_call(private, "call_03_dispute_escalation", agent_id="agent-q10")
    call_id = receipt["call_id"]
    items = [i for i in supervisor.get("/review-queue", params={"limit": 200}).json()["items"] if i["call_id"] == call_id]
    by_rule = {i["rule_id"]: i for i in items}
    mandate = by_rule.get("rule-mandate-dispute-escalation")
    calibration = by_rule.get("rule-e2e-calibration")
    assert mandate is not None and calibration is not None, sorted(by_rule)
    assert mandate["assigned_to_account_id"] == alice.account_id, f"SKILL_MATCHED should assign Alice: {mandate}"
    assert calibration["assigned_to_account_id"] is None and calibration["status"] == "PENDING", calibration

    # claim-next hands Bob the unassigned pool item, now IN_REVIEW and his.
    claimed = bob.post("/review-queue/claim-next").json()["item"]
    assert claimed is not None and claimed["id"] == calibration["id"], claimed
    assert claimed["status"] == "IN_REVIEW" and claimed["assigned_to_account_id"] == bob.account_id, claimed

    # Alice starts her assigned item; Bob cannot start it; stale item_version is a 409 conflict.
    assert bob.post(f"/review-queue/items/{mandate['id']}/start", json={"expected_item_version": mandate["item_version"]}).status_code == 403
    stale_start = alice.post(f"/review-queue/items/{mandate['id']}/start", json={"expected_item_version": mandate["item_version"] + 5})
    assert stale_start.status_code == 409 and error_code(stale_start) == "conflict", stale_start.text
    started = alice.post(f"/review-queue/items/{mandate['id']}/start", json={"expected_item_version": mandate["item_version"]})
    assert started.status_code == 200 and started.json()["status"] == "IN_REVIEW", started.text
    released = alice.post(f"/review-queue/items/{mandate['id']}/release", json={"expected_item_version": started.json()["item_version"], "note": "later"})
    assert released.status_code == 200 and released.json()["status"] == "PENDING" and released.json()["assigned_to_account_id"] is None, released.text

    # A supervisor assigns it back to Alice.
    assigned = supervisor.post(f"/review-queue/items/{mandate['id']}/assign",
                               json={"account_id": alice.account_id, "expected_item_version": released.json()["item_version"]})
    assert assigned.status_code == 200 and assigned.json()["assigned_to_account_id"] == alice.account_id, assigned.text

    # Resolve reads the call's review_version from GET /calls/{id}/review, not the item version.
    review = bob.get(f"/calls/{call_id}/review").json()
    body = {"status": "APPROVED", "reviewer_notes": "fine", "expected_item_version": claimed["item_version"],
            "expected_review_version": review["review_version"] + 1, "evaluation_version": claimed["evaluation_version"]}
    conflict = bob.post(f"/review-queue/items/{claimed['id']}/resolve", json=body)
    assert conflict.status_code == 409 and error_code(conflict) == "review_version_conflict", (conflict.status_code, conflict.text)
    body["expected_review_version"] = review["review_version"]
    resolved = bob.post(f"/review-queue/items/{claimed['id']}/resolve", json=body)
    assert resolved.status_code == 200 and resolved.json()["review_version"] == review["review_version"] + 1, resolved.text
    again = bob.post(f"/review-queue/items/{claimed['id']}/resolve", json=body)
    assert again.status_code == 409, again.text
    item = bob.get(f"/review-queue/items/{claimed['id']}").json()
    assert item["status"] == "APPROVED" and item["resolved_by_account_id"] == bob.account_id, item

    # A verdict override with a stale expected_version is refused the same way.
    evaluation = bob.get(f"/calls/{call_id}/evaluation").json()
    criterion = evaluation["verdicts"][0]["criterion_id"]
    stale_override = bob.post(f"/calls/{call_id}/verdicts/{criterion}",
                              json={"status": "FAIL", "expected_version": review["review_version"], "evaluation_version": claimed["evaluation_version"]})
    assert stale_override.status_code == 409 and error_code(stale_override) == "review_version_conflict", stale_override.text
    stats = supervisor.get("/review-queue/stats").json()
    assert stats["by_status"].get("APPROVED", 0) >= 1, stats

    # Delete the rule; a second delete is 404.
    assert supervisor.delete("/review-queue/rules/rule-e2e-calibration").status_code == 204
    assert supervisor.delete("/review-queue/rules/rule-e2e-calibration").status_code == 404


# --- q11: rubrics -------------------------------------------------------------------------------


def test_q11_rubric_draft_test_publish_and_retire(stack_factory):
    private = stack_factory(name="rubrics")
    supervisor = private.user("supervisor")
    reviewer = private.user("reviewer")
    receipt = settled_call(private, "call_01_compliant", agent_id="agent-q11")
    call_id = receipt["call_id"]

    current = supervisor.get(f"/rubrics/{RUBRIC}").json()
    assert current["ref"]["version"] == 1, current["ref"]
    assert supervisor.get(f"/rubrics/{RUBRIC}/draft").status_code == 404

    definition = dict(current["definition"], name=current["definition"]["name"] + " (e2e draft)")
    first = supervisor.put(f"/rubrics/{RUBRIC}/draft", json={"definition": definition, "expected_draft_revision": None})
    assert first.status_code == 200, first.text
    revision = first.json()["draft_revision"]
    assert first.json()["based_on_version"] == 1

    # Optimistic concurrency on the draft; reviewers cannot edit rubrics.
    stale = supervisor.put(f"/rubrics/{RUBRIC}/draft", json={"definition": definition, "expected_draft_revision": revision + 3})
    assert stale.status_code == 409 and error_code(stale) == "rubric_version_conflict", (stale.status_code, stale.text)
    assert reviewer.put(f"/rubrics/{RUBRIC}/draft", json={"definition": definition, "expected_draft_revision": revision}).status_code == 403
    definition["criteria"] = [dict(definition["criteria"][0], name=definition["criteria"][0]["name"] + " v2")] + definition["criteria"][1:]
    second = supervisor.put(f"/rubrics/{RUBRIC}/draft", json={"definition": definition, "expected_draft_revision": revision})
    assert second.status_code == 200 and second.json()["draft_revision"] > revision, second.text
    revision = second.json()["draft_revision"]

    # Draft test: a stale revision is refused; the real one reaches a terminal result without touching the call's QA.
    stale_test = supervisor.post(f"/rubrics/{RUBRIC}/draft/tests", json={"call_id": call_id, "expected_draft_revision": revision - 1}, idempotency_key=True)
    assert stale_test.status_code == 409 and error_code(stale_test) == "rubric_version_conflict", stale_test.text
    test = supervisor.post(f"/rubrics/{RUBRIC}/draft/tests", json={"call_id": call_id, "expected_draft_revision": revision}, idempotency_key=True)
    assert test.status_code == 201 and test.json()["kind"] == "qa_draft_test", test.text
    result = poll(lambda: supervisor.get(f"/reanalysis-requests/{test.json()['id']}/draft-result"),
                  lambda r: r.status_code == 200 and r.json()["state"] != "pending", timeout=30, what="draft test result")
    body = result.json()
    assert body["state"] == "available" and body["draft_revision"] == revision and body["scorecard"], body
    assert body["scorecard"]["rubric"]["draft_revision"] == revision, body["scorecard"]["rubric"]
    qa_group = next(g for g in supervisor.get(f"/calls/{call_id}").json()["results"] if g["kind"] == "qa")
    assert qa_group["version"] == 1 and qa_group["state"] == "available", qa_group

    # Publish creates version 2; version 1 is immutable.
    stale_publish = supervisor.post(f"/rubrics/{RUBRIC}/publish", json={"expected_current_version": 1, "expected_draft_revision": revision + 1})
    assert stale_publish.status_code == 409 and error_code(stale_publish) == "rubric_version_conflict", stale_publish.text
    published = supervisor.post(f"/rubrics/{RUBRIC}/publish", json={"expected_current_version": 1, "expected_draft_revision": revision, "notes": "e2e"})
    assert published.status_code == 201, published.text
    v2 = published.json()
    assert v2["ref"]["version"] == 2 and v2["ref"]["digest"] != current["ref"]["digest"] and v2["definition"]["name"] == definition["name"], v2["ref"]
    v1 = supervisor.get(f"/rubrics/{RUBRIC}/versions/1").json()
    assert v1["ref"] == current["ref"] and v1["definition"] == current["definition"], "version 1 changed after publish"
    versions = [v["ref"]["version"] for v in supervisor.get(f"/rubrics/{RUBRIC}/versions").json()["items"]]
    assert sorted(versions) == [1, 2], versions
    assert supervisor.get(f"/rubrics/{RUBRIC}").json()["ref"]["version"] == 2

    # Retire with a stale version is refused; with the current one it retires.
    assert supervisor.post(f"/rubrics/{RUBRIC}/retire", json={"expected_current_version": 1, "reason": "old"}).status_code == 409
    retired = supervisor.post(f"/rubrics/{RUBRIC}/retire", json={"expected_current_version": 2, "reason": "e2e retire"})
    assert retired.status_code == 200 and retired.json()["status"] == "retired", retired.text


# --- q12 and q15: metrics -----------------------------------------------------------------------


def test_q12_q15_metrics_ranges_and_hours_audited(stack_factory):
    private = stack_factory(name="metrics")
    supervisor = private.user("supervisor")
    reviewer = private.user("reviewer")
    before = datetime.now(timezone.utc) - timedelta(seconds=2)
    calls = [settled_call(private, sample, agent_id="agent-q12")["call_id"] for sample in ("call_01_compliant", "call_03_dispute_escalation")]
    after = datetime.now(timezone.utc) + timedelta(seconds=2)
    seconds = sum(reviewer.get(f"/calls/{c}").json()["call"]["duration_seconds"] for c in calls)
    assert seconds > 0

    executive = reviewer.get("/metrics/executive")
    assert executive.status_code == 200, executive.text
    body = executive.json()
    assert body["total_audited_calls"] == 2 and body["calls_pending_analysis"] == 0, body
    # q15: Hours audited is the sum of the audited calls' durations (not 0.0).
    assert body["total_hours_audited"] > 0, f"Hours audited is {body['total_hours_audited']} for {seconds:.1f}s of audited audio"
    assert abs(body["total_hours_audited"] - seconds / 3600.0) <= 0.005, (body["total_hours_audited"], seconds / 3600.0)

    in_range = reviewer.get("/metrics/executive", params={"start": before.isoformat(), "end": after.isoformat()}).json()
    assert in_range["total_audited_calls"] == 2 and abs(in_range["total_hours_audited"] - seconds / 3600.0) <= 0.005, in_range
    future = reviewer.get("/metrics/executive", params={"start": (after + timedelta(days=1)).isoformat()}).json()
    assert future["total_audited_calls"] == 0 and future["total_hours_audited"] == 0.0, future
    assert reviewer.get("/metrics/executive", params={"start": "not-a-date"}).status_code == 422

    rubric = reviewer.get(f"/metrics/rubrics/{RUBRIC}", params={"start": before.isoformat(), "end": after.isoformat()})
    assert rubric.status_code == 200 and rubric.json()["total_calls_evaluated"] == 2, rubric.text
    assert rubric.json()["criteria"], rubric.json()
    empty = reviewer.get(f"/metrics/rubrics/{RUBRIC}", params={"end": before.isoformat()}).json()
    assert empty["total_calls_evaluated"] == 0, empty
    assert reviewer.get("/metrics/rubrics/no-such-rubric").status_code == 404

    assert reviewer.get("/metrics/review-agreement").status_code == 403
    agreement = supervisor.get("/metrics/review-agreement", params={"rubric_id": RUBRIC, "start": before.isoformat()})
    assert agreement.status_code == 200, agreement.text
    assert {c["machine_decisions"] for c in agreement.json()["per_criterion"]} == {2}, agreement.json()


def test_q15_hours_audited_is_nonzero_for_a_single_short_audited_call(stack_factory):
    """Hours audited for one ~40 s audited call is ~0.011 h: the API value must be nonzero and exact
    to Store's 6 decimals (0.0036 s), so a short call is never rounded to 0.01 or 0.0 before it
    reaches the UI. (How Evaluate formats it is the Metrics UI's test.)"""
    private = stack_factory(name="hours")
    reviewer = private.user("reviewer")
    call_id = settled_call(private, "call_01_compliant", agent_id="agent-q15")["call_id"]
    seconds = reviewer.get(f"/calls/{call_id}").json()["call"]["duration_seconds"]
    hours = reviewer.get("/metrics/executive").json()["total_hours_audited"]
    assert hours > 0, f"Hours audited is {hours} for one audited call of {seconds:.1f}s"
    assert hours == round(seconds / 3600.0, 6), (hours, seconds / 3600.0)


# --- q16: agent name and extension on call rows -------------------------------------------------


def test_q16_agent_name_and_extension_reach_call_rows(stack, reviewer_session):
    agent = f"Samantha (104) {int(time.time() * 1000) % 100000}"
    receipt = settled_call(stack, "call_03_dispute_escalation", agent_id=agent, external_call_ref="pbx-q16")
    call_id = receipt["call_id"]

    rows = reviewer_session.get("/calls", params={"agent_id": agent, "limit": 50}).json()["items"]
    assert [r["call_id"] for r in rows] == [call_id], rows
    assert rows[0]["agent_id"] == agent
    assert reviewer_session.get(f"/calls/{call_id}").json()["call"]["agent_id"] == agent
    searched = reviewer_session.get("/calls", params={"text": "pbx-q16", "limit": 50}).json()["items"]
    assert call_id in {r["call_id"] for r in searched}, "text search covers the external call reference"
    items = [i for i in reviewer_session.get("/review-queue", params={"limit": 200}).json()["items"] if i["call_id"] == call_id]
    assert items and all(i["agent_id"] == agent for i in items), items

    # The Process CLI path (python -m call1.process ingest --agent-id) carries the agent too.
    from .stack import sample_path, unique_wav_copy

    copy = unique_wav_copy(sample_path("call_01_compliant"), stack.uploads_dir, 900001)
    cli = stack.process_cli("ingest", str(copy), "--agent-id", "Bob (202)", "--agent-channel", "0")
    cli_receipt = json.loads(cli.stdout)
    stack.wait_until_settled(cli_receipt["conversation_id"])
    cli_call = reviewer_session.get(f"/calls/{cli_receipt['call_id']}").json()["call"]
    assert cli_call["agent_id"] == "Bob (202)", cli_call


def _latest_cursor(session) -> str:
    return session.get("/changes", params={"limit": 1}).json()["latest_cursor"]


def _call_events(session, cursor: str, call_id: str):
    events = []
    while True:
        feed = session.get("/changes", params={"after": cursor, "kinds": ["call"], "limit": 1000}).json()
        events += [e for e in feed["events"] if e["call_id"] == call_id]
        if feed["next_cursor"] == cursor or not feed["events"]:
            return events
        cursor = feed["next_cursor"]


def _metadata_audits(admin, target_id: str):
    response = admin.get("/admin/audit", params={"action": "call_metadata_updated", "target_id": target_id, "limit": 50})
    assert response.status_code == 200, response.text
    return response.json()["items"]


def test_q16_reupload_with_agent_metadata_is_not_dropped(stack_factory):
    """Team decision 2 (2026-09-25, contract 1.1.0): re-uploading the same recording with different
    call metadata UPDATES the call's metadata. Process identifies a recording by its SHA-256, so the
    re-upload returns the existing conversation; Store merges the metadata (only the fields sent
    replace stored ones), writes a call_metadata_updated audit event (field names, never values) and
    a call change event with status metadata_updated, and reprocesses nothing."""
    private = stack_factory(name="agent")
    reviewer = private.user("reviewer")
    admin = private.admin()
    first = settled_call(private, "call_02_critical_breach", unique=False)
    call_id = first["call_id"]
    before = reviewer.get(f"/calls/{call_id}").json()
    assert before["call"]["agent_id"] == "Unknown", before["call"]
    jobs_before = private.store_get("/jobs", session="service", params={"conversation_id": first["conversation_id"], "limit": 200}).json()["items"]
    cursor = _latest_cursor(reviewer)

    second = private.ingest("call_02_critical_breach", unique=False, agent_id="Bob (202)")
    assert second["call_id"] == call_id and second["conversation_created"] is False, second
    assert second["graph_id"] == first["graph_id"] and second["graph_created"] is False, second
    if "metadata_updated" in second:  # Process's receipt reports it once Process carries the 1.1.0 fields
        assert second["metadata_updated"] is True and "agent_id" in second["updated_fields"], second
    after = reviewer.get(f"/calls/{call_id}").json()
    assert after["call"]["agent_id"] == "Bob (202)", (
        f"re-uploading the recording with agent 'Bob (202)' left the call as {after['call']['agent_id']!r}")
    rows = reviewer.get("/calls", params={"agent_id": "Bob (202)", "limit": 50}).json()["items"]
    assert [r["call_id"] for r in rows] == [call_id], rows
    items = reviewer.get("/review-queue", params={"call_id": call_id, "limit": 200}).json()["items"]
    assert all(i["agent_id"] == "Bob (202)" for i in items), items

    # Nothing reprocessed: the same jobs in the same states, the same results and review version.
    jobs_after = private.store_get("/jobs", session="service", params={"conversation_id": first["conversation_id"], "limit": 200}).json()["items"]
    assert [(j["id"], j["status"]) for j in jobs_after] == [(j["id"], j["status"]) for j in jobs_before]
    assert after["results"] == before["results"] and after["review_version"] == before["review_version"]
    assert after["evaluation"]["version"] == before["evaluation"]["version"]

    # One change event and one audit event, naming fields only.
    events = _call_events(reviewer, cursor, call_id)
    assert [e["status"] for e in events] == ["metadata_updated"], events
    audits = _metadata_audits(admin, call_id)
    assert len(audits) == 1, audits
    assert audits[0]["actor"]["kind"] == "process_service" and audits[0]["actor"]["installation_id"] == private.installation_id, audits[0]
    assert audits[0]["target"] == {"kind": "call", "id": call_id}
    assert audits[0]["details"]["conversation_id"] == first["conversation_id"] and "agent_id" in audits[0]["details"]["updated_fields"].split(",")
    assert "Bob" not in str(audits[0]["details"])

    # The same metadata again is a pure replay: no new audit event, the agent stays.
    private.ingest("call_02_critical_breach", unique=False, agent_id="Bob (202)")
    assert reviewer.get(f"/calls/{call_id}").json()["call"]["agent_id"] == "Bob (202)"
    assert len(_metadata_audits(admin, call_id)) == 1


def test_q16_display_name_and_extension_reach_rows_detail_and_search(stack, reviewer_session, admin_session):
    """Contract 1.1.0 agent identity, registered straight on Store with the service key (the Process
    upload form's fields are Process's test): the list row, call detail and text search carry
    agent_display_name and agent_extension, and a re-registration that changes only the extension
    updates just that field."""
    import hashlib

    from call1.contracts.calls import agent_label

    tag = f"{time.time_ns()}"
    source = {"kind": "api_upload", "content_digest": "sha256:" + hashlib.sha256(tag.encode()).hexdigest(),
              "received_at": datetime.now(timezone.utc).isoformat()}
    meta = {"agent_id": f"agt-{tag}", "agent_display_name": f"Samantha {tag}", "agent_extension": "104"}
    created = stack.store_post("/conversations", {"ingestion_kind": "call_audio", "source": source, "call_metadata": meta}, session="service")
    assert created.status_code == 200 and created.json()["created"] is True, created.text
    call_id = created.json()["conversation"]["call_id"]

    rows = reviewer_session.get("/calls", params={"agent_id": meta["agent_id"], "limit": 50}).json()["items"]
    assert [(r["call_id"], r["agent_display_name"], r["agent_extension"]) for r in rows] == [(call_id, meta["agent_display_name"], "104")], rows
    assert agent_label(rows[0]["agent_id"], rows[0]["agent_display_name"], rows[0]["agent_extension"]) == f"Samantha {tag} (104)"
    detail = reviewer_session.get(f"/calls/{call_id}").json()["call"]
    assert (detail["agent_display_name"], detail["agent_extension"]) == (meta["agent_display_name"], "104")
    searched = reviewer_session.get("/calls", params={"text": f"Samantha {tag}", "limit": 50}).json()["items"]
    assert [r["call_id"] for r in searched] == [call_id], "text search covers the display name"

    again = stack.store_post("/conversations", {"ingestion_kind": "call_audio", "source": source, "call_metadata": {"agent_extension": "105"}},
                             session="service")
    assert again.status_code == 200, again.text
    assert (again.json()["created"], again.json()["metadata_updated"], again.json()["updated_fields"]) == (False, True, ["agent_extension"])
    detail = reviewer_session.get(f"/calls/{call_id}").json()["call"]
    assert (detail["agent_id"], detail["agent_display_name"], detail["agent_extension"]) == (meta["agent_id"], meta["agent_display_name"], "105")
    audits = _metadata_audits(admin_session, call_id)
    assert [a["details"]["updated_fields"] for a in audits] == ["agent_extension"], audits
