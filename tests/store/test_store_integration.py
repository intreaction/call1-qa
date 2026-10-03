"""Cross-area flows through the real queue, results and auth areas together: no monkeypatched
hooks, rubric lookups or accounts. Process is played by ``QueueHarness`` over HTTP with a minted
service key, Evaluate by minted reviewer sessions.

* a rubric draft test: saveRubricDraft -> testRubricDraft -> claimReanalysisRequests -> draft
  graph -> completion in the draft slot -> getDraftTestResult, with the call's results untouched;
* a speaker correction: ASR publishes a transcript -> correctSpeaker -> the reanalysis request ->
  a speaker_attribution graph that fulfils it -> the new attribution on the transcript view;
* review-queue distribution across real reviewer accounts and their profiles.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.contents import QaScorecardContent
from call1.contracts.jobs import JobType

from .test_queue_harness import QueueHarness, content, job, pinned
from .test_results_harness import qa_scorecard, transcript

SEED = "call1_standard_v2"


@pytest.fixture
def real(client, store, clock, service_key, mint_service_key, mint_session) -> QueueHarness:
    """The queue harness without its fakes (no ``hooks`` or ``rubrics`` fixtures)."""
    return QueueHarness(client, store, clock, service_key, mint_service_key, mint_session)


def _claim_requests(real: QueueHarness):
    return real.post("/reanalysis-requests/claim", {"worker_id": "w1", "max_requests": 4}).json()["requests"]


def _scorecard(rubric: Dict[str, Any], **overrides) -> Dict[str, Any]:
    card = qa_scorecard([("REG-01", "PASS", 0.97)], **overrides).model_dump(mode="json")
    card["rubric"] = rubric
    return QaScorecardContent.model_validate(card).model_dump(mode="json")  # canonical: every default present


# --- rubric draft test --------------------------------------------------------------------------


def test_rubric_draft_test_end_to_end_leaves_the_call_untouched(real, client, supervisor_session):
    conversation = real.register()
    call_id = conversation["call_id"]
    current = client.get(f"/store/v1/rubrics/{SEED}", headers=supervisor_session.read_headers).json()
    definition = dict(current["definition"], name="Standard (draft)")
    draft = client.put(f"/store/v1/rubrics/{SEED}/draft", json={"definition": definition, "expected_draft_revision": 0},
                       headers=supervisor_session.headers)
    assert draft.status_code == 200, draft.text
    assert draft.json()["draft_revision"] == 1 and draft.json()["based_on_version"] == 1

    created = client.post(f"/store/v1/rubrics/{SEED}/draft/tests", json={"call_id": call_id, "expected_draft_revision": 1},
                          headers={**supervisor_session.headers, "Idempotency-Key": "studio-test-0001"})
    assert created.status_code == 201, created.text
    request = created.json()
    draft_ref = request["draft_rubric"]
    assert request["kind"] == "qa_draft_test" and draft_ref["rubric_id"] == SEED and draft_ref["draft_revision"] == 1
    snapshot = real.get(f"/artifacts/{draft_ref['snapshot_artifact_id']}").json()
    assert snapshot["kind"] == "rubric_snapshot" and snapshot["slot"].startswith(f"draft:{request['id']}:")

    [claimed] = _claim_requests(real)
    assert claimed["request"]["id"] == request["id"] and claimed["request"]["draft_rubric"] == draft_ref
    graph = real.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, key="studio-score-1", inputs=[pinned("rubric", snapshot)],
                                                parameters={"draft_rubric": draft_ref})],
                       key="studio-graph-1", reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"])
    assert real.get(f"/reanalysis-requests/{request['id']}").json()["status"] == "fulfilled"
    pending = real.get(f"/reanalysis-requests/{request['id']}/draft-result", headers=supervisor_session.read_headers).json()
    assert pending["state"] == "pending"

    score = real.claim_one(real.ids(graph)["score"])
    slot = f"draft:{request['id']}:"
    # A scorecard that names the published version instead of the draft under test is refused.
    wrong = _scorecard({"rubric_id": SEED, "rubric_version": 1, "digest": current["ref"]["digest"]})
    wrong_art = real.inline(conversation["id"], ArtifactKind.QA_SCORECARD, wrong, slot=slot, job_id=score["job"]["id"], token=score["claim_token"]).json()
    refused = real.post(f"/jobs/{score['job']['id']}/complete",
                        real.completion(score, [{"role": "scorecard", "artifact_id": wrong_art["id"], "checksum": wrong_art["checksum"]}], result=None),
                        expect=422).json()
    assert refused["code"] == "graph_invalid" and refused["details"]["reason"] == "scorecard_rubric_mismatch"

    right = _scorecard({"rubric_id": SEED, "draft_revision": 1, "digest": draft_ref["digest"]}, overall_score=88.0)
    right_art = real.inline(conversation["id"], ArtifactKind.QA_SCORECARD, right, slot=slot, job_id=score["job"]["id"], token=score["claim_token"]).json()
    receipt = real.complete(score, outputs=[{"role": "scorecard", "artifact_id": right_art["id"], "checksum": right_art["checksum"]}], result=None)
    assert receipt["result_version"] is None

    result = real.get(f"/reanalysis-requests/{request['id']}/draft-result", headers=supervisor_session.read_headers).json()
    assert result["state"] == "available" and result["draft_revision"] == 1 and result["scorecard"]["overall_score"] == 88.0
    assert result["scorecard"]["rubric"] == {"rubric_id": SEED, "rubric_version": None, "draft_revision": 1, "digest": draft_ref["digest"]}
    detail = client.get(f"/store/v1/calls/{call_id}", headers=supervisor_session.read_headers).json()
    assert detail["evaluation"] is None and {g["kind"]: g["state"] for g in detail["results"]}["qa"] == "disabled"
    assert client.get("/store/v1/review-queue", params={"call_id": call_id}, headers=supervisor_session.read_headers).json()["items"] == []


def test_a_live_scorecard_must_name_the_pinned_published_version(real, client, reviewer_session):
    conversation = real.register()
    snapshot = real.post(f"/conversations/{conversation['id']}/rubric-snapshots", {"rubric_id": SEED, "version": 1}, expect=201).json()
    ref = client.get(f"/store/v1/rubrics/{SEED}", headers=real.headers).json()["ref"]
    graph = real.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, key="live-score-1", inputs=[pinned("rubric", snapshot)],
                                                parameters={"rubric": ref})])
    score = real.claim_one(real.ids(graph)["score"])

    def attempt(rubric):
        art = real.inline(conversation["id"], ArtifactKind.QA_SCORECARD, _scorecard(rubric), job_id=score["job"]["id"], token=score["claim_token"]).json()
        return [{"role": "scorecard", "artifact_id": art["id"], "checksum": art["checksum"]}]

    for bad in ({"rubric_id": SEED, "rubric_version": 2, "digest": ref["digest"]},
                {"rubric_id": SEED, "rubric_version": 1, "digest": "sha256:" + "0" * 64},
                {"rubric_id": "someone_else", "rubric_version": 1, "digest": ref["digest"]}):
        refused = real.post(f"/jobs/{score['job']['id']}/complete", real.completion(score, attempt(bad)), expect=422).json()
        assert refused["details"]["reason"] == "scorecard_rubric_mismatch", bad
    receipt = real.complete(score, outputs=attempt({"rubric_id": SEED, "rubric_version": 1, "digest": ref["digest"]}))
    assert receipt["result_version"] == 1
    evaluation = client.get(f"/store/v1/calls/{conversation['call_id']}", headers=reviewer_session.read_headers).json()["evaluation"]
    assert evaluation["rubric"]["rubric_id"] == SEED and evaluation["rubric"]["rubric_version"] == 1
    listed = client.get("/store/v1/calls", params={"rubric_id": SEED}, headers=reviewer_session.read_headers).json()["items"]
    assert [i["call_id"] for i in listed] == [conversation["call_id"]]


# --- speaker correction --------------------------------------------------------------------------


def test_speaker_correction_flows_to_a_graph_and_a_new_attribution(real, client, reviewer_session):
    conversation = real.register()
    call_id = conversation["call_id"]
    audio = real.upload_audio(conversation["id"])
    graph = real.graph(conversation["id"], [job("asr", JobType.ASR, inputs=[pinned("audio", audio)])])
    said = transcript([("AGENT", "Thanks for calling."), ("AGENT", "Hi, I am the customer.")]).model_dump(mode="json")
    asr =real.claim_one(real.ids(graph)["asr"])
    real.complete(asr, outputs=real.outputs_for(asr, payloads={"transcript": said}))
    view = client.get(f"/store/v1/calls/{call_id}/transcript", headers=reviewer_session.read_headers)
    assert view.status_code == 200, view.text
    assert [t["speaker"] for t in view.json()["turns"]] == ["AGENT", "AGENT"]

    corrected = client.post(f"/store/v1/calls/{call_id}/speaker-corrections",
                            json={"correction": {"turn_id": 1, "speaker": "CALLER", "notes": "wrong channel"}, "expected_version": 0},
                            headers=reviewer_session.headers)
    assert corrected.status_code == 201, corrected.text
    request = corrected.json()
    assert request["kind"] == "speaker_correction" and request["status"] == "pending"

    [claimed] = _claim_requests(real)
    assert claimed["request"]["id"] == request["id"] and claimed["request"]["speaker_correction"]["turn_id"] == 1
    transcript_art = next(a for a in real.get(f"/conversations/{conversation['id']}/artifacts").json()["items"] if a["kind"] == "transcript")
    fix = real.graph(conversation["id"], [job("speakers", JobType.SPEAKER_ATTRIBUTION, key="speaker-fix-1", inputs=[pinned("transcript", transcript_art)],
                                              parameters={"speaker_correction": claimed["request"]["speaker_correction"]})],
                     key="speaker-graph-1", reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"])
    assert real.get(f"/reanalysis-requests/{request['id']}").json()["status"] == "fulfilled"
    attribution = content(ArtifactKind.SPEAKER_ATTRIBUTION, method="reviewer_correction",
                          assignments=[{"turn_id": 0, "speaker": "AGENT"}, {"turn_id": 1, "speaker": "CALLER"}])
    speakers = real.claim_one(real.ids(fix)["speakers"])
    assert speakers["job"]["selection"] is None  # a correction runs no model
    real.complete(speakers, outputs=real.outputs_for(speakers, payloads={"speaker_attribution": attribution}))

    after = client.get(f"/store/v1/calls/{call_id}/transcript", headers=reviewer_session.read_headers).json()
    assert [t["speaker"] for t in after["turns"]] == ["AGENT", "CALLER"]
    assert after["speaker_attribution_version"] == 1


# --- review-queue distribution with real accounts -------------------------------------------------


def test_review_items_are_distributed_across_real_reviewer_accounts(real, client, mint_session, supervisor_session, admin_session):
    ann = mint_session("reviewer", display_name="Ann")
    ben = mint_session("reviewer", display_name="Ben")
    # Only Ann and Ben take assignments; the supervisor and admin sessions opt out through the real profile route.
    for other in (supervisor_session, admin_session):
        opted = client.patch(f"/store/v1/admin/reviewer-profiles/{other.account_id}", json={"accepting_assignments": False},
                             headers=supervisor_session.headers)
        assert opted.status_code == 200, opted.text
    ref = client.get(f"/store/v1/rubrics/{SEED}", headers=real.headers).json()["ref"]
    assigned = []
    for i in range(4):
        conversation = real.register()
        snapshot = real.post(f"/conversations/{conversation['id']}/rubric-snapshots", {"rubric_id": SEED, "version": 1}, expect=201).json()
        graph = real.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, key=f"dist-score-{i}", inputs=[pinned("rubric", snapshot)],
                                                    parameters={"rubric": ref})])
        score = real.claim_one(real.ids(graph)["score"])
        card = qa_scorecard([("REG-01", "FAIL", 0.9)], rubric_id=SEED, digest=ref["digest"], overall_score=40, passed=False,
                            critical_failure=True).model_dump(mode="json")
        real.complete(score, outputs=real.outputs_for(score, payloads={"scorecard": card}))
        [item] = client.get("/store/v1/review-queue", params={"call_id": conversation["call_id"]}, headers=ann.read_headers).json()["items"]
        assert item["rule_id"] == "rule-triage-critical-lowconf"
        assigned.append(item["assigned_to_account_id"])
    # LEAST_OUTSTANDING alternates between the two accepting reviewers
    assert sorted(assigned) == sorted([ann.account_id, ben.account_id] * 2)
    profiles = {p["account_id"]: p for p in client.get("/store/v1/admin/reviewer-profiles", headers=supervisor_session.read_headers).json()["items"]}
    assert profiles[ann.account_id]["pending_count"] == 2 and profiles[ben.account_id]["pending_count"] == 2
    assert profiles[supervisor_session.account_id]["pending_count"] == 0
