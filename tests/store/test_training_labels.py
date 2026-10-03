"""The on-device training label log (contract 1.3.0, team decision 28; docs/OnDeviceTraining.md
sections 2 and 7.3, Store checks S1-S8), through the real queue, results and auth areas.

Process is played by ``QueueHarness`` over HTTP with a minted service key, Evaluate by minted
reviewer sessions. One call carries every label source: an ASR transcript, a model (diarization)
speaker attribution, an enrichment job's PII findings, a QA graph with a primary criterion job, an
escalation and the scorecard, and a v2 contact-signals merge with a multi-segment hit.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.common import ServiceScope
from call1.contracts.contents import QaScorecardContent, SignalHitPart
from call1.contracts.jobs import JobType
from call1.contracts.training import TrainingLabel, TrainingLabelPage, TrainingLabelKind
from call1.store import db
from call1.store.results import training_labels

from .test_queue_harness import QueueHarness, content, job, pinned, upstream_input
from .test_results_harness import qa_scorecard, transcript
from .test_signals_harness import TAXONOMY, V, hit, put_taxonomy, v2_content

SEED = "call1_standard_v2"
CRITERION = "REG-01"
LABELS = "/training/labels"
CALLER_SECRET = "my card is 4111 1111 1111 1111 and I am Jane Roe"
AGENT_LINE = "Thanks for calling Acme, this is Dana."
QUOTE = "I want to cancel my account"
REVIEWER_NOTE = "reviewer note: the agent did confirm it"
SIGNAL_NOTE = "signal note: caller clearly wants out"
CORRECTION_NOTE = "correction note: S2 is the caller"


@pytest.fixture
def real(client, store, clock, service_key, mint_service_key, mint_session) -> QueueHarness:
    return QueueHarness(client, store, clock, service_key, mint_service_key, mint_session)


def _labels(real: QueueHarness, *, expect: int = 200, headers=None, **params) -> Dict[str, Any]:
    response = real.client.get(V + LABELS, params=params, headers=headers if headers is not None else real.headers)
    assert response.status_code == expect, response.text
    return response.json()


def _rows(store) -> List[Dict[str, Any]]:
    with store.connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM results_training_labels ORDER BY seq").fetchall()]


def _audits(store, action: str = "training_labels_read") -> List[Dict[str, Any]]:
    with store.connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_events WHERE action = ? ORDER BY sequence", (action,)).fetchall()]


class Call:
    """One call with every label source, built through the real areas."""

    def __init__(self, real: QueueHarness, client, admin_session) -> None:
        self.real = real
        conversation = real.register()
        self.conversation_id, self.call_id = conversation["id"], conversation["call_id"]
        audio = real.upload_audio(self.conversation_id)
        ingest = real.graph(self.conversation_id, [
            job("asr", JobType.ASR, inputs=[pinned("audio", audio)], priority=5),
            job("speaker", JobType.SPEAKER_ATTRIBUTION, requires=["asr"],
                inputs=[pinned("audio", audio), upstream_input("transcript", "asr", "transcript")]),
            job("enrich", JobType.ENRICHMENT, requires=["asr"], inputs=[upstream_input("transcript", "asr", "transcript")]),
        ])
        ids = real.ids(ingest)
        said = transcript([("AGENT", AGENT_LINE), ("CALLER", CALLER_SECRET), ("CALLER", QUOTE)]).model_dump(mode="json")
        asr = real.claim_one(ids["asr"])
        real.complete(asr, outputs=real.outputs_for(asr, payloads={"transcript": said}))
        self.speaker_job_id = ids["speaker"]
        speaker = real.claim_one(ids["speaker"])
        attribution = content(ArtifactKind.SPEAKER_ATTRIBUTION, method="diarization", assignments=[
            {"turn_id": 0, "speaker": "AGENT", "speaker_cluster": "S1", "confidence": 0.8},
            {"turn_id": 1, "speaker": "AGENT", "speaker_cluster": "S2", "confidence": 0.8},
            {"turn_id": 2, "speaker": "AGENT", "speaker_cluster": "S2", "confidence": 0.8}])
        real.complete(speaker, outputs=real.outputs_for(speaker, payloads={"speaker_attribution": attribution}))
        self.enrich_job_id = ids["enrich"]
        real.run(ids["enrich"])
        arts = self.artifacts()
        self.transcript = arts["transcript"][-1]
        self.attribution = arts["speaker_attribution"][-1]
        self.enrichment = arts["enrichment"][-1]
        self.pii = arts["pii_findings"][-1]
        assert self.pii["producing_job_id"] == ids["enrich"]
        # QA: a primary criterion job, an escalation, and the scorecard that pins both assessments.
        self.snapshot = real.post(f"/conversations/{self.conversation_id}/rubric-snapshots", {"rubric_id": SEED, "version": 1}, expect=201).json()
        self.rubric_ref = client.get(f"{V}/rubrics/{SEED}", headers=real.headers).json()["ref"]
        qa_inputs = [pinned("transcript", self.transcript), pinned("speaker_attribution", self.attribution), pinned("enrichment", self.enrichment),
                     pinned("pii_findings", self.pii), pinned("rubric", self.snapshot)]
        params = {"rubric": self.rubric_ref, "criterion_id": CRITERION}
        qa = real.graph(self.conversation_id, [
            job("crit", JobType.QA_CRITERION, inputs=qa_inputs, parameters=params),
            job("esc", JobType.QA_ESCALATION, requires=["crit"], inputs=qa_inputs, parameters={**params, "escalation_trigger": "low_confidence"}),
            job("score", JobType.QA_SCORECARD, requires=["crit", "esc"], parameters={"rubric": self.rubric_ref}, slot="cpu",
                inputs=[pinned("rubric", self.snapshot), upstream_input(f"assessment:{CRITERION}", "crit", "assessment"),
                        upstream_input(f"escalation:{CRITERION}", "esc", "assessment")]),
        ])
        qa_ids = real.ids(qa)
        self.criterion_job_id, self.escalation_job_id = qa_ids["crit"], qa_ids["esc"]
        crit = real.claim_one(qa_ids["crit"])
        real.complete(crit, outputs=real.outputs_for(crit, slots={"assessment": CRITERION}, payloads={
            "assessment": content(ArtifactKind.QA_ASSESSMENT, criterion_id=CRITERION, reasoning=f"The agent said {AGENT_LINE}")}))
        esc = real.claim_one(qa_ids["esc"])
        real.complete(esc, outputs=real.outputs_for(esc, slots={"assessment": f"{CRITERION}:escalation"}, payloads={
            "assessment": content(ArtifactKind.QA_ASSESSMENT, criterion_id=CRITERION, assessment_kind="escalation", reasoning="Escalated.")}))
        score = real.claim_one(qa_ids["score"], slots=[{"memory_slot": "cpu", "count": 4}])
        card = qa_scorecard([(CRITERION, "PASS", 0.97)], evidence=AGENT_LINE).model_dump(mode="json")
        card["rubric"] = {"rubric_id": SEED, "rubric_version": 1, "digest": self.rubric_ref["digest"]}
        card = QaScorecardContent.model_validate(card).model_dump(mode="json")
        real.complete(score, outputs=real.outputs_for(score, payloads={"scorecard": card}))
        self.scorecard_job_id = qa_ids["score"]
        self.scorecard = self.artifacts()["qa_scorecard"][-1]
        # Contact Signals v2: one merged hit (anchor turn 1, part turn 2) published by a merge job.
        record = put_taxonomy(client, admin_session, TAXONOMY)
        self.taxonomy_version = record["current"]["version"]
        self.taxonomy_snapshot = real.post(f"/conversations/{self.conversation_id}/signal-taxonomy-snapshots", {"version": self.taxonomy_version},
                                           expect=201).json()
        anchor = hit(TAXONOMY, "intent", 1, sub="cancel_account", quote=QUOTE)
        anchor = anchor.model_copy(update={"parts": [SignalHitPart(turn_id=2, block=0, start=4.0, end=6.0, quote=QUOTE, char_start=0, char_end=len(QUOTE))],
                                           "span_end": 6.0})
        self.hit_id = anchor.id
        signals = v2_content(TAXONOMY, self.taxonomy_version, [anchor]).model_dump(mode="json")
        merge_graph = real.graph(self.conversation_id, [job("merge", JobType.CONTACT_SIGNALS_MERGE, slot="cpu", inputs=[
            pinned("transcript", self.transcript), pinned("speaker_attribution", self.attribution), pinned("pii_findings", self.pii),
            pinned("taxonomy", self.taxonomy_snapshot)])])
        self.merge_job_id = real.ids(merge_graph)["merge"]
        merge = real.claim_one(self.merge_job_id, slots=[{"memory_slot": "cpu", "count": 4}])
        real.complete(merge, outputs=real.outputs_for(merge, payloads={"contact_signals": signals}))
        self.contact_signals = self.artifacts()["contact_signals"][-1]

    def artifacts(self) -> Dict[str, List[Dict[str, Any]]]:
        items = self.real.get(f"/conversations/{self.conversation_id}/artifacts", params={"limit": 200}).json()["items"]
        out: Dict[str, List[Dict[str, Any]]] = {}
        for art in sorted(items, key=lambda a: (a["kind"], a["version"] or 0)):
            out.setdefault(art["kind"], []).append(art)
        return out

    # reviewer writes -----------------------------------------------------------------------------

    def review_version(self, session) -> int:
        return self.real.client.get(f"{V}/calls/{self.call_id}/review", headers=session.read_headers).json()["review_version"]

    def override(self, session, status: str = "FAIL", *, reason: str = "model_misread_evidence", expect: int = 200, expected_version=None):
        body = {"status": status, "reason_code": reason, "reviewer_notes": REVIEWER_NOTE, "evaluation_version": 1,
                "expected_version": self.review_version(session) if expected_version is None else expected_version}
        response = self.real.client.post(f"{V}/calls/{self.call_id}/verdicts/{CRITERION}", json=body, headers=session.headers)
        assert response.status_code == expect, response.text
        return response.json()

    def feedback(self, session, *, expected: int, expect: int = 200, **verdicts):
        body = {"expected_feedback_version": expected, "note": SIGNAL_NOTE, **verdicts}
        response = self.real.client.put(f"{V}/calls/{self.call_id}/signal-hits/{self.hit_id}/feedback", json=body, headers=session.headers)
        assert response.status_code == expect, response.text
        return response.json()

    def correct(self, session, turn_id: int = 1, speaker: str = "CALLER", *, apply_to_cluster: bool = True, expect: int = 201):
        body = {"correction": {"turn_id": turn_id, "speaker": speaker, "apply_to_cluster": apply_to_cluster, "notes": CORRECTION_NOTE},
                "expected_version": self.review_version(session)}
        response = self.real.client.post(f"{V}/calls/{self.call_id}/speaker-corrections", json=body, headers=session.headers)
        assert response.status_code == expect, response.text
        return response.json()


@pytest.fixture
def call(real, client, admin_session) -> Call:
    return Call(real, client, admin_session)


def _roles(item: Dict[str, Any]) -> Dict[str, str]:
    return {s["role"]: s["artifact_id"] for s in item["sources"]}


# --- S1: one row per label write, in its transaction ------------------------------------------------


def test_s1_each_label_write_appends_one_row_and_clearing_feedback_appends_withdrawn(call, real, store, reviewer_session):
    assert _rows(store) == []
    call.override(reviewer_session, "FAIL")
    saved = call.feedback(reviewer_session, expected=0, category_verdict="confirmed", subcategory_verdict="corrected",
                          corrected_subcategory_id="billing_question")
    request = call.correct(reviewer_session)
    call.feedback(reviewer_session, expected=saved["feedback_version"])  # clears both verdicts
    rows = _rows(store)
    assert [(r["kind"], r["withdrawn"]) for r in rows] == [("qa_verdict", 0), ("signal_hit", 0), ("speaker_role", 0), ("signal_hit", 1)]
    assert [r["seq"] for r in rows] == sorted(r["seq"] for r in rows)
    qa, signal, speaker, cleared = rows
    assert qa["subject"] == f"qa:{call.call_id}:{CRITERION}" and qa["source_artifact_id"] == call.scorecard["id"]
    assert json.loads(qa["label_json"])["original_status"] == "PASS" and json.loads(qa["label_json"])["status"] == "FAIL"
    assert signal["subject"] == f"signal:{call.call_id}:{call.hit_id}" == cleared["subject"]
    assert signal["source_artifact_id"] == call.contact_signals["id"]
    label = json.loads(signal["label_json"])
    assert label["spans"] == [{"turn_id": 1, "block": 0}, {"turn_id": 2, "block": 0}]
    assert label["subcategory_id"] == "cancel_account" and label["corrected_subcategory_id"] == "billing_question" and label["feedback_version"] == 1
    assert json.loads(cleared["label_json"])["category_verdict"] is None and json.loads(cleared["label_json"])["feedback_version"] == 2
    # The speaker row names the model's cluster for the turn, and the subject is the cluster.
    assert speaker["subject"] == f"speaker:{call.call_id}:S2" and speaker["source_artifact_id"] == call.attribution["id"]
    assert json.loads(speaker["label_json"]) == {"turn_id": 1, "speaker": "CALLER", "apply_to_cluster": True, "speaker_cluster": "S2",
                                                 "reanalysis_request_id": request["id"]}


def test_s1_a_failed_write_appends_nothing(call, real, store, reviewer_session, monkeypatch):
    call.override(reviewer_session, expected_version=99, expect=409)
    call.feedback(reviewer_session, expected=3, expect=409, category_verdict="dismissed")
    assert _rows(store) == []
    # A failure after the label row was appended rolls the row back with the reviewer's write.
    from call1.contracts.errors import ErrorCode
    from call1.store.errors import StoreError
    from call1.store.results import review_state

    def boom(*args, **kwargs):
        raise StoreError(ErrorCode.CONFLICT, "injected failure after the label row")

    monkeypatch.setattr(review_state, "write_result", boom)
    call.override(reviewer_session, expect=409)
    monkeypatch.undo()
    assert _rows(store) == []
    call.override(reviewer_session)
    assert [r["kind"] for r in _rows(store)] == ["qa_verdict"]


def test_s1_a_turn_only_correction_keys_on_the_turn(call, store, reviewer_session):
    call.correct(reviewer_session, turn_id=2, speaker="UNKNOWN", apply_to_cluster=False)
    [row] = _rows(store)
    assert row["subject"] == f"speaker:{call.call_id}:t2"
    assert json.loads(row["label_json"])["speaker_cluster"] == "S2" and json.loads(row["label_json"])["apply_to_cluster"] is False


# --- S2: the backfill ----------------------------------------------------------------------------------


def test_s2_the_backfill_imports_earlier_labels_in_order_once(call, real, store, reviewer_session, clock):
    call.override(reviewer_session, "FAIL")
    clock.advance(1)
    first = call.feedback(reviewer_session, expected=0, category_verdict="dismissed")
    clock.advance(1)
    call.correct(reviewer_session)
    clock.advance(1)
    call.override(reviewer_session, "NOT_APPLICABLE", reason="criterion_not_applicable")
    clock.advance(1)
    call.feedback(reviewer_session, expected=first["feedback_version"], category_verdict="confirmed", subcategory_verdict="confirmed")
    live = _rows(store)
    with store.connection() as conn:
        assert db.get_meta(conn, training_labels.META_BACKFILLED) is not None  # Store start ran it on the empty log
        assert training_labels.backfill(conn) == 0  # once
        with db.transaction(conn):
            conn.execute("DELETE FROM results_training_labels")
            conn.execute("DELETE FROM store_meta WHERE key = ?", (training_labels.META_BACKFILLED,))
        assert training_labels.backfill(conn) == 4
        assert training_labels.backfill(conn) == 0
    imported = _rows(store)
    keep = ("kind", "subject", "call_id", "conversation_id", "source_artifact_id", "label_json", "withdrawn", "recorded_at")
    # Overrides and corrections come back as logged; signal feedback only in its latest state.
    expected = [live[0], live[2], live[3], live[4]]
    assert [{k: r[k] for k in keep} for r in imported] == [{k: r[k] for k in keep} for r in expected]
    assert all(r["seq"] > live[-1]["seq"] for r in imported)


def test_s2_store_start_backfills_an_existing_dataset(tmp_path, clock):
    from fastapi.testclient import TestClient

    from call1.store.app import create_app
    from call1.store.config import StoreConfig

    config = StoreConfig.for_tests(tmp_path / "store")
    app = create_app(config, clock=clock)
    store = app.state.store
    with store.connection() as conn:
        with db.transaction(conn):
            conn.execute("DELETE FROM store_meta WHERE key = ?", (training_labels.META_BACKFILLED,))
    with TestClient(create_app(config, clock=clock)):
        pass
    with store.connection() as conn:
        assert db.get_meta(conn, training_labels.META_BACKFILLED) is not None


# --- S3: paging -------------------------------------------------------------------------------------


def test_s3_paging_kinds_counts_and_high_water(call, real, store, reviewer_session):
    empty = _labels(real, limit=0)
    assert empty == {"items": [], "next_after": 0, "count_after": 0, "high_water": 0}
    call.override(reviewer_session, "FAIL")
    saved = call.feedback(reviewer_session, expected=0, category_verdict="confirmed")
    call.correct(reviewer_session)
    call.override(reviewer_session, "PASS")
    call.feedback(reviewer_session, expected=saved["feedback_version"], category_verdict="dismissed")
    seqs = [r["seq"] for r in _rows(store)]
    top = seqs[-1]

    counts = _labels(real, limit=0)
    assert counts["items"] == [] and counts["count_after"] == 5 and counts["high_water"] == top and counts["next_after"] == 0
    first = _labels(real, limit=2)
    assert [i["seq"] for i in first["items"]] == seqs[:2] and first["next_after"] == seqs[1] and first["count_after"] == 5
    second = _labels(real, after=first["next_after"], limit=2)
    assert [i["seq"] for i in second["items"]] == seqs[2:4] and second["count_after"] == 3
    third = _labels(real, after=second["next_after"], limit=2)
    assert [i["seq"] for i in third["items"]] == seqs[4:] and third["count_after"] == 1
    done = _labels(real, after=third["next_after"])
    assert done == {"items": [], "next_after": top, "count_after": 0, "high_water": top}
    # A cursor from another dataset never passes the log.
    assert _labels(real, after=top + 50)["next_after"] == top

    signal_only = _labels(real, kinds="signal_hit")
    assert [i["kind"] for i in signal_only["items"]] == ["signal_hit", "signal_hit"] and signal_only["count_after"] == 2
    assert signal_only["high_water"] == top
    two_kinds = real.client.get(V + LABELS, params=[("kinds", "qa_verdict"), ("kinds", "speaker_role"), ("limit", "0")], headers=real.headers)
    assert two_kinds.status_code == 200 and two_kinds.json()["count_after"] == 3
    TrainingLabelPage.model_validate(first)
    for bad in ({"limit": 501}, {"limit": -1}, {"after": -1}, {"kinds": "nope"}):
        assert real.client.get(V + LABELS, params=bad, headers=real.headers).status_code == 422, bad


# --- S4: scope --------------------------------------------------------------------------------------


def test_s4_training_read_is_required_and_sessions_are_refused(call, real, mint_service_key, reviewer_session, admin_session):
    without = mint_service_key([s for s in ServiceScope if s is not ServiceScope.TRAINING_READ])
    refused = real.client.get(V + LABELS, headers=without.headers)
    assert refused.status_code == 403 and refused.json()["code"] == "insufficient_scope"
    assert refused.json()["details"]["required_scope"] == "training:read"
    only = mint_service_key([ServiceScope.TRAINING_READ])
    assert real.client.get(V + LABELS, headers=only.headers).status_code == 200
    for session in (reviewer_session, admin_session):
        assert real.client.get(V + LABELS, headers=session.read_headers).status_code in (401, 403)


def test_s4_training_read_is_a_default_process_scope():
    from call1 import launch
    from call1.store import __main__ as cli

    assert ServiceScope.TRAINING_READ in cli.PROCESS_DEFAULT_SCOPES and "training:read" in launch.PROCESS_SCOPES


# --- S5: no text ------------------------------------------------------------------------------------


def test_s5_pages_carry_no_text(call, real, store, reviewer_session):
    call.override(reviewer_session, "FAIL")
    call.feedback(reviewer_session, expected=0, category_verdict="confirmed", subcategory_verdict="confirmed")
    call.correct(reviewer_session)
    page = _labels(real)
    assert len(page["items"]) == 3
    raw = json.dumps(page) + json.dumps(_rows(store))
    for text in (CALLER_SECRET, "4111 1111", "Jane Roe", AGENT_LINE, QUOTE, REVIEWER_NOTE, SIGNAL_NOTE, CORRECTION_NOTE, "Escalated.",
                 reviewer_session.account_id):
        assert text not in raw, text
    for item in page["items"]:
        TrainingLabel.model_validate(item)


# --- S6: sources ------------------------------------------------------------------------------------


def test_s6_sources_resolve_per_kind(call, real, reviewer_session):
    call.override(reviewer_session, "FAIL", reason="evidence_not_in_transcript")
    call.feedback(reviewer_session, expected=0, category_verdict="dismissed")
    call.correct(reviewer_session)
    qa, signal, speaker = _labels(real)["items"]
    outputs = {o["role"]: o["artifact_id"] for o in real.job(call.criterion_job_id)["outputs"]}
    escalation = {o["role"]: o["artifact_id"] for o in real.job(call.escalation_job_id)["outputs"]}["assessment"]
    assessment, prompt = outputs["assessment"], outputs["prompt_input"]

    assert qa["kind"] == "qa_verdict" and qa["source_job_id"] == call.criterion_job_id
    assert qa["qa"]["reason_code"] == "evidence_not_in_transcript" and qa["qa"]["override_id"].startswith("ovr")
    assert _roles(qa) == {"transcript": call.transcript["id"], "speaker_attribution": call.attribution["id"], "rubric": call.snapshot["id"],
                          "enrichment": call.enrichment["id"], "assessment": assessment, "prompt_input": prompt,
                          "escalation_assessment": escalation, "pii_findings": call.pii["id"]}
    assert {s["role"]: s["kind"] for s in qa["sources"]}["escalation_assessment"] == "qa_assessment"
    assert all(s["checksum"].startswith("sha256:") for s in qa["sources"])

    assert signal["source_job_id"] == call.merge_job_id and signal["signal"]["category_id"] == "intent"
    assert signal["signal"]["spans"] == [{"turn_id": 1, "block": 0}, {"turn_id": 2, "block": 0}]
    assert _roles(signal) == {"transcript": call.transcript["id"], "speaker_attribution": call.attribution["id"], "pii_findings": call.pii["id"],
                              "taxonomy": call.taxonomy_snapshot["id"], "contact_signals": call.contact_signals["id"]}

    assert speaker["source_job_id"] == call.speaker_job_id and speaker["subject"].endswith(":S2")
    assert _roles(speaker) == {"transcript": call.transcript["id"], "speaker_attribution": call.attribution["id"], "pii_findings": call.pii["id"],
                               "enrichment": call.enrichment["id"]}


def test_s6_pii_findings_are_the_newest_for_the_labels_transcript(call, real, reviewer_session):
    """A newer enrichment for the same transcript wins; newer findings for another revision never do."""
    newer = real.graph(call.conversation_id, [job("enrich2", JobType.ENRICHMENT, slot="cpu", inputs=[pinned("transcript", call.transcript)])])
    enrich2 = real.claim_one(real.ids(newer)["enrich2"], slots=[{"memory_slot": "cpu", "count": 4}])
    real.complete(enrich2)
    bound = call.artifacts()["pii_findings"][-1]
    assert bound["producing_job_id"] == real.ids(newer)["enrich2"]
    unbound = real.graph(call.conversation_id, [job("enrich3", JobType.ENRICHMENT, slot="cpu", inputs=[pinned("transcript", call.transcript)])])
    claimed = real.claim_one(real.ids(unbound)["enrich3"], slots=[{"memory_slot": "cpu", "count": 4}])
    real.complete(claimed, outputs=real.outputs_for(claimed, payloads={"pii_findings": content(ArtifactKind.PII_FINDINGS)}))
    assert call.artifacts()["pii_findings"][-1]["id"] != bound["id"]
    call.override(reviewer_session)
    call.correct(reviewer_session)
    qa, speaker = _labels(real)["items"]
    assert _roles(qa)["pii_findings"] == bound["id"] != call.pii["id"]
    # The speaker's enrichment comes from the job that made those findings.
    assert _roles(speaker)["pii_findings"] == bound["id"]
    assert _roles(speaker)["enrichment"] == {o["role"]: o["artifact_id"] for o in real.job(real.ids(newer)["enrich2"])["outputs"]}["enrichment"]


def test_s6_a_second_correction_still_judges_the_models_attribution(call, real, reviewer_session):
    request = call.correct(reviewer_session)
    [claimed] = real.post("/reanalysis-requests/claim", {"worker_id": "w1", "max_requests": 4}).json()["requests"]
    fix = real.graph(call.conversation_id, [job("fix", JobType.SPEAKER_ATTRIBUTION, key="speaker-fix-t", slot="cpu",
                                                 inputs=[pinned("transcript", call.transcript), pinned("speaker_attribution", call.attribution)],
                                                 parameters={"speaker_correction": claimed["request"]["speaker_correction"]})],
                     key="speaker-fix-graph", reason="reanalysis", request_id=request["id"], claim_token=claimed["claim_token"])
    corrected = content(ArtifactKind.SPEAKER_ATTRIBUTION, method="reviewer_correction", corrected_turn_ids=[1, 2], assignments=[
        {"turn_id": 0, "speaker": "AGENT", "speaker_cluster": "S1"}, {"turn_id": 1, "speaker": "CALLER", "speaker_cluster": "S2"},
        {"turn_id": 2, "speaker": "CALLER", "speaker_cluster": "S2"}])
    job_ = real.claim_one(real.ids(fix)["fix"], slots=[{"memory_slot": "cpu", "count": 4}])
    real.complete(job_, outputs=real.outputs_for(job_, payloads={"speaker_attribution": corrected}))
    assert call.artifacts()["speaker_attribution"][-1]["id"] != call.attribution["id"]
    call.correct(reviewer_session, turn_id=0, speaker="AGENT")
    first, second = _labels(real)["items"]
    assert second["subject"] == f"speaker:{call.call_id}:S1" and second["source_job_id"] == call.speaker_job_id
    assert _roles(second)["speaker_attribution"] == call.attribution["id"]


def test_s6_an_unresolvable_source_leaves_sources_empty(real, client, reviewer_session):
    """A scorecard that pins no model assessment for the criterion (a deterministic check) resolves nothing."""
    conversation = real.register()
    snapshot = real.post(f"/conversations/{conversation['id']}/rubric-snapshots", {"rubric_id": SEED, "version": 1}, expect=201).json()
    ref = client.get(f"{V}/rubrics/{SEED}", headers=real.headers).json()["ref"]
    graph = real.graph(conversation["id"], [job("score", JobType.QA_SCORECARD, slot="cpu", inputs=[pinned("rubric", snapshot)], parameters={"rubric": ref})])
    score = real.claim_one(real.ids(graph)["score"], slots=[{"memory_slot": "cpu", "count": 4}])
    card = qa_scorecard([(CRITERION, "PASS", 0.97)]).model_dump(mode="json")
    card["rubric"] = {"rubric_id": SEED, "rubric_version": 1, "digest": ref["digest"]}
    real.complete(score, outputs=real.outputs_for(score, payloads={"scorecard": QaScorecardContent.model_validate(card).model_dump(mode="json")}))
    response = client.post(f"{V}/calls/{conversation['call_id']}/verdicts/{CRITERION}",
                           json={"status": "FAIL", "expected_version": 0, "evaluation_version": 1}, headers=reviewer_session.headers)
    assert response.status_code == 200, response.text
    [item] = _labels(real)["items"]
    assert item["kind"] == "qa_verdict" and item["sources"] == [] and item["source_job_id"] is None and item["withdrawn"] is False


def test_s6_withdrawn_rows_resolve_nothing(call, real, reviewer_session):
    saved = call.feedback(reviewer_session, expected=0, category_verdict="confirmed")
    call.feedback(reviewer_session, expected=saved["feedback_version"])
    live, withdrawn = _labels(real)["items"]
    assert live["withdrawn"] is False and live["sources"] and live["source_job_id"] == call.merge_job_id
    assert withdrawn["withdrawn"] is True and withdrawn["sources"] == [] and withdrawn["source_job_id"] is None
    assert withdrawn["subject"] == live["subject"]


def test_s6_v1_hits_are_not_logged(real, client, store, reviewer_session, admin_session):
    from .test_signals_harness import v1_content

    conversation = real.register()
    graph = real.graph(conversation["id"], [job("merge", JobType.CONTACT_SIGNALS_MERGE, slot="cpu")])
    merge = real.claim_one(real.ids(graph)["merge"], slots=[{"memory_slot": "cpu", "count": 4}])
    real.complete(merge, outputs=real.outputs_for(merge, payloads={"contact_signals": v1_content().model_dump(mode="json")}))
    hit_id = client.get(f"{V}/calls/{conversation['call_id']}/contact-signals", headers=reviewer_session.read_headers).json()["signals"][0]["id"]
    saved = client.put(f"{V}/calls/{conversation['call_id']}/signal-hits/{hit_id}/feedback",
                       json={"category_verdict": "confirmed", "expected_feedback_version": 0}, headers=reviewer_session.headers)
    assert saved.status_code == 200, saved.text
    assert _rows(store) == []


# --- S7: audit --------------------------------------------------------------------------------------


def test_s7_item_reads_are_audited_and_count_only_reads_are_not(call, real, store, reviewer_session):
    call.override(reviewer_session)
    call.correct(reviewer_session)
    _labels(real, limit=0)
    assert _audits(store) == []
    page = _labels(real, limit=1)
    _labels(real, after=page["next_after"])
    events = _audits(store)
    assert len(events) == 2
    first = json.loads(events[0]["details_json"])
    assert first == {"after": 0, "next_after": page["next_after"], "items": 1, "withdrawn": 0, "qa_verdict": 1, "signal_hit": 0, "speaker_role": 0}
    assert json.loads(events[1]["details_json"])["speaker_role"] == 1
    actor = json.loads(events[0]["actor_json"])
    assert actor["kind"] == "process_service" and actor["installation_id"] == real.installation_id
    assert events[0]["target_kind"] == "installation" and events[0]["target_id"] == real.installation_id
    with store.connection() as conn:
        from call1.store import audit

        assert audit.verify_chain(conn)


# --- S8: listJobs memory_slot --------------------------------------------------------------------------


def test_s8_list_jobs_filters_on_memory_slot(real):
    conversation = real.register()
    audio = real.upload_audio(conversation["id"])
    graph = real.graph(conversation["id"], [job("vad", JobType.VALIDATION_VAD, slot="cpu", inputs=[pinned("audio", audio)]),
                                            job("asr", JobType.ASR, inputs=[pinned("audio", audio)])])
    ids = real.ids(graph)

    def listed(**params):
        return [j["id"] for j in real.get("/jobs", params={"conversation_id": conversation["id"], **params}).json()["items"]]

    assert listed(memory_slot="local_memory") == [ids["asr"]]
    assert listed(memory_slot="cpu") == [ids["vad"]]
    assert listed(memory_slot="local_memory", status="QUEUED", limit=1) == [ids["asr"]]
    assert listed(memory_slot="outbound") == []
    assert set(listed()) == {ids["asr"], ids["vad"]}
    real.claim_one(ids["asr"])
    assert listed(memory_slot="local_memory", status="QUEUED") == [] and listed(memory_slot="local_memory", status="RUNNING") == [ids["asr"]]
    assert real.get("/jobs", params={"memory_slot": "nope"}, expect=422)
