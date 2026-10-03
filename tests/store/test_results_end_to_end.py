"""End to end through the real queue and auth areas over HTTP: Process registers a call, uploads its
audio, runs code-stage jobs and completes them; Evaluate then reads the projections the completion
transaction published, and the human review queue reacts to a new machine version.

Only code stages are used (validation_vad, qa_scorecard), so no model selection is needed."""

from __future__ import annotations

import hashlib
import io
import wave
from datetime import datetime, timezone
from itertools import count

from call1.contracts.common import canonical_digest
from call1.contracts.contents import VerdictStatus
from call1.contracts.usage import HardwareProfileFields, hardware_fingerprint

from .test_results_harness import qa_scorecard, validation_report, vad_metrics

V = "/store/v1"
_keys = count(1)


def _key(prefix: str) -> str:
    return f"{prefix}-{next(_keys):06d}"


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x01" * 8000)
    return buf.getvalue()


class Process:
    """A minimal Process client: just enough of the contract to drive code-stage jobs."""

    def __init__(self, client, key, clock) -> None:
        self.client = client
        self.clock = clock
        self.h = key.headers
        self.installation_id = key.installation_id
        fields = HardwareProfileFields(kind="process_host", source="measured", chip="Test CPU", memory_bytes=1 << 30, os_name="Linux",
                                       os_version="6")
        body = {**fields.model_dump(mode="json"), "fingerprint": hardware_fingerprint(fields)}
        r = client.put(f"{V}/hardware-profiles", json=body, headers=self.h)
        assert r.status_code in (200, 201), r.text
        self.hardware_profile_id = r.json()["id"]

    def ok(self, response, *codes):
        assert response.status_code in (codes or (200, 201)), response.text
        return response.json()

    def register(self, agent_id: str = "agent-e2e"):
        digest = "sha256:" + hashlib.sha256(_key("src").encode()).hexdigest()
        body = {"ingestion_kind": "call_audio", "source": {"kind": "api_upload", "content_digest": digest,
                                                           "received_at": datetime.now(timezone.utc).isoformat()},
                "call_metadata": {"agent_id": agent_id}}
        return self.ok(self.client.post(f"{V}/conversations", json=body, headers=self.h))["conversation"]

    def upload_audio(self, conversation_id: str, data: bytes):
        checksum = "sha256:" + hashlib.sha256(data).hexdigest()
        grant = self.ok(self.client.post(f"{V}/conversations/{conversation_id}/artifacts/uploads", json={
            "kind": "source_audio", "content_type": "audio/wav", "size_bytes": len(data), "checksum": checksum, "content_contract": "audio.v1",
            "sensitivity": "raw"}, headers=self.h))
        put = self.client.put(grant["url"], content=data, headers=grant.get("headers") or {})
        assert put.status_code in (200, 201, 204), put.text
        return self.ok(self.client.post(f"{V}/artifact-uploads/{grant['upload_id']}/commit", json={"checksum": checksum, "size_bytes": len(data)},
                                        headers=self.h))

    def graph(self, conversation_id: str, jobs):
        self.clock.advance(1)  # graphs are ordered by creation time (calls.result_state_inputs)
        return self.ok(self.client.post(f"{V}/conversations/{conversation_id}/job-graphs", json={
            "idempotency_key": _key("graph"), "reason": "ingest", "jobs": jobs}, headers=self.h))

    def claim(self, job_type: str):
        body = {"worker": {"worker_id": "w1", "installation_id": self.installation_id, "hardware_profile_id": self.hardware_profile_id,
                           "primary_host": True, "job_types": [job_type], "route_classes": ["appliance"],
                           "slot_offers": [{"memory_slot": "cpu", "count": 1}]}, "max_jobs": 1}
        claimed = self.ok(self.client.post(f"{V}/jobs/claim", json=body, headers=self.h))["jobs"]
        assert len(claimed) == 1, claimed
        return claimed[0]

    def output(self, conversation_id: str, claimed, kind: str, payload) -> dict:
        doc = payload.model_dump(mode="json")
        from call1.contracts.common import canonical_json
        art = self.ok(self.client.post(f"{V}/conversations/{conversation_id}/artifacts", json={
            "kind": kind, "content_type": "application/json", "size_bytes": len(canonical_json(doc)), "checksum": canonical_digest(doc),
            "content_contract": f"{kind}.v1", "sensitivity": "derived", "producing_job_id": claimed["job"]["id"], "payload": doc,
            "claim_token": claimed["claim_token"]}, headers=self.h))
        return art

    def complete(self, claimed, outputs, result=None):
        body = {"claim_token": claimed["claim_token"], "completion_key": _key("done"),
                "outputs": [{"role": role, "artifact_id": a["id"], "checksum": a["checksum"]} for role, a in outputs.items()],
                "usage": {"tokens_input": {"source": "unavailable"}, "tokens_output": {"source": "unavailable"}, "inference_seconds": 0.1,
                          "total_seconds": 0.2, "hardware_profile_id": self.hardware_profile_id, "outcome": "succeeded"},
                "provenance": {"worker_id": "w1", "installation_id": self.installation_id, "adapter_id": "test", "adapter_version": "1"},
                "result": result}
        return self.ok(self.client.post(f"{V}/jobs/{claimed['job']['id']}/complete", json=body, headers=self.h))

    def fail(self, claimed, code: str = "configuration_error"):
        body = {"claim_token": claimed["claim_token"], "completion_key": _key("fail"), "error_code": code,
                "usage": {"tokens_input": {"source": "unavailable"}, "tokens_output": {"source": "unavailable"}, "inference_seconds": 0,
                          "total_seconds": 0.1, "hardware_profile_id": self.hardware_profile_id, "outcome": "failed", "error_code": code},
                "provenance": {"worker_id": "w1", "installation_id": self.installation_id, "adapter_id": "test", "adapter_version": "1"}}
        return self.ok(self.client.post(f"{V}/jobs/{claimed['job']['id']}/fail", json=body, headers=self.h))


def _estimate():
    return {"size_class": "s", "memory_slot": "cpu"}


def _scorecard_job(snapshot, rubric):
    return {"ref": "scorecard", "job_type": "qa_scorecard", "idempotency_key": _key("job"), "resource_estimate": _estimate(),
            "parameters": {"rubric": rubric["ref"]},
            "inputs": [{"role": "rubric", "artifact": {"artifact_id": snapshot["id"], "checksum": snapshot["checksum"]}}]}


def test_process_to_evaluate_through_the_real_queue(client, clock, service_key, reviewer_session, mint_session):
    process = Process(client, service_key, clock)
    conversation = process.register()
    call_id, conversation_id = conversation["call_id"], conversation["id"]
    r = reviewer_session.read_headers

    listed = client.get(f"{V}/calls", headers=r).json()["items"]
    assert [(i["call_id"], i["transcript_state"], i["qa_state"]) for i in listed] == [(call_id, "disabled", "disabled")]

    audio = process.upload_audio(conversation_id, _wav())
    rubric = client.get(f"{V}/rubrics/call1_standard_v2", headers=process.h).json()
    snapshot = process.ok(client.post(f"{V}/conversations/{conversation_id}/rubric-snapshots", json={"rubric_id": "call1_standard_v2", "version": 1},
                                      headers=process.h))
    graph = process.graph(conversation_id, [
        {"ref": "vad", "job_type": "validation_vad", "idempotency_key": _key("job"), "resource_estimate": _estimate(),
         "inputs": [{"role": "audio", "artifact": {"artifact_id": audio["id"], "checksum": audio["checksum"]}}]},
        _scorecard_job(snapshot, rubric),
    ])
    assert graph["created"] is True
    detail = client.get(f"{V}/calls/{call_id}", headers=r).json()
    assert {g["kind"]: g["state"] for g in detail["results"]}["qa"] == "pending"

    vad_claim = process.claim("validation_vad")
    receipt = process.complete(vad_claim, {
        "validation_report": process.output(conversation_id, vad_claim, "validation_report", validation_report(1.0)),
        "vad_metrics": process.output(conversation_id, vad_claim, "vad_metrics", vad_metrics()),
    })
    assert receipt["result_version"] is None
    assert client.get(f"{V}/calls/{call_id}", headers=r).json()["call"]["duration_seconds"] == 1.0

    qa_claim = process.claim("qa_scorecard")
    card = qa_scorecard([("REG-01", VerdictStatus.FAIL, 0.95), ("SEC-01", VerdictStatus.PASS, 0.9)], digest=rubric["ref"]["digest"],
                        overall_score=55, passed=False, critical_failure=True, requires_human_review=True)
    receipt = process.complete(qa_claim, {"scorecard": process.output(conversation_id, qa_claim, "qa_scorecard", card)},
                               result={"kind": "qa", "state": "available"})
    assert receipt["result_version"] == 1

    detail = client.get(f"{V}/calls/{call_id}", headers=r).json()
    assert detail["evaluation"]["version"] == 1 and detail["evaluation"]["overall_score"] == 55
    assert {g["kind"]: g["state"] for g in detail["results"]}["qa"] == "available"
    assert detail["pending_work"]["settled"] is True
    items = client.get(f"{V}/review-queue", params={"call_id": call_id}, headers=r).json()["items"]
    assert [i["rule_id"] for i in items] == ["rule-triage-critical-lowconf"] and items[0]["evaluation_version"] == 1
    audio_bytes = client.get(f"{V}/calls/{call_id}/audio", headers=r)
    # No transcript (so no PII findings for one): audio is withheld, never served unmuted (contract 1.2.0).
    assert audio_bytes.status_code == 503 and audio_bytes.json()["details"]["reason"] == "pii_findings_pending", audio_bytes.text

    reviewer = mint_session("reviewer")
    override = client.post(f"{V}/calls/{call_id}/verdicts/REG-01", json={"status": "PASS", "expected_version": 0, "evaluation_version": 1},
                           headers=reviewer.headers)
    assert override.status_code == 200

    # A new machine version: the old item is superseded and the override becomes stale.
    second = process.graph(conversation_id, [_scorecard_job(snapshot, rubric)])
    qa_claim = process.claim("qa_scorecard")
    card = qa_scorecard([("REG-01", VerdictStatus.PASS, 0.95), ("SEC-01", VerdictStatus.PASS, 0.9)], digest=rubric["ref"]["digest"])
    assert process.complete(qa_claim, {"scorecard": process.output(conversation_id, qa_claim, "qa_scorecard", card)},
                            result={"kind": "qa", "state": "available"})["result_version"] == 2
    assert second["created"] is True
    old = client.get(f"{V}/review-queue/items/{items[0]['id']}", headers=r).json()
    assert old["status"] == "SUPERSEDED" and old["stale"] is True
    review = client.get(f"{V}/calls/{call_id}/review", headers=r).json()
    assert review["staleness"] == "stale" and review["current_evaluation_version"] == 2
    stale = client.post(f"{V}/calls/{call_id}/verdicts/SEC-01", json={"status": "FAIL", "expected_version": 1, "evaluation_version": 1},
                        headers=reviewer.headers)
    assert stale.status_code == 409 and stale.json()["details"]["current_evaluation_version"] == 2

    # A newer graph whose scorecard fails leaves version 2 shown with the failure code.
    process.graph(conversation_id, [_scorecard_job(snapshot, rubric)])
    process.fail(process.claim("qa_scorecard"), "configuration_error")
    qa = {g["kind"]: g for g in client.get(f"{V}/calls/{call_id}", headers=r).json()["results"]}["qa"]
    assert qa["state"] == "available" and qa["version"] == 2 and qa["failure_code"] == "configuration_error"
