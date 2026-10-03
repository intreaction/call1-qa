"""Shared harness for the queue-area tests (no tests here).

Everything goes through the HTTP API with a TestClient and a minted Process key, the way Process
will call Store. The results area's cross-area functions are replaced where a queue test needs
determinism: ``results.api`` rubric lookups return a fixed rubric, and the projection hooks are
recorders (``Hooks``) unless a test asks for the real ones. ``fake_result_groups`` derives group
states with ``calls.derive_result_state`` over the queue's own stakes, for progress tests.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlsplit

import pytest

from call1.contracts.artifacts import ARTIFACT_CONTENT_CONTRACTS, ArtifactKind
from call1.contracts.calls import ResultGroup, derive_result_state, result_state_inputs
from call1.contracts.catalog import ModelPurpose
from call1.contracts.common import canonical_digest, canonical_json
from call1.contracts.contents import (
    AsrVocabularyPassContent,
    AudioValidationContent,
    ContactSignalPass,
    ContactSignalsContent,
    ContactSignalsPassContent,
    ContactSignalsPassOutcome,
    EmbeddingsContent,
    EnrichmentContent,
    PiiFindingsContent,
    ModelAttemptView,
    PromptInputContent,
    QaAssessmentContent,
    QaScorecardContent,
    QaVerdictContent,
    ResultKind,
    ResultState,
    ScorecardRubricRef,
    SpeakerAttributionContent,
    SummaryContent,
    TextSentimentContent,
    ToneBlocksContent,
    TranscriptContent,
    TranscriptTurnContent,
    TurnEmbedding,
    TurnEnrichment,
    TurnPiiFindings,
    TurnSentiment,
    VadMetricsContent,
    VerdictStatus,
)
from call1.contracts.jobs import JOB_TYPE_RULES, JobType
from call1.contracts.rubrics import (
    RubricCheck,
    RubricCriterion,
    RubricDefinition,
    RubricSnapshotContent,
    RubricVersion,
    RubricVersionRef,
    RubricVersionStatus,
)
from call1.store.objects import sha256_checksum
from call1.store.queue import api as queue_api
from call1.store.results import api as results_api
from call1.store.results import projections

V = "/store/v1"
AUDIO = b"RIFF" + bytes(range(256)) * 16
RUBRIC_ID = "rubric-a"
CRITERION = "greeting"
DEFINITION = RubricDefinition(rubric_id=RUBRIC_ID, name="Rubric A", criteria=[RubricCriterion(criterion_id=CRITERION, name="Greeting", check=RubricCheck())])
DRAFT_DEFINITION = RubricDefinition(rubric_id=RUBRIC_ID, name="Rubric A (draft)", criteria=[RubricCriterion(criterion_id=CRITERION, name="Greeting v2", check=RubricCheck())])
RUBRIC_REF = RubricVersionRef(rubric_id=RUBRIC_ID, version=1, digest=canonical_digest(DEFINITION))
ENTRIES = [{"entry_id": f"mlx-{p.value}", "entry_version": 1} for p in ModelPurpose]


def path(url: str) -> str:
    parts = urlsplit(url)
    return parts.path + ("?" + parts.query if parts.query else "")


# --- contract payloads ----------------------------------------------------------------------


def route() -> Dict[str, Any]:
    return {"route_class": "appliance", "provider_type": "mlx", "destination_host": "in-process", "masked": True}


def selection(purpose: ModelPurpose) -> Dict[str, Any]:
    return {"catalog_entry": {"entry_id": f"mlx-{purpose.value}", "entry_version": 1}, "purpose": purpose.value, "model_family": "test",
            "model_revision": "rev-1", "adapter_id": "adapter", "adapter_version": "1", "output_contract": "x.v1",
            "provider_model_id": "model", "route": route()}


UNBOUND_TRANSCRIPT = {"artifact_id": "art_unbound", "checksum": "sha256:" + "0" * 64}
"""The default ``pii_findings`` transcript reference: no real transcript, so Store withholds text."""


def content(kind: ArtifactKind, *, now=None, **overrides) -> Dict[str, Any]:
    """A full canonical dump of a valid content document for ``kind``."""
    ts = now or "2026-09-25T12:00:00Z"
    turns = [TranscriptTurnContent(turn_id=0, speaker="AGENT", start_time=0.0, end_time=2.0, text="Hello, thanks for calling.")]
    models = {
        ArtifactKind.VALIDATION_REPORT: lambda: AudioValidationContent(container="wav", codec="pcm_s16le", sample_rate=16000, channels=1, channel_layout="MONO", duration_seconds=12.5),
        ArtifactKind.VAD_METRICS: lambda: VadMetricsContent(total_speech_duration=10, total_silence_duration=2.5, silence_ratio=0.2, overtalk_duration=0, overtalk_ratio=0),
        ArtifactKind.TRANSCRIPT: lambda: TranscriptContent(duration_seconds=12.5, is_redacted=False, turns=turns),
        ArtifactKind.ASR_BASE_TRANSCRIPT: lambda: TranscriptContent(duration_seconds=12.5, is_redacted=False, turns=turns),
        ArtifactKind.ASR_VOCABULARY_PASS: lambda: AsrVocabularyPassContent(engine="whisper-small", duration_seconds=12.5, channels=1, words=[]),
        ArtifactKind.SPEAKER_ATTRIBUTION: lambda: SpeakerAttributionContent(method="channel", assignments=[{"turn_id": 0, "speaker": "AGENT"}]),
        ArtifactKind.TONE_BLOCKS: lambda: ToneBlocksContent(blocks=[]),
        ArtifactKind.TEXT_SENTIMENT: lambda: TextSentimentContent(model="m", revision="r", turns=[TurnSentiment(turn_id=0, score=0.5, label="POSITIVE")]),
        ArtifactKind.EMBEDDINGS: lambda: EmbeddingsContent(scheme="hashing-projection-v1", dimensions=2, turn_vectors=[TurnEmbedding(turn_id=0, vector=[0.5, 0.5])]),
        ArtifactKind.ENRICHMENT: lambda: EnrichmentContent(turns=[TurnEnrichment(turn_id=0)]),
        ArtifactKind.PII_FINDINGS: lambda: PiiFindingsContent(transcript=overrides.pop("transcript_ref", UNBOUND_TRANSCRIPT), detector="stub",
                                                              detector_revision="stub-v1", turns=[TurnPiiFindings(turn_id=0)]),
        ArtifactKind.PROMPT_INPUT: lambda: PromptInputContent(template_id="t", template_version="1", prompt_digest=canonical_digest({"m": 1}), masked=True),
        ArtifactKind.QA_ASSESSMENT: lambda: QaAssessmentContent(
            criterion_id=CRITERION, assessment_kind="primary", status=VerdictStatus.PASS, confidence=0.9, reasoning="Greeted.",
            escalation_requested=False, attempt=ModelAttemptView(catalog_entry_id="mlx-semantic_qa", model_revision="rev-1", route_class="appliance",
                                                                 destination_host="in-process", status=VerdictStatus.PASS, reasoning="ok", latency_ms=5)),
        ArtifactKind.QA_VERDICT: lambda: QaVerdictContent(verdicts=[]),
        ArtifactKind.QA_SCORECARD: lambda: QaScorecardContent(rubric=ScorecardRubricRef(rubric_id=RUBRIC_ID, rubric_version=1, digest=RUBRIC_REF.digest),
                                                              overall_score=90, passed=True, critical_failure=False, requires_human_review=False,
                                                              verdicts=[], evaluated_at=ts),
        ArtifactKind.SUMMARY: lambda: SummaryContent.model_validate(overrides.pop("summary")),
        ArtifactKind.CONTACT_SIGNALS_PASS: lambda: ContactSignalsPassContent(pass_kind=overrides.pop("pass_kind", ContactSignalPass.LIFECYCLE), signals=[]),
        ArtifactKind.CONTACT_SIGNALS: lambda: ContactSignalsContent(completeness="complete", signals=[], passes=[
            ContactSignalsPassOutcome(pass_kind="lifecycle", included=True), ContactSignalsPassOutcome(pass_kind="resolution", included=True)],
            transcript_fingerprint=canonical_digest({"t": 1}), generated_at=ts),
    }
    model = models[kind]()
    data = model.model_dump(mode="json")
    data.update(overrides)
    return type(model).model_validate(data).model_dump(mode="json")


_DEFAULT_SLOT = {
    ArtifactKind.QA_ASSESSMENT: CRITERION,
    ArtifactKind.CONTACT_SIGNALS_PASS: "lifecycle:0",
}

_SENSITIVITY = {
    ArtifactKind.VALIDATION_REPORT: "derived", ArtifactKind.VAD_METRICS: "derived", ArtifactKind.EMBEDDINGS: "derived",
    ArtifactKind.PROMPT_INPUT: "derived", ArtifactKind.RUBRIC_SNAPSHOT: "derived", ArtifactKind.SOURCE_AUDIO: "raw",
    ArtifactKind.PII_FINDINGS: "raw", ArtifactKind.ASR_BASE_TRANSCRIPT: "raw", ArtifactKind.ASR_VOCABULARY_PASS: "raw",
}


def descriptor(kind: ArtifactKind, payload: Dict[str, Any], *, slot: str = "", job_id: Optional[str] = None, token: Optional[str] = None) -> Dict[str, Any]:
    data = canonical_json(payload)
    body = {"kind": kind.value, "slot": slot, "content_type": "application/json", "size_bytes": len(data), "checksum": sha256_checksum(data),
            "content_contract": ARTIFACT_CONTENT_CONTRACTS[kind], "sensitivity": _SENSITIVITY.get(kind, "masked"), "payload": payload}
    if job_id is not None:
        body["producing_job_id"] = job_id
        body["claim_token"] = token
    return body


def usage(outcome: str = "succeeded", error_code: Optional[str] = None, **extra) -> Dict[str, Any]:
    return {"tokens_input": {"count": 100, "source": "local_tokenizer"}, "tokens_output": {"count": 20, "source": "local_tokenizer"},
            "inference_seconds": 2.0, "total_seconds": 2.5, "hardware_profile_id": "hw_test", "outcome": outcome, "error_code": error_code, **extra}


def resource(slot: str = "local_memory", outbound: Optional[str] = None) -> Dict[str, Any]:
    return {"size_class": "s", "memory_slot": slot, "outbound_connection_ref": outbound}


def job(ref: str, job_type: JobType, *, key: Optional[str] = None, inputs: Sequence[Dict] = (), requires: Sequence[str] = (),
        after: Sequence[str] = (), requires_ids: Sequence[str] = (), after_ids: Sequence[str] = (), priority: int = 0,
        max_attempts: Optional[int] = None, parameters: Optional[Dict] = None, slot: str = "local_memory", sel: Any = "auto") -> Dict[str, Any]:
    rule = JOB_TYPE_RULES[job_type]
    params = dict(parameters or {})
    if rule.needs_rubric and "rubric" not in params and "draft_rubric" not in params:
        params["rubric"] = RUBRIC_REF.model_dump(mode="json")
    if job_type in (JobType.QA_CRITERION, JobType.QA_ESCALATION):
        params.setdefault("criterion_id", CRITERION)
    body = {"ref": ref, "job_type": job_type.value, "idempotency_key": key or f"job-{ref}-{job_type.value}", "priority": priority,
            "max_attempts": max_attempts, "inputs": list(inputs), "resource_estimate": resource(slot), "parameters": params,
            "requires_refs": list(requires), "after_refs": list(after), "requires_job_ids": list(requires_ids), "after_job_ids": list(after_ids)}
    if sel == "auto":
        sel = selection(rule.purpose) if (rule.purpose is not None and not params.get("speaker_correction")) else None
    body["selection"] = sel
    return body


def upstream_input(role: str, ref: str, output_role: str, *, optional: bool = False) -> Dict[str, Any]:
    return {"role": role, "upstream": {"ref": ref, "output_role": output_role}, "optional": optional}


def pinned(role: str, artifact: Dict[str, Any]) -> Dict[str, Any]:
    return {"role": role, "artifact": {"artifact_id": artifact["id"], "checksum": artifact["checksum"]}}


# --- fakes for the results area ---------------------------------------------------------------


class Hooks:
    """Recording stand-ins for the results projection hooks."""

    def __init__(self) -> None:
        self.completions: List = []
        self.failures: List = []
        self.registered: List = []
        self.raise_on_completion: Optional[BaseException] = None
        self.on_completion = None
        self._versions = itertools.count(1)

    def apply_completion(self, conn, completion):
        from call1.store.hooks import ProjectionOutcome

        if self.raise_on_completion is not None:
            raise self.raise_on_completion
        self.completions.append(completion)
        if self.on_completion is not None:
            self.on_completion(conn, completion)
        if completion.result is not None and completion.job.draft_test_request_id is None:
            return ProjectionOutcome(result_version=next(self._versions))
        return ProjectionOutcome()

    def on_job_failed(self, conn, failure):
        self.failures.append(failure)

    def on_conversation_registered(self, conn, conversation):
        self.registered.append(conversation)


def fake_result_groups(conn, conversation_id: str) -> List[ResultGroup]:
    """derive_result_state over the queue's stakes; 'published' = a live publisher SUCCEEDED."""
    stakes = queue_api.group_stakes(conn, conversation_id)
    pending = queue_api.reanalysis_pending_groups(conn, conversation_id)
    groups = []
    for kind in ResultKind:
        live = [s for s in stakes[kind] if not s.draft_test and s.publisher is not None]
        succeeded = [s for s in live if s.publisher.value == "succeeded"]
        newest_ok = max(succeeded, key=lambda s: s.graph_created_at, default=None)
        inputs = result_state_inputs(stakes[kind], ResultState.AVAILABLE if newest_ok else None,
                                     newest_ok.graph_created_at if newest_ok else None, kind in pending)
        groups.append(ResultGroup(kind=kind, state=derive_result_state(inputs)))
    return groups


def snapshot_content(rubric_id: str, version: int) -> Optional[RubricSnapshotContent]:
    if rubric_id != RUBRIC_ID or version != 1:
        return None
    return RubricSnapshotContent(source="published", rubric_id=RUBRIC_ID, rubric_version=1, digest=canonical_digest(DEFINITION), definition=DEFINITION)


# --- the harness ------------------------------------------------------------------------------


class QueueHarness:
    def __init__(self, client, store, clock, key, mint_service_key, mint_session) -> None:
        self.client = client
        self.store = store
        self.clock = clock
        self.key = key
        self.headers = key.headers
        self.installation_id = key.installation_id
        self.mint_service_key = mint_service_key
        self.mint_session = mint_session
        self._keys = itertools.count(1)
        self.draft_revision = 1

    # generic ------------------------------------------------------------------------------

    def post(self, url: str, body: Any = None, *, expect: Optional[int] = 200, headers: Optional[Dict] = None):
        response = self.client.post(V + url, json=body, headers=headers or self.headers)
        if expect is not None:
            assert response.status_code == expect, (response.status_code, response.text)
        return response

    def get(self, url: str, *, expect: Optional[int] = 200, headers: Optional[Dict] = None, params=None):
        response = self.client.get(V + url, headers=headers or self.headers, params=params)
        if expect is not None:
            assert response.status_code == expect, (response.status_code, response.text)
        return response

    # conversations and artifacts ---------------------------------------------------------------

    def register(self, *, etag: Optional[str] = None, object_key: str = "calls/1.wav", kind: str = "call_audio") -> Dict[str, Any]:
        body = {"ingestion_kind": kind, "source": {"kind": "s3_event", "bucket": "rec", "object_key": object_key,
                                                   "etag": etag or f"etag{next(self._keys)}", "received_at": "2026-09-25T11:59:00Z"}}
        if kind == "call_audio":
            body["call_metadata"] = {"agent_id": "agent-7"}
        return self.post("/conversations", body).json()["conversation"]

    def upload_audio(self, conversation_id: str, data: bytes = AUDIO) -> Dict[str, Any]:
        body = {"kind": "source_audio", "slot": "", "content_type": "audio/wav", "size_bytes": len(data), "checksum": sha256_checksum(data),
                "content_contract": "audio.v1", "sensitivity": "raw"}
        grant = self.post(f"/conversations/{conversation_id}/artifacts/uploads", body, expect=201).json()
        put = self.client.put(path(grant["url"]), content=data, headers={"Content-Type": "audio/wav"})
        assert put.status_code == 200, put.text
        return self.post(f"/artifact-uploads/{grant['upload_id']}/commit", {"checksum": body["checksum"], "size_bytes": len(data)}, expect=201).json()

    def inline(self, conversation_id: str, kind: ArtifactKind, payload: Optional[Dict] = None, *, slot: str = "", job_id: Optional[str] = None,
               token: Optional[str] = None, expect: Optional[int] = 201):
        body = descriptor(kind, payload if payload is not None else content(kind), slot=slot, job_id=job_id, token=token)
        return self.post(f"/conversations/{conversation_id}/artifacts", body, expect=expect)

    def rubric_snapshot(self, conversation_id: str) -> Dict[str, Any]:
        return self.post(f"/conversations/{conversation_id}/rubric-snapshots", {"rubric_id": RUBRIC_ID, "version": 1}, expect=201).json()

    # graphs ------------------------------------------------------------------------------------

    def graph(self, conversation_id: str, jobs: List[Dict], *, key: Optional[str] = None, reason: str = "ingest", expect: Optional[int] = 201,
              request_id: Optional[str] = None, claim_token: Optional[str] = None):
        body = {"idempotency_key": key or f"graph-key-{next(self._keys)}", "reason": reason, "jobs": jobs,
                "reanalysis_request_id": request_id, "reanalysis_claim_token": claim_token}
        response = self.post(f"/conversations/{conversation_id}/job-graphs", body, expect=expect)
        return response.json() if expect == 201 else response

    @staticmethod
    def ids(graph: Dict[str, Any]) -> Dict[str, str]:
        return {j["ref"]: j["job_id"] for j in graph["jobs"]}

    def job(self, job_id: str) -> Dict[str, Any]:
        return self.get(f"/jobs/{job_id}").json()

    def status(self, job_id: str) -> str:
        return self.job(job_id)["status"]

    # workers -----------------------------------------------------------------------------------

    def worker(self, *, job_types: Optional[Iterable[JobType]] = None, slots: Optional[List[Dict]] = None, primary_host: bool = True,
               worker_id: str = "w1", installation_id: Optional[str] = None, entries=None, route_classes=("appliance",)) -> Dict[str, Any]:
        return {"worker_id": worker_id, "installation_id": installation_id or self.installation_id, "hardware_profile_id": "hw_test",
                "primary_host": primary_host, "job_types": [t.value for t in (job_types or list(JobType))], "route_classes": list(route_classes),
                "qualified_entries": ENTRIES if entries is None else entries, "slot_offers": slots or [{"memory_slot": "local_memory", "count": 16}]}

    def claim(self, *, max_jobs: int = 16, expect: Optional[int] = 200, headers=None, conversation_id=None, lease_seconds=None, **worker) -> Dict[str, Any]:
        body = {"worker": self.worker(**worker), "max_jobs": max_jobs, "conversation_id": conversation_id, "lease_seconds": lease_seconds}
        response = self.post("/jobs/claim", body, expect=expect, headers=headers)
        return response.json() if expect == 200 else response

    def claim_one(self, job_id: str, **kw) -> Dict[str, Any]:
        """Claim ``job_id`` alone (one job of its type; the harness keeps priorities unambiguous)."""
        kw.setdefault("job_types", [JobType(self.job(job_id)["job_type"])])
        kw.setdefault("max_jobs", 1)
        claimed = self.claim(**kw)["jobs"]
        match = [c for c in claimed if c["job"]["id"] == job_id]
        assert match, f"{job_id} was not claimed (got {[c['job']['id'] for c in claimed]})"
        return match[0]

    def outputs_for(self, claimed: Dict[str, Any], *, slots: Optional[Dict[str, str]] = None, payloads: Optional[Dict[str, Dict]] = None) -> List[Dict]:
        job_ = claimed["job"]
        rule = JOB_TYPE_RULES[JobType(job_["job_type"])]
        draft = job_["parameters"].get("draft_rubric")
        outputs = []
        for role, kind in rule.outputs.items():
            slot = (slots or {}).get(role, _DEFAULT_SLOT.get(kind, ""))
            if draft is not None and not slot.startswith("draft:"):
                slot = f"draft:{self.draft_request_of(job_)}:{slot}"
            payload = (payloads or {}).get(role) or content(kind, **self.scorecard_rubric(job_, kind), **self.pii_transcript(claimed, kind))
            art = self.inline(job_["conversation_id"], kind, payload, slot=slot, job_id=job_["id"], token=claimed["claim_token"]).json()
            outputs.append({"role": role, "artifact_id": art["id"], "checksum": art["checksum"]})
        return outputs

    @staticmethod
    def pii_transcript(claimed: Dict[str, Any], kind: ArtifactKind) -> Dict[str, Any]:
        """``pii_findings`` (contract 1.2.0) name the transcript revision the job read, as Process's do."""
        if kind is not ArtifactKind.PII_FINDINGS:
            return {}
        found = next((i["artifact"] for i in claimed.get("inputs", []) if i["role"] == "transcript" and i.get("artifact")), None)
        return {"transcript_ref": {"artifact_id": found["id"], "checksum": found["checksum"]}} if found else {}

    @staticmethod
    def scorecard_rubric(job_: Dict[str, Any], kind: ArtifactKind) -> Dict[str, Any]:
        """A qa_scorecard names the rubric its job was pinned to (Store rejects a mismatch)."""
        if kind is not ArtifactKind.QA_SCORECARD:
            return {}
        params = job_["parameters"]
        draft, published = params.get("draft_rubric"), params.get("rubric")
        if draft is not None:
            return {"rubric": {"rubric_id": draft["rubric_id"], "draft_revision": draft["draft_revision"], "digest": draft["digest"]}}
        return {"rubric": {"rubric_id": published["rubric_id"], "rubric_version": published["version"], "digest": published["digest"]}}

    def draft_request_of(self, job_: Dict[str, Any]) -> str:
        from call1.store.queue.records import job_row

        with self.store.connection() as conn:
            return job_row(conn, job_["id"])["draft_test_request_id"]

    def completion(self, claimed: Dict[str, Any], outputs: List[Dict], *, key: Optional[str] = None, result: Any = "auto",
                   usage_body: Optional[Dict] = None, follow_on: Optional[Dict] = None, provenance: Optional[Dict] = None) -> Dict[str, Any]:
        job_ = claimed["job"]
        rule = JOB_TYPE_RULES[JobType(job_["job_type"])]
        if result == "auto":
            draft = job_["parameters"].get("draft_rubric") is not None
            result = {"kind": rule.publishes.value, "state": "available", "partial_reason": None} if (rule.publishes and not draft) else None
        prov = provenance or {"worker_id": "w1", "installation_id": self.installation_id, "adapter_id": "adapter", "adapter_version": "1",
                              "route": (job_["selection"] or {}).get("route")}
        return {"claim_token": claimed["claim_token"], "completion_key": key or f"done-{job_['id']}-{claimed['attempt_number']}",
                "outputs": outputs, "usage": usage_body or usage(), "provenance": prov, "result": result, "follow_on": follow_on}

    def complete(self, claimed: Dict[str, Any], *, expect: Optional[int] = 200, **kw):
        outputs = kw.pop("outputs", None) or self.outputs_for(claimed)
        response = self.post(f"/jobs/{claimed['job']['id']}/complete", self.completion(claimed, outputs, **kw), expect=expect)
        return response.json() if expect == 200 else response

    def run(self, job_id: str, **kw) -> Dict[str, Any]:
        """Claim one specific job and complete it."""
        return self.complete(self.claim_one(job_id), **kw)

    def failure(self, claimed: Dict[str, Any], code: str, *, key: Optional[str] = None) -> Dict[str, Any]:
        from call1.contracts.errors import JobErrorCode
        from call1.contracts.usage import usage_outcome_for

        outcome = usage_outcome_for(JobErrorCode(code)).value
        return {"claim_token": claimed["claim_token"], "completion_key": key or f"fail-{claimed['job']['id']}-{claimed['attempt_number']}",
                "error_code": code, "error_detail": "safe detail", "usage": usage(outcome, code),
                "provenance": {"worker_id": "w1", "installation_id": self.installation_id, "adapter_id": "adapter", "adapter_version": "1",
                               "route": (claimed["job"]["selection"] or {}).get("route")}}

    def fail(self, claimed: Dict[str, Any], code: str = "provider_error", *, expect: Optional[int] = 200, **kw):
        response = self.post(f"/jobs/{claimed['job']['id']}/fail", self.failure(claimed, code, **kw), expect=expect)
        return response.json() if expect == 200 else response

    def release(self, claimed: Dict[str, Any], disposition: str = "requeue", code: str = "resource_unavailable", *, key: Optional[str] = None,
                not_before: Optional[str] = None, expect: Optional[int] = 200):
        body = {"claim_token": claimed["claim_token"], "completion_key": key or f"rel-{claimed['job']['id']}-{claimed['attempt_number']}",
                "disposition": disposition, "reason_code": code, "detail": "slot lost", "not_before": not_before}
        response = self.post(f"/jobs/{claimed['job']['id']}/release", body, expect=expect)
        return response.json() if expect == 200 else response

    def heartbeat(self, claimed: Dict[str, Any], *, expect: Optional[int] = 200):
        response = self.post(f"/jobs/{claimed['job']['id']}/heartbeat", {"claim_token": claimed["claim_token"]}, expect=expect)
        return response.json() if expect == 200 else response

    def cancel(self, job_id: str, *, cascade: bool = True, expect: Optional[int] = 200):
        response = self.post(f"/jobs/{job_id}/cancel", {"reason": "operator cancelled", "cascade": cascade}, expect=expect)
        return response.json() if expect == 200 else response

    def retry(self, job_id: str, *, expect: Optional[int] = 200):
        response = self.post(f"/jobs/{job_id}/retry", {"reason": "fixed the model"}, expect=expect)
        return response.json() if expect == 200 else response

    # a standard ingest graph --------------------------------------------------------------------

    def ingest(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """validation_vad + asr on the uploaded audio, enrichment and text_sentiment on the transcript."""
        conversation_id = conversation_id or self.register()["id"]
        audio = self.upload_audio(conversation_id)
        graph = self.graph(conversation_id, [
            job("vad", JobType.VALIDATION_VAD, inputs=[pinned("audio", audio)]),
            job("asr", JobType.ASR, inputs=[pinned("audio", audio)], priority=5),
            job("enrich", JobType.ENRICHMENT, requires=["asr"], inputs=[upstream_input("transcript", "asr", "transcript")]),
            job("sentiment", JobType.TEXT_SENTIMENT, requires=["asr"], inputs=[upstream_input("transcript", "asr", "transcript")]),
        ])
        return {"conversation_id": conversation_id, "audio": audio, "graph": graph, "ids": self.ids(graph)}


@pytest.fixture
def hooks(monkeypatch) -> Hooks:
    recorder = Hooks()
    monkeypatch.setattr(projections, "apply_completion", recorder.apply_completion)
    monkeypatch.setattr(projections, "on_job_failed", recorder.on_job_failed)
    monkeypatch.setattr(projections, "on_conversation_registered", recorder.on_conversation_registered)
    return recorder


@pytest.fixture
def rubrics(monkeypatch):
    def rubric_version(conn, rubric_id, version):
        if rubric_id != RUBRIC_ID or version != 1:
            return None
        return RubricVersion(ref=RUBRIC_REF, definition=DEFINITION, status=RubricVersionStatus.ACTIVE, published_at="2026-09-01T00:00:00Z")

    def draft_snapshot_content(conn, rubric_id, *, expected_draft_revision):
        from call1.contracts.errors import ErrorCode
        from call1.store.errors import StoreError

        if rubric_id != RUBRIC_ID:
            raise StoreError(ErrorCode.NOT_FOUND, "no draft")
        if expected_draft_revision != 1:
            raise StoreError(ErrorCode.RUBRIC_VERSION_CONFLICT, "moved", details={"current_version": 1, "draft_revision": 1})
        return RubricSnapshotContent(source="draft", rubric_id=RUBRIC_ID, draft_revision=1, digest=canonical_digest(DRAFT_DEFINITION), definition=DRAFT_DEFINITION)

    monkeypatch.setattr(results_api, "rubric_snapshot_content", lambda conn, rubric_id, version: snapshot_content(rubric_id, version))
    monkeypatch.setattr(results_api, "rubric_version", rubric_version)
    monkeypatch.setattr(results_api, "draft_snapshot_content", draft_snapshot_content)
    monkeypatch.setattr(results_api, "result_groups", fake_result_groups)


@pytest.fixture
def q(client, store, clock, service_key, mint_service_key, mint_session, hooks, rubrics) -> QueueHarness:
    return QueueHarness(client, store, clock, service_key, mint_service_key, mint_session)
