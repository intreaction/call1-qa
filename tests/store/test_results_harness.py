"""Shared harness for the results-area tests (no tests here).

The queue and auth areas are built in parallel, so these tests stand in for them exactly where the
brief allows: ``FakeQueue`` monkeypatches ``call1.store.queue.api`` (conversations, group stakes,
pending work, linked artifacts, reanalysis requests) and drives the real projection hooks the way
the queue's completion and failure transactions do; ``FakeAccounts`` monkeypatches
``call1.store.auth.api`` with the accounts behind the minted test sessions.

The real ``auth.api`` works now. ``FakeAccounts`` stays because these tests pick the assignment
pool explicitly (``accounts.add``): with real accounts every minted session, supervisors and admins
included, would join it. Distribution over real accounts is covered in ``test_store_integration.py``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from call1.contracts.artifacts import ARTIFACT_CONTENT_CONTRACTS, Artifact, ArtifactKind, ArtifactStorage, Sensitivity
from call1.contracts.auth import AccountStatus, ReviewerAccount
from call1.contracts.calls import (
    CallMetadata,
    Conversation,
    GroupGraphStake,
    IngestionKind,
    PendingWorkIndicator,
    PublisherState,
    SourceKind,
    SourceReference,
)
from call1.contracts.common import ArtifactRef, canonical_json
from call1.contracts.contents import (
    AudioValidationContent,
    EmbeddingsContent,
    PiiFindingsContent,
    PiiSpanContent,
    QaScorecardContent,
    ResultKind,
    ScorecardRubricRef,
    SpeakerRole,
    TranscriptContent,
    TranscriptTurnContent,
    TurnPiiFindings,
    TurnEmbedding,
    VadMetricsContent,
    VerdictStatus,
    VerdictView,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JOB_TYPE_RULES, JobStatus, JobType, ReanalysisKind, ReanalysisRequest, ReanalysisStatus, ResultPublication
from call1.store import db
from call1.store.auth import api as auth_api
from call1.store.hooks import CompletedJob, FailedJob, JobSnapshot, LinkedOutput
from call1.store.ids import new_id
from call1.store.queue import api as queue_api
from call1 import embedding
from call1.store.results import projections

DEFAULT_RUBRIC_ID = "call1_standard_v2"

_SENSITIVITY = {
    ArtifactKind.SOURCE_AUDIO: Sensitivity.RAW,
    ArtifactKind.VALIDATION_REPORT: Sensitivity.DERIVED,
    ArtifactKind.VAD_METRICS: Sensitivity.DERIVED,
    ArtifactKind.EMBEDDINGS: Sensitivity.DERIVED,
    ArtifactKind.PROMPT_INPUT: Sensitivity.DERIVED,
}


@dataclass
class FakeJob:
    job_id: str
    job_type: JobType
    graph_id: str
    status: JobStatus = JobStatus.QUEUED
    dead_blocked: bool = False
    error_code: Optional[JobErrorCode] = None
    ended_at: int = 0


@dataclass
class FakeGraph:
    graph_id: str
    created_at: datetime
    draft_test_request_id: Optional[str] = None
    jobs: List[FakeJob] = field(default_factory=list)


@dataclass
class FakeConversation:
    conversation: Conversation
    graphs: List[FakeGraph] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    pending_reanalysis: set = field(default_factory=set)
    requests: List[ReanalysisRequest] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.conversation.id

    @property
    def call_id(self) -> str:
        return self.conversation.call_id


class FakeQueue:
    def __init__(self, store, clock, monkeypatch) -> None:
        self.store = store
        self.clock = clock
        self.conversations: Dict[str, FakeConversation] = {}
        self._versions: Dict[Tuple[str, str, str], int] = {}
        self._seq = itertools.count(1)
        self.auto_pii = True
        """Link empty PII findings (contract 1.2.0) for every transcript an ASR completion publishes, as
        if the ``enrichment`` job had run. Tests of the fail-closed path turn it off and link findings
        themselves with ``link_pii``."""
        for name in ("get_conversation", "get_conversation_by_call", "get_artifact", "list_linked_artifacts", "group_stakes",
                     "reanalysis_pending_groups", "pending_work", "create_reanalysis_request", "failure_codes",
                     "reanalysis_request_for_group", "job_type_active"):
            monkeypatch.setattr(queue_api, name, getattr(self, name))

    # --- the queue.api surface ---------------------------------------------------------------

    def get_conversation(self, conn, conversation_id):
        found = self.conversations.get(conversation_id)
        return found.conversation if found else None

    def get_conversation_by_call(self, conn, call_id):
        return next((c.conversation for c in self.conversations.values() if c.call_id == call_id), None)

    def get_artifact(self, conn, artifact_id):
        return next((a for c in self.conversations.values() for a in c.artifacts if a.id == artifact_id), None)

    def list_linked_artifacts(self, conn, conversation_id, *, kind=None, slot=None, include_superseded=False, include_draft_tests=False):
        arts = [a for a in self.conversations[conversation_id].artifacts if a.linked and (kind is None or a.kind is kind)
                and (slot is None or a.slot == slot)]
        if not include_superseded:
            newest: Dict[Tuple[str, str], Artifact] = {}
            for art in arts:
                key = (art.kind.value, art.slot)
                if key not in newest or (art.version or 0) > (newest[key].version or 0):
                    newest[key] = art
            arts = list(newest.values())
        return arts

    def group_stakes(self, conn, conversation_id):
        out: Dict[ResultKind, List[GroupGraphStake]] = {}
        for graph in self.conversations[conversation_id].graphs:
            for kind in ResultKind:
                jobs = [j for j in graph.jobs if JOB_TYPE_RULES[j.job_type].group is kind]
                if not jobs:
                    continue
                publisher = next((j for j in jobs if JOB_TYPE_RULES[j.job_type].publishes is kind), None)
                state = None
                if publisher is not None:
                    if publisher.status is JobStatus.SUCCEEDED:
                        state = PublisherState.SUCCEEDED
                    elif publisher.status in (JobStatus.FAILED, JobStatus.CANCELLED) or publisher.dead_blocked:
                        state = PublisherState.ENDED_WITHOUT_RESULT
                    else:
                        state = PublisherState.IN_PROGRESS
                out.setdefault(kind, []).append(GroupGraphStake(graph_id=graph.graph_id, graph_created_at=graph.created_at,
                                                                draft_test=graph.draft_test_request_id is not None,
                                                                has_group_jobs=True, publisher=state))
        return out

    def failure_codes(self, conn, conversation_id):
        """The newest live graph's publisher ended without a result: its code, else the graph's first failure."""
        out = {kind: None for kind in ResultKind}
        live = [g for g in self.conversations[conversation_id].graphs if g.draft_test_request_id is None]
        for kind in ResultKind:
            graphs = [g for g in live if any(JOB_TYPE_RULES[j.job_type].publishes is kind for j in g.jobs)]
            if not graphs:
                continue
            graph = max(graphs, key=lambda g: g.created_at)
            publisher = next(j for j in graph.jobs if JOB_TYPE_RULES[j.job_type].publishes is kind)
            if publisher.status in (JobStatus.FAILED, JobStatus.CANCELLED) or publisher.dead_blocked:
                failed = sorted((j for j in graph.jobs if j.error_code is not None), key=lambda j: j.ended_at)
                out[kind] = publisher.error_code or (failed[0].error_code if failed else None)
        return out

    def reanalysis_request_for_group(self, conn, conversation_id, kind):
        from call1.contracts.jobs import REANALYSIS_KIND_AFFECTS
        for request in reversed(self.conversations[conversation_id].requests):
            if kind in REANALYSIS_KIND_AFFECTS[request.kind]:
                return request.id
        return None

    def job_type_active(self, conn, conversation_id, job_type):
        return any(j.job_type is job_type and not j.dead_blocked and j.status not in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)
                   for g in self.conversations[conversation_id].graphs if g.draft_test_request_id is None for j in g.jobs)

    def reanalysis_pending_groups(self, conn, conversation_id):
        return frozenset(self.conversations[conversation_id].pending_reanalysis)

    def pending_work(self, conn, conversation_id):
        jobs = [j for g in self.conversations[conversation_id].graphs if g.draft_test_request_id is None for j in g.jobs]
        count = lambda s: sum(1 for j in jobs if j.status is s)  # noqa: E731
        blocked = count(JobStatus.BLOCKED)
        return PendingWorkIndicator(jobs_total=len(jobs), jobs_succeeded=count(JobStatus.SUCCEEDED), jobs_running=count(JobStatus.RUNNING),
                                    jobs_queued=count(JobStatus.QUEUED), jobs_blocked=blocked, jobs_failed=count(JobStatus.FAILED),
                                    jobs_cancelled=count(JobStatus.CANCELLED),
                                    settled=all(j.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED) or j.dead_blocked for j in jobs))

    def create_reanalysis_request(self, conn, *, conversation_id, kind, requested_by, idempotency_key=None, speaker_correction=None, reason=None):
        conv = self.conversations[conversation_id]
        request = ReanalysisRequest(id=new_id("rq"), call_id=conv.call_id, conversation_id=conversation_id, kind=kind,
                                    speaker_correction=speaker_correction, note=reason, status=ReanalysisStatus.PENDING,
                                    requested_by_account_id=requested_by.account_id, requested_at=conn.now(),
                                    idempotency_key=idempotency_key or new_id("idem"))
        conv.requests.append(request)
        from call1.contracts.jobs import REANALYSIS_KIND_AFFECTS
        conv.pending_reanalysis |= set(REANALYSIS_KIND_AFFECTS[ReanalysisKind(kind)])
        return request

    # --- driving the hooks ---------------------------------------------------------------------

    def register(self, *, agent_id: str = "agent-7", external_call_ref: Optional[str] = None) -> FakeConversation:
        conv_id, call_id = new_id("conv"), new_id("call")
        digest = "sha256:" + format(next(self._seq), "064x")
        conversation = Conversation(
            id=conv_id, ingestion_kind=IngestionKind.CALL_AUDIO,
            source=SourceReference(kind=SourceKind.API_UPLOAD, content_digest=digest, received_at=self.clock.now()),
            call_id=call_id, call_metadata=CallMetadata(agent_id=agent_id, external_call_ref=external_call_ref),
            created_at=self.clock.now(),
        )
        fake = FakeConversation(conversation)
        self.conversations[conv_id] = fake
        with self.store.connection() as conn, db.transaction(conn):
            projections.on_conversation_registered(conn, conversation)
        self.clock.advance(1)
        return fake

    def graph(self, conv: FakeConversation, job_types: Sequence[JobType], *, draft_test_request_id: Optional[str] = None) -> FakeGraph:
        graph = FakeGraph(new_id("grf"), self.clock.now(), draft_test_request_id)
        for job_type in job_types:
            graph.jobs.append(FakeJob(new_id("job"), job_type, graph.graph_id, JobStatus.QUEUED))
        conv.graphs.append(graph)
        self.clock.advance(1)
        return graph

    def job(self, graph: FakeGraph, job_type: JobType) -> FakeJob:
        return next(j for j in graph.jobs if j.job_type is job_type and j.status is not JobStatus.SUCCEEDED) \
            if any(j.job_type is job_type and j.status is not JobStatus.SUCCEEDED for j in graph.jobs) \
            else next(j for j in graph.jobs if j.job_type is job_type)

    def _snapshot(self, conv: FakeConversation, graph: FakeGraph, job: FakeJob) -> JobSnapshot:
        return JobSnapshot(job_id=job.job_id, conversation_id=conv.id, call_id=conv.call_id, graph_id=graph.graph_id,
                           graph_created_at=graph.created_at, job_type=job.job_type, status=job.status, attempt_count=1,
                           draft_test_request_id=graph.draft_test_request_id)

    def artifact(self, conv: FakeConversation, kind: ArtifactKind, payload, *, slot: str = "", job_id: Optional[str] = None,
                 content_type: str = "application/json") -> Artifact:
        data = payload if isinstance(payload, bytes) else canonical_json(payload.model_dump(mode="json"))
        stored = self.store.objects.put_bytes(data)
        key = (conv.id, kind.value, slot)
        self._versions[key] = self._versions.get(key, 0) + 1
        art = Artifact(kind=kind, slot=slot, content_type=content_type, size_bytes=stored.size_bytes, checksum=stored.checksum,
                       content_contract=ARTIFACT_CONTENT_CONTRACTS[kind], sensitivity=_SENSITIVITY.get(kind, Sensitivity.RAW),
                       producing_job_id=job_id, id=new_id("art"), conversation_id=conv.id, linked=True,
                       linked_by_receipt_id=new_id("rcpt") if job_id else None, version=self._versions[key],
                       storage=ArtifactStorage.OBJECT, committed_at=self.clock.now())
        conv.artifacts.append(art)
        return art

    def complete(self, conv: FakeConversation, graph: FakeGraph, job_type: JobType, outputs: Dict[str, object], *,
                 result: Optional[ResultPublication] = None, slots: Optional[Dict[str, str]] = None) -> Optional[int]:
        job = self.job(graph, job_type)
        rule = JOB_TYPE_RULES[job_type]
        linked = []
        for role, payload in outputs.items():
            slot = (slots or {}).get(role, "")
            if graph.draft_test_request_id:
                slot = f"draft:{graph.draft_test_request_id}:{slot}"
            kind = {**rule.outputs, **rule.optional_outputs}[role]  # optional roles (1.3.0): the asr job's base_transcript, vocabulary_pass
            linked.append(LinkedOutput(role=role, artifact=self.artifact(conv, kind, payload, slot=slot, job_id=job.job_id)))
        if result is None and rule.publishes is not None and not graph.draft_test_request_id:
            result = ResultPublication(kind=rule.publishes)
        job.status = JobStatus.SUCCEEDED
        if self.auto_pii and job_type is JobType.ASR and not graph.draft_test_request_id:
            for output in linked:
                if output.artifact.kind is ArtifactKind.TRANSCRIPT and output.artifact.slot == "":
                    self.link_pii(conv, output.artifact)
        completion = CompletedJob(job=self._snapshot(conv, graph, job), attempt_number=1, receipt_id=new_id("rcpt"), outputs=tuple(linked),
                                  result=result, completed_at=self.clock.now())
        with self.store.connection() as conn, db.transaction(conn):
            outcome = projections.apply_completion(conn, completion)
        self.clock.advance(1)
        return outcome.result_version

    def link_pii(self, conv: FakeConversation, transcript_artifact: Artifact, spans: Optional[Dict[int, list]] = None) -> Artifact:
        """Link ``pii_findings`` bound to ``transcript_artifact`` (``spans``: turn_id -> [(category, text, start)])
        straight into the results index, as the enrichment job's completion would."""
        turns = [TurnPiiFindings(turn_id=turn_id, spans=[PiiSpanContent(start=start, end=start + len(text), category=category, text=text)
                                                         for category, text, start in found])
                 for turn_id, found in sorted((spans or {}).items())]
        findings = PiiFindingsContent(transcript=ArtifactRef(artifact_id=transcript_artifact.id, checksum=transcript_artifact.checksum),
                                      detector="stub", detector_revision="stub-v1", turns=turns)
        art = self.artifact(conv, ArtifactKind.PII_FINDINGS, findings)
        with self.store.connection() as conn, db.transaction(conn):
            conn.execute(
                "INSERT INTO results_artifacts (artifact_id, conversation_id, kind, slot, version, checksum, content_type, job_id, graph_id, committed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)",
                (art.id, conv.id, art.kind.value, art.slot, art.version, art.checksum, art.content_type, db.ts(self.clock.now())))
        return art

    def fail(self, conv: FakeConversation, graph: FakeGraph, job_type: JobType, code: JobErrorCode = JobErrorCode.PROVIDER_ERROR, *,
             terminal: bool = True, status: JobStatus = JobStatus.FAILED) -> None:
        job = self.job(graph, job_type)
        job.status = status if terminal else JobStatus.QUEUED
        if terminal:
            job.error_code, job.ended_at = code, next(self._seq)
        failure = FailedJob(job=self._snapshot(conv, graph, job), attempt_number=1, status=job.status, error_code=code, terminal=terminal,
                            occurred_at=self.clock.now())
        with self.store.connection() as conn, db.transaction(conn):
            projections.on_job_failed(conn, failure)
        self.clock.advance(1)

    def dead_block(self, graph: FakeGraph, job_type: JobType) -> None:
        job = self.job(graph, job_type)
        job.status = JobStatus.BLOCKED
        job.dead_blocked = True

    # --- content builders ----------------------------------------------------------------------

    def ingest_qa(self, conv: FakeConversation, verdicts: Sequence[Tuple[str, VerdictStatus, float]], **scorecard) -> Tuple[FakeGraph, int]:
        graph = self.graph(conv, [JobType.QA_SCORECARD])
        version = self.complete(conv, graph, JobType.QA_SCORECARD, {"scorecard": qa_scorecard(verdicts, evaluated_at=self.clock.now(), **scorecard)})
        return graph, version


class FakeAccounts:
    def __init__(self, monkeypatch) -> None:
        self.accounts: Dict[str, ReviewerAccount] = {}
        monkeypatch.setattr(auth_api, "get_account", lambda conn, account_id: self.accounts.get(account_id))
        monkeypatch.setattr(auth_api, "list_accounts", lambda conn, role=None, status=None: [
            a for a in self.accounts.values() if (role is None or a.role is role) and (status is None or a.status is status)])

    def add(self, session) -> None:
        p = session.principal
        self.accounts[p.account_id] = ReviewerAccount(id=p.account_id, email=p.email, display_name=p.display_name, role=p.role,
                                                      status=AccountStatus.ACTIVE, created_at=datetime(2026, 9, 1, tzinfo=timezone.utc), authenticator_count=1)


# --- content factories ------------------------------------------------------------------------


def transcript(turns: Sequence[Tuple[str, str]], *, word_turns: Optional[Dict[int, list]] = None) -> TranscriptContent:
    out = []
    t = 0.0
    for i, (speaker, text) in enumerate(turns):
        out.append(TranscriptTurnContent(turn_id=i, speaker=SpeakerRole(speaker), start_time=t, end_time=t + 2.0, text=text,
                                         word_timestamps=(word_turns or {}).get(i)))
        t += 2.0
    return TranscriptContent(duration_seconds=t, language="en", is_redacted=False, turns=out)


def qa_scorecard(verdicts: Sequence[Tuple[str, VerdictStatus, float]], *, rubric_id: str = DEFAULT_RUBRIC_ID, rubric_version: int = 1,
                 digest: Optional[str] = None, overall_score: float = 90.0, passed: bool = True, critical_failure: bool = False,
                 requires_human_review: bool = False, evaluated_at: Optional[datetime] = None, evidence: str = "thanks for calling") -> QaScorecardContent:
    return QaScorecardContent(
        rubric=ScorecardRubricRef(rubric_id=rubric_id, rubric_version=rubric_version, digest=digest or "sha256:" + "0" * 64),
        overall_score=overall_score, passed=passed, critical_failure=critical_failure, requires_human_review=requires_human_review,
        verdicts=[VerdictView(criterion_id=cid, criterion_name=f"Criterion {cid}", status=status, confidence=conf, reasoning=f"because {evidence}",
                              quoted_evidence=evidence) for cid, status, conf in verdicts],
        evaluated_at=evaluated_at or datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc),
    )


def validation_report(duration: float = 12.0) -> AudioValidationContent:
    from call1.contracts.contents import AudioChannelLayout
    return AudioValidationContent(container="wav", codec="pcm_s16le", sample_rate=8000, channels=1, channel_layout=AudioChannelLayout.MONO,
                                  duration_seconds=duration)


def vad_metrics() -> VadMetricsContent:
    return VadMetricsContent(total_speech_duration=10.0, total_silence_duration=2.0, silence_ratio=0.17, overtalk_duration=0.5, overtalk_ratio=0.04)


def embeddings_for(content: TranscriptContent) -> EmbeddingsContent:
    """What a fake-handler Process writes: the fake search embedder's vectors (the suite's Store uses
    the same backend, tests/conftest.py)."""
    fake = embedding.get_embedder("fake")
    return EmbeddingsContent(scheme=fake.scheme, dimensions=fake.dimensions, turn_vectors=[
        TurnEmbedding(turn_id=t.turn_id, vector=v) for t, v in zip(content.turns, fake.embed_documents([t.text for t in content.turns]))])


def legacy_embeddings_for(content: TranscriptContent, dims: int = 8) -> EmbeddingsContent:
    """An artifact from before contract 1.2.0: scheme ``hashing-projection-v1``, never ranked."""
    return EmbeddingsContent(scheme="hashing-projection-v1", dimensions=dims, turn_vectors=[
        TurnEmbedding(turn_id=t.turn_id, vector=[1.0] + [0.0] * (dims - 1)) for t in content.turns])


@pytest.fixture
def accounts(monkeypatch) -> FakeAccounts:
    return FakeAccounts(monkeypatch)


@pytest.fixture
def fq(store, clock, monkeypatch, accounts) -> FakeQueue:
    return FakeQueue(store, clock, monkeypatch)


def later(clock, seconds: float = 1) -> datetime:
    return clock.advance(seconds)


__all__ = ["FakeQueue", "FakeAccounts", "transcript", "qa_scorecard", "validation_report", "vad_metrics", "embeddings_for", "legacy_embeddings_for", "accounts", "fq",
           "DEFAULT_RUBRIC_ID", "timedelta"]
