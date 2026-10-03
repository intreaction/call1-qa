"""Queue rows to contract models, and the in-memory dependency view of one conversation.

Everything here reads; nothing writes. ``ConversationJobs`` loads a conversation's jobs and edges
once and answers the questions the rest of the area asks repeatedly: is a job dead-blocked, why is
it blocked, is it ready, where does a group's publisher stand.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from call1.contracts.artifacts import Artifact
from call1.contracts.calls import Conversation, PublisherState
from call1.contracts.catalog import FrozenSelection
from call1.contracts.errors import ErrorCode, JobErrorCode
from call1.contracts.jobs import (
    JOB_TYPE_RULES,
    TERMINAL_STATUSES,
    Attempt,
    AttemptProvenance,
    AttemptStatus,
    BlockingReason,
    EdgeKind,
    Job,
    JobInput,
    JobOutput,
    JobParameters,
    JobStatus,
    JobType,
    LeaseInfo,
    ReanalysisRequest,
    ReanalysisStatus,
    ResolvedInputRef,
    ResourceEstimate,
    WaitingReason,
)
from call1.contracts.rubrics import DraftRubricRef, RubricVersionRef
from call1.contracts.jobs import SpeakerCorrection
from call1.contracts.usage import UsageRecord

from .. import db
from ..errors import StoreError, not_found

ENDED_WITHOUT_RESULT = frozenset({JobStatus.FAILED, JobStatus.CANCELLED})


# --- conversations ------------------------------------------------------------------------


def conversation_from_row(row) -> Conversation:
    return Conversation.model_validate({
        "id": row["id"],
        "ingestion_kind": row["ingestion_kind"],
        "source": db.loads(row["source_json"]),
        "call_id": row["call_id"],
        "call_metadata": db.loads(row["call_metadata_json"]),
        "created_at": row["created_at"],
        "registered_by_installation_id": row["registered_by_installation_id"],
    })


def conversation_row(conn, conversation_id: str):
    return conn.execute("SELECT * FROM q_conversations WHERE id = ?", (conversation_id,)).fetchone()


def require_conversation_row(conn, conversation_id: str):
    row = conversation_row(conn, conversation_id)
    if row is None:
        raise not_found("Conversation", conversation_id=conversation_id)
    return row


def conversation_by_call_row(conn, call_id: str):
    return conn.execute("SELECT * FROM q_conversations WHERE call_id = ?", (call_id,)).fetchone()


def require_call_row(conn, call_id: str):
    row = conversation_by_call_row(conn, call_id)
    if row is None:
        raise not_found("Call", call_id=call_id)
    return row


def call_id_of(conn, conversation_id: str) -> Optional[str]:
    row = conn.execute("SELECT call_id FROM q_conversations WHERE id = ?", (conversation_id,)).fetchone()
    return None if row is None else row["call_id"]


# --- artifacts ----------------------------------------------------------------------------


def artifact_from_row(row) -> Artifact:
    return Artifact.model_validate({
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "kind": row["kind"],
        "slot": row["slot"],
        "content_type": row["content_type"],
        "size_bytes": row["size_bytes"],
        "checksum": row["checksum"],
        "content_contract": row["content_contract"],
        "sensitivity": row["sensitivity"],
        "producing_job_id": row["producing_job_id"],
        "labels": db.loads(row["labels_json"]) or {},
        "linked": bool(row["linked"]),
        "linked_by_receipt_id": row["linked_by_receipt_id"],
        "version": row["version"],
        "storage": row["storage"],
        "committed_at": row["committed_at"],
        "superseded_by": row["superseded_by"],
    })


def artifact_row(conn, artifact_id: str):
    return conn.execute("SELECT * FROM q_artifacts WHERE id = ?", (artifact_id,)).fetchone()


def require_artifact_row(conn, artifact_id: str):
    row = artifact_row(conn, artifact_id)
    if row is None:
        raise not_found("Artifact", artifact_id=artifact_id)
    return row


# --- jobs -----------------------------------------------------------------------------------


def job_row(conn, job_id: str):
    return conn.execute("SELECT * FROM q_jobs WHERE id = ?", (job_id,)).fetchone()


def require_job_row(conn, job_id: str):
    row = job_row(conn, job_id)
    if row is None:
        raise not_found("Job", job_id=job_id)
    return row


def job_inputs(row) -> List[JobInput]:
    return [JobInput.model_validate(item) for item in db.loads(row["inputs_json"]) or []]


def job_outputs(row) -> List[JobOutput]:
    return [JobOutput.model_validate(item) for item in db.loads(row["outputs_json"]) or []]


def job_selection(row) -> Optional[FrozenSelection]:
    data = db.loads(row["selection_json"])
    return None if data is None else FrozenSelection.model_validate(data)


def job_parameters(row) -> JobParameters:
    return JobParameters.model_validate(db.loads(row["parameters_json"]) or {})


def job_resource_estimate(row) -> ResourceEstimate:
    return ResourceEstimate.model_validate(db.loads(row["resource_estimate_json"]))


@dataclass(frozen=True)
class JobNode:
    id: str
    status: JobStatus
    job_type: JobType
    graph_id: str
    error_code: Optional[JobErrorCode]
    draft_test: bool
    created_at: str


class ConversationJobs:
    """Every job and edge of one conversation, loaded once (no cross-conversation edges exist)."""

    def __init__(self, conn, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.nodes: Dict[str, JobNode] = {}
        for row in conn.execute(
            "SELECT id, status, job_type, graph_id, error_code, draft_test_request_id, created_at FROM q_jobs WHERE conversation_id = ?",
            (conversation_id,),
        ):
            self.nodes[row["id"]] = JobNode(
                id=row["id"], status=JobStatus(row["status"]), job_type=JobType(row["job_type"]), graph_id=row["graph_id"],
                error_code=JobErrorCode(row["error_code"]) if row["error_code"] else None,
                draft_test=row["draft_test_request_id"] is not None, created_at=row["created_at"],
            )
        self.upstreams: Dict[str, List[Tuple[str, EdgeKind]]] = {job_id: [] for job_id in self.nodes}
        self.downstreams: Dict[str, List[Tuple[str, EdgeKind]]] = {job_id: [] for job_id in self.nodes}
        for row in conn.execute(
            "SELECT e.job_id, e.upstream_job_id, e.kind FROM q_edges e JOIN q_jobs j ON j.id = e.job_id WHERE j.conversation_id = ?",
            (conversation_id,),
        ):
            kind = EdgeKind(row["kind"])
            self.upstreams.setdefault(row["job_id"], []).append((row["upstream_job_id"], kind))
            self.downstreams.setdefault(row["upstream_job_id"], []).append((row["job_id"], kind))
        self._dead: Dict[str, bool] = {}

    def dead(self, job_id: str) -> bool:
        """BLOCKED behind an upstream that cannot finish: a requires-edge upstream that is FAILED
        or CANCELLED, or any upstream that is itself dead-blocked (transitively)."""
        if job_id in self._dead:
            return self._dead[job_id]
        stack: List[Tuple[str, bool]] = [(job_id, False)]
        visiting: Set[str] = set()
        while stack:
            current, expanded = stack.pop()
            if current in self._dead:
                continue
            node = self.nodes.get(current)
            if node is None or node.status is not JobStatus.BLOCKED:
                self._dead[current] = False
                continue
            ups = self.upstreams.get(current, [])
            if not expanded:
                if current in visiting:  # a cycle cannot be created; treat defensively as not dead
                    self._dead[current] = False
                    continue
                visiting.add(current)
                stack.append((current, True))
                stack.extend((up, False) for up, _ in ups if up not in self._dead)
                continue
            result = False
            for up, kind in ups:
                upstream = self.nodes.get(up)
                if upstream is None:
                    continue
                if kind is EdgeKind.REQUIRES and upstream.status in ENDED_WITHOUT_RESULT:
                    result = True
                    break
                if self._dead.get(up, False):
                    result = True
                    break
            self._dead[current] = result
        return self._dead[job_id]

    @staticmethod
    def satisfied(status: JobStatus, kind: EdgeKind) -> bool:
        return status is JobStatus.SUCCEEDED if kind is EdgeKind.REQUIRES else status in TERMINAL_STATUSES

    def blocking(self, job_id: str) -> List[BlockingReason]:
        reasons = []
        for up, kind in self.upstreams.get(job_id, []):
            node = self.nodes[up]
            if self.satisfied(node.status, kind):
                continue
            dead = (kind is EdgeKind.REQUIRES and node.status in ENDED_WITHOUT_RESULT) or self.dead(up)
            reasons.append(BlockingReason(job_id=up, job_type=node.job_type, status=node.status, edge=kind, dead=dead))
        return reasons

    def publisher_state(self, job_id: str) -> PublisherState:
        node = self.nodes[job_id]
        if node.status is JobStatus.SUCCEEDED:
            return PublisherState.SUCCEEDED
        if node.status in ENDED_WITHOUT_RESULT or self.dead(job_id):
            return PublisherState.ENDED_WITHOUT_RESULT
        return PublisherState.IN_PROGRESS

    def failure_code(self, job_id: str) -> Optional[JobErrorCode]:
        """The job's own error code, or the first ended upstream's on its requires chain."""
        seen: Set[str] = set()
        queue = [job_id]
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            node = self.nodes.get(current)
            if node is None:
                continue
            if node.status in ENDED_WITHOUT_RESULT and node.error_code is not None:
                return node.error_code
            ups = sorted(self.upstreams.get(current, []), key=lambda item: (self.nodes[item[0]].created_at, item[0]))
            queue.extend(up for up, _ in ups)
        return None

    def settled(self) -> bool:
        return all(n.status in TERMINAL_STATUSES or self.dead(n.id) for n in self.nodes.values() if not n.draft_test)


def waiting_reason(row, cj: ConversationJobs, now: datetime) -> Optional[WaitingReason]:
    status = JobStatus(row["status"])
    if status is JobStatus.BLOCKED:
        return WaitingReason.DEAD_BLOCKED if cj.dead(row["id"]) else WaitingReason.WAITING_FOR_DEPENDENCIES
    if status is JobStatus.QUEUED:
        next_run = db.parse_ts(row["next_run_at"])
        if next_run is not None and next_run > now:
            return WaitingReason.DEFERRED_BY_WORKER if row["wait_cause"] == WaitingReason.DEFERRED_BY_WORKER.value else WaitingReason.RETRY_BACKOFF
        return WaitingReason.WAITING_FOR_WORKER
    return None


def job_from_row(row, cj: ConversationJobs, now: datetime) -> Job:
    status = JobStatus(row["status"])
    ups = cj.upstreams.get(row["id"], [])
    lease = None
    if status is JobStatus.RUNNING:
        lease = LeaseInfo(
            worker_id=row["lease_worker_id"], installation_id=row["lease_installation_id"], attempt_number=row["lease_attempt_number"],
            granted_at=db.parse_ts(row["lease_granted_at"]), expires_at=db.parse_ts(row["lease_expires_at"]),
            claim_token_hash=row["lease_claim_token_hash"],
        )
    return Job(
        id=row["id"],
        conversation_id=row["conversation_id"],
        graph_id=row["graph_id"],
        job_type=JobType(row["job_type"]),
        execution_class=row["execution_class"],
        status=status,
        priority=row["priority"],
        attempt_count=row["attempt_count"],
        claim_count=row["claim_count"],
        max_attempts=row["max_attempts"],
        retry_generation=row["retry_generation"],
        next_run_at=db.parse_ts(row["next_run_at"]) if status is JobStatus.QUEUED else None,
        waiting_reason=waiting_reason(row, cj, now),
        lease=lease,
        cancel_requested=bool(row["cancel_requested"]),
        blocking=cj.blocking(row["id"]) if status is JobStatus.BLOCKED else [],
        error_code=JobErrorCode(row["error_code"]) if row["error_code"] else None,
        error_detail=row["error_detail"],
        inputs=job_inputs(row),
        resolved_inputs=[ResolvedInputRef.model_validate(i) for i in db.loads(row["resolved_inputs_json"]) or []],
        requires_job_ids=[up for up, kind in ups if kind is EdgeKind.REQUIRES],
        after_job_ids=[up for up, kind in ups if kind is EdgeKind.AFTER],
        selection=job_selection(row),
        resource_estimate=job_resource_estimate(row),
        parameters=job_parameters(row),
        idempotency_key=row["idempotency_key"],
        outputs=job_outputs(row),
        result_version=row["result_version"],
        created_at=db.parse_ts(row["created_at"]),
        updated_at=db.parse_ts(row["updated_at"]),
        completed_at=db.parse_ts(row["completed_at"]),
    )


class JobReader:
    """Builds ``Job`` models, loading each conversation's dependency view once."""

    def __init__(self, conn) -> None:
        self.conn = conn
        self._views: Dict[str, ConversationJobs] = {}

    def view(self, conversation_id: str) -> ConversationJobs:
        if conversation_id not in self._views:
            self._views[conversation_id] = ConversationJobs(self.conn, conversation_id)
        return self._views[conversation_id]

    def job(self, row) -> Job:
        return job_from_row(row, self.view(row["conversation_id"]), self.conn.now())

    def job_by_id(self, job_id: str) -> Job:
        return self.job(require_job_row(self.conn, job_id))

    def jobs(self, rows: Iterable) -> List[Job]:
        return [self.job(row) for row in rows]


def load_job(conn, job_id: str) -> Job:
    return JobReader(conn).job_by_id(job_id)


# --- attempts -------------------------------------------------------------------------------


def attempt_from_row(row) -> Attempt:
    provenance = db.loads(row["provenance_json"])
    return Attempt(
        job_id=row["job_id"],
        attempt_number=row["attempt_number"],
        status=AttemptStatus(row["status"]),
        counts_as_attempt=bool(row["counts_as_attempt"]),
        started_at=db.parse_ts(row["started_at"]),
        ended_at=db.parse_ts(row["ended_at"]),
        worker_id=row["worker_id"],
        installation_id=row["installation_id"],
        claim_token_hash=row["claim_token_hash"],
        error_code=JobErrorCode(row["error_code"]) if row["error_code"] else None,
        error_detail=row["error_detail"],
        provenance=AttemptProvenance.model_validate(provenance) if provenance is not None else None,
        usage_record_id=row["usage_record_id"],
        resource_estimate=ResourceEstimate.model_validate(db.loads(row["resource_estimate_json"])),
    )


def usage_from_row(row) -> UsageRecord:
    return UsageRecord.model_validate(db.loads(row["record_json"]))


# --- graphs ---------------------------------------------------------------------------------


def graph_row(conn, graph_id: str):
    return conn.execute("SELECT * FROM q_graphs WHERE id = ?", (graph_id,)).fetchone()


# --- reanalysis requests --------------------------------------------------------------------


def effective_request_status(row, now: datetime) -> ReanalysisStatus:
    """A claimed request whose claim lease ran out is pending again (``claimed -> pending``)."""
    status = ReanalysisStatus(row["status"])
    if status is ReanalysisStatus.CLAIMED:
        expires = db.parse_ts(row["claim_expires_at"])
        if expires is not None and expires <= now:
            return ReanalysisStatus.PENDING
    return status


def request_from_row(row, now: datetime) -> ReanalysisRequest:
    status = effective_request_status(row, now)
    claimed = status is ReanalysisStatus.CLAIMED
    rubric = db.loads(row["rubric_json"])
    draft = db.loads(row["draft_rubric_json"])
    correction = db.loads(row["speaker_correction_json"])
    return ReanalysisRequest(
        id=row["id"],
        call_id=row["call_id"],
        conversation_id=row["conversation_id"],
        kind=row["kind"],
        rubric=RubricVersionRef.model_validate(rubric) if rubric else None,
        draft_rubric=DraftRubricRef.model_validate(draft) if draft else None,
        speaker_correction=SpeakerCorrection.model_validate(correction) if correction else None,
        note=row["note"],
        status=status,
        requested_by_account_id=row["requested_by_account_id"],
        requested_at=db.parse_ts(row["requested_at"]),
        claimed_by_installation_id=row["claimed_by_installation_id"] if claimed else None,
        claim_expires_at=db.parse_ts(row["claim_expires_at"]) if claimed else None,
        graph_id=row["graph_id"],
        draft_result_artifact_id=row["draft_result_artifact_id"],
        rejected_reason=row["rejected_reason"],
        idempotency_key=row["idempotency_key"],
        priority=int(row["priority"] or 0),
        rescore_signals=bool(row["rescore_signals"]),
        signal_taxonomy_version=row["signal_taxonomy_version"],
        signal_pipeline=row["signal_pipeline"],
        signal_backfill_id=row["signal_backfill_id"],
        signal_preview_id=row["signal_preview_id"],
        signal_taxonomy_snapshot_artifact_id=row["signal_taxonomy_snapshot_artifact_id"],
        preview_result_artifact_id=row["preview_result_artifact_id"],
    )


def request_row(conn, request_id: str):
    return conn.execute("SELECT * FROM q_reanalysis_requests WHERE id = ?", (request_id,)).fetchone()


def require_request_row(conn, request_id: str):
    row = request_row(conn, request_id)
    if row is None:
        raise not_found("Reanalysis request", request_id=request_id)
    return row


def graph_invalid(message: str, **details) -> StoreError:
    return StoreError(ErrorCode.GRAPH_INVALID, message, details=details)


def rule_for(job_type: JobType):
    return JOB_TYPE_RULES[JobType(job_type)]


def ids_sequence(values: Sequence[str]) -> str:
    return ",".join("?" for _ in values)
