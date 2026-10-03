"""Payload types of the cross-area hooks (owned by core; both sides import them).

The queue area calls the results area's projection hooks (``call1/store/results/projections.py``)
*inside* its own transactions, so a completion's outputs, usage, receipt, projection, dependent
release and change events commit together (architecture rule 6). These dataclasses are the
arguments. They are Store-internal (not contract models) and change only by agreement between the
queue and results owners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

from call1.contracts.artifacts import Artifact
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobStatus, JobType, ResultPublication


@dataclass(frozen=True)
class JobSnapshot:
    """The job row as the hook sees it, inside the transaction that changed it."""

    job_id: str
    conversation_id: str
    call_id: Optional[str]
    graph_id: str
    graph_created_at: datetime
    job_type: JobType
    status: JobStatus
    attempt_count: int
    draft_test_request_id: Optional[str] = None
    """Set when the graph fulfils a qa_draft_test request: never project it as the call's result."""


@dataclass(frozen=True)
class LinkedOutput:
    """One output the completion just linked (it now has its per-slot version)."""

    role: str
    artifact: Artifact


@dataclass(frozen=True)
class CompletedJob:
    job: JobSnapshot
    attempt_number: int
    receipt_id: str
    outputs: Tuple[LinkedOutput, ...]
    result: Optional[ResultPublication]
    """``CompletionRequest.result``: present exactly for the group's publishing job type."""
    completed_at: datetime
    superseded_artifact_ids: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProjectionOutcome:
    """What the projection tells the completion receipt."""

    result_version: Optional[int] = None
    """``CompletionReceipt.result_version``: the version the projection published, if any."""


@dataclass(frozen=True)
class FailedJob:
    """A job attempt or job ended without a result: /fail (retry scheduled or terminal), lease
    expiry, a release with ``reject``, admission rejection, or cancel (each cascaded job too)."""

    job: JobSnapshot
    attempt_number: Optional[int]
    status: JobStatus
    """The job's status after the change: QUEUED (retry scheduled), FAILED or CANCELLED."""
    error_code: Optional[JobErrorCode]
    terminal: bool
    occurred_at: datetime
