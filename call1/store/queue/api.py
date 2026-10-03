"""Functions the queue area offers the other Store areas.

Other areas import only this module from ``call1.store.queue``, never its tables or internals.
Every function takes the caller's connection and runs inside the caller's transaction when there
is one (writes nest as a SAVEPOINT). The first eight are the agreed interface; the rest are
additions the results area may use: ``failure_codes``, ``reanalysis_request_for_group``,
``graph_created_at``, ``conversation_settled``, ``job_type_active``, (1.3.0) ``latest_compare_preview`` and (1.3.0, on-device
training) ``get_job``. Artifact bytes come from the object store:
``store.objects.read_bytes(artifact.checksum)``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, FrozenSet, List, Optional

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.calls import Conversation, GroupGraphStake, PendingWorkIndicator
from call1.contracts.contents import ResultKind
from call1.contracts.errors import JobErrorCode
from call1.contracts.events import Actor
from call1.contracts.jobs import TERMINAL_STATUSES, Job, JobType, ReanalysisKind, ReanalysisRequest, SpeakerCorrection

from .. import db
from ..db import StoreConnection
from ..errors import not_found
from . import progress, reanalysis
from .artifacts import list_linked
from .records import ConversationJobs, JobReader, artifact_from_row, artifact_row, conversation_by_call_row, conversation_from_row, conversation_row, job_row


def get_conversation(conn: StoreConnection, conversation_id: str) -> Optional[Conversation]:
    row = conversation_row(conn, conversation_id)
    return None if row is None else conversation_from_row(row)


def get_conversation_by_call(conn: StoreConnection, call_id: str) -> Optional[Conversation]:
    row = conversation_by_call_row(conn, call_id)
    return None if row is None else conversation_from_row(row)


def get_artifact(conn: StoreConnection, artifact_id: str) -> Optional[Artifact]:
    """Any committed artifact (linked or not). Its bytes: ``store.objects.open(artifact.checksum)``."""
    row = artifact_row(conn, artifact_id)
    return None if row is None else artifact_from_row(row)


def list_linked_artifacts(conn: StoreConnection, conversation_id: str, *, kind: Optional[ArtifactKind] = None,
                          slot: Optional[str] = None, include_superseded: bool = False,
                          include_draft_tests: bool = False) -> List[Artifact]:
    """Linked artifacts of a conversation (current versions unless ``include_superseded``),
    ordered by kind, slot and version."""
    return list_linked(conn, conversation_id, kind=kind, slot=slot, include_superseded=include_superseded, include_draft_tests=include_draft_tests)


def group_stakes(conn: StoreConnection, conversation_id: str) -> Dict[ResultKind, List[GroupGraphStake]]:
    """Per result group, what each job graph contributes (``calls.result_state_inputs``' input).
    Every ``ResultKind`` is a key; draft-test graphs are included with ``draft_test=True``."""
    return progress.group_stakes(conn, conversation_id)


def reanalysis_pending_groups(conn: StoreConnection, conversation_id: str) -> FrozenSet[ResultKind]:
    """Groups a pending or claimed reanalysis request affects (``jobs.REANALYSIS_KIND_AFFECTS``)."""
    return reanalysis.pending_groups(conn, conversation_id)


def pending_work(conn: StoreConnection, conversation_id: str) -> PendingWorkIndicator:
    """Job counts outside draft-test graphs, and whether the conversation is settled."""
    return progress.pending_work(conn, conversation_id)


def create_reanalysis_request(conn: StoreConnection, *, conversation_id: str, kind: ReanalysisKind, requested_by: Actor,
                              idempotency_key: Optional[str] = None, speaker_correction: Optional[SpeakerCorrection] = None,
                              reason: Optional[str] = None) -> ReanalysisRequest:
    """Create a durable reanalysis request in the caller's transaction and append its change
    event. Used by the results area's ``correctSpeaker`` (kind ``speaker_correction``). With an
    ``idempotency_key`` a repeat returns the same request (scoped to the actor and call)."""
    row = conversation_row(conn, conversation_id)
    if row is None:
        raise not_found("Conversation", conversation_id=conversation_id)
    return reanalysis.create_for_area(conn, conversation_row=row, kind=ReanalysisKind(kind), requested_by=requested_by,
                                      idempotency_key=idempotency_key, speaker_correction=speaker_correction, reason=reason)


# --- additions (offered to the results area) --------------------------------------------------


def failure_codes(conn: StoreConnection, conversation_id: str) -> Dict[ResultKind, Optional[JobErrorCode]]:
    """``ResultGroup.failure_code`` per group: when the newest live graph's publisher ended without
    a result, its error code or the first ended upstream's (draft-test graphs never count)."""
    return progress.group_failure_codes(conn, conversation_id)


def reanalysis_request_for_group(conn: StoreConnection, conversation_id: str, kind: ResultKind) -> Optional[str]:
    """``ResultGroup.reanalysis_request_id``: the newest pending or claimed request affecting ``kind``."""
    return reanalysis.active_request_for_group(conn, conversation_id, kind)


def graph_created_at(conn: StoreConnection, graph_id: str) -> Optional[datetime]:
    """When a graph was created (``result_state_inputs``' ``published_graph_created_at``)."""
    row = conn.execute("SELECT created_at FROM q_graphs WHERE id = ?", (graph_id,)).fetchone()
    return None if row is None else db.parse_ts(row["created_at"])


def conversation_settled(conn: StoreConnection, conversation_id: str) -> bool:
    return ConversationJobs(conn, conversation_id).settled()


def job_type_active(conn: StoreConnection, conversation_id: str, job_type: JobType) -> bool:
    """Whether a live (non-draft-test) job of ``job_type`` for the conversation can still finish:
    not terminal and not dead-blocked behind a failed or cancelled upstream."""
    view = ConversationJobs(conn, conversation_id)
    return any(n.job_type is job_type and not n.draft_test and n.status not in TERMINAL_STATUSES and not view.dead(n.id)
               for n in view.nodes.values())


def latest_compare_preview(conn: StoreConnection, call_id: str) -> Optional[str]:
    """Contract 1.3.0: the newest compare preview with an available v2 result for the call
    (``ContactSignalsView.comparison_preview_id``, admins only)."""
    from .signals import latest_compare_preview as latest

    return latest(conn, call_id)


def get_job(conn: StoreConnection, job_id: str) -> Optional[Job]:
    """Contract 1.3.0 (on-device training): any job as ``getJob`` returns it (resolved inputs, outputs,
    parameters), or None. The results area's training-label reads resolve a label's source job and
    artifacts through this (docs/OnDeviceTraining.md section 2.3)."""
    row = job_row(conn, job_id)
    return None if row is None else JobReader(conn).job(row)


def import_demo_snapshot(conn, store, registration, contents):
    """Host-only synthetic history import. Linked snapshots have no fabricated worker jobs.

    Results owns the projections; queue owns registration and artifact validation/storage.
    The outer transaction must include both. Normal installs cannot use this entry point.
    """
    if not store.config.demo_mode:
        raise ValueError("Synthetic sessions require CALL1_STORE_DEMO=1")
    from call1.contracts.artifacts import ARTIFACT_CONTENT_CONTRACTS, InlineArtifactCreate, Sensitivity
    from call1.contracts.common import ArtifactRef, canonical_digest, canonical_json
    from call1.contracts.contents import PiiFindingsContent
    from ..principals import ServiceKeyPrincipal
    from .reads import register_conversation
    from .artifacts import create_inline

    principal = ServiceKeyPrincipal(key_id="demo-history", installation_id="demo-history", scopes=frozenset())
    registered = register_conversation(conn, principal, registration)
    if not registered.created:
        return registered.conversation, []
    artifacts = []
    contents = list(contents)
    for kind, model in contents:
        payload = model.model_dump(mode="json")
        artifacts.append(create_inline(conn, store, registered.conversation.id, InlineArtifactCreate(
            kind=kind, content_type="application/json", content_contract=ARTIFACT_CONTENT_CONTRACTS[kind],
            sensitivity=Sensitivity.RAW if kind is ArtifactKind.PII_FINDINGS else Sensitivity.DERIVED, checksum=canonical_digest(payload), size_bytes=len(canonical_json(payload)), payload=payload,
        )))
        if kind is ArtifactKind.TRANSCRIPT:
            transcript = artifacts[-1]
            contents.append((ArtifactKind.PII_FINDINGS, PiiFindingsContent(
                transcript=ArtifactRef(artifact_id=transcript.id, checksum=transcript.checksum),
                detector="synthetic-demo-fixture", detector_revision="v1", turns=[])))
    return registered.conversation, artifacts
