"""Reanalysis requests and draft tests (Evaluate -> Store -> Process).

A request is durable: Evaluate creates it (header idempotency, scoped to the session and call),
Process claims it under ``REANALYSIS_CLAIM_LEASE`` and fulfils it by creating the graph that
carries its claim token (``lifecycle.create_graph``), so one request yields at most one graph.
``REANALYSIS_TRANSITIONS`` is the state machine; an expired claim returns to ``pending``.

A draft test (``testRubricDraft``) snapshots the stored draft into ``draft:<request_id>:...`` in the
same transaction and creates a ``qa_draft_test`` request; its scorecard is recorded on the request
(``draft_result_artifact_id``) and never projected as the call's QA.

Contract 1.3.0 (Contact Signals v2, docs/ContactSignalsV2.md section 7.5):

* every request has a ``priority`` (+5 taxonomy previews, -10 backfills and compares, 0 otherwise)
  and claims order by ``priority DESC, requested_at, id``; ``kinds`` filters a claim;
* for kinds that run contact signals (``SIGNAL_REANALYSIS_KINDS``) Store resolves
  ``signal_taxonomy_version`` (the current version) and ``signal_pipeline`` (v2 for previews;
  otherwise the settings, with shadow building v1) at creation;
* at most one pending ``contact_signals`` request exists per call: a new one *widens* a pending,
  unclaimed request (latest taxonomy version, higher priority, ``rescore_signals`` or-ed) instead of
  answering 409;
* ``contact_signals_preview`` is the second draft-test kind (``DRAFT_TEST_KINDS``): its merge output
  is recorded on the request (``record_preview_result``). Previews, compares and backfills are
  created by ``signals.py``.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import FrozenSet, List, Optional

from call1.contracts.common import Page, canonical_digest
from call1.contracts.contents import QaScorecardContent, ResultKind
from call1.contracts.errors import ErrorCode
from call1.contracts.events import Actor, AuditAction
from call1.contracts.jobs import (
    REANALYSIS_KIND_AFFECTS,
    REANALYSIS_PRIORITY_DEFAULT,
    SIGNAL_REANALYSIS_KINDS,
    ClaimedReanalysisRequest,
    DraftTestRequest,
    DraftTestResult,
    JobStatus,
    JobType,
    ReanalysisClaimRequest,
    ReanalysisClaimResponse,
    ReanalysisKind,
    ReanalysisReject,
    ReanalysisRequest,
    ReanalysisRequestCreate,
    ReanalysisStatus,
    SpeakerCorrection,
)
from call1.contracts.rubrics import DraftRubricRef, RubricVersionRef

from .. import audit, db
from ..errors import StoreError, not_found
from ..ids import new_id
from ..principals import ServiceKeyPrincipal, SessionPrincipal, generate_secret, hash_secret, secrets_equal
from .changes import ChangeLog
from .records import (
    ConversationJobs,
    artifact_row,
    effective_request_status,
    request_from_row,
    require_call_row,
    require_request_row,
)

IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
ACTIVE = (ReanalysisStatus.PENDING.value, ReanalysisStatus.CLAIMED.value)


def check_idempotency_key(key: str) -> str:
    if not IDEMPOTENCY_KEY.match(key or ""):
        raise StoreError(ErrorCode.VALIDATION_FAILED, "Idempotency-Key must be 8-128 characters of [A-Za-z0-9._:-]",
                         details={"field": "Idempotency-Key", "reason": "invalid_idempotency_key"})
    return key


def _existing(conn, scope: str, key: str, digest: str):
    row = conn.execute("SELECT * FROM q_reanalysis_requests WHERE idempotency_scope = ? AND idempotency_key = ?", (scope, key)).fetchone()
    if row is not None and row["request_digest"] != digest:
        raise StoreError(ErrorCode.IDEMPOTENCY_KEY_REUSED, "This Idempotency-Key was used with a different request", details={"original_id": row["id"]})
    return row


def resolve_signals(conn, kind: ReanalysisKind):
    """(signal_taxonomy_version, signal_pipeline) Store resolves at creation, or (None, None) for a
    kind whose graph runs no contact signals."""
    from ..results import api as results_api

    if kind not in SIGNAL_REANALYSIS_KINDS:
        return None, None
    version = results_api.current_signal_taxonomy(conn).version
    if kind is ReanalysisKind.CONTACT_SIGNALS_PREVIEW:
        return version, "v2"
    return version, "v2" if results_api.signal_settings(conn).pipeline == "v2" else "v1"


def insert_request(conn, *, conversation_row, kind: ReanalysisKind, scope: str, key: str, digest: str, requested_by_account_id: Optional[str],
                   rubric: Optional[RubricVersionRef] = None, draft_rubric: Optional[DraftRubricRef] = None,
                   speaker_correction: Optional[SpeakerCorrection] = None, note: Optional[str] = None,
                   request_id: Optional[str] = None, changes: ChangeLog, priority: int = REANALYSIS_PRIORITY_DEFAULT,
                   rescore_signals: bool = False, signal_backfill_id: Optional[str] = None, signal_preview_id: Optional[str] = None,
                   signal_taxonomy_snapshot_artifact_id: Optional[str] = None):
    """Insert a pending request in the caller's transaction and record its change event."""
    request_id = request_id or new_id("rq")
    now = db.ts(conn.now())
    taxonomy_version, pipeline = resolve_signals(conn, kind)
    conn.execute(
        "INSERT INTO q_reanalysis_requests (id, conversation_id, call_id, kind, status, rubric_json, draft_rubric_json, speaker_correction_json, note, "
        "requested_by_account_id, requested_at, idempotency_scope, idempotency_key, request_digest, updated_at, priority, rescore_signals, "
        "signal_taxonomy_version, signal_pipeline, signal_backfill_id, signal_preview_id, signal_taxonomy_snapshot_artifact_id) "
        "VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (request_id, conversation_row["id"], conversation_row["call_id"], kind.value, db.dumps(rubric) if rubric else None,
         db.dumps(draft_rubric) if draft_rubric else None, db.dumps(speaker_correction) if speaker_correction else None, note,
         requested_by_account_id, now, scope, key, digest, now, priority, 1 if rescore_signals else 0, taxonomy_version, pipeline,
         signal_backfill_id, signal_preview_id, signal_taxonomy_snapshot_artifact_id),
    )
    changes.request(request_id, ReanalysisStatus.PENDING, conversation_row["id"], conversation_row["call_id"])
    changes.conversation(conversation_row["id"])
    return require_request_row(conn, request_id)


def create_for_area(conn, *, conversation_row, kind: ReanalysisKind, requested_by: Actor, idempotency_key: Optional[str],
                    speaker_correction: Optional[SpeakerCorrection], reason: Optional[str]) -> ReanalysisRequest:
    """``queue.api.create_reanalysis_request`` (the results area's ``correctSpeaker``)."""
    if conversation_row["call_id"] is None:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "Only call conversations take reanalysis requests", details={"reason": "not_a_call"})
    if kind is ReanalysisKind.QA_DRAFT_TEST:
        raise ValueError("draft tests are created by testRubricDraft")
    if kind is ReanalysisKind.CONTACT_SIGNALS_PREVIEW:
        raise ValueError("signal previews are created by createSignalPreview or a compare backfill")
    if (kind is ReanalysisKind.SPEAKER_CORRECTION) != (speaker_correction is not None):
        raise ValueError("speaker correction requests carry the correction; others do not")
    scope = f"area:{requested_by.account_id or requested_by.kind.value}:call:{conversation_row['call_id']}"
    key = check_idempotency_key(idempotency_key) if idempotency_key else "auto:" + new_id("rq")
    digest = canonical_digest({"kind": kind.value, "speaker_correction": speaker_correction.model_dump(mode="json") if speaker_correction else None,
                               "note": reason})
    with db.transaction(conn):
        existing = _existing(conn, scope, key, digest)
        if existing is not None:
            return request_from_row(existing, conn.now())
        changes = ChangeLog(conn)
        row = insert_request(conn, conversation_row=conversation_row, kind=kind, scope=scope, key=key, digest=digest,
                             requested_by_account_id=requested_by.account_id, speaker_correction=speaker_correction, note=reason, changes=changes)
        changes.flush()
        return request_from_row(row, conn.now())


def request_reanalysis(conn, store, principal: SessionPrincipal, call_id: str, body: ReanalysisRequestCreate, key: str) -> ReanalysisRequest:
    """``requestReanalysis``: durable and idempotent; one active request per call and kind."""
    from ..results import api as results_api

    check_idempotency_key(key)
    if body.kind is ReanalysisKind.SPEAKER_CORRECTION:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "Speaker corrections go through POST /calls/{call_id}/speaker-corrections",
                         details={"field": "kind", "reason": "use_speaker_corrections"})
    scope = f"session:{principal.session_id}:call:{call_id}"
    digest = canonical_digest({"operation": "requestReanalysis", "call_id": call_id, "body": body.model_dump(mode="json")})
    with db.transaction(conn):
        conversation = require_call_row(conn, call_id)
        existing = _existing(conn, scope, key, digest)
        if existing is not None:
            return request_from_row(existing, conn.now())
        now = conn.now()
        if body.kind is ReanalysisKind.CONTACT_SIGNALS:
            widened = widen_pending_signals(conn, call_id, priority=REANALYSIS_PRIORITY_DEFAULT, rescore_signals=body.rescore_signals)
            if widened is not None:
                audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.REANALYSIS_REQUESTED, target_kind="call", target_id=call_id,
                             details={"request_id": widened["id"], "kind": body.kind.value, "widened": True})
                return request_from_row(widened, conn.now())
        for other in conn.execute("SELECT * FROM q_reanalysis_requests WHERE call_id = ? AND kind = ? AND status IN (?, ?)",
                                  (call_id, body.kind.value, *ACTIVE)).fetchall():
            if effective_request_status(other, now) in (ReanalysisStatus.PENDING, ReanalysisStatus.CLAIMED):
                raise StoreError(ErrorCode.CONFLICT, "A reanalysis of this kind is already pending for this call",
                                 details={"reason": "already_pending", "existing_request_id": other["id"]})
        if body.rubric is not None:
            version = results_api.rubric_version(conn, body.rubric.rubric_id, body.rubric.version)
            if version is None:
                raise not_found("Rubric version", rubric_id=body.rubric.rubric_id, version=body.rubric.version)
            if version.ref.digest != body.rubric.digest:
                raise StoreError(ErrorCode.CONFLICT, "The rubric digest differs from the published version's", details={"reason": "rubric_digest_mismatch"})
        changes = ChangeLog(conn)
        row = insert_request(conn, conversation_row=conversation, kind=body.kind, scope=scope, key=key, digest=digest,
                             requested_by_account_id=principal.account_id, rubric=body.rubric, note=body.note, changes=changes,
                             rescore_signals=body.rescore_signals)
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.REANALYSIS_REQUESTED, target_kind="call", target_id=call_id,
                     details={"request_id": row["id"], "kind": body.kind.value})
        changes.flush()
        return request_from_row(row, conn.now())


def create_draft_test(conn, store, principal: SessionPrincipal, rubric_id: str, body: DraftTestRequest, key: str) -> ReanalysisRequest:
    """``testRubricDraft``: snapshot the stored draft and create a ``qa_draft_test`` request."""
    from ..results import api as results_api
    from .artifacts import mint_draft_snapshot

    check_idempotency_key(key)
    scope = f"session:{principal.session_id}:call:{body.call_id}"
    digest = canonical_digest({"operation": "testRubricDraft", "rubric_id": rubric_id, "body": body.model_dump(mode="json")})
    with db.transaction(conn):
        conversation = require_call_row(conn, body.call_id)
        existing = _existing(conn, scope, key, digest)
        if existing is not None:
            return request_from_row(existing, conn.now())
        content = results_api.draft_snapshot_content(conn, rubric_id, expected_draft_revision=body.expected_draft_revision)
        if content.source != "draft" or content.rubric_id != rubric_id or content.draft_revision is None:
            raise StoreError(ErrorCode.CONFLICT, "The draft snapshot does not describe this rubric's draft", details={"reason": "snapshot_mismatch"})
        request_id = new_id("rq")
        snapshot = mint_draft_snapshot(conn, store, conversation["id"], request_id, content)
        draft = DraftRubricRef(rubric_id=rubric_id, draft_revision=content.draft_revision, snapshot_artifact_id=snapshot.id, digest=content.digest)
        changes = ChangeLog(conn)
        row = insert_request(conn, conversation_row=conversation, kind=ReanalysisKind.QA_DRAFT_TEST, scope=scope, key=key, digest=digest,
                             requested_by_account_id=principal.account_id, draft_rubric=draft, note=body.note, request_id=request_id, changes=changes)
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.REANALYSIS_REQUESTED, target_kind="call", target_id=body.call_id,
                     details={"request_id": request_id, "kind": ReanalysisKind.QA_DRAFT_TEST.value, "rubric_id": rubric_id,
                              "draft_revision": content.draft_revision})
        changes.flush()
        return request_from_row(row, conn.now())


def draft_result(conn, store, request_id: str) -> DraftTestResult:
    """``getDraftTestResult``: pending until the draft graph's scorecard succeeds or ends."""
    row = require_request_row(conn, request_id)
    if row["kind"] != ReanalysisKind.QA_DRAFT_TEST.value:
        raise not_found("Draft test", request_id=request_id)
    request = request_from_row(row, conn.now())
    draft = request.draft_rubric
    state, failure, scorecard = "pending", None, None
    if row["draft_result_artifact_id"] is not None:
        art = artifact_row(conn, row["draft_result_artifact_id"])
        from ..results import api as results_api

        scorecard = QaScorecardContent.model_validate(json.loads(store.objects.read_bytes(art["checksum"])))
        # Masked like every reviewer read (contract 1.2.0: rule values plus the PII findings).
        scorecard = results_api.masked_scorecard(conn, row["conversation_id"], scorecard)
        state = "available"
    elif request.status is ReanalysisStatus.REJECTED:
        state = "failed"
    elif row["graph_id"] is not None:
        scorecards = conn.execute("SELECT id, conversation_id FROM q_jobs WHERE graph_id = ? AND job_type = ? ORDER BY created_at DESC, id DESC",
                                  (row["graph_id"], JobType.QA_SCORECARD.value)).fetchall()
        if scorecards:
            view = ConversationJobs(conn, scorecards[0]["conversation_id"])
            node = view.nodes[scorecards[0]["id"]]
            if node.status in (JobStatus.FAILED, JobStatus.CANCELLED) or view.dead(node.id):
                state, failure = "failed", view.failure_code(node.id)
    return DraftTestResult(request_id=request.id, call_id=request.call_id, rubric_id=draft.rubric_id, draft_revision=draft.draft_revision,
                           state=state, failure_code=failure, scorecard=scorecard)


def widen_pending_signals(conn, call_id: str, *, priority: int, rescore_signals: bool):
    """Widen the call's pending, unclaimed ``contact_signals`` request (caller's transaction): the
    latest taxonomy version and pipeline, the higher priority, ``rescore_signals`` or-ed. Returns
    the widened row, or None when there is none to widen."""
    now = conn.now()
    for row in conn.execute("SELECT * FROM q_reanalysis_requests WHERE call_id = ? AND kind = ? AND status = 'pending' ORDER BY requested_at, id",
                            (call_id, ReanalysisKind.CONTACT_SIGNALS.value)).fetchall():
        if effective_request_status(row, now) is not ReanalysisStatus.PENDING:
            continue
        version, pipeline = resolve_signals(conn, ReanalysisKind.CONTACT_SIGNALS)
        conn.execute("UPDATE q_reanalysis_requests SET signal_taxonomy_version = ?, signal_pipeline = ?, priority = MAX(priority, ?), "
                     "rescore_signals = MAX(rescore_signals, ?), updated_at = ? WHERE id = ?",
                     (version, pipeline, priority, 1 if rescore_signals else 0, db.ts(now), row["id"]))
        changes = ChangeLog(conn)
        changes.request(row["id"], ReanalysisStatus.PENDING, row["conversation_id"], row["call_id"])
        changes.flush()
        return require_request_row(conn, row["id"])
    return None


def record_preview_result(conn, request_id: str, artifact_id: str, changes: ChangeLog) -> None:
    """A contact_signals_preview graph's merge succeeded: its contact_signals artifact (in the
    request's draft slot) is the preview result. Nothing is projected."""
    row = require_request_row(conn, request_id)
    conn.execute("UPDATE q_reanalysis_requests SET preview_result_artifact_id = ?, updated_at = ? WHERE id = ?", (artifact_id, db.ts(conn.now()), request_id))
    changes.request(request_id, row["status"], row["conversation_id"], row["call_id"])


def record_draft_result(conn, request_id: str, artifact_id: str, changes: ChangeLog) -> None:
    row = require_request_row(conn, request_id)
    conn.execute("UPDATE q_reanalysis_requests SET draft_result_artifact_id = ?, updated_at = ? WHERE id = ?", (artifact_id, db.ts(conn.now()), request_id))
    changes.request(request_id, row["status"], row["conversation_id"], row["call_id"])


def list_for_call(conn, call_id: str) -> Page[ReanalysisRequest]:
    require_call_row(conn, call_id)
    now = conn.now()
    rows = conn.execute("SELECT * FROM q_reanalysis_requests WHERE call_id = ? ORDER BY requested_at DESC, id DESC", (call_id,)).fetchall()
    return Page[ReanalysisRequest](items=[request_from_row(r, now) for r in rows], next_page_token=None)


def get(conn, request_id: str) -> ReanalysisRequest:
    return request_from_row(require_request_row(conn, request_id), conn.now())


def require_request(conn, request_id: str):
    return require_request_row(conn, request_id)


def expire_claims(conn, changes: ChangeLog) -> int:
    """``claimed -> pending`` for every claim past its lease."""
    now = db.ts(conn.now())
    rows = conn.execute("SELECT * FROM q_reanalysis_requests WHERE status = 'claimed' AND claim_expires_at <= ?", (now,)).fetchall()
    for row in rows:
        conn.execute("UPDATE q_reanalysis_requests SET status = 'pending', claimed_by_installation_id = NULL, claim_worker_id = NULL, "
                     "claim_token_hash = NULL, claim_expires_at = NULL, updated_at = ? WHERE id = ?", (now, row["id"]))
        changes.request(row["id"], ReanalysisStatus.PENDING, row["conversation_id"], row["call_id"])
    return len(rows)


def claim(conn, store, principal: ServiceKeyPrincipal, body: ReanalysisClaimRequest) -> ReanalysisClaimResponse:
    """``claimReanalysisRequests``: highest priority first, then oldest (``jobs.reanalysis_claim_order_key``),
    each under its own claim token; ``kinds`` limits the claim to those kinds."""
    lease = store.config.parameters.reanalysis_claim_lease_seconds
    out: List[ClaimedReanalysisRequest] = []
    with db.transaction(conn):
        changes = ChangeLog(conn)
        expire_claims(conn, changes)
        now = conn.now()
        expires = now + timedelta(seconds=lease)
        kinds = [k.value for k in body.kinds] if body.kinds is not None else None
        sql = "SELECT * FROM q_reanalysis_requests WHERE status = 'pending'"
        args: list = []
        if kinds is not None:
            sql += " AND kind IN (" + ",".join("?" for _ in kinds) + ")"
            args.extend(kinds)
        rows = conn.execute(sql + " ORDER BY priority DESC, requested_at, id LIMIT ?", (*args, body.max_requests)).fetchall()
        tokens = []
        for row in rows:
            token = generate_secret(32)
            conn.execute("UPDATE q_reanalysis_requests SET status = 'claimed', claimed_by_installation_id = ?, claim_worker_id = ?, claim_token_hash = ?, "
                         "claim_expires_at = ?, updated_at = ? WHERE id = ? AND status = 'pending'",
                         (principal.installation_id, body.worker_id, hash_secret(token), db.ts(expires), db.ts(now), row["id"]))
            changes.request(row["id"], ReanalysisStatus.CLAIMED, row["conversation_id"], row["call_id"])
            tokens.append((row["id"], token))
        changes.flush()
        for request_id, token in tokens:
            out.append(ClaimedReanalysisRequest(request=request_from_row(require_request_row(conn, request_id), now), claim_token=token, claim_expires_at=expires))
    return ReanalysisClaimResponse(requests=out)


def require_claim(conn, request_id: str, claim_token: str, conversation_id: str):
    """The request a reanalysis graph fulfils, if ``claim_token`` is its active claim."""
    row = require_request_row(conn, request_id)
    if row["conversation_id"] != conversation_id:
        raise StoreError(ErrorCode.GRAPH_INVALID, "The reanalysis request belongs to another conversation", details={"reason": "request_for_other_conversation"})
    status = effective_request_status(row, conn.now())
    if status in (ReanalysisStatus.FULFILLED, ReanalysisStatus.REJECTED):
        raise StoreError(ErrorCode.INVALID_TRANSITION, f"The reanalysis request is already {status.value}", details={"status": status.value})
    if status is not ReanalysisStatus.CLAIMED or not secrets_equal(row["claim_token_hash"] or "", hash_secret(claim_token or "")):
        raise StoreError(ErrorCode.CLAIM_TOKEN_STALE, "The reanalysis claim token is not the request's active claim", details={"status": status.value})
    return row


def fulfil(conn, row, graph_id: str, changes: ChangeLog) -> None:
    conn.execute("UPDATE q_reanalysis_requests SET status = 'fulfilled', graph_id = ?, claim_token_hash = NULL, updated_at = ? WHERE id = ?",
                 (graph_id, db.ts(conn.now()), row["id"]))
    changes.request(row["id"], ReanalysisStatus.FULFILLED, row["conversation_id"], row["call_id"])


def reject(conn, store, principal: ServiceKeyPrincipal, request_id: str, body: ReanalysisReject) -> ReanalysisRequest:
    """``rejectReanalysisRequest``: a claimed request ends ``rejected`` with a safe reason."""
    with db.transaction(conn):
        row = require_request_row(conn, request_id)
        status = effective_request_status(row, conn.now())
        if status in (ReanalysisStatus.FULFILLED, ReanalysisStatus.REJECTED):
            raise StoreError(ErrorCode.INVALID_TRANSITION, f"The reanalysis request is already {status.value}", details={"status": status.value})
        if status is not ReanalysisStatus.CLAIMED or not secrets_equal(row["claim_token_hash"] or "", hash_secret(body.claim_token or "")):
            raise StoreError(ErrorCode.CLAIM_TOKEN_STALE, "The reanalysis claim token is not the request's active claim", details={"status": status.value})
        conn.execute("UPDATE q_reanalysis_requests SET status = 'rejected', rejected_reason = ?, claim_token_hash = NULL, updated_at = ? WHERE id = ?",
                     (body.reason, db.ts(conn.now()), request_id))
        changes = ChangeLog(conn)
        changes.request(request_id, ReanalysisStatus.REJECTED, row["conversation_id"], row["call_id"])
        changes.conversation(row["conversation_id"])
        changes.flush()
        return request_from_row(require_request_row(conn, request_id), conn.now())


def pending_groups(conn, conversation_id: str) -> FrozenSet[ResultKind]:
    """Groups a pending or claimed request affects (``REANALYSIS_KIND_AFFECTS``)."""
    groups = set()
    for row in conn.execute("SELECT kind FROM q_reanalysis_requests WHERE conversation_id = ? AND status IN (?, ?)", (conversation_id, *ACTIVE)):
        groups |= REANALYSIS_KIND_AFFECTS[ReanalysisKind(row["kind"])]
    return frozenset(groups)


def active_request_for_group(conn, conversation_id: str, kind: ResultKind) -> Optional[str]:
    """The newest pending or claimed request affecting ``kind`` (``ResultGroup.reanalysis_request_id``)."""
    for row in conn.execute("SELECT id, kind FROM q_reanalysis_requests WHERE conversation_id = ? AND status IN (?, ?) ORDER BY requested_at DESC, id DESC",
                            (conversation_id, *ACTIVE)):
        if kind in REANALYSIS_KIND_AFFECTS[ReanalysisKind(row["kind"])]:
            return row["id"]
    return None
