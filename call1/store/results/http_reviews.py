"""Human review decisions on a call: overrides, escalation resolution, retention, speaker corrections."""

from __future__ import annotations

from typing import Annotated, List

from fastapi import Depends, Query

from call1.contracts.common import Page
from call1.contracts.contents import QaScorecardContent, ResultKind
from call1.contracts.errors import ErrorCode
from call1.contracts.events import AuditAction
from call1.contracts.jobs import ReanalysisKind, ReanalysisRequest
from call1.contracts.reviews import (
    CallReviewState,
    EscalationListItem,
    EscalationQuery,
    EscalationResolution,
    EscalationStatus,
    ReviewHistoryEntry,
    ReviewHistoryKind,
    ReviewStaleness,
    ReviewWriteResult,
    RetainReviewRequest,
    SpeakerCorrectionRequest,
    VerdictOverride,
)

from .. import audit, db, pagination
from ..db import StoreConnection, read_snapshot, transaction
from ..deps import current_principal, get_conn
from ..errors import StoreError, not_found
from ..ids import new_id
from ..principals import Principal, require_session
from ..queue import api as queue_api
from . import content, records, review_state, training_labels
from .routes import router


def _call(conn: StoreConnection, call_id: str):
    row = records.call_row(conn, call_id)
    if row is None:
        raise not_found("Call", call_id=call_id)
    return row


@router.operation("getReviewState")
def get_review_state(call_id: str, conn: StoreConnection = Depends(get_conn)) -> CallReviewState:
    with read_snapshot(conn):
        _call(conn, call_id)
        state = review_state.review_state(conn, call_id)
    if state is None:
        raise not_found("Review", call_id=call_id)
    return state


@router.operation("listReviewHistory")
def list_review_history(call_id: str, conn: StoreConnection = Depends(get_conn)) -> Page[ReviewHistoryEntry]:
    with read_snapshot(conn):
        _call(conn, call_id)
        return Page[ReviewHistoryEntry](items=review_state.history(conn, call_id), next_page_token=None)


@router.operation("overrideVerdict")
def override_verdict(call_id: str, criterion_id: str, body: VerdictOverride, conn: StoreConnection = Depends(get_conn),
                     principal: Principal = Depends(current_principal)) -> ReviewWriteResult:
    session = require_session(principal)
    with transaction(conn):
        row = _call(conn, call_id)
        state = review_state.state_row(conn, call_id)
        review_state.check_review_version(state, body.expected_version)
        version = review_state.check_evaluation_version(conn, call_id, body.evaluation_version)
        pub = records.publication(conn, row["conversation_id"], ResultKind.QA, version)
        scorecard = content.read_model(conn, pub.checksum, QaScorecardContent)
        verdict = next((v for v in scorecard.verdicts if v.criterion_id == criterion_id), None)
        if verdict is None:
            raise not_found("Criterion verdict", call_id=call_id, criterion_id=criterion_id[:200])
        new_version = review_state.decided(conn, call_id, version)
        override_id = new_id("ovr")
        conn.execute(
            "INSERT INTO results_verdict_overrides (id, call_id, criterion_id, evaluation_version, original_status, status, reason_code, "
            "reviewer_notes, account_id, review_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (override_id, call_id, criterion_id, version, verdict.status.value, body.status.value,
             body.reason_code.value if body.reason_code else None, body.reviewer_notes, session.account_id, new_version, db.ts(conn.now())),
        )
        # On-device training (1.3.0, decision 28): the label log row, in this transaction. IDs and enums only.
        training_labels.record_verdict_override(conn, call_id=call_id, conversation_id=row["conversation_id"], override_id=override_id,
                                                criterion_id=criterion_id, evaluation_version=version, original_status=verdict.status,
                                                status=body.status, reason_code=body.reason_code)
        review_state.add_history(conn, call_id, ReviewHistoryKind.VERDICT_OVERRIDE, account_id=session.account_id, review_version=new_version,
                                 evaluation_version=version, payload={"criterion_id": criterion_id, "original_status": verdict.status.value,
                                                                      "status": body.status.value,
                                                                      "reason_code": body.reason_code.value if body.reason_code else None})
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.VERDICT_OVERRIDDEN, target_kind="call", target_id=call_id,
                     details={"criterion_id": criterion_id[:200], "evaluation_version": version, "status": body.status.value,
                              "original_status": verdict.status.value, "review_version": new_version})
        return review_state.write_result(conn, call_id, row["conversation_id"], new_version, "verdict_overridden")


@router.operation("resolveEscalation")
def resolve_escalation(call_id: str, body: EscalationResolution, conn: StoreConnection = Depends(get_conn),
                       principal: Principal = Depends(current_principal)) -> ReviewWriteResult:
    session = require_session(principal)
    with transaction(conn):
        row = _call(conn, call_id)
        state = review_state.state_row(conn, call_id)
        review_state.check_review_version(state, body.expected_version)
        version = review_state.check_evaluation_version(conn, call_id, body.evaluation_version)
        if EscalationStatus(state["escalation_status"]) is EscalationStatus.NONE:
            raise StoreError(ErrorCode.INVALID_TRANSITION, "This call has no escalation to resolve", details={"escalation_status": "NONE"})
        now = db.ts(conn.now())
        new_version = review_state.decided(conn, call_id, version, escalation_status=body.escalation_status.value,
                                           escalation_evaluation_version=version, escalation_resolved_by=session.account_id,
                                           escalation_resolved_at=now, reviewer_notes=body.reviewer_notes)
        review_state.add_history(conn, call_id, ReviewHistoryKind.ESCALATION_RESOLUTION, account_id=session.account_id,
                                 review_version=new_version, evaluation_version=version,
                                 payload={"escalation_status": body.escalation_status.value, "previous_status": state["escalation_status"]})
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.ESCALATION_RESOLVED, target_kind="call", target_id=call_id,
                     details={"escalation_status": body.escalation_status.value, "evaluation_version": version, "review_version": new_version})
        return review_state.write_result(conn, call_id, row["conversation_id"], new_version, "escalation_resolved")


@router.operation("retainReview")
def retain_review(call_id: str, body: RetainReviewRequest, conn: StoreConnection = Depends(get_conn),
                  principal: Principal = Depends(current_principal)) -> ReviewWriteResult:
    session = require_session(principal)
    with transaction(conn):
        row = _call(conn, call_id)
        state = review_state.state_row(conn, call_id)
        review_state.check_review_version(state, body.expected_version)
        if ReviewStaleness(state["staleness"]) is not ReviewStaleness.STALE:
            raise StoreError(ErrorCode.INVALID_TRANSITION, "Only a stale review can be retained", details={"staleness": state["staleness"]})
        new_version = review_state.bump(conn, call_id, staleness=ReviewStaleness.RETAINED.value, retained_by=session.account_id,
                                        retained_at=db.ts(conn.now()))
        current = review_state.current_evaluation_version(conn, call_id)
        review_state.add_history(conn, call_id, ReviewHistoryKind.REVIEW_RETAINED, account_id=session.account_id, review_version=new_version,
                                 evaluation_version=current, payload={"reviewed_evaluation_version": state["reviewed_evaluation_version"],
                                                                      "note": (body.note or "")[:500] or None})
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.REVIEW_RETAINED, target_kind="call", target_id=call_id,
                     details={"current_evaluation_version": current, "reviewed_evaluation_version": state["reviewed_evaluation_version"],
                              "review_version": new_version})
        return review_state.write_result(conn, call_id, row["conversation_id"], new_version, "retained")


@router.operation("correctSpeaker")
def correct_speaker(call_id: str, body: SpeakerCorrectionRequest, conn: StoreConnection = Depends(get_conn),
                    principal: Principal = Depends(current_principal)) -> ReanalysisRequest:
    session = require_session(principal)
    with transaction(conn):
        row = _call(conn, call_id)
        state = review_state.state_row(conn, call_id)
        review_state.check_review_version(state, body.expected_version)
        _, transcript = records.transcript_content(conn, row["conversation_id"])
        if transcript is None:
            raise not_found("Transcript", call_id=call_id)
        if not any(t.turn_id == body.correction.turn_id for t in transcript.turns):
            raise StoreError(ErrorCode.VALIDATION_FAILED, "The transcript has no such turn", details={"field": "correction.turn_id"})
        new_version = review_state.bump(conn, call_id)
        request = queue_api.create_reanalysis_request(
            conn, conversation_id=row["conversation_id"], kind=ReanalysisKind.SPEAKER_CORRECTION, requested_by=audit.actor_for(principal),
            idempotency_key=f"speaker-correction:{call_id}:r{new_version}", speaker_correction=body.correction, reason=body.correction.notes)
        # On-device training (1.3.0, decision 28): the label log row, in this transaction. No notes are copied.
        training_labels.record_speaker_correction(conn, call_id=call_id, conversation_id=row["conversation_id"], turn_id=body.correction.turn_id,
                                                  speaker=body.correction.speaker, apply_to_cluster=body.correction.apply_to_cluster,
                                                  reanalysis_request_id=request.id)
        review_state.add_history(conn, call_id, ReviewHistoryKind.SPEAKER_CORRECTION, account_id=session.account_id, review_version=new_version,
                                 evaluation_version=review_state.current_evaluation_version(conn, call_id),
                                 payload={"turn_id": body.correction.turn_id, "speaker": body.correction.speaker,
                                          "apply_to_cluster": body.correction.apply_to_cluster, "reanalysis_request_id": request.id})
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.REANALYSIS_REQUESTED, target_kind="call", target_id=call_id,
                     details={"kind": ReanalysisKind.SPEAKER_CORRECTION.value, "reanalysis_request_id": request.id, "review_version": new_version})
        review_state.write_result(conn, call_id, row["conversation_id"], new_version, "speaker_correction_requested")
        return request


@router.operation("listEscalations")
def list_escalations(query: Annotated[EscalationQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> Page[EscalationListItem]:
    where: List[str] = ["c.evaluation_version IS NOT NULL"]
    args: list = []
    if query.status is not None:
        where.append("r.escalation_status = ?")
        args.append(query.status.value)
    else:
        where.append("r.escalation_status != 'NONE'")
    after = pagination.decode(query.page_token, 2)
    if after is not None:
        where.append("(c.created_at < ? OR (c.created_at = ? AND c.call_id < ?))")
        args.extend([after[0], after[0], after[1]])
    sql = ("SELECT c.*, r.escalation_status AS esc, r.review_version AS rv FROM results_calls c JOIN results_review_state r ON r.call_id = c.call_id "
           "WHERE " + " AND ".join(where) + " ORDER BY c.created_at DESC, c.call_id DESC LIMIT ?")
    with read_snapshot(conn):
        rows = conn.execute(sql, (*args, query.limit + 1)).fetchall()
    items = [
        EscalationListItem(call_id=r["call_id"], agent_id=r["agent_id"], agent_display_name=r["agent_display_name"],
                           agent_extension=r["agent_extension"], duration_seconds=float(r["duration_seconds"] or 0.0),
                           overall_score=r["overall_score"], escalation_status=EscalationStatus(r["esc"]),
                           critical_failure=bool(r["critical_failure"]), evaluation_version=r["evaluation_version"], review_version=r["rv"],
                           created_at=db.parse_ts(r["created_at"]))
        for r in rows[: query.limit]
    ]
    token = pagination.encode(rows[query.limit - 1]["created_at"], rows[query.limit - 1]["call_id"]) if len(rows) > query.limit else None
    return Page[EscalationListItem](items=items, next_page_token=token)
