"""The human review queue (never the processing job queue), its rules, and reviewer profiles."""

from __future__ import annotations

from typing import Annotated, Dict, List

from fastapi import Depends, Query

from call1.contracts.auth import AccountStatus, Permission
from call1.contracts.common import Page
from call1.contracts.errors import ErrorCode
from call1.contracts.events import AuditAction
from call1.contracts.reviews import (
    AssignRequest,
    ClaimNextResponse,
    ReleaseRequest,
    ResolveRequest,
    ReviewerProfile,
    ReviewerProfileUpdate,
    ReviewHistoryKind,
    ReviewQueueItem,
    ReviewQueueQuery,
    ReviewQueueRuleRecord,
    ReviewQueueRuleSave,
    ReviewQueueStats,
    ReviewQueueStatus,
    ReviewWriteResult,
    StartReviewRequest,
)

from .. import audit, db, pagination
from ..auth import api as auth_api
from ..db import StoreConnection, read_snapshot, transaction
from ..deps import current_principal, get_conn
from ..errors import StoreError, not_found
from ..principals import Principal, SessionPrincipal, require_permission, require_session
from . import review_queue, review_state
from .routes import router


def _item(conn: StoreConnection, item_id: str):
    row = review_queue.get_item_row(conn, item_id)
    if row is None:
        raise not_found("Review queue item", item_id=item_id)
    return row


def _check_item_version(row, expected: int) -> None:
    if int(row["item_version"]) != expected:
        raise StoreError(ErrorCode.CONFLICT, "The queue item changed since you read it",
                         details={"current_item_version": int(row["item_version"]), "reason": "item_version"})


def _require_status(row, *allowed: ReviewQueueStatus) -> None:
    status = ReviewQueueStatus(row["status"])
    if status is ReviewQueueStatus.SUPERSEDED:
        raise StoreError(ErrorCode.CONFLICT, "A newer machine result superseded this item; open its replacement",
                         details={"reason": "superseded", "superseded_by_item_id": row["superseded_by_item_id"]})
    if status not in allowed:
        raise StoreError(ErrorCode.INVALID_TRANSITION, f"The item is {status.value}", details={"status": status.value})


def _changed(conn: StoreConnection, item_id: str) -> ReviewQueueItem:
    row = review_queue.get_item_row(conn, item_id)
    review_queue.item_changed(conn, row)
    return review_queue.item_from_row(row)


@router.operation("listReviewQueue")
def list_review_queue(query: Annotated[ReviewQueueQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> Page[ReviewQueueItem]:
    where: List[str] = []
    args: list = []
    after = pagination.decode(query.page_token, 2)
    if after is not None:
        where.append("(created_at < ? OR (created_at = ? AND id < ?))")
        args.extend([after[0], after[0], after[1]])
    if query.status is not None:
        where.append("status = ?")
        args.append(query.status.value)
    if query.stream is not None:
        where.append("stream = ?")
        args.append(query.stream.value)
    if query.assigned_to_account_id is not None:
        where.append("assigned_to_account_id = ?")
        args.append(query.assigned_to_account_id)
    if query.unassigned_only:
        where.append("assigned_to_account_id IS NULL")
    if query.call_id is not None:
        where.append("call_id = ?")
        args.append(query.call_id)
    if not query.include_stale:
        where.append("stale = 0")
    sql = "SELECT * FROM results_review_items" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC, id DESC LIMIT ?"
    with read_snapshot(conn):
        rows = conn.execute(sql, (*args, query.limit + 1)).fetchall()
    items = [review_queue.item_from_row(r) for r in rows[: query.limit]]
    token = pagination.encode(rows[query.limit - 1]["created_at"], rows[query.limit - 1]["id"]) if len(rows) > query.limit else None
    return Page[ReviewQueueItem](items=items, next_page_token=token)


@router.operation("getReviewQueueStats")
def get_review_queue_stats(conn: StoreConnection = Depends(get_conn)) -> ReviewQueueStats:
    with read_snapshot(conn):
        rows = conn.execute("SELECT status, stream, assigned_to_account_id, stale FROM results_review_items").fetchall()
    by_stream: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    for r in rows:
        by_stream[r["stream"]] = by_stream.get(r["stream"], 0) + 1
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return ReviewQueueStats(
        total=len(rows),
        pending=by_status.get("PENDING", 0),
        in_review=by_status.get("IN_REVIEW", 0),
        unassigned=sum(1 for r in rows if r["status"] == "PENDING" and r["assigned_to_account_id"] is None),
        stale=sum(1 for r in rows if r["stale"]),
        by_stream=by_stream,
        by_status=by_status,
    )


@router.operation("claimNextReview")
def claim_next_review(conn: StoreConnection = Depends(get_conn), principal: Principal = Depends(current_principal)) -> ClaimNextResponse:
    session = require_session(principal)
    with transaction(conn):
        row = conn.execute(
            "SELECT * FROM results_review_items WHERE status = 'PENDING' AND assigned_to_account_id IS NULL AND stale = 0 "
            "ORDER BY urgency_score DESC, created_at ASC, id ASC LIMIT 1").fetchone()
        if row is None:
            return ClaimNextResponse(item=None)
        now = db.ts(conn.now())
        conn.execute(
            "UPDATE results_review_items SET status = 'IN_REVIEW', assigned_to_account_id = ?, assigned_display_name = ?, assigned_at = ?, "
            "started_at = ?, item_version = item_version + 1 WHERE id = ?",
            (session.account_id, session.display_name, now, now, row["id"]))
        return ClaimNextResponse(item=_changed(conn, row["id"]))


@router.operation("getReviewQueueItem")
def get_review_queue_item(item_id: str, conn: StoreConnection = Depends(get_conn)) -> ReviewQueueItem:
    with read_snapshot(conn):
        return review_queue.item_from_row(_item(conn, item_id))


@router.operation("assignReviewQueueItem")
def assign_review_queue_item(item_id: str, body: AssignRequest, conn: StoreConnection = Depends(get_conn)) -> ReviewQueueItem:
    with transaction(conn):
        row = _item(conn, item_id)
        _check_item_version(row, body.expected_item_version)
        _require_status(row, ReviewQueueStatus.PENDING)
        name = None
        if body.account_id is not None:
            account = auth_api.get_account(conn, body.account_id)
            if account is None or account.status is not AccountStatus.ACTIVE:
                raise not_found("Active reviewer account", account_id=body.account_id)
            name = account.display_name
        conn.execute(
            "UPDATE results_review_items SET assigned_to_account_id = ?, assigned_display_name = ?, assigned_at = ?, item_version = item_version + 1 "
            "WHERE id = ?", (body.account_id, name, db.ts(conn.now()) if body.account_id else None, item_id))
        return _changed(conn, item_id)


@router.operation("startReview")
def start_review(item_id: str, body: StartReviewRequest, conn: StoreConnection = Depends(get_conn),
                 principal: Principal = Depends(current_principal)) -> ReviewQueueItem:
    session = require_session(principal)
    with transaction(conn):
        row = _item(conn, item_id)
        _check_item_version(row, body.expected_item_version)
        _require_status(row, ReviewQueueStatus.PENDING)
        if row["assigned_to_account_id"] not in (None, session.account_id):
            raise StoreError(ErrorCode.FORBIDDEN, "This item is assigned to another reviewer", details={"reason": "assigned_to_another_reviewer"})
        now = db.ts(conn.now())
        conn.execute(
            "UPDATE results_review_items SET status = 'IN_REVIEW', assigned_to_account_id = ?, assigned_display_name = ?, "
            "assigned_at = COALESCE(assigned_at, ?), started_at = ?, item_version = item_version + 1 WHERE id = ?",
            (session.account_id, row["assigned_display_name"] or session.display_name, now, now, item_id))
        return _changed(conn, item_id)


def _holder_or_any(session: SessionPrincipal, row) -> None:
    if row["assigned_to_account_id"] != session.account_id:
        require_permission(session, Permission.RESOLVE_ANY_REVIEW)


@router.operation("releaseReview")
def release_review(item_id: str, body: ReleaseRequest, conn: StoreConnection = Depends(get_conn),
                   principal: Principal = Depends(current_principal)) -> ReviewQueueItem:
    session = require_session(principal)
    with transaction(conn):
        row = _item(conn, item_id)
        _check_item_version(row, body.expected_item_version)
        _require_status(row, ReviewQueueStatus.IN_REVIEW)
        _holder_or_any(session, row)
        conn.execute(
            "UPDATE results_review_items SET status = 'PENDING', assigned_to_account_id = NULL, assigned_display_name = NULL, assigned_at = NULL, "
            "started_at = NULL, reviewer_notes = COALESCE(?, reviewer_notes), item_version = item_version + 1 WHERE id = ?", (body.note, item_id))
        return _changed(conn, item_id)


@router.operation("resolveReview")
def resolve_review(item_id: str, body: ResolveRequest, conn: StoreConnection = Depends(get_conn),
                   principal: Principal = Depends(current_principal)) -> ReviewWriteResult:
    session = require_session(principal)
    with transaction(conn):
        row = _item(conn, item_id)
        _check_item_version(row, body.expected_item_version)
        _require_status(row, ReviewQueueStatus.IN_REVIEW)
        _holder_or_any(session, row)
        call_id = row["call_id"]
        state = review_state.state_row(conn, call_id)
        review_state.check_review_version(state, body.expected_review_version)
        if body.evaluation_version != int(row["evaluation_version"]):
            raise StoreError(ErrorCode.CONFLICT, "The item was created for a different machine result version",
                             details={"current_evaluation_version": review_state.current_evaluation_version(conn, call_id),
                                      "reason": "item_evaluation_version"})
        version = review_state.check_evaluation_version(conn, call_id, body.evaluation_version)
        now = db.ts(conn.now())
        conn.execute(
            "UPDATE results_review_items SET status = ?, resolved_at = ?, resolved_by_account_id = ?, reviewer_notes = ?, "
            "item_version = item_version + 1 WHERE id = ?", (body.status.value, now, session.account_id, body.reviewer_notes, item_id))
        new_version = review_state.decided(conn, call_id, version)
        review_state.add_history(conn, call_id, ReviewHistoryKind.QUEUE_RESOLUTION, account_id=session.account_id, review_version=new_version,
                                 evaluation_version=version, payload={"item_id": item_id, "rule_id": row["rule_id"], "status": body.status.value})
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.REVIEW_RESOLVED, target_kind="review_queue_item", target_id=item_id,
                     details={"call_id": call_id, "status": body.status.value, "evaluation_version": version, "review_version": new_version,
                              "resolved_for_other": row["assigned_to_account_id"] != session.account_id})
        _changed(conn, item_id)
        return review_state.write_result(conn, call_id, row["conversation_id"], new_version, "queue_item_resolved")


@router.operation("listReviewQueueRules")
def list_review_queue_rules(conn: StoreConnection = Depends(get_conn)) -> Page[ReviewQueueRuleRecord]:
    with read_snapshot(conn):
        return Page[ReviewQueueRuleRecord](items=review_queue.list_rules(conn), next_page_token=None)


@router.operation("saveReviewQueueRule")
def save_review_queue_rule(rule_id: str, body: ReviewQueueRuleSave, conn: StoreConnection = Depends(get_conn),
                           principal: Principal = Depends(current_principal)) -> ReviewQueueRuleRecord:
    session = require_session(principal)
    if body.rule.id != rule_id:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "rule.id must equal the rule in the path", details={"field": "rule.id"})
    with transaction(conn):
        existing = conn.execute("SELECT rule_version FROM results_queue_rules WHERE id = ?", (rule_id,)).fetchone()
        current = int(existing["rule_version"]) if existing else 0
        if body.expected_rule_version != current:
            raise StoreError(ErrorCode.CONFLICT, "The rule changed since you read it", details={"current_rule_version": current, "reason": "rule_version"})
        now = db.ts(conn.now())
        conn.execute(
            "INSERT INTO results_queue_rules (id, rank, enabled, rule_json, rule_version, updated_at, updated_by_account_id) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET rank = excluded.rank, enabled = excluded.enabled, rule_json = excluded.rule_json, "
            "rule_version = excluded.rule_version, updated_at = excluded.updated_at, updated_by_account_id = excluded.updated_by_account_id",
            (rule_id, body.rule.rank, 1 if body.rule.enabled else 0, db.dumps(body.rule), current + 1, now, session.account_id))
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.QUEUE_RULE_CHANGED, target_kind="queue_rule", target_id=rule_id,
                     details={"change": "created" if current == 0 else "updated", "rule_version": current + 1, "stream": body.rule.stream.value,
                              "enabled": body.rule.enabled})
        return review_queue.get_rule(conn, rule_id)


@router.operation("deleteReviewQueueRule")
def delete_review_queue_rule(rule_id: str, conn: StoreConnection = Depends(get_conn), principal: Principal = Depends(current_principal)) -> None:
    require_session(principal)
    with transaction(conn):
        existing = conn.execute("SELECT rule_version FROM results_queue_rules WHERE id = ?", (rule_id,)).fetchone()
        if existing is None:
            raise not_found("Review queue rule", rule_id=rule_id)
        conn.execute("DELETE FROM results_queue_rules WHERE id = ?", (rule_id,))
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.QUEUE_RULE_CHANGED, target_kind="queue_rule", target_id=rule_id,
                     details={"change": "deleted", "rule_version": int(existing["rule_version"])})
    return None


@router.operation("listReviewerProfiles")
def list_reviewer_profiles(conn: StoreConnection = Depends(get_conn)) -> Page[ReviewerProfile]:
    with read_snapshot(conn):
        return Page[ReviewerProfile](items=review_queue.list_profiles(conn), next_page_token=None)


@router.operation("updateReviewerProfile")
def update_reviewer_profile(account_id: str, body: ReviewerProfileUpdate, conn: StoreConnection = Depends(get_conn)) -> ReviewerProfile:
    with transaction(conn):
        account = auth_api.get_account(conn, account_id)
        if account is None:
            raise not_found("Reviewer account", account_id=account_id)
        current = review_queue.profile_for(conn, account_id, account.display_name)
        skills = body.skills if body.skills is not None else current.skills
        weight = body.capacity_weight if body.capacity_weight is not None else current.capacity_weight
        accepting = body.accepting_assignments if body.accepting_assignments is not None else current.accepting_assignments
        conn.execute(
            "INSERT INTO results_reviewer_profiles (account_id, skills_json, capacity_weight, accepting_assignments, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(account_id) DO UPDATE SET skills_json = excluded.skills_json, capacity_weight = excluded.capacity_weight, "
            "accepting_assignments = excluded.accepting_assignments, updated_at = excluded.updated_at",
            (account_id, db.dumps(list(skills)), weight, 1 if accepting else 0, db.ts(conn.now())))
        return review_queue.profile_for(conn, account_id, account.display_name)
