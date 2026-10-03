"""Call-level review state: expected-version writes, history, and staleness on new machine versions.

Machine results and human decisions are separate records (contract README, "Reviews and expected
versions"). ``review_version`` is bumped by every human write; each write carries the version it
was read against (409 ``review_version_conflict`` with ``current_version`` otherwise) and, where it
judges the machine result, the evaluation version it judged, which must be the call's current one
(409 ``conflict`` with ``current_evaluation_version``). A new machine version marks the review
``stale`` when any decision referenced an older version; Store-side staleness changes do not bump
``review_version`` (they are not human writes) but are recorded in the history and the feed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from call1.contracts.errors import ErrorCode
from call1.contracts.events import ChangeKind
from call1.contracts.reviews import (
    CallReviewState,
    EscalationStatus,
    ReviewHistoryEntry,
    ReviewHistoryKind,
    ReviewStaleness,
    ReviewWriteResult,
    VerdictOverrideRecord,
)
from call1.contracts.contents import VerdictStatus

from .. import db, feed
from ..db import StoreConnection
from ..errors import StoreError
from ..ids import new_id


def state_row(conn: StoreConnection, call_id: str):
    return conn.execute("SELECT * FROM results_review_state WHERE call_id = ?", (call_id,)).fetchone()


def current_evaluation_version(conn: StoreConnection, call_id: str) -> Optional[int]:
    row = conn.execute("SELECT evaluation_version FROM results_calls WHERE call_id = ?", (call_id,)).fetchone()
    return None if row is None or row["evaluation_version"] is None else int(row["evaluation_version"])


def check_review_version(row, expected: int) -> None:
    current = int(row["review_version"])
    if expected != current:
        raise StoreError(ErrorCode.REVIEW_VERSION_CONFLICT, "The call's review changed since you read it; re-read and decide again",
                         details={"current_version": current})


def check_evaluation_version(conn: StoreConnection, call_id: str, judged: int) -> int:
    current = current_evaluation_version(conn, call_id)
    if current is None or judged != current:
        raise StoreError(ErrorCode.CONFLICT, "The decision names a machine result version that is not the call's current one",
                         details={"current_evaluation_version": current, "reason": "not_current_evaluation_version"})
    return current


def add_history(conn: StoreConnection, call_id: str, kind: ReviewHistoryKind, *, account_id: Optional[str], review_version: int,
                evaluation_version: Optional[int], payload: Optional[Dict[str, Any]] = None) -> None:
    conn.execute(
        "INSERT INTO results_review_history (id, call_id, kind, account_id, review_version, evaluation_version, payload_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (new_id("rh"), call_id, kind.value, account_id, review_version, evaluation_version, db.dumps(payload or {}), db.ts(conn.now())),
    )


def bump(conn: StoreConnection, call_id: str, **fields: Any) -> int:
    """Bump review_version (a human write), set ``fields``, and return the new version."""
    assignments = ", ".join(f"{name} = ?" for name in fields)
    conn.execute(
        f"UPDATE results_review_state SET review_version = review_version + 1, updated_at = ?{', ' + assignments if assignments else ''} WHERE call_id = ?",
        (db.ts(conn.now()), *fields.values(), call_id),
    )
    return int(state_row(conn, call_id)["review_version"])


def write_result(conn: StoreConnection, call_id: str, conversation_id: str, review_version: int, status: str) -> ReviewWriteResult:
    cursor = feed.append(conn, ChangeKind.REVIEW, call_id, review_version, status, conversation_id=conversation_id, call_id=call_id)
    return ReviewWriteResult(call_id=call_id, review_version=review_version, change_cursor=cursor, committed_at=conn.now())


def _override(row) -> VerdictOverrideRecord:
    return VerdictOverrideRecord(
        id=row["id"], call_id=row["call_id"], criterion_id=row["criterion_id"], evaluation_version=row["evaluation_version"],
        original_status=VerdictStatus(row["original_status"]), status=VerdictStatus(row["status"]), reason_code=row["reason_code"],
        reviewer_notes=row["reviewer_notes"], account_id=row["account_id"], review_version=row["review_version"],
        created_at=db.parse_ts(row["created_at"]),
    )


def review_state(conn: StoreConnection, call_id: str) -> Optional[CallReviewState]:
    row = state_row(conn, call_id)
    if row is None:
        return None
    reviewed = row["reviewed_evaluation_version"]
    overrides: List[VerdictOverrideRecord] = []
    if reviewed is not None:
        overrides = [_override(r) for r in conn.execute(
            "SELECT * FROM results_verdict_overrides WHERE call_id = ? AND evaluation_version = ? ORDER BY created_at, id",
            (call_id, reviewed)).fetchall()]
    return CallReviewState(
        call_id=call_id,
        review_version=row["review_version"],
        current_evaluation_version=current_evaluation_version(conn, call_id),
        reviewed_evaluation_version=reviewed,
        staleness=ReviewStaleness(row["staleness"]),
        escalation_status=EscalationStatus(row["escalation_status"]),
        escalation_resolved_by_account_id=row["escalation_resolved_by"],
        escalation_resolved_at=db.parse_ts(row["escalation_resolved_at"]),
        reviewer_notes=row["reviewer_notes"],
        overrides=overrides,
        retained_by_account_id=row["retained_by"],
        retained_at=db.parse_ts(row["retained_at"]),
        updated_at=db.parse_ts(row["updated_at"]),
    )


def history(conn: StoreConnection, call_id: str) -> List[ReviewHistoryEntry]:
    rows = conn.execute("SELECT * FROM results_review_history WHERE call_id = ? ORDER BY created_at DESC, id DESC", (call_id,)).fetchall()
    return [
        ReviewHistoryEntry(id=r["id"], call_id=r["call_id"], kind=ReviewHistoryKind(r["kind"]), account_id=r["account_id"],
                           review_version=r["review_version"], evaluation_version=r["evaluation_version"],
                           payload=db.loads(r["payload_json"]), created_at=db.parse_ts(r["created_at"]))
        for r in rows
    ]


def on_new_evaluation(conn: StoreConnection, call_id: str, conversation_id: str, version: int, requires_human_review: bool) -> None:
    """Apply the stale-write rule to the call-level review when a new machine version commits."""
    row = state_row(conn, call_id)
    if row is None:
        return
    now = db.ts(conn.now())
    reviewed = row["reviewed_evaluation_version"]
    decided_older = reviewed is not None and int(reviewed) < version
    escalation = EscalationStatus(row["escalation_status"])
    resolved_older = escalation in (EscalationStatus.APPROVED, EscalationStatus.OVERRIDDEN) and (
        row["escalation_evaluation_version"] is None or int(row["escalation_evaluation_version"]) < version)
    if escalation is EscalationStatus.NONE and requires_human_review:
        escalation = EscalationStatus.PENDING
    elif escalation is EscalationStatus.PENDING and not requires_human_review:
        escalation = EscalationStatus.NONE
    staleness = ReviewStaleness(row["staleness"])
    if (decided_older or resolved_older) and staleness is not ReviewStaleness.STALE:
        staleness = ReviewStaleness.STALE
        add_history(conn, call_id, ReviewHistoryKind.MARKED_STALE, account_id=None, review_version=int(row["review_version"]),
                    evaluation_version=version, payload={"reviewed_evaluation_version": reviewed})
    conn.execute("UPDATE results_review_state SET escalation_status = ?, staleness = ?, updated_at = ? WHERE call_id = ?",
                 (escalation.value, staleness.value, now, call_id))
    feed.append(conn, ChangeKind.REVIEW, call_id, int(row["review_version"]), staleness.value, conversation_id=conversation_id, call_id=call_id)


def decided(conn: StoreConnection, call_id: str, evaluation_version: int, **extra: Any) -> int:
    """Record a human decision against ``evaluation_version``: bump, mark current, return version."""
    return bump(conn, call_id, reviewed_evaluation_version=evaluation_version, staleness=ReviewStaleness.CURRENT.value, **extra)


def now_ts(conn: StoreConnection) -> datetime:
    return conn.now()
