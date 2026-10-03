"""Rubrics: one editable draft per rubric, immutable published versions, retirement.

A published version is immutable: its ``definition`` and ``digest`` (``canonical_digest``) never
change; retirement only flips its status. Drafts are single-writer with a ``draft_revision``
counter that never repeats within a rubric (``last_draft_revision``), so a draft-test snapshot's
revision always names exactly one saved draft. 040_results.sql seeds the pre-split default rubric
as version 1 of ``call1_standard_v2``.
"""

from __future__ import annotations

from typing import List, Optional

from call1.contracts.common import canonical_digest
from call1.contracts.errors import ErrorCode
from call1.contracts.events import ChangeKind
from call1.contracts.rubrics import (
    RubricDefinition,
    RubricDraft,
    RubricSnapshotContent,
    RubricSummary,
    RubricVersion,
    RubricVersionRef,
    RubricVersionStatus,
)

from .. import db, feed
from ..db import StoreConnection
from ..errors import StoreError, not_found


def rubric_row(conn: StoreConnection, rubric_id: str):
    return conn.execute("SELECT * FROM results_rubrics WHERE rubric_id = ?", (rubric_id,)).fetchone()


def _version(row) -> RubricVersion:
    definition = RubricDefinition.model_validate(db.loads(row["definition_json"]))
    return RubricVersion(
        ref=RubricVersionRef(rubric_id=row["rubric_id"], version=row["version"], digest=row["digest"]),
        definition=definition,
        status=RubricVersionStatus(row["status"]),
        published_at=db.parse_ts(row["published_at"]),
        published_by_account_id=row["published_by_account_id"],
        notes=row["notes"],
    )


def get_version(conn: StoreConnection, rubric_id: str, version: int) -> Optional[RubricVersion]:
    row = conn.execute("SELECT * FROM results_rubric_versions WHERE rubric_id = ? AND version = ?", (rubric_id, version)).fetchone()
    return None if row is None else _version(row)


def current_version(conn: StoreConnection, rubric_id: str) -> Optional[RubricVersion]:
    rubric = rubric_row(conn, rubric_id)
    if rubric is None or rubric["current_version"] is None:
        return None
    return get_version(conn, rubric_id, int(rubric["current_version"]))


def list_versions(conn: StoreConnection, rubric_id: str) -> List[RubricVersion]:
    rows = conn.execute("SELECT * FROM results_rubric_versions WHERE rubric_id = ? ORDER BY version DESC", (rubric_id,)).fetchall()
    return [_version(r) for r in rows]


def draft_row(conn: StoreConnection, rubric_id: str):
    return conn.execute("SELECT * FROM results_rubric_drafts WHERE rubric_id = ?", (rubric_id,)).fetchone()


def _draft(row) -> RubricDraft:
    return RubricDraft(
        rubric_id=row["rubric_id"],
        definition=RubricDefinition.model_validate(db.loads(row["definition_json"])),
        draft_revision=row["draft_revision"],
        based_on_version=row["based_on_version"],
        updated_at=db.parse_ts(row["updated_at"]),
        updated_by_account_id=row["updated_by_account_id"],
    )


def get_draft(conn: StoreConnection, rubric_id: str) -> Optional[RubricDraft]:
    row = draft_row(conn, rubric_id)
    return None if row is None else _draft(row)


def _conflict(conn: StoreConnection, rubric_id: str, message: str) -> StoreError:
    rubric = rubric_row(conn, rubric_id)
    draft = draft_row(conn, rubric_id)
    return StoreError(ErrorCode.RUBRIC_VERSION_CONFLICT, message, details={
        "current_version": int(rubric["current_version"]) if rubric is not None and rubric["current_version"] is not None else 0,
        "draft_revision": int(draft["draft_revision"]) if draft is not None else None,
    })


def summaries(conn: StoreConnection) -> List[RubricSummary]:
    out: List[RubricSummary] = []
    for rubric in conn.execute("SELECT * FROM results_rubrics ORDER BY rubric_id").fetchall():
        rubric_id = rubric["rubric_id"]
        current = current_version(conn, rubric_id)
        draft = get_draft(conn, rubric_id)
        definition = current.definition if current else (draft.definition if draft else None)
        if definition is None:
            continue
        out.append(RubricSummary(
            rubric_id=rubric_id,
            name=definition.name,
            description=definition.description,
            category=definition.category,
            pass_threshold=definition.pass_threshold,
            criteria_count=len(definition.criteria),
            current_version=current.ref.version if current else None,
            current_digest=current.ref.digest if current else None,
            has_draft=draft is not None,
            updated_at=db.parse_ts(rubric["updated_at"]),
        ))
    return out


def is_retired(conn: StoreConnection, rubric_id: str) -> bool:
    current = current_version(conn, rubric_id)
    return current is not None and current.status is RubricVersionStatus.RETIRED


def save_draft(conn: StoreConnection, rubric_id: str, definition: RubricDefinition, expected: Optional[int], account_id: str) -> RubricDraft:
    if definition.rubric_id != rubric_id:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "definition.rubric_id must equal the rubric in the path", details={"field": "definition.rubric_id"})
    now = db.ts(conn.now())
    rubric = rubric_row(conn, rubric_id)
    draft = draft_row(conn, rubric_id)
    expected = expected or 0
    if draft is None and expected != 0:
        raise _conflict(conn, rubric_id, "There is no draft at that revision; create one with expected_draft_revision 0")
    if draft is not None and expected != int(draft["draft_revision"]):
        raise _conflict(conn, rubric_id, "The draft changed since you read it")
    if rubric is None:
        conn.execute("INSERT INTO results_rubrics (rubric_id, current_version, last_draft_revision, created_at, updated_at) VALUES (?, NULL, 0, ?, ?)",
                     (rubric_id, now, now))
        rubric = rubric_row(conn, rubric_id)
    revision = int(rubric["last_draft_revision"]) + 1
    conn.execute("UPDATE results_rubrics SET last_draft_revision = ?, updated_at = ? WHERE rubric_id = ?", (revision, now, rubric_id))
    based_on = int(draft["based_on_version"]) if draft is not None and draft["based_on_version"] is not None else rubric["current_version"]
    conn.execute(
        "INSERT INTO results_rubric_drafts (rubric_id, definition_json, draft_revision, based_on_version, updated_at, updated_by_account_id) "
        "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(rubric_id) DO UPDATE SET definition_json = excluded.definition_json, "
        "draft_revision = excluded.draft_revision, based_on_version = excluded.based_on_version, updated_at = excluded.updated_at, "
        "updated_by_account_id = excluded.updated_by_account_id",
        (rubric_id, db.dumps(definition), revision, based_on, now, account_id),
    )
    feed.append(conn, ChangeKind.RUBRIC, rubric_id, revision, "draft_saved")
    return get_draft(conn, rubric_id)


def discard_draft(conn: StoreConnection, rubric_id: str) -> None:
    draft = draft_row(conn, rubric_id)
    if draft is None:
        raise not_found("Rubric draft", rubric_id=rubric_id)
    conn.execute("DELETE FROM results_rubric_drafts WHERE rubric_id = ?", (rubric_id,))
    conn.execute("UPDATE results_rubrics SET updated_at = ? WHERE rubric_id = ?", (db.ts(conn.now()), rubric_id))
    rubric = rubric_row(conn, rubric_id)
    if rubric["current_version"] is None:
        conn.execute("DELETE FROM results_rubrics WHERE rubric_id = ?", (rubric_id,))
    feed.append(conn, ChangeKind.RUBRIC, rubric_id, int(draft["draft_revision"]), "draft_discarded")


def _validate_publishable(definition: RubricDefinition) -> None:
    if not definition.criteria:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "A published rubric needs at least one criterion", details={"field": "definition.criteria"})
    if sum(c.weight for c in definition.criteria) <= 0:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "A published rubric needs a positive total weight", details={"field": "definition.criteria.weight"})


def publish(conn: StoreConnection, rubric_id: str, *, expected_current_version: int, expected_draft_revision: int, notes: Optional[str],
            account_id: str) -> RubricVersion:
    rubric = rubric_row(conn, rubric_id)
    draft = draft_row(conn, rubric_id)
    if rubric is None or draft is None:
        raise not_found("Rubric draft", rubric_id=rubric_id)
    current = int(rubric["current_version"] or 0)
    if expected_current_version != current:
        raise _conflict(conn, rubric_id, "A newer version was published since you read the rubric")
    if expected_draft_revision != int(draft["draft_revision"]):
        raise _conflict(conn, rubric_id, "The draft changed since you read it")
    definition = RubricDefinition.model_validate(db.loads(draft["definition_json"]))
    _validate_publishable(definition)
    version = current + 1
    now = db.ts(conn.now())
    conn.execute(
        "INSERT INTO results_rubric_versions (rubric_id, version, digest, definition_json, status, published_at, published_by_account_id, notes) "
        "VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
        (rubric_id, version, canonical_digest(definition), db.dumps(definition), now, account_id, notes),
    )
    conn.execute("UPDATE results_rubrics SET current_version = ?, updated_at = ? WHERE rubric_id = ?", (version, now, rubric_id))
    conn.execute("DELETE FROM results_rubric_drafts WHERE rubric_id = ?", (rubric_id,))
    feed.append(conn, ChangeKind.RUBRIC, rubric_id, version, "published")
    return get_version(conn, rubric_id, version)


def retire(conn: StoreConnection, rubric_id: str, *, expected_current_version: int) -> RubricVersion:
    rubric = rubric_row(conn, rubric_id)
    if rubric is None or rubric["current_version"] is None:
        raise not_found("Rubric", rubric_id=rubric_id)
    current = int(rubric["current_version"])
    if expected_current_version != current:
        raise _conflict(conn, rubric_id, "A newer version was published since you read the rubric")
    now = db.ts(conn.now())
    conn.execute("UPDATE results_rubric_versions SET status = 'retired' WHERE rubric_id = ? AND version = ?", (rubric_id, current))
    conn.execute("UPDATE results_rubrics SET updated_at = ? WHERE rubric_id = ?", (now, rubric_id))
    feed.append(conn, ChangeKind.RUBRIC, rubric_id, current, "retired")
    return get_version(conn, rubric_id, current)


def published_snapshot(conn: StoreConnection, rubric_id: str, version: int) -> Optional[RubricSnapshotContent]:
    found = get_version(conn, rubric_id, version)
    if found is None:
        return None
    return RubricSnapshotContent(source="published", rubric_id=rubric_id, rubric_version=version, digest=found.ref.digest,
                                 definition=found.definition)


def draft_snapshot(conn: StoreConnection, rubric_id: str, *, expected_draft_revision: int) -> RubricSnapshotContent:
    draft = draft_row(conn, rubric_id)
    if draft is None:
        raise not_found("Rubric draft", rubric_id=rubric_id)
    if int(draft["draft_revision"]) != expected_draft_revision:
        raise _conflict(conn, rubric_id, "The draft changed since you read it")
    definition = RubricDefinition.model_validate(db.loads(draft["definition_json"]))
    return RubricSnapshotContent(source="draft", rubric_id=rubric_id, draft_revision=int(draft["draft_revision"]),
                                 digest=canonical_digest(definition), definition=definition)
