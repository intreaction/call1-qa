"""Rubrics (drafts, publish, versions, retire) and metrics."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query

from call1.contracts.common import Page
from call1.contracts.events import AuditAction
from call1.contracts.metrics import ExecutiveMetrics, MetricsQuery, ReviewAgreementMetrics, RubricMetrics
from call1.contracts.rubrics import (
    RubricDraft,
    RubricDraftSave,
    RubricListQuery,
    RubricPublishRequest,
    RubricRetireRequest,
    RubricSummary,
    RubricVersion,
)

from .. import audit, pagination
from ..db import StoreConnection, read_snapshot, transaction
from ..deps import current_principal, get_conn
from ..errors import not_found
from ..principals import Principal, require_session
from . import metrics, rubric_store
from .routes import router


@router.operation("listRubrics")
def list_rubrics(query: Annotated[RubricListQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> Page[RubricSummary]:
    with read_snapshot(conn):
        items = rubric_store.summaries(conn)
        if not query.include_retired:
            items = [s for s in items if not rubric_store.is_retired(conn, s.rubric_id)]
    if query.category is not None:
        items = [s for s in items if s.category is query.category]
    after = pagination.decode(query.page_token, 1)
    if after is not None:
        items = [s for s in items if s.rubric_id > after[0]]
    page = items[: query.limit]
    token = pagination.encode(page[-1].rubric_id) if len(items) > query.limit else None
    return Page[RubricSummary](items=page, next_page_token=token)


@router.operation("getRubric")
def get_rubric(rubric_id: str, conn: StoreConnection = Depends(get_conn)) -> RubricVersion:
    with read_snapshot(conn):
        found = rubric_store.current_version(conn, rubric_id)
    if found is None:
        raise not_found("Rubric", rubric_id=rubric_id)
    return found


@router.operation("listRubricVersions")
def list_rubric_versions(rubric_id: str, conn: StoreConnection = Depends(get_conn)) -> Page[RubricVersion]:
    with read_snapshot(conn):
        if rubric_store.rubric_row(conn, rubric_id) is None:
            raise not_found("Rubric", rubric_id=rubric_id)
        return Page[RubricVersion](items=rubric_store.list_versions(conn, rubric_id), next_page_token=None)


@router.operation("getRubricVersion")
def get_rubric_version(rubric_id: str, version: int, conn: StoreConnection = Depends(get_conn)) -> RubricVersion:
    with read_snapshot(conn):
        found = rubric_store.get_version(conn, rubric_id, version)
    if found is None:
        raise not_found("Rubric version", rubric_id=rubric_id, version=version)
    return found


@router.operation("getRubricDraft")
def get_rubric_draft(rubric_id: str, conn: StoreConnection = Depends(get_conn)) -> RubricDraft:
    with read_snapshot(conn):
        found = rubric_store.get_draft(conn, rubric_id)
    if found is None:
        raise not_found("Rubric draft", rubric_id=rubric_id)
    return found


@router.operation("saveRubricDraft")
def save_rubric_draft(rubric_id: str, body: RubricDraftSave, conn: StoreConnection = Depends(get_conn),
                      principal: Principal = Depends(current_principal)) -> RubricDraft:
    session = require_session(principal)
    with transaction(conn):
        return rubric_store.save_draft(conn, rubric_id, body.definition, body.expected_draft_revision, session.account_id)


@router.operation("discardRubricDraft")
def discard_rubric_draft(rubric_id: str, conn: StoreConnection = Depends(get_conn), principal: Principal = Depends(current_principal)) -> None:
    require_session(principal)
    with transaction(conn):
        rubric_store.discard_draft(conn, rubric_id)
    return None


@router.operation("publishRubric")
def publish_rubric(rubric_id: str, body: RubricPublishRequest, conn: StoreConnection = Depends(get_conn),
                   principal: Principal = Depends(current_principal)) -> RubricVersion:
    session = require_session(principal)
    with transaction(conn):
        published = rubric_store.publish(conn, rubric_id, expected_current_version=body.expected_current_version,
                                         expected_draft_revision=body.expected_draft_revision, notes=body.notes, account_id=session.account_id)
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.RUBRIC_PUBLISHED, target_kind="rubric", target_id=rubric_id,
                     details={"version": published.ref.version, "digest": published.ref.digest, "draft_revision": body.expected_draft_revision,
                              "criteria_count": len(published.definition.criteria)})
        return published


@router.operation("retireRubric")
def retire_rubric(rubric_id: str, body: RubricRetireRequest, conn: StoreConnection = Depends(get_conn),
                  principal: Principal = Depends(current_principal)) -> RubricVersion:
    require_session(principal)
    with transaction(conn):
        retired = rubric_store.retire(conn, rubric_id, expected_current_version=body.expected_current_version)
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.RUBRIC_RETIRED, target_kind="rubric", target_id=rubric_id,
                     details={"version": retired.ref.version, "reason": body.reason[:500]})
        return retired


@router.operation("getExecutiveMetrics")
def get_executive_metrics(query: Annotated[MetricsQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> ExecutiveMetrics:
    return metrics.executive(conn, query)


@router.operation("getRubricMetrics")
def get_rubric_metrics(rubric_id: str, query: Annotated[MetricsQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> RubricMetrics:
    return metrics.rubric(conn, rubric_id, query)


@router.operation("getReviewAgreement")
def get_review_agreement(query: Annotated[MetricsQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> ReviewAgreementMetrics:
    return metrics.review_agreement(conn, query)
