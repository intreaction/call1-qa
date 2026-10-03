"""Calls and result projections, audio playback, semantic search."""

from __future__ import annotations

from typing import Annotated, List

from fastapi import Depends, Query
from starlette.responses import Response

from call1.contracts.calls import (
    CallDetail,
    CallListItem,
    CallListQuery,
    ContactSignalsView,
    EvaluationView,
    SemanticSearchQuery,
    SemanticSearchResponse,
    SummaryView,
    TranscriptView,
)
from call1.contracts.auth import Permission
from call1.contracts.common import Page

from .. import db, feed, pagination
from ..context import Store
from ..db import StoreConnection, read_snapshot
from ..deps import current_principal, get_conn, get_store
from ..principals import Principal, SessionPrincipal
from ..errors import not_found
from ..queue import api as queue_api
from . import audio, records, search, signals
from .routes import router


def _call(conn: StoreConnection, call_id: str):
    row = records.call_row(conn, call_id)
    if row is None:
        raise not_found("Call", call_id=call_id)
    return row


@router.operation("listCalls")
def list_calls(query: Annotated[CallListQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> Page[CallListItem]:
    where: List[str] = []
    args: list = []
    after = pagination.decode(query.page_token, 2)
    if after is not None:
        where.append("(created_at < ? OR (created_at = ? AND call_id < ?))")
        args.extend([after[0], after[0], after[1]])
    if query.agent_id is not None:
        where.append("agent_id = ?")
        args.append(query.agent_id)
    if query.needs_review is not None:
        where.append("requires_human_review = ?")
        args.append(1 if query.needs_review else 0)
    if query.rubric_id is not None:
        where.append("rubric_id = ?")
        args.append(query.rubric_id)
    if query.created_after is not None:
        where.append("created_at >= ?")
        args.append(db.ts(query.created_after))
    if query.created_before is not None:
        where.append("created_at < ?")
        args.append(db.ts(query.created_before))
    signal_where, signal_args = signals.list_filters(conn, category=query.signal_category, subcategory=query.signal_subcategory,
                                                     alert=query.signal_alert)
    where.extend(signal_where)
    args.extend(signal_args)
    if query.text:
        # 1.1.0: agent ID, display name, extension and external call reference (not transcript search).
        columns = ("agent_id", "agent_display_name", "agent_extension", "external_call_ref")
        where.append("(" + " OR ".join(f"{column} LIKE ? ESCAPE '\\'" for column in columns) + ")")
        pattern = "%" + query.text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        args.extend([pattern] * len(columns))
    sql = "SELECT * FROM results_calls" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC, call_id DESC LIMIT ?"
    with read_snapshot(conn):
        rows = conn.execute(sql, (*args, query.limit + 1)).fetchall()
        context = signals.ListContext.load(conn)
        items = [records.call_list_item(conn, r, context) for r in rows[: query.limit]]
    token = pagination.encode(rows[query.limit - 1]["created_at"], rows[query.limit - 1]["call_id"]) if len(rows) > query.limit else None
    return Page[CallListItem](items=items, next_page_token=token)


@router.operation("getCall")
def get_call(call_id: str, conn: StoreConnection = Depends(get_conn)) -> CallDetail:
    with read_snapshot(conn):
        row = _call(conn, call_id)
        conversation_id = row["conversation_id"]
        return CallDetail(
            call=records.call_record_view(row),
            results=records.result_groups(conn, conversation_id),
            pending_work=queue_api.pending_work(conn, conversation_id),
            review_version=records.review_version_of(conn, call_id),
            evaluation=records.evaluation_view(conn, call_id, conversation_id),
            change_cursor=feed.latest_cursor(conn),
        )


@router.operation("getTranscript")
def get_transcript(call_id: str, conn: StoreConnection = Depends(get_conn)) -> TranscriptView:
    with read_snapshot(conn):
        row = _call(conn, call_id)
        view = records.transcript_view(conn, call_id, row["conversation_id"])
    if view is None:
        raise not_found("Transcript", call_id=call_id)
    return view


@router.operation("getEvaluation")
def get_evaluation(call_id: str, conn: StoreConnection = Depends(get_conn)) -> EvaluationView:
    with read_snapshot(conn):
        row = _call(conn, call_id)
        view = records.evaluation_view(conn, call_id, row["conversation_id"])
    if view is None:
        raise not_found("Evaluation", call_id=call_id)
    return view


@router.operation("getEvaluationVersion")
def get_evaluation_version(call_id: str, version: int, conn: StoreConnection = Depends(get_conn)) -> EvaluationView:
    with read_snapshot(conn):
        row = _call(conn, call_id)
        view = records.evaluation_view(conn, call_id, row["conversation_id"], version=version)
    if view is None:
        raise not_found("Evaluation version", call_id=call_id, version=version)
    return view


@router.operation("getSummary")
def get_summary(call_id: str, conn: StoreConnection = Depends(get_conn)) -> SummaryView:
    with read_snapshot(conn):
        row = _call(conn, call_id)
        view = records.summary_view(conn, call_id, row["conversation_id"])
    if view is None:
        raise not_found("Summary", call_id=call_id)
    return view


@router.operation("getContactSignals")
def get_contact_signals(call_id: str, conn: StoreConnection = Depends(get_conn),
                        principal: Principal = Depends(current_principal)) -> ContactSignalsView:
    admin = isinstance(principal, SessionPrincipal) and principal.can(Permission.MANAGE_SIGNALS)
    with read_snapshot(conn):
        row = _call(conn, call_id)
        view = records.contact_signals_view(conn, call_id, row["conversation_id"], admin=admin)
    if view is None:
        raise not_found("Contact signals", call_id=call_id)
    return view


@router.operation("getCallAudio")
def get_call_audio(call_id: str, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn)) -> Response:
    return audio.call_audio(store, conn, call_id)


@router.operation("semanticSearch")
def semantic_search(body: SemanticSearchQuery, conn: StoreConnection = Depends(get_conn)) -> SemanticSearchResponse:
    return search.search(conn, body)
