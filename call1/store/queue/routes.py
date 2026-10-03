"""Queue-area handlers: one per contract operation this area owns (``call1.store.routing._QUEUE``).

Handlers are thin: they bind the contract's path, query and body models and hand off to the area
modules (``reads``, ``artifacts``, ``lifecycle``, ``reanalysis``, ``usage``, ``catalog``,
``progress``). ``create_app`` adds the method, path, status code, response model and principal
guard, and checks each signature against the contract route at startup. The processing queue here
is never the human review queue (that is the results area).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Query
from starlette.responses import JSONResponse, Response

from call1.contracts import artifacts, calls, catalog, jobs, rubrics, signals, usage
from call1.contracts.common import Page
from call1.contracts.events import AuditAction
from call1.store.routing import Area, AreaRouter

from .. import audit, db, devmode
from ..context import Store
from ..db import StoreConnection
from ..deps import current_principal, get_conn, get_store
from ..principals import Principal, require_service_key, require_session

router = AreaRouter(Area.QUEUE)

StoreDep = Annotated[Store, Depends(get_store)]
ConnDep = Annotated[StoreConnection, Depends(get_conn)]
PrincipalDep = Annotated[Principal, Depends(current_principal)]
IdempotencyKeyHeader = Annotated[str, Header(alias="Idempotency-Key")]

from . import artifacts as artifact_service  # noqa: E402
from . import catalog as catalog_service  # noqa: E402
from . import lifecycle, progress, reads, reanalysis  # noqa: E402
from . import signals as signal_service  # noqa: E402
from . import usage as usage_service  # noqa: E402


# --- conversations and artifacts -----------------------------------------------------------------


@router.operation("registerConversation")
def register_conversation(body: calls.ConversationRegistration, conn: ConnDep, principal: PrincipalDep) -> calls.ConversationRegistered:
    return reads.register_conversation(conn, require_service_key(principal), body)


@router.operation("getConversation")
def get_conversation(conversation_id: str, conn: ConnDep) -> calls.Conversation:
    with db.read_snapshot(conn):
        return reads.get_conversation(conn, conversation_id)


@router.operation("createInlineArtifact")
def create_inline_artifact(conversation_id: str, body: artifacts.InlineArtifactCreate, store: StoreDep, conn: ConnDep) -> artifacts.Artifact:
    return artifact_service.create_inline(conn, store, conversation_id, body)


@router.operation("mintRubricSnapshot")
def mint_rubric_snapshot(conversation_id: str, body: rubrics.RubricSnapshotRequest, store: StoreDep, conn: ConnDep) -> artifacts.Artifact:
    return artifact_service.mint_rubric_snapshot(conn, store, conversation_id, body)


@router.operation("mintSignalTaxonomySnapshot")
def mint_signal_taxonomy_snapshot(conversation_id: str, body: signals.SignalTaxonomySnapshotRequest, store: StoreDep, conn: ConnDep) -> artifacts.Artifact:
    return artifact_service.mint_signal_taxonomy_snapshot(conn, store, conversation_id, body)


@router.operation("createUploadGrant")
def create_upload_grant(conversation_id: str, body: artifacts.UploadGrantRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> JSONResponse:
    grant = artifact_service.create_upload_grant(conn, store, require_service_key(principal), conversation_id, body)
    return devmode.respond(artifacts.UploadGrant, grant, store.config, status_code=201)


@router.operation("commitUpload")
def commit_upload(upload_id: str, body: artifacts.UploadCommit, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> artifacts.Artifact:
    return artifact_service.commit_upload(conn, store, principal, upload_id, body)


@router.operation("listArtifacts")
def list_artifacts(conversation_id: str, query: Annotated[artifacts.ArtifactListQuery, Query()], conn: ConnDep) -> Page[artifacts.Artifact]:
    with db.read_snapshot(conn):
        return artifact_service.list_artifacts(conn, conversation_id, query)


@router.operation("getArtifact")
def get_artifact(artifact_id: str, conn: ConnDep) -> artifacts.Artifact:
    from .records import artifact_from_row, require_artifact_row

    return artifact_from_row(require_artifact_row(conn, artifact_id))


@router.operation("getContentGrant")
def get_content_grant(artifact_id: str, store: StoreDep, conn: ConnDep) -> JSONResponse:
    return devmode.respond(artifacts.ContentGrant, artifact_service.content_grant(conn, store, artifact_id), store.config)


@router.operation("getArtifactContent")
def get_artifact_content(artifact_id: str, store: StoreDep, conn: ConnDep) -> Response:
    from .records import require_artifact_row

    row = require_artifact_row(conn, artifact_id)
    return store.objects.file_response(row["checksum"], row["content_type"])


# --- the processing queue -----------------------------------------------------------------------


@router.operation("createJobGraph")
def create_job_graph(conversation_id: str, body: jobs.JobGraphRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.JobGraph:
    return lifecycle.create_graph(conn, store, require_service_key(principal), conversation_id, body)


@router.operation("getJobGraph")
def get_job_graph(graph_id: str, conn: ConnDep) -> jobs.JobGraph:
    with db.read_snapshot(conn):
        return reads.get_graph(conn, graph_id)


@router.operation("claimJobs")
def claim_jobs(body: jobs.ClaimRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.ClaimResponse:
    response = lifecycle.claim(conn, store, require_service_key(principal), body)
    lifecycle.maybe_sweep(store)  # orphan artifacts and expired upload sessions, at most every few minutes
    return response


@router.operation("listJobs")
def list_jobs(query: Annotated[jobs.JobListQuery, Query()], conn: ConnDep) -> Page[jobs.Job]:
    with db.read_snapshot(conn):
        return reads.list_jobs(conn, query)


@router.operation("getJob")
def get_job(job_id: str, conn: ConnDep) -> jobs.Job:
    with db.read_snapshot(conn):
        return reads.get_job(conn, job_id)


@router.operation("listAttempts")
def list_attempts(job_id: str, conn: ConnDep) -> Page[jobs.Attempt]:
    with db.read_snapshot(conn):
        return reads.list_attempts(conn, job_id)


@router.operation("heartbeat")
def heartbeat(job_id: str, body: jobs.HeartbeatRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.HeartbeatResponse:
    return lifecycle.heartbeat(conn, store, require_service_key(principal), job_id, body)


@router.operation("completeJob")
def complete_job(job_id: str, body: jobs.CompletionRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.CompletionReceipt:
    return lifecycle.complete(conn, store, require_service_key(principal), job_id, body)


@router.operation("failJob")
def fail_job(job_id: str, body: jobs.FailureRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.FailureReceipt:
    return lifecycle.fail(conn, store, require_service_key(principal), job_id, body)


@router.operation("releaseJob")
def release_job(job_id: str, body: jobs.JobReleaseRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.JobReleaseReceipt:
    return lifecycle.release(conn, store, require_service_key(principal), job_id, body)


@router.operation("attachLateUsage")
def attach_late_usage(job_id: str, attempt_number: int, body: usage.LateUsageReport, conn: ConnDep) -> usage.UsageRecord:
    return usage_service.attach_late_usage(conn, job_id, attempt_number, body)


@router.operation("retryJob")
def retry_job(job_id: str, body: jobs.RetryRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.Job:
    return lifecycle.retry(conn, store, principal, job_id, body)


@router.operation("cancelJob")
def cancel_job(job_id: str, body: jobs.CancelRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.CancelResponse:
    return lifecycle.cancel(conn, store, principal, job_id, body)


@router.operation("getGroupProgress")
def get_group_progress(conversation_id: str, conn: ConnDep) -> jobs.JobGroupProgress:
    with db.read_snapshot(conn):
        return progress.group_progress(conn, conversation_id)


@router.operation("listConversationUsage")
def list_conversation_usage(conversation_id: str, conn: ConnDep) -> Page[usage.UsageRecord]:
    with db.read_snapshot(conn):
        return usage_service.conversation_usage(conn, conversation_id)


# --- reanalysis requests and draft tests -----------------------------------------------------------


@router.operation("requestReanalysis")
def request_reanalysis(call_id: str, body: jobs.ReanalysisRequestCreate, idempotency_key: IdempotencyKeyHeader, store: StoreDep,
                       conn: ConnDep, principal: PrincipalDep) -> jobs.ReanalysisRequest:
    return reanalysis.request_reanalysis(conn, store, require_session(principal), call_id, body, idempotency_key)


@router.operation("testRubricDraft")
def test_rubric_draft(rubric_id: str, body: jobs.DraftTestRequest, idempotency_key: IdempotencyKeyHeader, store: StoreDep,
                      conn: ConnDep, principal: PrincipalDep) -> jobs.ReanalysisRequest:
    return reanalysis.create_draft_test(conn, store, require_session(principal), rubric_id, body, idempotency_key)


test_rubric_draft.__test__ = False  # not a pytest test, whatever imports it


@router.operation("getDraftTestResult")
def get_draft_test_result(request_id: str, store: StoreDep, conn: ConnDep) -> jobs.DraftTestResult:
    with db.read_snapshot(conn):
        return reanalysis.draft_result(conn, store, request_id)


@router.operation("listReanalysisRequests")
def list_reanalysis_requests(call_id: str, conn: ConnDep) -> Page[jobs.ReanalysisRequest]:
    with db.read_snapshot(conn):
        return reanalysis.list_for_call(conn, call_id)


@router.operation("getReanalysisRequest")
def get_reanalysis_request(request_id: str, conn: ConnDep) -> jobs.ReanalysisRequest:
    with db.read_snapshot(conn):
        return reanalysis.get(conn, request_id)


@router.operation("claimReanalysisRequests")
def claim_reanalysis_requests(body: jobs.ReanalysisClaimRequest, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.ReanalysisClaimResponse:
    return reanalysis.claim(conn, store, require_service_key(principal), body)


@router.operation("rejectReanalysisRequest")
def reject_reanalysis_request(request_id: str, body: jobs.ReanalysisReject, store: StoreDep, conn: ConnDep, principal: PrincipalDep) -> jobs.ReanalysisRequest:
    return reanalysis.reject(conn, store, require_service_key(principal), request_id, body)


# --- contact signals v2: previews, compares and backfills (1.3.0) ----------------------------------


@router.operation("createSignalPreview")
def create_signal_preview(body: signals.SignalPreviewCreate, idempotency_key: IdempotencyKeyHeader, store: StoreDep, conn: ConnDep,
                          principal: PrincipalDep) -> signals.SignalPreview:
    return signal_service.create_preview(conn, store, require_session(principal), body, idempotency_key)


@router.operation("getSignalPreview")
def get_signal_preview(preview_id: str, store: StoreDep, conn: ConnDep) -> signals.SignalPreview:
    with db.read_snapshot(conn):
        return signal_service.get_preview(conn, store, preview_id)


@router.operation("createSignalBackfill")
def create_signal_backfill(body: signals.SignalBackfillCreate, idempotency_key: IdempotencyKeyHeader, store: StoreDep, conn: ConnDep,
                           principal: PrincipalDep) -> signals.SignalBackfill:
    return signal_service.create_backfill(conn, store, require_session(principal), body, idempotency_key)


# --- usage, hardware and catalog -----------------------------------------------------------------


@router.operation("usageReport")
def usage_report(body: usage.UsageReportQuery, conn: ConnDep, principal: PrincipalDep) -> usage.UsageReport:
    with db.transaction(conn):
        report = usage_service.build_report(conn, body)
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.USAGE_REPORT_EXPORTED, target_kind="usage_report", target_id="report",
                     details={"format": "json", "start": db.ts(body.start), "end": db.ts(body.end), "rows": len(report.rows)})
    return report


@router.operation("listUsageRecords")
def list_usage_records(query: Annotated[usage.UsageRecordQuery, Query()], conn: ConnDep) -> Page[usage.UsageRecord]:
    with db.read_snapshot(conn):
        return usage_service.list_records(conn, query)


@router.operation("exportUsageRecordsCsv")
def export_usage_records_csv(body: usage.UsageReportQuery, conn: ConnDep, principal: PrincipalDep) -> Response:
    with db.transaction(conn):
        records = usage_service.report_records(conn, body)
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.USAGE_REPORT_EXPORTED, target_kind="usage_report", target_id="records.csv",
                     details={"format": "csv", "start": db.ts(body.start), "end": db.ts(body.end), "rows": len(records)})
    return Response(usage_service.csv_export(records), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="usage-records.csv"'})


@router.operation("getUsageMedians")
def get_usage_medians(query: Annotated[usage.UsageMediansQuery, Query()], conn: ConnDep) -> usage.UsageMedians:
    with db.read_snapshot(conn):
        return usage_service.medians(conn, query)


@router.operation("upsertHardwareProfile")
def upsert_hardware_profile(body: usage.HardwareProfileInput, conn: ConnDep) -> usage.HardwareProfile:
    return usage_service.upsert_hardware_profile(conn, body)


@router.operation("listHardwareProfiles")
def list_hardware_profiles(conn: ConnDep) -> Page[usage.HardwareProfile]:
    with db.read_snapshot(conn):
        return usage_service.list_hardware_profiles(conn)


@router.operation("publishCatalogSnapshot")
def publish_catalog_snapshot(body: catalog.CatalogSnapshot, conn: ConnDep, principal: PrincipalDep) -> catalog.CatalogSnapshot:
    return catalog_service.publish(conn, require_service_key(principal), body)


@router.operation("listCatalogSnapshots")
def list_catalog_snapshots(conn: ConnDep) -> Page[catalog.CatalogSnapshot]:
    with db.read_snapshot(conn):
        return catalog_service.list_snapshots(conn)
