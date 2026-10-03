"""The processing-job lifecycle: graph creation, admission, atomic claims, heartbeats, completion,
failure, release, lease expiry, manual retry and cancel (contract README, "Lease and claim
semantics"; ``jobs.JOB_TRANSITIONS`` is the state machine).

Every write runs in one ``BEGIN IMMEDIATE`` transaction, so SQLite serializes it against every
other write: two claimers never see the same QUEUED job, and a completion commits its outputs,
usage row, receipt, projection, follow-on jobs, dependent release and change events together or
not at all (architecture rule 6). Lease expiry is applied lazily: every claim call sweeps all
expired leases, and heartbeat, complete, fail and release first expire their own job's lease if it
is past ``expires_at + LEASE_GRACE`` (in its own transaction, so the expiry sticks even when the
request is then refused as stale).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from call1.contracts.artifacts import DRAFT_TEST_SLOT_PREFIX, is_draft_test_slot
from call1.contracts.common import canonical_digest
from call1.contracts.errors import JOB_ERROR_CLASSES, ErrorCode, JobErrorClass, JobErrorCode
from call1.contracts.events import AuditAction
from call1.contracts.artifacts import ArtifactKind
from call1.contracts.contents import QaScorecardContent
from call1.contracts.jobs import (
    JOB_TYPE_RULES,
    TERMINAL_STATUSES,
    CancelRequest,
    CancelResponse,
    ClaimedJob,
    ClaimRequest,
    ClaimResponse,
    CompletionReceipt,
    CompletionRequest,
    ExecutionClass,
    FailureReceipt,
    FailureRequest,
    GraphReason,
    HeartbeatRequest,
    HeartbeatResponse,
    Job,
    JobGraph,
    JobGraphRequest,
    JobInput,
    JobOutput,
    JobReleaseReceipt,
    JobReleaseRequest,
    JobStatus,
    JobType,
    DRAFT_TEST_KINDS,
    ReanalysisKind,
    ResolvedInput,
    RetryRequest,
    UpstreamOutcome,
    UpstreamOutput,
)
from call1.contracts.rubrics import DraftRubricRef
from call1.contracts.usage import UsageOutcome

from .. import audit, db
from ..errors import StoreError
from ..hooks import CompletedJob, FailedJob, JobSnapshot, LinkedOutput
from ..ids import new_id
from ..principals import (
    Principal,
    ServiceKeyPrincipal,
    generate_secret,
    hash_secret,
    require_own_installation,
    secrets_equal,
)
from . import reanalysis
from .artifacts import link
from .changes import ChangeLog
from .graphs import (
    downstream_closure,
    graph_model,
    insert_prepared,
    prepare_jobs,
    reaches,
    release_dependents,
    route_permitted,
    try_release,
)
from .records import (
    JobReader,
    artifact_from_row,
    artifact_row,
    call_id_of,
    graph_invalid,
    graph_row,
    job_inputs,
    job_parameters,
    job_selection,
    require_conversation_row,
    require_job_row,
)
from .usage import record_attempt_usage, synthesize_abandoned


def _projections():
    from ..results import projections

    return projections


# --- helpers --------------------------------------------------------------------------------


def backoff_seconds(params, attempts_used: int) -> float:
    delay = params.retry_backoff_base_seconds * (params.retry_backoff_factor ** max(0, attempts_used - 1))
    return float(min(delay, params.retry_backoff_max_seconds))


def snapshot(conn, row) -> JobSnapshot:
    graph = graph_row(conn, row["graph_id"])
    return JobSnapshot(
        job_id=row["id"], conversation_id=row["conversation_id"], call_id=call_id_of(conn, row["conversation_id"]),
        graph_id=row["graph_id"], graph_created_at=db.parse_ts(graph["created_at"]), job_type=JobType(row["job_type"]),
        status=JobStatus(row["status"]), attempt_count=row["attempt_count"], draft_test_request_id=row["draft_test_request_id"],
    )


def _notify_failed(conn, job_id: str, attempt_number: Optional[int], error_code: Optional[JobErrorCode]) -> None:
    row = require_job_row(conn, job_id)
    status = JobStatus(row["status"])
    _projections().on_job_failed(conn, FailedJob(
        job=snapshot(conn, row), attempt_number=attempt_number, status=status, error_code=error_code,
        terminal=status in TERMINAL_STATUSES, occurred_at=conn.now(),
    ))


def _stale(conn, row) -> StoreError:
    return StoreError(ErrorCode.CLAIM_TOKEN_STALE, "The claim token is not the job's active claim",
                      details={"current_attempt_number": row["claim_count"] or None, "status": row["status"], "your_attempt_outcome": None})


def check_active_claim(conn, row, claim_token: str) -> None:
    """Raise 409 ``claim_token_stale`` unless ``claim_token`` is the job's active claim."""
    token_hash = hash_secret(claim_token or "")
    if row["status"] == JobStatus.RUNNING.value and row["lease_claim_token_hash"] and secrets_equal(row["lease_claim_token_hash"], token_hash):
        return
    attempt = conn.execute("SELECT status FROM q_attempts WHERE job_id = ? AND claim_token_hash = ?", (row["id"], token_hash)).fetchone()
    error = _stale(conn, row)
    error.details["your_attempt_outcome"] = attempt["status"] if attempt is not None else None
    raise error


def _require_claim_installation(principal: Principal, row) -> None:
    if not isinstance(principal, ServiceKeyPrincipal) or not secrets_equal(principal.installation_id, row["lease_installation_id"] or ""):
        raise StoreError(ErrorCode.FORBIDDEN, "A Process key acts only on claims of its own installation", details={"reason": "installation_mismatch"})


def _lookup_receipt(conn, job_id: str, key: str, operation: str, digest: str, model):
    row = conn.execute("SELECT * FROM q_receipts WHERE job_id = ? AND completion_key = ?", (job_id, key)).fetchone()
    if row is None:
        return None
    if row["operation"] != operation or row["request_digest"] != digest:
        raise StoreError(ErrorCode.COMPLETION_KEY_REUSED, "This completion key already ended a claim with a different request",
                         details={"original_receipt_id": row["receipt_id"], "original_operation": row["operation"]})
    data = db.loads(row["receipt_json"])
    data["replayed"] = True
    return model.model_validate(data)


def _store_receipt(conn, *, job_id: str, key: str, receipt_id: str, operation: str, digest: str, attempt_number: int, receipt) -> None:
    conn.execute(
        "INSERT INTO q_receipts (job_id, completion_key, receipt_id, operation, request_digest, attempt_number, receipt_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, key, receipt_id, operation, digest, attempt_number, db.dumps(receipt), db.ts(conn.now())),
    )


def _attempt_row(conn, job_id: str, attempt_number: int):
    return conn.execute("SELECT * FROM q_attempts WHERE job_id = ? AND attempt_number = ?", (job_id, attempt_number)).fetchone()


_CLEAR_LEASE = ("lease_worker_id = NULL, lease_installation_id = NULL, lease_attempt_number = NULL, lease_granted_at = NULL, "
                "lease_expires_at = NULL, lease_seconds = NULL, lease_claim_token_hash = NULL")


# --- lease expiry and admission -------------------------------------------------------------


def _expire(conn, store, row, changes: ChangeLog) -> None:
    params = store.config.parameters
    now = conn.now()
    attempt = _attempt_row(conn, row["id"], row["lease_attempt_number"])
    usage = synthesize_abandoned(conn, row, attempt)
    conn.execute("UPDATE q_attempts SET status = 'lease_expired', ended_at = ?, error_code = ?, usage_record_id = ? WHERE job_id = ? AND attempt_number = ?",
                 (db.ts(now), JobErrorCode.LEASE_EXPIRED.value, usage.id, row["id"], row["lease_attempt_number"]))
    next_run = None
    if row["cancel_requested"]:
        status, code = JobStatus.CANCELLED, JobErrorCode.CANCELLED
    elif row["attempt_count"] < row["max_attempts"]:
        status, code = JobStatus.QUEUED, JobErrorCode.LEASE_EXPIRED
        next_run = now + timedelta(seconds=backoff_seconds(params, row["attempt_count"]))
    else:
        status, code = JobStatus.FAILED, JobErrorCode.LEASE_EXPIRED
    terminal = status in TERMINAL_STATUSES
    conn.execute(
        f"UPDATE q_jobs SET status = ?, {_CLEAR_LEASE}, error_code = ?, error_detail = ?, next_run_at = ?, wait_cause = ?, "
        "queued_at = CASE WHEN ? THEN ? ELSE queued_at END, completed_at = ?, updated_at = ? WHERE id = ?",
        (status.value, code.value, "Lease expired without a heartbeat", db.ts(next_run) if next_run else None,
         "retry_backoff" if next_run else None, status is JobStatus.QUEUED, db.ts(now), db.ts(now) if terminal else None, db.ts(now), row["id"]),
    )
    changes.job(row["id"], status, row["conversation_id"])
    if terminal:
        release_dependents(conn, row["id"], changes)
    _notify_failed(conn, row["id"], row["lease_attempt_number"], JobErrorCode.LEASE_EXPIRED)


def expire_leases(conn, store, changes: ChangeLog, *, job_id: Optional[str] = None) -> List[str]:
    """Expire every RUNNING lease past ``expires_at + LEASE_GRACE`` (or just ``job_id``'s)."""
    cutoff = db.ts(conn.now() - timedelta(seconds=store.config.parameters.lease_grace_seconds))
    sql = "SELECT * FROM q_jobs WHERE status = 'RUNNING' AND lease_expires_at < ?"
    args: Tuple = (cutoff,)
    if job_id is not None:
        sql += " AND id = ?"
        args = (cutoff, job_id)
    rows = conn.execute(sql + " ORDER BY lease_expires_at, id", args).fetchall()
    for row in rows:
        _expire(conn, store, row, changes)
    return [row["id"] for row in rows]


def expire_job_if_due(conn, store, job_id: str) -> bool:
    with db.transaction(conn):
        changes = ChangeLog(conn)
        expired = expire_leases(conn, store, changes, job_id=job_id)
        if expired:
            changes.flush()
    return bool(expired)


def admit(conn, store, changes: ChangeLog) -> List[str]:
    """Store-side admission: QUEUED jobs whose frozen route is no longer permitted fail with
    ``route_disabled`` (no attempt consumed). Stage 2 permits the appliance route only, which
    graph creation already enforces, so this finds nothing until admin state exists."""
    failed = []
    now = db.ts(conn.now())
    for row in conn.execute("SELECT * FROM q_jobs WHERE status = 'QUEUED' AND route_class IS NOT NULL").fetchall():
        if route_permitted(row["route_class"]):
            continue
        conn.execute("UPDATE q_jobs SET status = 'FAILED', error_code = ?, error_detail = ?, completed_at = ?, updated_at = ?, next_run_at = NULL WHERE id = ?",
                     (JobErrorCode.ROUTE_DISABLED.value, "Route disabled in admin state", now, now, row["id"]))
        changes.job(row["id"], JobStatus.FAILED, row["conversation_id"])
        release_dependents(conn, row["id"], changes)
        _notify_failed(conn, row["id"], None, JobErrorCode.ROUTE_DISABLED)
        failed.append(row["id"])
    return failed


def sweep(store) -> Dict[str, int]:
    """Maintenance: expire leases, apply admission, expire reanalysis claims, drop old orphans and
    expired upload sessions. Claims run it at most every ``SWEEP_INTERVAL_SECONDS``
    (``maybe_sweep``); a host command or test can call it directly."""
    from .artifacts import sweep_orphans

    with store.connection() as conn:
        with db.transaction(conn):
            changes = ChangeLog(conn)
            expired = expire_leases(conn, store, changes)
            rejected = admit(conn, store, changes)
            released_claims = reanalysis.expire_claims(conn, changes)
            changes.flush()
        orphans = sweep_orphans(conn, store)
        uploads = store.objects.sweep_expired_uploads(conn)
    return {"leases_expired": len(expired), "admission_rejected": len(rejected), "reanalysis_claims_expired": released_claims,
            "orphans_deleted": orphans, "uploads_expired": uploads}


SWEEP_INTERVAL_SECONDS = 600
_last_sweep: Dict[int, float] = {}


def maybe_sweep(store) -> bool:
    """Run ``sweep`` when the last one (for this Store) is older than ``SWEEP_INTERVAL_SECONDS``."""
    now = store.clock.now().timestamp()
    last = _last_sweep.get(id(store))
    if last is not None and now - last < SWEEP_INTERVAL_SECONDS:
        return False
    _last_sweep[id(store)] = now
    sweep(store)
    return True


# --- graphs ---------------------------------------------------------------------------------


def create_graph(conn, store, principal: ServiceKeyPrincipal, conversation_id: str, body: JobGraphRequest) -> JobGraph:
    """``createJobGraph``: idempotent by key; a reanalysis graph also fulfils its request."""
    digest = canonical_digest(body.model_dump(mode="json"))
    with db.transaction(conn):
        require_conversation_row(conn, conversation_id)
        existing = conn.execute("SELECT * FROM q_graphs WHERE installation_id = ? AND conversation_id = ? AND idempotency_key = ?",
                                (principal.installation_id, conversation_id, body.idempotency_key)).fetchone()
        if existing is not None:
            if existing["request_digest"] != digest:
                raise StoreError(ErrorCode.IDEMPOTENCY_KEY_REUSED, "This graph idempotency key was used with a different request",
                                 details={"original_id": existing["id"]})
            return graph_model(conn, existing["id"], created=False)
        request = None
        draft_rubric: Optional[DraftRubricRef] = None
        draft_request_id: Optional[str] = None
        signal_request = None
        if body.reason is GraphReason.REANALYSIS:
            request = reanalysis.require_claim(conn, body.reanalysis_request_id, body.reanalysis_claim_token, conversation_id)
            if ReanalysisKind(request["kind"]) in DRAFT_TEST_KINDS:
                draft_request_id = request["id"]
            if request["kind"] == ReanalysisKind.QA_DRAFT_TEST.value:
                draft_rubric = DraftRubricRef.model_validate(db.loads(request["draft_rubric_json"]))
            if request["kind"] == ReanalysisKind.CONTACT_SIGNALS_PREVIEW.value:
                signal_request = request
        prepared = prepare_jobs(conn, store, conversation_id=conversation_id, installation_id=principal.installation_id,
                                definitions=body.jobs, draft_rubric=draft_rubric, draft_test_request_id=draft_request_id,
                                signal_request=signal_request)
        graph_id = new_id("grf")
        conn.execute(
            "INSERT INTO q_graphs (id, conversation_id, installation_id, idempotency_key, request_digest, reason, reanalysis_request_id, "
            "draft_test_request_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (graph_id, conversation_id, principal.installation_id, body.idempotency_key, digest, body.reason.value,
             body.reanalysis_request_id, draft_request_id, db.ts(conn.now())),
        )
        changes = ChangeLog(conn)
        insert_prepared(conn, prepared=prepared, graph_id=graph_id, conversation_id=conversation_id, installation_id=principal.installation_id,
                        draft_test_request_id=draft_request_id, changes=changes, default_max_attempts=store.config.parameters.default_max_attempts)
        if request is not None:
            reanalysis.fulfil(conn, request, graph_id, changes)
        changes.flush()
        return graph_model(conn, graph_id, created=True)


# --- claim ----------------------------------------------------------------------------------


def _capable(row, principal: ServiceKeyPrincipal, worker, types, routes, entries) -> bool:
    if row["job_type"] not in types:
        return False
    if row["execution_class"] == ExecutionClass.PRIMARY_HOST.value and not (worker.primary_host and principal.primary_host):
        return False
    if row["route_class"] is not None:
        if row["route_class"] not in routes or (row["catalog_entry_id"], row["catalog_entry_version"]) not in entries:
            return False
    return True


def _offer_for(row, offers, remaining) -> Optional[int]:
    for index, offer in enumerate(offers):
        if remaining[index] <= 0 or offer.memory_slot.value != row["memory_slot"]:
            continue
        if (offer.outbound_connection_ref or None) != (row["outbound_connection_ref"] or None):
            continue
        return index
    return None


def _claimed_job(conn, reader: JobReader, job_id: str, offer_index: int, token: str) -> ClaimedJob:
    row = require_job_row(conn, job_id)
    job = reader.job(row)
    resolved = {r.role: r for r in job.resolved_inputs}
    inputs = []
    for job_input in job.inputs:
        ref = resolved.get(job_input.role)
        artifact = artifact_from_row(artifact_row(conn, ref.artifact_id)) if ref is not None else None
        inputs.append(ResolvedInput(role=job_input.role, artifact=artifact))
    view = reader.view(row["conversation_id"])
    upstream = []
    for up, kind in view.upstreams.get(job_id, []):
        node = view.nodes[up]
        upstream.append(UpstreamOutcome(job_id=up, job_type=node.job_type, edge=kind, status=node.status, error_code=node.error_code))
    return ClaimedJob(
        job=job, attempt_number=row["lease_attempt_number"], claim_token=token, lease_expires_at=db.parse_ts(row["lease_expires_at"]),
        slot_offer_index=offer_index, final_attempt=row["attempt_count"] >= row["max_attempts"], inputs=inputs, upstream=upstream,
    )


# Claim order: call by call, not stage by stage. A job's own ``priority`` is Process's stage
# priority (asr 50 ... summary 10) plus its reanalysis request's offset, so ordering by it alone
# advances a whole batch one stage at a time and every call finishes at the end. Instead:
#   1. the ``conversation_id`` affinity hint, when the worker sends one (unchanged);
#   2. the band: the priority of the reanalysis request the graph fulfils (+5 previews, 0 ingest and
#      ordinary reanalysis, -10 backfills and compares), 0 for a graph with no request;
#   3. the graph that entered the queue first (created_at, then insertion order), so within a band
#      the oldest call's ready jobs go before a newer call's;
#   4. within that graph, the job's own priority, then age and id (the old order).
# Eligibility is unchanged: capability, route, catalog entry and slot filtering skip a job the
# worker cannot take, so a newer call's ready jobs still fill slots the older call cannot use.
CLAIM_CANDIDATES_SQL = (
    "SELECT j.* FROM q_jobs j JOIN q_graphs g ON g.id = j.graph_id "
    "LEFT JOIN q_reanalysis_requests r ON r.id = g.reanalysis_request_id "
    "WHERE j.status = 'QUEUED' AND (j.next_run_at IS NULL OR j.next_run_at <= ?) "
    "ORDER BY CASE WHEN j.conversation_id = ? THEN 0 ELSE 1 END, COALESCE(r.priority, 0) DESC, g.created_at, g.rowid, "
    "j.priority DESC, j.created_at, j.id"
)


def claim(conn, store, principal: ServiceKeyPrincipal, body: ClaimRequest) -> ClaimResponse:
    """``claimJobs``: atomically claim eligible jobs against the offered slots."""
    worker = body.worker
    require_own_installation(principal, worker.installation_id)
    if worker.primary_host and not principal.primary_host:
        raise StoreError(ErrorCode.FORBIDDEN, "Only the designated primary host may claim as primary_host", details={"reason": "not_primary_host"})
    params = store.config.parameters
    lease_seconds = min(body.lease_seconds or params.lease_duration_seconds, params.lease_duration_seconds)
    limit = min(body.max_jobs, params.max_claim_batch)
    types = {t.value for t in worker.job_types}
    routes = {r.value for r in worker.route_classes}
    entries = {(e.entry_id, e.entry_version) for e in worker.qualified_entries}
    offers = list(worker.slot_offers)
    remaining = [offer.count for offer in offers]
    claimed: List[Tuple[str, int, str]] = []
    lacked_capability = lacked_slot = False
    with db.transaction(conn):
        changes = ChangeLog(conn)
        expire_leases(conn, store, changes)
        admit(conn, store, changes)
        now = conn.now()
        candidates = conn.execute(CLAIM_CANDIDATES_SQL, (db.ts(now), body.conversation_id or "")).fetchall()
        for row in candidates:
            if len(claimed) >= limit:
                break
            if not _capable(row, principal, worker, types, routes, entries):
                lacked_capability = True
                continue
            index = _offer_for(row, offers, remaining)
            if index is None:
                lacked_slot = True
                continue
            token = generate_secret(32)
            expires = now + timedelta(seconds=lease_seconds)
            attempt_number = row["claim_count"] + 1
            updated = conn.execute(
                "UPDATE q_jobs SET status = 'RUNNING', attempt_count = attempt_count + 1, claim_count = claim_count + 1, lease_worker_id = ?, "
                "lease_installation_id = ?, lease_attempt_number = ?, lease_granted_at = ?, lease_expires_at = ?, lease_seconds = ?, "
                "lease_claim_token_hash = ?, next_run_at = NULL, wait_cause = NULL, updated_at = ? WHERE id = ? AND status = 'QUEUED'",
                (worker.worker_id, principal.installation_id, attempt_number, db.ts(now), db.ts(expires), lease_seconds,
                 hash_secret(token), db.ts(now), row["id"]),
            )
            if updated.rowcount != 1:  # impossible under BEGIN IMMEDIATE; never hand out a claim we did not take
                continue
            queued_at = db.parse_ts(row["queued_at"])
            conn.execute(
                "INSERT INTO q_attempts (job_id, attempt_number, status, counts_as_attempt, started_at, worker_id, installation_id, hardware_profile_id, "
                "claim_token_hash, slot_offer_index, lease_granted_at, lease_expires_at, queue_wait_seconds, resource_estimate_json) "
                "VALUES (?, ?, 'running', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row["id"], attempt_number, db.ts(now), worker.worker_id, principal.installation_id, worker.hardware_profile_id,
                 hash_secret(token), index, db.ts(now), db.ts(expires),
                 max(0.0, (now - queued_at).total_seconds()) if queued_at else None, row["resource_estimate_json"]),
            )
            remaining[index] -= 1
            claimed.append((row["id"], index, token))
            changes.job(row["id"], JobStatus.RUNNING, row["conversation_id"])
        changes.flush()
        reader = JobReader(conn)
        jobs = [_claimed_job(conn, reader, job_id, index, token) for job_id, index, token in claimed]
    reason = None
    if not jobs:
        if lacked_slot:
            reason = "ready jobs need a slot you did not offer"
        elif lacked_capability:
            reason = "ready jobs need capabilities this worker lacks"
        else:
            reason = "no ready jobs"
    return ClaimResponse(jobs=jobs, lease_duration_seconds=lease_seconds,
                         heartbeat_interval_seconds=min(params.heartbeat_interval_seconds, max(5, lease_seconds // 3)),
                         no_eligible_reason=reason)


# --- heartbeat ------------------------------------------------------------------------------


def heartbeat(conn, store, principal: ServiceKeyPrincipal, job_id: str, body: HeartbeatRequest) -> HeartbeatResponse:
    with db.read_snapshot(conn):
        require_job_row(conn, job_id)
    expire_job_if_due(conn, store, job_id)
    with db.transaction(conn):
        row = require_job_row(conn, job_id)
        check_active_claim(conn, row, body.claim_token)
        _require_claim_installation(principal, row)
        now = conn.now()
        expires = now + timedelta(seconds=row["lease_seconds"] or store.config.parameters.lease_duration_seconds)
        conn.execute("UPDATE q_jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?", (db.ts(expires), db.ts(now), job_id))
        conn.execute("UPDATE q_attempts SET lease_expires_at = ? WHERE job_id = ? AND attempt_number = ?", (db.ts(expires), job_id, row["lease_attempt_number"]))
    return HeartbeatResponse(lease_expires_at=expires, cancel_requested=bool(row["cancel_requested"]))


# --- completion -----------------------------------------------------------------------------


def _verify_outputs(conn, row, rule, outputs: List[JobOutput], attempt_number: int):
    """Every role of ``rule.outputs`` exactly once, plus any role of ``rule.optional_outputs`` (1.3.0,
    decision 33: the ``asr`` job's ``base_transcript`` and ``vocabulary_pass``) at most once; nothing
    else. Required roles come first in the result, so ``linked[0]`` stays the job's own output."""
    expected = rule.outputs
    optional = rule.optional_outputs
    given = {o.role: o for o in outputs}
    if len(given) != len(outputs) or not set(expected) <= set(given) or not set(given) <= set(expected) | set(optional):
        raise graph_invalid("A completion links exactly one artifact per output role the job type declares, and at most one per optional role",
                            reason="outputs_mismatch", expected_roles=",".join(sorted(expected)), optional_roles=",".join(sorted(optional)))
    draft = row["draft_test_request_id"]
    verified = []
    for role, kind in [*expected.items(), *((r, k) for r, k in optional.items() if r in given)]:
        output = given[role]
        art = artifact_row(conn, output.artifact_id)
        if art is None:
            raise graph_invalid(f"Output {role}: artifact not found", reason="unknown_artifact", role=role, artifact_id=output.artifact_id)
        if art["checksum"] != output.checksum:
            raise StoreError(ErrorCode.CHECKSUM_MISMATCH, f"Output {role}: checksum differs from the committed artifact", details={"role": role, "reason": "output_checksum_differs"})
        if art["conversation_id"] != row["conversation_id"] or art["kind"] != kind.value:
            raise graph_invalid(f"Output {role}: expected a {kind.value} artifact of this conversation", reason="output_kind", role=role)
        if art["producing_job_id"] != row["id"] or art["producing_attempt_number"] != attempt_number:
            raise graph_invalid(f"Output {role}: not produced by this job under this claim", reason="not_produced_under_this_claim", role=role)
        if art["linked"]:
            raise graph_invalid(f"Output {role}: already linked", reason="already_linked", role=role)
        if draft is not None and not art["slot"].startswith(f"{DRAFT_TEST_SLOT_PREFIX}{draft}:"):
            raise graph_invalid(f"Output {role}: a draft-test output lives in its request's draft slot", reason="draft_slot_required", role=role)
        if draft is None and is_draft_test_slot(art["slot"]):
            raise graph_invalid(f"Output {role}: draft slots belong to draft-test graphs", reason="draft_slot_reserved", role=role)
        verified.append((role, art))
    return verified


def _verify_scorecard_rubric(store, row, outputs) -> None:
    """A qa_scorecard output names the rubric its job was pinned to: the published version, or the
    draft revision under test, and the same digest. The projection copies this ref onto the call."""
    params = job_parameters(row)
    for role, art in outputs:
        if art["kind"] != ArtifactKind.QA_SCORECARD.value:
            continue
        try:
            scored = QaScorecardContent.model_validate_json(store.objects.read_bytes(art["checksum"])).rubric
        except ValueError:
            raise graph_invalid(f"Output {role}: not a valid qa_scorecard document", reason="scorecard_invalid", role=role) from None
        if params.draft_rubric is not None:
            pinned = params.draft_rubric
            matches = (scored.rubric_id == pinned.rubric_id and scored.rubric_version is None
                       and scored.draft_revision == pinned.draft_revision and scored.digest == pinned.digest)
            expected = f"{pinned.rubric_id} draft revision {pinned.draft_revision}"
        elif params.rubric is not None:
            pinned = params.rubric
            matches = (scored.rubric_id == pinned.rubric_id and scored.draft_revision is None
                       and scored.rubric_version == pinned.version and scored.digest == pinned.digest)
            expected = f"{pinned.rubric_id} version {pinned.version}"
        else:  # JobParameters guarantees QA types pin one; nothing to compare against otherwise
            continue
        if not matches:
            raise graph_invalid(f"Output {role}: the scorecard must name the job's rubric ({expected}) and its digest",
                                reason="scorecard_rubric_mismatch", role=role)


def _verify_result(row, rule, result) -> None:
    if row["draft_test_request_id"] is not None:
        if result is not None:
            raise graph_invalid("A draft-test graph never publishes a result", reason="result_in_draft_test")
        return
    if rule.publishes is None:
        if result is not None:
            raise graph_invalid(f"{row['job_type']} does not publish a result group", reason="result_not_published_by_type")
    elif result is None or result.kind is not rule.publishes:
        raise graph_invalid(f"{row['job_type']} completions publish the {rule.publishes.value} result", reason="result_required", kind=rule.publishes.value)


def _verify_usage(row, rule, usage) -> None:
    if usage.outcome not in rule.completion_outcomes:
        raise StoreError(ErrorCode.VALIDATION_FAILED, f"A {row['job_type']} completion cannot carry usage outcome {usage.outcome.value}",
                         details={"field": "usage.outcome", "reason": "outcome_not_allowed"})
    if usage.outcome is UsageOutcome.FAILED and row["attempt_count"] < row["max_attempts"]:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "A provider failure completes FLAGGED only on the claim's final attempt; report it with /fail",
                         details={"field": "usage.outcome", "reason": "not_final_attempt"})


def _verify_route(row, provenance) -> None:
    selection = job_selection(row)
    if selection is None:
        if provenance.route is not None:
            raise StoreError(ErrorCode.ROUTE_NOT_PERMITTED, "A code stage records no model route", details={"reason": "route_on_code_stage"})
        return
    if provenance.route is None or provenance.route.model_dump(mode="json") != selection.route.model_dump(mode="json"):
        raise StoreError(ErrorCode.ROUTE_NOT_PERMITTED, "The attempt's route differs from the job's frozen route", details={"reason": "route_differs_from_frozen"})


def _prepare_follow_on(conn, store, row, principal: ServiceKeyPrincipal, follow_on):
    if follow_on is None:
        return None
    draft_rubric = None
    signal_request = None
    if row["draft_test_request_id"] is not None:
        request = reanalysis.require_request(conn, row["draft_test_request_id"])
        if request["kind"] == ReanalysisKind.QA_DRAFT_TEST.value:
            draft_rubric = DraftRubricRef.model_validate(db.loads(request["draft_rubric_json"]))
        elif request["kind"] == ReanalysisKind.CONTACT_SIGNALS_PREVIEW.value:
            signal_request = request
    prepared = prepare_jobs(conn, store, conversation_id=row["conversation_id"], installation_id=principal.installation_id,
                            definitions=follow_on.jobs, draft_rubric=draft_rubric, draft_test_request_id=row["draft_test_request_id"],
                            signal_request=signal_request)
    by_ref = {p.definition.ref: p for p in prepared}
    extra = {p.job_id: p.requires + p.after for p in prepared if not p.existing}
    bindings = []
    bound_roles: Dict[str, set] = {}
    for dep in follow_on.add_dependencies:
        dependent = conn.execute("SELECT * FROM q_jobs WHERE id = ?", (dep.dependent_job_id,)).fetchone()
        if dependent is None or dependent["conversation_id"] != row["conversation_id"]:
            raise graph_invalid("add_dependencies names a job that is not in this conversation", reason="unknown_dependent", job_id=dep.dependent_job_id)
        if dependent["status"] != JobStatus.BLOCKED.value:
            raise graph_invalid("A new dependency applies only to a BLOCKED dependent", reason="dependent_not_blocked", job_id=dep.dependent_job_id)
        edge = conn.execute("SELECT kind FROM q_edges WHERE job_id = ? AND upstream_job_id = ?", (dependent["id"], row["id"])).fetchone()
        if edge is None or edge["kind"] != "requires":
            raise graph_invalid("A new dependency applies only to a job that directly requires the completing job", reason="not_a_direct_dependent", job_id=dep.dependent_job_id)
        new = by_ref[dep.requires_ref]
        if new.job_id == dependent["id"] or reaches(conn, [new.job_id], dependent["id"], extra):
            raise graph_invalid("The new dependency would create a cycle", reason="dependency_cycle", job_id=dep.dependent_job_id)
        if dep.input_role is not None:
            roles = bound_roles.setdefault(dependent["id"], {i.role for i in job_inputs(dependent)})
            if dep.input_role in roles:
                raise graph_invalid(f"Input role {dep.input_role} already exists on the dependent", reason="input_role_exists", job_id=dep.dependent_job_id)
            roles.add(dep.input_role)
        bindings.append((dependent["id"], new.job_id, dep))
    return prepared, bindings


def _insert_follow_on(conn, store, row, principal: ServiceKeyPrincipal, follow, changes: ChangeLog) -> List[str]:
    prepared, bindings = follow
    created, _queued = insert_prepared(conn, prepared=prepared, graph_id=row["graph_id"], conversation_id=row["conversation_id"],
                                       installation_id=principal.installation_id, draft_test_request_id=row["draft_test_request_id"],
                                       changes=changes, default_max_attempts=store.config.parameters.default_max_attempts)
    for dependent_id, new_job_id, dep in bindings:
        conn.execute("INSERT INTO q_edges (job_id, upstream_job_id, kind) VALUES (?, ?, 'requires') "
                     "ON CONFLICT (job_id, upstream_job_id) DO UPDATE SET kind = 'requires'", (dependent_id, new_job_id))
        if dep.input_role is not None:
            dependent = require_job_row(conn, dependent_id)
            inputs = job_inputs(dependent)
            inputs.append(JobInput(role=dep.input_role, upstream=UpstreamOutput(job_id=new_job_id, output_role=dep.output_role)))
            conn.execute("UPDATE q_jobs SET inputs_json = ?, updated_at = ? WHERE id = ?",
                         (db.dumps([i.model_dump(mode="json") for i in inputs]), db.ts(conn.now()), dependent_id))
        changes.job(dependent_id, JobStatus.BLOCKED, row["conversation_id"])
    return created


def complete(conn, store, principal: ServiceKeyPrincipal, job_id: str, body: CompletionRequest) -> CompletionReceipt:
    """``completeJob``: ``jobs.COMPLETION_TRANSACTION_STEPS`` in one transaction."""
    digest = canonical_digest(body.model_dump(mode="json"))
    with db.read_snapshot(conn):
        require_job_row(conn, job_id)
        replay = _lookup_receipt(conn, job_id, body.completion_key, "complete", digest, CompletionReceipt)
    if replay is not None:
        return replay
    expire_job_if_due(conn, store, job_id)
    with db.transaction(conn):
        # 1. the completion key (again, inside the writing transaction)
        replay = _lookup_receipt(conn, job_id, body.completion_key, "complete", digest, CompletionReceipt)
        if replay is not None:
            return replay
        row = require_job_row(conn, job_id)
        # 2. the active claim, then cancel
        check_active_claim(conn, row, body.claim_token)
        if row["cancel_requested"]:
            raise StoreError(ErrorCode.JOB_CANCELLING, "Cancel was requested; report failure code cancelled instead")
        _require_claim_installation(principal, row)
        require_own_installation(principal, body.provenance.installation_id)
        # 3. outputs, result, usage outcome and provenance
        rule = JOB_TYPE_RULES[JobType(row["job_type"])]
        attempt_number = row["lease_attempt_number"]
        attempt = _attempt_row(conn, job_id, attempt_number)
        outputs = _verify_outputs(conn, row, rule, body.outputs, attempt_number)
        _verify_scorecard_rubric(store, row, outputs)
        _verify_result(row, rule, body.result)
        _verify_usage(row, rule, body.usage)
        _verify_route(row, body.provenance)
        follow = _prepare_follow_on(conn, store, row, principal, body.follow_on)
        now = conn.now()
        receipt_id = new_id("rcpt")
        changes = ChangeLog(conn)
        # 4. link the outputs
        linked: List[LinkedOutput] = []
        superseded: List[str] = []
        for role, art in outputs:
            artifact, previous = link(conn, art["id"], receipt_id=receipt_id)
            linked.append(LinkedOutput(role=role, artifact=artifact))
            if previous is not None:
                superseded.append(previous)
        # 5. usage row, provenance, (receipt below, once the cursor is known)
        usage = record_attempt_usage(conn, row, attempt, body.usage, body.provenance)
        conn.execute("UPDATE q_attempts SET status = 'succeeded', ended_at = ?, provenance_json = ?, usage_record_id = ?, error_code = ? "
                     "WHERE job_id = ? AND attempt_number = ?",
                     (db.ts(now), db.dumps(body.provenance), usage.id, body.usage.error_code.value if body.usage.error_code else None, job_id, attempt_number))
        # 6. the job SUCCEEDED
        job_outputs = [JobOutput(role=o.role, artifact_id=o.artifact.id, checksum=o.artifact.checksum) for o in linked]
        conn.execute(
            f"UPDATE q_jobs SET status = 'SUCCEEDED', {_CLEAR_LEASE}, outputs_json = ?, result_json = ?, error_code = NULL, error_detail = NULL, "
            "next_run_at = NULL, wait_cause = NULL, completed_at = ?, updated_at = ? WHERE id = ?",
            (db.dumps([o.model_dump(mode="json") for o in job_outputs]), db.dumps(body.result) if body.result else None, db.ts(now), db.ts(now), job_id),
        )
        changes.job(job_id, JobStatus.SUCCEEDED, row["conversation_id"])
        # 7. the projection (results skips it for a draft-test graph); a draft test's scorecard is recorded on its request
        done = require_job_row(conn, job_id)
        outcome = _projections().apply_completion(conn, CompletedJob(
            job=snapshot(conn, done), attempt_number=attempt_number, receipt_id=receipt_id, outputs=tuple(linked), result=body.result,
            completed_at=now, superseded_artifact_ids=tuple(superseded),
        ))
        result_version = None
        if row["draft_test_request_id"] is None and outcome is not None:
            result_version = outcome.result_version
            if result_version is not None:
                conn.execute("UPDATE q_jobs SET result_version = ? WHERE id = ?", (result_version, job_id))
        elif row["draft_test_request_id"] is not None and row["job_type"] == JobType.QA_SCORECARD.value:
            reanalysis.record_draft_result(conn, row["draft_test_request_id"], linked[0].artifact.id, changes)
        elif row["draft_test_request_id"] is not None and row["job_type"] == JobType.CONTACT_SIGNALS_MERGE.value:
            reanalysis.record_preview_result(conn, row["draft_test_request_id"], linked[0].artifact.id, changes)
        if row["draft_test_request_id"] is None and row["job_type"] == JobType.CONTACT_SIGNALS_MERGE.value and result_version is not None:
            # Shadow mode (section 14): a v1 publish gets its v2 compare companion in this transaction.
            from .signals import shadow_companion

            shadow_companion(conn, store, row["conversation_id"], linked[0].artifact, changes)
        # 8-9. follow-on jobs, their edges and input bindings, and their initial status
        created = _insert_follow_on(conn, store, row, principal, follow, changes) if follow is not None else []
        # 10. release dependents, change events, the receipt
        released = release_dependents(conn, job_id, changes)
        cursor = changes.flush()
        receipt = CompletionReceipt(
            receipt_id=receipt_id, job_id=job_id, attempt_number=attempt_number, completion_key=body.completion_key, status=JobStatus.SUCCEEDED,
            result_version=result_version, linked_artifact_ids=[o.artifact.id for o in linked], released_job_ids=released,
            created_job_ids=created, usage_record_id=usage.id, change_cursor=cursor, committed_at=now, replayed=False,
        )
        _store_receipt(conn, job_id=job_id, key=body.completion_key, receipt_id=receipt_id, operation="complete", digest=digest,
                       attempt_number=attempt_number, receipt=receipt)
    return receipt


# --- failure and release --------------------------------------------------------------------


def fail(conn, store, principal: ServiceKeyPrincipal, job_id: str, body: FailureRequest) -> FailureReceipt:
    """``failJob``: record the attempt's usage and apply ``JOB_ERROR_CLASSES``."""
    digest = canonical_digest(body.model_dump(mode="json"))
    with db.read_snapshot(conn):
        require_job_row(conn, job_id)
        replay = _lookup_receipt(conn, job_id, body.completion_key, "fail", digest, FailureReceipt)
    if replay is not None:
        return replay
    expire_job_if_due(conn, store, job_id)
    params = store.config.parameters
    with db.transaction(conn):
        replay = _lookup_receipt(conn, job_id, body.completion_key, "fail", digest, FailureReceipt)
        if replay is not None:
            return replay
        row = require_job_row(conn, job_id)
        check_active_claim(conn, row, body.claim_token)
        _require_claim_installation(principal, row)
        if body.provenance is not None:
            require_own_installation(principal, body.provenance.installation_id)
        now = conn.now()
        attempt_number = row["lease_attempt_number"]
        attempt = _attempt_row(conn, job_id, attempt_number)
        error_class = JOB_ERROR_CLASSES[body.error_code]
        next_run = None
        if row["cancel_requested"]:
            status = JobStatus.CANCELLED
        elif error_class is JobErrorClass.TRANSIENT and row["attempt_count"] < row["max_attempts"]:
            status = JobStatus.QUEUED
            next_run = now + timedelta(seconds=backoff_seconds(params, row["attempt_count"]))
        else:
            status = JobStatus.FAILED
        terminal = status in TERMINAL_STATUSES
        usage = record_attempt_usage(conn, row, attempt, body.usage, body.provenance)
        conn.execute(
            "UPDATE q_attempts SET status = ?, ended_at = ?, error_code = ?, error_detail = ?, provenance_json = ?, usage_record_id = ? "
            "WHERE job_id = ? AND attempt_number = ?",
            ("cancelled" if status is JobStatus.CANCELLED else "failed", db.ts(now), body.error_code.value, body.error_detail,
             db.dumps(body.provenance) if body.provenance else None, usage.id, job_id, attempt_number),
        )
        conn.execute(
            f"UPDATE q_jobs SET status = ?, {_CLEAR_LEASE}, error_code = ?, error_detail = ?, next_run_at = ?, wait_cause = ?, "
            "queued_at = CASE WHEN ? THEN ? ELSE queued_at END, completed_at = ?, updated_at = ? WHERE id = ?",
            (status.value, body.error_code.value, body.error_detail, db.ts(next_run) if next_run else None, "retry_backoff" if next_run else None,
             status is JobStatus.QUEUED, db.ts(now), db.ts(now) if terminal else None, db.ts(now), job_id),
        )
        changes = ChangeLog(conn)
        changes.job(job_id, status, row["conversation_id"])
        released = release_dependents(conn, job_id, changes) if terminal else []
        _notify_failed(conn, job_id, attempt_number, body.error_code)
        cursor = changes.flush()
        receipt_id = new_id("rcpt")
        receipt = FailureReceipt(
            receipt_id=receipt_id, job_id=job_id, attempt_number=attempt_number, completion_key=body.completion_key, status=status,
            error_class=error_class, next_run_at=next_run, attempts_remaining=max(0, row["max_attempts"] - row["attempt_count"]),
            # Pro1 is closed in Stage 2 (no call1_confidential job can exist), so no failure blocks its connection.
            pro1_connection_blocked=False, released_job_ids=released, usage_record_id=usage.id, change_cursor=cursor,
            committed_at=now, replayed=False,
        )
        _store_receipt(conn, job_id=job_id, key=body.completion_key, receipt_id=receipt_id, operation="fail", digest=digest,
                       attempt_number=attempt_number, receipt=receipt)
    return receipt


def release(conn, store, principal: ServiceKeyPrincipal, job_id: str, body: JobReleaseRequest) -> JobReleaseReceipt:
    """``releaseJob``: end a claim before inference started; the attempt is refunded, no usage row."""
    digest = canonical_digest(body.model_dump(mode="json"))
    with db.read_snapshot(conn):
        require_job_row(conn, job_id)
        replay = _lookup_receipt(conn, job_id, body.completion_key, "release", digest, JobReleaseReceipt)
    if replay is not None:
        return replay
    expire_job_if_due(conn, store, job_id)
    with db.transaction(conn):
        replay = _lookup_receipt(conn, job_id, body.completion_key, "release", digest, JobReleaseReceipt)
        if replay is not None:
            return replay
        row = require_job_row(conn, job_id)
        check_active_claim(conn, row, body.claim_token)
        _require_claim_installation(principal, row)
        if row["cancel_requested"]:
            raise StoreError(ErrorCode.INVALID_TRANSITION, "Cancel was requested; report failure code cancelled instead of releasing",
                             details={"reason": "cancel_requested"})
        now = conn.now()
        attempt_number = row["lease_attempt_number"]
        conn.execute("UPDATE q_attempts SET status = 'released', counts_as_attempt = 0, ended_at = ?, error_code = ?, error_detail = ? "
                     "WHERE job_id = ? AND attempt_number = ?", (db.ts(now), body.reason_code.value, body.detail, job_id, attempt_number))
        next_run = None
        if body.disposition == "requeue":
            status = JobStatus.QUEUED
            if body.not_before is not None and body.not_before > now:
                next_run = body.not_before
            conn.execute(
                f"UPDATE q_jobs SET status = 'QUEUED', attempt_count = attempt_count - 1, {_CLEAR_LEASE}, next_run_at = ?, wait_cause = ?, updated_at = ? WHERE id = ?",
                (db.ts(next_run) if next_run else None, "deferred_by_worker" if next_run else None, db.ts(now), job_id),
            )
        else:
            status = JobStatus.FAILED
            conn.execute(
                f"UPDATE q_jobs SET status = 'FAILED', attempt_count = attempt_count - 1, {_CLEAR_LEASE}, error_code = ?, error_detail = ?, "
                "next_run_at = NULL, wait_cause = NULL, completed_at = ?, updated_at = ? WHERE id = ?",
                (body.reason_code.value, body.detail, db.ts(now), db.ts(now), job_id),
            )
        changes = ChangeLog(conn)
        changes.job(job_id, status, row["conversation_id"])
        if status is JobStatus.FAILED:
            release_dependents(conn, job_id, changes)
            _notify_failed(conn, job_id, attempt_number, body.reason_code)
        cursor = changes.flush()
        after = require_job_row(conn, job_id)
        receipt = JobReleaseReceipt(
            job_id=job_id, attempt_number=attempt_number, completion_key=body.completion_key, status=status, attempt_count=after["attempt_count"],
            next_run_at=next_run, change_cursor=cursor, committed_at=now, replayed=False,
        )
        _store_receipt(conn, job_id=job_id, key=body.completion_key, receipt_id=new_id("rcpt"), operation="release", digest=digest,
                       attempt_number=attempt_number, receipt=receipt)
    return receipt


# --- manual retry and cancel -----------------------------------------------------------------


def retry(conn, store, principal: Principal, job_id: str, body: RetryRequest) -> Job:
    """``retryJob``: FAILED -> QUEUED (or BLOCKED while an upstream is unsatisfied), plus
    ``RETRY_ATTEMPT_GRANT`` attempts and a new ``retry_generation``. Nothing else reruns."""
    with db.transaction(conn):
        row = require_job_row(conn, job_id)
        if row["status"] != JobStatus.FAILED.value:
            raise StoreError(ErrorCode.INVALID_TRANSITION, f"Only a FAILED job can be retried (this one is {row['status']})",
                             details={"status": row["status"]})
        now = db.ts(conn.now())
        grant = store.config.parameters.retry_attempt_grant
        conn.execute(
            "UPDATE q_jobs SET status = 'BLOCKED', max_attempts = max_attempts + ?, retry_generation = retry_generation + 1, cancel_requested = 0, error_code = NULL, "
            "error_detail = NULL, completed_at = NULL, next_run_at = NULL, wait_cause = NULL, resolved_inputs_json = '[]', updated_at = ? WHERE id = ?",
            (grant, now, job_id),
        )
        try_release(conn, job_id)
        after = require_job_row(conn, job_id)
        changes = ChangeLog(conn)
        changes.job(job_id, after["status"], row["conversation_id"])
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.JOB_RETRIED, target_kind="job", target_id=job_id,
                     details={"retry_generation": after["retry_generation"], "status": after["status"], "max_attempts": after["max_attempts"],
                              "reason": body.reason[:200]})
        changes.flush()
        return JobReader(conn).job(after)


def cancel(conn, store, principal: Principal, job_id: str, body: CancelRequest) -> CancelResponse:
    """``cancelJob``: BLOCKED/QUEUED jobs are CANCELLED now; a RUNNING job gets
    ``cancel_requested`` and ends on its worker's ``cancelled`` failure (or lease expiry). With
    ``cascade`` every dependent is cancelled; without it, after-edge dependents are released."""
    with db.transaction(conn):
        row = require_job_row(conn, job_id)
        status = JobStatus(row["status"])
        if status in TERMINAL_STATUSES:
            raise StoreError(ErrorCode.INVALID_TRANSITION, f"A {status.value} job cannot be cancelled", details={"status": status.value})
        now = db.ts(conn.now())
        changes = ChangeLog(conn)
        cancelled: List[str] = []

        def cancel_now(target) -> None:
            conn.execute("UPDATE q_jobs SET status = 'CANCELLED', error_code = ?, error_detail = ?, next_run_at = NULL, wait_cause = NULL, "
                         "completed_at = ?, updated_at = ? WHERE id = ?",
                         (JobErrorCode.CANCELLED.value, "Cancelled: " + body.reason[:180], now, now, target["id"]))
            changes.job(target["id"], JobStatus.CANCELLED, target["conversation_id"])
            cancelled.append(target["id"])

        if status is JobStatus.RUNNING:
            conn.execute("UPDATE q_jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?", (now, job_id))
            changes.job(job_id, JobStatus.RUNNING, row["conversation_id"])
        else:
            cancel_now(row)
        if body.cascade:
            for dependent_id in downstream_closure(conn, job_id):
                dependent = require_job_row(conn, dependent_id)
                if dependent["status"] in (JobStatus.BLOCKED.value, JobStatus.QUEUED.value):
                    cancel_now(dependent)
        elif status is not JobStatus.RUNNING:
            release_dependents(conn, job_id, changes)
        for cancelled_id in cancelled:
            _notify_failed(conn, cancelled_id, None, JobErrorCode.CANCELLED)
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.JOB_CANCELLED, target_kind="job", target_id=job_id,
                     details={"cascade": body.cascade, "cancelled_count": len(cancelled), "cancel_requested": status is JobStatus.RUNNING,
                              "reason": body.reason[:200]})
        changes.flush()
        job = JobReader(conn).job(require_job_row(conn, job_id))
    return CancelResponse(job=job, cancelled_job_ids=cancelled)
