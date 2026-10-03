"""Usage rows (one per attempt), late measurements, the usage report, the CSV export, per-route
medians and hardware profiles.

Identity fields of a usage row come from the job and attempt, never from Process input: the job
type, purpose, frozen catalog entry and route. A completion or failure writes the row; a lease
expiry synthesizes an ``abandoned`` one (tokens unavailable, durations from the lease, hardware
from the claim), which the late worker may fill in once.
"""

from __future__ import annotations

import csv
import io
import statistics
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

from call1.contracts.common import Page, canonical_digest
from call1.contracts.errors import ErrorCode, JobErrorCode
from call1.contracts.jobs import JOB_TYPE_RULES, AttemptProvenance, JobType
from call1.contracts.usage import (
    USAGE_CSV_COLUMNS,
    BillingUnit,
    HardwareProfile,
    HardwareProfileInput,
    LateUsageReport,
    PerScoredCallRollup,
    TokenCount,
    TokenSource,
    UsageGroupBy,
    UsageMedianRow,
    UsageMedians,
    UsageMediansQuery,
    UsageOutcome,
    UsageRecord,
    UsageRecordedBy,
    UsageRecordInput,
    UsageRecordQuery,
    UsageReport,
    UsageReportQuery,
    UsageRollupRow,
)

from .. import db, pagination
from ..errors import StoreError, not_found
from ..ids import new_id
from ..principals import hash_secret, secrets_equal
from .records import job_selection, require_conversation_row, require_job_row, usage_from_row

_MEASUREMENTS = [name for name in UsageRecordInput.model_fields]


def _identity(job_row) -> Dict[str, Any]:
    selection = job_selection(job_row)
    rule = JOB_TYPE_RULES[JobType(job_row["job_type"])]
    route = selection.route if selection is not None else None
    return {
        "job_id": job_row["id"],
        "conversation_id": job_row["conversation_id"],
        "job_type": job_row["job_type"],
        "purpose": rule.purpose.value if (selection is not None and rule.purpose is not None) else None,
        "catalog_entry": selection.catalog_entry.model_dump(mode="json") if selection is not None else None,
        "model_revision": selection.model_revision if selection is not None else None,
        "route_class": route.route_class.value if route is not None else None,
        "provider_connection_ref": route.provider_connection_ref if route is not None else None,
        "destination_host": route.destination_host if route is not None else None,
    }


def _insert(conn, record: UsageRecord, *, late_digest: Optional[str] = None) -> UsageRecord:
    conn.execute(
        "INSERT INTO q_usage_records (id, job_id, attempt_number, conversation_id, recorded_at, route_class, purpose, outcome, recorded_by, "
        "late_input_digest, record_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (record.id, record.job_id, record.attempt_number, record.conversation_id, db.ts(record.recorded_at),
         record.route_class.value if record.route_class else None, record.purpose.value if record.purpose else None,
         record.outcome.value, record.recorded_by.value, late_digest, db.dumps(record)),
    )
    return record


def record_attempt_usage(conn, job_row, attempt_row, usage: UsageRecordInput, provenance: Optional[AttemptProvenance]) -> UsageRecord:
    """The usage row a completion or failure writes for the attempt it ends."""
    data = usage.model_dump(mode="json")
    if data.get("queue_wait_seconds") is None and attempt_row["queue_wait_seconds"] is not None:
        data["queue_wait_seconds"] = attempt_row["queue_wait_seconds"]
    identity = _identity(job_row)
    if provenance is not None and provenance.model_revision:
        identity["model_revision"] = provenance.model_revision
    record = UsageRecord.model_validate({
        **data, **identity, "id": new_id("use"), "attempt_number": attempt_row["attempt_number"],
        "recorded_at": db.ts(conn.now()), "recorded_by": UsageRecordedBy.PROCESS.value,
    })
    return _insert(conn, record)


def synthesize_abandoned(conn, job_row, attempt_row) -> UsageRecord:
    """Lease expiry: tokens unavailable, durations from the lease, hardware from the claim."""
    granted = db.parse_ts(attempt_row["lease_granted_at"])
    expired = db.parse_ts(job_row["lease_expires_at"]) or db.parse_ts(attempt_row["lease_expires_at"])
    seconds = max(0.0, (expired - granted).total_seconds()) if granted and expired else 0.0
    unavailable = {"count": None, "source": TokenSource.UNAVAILABLE.value}
    record = UsageRecord.model_validate({
        **_identity(job_row), "id": new_id("use"), "attempt_number": attempt_row["attempt_number"],
        "tokens_input": unavailable, "tokens_output": unavailable, "queue_wait_seconds": attempt_row["queue_wait_seconds"],
        "inference_seconds": seconds, "total_seconds": seconds, "hardware_profile_id": attempt_row["hardware_profile_id"],
        "outcome": UsageOutcome.ABANDONED.value, "error_code": JobErrorCode.LEASE_EXPIRED.value,
        "recorded_by": UsageRecordedBy.STORE_SYNTHESIZED.value, "recorded_at": db.ts(conn.now()),
    })
    return _insert(conn, record)


def attach_late_usage(conn, job_id: str, attempt_number: int, body: LateUsageReport) -> UsageRecord:
    """``attachLateUsage``: once per lease-expired attempt, with that attempt's own claim token."""
    digest = canonical_digest(body.usage.model_dump(mode="json"))
    with db.transaction(conn):
        job = require_job_row(conn, job_id)
        attempt = conn.execute("SELECT * FROM q_attempts WHERE job_id = ? AND attempt_number = ?", (job_id, attempt_number)).fetchone()
        if attempt is None:
            raise not_found("Attempt", job_id=job_id, attempt_number=attempt_number)
        if not secrets_equal(attempt["claim_token_hash"], hash_secret(body.claim_token or "")):
            raise StoreError(ErrorCode.CLAIM_TOKEN_STALE, "The token is not the one this attempt was claimed with",
                             details={"current_attempt_number": job["claim_count"] or None, "status": job["status"], "your_attempt_outcome": None})
        if attempt["status"] != "lease_expired":
            raise StoreError(ErrorCode.CONFLICT, "Late usage attaches only to an attempt that ended by lease expiry", details={"reason": "not_lease_expired", "attempt_status": attempt["status"]})
        row = conn.execute("SELECT * FROM q_usage_records WHERE job_id = ? AND attempt_number = ?", (job_id, attempt_number)).fetchone()
        if row["recorded_by"] == UsageRecordedBy.PROCESS_LATE.value:
            if row["late_input_digest"] == digest:
                return usage_from_row(row)
            raise StoreError(ErrorCode.CONFLICT, "Late usage was already attached to this attempt", details={"reason": "already_attached", "usage_record_id": row["id"]})
        current = usage_from_row(row)
        measured = body.usage.model_dump(mode="json")
        updated = UsageRecord.model_validate({
            **current.model_dump(mode="json"),
            **{k: measured[k] for k in _MEASUREMENTS if k not in ("outcome", "error_code", "hardware_profile_id")},
            "hardware_profile_id": measured["hardware_profile_id"],
            "outcome": UsageOutcome.ABANDONED.value, "error_code": current.error_code.value if current.error_code else None,
            "recorded_by": UsageRecordedBy.PROCESS_LATE.value,
        })
        conn.execute("UPDATE q_usage_records SET recorded_by = ?, late_input_digest = ?, record_json = ? WHERE id = ?",
                     (updated.recorded_by.value, digest, db.dumps(updated), row["id"]))
        return updated


# --- reads -----------------------------------------------------------------------------------


def conversation_usage(conn, conversation_id: str) -> Page[UsageRecord]:
    require_conversation_row(conn, conversation_id)
    rows = conn.execute("SELECT * FROM q_usage_records WHERE conversation_id = ? ORDER BY recorded_at, id", (conversation_id,)).fetchall()
    return Page[UsageRecord](items=[usage_from_row(r) for r in rows], next_page_token=None)


def _range_rows(conn, *, start: Optional[datetime], end: Optional[datetime], route_class=None, purpose=None,
                conversation_id: Optional[str] = None, outcome=None, after: Optional[List[Any]] = None, limit: Optional[int] = None):
    where, args = [], []
    if start is not None:
        where.append("recorded_at >= ?")
        args.append(db.ts(start))
    if end is not None:
        where.append("recorded_at < ?")
        args.append(db.ts(end))
    if route_class is not None:
        where.append("route_class = ?")
        args.append(route_class.value)
    if purpose is not None:
        where.append("purpose = ?")
        args.append(purpose.value)
    if conversation_id is not None:
        where.append("conversation_id = ?")
        args.append(conversation_id)
    if outcome is not None:
        where.append("outcome = ?")
        args.append(outcome.value)
    if after is not None:
        where.append("(recorded_at > ? OR (recorded_at = ? AND id > ?))")
        args.extend([after[0], after[0], after[1]])
    sql = "SELECT * FROM q_usage_records" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY recorded_at, id"
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    return conn.execute(sql, args).fetchall()


def list_records(conn, query: UsageRecordQuery) -> Page[UsageRecord]:
    after = pagination.decode(query.page_token, 2)
    rows = _range_rows(conn, start=query.start, end=query.end, route_class=query.route_class, purpose=query.purpose,
                       conversation_id=query.conversation_id, outcome=query.outcome, after=after, limit=query.limit + 1)
    items = [usage_from_row(r) for r in rows[: query.limit]]
    token = pagination.encode(rows[query.limit - 1]["recorded_at"], rows[query.limit - 1]["id"]) if len(rows) > query.limit else None
    return Page[UsageRecord](items=items, next_page_token=token)


def report_records(conn, query: UsageReportQuery) -> List[UsageRecord]:
    rows = _range_rows(conn, start=query.start, end=query.end, route_class=query.route_class, purpose=query.purpose)
    return [usage_from_row(r) for r in rows]


def _group_value(record: UsageRecord, key: UsageGroupBy) -> str:
    if key is UsageGroupBy.ROUTE_CLASS:
        return record.route_class.value if record.route_class else "none"
    if key is UsageGroupBy.MODEL:
        return f"{record.catalog_entry.entry_id}@{record.catalog_entry.entry_version}" if record.catalog_entry else "none"
    if key is UsageGroupBy.PURPOSE:
        return record.purpose.value if record.purpose else "none"
    if key is UsageGroupBy.HARDWARE_PROFILE:
        return record.hardware_profile_id
    return record.outcome.value


def _tokens(count: TokenCount) -> int:
    return count.count or 0


def rollup(records: Iterable[UsageRecord], group_by: List[UsageGroupBy]) -> List[UsageRollupRow]:
    groups: "OrderedDict[Tuple[str, ...], List[UsageRecord]]" = OrderedDict()
    for record in sorted(records, key=lambda r: tuple(_group_value(r, k) for k in group_by)):
        groups.setdefault(tuple(_group_value(record, k) for k in group_by), []).append(record)
    rows = []
    for key, items in groups.items():
        units: Dict[str, float] = defaultdict(float)
        for record in items:
            for unit in record.billing_units:
                units[unit.unit] += unit.quantity
        outcomes = [r.outcome for r in items]
        rows.append(UsageRollupRow(
            keys={k.value: v for k, v in zip(group_by, key)},
            attempts=len(items),
            succeeded=outcomes.count(UsageOutcome.SUCCEEDED),
            failed=outcomes.count(UsageOutcome.FAILED),
            validation_rejected=outcomes.count(UsageOutcome.VALIDATION_REJECTED),
            cancelled=outcomes.count(UsageOutcome.CANCELLED),
            tokens_input_total=sum(_tokens(r.tokens_input) for r in items),
            tokens_output_total=sum(_tokens(r.tokens_output) for r in items),
            attempts_with_unavailable_tokens=sum(1 for r in items if TokenSource.UNAVAILABLE in (r.tokens_input.source, r.tokens_output.source)),
            inference_seconds_total=sum(r.inference_seconds for r in items),
            audio_seconds_total=sum(r.audio_seconds_processed or 0.0 for r in items),
            billing_units=[BillingUnit(unit=u, quantity=q) for u, q in sorted(units.items())],
        ))
    return rows


def per_scored_call(records: List[UsageRecord]) -> PerScoredCallRollup:
    scored = {r.conversation_id for r in records if r.job_type == JobType.QA_SCORECARD.value and r.outcome is UsageOutcome.SUCCEEDED}
    escalations = sum(1 for r in records if r.job_type == JobType.QA_ESCALATION.value)
    tokens: Dict[str, int] = defaultdict(int)
    seconds: Dict[str, float] = defaultdict(float)
    for r in records:
        tokens[r.route_class.value if r.route_class else "none"] += _tokens(r.tokens_input) + _tokens(r.tokens_output)
        seconds[r.hardware_profile_id] += r.inference_seconds
    n = len(scored)
    rejected = sum(1 for r in records if r.outcome is UsageOutcome.VALIDATION_REJECTED)
    return PerScoredCallRollup(
        scored_calls=n,
        attempts_per_scored_call=(len(records) / n) if n else 0.0,
        escalations_per_scored_call=(escalations / n) if n else 0.0,
        validation_reject_rate=(rejected / len(records)) if records else 0.0,
        tokens_by_route_class=dict(tokens),
        inference_seconds_by_hardware_profile=dict(seconds),
    )


def build_report(conn, query: UsageReportQuery) -> UsageReport:
    records = report_records(conn, query)
    rows = rollup(records, list(query.group_by))
    # The price table is deferred in Stage 2, so no row can be priced: estimates stay null.
    estimates = [None for _ in rows] if query.include_estimates else None
    return UsageReport(query=query, rows=rows, per_scored_call=per_scored_call(records), estimates_by_row=estimates, generated_at=conn.now())


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def csv_export(records: List[UsageRecord]) -> str:
    """RFC 4180, header line, empty cells for nulls, billing units as ``unit=quantity;...``."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(USAGE_CSV_COLUMNS)
    for r in records:
        values = {
            "id": r.id, "recorded_at": db.ts(r.recorded_at), "conversation_id": r.conversation_id, "job_id": r.job_id,
            "attempt_number": r.attempt_number, "job_type": r.job_type, "purpose": r.purpose.value if r.purpose else None,
            "catalog_entry_id": r.catalog_entry.entry_id if r.catalog_entry else None,
            "catalog_entry_version": r.catalog_entry.entry_version if r.catalog_entry else None,
            "model_revision": r.model_revision, "provider_reported_model_id": r.provider_reported_model_id,
            "route_class": r.route_class.value if r.route_class else None, "provider_connection_ref": r.provider_connection_ref,
            "destination_host": r.destination_host, "hardware_profile_id": r.hardware_profile_id, "outcome": r.outcome.value,
            "error_code": r.error_code.value if r.error_code else None, "recorded_by": r.recorded_by.value,
            "tokens_input": r.tokens_input.count, "tokens_input_source": r.tokens_input.source.value,
            "tokens_output": r.tokens_output.count, "tokens_output_source": r.tokens_output.source.value,
            "queue_wait_seconds": r.queue_wait_seconds, "slot_wait_seconds": r.slot_wait_seconds, "model_load_seconds": r.model_load_seconds,
            "inference_seconds": r.inference_seconds, "total_seconds": r.total_seconds, "session_setup_seconds": r.session_setup_seconds,
            "session_id": r.session_id, "audio_seconds_processed": r.audio_seconds_processed, "peak_memory_bytes": r.peak_memory_bytes,
            "billing_units": ";".join(f"{u.unit}={u.quantity}" for u in r.billing_units) or None,
        }
        writer.writerow([_cell(values[c]) for c in USAGE_CSV_COLUMNS])
    return buffer.getvalue()


def medians(conn, query: UsageMediansQuery) -> UsageMedians:
    now = conn.now()
    since = query.since or (now - timedelta(days=30))
    rows = _range_rows(conn, start=since, end=None, purpose=query.purpose)
    groups: "OrderedDict[Tuple, List[UsageRecord]]" = OrderedDict()
    for record in (usage_from_row(r) for r in rows):
        if record.catalog_entry is None or record.purpose is None or record.route_class is None:
            continue
        key = (record.catalog_entry.entry_id, record.catalog_entry.entry_version, record.purpose.value, record.route_class.value,
               record.destination_host, record.hardware_profile_id)
        groups.setdefault(key, []).append(record)

    def median(values: List[float]) -> Optional[float]:
        return float(statistics.median(values)) if values else None

    out = []
    for key in sorted(groups, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        items = groups[key]
        succeeded = [r for r in items if r.outcome is UsageOutcome.SUCCEEDED]
        out.append(UsageMedianRow(
            catalog_entry={"entry_id": key[0], "entry_version": key[1]}, purpose=key[2], route_class=key[3], destination_host=key[4],
            hardware_profile_id=key[5], attempts=len(items), succeeded=len(succeeded),
            median_inference_seconds=median([r.inference_seconds for r in succeeded]),
            median_total_seconds=median([r.total_seconds for r in succeeded]),
            median_tokens_input=median([float(r.tokens_input.count) for r in succeeded if r.tokens_input.count is not None]),
            median_tokens_output=median([float(r.tokens_output.count) for r in succeeded if r.tokens_output.count is not None]),
            validation_reject_rate=sum(1 for r in items if r.outcome is UsageOutcome.VALIDATION_REJECTED) / len(items),
        ))
    return UsageMedians(since=since, rows=out, generated_at=now)


# --- hardware profiles -------------------------------------------------------------------------


def _profile(row) -> HardwareProfile:
    return HardwareProfile.model_validate({**db.loads(row["fields_json"]), "fingerprint": row["fingerprint"], "id": row["id"],
                                           "first_seen_at": row["first_seen_at"], "last_seen_at": row["last_seen_at"]})


def upsert_hardware_profile(conn, body: HardwareProfileInput) -> HardwareProfile:
    now = db.ts(conn.now())
    fields = body.model_dump(mode="json")
    fields.pop("fingerprint")
    with db.transaction(conn):
        row = conn.execute("SELECT * FROM q_hardware_profiles WHERE fingerprint = ?", (body.fingerprint,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO q_hardware_profiles (id, fingerprint, fields_json, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?)",
                         (new_id("hw"), body.fingerprint, db.dumps(fields), now, now))
        else:
            conn.execute("UPDATE q_hardware_profiles SET last_seen_at = ? WHERE id = ?", (now, row["id"]))
        return _profile(conn.execute("SELECT * FROM q_hardware_profiles WHERE fingerprint = ?", (body.fingerprint,)).fetchone())


def list_hardware_profiles(conn) -> Page[HardwareProfile]:
    rows = conn.execute("SELECT * FROM q_hardware_profiles ORDER BY first_seen_at, id").fetchall()
    return Page[HardwareProfile](items=[_profile(r) for r in rows], next_page_token=None)
