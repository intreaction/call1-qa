"""Contact Signals v2 in the queue area: previews, compares, backfills and shadow-mode companions
(contract 1.3.0; docs/ContactSignalsV2.md sections 7.5 and 14).

* **Preview** (``createSignalPreview``, admin): Store checks the (usually unsaved) taxonomy with the
  save-time caps and text detectors (through ``results/api.py``), mints a preview snapshot per call
  in ``draft:<request_id>:signals:preview`` and creates one ``contact_signals_preview`` request per
  call (priority +5). The graphs are draft-test graphs: outputs in draft slots, nothing projected.
* **Compare**: a compare backfill, or a shadow-mode companion that ``lifecycle.complete`` creates
  in the transaction that publishes a v1 ``contact_signals`` result from a non-draft graph. Each is
  a ``contact_signals_preview`` request (priority -10) pinned to the published version's snapshot.
* **Rescore backfill**: one ``contact_signals`` request per call whose result a digest-driven update
  would change (or every call with ``rescore_signals``); a pending, unclaimed request is widened
  instead. Priority -10, so backfills never delay new calls.
* ``getSignalPreview`` reads each call's state, masked result and diff against the call's
  published signals through the results area's API.

The taxonomy, settings and results are read only through ``results/api.py``.
"""

from __future__ import annotations

import json
from typing import List, Optional

from call1.contracts.artifacts import Artifact
from call1.contracts.common import canonical_digest
from call1.contracts.contents import ContactSignalsContent
from call1.contracts.errors import ErrorCode
from call1.contracts.events import AuditAction
from call1.contracts.jobs import (
    REANALYSIS_PRIORITY_SIGNAL_BACKFILL,
    REANALYSIS_PRIORITY_SIGNAL_PREVIEW,
    JobStatus,
    JobType,
    ReanalysisKind,
    ReanalysisStatus,
)
from call1.contracts.signals import (
    SignalBackfill,
    SignalBackfillCreate,
    SignalPreview,
    SignalPreviewCall,
    SignalPreviewCreate,
    SignalTaxonomyRef,
    SignalTaxonomySnapshotContent,
    taxonomy_digest,
)

from .. import audit, db
from ..errors import StoreError, not_found
from ..ids import new_id
from ..principals import SessionPrincipal
from . import reanalysis
from .artifacts import ensure_published_signal_snapshot, mint_preview_signal_snapshot
from .changes import ChangeLog
from .records import ConversationJobs, artifact_row, effective_request_status, require_call_row, request_from_row


def _results():
    from ..results import api as results_api

    return results_api


def _insert_preview(conn, *, source: str, ref: SignalTaxonomyRef, account_id: Optional[str], scope: Optional[str] = None,
                    key: Optional[str] = None, digest: Optional[str] = None) -> str:
    preview_id = new_id("prv")
    conn.execute("INSERT INTO q_signal_previews (id, source, taxonomy_version, taxonomy_digest, created_at, created_by_account_id, "
                 "idempotency_scope, idempotency_key, request_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 (preview_id, source, ref.version, ref.digest, db.ts(conn.now()), account_id, scope, key, digest))
    return preview_id


def _preview_request(conn, *, conversation_row, preview_id: str, snapshot: Artifact, request_id: str, priority: int,
                     account_id: Optional[str], backfill_id: Optional[str], changes: ChangeLog, key_prefix: str):
    key = f"{key_prefix}.{conversation_row['call_id']}"[:128]
    return reanalysis.insert_request(
        conn, conversation_row=conversation_row, kind=ReanalysisKind.CONTACT_SIGNALS_PREVIEW, scope=f"signals:preview:{preview_id}", key=key,
        digest=canonical_digest({"preview_id": preview_id, "call_id": conversation_row["call_id"]}), requested_by_account_id=account_id,
        request_id=request_id, changes=changes, priority=priority, signal_backfill_id=backfill_id, signal_preview_id=preview_id,
        signal_taxonomy_snapshot_artifact_id=snapshot.id)


# --- previews ------------------------------------------------------------------------------------


def create_preview(conn, store, principal: SessionPrincipal, body: SignalPreviewCreate, key: str) -> SignalPreview:
    """``createSignalPreview``: header-idempotent per session."""
    results = _results()
    reanalysis.check_idempotency_key(key)
    params = store.config.parameters
    scope = f"session:{principal.session_id}:signals:previews"
    digest = canonical_digest({"operation": "createSignalPreview", "body": body.model_dump(mode="json")})
    with db.transaction(conn):
        existing = conn.execute("SELECT * FROM q_signal_previews WHERE idempotency_scope = ? AND idempotency_key = ?", (scope, key)).fetchone()
        if existing is not None:
            if existing["request_digest"] != digest:
                raise StoreError(ErrorCode.IDEMPOTENCY_KEY_REUSED, "This Idempotency-Key was used with a different request",
                                 details={"original_id": existing["id"]})
            return get_preview(conn, store, existing["id"])
        if len(body.call_ids) > params.signal_preview_max_calls:
            raise StoreError(ErrorCode.VALIDATION_FAILED, f"A preview covers at most {params.signal_preview_max_calls} calls",
                             details={"field": "call_ids", "reason": "cap", "cap": "signal_preview_max_calls", "limit": params.signal_preview_max_calls})
        current = results.current_signal_taxonomy(conn)
        taxonomy = body.taxonomy if body.taxonomy is not None else current.taxonomy
        results.check_signal_taxonomy(taxonomy, params)
        tax_digest = taxonomy_digest(taxonomy)
        ref = SignalTaxonomyRef(version=current.version if tax_digest == current.digest else None, digest=tax_digest)
        conversations = [require_call_row(conn, call_id) for call_id in body.call_ids]
        preview_id = _insert_preview(conn, source="preview", ref=ref, account_id=principal.account_id, scope=scope, key=key, digest=digest)
        settings = results.signal_settings(conn)
        changes = ChangeLog(conn)
        for conversation in conversations:
            request_id = new_id("rq")
            content = SignalTaxonomySnapshotContent(source="preview", taxonomy_ref=SignalTaxonomyRef(version=None, digest=tax_digest),
                                                    taxonomy=taxonomy, settings=settings, preview_id=preview_id)
            snapshot = mint_preview_signal_snapshot(conn, store, conversation["id"], request_id, content)
            _preview_request(conn, conversation_row=conversation, preview_id=preview_id, snapshot=snapshot, request_id=request_id,
                             priority=REANALYSIS_PRIORITY_SIGNAL_PREVIEW, account_id=principal.account_id, backfill_id=None, changes=changes,
                             key_prefix=f"preview.{preview_id}")
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.SIGNAL_PREVIEW_REQUESTED, target_kind="signal_preview",
                     target_id=preview_id, details={"preview_id": preview_id, "taxonomy_digest": tax_digest, "call_count": len(conversations)})
        changes.flush()
        return get_preview(conn, store, preview_id)


def _call_state(conn, store, row):
    """(state, failure_code, result, diff) of one preview request."""
    results = _results()
    now = conn.now()
    status = effective_request_status(row, now)
    if row["preview_result_artifact_id"] is not None:
        art = artifact_row(conn, row["preview_result_artifact_id"])
        content = ContactSignalsContent.model_validate(json.loads(store.objects.read_bytes(art["checksum"])))
        masked = results.masked_signal_result(conn, row["conversation_id"], content)
        return "available", None, masked, results.signal_preview_diff(conn, row["conversation_id"], content)
    if status is ReanalysisStatus.REJECTED:
        return "failed", None, None, None
    if row["graph_id"] is not None:
        merges = conn.execute("SELECT id FROM q_jobs WHERE graph_id = ? AND job_type = ? ORDER BY created_at DESC, id DESC",
                              (row["graph_id"], JobType.CONTACT_SIGNALS_MERGE.value)).fetchall()
        if merges:
            view = ConversationJobs(conn, row["conversation_id"])
            node = view.nodes[merges[0]["id"]]
            if node.status in (JobStatus.FAILED, JobStatus.CANCELLED) or view.dead(node.id):
                return "failed", view.failure_code(node.id), None, None
    return "pending", None, None, None


def get_preview(conn, store, preview_id: str) -> SignalPreview:
    preview = conn.execute("SELECT * FROM q_signal_previews WHERE id = ?", (preview_id,)).fetchone()
    if preview is None:
        raise not_found("Signal preview", preview_id=preview_id)
    calls = []
    for row in conn.execute("SELECT * FROM q_reanalysis_requests WHERE signal_preview_id = ? ORDER BY requested_at, id", (preview_id,)).fetchall():
        state, failure, result, diff = _call_state(conn, store, row)
        calls.append(SignalPreviewCall(call_id=row["call_id"], request_id=row["id"], state=state, failure_code=failure, result=result, diff=diff))
    return SignalPreview(id=preview["id"], source=preview["source"],
                         taxonomy_ref=SignalTaxonomyRef(version=preview["taxonomy_version"], digest=preview["taxonomy_digest"]), calls=calls,
                         options_trimmed=db.loads(preview["options_trimmed_json"]) or [], created_at=db.parse_ts(preview["created_at"]),
                         created_by_account_id=preview["created_by_account_id"])


# --- compares ------------------------------------------------------------------------------------


def _compare_request(conn, store, conversation_row, preview_id: str, version: int, *, account_id: Optional[str],
                     backfill_id: Optional[str], changes: ChangeLog):
    snapshot = ensure_published_signal_snapshot(conn, store, conversation_row["id"], version)
    return _preview_request(conn, conversation_row=conversation_row, preview_id=preview_id, snapshot=snapshot, request_id=new_id("rq"),
                            priority=REANALYSIS_PRIORITY_SIGNAL_BACKFILL, account_id=account_id, backfill_id=backfill_id, changes=changes,
                            key_prefix=f"compare.{preview_id}")


def shadow_companion(conn, store, conversation_id: str, published: Artifact, changes: ChangeLog) -> Optional[str]:
    """Shadow mode (section 14): a v1 ``contact_signals`` publish from a non-draft graph gets one
    ``contact_signals_preview`` compare request (priority -10), in the publishing transaction. One
    per published artifact (idempotent by the artifact ID). Returns the preview ID, or None."""
    results = _results()
    if results.signal_settings(conn).pipeline != "shadow":
        return None
    content = ContactSignalsContent.model_validate(json.loads(store.objects.read_bytes(published.checksum)))
    if content.pipeline != "v1":
        return None
    conversation = conn.execute("SELECT * FROM q_conversations WHERE id = ?", (conversation_id,)).fetchone()
    if conversation is None or conversation["call_id"] is None:
        return None
    key = f"shadow.{published.id}"
    found = conn.execute("SELECT id FROM q_signal_previews WHERE idempotency_scope = 'store:shadow' AND idempotency_key = ?", (key,)).fetchone()
    if found is not None:
        return found["id"]
    current = results.current_signal_taxonomy(conn)
    preview_id = _insert_preview(conn, source="compare", ref=current.ref, account_id=None, scope="store:shadow", key=key, digest=key)
    _compare_request(conn, store, conversation, preview_id, current.version, account_id=None, backfill_id=None, changes=changes)
    return preview_id


def latest_compare_preview(conn, call_id: str) -> Optional[str]:
    """The newest compare preview with an available result for the call (``comparison_preview_id``)."""
    row = conn.execute(
        "SELECT r.signal_preview_id FROM q_reanalysis_requests r JOIN q_signal_previews p ON p.id = r.signal_preview_id "
        "WHERE r.call_id = ? AND p.source = 'compare' AND r.preview_result_artifact_id IS NOT NULL ORDER BY r.requested_at DESC, r.id DESC LIMIT 1",
        (call_id,)).fetchone()
    return None if row is None else row["signal_preview_id"]


# --- backfills -----------------------------------------------------------------------------------


def _backfill(row) -> SignalBackfill:
    return SignalBackfill(id=row["id"], mode=row["mode"], taxonomy_version=row["taxonomy_version"], calls_matched=row["calls_matched"],
                          requests_created=row["requests_created"], calls_skipped=row["calls_skipped"], preview_id=row["preview_id"],
                          created_at=db.parse_ts(row["created_at"]), created_by_account_id=row["created_by_account_id"])


def create_backfill(conn, store, principal: SessionPrincipal, body: SignalBackfillCreate, key: str) -> SignalBackfill:
    """``createSignalBackfill``: header-idempotent per session."""
    results = _results()
    reanalysis.check_idempotency_key(key)
    limit = store.config.parameters.signal_backfill_max_calls
    if body.max_calls > limit:
        raise StoreError(ErrorCode.VALIDATION_FAILED, f"A backfill requests at most {limit} calls",
                         details={"field": "max_calls", "reason": "cap", "cap": "signal_backfill_max_calls", "limit": limit})
    scope = f"session:{principal.session_id}:signals:backfills"
    digest = canonical_digest({"operation": "createSignalBackfill", "body": body.model_dump(mode="json")})
    with db.transaction(conn):
        existing = conn.execute("SELECT * FROM q_signal_backfills WHERE idempotency_scope = ? AND idempotency_key = ?", (scope, key)).fetchone()
        if existing is not None:
            if existing["request_digest"] != digest:
                raise StoreError(ErrorCode.IDEMPOTENCY_KEY_REUSED, "This Idempotency-Key was used with a different request",
                                 details={"original_id": existing["id"]})
            return _backfill(existing)
        current = results.current_signal_taxonomy(conn)
        candidates = results.signal_backfill_candidates(conn, body.created_after, body.created_before, body.max_calls, body.mode,
                                                        rescore_signals=body.rescore_signals)
        backfill_id = new_id("bkf")
        preview_id = None
        if body.mode == "compare":
            preview_id = _insert_preview(conn, source="compare", ref=current.ref, account_id=principal.account_id)
        changes = ChangeLog(conn)
        created = skipped = 0
        for candidate in candidates:
            if not candidate.needs_update:
                skipped += 1
                continue
            conversation = require_call_row(conn, candidate.call_id)
            if body.mode == "compare":
                _compare_request(conn, store, conversation, preview_id, current.version, account_id=principal.account_id,
                                 backfill_id=backfill_id, changes=changes)
                created += 1
                continue
            if reanalysis.widen_pending_signals(conn, candidate.call_id, priority=REANALYSIS_PRIORITY_SIGNAL_BACKFILL,
                                                rescore_signals=body.rescore_signals) is not None:
                skipped += 1
                continue
            active = conn.execute("SELECT 1 FROM q_reanalysis_requests WHERE call_id = ? AND kind = ? AND status IN ('pending', 'claimed')",
                                  (candidate.call_id, ReanalysisKind.CONTACT_SIGNALS.value)).fetchone()
            if active is not None:  # claimed and underway: it will pick up the current taxonomy on its own graph
                skipped += 1
                continue
            reanalysis.insert_request(
                conn, conversation_row=conversation, kind=ReanalysisKind.CONTACT_SIGNALS, scope=f"signals:backfill:{backfill_id}",
                key=f"backfill.{backfill_id}.{candidate.call_id}"[:128],
                digest=canonical_digest({"backfill_id": backfill_id, "call_id": candidate.call_id}), requested_by_account_id=principal.account_id,
                changes=changes, priority=REANALYSIS_PRIORITY_SIGNAL_BACKFILL, rescore_signals=body.rescore_signals, signal_backfill_id=backfill_id)
            created += 1
        conn.execute(
            "INSERT INTO q_signal_backfills (id, mode, rescore_signals, created_after, created_before, max_calls, taxonomy_version, calls_matched, "
            "requests_created, calls_skipped, preview_id, created_at, created_by_account_id, idempotency_scope, idempotency_key, request_digest) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (backfill_id, body.mode, 1 if body.rescore_signals else 0, db.ts(body.created_after),
             db.ts(body.created_before) if body.created_before else None, body.max_calls, current.version, len(candidates), created, skipped,
             preview_id, db.ts(conn.now()), principal.account_id, scope, key, digest))
        window = db.ts(body.created_after) + ".." + (db.ts(body.created_before) if body.created_before else "")
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.SIGNAL_BACKFILL_REQUESTED, target_kind="signal_backfill",
                     target_id=backfill_id, details={"backfill_id": backfill_id, "mode": body.mode, "window": window, "requests_created": created,
                                                     "calls_matched": len(candidates), "rescore_signals": body.rescore_signals})
        changes.flush()
        return _backfill(conn.execute("SELECT * FROM q_signal_backfills WHERE id = ?", (backfill_id,)).fetchone())


def preview_requests(conn, preview_id: str) -> List:
    return [request_from_row(r, conn.now()) for r in conn.execute("SELECT * FROM q_reanalysis_requests WHERE signal_preview_id = ?", (preview_id,))]
