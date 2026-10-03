"""Artifacts of conversations: inline JSON, verified uploads, Store-minted rubric snapshots, linking
and per-slot versions, listings, content grants and orphan cleanup.

Rules (contract README, "Artifacts, slots and content"):

* JSON content is exactly ``artifacts.canonical_content`` of the payload; Store stores those bytes
  and never normalizes, so the producer's ``canonical_digest(payload)`` is the checksum.
* An artifact with no producing job (sources, Store-minted snapshots, migrated artifacts) is linked
  at commit: it gets the next version of its ``(conversation, kind, slot)`` and supersedes the
  previous linked one. A job output stays unlinked until the completion that references it.
* Natural idempotency: a job output is keyed by (producing job, the attempt its claim token
  belongs to, kind, slot, checksum); any other artifact by (conversation, kind, slot, checksum).
* A draft-test graph's outputs live in ``draft:<request_id>:`` slots, and nothing else does.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from call1.contracts.artifacts import (
    ARTIFACT_CONTENT_CONTRACTS,
    DRAFT_TEST_SLOT_PREFIX,
    Artifact,
    ArtifactDescriptor,
    ArtifactKind,
    ArtifactListQuery,
    ArtifactStorage,
    InlineArtifactCreate,
    Sensitivity,
    UploadCommit,
    UploadGrantRequest,
    canonical_content,
    content_model_for,
    draft_rubric_snapshot_slot,
    is_draft_test_slot,
    preview_signal_taxonomy_snapshot_slot,
    rubric_snapshot_slot,
    signal_taxonomy_snapshot_slot,
)
from call1.contracts.common import Page, ServiceScope
from call1.contracts.errors import ErrorCode
from call1.contracts.jobs import JOB_TYPE_RULES, JobStatus, JobType
from call1.contracts.rubrics import RubricSnapshotContent, RubricSnapshotRequest
from call1.contracts.signals import SignalTaxonomySnapshotContent, SignalTaxonomySnapshotRequest, signal_taxonomy_snapshot_current

from .. import db, pagination
from ..errors import StoreError, not_found
from ..ids import new_id
from ..objects import sha256_checksum
from ..principals import Principal, ServiceKeyPrincipal, hash_secret, secrets_equal
from .records import artifact_from_row, artifact_row, require_artifact_row, require_conversation_row

UPLOAD_PURPOSE = "conversation_artifact"
JSON_CONTENT_TYPE = "application/json"
RUBRIC_SNAPSHOT_CONTRACT = ARTIFACT_CONTENT_CONTRACTS[ArtifactKind.RUBRIC_SNAPSHOT]
SIGNAL_SNAPSHOT_CONTRACT = ARTIFACT_CONTENT_CONTRACTS[ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT]


def _validation(message: str, **details) -> StoreError:
    return StoreError(ErrorCode.VALIDATION_FAILED, message, details=details)


# --- producing-job checks -------------------------------------------------------------------


def _claim_attempt(conn, job, claim_token: str) -> Tuple[int, bool]:
    """(attempt number the token belongs to, whether it is the job's active claim)."""
    token_hash = hash_secret(claim_token or "")
    row = conn.execute("SELECT attempt_number FROM q_attempts WHERE job_id = ? AND claim_token_hash = ?", (job["id"], token_hash)).fetchone()
    if row is None:
        raise StoreError(ErrorCode.CLAIM_TOKEN_STALE, "The claim token is not a claim of this job",
                         details={"current_attempt_number": job["claim_count"] or None, "status": job["status"], "your_attempt_outcome": None})
    active = (job["status"] == JobStatus.RUNNING.value and job["lease_claim_token_hash"] is not None
              and secrets_equal(job["lease_claim_token_hash"], token_hash))
    return int(row["attempt_number"]), active


def _stale(conn, job, attempt_number: int) -> StoreError:
    row = conn.execute("SELECT status FROM q_attempts WHERE job_id = ? AND attempt_number = ?", (job["id"], attempt_number)).fetchone()
    return StoreError(ErrorCode.CLAIM_TOKEN_STALE, "The claim token is no longer the job's active claim",
                      details={"current_attempt_number": job["claim_count"] or None, "status": job["status"],
                               "your_attempt_outcome": row["status"] if row is not None else None})


def check_slot_rules(slot: str, draft_test_request_id: Optional[str]) -> None:
    if draft_test_request_id is not None:
        prefix = f"{DRAFT_TEST_SLOT_PREFIX}{draft_test_request_id}:"
        if not slot.startswith(prefix):
            raise _validation("A draft-test graph writes its artifacts in its request's draft slots", reason="draft_slot_required", slot_prefix=prefix)
    elif is_draft_test_slot(slot):
        raise _validation("Draft-test slots belong to draft-test graphs only", reason="draft_slot_reserved")


def _job_output_context(conn, conversation_id: str, descriptor: ArtifactDescriptor, claim_token: Optional[str]):
    """For a job output: (job row, attempt number, active). For anything else: (None, None, False)."""
    if descriptor.producing_job_id is None:
        check_slot_rules(descriptor.slot, None)
        return None, None, False
    job = conn.execute("SELECT * FROM q_jobs WHERE id = ?", (descriptor.producing_job_id,)).fetchone()
    if job is None:
        raise not_found("Producing job", job_id=descriptor.producing_job_id)
    if job["conversation_id"] != conversation_id:
        raise _validation("The producing job belongs to another conversation", reason="job_not_in_conversation")
    rule = JOB_TYPE_RULES[JobType(job["job_type"])]
    declared = set(rule.outputs.values()) | set(rule.optional_outputs.values())
    if descriptor.kind not in declared:
        raise _validation(f"{job['job_type']} jobs do not produce {descriptor.kind.value} artifacts", reason="kind_not_an_output")
    check_slot_rules(descriptor.slot, job["draft_test_request_id"])
    attempt_number, active = _claim_attempt(conn, job, claim_token or "")
    return job, attempt_number, active


def find_natural(conn, conversation_id: str, descriptor: ArtifactDescriptor, attempt_number: Optional[int]):
    if descriptor.producing_job_id is not None:
        return conn.execute(
            "SELECT * FROM q_artifacts WHERE producing_job_id = ? AND producing_attempt_number = ? AND kind = ? AND slot = ? AND checksum = ?",
            (descriptor.producing_job_id, attempt_number, descriptor.kind.value, descriptor.slot, descriptor.checksum),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM q_artifacts WHERE conversation_id = ? AND producing_job_id IS NULL AND kind = ? AND slot = ? AND checksum = ? "
        "ORDER BY committed_at, id LIMIT 1",
        (conversation_id, descriptor.kind.value, descriptor.slot, descriptor.checksum),
    ).fetchone()


# --- writing and linking --------------------------------------------------------------------


def link(conn, artifact_id: str, *, receipt_id: Optional[str] = None) -> Tuple[Artifact, Optional[str]]:
    """Link one committed artifact: next version of its slot, supersede the previous linked one.
    Returns the linked artifact and the ID it superseded (if any)."""
    row = require_artifact_row(conn, artifact_id)
    if row["linked"]:
        return artifact_from_row(row), None
    current = conn.execute(
        "SELECT id, version FROM q_artifacts WHERE conversation_id IS ? AND kind = ? AND slot = ? AND linked = 1 ORDER BY version DESC LIMIT 1",
        (row["conversation_id"], row["kind"], row["slot"]),
    ).fetchone()
    version = (int(current["version"]) + 1) if current is not None else 1
    superseded = None
    if current is not None:
        conn.execute("UPDATE q_artifacts SET superseded_by = ? WHERE id = ? AND superseded_by IS NULL", (artifact_id, current["id"]))
        superseded = current["id"]
    conn.execute("UPDATE q_artifacts SET linked = 1, version = ?, linked_by_receipt_id = ? WHERE id = ?", (version, receipt_id, artifact_id))
    return artifact_from_row(require_artifact_row(conn, artifact_id)), superseded


def insert_artifact(conn, *, artifact_id: str, conversation_id: str, descriptor: ArtifactDescriptor, attempt_number: Optional[int],
                    storage: ArtifactStorage, upload_id: Optional[str] = None) -> Artifact:
    """Insert a committed artifact; link it at once when it has no producing job."""
    conn.execute(
        "INSERT INTO q_artifacts (id, conversation_id, kind, slot, content_type, size_bytes, checksum, content_contract, sensitivity, "
        "producing_job_id, producing_attempt_number, labels_json, linked, storage, committed_at, upload_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)",
        (artifact_id, conversation_id, descriptor.kind.value, descriptor.slot, descriptor.content_type, descriptor.size_bytes,
         descriptor.checksum, descriptor.content_contract, descriptor.sensitivity.value, descriptor.producing_job_id, attempt_number,
         db.dumps(descriptor.labels), storage.value, db.ts(conn.now()), upload_id),
    )
    if descriptor.producing_job_id is None:
        return link(conn, artifact_id)[0]
    return artifact_from_row(require_artifact_row(conn, artifact_id))


def _descriptor_of(body) -> ArtifactDescriptor:
    return ArtifactDescriptor.model_validate({k: getattr(body, k) for k in ArtifactDescriptor.model_fields})


def create_inline(conn, store, conversation_id: str, body: InlineArtifactCreate) -> Artifact:
    """``createInlineArtifact``: validate, deduplicate naturally, store the canonical bytes."""
    data = canonical_content(body.content_contract, body.payload)
    if len(data) > store.config.parameters.inline_artifact_max_bytes:
        raise StoreError(ErrorCode.PAYLOAD_TOO_LARGE, "Inline artifacts are limited; use an upload grant",
                         details={"max_bytes": store.config.parameters.inline_artifact_max_bytes})
    checksum = sha256_checksum(data)
    if checksum != body.checksum:
        raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "checksum is not canonical_digest(payload)", details={"reason": "checksum_differs", "computed": checksum})
    if body.size_bytes != len(data):
        raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "size_bytes is not the length of the canonical bytes", details={"reason": "size_differs", "computed_size": len(data)})
    descriptor = _descriptor_of(body)
    with db.transaction(conn):
        require_conversation_row(conn, conversation_id)
        job, attempt_number, active = _job_output_context(conn, conversation_id, descriptor, body.claim_token)
        existing = find_natural(conn, conversation_id, descriptor, attempt_number)
        if existing is not None:
            return artifact_from_row(existing)
        if job is not None and not active:
            raise _stale(conn, job, attempt_number)
        store.objects.put_bytes(data)
        return insert_artifact(conn, artifact_id=new_id("art"), conversation_id=conversation_id, descriptor=descriptor,
                               attempt_number=attempt_number, storage=ArtifactStorage.INLINE)


def create_upload_grant(conn, store, principal: ServiceKeyPrincipal, conversation_id: str, body: UploadGrantRequest) -> Dict[str, Any]:
    """``createUploadGrant``: a one-time upload session reserving the artifact's ID."""
    descriptor = _descriptor_of(body)
    with db.transaction(conn):
        require_conversation_row(conn, conversation_id)
        job, attempt_number, active = _job_output_context(conn, conversation_id, descriptor, body.claim_token)
        existing = find_natural(conn, conversation_id, descriptor, attempt_number)
        if existing is None and job is not None and not active:
            raise _stale(conn, job, attempt_number)
        artifact_id = existing["id"] if existing is not None else _pending_reservation(conn, conversation_id, descriptor, attempt_number) or new_id("art")
        ticket = store.objects.create_upload(
            conn, purpose=UPLOAD_PURPOSE, expected_checksum=descriptor.checksum, expected_size=descriptor.size_bytes,
            content_type=descriptor.content_type, required_scope=ServiceScope.ARTIFACTS_WRITE, installation_id=principal.installation_id,
            metadata={"artifact_id": artifact_id, "conversation_id": conversation_id, "attempt_number": attempt_number,
                      "descriptor": descriptor.model_dump(mode="json")},
        )
    return {"upload_id": ticket.upload_id, "artifact_id": artifact_id, "method": "PUT", "url": ticket.url, "headers": ticket.headers,
            "expires_at": ticket.expires_at, "max_bytes": ticket.max_bytes}


def _pending_reservation(conn, conversation_id: str, descriptor: ArtifactDescriptor, attempt_number: Optional[int]) -> Optional[str]:
    """The artifact ID an unexpired, uncommitted grant for the same natural key already reserved."""
    now = db.ts(conn.now())
    for row in conn.execute("SELECT metadata_json FROM object_uploads WHERE purpose = ? AND status != 'committed' AND expires_at >= ? "
                            "AND expected_checksum = ? ORDER BY created_at", (UPLOAD_PURPOSE, now, descriptor.checksum)):
        meta = db.loads(row["metadata_json"]) or {}
        other = meta.get("descriptor") or {}
        if (meta.get("conversation_id") == conversation_id and meta.get("attempt_number") == attempt_number
                and other.get("kind") == descriptor.kind.value and other.get("slot") == descriptor.slot
                and other.get("producing_job_id") == descriptor.producing_job_id):
            return meta.get("artifact_id")
    return None


def _json_validator(contract: str):
    def check(data: bytes) -> None:
        try:
            payload = json.loads(data.decode("utf-8"))
            canonical = canonical_content(contract, payload)
        except (ValueError, UnicodeDecodeError):
            raise _validation(f"Uploaded bytes are not a valid {contract} document", reason="not_canonical_content") from None
        if canonical != data:
            raise _validation(f"Uploaded bytes are not the canonical {contract} document", reason="not_canonical_content")

    return check


def commit_upload(conn, store, principal: Principal, upload_id: str, body: UploadCommit) -> Artifact:
    """``commitUpload``: verify the bytes against the grant, then commit the artifact row."""
    with db.transaction(conn):
        record = store.objects.get_upload(conn, upload_id)
        if record is None or record.purpose != UPLOAD_PURPOSE:
            raise not_found("Upload", upload_id=upload_id)
        if not isinstance(principal, ServiceKeyPrincipal) or not principal.has_scope(record.required_scope):
            raise StoreError(ErrorCode.INSUFFICIENT_SCOPE, "The key must hold the scope that created the grant",
                             details={"required_scope": record.required_scope.value if record.required_scope else None})
        meta = record.metadata
        descriptor = ArtifactDescriptor.model_validate(meta["descriptor"])
        artifact_id = meta["artifact_id"]
        existing = artifact_row(conn, artifact_id)
        if existing is None:
            existing = find_natural(conn, meta["conversation_id"], descriptor, meta.get("attempt_number"))
        if existing is not None:
            # Natural replay: the artifact is already committed (by this upload or an identical one).
            if existing["checksum"] != body.checksum or existing["size_bytes"] != body.size_bytes:
                raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "Commit differs from the committed artifact", details={"reason": "commit_differs_from_grant"})
            if record.status == "received":
                store.objects.commit_upload(conn, upload_id, checksum=body.checksum, size_bytes=body.size_bytes)
            return artifact_from_row(existing)
        model = content_model_for(descriptor.kind)
        validate = _json_validator(descriptor.content_contract) if model is not None else None
        store.objects.commit_upload(conn, upload_id, checksum=body.checksum, size_bytes=body.size_bytes, validate=validate)
        return insert_artifact(conn, artifact_id=artifact_id, conversation_id=meta["conversation_id"], descriptor=descriptor,
                               attempt_number=meta.get("attempt_number"), storage=ArtifactStorage.OBJECT, upload_id=upload_id)


# --- Store-minted rubric snapshots ----------------------------------------------------------


def _snapshot_descriptor(content: RubricSnapshotContent, slot: str, labels: Dict[str, str]) -> Tuple[ArtifactDescriptor, bytes]:
    data = canonical_content(RUBRIC_SNAPSHOT_CONTRACT, content.model_dump(mode="json"))
    descriptor = ArtifactDescriptor(
        kind=ArtifactKind.RUBRIC_SNAPSHOT, slot=slot, content_type=JSON_CONTENT_TYPE, size_bytes=len(data), checksum=sha256_checksum(data),
        content_contract=RUBRIC_SNAPSHOT_CONTRACT, sensitivity=Sensitivity.DERIVED, labels=labels,
    )
    return descriptor, data


def mint_rubric_snapshot(conn, store, conversation_id: str, body: RubricSnapshotRequest) -> Artifact:
    """``mintRubricSnapshot``: idempotent per (conversation, rubric_id, version)."""
    from ..results import api as results_api

    slot = rubric_snapshot_slot(body.rubric_id, body.version)
    with db.transaction(conn):
        require_conversation_row(conn, conversation_id)
        existing = conn.execute(
            "SELECT * FROM q_artifacts WHERE conversation_id = ? AND kind = ? AND slot = ? AND linked = 1 ORDER BY version DESC LIMIT 1",
            (conversation_id, ArtifactKind.RUBRIC_SNAPSHOT.value, slot),
        ).fetchone()
        if existing is not None:
            return artifact_from_row(existing)
        content = results_api.rubric_snapshot_content(conn, body.rubric_id, body.version)
        if content is None:
            raise not_found("Rubric version", rubric_id=body.rubric_id, version=body.version)
        descriptor, data = _snapshot_descriptor(content, slot, {"rubric_id": body.rubric_id, "rubric_version": str(body.version)})
        store.objects.put_bytes(data)
        return insert_artifact(conn, artifact_id=new_id("art"), conversation_id=conversation_id, descriptor=descriptor,
                               attempt_number=None, storage=ArtifactStorage.INLINE)


def mint_draft_snapshot(conn, store, conversation_id: str, request_id: str, content: RubricSnapshotContent) -> Artifact:
    """The draft snapshot ``testRubricDraft`` mints in the request's draft slot (caller's transaction)."""
    slot = draft_rubric_snapshot_slot(request_id, content.rubric_id, content.draft_revision)
    descriptor, data = _snapshot_descriptor(content, slot, {"rubric_id": content.rubric_id, "draft_revision": str(content.draft_revision)})
    store.objects.put_bytes(data)
    return insert_artifact(conn, artifact_id=new_id("art"), conversation_id=conversation_id, descriptor=descriptor,
                           attempt_number=None, storage=ArtifactStorage.INLINE)


# --- Store-minted signal taxonomy snapshots (contract 1.3.0) ------------------------------------


def _signal_snapshot(conn, store, conversation_id: str, content: SignalTaxonomySnapshotContent, slot: str) -> Artifact:
    data = canonical_content(SIGNAL_SNAPSHOT_CONTRACT, content.model_dump(mode="json"))
    labels = {"taxonomy_digest": content.taxonomy_ref.digest}
    if content.taxonomy_ref.version is not None:
        labels["taxonomy_version"] = str(content.taxonomy_ref.version)
    descriptor = ArtifactDescriptor(
        kind=ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, slot=slot, content_type=JSON_CONTENT_TYPE, size_bytes=len(data), checksum=sha256_checksum(data),
        content_contract=SIGNAL_SNAPSHOT_CONTRACT, sensitivity=Sensitivity.DERIVED, labels=labels,
    )
    store.objects.put_bytes(data)
    return insert_artifact(conn, artifact_id=new_id("art"), conversation_id=conversation_id, descriptor=descriptor,
                           attempt_number=None, storage=ArtifactStorage.INLINE)


def ensure_published_signal_snapshot(conn, store, conversation_id: str, version: int) -> Artifact:
    """The linked snapshot of published taxonomy ``version`` with the current settings, in slot
    ``signals:v<version>`` (the caller's transaction). The linked one is reused while it is current
    (``signals.signal_taxonomy_snapshot_current``); after a settings change a new version is linked
    in the same slot. 404 for an unknown version, 409 ``conflict`` (reason ``redacted``) for a
    version whose text was redacted."""
    from ..results import api as results_api

    found = results_api.signal_taxonomy_version(conn, version)
    if found is None:
        raise not_found("Signal taxonomy version", version=version)
    if found.text_redacted:
        raise StoreError(ErrorCode.CONFLICT, "A redacted taxonomy version cannot be minted into a snapshot", details={"reason": "redacted", "version": version})
    settings = results_api.signal_settings(conn)
    slot = signal_taxonomy_snapshot_slot(version)
    existing = conn.execute(
        "SELECT * FROM q_artifacts WHERE conversation_id = ? AND kind = ? AND slot = ? AND linked = 1 ORDER BY version DESC LIMIT 1",
        (conversation_id, ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT.value, slot),
    ).fetchone()
    if existing is not None:
        linked = SignalTaxonomySnapshotContent.model_validate(json.loads(store.objects.read_bytes(existing["checksum"])))
        if signal_taxonomy_snapshot_current(linked, found.ref, settings):
            return artifact_from_row(existing)
    content = SignalTaxonomySnapshotContent(source="published", taxonomy_ref=found.ref, taxonomy=found.taxonomy, settings=settings)
    return _signal_snapshot(conn, store, conversation_id, content, slot)


def mint_signal_taxonomy_snapshot(conn, store, conversation_id: str, body: SignalTaxonomySnapshotRequest) -> Artifact:
    """``mintSignalTaxonomySnapshot``: idempotent per (conversation, version, current settings)."""
    with db.transaction(conn):
        require_conversation_row(conn, conversation_id)
        return ensure_published_signal_snapshot(conn, store, conversation_id, body.version)


def mint_preview_signal_snapshot(conn, store, conversation_id: str, request_id: str, content: SignalTaxonomySnapshotContent) -> Artifact:
    """A preview request's snapshot of an (unsaved) taxonomy, in ``draft:<request_id>:signals:preview``
    (the caller's transaction)."""
    return _signal_snapshot(conn, store, conversation_id, content, preview_signal_taxonomy_snapshot_slot(request_id))


def read_json(store, artifact: Artifact) -> Any:
    return json.loads(store.objects.read_bytes(artifact.checksum).decode("utf-8"))


# --- reads ------------------------------------------------------------------------------------


def list_artifacts(conn, conversation_id: str, query: ArtifactListQuery) -> Page[Artifact]:
    require_conversation_row(conn, conversation_id)
    where = ["conversation_id = ?"]
    args: List[Any] = [conversation_id]
    if query.kind is not None:
        where.append("kind = ?")
        args.append(query.kind.value)
    if query.slot is not None:
        where.append("slot = ?")
        args.append(query.slot)
    if not query.include_superseded:
        where.append("superseded_by IS NULL")
    if not query.include_unlinked:
        where.append("linked = 1")
    if not (query.include_draft_tests or (query.slot is not None and is_draft_test_slot(query.slot))):
        where.append("slot NOT LIKE 'draft:%'")
    after = pagination.decode(query.page_token, 2)
    if after is not None:
        where.append("(committed_at > ? OR (committed_at = ? AND id > ?))")
        args.extend([after[0], after[0], after[1]])
    rows = conn.execute(f"SELECT * FROM q_artifacts WHERE {' AND '.join(where)} ORDER BY committed_at, id LIMIT ?", (*args, query.limit + 1)).fetchall()
    items = [artifact_from_row(r) for r in rows[: query.limit]]
    token = pagination.encode(rows[query.limit - 1]["committed_at"], rows[query.limit - 1]["id"]) if len(rows) > query.limit else None
    return Page[Artifact](items=items, next_page_token=token)


def list_linked(conn, conversation_id: str, *, kind: Optional[ArtifactKind] = None, slot: Optional[str] = None,
                include_superseded: bool = False, include_draft_tests: bool = False) -> List[Artifact]:
    where = ["conversation_id = ?", "linked = 1"]
    args: List[Any] = [conversation_id]
    if kind is not None:
        where.append("kind = ?")
        args.append(ArtifactKind(kind).value)
    if slot is not None:
        where.append("slot = ?")
        args.append(slot)
    if not include_superseded:
        where.append("superseded_by IS NULL")
    if not (include_draft_tests or (slot is not None and is_draft_test_slot(slot))):
        where.append("slot NOT LIKE 'draft:%'")
    rows = conn.execute(f"SELECT * FROM q_artifacts WHERE {' AND '.join(where)} ORDER BY kind, slot, version", args).fetchall()
    return [artifact_from_row(r) for r in rows]


def content_grant(conn, store, artifact_id: str) -> Dict[str, Any]:
    row = require_artifact_row(conn, artifact_id)
    ticket = store.objects.content_grant(conn, artifact_id=row["id"], checksum=row["checksum"], content_type=row["content_type"])
    return {"artifact_id": row["id"], "url": ticket.url, "expires_at": ticket.expires_at, "checksum": row["checksum"],
            "content_type": row["content_type"], "size_bytes": row["size_bytes"]}


# --- orphans ---------------------------------------------------------------------------------


def sweep_orphans(conn, store) -> int:
    """Delete committed job outputs no completion linked, once ``ORPHAN_ARTIFACT_RETENTION`` has
    passed and the attempt that produced them has ended. Object bytes go too when no other
    artifact row shares the checksum. Returns the number of artifacts removed."""
    retention = store.config.parameters.orphan_artifact_retention_seconds
    cutoff = db.ts(conn.now() - timedelta(seconds=retention))
    removed: List[str] = []
    with db.transaction(conn):
        rows = conn.execute(
            "SELECT a.id, a.checksum FROM q_artifacts a LEFT JOIN q_attempts t ON t.job_id = a.producing_job_id "
            "AND t.attempt_number = a.producing_attempt_number WHERE a.linked = 0 AND a.committed_at < ? "
            "AND (t.status IS NULL OR t.status != 'running')",
            (cutoff,),
        ).fetchall()
        for row in rows:
            conn.execute("DELETE FROM q_artifacts WHERE id = ?", (row["id"],))
            if conn.execute("SELECT 1 FROM q_artifacts WHERE checksum = ? LIMIT 1", (row["checksum"],)).fetchone() is None:
                removed.append(row["checksum"])
    for checksum in removed:
        store.objects.delete(checksum)
    return len(rows)
