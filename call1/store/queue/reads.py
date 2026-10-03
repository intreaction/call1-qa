"""Queue reads: conversations, graphs, jobs and attempts."""

from __future__ import annotations

from typing import Any, List

from call1.contracts.calls import CallMetadata, Conversation, ConversationRegistered, ConversationRegistration, IngestionKind, merge_call_metadata
from call1.contracts.common import Page
from call1.contracts.events import AuditAction
from call1.contracts.jobs import Attempt, Job, JobGraph, JobListQuery

from .. import audit, db, pagination
from ..errors import not_found
from ..ids import new_id
from ..principals import ServiceKeyPrincipal
from .graphs import graph_model
from .records import JobReader, attempt_from_row, conversation_from_row, require_conversation_row, require_job_row


def register_conversation(conn, principal: ServiceKeyPrincipal, body: ConversationRegistration) -> ConversationRegistered:
    """``registerConversation``: natural idempotency by the source's dedup identity. A new
    conversation is announced to the results area (``on_conversation_registered``) in the same
    transaction, so a call lists as Analyzing at once.

    A known identity returns the existing conversation. Since contract 1.1.0 a re-registration whose
    ``call_metadata`` changes the stored metadata (``calls.merge_call_metadata``: only the fields the
    body sets replace stored ones) updates it in the same transaction: the conversation row, the
    results projections and the ``call`` change event (``on_call_metadata_updated``), and a
    ``call_metadata_updated`` audit event naming the changed fields, never their values. Nothing is
    reprocessed. Identical metadata is a pure replay with no event."""
    from ..results import projections

    identity = body.source.dedup_identity
    with db.transaction(conn):
        row = conn.execute("SELECT * FROM q_conversations WHERE dedup_identity = ?", (identity,)).fetchone()
        if row is not None:
            return _reregister(conn, principal, row, body)
        conversation_id = new_id("conv")
        call_id = new_id("call") if body.ingestion_kind is IngestionKind.CALL_AUDIO else None
        conn.execute(
            "INSERT INTO q_conversations (id, dedup_identity, ingestion_kind, source_json, call_id, call_metadata_json, created_at, "
            "registered_by_installation_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, identity, body.ingestion_kind.value, db.dumps(body.source), call_id,
             db.dumps(body.call_metadata) if body.call_metadata is not None else None, db.ts(conn.now()), principal.installation_id),
        )
        conversation = conversation_from_row(require_conversation_row(conn, conversation_id))
        projections.on_conversation_registered(conn, conversation)
        return ConversationRegistered(conversation=conversation, created=True)


def _reregister(conn, principal: ServiceKeyPrincipal, row, body: ConversationRegistration) -> ConversationRegistered:
    """The known-identity branch of ``register_conversation`` (inside its transaction)."""
    from ..results import projections

    existing = conversation_from_row(row)
    if body.call_metadata is None:  # the body names no metadata at all: nothing to merge
        return ConversationRegistered(conversation=existing, created=False)
    # With nothing stored, the call shows CallMetadata's defaults ("Unknown"), so compare against
    # those: re-sending the defaults is a replay, not an update. The merged values equal
    # merge_call_metadata(None, incoming) field for field.
    stored = existing.call_metadata if existing.call_metadata is not None else CallMetadata()
    merged, changed = merge_call_metadata(stored, body.call_metadata)
    if not changed:
        return ConversationRegistered(conversation=existing, created=False)
    conn.execute("UPDATE q_conversations SET call_metadata_json = ? WHERE id = ?", (db.dumps(merged), existing.id))
    conversation = conversation_from_row(require_conversation_row(conn, existing.id))
    projections.on_call_metadata_updated(conn, conversation, changed)
    target_kind, target_id = ("call", conversation.call_id) if conversation.call_id else ("conversation", conversation.id)
    audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.CALL_METADATA_UPDATED, target_kind=target_kind, target_id=target_id,
                 details={"conversation_id": conversation.id, "updated_fields": ",".join(changed)})
    return ConversationRegistered(conversation=conversation, created=False, metadata_updated=True, updated_fields=changed)


def get_conversation(conn, conversation_id: str) -> Conversation:
    return conversation_from_row(require_conversation_row(conn, conversation_id))


def get_graph(conn, graph_id: str) -> JobGraph:
    if conn.execute("SELECT 1 FROM q_graphs WHERE id = ?", (graph_id,)).fetchone() is None:
        raise not_found("Job graph", graph_id=graph_id)
    return graph_model(conn, graph_id, created=False)


def get_job(conn, job_id: str) -> Job:
    return JobReader(conn).job_by_id(job_id)


def list_jobs(conn, query: JobListQuery) -> Page[Job]:
    """Ordered ``priority desc, created_at asc, id asc`` (claims walk call by call instead:
    ``lifecycle.CLAIM_CANDIDATES_SQL``). ``memory_slot`` (1.3.0) filters on
    ``q_jobs.memory_slot``; Process's on-device training start check asks for ``local_memory`` jobs that are
    QUEUED or RUNNING with ``limit=1``."""
    where: List[str] = []
    args: List[Any] = []
    for column, value in (("conversation_id", query.conversation_id), ("graph_id", query.graph_id),
                          ("status", query.status.value if query.status else None), ("job_type", query.job_type.value if query.job_type else None),
                          ("memory_slot", query.memory_slot.value if query.memory_slot else None)):
        if value is not None:
            where.append(f"{column} = ?")
            args.append(value)
    after = pagination.decode(query.page_token, 3)
    if after is not None:
        priority, created_at, job_id = int(after[0]), after[1], after[2]
        where.append("(priority < ? OR (priority = ? AND (created_at > ? OR (created_at = ? AND id > ?))))")
        args.extend([priority, priority, created_at, created_at, job_id])
    sql = "SELECT * FROM q_jobs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY priority DESC, created_at, id LIMIT ?"
    rows = conn.execute(sql, (*args, query.limit + 1)).fetchall()
    page = rows[: query.limit]
    token = pagination.encode(page[-1]["priority"], page[-1]["created_at"], page[-1]["id"]) if len(rows) > query.limit else None
    return Page[Job](items=JobReader(conn).jobs(page), next_page_token=token)


def list_attempts(conn, job_id: str) -> Page[Attempt]:
    require_job_row(conn, job_id)
    rows = conn.execute("SELECT * FROM q_attempts WHERE job_id = ? ORDER BY attempt_number", (job_id,)).fetchall()
    return Page[Attempt](items=[attempt_from_row(r) for r in rows], next_page_token=None)
