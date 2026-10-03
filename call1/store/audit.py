"""The hash-chained audit log (``events.AuditEvent``).

Every route the contract marks ``audited`` writes one event with ``audit.append(conn, ...)`` inside
the transaction that made the change. ``sequence`` is dense and monotonic; each body carries the
previous event's digest and ``event_digest = canonical_digest(body)``, so the log is
tamper-evident (``verify_chain``). Details are safe scalars only: never content or secrets.
"""

from __future__ import annotations

from typing import List, Mapping, Optional

from call1.contracts.common import JsonScalar, Page
from call1.contracts.events import Actor, ActorKind, AuditAction, AuditEvent, AuditEventBody, AuditQuery, AuditTarget, audit_event_digest

from . import db, pagination
from .ids import new_id
from .principals import AnonymousPrincipal, Principal, ServiceKeyPrincipal, SessionPrincipal


def actor_for(principal: Principal) -> Actor:
    """The audit actor for the request's principal."""
    if isinstance(principal, SessionPrincipal):
        return Actor(kind=ActorKind.REVIEWER, account_id=principal.account_id, session_id=principal.session_id, display=principal.display_name)
    if isinstance(principal, ServiceKeyPrincipal):
        return Actor(kind=ActorKind.PROCESS_SERVICE, installation_id=principal.installation_id, display=principal.key_prefix or None)
    if isinstance(principal, AnonymousPrincipal):
        return Actor(kind=ActorKind.STORE_SYSTEM, display="anonymous request")
    raise TypeError(f"unknown principal {principal!r}")


def installer_actor(os_user: str) -> Actor:
    """Host commands run as the Store OS user (``python -m call1.store ...``)."""
    return Actor(kind=ActorKind.INSTALLER, display=os_user[:200] or "installer")


def break_glass_actor(os_user: str) -> Actor:
    return Actor(kind=ActorKind.BREAK_GLASS, display=os_user[:200] or "break-glass")


SYSTEM_ACTOR = Actor(kind=ActorKind.STORE_SYSTEM, display="Store")


def append(conn, *, actor: Actor, action: AuditAction, target_kind: str, target_id: str,
           details: Optional[Mapping[str, JsonScalar]] = None) -> AuditEvent:
    """Write one audit event in the caller's open transaction and return it."""
    if not conn.in_transaction:
        raise RuntimeError("audit.append must run inside the transaction that made the change")
    last = conn.execute("SELECT sequence, event_digest FROM audit_events ORDER BY sequence DESC LIMIT 1").fetchone()
    body = AuditEventBody(
        id=new_id("evt"),
        sequence=(int(last["sequence"]) + 1) if last else 1,
        occurred_at=conn.now(),
        actor=actor,
        action=AuditAction(action),
        target=AuditTarget(kind=target_kind, id=target_id),
        details=dict(details or {}),
        previous_event_digest=last["event_digest"] if last else None,
    )
    event = AuditEvent(**body.model_dump(), event_digest=audit_event_digest(body))
    conn.execute(
        "INSERT INTO audit_events (sequence, id, occurred_at, actor_json, actor_kind, action, target_kind, target_id, details_json, "
        "previous_event_digest, event_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (event.sequence, event.id, db.ts(event.occurred_at), db.dumps(event.actor), event.actor.kind.value, event.action.value,
         event.target.kind, event.target.id, db.dumps(event.details), event.previous_event_digest, event.event_digest),
    )
    return event


def _from_row(row) -> AuditEvent:
    return AuditEvent(
        id=row["id"],
        sequence=row["sequence"],
        occurred_at=db.parse_ts(row["occurred_at"]),
        actor=Actor.model_validate(db.loads(row["actor_json"])),
        action=AuditAction(row["action"]),
        target=AuditTarget(kind=row["target_kind"], id=row["target_id"]),
        details=db.loads(row["details_json"]),
        previous_event_digest=row["previous_event_digest"],
        event_digest=row["event_digest"],
    )


def get(conn, event_id: str) -> Optional[AuditEvent]:
    row = conn.execute("SELECT * FROM audit_events WHERE id = ?", (event_id,)).fetchone()
    return None if row is None else _from_row(row)


def list_events(conn, query: AuditQuery) -> Page[AuditEvent]:
    """Newest first, filtered by the query, paged by sequence."""
    where: List[str] = []
    args: List[object] = []
    before = pagination.decode(query.page_token, 1)
    if before is not None:
        where.append("sequence < ?")
        args.append(int(before[0]))
    if query.action is not None:
        where.append("action = ?")
        args.append(query.action.value)
    if query.actor_kind is not None:
        where.append("actor_kind = ?")
        args.append(query.actor_kind.value)
    if query.target_kind is not None:
        where.append("target_kind = ?")
        args.append(query.target_kind)
    if query.target_id is not None:
        where.append("target_id = ?")
        args.append(query.target_id)
    if query.since is not None:
        where.append("occurred_at >= ?")
        args.append(db.ts(query.since))
    if query.until is not None:
        where.append("occurred_at < ?")
        args.append(db.ts(query.until))
    sql = "SELECT * FROM audit_events" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY sequence DESC LIMIT ?"
    rows = conn.execute(sql, (*args, query.limit + 1)).fetchall()
    items = [_from_row(r) for r in rows[: query.limit]]
    token = pagination.encode(items[-1].sequence) if len(rows) > query.limit else None
    return Page[AuditEvent](items=items, next_page_token=token)


def verify_chain(conn) -> bool:
    """Recompute every digest and link; False on any break."""
    previous: Optional[str] = None
    expected_sequence = 1
    for row in conn.execute("SELECT * FROM audit_events ORDER BY sequence"):
        try:
            event = _from_row(row)  # the model validator recomputes event_digest
        except ValueError:
            return False
        if event.sequence != expected_sequence or event.previous_event_digest != previous:
            return False
        previous = event.event_digest
        expected_sequence += 1
    return True
