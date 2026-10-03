"""Catalog snapshots: one read-only snapshot per Process installation, replaced whole on publish.

Process owns the catalog; Store keeps what Process published so Evaluate and admin screens can
show routes and qualification without reaching Process. Publishing is audited, and an entry that
names a Call1-operated destination on any route but ``call1_confidential`` is refused
(``route_not_permitted``, the Stage 0 host rule).
"""

from __future__ import annotations

from call1.contracts.catalog import CatalogSnapshot
from call1.contracts.common import Page, canonical_digest
from call1.contracts.custody import IN_PROCESS_DESTINATION, RouteClass, call1_operated_host, normalize_host
from call1.contracts.errors import ErrorCode
from call1.contracts.events import AuditAction

from .. import audit, db
from ..errors import StoreError
from ..principals import ServiceKeyPrincipal, require_own_installation
from .changes import ChangeLog


def publish(conn, principal: ServiceKeyPrincipal, body: CatalogSnapshot) -> CatalogSnapshot:
    require_own_installation(principal, body.installation_id)
    for entry in body.entries:
        host = normalize_host(entry.destination_host)
        if entry.route_class is not RouteClass.CALL1_CONFIDENTIAL and host != IN_PROCESS_DESTINATION and call1_operated_host(host):
            raise StoreError(ErrorCode.ROUTE_NOT_PERMITTED, "A Call1-operated host is reachable only through the attested call1_confidential route",
                             details={"entry_id": entry.entry.entry_id, "reason": "call1_operated_destination"})
    digest = canonical_digest(body.model_dump(mode="json"))
    with db.transaction(conn):
        row = conn.execute("SELECT digest FROM q_catalog_snapshots WHERE installation_id = ?", (body.installation_id,)).fetchone()
        if row is not None and row["digest"] == digest:
            return body
        now = db.ts(conn.now())
        conn.execute(
            "INSERT INTO q_catalog_snapshots (installation_id, snapshot_json, digest, published_at, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (installation_id) DO UPDATE SET snapshot_json = excluded.snapshot_json, digest = excluded.digest, "
            "published_at = excluded.published_at, updated_at = excluded.updated_at",
            (body.installation_id, db.dumps(body), digest, db.ts(body.published_at), now),
        )
        audit.append(conn, actor=audit.actor_for(principal), action=AuditAction.CATALOG_SNAPSHOT_PUBLISHED, target_kind="installation",
                     target_id=body.installation_id, details={"catalog_version": body.catalog_version, "entries": len(body.entries), "digest": digest})
        changes = ChangeLog(conn)
        changes.catalog(body.installation_id)
        changes.flush()
    return body


def list_snapshots(conn) -> Page[CatalogSnapshot]:
    rows = conn.execute("SELECT snapshot_json FROM q_catalog_snapshots ORDER BY installation_id").fetchall()
    return Page[CatalogSnapshot](items=[CatalogSnapshot.model_validate(db.loads(r["snapshot_json"])) for r in rows], next_page_token=None)
