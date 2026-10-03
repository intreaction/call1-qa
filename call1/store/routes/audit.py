"""``GET /store/v1/admin/audit`` (listAuditEvents): the hash-chained audit log, newest first."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query

from call1.contracts.common import Page
from call1.contracts.events import AuditEvent, AuditQuery

from .. import audit
from ..db import StoreConnection, read_snapshot
from ..deps import get_conn
from . import router


@router.operation("listAuditEvents")
def list_audit_events(query: Annotated[AuditQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> Page[AuditEvent]:
    with read_snapshot(conn):
        return audit.list_events(conn, query)
