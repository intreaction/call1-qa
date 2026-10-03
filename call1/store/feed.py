"""The change feed (``GET /store/v1/changes``): append inside the writing transaction, read by cursor.

Every area that changes something a client follows calls ``feed.append(conn, kind, resource_id,
version, status, conversation_id=..., call_id=...)`` inside the same transaction as the change, so
the event commits with it or not at all. Events carry IDs, versions and statuses, never content.

Cursors are ``<feed_epoch>-<seq>``. ``seq`` is assigned inside the transaction and SQLite
serializes writers, so cursor order is commit order and no event appears below a cursor already
served. A cursor from another epoch (before a restore) or ahead of the latest is 410
``cursor_unknown``; one older than the pruned horizon is 410 ``cursor_expired``.
"""

from __future__ import annotations

import re
from datetime import timedelta
from typing import List, Optional, Sequence

from call1.contracts.errors import ErrorCode
from call1.contracts.events import CHANGE_KINDS_BY_PRINCIPAL, ChangeEvent, ChangeFeed, ChangeKind

from . import db
from .errors import StoreError

_CURSOR = re.compile(r"^(ep[0-9a-f]{12})-(\d{1,19})$")
META_PRUNED_THROUGH = "feed_pruned_through"


def feed_epoch(conn) -> str:
    epoch = db.get_meta(conn, db.META_FEED_EPOCH)
    if epoch is None:
        raise RuntimeError("feed epoch missing: Database.initialize() was not run")
    return epoch


def format_cursor(epoch: str, seq: int) -> str:
    return f"{epoch}-{seq:012d}"


def parse_cursor(conn, cursor: str) -> int:
    """The sequence number of a cursor from this epoch; 410 ``cursor_unknown`` otherwise."""
    match = _CURSOR.match(cursor)
    epoch = feed_epoch(conn)
    if not match or match.group(1) != epoch:
        raise StoreError(ErrorCode.CURSOR_UNKNOWN, "Cursor is from another feed epoch; re-snapshot", details={"feed_epoch": epoch})
    return int(match.group(2))


def _latest_seq(conn) -> int:
    row = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'change_events'").fetchone()
    return int(row["seq"]) if row is not None else 0


def latest_cursor(conn) -> str:
    return format_cursor(feed_epoch(conn), _latest_seq(conn))


def append(conn, kind: ChangeKind, resource_id: str, version: Optional[int] = None, status: Optional[str] = None, *,
           conversation_id: Optional[str] = None, call_id: Optional[str] = None) -> str:
    """Record one change in the caller's open transaction and return its cursor."""
    if not conn.in_transaction:
        raise RuntimeError("feed.append must run inside the transaction that made the change")
    kind = ChangeKind(kind)
    cur = conn.execute(
        "INSERT INTO change_events (occurred_at, kind, resource_id, conversation_id, call_id, version, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (db.ts(conn.now()), kind.value, resource_id, conversation_id, call_id, version, None if status is None else str(status)[:200]),
    )
    return format_cursor(feed_epoch(conn), int(cur.lastrowid))


def kinds_for(audience: str) -> List[ChangeKind]:
    """The kinds a principal's feed may contain (``events.CHANGE_KINDS_BY_PRINCIPAL``)."""
    return list(CHANGE_KINDS_BY_PRINCIPAL.get(audience, []))


def read(conn, *, audience: str, after: Optional[str], limit: int, kinds: Optional[Sequence[ChangeKind]], retention_seconds: int) -> ChangeFeed:
    """Events after ``after`` visible to ``audience`` (a role value or ``process_service_key``)."""
    allowed = set(kinds_for(audience))
    wanted = [k for k in (ChangeKind(k) for k in kinds)] if kinds else list(allowed)
    selected = sorted({k.value for k in wanted if k in allowed})
    with db.read_snapshot(conn):
        epoch = feed_epoch(conn)
        latest = _latest_seq(conn)
        pruned_through = int(db.get_meta(conn, META_PRUNED_THROUGH) or 0)
        start = parse_cursor(conn, after) if after is not None else pruned_through
        if start > latest:
            raise StoreError(ErrorCode.CURSOR_UNKNOWN, "Cursor is ahead of the feed; re-snapshot", details={"feed_epoch": epoch})
        if start < pruned_through:
            raise StoreError(ErrorCode.CURSOR_EXPIRED, "Cursor is older than the feed keeps", details={"oldest_cursor": format_cursor(epoch, pruned_through)})
        rows = []
        if selected:
            marks = ",".join("?" for _ in selected)
            rows = conn.execute(
                f"SELECT * FROM change_events WHERE seq > ? AND seq <= ? AND kind IN ({marks}) ORDER BY seq LIMIT ?",
                (start, latest, *selected, limit),
            ).fetchall()
        next_seq = int(rows[-1]["seq"]) if len(rows) == limit else latest
        now = conn.now()
    events = [
        ChangeEvent(
            cursor=format_cursor(epoch, int(row["seq"])),
            occurred_at=db.parse_ts(row["occurred_at"]),
            kind=ChangeKind(row["kind"]),
            resource_id=row["resource_id"],
            conversation_id=row["conversation_id"],
            call_id=row["call_id"],
            version=row["version"],
            status=row["status"],
        )
        for row in rows
    ]
    return ChangeFeed(
        events=events,
        next_cursor=format_cursor(epoch, next_seq),
        latest_cursor=format_cursor(epoch, latest),
        feed_epoch=epoch,
        retention_until=now - timedelta(seconds=retention_seconds),
    )


def prune(conn, *, retention_seconds: int) -> int:
    """Delete events older than the retention window; cursors below them become expired."""
    cutoff = db.ts(conn.now() - timedelta(seconds=retention_seconds))
    with db.transaction(conn):
        row = conn.execute("SELECT MAX(seq) AS s FROM change_events WHERE occurred_at < ?", (cutoff,)).fetchone()
        through = int(row["s"] or 0)
        if not through:
            return 0
        deleted = conn.execute("DELETE FROM change_events WHERE seq <= ?", (through,)).rowcount
        conn.execute("INSERT INTO store_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (META_PRUNED_THROUGH, str(through)))
        return deleted


def start_new_epoch(conn) -> str:
    """Begin a new feed epoch (every restore does this, Stage 4). Old cursors become unknown."""
    epoch = db.new_feed_epoch()
    with db.transaction(conn):
        conn.execute("UPDATE store_meta SET value = ? WHERE key = ?", (epoch, db.META_FEED_EPOCH))
    return epoch


__all__ = ["append", "read", "latest_cursor", "feed_epoch", "format_cursor", "parse_cursor", "prune", "start_new_epoch", "kinds_for"]
