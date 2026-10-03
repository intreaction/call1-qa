"""Server-side reviewer sessions: creation, lookup for the principal guard, and the contract views.

A session row keeps ``sha256(cookie)`` and ``sha256(csrf)`` only. Expiry follows
``ContractParameters``: ``session_idle_lifetime_seconds`` slides on use (at most one write a
minute per session) and never passes ``session_absolute_lifetime_seconds`` from sign-in. The
account's role and status are read on every request, so a demotion applies from the next request
and a disabled account is refused at once.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from call1.contracts.auth import ROLE_PERMISSIONS, Permission, SessionInfo, SessionListItem
from call1.contracts.common import ReviewerRole
from call1.contracts.errors import ErrorCode

from .. import db
from ..errors import StoreError
from ..ids import new_id
from ..principals import SessionPrincipal, generate_secret, hash_secret, secrets_equal
from . import crypto

SLIDE_EVERY = timedelta(seconds=60)
"""Idle expiry and last_seen_at are rewritten at most this often per session."""


def create_session(conn, parameters, *, account_id: str, authenticator_id: str, now: datetime) -> Tuple[str, str, str]:
    """Insert a session in the caller's transaction. Returns (session_id, cookie_value, csrf_token)."""
    session_id = new_id("sess")
    cookie = generate_secret(32)
    csrf = crypto.csrf_token_for(conn, cookie)
    absolute = now + timedelta(seconds=parameters.session_absolute_lifetime_seconds)
    idle = min(now + timedelta(seconds=parameters.session_idle_lifetime_seconds), absolute)
    conn.execute(
        "INSERT INTO auth_sessions (id, account_id, cookie_hash, csrf_hash, authenticator_id, created_at, last_seen_at, idle_expires_at, "
        "absolute_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, account_id, hash_secret(cookie), hash_secret(csrf), authenticator_id, db.ts(now), db.ts(now), db.ts(idle), db.ts(absolute)),
    )
    return session_id, cookie, csrf


_LOOKUP = (
    "SELECT s.*, a.email, a.display_name, a.role, a.status FROM auth_sessions s JOIN auth_accounts a ON a.id = s.account_id "
    "WHERE s.cookie_hash = ?"
)


def authenticate(conn, parameters, cookie_value: str, now: datetime) -> Optional[SessionPrincipal]:
    """The guard's session lookup (``AuthBackend.authenticate_session``)."""
    row = conn.execute(_LOOKUP, (hash_secret(cookie_value),)).fetchone()
    if row is None:
        return None
    if row["status"] == "disabled":
        raise StoreError(ErrorCode.ACCOUNT_DISABLED, "This account is disabled; ask an administrator")
    if row["revoked_at"] is not None or row["status"] != "active":
        return None
    idle = db.parse_ts(row["idle_expires_at"])
    absolute = db.parse_ts(row["absolute_expires_at"])
    if now >= idle or now >= absolute:
        raise StoreError(ErrorCode.SESSION_EXPIRED, "Session expired; sign in again")
    last_seen = db.parse_ts(row["last_seen_at"])
    if now - last_seen >= SLIDE_EVERY:
        new_idle = min(now + timedelta(seconds=parameters.session_idle_lifetime_seconds), absolute)
        with db.transaction(conn):
            conn.execute("UPDATE auth_sessions SET last_seen_at = ?, idle_expires_at = ? WHERE id = ? AND revoked_at IS NULL",
                         (db.ts(now), db.ts(new_idle), row["id"]))
    return SessionPrincipal(
        session_id=row["id"],
        account_id=row["account_id"],
        email=row["email"],
        display_name=row["display_name"],
        role=ReviewerRole(row["role"]),
        authenticator_id=row["authenticator_id"],
    )


def verify_csrf(conn, session_id: str, header_value: Optional[str]) -> bool:
    if not header_value:
        return False
    row = conn.execute("SELECT csrf_hash FROM auth_sessions WHERE id = ? AND revoked_at IS NULL", (session_id,)).fetchone()
    if row is None:
        return False
    return secrets_equal(row["csrf_hash"], hash_secret(header_value))


def session_row(conn, session_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM auth_sessions WHERE id = ?", (session_id,)).fetchone()


def permissions_of(role: ReviewerRole) -> List[Permission]:
    granted = ROLE_PERMISSIONS[ReviewerRole(role)]
    return [p for p in Permission if p in granted]


def session_info(conn, session_id: str, csrf_token: str) -> SessionInfo:
    """``SessionInfo`` for a live session, with the account's current role and permissions."""
    row = conn.execute(
        "SELECT s.*, a.email, a.display_name, a.role, (SELECT COUNT(*) FROM auth_credentials c WHERE c.account_id = a.id "
        "AND c.revoked_at IS NULL) AS credential_count FROM auth_sessions s JOIN auth_accounts a ON a.id = s.account_id WHERE s.id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise StoreError(ErrorCode.UNAUTHENTICATED, "Session not recognised; sign in again")
    role = ReviewerRole(row["role"])
    return SessionInfo(
        session_id=row["id"],
        account_id=row["account_id"],
        email=row["email"],
        display_name=row["display_name"],
        role=role,
        permissions=permissions_of(role),
        created_at=db.parse_ts(row["created_at"]),
        last_seen_at=db.parse_ts(row["last_seen_at"]),
        idle_expires_at=db.parse_ts(row["idle_expires_at"]),
        absolute_expires_at=db.parse_ts(row["absolute_expires_at"]),
        authenticator_id_used=row["authenticator_id"],
        prompt_second_authenticator=int(row["credential_count"]) < 2,
        csrf_token=csrf_token,
    )


def list_items(conn, account_id: str, *, current_session_id: str, now: datetime) -> List[SessionListItem]:
    """Live (not revoked, not expired) sessions of an account, newest first. Handles only."""
    rows = conn.execute(
        "SELECT * FROM auth_sessions WHERE account_id = ? AND revoked_at IS NULL AND idle_expires_at > ? AND absolute_expires_at > ? "
        "ORDER BY created_at DESC, id DESC",
        (account_id, db.ts(now), db.ts(now)),
    ).fetchall()
    return [
        SessionListItem(
            session_id=r["id"],
            account_id=r["account_id"],
            created_at=db.parse_ts(r["created_at"]),
            last_seen_at=db.parse_ts(r["last_seen_at"]),
            idle_expires_at=db.parse_ts(r["idle_expires_at"]),
            absolute_expires_at=db.parse_ts(r["absolute_expires_at"]),
            current=r["id"] == current_session_id,
        )
        for r in rows
    ]
