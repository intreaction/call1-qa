"""Row mapping and queries over the auth tables (``migrations/030_auth.sql``).

Only the auth area imports this module. Other areas use ``call1.store.auth.api``. Every function
takes the caller's connection and runs inside whatever transaction the caller holds.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

from call1.contracts.auth import (
    AccountStatus,
    AuthenticatorTransport,
    BreakGlassRecord,
    Invitation,
    InvitationDelivery,
    InvitationStatus,
    ProcessInstallation,
    ReviewerAccount,
    ServiceKeyRecord,
    SetupCodePurpose,
    SetupCodeRecord,
    WebAuthnCredentialRecord,
)
from call1.contracts.common import ReviewerRole, ServiceScope

from .. import db

ACCOUNT_SELECT = (
    "SELECT a.*, (SELECT COUNT(*) FROM auth_credentials c WHERE c.account_id = a.id AND c.revoked_at IS NULL) "
    "AS authenticator_count FROM auth_accounts a"
)


def email_key(email: str) -> str:
    """Accounts and invitations match email case-insensitively."""
    return email.strip().lower()


def _bool(value) -> Optional[bool]:
    return None if value is None else bool(value)


# --- accounts ---------------------------------------------------------------------------------


def account_from_row(row: sqlite3.Row) -> ReviewerAccount:
    return ReviewerAccount(
        id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        role=ReviewerRole(row["role"]),
        status=AccountStatus(row["status"]),
        created_at=db.parse_ts(row["created_at"]),
        last_sign_in_at=db.parse_ts(row["last_sign_in_at"]),
        authenticator_count=int(row["authenticator_count"]),
        legacy_auditor_id=row["legacy_auditor_id"],
    )


def account_row(conn, account_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(ACCOUNT_SELECT + " WHERE a.id = ?", (account_id,)).fetchone()


def account_row_by_email(conn, email: str) -> Optional[sqlite3.Row]:
    return conn.execute(ACCOUNT_SELECT + " WHERE a.email_key = ?", (email_key(email),)).fetchone()


def get_account(conn, account_id: str) -> Optional[ReviewerAccount]:
    row = account_row(conn, account_id)
    return None if row is None else account_from_row(row)


def list_accounts(conn, *, role: Optional[ReviewerRole] = None, status: Optional[AccountStatus] = None,
                  after_id: Optional[str] = None, limit: Optional[int] = None) -> List[ReviewerAccount]:
    where, args = [], []
    if role is not None:
        where.append("a.role = ?")
        args.append(ReviewerRole(role).value)
    if status is not None:
        where.append("a.status = ?")
        args.append(AccountStatus(status).value)
    if after_id is not None:
        where.append("a.id > ?")
        args.append(after_id)
    sql = ACCOUNT_SELECT + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY a.id"
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    return [account_from_row(r) for r in conn.execute(sql, args).fetchall()]


def insert_account(conn, *, account_id: str, email: str, display_name: str, role: ReviewerRole, status: AccountStatus,
                   user_handle: str, now: datetime) -> None:
    conn.execute(
        "INSERT INTO auth_accounts (id, email, email_key, display_name, role, status, user_handle, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (account_id, email.strip(), email_key(email), display_name, ReviewerRole(role).value, AccountStatus(status).value,
         user_handle, db.ts(now), db.ts(now)),
    )


def update_account(conn, account_id: str, now: datetime, **fields) -> None:
    """Set the given columns (display_name, role, status, last_sign_in_at) and updated_at."""
    columns, args = [], []
    for name, value in fields.items():
        if name not in ("display_name", "role", "status", "last_sign_in_at"):
            raise ValueError(f"not an updatable account column: {name}")
        if isinstance(value, datetime):
            value = db.ts(value)
        elif hasattr(value, "value"):
            value = value.value
        columns.append(f"{name} = ?")
        args.append(value)
    columns.append("updated_at = ?")
    args.append(db.ts(now))
    conn.execute(f"UPDATE auth_accounts SET {', '.join(columns)} WHERE id = ?", (*args, account_id))


def active_admin_ids(conn, *, excluding: Sequence[str] = ()) -> List[str]:
    """Active admin accounts that can still sign in (at least one live credential)."""
    rows = conn.execute(
        "SELECT a.id FROM auth_accounts a WHERE a.role = 'admin' AND a.status = 'active' AND EXISTS "
        "(SELECT 1 FROM auth_credentials c WHERE c.account_id = a.id AND c.revoked_at IS NULL)"
    ).fetchall()
    return [r["id"] for r in rows if r["id"] not in excluding]


def any_active_admin(conn) -> bool:
    row = conn.execute("SELECT 1 FROM auth_accounts WHERE role = 'admin' AND status = 'active' LIMIT 1").fetchone()
    return row is not None


# --- credentials ------------------------------------------------------------------------------

_KNOWN_TRANSPORTS = {t.value for t in AuthenticatorTransport}


def known_transports(values: Iterable[str]) -> List[str]:
    """Browsers report open-ended transport strings; Store keeps the ones it knows, once each."""
    out: List[str] = []
    for value in values:
        if value in _KNOWN_TRANSPORTS and value not in out:
            out.append(value)
    return out


def credential_from_row(row: sqlite3.Row) -> WebAuthnCredentialRecord:
    return WebAuthnCredentialRecord(
        id=row["id"],
        account_id=row["account_id"],
        credential_id=row["credential_id"],
        public_key_cose=row["public_key_cose"],
        sign_count=int(row["sign_count"]),
        transports=[AuthenticatorTransport(t) for t in db.loads(row["transports_json"]) or []],
        aaguid=row["aaguid"],
        nickname=row["nickname"],
        backup_eligible=_bool(row["backup_eligible"]),
        backup_state=_bool(row["backup_state"]),
        created_at=db.parse_ts(row["created_at"]),
        last_used_at=db.parse_ts(row["last_used_at"]),
    )


def live_credential_rows(conn, account_id: str) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM auth_credentials WHERE account_id = ? AND revoked_at IS NULL ORDER BY created_at, id", (account_id,)
    ).fetchall()


def live_credential_row(conn, account_id: str, authenticator_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM auth_credentials WHERE id = ? AND account_id = ? AND revoked_at IS NULL", (authenticator_id, account_id)
    ).fetchone()


def credential_row_by_credential_id(conn, credential_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM auth_credentials WHERE credential_id = ? AND revoked_at IS NULL", (credential_id,)).fetchone()


def insert_credential(conn, *, authenticator_id: str, account_id: str, credential_id: str, public_key_cose: str, sign_count: int,
                      transports: Sequence[str], aaguid: Optional[str], nickname: Optional[str], backup_eligible: Optional[bool],
                      backup_state: Optional[bool], now: datetime) -> None:
    conn.execute(
        "INSERT INTO auth_credentials (id, account_id, credential_id, public_key_cose, sign_count, transports_json, aaguid, nickname, "
        "backup_eligible, backup_state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (authenticator_id, account_id, credential_id, public_key_cose, int(sign_count), db.dumps(known_transports(transports)), aaguid,
         nickname, None if backup_eligible is None else int(backup_eligible), None if backup_state is None else int(backup_state), db.ts(now)),
    )


def revoke_credentials(conn, account_id: str, now: datetime, reason: str, *, authenticator_id: Optional[str] = None) -> int:
    """Revoke one live credential of the account, or all of them. Returns how many were revoked."""
    sql = "UPDATE auth_credentials SET revoked_at = ?, revoked_reason = ? WHERE account_id = ? AND revoked_at IS NULL"
    args: list = [db.ts(now), reason, account_id]
    if authenticator_id is not None:
        sql += " AND id = ?"
        args.append(authenticator_id)
    return conn.execute(sql, args).rowcount


# --- sessions ---------------------------------------------------------------------------------


def revoke_sessions(conn, account_id: str, now: datetime, reason: str, *, authenticator_id: Optional[str] = None,
                    except_session_id: Optional[str] = None, session_id: Optional[str] = None) -> int:
    """Revoke live sessions of an account (optionally only those signed in with one authenticator,
    or one session). Returns how many were revoked."""
    sql = "UPDATE auth_sessions SET revoked_at = ?, revoked_reason = ? WHERE account_id = ? AND revoked_at IS NULL"
    args: list = [db.ts(now), reason, account_id]
    if authenticator_id is not None:
        sql += " AND authenticator_id = ?"
        args.append(authenticator_id)
    if except_session_id is not None:
        sql += " AND id != ?"
        args.append(except_session_id)
    if session_id is not None:
        sql += " AND id = ?"
        args.append(session_id)
    return conn.execute(sql, args).rowcount


# --- invitations ------------------------------------------------------------------------------


def invitation_status(row: sqlite3.Row, now: datetime) -> InvitationStatus:
    status = InvitationStatus(row["status"])
    if status is InvitationStatus.PENDING and db.parse_ts(row["expires_at"]) <= now:
        return InvitationStatus.EXPIRED
    return status


def invitation_from_row(row: sqlite3.Row, now: datetime) -> Invitation:
    return Invitation(
        id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        role=ReviewerRole(row["role"]),
        status=invitation_status(row, now),
        delivery=InvitationDelivery(row["delivery"]),
        token_hash=row["token_hash"],
        issued_by_account_id=row["issued_by_account_id"],
        issued_at=db.parse_ts(row["issued_at"]),
        expires_at=db.parse_ts(row["expires_at"]),
        redeemed_at=db.parse_ts(row["redeemed_at"]),
        account_id=row["account_id"],
        reinvite_of_account_id=row["reinvite_of_account_id"],
    )


def invitation_row(conn, invitation_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM auth_invitations WHERE id = ?", (invitation_id,)).fetchone()


def invitation_row_by_hash(conn, token_hash: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM auth_invitations WHERE token_hash = ?", (token_hash,)).fetchone()


def revoke_pending_invitations(conn, now: datetime, *, email: Optional[str] = None, reinvite_of_account_id: Optional[str] = None) -> List[str]:
    """Revoke pending invitations for an email or for a re-invited account; returns their IDs."""
    if email is None and reinvite_of_account_id is None:
        raise ValueError("name an email or an account")
    where = ["status = 'pending'"]
    args: list = []
    if email is not None:
        where.append("email_key = ?")
        args.append(email_key(email))
    if reinvite_of_account_id is not None:
        where.append("reinvite_of_account_id = ?")
        args.append(reinvite_of_account_id)
    ids = [r["id"] for r in conn.execute("SELECT id FROM auth_invitations WHERE " + " AND ".join(where), args).fetchall()]
    for invitation_id in ids:
        conn.execute("UPDATE auth_invitations SET status = 'revoked', revoked_at = ? WHERE id = ?", (db.ts(now), invitation_id))
    return ids


# --- setup codes and break-glass --------------------------------------------------------------


def setup_code_from_row(row: sqlite3.Row) -> SetupCodeRecord:
    return SetupCodeRecord(
        id=row["id"],
        purpose=SetupCodePurpose(row["purpose"]),
        email=row["email"],
        display_name=row["display_name"],
        target_account_id=row["target_account_id"],
        code_hash=row["code_hash"],
        issued_at=db.parse_ts(row["issued_at"]),
        expires_at=db.parse_ts(row["expires_at"]),
        used_at=db.parse_ts(row["used_at"]),
        enrolled_account_id=row["enrolled_account_id"],
        audit_event_id=row["audit_event_id"],
    )


def break_glass_from_row(row: sqlite3.Row) -> BreakGlassRecord:
    """``audit_event_id`` is the ``break_glass_used`` event once redeemed, else ``setup_code_issued``."""
    return BreakGlassRecord(
        setup_code_id=row["id"],
        os_user=row["os_user"][:200] or "unknown",
        target_account_id=row["target_account_id"],
        issued_at=db.parse_ts(row["issued_at"]),
        used_at=db.parse_ts(row["used_at"]),
        enrolled_account_id=row["enrolled_account_id"],
        revoked_credential_count=int(row["revoked_credential_count"]),
        audit_event_id=row["redeemed_audit_event_id"] or row["audit_event_id"],
    )


def setup_code_row(conn, setup_code_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM auth_setup_codes WHERE id = ?", (setup_code_id,)).fetchone()


def setup_code_row_by_hash(conn, code_hash: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM auth_setup_codes WHERE code_hash = ?", (code_hash,)).fetchone()


# --- installations and service keys -----------------------------------------------------------


def installation_from_row(row: sqlite3.Row) -> ProcessInstallation:
    return ProcessInstallation(
        id=row["id"],
        label=row["label"],
        primary_host=bool(row["primary_host"]),
        created_at=db.parse_ts(row["created_at"]),
        created_by_account_id=row["created_by_account_id"],
        retired_at=db.parse_ts(row["retired_at"]),
    )


def installation_row(conn, installation_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM auth_installations WHERE id = ?", (installation_id,)).fetchone()


def active_installation_by_label(conn, label: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM auth_installations WHERE label = ? AND retired_at IS NULL ORDER BY created_at, id LIMIT 1", (label,)
    ).fetchone()


def active_primary_installation(conn, *, excluding: Optional[str] = None) -> Optional[sqlite3.Row]:
    sql = "SELECT * FROM auth_installations WHERE primary_host = 1 AND retired_at IS NULL"
    args: list = []
    if excluding is not None:
        sql += " AND id != ?"
        args.append(excluding)
    return conn.execute(sql + " ORDER BY created_at, id LIMIT 1", args).fetchone()


def insert_installation(conn, *, installation_id: str, label: str, primary_host: bool, created_by_account_id: Optional[str], now: datetime) -> None:
    conn.execute(
        "INSERT INTO auth_installations (id, label, primary_host, created_at, created_by_account_id) VALUES (?, ?, ?, ?, ?)",
        (installation_id, label, int(bool(primary_host)), db.ts(now), created_by_account_id),
    )


def service_key_from_row(row: sqlite3.Row) -> ServiceKeyRecord:
    return ServiceKeyRecord(
        id=row["id"],
        installation_id=row["installation_id"],
        label=row["label"],
        scopes=[ServiceScope(s) for s in db.loads(row["scopes_json"])],
        key_prefix=row["key_prefix"],
        key_hash=row["key_hash"],
        created_at=db.parse_ts(row["created_at"]),
        created_by_account_id=row["created_by_account_id"],
        expires_at=db.parse_ts(row["expires_at"]),
        revoked_at=db.parse_ts(row["revoked_at"]),
        rotated_from_key_id=row["rotated_from_key_id"],
        superseded_by_key_id=row["superseded_by_key_id"],
        grace_until=db.parse_ts(row["grace_until"]),
        last_used_at=db.parse_ts(row["last_used_at"]),
    )


def service_key_row(conn, key_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM auth_service_keys WHERE id = ?", (key_id,)).fetchone()


def insert_service_key(conn, *, key_id: str, installation_id: str, label: str, scopes: Sequence[ServiceScope], key_prefix: str,
                       key_hash: str, created_by_account_id: Optional[str], expires_at: Optional[datetime],
                       rotated_from_key_id: Optional[str], now: datetime) -> None:
    ordered = [s.value for s in ServiceScope if s in {ServiceScope(x) for x in scopes}]
    conn.execute(
        "INSERT INTO auth_service_keys (id, installation_id, label, scopes_json, key_prefix, key_hash, created_at, created_by_account_id, "
        "expires_at, rotated_from_key_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (key_id, installation_id, label, db.dumps(ordered), key_prefix, key_hash, db.ts(now), created_by_account_id,
         db.ts(expires_at) if expires_at else None, rotated_from_key_id),
    )


def key_usable(row: sqlite3.Row, now: datetime) -> bool:
    """A key authenticates while it is not revoked, not expired and inside any rotation grace."""
    if row["revoked_at"] is not None and db.parse_ts(row["revoked_at"]) <= now:
        return False
    if row["expires_at"] is not None and db.parse_ts(row["expires_at"]) <= now:
        return False
    if row["grace_until"] is not None and db.parse_ts(row["grace_until"]) <= now:
        return False
    return True
