"""Identity operations shared by the admin routes and the host commands.

Each function runs in the caller's open transaction and writes its audit event there, so a change
and its audit record commit together or not at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional, Sequence, Tuple

from call1.contracts.auth import (
    ProcessInstallation,
    ServiceKeyIssued,
    SetupCodeIssueRequest,
    SetupCodePurpose,
    SetupCodeRecord,
)
from call1.contracts.common import ServiceScope
from call1.contracts.errors import ErrorCode
from call1.contracts.events import Actor, AuditAction

from .. import audit, db
from ..errors import StoreError
from ..ids import new_id
from ..principals import generate_service_key_token, hash_secret
from . import crypto, records


def require_open(conn) -> None:
    if not conn.in_transaction:
        raise RuntimeError("identity operations run inside the caller's transaction")


# --- admins -----------------------------------------------------------------------------------


def ensure_admin_remains(conn, account_id: str) -> None:
    """409 ``conflict`` when ``account_id`` is the last active admin who can still sign in."""
    row = records.account_row(conn, account_id)
    if row is None or row["role"] != "admin" or row["status"] != "active":
        return
    if not records.active_admin_ids(conn, excluding=[account_id]):
        raise StoreError(ErrorCode.CONFLICT, "This is the last active administrator; enroll another admin first",
                         details={"reason": "last_admin"})


# --- setup codes ------------------------------------------------------------------------------


def issue_setup_code(conn, parameters, request: SetupCodeIssueRequest, *, now: datetime) -> Tuple[str, SetupCodeRecord]:
    """Issue a single-use first-admin or break-glass code. ``first_admin`` is refused (409) while an
    active admin exists or when the email already has an account; ``break_glass`` with a target
    needs that account and its email. Writes ``setup_code_issued``."""
    require_open(conn)
    existing = records.account_row_by_email(conn, request.email)
    if request.purpose is SetupCodePurpose.FIRST_ADMIN:
        if records.any_active_admin(conn):
            raise StoreError(ErrorCode.CONFLICT, "An active administrator exists; use --purpose break_glass for recovery",
                             details={"reason": "active_admin_exists"})
        if existing is not None:
            raise StoreError(ErrorCode.CONFLICT, "An account with this email exists; use --purpose break_glass --target-account-id",
                             details={"reason": "account_exists", "account_id": existing["id"]})
    elif request.target_account_id is not None:
        target = records.account_row(conn, request.target_account_id)
        if target is None:
            raise StoreError(ErrorCode.NOT_FOUND, "No account with that ID", details={"target_account_id": request.target_account_id})
        if target["email_key"] != records.email_key(request.email):
            raise StoreError(ErrorCode.VALIDATION_FAILED, "The email must match the target account's email",
                             details={"field": "email", "reason": "target_email_mismatch"})
    elif existing is not None:
        raise StoreError(ErrorCode.CONFLICT, "An account with this email exists; name it with --target-account-id",
                         details={"reason": "account_exists", "account_id": existing["id"]})
    code = crypto.new_setup_code()
    code_id = new_id("sc")
    actor = audit.installer_actor(request.os_user) if request.purpose is SetupCodePurpose.FIRST_ADMIN else audit.break_glass_actor(request.os_user)
    event = audit.append(conn, actor=actor, action=AuditAction.SETUP_CODE_ISSUED, target_kind="setup_code", target_id=code_id,
                         details={"purpose": request.purpose.value, "target_account_id": request.target_account_id})
    expires = now + timedelta(seconds=parameters.setup_code_lifetime_seconds)
    conn.execute(
        "INSERT INTO auth_setup_codes (id, purpose, email, display_name, target_account_id, code_hash, os_user, issued_at, expires_at, "
        "audit_event_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (code_id, request.purpose.value, request.email.strip(), request.display_name, request.target_account_id,
         hash_secret(crypto.normalize_setup_code(code)), request.os_user, db.ts(now), db.ts(expires), event.id),
    )
    return code, records.setup_code_from_row(records.setup_code_row(conn, code_id))


# --- installations ----------------------------------------------------------------------------


def register_installation(conn, *, label: str, primary_host: bool, actor: Actor, created_by_account_id: Optional[str],
                          now: datetime) -> ProcessInstallation:
    """Register an installation. 409 when an active one has the label, or when ``primary_host``
    is asked while another active installation holds it. Writes ``installation_registered``."""
    require_open(conn)
    if records.active_installation_by_label(conn, label) is not None:
        raise StoreError(ErrorCode.CONFLICT, "An active installation already has this label", details={"reason": "label_in_use"})
    if primary_host:
        current = records.active_primary_installation(conn)
        if current is not None:
            raise StoreError(ErrorCode.CONFLICT, "Another active installation is the primary host; retire it first",
                             details={"reason": "primary_host_taken", "primary_installation_id": current["id"]})
    installation_id = new_id("inst")
    records.insert_installation(conn, installation_id=installation_id, label=label, primary_host=primary_host,
                                created_by_account_id=created_by_account_id, now=now)
    audit.append(conn, actor=actor, action=AuditAction.INSTALLATION_REGISTERED, target_kind="installation", target_id=installation_id,
                 details={"primary_host": bool(primary_host)})
    return records.installation_from_row(records.installation_row(conn, installation_id))


def retire_installation(conn, installation_id: str, *, reason: str, actor: Actor, now: datetime) -> ProcessInstallation:
    require_open(conn)
    row = records.installation_row(conn, installation_id)
    if row is None:
        raise StoreError(ErrorCode.NOT_FOUND, "Installation not found", details={"installation_id": installation_id})
    if row["retired_at"] is not None:
        raise StoreError(ErrorCode.INVALID_TRANSITION, "The installation is already retired", details={"status": "retired"})
    conn.execute("UPDATE auth_installations SET retired_at = ?, retire_reason = ? WHERE id = ?", (db.ts(now), reason, installation_id))
    revoked = conn.execute(
        "UPDATE auth_service_keys SET revoked_at = ?, revoke_reason = ? WHERE installation_id = ? AND revoked_at IS NULL",
        (db.ts(now), "installation retired", installation_id),
    ).rowcount
    audit.append(conn, actor=actor, action=AuditAction.INSTALLATION_RETIRED, target_kind="installation", target_id=installation_id,
                 details={"revoked_key_count": revoked, "reason": reason})
    return records.installation_from_row(records.installation_row(conn, installation_id))


# --- service keys -----------------------------------------------------------------------------


def issue_key(conn, *, installation_id: str, label: str, scopes: Sequence[ServiceScope], expires_at: Optional[datetime], actor: Actor,
              created_by_account_id: Optional[str], now: datetime, rotated_from_key_id: Optional[str] = None,
              action: AuditAction = AuditAction.SERVICE_KEY_ISSUED) -> ServiceKeyIssued:
    """Issue a key to a registered, non-retired installation (404 otherwise). The token is in the
    result exactly once; Store keeps ``sha256(token)``."""
    require_open(conn)
    installation = records.installation_row(conn, installation_id)
    if installation is None or installation["retired_at"] is not None:
        raise StoreError(ErrorCode.NOT_FOUND, "No registered, active installation with that ID", details={"installation_id": installation_id})
    if expires_at is not None and expires_at <= now:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "expires_at must be in the future", details={"field": "expires_at"})
    scope_list = [ServiceScope(s) for s in scopes]
    if not scope_list:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "A key needs at least one scope", details={"field": "scopes"})
    token, prefix = generate_service_key_token()
    key_id = new_id("key")
    records.insert_service_key(conn, key_id=key_id, installation_id=installation_id, label=label, scopes=scope_list, key_prefix=prefix,
                               key_hash=hash_secret(token), created_by_account_id=created_by_account_id, expires_at=expires_at,
                               rotated_from_key_id=rotated_from_key_id, now=now)
    details = {"installation_id": installation_id, "key_prefix": prefix, "scopes": " ".join(s.value for s in scope_list)}
    if rotated_from_key_id is not None:
        details["rotated_from_key_id"] = rotated_from_key_id
    audit.append(conn, actor=actor, action=action, target_kind="service_key", target_id=key_id, details=details)
    return ServiceKeyIssued(key=records.service_key_from_row(records.service_key_row(conn, key_id)), token=token)


def rotate_key(conn, key_id: str, *, grace_seconds: int, actor: Actor, created_by_account_id: Optional[str], now: datetime) -> ServiceKeyIssued:
    """Replace a key with the same installation, scopes, label and expiry (a rotation never adds
    scopes or extends access). The old key works until ``grace_until``; 0 revokes it at once."""
    require_open(conn)
    old = records.service_key_row(conn, key_id)
    if old is None:
        raise StoreError(ErrorCode.NOT_FOUND, "Service key not found", details={"key_id": key_id})
    if old["superseded_by_key_id"] is not None:
        raise StoreError(ErrorCode.CONFLICT, "This key was already rotated", details={"reason": "already_rotated", "superseded_by_key_id": old["superseded_by_key_id"]})
    if not records.key_usable(old, now):
        raise StoreError(ErrorCode.CONFLICT, "Only a working key can be rotated", details={"reason": "key_not_active"})
    installation = records.installation_row(conn, old["installation_id"])
    if installation is None or installation["retired_at"] is not None:
        raise StoreError(ErrorCode.CONFLICT, "The key's installation is retired", details={"reason": "installation_retired"})
    issued = issue_key(conn, installation_id=old["installation_id"], label=old["label"], scopes=[ServiceScope(s) for s in db.loads(old["scopes_json"])],
                       expires_at=db.parse_ts(old["expires_at"]), actor=actor, created_by_account_id=created_by_account_id, now=now,
                       rotated_from_key_id=key_id, action=AuditAction.SERVICE_KEY_ROTATED)
    grace_until = now + timedelta(seconds=grace_seconds)
    conn.execute(
        "UPDATE auth_service_keys SET superseded_by_key_id = ?, grace_until = ?, revoked_at = ?, revoke_reason = ? WHERE id = ?",
        (issued.key.id, db.ts(grace_until), db.ts(now) if grace_seconds == 0 else None, "rotated" if grace_seconds == 0 else None, key_id),
    )
    return issued


def revoke_key(conn, key_id: str, *, reason: str, actor: Actor, now: datetime):
    require_open(conn)
    row = records.service_key_row(conn, key_id)
    if row is None:
        raise StoreError(ErrorCode.NOT_FOUND, "Service key not found", details={"key_id": key_id})
    if row["revoked_at"] is None:
        conn.execute("UPDATE auth_service_keys SET revoked_at = ?, revoke_reason = ? WHERE id = ?", (db.ts(now), reason, key_id))
        audit.append(conn, actor=actor, action=AuditAction.SERVICE_KEY_REVOKED, target_kind="service_key", target_id=key_id,
                     details={"installation_id": row["installation_id"], "key_prefix": row["key_prefix"], "reason": reason})
    return records.service_key_from_row(records.service_key_row(conn, key_id))


def installer_key(conn, *, installation_label: str, scopes: Iterable[ServiceScope], primary_host: Optional[bool], os_user: str,
                  now: datetime) -> ServiceKeyIssued:
    """The host command: register the installation if no active one has this label, then issue a
    key. ``primary_host`` None means primary when no active installation is (the one-computer
    install). An explicit True on an existing installation promotes it when no other is primary."""
    require_open(conn)
    actor = audit.installer_actor(os_user)
    row = records.active_installation_by_label(conn, installation_label)
    if row is None:
        wants_primary = primary_host if primary_host is not None else records.active_primary_installation(conn) is None
        installation = register_installation(conn, label=installation_label, primary_host=bool(wants_primary), actor=actor,
                                             created_by_account_id=None, now=now)
        installation_id = installation.id
    else:
        installation_id = row["id"]
        if primary_host and not row["primary_host"]:
            current = records.active_primary_installation(conn, excluding=installation_id)
            if current is not None:
                raise StoreError(ErrorCode.CONFLICT, "Another active installation is the primary host",
                                 details={"reason": "primary_host_taken", "primary_installation_id": current["id"]})
            conn.execute("UPDATE auth_installations SET primary_host = 1 WHERE id = ?", (installation_id,))
    return issue_key(conn, installation_id=installation_id, label=f"{installation_label[:180]} (installer)", scopes=list(scopes),
                     expires_at=None, actor=actor, created_by_account_id=None, now=now)
