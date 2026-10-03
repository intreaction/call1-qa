"""Admin identity: accounts, invitations, setup-code and break-glass listings, installations and
service keys. Every write is audited in its own transaction and names the individual admin.

Rules beyond the route table:

* The last active admin who can still sign in cannot be demoted, disabled or re-invited
  (409 ``conflict``, ``reason: last_admin``); break-glass on the Store host is the recovery path.
* Disabling an account revokes all its sessions and its pending re-invitations in the same
  transaction. A disabled account with no authenticator left cannot be re-enabled (409); re-invite it.
* A re-invite revokes every credential and session of the account at issue and sets it
  ``reinvite_required``; the account keeps its ID, history and user handle.
* Revoking an account's last authenticator makes an active account ``reinvite_required`` and ends
  its sessions; revoking any authenticator ends the sessions it signed in.
* ``smtp_relay`` delivery needs the customer's relay in admin state, which Stage 2 does not have:
  409 ``conflict`` (``reason: smtp_relay_not_configured``). ``out_of_band`` returns the link once.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import Depends, Query

from call1.contracts.auth import (
    AccountListQuery,
    AccountStatus,
    AccountUpdate,
    BreakGlassRecord,
    InstallationCreate,
    InstallationRetire,
    Invitation,
    InvitationCreate,
    InvitationDelivery,
    InvitationIssued,
    InvitationListQuery,
    InvitationStatus,
    ProcessInstallation,
    ReviewerAccount,
    ServiceKeyCreate,
    ServiceKeyIssued,
    ServiceKeyRecord,
    ServiceKeyRevoke,
    ServiceKeyRotate,
    SetupCodeRecord,
    WebAuthnCredentialRecord,
)
from call1.contracts.common import Page
from call1.contracts.errors import ErrorCode
from call1.contracts.events import AuditAction

from .. import audit, db, pagination
from ..context import Store
from ..db import StoreConnection
from ..deps import current_principal, get_conn, get_store
from ..errors import StoreError
from ..ids import new_id
from ..principals import Principal, generate_secret, hash_secret, require_session
from . import identity, records
from .routes import router


def _account_or_404(conn, account_id: str):
    row = records.account_row(conn, account_id)
    if row is None:
        raise StoreError(ErrorCode.NOT_FOUND, "Account not found", details={"account_id": account_id})
    return row


# --- accounts ---------------------------------------------------------------------------------


@router.operation("listAccounts")
def list_accounts(query: Annotated[AccountListQuery, Query()], conn: StoreConnection = Depends(get_conn)) -> Page[ReviewerAccount]:
    after = pagination.decode(query.page_token, 1)
    with db.read_snapshot(conn):
        items = records.list_accounts(conn, role=query.role, status=query.status, after_id=after[0] if after else None, limit=query.limit + 1)
    token = pagination.encode(items[query.limit - 1].id) if len(items) > query.limit else None
    return Page[ReviewerAccount](items=items[: query.limit], next_page_token=token)


@router.operation("getAccount")
def get_account(account_id: str, conn: StoreConnection = Depends(get_conn)) -> ReviewerAccount:
    with db.read_snapshot(conn):
        return records.account_from_row(_account_or_404(conn, account_id))


@router.operation("updateAccount")
def update_account(account_id: str, body: AccountUpdate, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                   principal: Principal = Depends(current_principal)) -> ReviewerAccount:
    admin = require_session(principal)
    now = store.clock.now()
    with db.transaction(conn):
        row = _account_or_404(conn, account_id)
        changes, details = {}, {"reason": body.reason}
        if body.display_name is not None and body.display_name != row["display_name"]:
            changes["display_name"] = body.display_name
            details["display_name_changed"] = True
        role_change = body.role is not None and body.role.value != row["role"]
        disabling = body.status == AccountStatus.DISABLED.value and row["status"] != AccountStatus.DISABLED.value
        if role_change or disabling:
            identity.ensure_admin_remains(conn, account_id)
        if role_change:
            changes["role"] = body.role
            details.update(previous_role=row["role"], role=body.role.value)
        if disabling:
            changes["status"] = AccountStatus.DISABLED
            details["revoked_session_count"] = records.revoke_sessions(conn, account_id, now, "account disabled")
            details["revoked_invitation_count"] = len(records.revoke_pending_invitations(conn, now, reinvite_of_account_id=account_id))
        elif body.status == AccountStatus.ACTIVE.value and row["status"] != AccountStatus.ACTIVE.value:
            if row["status"] != AccountStatus.DISABLED.value:
                raise StoreError(ErrorCode.CONFLICT, "This account must enroll through its invitation first",
                                 details={"reason": row["status"]})
            if not records.live_credential_rows(conn, account_id):
                raise StoreError(ErrorCode.CONFLICT, "The account has no authenticator left; re-invite it instead",
                                 details={"reason": "no_authenticators"})
            changes["status"] = AccountStatus.ACTIVE
        if changes:
            records.update_account(conn, account_id, now, **changes)
            if "status" in changes:
                details["status"] = changes["status"].value
            action = AuditAction.ACCOUNT_DISABLED if disabling else AuditAction.ACCOUNT_UPDATED
            audit.append(conn, actor=audit.actor_for(admin), action=action, target_kind="account", target_id=account_id, details=details)
        return records.get_account(conn, account_id)


@router.operation("listAccountAuthenticators")
def list_account_authenticators(account_id: str, conn: StoreConnection = Depends(get_conn)) -> Page[WebAuthnCredentialRecord]:
    with db.read_snapshot(conn):
        _account_or_404(conn, account_id)
        rows = records.live_credential_rows(conn, account_id)
    return Page[WebAuthnCredentialRecord](items=[records.credential_from_row(r) for r in rows])


@router.operation("revokeAccountAuthenticator")
def revoke_account_authenticator(account_id: str, authenticator_id: str, store: Store = Depends(get_store),
                                 conn: StoreConnection = Depends(get_conn), principal: Principal = Depends(current_principal)) -> None:
    admin = require_session(principal)
    now = store.clock.now()
    with db.transaction(conn):
        row = _account_or_404(conn, account_id)
        if records.live_credential_row(conn, account_id, authenticator_id) is None:
            raise StoreError(ErrorCode.NOT_FOUND, "Authenticator not found", details={"authenticator_id": authenticator_id})
        records.revoke_credentials(conn, account_id, now, "revoked by admin", authenticator_id=authenticator_id)
        ended = records.revoke_sessions(conn, account_id, now, "authenticator revoked", authenticator_id=authenticator_id)
        remaining = len(records.live_credential_rows(conn, account_id))
        details = {"account_id": account_id, "remaining_authenticators": remaining}
        if remaining == 0 and row["status"] == AccountStatus.ACTIVE.value:
            records.update_account(conn, account_id, now, status=AccountStatus.REINVITE_REQUIRED)
            ended += records.revoke_sessions(conn, account_id, now, "no authenticators left")
            details["status"] = AccountStatus.REINVITE_REQUIRED.value
        details["revoked_session_count"] = ended
        audit.append(conn, actor=audit.actor_for(admin), action=AuditAction.AUTHENTICATOR_REMOVED, target_kind="authenticator",
                     target_id=authenticator_id, details=details)


@router.operation("revokeAccountSessions")
def revoke_account_sessions(account_id: str, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                            principal: Principal = Depends(current_principal)) -> None:
    admin = require_session(principal)
    with db.transaction(conn):
        _account_or_404(conn, account_id)
        ended = records.revoke_sessions(conn, account_id, store.clock.now(), "revoked by admin")
        audit.append(conn, actor=audit.actor_for(admin), action=AuditAction.SESSION_REVOKED, target_kind="account", target_id=account_id,
                     details={"revoked_session_count": ended})


# --- invitations ------------------------------------------------------------------------------


@router.operation("createInvitation")
def create_invitation(body: InvitationCreate, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                      principal: Principal = Depends(current_principal)) -> InvitationIssued:
    admin = require_session(principal)
    if body.delivery is InvitationDelivery.SMTP_RELAY:
        raise StoreError(ErrorCode.CONFLICT, "No SMTP relay is configured (admin state is not in Stage 2); use out_of_band delivery",
                         details={"reason": "smtp_relay_not_configured"})
    now = store.clock.now()
    token = generate_secret(32)
    invitation_id = new_id("inv")
    with db.transaction(conn):
        details = {"role": body.role.value, "delivery": body.delivery.value}
        account_id = None
        if body.reinvite_of_account_id is not None:
            target = records.account_row(conn, body.reinvite_of_account_id)
            if target is None:
                raise StoreError(ErrorCode.VALIDATION_FAILED, "reinvite_of_account_id names no account",
                                 details={"field": "reinvite_of_account_id", "reason": "account_not_found"})
            if target["email_key"] != records.email_key(body.email):
                raise StoreError(ErrorCode.VALIDATION_FAILED, "A re-invite goes to the account's own email",
                                 details={"field": "email", "reason": "reinvite_email_mismatch"})
            identity.ensure_admin_remains(conn, target["id"])
            account_id = target["id"]
            records.revoke_pending_invitations(conn, now, email=body.email)
            details["revoked_credential_count"] = records.revoke_credentials(conn, account_id, now, "re-invited")
            details["revoked_session_count"] = records.revoke_sessions(conn, account_id, now, "re-invited")
            details["account_id"] = account_id
            records.update_account(conn, account_id, now, status=AccountStatus.REINVITE_REQUIRED)
            action = AuditAction.REINVITE_ISSUED
        else:
            existing = records.account_row_by_email(conn, body.email)
            if existing is not None:
                raise StoreError(ErrorCode.CONFLICT, "An account with this email exists; re-invite it instead",
                                 details={"reason": "account_exists", "account_id": existing["id"]})
            pending = conn.execute("SELECT id FROM auth_invitations WHERE email_key = ? AND status = 'pending' AND expires_at > ?",
                                   (records.email_key(body.email), db.ts(now))).fetchone()
            if pending is not None:
                raise StoreError(ErrorCode.CONFLICT, "A pending invitation exists for this email; revoke it first",
                                 details={"reason": "invitation_pending", "invitation_id": pending["id"]})
            action = AuditAction.INVITATION_ISSUED
        expires = now + timedelta(seconds=store.config.parameters.invitation_lifetime_seconds)
        conn.execute(
            "INSERT INTO auth_invitations (id, email, email_key, display_name, role, status, delivery, token_hash, issued_by_account_id, "
            "issued_at, expires_at, account_id, reinvite_of_account_id) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)",
            (invitation_id, body.email.strip(), records.email_key(body.email), body.display_name, body.role.value, body.delivery.value,
             hash_secret(token), admin.account_id, db.ts(now), db.ts(expires), account_id, body.reinvite_of_account_id),
        )
        audit.append(conn, actor=audit.actor_for(admin), action=action, target_kind="invitation", target_id=invitation_id, details=details)
        invitation = records.invitation_from_row(records.invitation_row(conn, invitation_id), now)
    return InvitationIssued(invitation=invitation, invitation_url=f"{store.config.public_base_url}/enroll#{token}")


@router.operation("listInvitations")
def list_invitations(query: Annotated[InvitationListQuery, Query()], store: Store = Depends(get_store),
                     conn: StoreConnection = Depends(get_conn)) -> Page[Invitation]:
    now = store.clock.now()
    where, args = [], []
    before = pagination.decode(query.page_token, 1)
    if before is not None:
        where.append("id < ?")
        args.append(before[0])
    if query.status is InvitationStatus.PENDING:
        where.append("status = 'pending' AND expires_at > ?")
        args.append(db.ts(now))
    elif query.status is InvitationStatus.EXPIRED:
        where.append("status = 'pending' AND expires_at <= ?")
        args.append(db.ts(now))
    elif query.status is not None:
        where.append("status = ?")
        args.append(query.status.value)
    sql = "SELECT * FROM auth_invitations" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ?"
    with db.read_snapshot(conn):
        rows = conn.execute(sql, (*args, query.limit + 1)).fetchall()
    items = [records.invitation_from_row(r, now) for r in rows[: query.limit]]
    token = pagination.encode(items[-1].id) if len(rows) > query.limit else None
    return Page[Invitation](items=items, next_page_token=token)


@router.operation("revokeInvitation")
def revoke_invitation(invitation_id: str, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                      principal: Principal = Depends(current_principal)) -> None:
    admin = require_session(principal)
    now = store.clock.now()
    with db.transaction(conn):
        row = records.invitation_row(conn, invitation_id)
        if row is None:
            raise StoreError(ErrorCode.NOT_FOUND, "Invitation not found", details={"invitation_id": invitation_id})
        status = records.invitation_status(row, now)
        if status is not InvitationStatus.PENDING:
            raise StoreError(ErrorCode.INVITATION_INVALID, "Only a pending invitation can be revoked", details={"status": status.value})
        conn.execute("UPDATE auth_invitations SET status = 'revoked', revoked_at = ? WHERE id = ?", (db.ts(now), invitation_id))
        audit.append(conn, actor=audit.actor_for(admin), action=AuditAction.INVITATION_REVOKED, target_kind="invitation", target_id=invitation_id,
                     details={"reinvite_of_account_id": row["reinvite_of_account_id"]})


# --- setup codes and break-glass (issued only on the Store host) ------------------------------


@router.operation("listSetupCodes")
def list_setup_codes(conn: StoreConnection = Depends(get_conn)) -> Page[SetupCodeRecord]:
    with db.read_snapshot(conn):
        rows = conn.execute("SELECT * FROM auth_setup_codes ORDER BY issued_at DESC, id DESC").fetchall()
    return Page[SetupCodeRecord](items=[records.setup_code_from_row(r) for r in rows])


@router.operation("listBreakGlass")
def list_break_glass(conn: StoreConnection = Depends(get_conn)) -> Page[BreakGlassRecord]:
    with db.read_snapshot(conn):
        rows = conn.execute("SELECT * FROM auth_setup_codes WHERE purpose = 'break_glass' ORDER BY issued_at DESC, id DESC").fetchall()
    return Page[BreakGlassRecord](items=[records.break_glass_from_row(r) for r in rows])


# --- installations and service keys ----------------------------------------------------------


@router.operation("registerInstallation")
def register_installation(body: InstallationCreate, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                          principal: Principal = Depends(current_principal)) -> ProcessInstallation:
    admin = require_session(principal)
    with db.transaction(conn):
        return identity.register_installation(conn, label=body.label, primary_host=body.primary_host, actor=audit.actor_for(admin),
                                              created_by_account_id=admin.account_id, now=store.clock.now())


@router.operation("listInstallations")
def list_installations(conn: StoreConnection = Depends(get_conn)) -> Page[ProcessInstallation]:
    with db.read_snapshot(conn):
        rows = conn.execute("SELECT * FROM auth_installations ORDER BY created_at, id").fetchall()
    return Page[ProcessInstallation](items=[records.installation_from_row(r) for r in rows])


@router.operation("retireInstallation")
def retire_installation(installation_id: str, body: InstallationRetire, store: Store = Depends(get_store),
                        conn: StoreConnection = Depends(get_conn), principal: Principal = Depends(current_principal)) -> ProcessInstallation:
    admin = require_session(principal)
    with db.transaction(conn):
        return identity.retire_installation(conn, installation_id, reason=body.reason, actor=audit.actor_for(admin), now=store.clock.now())


@router.operation("createServiceKey")
def create_service_key(body: ServiceKeyCreate, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                       principal: Principal = Depends(current_principal)) -> ServiceKeyIssued:
    admin = require_session(principal)
    with db.transaction(conn):
        return identity.issue_key(conn, installation_id=body.installation_id, label=body.label, scopes=body.scopes, expires_at=body.expires_at,
                                  actor=audit.actor_for(admin), created_by_account_id=admin.account_id, now=store.clock.now())


@router.operation("listServiceKeys")
def list_service_keys(conn: StoreConnection = Depends(get_conn)) -> Page[ServiceKeyRecord]:
    with db.read_snapshot(conn):
        rows = conn.execute("SELECT * FROM auth_service_keys ORDER BY created_at, id").fetchall()
    return Page[ServiceKeyRecord](items=[records.service_key_from_row(r) for r in rows])


@router.operation("rotateServiceKey")
def rotate_service_key(key_id: str, body: ServiceKeyRotate, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                       principal: Principal = Depends(current_principal)) -> ServiceKeyIssued:
    admin = require_session(principal)
    grace = body.grace_seconds if body.grace_seconds is not None else store.config.parameters.service_key_rotation_grace_seconds
    with db.transaction(conn):
        return identity.rotate_key(conn, key_id, grace_seconds=grace, actor=audit.actor_for(admin), created_by_account_id=admin.account_id,
                                   now=store.clock.now())


@router.operation("revokeServiceKey")
def revoke_service_key(key_id: str, body: ServiceKeyRevoke, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                       principal: Principal = Depends(current_principal)) -> ServiceKeyRecord:
    admin = require_session(principal)
    with db.transaction(conn):
        return identity.revoke_key(conn, key_id, reason=body.reason, actor=audit.actor_for(admin), now=store.clock.now())

