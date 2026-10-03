"""Passkey ceremonies and the signed-in reviewer's own session, sessions and authenticators.

Enrollment (``/auth/enroll/*``) redeems exactly one invitation token or setup code; sign-in
(``/auth/sign-in/*``) is account-first with decoys for unknown accounts. Both finishes consume the
ceremony before verifying, create a server-side session and set the session cookie. Adding an
authenticator is a step-up ceremony: a fresh user-verified assertion from an existing credential,
then the new registration, both within the ceremony's lifetime.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import Depends, Request, Response

from call1.contracts.auth import (
    AddAuthenticatorBeginRequest,
    AddAuthenticatorBeginResponse,
    AddAuthenticatorFinishRequest,
    AccountStatus,
    AuthenticationBeginRequest,
    AuthenticationBeginResponse,
    AuthenticationFinishRequest,
    CredentialUpdate,
    EnrollmentBeginRequest,
    InvitationStatus,
    RegistrationBeginResponse,
    RegistrationFinishRequest,
    RegistrationFinishResponse,
    SessionInfo,
    SessionListItem,
    SetupCodePurpose,
    SignInResponse,
    SignOutResponse,
    WebAuthnCredentialRecord,
)
from call1.contracts.common import Page, ReviewerRole
from call1.contracts.errors import ErrorCode
from call1.contracts.events import Actor, ActorKind, AuditAction

from .. import audit, db
from ..context import Store
from ..db import StoreConnection
from ..deps import current_principal, get_conn, get_store
from ..errors import StoreError
from ..ids import new_id
from ..principals import Principal, clear_session_cookie, hash_secret, require_session, set_session_cookie
from . import crypto, passkeys, ratelimit, records, sessions
from .routes import router


def client_key(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def canonical_credential_id(raw_id: str) -> str:
    try:
        return crypto.b64url(crypto.b64url_decode(raw_id))
    except (ValueError, TypeError):
        raise passkeys.verification_failed("credential_id_malformed") from None


def _invitation_invalid(reason: str) -> StoreError:
    return StoreError(ErrorCode.INVITATION_INVALID, "This invitation is no longer valid; ask an administrator for a new one", details={"reason": reason})


def _setup_code_invalid(reason: str) -> StoreError:
    return StoreError(ErrorCode.SETUP_CODE_INVALID, "This setup code is no longer valid; issue a new one on the Store host", details={"reason": reason})


def _reviewer_actor(account_id: str, session_id: Optional[str], display_name: str) -> Actor:
    return Actor(kind=ActorKind.REVIEWER, account_id=account_id, session_id=session_id, display=display_name[:200])


def _start_session(conn, store: Store, response: Response, *, account_id: str, authenticator_id: str, now: datetime) -> SessionInfo:
    records.update_account(conn, account_id, now, last_sign_in_at=now)
    session_id, cookie, csrf = sessions.create_session(conn, store.config.parameters, account_id=account_id,
                                                       authenticator_id=authenticator_id, now=now)
    info = sessions.session_info(conn, session_id, csrf)
    set_session_cookie(response, store.config, cookie, max_age=store.config.parameters.session_absolute_lifetime_seconds)
    return info


# --- enrollment -------------------------------------------------------------------------------


def _check_invitation(conn, row, now: datetime) -> None:
    if row is None:
        raise _invitation_invalid("unknown")
    status = records.invitation_status(row, now)
    if status is not InvitationStatus.PENDING:
        raise _invitation_invalid(status.value)
    if row["reinvite_of_account_id"] is not None:
        target = records.account_row(conn, row["reinvite_of_account_id"])
        if target is None or target["status"] != AccountStatus.REINVITE_REQUIRED.value:
            raise _invitation_invalid("account_not_awaiting_reinvite")
    elif records.account_row_by_email(conn, row["email"]) is not None:
        raise _invitation_invalid("account_exists")


def _check_setup_code(conn, row, now: datetime) -> None:
    if row is None:
        raise _setup_code_invalid("unknown")
    if row["used_at"] is not None:
        raise _setup_code_invalid("used")
    if db.parse_ts(row["expires_at"]) <= now:
        raise _setup_code_invalid("expired")
    if row["target_account_id"] is not None:
        if records.account_row(conn, row["target_account_id"]) is None:
            raise _setup_code_invalid("target_account_missing")
        return
    if row["purpose"] == SetupCodePurpose.FIRST_ADMIN.value and records.any_active_admin(conn):
        raise _setup_code_invalid("active_admin_exists")
    if records.account_row_by_email(conn, row["email"]) is not None:
        raise _setup_code_invalid("account_exists")


@router.operation("enrollBegin")
def enroll_begin(body: EnrollmentBeginRequest, request: Request, store: Store = Depends(get_store),
                 conn: StoreConnection = Depends(get_conn)) -> RegistrationBeginResponse:
    ratelimit.limiter_for(store).check(ratelimit.ENROLL_PER_CLIENT, client_key(request))
    now = store.clock.now()
    with db.transaction(conn):
        existing = None
        if body.invitation_token:
            invitation = records.invitation_row_by_hash(conn, hash_secret(body.invitation_token))
            _check_invitation(conn, invitation, now)
            email, display_name = invitation["email"], invitation["display_name"]
            links = {"invitation_id": invitation["id"]}
            if invitation["reinvite_of_account_id"] is not None:
                existing = records.account_row(conn, invitation["reinvite_of_account_id"])
        else:
            code = records.setup_code_row_by_hash(conn, hash_secret(crypto.normalize_setup_code(body.setup_code or "")))
            _check_setup_code(conn, code, now)
            email, display_name = code["email"], code["display_name"]
            links = {"setup_code_id": code["id"]}
            if code["target_account_id"] is not None:
                existing = records.account_row(conn, code["target_account_id"])
        if existing is not None:
            ceremony = passkeys.begin(conn, store.config, kind="enroll", now=now, account_id=existing["id"], **links)
            user_handle = existing["user_handle"]
        else:
            user_handle = crypto.new_user_handle()
            ceremony = passkeys.begin(conn, store.config, kind="enroll", now=now, new_account_id=new_id("acct"),
                                      new_user_handle=user_handle, **links)
    options = passkeys.creation_options(store.config, challenge=ceremony["challenge"], user_handle=user_handle, email=email,
                                        display_name=display_name)
    return RegistrationBeginResponse(ceremony_id=ceremony["id"], options=options, expires_at=db.parse_ts(ceremony["expires_at"]))


def _redeem_invitation(conn, ceremony, now: datetime) -> tuple:
    """Returns (account_id, audit plan) after applying the invitation to its account."""
    invitation = records.invitation_row(conn, ceremony["invitation_id"])
    _check_invitation(conn, invitation, now)
    role = ReviewerRole(invitation["role"])
    if invitation["reinvite_of_account_id"] is not None:
        account_id = invitation["reinvite_of_account_id"]
        if ceremony["account_id"] != account_id:
            raise _invitation_invalid("ceremony_mismatch")
        records.revoke_credentials(conn, account_id, now, "re-enrolled by invitation")
        records.revoke_sessions(conn, account_id, now, "re-enrolled by invitation")
        records.update_account(conn, account_id, now, role=role, display_name=invitation["display_name"], status=AccountStatus.ACTIVE)
        created = False
    else:
        account_id = ceremony["new_account_id"]
        records.insert_account(conn, account_id=account_id, email=invitation["email"], display_name=invitation["display_name"], role=role,
                               status=AccountStatus.ACTIVE, user_handle=ceremony["new_user_handle"], now=now)
        created = True
    conn.execute("UPDATE auth_invitations SET status = 'redeemed', redeemed_at = ?, account_id = ? WHERE id = ? AND status = 'pending'",
                 (db.ts(now), account_id, invitation["id"]))
    return account_id, {"created": created, "invitation_id": invitation["id"], "role": role.value}


def _redeem_setup_code(conn, ceremony, now: datetime) -> tuple:
    code = records.setup_code_row(conn, ceremony["setup_code_id"])
    _check_setup_code(conn, code, now)
    revoked = 0
    if code["target_account_id"] is not None:
        account_id = code["target_account_id"]
        if ceremony["account_id"] != account_id:
            raise _setup_code_invalid("ceremony_mismatch")
        revoked = records.revoke_credentials(conn, account_id, now, "break-glass re-enrollment")
        records.revoke_sessions(conn, account_id, now, "break-glass re-enrollment")
        records.revoke_pending_invitations(conn, now, reinvite_of_account_id=account_id)
        records.update_account(conn, account_id, now, role=ReviewerRole.ADMIN, display_name=code["display_name"], status=AccountStatus.ACTIVE)
        created = False
    else:
        account_id = ceremony["new_account_id"]
        records.insert_account(conn, account_id=account_id, email=code["email"], display_name=code["display_name"], role=ReviewerRole.ADMIN,
                               status=AccountStatus.ACTIVE, user_handle=ceremony["new_user_handle"], now=now)
        created = True
    conn.execute("UPDATE auth_setup_codes SET used_at = ?, enrolled_account_id = ?, revoked_credential_count = ? WHERE id = ? AND used_at IS NULL",
                 (db.ts(now), account_id, revoked, code["id"]))
    return account_id, {"created": created, "setup_code_id": code["id"], "purpose": code["purpose"], "os_user": code["os_user"],
                        "revoked_credential_count": revoked}


@router.operation("enrollFinish")
def enroll_finish(body: RegistrationFinishRequest, request: Request, response: Response, store: Store = Depends(get_store),
                  conn: StoreConnection = Depends(get_conn)) -> RegistrationFinishResponse:
    config = store.config
    passkeys.check_request_origin(config, request.headers.get("origin"))
    now = store.clock.now()
    ceremony = passkeys.consume(conn, body.ceremony_id, kind="enroll", now=now)
    verified = passkeys.verify_registration(config, body.credential, challenge=ceremony["challenge"])
    with db.transaction(conn):
        if records.credential_row_by_credential_id(conn, verified.credential_id) is not None:
            raise passkeys.verification_failed("credential_already_registered")
        if ceremony["invitation_id"] is not None:
            account_id, plan = _redeem_invitation(conn, ceremony, now)
        else:
            account_id, plan = _redeem_setup_code(conn, ceremony, now)
        authenticator_id = new_id("cred")
        records.insert_credential(conn, authenticator_id=authenticator_id, account_id=account_id, credential_id=verified.credential_id,
                                  public_key_cose=verified.public_key_cose, sign_count=verified.sign_count, transports=verified.transports,
                                  aaguid=verified.aaguid, nickname=body.nickname, backup_eligible=verified.backup_eligible,
                                  backup_state=verified.backup_state, now=now)
        info = _start_session(conn, store, response, account_id=account_id, authenticator_id=authenticator_id, now=now)
        if plan.get("purpose") == SetupCodePurpose.BREAK_GLASS.value:
            actor = Actor(kind=ActorKind.BREAK_GLASS, account_id=account_id, session_id=info.session_id, display=(plan["os_user"] or "break-glass")[:200])
        else:
            actor = _reviewer_actor(account_id, info.session_id, info.display_name)
        if plan["created"]:
            audit.append(conn, actor=actor, action=AuditAction.ACCOUNT_CREATED, target_kind="account", target_id=account_id,
                         details={"role": info.role.value, "via": "invitation" if "invitation_id" in plan else plan["purpose"]})
        if "invitation_id" in plan:
            audit.append(conn, actor=actor, action=AuditAction.INVITATION_REDEEMED, target_kind="invitation", target_id=plan["invitation_id"],
                         details={"account_id": account_id, "role": plan["role"]})
        else:
            action = AuditAction.BREAK_GLASS_USED if plan["purpose"] == SetupCodePurpose.BREAK_GLASS.value else AuditAction.SETUP_CODE_REDEEMED
            event = audit.append(conn, actor=actor, action=action, target_kind="setup_code", target_id=plan["setup_code_id"],
                                 details={"account_id": account_id, "purpose": plan["purpose"],
                                          "revoked_credential_count": plan["revoked_credential_count"]})
            conn.execute("UPDATE auth_setup_codes SET redeemed_audit_event_id = ? WHERE id = ?", (event.id, plan["setup_code_id"]))
        audit.append(conn, actor=actor, action=AuditAction.AUTHENTICATOR_ADDED, target_kind="authenticator", target_id=authenticator_id,
                     details={"account_id": account_id, "enrollment": True})
        account = records.get_account(conn, account_id)
        credential = records.credential_from_row(records.live_credential_row(conn, account_id, authenticator_id))
    return RegistrationFinishResponse(account=account, credential=credential, signed_in=SignInResponse(session=info))


# --- sign-in ----------------------------------------------------------------------------------


@router.operation("signInBegin")
def sign_in_begin(body: AuthenticationBeginRequest, request: Request, store: Store = Depends(get_store),
                  conn: StoreConnection = Depends(get_conn)) -> AuthenticationBeginResponse:
    key = records.email_key(body.email)
    limiter = ratelimit.limiter_for(store)
    limiter.check(ratelimit.SIGN_IN_PER_CLIENT, client_key(request))
    limiter.check(ratelimit.SIGN_IN_PER_EMAIL, ratelimit.email_bucket(key))
    now = store.clock.now()
    with db.transaction(conn):
        decoys = crypto.decoy_descriptors(conn, key)  # computed on every path, so timing does not depend on the account
        account = records.account_row_by_email(conn, body.email)
        allow, bound = decoys, None
        if account is not None and account["status"] == AccountStatus.ACTIVE.value:
            live = records.live_credential_rows(conn, account["id"])
            if live:
                allow, bound = [passkeys.descriptor(row) for row in live], account["id"]
        ceremony = passkeys.begin(conn, store.config, kind="sign_in", now=now, account_id=bound)
    options = passkeys.request_options(store.config, challenge=ceremony["challenge"], allow=allow)
    return AuthenticationBeginResponse(ceremony_id=ceremony["id"], options=options, expires_at=db.parse_ts(ceremony["expires_at"]))


@router.operation("signInFinish")
def sign_in_finish(body: AuthenticationFinishRequest, request: Request, response: Response, store: Store = Depends(get_store),
                   conn: StoreConnection = Depends(get_conn)) -> SignInResponse:
    config = store.config
    passkeys.check_request_origin(config, request.headers.get("origin"))
    now = store.clock.now()
    ceremony = passkeys.consume(conn, body.ceremony_id, kind="sign_in", now=now)
    credential_id = canonical_credential_id(body.credential.rawId)
    if ceremony["account_id"] is None:
        passkeys.check_client_data(config, body.credential.response.clientDataJSON, expected_type="webauthn.get")
        raise passkeys.verification_failed("credential_not_registered")
    with db.transaction(conn):
        account = records.account_row(conn, ceremony["account_id"])
        if account is None:
            raise passkeys.verification_failed("credential_not_registered")
        if account["status"] == AccountStatus.DISABLED.value:
            raise StoreError(ErrorCode.ACCOUNT_DISABLED, "This account is disabled; ask an administrator")
        stored = records.credential_row_by_credential_id(conn, credential_id)
        if account["status"] != AccountStatus.ACTIVE.value or stored is None or stored["account_id"] != account["id"]:
            raise passkeys.verification_failed("credential_not_registered")
        assertion = passkeys.verify_assertion(config, body.credential, challenge=ceremony["challenge"], stored=stored,
                                              user_handle=account["user_handle"])
        passkeys.record_use(conn, stored, assertion, now)
        info = _start_session(conn, store, response, account_id=account["id"], authenticator_id=stored["id"], now=now)
    return SignInResponse(session=info)


# --- the current session ----------------------------------------------------------------------


@router.operation("getSession")
def get_session(request: Request, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                principal: Principal = Depends(current_principal)) -> SessionInfo:
    session = require_session(principal)
    cookie = request.cookies.get(store.config.cookie_name) or ""
    with db.read_snapshot(conn):
        return sessions.session_info(conn, session.session_id, crypto.csrf_token_for(conn, cookie))


@router.operation("signOut")
def sign_out(response: Response, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
             principal: Principal = Depends(current_principal)) -> SignOutResponse:
    session = require_session(principal)
    with db.transaction(conn):
        records.revoke_sessions(conn, session.account_id, store.clock.now(), "signed out", session_id=session.session_id)
    clear_session_cookie(response, store.config)
    return SignOutResponse()


# --- add another authenticator (step-up) ------------------------------------------------------


@router.operation("addAuthenticatorBegin")
def add_authenticator_begin(body: AddAuthenticatorBeginRequest, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                            principal: Principal = Depends(current_principal)) -> AddAuthenticatorBeginResponse:
    session = require_session(principal)
    now = store.clock.now()
    with db.transaction(conn):
        account = records.account_row(conn, session.account_id)
        live = records.live_credential_rows(conn, session.account_id)
        if account is None or not live:
            raise StoreError(ErrorCode.FORBIDDEN, "No existing authenticator to confirm with; ask an administrator to re-invite you")
        ceremony = passkeys.begin(conn, store.config, kind="add_authenticator", now=now, account_id=session.account_id,
                                  session_id=session.session_id, nickname=body.nickname, step_up=True)
    existing = [passkeys.descriptor(row) for row in live]
    return AddAuthenticatorBeginResponse(
        ceremony_id=ceremony["id"],
        reauthentication=passkeys.request_options(store.config, challenge=ceremony["step_up_challenge"], allow=existing),
        options=passkeys.creation_options(store.config, challenge=ceremony["challenge"], user_handle=account["user_handle"],
                                          email=account["email"], display_name=account["display_name"], exclude=existing),
        expires_at=db.parse_ts(ceremony["expires_at"]),
    )


@router.operation("addAuthenticatorFinish")
def add_authenticator_finish(body: AddAuthenticatorFinishRequest, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                             principal: Principal = Depends(current_principal)) -> RegistrationFinishResponse:
    session = require_session(principal)
    config = store.config
    now = store.clock.now()
    ceremony = passkeys.consume(conn, body.ceremony_id, kind="add_authenticator", now=now)
    if ceremony["session_id"] != session.session_id or ceremony["account_id"] != session.account_id:
        raise passkeys.verification_failed("ceremony_not_this_session")
    reauth_id = canonical_credential_id(body.reauthentication.rawId)
    with db.transaction(conn):
        account = records.account_row(conn, session.account_id)
        stored = records.credential_row_by_credential_id(conn, reauth_id)
        if account is None or stored is None or stored["account_id"] != session.account_id:
            raise passkeys.verification_failed("reauthentication_credential_not_registered")
        assertion = passkeys.verify_assertion(config, body.reauthentication, challenge=ceremony["step_up_challenge"], stored=stored,
                                              user_handle=account["user_handle"])
        passkeys.record_use(conn, stored, assertion, now)
        verified = passkeys.verify_registration(config, body.credential, challenge=ceremony["challenge"])
        if records.credential_row_by_credential_id(conn, verified.credential_id) is not None:
            raise passkeys.verification_failed("credential_already_registered")
        authenticator_id = new_id("cred")
        records.insert_credential(conn, authenticator_id=authenticator_id, account_id=session.account_id, credential_id=verified.credential_id,
                                  public_key_cose=verified.public_key_cose, sign_count=verified.sign_count, transports=verified.transports,
                                  aaguid=verified.aaguid, nickname=body.nickname or ceremony["nickname"],
                                  backup_eligible=verified.backup_eligible, backup_state=verified.backup_state, now=now)
        audit.append(conn, actor=audit.actor_for(session), action=AuditAction.AUTHENTICATOR_ADDED, target_kind="authenticator",
                     target_id=authenticator_id, details={"account_id": session.account_id, "confirmed_with": stored["id"]})
        result = RegistrationFinishResponse(
            account=records.get_account(conn, session.account_id),
            credential=records.credential_from_row(records.live_credential_row(conn, session.account_id, authenticator_id)),
            signed_in=None,
        )
    return result


# --- own authenticators -----------------------------------------------------------------------


@router.operation("listOwnAuthenticators")
def list_own_authenticators(conn: StoreConnection = Depends(get_conn), principal: Principal = Depends(current_principal)) -> Page[WebAuthnCredentialRecord]:
    session = require_session(principal)
    with db.read_snapshot(conn):
        rows = records.live_credential_rows(conn, session.account_id)
    return Page[WebAuthnCredentialRecord](items=[records.credential_from_row(r) for r in rows])


def _own_credential(conn, account_id: str, authenticator_id: str):
    row = records.live_credential_row(conn, account_id, authenticator_id)
    if row is None:
        raise StoreError(ErrorCode.NOT_FOUND, "Authenticator not found", details={"authenticator_id": authenticator_id})
    return row


@router.operation("renameOwnAuthenticator")
def rename_own_authenticator(authenticator_id: str, body: CredentialUpdate, conn: StoreConnection = Depends(get_conn),
                             principal: Principal = Depends(current_principal)) -> WebAuthnCredentialRecord:
    session = require_session(principal)
    with db.transaction(conn):
        _own_credential(conn, session.account_id, authenticator_id)
        conn.execute("UPDATE auth_credentials SET nickname = ? WHERE id = ?", (body.nickname, authenticator_id))
        return records.credential_from_row(records.live_credential_row(conn, session.account_id, authenticator_id))


@router.operation("removeOwnAuthenticator")
def remove_own_authenticator(authenticator_id: str, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                             principal: Principal = Depends(current_principal)) -> None:
    session = require_session(principal)
    now = store.clock.now()
    with db.transaction(conn):
        _own_credential(conn, session.account_id, authenticator_id)
        if len(records.live_credential_rows(conn, session.account_id)) <= 1:
            raise StoreError(ErrorCode.CONFLICT, "You cannot remove your last authenticator; add another one first",
                             details={"reason": "last_authenticator"})
        records.revoke_credentials(conn, session.account_id, now, "removed by owner", authenticator_id=authenticator_id)
        ended = records.revoke_sessions(conn, session.account_id, now, "authenticator removed", authenticator_id=authenticator_id,
                                        except_session_id=session.session_id)
        audit.append(conn, actor=audit.actor_for(session), action=AuditAction.AUTHENTICATOR_REMOVED, target_kind="authenticator",
                     target_id=authenticator_id, details={"account_id": session.account_id, "revoked_session_count": ended})


# --- own sessions -----------------------------------------------------------------------------


@router.operation("listOwnSessions")
def list_own_sessions(store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                      principal: Principal = Depends(current_principal)) -> Page[SessionListItem]:
    session = require_session(principal)
    with db.read_snapshot(conn):
        items = sessions.list_items(conn, session.account_id, current_session_id=session.session_id, now=store.clock.now())
    return Page[SessionListItem](items=items)


@router.operation("revokeOwnSession")
def revoke_own_session(session_id: str, response: Response, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn),
                       principal: Principal = Depends(current_principal)) -> None:
    session = require_session(principal)
    with db.transaction(conn):
        if records.revoke_sessions(conn, session.account_id, store.clock.now(), "revoked by owner", session_id=session_id) != 1:
            raise StoreError(ErrorCode.NOT_FOUND, "Session not found", details={"session_id": session_id})
        audit.append(conn, actor=audit.actor_for(session), action=AuditAction.SESSION_REVOKED, target_kind="session", target_id=session_id,
                     details={"account_id": session.account_id, "own": True})
    if session_id == session.session_id:
        clear_session_cookie(response, store.config)
