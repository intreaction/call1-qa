"""DEMO MODE: persona sign-in without a passkey, for class demos on ``localhost`` only.

Off unless ``CALL1_STORE_DEMO=1``, and ``StoreConfig`` refuses it outside dev mode (plain
``http://localhost`` on a loopback bind), so an installed Store can never run it. The passkey
ceremonies are untouched and keep working beside it.

Two routes, outside ``/store/v1`` because they are not part of the contract:

* ``GET /demo/status`` -> ``{"demo": true, "label": ..., "personas": [...]}``
* ``POST /demo/sign-in`` ``{"persona": "admin" | "supervisor" | "reviewer"}`` -> the same
  ``SignInResponse`` body a passkey sign-in returns (``{"session": SessionInfo}``) plus
  ``"demo": true`` and ``"persona"``, and the same server-side session cookie. The CSRF token is
  ``session.csrf_token``, exactly as after a passkey sign-in, and every ``/store/v1`` route then
  enforces roles, CSRF and origin as usual.

With demo mode off both answer 404 ``not_found`` (they are registered either way, so the Evaluate
catch-all never serves its page for them).

Sign-in creates the persona's account on first use (idempotent: it is keyed by the persona's
fixed email) with one placeholder authenticator record that holds no key, because a session row
must name the authenticator it was issued for. Nothing can sign in with that record through the
passkey routes. On later sign-ins the persona's role is restored if an admin changed it (unless
that would demote the last admin: 409 ``last_admin``), a ``reinvite_required`` persona is
reactivated, and a disabled persona gets 403 ``account_disabled``, as a passkey sign-in would.

Audit: account creation (``account_created``), a placeholder record (``authenticator_added``), a
role or status restore and every sign-in (``account_updated``) all carry
``details.demo_mode = true`` and ``details.demo_persona``. The contract has no sign-in action, so
a demo sign-in is ``account_updated`` (it sets ``last_sign_in_at``) with
``details.event = "demo_sign_in"``; the actor is the persona's reviewer session.

Guards beyond the dev-mode config rule: the request's ``Host`` must be a loopback name
(``localhost``, ``127.0.0.1``, ``::1``), which defeats DNS rebinding, and a browser ``Origin``, when
sent, must be a Store origin (403 ``origin_not_allowed``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict

from call1.contracts.auth import AccountStatus, SignInResponse
from call1.contracts.common import ReviewerRole
from call1.contracts.errors import ErrorCode
from call1.contracts.events import Actor, ActorKind, AuditAction

from .. import audit, db
from ..context import Store
from ..db import StoreConnection
from ..deps import get_conn, get_store
from ..errors import StoreError
from ..ids import new_id
from ..principals import set_session_cookie
from . import crypto, identity, passkeys, records, sessions

DEMO_LABEL = "Demo mode: passkeys bypassed on localhost. Not for real data."
PLACEHOLDER_NICKNAME = "Demo mode (no passkey)"
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]")
DEMO_ACTOR = Actor(kind=ActorKind.STORE_SYSTEM, display="Demo mode")


@dataclass(frozen=True)
class Persona:
    key: str
    display_name: str
    role: ReviewerRole
    email: str

    def as_json(self) -> dict:
        return {"persona": self.key, "display_name": self.display_name, "role": self.role.value, "email": self.email}


PERSONAS: Dict[str, Persona] = {
    p.key: p
    for p in (
        Persona("admin", "Demo Admin", ReviewerRole.ADMIN, "demo.admin@call1-demo.example"),
        Persona("supervisor", "Demo Supervisor", ReviewerRole.SUPERVISOR, "demo.supervisor@call1-demo.example"),
        Persona("reviewer", "Demo Reviewer", ReviewerRole.REVIEWER, "demo.reviewer@call1-demo.example"),
    )
}


class DemoSignInRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona: Literal["admin", "supervisor", "reviewer"]


def demo_enabled(store: Store) -> bool:
    return bool(store.config.demo_mode and store.config.dev_mode)


def require_demo(store: Store = Depends(get_store)) -> None:
    """404 unless demo mode is on. Runs before the body is read, so an off Store answers 404."""
    if not demo_enabled(store):
        raise StoreError(ErrorCode.NOT_FOUND, "No such route")


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _check_request(request: Request, store: Store) -> None:
    host = (request.url.hostname or "").lower()
    if host not in LOOPBACK_HOSTS:
        raise StoreError(ErrorCode.FORBIDDEN, "Demo sign-in answers only on localhost", details={"reason": "demo_host_not_loopback"})
    passkeys.check_request_origin(store.config, request.headers.get("origin"))


def _details(persona: Persona, **extra) -> dict:
    return {"demo_mode": True, "demo_persona": persona.key, **extra}


def ensure_persona(conn, persona: Persona, now) -> tuple:
    """(account_id, authenticator_id) for the persona, creating or restoring it. Caller's transaction."""
    identity.require_open(conn)
    row = records.account_row_by_email(conn, persona.email)
    if row is None:
        account_id = new_id("acct")
        records.insert_account(conn, account_id=account_id, email=persona.email, display_name=persona.display_name, role=persona.role,
                               status=AccountStatus.ACTIVE, user_handle=crypto.new_user_handle(), now=now)
        audit.append(conn, actor=DEMO_ACTOR, action=AuditAction.ACCOUNT_CREATED, target_kind="account", target_id=account_id,
                     details=_details(persona, role=persona.role.value))
    else:
        account_id = row["id"]
        if row["status"] == AccountStatus.DISABLED.value:
            raise StoreError(ErrorCode.ACCOUNT_DISABLED, "This account is disabled; ask an administrator")
        changes = {}
        if row["role"] != persona.role.value:
            identity.ensure_admin_remains(conn, account_id)
            changes["role"] = persona.role
        if row["status"] != AccountStatus.ACTIVE.value:
            changes["status"] = AccountStatus.ACTIVE
        if changes:
            records.update_account(conn, account_id, now, **changes)
            audit.append(conn, actor=DEMO_ACTOR, action=AuditAction.ACCOUNT_UPDATED, target_kind="account", target_id=account_id,
                         details=_details(persona, event="demo_restored", fields=",".join(sorted(changes))))
    live = records.live_credential_rows(conn, account_id)
    if live:
        return account_id, live[0]["id"]
    authenticator_id = new_id("cred")
    records.insert_credential(conn, authenticator_id=authenticator_id, account_id=account_id,
                              credential_id=crypto.b64url(b"call1-demo-placeholder:" + authenticator_id.encode()),
                              public_key_cose=crypto.b64url(b"demo-mode-placeholder-not-a-key"), sign_count=0, transports=[],
                              aaguid=None, nickname=PLACEHOLDER_NICKNAME, backup_eligible=False, backup_state=False, now=now)
    audit.append(conn, actor=DEMO_ACTOR, action=AuditAction.AUTHENTICATOR_ADDED, target_kind="authenticator", target_id=authenticator_id,
                 details=_details(persona, placeholder=True, account_id=account_id))
    return account_id, authenticator_id


router = APIRouter(prefix="/demo", dependencies=[Depends(require_demo)], include_in_schema=False)


@router.get("/status")
def demo_status(response: Response) -> dict:
    _no_store(response)
    return {"demo": True, "label": DEMO_LABEL, "personas": [p.as_json() for p in PERSONAS.values()]}


@router.post("/sign-in")
def demo_sign_in(body: DemoSignInRequest, request: Request, response: Response, store: Store = Depends(get_store),
                 conn: StoreConnection = Depends(get_conn)) -> dict:
    _check_request(request, store)
    _no_store(response)
    persona = PERSONAS[body.persona]
    config = store.config
    now = store.clock.now()
    with db.transaction(conn):
        account_id, authenticator_id = ensure_persona(conn, persona, now)
        records.update_account(conn, account_id, now, last_sign_in_at=now)
        session_id, cookie, csrf = sessions.create_session(conn, config.parameters, account_id=account_id,
                                                           authenticator_id=authenticator_id, now=now)
        actor = Actor(kind=ActorKind.REVIEWER, account_id=account_id, session_id=session_id, display=f"{persona.display_name} (demo mode)")
        audit.append(conn, actor=actor, action=AuditAction.ACCOUNT_UPDATED, target_kind="account", target_id=account_id,
                     details=_details(persona, event="demo_sign_in"))
        info = sessions.session_info(conn, session_id, csrf)
    set_session_cookie(response, config, cookie, max_age=config.parameters.session_absolute_lifetime_seconds)
    signed_in = SignInResponse(session=info).model_dump(mode="json")
    return {**signed_in, "demo": True, "persona": persona.key}


__all__ = ["router", "PERSONAS", "Persona", "DEMO_LABEL", "demo_enabled", "ensure_persona"]
