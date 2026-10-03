"""FastAPI dependencies: the Store, the per-request connection, and the principal guard.

``create_app`` attaches ``guard_for(route)`` to every handled contract route, so a handler only
runs for a principal the route's ``PrincipalRule`` list admits. The guard, in order:

1. ``[ANONYMOUS]`` routes: no credential is read; the principal is anonymous.
2. ``Authorization`` present: it must be a valid service key (401 ``unauthenticated``), the route
   must accept service keys (403 ``forbidden``), and the key must hold one of the route's scopes
   (403 ``insufficient_scope``).
3. Session cookie present: the session must be live (401 ``unauthenticated``; the backend raises
   ``session_expired`` / ``account_disabled``), the route must accept sessions (403 ``forbidden``),
   the current role must satisfy a rule (403 ``insufficient_role``), and on a state-changing method
   ``Origin`` (when sent) must be allowed (403 ``origin_not_allowed``) and ``X-Call1-CSRF`` must
   match (403 ``csrf_failed``).
4. Nothing presented: 401 ``unauthenticated``.

Object rules (``route.object_rule``: own installation, the scope that created an upload, the item's
assignee, ``resolve_any_review``) stay with the handler; see the helpers in ``principals``.
"""

from __future__ import annotations

from typing import Callable, Iterator

from fastapi import Depends, Request

from call1.contracts.api import Route
from call1.contracts.common import PrincipalKind, role_satisfies
from call1.contracts.errors import ErrorCode

from .context import Store
from .db import StoreConnection
from .errors import StoreError
from .principals import ANONYMOUS_PRINCIPAL, SAFE_METHODS, Principal, parse_bearer


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_conn(store: Store = Depends(get_store)) -> Iterator[StoreConnection]:
    """One connection per request, shared by the guard and the handler, closed afterwards."""
    conn = store.db.connect()
    try:
        yield conn
    finally:
        conn.close()


def current_principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise RuntimeError("route registered without its principal guard")
    return principal


def authenticate(request: Request, store: Store, conn: StoreConnection, route: Route) -> Principal:
    rules = route.principals
    process_rules = [r for r in rules if r.kind is PrincipalKind.PROCESS_SERVICE_KEY]
    session_rules = [r for r in rules if r.kind is PrincipalKind.REVIEWER_SESSION]
    now = store.clock.now()

    authorization = request.headers.get("authorization")
    if authorization is not None:
        token = parse_bearer(authorization)
        principal = store.auth.authenticate_service_key(conn, token, now) if token else None
        if principal is None:
            raise StoreError(ErrorCode.UNAUTHENTICATED, "Service key not recognised", headers={"WWW-Authenticate": "Bearer"})
        if not process_rules:
            raise StoreError(ErrorCode.FORBIDDEN, "This route does not accept Process service keys")
        if not any(principal.has_scope(r.scope) for r in process_rules):
            needed = "|".join(r.scope.value for r in process_rules if r.scope is not None)
            raise StoreError(ErrorCode.INSUFFICIENT_SCOPE, "The service key lacks the scope this route needs", details={"required_scope": needed})
        return principal

    cookie = request.cookies.get(store.config.cookie_name)
    if cookie:
        principal = store.auth.authenticate_session(conn, cookie, now)
        if principal is None:
            raise StoreError(ErrorCode.UNAUTHENTICATED, "Session not recognised; sign in again")
        if not session_rules:
            raise StoreError(ErrorCode.FORBIDDEN, "This route does not accept reviewer sessions")
        if not any(role_satisfies(principal.role, r.min_role) and (r.permission is None or principal.can(r.permission)) for r in session_rules):
            rule = session_rules[0]
            raise StoreError(ErrorCode.INSUFFICIENT_ROLE, "Your role does not allow this",
                             details={"required_role": rule.min_role.value if rule.min_role else None, "required_permission": rule.permission.value if rule.permission else None})
        if request.method.upper() not in SAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None and origin.rstrip("/") not in store.config.allowed_origins:
                raise StoreError(ErrorCode.ORIGIN_NOT_ALLOWED, "Request origin is not the Store origin")
            if not store.auth.verify_csrf(conn, principal, request.headers.get(store.config.csrf_header)):
                raise StoreError(ErrorCode.CSRF_FAILED, "Missing or wrong X-Call1-CSRF header")
        return principal

    if any(r.kind is PrincipalKind.ANONYMOUS for r in rules):
        return ANONYMOUS_PRINCIPAL
    raise StoreError(ErrorCode.UNAUTHENTICATED, "Sign in, or present a Process service key")


def guard_for(route: Route) -> Callable:
    """The dependency that enforces ``route.principals`` and stores the principal on the request."""
    if all(rule.kind is PrincipalKind.ANONYMOUS for rule in route.principals):
        def anonymous_guard(request: Request) -> None:
            request.state.principal = ANONYMOUS_PRINCIPAL

        return anonymous_guard

    def guard(request: Request, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn)) -> None:
        request.state.principal = authenticate(request, store, conn, route)

    guard.__name__ = f"guard_{route.operation_id}"
    return guard
