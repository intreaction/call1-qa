"""Who is calling: the three Store principals, the auth-owner interface, and the secret helpers.

The contract names three principals (``PrincipalKind``) and each route's ``PrincipalRule`` list
(``api.ROUTES[...].principals``). ``call1.store.deps.guard_for(route)`` enforces them before any
handler runs; handlers read the result with ``Depends(current_principal)``:

* ``ServiceKeyPrincipal``: a Process installation, ``Authorization: Bearer c1sk_<prefix>_<secret>``.
  Store keeps ``sha256(token)`` only. The route needs one of its listed scopes. Every
  ``installation_id`` a request names must be the key's own (``require_own_installation``).
* ``SessionPrincipal``: a signed-in reviewer, by session cookie. Role and permissions are the
  account's current ones (``ROLE_PERMISSIONS``), re-read on every request by the auth backend.
  State-changing requests also need ``X-Call1-CSRF`` and, when the browser sends ``Origin``, an
  allowed origin.
* ``AnonymousPrincipal``: no credential; only the routes whose rule list is ``[ANONYMOUS]``.

``AuthBackend`` is the interface the **auth owner** implements (``call1/store/auth/backend.py``):
key lookup, session lookup and the CSRF check, plus two TEST-ONLY minting methods the fixtures in
``tests/store/conftest.py`` use. No HTTP route mints anything.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import string
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Dict, FrozenSet, Iterable, Optional, Protocol, Union, runtime_checkable

from starlette.responses import Response

from call1.contracts.auth import ROLE_PERMISSIONS, Permission
from call1.contracts.common import PrincipalKind, ReviewerRole, ServiceScope
from call1.contracts.errors import ErrorCode

from .config import StoreConfig
from .errors import StoreError

SERVICE_KEY_TOKEN = re.compile(r"^(c1sk_[A-Za-z0-9]{6})_[A-Za-z0-9_-]{43,}$")
"""The ``ServiceKeyIssued.token`` form; group 1 is the non-secret ``key_prefix``."""

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_ALNUM = string.ascii_letters + string.digits


# --- secrets --------------------------------------------------------------------------------


def hash_secret(value: str) -> str:
    """``sha256:<hex>`` of a token, cookie value, CSRF token, invitation token or setup code."""
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def secrets_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def generate_secret(nbytes: int = 32) -> str:
    """URL-safe random secret (base64url, 43 characters for 32 bytes)."""
    return secrets.token_urlsafe(nbytes)


def generate_service_key_token() -> tuple[str, str]:
    """A new ``c1sk_<prefix>_<secret>`` token and its prefix. Store keeps ``hash_secret(token)``."""
    prefix = "c1sk_" + "".join(secrets.choice(_ALNUM) for _ in range(6))
    return f"{prefix}_{generate_secret(32)}", prefix


def parse_bearer(header_value: Optional[str]) -> Optional[str]:
    """The service-key token of an ``Authorization: Bearer`` header, or None if malformed."""
    if not header_value:
        return None
    scheme, _, token = header_value.strip().partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not SERVICE_KEY_TOKEN.match(token):
        return None
    return token


# --- principals -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AnonymousPrincipal:
    kind: ClassVar[PrincipalKind] = PrincipalKind.ANONYMOUS


@dataclass(frozen=True)
class ServiceKeyPrincipal:
    key_id: str
    installation_id: str
    scopes: FrozenSet[ServiceScope]
    primary_host: bool = False
    key_prefix: str = ""
    kind: ClassVar[PrincipalKind] = PrincipalKind.PROCESS_SERVICE_KEY

    def has_scope(self, scope: Optional[ServiceScope]) -> bool:
        return scope is not None and scope in self.scopes

    @property
    def feed_audience(self) -> str:
        return PrincipalKind.PROCESS_SERVICE_KEY.value


@dataclass(frozen=True)
class SessionPrincipal:
    session_id: str
    account_id: str
    email: str
    display_name: str
    role: ReviewerRole
    authenticator_id: str = ""
    kind: ClassVar[PrincipalKind] = PrincipalKind.REVIEWER_SESSION

    @property
    def permissions(self) -> FrozenSet[Permission]:
        return ROLE_PERMISSIONS[self.role]

    def can(self, permission: Permission) -> bool:
        return permission in self.permissions

    @property
    def feed_audience(self) -> str:
        return self.role.value


Principal = Union[AnonymousPrincipal, ServiceKeyPrincipal, SessionPrincipal]
ANONYMOUS_PRINCIPAL = AnonymousPrincipal()


# --- object-rule helpers for handlers -------------------------------------------------------


def require_own_installation(principal: Principal, installation_id: Optional[str]) -> None:
    """Contract OWN_INSTALLATION: a named installation must be the calling key's (403 forbidden)."""
    if installation_id is None:
        return
    if not isinstance(principal, ServiceKeyPrincipal) or not secrets_equal(installation_id, principal.installation_id):
        raise StoreError(ErrorCode.FORBIDDEN, "A Process key acts only as its own installation", details={"reason": "installation_mismatch"})


def require_permission(principal: Principal, permission: Permission) -> SessionPrincipal:
    """An object rule's extra permission (e.g. ``resolve_any_review``)."""
    if not isinstance(principal, SessionPrincipal):
        raise StoreError(ErrorCode.FORBIDDEN, "A reviewer session is required")
    if not principal.can(permission):
        raise StoreError(ErrorCode.INSUFFICIENT_ROLE, "Your role does not allow this", details={"required_permission": permission.value})
    return principal


def require_session(principal: Principal) -> SessionPrincipal:
    if not isinstance(principal, SessionPrincipal):
        raise StoreError(ErrorCode.FORBIDDEN, "A reviewer session is required")
    return principal


def require_service_key(principal: Principal) -> ServiceKeyPrincipal:
    if not isinstance(principal, ServiceKeyPrincipal):
        raise StoreError(ErrorCode.FORBIDDEN, "A Process service key is required")
    return principal


# --- session cookie -------------------------------------------------------------------------


def set_session_cookie(response: Response, config: StoreConfig, value: str, *, max_age: int) -> None:
    """Set the session cookie: ``__Host-call1_session`` (Secure) normally, ``call1_session``
    without Secure in dev mode (config docstring). HttpOnly, SameSite=Strict, Path=/, no Domain."""
    response.set_cookie(config.cookie_name, value, max_age=max_age, path="/", secure=config.cookie_secure, httponly=True, samesite="strict")


def clear_session_cookie(response: Response, config: StoreConfig) -> None:
    response.delete_cookie(config.cookie_name, path="/", secure=config.cookie_secure, httponly=True, samesite="strict")


# --- the auth-owner interface ---------------------------------------------------------------


@dataclass(frozen=True)
class MintedServiceKey:
    """TEST-ONLY: a service key minted directly by the backend (no HTTP route)."""

    token: str
    principal: ServiceKeyPrincipal

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    @property
    def installation_id(self) -> str:
        return self.principal.installation_id


@dataclass(frozen=True)
class MintedSession:
    """TEST-ONLY: a reviewer session minted directly by the backend (no ceremony)."""

    cookie_name: str
    cookie_value: str
    csrf_token: str
    principal: SessionPrincipal
    csrf_header: str = "X-Call1-CSRF"

    @property
    def read_headers(self) -> Dict[str, str]:
        """For GET requests: the cookie only."""
        return {"Cookie": f"{self.cookie_name}={self.cookie_value}"}

    @property
    def headers(self) -> Dict[str, str]:
        """For state-changing requests: the cookie and the CSRF header."""
        return {**self.read_headers, self.csrf_header: self.csrf_token}

    @property
    def account_id(self) -> str:
        return self.principal.account_id


@runtime_checkable
class AuthBackend(Protocol):
    """Implemented by the auth owner (``call1/store/auth/backend.py``). Every method receives the
    request's connection; lookups run on each request (no caching of roles or key status)."""

    def authenticate_service_key(self, conn, token: str, now: datetime) -> Optional[ServiceKeyPrincipal]:
        """Look up ``hash_secret(token)``. Return None when unknown, revoked, expired, past its
        rotation ``grace_until``, or its installation is retired (the guard answers 401)."""

    def authenticate_session(self, conn, cookie_value: str, now: datetime) -> Optional[SessionPrincipal]:
        """Look up ``hash_secret(cookie_value)``. Return None for an unknown or revoked session
        (401 ``unauthenticated``). Raise ``StoreError(SESSION_EXPIRED)`` past the idle or absolute
        expiry and ``StoreError(ACCOUNT_DISABLED)`` for a disabled account. Return the account's
        *current* role. May slide the idle expiry (a short write)."""

    def verify_csrf(self, conn, principal: SessionPrincipal, header_value: Optional[str]) -> bool:
        """True when ``header_value`` is this session's CSRF token (constant-time compare of hashes)."""

    def mint_service_key_for_tests(self, conn, *, scopes: Iterable[ServiceScope], installation_id: Optional[str] = None,
                                   primary_host: bool = False, label: str = "test key") -> MintedServiceKey:
        """TEST-ONLY. Create (or reuse, when ``installation_id`` names one) an installation and issue
        a key to it without an HTTP route. Commits its own transaction."""

    def mint_session_for_tests(self, conn, *, role: ReviewerRole, email: Optional[str] = None,
                               display_name: Optional[str] = None) -> MintedSession:
        """TEST-ONLY. Create an active account with ``role`` (and one authenticator record) and a
        signed-in session for it. Commits its own transaction."""

