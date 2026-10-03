"""The ``AuthBackend`` the principal guard calls on every request (``principals.AuthBackend``).

``SqliteAuthBackend`` works over the 030_auth tables:

* ``authenticate_service_key``: ``hash_secret(token)`` lookup; None when unknown, revoked, expired,
  past its rotation ``grace_until``, or its installation is retired. ``primary_host`` and the
  scopes are read from the rows on every request. ``last_used_at`` is written at most once a minute.
* ``authenticate_session``: ``hash_secret(cookie)`` lookup (``sessions.authenticate``): None when
  unknown or revoked, ``account_disabled`` for a disabled account, ``session_expired`` past the idle
  or absolute expiry; the role is the account's current one; the idle expiry slides.
* ``verify_csrf``: constant-time comparison of ``sha256(header)`` with the session's CSRF hash.
* ``mint_service_key_for_tests`` / ``mint_session_for_tests``: TEST-ONLY (no HTTP route reaches
  them). They write real rows, so fixtures exercise the real lookups.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Optional

from call1.contracts.auth import AccountStatus
from call1.contracts.common import ReviewerRole, ServiceScope

from .. import db
from ..ids import new_id
from ..principals import (
    AuthBackend,
    MintedServiceKey,
    MintedSession,
    ServiceKeyPrincipal,
    generate_service_key_token,
    hash_secret,
)
from . import crypto, records, sessions

LAST_USED_EVERY = timedelta(seconds=60)


class SqliteAuthBackend:
    def __init__(self, store) -> None:
        self._store = store

    @property
    def _parameters(self):
        return self._store.config.parameters

    # --- lookups ---------------------------------------------------------------------------

    def authenticate_service_key(self, conn, token: str, now: datetime) -> Optional[ServiceKeyPrincipal]:
        row = conn.execute(
            "SELECT k.*, i.primary_host, i.retired_at AS installation_retired_at FROM auth_service_keys k "
            "JOIN auth_installations i ON i.id = k.installation_id WHERE k.key_hash = ?",
            (hash_secret(token),),
        ).fetchone()
        if row is None or row["installation_retired_at"] is not None or not records.key_usable(row, now):
            return None
        last_used = db.parse_ts(row["last_used_at"])
        if last_used is None or now - last_used >= LAST_USED_EVERY:
            with db.transaction(conn):
                conn.execute("UPDATE auth_service_keys SET last_used_at = ? WHERE id = ?", (db.ts(now), row["id"]))
        return ServiceKeyPrincipal(
            key_id=row["id"],
            installation_id=row["installation_id"],
            scopes=frozenset(ServiceScope(s) for s in db.loads(row["scopes_json"])),
            primary_host=bool(row["primary_host"]),
            key_prefix=row["key_prefix"],
        )

    def authenticate_session(self, conn, cookie_value: str, now: datetime):
        return sessions.authenticate(conn, self._parameters, cookie_value, now)

    def verify_csrf(self, conn, principal, header_value: Optional[str]) -> bool:
        return sessions.verify_csrf(conn, principal.session_id, header_value)

    # --- TEST-ONLY minting -----------------------------------------------------------------

    def mint_service_key_for_tests(self, conn, *, scopes: Iterable[ServiceScope], installation_id: Optional[str] = None,
                                   primary_host: bool = False, label: str = "test key") -> MintedServiceKey:
        """TEST-ONLY. Unlike the admin route and the host command, this does not enforce the
        one-primary-host rule, so fixtures can mint several primary keys."""
        now = self._store.clock.now()
        token, prefix = generate_service_key_token()
        scope_list = [ServiceScope(s) for s in scopes]
        with db.transaction(conn):
            installation_id = installation_id or new_id("inst")
            if records.installation_row(conn, installation_id) is None:
                records.insert_installation(conn, installation_id=installation_id, label=label[:200] or "test installation",
                                            primary_host=primary_host, created_by_account_id=None, now=now)
            elif primary_host:
                conn.execute("UPDATE auth_installations SET primary_host = 1 WHERE id = ?", (installation_id,))
            key_id = new_id("key")
            records.insert_service_key(conn, key_id=key_id, installation_id=installation_id, label=label[:200] or "test key", scopes=scope_list,
                                       key_prefix=prefix, key_hash=hash_secret(token), created_by_account_id=None, expires_at=None,
                                       rotated_from_key_id=None, now=now)
            row = records.installation_row(conn, installation_id)
        principal = ServiceKeyPrincipal(key_id=key_id, installation_id=installation_id, scopes=frozenset(scope_list),
                                        primary_host=bool(row["primary_host"]), key_prefix=prefix)
        return MintedServiceKey(token=token, principal=principal)

    def mint_session_for_tests(self, conn, *, role: ReviewerRole, email: Optional[str] = None,
                               display_name: Optional[str] = None) -> MintedSession:
        """TEST-ONLY. Creates an active account with ``role`` and one (non-signing) authenticator
        record, or reuses the account that has ``email`` (setting its role and status active), and
        signs it in."""
        config = self._store.config
        now = self._store.clock.now()
        role = ReviewerRole(role)
        with db.transaction(conn):
            existing = records.account_row_by_email(conn, email) if email else None
            if existing is not None:
                account_id = existing["id"]
                records.update_account(conn, account_id, now, role=role, status=AccountStatus.ACTIVE,
                                       **({"display_name": display_name} if display_name else {}))
            else:
                account_id = new_id("acct")
                records.insert_account(conn, account_id=account_id, email=email or f"{role.value}.{account_id[-6:]}@example.com",
                                       display_name=display_name or f"Test {role.value}", role=role, status=AccountStatus.ACTIVE,
                                       user_handle=crypto.new_user_handle(), now=now)
            live = records.live_credential_rows(conn, account_id)
            if live:
                authenticator_id = live[0]["id"]
            else:
                authenticator_id = new_id("cred")
                records.insert_credential(conn, authenticator_id=authenticator_id, account_id=account_id,
                                          credential_id=crypto.b64url(b"test-credential:" + authenticator_id.encode()),
                                          public_key_cose=crypto.b64url(b"test-only-not-a-key"), sign_count=0, transports=["usb"],
                                          aaguid=None, nickname="Test authenticator", backup_eligible=False, backup_state=False, now=now)
            records.update_account(conn, account_id, now, last_sign_in_at=now)
            session_id, cookie, csrf = sessions.create_session(conn, config.parameters, account_id=account_id,
                                                               authenticator_id=authenticator_id, now=now)
            principal = sessions.authenticate(conn, config.parameters, cookie, now)
        return MintedSession(cookie_name=config.cookie_name, cookie_value=cookie, csrf_token=csrf, principal=principal,
                             csrf_header=config.csrf_header)


def build_auth_backend(store) -> AuthBackend:
    """The backend ``Store.open`` installs. Creates the area keys up front so request paths never
    have to write them."""
    with store.connection() as conn:
        with db.transaction(conn):
            crypto.area_key(conn, crypto.CSRF_KEY)
            crypto.area_key(conn, crypto.DECOY_KEY)
    return SqliteAuthBackend(store)
