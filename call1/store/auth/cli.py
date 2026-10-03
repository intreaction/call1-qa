"""Host commands behind ``python -m call1.store setup-code`` and ``issue-service-key``.

Run on the Store host as the Store OS user; never reachable over HTTP. Each runs in one transaction
with its audit event (``audit.installer_actor`` / ``audit.break_glass_actor``). ``__main__`` handles
argument parsing, printing and writing Process's config file.

A refusal (for example a first-admin code while an active admin exists) raises
``HostCommandRefused``: a ``StoreError`` with the contract code, which is also a ``ConfigError`` so
``__main__`` prints it and exits 2 instead of showing a traceback.
"""

from __future__ import annotations

import getpass
from typing import Iterable, Optional, Tuple

from call1.contracts.auth import ServiceKeyIssued, SetupCodeIssueRequest, SetupCodeRecord
from call1.contracts.common import ServiceScope

from .. import db
from ..config import ConfigError
from ..errors import StoreError
from . import identity


class HostCommandRefused(StoreError, ConfigError):
    """A host command refused its request; ``code`` and ``details`` are the contract's."""


def _refused(exc: StoreError) -> HostCommandRefused:
    return HostCommandRefused(exc.code, exc.message, details=exc.details)


def issue_setup_code(store, request: SetupCodeIssueRequest) -> Tuple[str, SetupCodeRecord]:
    """Issue a single-use first-admin or break-glass code (``SETUP_CODE_LIFETIME``). Store keeps
    ``hash_secret(code)``. ``first_admin`` is refused (``CONFLICT``) while an active admin exists.
    Writes ``setup_code_issued``. Returns the code (printed once) and its record."""
    with store.connection() as conn:
        try:
            with db.transaction(conn):
                return identity.issue_setup_code(conn, store.config.parameters, request, now=store.clock.now())
        except StoreError as exc:
            raise _refused(exc) from None


def issue_service_key(store, *, installation_label: str, scopes: Iterable[ServiceScope], primary_host: Optional[bool],
                      os_user: Optional[str] = None) -> ServiceKeyIssued:
    """Register the installation if no active one has this label, then issue a key with
    ``scopes``. ``primary_host`` None means: primary when no active installation holds it (the
    one-computer install), so ML jobs have a host; at most one active installation is primary.
    Writes ``installation_registered`` (when new) and ``service_key_issued`` with an installer
    actor. Returns the one-time ``ServiceKeyIssued``."""
    user = os_user or getpass.getuser() or "installer"
    with store.connection() as conn:
        try:
            with db.transaction(conn):
                return identity.installer_key(conn, installation_label=installation_label, scopes=scopes, primary_host=primary_host,
                                              os_user=user, now=store.clock.now())
        except StoreError as exc:
            raise _refused(exc) from None
