"""Auth-area secrets and derivations: area keys, CSRF tokens, sign-in decoys, setup codes, challenges.

* **Area keys** (``auth_settings``): random 32-byte keys generated on first use, one for deriving
  CSRF tokens and one for sign-in decoys. They never leave the Store database.
* **CSRF token**: ``base64url(HMAC-SHA256(csrf_key, cookie_value))``. It is bound to the session
  (only the holder of the cookie can recompute it), so ``GET /auth/session`` can return it again
  although Store stores only ``sha256(csrf)``.
* **Decoys** (contract ``CredentialRequestOptions``): for an unknown, disabled or re-invite-pending
  email, one or two credential descriptors derived as HMAC(decoy_key, email), with plausible ID
  lengths and transports. They are stable per email, so repeated sign-in attempts cannot tell a
  decoy from a real account.
* **Setup codes**: 20 characters from an unambiguous alphabet, printed in groups of five (about 100
  bits). Input is normalized (case, spaces and hyphens ignored) before hashing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import List, Tuple

from call1.contracts.auth import AuthenticatorTransport, CredentialDescriptor

CSRF_KEY = "csrf_key"
DECOY_KEY = "decoy_key"

_SETUP_ALPHABET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"  # no 0/O, 1/I/L, U
_SETUP_GROUPS = 4
_SETUP_GROUP_LEN = 5

_DECOY_ID_LENGTHS = (16, 20, 32, 64)
_DECOY_TRANSPORTS: Tuple[Tuple[AuthenticatorTransport, ...], ...] = (
    (AuthenticatorTransport.USB,),
    (AuthenticatorTransport.NFC, AuthenticatorTransport.USB),
    (AuthenticatorTransport.HYBRID, AuthenticatorTransport.INTERNAL),
    (AuthenticatorTransport.INTERNAL,),
    (AuthenticatorTransport.USB, AuthenticatorTransport.NFC),
)


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def new_challenge() -> str:
    """A single-use WebAuthn challenge: 32 random bytes, base64url."""
    return b64url(secrets.token_bytes(32))


def new_user_handle() -> str:
    """An opaque WebAuthn user handle (never the email): 32 random bytes, base64url."""
    return b64url(secrets.token_bytes(32))


def area_key(conn, name: str) -> bytes:
    """The area key ``name``, created on first use. Safe under concurrent first use."""
    row = conn.execute("SELECT value FROM auth_settings WHERE key = ?", (name,)).fetchone()
    if row is None:
        conn.execute("INSERT OR IGNORE INTO auth_settings (key, value) VALUES (?, ?)", (name, secrets.token_hex(32)))
        row = conn.execute("SELECT value FROM auth_settings WHERE key = ?", (name,)).fetchone()
    return bytes.fromhex(row["value"])


def csrf_token_for(conn, cookie_value: str) -> str:
    return b64url(hmac.new(area_key(conn, CSRF_KEY), b"csrf|" + cookie_value.encode("utf-8"), hashlib.sha256).digest())


def decoy_descriptors(conn, email_key: str) -> List[CredentialDescriptor]:
    key = area_key(conn, DECOY_KEY)
    head = hmac.new(key, b"decoy-count|" + email_key.encode("utf-8"), hashlib.sha256).digest()
    count = 1 + (head[0] & 1)
    out: List[CredentialDescriptor] = []
    for index in range(count):
        label = f"decoy|{index}|".encode("ascii") + email_key.encode("utf-8")
        body = hmac.new(key, label, hashlib.sha512).digest()
        shape = hmac.new(key, b"shape|" + label, hashlib.sha256).digest()
        length = _DECOY_ID_LENGTHS[shape[0] % len(_DECOY_ID_LENGTHS)]
        transports = _DECOY_TRANSPORTS[shape[1] % len(_DECOY_TRANSPORTS)]
        out.append(CredentialDescriptor(id=b64url(body[:length]), transports=list(transports)))
    return out


def new_setup_code() -> str:
    """A printable one-time code, e.g. ``K7QRM-2XWPA-HN9TD-3FVCE``."""
    chars = "".join(secrets.choice(_SETUP_ALPHABET) for _ in range(_SETUP_GROUPS * _SETUP_GROUP_LEN))
    return "-".join(chars[i:i + _SETUP_GROUP_LEN] for i in range(0, len(chars), _SETUP_GROUP_LEN))


def normalize_setup_code(code: str) -> str:
    return "".join(ch for ch in code.upper() if ch not in " -\t\r\n")
