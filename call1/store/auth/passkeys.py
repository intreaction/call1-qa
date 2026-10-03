"""WebAuthn ceremonies on the ``webauthn`` 3.x library: options, single-use challenges, verification.

Every ceremony is a row in ``auth_ceremonies`` with a random 32-byte challenge that expires after
``CHALLENGE_LIFETIME`` (``webauthn_challenge_lifetime_seconds``). Finish **consumes** the ceremony
in its own committed transaction before verifying anything, so a challenge is used at most once
whether the verification then passes or fails, and two concurrent finishes cannot both win.

Verification checks, in order: the HTTP ``Origin`` header (when the browser sends one) and the
``clientDataJSON`` origin are Store origins (403 ``origin_not_allowed``); then the library checks
the ceremony type, the challenge, the RP ID hash (the fixed Store hostname, ``localhost`` in dev
mode), user presence, **user verification** (required), the signature and the signature counter
(401 ``webauthn_verification_failed``). Store requests attestation ``none`` and keeps no
attestation trust decision, so records carry ``attestation_format: none``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Sequence

from webauthn import verify_authentication_response, verify_registration_response
from webauthn.helpers.cose import COSEAlgorithmIdentifier
from webauthn.helpers.structs import CredentialDeviceType

from call1.contracts.auth import (
    AuthenticationCredentialJSON,
    AuthenticatorSelection,
    AuthenticatorTransport,
    CredentialCreationOptions,
    CredentialDescriptor,
    CredentialRequestOptions,
    PubKeyCredParam,
    RegistrationCredentialJSON,
    RelyingParty,
    UserEntity,
)
from call1.contracts.errors import ErrorCode

from .. import db
from ..config import StoreConfig
from ..errors import StoreError
from ..ids import new_id
from . import crypto

RP_NAME = "Call1 Store"
PUB_KEY_ALGS = (COSEAlgorithmIdentifier.ECDSA_SHA_256, COSEAlgorithmIdentifier.RSASSA_PKCS1_v1_5_SHA_256)
CEREMONY_RETENTION = timedelta(days=1)


def verification_failed(reason: str) -> StoreError:
    return StoreError(ErrorCode.WEBAUTHN_VERIFICATION_FAILED, "Passkey verification failed; start again", details={"reason": reason})


# --- options ----------------------------------------------------------------------------------


def timeout_ms(config: StoreConfig) -> int:
    return config.parameters.webauthn_challenge_lifetime_seconds * 1000


def descriptor(row: sqlite3.Row) -> CredentialDescriptor:
    transports = [AuthenticatorTransport(t) for t in db.loads(row["transports_json"]) or []]
    return CredentialDescriptor(id=row["credential_id"], transports=transports)


def creation_options(config: StoreConfig, *, challenge: str, user_handle: str, email: str, display_name: str,
                     exclude: Sequence[CredentialDescriptor] = ()) -> CredentialCreationOptions:
    return CredentialCreationOptions(
        rp=RelyingParty(id=config.rp_id, name=RP_NAME),
        user=UserEntity(id=user_handle, name=email, displayName=display_name),
        challenge=challenge,
        pubKeyCredParams=[PubKeyCredParam(alg=int(a)) for a in PUB_KEY_ALGS],
        timeout=timeout_ms(config),
        excludeCredentials=list(exclude),
        authenticatorSelection=AuthenticatorSelection(),
        attestation="none",
    )


def request_options(config: StoreConfig, *, challenge: str, allow: Sequence[CredentialDescriptor]) -> CredentialRequestOptions:
    return CredentialRequestOptions(challenge=challenge, rpId=config.rp_id, allowCredentials=list(allow), timeout=timeout_ms(config))


# --- ceremonies -------------------------------------------------------------------------------


def begin(conn, config: StoreConfig, *, kind: str, now: datetime, account_id: Optional[str] = None, session_id: Optional[str] = None,
          invitation_id: Optional[str] = None, setup_code_id: Optional[str] = None, new_account_id: Optional[str] = None,
          new_user_handle: Optional[str] = None, nickname: Optional[str] = None, step_up: bool = False) -> sqlite3.Row:
    """Insert a ceremony in the caller's transaction and return its row."""
    ceremony_id = new_id("cer")
    expires = now + timedelta(seconds=config.parameters.webauthn_challenge_lifetime_seconds)
    conn.execute("DELETE FROM auth_ceremonies WHERE expires_at < ?", (db.ts(now - CEREMONY_RETENTION),))
    conn.execute(
        "INSERT INTO auth_ceremonies (id, kind, challenge, step_up_challenge, account_id, session_id, invitation_id, setup_code_id, "
        "new_account_id, new_user_handle, nickname, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ceremony_id, kind, crypto.new_challenge(), crypto.new_challenge() if step_up else None, account_id, session_id, invitation_id,
         setup_code_id, new_account_id, new_user_handle, nickname, db.ts(now), db.ts(expires)),
    )
    return conn.execute("SELECT * FROM auth_ceremonies WHERE id = ?", (ceremony_id,)).fetchone()


def consume(conn, ceremony_id: str, *, kind: str, now: datetime) -> sqlite3.Row:
    """Mark the ceremony used (committed at once) and return it; 401 when it is unknown, of another
    kind, already used or expired."""
    with db.transaction(conn):
        row = conn.execute("SELECT * FROM auth_ceremonies WHERE id = ?", (ceremony_id,)).fetchone()
        if row is None or row["kind"] != kind:
            raise verification_failed("unknown_ceremony")
        if row["used_at"] is not None:
            raise verification_failed("challenge_already_used")
        if db.parse_ts(row["expires_at"]) <= now:
            raise verification_failed("challenge_expired")
        updated = conn.execute("UPDATE auth_ceremonies SET used_at = ? WHERE id = ? AND used_at IS NULL", (db.ts(now), ceremony_id)).rowcount
        if updated != 1:
            raise verification_failed("challenge_already_used")
    return row


# --- verification -----------------------------------------------------------------------------


def check_request_origin(config: StoreConfig, origin_header: Optional[str]) -> None:
    """The browser's ``Origin`` header, when sent, must be a Store origin."""
    if origin_header is not None and origin_header.rstrip("/") not in config.allowed_origins:
        raise StoreError(ErrorCode.ORIGIN_NOT_ALLOWED, "Request origin is not the Store origin")


def check_client_data(config: StoreConfig, client_data_b64: str, *, expected_type: str) -> None:
    """Pre-check ``clientDataJSON``: a foreign origin (or a cross-origin iframe) is 403
    ``origin_not_allowed``; malformed data is 401."""
    try:
        data = json.loads(crypto.b64url_decode(client_data_b64))
    except (ValueError, TypeError):
        raise verification_failed("client_data_malformed") from None
    if not isinstance(data, dict) or data.get("type") != expected_type:
        raise verification_failed("client_data_type")
    origin = data.get("origin")
    if not isinstance(origin, str) or origin.rstrip("/") not in config.allowed_origins or data.get("crossOrigin") is True:
        raise StoreError(ErrorCode.ORIGIN_NOT_ALLOWED, "The passkey ceremony ran on an origin that is not the Store origin")


class VerifiedCredential:
    """What Store keeps from a verified registration."""

    def __init__(self, *, credential_id: str, public_key_cose: str, sign_count: int, aaguid: Optional[str], transports: List[str],
                 backup_eligible: bool, backup_state: bool) -> None:
        self.credential_id = credential_id
        self.public_key_cose = public_key_cose
        self.sign_count = sign_count
        self.aaguid = aaguid
        self.transports = transports
        self.backup_eligible = backup_eligible
        self.backup_state = backup_state


def verify_registration(config: StoreConfig, credential: RegistrationCredentialJSON, *, challenge: str) -> VerifiedCredential:
    check_client_data(config, credential.response.clientDataJSON, expected_type="webauthn.create")
    try:
        verified = verify_registration_response(
            credential=credential.model_dump(mode="json", exclude_none=True),
            expected_challenge=crypto.b64url_decode(challenge),
            expected_rp_id=config.rp_id,
            expected_origin=list(config.allowed_origins),
            require_user_presence=True,
            require_user_verification=True,
            supported_pub_key_algs=list(PUB_KEY_ALGS),
        )
    except Exception:  # the library raises several types; none of their text reaches the client
        raise verification_failed("registration_invalid") from None
    aaguid = verified.aaguid if verified.aaguid and len(verified.aaguid) == 36 else None
    return VerifiedCredential(
        credential_id=crypto.b64url(verified.credential_id),
        public_key_cose=crypto.b64url(verified.credential_public_key),
        sign_count=int(verified.sign_count),
        aaguid=aaguid,
        transports=list(credential.response.transports),
        backup_eligible=verified.credential_device_type == CredentialDeviceType.MULTI_DEVICE,
        backup_state=bool(verified.credential_backed_up),
    )


class VerifiedAssertion:
    def __init__(self, *, new_sign_count: int, backup_state: bool) -> None:
        self.new_sign_count = new_sign_count
        self.backup_state = backup_state


def verify_assertion(config: StoreConfig, credential: AuthenticationCredentialJSON, *, challenge: str, stored: sqlite3.Row,
                     user_handle: str) -> VerifiedAssertion:
    """Verify an assertion from the stored credential row ``stored`` (already matched by ID)."""
    check_client_data(config, credential.response.clientDataJSON, expected_type="webauthn.get")
    if credential.response.userHandle is not None and credential.response.userHandle != user_handle:
        raise verification_failed("user_handle_mismatch")
    try:
        verified = verify_authentication_response(
            credential=credential.model_dump(mode="json", exclude_none=True),
            expected_challenge=crypto.b64url_decode(challenge),
            expected_rp_id=config.rp_id,
            expected_origin=list(config.allowed_origins),
            credential_public_key=crypto.b64url_decode(stored["public_key_cose"]),
            credential_current_sign_count=int(stored["sign_count"]),
            require_user_verification=True,
        )
    except Exception:
        raise verification_failed("assertion_invalid") from None
    return VerifiedAssertion(new_sign_count=int(verified.new_sign_count), backup_state=bool(verified.credential_backed_up))


def record_use(conn, stored: sqlite3.Row, assertion: VerifiedAssertion, now: datetime) -> None:
    conn.execute(
        "UPDATE auth_credentials SET sign_count = ?, backup_state = ?, last_used_at = ? WHERE id = ?",
        (max(int(stored["sign_count"]), assertion.new_sign_count), int(assertion.backup_state), db.ts(now), stored["id"]),
    )
