"""A software WebAuthn authenticator for Store tests (P-256, attestation ``none``).

It produces exactly the JSON ``@simplewebauthn/browser`` sends (``startRegistration`` /
``startAuthentication``), so tests run full enrollment, sign-in and step-up ceremonies against the
real ``webauthn`` verification. Knobs make it misbehave on purpose: a foreign origin or RP ID,
no user verification, a replayed or stale signature counter, a tampered signature.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cbor2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

ORIGIN = "http://localhost:8010"
RP_ID = "localhost"

FLAG_UP = 0x01
FLAG_UV = 0x04
FLAG_BE = 0x08
FLAG_BS = 0x10
FLAG_AT = 0x40


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


@dataclass
class SoftCredential:
    credential_id: bytes
    private_key: ec.EllipticCurvePrivateKey
    rp_id: str
    user_handle: str
    sign_count: int = 0

    @property
    def id(self) -> str:
        return b64url(self.credential_id)

    def cose_public_key(self) -> bytes:
        numbers = self.private_key.public_key().public_numbers()
        return cbor2.dumps({1: 2, 3: -7, -1: 1, -2: numbers.x.to_bytes(32, "big"), -3: numbers.y.to_bytes(32, "big")})


@dataclass
class SoftAuthenticator:
    """One authenticator (a security key or a platform passkey) holding credentials by RP ID."""

    transports: List[str] = field(default_factory=lambda: ["usb", "nfc"])
    counter: bool = True
    """True: a signature counter that increments on every assertion (security keys). False: always 0 (synced passkeys)."""
    credentials: Dict[str, SoftCredential] = field(default_factory=dict)

    @staticmethod
    def _client_data(kind: str, challenge: str, origin: str, cross_origin: bool = False) -> bytes:
        return json.dumps({"type": kind, "challenge": challenge, "origin": origin, "crossOrigin": cross_origin}, separators=(",", ":")).encode()

    def register(self, options: dict, *, origin: str = ORIGIN, rp_id: Optional[str] = None, user_verified: bool = True,
                 challenge: Optional[str] = None, credential_id: Optional[bytes] = None) -> dict:
        """``startRegistration(options)``: create a credential and return the registration JSON."""
        rp_id = rp_id or options["rp"]["id"]
        cred = SoftCredential(credential_id=credential_id or os.urandom(32), private_key=ec.generate_private_key(ec.SECP256R1()),
                              rp_id=rp_id, user_handle=options["user"]["id"])
        client_data = self._client_data("webauthn.create", challenge or options["challenge"], origin)
        flags = FLAG_UP | FLAG_AT | (FLAG_UV if user_verified else 0)
        attested = b"\x00" * 16 + struct.pack(">H", len(cred.credential_id)) + cred.credential_id + cred.cose_public_key()
        auth_data = hashlib.sha256(rp_id.encode()).digest() + bytes([flags]) + struct.pack(">I", 0) + attested
        attestation_object = cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth_data})
        self.credentials[cred.id] = cred
        return {
            "id": cred.id,
            "rawId": cred.id,
            "type": "public-key",
            "response": {
                "clientDataJSON": b64url(client_data),
                "attestationObject": b64url(attestation_object),
                "transports": list(self.transports) + ["some-future-transport"],
                "publicKeyAlgorithm": -7,
                "authenticatorData": b64url(auth_data),
            },
            "clientExtensionResults": {},
            "authenticatorAttachment": "cross-platform",
        }

    def assert_(self, options: dict, *, origin: str = ORIGIN, rp_id: Optional[str] = None, user_verified: bool = True,
                challenge: Optional[str] = None, credential: Optional[SoftCredential] = None, sign_count: Optional[int] = None,
                tamper: bool = False) -> dict:
        """``startAuthentication(options)``: sign with the first allowed credential this authenticator holds."""
        if credential is None:
            allowed = [c["id"] for c in options.get("allowCredentials", [])]
            credential = next((self.credentials[i] for i in allowed if i in self.credentials), None)
            if credential is None:
                raise LookupError("this authenticator holds none of the allowed credentials")
        rp_id = rp_id or options.get("rpId") or credential.rp_id
        if sign_count is None:
            if self.counter:
                credential.sign_count += 1
            sign_count = credential.sign_count
        client_data = self._client_data("webauthn.get", challenge or options["challenge"], origin)
        flags = FLAG_UP | (FLAG_UV if user_verified else 0)
        auth_data = hashlib.sha256(rp_id.encode()).digest() + bytes([flags]) + struct.pack(">I", sign_count)
        signature = credential.private_key.sign(auth_data + hashlib.sha256(client_data).digest(), ec.ECDSA(hashes.SHA256()))
        if tamper:
            signature = signature[:-1] + bytes([signature[-1] ^ 0x01])
        return {
            "id": credential.id,
            "rawId": credential.id,
            "type": "public-key",
            "response": {
                "clientDataJSON": b64url(client_data),
                "authenticatorData": b64url(auth_data),
                "signature": b64url(signature),
                "userHandle": credential.user_handle,
            },
            "clientExtensionResults": {},
            "authenticatorAttachment": "cross-platform",
        }
