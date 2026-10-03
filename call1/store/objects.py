"""Content-addressed local object store, upload sessions and download grants.

Only Store opens this directory (architecture rule 1). Objects are addressed by
``sha256:<hex>`` (the contract's ``Sha256Digest``) and stored once at
``objects/sha256/<hex[:2]>/<hex[2:4]>/<hex>``; writes go through ``objects/staging`` and an atomic
rename, so a reader never sees a partial object. Nothing outside Store ever sees a path.

Uploads (behind ``artifacts.UploadGrant``)
    1. The owning area calls ``store.objects.create_upload(conn, ...)`` inside its request
       transaction and gets an ``UploadTicket`` (upload ID, one-time URL, headers, expiry).
    2. The client PUTs the bytes to the ticket URL (``routes/transfer.py``), which streams them into
       staging, hashing as it goes.
    3. The commit route calls ``store.objects.commit_upload(conn, upload_id, checksum=..,
       size_bytes=.., validate=..)``, which checks the declared checksum and size against the grant
       and the received bytes, optionally runs ``validate(bytes)`` (JSON kinds:
       ``artifacts.canonical_content``), moves the bytes into the store and marks the session
       committed. A repeated commit returns the committed record (natural idempotency by upload ID).
       The owning area then writes its artifact row in the same transaction.

Downloads (behind ``artifacts.ContentGrant``)
    ``store.objects.content_grant(conn, ...)`` signs a short-lived, object-scoped URL with a
    per-dataset secret (HMAC-SHA256); ``routes/transfer.py`` verifies it and streams the object
    (Range supported).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Callable, Optional

from starlette.responses import FileResponse

from call1.contracts.common import ContractParameters, ServiceScope
from call1.contracts.errors import ErrorCode

from . import db
from .errors import StoreError
from .ids import new_id
from .principals import generate_secret, hash_secret, secrets_equal

TRANSFER_PREFIX = "/store/transfer"
TRANSFER_UPLOADS = f"{TRANSFER_PREFIX}/uploads"
TRANSFER_DOWNLOADS = f"{TRANSFER_PREFIX}/downloads"
_CHUNK = 1024 * 1024


def _hex_of(checksum: str) -> str:
    if not checksum.startswith("sha256:") or len(checksum) != 71:
        raise ValueError(f"not a sha256 checksum: {checksum[:80]!r}")
    hexpart = checksum[7:]
    int(hexpart, 16)
    return hexpart.lower()


def sha256_checksum(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class StoredObject:
    checksum: str
    size_bytes: int


@dataclass(frozen=True)
class UploadTicket:
    """What the owning area needs to build ``artifacts.UploadGrant``. ``token`` is one-time."""

    upload_id: str
    token: str
    url: str
    headers: dict
    expires_at: datetime
    max_bytes: int


@dataclass(frozen=True)
class UploadRecord:
    upload_id: str
    purpose: str
    required_scope: Optional[ServiceScope]
    installation_id: Optional[str]
    expected_checksum: str
    expected_size: int
    content_type: str
    metadata: dict
    status: str
    received_checksum: Optional[str]
    received_size: Optional[int]
    created_at: datetime
    expires_at: datetime
    committed_at: Optional[datetime]

    @classmethod
    def from_row(cls, row) -> "UploadRecord":
        return cls(
            upload_id=row["upload_id"],
            purpose=row["purpose"],
            required_scope=ServiceScope(row["required_scope"]) if row["required_scope"] else None,
            installation_id=row["installation_id"],
            expected_checksum=row["expected_checksum"],
            expected_size=row["expected_size"],
            content_type=row["content_type"],
            metadata=db.loads(row["metadata_json"]) or {},
            status=row["status"],
            received_checksum=row["received_checksum"],
            received_size=row["received_size"],
            created_at=db.parse_ts(row["created_at"]),
            expires_at=db.parse_ts(row["expires_at"]),
            committed_at=db.parse_ts(row["committed_at"]),
        )


@dataclass(frozen=True)
class DownloadTicket:
    url: str
    expires_at: datetime


class ObjectStore:
    def __init__(self, root: Path, *, base_url: str, parameters: ContractParameters, max_upload_bytes: int) -> None:
        self.root = Path(root)
        self.base_url = base_url.rstrip("/")
        self.parameters = parameters
        self.max_upload_bytes = max_upload_bytes
        self._objects = self.root / "sha256"
        self._staging = self.root / "staging"
        for directory in (self.root, self._objects, self._staging):
            directory.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass

    # --- content-addressed objects ---------------------------------------------------------

    def _path(self, checksum: str) -> Path:
        hexpart = _hex_of(checksum)
        return self._objects / hexpart[:2] / hexpart[2:4] / hexpart

    def exists(self, checksum: str) -> bool:
        return self._path(checksum).is_file()

    def size(self, checksum: str) -> int:
        return self._path(checksum).stat().st_size

    def put_bytes(self, data: bytes) -> StoredObject:
        checksum = sha256_checksum(data)
        target = self._path(checksum)
        if not target.is_file():
            fd, tmp = tempfile.mkstemp(dir=self._staging, prefix="put-")
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self._install(Path(tmp), target)
        return StoredObject(checksum, len(data))

    def put_file(self, source: Path) -> StoredObject:
        """Hash ``source`` and move it into the store (it is consumed)."""
        digest, size = _hash_file(source)
        checksum = "sha256:" + digest
        target = self._path(checksum)
        if target.is_file():
            source.unlink(missing_ok=True)
        else:
            self._install(source, target)
        return StoredObject(checksum, size)

    def _install(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass

    def read_bytes(self, checksum: str) -> bytes:
        try:
            return self._path(checksum).read_bytes()
        except FileNotFoundError:
            raise StoreError(ErrorCode.NOT_FOUND, "Object bytes not found", details={"checksum": checksum}) from None

    def open(self, checksum: str) -> BinaryIO:
        try:
            return self._path(checksum).open("rb")
        except FileNotFoundError:
            raise StoreError(ErrorCode.NOT_FOUND, "Object bytes not found", details={"checksum": checksum}) from None

    def file_response(self, checksum: str, media_type: str, *, filename: Optional[str] = None) -> FileResponse:
        """Stream an object (HTTP Range supported by Starlette's FileResponse)."""
        path = self._path(checksum)
        if not path.is_file():
            raise StoreError(ErrorCode.NOT_FOUND, "Object bytes not found", details={"checksum": checksum})
        headers = {"ETag": f'"{_hex_of(checksum)}"', "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}
        return FileResponse(path, media_type=media_type, filename=filename, headers=headers)

    def verify(self, checksum: str) -> bool:
        path = self._path(checksum)
        return path.is_file() and "sha256:" + _hash_file(path)[0] == checksum

    def delete(self, checksum: str) -> bool:
        path = self._path(checksum)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    # --- upload sessions -------------------------------------------------------------------

    def staging_path(self, upload_id: str) -> Path:
        if not upload_id.replace("_", "").isalnum():
            raise ValueError("bad upload id")
        return self._staging / f"{upload_id}.part"

    def create_upload(self, conn, *, purpose: str, expected_checksum: str, expected_size: int, content_type: str,
                      required_scope: Optional[ServiceScope] = None, installation_id: Optional[str] = None,
                      metadata: Optional[dict] = None) -> UploadTicket:
        _hex_of(expected_checksum)
        if expected_size > self.max_upload_bytes:
            raise StoreError(ErrorCode.PAYLOAD_TOO_LARGE, "Object is larger than Store accepts", details={"max_bytes": self.max_upload_bytes})
        upload_id = new_id("upl")
        token = generate_secret(32)
        now = conn.now()
        expires = now + timedelta(seconds=self.parameters.upload_grant_lifetime_seconds)
        with db.transaction(conn):
            conn.execute(
                "INSERT INTO object_uploads (upload_id, token_hash, purpose, required_scope, installation_id, expected_checksum, "
                "expected_size, content_type, metadata_json, status, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (upload_id, hash_secret(token), purpose, required_scope.value if required_scope else None, installation_id,
                 expected_checksum, expected_size, content_type, db.dumps(metadata or {}), db.ts(now), db.ts(expires)),
            )
        url = f"{self.base_url}{TRANSFER_UPLOADS}/{upload_id}?token={token}"
        headers = {"Content-Type": content_type, "Content-Length": str(expected_size)}
        return UploadTicket(upload_id, token, url, headers, expires, max(expected_size, 1))

    def get_upload(self, conn, upload_id: str) -> Optional[UploadRecord]:
        row = conn.execute("SELECT * FROM object_uploads WHERE upload_id = ?", (upload_id,)).fetchone()
        return None if row is None else UploadRecord.from_row(row)

    def authorize_transfer(self, conn, upload_id: str, token: str) -> UploadRecord:
        """Check a PUT against its session: known, token matches, not expired, not committed."""
        row = conn.execute("SELECT * FROM object_uploads WHERE upload_id = ?", (upload_id,)).fetchone()
        if row is None or not secrets_equal(row["token_hash"], hash_secret(token or "")):
            raise StoreError(ErrorCode.NOT_FOUND, "Upload not found")
        record = UploadRecord.from_row(row)
        if record.status == "committed":
            raise StoreError(ErrorCode.CONFLICT, "Upload already committed", details={"reason": "committed"})
        if conn.now() > record.expires_at:
            raise StoreError(ErrorCode.UPLOAD_EXPIRED, "Upload grant expired; request a new one")
        return record

    def record_received(self, conn, upload_id: str, checksum: str, size: int) -> None:
        with db.transaction(conn):
            conn.execute(
                "UPDATE object_uploads SET status = 'received', received_checksum = ?, received_size = ? WHERE upload_id = ? AND status != 'committed'",
                (checksum, size, upload_id),
            )

    def commit_upload(self, conn, upload_id: str, *, checksum: str, size_bytes: int,
                      validate: Optional[Callable[[bytes], None]] = None) -> UploadRecord:
        """Verify and commit an upload; call inside the owning area's transaction."""
        with db.transaction(conn):
            record = self.get_upload(conn, upload_id)
            if record is None:
                raise StoreError(ErrorCode.NOT_FOUND, "Upload not found")
            declared_ok = checksum == record.expected_checksum and size_bytes == record.expected_size
            if record.status == "committed":
                if not declared_ok:
                    raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "Commit differs from the committed upload", details={"reason": "commit_differs_from_grant"})
                return record
            if conn.now() > record.expires_at:
                raise StoreError(ErrorCode.UPLOAD_EXPIRED, "Upload grant expired; request a new one")
            if not declared_ok:
                raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "Declared checksum or size differs from the grant", details={"reason": "commit_differs_from_grant"})
            if record.status != "received":
                raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "No bytes were uploaded for this grant", details={"reason": "no_bytes_received"})
            if record.received_checksum != record.expected_checksum or record.received_size != record.expected_size:
                raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "Uploaded bytes do not match the declared checksum", details={"reason": "uploaded_bytes_differ", "received_size": record.received_size})
            staging = self.staging_path(upload_id)
            if staging.is_file():
                digest, size = _hash_file(staging)
                if "sha256:" + digest != record.expected_checksum or size != record.expected_size:
                    raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "Uploaded bytes do not match the declared checksum", details={"reason": "uploaded_bytes_differ"})
                if validate is not None:
                    validate(staging.read_bytes())
                self.put_file(staging)
            elif self.verify(record.expected_checksum):
                # A previous commit moved the bytes, then its transaction rolled back.
                if validate is not None:
                    validate(self.read_bytes(record.expected_checksum))
            else:
                raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "No bytes were uploaded for this grant", details={"reason": "no_bytes_received"})
            now = conn.now()
            conn.execute("UPDATE object_uploads SET status = 'committed', committed_at = ? WHERE upload_id = ?", (db.ts(now), upload_id))
            return self.get_upload(conn, upload_id)

    def sweep_expired_uploads(self, conn) -> int:
        """Forget uncommitted sessions past their expiry and delete their staged bytes."""
        now = db.ts(conn.now())
        with db.transaction(conn):
            rows = conn.execute("SELECT upload_id FROM object_uploads WHERE status != 'committed' AND expires_at < ?", (now,)).fetchall()
            for row in rows:
                self.staging_path(row["upload_id"]).unlink(missing_ok=True)
            conn.execute("DELETE FROM object_uploads WHERE status != 'committed' AND expires_at < ?", (now,))
        return len(rows)

    # --- download grants -------------------------------------------------------------------

    def content_grant(self, conn, *, artifact_id: str, checksum: str, content_type: str) -> DownloadTicket:
        expires = conn.now() + timedelta(seconds=self.parameters.download_grant_lifetime_seconds)
        payload = {"a": artifact_id, "c": _hex_of(checksum), "t": content_type, "e": int(expires.timestamp())}
        body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        token = f"{body}.{_b64(self._sign(conn, body))}"
        return DownloadTicket(f"{self.base_url}{TRANSFER_DOWNLOADS}/{token}", expires)

    def verify_download_token(self, conn, token: str) -> tuple[str, str, str]:
        """(artifact_id, checksum, content_type) of a valid, unexpired grant token."""
        body, _, signature = token.partition(".")
        try:
            expected = _b64(self._sign(conn, body))
            if not signature or not hmac.compare_digest(signature, expected):
                raise ValueError
            payload = json.loads(_unb64(body))
            checksum = "sha256:" + payload["c"]
            _hex_of(checksum)
        except (ValueError, KeyError, TypeError):
            raise StoreError(ErrorCode.FORBIDDEN, "Download grant is not valid", details={"reason": "bad_grant"}) from None
        if conn.now().timestamp() > payload["e"]:
            raise StoreError(ErrorCode.FORBIDDEN, "Download grant expired; request a new one", details={"reason": "grant_expired"})
        return payload["a"], checksum, payload["t"]

    def _sign(self, conn, body: str) -> bytes:
        secret = db.get_meta(conn, db.META_TRANSFER_SECRET)
        if not secret:
            raise RuntimeError("transfer secret missing: Database.initialize() was not run")
        return hmac.new(bytes.fromhex(secret), body.encode(), hashlib.sha256).digest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


__all__ = [
    "ObjectStore", "StoredObject", "UploadTicket", "UploadRecord", "DownloadTicket", "sha256_checksum",
    "TRANSFER_PREFIX", "TRANSFER_UPLOADS", "TRANSFER_DOWNLOADS",
]
