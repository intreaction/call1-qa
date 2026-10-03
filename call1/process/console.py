"""The loopback console credential (``ConsoleCredentialDescriptor`` in the contract).

It authenticates the operator's console to Process only (never to Store: no Store route accepts
it). Process stores only its SHA-256 hash in the protected config file (mode 0600); the token is
shown once when it is issued, as a console URL whose fragment carries it
(``http://127.0.0.1:8020/#console_token=...``, never sent to the server in a URL). The console
sends it as ``X-Call1-Console-Token`` on every write; reads need only loopback.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .config import ProcessConfig, read_config_file, update_config_file

TOKEN_PREFIX = "c1con_"
HEADER = "X-Call1-Console-Token"


def hash_token(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue(config: ProcessConfig, *, rotate: bool = False) -> Tuple[str, Dict[str, Any]]:
    """Create (or, with ``rotate``, replace) the console credential. Returns the token, once."""
    current = read_config_file(config.config_path).get("console_credential")
    if current and not rotate:
        raise FileExistsError("A console credential already exists; rotate it to issue a new one")
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc).isoformat()
    record = {"credential_hash": hash_token(token), "created_at": (current or {}).get("created_at") or now,
              "rotated_at": now if current else None, "installation_id": config.installation_id, "loopback_only": True}
    update_config_file(config.config_path, {"console_credential": record})
    return token, record


def console_url(config: ProcessConfig, token: Optional[str] = None) -> str:
    base = f"http://{'127.0.0.1' if config.bind_host == 'localhost' else config.bind_host}:{config.port}/"
    if config.bind_host == "::1":
        base = f"http://[::1]:{config.port}/"
    return base + (f"#console_token={token}" if token else "")


class ConsoleAuth:
    """Checks the console token against the hash in the config file, re-reading the file when it
    changes (``console-token --rotate`` takes effect without a restart)."""

    def __init__(self, config_path: Path, initial: Optional[Dict[str, Any]] = None) -> None:
        self.config_path = Path(config_path)
        self._mtime: Optional[float] = None
        self._record: Optional[Dict[str, Any]] = dict(initial) if initial else None

    def record(self) -> Optional[Dict[str, Any]]:
        try:
            mtime = self.config_path.stat().st_mtime
        except OSError:
            return self._record
        if mtime != self._mtime:
            self._mtime = mtime
            self._record = read_config_file(self.config_path).get("console_credential") or None
        return self._record

    @property
    def configured(self) -> bool:
        record = self.record()
        return bool(record and record.get("credential_hash"))

    def check(self, token: Optional[str]) -> bool:
        record = self.record()
        if not token or not record or not record.get("credential_hash"):
            return False
        return hmac.compare_digest(hash_token(token), str(record["credential_hash"]))
