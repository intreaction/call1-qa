"""Store configuration: where its data lives, the fixed hostname, the port, and dev mode.

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| ``CALL1_STORE_DATA`` | ``data/store`` | Store's data directory: ``store.db`` (SQLite, WAL) and ``objects/`` |
| ``CALL1_STORE_HOSTNAME`` | ``localhost`` | The fixed Store hostname (the WebAuthn relying-party ID) |
| ``CALL1_STORE_PORT`` | ``8010`` | Port ``serve`` listens on |
| ``CALL1_STORE_BIND`` | ``127.0.0.1`` | Interface ``serve`` binds; dev mode refuses anything but loopback |
| ``CALL1_STORE_DEV`` | on when the hostname is ``localhost`` | Dev mode: plain ``http://localhost`` (see below) |
| ``CALL1_STORE_DEV_ORIGINS`` | none | Extra ``http://localhost:<port>`` origins accepted in dev mode (the Vite dev server) |
| ``CALL1_STORE_PUBLIC_URL`` | derived | The URL clients use to reach Store, when it differs from ``https://<hostname>:<port>`` (a TLS proxy on 443) |
| ``CALL1_STORE_TLS_CERT`` / ``CALL1_STORE_TLS_KEY`` | none | Certificate and key ``serve`` hands to uvicorn outside dev mode |
| ``CALL1_STORE_PARAMETERS`` | contract defaults | JSON object overriding ``ContractParameters`` fields (tests, tuning) |
| ``CALL1_STORE_DEMO`` | off | DEMO MODE: persona sign-in without a passkey (``auth/demo.py``); dev mode only |

**Dev mode** exists because Stage 2 ships passkeys on ``localhost``, which is a WebAuthn secure
context over plain HTTP, while the contract's hostname rule rejects ``localhost`` and every URL it
describes is ``https://``. In dev mode Store serves ``http://localhost:<port>`` on loopback only,
names its session cookie ``call1_session`` (no ``__Host-`` prefix, no ``Secure``; everything else
as ``auth.SESSION_COOKIE``), uses ``localhost`` as the relying-party ID, and emits ``http://`` grant
URLs. ``call1.store.devmode`` is the one place those values cross the contract models. Outside dev
mode the hostname must pass ``validate_store_hostname`` and everything is exactly the contract.

**Demo mode** (``CALL1_STORE_DEMO=1``) adds ``/demo/status`` and ``/demo/sign-in`` for class demos
(``call1.store.auth.demo``). It exists only on top of dev mode: with dev mode off, ``StoreConfig``
refuses it, so ``serve`` never starts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Optional, Tuple
from urllib.parse import urlsplit

from call1.contracts.auth import SESSION_COOKIE
from call1.contracts.common import CONTRACT_PARAMETERS, ContractParameters, validate_store_hostname

DEFAULT_DATA_DIR = Path("data/store")
DEFAULT_HOSTNAME = "localhost"
DEFAULT_PORT = 8010
DEFAULT_BIND = "127.0.0.1"
DEFAULT_MAINTENANCE_SECONDS = 60.0
DEV_HOSTNAME = "localhost"
DEV_COOKIE_NAME = "call1_session"
LOOPBACK_BINDS = ("127.0.0.1", "::1", "localhost")


class ConfigError(ValueError):
    """The Store configuration is inconsistent; the message says which setting to change."""


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class StoreConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    hostname: str = DEFAULT_HOSTNAME
    port: int = DEFAULT_PORT
    bind_host: str = DEFAULT_BIND
    dev_mode: Optional[bool] = None
    """None: on exactly when ``hostname`` is ``localhost``."""
    dev_origins: Tuple[str, ...] = ()
    public_url: Optional[str] = None
    tls_cert_file: Optional[Path] = None
    tls_key_file: Optional[Path] = None
    parameters: ContractParameters = field(default_factory=lambda: CONTRACT_PARAMETERS)
    max_upload_bytes: int = 4 * 1024 ** 3
    demo_mode: bool = False
    """DEMO MODE (``CALL1_STORE_DEMO``): persona sign-in without a passkey. Requires dev mode."""
    maintenance_interval_seconds: float = 0.0
    """How often the running app sweeps expired leases/claims/uploads and prunes the change feed.
    0 turns the background task off (tests call ``maintenance.run_once`` directly)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser().resolve())
        host = self.hostname.strip().rstrip(".").lower()
        dev = (host == DEV_HOSTNAME) if self.dev_mode is None else bool(self.dev_mode)
        object.__setattr__(self, "dev_mode", dev)
        object.__setattr__(self, "demo_mode", bool(self.demo_mode))
        if self.demo_mode and not dev:
            raise ConfigError("CALL1_STORE_DEMO works only in dev mode (http://localhost on a loopback bind); "
                              "unset CALL1_STORE_DEMO or set CALL1_STORE_HOSTNAME=localhost")
        if dev:
            if host != DEV_HOSTNAME:
                raise ConfigError("dev mode serves http://localhost only; set CALL1_STORE_HOSTNAME=localhost or turn CALL1_STORE_DEV off")
            if self.bind_host not in LOOPBACK_BINDS:
                raise ConfigError("dev mode is plain HTTP, so it binds loopback only (CALL1_STORE_BIND=127.0.0.1)")
            for origin in self.dev_origins:
                parts = urlsplit(origin)
                if parts.scheme != "http" or parts.hostname != "localhost" or parts.path not in ("", "/"):
                    raise ConfigError(f"dev origin {origin!r} must be http://localhost:<port>")
        else:
            try:
                host = validate_store_hostname(host)
            except ValueError as exc:
                raise ConfigError(f"CALL1_STORE_HOSTNAME: {exc} (use localhost for dev mode)") from None
            if self.dev_origins:
                raise ConfigError("CALL1_STORE_DEV_ORIGINS applies to dev mode only")
            if self.public_url is not None:
                parts = urlsplit(self.public_url)
                if parts.scheme != "https" or (parts.hostname or "").lower() != host or parts.path not in ("", "/"):
                    raise ConfigError("CALL1_STORE_PUBLIC_URL must be https://<store-hostname>[:port]")
        object.__setattr__(self, "hostname", host)
        if not (0 < int(self.port) < 65536):
            raise ConfigError("CALL1_STORE_PORT must be a TCP port")
        if self.maintenance_interval_seconds < 0:
            raise ConfigError("CALL1_STORE_MAINTENANCE_SECONDS must be 0 (off) or a positive number of seconds")

    # --- derived values -------------------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self.data_dir / "store.db"

    @property
    def objects_dir(self) -> Path:
        return self.data_dir / "objects"

    @property
    def public_base_url(self) -> str:
        """Origin clients use to reach Store, without a trailing slash."""
        if self.public_url:
            return self.public_url.rstrip("/")
        if self.dev_mode:
            return f"http://localhost:{self.port}"
        return f"https://{self.hostname}" + ("" if self.port == 443 else f":{self.port}")

    @property
    def rp_id(self) -> str:
        return self.hostname

    @property
    def allowed_origins(self) -> Tuple[str, ...]:
        return (self.public_base_url,) + tuple(o.rstrip("/") for o in self.dev_origins)

    @property
    def cookie_name(self) -> str:
        return DEV_COOKIE_NAME if self.dev_mode else SESSION_COOKIE.name

    @property
    def cookie_secure(self) -> bool:
        return not self.dev_mode

    @property
    def csrf_header(self) -> str:
        return SESSION_COOKIE.csrf_header

    def with_overrides(self, **changes) -> "StoreConfig":
        return replace(self, **changes)

    # --- construction ---------------------------------------------------------------------

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "StoreConfig":
        env = os.environ if env is None else env
        dev_raw = env.get("CALL1_STORE_DEV")
        params = CONTRACT_PARAMETERS
        if env.get("CALL1_STORE_PARAMETERS"):
            try:
                overrides = json.loads(env["CALL1_STORE_PARAMETERS"])
            except json.JSONDecodeError as exc:
                raise ConfigError(f"CALL1_STORE_PARAMETERS is not JSON: {exc}") from None
            params = ContractParameters.model_validate({**CONTRACT_PARAMETERS.model_dump(), **overrides})
        cert = env.get("CALL1_STORE_TLS_CERT")
        key = env.get("CALL1_STORE_TLS_KEY")
        origins = tuple(o.strip() for o in env.get("CALL1_STORE_DEV_ORIGINS", "").split(",") if o.strip())
        try:
            port = int(env.get("CALL1_STORE_PORT", DEFAULT_PORT))
        except ValueError:
            raise ConfigError("CALL1_STORE_PORT must be an integer") from None
        try:
            maintenance = float(env.get("CALL1_STORE_MAINTENANCE_SECONDS", DEFAULT_MAINTENANCE_SECONDS))
        except ValueError:
            raise ConfigError("CALL1_STORE_MAINTENANCE_SECONDS must be a number") from None
        return cls(
            data_dir=Path(env.get("CALL1_STORE_DATA", str(DEFAULT_DATA_DIR))),
            hostname=env.get("CALL1_STORE_HOSTNAME", DEFAULT_HOSTNAME),
            port=port,
            bind_host=env.get("CALL1_STORE_BIND", DEFAULT_BIND),
            dev_mode=None if dev_raw is None or dev_raw == "" else _truthy(dev_raw),
            dev_origins=origins,
            public_url=env.get("CALL1_STORE_PUBLIC_URL") or None,
            tls_cert_file=Path(cert) if cert else None,
            tls_key_file=Path(key) if key else None,
            parameters=params,
            demo_mode=_truthy(env.get("CALL1_STORE_DEMO", "")),
            maintenance_interval_seconds=maintenance,
        )

    @classmethod
    def for_tests(cls, data_dir: Path, **overrides) -> "StoreConfig":
        """A dev-mode config rooted in a temporary directory."""
        return cls(data_dir=Path(data_dir), **overrides)
