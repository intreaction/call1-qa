"""Process configuration: the Store connection, the service key, the loopback bind, scratch space
and worker tuning.

The config file is JSON, mode 0600, at ``CALL1_PROCESS_CONFIG`` (default
``data/process/config.json``). ``python -m call1.store issue-service-key --installation <name>``
writes the first four keys; everything else is optional:

| Key | Default | Meaning |
|---|---|---|
| ``store_url`` | none | Store's base URL. ``https://`` always, or ``http://localhost`` in Store dev mode |
| ``installation_id`` / ``service_key_id`` / ``service_key`` | none | The Process installation and its ``c1sk_`` key |
| ``bind`` / ``port`` | ``127.0.0.1`` / ``8020`` | Where ``serve`` listens; loopback only |
| ``data_dir`` | ``data/process`` | Scratch (``scratch/``), the completion spool (``spool/``) and the ingest ledger |
| ``handlers`` | ``real`` | ``fake`` or ``real`` (``CALL1_PROCESS_HANDLERS`` overrides) |
| ``worker_id`` | ``<hostname>-process`` | Recorded on every claim and attempt |
| ``primary_host`` | ``true`` | Offer primary-host (ML) jobs; switched off automatically when Store says the key is not primary |
| ``rubric_id`` | ``call1_standard_v2`` | The published rubric new recordings are scored with |
| ``model_defaults`` | catalog defaults | Purpose to catalog entry ID (``SET_MODEL_DEFAULTS``) |
| ``escalation_entry_id`` | none | Catalog entry for QA escalations when a criterion does not name one (none: no escalation, as before the split) |
| ``slots`` | ``{"cpu_io": 4, "torch": 1, "mlx": 1, "outbound": 2}`` | Local resource slots; ``mlx`` is always 1 (the shared unified-memory slot) |
| ``stages`` | all on | ``{"summary": true, "contact_signals": true, "embeddings": true}`` |
| ``summary_batch_turns`` | ``60`` | Turns per summary segment (the pre-split ``batch_turns``) |
| ``mask_model_text`` | ``store`` | Mask the text-model prompts (QA, summary, contact signals) on the appliance route: ``store`` follows Store's text masking (``MaskingSettings.mask_reviewer_reads``, the legacy ``redaction.text``, on by default), ``on`` or ``off`` override it. Non-appliance routes are always masked per ``masked_route_classes`` |
| ``scratch_retention_hours`` / ``scratch_max_bytes`` | ``24`` / 5 GiB | Bounded scratch retention |
| ``poll_interval_seconds`` | ``2`` | Idle claim polling |
| ``shutdown_grace_seconds`` | ``30`` | How long a graceful stop waits for running jobs |
| ``console_credential`` | none | ``{credential_hash, created_at, rotated_at}``: the loopback console credential, stored hashed |
| ``training`` | off | On-device training of the customer LoRA (decision 28): ``enabled``, ``schedule`` ``{frequency, weekday, time}``, ``min_new_labels``, ``max_duration_minutes``, ``only_when_idle``, and the config-only knobs. ``training/settings.py`` documents every key; the console's Settings tab writes it |

Environment overrides: ``CALL1_PROCESS_CONFIG``, ``CALL1_PROCESS_HANDLERS``, ``CALL1_PROCESS_DATA``,
``CALL1_PROCESS_PORT``, ``CALL1_PROCESS_BIND``, ``CALL1_PROCESS_WORKER_ID``,
``CALL1_PROCESS_STORE_URL``, ``CALL1_PROCESS_MASK_MODEL_TEXT``, ``CALL1_PROCESS_TRAINER`` (``fake`` or
``mlx_lm``: the on-device trainer).
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

DEFAULT_CONFIG_PATH = Path("data/process/config.json")
DEFAULT_DATA_DIR = Path("data/process")
DEFAULT_PORT = 8020
DEFAULT_BIND = "127.0.0.1"
DEFAULT_RUBRIC_ID = "call1_standard_v2"
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
HANDLER_MODES = ("fake", "real")
MASK_MODEL_TEXT_MODES = ("store", "on", "off")


class ConfigError(ValueError):
    """The Process configuration is missing or inconsistent; the message says what to change."""


@dataclass(frozen=True)
class SlotSizes:
    """Local resource slots. ``mlx`` is the one shared unified-memory inference slot (MLX and
    appliance Ollama); ``torch`` runs the tone and sentiment models; ``cpu_io`` runs code stages and
    audio validation; ``outbound`` is the pool size per outbound connection (customer LAN, BYOK,
    Pro1), none of which Stage 2 admits."""

    cpu_io: int = 4
    torch: int = 1
    mlx: int = 1
    outbound: int = 2

    def __post_init__(self) -> None:
        if self.mlx != 1:
            raise ConfigError("slots.mlx is the one shared unified-memory slot and must be 1")
        for name in ("cpu_io", "torch", "outbound"):
            value = getattr(self, name)
            if not isinstance(value, int) or not 0 <= value <= 64:
                raise ConfigError(f"slots.{name} must be an integer from 0 to 64")


@dataclass(frozen=True)
class Stages:
    summary: bool = True
    contact_signals: bool = True
    embeddings: bool = True


def _is_loopback_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in LOOPBACK_HOSTS


def validate_store_url(url: str) -> str:
    """HTTPS always; plain HTTP only to a loopback Store in dev mode (the Store README's documented
    deviation). No credentials, query or fragment."""
    value = (url or "").strip().rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        raise ConfigError("store_url must be an https:// URL (or http://localhost for a dev-mode Store)")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ConfigError("store_url carries no credentials, query or fragment")
    if parts.scheme == "http" and not _is_loopback_url(value):
        raise ConfigError("Plain HTTP is allowed only to a dev-mode Store on localhost; use https://")
    return value


@dataclass(frozen=True)
class ProcessConfig:
    config_path: Path = DEFAULT_CONFIG_PATH
    store_url: Optional[str] = None
    installation_id: Optional[str] = None
    service_key_id: Optional[str] = None
    service_key: Optional[str] = field(default=None, repr=False)
    bind_host: str = DEFAULT_BIND
    port: int = DEFAULT_PORT
    data_dir: Path = DEFAULT_DATA_DIR
    handlers: str = "real"
    worker_id: str = ""
    primary_host: bool = True
    rubric_id: str = DEFAULT_RUBRIC_ID
    model_defaults: Mapping[str, str] = field(default_factory=dict)
    escalation_entry_id: Optional[str] = None
    slots: SlotSizes = field(default_factory=SlotSizes)
    stages: Stages = field(default_factory=Stages)
    summary_batch_turns: int = 60
    mask_model_text: str = "store"
    scratch_retention_seconds: float = 24 * 3600
    scratch_max_bytes: int = 5 * 1024 ** 3
    poll_interval_seconds: float = 2.0
    shutdown_grace_seconds: float = 30.0
    console_credential: Optional[Mapping[str, Any]] = None
    training: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path).expanduser())
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser())
        if self.store_url is not None:
            object.__setattr__(self, "store_url", validate_store_url(self.store_url))
        if self.bind_host not in LOOPBACK_HOSTS:
            raise ConfigError("Process serves its console on loopback only (bind 127.0.0.1)")
        if not 0 < int(self.port) < 65536:
            raise ConfigError("port must be a TCP port")
        if self.handlers not in HANDLER_MODES:
            raise ConfigError(f"handlers must be one of {', '.join(HANDLER_MODES)}")
        if not self.worker_id:
            object.__setattr__(self, "worker_id", default_worker_id())
        if self.mask_model_text not in MASK_MODEL_TEXT_MODES:
            raise ConfigError(f"mask_model_text must be one of {', '.join(MASK_MODEL_TEXT_MODES)}")
        if not 1 <= int(self.summary_batch_turns) <= 500:
            raise ConfigError("summary_batch_turns must be from 1 to 500")
        if self.poll_interval_seconds <= 0 or self.shutdown_grace_seconds < 0:
            raise ConfigError("poll_interval_seconds must be positive and shutdown_grace_seconds not negative")
        from .training.settings import SettingsError, TrainingSettings

        try:
            TrainingSettings.from_mapping(self.training)
        except SettingsError as exc:
            raise ConfigError(str(exc)) from None

    # --- derived -----------------------------------------------------------------------------

    @property
    def configured(self) -> bool:
        """True once the Store URL and a service key are present."""
        return bool(self.store_url and self.service_key)

    @property
    def dev_store(self) -> bool:
        """A plain-HTTP loopback Store (Store dev mode)."""
        return bool(self.store_url) and self.store_url.startswith("http://")

    @property
    def scratch_dir(self) -> Path:
        return self.data_dir / "scratch"

    @property
    def spool_dir(self) -> Path:
        return self.data_dir / "spool"

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "conversations.jsonl"

    def evaluate_url(self, call_id: Optional[str] = None) -> Optional[str]:
        """Deep link into Evaluate (served by Store at ``/``)."""
        if not self.store_url:
            return None
        return f"{self.store_url}/#/calls/{call_id}" if call_id else f"{self.store_url}/"

    def with_overrides(self, **changes) -> "ProcessConfig":
        return replace(self, **changes)

    def require_configured(self) -> None:
        if not self.store_url:
            raise ConfigError(f"No store_url in {self.config_path}. On the Store host run: "
                              "python -m call1.store issue-service-key --installation <name>")
        if not self.service_key:
            raise ConfigError(f"No service key in {self.config_path}. On the Store host run: "
                              "python -m call1.store issue-service-key --installation <name>")

    # --- construction ------------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[Path] = None, env: Optional[Mapping[str, str]] = None) -> "ProcessConfig":
        env = os.environ if env is None else env
        path = Path(path or env.get("CALL1_PROCESS_CONFIG") or DEFAULT_CONFIG_PATH).expanduser()
        raw = read_config_file(path)
        return cls.from_mapping(raw, path=path, env=env)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, path: Path = DEFAULT_CONFIG_PATH, env: Optional[Mapping[str, str]] = None) -> "ProcessConfig":
        env = {} if env is None else env
        try:
            slots = SlotSizes(**{k: int(v) for k, v in (raw.get("slots") or {}).items()})
        except TypeError as exc:
            raise ConfigError(f"slots: {exc}") from None
        try:
            stages = Stages(**{k: bool(v) for k, v in (raw.get("stages") or {}).items()})
        except TypeError as exc:
            raise ConfigError(f"stages: {exc}") from None
        try:
            port = int(env.get("CALL1_PROCESS_PORT") or raw.get("port") or DEFAULT_PORT)
        except ValueError:
            raise ConfigError("port must be an integer") from None
        defaults = raw.get("model_defaults") or {}
        if not isinstance(defaults, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in defaults.items()):
            raise ConfigError("model_defaults maps a purpose to a catalog entry ID")
        retention_hours = raw.get("scratch_retention_hours", 24)
        return cls(
            config_path=path,
            store_url=env.get("CALL1_PROCESS_STORE_URL") or raw.get("store_url"),
            installation_id=raw.get("installation_id"),
            service_key_id=raw.get("service_key_id"),
            service_key=raw.get("service_key"),
            bind_host=env.get("CALL1_PROCESS_BIND") or raw.get("bind") or DEFAULT_BIND,
            port=port,
            data_dir=Path(env.get("CALL1_PROCESS_DATA") or raw.get("data_dir") or DEFAULT_DATA_DIR),
            handlers=(env.get("CALL1_PROCESS_HANDLERS") or raw.get("handlers") or "real").strip().lower(),
            worker_id=env.get("CALL1_PROCESS_WORKER_ID") or raw.get("worker_id") or "",
            primary_host=bool(raw.get("primary_host", True)),
            rubric_id=raw.get("rubric_id") or DEFAULT_RUBRIC_ID,
            model_defaults=dict(defaults),
            escalation_entry_id=raw.get("escalation_entry_id") or None,
            slots=slots,
            stages=stages,
            summary_batch_turns=int(raw.get("summary_batch_turns", 60)),
            mask_model_text=_mask_mode(env.get("CALL1_PROCESS_MASK_MODEL_TEXT") or raw.get("mask_model_text")),
            scratch_retention_seconds=float(retention_hours) * 3600,
            scratch_max_bytes=int(raw.get("scratch_max_bytes", 5 * 1024 ** 3)),
            poll_interval_seconds=float(raw.get("poll_interval_seconds", 2.0)),
            shutdown_grace_seconds=float(raw.get("shutdown_grace_seconds", 30.0)),
            console_credential=raw.get("console_credential") or None,
            training=_training(raw.get("training")),
        )


def _mask_mode(value: Any) -> str:
    """``store`` (default), ``on`` or ``off``; JSON booleans are accepted for ``on``/``off``."""
    if value is None or value == "":
        return "store"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value).strip().lower()


def _training(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("training must be an object")
    return dict(value)


def default_worker_id() -> str:
    host = socket.gethostname().split(".")[0] or "host"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in host)[:48] or "host"
    return f"{safe}-process"


def read_config_file(path: Path) -> Dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must hold a JSON object")
    return data


def update_config_file(path: Path, values: Mapping[str, Any]) -> Path:
    """Merge ``values`` into the JSON config atomically and leave it mode 0600 (the same rule the
    Store host command uses when it writes the service key here)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = read_config_file(path)
    current.update(values)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(current, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path
