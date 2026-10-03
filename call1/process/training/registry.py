"""Versioned customer adapters, the active pointer, rollback and retention (docs/OnDeviceTraining.md
sections 4.5 and 5.1).

::

    <data_dir>/adapters/                     0700
      active.json    {"version", "base", "base_fingerprint", "tasks", "activated_at", "previous"}
      ft-20260927T090012Z/
        adapters.safetensors  adapter_config.json  manifest.json

* **Promote** writes ``manifest.json`` into the candidate, moves it to ``adapters/<version>/`` and
  replaces ``active.json`` atomically (a temporary file and ``os.replace``), so the pointer is never
  half-written. ``previous`` records the version it replaced.
* **Rollback** is activation of a kept version, or ``None`` for the base. It runs no evaluation and
  is refused for a version whose ``base_fingerprint`` no longer matches the installed base.
* **Retention** keeps the newest ``keep_versions`` promoted versions, never the active one or its
  ``previous``.
* **Resolution** (``resolve``) is what a job's LLM transport asks once per job: the active adapter's
  directory and version when the job's task is one the adapter was measured on and the base is
  unchanged; otherwise ``None`` (the base). A missing or corrupt adapter is logged once and falls
  back to the base: a job never fails over an adapter.

``current()`` is the process-wide registry the transports read; ``ProcessRuntime`` sets it with
``set_current``. Before that it resolves nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("call1.process.training.registry")

INCLUDED_MODEL_NAME = "gemma-4-e2b-it"
ADAPTER_FILES = ("adapters.safetensors", "adapter_config.json")
WEIGHT_SUFFIXES = (".safetensors", ".npz", ".gguf", ".bin")
TASKS = ("signal_stage1", "signal_stage2", "qa_verdict", "speaker_roles")


class RegistryError(Exception):
    """An activation the registry refuses. ``code`` is the console's error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def version_id(now: Optional[datetime] = None) -> str:
    """``ft-<UTC yyyymmddTHHMMSSZ>``."""
    return "ft-" + (now or utcnow()).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) and leave it mode 0700."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


_fingerprints: Dict[str, Tuple[Tuple[float, ...], str]] = {}
_fingerprint_lock = threading.Lock()


def base_fingerprint(base: Path) -> Optional[str]:
    """``sha256`` over the base model's ``config.json`` bytes and the sorted (name, size) of its
    weight files. Cheap, and cached by the directory's and config's mtimes. None when the base has no
    ``config.json`` (not installed)."""
    base = Path(base)
    config = base / "config.json"
    try:
        stamp = (base.stat().st_mtime, config.stat().st_mtime)
    except OSError:
        return None
    key = str(base.resolve())
    with _fingerprint_lock:
        cached = _fingerprints.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    digest = hashlib.sha256(config.read_bytes())
    weights = sorted((p.name, p.stat().st_size) for p in base.iterdir() if p.is_file() and p.suffix in WEIGHT_SUFFIXES)
    digest.update(json.dumps(weights).encode("utf-8"))
    value = "sha256:" + digest.hexdigest()
    with _fingerprint_lock:
        _fingerprints[key] = (stamp, value)
    return value


def adapter_complete(path: Path) -> bool:
    return all((Path(path) / name).is_file() for name in ADAPTER_FILES)


class AdapterRegistry:
    """The adapter directory of one Process host. ``base`` is the included model's directory."""

    def __init__(self, root: Path, base: Optional[Path] = None) -> None:
        self.root = Path(root)
        self.base = Path(base) if base is not None else None
        self._lock = threading.RLock()
        self._warned: set = set()
        self.set_aside_reason: Optional[str] = None

    # --- reads -------------------------------------------------------------------------------

    @property
    def active_path(self) -> Path:
        return self.root / "active.json"

    def active(self) -> Optional[Dict[str, Any]]:
        """The active pointer, or None when the base runs (no pointer, or ``version: null``)."""
        try:
            data = json.loads(self.active_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and data.get("version") else None

    def pointer(self) -> Dict[str, Any]:
        """The raw pointer, including ``previous`` when the base is active."""
        try:
            data = json.loads(self.active_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def manifest(self, version: str) -> Optional[Dict[str, Any]]:
        if not _valid_version(version):
            return None
        try:
            return json.loads((self.root / version / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def versions(self) -> List[Dict[str, Any]]:
        """Every kept (promoted) version's manifest, newest first."""
        out = []
        if not self.root.is_dir():
            return out
        for child in sorted(self.root.iterdir(), reverse=True):
            if child.is_dir() and _valid_version(child.name):
                manifest = self.manifest(child.name)
                if manifest is not None:
                    out.append(manifest)
        return out

    def fingerprint(self) -> Optional[str]:
        return base_fingerprint(self.base) if self.base is not None else None

    def stale(self) -> bool:
        """True when an adapter is active but was trained on another base (the included model was
        updated): it is set aside, and the next scheduled occurrence retrains regardless of
        ``min_new_labels``."""
        active = self.active()
        if active is None:
            return False
        current = self.fingerprint()
        return current is not None and active.get("base_fingerprint") != current

    # --- resolution (once per job) -----------------------------------------------------------

    def resolve(self, task: Optional[str], base_path: Optional[str]) -> Optional[Tuple[str, str]]:
        """(adapter directory, version) for a job of ``task`` on the text model at ``base_path``, or
        None for the base. Never raises: a job never fails over an adapter."""
        try:
            return self._resolve(task, base_path)
        except Exception as exc:  # pragma: no cover - defensive
            self._warn_once("resolve", f"adapter resolution failed ({type(exc).__name__}); running the base model")
            return None

    def _resolve(self, task: Optional[str], base_path: Optional[str]) -> Optional[Tuple[str, str]]:
        if task is None or not base_path or Path(base_path).name != INCLUDED_MODEL_NAME:
            return None
        active = self.active()
        if active is None or task not in (active.get("tasks") or []):
            return None
        version = str(active["version"])
        current = base_fingerprint(Path(base_path))
        if current is not None and active.get("base_fingerprint") != current:
            self.set_aside_reason = "Adapter set aside: the base model changed; retrain"
            self._warn_once(f"stale:{version}", f"adapter {version} was trained on another base model; running the base model")
            return None
        path = self.root / version
        if not _valid_version(version) or not adapter_complete(path):
            self._warn_once(f"missing:{version}", f"adapter {version} is missing or incomplete; running the base model")
            return None
        return str(path), version

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message)

    def new_version(self, now: Optional[datetime] = None) -> str:
        """A fresh ``ft-<UTC yyyymmddTHHMMSSZ>``, one second later while that name is taken."""
        moment = now or utcnow()
        version = version_id(moment)
        while (self.root / version).exists():
            moment += timedelta(seconds=1)
            version = version_id(moment)
        return version

    # --- writes ------------------------------------------------------------------------------

    def promote(self, candidate: Path, manifest: Dict[str, Any], *, keep_versions: int = 5) -> Dict[str, Any]:
        """Move the evaluated candidate to ``<root>/<version>/`` with its manifest and make it
        active. Returns the new pointer."""
        with self._lock:
            private_dir(self.root)
            version = str(manifest["version"])
            if not _valid_version(version):
                raise RegistryError("invalid_version", f"{version} is not an adapter version")
            if not adapter_complete(candidate):
                raise RegistryError("candidate_incomplete", "the candidate has no adapters.safetensors and adapter_config.json")
            dest = self.root / version
            if dest.exists():
                raise RegistryError("version_exists", f"{version} already exists")
            write_json_atomic(Path(candidate) / "manifest.json", manifest)
            shutil.move(str(candidate), str(dest))
            os.chmod(dest, 0o700)
            before = self.active()
            pointer = {"version": version, "base": manifest.get("base") or INCLUDED_MODEL_NAME,
                       "base_fingerprint": manifest.get("base_fingerprint"), "tasks": list(manifest.get("tasks") or []),
                       "activated_at": utcnow().isoformat(), "previous": before.get("version") if before else None}
            write_json_atomic(self.active_path, pointer)
            self.set_aside_reason = None
            self.prune(keep_versions)
            return pointer

    def activate(self, version: Optional[str]) -> Dict[str, Any]:
        """Rollback or reactivation: make ``version`` active, or the base with ``None``. No
        evaluation runs. Refused for an unknown version (``not_found``) and for one trained on
        another base (``stale_base``)."""
        with self._lock:
            before = self.active()
            if version is None:
                pointer: Dict[str, Any] = {"version": None, "base": INCLUDED_MODEL_NAME, "base_fingerprint": None, "tasks": [],
                                           "activated_at": utcnow().isoformat(), "previous": before.get("version") if before else None}
            else:
                manifest = self.manifest(version)
                if manifest is None or not adapter_complete(self.root / version):
                    raise RegistryError("not_found", f"No kept adapter version {version}")
                current = self.fingerprint()
                if current is not None and manifest.get("base_fingerprint") != current:
                    raise RegistryError("stale_base", f"{version} was trained on another version of the base model; retrain instead")
                pointer = {"version": version, "base": manifest.get("base") or INCLUDED_MODEL_NAME,
                           "base_fingerprint": manifest.get("base_fingerprint"), "tasks": list(manifest.get("tasks") or []),
                           "activated_at": utcnow().isoformat(), "previous": before.get("version") if before else None}
            private_dir(self.root)
            write_json_atomic(self.active_path, pointer)
            self.set_aside_reason = None
            return pointer

    def prune(self, keep_versions: int) -> List[str]:
        """Delete promoted versions beyond the newest ``keep_versions``, never the active one or its
        ``previous``. Returns the deleted versions."""
        with self._lock:
            pointer = self.pointer()
            protected = {pointer.get("version"), pointer.get("previous")}
            kept = 0
            removed = []
            for manifest in self.versions():  # newest first
                version = manifest.get("version")
                if version in protected or kept < keep_versions:
                    kept += 1
                    continue
                shutil.rmtree(self.root / str(version), ignore_errors=True)
                removed.append(str(version))
            return removed


def _valid_version(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("ft-") and len(value) <= 64 and all(c.isalnum() or c in "-_" for c in value)


class _NoRegistry:
    """Before Process sets a registry: every job runs the base."""

    set_aside_reason = None

    def resolve(self, task: Optional[str], base_path: Optional[str]) -> Optional[Tuple[str, str]]:
        return None


_current: Any = _NoRegistry()


def current():
    """The registry the LLM transports resolve adapters from."""
    return _current


def set_current(registry: Optional[AdapterRegistry]) -> None:
    global _current
    _current = registry if registry is not None else _NoRegistry()


def lora_suffix(model_revision: Optional[str]) -> str:
    """The ``+lora.<version>`` part of a model revision (``+lora.env`` for the manual override), or
    "" when the base answered."""
    if not model_revision or "+lora." not in model_revision:
        return ""
    return model_revision[model_revision.index("+lora."):]


def task_for(job_type: Any) -> Optional[str]:
    """The adapter task of a job type (section 5.1); None means the base. Summaries and stage-3
    extraction always run on the base, because nothing measures them."""
    value = getattr(job_type, "value", job_type)
    return {"contact_signals_categorize": "signal_stage1", "contact_signals_subcategorize": "signal_stage2",
            "qa_criterion": "qa_verdict", "qa_escalation": "qa_verdict", "speaker_attribution": "speaker_roles"}.get(value)


__all__ = ["AdapterRegistry", "RegistryError", "TASKS", "adapter_complete", "base_fingerprint", "current", "lora_suffix", "private_dir", "set_current",
           "task_for", "version_id", "write_json_atomic"]
