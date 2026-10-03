"""The running Process: its Store connection, catalog, handlers, worker, reanalysis consumer and
scratch janitor, built from ``ProcessConfig``.

``connect()`` does the start-up handshake: check the contract version (refuse another major), read
admin state (501 until Store implements it: the contract defaults apply), publish the hardware
profile and the catalog snapshot, build the worker and replay the completion spool. ``start()``
runs that handshake in the background, retrying while Store is unreachable, then starts the worker
threads, the reanalysis consumer and the janitor. ``stop()`` is the graceful shutdown.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from call1.contracts.admin import MaskingSettings
from call1.contracts.common import CONTRACT_VERSION
from call1.contracts.jobs import JobGroupProgress

from . import PROCESS_VERSION, hardware
from .catalog import ProcessCatalog, seeded_catalog
from .config import ConfigError, ProcessConfig
from .graph import GraphPlanner
from .handlers import HandlerRegistry, build_registry
from .ingest import Ingestor
from .ledger import Ledger
from .reanalysis import ReanalysisConsumer
from .scratch import Scratch, Spool
from .store_client import ContractMismatch, StoreClient, StoreError, StoreUnavailable
from .worker import Worker

log = logging.getLogger("call1.process")

GROUP_LABELS = {"transcript": "Transcript", "tone": "Tone", "text_sentiment": "Sentiment", "qa": "QA", "summary": "Summary",
                "contact_signals": "Signals"}
STATE_WORDS = {"pending": "analyzing", "available": "ready", "partial": "partial", "stale": "stale", "failed": "needs attention",
               "disabled": "off"}


def included_model_path():
    """The included model's weights directory on this host (``CALL1_MLX_TEXT_PATH`` or
    ``<models>/gemma-4-e2b-it``): the base every customer adapter trains over."""
    from .catalog import BUNDLED_LLM_ENTRY_ID
    from .handlers.real.paths import weights_path

    return weights_path(seeded_catalog(mode="real").get(BUNDLED_LLM_ENTRY_ID))


def progress_line(progress: JobGroupProgress) -> str:
    """The compact per-recording line the plan asks for: 'Transcript ready · QA 7/10 · Summary 2/4'."""
    parts = []
    for group in progress.groups:
        label = GROUP_LABELS.get(group.kind.value, group.kind.value)
        state = group.state.value
        if state == "disabled":
            continue
        if state in ("pending", "stale") and group.total:
            parts.append(f"{label} {group.succeeded}/{group.total}")
        else:
            parts.append(f"{label} {STATE_WORDS.get(state, state)}")
    return " · ".join(parts)


class ProcessRuntime:
    def __init__(self, config: ProcessConfig, *, http: Optional[httpx.Client] = None, client: Optional[StoreClient] = None,
                 registry: Optional[HandlerRegistry] = None, fake_behavior: Any = None) -> None:
        self.config = config
        self.scratch = Scratch(config.scratch_dir, retention_seconds=config.scratch_retention_seconds, max_bytes=config.scratch_max_bytes)
        self.spool = Spool(config.spool_dir)
        self.ledger = Ledger(config.ledger_path)
        if client is None and config.store_url:
            client = StoreClient(config.store_url, config.service_key, http=http)
        self.client = client
        catalog = seeded_catalog(mode=config.handlers, overrides=config.model_defaults)
        self.registry = registry or build_registry(config.handlers, config=config, catalog=catalog, fake_behavior=fake_behavior)
        if self.registry.entry_status is not None:
            catalog = catalog.with_status(self.registry.entry_status, getattr(self.registry, "entry_detail", None))
        self.catalog: ProcessCatalog = catalog
        self.masking = MaskingSettings()
        self.admin_state_source = "contract defaults"
        self.planner = GraphPlanner(self.catalog, config, self.masking)
        self.ingestor = Ingestor(config, client, self.planner, self.ledger, self.scratch) if client is not None else None
        self.worker: Optional[Worker] = None
        self.reanalysis: Optional[ReanalysisConsumer] = None
        self.hardware_profile_id: Optional[str] = None
        self.catalog_published: Optional[Dict[str, str]] = None
        self.state = "stopped" if config.configured else "not_configured"
        self.store_error: Optional[Dict[str, Any]] = None
        self.started_at: Optional[str] = None
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._connect_lock = threading.Lock()
        # On-device training (decision 28): the adapter registry the LLM transports resolve from, and
        # the scheduler the console's Settings tab drives (docs/OnDeviceTraining.md sections 3-5).
        from .training import registry as adapter_registry
        from .training.scheduler import TrainingService

        base_model = included_model_path()
        self.adapters = adapter_registry.AdapterRegistry(config.data_dir / "adapters", base=base_model)
        adapter_registry.set_current(self.adapters)
        self.training = TrainingService(config, client=client, registry=self.adapters, base_model=base_model, worker=lambda: self.worker,
                                        ledger=self.ledger, catalog=self.catalog, primary_host=lambda: self.config.primary_host)

    # --- handshake ---------------------------------------------------------------------------

    def require_client(self) -> StoreClient:
        self.config.require_configured()
        if not self.config.installation_id:
            raise ConfigError(f"No installation_id in {self.config.config_path}; reissue the key with python -m call1.store issue-service-key")
        assert self.client is not None
        return self.client

    def check_store(self) -> None:
        """The contract check alone (enough for ingest)."""
        client = self.require_client()
        client.check_contract()

    def connect(self) -> Worker:
        with self._connect_lock:
            if self.worker is not None:
                return self.worker
            client = self.require_client()
            self.state = "connecting"
            client.check_contract()
            try:
                admin = client.admin_state()
            except StoreError as exc:
                log.info("admin state not readable (%s); using the contract defaults", exc.code)
                admin = None
            if admin and isinstance(admin.get("masking"), dict):
                try:
                    self.masking = MaskingSettings.model_validate(admin["masking"])
                    self.admin_state_source = "store"
                except ValueError:
                    pass
            self.planner.masking = self.masking
            profile = client.upsert_hardware_profile(hardware.profile_input())
            self.hardware_profile_id = profile.id
            self.publish_catalog()
            self.worker = Worker(client=client, registry=self.registry, catalog=self.catalog, planner=self.planner,
                                 installation_id=str(self.config.installation_id), hardware_profile_id=profile.id, worker_id=self.config.worker_id,
                                 slots=self.config.slots, scratch=self.scratch, spool=self.spool, primary_host=self.config.primary_host,
                                 poll_interval=self.config.poll_interval_seconds)
            self.reanalysis = ReanalysisConsumer(self.config, client, self.planner, self.ledger, self.config.worker_id)
            replayed = self.worker.recover()
            if replayed:
                log.info("replayed %d spooled completion(s)", replayed)
            self.state = "connected"
            self.store_error = None
            return self.worker

    def publish_catalog(self) -> Dict[str, str]:
        client = self.require_client()
        snapshot = self.catalog.snapshot(str(self.config.installation_id), datetime.now(timezone.utc))
        client.publish_catalog(snapshot)
        self.catalog_published = {"catalog_version": snapshot.catalog_version, "published_at": snapshot.published_at.isoformat()}
        return self.catalog_published

    # --- background --------------------------------------------------------------------------

    def start(self) -> None:
        if not self.config.configured:
            self.state = "not_configured"
            return
        self._stop.clear()
        self.started_at = datetime.now(timezone.utc).isoformat()
        thread = threading.Thread(target=self._run, name="call1-process-runtime", daemon=True)
        thread.start()
        self._threads.append(thread)
        self.training.start()

    def _run(self) -> None:
        delay = 2.0
        while not self._stop.is_set():
            try:
                self.connect()
                break
            except ContractMismatch as exc:
                self.state = "contract_mismatch"
                self.store_error = {"code": "contract_mismatch", "message": str(exc)}
                delay = 60.0
            except StoreUnavailable as exc:
                self.state = "store_unreachable"
                self.store_error = exc.to_dict()
            except StoreError as exc:
                self.state = "store_refused"
                self.store_error = exc.to_dict()
                delay = 30.0
            except ConfigError as exc:
                self.state = "not_configured"
                self.store_error = {"code": "not_configured", "message": str(exc)}
                return
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("Process start-up failed")
                self.state = "error"
                self.store_error = {"code": "error", "message": type(exc).__name__}
            self._stop.wait(delay)
            delay = min(60.0, delay * 2)
        if self._stop.is_set() or self.worker is None:
            return
        self.worker.start()
        self.state = "running"
        for target, name in ((self._reanalysis_loop, "call1-reanalysis"), (self._janitor_loop, "call1-janitor")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def _reanalysis_loop(self) -> None:
        while not self._stop.wait(max(2.0, self.config.poll_interval_seconds * 2)):
            if self.reanalysis is None:
                continue
            if self.worker is not None and self.worker.claims_paused is not None:
                continue  # on-device training paused every claim (docs/OnDeviceTraining.md section 3.3)
            try:
                self.reanalysis.poll_once()
            except StoreError as exc:
                self.reanalysis.last_error = f"claim: {exc.code}"
            except Exception:  # pragma: no cover - defensive
                log.exception("reanalysis loop")

    def _janitor_loop(self) -> None:
        while True:
            try:
                keep = set()
                if self.worker is not None:
                    keep = {p for p in (self.config.scratch_dir / "jobs").glob("*") if any(p.name.startswith(r.claimed.job.id) for r in self.worker.running())}
                self.scratch.sweep(keep=keep)
            except Exception:  # pragma: no cover - defensive
                log.exception("scratch janitor")
            if self._stop.wait(3600):
                return

    def stop(self) -> None:
        self._stop.set()
        self.training.stop()
        if self.worker is not None:
            self.worker.stop(self.config.shutdown_grace_seconds)
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads = []
        if self.state not in ("not_configured",):
            self.state = "stopped"

    # --- views -------------------------------------------------------------------------------

    def overview(self) -> Dict[str, Any]:
        store: Dict[str, Any] = {"url": self.config.store_url, "dev_mode": self.config.dev_store, "state": self.state,
                                 "error": self.store_error, "process_contract_version": CONTRACT_VERSION}
        if self.client is not None and self.client.contract is not None:
            params = self.client.parameters
            store.update(contract_version=self.client.contract.contract_version, compatible=True,
                         parameters={"lease_duration_seconds": params.lease_duration_seconds,
                                     "heartbeat_interval_seconds": params.heartbeat_interval_seconds,
                                     "max_claim_batch": params.max_claim_batch, "default_max_attempts": params.default_max_attempts,
                                     "inline_artifact_max_bytes": params.inline_artifact_max_bytes})
        else:
            store.update(contract_version=None, compatible=None)
        return {
            "app": "Call1 Process", "version": PROCESS_VERSION, "state": self.state, "started_at": self.started_at,
            "installation_id": self.config.installation_id, "worker_id": self.config.worker_id,
            "bind": f"{self.config.bind_host}:{self.config.port}", "store": store,
            "handlers": {"mode": self.registry.mode, "notes": list(self.registry.notes),
                         "missing_job_types": [t.value for t in self.registry.missing()]},
            "worker": self.worker.describe() if self.worker is not None else None,
            "slots": [p.describe() for p in self.worker.pools] if self.worker is not None else [
                {"pool": "mlx", "size": self.config.slots.mlx}, {"pool": "torch", "size": self.config.slots.torch},
                {"pool": "cpu_io", "size": self.config.slots.cpu_io}],
            "reanalysis": ({"handled": self.reanalysis.handled, "rejected": self.reanalysis.rejected, "last_error": self.reanalysis.last_error}
                           if self.reanalysis is not None else None),
            "catalog": {"version": self.catalog.version(), "published": self.catalog_published},
            "admin_state": self.admin_state_source, "hardware_profile_id": self.hardware_profile_id,
            "scratch_bytes": self.scratch.usage_bytes(), "conversations": self.ledger.count(),
            "evaluate_url": self.config.evaluate_url(),
            "training": self.training_summary(),
        }

    def training_summary(self) -> Dict[str, Any]:
        """The Overview's line about on-device training (the Settings tab has the rest)."""
        active = self.adapters.active()
        return {"enabled": self.training.settings.enabled, "active_version": active.get("version") if active else None,
                "claims_paused": self.worker.claims_paused if self.worker is not None else None}

    def wait_until(self, predicate, timeout: float, interval: float = 0.05) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return predicate()
