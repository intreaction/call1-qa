"""``TrainingService``: the settings, the schedule, the start rule and the claim pause
(docs/OnDeviceTraining.md section 3), and the console's view of all of it (section 6.1).

**Schedule.** The ``call1-training`` thread wakes every 30 s. A due occurrence (daily, or weekly on a
weekday, at ``HH:MM`` in the host's time zone; DST through ``zoneinfo``, a nonexistent local time
runs at the first valid minute after it) opens a start window of 4 hours (``START_WINDOW``). Its
label threshold is checked once, with a count-only read (``listTrainingLabels?after=<cursor>&limit=0``):
below ``min_new_labels`` the occurrence is recorded in ``state.json`` as a *check* and no run is
written; at or above it a run waits for the start rule, rechecking every 60 s, until the window
closes (then ``skipped: busy``). An occurrence missed while the host was down still runs while its
window is open. At most one run exists at a time; occurrences never queue up.

**Start rule** (every run, including Train now): the local ``mlx`` and ``torch`` pools are empty, and
Store lists no ``QUEUED`` or ``RUNNING`` job with ``memory_slot = local_memory``. ``BLOCKED`` jobs do
not count. Scheduled runs with ``only_when_idle`` also need every pool empty, no ``QUEUED`` or
``RUNNING`` job of any slot, and no recording ingested in the last 15 minutes.

**Pause protocol.** Before the pause only Store's side of the rule is checked: a pool loop reserves
its slots for every claim round-trip, so an empty claim in flight would read as a busy pool.
``worker.pause_claims`` stops every pool and the reanalysis consumer from claiming (heartbeats and
the spool keep running); claims in flight and running jobs get up to 10 minutes to finish, else
claims resume and the run waits again; the rule is checked again (work may have arrived between the
check and the pause); then the runner runs with ``inference_lock`` held from the first GPU use to
the last. Claims always resume, in a ``finally``.

**Train now** queues a manual run (409 ``training_busy`` when one is queued or active): it ignores
the schedule and ``min_new_labels``, obeys the hard rule within a 4-hour window, and does not apply
``only_when_idle``. **Cancel** is idempotent: a waiting run is dropped, a running one's subprocess
gets ``SIGTERM`` then ``SIGKILL`` after 10 s, and the run ends ``cancelled`` with the active
adapter untouched.

Files (``<data_dir>/training/``, 0700): ``state.json`` (the label cursor, the last check, the
measurements, the run in progress), ``runs.jsonl`` (one record per run: counts, metrics and
reasons, never text) and ``lock``.

**Ownership.** One process on the host owns the training directory: the one holding an exclusive
``flock`` on ``lock`` (``claim_ownership``). ``serve`` claims it when its scheduler starts; ``training
run`` claims it, and refuses while a ``serve`` holds it (Train now in that serve's console is the
way to train then, so its claim pause covers the run). Constructing a ``TrainingService`` (every
CLI command builds one) never recovers, pauses or deletes anything. Only the owner recovers: a run
found in ``state.json`` when ownership is claimed was interrupted, so it is recorded
``interrupted`` and the work directory is deleted. Without ownership the service is read-only:
``request_run`` and ``tick`` refuse or do nothing, and ``describe`` shows the owner's run from
``state.json``.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import platform
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import settings as settings_mod
from .registry import AdapterRegistry, private_dir, write_json_atomic
from .runner import RunContext, Runner, TERMINAL, new_record
from .settings import SettingsError, TrainingSettings

log = logging.getLogger("call1.process.training")

START_WINDOW = timedelta(hours=4)
TICK_SECONDS = 30.0
RECHECK_SECONDS = 60.0
DRAIN_TIMEOUT_SECONDS = 600.0
INGEST_QUIET = timedelta(minutes=15)
HISTORY_LIMIT = 20
LABEL_COUNT_TTL_SECONDS = 10.0
OWNED_ELSEWHERE = ("Another Process on this host (python -m call1.process serve) owns on-device training; use Train now in its console, "
                   "or stop it first")
TRAINING_READ_HELP = ("This Process service key lacks the training:read scope, so on-device training cannot read reviewer labels. On the "
                      "Store host, issue a key that holds it: python -m call1.store issue-service-key --installation <name> --scope "
                      "training:read plus the default Process scopes (see call1/store/README.md).")


class TrainingBusy(Exception):
    """A run is already queued or active (409 ``training_busy``)."""


class TrainingUnavailable(Exception):
    """Training cannot run on this host (409 ``training_unavailable``); the message says why."""


class RunNotFound(Exception):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def local_zone() -> tzinfo:
    """The Process host's IANA time zone: ``TZ``, else ``/etc/localtime``'s target, else UTC."""
    name = os.environ.get("TZ", "").lstrip(":")
    candidates = [name] if name else []
    try:
        target = os.path.realpath("/etc/localtime")
        if "zoneinfo/" in target:
            candidates.append(target.split("zoneinfo/", 1)[1])
    except OSError:  # pragma: no cover
        pass
    for candidate in candidates:
        try:
            return ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError):
            continue
    return ZoneInfo("UTC")


def zone_name(tz: tzinfo) -> str:
    return getattr(tz, "key", None) or str(tz)


def mlx_available() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64" and os.getenv("CALL1_BACKEND", "") == "mlx"


# --- schedule arithmetic -------------------------------------------------------------------------


def occurrence_on(day: date, settings: TrainingSettings, tz: tzinfo) -> datetime:
    """The scheduled local time on ``day`` as an aware datetime. A nonexistent local time (a DST
    gap) runs at the first valid minute after it; an ambiguous one at its first occurrence."""
    naive = datetime(day.year, day.month, day.day, settings.schedule.hour, settings.schedule.minute)
    for _ in range(24 * 60):
        aware = naive.replace(tzinfo=tz, fold=0)
        if aware.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None) == naive:
            return aware
        naive += timedelta(minutes=1)
    return naive.replace(tzinfo=tz)  # pragma: no cover - no zone has a day-long gap


def _scheduled_day(day: date, settings: TrainingSettings) -> bool:
    return settings.schedule.frequency == "daily" or day.weekday() == settings.schedule.weekday


def latest_occurrence(now: datetime, settings: TrainingSettings, tz: tzinfo) -> Optional[datetime]:
    """The most recent occurrence at or before ``now``."""
    today = now.astimezone(tz).date()
    for back in range(0, 9):
        day = today - timedelta(days=back)
        if not _scheduled_day(day, settings):
            continue
        occurrence = occurrence_on(day, settings, tz)
        if occurrence <= now:
            return occurrence
    return None


def next_occurrence(now: datetime, settings: TrainingSettings, tz: tzinfo) -> datetime:
    """The first occurrence strictly after ``now``."""
    today = now.astimezone(tz).date()
    for ahead in range(0, 9):
        day = today + timedelta(days=ahead)
        if not _scheduled_day(day, settings):
            continue
        occurrence = occurrence_on(day, settings, tz)
        if occurrence > now:
            return occurrence
    raise AssertionError("unreachable")  # pragma: no cover


# --- the service ---------------------------------------------------------------------------------


@dataclass
class ActiveRun:
    record: Dict[str, Any]
    scheduled: bool
    window_end: datetime
    cancel: threading.Event = field(default_factory=threading.Event)
    phase: str = "waiting"
    detail: str = "Waiting to start"
    runner: Optional[Runner] = None
    thread: Optional[threading.Thread] = None

    @property
    def run_id(self) -> str:
        return self.record["run_id"]


class TrainingService:
    def __init__(self, config, *, client, registry: AdapterRegistry, base_model: Path, worker: Callable[[], Any] = lambda: None,
                 ledger=None, catalog=None, primary_host: Callable[[], bool] = lambda: True, tz: Optional[tzinfo] = None,
                 now: Callable[[], datetime] = utcnow, count_tokens: Optional[Callable[[str], int]] = None,
                 generator_factory: Optional[Callable[..., Any]] = None, tick_seconds: float = TICK_SECONDS,
                 recheck_seconds: float = RECHECK_SECONDS, drain_timeout: float = DRAIN_TIMEOUT_SECONDS, kill_grace: float = 10.0,
                 min_free_bytes: Optional[int] = None, lock=None) -> None:
        from call1.pipeline.inference import inference_lock

        self.config = config
        self.client = client
        self.registry = registry
        # absolute: the trainer and the generation worker run with the run's work directory as cwd,
        # where a relative data/models path would not resolve (mlx_lm then treats it as a Hub repo ID)
        self.base_model = Path(base_model).expanduser().absolute()
        self.worker = worker
        self.ledger = ledger
        self.catalog = catalog
        self.primary_host = primary_host
        self.tz = tz or local_zone()
        self.now = now
        self._count_tokens = count_tokens
        self.generator_factory = generator_factory
        self.tick_seconds = tick_seconds
        self.recheck_seconds = recheck_seconds
        self.drain_timeout = drain_timeout
        self.kill_grace = kill_grace
        self.min_free_bytes = min_free_bytes
        self.inference_lock = lock if lock is not None else inference_lock
        self.root = Path(config.data_dir) / "training"
        self.state_path = self.root / "state.json"
        self.runs_path = self.root / "runs.jsonl"
        self.work_root = self.root / "work"
        self.settings = TrainingSettings.from_mapping(config.training)
        self._lock = threading.RLock()
        self._file_lock = threading.Lock()
        self._run: Optional[ActiveRun] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._labels_cache: Tuple[float, Dict[str, Any]] = (0.0, {})
        self._lock_file = None  # the open ``lock`` file while this process owns the directory

    # --- ownership -----------------------------------------------------------------------------

    @property
    def owned(self) -> bool:
        return self._lock_file is not None

    def claim_ownership(self) -> bool:
        """Take the exclusive ``flock`` on ``<data_dir>/training/lock`` (non-blocking). The first
        successful claim recovers an interrupted run. False when another process (a running
        ``serve``) holds it."""
        with self._lock:
            if self._lock_file is not None:
                return True
            private_dir(self.root)
            fd = os.open(self.root / "lock", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                return False
            self._lock_file = fd
            self.recover()
            return True

    def release_ownership(self) -> None:
        with self._lock:
            fd, self._lock_file = self._lock_file, None
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def __del__(self) -> None:  # pragma: no cover - best effort; the OS drops the flock at exit anyway
        fd = getattr(self, "_lock_file", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    # --- files -------------------------------------------------------------------------------

    def _read_state(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_state(self, **changes: Any) -> Dict[str, Any]:
        with self._file_lock:
            state = self._read_state()
            state.update(changes)
            private_dir(self.root)
            write_json_atomic(self.state_path, state)
            return state

    def state(self) -> Dict[str, Any]:
        with self._file_lock:
            return self._read_state()

    def _append_run(self, record: Dict[str, Any]) -> None:
        with self._file_lock:
            private_dir(self.root)
            fd = os.open(self.runs_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Finished runs, newest first."""
        try:
            lines = self.runs_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
            if len(out) >= limit:
                break
        return out

    def runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """The run in progress (if any), then finished runs, newest first."""
        items: List[Dict[str, Any]] = []
        with self._lock:
            if self._run is not None:
                items.append(self._live_record(self._run))
        return (items + self.history(limit))[:max(1, limit)]

    def recover(self) -> Optional[Dict[str, Any]]:
        """Mark a run left in ``state.json`` by a stop or crash ``interrupted``, delete every work
        directory and discard any candidate. The pause was in memory, so claims are not paused.
        Only the owner recovers (``claim_ownership`` calls this); anyone else gets None."""
        if not self.owned:
            return None
        state = self.state()
        stale = state.get("active_run")
        record = None
        if isinstance(stale, dict) and stale.get("run_id"):
            record = dict(stale, status="interrupted", reason="Process stopped during the run", ended_at=utcnow().isoformat())
            self._append_run(record)
            self._write_state(active_run=None)
        if self.work_root.exists():
            shutil.rmtree(self.work_root, ignore_errors=True)
        return record

    # --- availability --------------------------------------------------------------------------

    @property
    def trainer(self) -> str:
        return self.settings.effective_trainer(self.config.handlers)

    def unavailable_reason(self) -> Optional[str]:
        if not self.config.configured or self.client is None:
            return "The Store connection is not configured"
        worker = self.worker()
        primary = self.primary_host() and (worker is None or worker.primary_host)
        if not primary:
            return "This Process is not the primary host"
        if self.trainer != "fake":
            if not mlx_available():
                return "MLX is not available on this host (Apple Silicon with CALL1_BACKEND=mlx)"
            if not (self.base_model / "config.json").is_file():
                return "The included model is not installed on this host"
        return None

    # --- settings ------------------------------------------------------------------------------

    def update_settings(self, body: Dict[str, Any]) -> TrainingSettings:
        with self._lock:
            updated = self.settings.with_console_update(body)
            if updated.enabled:
                reason = self.unavailable_reason()
                if reason:
                    raise TrainingUnavailable(reason)
            settings_mod.save(self.config.config_path, updated)
            self.settings = updated
            return updated

    # --- the scheduler thread ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self.claim_ownership():
            log.error("on-device training: another Process on this host owns %s; this one does not schedule runs", self.root)
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="call1-training", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        self._stop.set()
        with self._lock:
            active = self._run
        if active is not None:
            active.cancel.set()
            if active.thread is not None:
                active.thread.join(timeout=timeout)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.release_ownership()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # pragma: no cover - defensive
                log.exception("training scheduler")
            self._stop.wait(self.tick_seconds)

    def tick(self, now: Optional[datetime] = None) -> Optional[str]:
        """One scheduler pass. Returns what it did: ``None`` (nothing due), ``check``, ``missed``,
        ``queued`` or ``error``."""
        now = now or self.now()
        with self._lock:
            settings = self.settings
            if self._run is not None or not settings.enabled or self.unavailable_reason():
                return None
            if not self.claim_ownership():
                return None
            occurrence = latest_occurrence(now, settings, self.tz)
            if occurrence is None:
                return None
            key = occurrence.isoformat()
            state = self.state()
            if state.get("last_occurrence") == key:
                return None
            if now >= occurrence + START_WINDOW:
                self._write_state(last_occurrence=key)
                return "missed"
            force = bool(self.registry.stale())
            count = None
            if not force:
                try:
                    count = self.client.list_training_labels(after=int(state.get("cursor") or 0), limit=0, retry=False, timeout=10).count_after
                except Exception as exc:
                    self._write_state(last_check={"at": now.isoformat(), "occurrence": key, "error": getattr(exc, "code", type(exc).__name__)})
                    return "error"
                if count < settings.min_new_labels:
                    self._write_state(last_occurrence=key, last_check={"at": now.isoformat(), "occurrence": key, "new_labels": count,
                                                                       "min_new_labels": settings.min_new_labels, "started": False})
                    return "check"
            self._write_state(last_occurrence=key, last_check={"at": now.isoformat(), "occurrence": key, "new_labels": count,
                                                               "min_new_labels": settings.min_new_labels, "started": True,
                                                               "base_changed": force})
            self._queue("schedule", occurrence + START_WINDOW, scheduled=True)
            return "queued"

    # --- runs ----------------------------------------------------------------------------------

    def request_run(self, trigger: str = "manual") -> Dict[str, Any]:
        """Train now: queue a manual run (ignores the schedule and ``min_new_labels``)."""
        with self._lock:
            if self._run is not None:
                raise TrainingBusy("A training run is already queued or running")
            reason = self.unavailable_reason()
            if reason:
                raise TrainingUnavailable(reason)
            if not self.claim_ownership():
                raise TrainingUnavailable(OWNED_ELSEWHERE)
            active = self._queue(trigger, self.now() + START_WINDOW, scheduled=False)
            return self._live_record(active)

    def _queue(self, trigger: str, window_end: datetime, *, scheduled: bool) -> ActiveRun:
        run_id = "tr-" + utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        record = new_record(run_id, trigger)
        active = ActiveRun(record=record, scheduled=scheduled, window_end=window_end)
        self._run = active
        self._write_state(active_run=record)
        active.thread = threading.Thread(target=self._run_thread, args=(active,), name=f"call1-training-{run_id}", daemon=True)
        active.thread.start()
        return active

    def cancel(self, run_id: str) -> Dict[str, Any]:
        """Idempotent: cancel the queued or running run, or return the finished one's record."""
        with self._lock:
            active = self._run
            if active is not None and active.run_id == run_id:
                active.cancel.set()
                active.detail = "Cancelling"
                return self._live_record(active)
        for record in self.history(10_000):
            if record.get("run_id") == run_id:
                return record
        raise RunNotFound(run_id)

    def wait(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """Wait for the current run to end (tests and the CLI); returns its record."""
        with self._lock:
            active = self._run
        if active is None or active.thread is None:
            return None
        active.thread.join(timeout=timeout)
        return active.record

    def _run_thread(self, active: ActiveRun) -> None:
        try:
            self._wait_and_run(active)
        except Exception:  # pragma: no cover - defensive
            log.exception("training run %s", active.run_id)
            self._finish(active, "failed", "error")
        finally:
            with self._lock:
                record = active.record
                if record.get("status") not in TERMINAL:
                    self._finish(active, "failed", "error")
                self._append_run(record)
                changes: Dict[str, Any] = {"active_run": None}
                cursor = (record.get("label_cursor") or {}).get("to")
                if cursor is not None and record["status"] in ("promoted", "rejected", "skipped"):
                    changes["cursor"] = cursor
                if active.runner is not None:
                    changes.update({k: v for k, v in active.runner.ctx.measurements.items() if k in ("it_per_s", "s_per_prompt")})
                self._write_state(**changes)
                self._run = None
                self._labels_cache = (0.0, {})

    def _finish(self, active: ActiveRun, status: str, reason: str) -> None:
        active.phase = status
        active.detail = reason
        active.record.update(status=status, reason=reason, ended_at=utcnow().isoformat())

    def _wait_and_run(self, active: ActiveRun) -> None:
        settings = self.settings
        run_id = active.run_id
        last_reason = ""
        backoff = False
        while True:
            if backoff:
                # claims are resumed while the run waits again
                active.cancel.wait(self.recheck_seconds)
                backoff = False
            if active.cancel.is_set():
                self._finish(active, "cancelled", "cancelled while waiting")
                return
            if self.now() >= active.window_end:
                self._finish(active, "skipped", f"busy: {last_reason or 'the start window closed'}")
                return
            idle_mode = active.scheduled and settings.only_when_idle
            # Store's view only: the local pools are checked after the pause and drain, because a
            # pool loop reserves its slots for each claim round-trip, so an empty claim in flight
            # would read as a running model job and cost a needless recheck interval.
            ok, why = self.start_rule(idle_mode, local=False)
            if not ok:
                last_reason = why
                active.phase, active.detail = "waiting", f"Waiting: {why}"
                active.record["status"] = "waiting"
                backoff = True
                continue
            worker = self.worker()
            max_duration = timedelta(minutes=settings.max_duration_minutes)
            deadline = time.monotonic() + max_duration.total_seconds()
            worker.pause_claims(run_id, until=self.now() + max_duration)
            try:
                active.phase, active.detail = "pausing", "Pausing claims and waiting for running jobs"
                active.record["status"] = "pausing"
                if not self._drain(worker, active):
                    last_reason = ("a model job is running" if worker.busy_pools(["mlx", "torch"])
                                   else "running jobs did not finish within 10 minutes")
                    active.phase, active.detail = "waiting", f"Waiting: {last_reason}"
                    active.record["status"] = "waiting"
                    backoff = not active.cancel.is_set()
                    continue
                ok, why = self.start_rule(idle_mode)
                if not ok:  # work arrived between the check and the pause
                    last_reason = why
                    active.phase, active.detail = "waiting", f"Waiting: {why}"
                    active.record["status"] = "waiting"
                    backoff = True
                    continue
                with self.inference_lock:  # nothing in Process takes the GPU until the evaluation ends
                    runner = Runner(active.record, self._context(settings), active.cancel,
                                    label_cursor_from=int(self.state().get("cursor") or 0))
                    active.runner = runner
                    runner.execute(deadline)
                active.phase, active.detail = active.record["status"], active.record.get("reason") or ""
                return
            finally:
                worker.resume_claims(run_id)

    def _drain(self, worker, active: ActiveRun) -> bool:
        deadline = time.monotonic() + self.drain_timeout
        while time.monotonic() < deadline:
            if active.cancel.is_set():
                return False
            if worker.idle():
                return True
            active.cancel.wait(0.05)
        return worker.idle()

    def start_rule(self, idle_mode: bool, *, local: bool = True) -> Tuple[bool, str]:
        """Whether a run may start now, and why not (section 3.3). ``local=False`` skips this
        Process's own pools: the check before the claim pause, where a claim round-trip in flight
        holds a pool's slots without running anything. The check after the drain is always full."""
        from call1.contracts.jobs import JobStatus, MemorySlot

        from ..store_client import StoreError, StoreUnavailable

        worker = self.worker()
        if worker is None:
            return False, "Process is not connected to Store yet"
        if local and worker.busy_pools(["mlx", "torch"]):
            return False, "a model job is running"
        if local and idle_mode and not worker.idle():
            return False, "jobs are running"
        try:
            for status in (JobStatus.QUEUED, JobStatus.RUNNING):
                if self.client.list_jobs(status=status.value, memory_slot=MemorySlot.LOCAL_MEMORY.value, limit=1, retry=False, timeout=10):
                    return False, f"model jobs {status.value.lower()}"
            if idle_mode:
                for status in (JobStatus.QUEUED, JobStatus.RUNNING):
                    if self.client.list_jobs(status=status.value, limit=1, retry=False, timeout=10):
                        return False, f"jobs {status.value.lower()}"
        except StoreUnavailable:
            return False, "Store is unreachable"
        except StoreError as exc:
            return False, f"Store refused the job check ({exc.code})"
        if idle_mode and self.ledger is not None:
            last = self.ledger.latest_at() if hasattr(self.ledger, "latest_at") else None
            if last is not None and self.now() - last < INGEST_QUIET:
                return False, "a recording was ingested in the last 15 minutes"
        return True, ""

    def _context(self, settings: TrainingSettings) -> RunContext:
        state = self.state()
        measurements = {k: float(state[k]) for k in ("it_per_s", "s_per_prompt") if isinstance(state.get(k), (int, float))}
        extra: Dict[str, Any] = {}
        if self.min_free_bytes is not None:
            extra["min_free_bytes"] = self.min_free_bytes
        return RunContext(client=self.client, registry=self.registry, settings=settings, trainer=self.trainer, base_model=self.base_model,
                          installation_id=str(self.config.installation_id or ""), work_root=self.work_root, count_tokens=self.count_tokens(),
                          measurements=measurements, generator_factory=self._generator_factory(), catalog=self.catalog,
                          kill_grace=self.kill_grace, **extra)

    def count_tokens(self) -> Callable[[str], int]:
        if self._count_tokens is None:
            from ..catalog import BUNDLED_LLM_ENTRY_ID, seeded_catalog
            from ..handlers.real.signals_v2 import gemma_token_counter

            self._count_tokens = gemma_token_counter(seeded_catalog(mode="real").get(BUNDLED_LLM_ENTRY_ID))
        return self._count_tokens

    def _generator_factory(self):
        if self.generator_factory is not None:
            return self.generator_factory
        if self.trainer != "fake":
            return None  # the runner uses the subprocess generation worker

        def fake(data, runner):
            from .generate import FakeGenerator

            return FakeGenerator(data.prompts, data.items, self.next_fake_outcome())
        return fake

    def next_fake_outcome(self) -> str:
        """``CALL1_FAKE_TRAINING_OUTCOMES`` (``promote,reject,invalid,crash``), one consumed per run
        that reaches evaluation; ``promote`` once they are used up."""
        outcomes = [o.strip() for o in (os.getenv("CALL1_FAKE_TRAINING_OUTCOMES") or "").split(",") if o.strip()]
        used = int(self.state().get("fake_outcomes_used") or 0)
        if used < len(outcomes):
            self._write_state(fake_outcomes_used=used + 1)
            return outcomes[used]
        return "promote"

    # --- activation ----------------------------------------------------------------------------

    def activate(self, version: Optional[str]) -> Dict[str, Any]:
        """Rollback or reactivation (no evaluation); refused while a run is queued or active."""
        with self._lock:
            if self._run is not None:
                raise TrainingBusy("A training run is queued or running; cancel it or wait for it to finish")
            if not self.owned and isinstance(self.state().get("active_run"), dict):
                raise TrainingBusy("A training run is queued or running in another Process on this host")
            return self.registry.activate(version)

    # --- the console's view --------------------------------------------------------------------

    def _live_record(self, active: ActiveRun) -> Dict[str, Any]:
        record = dict(active.record)
        runner = active.runner
        record["phase"], record["detail"] = (runner.phase, runner.detail) if runner is not None else (active.phase, active.detail)
        return record

    def label_counts(self) -> Dict[str, Any]:
        """Count-only reads (not audited): the label total and the rows since the last run's cursor."""
        from ..store_client import StoreError

        cached_at, cached = self._labels_cache
        if cached and time.monotonic() - cached_at < LABEL_COUNT_TTL_SECONDS:
            return cached
        out: Dict[str, Any] = {"total": None, "new_since_last_run": None, "error": None}
        if self.client is None or not self.config.configured:
            out["error"] = {"code": "not_configured", "message": "The Store connection is not configured"}
            return out
        try:
            total = self.client.list_training_labels(after=0, limit=0, retry=False, timeout=3).count_after
            cursor = int(self.state().get("cursor") or 0)
            out["total"] = total
            out["new_since_last_run"] = self.client.list_training_labels(after=cursor, limit=0, retry=False, timeout=3).count_after if cursor else total
        except StoreError as exc:
            message = TRAINING_READ_HELP if exc.code == "insufficient_scope" else (exc.message or exc.code)
            out["error"] = {"code": exc.code, "message": message}
        self._labels_cache = (time.monotonic(), out)
        return out

    def describe(self) -> Dict[str, Any]:
        now = self.now()
        settings = self.settings
        reason = self.unavailable_reason()
        state = self.state()
        with self._lock:
            active = self._run
            status: Dict[str, Any]
            if active is not None:
                runner = active.runner
                phase, detail = (runner.phase, runner.detail) if runner is not None else (active.phase, active.detail)
                progress = dict(runner.progress) if runner is not None else {"iteration": None, "iterations": None, "train_loss": None, "val_loss": None}
                status = {"phase": phase, "run_id": active.run_id, "trigger": active.record.get("trigger"), "detail": detail, "progress": progress}
            elif not self.owned and isinstance(state.get("active_run"), dict) and state["active_run"].get("run_id"):
                elsewhere = state["active_run"]  # the owner's run, as its state.json records it
                status = {"phase": str(elsewhere.get("status") or "waiting"), "run_id": elsewhere["run_id"], "trigger": elsewhere.get("trigger"),
                          "detail": "Running in another Process on this host", "progress": None}
            else:
                status = {"phase": "idle", "run_id": None, "trigger": None, "detail": "", "progress": None}
        worker = self.worker()
        status["claims_paused"] = worker.claims_paused if worker is not None else None
        pointer = self.registry.active()
        active_view = None
        if pointer is not None:
            manifest = self.registry.manifest(str(pointer["version"])) or {}
            active_view = dict(pointer, eval=manifest.get("eval"), decision=manifest.get("decision"))
        labels = self.label_counts()
        notices: List[Dict[str, str]] = []
        if reason:
            notices.append({"code": "training_unavailable", "message": reason})
        if (labels.get("error") or {}).get("code") == "insufficient_scope":
            notices.append({"code": "insufficient_scope", "message": TRAINING_READ_HELP})
        last = next((r for r in self.history(20) if r.get("status") not in ("waiting",)), None)
        too_long = int(((last or {}).get("notes") or {}).get("qa_too_long") or 0)
        if too_long:
            notices.append({"code": "qa_too_long", "message": f"{too_long} QA labels were too long for this host's training budget "
                                                               f"({settings.max_seq_length} tokens)"})
        if self.registry.stale():
            notices.append({"code": "base_changed", "message": "Adapter set aside: the base model changed; retrain"})
        return {
            "available": reason is None, "unavailable_reason": reason, "settings": settings.console_view(), "trainer": self.trainer,
            "timezone": zone_name(self.tz),
            "next_run_at": next_occurrence(now, settings, self.tz).isoformat() if settings.enabled else None,
            "labels": labels, "status": status, "last_check": state.get("last_check"), "active": active_view,
            "versions": self.registry.versions(), "runs": self.runs(HISTORY_LIMIT), "notices": notices,
        }


__all__ = ["OWNED_ELSEWHERE", "RunNotFound", "START_WINDOW", "SettingsError", "TrainingBusy", "TrainingService", "TrainingUnavailable", "latest_occurrence",
           "local_zone", "next_occurrence", "occurrence_on"]
