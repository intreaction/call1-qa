"""The worker loop: resource slots, claims, heartbeats, handler execution and publication.

Local resource slots (``config.slots``) decide how many jobs of each kind run at once. Each pool
claims only against the slots it has free, offering the contract memory slots its jobs use:

=========  ============  =====================================================================
pool       offers        job types
=========  ============  =====================================================================
``mlx``    local_memory  every type (MLX/Ollama entries); size 1, the shared unified-memory slot
``torch``  cpu           ``acoustic_tone``, ``text_sentiment`` (the torch models) and the v2
                        classifier types (``TORCH_TYPES``); on Gemma those freeze
                        ``local_memory`` and ``mlx`` claims them, so ``torch`` gets only a
                        stage-1 re-derive (no model)
``cpu_io`` cpu, io       every other type: code stages, assemblies, audio validation
outbound   outbound      one pool per admin-state connection (none in Stage 2: appliance only)
=========  ============  =====================================================================

For every claimed job the worker checks it has a handler and a usable catalog entry (else it
*releases* the claim with ``reject``), fetches and verifies the inputs (else ``requeue``), runs the
handler while a heartbeat thread renews the lease, then uploads each output as an artifact (small
JSON inline, anything larger through an upload grant) and completes the job with the usage row,
provenance, result publication and any follow-on jobs. Failures go to ``/fail`` with a contract
``JobErrorCode``.

Completion, failure and release requests are spooled before they are sent. When Store cannot be
reached the request stays in the spool and the job's claim stays alive: the heartbeat thread keeps
renewing the lease of every job whose completion or failure is waiting for delivery, and a spool
thread replays the spool (same key, same body) every few seconds and as soon as a pool sees Store
again. So a Store outage shorter than the lease costs no work. A spool left by a crash is replayed
when Process next connects; Store then applies it if the claim is still active, returns the
original receipt if it had committed, or refuses the stale claim (the measurements become late
usage).

A job can also finish while Store is away, before its outputs are uploaded (the usual case: the
outage outlasts the handler, not just the ``/complete`` call). The finished result is then spooled
as a *deferred publication* (``_defer_publication``): the claim, measurements and output files go to
the spool, the slot is freed, and the claim is heartbeated like any undelivered completion. The
replay (``_replay_publication``) first rechecks the claim with a heartbeat: a stale claim becomes
late usage, a requested cancel becomes a ``cancelled`` failure without uploading anything, and
otherwise it plans the follow-ons, uploads the outputs (naturally idempotent) and completes the
job. The same happens after a restart, which also resumes heartbeating the inherited claim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel

from call1.contracts.artifacts import (
    ALLOWED_SENSITIVITY,
    ARTIFACT_CONTENT_CONTRACTS,
    ARTIFACT_CONTENT_MODELS,
    SLOT_MAX_LENGTH,
    Artifact,
    ArtifactKind,
    InlineArtifactCreate,
    Sensitivity,
    UploadGrantRequest,
    draft_test_slot,
)
from call1.contracts.common import ArtifactRef, canonical_digest, canonical_json
from call1.contracts.contents import (
    EscalationTrigger,
    ModelAttemptView,
    PromptInputContent,
    QaAssessmentContent,
    SpeakerRole,
    SummarySynthesisContent,
    TranscriptContent,
    VerdictStatus,
)
from call1.contracts.custody import RouteClass
from call1.contracts.errors import PROVIDER_FAILURE_CODES, JobErrorCode
from call1.contracts.jobs import (
    JOB_TYPE_RULES,
    AttemptProvenance,
    ClaimedJob,
    ClaimRequest,
    CompletionRequest,
    ExecutionClass,
    FailureRequest,
    FollowOnJobs,
    HeartbeatRequest,
    JobOutput,
    JobReleaseRequest,
    JobStatus,
    JobType,
    MemorySlot,
    ProgressNote,
    ResultPublication,
    SlotOffer,
    WorkerCapabilities,
)
from call1.contracts.contents import ResultState
from call1.contracts.usage import (
    BillingUnit,
    LateUsageReport,
    TokenCount,
    TokenSource,
    UsageOutcome,
    UsageRecordInput,
    usage_outcome_for,
)

from .catalog import ProcessCatalog
from .graph import GraphPlanner, PlanError
from .handlers import HandlerRegistry
from .handlers.base import HandlerError, HandlerJob, HandlerResult, InputArtifact, JobCancelled, Output, ReleaseJob, Usage
from .scratch import Scratch, Spool
from .store_client import StoreClient, StoreError, StoreUnavailable

log = logging.getLogger("call1.process.worker")

QA_ASSESSMENT_TYPES = frozenset({JobType.QA_CRITERION, JobType.QA_ESCALATION})
SPOOL_REPLAY_INTERVAL_SECONDS = 5.0

SpoolKey = Tuple[str, int, str]
TORCH_TYPES = frozenset({JobType.ACOUSTIC_TONE, JobType.TEXT_SENTIMENT, JobType.CONTACT_SIGNALS_CATEGORIZE,
                         JobType.CONTACT_SIGNALS_SUBCATEGORIZE})
"""Job types the ``torch`` pool may claim: the tone and sentiment models and the Contact Signals v2
classifier types (stages 1 and 2, section 8.5). On Gemma (decision 24) the classifier stages freeze
the ``local_memory`` slot, so the ``mlx`` pool claims them; the ``torch`` pool gets only a stage-1
re-derive (``graph.CODE_SLOT``, CPU and no model) and would serve a later torch "system one" engine.
Every model run loads, runs and releases inside ``inference_lock``."""
ALL_TYPES = frozenset(JobType)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --- slots -------------------------------------------------------------------------------------


class SlotPool:
    """A counted local resource. ``reserve`` takes every free slot before a claim; unused ones go
    back at once, and each claimed job returns its slot when it ends."""

    def __init__(self, name: str, size: int, memory_slots: Sequence[MemorySlot], job_types: FrozenSet[JobType],
                 outbound_ref: Optional[str] = None) -> None:
        self.name = name
        self.size = size
        self.memory_slots = list(memory_slots)
        self.job_types = job_types
        self.outbound_ref = outbound_ref
        self.in_use = 0
        self._cond = threading.Condition()

    def free_count(self) -> int:
        with self._cond:
            return self.size - self.in_use

    def reserve(self) -> int:
        with self._cond:
            n = self.size - self.in_use
            self.in_use += n
            return n

    def free(self, n: int = 1) -> None:
        if n <= 0:
            return
        with self._cond:
            self.in_use = max(0, self.in_use - n)
            self._cond.notify_all()

    def wait_for_free(self, timeout: float) -> None:
        with self._cond:
            if self.in_use >= self.size:
                self._cond.wait(timeout)

    def describe(self) -> Dict[str, Any]:
        return {"pool": self.name, "size": self.size, "in_use": self.in_use, "memory_slots": [m.value for m in self.memory_slots],
                "outbound_connection_ref": self.outbound_ref}


def build_pools(slots, outbound_refs: Sequence[str] = ()) -> List[SlotPool]:
    pools = [
        SlotPool("mlx", slots.mlx, [MemorySlot.LOCAL_MEMORY], ALL_TYPES),
        SlotPool("torch", slots.torch, [MemorySlot.CPU], TORCH_TYPES),
        SlotPool("cpu_io", slots.cpu_io, [MemorySlot.CPU, MemorySlot.IO], ALL_TYPES - TORCH_TYPES),
    ]
    for ref in outbound_refs:
        pools.append(SlotPool(f"outbound:{ref}", slots.outbound, [MemorySlot.OUTBOUND], ALL_TYPES, outbound_ref=ref))
    return [p for p in pools if p.size > 0]


# --- outputs -----------------------------------------------------------------------------------


def _bounded_slot(value: str) -> str:
    if len(value) <= SLOT_MAX_LENGTH:
        return value
    return value[:SLOT_MAX_LENGTH - 17] + "." + hashlib.sha256(value.encode()).hexdigest()[:16]


def default_slot(job_type: JobType, parameters, kind: ArtifactKind, content: Any) -> str:
    """The artifact slot of a job output (versions and supersession are per slot)."""
    cid = parameters.criterion_id
    if job_type is JobType.QA_CRITERION:
        main = str(cid)
    elif job_type is JobType.QA_ESCALATION:
        main = f"escalation:{cid}"
    elif job_type is JobType.SUMMARY_SEGMENT:
        main = f"segment:{parameters.segment.index if parameters.segment else 0}"
    elif job_type is JobType.SUMMARY_SYNTHESIS:
        if isinstance(content, SummarySynthesisContent) and not content.final:
            main = f"synthesis:{min(content.segment_indexes)}-{max(content.segment_indexes)}"
        else:
            main = "synthesis:final"
    elif job_type in (JobType.CONTACT_SIGNALS_LIFECYCLE, JobType.CONTACT_SIGNALS_RESOLUTION):
        kind_name = parameters.pass_kind.value if parameters.pass_kind else job_type.value
        window = parameters.window
        main = f"{kind_name}:{window.turn_start}-{window.turn_end}" if window else f"{kind_name}:0"
    else:
        main = ""
    return _bounded_slot(main)


def output_roles_ok(rule, outputs: Dict[str, Any]) -> bool:
    """Every required output role of the job type, plus any of its ``optional_outputs`` (contract 1.3.0:
    the ``asr`` job's ``base_transcript`` and ``vocabulary_pass``), and nothing else."""
    roles = set(outputs)
    return set(rule.outputs) <= roles and not (roles - set(rule.outputs) - set(rule.optional_outputs))


def output_kinds(rule, outputs: Dict[str, Any]) -> List[Tuple[str, ArtifactKind]]:
    """(role, kind) of each output to upload: the required roles in declaration order, then the
    optional roles the handler returned."""
    return list(rule.outputs.items()) + [(role, kind) for role, kind in rule.optional_outputs.items() if role in outputs]


def default_sensitivity(kind: ArtifactKind, content: Any, masked_route: bool) -> Sensitivity:
    fixed = ALLOWED_SENSITIVITY.get(kind)
    if fixed and len(fixed) == 1:
        return next(iter(fixed))
    if kind is ArtifactKind.TRANSCRIPT:
        return Sensitivity.MASKED if isinstance(content, TranscriptContent) and content.is_redacted else Sensitivity.RAW
    if kind in (ArtifactKind.SPEAKER_ATTRIBUTION, ArtifactKind.TONE_BLOCKS, ArtifactKind.TEXT_SENTIMENT):
        return Sensitivity.DERIVED
    return Sensitivity.MASKED if masked_route else Sensitivity.RAW


def draft_request_id(parameters) -> Optional[str]:
    value = parameters.extra.get("draft_test_request_id")
    return str(value) if value else None


def _unreachable(exc: StoreError) -> bool:
    """Store could not be reached (or a gateway in front of it could not): worth waiting for."""
    return isinstance(exc, StoreUnavailable) or exc.status in (502, 503, 504)


def _usage_dump(usage: Usage) -> Dict[str, Any]:
    return {
        "tokens_input": usage.tokens_input.model_dump(mode="json") if usage.tokens_input else None,
        "tokens_output": usage.tokens_output.model_dump(mode="json") if usage.tokens_output else None,
        "model_load_seconds": usage.model_load_seconds, "inference_seconds": usage.inference_seconds,
        "audio_seconds_processed": usage.audio_seconds_processed, "peak_memory_bytes": usage.peak_memory_bytes,
        "provider_reported_model_id": usage.provider_reported_model_id,
        "billing_units": [unit.model_dump(mode="json") for unit in usage.billing_units],
    }


def _usage_load(data: Dict[str, Any]) -> Usage:
    return Usage(
        tokens_input=TokenCount.model_validate(data["tokens_input"]) if data.get("tokens_input") else None,
        tokens_output=TokenCount.model_validate(data["tokens_output"]) if data.get("tokens_output") else None,
        model_load_seconds=data.get("model_load_seconds"), inference_seconds=data.get("inference_seconds"),
        audio_seconds_processed=data.get("audio_seconds_processed"), peak_memory_bytes=data.get("peak_memory_bytes"),
        provider_reported_model_id=data.get("provider_reported_model_id"),
        billing_units=[BillingUnit.model_validate(unit) for unit in data.get("billing_units") or []],
    )


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


# --- running jobs ------------------------------------------------------------------------------


@dataclass
class RunningJob:
    claimed: ClaimedJob
    pool: str
    started_monotonic: float
    heartbeat_interval: float
    next_heartbeat: float
    cancel: threading.Event = field(default_factory=threading.Event)
    cancel_requested: bool = False
    lost: bool = False
    inference_started: bool = False
    progress_fraction: Optional[float] = None
    progress_note: Optional[str] = None
    # What this finished attempt is waiting to deliver, kept in the spool: "publish" (its outputs
    # could not be uploaded yet), "complete" or "fail".
    awaiting_delivery: Optional[str] = None
    # Its outputs are already in the spool (a replay that is deferred again does not rewrite them).
    publication_spooled: bool = False

    def set_progress(self, fraction: Optional[float], note: Optional[str]) -> None:
        self.progress_fraction = fraction
        self.progress_note = (note or None) and note[:500]

    def describe(self) -> Dict[str, Any]:
        job = self.claimed.job
        return {"job_id": job.id, "job_type": job.job_type.value, "conversation_id": job.conversation_id, "pool": self.pool,
                "attempt_number": self.claimed.attempt_number, "running_seconds": round(time.monotonic() - self.started_monotonic, 1),
                "cancel_requested": self.cancel_requested, "progress": self.progress_fraction, "awaiting_delivery": self.awaiting_delivery}


@dataclass
class WorkerStats:
    claimed: int = 0
    succeeded: int = 0
    failed: int = 0
    released: int = 0
    lost: int = 0
    last_claim_at: Optional[str] = None
    last_error: Optional[str] = None


class Worker:
    def __init__(self, *, client: StoreClient, registry: HandlerRegistry, catalog: ProcessCatalog, planner: GraphPlanner,
                 installation_id: str, hardware_profile_id: str, worker_id: str, slots, scratch: Scratch, spool: Spool,
                 primary_host: bool = True, poll_interval: float = 2.0, outbound_refs: Sequence[str] = (),
                 replay_interval: float = SPOOL_REPLAY_INTERVAL_SECONDS) -> None:
        self.client = client
        self.registry = registry
        self.catalog = catalog
        self.planner = planner
        self.installation_id = installation_id
        self.hardware_profile_id = hardware_profile_id
        self.worker_id = worker_id
        self.scratch = scratch
        self.spool = spool
        self.primary_host = primary_host
        self.poll_interval = poll_interval
        self.replay_interval = replay_interval
        self.pools = build_pools(slots, outbound_refs)
        self.stats = WorkerStats()
        self.state = "stopped"
        self._running: Dict[str, RunningJob] = {}
        # Attempts whose completion or failure is in the spool, not yet answered by Store. Their
        # claims are heartbeated until the spool delivers the request or Store refuses it.
        self._undelivered: Dict[Tuple[str, int], RunningJob] = {}
        self._inflight: Set[SpoolKey] = set()  # spool entries _send is delivering right now
        self._lock = threading.Lock()
        self._replay_lock = threading.Lock()
        self._replay_now = threading.Event()
        self._submit_lock = threading.Lock()
        self._stopping = threading.Event()
        self._beats_done = threading.Event()
        self._threads: List[threading.Thread] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        self._heartbeat_interval = float(client.parameters.heartbeat_interval_seconds)
        # On-device training's claim pause (docs/OnDeviceTraining.md section 3.3): while set, no pool
        # and no reanalysis consumer claims. Heartbeats and the spool keep running.
        self._pause_lock = threading.Lock()
        self._claims_paused: Optional[Dict[str, Any]] = None
        # Wakes idle pool loops when a delivery released or created jobs (latency only: the
        # poll_interval poll stays as the fallback, e.g. for jobs another host released).
        self._work_cond = threading.Condition()
        self._work_generation = 0

    # --- the training pause ------------------------------------------------------------------

    def pause_claims(self, run_id: str, until: Optional[datetime] = None) -> None:
        """Stop every pool loop (and, through ``claims_paused``, the reanalysis consumer) from
        claiming. Jobs already claimed run to their end; ``idle()`` says when none is left."""
        with self._pause_lock:
            self._claims_paused = {"run_id": run_id, "since": utcnow().isoformat(), "until": until.isoformat() if until else None}
        log.info("claims paused for on-device training run %s", run_id)

    def resume_claims(self, run_id: Optional[str] = None) -> None:
        """Resume claiming. With ``run_id``, only a pause that run set is lifted."""
        with self._pause_lock:
            if self._claims_paused is None or (run_id is not None and self._claims_paused.get("run_id") != run_id):
                return
            self._claims_paused = None
        log.info("claims resumed")

    @property
    def claims_paused(self) -> Optional[Dict[str, Any]]:
        with self._pause_lock:
            return dict(self._claims_paused) if self._claims_paused else None

    def busy_pools(self, names: Optional[Sequence[str]] = None) -> List[str]:
        """Pools with a slot in use: a claim in flight or a claimed job not yet ended. ``names``
        limits the check (the training start rule asks ``mlx`` and ``torch``)."""
        return [p.name for p in self.pools if (names is None or p.name in names) and p.free_count() < p.size]

    def idle(self) -> bool:
        """No claim in flight and no claimed job running or awaiting its slot's release."""
        return not self.busy_pools() and not self.running()

    # --- capabilities and claims -------------------------------------------------------------

    def offered_types(self, pool: SlotPool) -> List[JobType]:
        types = set(pool.job_types) & set(self.registry.job_types())
        if not self.primary_host:
            types = {t for t in types if JOB_TYPE_RULES[t].execution_class is not ExecutionClass.PRIMARY_HOST}
        return sorted(types, key=lambda t: t.value)

    def capabilities(self, pool: SlotPool, count: int) -> WorkerCapabilities:
        route_classes = [RouteClass.APPLIANCE]
        return WorkerCapabilities(
            worker_id=self.worker_id, installation_id=self.installation_id, hardware_profile_id=self.hardware_profile_id,
            primary_host=self.primary_host, job_types=self.offered_types(pool), route_classes=route_classes,
            qualified_entries=self.catalog.qualified_entries(),
            slot_offers=[SlotOffer(memory_slot=m, outbound_connection_ref=pool.outbound_ref, count=count) for m in pool.memory_slots],
        )

    def claim(self, pool: SlotPool) -> List[ClaimedJob]:
        """Reserve the pool's free slots, claim at most that many jobs, give back the rest."""
        if not self.offered_types(pool) or self.claims_paused is not None:
            return []
        reserved = pool.reserve()
        if reserved <= 0:
            return []
        claimed: List[ClaimedJob] = []
        try:
            limit = min(reserved, self.client.parameters.max_claim_batch, 64)
            request = ClaimRequest(worker=self.capabilities(pool, reserved), max_jobs=limit)
            response = self.client.claim_jobs(request)
            claimed = list(response.jobs)
            self._heartbeat_interval = float(response.heartbeat_interval_seconds)
            if claimed:
                self.stats.claimed += len(claimed)
                self.stats.last_claim_at = utcnow().isoformat()
        except StoreError as exc:
            if exc.code == "forbidden" and exc.details.get("reason") == "not_primary_host":
                log.warning("Store says this installation is not the primary host; offering only non-ML jobs")
                self.primary_host = False
            else:
                self.stats.last_error = f"claim: {exc.code}"
                raise
        finally:
            pool.free(reserved - len(claimed))
        return claimed

    # --- execution ---------------------------------------------------------------------------

    def execute(self, claimed: ClaimedJob, pool: SlotPool) -> str:
        """Run one claimed job to its end (complete, fail or release). Returns the outcome."""
        try:
            return self._execute(claimed, pool.name)
        except Exception:  # never let one job take the loop down
            log.exception("job %s: unexpected worker error", claimed.job.id)
            self.stats.last_error = "worker error"
            return "error"
        finally:
            pool.free(1)

    def _execute(self, claimed: ClaimedJob, pool_name: str) -> str:
        job = claimed.job
        now = time.monotonic()
        running = RunningJob(claimed=claimed, pool=pool_name, started_monotonic=now, heartbeat_interval=self._heartbeat_interval,
                             next_heartbeat=now + self._heartbeat_interval)
        with self._lock:
            self._running[job.id] = running
        scratch_dir = self.scratch.attempt_dir(job.id, claimed.attempt_number)
        try:
            return self._run(claimed, running, scratch_dir)
        finally:
            with self._lock:
                self._running.pop(job.id, None)
            Scratch.remove(scratch_dir)

    def _fetch(self, artifact: Artifact, dest: Optional[Path]) -> Union[bytes, Path]:
        return self.client.download(artifact, dest)

    def _run(self, claimed: ClaimedJob, running: RunningJob, scratch_dir: Path) -> str:
        job = claimed.job
        handler = self.registry.get(job.job_type)
        if handler is None:
            return self._release(claimed, "reject", JobErrorCode.MODEL_UNAVAILABLE, "no handler for this job type on this Process host")
        if self._stopping.is_set():
            return self._release(claimed, "requeue", JobErrorCode.RESOURCE_UNAVAILABLE, "Process is stopping")
        entry = None
        if job.selection is not None:
            entry = self.catalog.by_ref(job.selection.catalog_entry)
            if entry is None:
                return self._release(claimed, "reject", JobErrorCode.MODEL_UNAVAILABLE, "the frozen catalog entry is not installed here")
            if not self.catalog.usable(entry, job.selection.purpose):
                return self._release(claimed, "reject", JobErrorCode.MODEL_UNQUALIFIED, "the frozen catalog entry is not qualified here")
        inputs: Dict[str, Optional[InputArtifact]] = {}
        for resolved in claimed.inputs:
            inputs[resolved.role] = InputArtifact(resolved.role, resolved.artifact, self._fetch, scratch_dir) if resolved.artifact else None
        hjob = HandlerJob(claimed, inputs, scratch_dir, catalog_entry=entry, cancel_event=running.cancel, progress=running.set_progress,
                          call_metadata=lambda: self.client.get_conversation(job.conversation_id).call_metadata)
        try:
            handler.ready(hjob)
        except ReleaseJob as release:
            return self._release(claimed, release.disposition, release.code, release.detail, release.not_before_seconds)
        try:
            for item in inputs.values():
                if item is None:
                    continue
                if item.artifact.content_type == "application/json" and item.artifact.size_bytes <= (8 << 20):
                    item.read_bytes()
                else:
                    item.path()
        except (StoreError, OSError) as exc:
            log.warning("job %s: input fetch failed (%s)", job.id, getattr(exc, "code", type(exc).__name__))
            return self._release(claimed, "requeue", JobErrorCode.INPUT_UNAVAILABLE, "an input could not be fetched from Store", 30)
        running.inference_started = True
        started = time.monotonic()
        try:
            result = handler.run(hjob)
        except ReleaseJob as release:
            return self._release(claimed, release.disposition, release.code, release.detail, release.not_before_seconds)
        except JobCancelled:
            return self._after_cancel(claimed, running, handler, time.monotonic() - started)
        except HandlerError as exc:
            if running.lost:
                return self._lost(claimed, exc.usage, time.monotonic() - started, running)
            return self._handler_failed(claimed, hjob, handler, exc, time.monotonic() - started, running)
        except Exception as exc:
            log.exception("job %s: handler %s raised", job.id, handler.adapter_id)
            if running.lost:
                return self._lost(claimed, None, time.monotonic() - started, running)
            return self._fail(claimed, handler, JobErrorCode.WORKER_CRASHED, f"the handler raised {type(exc).__name__}", None,
                              time.monotonic() - started, running)
        run_seconds = time.monotonic() - started
        if running.lost:
            return self._lost(claimed, result.usage, run_seconds, running)
        if running.cancel_requested:
            return self._after_cancel(claimed, running, handler, run_seconds)
        return self._publish(claimed, hjob, handler, result, run_seconds, running)

    # --- publication --------------------------------------------------------------------------

    def _upload(self, claimed: ClaimedJob, role: str, kind: ArtifactKind, output: Output) -> Artifact:
        job = claimed.job
        contract = ARTIFACT_CONTENT_CONTRACTS[kind]
        model = ARTIFACT_CONTENT_MODELS[contract]
        masked = bool(job.selection and job.selection.route.masked)
        slot = output.slot if output.slot is not None else default_slot(job.job_type, job.parameters, kind, output.content)
        draft = draft_request_id(job.parameters)
        if draft:
            slot = draft_test_slot(draft, slot)
        sensitivity = output.sensitivity or default_sensitivity(kind, output.content, masked)
        if model is not None:
            content = output.content
            if not isinstance(content, BaseModel):
                raw = content.read_bytes() if isinstance(content, Path) else bytes(content)
                content = model.model_validate(json.loads(raw.decode("utf-8")))
            if not isinstance(content, model):
                content = model.model_validate(content.model_dump(mode="json"))
            payload = content.model_dump(mode="json")
            data = canonical_json(payload)
            checksum = "sha256:" + hashlib.sha256(data).hexdigest()
            if len(data) <= self.client.parameters.inline_artifact_max_bytes:
                body = InlineArtifactCreate(kind=kind, slot=slot, content_type="application/json", size_bytes=len(data), checksum=checksum,
                                            content_contract=contract, sensitivity=sensitivity, producing_job_id=job.id, labels=dict(output.labels),
                                            payload=payload, claim_token=claimed.claim_token)
                return self.client.create_inline_artifact(job.conversation_id, body)
            source: Union[bytes, Path] = data
            content_type = "application/json"
        else:
            raw_source = output.content
            if isinstance(raw_source, BaseModel):
                raise TypeError(f"{kind.value} is opaque content; return bytes or a file path")
            source = Path(raw_source) if isinstance(raw_source, Path) else bytes(raw_source)
            digest = hashlib.sha256()
            size = 0
            if isinstance(source, Path):
                with source.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
                        size += len(chunk)
            else:
                digest.update(source)
                size = len(source)
            checksum = "sha256:" + digest.hexdigest()
            data = b""
            content_type = output.content_type or "application/octet-stream"
        size_bytes = len(data) if model is not None else size
        request = UploadGrantRequest(kind=kind, slot=slot, content_type=content_type, size_bytes=size_bytes, checksum=checksum,
                                     content_contract=contract, sensitivity=sensitivity, producing_job_id=job.id, labels=dict(output.labels),
                                     claim_token=claimed.claim_token)
        return self.client.upload_artifact(job.conversation_id, request, source)

    def _usage(self, usage: Optional[Usage], outcome: UsageOutcome, error_code: Optional[JobErrorCode], run_seconds: float,
               running: RunningJob) -> UsageRecordInput:
        usage = usage or Usage()
        unavailable = TokenCount(source=TokenSource.UNAVAILABLE)
        inference = usage.inference_seconds if usage.inference_seconds is not None else run_seconds
        total = max(time.monotonic() - running.started_monotonic, inference)
        return UsageRecordInput(
            tokens_input=usage.tokens_input or unavailable, tokens_output=usage.tokens_output or unavailable, slot_wait_seconds=0,
            model_load_seconds=usage.model_load_seconds, inference_seconds=round(max(0.0, inference), 6), total_seconds=round(total, 6),
            audio_seconds_processed=usage.audio_seconds_processed, peak_memory_bytes=usage.peak_memory_bytes,
            hardware_profile_id=self.hardware_profile_id, outcome=outcome, error_code=error_code, billing_units=list(usage.billing_units),
            provider_reported_model_id=usage.provider_reported_model_id,
        )

    def _provenance(self, claimed: ClaimedJob, handler, model_revision: Optional[str] = None,
                    provider_model_id: Optional[str] = None) -> AttemptProvenance:
        job = claimed.job
        selection = job.selection
        return AttemptProvenance(
            worker_id=self.worker_id, installation_id=self.installation_id, adapter_id=handler.adapter_id, adapter_version=handler.adapter_version,
            model_revision=(model_revision or selection.model_revision) if selection else None, provider_reported_model_id=provider_model_id,
            route=selection.route if selection else None, resource_estimate_used=job.resource_estimate,
        )

    def _publish(self, claimed: ClaimedJob, hjob: HandlerJob, handler, result: HandlerResult, run_seconds: float, running: RunningJob,
                 *, outcome: UsageOutcome = UsageOutcome.SUCCEEDED, error_code: Optional[JobErrorCode] = None) -> str:
        job = claimed.job
        rule = JOB_TYPE_RULES[job.job_type]
        if not output_roles_ok(rule, result.outputs):
            log.error("job %s: handler returned outputs %s, expected %s (optional %s)", job.id, sorted(result.outputs), sorted(rule.outputs),
                      sorted(rule.optional_outputs))
            return self._fail(claimed, handler, JobErrorCode.STORE_PUBLICATION_FAILED, "the handler did not return the job type's outputs",
                              result.usage, run_seconds, running)
        # The outcome before follow-on planning: a deferred publication replays from here.
        base_outcome, base_error = outcome, error_code
        follow_on: Optional[FollowOnJobs] = None
        try:
            if job.job_type in QA_ASSESSMENT_TYPES:
                outcome, error_code, follow_on = self._qa_outcome(claimed, hjob, result, outcome, error_code)
            elif job.job_type is JobType.ASR:
                follow_on = self._summary_follow_on(claimed, result)
        except HandlerError as exc:
            return self._fail(claimed, handler, exc.code, exc.detail, result.usage, run_seconds, running)
        except (PlanError, StoreError) as exc:
            if isinstance(exc, StoreError) and exc.code == "claim_token_stale":
                return self._lost(claimed, result.usage, run_seconds, running)
            if isinstance(exc, StoreError) and _unreachable(exc):
                log.warning("job %s: Store unreachable while planning follow-ons; the finished result is spooled", job.id)
                return self._defer_publication(claimed, handler, result, run_seconds, running, base_outcome, base_error)
            log.error("job %s: follow-on planning failed: %s", job.id, exc)
            return self._fail(claimed, handler, JobErrorCode.CONFIGURATION_ERROR, "follow-on jobs could not be planned", result.usage,
                              run_seconds, running)
        try:
            outputs = []
            for role, kind in output_kinds(rule, result.outputs):
                artifact = self._upload(claimed, role, kind, result.outputs[role])
                outputs.append(JobOutput(role=role, artifact_id=artifact.id, checksum=artifact.checksum))
        except StoreError as exc:
            if exc.code == "claim_token_stale":
                return self._lost(claimed, result.usage, run_seconds, running)
            if _unreachable(exc):
                # Keep the finished work: spool the outputs, keep the claim alive, upload on replay.
                log.warning("job %s: output upload failed (%s); the finished result is spooled", job.id, exc.code)
                return self._defer_publication(claimed, handler, result, run_seconds, running, base_outcome, base_error)
            log.error("job %s: output upload failed: %s", job.id, exc)
            return self._fail(claimed, handler, JobErrorCode.STORE_PUBLICATION_FAILED, "an output could not be stored",
                              result.usage, run_seconds, running)
        except (ValueError, TypeError) as exc:
            log.error("job %s: output is not valid contract content: %s", job.id, type(exc).__name__)
            return self._fail(claimed, handler, JobErrorCode.STORE_PUBLICATION_FAILED, "an output is not valid contract content",
                              result.usage, run_seconds, running)
        publication = None
        if rule.publishes is not None and not draft_request_id(job.parameters):
            publication = ResultPublication(kind=rule.publishes, state=ResultState.PARTIAL if result.partial_reason else ResultState.AVAILABLE,
                                            partial_reason=result.partial_reason[:200] if result.partial_reason else None)
        body = CompletionRequest(
            claim_token=claimed.claim_token, completion_key=f"complete:{job.id}:{claimed.attempt_number}", outputs=outputs,
            usage=self._usage(result.usage, outcome, error_code, run_seconds, running),
            provenance=self._provenance(claimed, handler, result.model_revision, result.usage.provider_reported_model_id),
            result=publication, follow_on=follow_on,
        )
        return self._send(claimed, "complete", body, running, handler)

    # --- deferred publication (Store unreachable when the handler finished) ---------------------

    def _defer_publication(self, claimed: ClaimedJob, handler, result: HandlerResult, run_seconds: float, running: RunningJob,
                           outcome: UsageOutcome, error_code: Optional[JobErrorCode]) -> str:
        """Keep a finished attempt's work through a Store outage. Its outputs and the facts needed to
        publish them go to the spool (``<job>.<attempt>.publish.json`` plus ``outputs/<job>.<attempt>/``),
        the claim is heartbeated like any undelivered completion, and ``recover`` rechecks the claim,
        uploads the outputs and completes the job once Store answers again (after a restart too)."""
        job = claimed.job
        attempt = claimed.attempt_number
        key: SpoolKey = (job.id, attempt, "publish")
        with self._lock:
            self._inflight.add(key)
        try:
            if not running.publication_spooled:
                try:
                    self._spool_publication(claimed, handler, result, run_seconds, running, outcome, error_code)
                except (OSError, ValueError, TypeError) as exc:
                    log.error("job %s: the finished result could not be spooled (%s)", job.id, type(exc).__name__)
                    self.spool.discard_publication(job.id, attempt)
                    return self._fail(claimed, handler, JobErrorCode.STORE_PUBLICATION_FAILED, "the finished result could not be spooled",
                                      result.usage, run_seconds, running)
                running.publication_spooled = True
            self.stats.last_error = "upload: store_unavailable"
            running.awaiting_delivery = "publish"
            with self._lock:
                self._undelivered[(job.id, attempt)] = running
            return "spooled"
        finally:
            with self._lock:
                self._inflight.discard(key)

    def _spool_publication(self, claimed: ClaimedJob, handler, result: HandlerResult, run_seconds: float, running: RunningJob,
                           outcome: UsageOutcome, error_code: Optional[JobErrorCode]) -> None:
        job = claimed.job
        attempt = claimed.attempt_number
        folder = self.spool.outputs_dir(job.id, attempt)
        outputs: Dict[str, Any] = {}
        for index, role in enumerate(sorted(result.outputs)):
            output = result.outputs[role]
            name = f"{index}.bin"
            target = folder / name
            content = output.content
            if isinstance(content, BaseModel):
                _write_private(target, canonical_json(content.model_dump(mode="json")))
            elif isinstance(content, Path):
                if content.resolve() != target.resolve():
                    shutil.copyfile(content, target)
                    os.chmod(target, 0o600)
            else:
                _write_private(target, bytes(content))
            outputs[role] = {"file": name, "content_type": output.content_type,
                             "sensitivity": output.sensitivity.value if output.sensitivity else None,
                             "slot": output.slot, "labels": dict(output.labels)}
        body = {
            "claimed": claimed.model_dump(mode="json"),
            "adapter_id": handler.adapter_id, "adapter_version": handler.adapter_version,
            "outcome": outcome.value, "error_code": error_code.value if error_code else None,
            "run_seconds": run_seconds, "elapsed_seconds": time.monotonic() - running.started_monotonic,
            "model_revision": result.model_revision, "partial_reason": result.partial_reason,
            "usage": _usage_dump(result.usage or Usage()), "outputs": outputs,
        }
        # The entry is written last (atomically), so an entry always has its output files.
        self.spool.write(job.id, attempt, "publish", body)

    def _load_publication(self, job_id: str, attempt: int, body: Dict[str, Any]) -> Tuple[ClaimedJob, HandlerResult]:
        claimed = ClaimedJob.model_validate(body["claimed"])
        if claimed.job.id != job_id or claimed.attempt_number != attempt:
            raise ValueError("spooled publication does not match its entry")
        rule = JOB_TYPE_RULES[claimed.job.job_type]
        folder = self.spool.outputs_dir(job_id, attempt, create=False)
        outputs: Dict[str, Output] = {}
        for role, meta in body["outputs"].items():
            kind = rule.outputs.get(role) or rule.optional_outputs[role]
            path = folder / str(meta["file"])
            if path.parent != folder:
                raise ValueError("spooled output path escapes its folder")
            model = ARTIFACT_CONTENT_MODELS[ARTIFACT_CONTENT_CONTRACTS[kind]]
            if model is not None:
                content: Any = model.model_validate(json.loads(path.read_bytes().decode("utf-8")))
            elif path.is_file():
                content = path
            else:
                raise FileNotFoundError(str(path))
            outputs[role] = Output(content, content_type=meta.get("content_type"),
                                   sensitivity=Sensitivity(meta["sensitivity"]) if meta.get("sensitivity") else None,
                                   slot=meta.get("slot"), labels=dict(meta.get("labels") or {}))
        result = HandlerResult(outputs=outputs, usage=_usage_load(body.get("usage") or {}), model_revision=body.get("model_revision"),
                               partial_reason=body.get("partial_reason"))
        return claimed, result

    def _replay_publication(self, job_id: str, attempt: int, body: Dict[str, Any]) -> str:
        """Finish a deferred publication: recheck the claim (stale → late usage; cancel requested →
        a ``cancelled`` failure), then upload the outputs and complete the job. Returns
        ``"unavailable"`` while Store still cannot be reached (the entry stays in the spool)."""
        try:
            claimed, result = self._load_publication(job_id, attempt, body)
            outcome = UsageOutcome(body.get("outcome") or UsageOutcome.SUCCEEDED.value)
            error_code = JobErrorCode(body["error_code"]) if body.get("error_code") else None
            run_seconds = float(body.get("run_seconds") or 0.0)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            log.error("spooled publication for %s is unreadable (%s); dropped, so its lease expires and Store retries the job",
                      job_id, type(exc).__name__)
            self.spool.discard_publication(job_id, attempt)
            self._delivered(job_id, attempt)
            return "dropped"
        handler = SimpleNamespace(adapter_id=str(body.get("adapter_id") or "unknown"), adapter_version=str(body.get("adapter_version") or "0"))
        with self._lock:
            running = self._undelivered.get((job_id, attempt))
            if running is None:  # after a restart: heartbeat the claim again while it waits
                now = time.monotonic()
                running = RunningJob(claimed=claimed, pool="spool", started_monotonic=now - float(body.get("elapsed_seconds") or 0.0),
                                     heartbeat_interval=self._heartbeat_interval, next_heartbeat=now + self._heartbeat_interval,
                                     awaiting_delivery="publish", publication_spooled=True)
                self._undelivered[(job_id, attempt)] = running
        # Recheck the claim before uploading anything.
        try:
            beat = self.client.heartbeat(job_id, HeartbeatRequest(claim_token=claimed.claim_token))
            running.next_heartbeat = time.monotonic() + running.heartbeat_interval
        except StoreError as exc:
            if _unreachable(exc):
                return "unavailable"
            if exc.code == "claim_token_stale":
                outcome_text = self._lost(claimed, result.usage, run_seconds, running)
            else:
                log.warning("spooled publication for %s not applied: %s", job_id, exc.code)
                outcome_text = "refused"
            self.spool.discard_publication(job_id, attempt)
            self._delivered(job_id, attempt)
            return outcome_text
        if beat.cancel_requested:
            running.cancel_requested = True
            outcome_text = self._fail(claimed, handler, JobErrorCode.CANCELLED, "cancel requested", result.usage, run_seconds, running)
        else:
            scratch_dir = self.scratch.attempt_dir(job_id, attempt)
            try:
                job = claimed.job
                entry = self.catalog.by_ref(job.selection.catalog_entry) if job.selection is not None else None
                inputs: Dict[str, Optional[InputArtifact]] = {
                    r.role: (InputArtifact(r.role, r.artifact, self._fetch, scratch_dir) if r.artifact else None) for r in claimed.inputs}
                hjob = HandlerJob(claimed, inputs, scratch_dir, catalog_entry=entry, cancel_event=running.cancel, progress=running.set_progress)
                outcome_text = self._publish(claimed, hjob, handler, result, run_seconds, running, outcome=outcome, error_code=error_code)
            finally:
                Scratch.remove(scratch_dir)
            if outcome_text == "spooled" and running.awaiting_delivery == "publish":
                return "unavailable"  # deferred again; the entry and its outputs stay
        self.spool.discard_publication(job_id, attempt)
        if outcome_text != "spooled":
            self._delivered(job_id, attempt)
        return outcome_text

    def _qa_outcome(self, claimed: ClaimedJob, hjob: HandlerJob, result: HandlerResult, outcome: UsageOutcome,
                    error_code: Optional[JobErrorCode]):
        """Usage outcome for an assessment, and the escalation follow-on when its trigger fires."""
        job = claimed.job
        output = result.outputs["assessment"]
        assessment = output.content
        if not isinstance(assessment, QaAssessmentContent):
            assessment = QaAssessmentContent.model_validate(assessment.model_dump(mode="json") if isinstance(assessment, BaseModel)
                                                            else json.loads(bytes(assessment).decode()))  # type: ignore[arg-type]
        if outcome is UsageOutcome.SUCCEEDED and assessment.trigger is EscalationTrigger.INVALID_ANSWER:
            outcome, error_code = UsageOutcome.VALIDATION_REJECTED, JobErrorCode.VALIDATION_REJECTED
        follow_on = None
        if job.job_type is JobType.QA_CRITERION:
            criterion = hjob.criterion()
            reason = assessment.trigger or EscalationTrigger.ALWAYS
            when = criterion.check.escalation_when
            if reason in when or EscalationTrigger.ALWAYS in when:
                scorecard_id = self._dependent_job_id(job.graph_id, str(job.parameters.extra.get("scorecard_ref") or ""))
                if scorecard_id:
                    pinned = [(r.role, ArtifactRef(artifact_id=r.artifact.id, checksum=r.artifact.checksum)) for r in claimed.inputs if r.artifact]
                    follow_on = self.planner.escalation_follow_on(
                        job_id=job.id, parameters=job.parameters, pinned_inputs=pinned, trigger=reason, scorecard_job_id=scorecard_id,
                        audio_seconds=job.resource_estimate.audio_seconds)
            update = {"escalation_requested": follow_on is not None}
            if follow_on is not None and assessment.trigger is None:
                update["trigger"] = EscalationTrigger.ALWAYS
            assessment = assessment.model_copy(update=update)
        elif assessment.escalation_requested:
            assessment = assessment.model_copy(update={"escalation_requested": False})
        output.content = QaAssessmentContent.model_validate(assessment.model_dump(mode="json"))
        return outcome, error_code, follow_on

    def _dependent_job_id(self, graph_id: str, ref: str) -> Optional[str]:
        if not ref:
            return None
        graph = self.client.get_job_graph(graph_id)
        found = next((j for j in graph.jobs if j.ref == ref), None)
        if found is None or found.status is not JobStatus.BLOCKED:
            return None
        return found.job_id

    def _summary_follow_on(self, claimed: ClaimedJob, result: HandlerResult) -> Optional[FollowOnJobs]:
        job = claimed.job
        if not job.parameters.extra.get("summary_plan"):
            return None
        transcript = result.outputs["transcript"].content
        if not isinstance(transcript, TranscriptContent):
            transcript = TranscriptContent.model_validate(transcript.model_dump(mode="json") if isinstance(transcript, BaseModel)
                                                          else json.loads(bytes(transcript).decode()))  # type: ignore[arg-type]
        graph = self.client.get_job_graph(job.graph_id)
        return self.planner.summary_follow_on(job.id, dict(job.parameters.extra), transcript, graph, job.resource_estimate.audio_seconds)

    # --- failure paths -----------------------------------------------------------------------

    def _handler_failed(self, claimed: ClaimedJob, hjob: HandlerJob, handler, exc: HandlerError, run_seconds: float, running: RunningJob) -> str:
        job = claimed.job
        if job.job_type in QA_ASSESSMENT_TYPES and exc.code in PROVIDER_FAILURE_CODES and claimed.final_attempt:
            route = job.selection.route if job.selection else None
            if not exc.code.value.startswith("pro1_") and (route is None or route.route_class is not RouteClass.CALL1_CONFIDENTIAL):
                try:
                    result = self._provider_failure_result(claimed, hjob, exc, run_seconds)
                except HandlerError:
                    result = None
                if result is not None:
                    return self._publish(claimed, hjob, handler, result, run_seconds, running, outcome=UsageOutcome.FAILED, error_code=exc.code)
        return self._fail(claimed, handler, exc.code, exc.detail, exc.usage, run_seconds, running)

    def _provider_failure_result(self, claimed: ClaimedJob, hjob: HandlerJob, exc: HandlerError, run_seconds: float) -> HandlerResult:
        """A final-attempt provider failure recorded as a FLAGGED assessment (the pre-split
        router's behaviour), with the escalation it may trigger."""
        job = claimed.job
        criterion = hjob.criterion()
        selection = job.selection
        reasoning = "The model provider failed on the final attempt. Human review is required."
        attempt = ModelAttemptView(
            job_id=job.id, attempt_number=claimed.attempt_number, catalog_entry_id=selection.catalog_entry.entry_id if selection else "unknown",
            model_revision=selection.model_revision if selection else "unknown", route_class=selection.route.route_class.value if selection else "appliance",
            destination_host=selection.route.destination_host if selection else "in-process", status=VerdictStatus.FLAGGED, reasoning=reasoning,
            trigger=EscalationTrigger.PROVIDER_ERROR, latency_ms=int(run_seconds * 1000), error_code=exc.code)
        assessment = QaAssessmentContent(
            criterion_id=criterion.criterion_id, assessment_kind="escalation" if job.job_type is JobType.QA_ESCALATION else "primary",
            status=VerdictStatus.FLAGGED, confidence=0.0, reasoning=reasoning, speaker=criterion.check.speaker or SpeakerRole.AGENT,
            trigger=EscalationTrigger.PROVIDER_ERROR, escalation_requested=False, attempt=attempt)
        prompt = exc.outputs.get("prompt_input") or Output(PromptInputContent(
            template_id=f"{job.job_type.value}.unsent", template_version="1", prompt_digest=canonical_digest([]),
            masked=bool(selection and selection.route.masked), inputs=[i.ref for i in hjob.inputs.values() if i is not None]))
        return HandlerResult(outputs={"assessment": Output(assessment), "prompt_input": prompt}, usage=exc.usage or Usage())

    def _after_cancel(self, claimed: ClaimedJob, running: RunningJob, handler, run_seconds: float) -> str:
        if running.lost:
            return self._lost(claimed, None, run_seconds, running)
        return self._fail(claimed, handler, JobErrorCode.CANCELLED, "cancel requested", None, run_seconds, running)

    def _fail(self, claimed: ClaimedJob, handler, code: JobErrorCode, detail: Optional[str], usage: Optional[Usage], run_seconds: float,
              running: RunningJob) -> str:
        job = claimed.job
        if code is JobErrorCode.LEASE_EXPIRED:
            code = JobErrorCode.WORKER_CRASHED
        body = FailureRequest(
            claim_token=claimed.claim_token, completion_key=f"fail:{job.id}:{claimed.attempt_number}", error_code=code,
            error_detail=(detail or None) and detail[:500], usage=self._usage(usage, usage_outcome_for(code), code, run_seconds, running),
            provenance=self._provenance(claimed, handler) if handler is not None else None,
        )
        return self._send(claimed, "fail", body, running, handler)

    def _release(self, claimed: ClaimedJob, disposition: str, code: JobErrorCode, detail: Optional[str] = None,
                 not_before_seconds: Optional[float] = None) -> str:
        job = claimed.job
        try:
            body = JobReleaseRequest(
                claim_token=claimed.claim_token, completion_key=f"release:{job.id}:{claimed.attempt_number}", disposition=disposition,
                reason_code=code, detail=(detail or None) and detail[:500],
                not_before=(utcnow() + timedelta(seconds=not_before_seconds)) if disposition == "requeue" and not_before_seconds else None)
        except ValueError:
            running = RunningJob(claimed=claimed, pool="-", started_monotonic=time.monotonic(), heartbeat_interval=0, next_heartbeat=0)
            return self._fail(claimed, self.registry.get(job.job_type), code, detail, None, 0.0, running)
        return self._send(claimed, "release", body, None, None)

    def _lost(self, claimed: ClaimedJob, usage: Optional[Usage], run_seconds: float, running: RunningJob) -> str:
        """The claim is no longer active (lease expired): attach this attempt's measurements to the
        abandoned row Store synthesized, once, and publish nothing."""
        self.stats.lost += 1
        try:
            report = LateUsageReport(claim_token=claimed.claim_token,
                                     usage=self._usage(usage, UsageOutcome.SUCCEEDED, None, run_seconds, running))
            self.client.attach_late_usage(claimed.job.id, claimed.attempt_number, report)
        except StoreError as exc:
            log.info("job %s: late usage not attached (%s)", claimed.job.id, exc.code)
        return "lost"

    # --- sending with the spool ----------------------------------------------------------------

    def _send(self, claimed: ClaimedJob, operation: str, body, running: Optional[RunningJob], handler) -> str:
        job = claimed.job
        attempt = claimed.attempt_number
        key: SpoolKey = (job.id, attempt, operation)
        with self._lock:
            self._inflight.add(key)
        try:
            self.spool.write(job.id, attempt, operation, body.model_dump(mode="json"))
            call: Callable = {"complete": self.client.complete_job, "fail": self.client.fail_job, "release": self.client.release_job}[operation]
            try:
                receipt = call(job.id, body)
            except StoreError as exc:
                if isinstance(exc, StoreUnavailable) or (exc.status or 0) >= 500:
                    self.stats.last_error = f"{operation}: {exc.code}"
                    log.warning("job %s: %s not delivered (%s); kept in the spool and replayed while Process runs", job.id, operation, exc.code)
                    if running is not None and operation != "release":
                        # Keep the claim alive until the spool delivers the result, so a short
                        # outage does not expire the lease and redo the work.
                        running.awaiting_delivery = operation
                        with self._lock:
                            self._undelivered[(job.id, attempt)] = running
                    return "spooled"
                self.spool.remove(job.id, attempt, operation)
                if exc.code == "job_cancelling" and operation == "complete" and running is not None:
                    return self._fail(claimed, handler, JobErrorCode.CANCELLED, "cancel requested", None, 0.0, running)
                if exc.code == "claim_token_stale":
                    if running is not None and operation != "release":
                        return self._lost(claimed, None, 0.0, running)
                    self.stats.lost += 1
                    return "lost"
                self.stats.last_error = f"{operation}: {exc.code}"
                log.error("job %s: Store refused %s: %s %s", job.id, operation, exc.code, exc.details)
                if operation == "complete" and running is not None:
                    return self._fail(claimed, handler, JobErrorCode.STORE_PUBLICATION_FAILED, f"Store refused the completion ({exc.code})",
                                      None, 0.0, running)
                return "refused"
            self.spool.remove(job.id, attempt, operation)
            self._count_delivered(operation)
            self._wake_if_ready(receipt)
            return {"complete": "succeeded", "fail": "failed", "release": "released"}[operation]
        finally:
            with self._lock:
                self._inflight.discard(key)

    def _count_delivered(self, operation: str) -> None:
        if operation == "complete":
            self.stats.succeeded += 1
        elif operation == "fail":
            self.stats.failed += 1
        else:
            self.stats.released += 1

    def _wake_if_ready(self, receipt) -> None:
        """A completion or failure that released dependents (or created follow-on jobs) made new
        work claimable now: wake the idle pool loops instead of leaving it to the next poll."""
        if getattr(receipt, "released_job_ids", None) or getattr(receipt, "created_job_ids", None):
            self.notify_work()

    def notify_work(self) -> None:
        """Wake every pool loop waiting in ``_wait_for_work``."""
        with self._work_cond:
            self._work_generation += 1
            self._work_cond.notify_all()

    def _wait_for_work(self, seen: int) -> None:
        """Idle wait: up to ``poll_interval``, cut short by ``notify_work`` (including one that
        arrived after ``seen`` was read, while the claim was in flight) or by ``stop``."""
        with self._work_cond:
            self._work_cond.wait_for(lambda: self._work_generation != seen or self._stopping.is_set(), timeout=self.poll_interval)

    def _delivered(self, job_id: str, attempt: int) -> None:
        with self._lock:
            self._undelivered.pop((job_id, attempt), None)

    def recover(self) -> int:
        """Replay spooled completion, failure and release requests (same key, same body). Called
        at connect, periodically by the spool thread, and when a pool sees Store again. Returns
        how many Store accepted; stops at the first sign Store is still unreachable."""
        models = {"complete": CompletionRequest, "fail": FailureRequest, "release": JobReleaseRequest}
        calls = {"complete": self.client.complete_job, "fail": self.client.fail_job, "release": self.client.release_job}
        replayed = 0
        with self._replay_lock:
            self.spool.sweep_outputs()
            for item in list(self.spool.pending()):
                operation, job_id, attempt = item.get("operation"), item.get("job_id"), int(item.get("attempt_number") or 0)
                if (operation not in models and operation != "publish") or not job_id:
                    continue
                with self._lock:
                    if (job_id, attempt, operation) in self._inflight:
                        continue  # _send is delivering this one itself
                if operation == "publish":
                    outcome = self._replay_publication(job_id, attempt, item.get("body") or {})
                    if outcome == "unavailable":
                        log.warning("spool replay stopped: Store unavailable")
                        return replayed
                    if outcome in ("succeeded", "failed"):
                        replayed += 1
                    continue
                try:
                    body = models[operation].model_validate(item["body"])
                except ValueError:
                    self.spool.remove(job_id, attempt, operation)
                    self._delivered(job_id, attempt)
                    continue
                try:
                    receipt = calls[operation](job_id, body)
                    replayed += 1
                    self._count_delivered(operation)
                    self._wake_if_ready(receipt)
                except StoreError as exc:
                    if isinstance(exc, StoreUnavailable) or (exc.status or 0) >= 500:
                        log.warning("spool replay stopped: Store unavailable")
                        return replayed
                    if exc.code == "claim_token_stale" and operation != "release":
                        self.stats.lost += 1
                        try:
                            self.client.attach_late_usage(job_id, attempt, LateUsageReport(claim_token=body.claim_token, usage=body.usage))
                        except StoreError:
                            pass
                    elif exc.code == "job_cancelling" and operation == "complete":
                        # Cancel arrived while the completion waited: report the attempt cancelled.
                        self.spool.write(job_id, attempt, "fail", self._cancelled_failure(job_id, attempt, body).model_dump(mode="json"))
                        self._replay_now.set()
                    log.info("spooled %s for %s not applied: %s", operation, job_id, exc.code)
                self.spool.remove(job_id, attempt, operation)
                if not (operation == "complete" and self.spool.has(job_id, attempt, "fail")):
                    self._delivered(job_id, attempt)
        return replayed

    @staticmethod
    def _cancelled_failure(job_id: str, attempt: int, completion: CompletionRequest) -> FailureRequest:
        usage = completion.usage.model_copy(update={"outcome": usage_outcome_for(JobErrorCode.CANCELLED), "error_code": JobErrorCode.CANCELLED})
        return FailureRequest(claim_token=completion.claim_token, completion_key=f"fail:{job_id}:{attempt}", error_code=JobErrorCode.CANCELLED,
                              error_detail="cancel requested", usage=usage, provenance=completion.provenance)

    def request_replay(self) -> None:
        """Ask the spool thread to replay now (a pool just reached Store again)."""
        self._replay_now.set()

    def _spool_loop(self) -> None:
        while not self._stopping.is_set():
            self._replay_now.wait(self.replay_interval)
            self._replay_now.clear()
            if self._stopping.is_set():
                return
            try:
                if self.spool.count():
                    replayed = self.recover()
                    if replayed:
                        log.info("replayed %d spooled request(s)", replayed)
            except Exception:  # pragma: no cover - defensive
                log.exception("spool replay")

    # --- heartbeats --------------------------------------------------------------------------

    def running(self) -> List[RunningJob]:
        with self._lock:
            return list(self._running.values())

    def undelivered(self) -> List[RunningJob]:
        """Attempts that finished but whose completion or failure is still in the spool."""
        with self._lock:
            return list(self._undelivered.values())

    def heartbeat_due(self, force: bool = False) -> int:
        sent = 0
        now = time.monotonic()
        for running in self.running() + self.undelivered():
            if running.lost or (not force and now < running.next_heartbeat):
                continue
            claimed = running.claimed
            progress = None
            if running.progress_fraction is not None or running.progress_note:
                progress = ProgressNote(fraction=running.progress_fraction, note=running.progress_note)
            try:
                response = self.client.heartbeat(claimed.job.id, HeartbeatRequest(claim_token=claimed.claim_token, progress=progress))
                sent += 1
                running.next_heartbeat = now + running.heartbeat_interval
                if response.cancel_requested:
                    running.cancel_requested = True
                    running.cancel.set()
            except StoreError as exc:
                if exc.code == "claim_token_stale":
                    running.lost = True
                    running.cancel.set()
                    if running.awaiting_delivery:
                        # The spool replay will be refused as stale and turned into late usage.
                        self.request_replay()
                else:
                    running.next_heartbeat = now + min(10.0, running.heartbeat_interval)
        return sent

    # --- loops -------------------------------------------------------------------------------

    def drain(self, max_rounds: int = 10_000) -> int:
        """Synchronously claim and run jobs until nothing is claimable (tests, ``drain`` CLI).
        Leases are renewed by a heartbeat thread for the duration."""
        done = 0
        beats = threading.Event()
        beat = None
        if not self._threads:
            beat = threading.Thread(target=self._heartbeat_loop, args=(beats,), name="call1-heartbeat-drain", daemon=True)
            beat.start()
        try:
            for _ in range(max_rounds):
                progressed = False
                for pool in self.pools:
                    for claimed in self.claim(pool):
                        self.execute(claimed, pool)
                        done += 1
                        progressed = True
                if not progressed:
                    return done
            return done
        finally:
            beats.set()
            if beat is not None:
                beat.join(timeout=2.0)

    def start(self) -> None:
        if self._threads:
            return
        self._stopping.clear()
        self._beats_done.clear()
        self.state = "running"
        self._executor = ThreadPoolExecutor(max_workers=max(1, sum(p.size for p in self.pools)), thread_name_prefix="call1-job")
        for pool in self.pools:
            thread = threading.Thread(target=self._pool_loop, args=(pool,), name=f"call1-claim-{pool.name}", daemon=True)
            thread.start()
            self._threads.append(thread)
        beat = threading.Thread(target=self._heartbeat_loop, args=(self._beats_done,), name="call1-heartbeat", daemon=True)
        beat.start()
        self._threads.append(beat)
        replay = threading.Thread(target=self._spool_loop, name="call1-spool", daemon=True)
        replay.start()
        self._threads.append(replay)

    def _pool_loop(self, pool: SlotPool) -> None:
        backoff = self.poll_interval
        while not self._stopping.is_set():
            if self.claims_paused is not None:
                self._stopping.wait(self.poll_interval)
                continue
            if pool.free_count() <= 0:
                pool.wait_for_free(self.poll_interval)
                continue
            seen = self._work_generation
            try:
                claimed = self.claim(pool)
                backoff = self.poll_interval
                if self.state == "store_unreachable":
                    self.state = "running"
                    self.request_replay()  # deliver what the outage left in the spool
            except StoreUnavailable:
                self.state = "store_unreachable"
                self._stopping.wait(backoff)
                backoff = min(60.0, backoff * 2)
                continue
            except Exception as exc:
                log.warning("claim on %s failed: %s", pool.name, exc)
                self._stopping.wait(self.poll_interval)
                continue
            self._dispatch(claimed, pool)
            if not claimed:
                self._wait_for_work(seen)

    def _dispatch(self, claimed: Sequence[ClaimedJob], pool: SlotPool) -> None:
        """Hand claimed jobs to the executor. A job that cannot run because stop() is shutting the
        executor down (or cancelled it before it started) is released for requeue, and its slot
        returned, so Store does not hold the lease until it expires."""
        for job in claimed:
            future: Optional[Future] = None
            with self._submit_lock:
                executor = self._executor
                if executor is not None and not self._stopping.is_set():
                    try:
                        future = executor.submit(self.execute, job, pool)
                    except RuntimeError:  # shut down between the check and the submit
                        future = None
            if future is None:
                self._abandon(job, pool)
            else:
                future.add_done_callback(lambda f, job=job: self._abandon(job, pool) if f.cancelled() else None)

    def _abandon(self, claimed: ClaimedJob, pool: SlotPool) -> None:
        try:
            self._release(claimed, "requeue", JobErrorCode.RESOURCE_UNAVAILABLE, "Process is stopping")
        except Exception:  # pragma: no cover - best effort on the way out
            log.exception("job %s: could not release the unstarted claim", claimed.job.id)
        finally:
            pool.free(1)

    def _heartbeat_loop(self, done: threading.Event) -> None:
        # In start(): runs until stop() has dealt with every running job, so leases stay alive
        # while stop() waits for them to finish.
        while not done.wait(1.0):
            try:
                self.heartbeat_due()
            except Exception:  # pragma: no cover - defensive
                log.exception("heartbeat loop")

    def stop(self, grace: float = 30.0) -> None:
        """Stop claiming, wait up to ``grace`` seconds for running jobs, then report each job still
        running as ``worker_crashed`` (transient: Store retries it) and abandon its thread."""
        if self.state == "stopped" and not self._threads:
            return
        self.state = "stopping"
        self._stopping.set()
        self.notify_work()  # idle pool loops see _stopping now
        deadline = time.monotonic() + grace
        while self.running() and time.monotonic() < deadline:
            time.sleep(0.05)
        for running in self.running():
            running.lost = True
            running.cancel.set()
            claimed = running.claimed
            try:
                handler = self.registry.get(claimed.job.job_type)
                self._fail(claimed, handler, JobErrorCode.WORKER_CRASHED, "Process stopped before the attempt finished", None,
                           time.monotonic() - running.started_monotonic, running)
            except Exception:  # pragma: no cover - best effort on the way out
                log.exception("job %s: could not report the interrupted attempt", claimed.job.id)
        self._beats_done.set()
        self._replay_now.set()  # wake the spool thread so it sees _stopping
        with self._submit_lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = []
        self.state = "stopped"

    def describe(self) -> Dict[str, Any]:
        return {
            "state": self.state, "worker_id": self.worker_id, "primary_host": self.primary_host, "handlers": self.registry.mode,
            "stats": self.stats.__dict__.copy(), "pools": [dict(p.describe(), job_types=[t.value for t in self.offered_types(p)]) for p in self.pools],
            "running": [r.describe() for r in self.running()], "spooled": self.spool.count(),
            "awaiting_delivery": [r.describe() for r in self.undelivered()], "claims_paused": self.claims_paused,
        }


__all__ = ["Worker", "SlotPool", "build_pools", "default_slot", "default_sensitivity"]
