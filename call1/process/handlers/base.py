"""The narrow interface between the worker loop and a stage handler.

A handler turns **one claimed job plus its resolved input artifacts** into **the canonical output
artifact(s) of its job type plus usage measurements**. It never talks to Store, never chooses
slots, keys or result states, and never builds follow-on jobs: the worker core owns all of that.

.. code-block:: python

    class MyAsr(Handler):
        job_type = JobType.ASR
        adapter_id = "call1.mlx.parakeet"
        adapter_version = "1"

        def run(self, job: HandlerJob) -> HandlerResult:
            audio_path = job.require("audio").path()          # downloaded, checksum-verified
            ...                                               # job.check_cancelled() between steps
            return HandlerResult(
                outputs={"transcript": Output(TranscriptContent(...))},
                usage=Usage(audio_seconds_processed=42.0, model_load_seconds=1.2),
            )

Rules the core enforces:

* ``outputs`` has exactly one entry per output role of the job type
  (``JOB_TYPE_RULES[job_type].outputs``), plus at most one per role of its ``optional_outputs``
  (contract 1.3.0: ``asr``'s ``base_transcript`` and ``vocabulary_pass``). JSON kinds are contract content models; the core dumps
  them in canonical form, picks the slot (``Output.slot`` overrides it), wraps draft-test slots,
  and uploads inline (small) or through an upload grant (large).
* Raise ``HandlerError(code)`` with a contract ``JobErrorCode`` to fail the attempt. For
  ``qa_criterion``/``qa_escalation`` a code in ``PROVIDER_FAILURE_CODES`` on the claim's final
  attempt is turned into a FLAGGED assessment (``trigger: provider_error``) by the core; pass the
  attempt's ``prompt_input`` in ``HandlerError.outputs`` when one was built.
* Raise ``ReleaseJob`` only **before inference starts** (model not installed, context too long,
  a local resource lost). It returns the claim without consuming an attempt.
* Call ``job.check_cancelled()`` between expensive steps; it raises ``JobCancelled`` when Store
  asked for cancellation (seen on heartbeat) or the claim was lost.
* A QA assessment that fails schema or quote validation is an outcome: return a FLAGGED
  ``QaAssessmentContent`` with ``trigger: invalid_answer``. The core records usage outcome
  ``validation_rejected``, sets ``escalation_requested`` and adds the escalation follow-on when the
  criterion's ``escalation_when`` lists the trigger.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

from call1.contracts.artifacts import Artifact, Sensitivity, content_model_for
from call1.contracts.catalog import FrozenSelection
from call1.contracts.common import ArtifactRef
from call1.contracts.contents import ScorecardRubricRef, SpeakerAttributionContent, TranscriptContent
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import ClaimedJob, Job, JobParameters, JobType, UpstreamOutcome
from call1.contracts.rubrics import RubricCriterion, RubricDefinition, RubricSnapshotContent
from call1.contracts.usage import BillingUnit, TokenCount

from ..transcripts import apply_attribution


# --- errors a handler may raise -------------------------------------------------------------


class HandlerError(Exception):
    """Fail this attempt with a contract ``JobErrorCode``. ``detail`` is safe text only: never
    transcript content, prompts, provider bodies, URLs with credentials or key material."""

    def __init__(self, code: JobErrorCode, detail: Optional[str] = None, *, outputs: Optional[Dict[str, "Output"]] = None,
                 usage: Optional["Usage"] = None) -> None:
        super().__init__(f"{code.value}: {detail or ''}")
        self.code = JobErrorCode(code)
        self.detail = detail
        self.outputs = dict(outputs or {})
        self.usage = usage


class ReleaseJob(Exception):
    """End the claim before inference started, without consuming an attempt. ``requeue`` takes
    ``resource_unavailable`` or ``input_unavailable``; ``reject`` takes a configuration code
    (``model_unavailable``, ``model_unqualified``, ``context_limit_exceeded``, ``credential_missing``...)."""

    def __init__(self, disposition: Literal["requeue", "reject"], code: JobErrorCode, detail: Optional[str] = None,
                 not_before_seconds: Optional[float] = None) -> None:
        super().__init__(f"{disposition} {code.value}: {detail or ''}")
        self.disposition = disposition
        self.code = JobErrorCode(code)
        self.detail = detail
        self.not_before_seconds = not_before_seconds


class JobCancelled(Exception):
    """Store asked for cancellation, or this claim is no longer the job's active claim."""


# --- inputs ---------------------------------------------------------------------------------


class InputArtifact:
    """One resolved input: the committed artifact record plus lazy, checksum-verified content."""

    def __init__(self, role: str, artifact: Artifact, fetch: Callable[[Artifact, Optional[Path]], Union[bytes, Path]], scratch: Path) -> None:
        self.role = role
        self.artifact = artifact
        self._fetch = fetch
        self._scratch = scratch
        self._bytes: Optional[bytes] = None
        self._path: Optional[Path] = None
        self._content: Optional[BaseModel] = None
        self._lock = threading.Lock()

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(artifact_id=self.artifact.id, checksum=self.artifact.checksum)

    def read_bytes(self) -> bytes:
        with self._lock:
            if self._bytes is None:
                if self._path is not None:
                    self._bytes = self._path.read_bytes()
                else:
                    self._bytes = self._fetch(self.artifact, None)  # type: ignore[assignment]
            return self._bytes  # type: ignore[return-value]

    def path(self) -> Path:
        """The content as a file in this attempt's scratch directory (audio, large JSON)."""
        with self._lock:
            if self._path is None:
                suffix = {"audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/flac": ".flac", "audio/ogg": ".ogg",
                          "audio/mp4": ".m4a", "application/json": ".json"}.get(self.artifact.content_type, ".bin")
                dest = self._scratch / f"{self.role.replace(':', '_')}-{self.artifact.id}{suffix}"
                if self._bytes is not None:
                    dest.write_bytes(self._bytes)
                    self._path = dest
                else:
                    self._path = self._fetch(self.artifact, dest)  # type: ignore[assignment]
            return self._path  # type: ignore[return-value]

    def json(self) -> Any:
        return json.loads(self.read_bytes().decode("utf-8"))

    def content(self) -> BaseModel:
        """The artifact parsed as its kind's contract content model."""
        if self._content is None:
            model = content_model_for(self.artifact.kind)
            if model is None:
                raise TypeError(f"{self.artifact.kind.value} is not JSON content")
            self._content = model.model_validate(self.json())
        return self._content


class HandlerJob:
    """What a handler sees: the claimed job, its inputs, the frozen selection and helpers."""

    def __init__(self, claimed: ClaimedJob, inputs: Dict[str, Optional[InputArtifact]], scratch_dir: Path, *,
                 catalog_entry: Any = None, cancel_event: Optional[threading.Event] = None,
                 progress: Optional[Callable[[Optional[float], Optional[str]], None]] = None, log: Optional[logging.Logger] = None,
                 call_metadata: Optional[Callable[[], Any]] = None) -> None:
        self.claimed = claimed
        self._call_metadata_loader = call_metadata
        self._call_metadata: Any = None
        self._call_metadata_loaded = False
        self.inputs = inputs
        self.scratch_dir = scratch_dir
        self.catalog_entry = catalog_entry
        self._cancel = cancel_event or threading.Event()
        self._progress = progress
        self.log = log or logging.getLogger(f"call1.process.handler.{claimed.job.job_type.value}")
        self._rubric: Optional[RubricSnapshotContent] = None

    # identity
    @property
    def job(self) -> Job:
        return self.claimed.job

    @property
    def job_type(self) -> JobType:
        return self.claimed.job.job_type

    @property
    def parameters(self) -> JobParameters:
        return self.claimed.job.parameters

    @property
    def selection(self) -> Optional[FrozenSelection]:
        return self.claimed.job.selection

    @property
    def attempt_number(self) -> int:
        return self.claimed.attempt_number

    @property
    def final_attempt(self) -> bool:
        return self.claimed.final_attempt

    @property
    def upstream(self) -> List[UpstreamOutcome]:
        return list(self.claimed.upstream)

    def call_metadata(self) -> Any:
        """The conversation's current ``CallMetadata`` (read from Store once, on first use), or None
        when this runner has no Store or the read fails. Masking uses ``agent_display_name``."""
        if not self._call_metadata_loaded:
            self._call_metadata_loaded = True
            if self._call_metadata_loader is not None:
                try:
                    self._call_metadata = self._call_metadata_loader()
                except Exception as exc:  # the name only narrows masking; never fail the job over it
                    self.log.warning("call metadata unavailable (%s)", type(exc).__name__)
        return self._call_metadata

    # inputs
    def input(self, role: str) -> Optional[InputArtifact]:
        return self.inputs.get(role)

    def require(self, role: str) -> InputArtifact:
        item = self.inputs.get(role)
        if item is None:
            raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, f"input {role} is missing")
        return item

    def inputs_with_prefix(self, prefix: str) -> Dict[str, InputArtifact]:
        return {role: item for role, item in self.inputs.items() if role.startswith(prefix) and item is not None}

    def transcript(self) -> TranscriptContent:
        """The ``transcript`` input with the ``speaker_attribution`` input (when present) applied."""
        transcript = self.require("transcript").content()
        attribution = self.input("speaker_attribution")
        return apply_attribution(transcript, attribution.content() if attribution is not None else None)  # type: ignore[arg-type]

    def attribution(self) -> Optional[SpeakerAttributionContent]:
        item = self.input("speaker_attribution")
        return item.content() if item is not None else None  # type: ignore[return-value]

    def rubric_snapshot(self) -> RubricSnapshotContent:
        if self._rubric is None:
            self._rubric = self.require("rubric").content()  # type: ignore[assignment]
        return self._rubric  # type: ignore[return-value]

    def rubric(self) -> RubricDefinition:
        return self.rubric_snapshot().definition

    def scorecard_rubric_ref(self) -> ScorecardRubricRef:
        snap = self.rubric_snapshot()
        return ScorecardRubricRef(rubric_id=snap.rubric_id, rubric_version=snap.rubric_version, draft_revision=snap.draft_revision,
                                  digest=snap.digest)

    def criterion(self) -> RubricCriterion:
        cid = self.parameters.criterion_id
        found = next((c for c in self.rubric().criteria if c.criterion_id == cid), None)
        if found is None:
            raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, "the job's criterion is not in its rubric snapshot")
        return found

    # control
    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled()

    def progress(self, fraction: Optional[float] = None, note: Optional[str] = None) -> None:
        """Reported with the next heartbeat (safe text only)."""
        if self._progress is not None:
            self._progress(fraction, note)


# --- results --------------------------------------------------------------------------------


@dataclass
class Output:
    """One output artifact. ``content`` is a contract content model for JSON kinds, or bytes / a
    file path for opaque kinds (audio). ``sensitivity`` and ``slot`` default per kind and job."""

    content: Union[BaseModel, bytes, Path]
    content_type: Optional[str] = None
    sensitivity: Optional[Sensitivity] = None
    slot: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Usage:
    """Measurements of this attempt. Token counts are ``None`` when unknown (recorded as
    ``unavailable``, never estimated silently). ``inference_seconds`` defaults to the wall time of
    ``run``."""

    tokens_input: Optional[TokenCount] = None
    tokens_output: Optional[TokenCount] = None
    model_load_seconds: Optional[float] = None
    inference_seconds: Optional[float] = None
    audio_seconds_processed: Optional[float] = None
    peak_memory_bytes: Optional[int] = None
    provider_reported_model_id: Optional[str] = None
    billing_units: List[BillingUnit] = field(default_factory=list)


@dataclass
class HandlerResult:
    outputs: Dict[str, Output]
    usage: Usage = field(default_factory=Usage)
    model_revision: Optional[str] = None
    """The exact revision used, when it differs from the frozen selection's."""
    partial_reason: Optional[str] = None
    """Publishing job types only: publish the result as ``partial`` with this reason."""


class Handler:
    """Base class. Subclasses set ``job_type``, ``adapter_id`` and ``adapter_version`` and
    implement ``run``. ``ready`` may refuse an entry this handler cannot serve here (the worker
    then releases the claim with ``reject``)."""

    job_type: ClassVar[JobType]
    adapter_id: ClassVar[str] = "call1.handler"
    adapter_version: ClassVar[str] = "1"

    def run(self, job: HandlerJob) -> HandlerResult:  # pragma: no cover - interface
        raise NotImplementedError

    def ready(self, job: HandlerJob) -> None:
        """Called before inputs are fetched. Raise ``ReleaseJob`` to refuse the claim."""
        return None
