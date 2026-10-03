"""The processing job queue: graphs, dependencies, claims, leases, heartbeats, completion receipts,
retries, releases, cancellation, reanalysis requests and group progress.

This is Process's queue. It is not the human review queue (``reviews.py``); the two share no type
and no endpoint, and tests enforce that. Store's records are authoritative for every transition.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from .artifacts import Artifact, ArtifactKind
from .catalog import CatalogEntryRef, FrozenSelection, ModelPurpose
from .common import ArtifactRef, ChangeCursor, ContractModel, IdempotencyKey, JsonScalar, PageQuery, ResourceId, SafeText, Sha256Digest, ShortText, Timestamp
from .contents import ContactSignalPass, QaScorecardContent, ResultKind, ResultState, TurnWindow
from .custody import Pro1AttemptFailureEvidence, Pro1AttestationRecord, RouteClass, RouteRecord
from .errors import JOB_ERROR_CLASSES, PRO1_PROVIDER_FAILURE_CODES, PROVIDER_FAILURE_CODES, JobErrorClass, JobErrorCode
from .rubrics import DraftRubricRef, RubricVersionRef
from .signals import SIGNAL_TAXONOMY_INPUT_ROLE, SignalSpanKey
from .vocabulary import AsrVocabularyParameters
from .usage import UsageOutcome, UsageRecordInput, usage_outcome_for


# --- Status and transitions ---------------------------------------------------------------


class JobStatus(str, Enum):
    BLOCKED = "BLOCKED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    WAITING_PROVIDER = "WAITING_PROVIDER"
    """Reserved for a future asynchronous provider route (FutureProviders.md). v1 never enters it."""


TERMINAL_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})
RESERVED_STATUSES = frozenset({JobStatus.WAITING_PROVIDER})


class JobTrigger(str, Enum):
    DEPENDENCIES_SATISFIED = "dependencies_satisfied"
    CLAIM = "claim"
    ADMISSION_REJECTED = "admission_rejected"
    COMPLETE = "complete"
    FAIL_RETRYABLE = "fail_retryable"
    FAIL_TERMINAL = "fail_terminal"
    RELEASE_REQUEUE = "release_requeue"
    RELEASE_REJECT = "release_reject"
    LEASE_EXPIRED_RETRYABLE = "lease_expired_retryable"
    LEASE_EXHAUSTED = "lease_exhausted"
    CANCEL = "cancel"
    CANCEL_ACKNOWLEDGED = "cancel_acknowledged"
    LEASE_EXPIRED_AFTER_CANCEL = "lease_expired_after_cancel"
    MANUAL_RETRY = "manual_retry"
    PROVIDER_ASYNC_SUBMITTED = "provider_async_submitted"
    PROVIDER_ASYNC_RESULT = "provider_async_result"


class TransitionActor(str, Enum):
    STORE = "store"
    PROCESS = "process"
    OPERATOR = "operator"


class JobTransition(ContractModel):
    from_status: JobStatus
    to_status: JobStatus
    trigger: JobTrigger
    actor: TransitionActor
    consumes_attempt: bool = Field(default=False, description="True only for the claim; a release refunds it.")
    reserved: bool = Field(default=False, description="True for transitions v1 never performs.")
    note: SafeText


JOB_TRANSITIONS: List[JobTransition] = [
    JobTransition(from_status=JobStatus.BLOCKED, to_status=JobStatus.QUEUED, trigger=JobTrigger.DEPENDENCIES_SATISFIED, actor=TransitionActor.STORE, note="Every success edge's upstream SUCCEEDED, every after edge's upstream is terminal, and every upstream-output input is resolved and pinned; done in the upstream completion (or failure) transaction."),
    JobTransition(from_status=JobStatus.BLOCKED, to_status=JobStatus.CANCELLED, trigger=JobTrigger.CANCEL, actor=TransitionActor.OPERATOR, note="Cancel, including cascade from a cancelled upstream job."),
    JobTransition(from_status=JobStatus.QUEUED, to_status=JobStatus.RUNNING, trigger=JobTrigger.CLAIM, actor=TransitionActor.PROCESS, consumes_attempt=True, note="Atomic claim: one lease, one claim token, attempt_count + 1, claim_count + 1."),
    JobTransition(from_status=JobStatus.QUEUED, to_status=JobStatus.FAILED, trigger=JobTrigger.ADMISSION_REJECTED, actor=TransitionActor.STORE, note="Store-side admission, evaluated for every QUEUED job on each claim call (whoever asks) and whenever admin state or the Pro1 connection changes: its frozen route's opt-in is disabled or its connection no longer in admin state (route_disabled), or it is call1_confidential and the Pro1 connection is blocked (pro1_connection_blocked). No attempt consumed; manual retry after the admin fix. A pending_verification connection only holds the job (waiting_reason pro1_pending_verification)."),
    JobTransition(from_status=JobStatus.QUEUED, to_status=JobStatus.CANCELLED, trigger=JobTrigger.CANCEL, actor=TransitionActor.OPERATOR, note="Cancel before any claim."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.SUCCEEDED, trigger=JobTrigger.COMPLETE, actor=TransitionActor.PROCESS, note="Idempotent completion with the active claim token and no cancel requested; commits outputs, usage, receipt, follow-on jobs and dependent release together (COMPLETION_TRANSACTION_STEPS)."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.QUEUED, trigger=JobTrigger.FAIL_RETRYABLE, actor=TransitionActor.PROCESS, note="Transient error class and attempts remain; next_run_at set by backoff."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.FAILED, trigger=JobTrigger.FAIL_TERMINAL, actor=TransitionActor.PROCESS, note="Configuration, definitive or terminal error class, or attempts exhausted. A definitive Pro1 code also blocks the Pro1 connection in the same transaction."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.QUEUED, trigger=JobTrigger.RELEASE_REQUEUE, actor=TransitionActor.PROCESS, note="Released before inference started (slot lost, input fetch failed, temporary pressure): attempt refunded, no usage row, next_run_at from not_before."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.FAILED, trigger=JobTrigger.RELEASE_REJECT, actor=TransitionActor.PROCESS, note="Refused before inference started for a configuration reason Process detects (credential missing, key anchor unavailable, model not installed or unqualified, context limit): attempt refunded, no usage row."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.QUEUED, trigger=JobTrigger.LEASE_EXPIRED_RETRYABLE, actor=TransitionActor.STORE, note="Lease expired past LEASE_GRACE and attempts remain; Store synthesizes the attempt's abandoned usage row; the stale claim token is rejected thereafter."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.FAILED, trigger=JobTrigger.LEASE_EXHAUSTED, actor=TransitionActor.STORE, note="Lease expired and no attempts remain; error_code lease_expired; abandoned usage row synthesized."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.CANCELLED, trigger=JobTrigger.CANCEL_ACKNOWLEDGED, actor=TransitionActor.PROCESS, note="Worker saw cancel_requested (heartbeat, or job_cancelling on completion) and reported failure code cancelled."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.CANCELLED, trigger=JobTrigger.LEASE_EXPIRED_AFTER_CANCEL, actor=TransitionActor.STORE, note="Cancel was requested and the lease then expired; abandoned usage row synthesized."),
    JobTransition(from_status=JobStatus.FAILED, to_status=JobStatus.QUEUED, trigger=JobTrigger.MANUAL_RETRY, actor=TransitionActor.OPERATOR, note="Explicit retry of this job only; grants RETRY_ATTEMPT_GRANT attempts; dependencies already satisfied."),
    JobTransition(from_status=JobStatus.FAILED, to_status=JobStatus.BLOCKED, trigger=JobTrigger.MANUAL_RETRY, actor=TransitionActor.OPERATOR, note="Explicit retry while an upstream job is not yet satisfied."),
    JobTransition(from_status=JobStatus.RUNNING, to_status=JobStatus.WAITING_PROVIDER, trigger=JobTrigger.PROVIDER_ASYNC_SUBMITTED, actor=TransitionActor.PROCESS, reserved=True, note="Reserved for an asynchronous provider route; not performed in v1."),
    JobTransition(from_status=JobStatus.WAITING_PROVIDER, to_status=JobStatus.RUNNING, trigger=JobTrigger.PROVIDER_ASYNC_RESULT, actor=TransitionActor.PROCESS, reserved=True, note="Reserved; not performed in v1."),
    JobTransition(from_status=JobStatus.WAITING_PROVIDER, to_status=JobStatus.CANCELLED, trigger=JobTrigger.CANCEL, actor=TransitionActor.OPERATOR, reserved=True, note="Reserved; not performed in v1."),
]
"""The complete job state machine. Any transition not listed is ``invalid_transition``. CANCELLED
is final: work cancelled by mistake is rerun as a new graph (reason retry_group)."""


# --- Job types ----------------------------------------------------------------------------


class JobType(str, Enum):
    VALIDATION_VAD = "validation_vad"
    ASR = "asr"
    SPEAKER_ATTRIBUTION = "speaker_attribution"
    ACOUSTIC_TONE = "acoustic_tone"
    TEXT_SENTIMENT = "text_sentiment"
    EMBEDDINGS = "embeddings"
    ENRICHMENT = "enrichment"
    QA_DETERMINISTIC = "qa_deterministic"
    QA_CRITERION = "qa_criterion"
    QA_ESCALATION = "qa_escalation"
    QA_SCORECARD = "qa_scorecard"
    SUMMARY_SEGMENT = "summary_segment"
    SUMMARY_SYNTHESIS = "summary_synthesis"
    SUMMARY_ASSEMBLY = "summary_assembly"
    CONTACT_SIGNALS_LIFECYCLE = "contact_signals_lifecycle"
    CONTACT_SIGNALS_RESOLUTION = "contact_signals_resolution"
    CONTACT_SIGNALS_MERGE = "contact_signals_merge"
    CONTACT_SIGNALS_CATEGORIZE = "contact_signals_categorize"
    """Added in 1.3.0: Contact Signals v2 stage 1, one category pick per ~7 s segment (primary host)."""
    CONTACT_SIGNALS_SUBCATEGORIZE = "contact_signals_subcategorize"
    """Added in 1.3.0: stage 2, one subcategory pick per isolated span (primary host)."""
    CONTACT_SIGNALS_EXTRACT = "contact_signals_extract"
    """Added in 1.3.0: stage 3, admin-defined fields on spans whose node has fields or narrow_quote (LLM route)."""


SIGNAL_V2_JOB_TYPES = frozenset({JobType.CONTACT_SIGNALS_CATEGORIZE, JobType.CONTACT_SIGNALS_SUBCATEGORIZE, JobType.CONTACT_SIGNALS_EXTRACT})
"""Added in 1.3.0. The v2 stage job types. Each pins a signal_taxonomy_snapshot (``needs_signal_taxonomy``);
the existing ``contact_signals_merge`` publishes their result into the same group. v1 and v2 never
mix in one graph."""


class ExecutionClass(str, Enum):
    PRIMARY_HOST = "primary_host"
    """ML and deterministic stages: run only on the designated primary Process host."""
    LLM_ROUTE = "llm_route"
    """One inference request over the job's frozen route."""
    ASSEMBLY = "assembly"
    """Deterministic assembly of committed artifacts; any qualified Process worker."""


_SUCCESS_ONLY = [UsageOutcome.SUCCEEDED]
_ASSESSMENT = [UsageOutcome.SUCCEEDED, UsageOutcome.VALIDATION_REJECTED, UsageOutcome.FAILED]


class JobTypeRule(ContractModel):
    job_type: JobType
    execution_class: ExecutionClass
    purpose: Optional[ModelPurpose]
    outputs: Dict[str, ArtifactKind] = Field(description="Output role to artifact kind. A completion links exactly one committed artifact per role.")
    group: Optional[ResultKind] = Field(description="The result group whose progress this job counts toward; null for supporting work (embeddings), counted only in pending_work.")
    publishes: Optional[ResultKind] = Field(default=None, description="Set on the one job type per group whose completion publishes the group's result version.")
    completion_outcomes: List[UsageOutcome] = Field(default_factory=lambda: list(_SUCCESS_ONLY), description="Usage outcomes a completion of this type may carry. QA assessments complete FLAGGED on an invalid answer or a final-attempt provider failure.")
    needs_rubric: bool = False
    needs_signal_taxonomy: bool = Field(default=False, description="Added in 1.3.0. The job freezes parameters.signals and pins one signal_taxonomy_snapshot under input role 'taxonomy'; Store checks its digest against parameters.signals.taxonomy_digest at graph creation (graph_invalid).")
    optional_outputs: Dict[str, ArtifactKind] = Field(default_factory=dict, description="Added in 1.3.0 (decision 33). Output roles a completion may link at most once each, besides every role in outputs. Never a publishing output and never an upstream input of another job (graph_invalid).")

    @model_validator(mode="after")
    def _optional_roles(self):
        if set(self.optional_outputs) & set(self.outputs):
            raise ValueError("an output role is required or optional, not both")
        return self


def _rule(job_type, execution_class, purpose, outputs, group, publishes=None, completion_outcomes=_SUCCESS_ONLY, needs_rubric=False, needs_signal_taxonomy=False, optional_outputs=None):
    return JobTypeRule(job_type=job_type, execution_class=execution_class, purpose=purpose, outputs=outputs, group=group, publishes=publishes, completion_outcomes=list(completion_outcomes), needs_rubric=needs_rubric, needs_signal_taxonomy=needs_signal_taxonomy, optional_outputs=dict(optional_outputs or {}))


ASR_BASE_TRANSCRIPT_ROLE = "base_transcript"
"""Added in 1.3.0 (decision 33). Optional ``asr`` output: the base engine's own transcript, linked
whenever the job ran with ``parameters.asr_vocabulary``."""

ASR_VOCABULARY_PASS_ROLE = "vocabulary_pass"
"""Added in 1.3.0 (decision 33). Optional ``asr`` output: the raw vocabulary pass, linked only when
it produced output (absent on a ``base_only`` transcript)."""

_P = ExecutionClass.PRIMARY_HOST
_L = ExecutionClass.LLM_ROUTE
_A = ExecutionClass.ASSEMBLY
_K = ArtifactKind
_G = ResultKind

JOB_TYPE_RULES: Dict[JobType, JobTypeRule] = {r.job_type: r for r in [
    _rule(JobType.VALIDATION_VAD, _P, None, {"validation_report": _K.VALIDATION_REPORT, "vad_metrics": _K.VAD_METRICS}, _G.TRANSCRIPT),
    _rule(JobType.ASR, _P, ModelPurpose.ASR, {"transcript": _K.TRANSCRIPT}, _G.TRANSCRIPT, publishes=_G.TRANSCRIPT,
          optional_outputs={ASR_BASE_TRANSCRIPT_ROLE: _K.ASR_BASE_TRANSCRIPT, ASR_VOCABULARY_PASS_ROLE: _K.ASR_VOCABULARY_PASS}),
    _rule(JobType.SPEAKER_ATTRIBUTION, _P, ModelPurpose.SPEAKER_DIARIZATION, {"speaker_attribution": _K.SPEAKER_ATTRIBUTION}, _G.TRANSCRIPT),
    _rule(JobType.ENRICHMENT, _P, None, {"enrichment": _K.ENRICHMENT, "pii_findings": _K.PII_FINDINGS}, _G.TRANSCRIPT),
    _rule(JobType.ACOUSTIC_TONE, _P, ModelPurpose.ACOUSTIC_TONE, {"tone_blocks": _K.TONE_BLOCKS}, _G.TONE, publishes=_G.TONE),
    _rule(JobType.TEXT_SENTIMENT, _P, ModelPurpose.TEXT_SENTIMENT, {"text_sentiment": _K.TEXT_SENTIMENT}, _G.TEXT_SENTIMENT, publishes=_G.TEXT_SENTIMENT),
    _rule(JobType.EMBEDDINGS, _P, ModelPurpose.EMBEDDINGS, {"embeddings": _K.EMBEDDINGS}, None),
    _rule(JobType.QA_DETERMINISTIC, _P, None, {"verdicts": _K.QA_VERDICT}, _G.QA, needs_rubric=True),
    _rule(JobType.QA_CRITERION, _L, ModelPurpose.SEMANTIC_QA, {"assessment": _K.QA_ASSESSMENT, "prompt_input": _K.PROMPT_INPUT}, _G.QA, completion_outcomes=_ASSESSMENT, needs_rubric=True),
    _rule(JobType.QA_ESCALATION, _L, ModelPurpose.SEMANTIC_QA, {"assessment": _K.QA_ASSESSMENT, "prompt_input": _K.PROMPT_INPUT}, _G.QA, completion_outcomes=_ASSESSMENT, needs_rubric=True),
    _rule(JobType.QA_SCORECARD, _A, None, {"scorecard": _K.QA_SCORECARD}, _G.QA, publishes=_G.QA, needs_rubric=True),
    _rule(JobType.SUMMARY_SEGMENT, _L, ModelPurpose.SUMMARY, {"segment": _K.SUMMARY_SEGMENT, "prompt_input": _K.PROMPT_INPUT}, _G.SUMMARY),
    _rule(JobType.SUMMARY_SYNTHESIS, _L, ModelPurpose.SUMMARY, {"synthesis": _K.SUMMARY_SYNTHESIS, "prompt_input": _K.PROMPT_INPUT}, _G.SUMMARY),
    _rule(JobType.SUMMARY_ASSEMBLY, _A, None, {"summary": _K.SUMMARY}, _G.SUMMARY, publishes=_G.SUMMARY),
    _rule(JobType.CONTACT_SIGNALS_LIFECYCLE, _L, ModelPurpose.CONTACT_SIGNALS, {"pass": _K.CONTACT_SIGNALS_PASS, "prompt_input": _K.PROMPT_INPUT}, _G.CONTACT_SIGNALS),
    _rule(JobType.CONTACT_SIGNALS_RESOLUTION, _L, ModelPurpose.CONTACT_SIGNALS, {"pass": _K.CONTACT_SIGNALS_PASS, "prompt_input": _K.PROMPT_INPUT}, _G.CONTACT_SIGNALS),
    _rule(JobType.CONTACT_SIGNALS_MERGE, _A, None, {"contact_signals": _K.CONTACT_SIGNALS}, _G.CONTACT_SIGNALS, publishes=_G.CONTACT_SIGNALS),
    _rule(JobType.CONTACT_SIGNALS_CATEGORIZE, _P, ModelPurpose.SIGNAL_CATEGORY, {"categories": _K.SIGNAL_CATEGORIES}, _G.CONTACT_SIGNALS, needs_signal_taxonomy=True),
    _rule(JobType.CONTACT_SIGNALS_SUBCATEGORIZE, _P, ModelPurpose.SIGNAL_SUBCATEGORY, {"subcategories": _K.SIGNAL_SUBCATEGORIES}, _G.CONTACT_SIGNALS, needs_signal_taxonomy=True),
    _rule(JobType.CONTACT_SIGNALS_EXTRACT, _L, ModelPurpose.SIGNAL_EXTRACTION, {"extraction": _K.SIGNAL_EXTRACTION, "prompt_input": _K.PROMPT_INPUT}, _G.CONTACT_SIGNALS, needs_signal_taxonomy=True),
]}
"""Job types as data. Every job counts toward its ``group`` (or pending_work only); exactly one type
per group ``publishes``. Merge and assembly jobs that may publish a partial result use ``after``
edges on the passes they combine (see README, "Dependencies")."""


# --- Job definition -----------------------------------------------------------------------

_REF = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class EdgeKind(str, Enum):
    REQUIRES = "requires"
    """The upstream must SUCCEED. While it is FAILED or CANCELLED the dependent stays BLOCKED (dead-blocked)."""
    AFTER = "after"
    """The upstream must be terminal (SUCCEEDED, FAILED or CANCELLED). Used by merges that can publish partial."""


class UpstreamOutput(ContractModel):
    """An input that is an output of another job: in this request (``ref``) or already existing
    (``job_id``), by its output role. Store resolves it to the committed, linked output artifact and
    pins its checksum when it releases the dependent."""

    ref: Optional[str] = Field(default=None, pattern=_REF.pattern)
    job_id: Optional[ResourceId] = None
    output_role: ShortText

    @model_validator(mode="after")
    def _one(self):
        if (self.ref is None) == (self.job_id is None):
            raise ValueError("an upstream output names exactly one of ref or job_id")
        return self


class JobInput(ContractModel):
    """One input role of a job: a committed artifact pinned by checksum, or an upstream job's output."""

    role: ShortText = Field(description="What the input is to this job: audio, transcript, rubric, segment, assessment ...")
    artifact: Optional[ArtifactRef] = Field(default=None, description="An existing committed artifact, pinned by checksum; Store rejects a mismatch at graph creation.")
    upstream: Optional[UpstreamOutput] = None
    optional: bool = Field(default=False, description="Only on an after edge: resolved when the upstream SUCCEEDED, absent otherwise.")

    @model_validator(mode="after")
    def _one_source(self):
        if (self.artifact is None) == (self.upstream is None):
            raise ValueError("an input is exactly one of a pinned artifact or an upstream output")
        if self.optional and self.upstream is None:
            raise ValueError("only an upstream output can be optional")
        return self


class SizeClass(str, Enum):
    XS = "xs"
    S = "s"
    M = "m"
    L = "l"
    XL = "xl"


class MemorySlot(str, Enum):
    LOCAL_MEMORY = "local_memory"
    """The one shared unified-memory inference slot (included MLX and appliance Ollama)."""
    CPU = "cpu"
    IO = "io"
    OUTBOUND = "outbound"
    """One pool per customer-LAN host, Pro1 connection or BYOK connection."""


class ResourceEstimate(ContractModel):
    """Admission estimate from frozen inputs and the selected model; retained per attempt."""

    size_class: SizeClass
    memory_slot: MemorySlot
    outbound_connection_ref: Optional[ResourceId] = Field(default=None, description="Which outbound pool (the admin-state connection ref), when memory_slot is outbound.")
    estimated_input_tokens: Optional[int] = Field(default=None, ge=0)
    context_limit_tokens: Optional[int] = Field(default=None, ge=0)
    output_token_limit: Optional[int] = Field(default=None, ge=0)
    audio_seconds: Optional[float] = Field(default=None, ge=0)
    estimated_runtime_seconds: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _outbound_ref(self):
        if (self.memory_slot is MemorySlot.OUTBOUND) != (self.outbound_connection_ref is not None):
            raise ValueError("outbound slots name their connection; other slots do not")
        return self


class SegmentSpec(ContractModel):
    index: int = Field(ge=0)
    window: TurnWindow


class SpeakerCorrection(ContractModel):
    """Relabel one turn (or its whole speaker cluster); mirrors the pre-split per-turn correction.
    Transcription is not repeated; downstream analysis is."""

    turn_id: int = Field(ge=0)
    speaker: ShortText = Field(pattern=r"^(AGENT|CALLER|UNKNOWN)$")
    apply_to_cluster: bool = False
    notes: Optional[SafeText] = None


class SignalJobParameters(ContractModel):
    """Added in 1.3.0. What a Contact Signals v2 job (and a v2 merge) freezes. ``span_keys`` are IDs,
    never text."""

    taxonomy_digest: Sha256Digest = Field(description="The pinned signal_taxonomy_snapshot's taxonomy digest; Store checks it at graph creation.")
    stage1_mode: Literal["run", "rederive"] = Field(default="run", description="contact_signals_categorize only. rederive rebuilds spans from the previous categories artifact's stored scores (a threshold-only edit): no model, so the job takes no selection.")
    span_keys: Optional[List[SignalSpanKey]] = Field(default=None, max_length=512, description="subcategorize and extract only: the spans to (re)run; null means every span.")
    fallback_entry_id: Optional[ShortText] = Field(default=None, description="contact_signals_extract only: the declared in-job fallback entry, frozen by the planner (section 5.7).")
    preview_id: Optional[ResourceId] = Field(default=None, description="Set in a contact_signals_preview graph.")

    @model_validator(mode="after")
    def _keys(self):
        if self.span_keys is not None and len(self.span_keys) != len(set(self.span_keys)):
            raise ValueError("span_keys lists each span once")
        return self


class JobParameters(ContractModel):
    """Non-secret settings needed to reproduce the job. Never prompts, transcript text or credentials.
    The ASR vocabulary (1.3.0) is the one customer-authored text here: business terms that
    ``vocabulary.vocabulary_term_problem`` accepts, so never a digit."""

    rubric: Optional[RubricVersionRef] = Field(default=None, description="QA types: the published version scored with.")
    draft_rubric: Optional[DraftRubricRef] = Field(default=None, description="QA types in a draft-test graph only: the draft snapshot scored with.")
    criterion_id: Optional[ShortText] = None
    escalation_trigger: Optional[ShortText] = Field(default=None, description="For qa_escalation: the configured trigger that fired.")
    segment: Optional[SegmentSpec] = None
    pass_kind: Optional[ContactSignalPass] = None
    window: Optional[TurnWindow] = Field(default=None, description="Bounded transcript window for a split contact-signal pass. Never set on a v2 job: v2 segments are rows inside one job, not graph windows.")
    speaker_correction: Optional[SpeakerCorrection] = Field(default=None, description="speaker_attribution only: the reviewer's correction this job applies, frozen from the speaker_correction reanalysis request. The job then writes a speaker_attribution with method reviewer_correction, runs no model and takes no selection.")
    signals: Optional[SignalJobParameters] = Field(default=None, description="Added in 1.3.0. Contact Signals v2 jobs (required) and a v2 contact_signals_merge (optional).")
    asr_vocabulary: Optional[AsrVocabularyParameters] = Field(default=None, description="Added in 1.3.0 (decision 33). asr only: run dual transcription with this vocabulary (vocabulary.AsrVocabularyParameters). Absent means the base engine alone.")
    extra: Dict[str, JsonScalar] = Field(default_factory=dict, description="Bounded scalar settings (at most 32 keys).")

    @model_validator(mode="after")
    def _bounded(self):
        if len(self.extra) > 32 or any(len(k) > 64 for k in self.extra):
            raise ValueError("extra is a small set of scalar settings")
        if self.rubric is not None and self.draft_rubric is not None:
            raise ValueError("a job scores with a published version or a draft, not both")
        return self


class JobDefinition(ContractModel):
    """One job inside a graph request. Store assigns the job ID."""

    ref: str = Field(pattern=_REF.pattern, description="Name local to this graph request, used for dependency edges.")
    job_type: JobType
    idempotency_key: IdempotencyKey = Field(description="Stable per logical job definition; a replay returns the existing job.")
    priority: int = Field(default=0, description="Higher values are claimed first.")
    max_attempts: Optional[int] = Field(default=None, ge=1, description="Defaults to DEFAULT_MAX_ATTEMPTS.")
    inputs: List[JobInput] = Field(default_factory=list)
    selection: Optional[FrozenSelection] = Field(default=None, description="Required for model-backed job types; absent for code stages.")
    resource_estimate: ResourceEstimate
    parameters: JobParameters = Field(default_factory=JobParameters)
    requires_refs: List[str] = Field(default_factory=list, description="Refs in this request that must SUCCEED first.")
    requires_job_ids: List[ResourceId] = Field(default_factory=list, description="Existing jobs of the same conversation that must SUCCEED first.")
    after_refs: List[str] = Field(default_factory=list, description="Refs in this request that must be terminal first (any outcome).")
    after_job_ids: List[ResourceId] = Field(default_factory=list, description="Existing jobs of the same conversation that must be terminal first.")

    @property
    def execution_class(self) -> ExecutionClass:
        return JOB_TYPE_RULES[self.job_type].execution_class

    @property
    def model_backed(self) -> bool:
        """True when the job runs a model and so freezes a selection. A speaker_attribution job
        that applies a reviewer's correction is a code stage, and so (1.3.0) is a
        contact_signals_categorize job in ``rederive`` mode."""
        if JOB_TYPE_RULES[self.job_type].purpose is None or self.parameters.speaker_correction is not None:
            return False
        signals = self.parameters.signals
        return not (self.job_type is JobType.CONTACT_SIGNALS_CATEGORIZE and signals is not None and signals.stage1_mode == "rederive")

    def _check_signal_parameters(self, rule: "JobTypeRule") -> None:
        signals = self.parameters.signals
        jt = self.job_type
        if signals is not None and jt not in SIGNAL_V2_JOB_TYPES and jt is not JobType.CONTACT_SIGNALS_MERGE:
            raise ValueError("only Contact Signals v2 jobs and the merge carry parameters.signals")
        if rule.needs_signal_taxonomy:
            if signals is None:
                raise ValueError(f"{jt.value} freezes parameters.signals with its taxonomy digest")
            if self.parameters.window is not None or self.parameters.pass_kind is not None:
                raise ValueError("a v2 job is never a windowed v1 pass")
            pinned = [i for i in self.inputs if i.role == SIGNAL_TAXONOMY_INPUT_ROLE]
            if len(pinned) != 1 or pinned[0].artifact is None:
                raise ValueError(f"{jt.value} pins one signal_taxonomy_snapshot artifact under input role '{SIGNAL_TAXONOMY_INPUT_ROLE}'")
        if signals is None:
            return
        if signals.stage1_mode == "rederive" and jt is not JobType.CONTACT_SIGNALS_CATEGORIZE:
            raise ValueError("only contact_signals_categorize re-derives")
        if signals.span_keys is not None and jt not in (JobType.CONTACT_SIGNALS_SUBCATEGORIZE, JobType.CONTACT_SIGNALS_EXTRACT):
            raise ValueError("span_keys narrow subcategorize and extract only")
        if signals.fallback_entry_id is not None and jt is not JobType.CONTACT_SIGNALS_EXTRACT:
            raise ValueError("only contact_signals_extract declares an in-job fallback entry")

    @model_validator(mode="after")
    def _selection_matches_type(self):
        rule = JOB_TYPE_RULES[self.job_type]
        if self.parameters.speaker_correction is not None and self.job_type is not JobType.SPEAKER_ATTRIBUTION:
            raise ValueError("only a speaker_attribution job applies a speaker correction")
        if self.parameters.asr_vocabulary is not None and self.job_type is not JobType.ASR:
            raise ValueError("only an asr job runs dual transcription (parameters.asr_vocabulary)")
        if not self.model_backed and self.selection is not None:
            raise ValueError(f"{self.job_type.value} is a code stage here and takes no model selection")
        if self.model_backed:
            if self.selection is None:
                raise ValueError(f"{self.job_type.value} freezes a model selection")
            if self.selection.purpose is not rule.purpose:
                raise ValueError(f"{self.job_type.value} selects a {rule.purpose.value} entry")
            if rule.execution_class is ExecutionClass.PRIMARY_HOST and self.selection.route.route_class is not RouteClass.APPLIANCE:
                raise ValueError("ML stages run on the primary Process host; audio never leaves the site")
        if self.ref in self.requires_refs or self.ref in self.after_refs:
            raise ValueError("a job cannot depend on itself")
        if set(self.requires_refs) & set(self.after_refs) or set(self.requires_job_ids) & set(self.after_job_ids):
            raise ValueError("an upstream is on one edge kind only")
        if self.job_type in (JobType.QA_CRITERION, JobType.QA_ESCALATION) and self.parameters.criterion_id is None:
            raise ValueError("qa_criterion and qa_escalation name their criterion")
        if rule.needs_rubric and (self.parameters.rubric is None) == (self.parameters.draft_rubric is None):
            raise ValueError(f"{self.job_type.value} scores with exactly one published rubric version or draft snapshot")
        self._check_signal_parameters(rule)
        roles = [i.role for i in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("input roles are unique within a job")
        for item in self.inputs:
            if item.upstream is None:
                continue
            up = item.upstream
            on_requires = (up.ref in self.requires_refs) if up.ref else (up.job_id in self.requires_job_ids)
            on_after = (up.ref in self.after_refs) if up.ref else (up.job_id in self.after_job_ids)
            if not (on_requires or on_after):
                raise ValueError(f"input {item.role} reads an upstream this job does not depend on")
            if on_after != item.optional:
                raise ValueError(f"input {item.role}: inputs from after edges are optional, inputs from success edges are not")
        return self


class GraphReason(str, Enum):
    INGEST = "ingest"
    REANALYSIS = "reanalysis"
    RETRY_GROUP = "retry_group"
    MIGRATION = "migration"


def _check_graph(jobs: List[JobDefinition]) -> None:
    by_ref = {job.ref: job for job in jobs}
    if len(by_ref) != len(jobs):
        raise ValueError("job refs must be unique within a graph request")
    edges = {job.ref: list(job.requires_refs) + list(job.after_refs) for job in jobs}
    for ref, deps in edges.items():
        unknown = [d for d in deps if d not in by_ref]
        if unknown:
            raise ValueError(f"{ref} depends on unknown ref(s) {unknown}")
    for job in jobs:
        for item in job.inputs:
            if item.upstream is not None and item.upstream.ref is not None:
                upstream = by_ref[item.upstream.ref]
                if item.upstream.output_role not in JOB_TYPE_RULES[upstream.job_type].outputs:
                    raise ValueError(f"{job.ref}.{item.role}: {upstream.job_type.value} has no output role {item.upstream.output_role}")
    state: Dict[str, int] = {}

    def visit(node: str) -> None:
        if state.get(node) == 1:
            raise ValueError("dependency cycle detected")
        if state.get(node) == 2:
            return
        state[node] = 1
        for dep in edges[node]:
            visit(dep)
        state[node] = 2

    for ref in edges:
        visit(ref)


class JobGraphRequest(ContractModel):
    """Idempotent graph creation. All jobs belong to the conversation in the path; Store rejects
    ``requires_job_ids``/``after_job_ids`` from any other conversation (no cross-conversation edges)
    and an upstream ``job_id`` output role the upstream's type does not declare (graph_invalid).

    A reanalysis graph carries the reanalysis claim token: Store creates the graph and fulfils the
    request in one transaction and refuses a stale claim (claim_token_stale), so one request yields
    at most one graph. The idempotency key is looked up first, so a replay returns the original
    graph even after the request is fulfilled.

    Every QA job (``JobTypeRule.needs_rubric``) pins its ``rubric_snapshot`` under input role
    ``rubrics.RUBRIC_INPUT_ROLE``; Store checks that the snapshot names the job's rubric, version or
    draft revision, and digest (graph_invalid). A graph that fulfils a ``qa_draft_test`` request is a
    draft-test graph: its jobs write in ``draft:<request_id>:`` slots and never count toward the
    call's result groups (``calls.result_state_inputs``)."""

    idempotency_key: IdempotencyKey
    reason: GraphReason
    reanalysis_request_id: Optional[ResourceId] = None
    reanalysis_claim_token: Optional[str] = None
    jobs: List[JobDefinition] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def _valid_graph(self):
        _check_graph(self.jobs)
        reanalysis = self.reason is GraphReason.REANALYSIS
        if reanalysis != (self.reanalysis_request_id is not None) or reanalysis != (self.reanalysis_claim_token is not None):
            raise ValueError("a reanalysis graph, and only one, names its request and the request's claim token")
        return self


class JobEdge(ContractModel):
    job_id: ResourceId
    upstream_job_id: ResourceId
    kind: EdgeKind


class JobRef(ContractModel):
    ref: str
    job_id: ResourceId
    job_type: JobType
    status: JobStatus


class JobGraph(ContractModel):
    graph_id: ResourceId
    conversation_id: ResourceId
    reason: GraphReason
    reanalysis_request_id: Optional[ResourceId] = None
    created_at: Timestamp
    created: bool = Field(description="False on an idempotent replay; the original graph is returned.")
    jobs: List[JobRef]
    edges: List[JobEdge] = Field(description="Every dependency edge of the graph's jobs, including edges to jobs of earlier graphs.")


# --- Job record ---------------------------------------------------------------------------


class LeaseInfo(ContractModel):
    worker_id: ShortText
    installation_id: ResourceId
    attempt_number: int = Field(ge=1, description="Sequence number of this claim on the job; never reused, including after a release.")
    granted_at: Timestamp
    expires_at: Timestamp
    claim_token_hash: Sha256Digest = Field(description="SHA-256 of the claim token. The token itself is never stored.")


class BlockingReason(ContractModel):
    job_id: ResourceId
    job_type: JobType
    status: JobStatus
    edge: EdgeKind
    dead: bool = Field(description="True when this upstream is FAILED or CANCELLED on a requires edge (or is itself dead-blocked): nothing will release the dependent until that upstream is retried.")


class WaitingReason(str, Enum):
    """Why a QUEUED or BLOCKED job is not running, for the compact progress view."""

    WAITING_FOR_DEPENDENCIES = "waiting_for_dependencies"
    DEAD_BLOCKED = "dead_blocked"
    RETRY_BACKOFF = "retry_backoff"
    DEFERRED_BY_WORKER = "deferred_by_worker"
    WAITING_FOR_WORKER = "waiting_for_worker"
    """No worker whose capabilities match (route, qualified entry, slot) has claimed it yet."""
    PRO1_PENDING_VERIFICATION = "pro1_pending_verification"


class JobOutput(ContractModel):
    role: ShortText
    artifact_id: ResourceId
    checksum: Sha256Digest


class ResolvedInputRef(ContractModel):
    role: ShortText
    artifact_id: ResourceId
    checksum: Sha256Digest


class Job(ContractModel):
    id: ResourceId
    conversation_id: ResourceId
    graph_id: ResourceId
    job_type: JobType
    execution_class: ExecutionClass
    status: JobStatus
    priority: int
    attempt_count: int = Field(ge=0, description="Inference executions started: claims that were not released.")
    claim_count: int = Field(ge=0, description="Every claim, including released ones.")
    max_attempts: int = Field(ge=1)
    retry_generation: int = Field(ge=0, description="Incremented by each manual retry.")
    next_run_at: Optional[Timestamp] = Field(default=None, description="Earliest claim time while QUEUED (backoff or a worker's not_before).")
    waiting_reason: Optional[WaitingReason] = None
    lease: Optional[LeaseInfo] = None
    cancel_requested: bool = False
    blocking: List[BlockingReason] = Field(default_factory=list, description="Why a BLOCKED job is blocked.")
    error_code: Optional[JobErrorCode] = None
    error_detail: Optional[SafeText] = None
    inputs: List[JobInput]
    resolved_inputs: List[ResolvedInputRef] = Field(default_factory=list, description="Pinned when the job becomes QUEUED: every input's artifact and checksum (optional inputs of failed upstreams omitted).")
    requires_job_ids: List[ResourceId] = Field(default_factory=list)
    after_job_ids: List[ResourceId] = Field(default_factory=list)
    selection: Optional[FrozenSelection] = None
    resource_estimate: ResourceEstimate
    parameters: JobParameters
    idempotency_key: IdempotencyKey
    outputs: List[JobOutput] = Field(default_factory=list)
    result_version: Optional[int] = Field(default=None, ge=1)
    created_at: Timestamp
    updated_at: Timestamp
    completed_at: Optional[Timestamp] = None

    @model_validator(mode="after")
    def _status_consistency(self):
        if self.status is JobStatus.RUNNING and self.lease is None:
            raise ValueError("a RUNNING job holds a lease")
        if self.status is not JobStatus.RUNNING and self.lease is not None:
            raise ValueError("only a RUNNING job holds a lease")
        if self.status is JobStatus.SUCCEEDED and not self.outputs:
            raise ValueError("a SUCCEEDED job links its committed outputs")
        if self.attempt_count > self.claim_count:
            raise ValueError("attempts are claims that were not released")
        return self


class JobListQuery(PageQuery):
    conversation_id: Optional[ResourceId] = None
    graph_id: Optional[ResourceId] = None
    status: Optional[JobStatus] = None
    job_type: Optional[JobType] = None
    memory_slot: Optional[MemorySlot] = Field(default=None, description="Added in 1.3.0 (on-device training): only jobs that need this memory slot. Process's training start check asks for local_memory jobs that are QUEUED or RUNNING with limit 1.")


# --- Claim, heartbeat, completion ---------------------------------------------------------


class SlotOffer(ContractModel):
    """Slots a worker reserved before asking. Store returns at most ``count`` jobs for this offer,
    each needing exactly this memory slot and, for outbound, exactly this connection."""

    memory_slot: MemorySlot
    outbound_connection_ref: Optional[ResourceId] = None
    count: int = Field(ge=1)

    @model_validator(mode="after")
    def _outbound(self):
        if (self.memory_slot is MemorySlot.OUTBOUND) != (self.outbound_connection_ref is not None):
            raise ValueError("outbound offers name their connection; other offers do not")
        return self


class WorkerCapabilities(ContractModel):
    worker_id: ShortText
    installation_id: ResourceId = Field(description="Must equal the calling key's installation (403 forbidden otherwise).")
    hardware_profile_id: ResourceId
    primary_host: bool = Field(description="True only for the designated primary Process host, which alone may claim primary_host jobs.")
    job_types: List[JobType] = Field(min_length=1)
    route_classes: List[RouteClass] = Field(default_factory=list, description="Routes this worker is configured and permitted for.")
    qualified_entries: List[CatalogEntryRef] = Field(default_factory=list, description="Catalog entries installed and qualified here; a job is eligible only if its frozen entry is listed.")
    slot_offers: List[SlotOffer] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_offers(self):
        keys = [(o.memory_slot, o.outbound_connection_ref) for o in self.slot_offers]
        if len(keys) != len(set(keys)):
            raise ValueError("one offer per slot and connection")
        return self


class ClaimRequest(ContractModel):
    worker: WorkerCapabilities
    max_jobs: int = Field(default=1, ge=1, le=64, description="At most MAX_CLAIM_BATCH and at most the sum of the offers' counts.")
    lease_seconds: Optional[int] = Field(default=None, ge=30, description="Defaults to LEASE_DURATION; Store may shorten it.")
    conversation_id: Optional[ResourceId] = Field(default=None, description="Optional affinity: prefer this conversation's ready jobs.")

    @model_validator(mode="after")
    def _within_offers(self):
        if self.max_jobs > sum(o.count for o in self.worker.slot_offers):
            raise ValueError("max_jobs cannot exceed the slots offered")
        return self


ClaimToken = str


class ResolvedInput(ContractModel):
    role: ShortText
    artifact: Optional[Artifact] = Field(description="Null only for an optional input whose after-edge upstream did not succeed.")


class UpstreamOutcome(ContractModel):
    job_id: ResourceId
    job_type: JobType
    edge: EdgeKind
    status: JobStatus
    error_code: Optional[JobErrorCode] = None


class ClaimedJob(ContractModel):
    job: Job
    attempt_number: int = Field(ge=1, description="This claim's sequence number (LeaseInfo.attempt_number).")
    claim_token: str = Field(pattern=r"^[A-Za-z0-9_-]{32,128}$", description="Capability for this claim only. Sent on heartbeat, completion, failure and release; Store stores only its hash.")
    lease_expires_at: Timestamp
    slot_offer_index: int = Field(ge=0, description="The index in WorkerCapabilities.slot_offers this job consumed.")
    final_attempt: bool = Field(description="True when attempt_count == max_attempts after this claim: a QA assessment then records a provider failure as FLAGGED instead of failing.")
    inputs: List[ResolvedInput] = Field(description="Every input resolved to its committed artifact record; content is fetched through the artifact routes.")
    upstream: List[UpstreamOutcome] = Field(default_factory=list, description="The outcome of every upstream, so a merge on after edges knows what is missing.")


class ClaimResponse(ContractModel):
    jobs: List[ClaimedJob]
    lease_duration_seconds: int
    heartbeat_interval_seconds: int
    no_eligible_reason: Optional[ShortText] = Field(default=None, description="Why nothing was returned, when jobs is empty (e.g. 'no ready jobs', 'ready jobs need a slot you did not offer').")


class ProgressNote(ContractModel):
    fraction: Optional[float] = Field(default=None, ge=0, le=1)
    note: Optional[SafeText] = None


class HeartbeatRequest(ContractModel):
    claim_token: str
    progress: Optional[ProgressNote] = None


class HeartbeatResponse(ContractModel):
    lease_expires_at: Timestamp
    cancel_requested: bool


class AttemptProvenance(ContractModel):
    """Who ran the attempt and exactly how. On the Pro1 route it carries the attestation record and
    key-release reference once verification passed, and ``pro1_failure`` when the attempt failed or
    completed FLAGGED on a Pro1 provider failure (with both, when that happened after key release;
    ``failed_check: interrupted`` when a cancel, crash, input or publication failure stopped it
    before verification); on every other route none of them."""

    worker_id: ShortText
    installation_id: ResourceId = Field(description="Must equal the calling key's installation.")
    adapter_id: ShortText
    adapter_version: ShortText
    model_revision: Optional[ShortText] = Field(default=None, description="Exact model revision used; absent for code stages.")
    provider_reported_model_id: Optional[ShortText] = None
    route: Optional[RouteRecord] = Field(default=None, description="The route the attempt used; equals the frozen route. Absent for code stages.")
    attestation: Optional[Pro1AttestationRecord] = None
    key_release_ref: Optional[ResourceId] = None
    pro1_failure: Optional[Pro1AttemptFailureEvidence] = None
    resource_estimate_used: Optional[ResourceEstimate] = None

    @model_validator(mode="after")
    def _pro1_evidence(self):
        confidential = self.route is not None and self.route.route_class is RouteClass.CALL1_CONFIDENTIAL
        if confidential:
            if self.attestation is None and self.pro1_failure is None:
                raise ValueError("a call1_confidential attempt records its attestation evidence, or the evidence it rejected")
            if (self.attestation is None) != (self.key_release_ref is None):
                raise ValueError("a verified attempt names its key release; an attempt that never reached key release does not")
            if self.attestation is not None and self.attestation.key_release_ref != self.key_release_ref:
                raise ValueError("the attestation record and the attempt name the same key release")
        elif self.attestation is not None or self.key_release_ref is not None or self.pro1_failure is not None:
            raise ValueError("only the call1_confidential route carries attestation, key-release or Pro1 failure records")
        return self


class NewDependency(ContractModel):
    """Make an existing BLOCKED job that directly requires the completing job also require a job
    created in the same completion (success edge). Anything else is graph_invalid.

    With ``input_role`` and ``output_role``, Store also appends
    ``JobInput{role: input_role, upstream: {job_id: <the new job>, output_role}}`` to the dependent
    in the same transaction, and resolves and pins it when it releases the dependent, like any other
    upstream-output input. A triggered escalation uses this to hand its ``assessment`` to the
    scorecard (for example ``input_role: "escalation:greeting"``)."""

    dependent_job_id: ResourceId
    requires_ref: str = Field(pattern=_REF.pattern)
    input_role: Optional[ShortText] = Field(default=None, description="New input role on the dependent; must not already exist there (graph_invalid otherwise).")
    output_role: Optional[ShortText] = Field(default=None, description="The output role of the new job that the input reads; one its job type declares.")

    @model_validator(mode="after")
    def _binding(self):
        if (self.input_role is None) != (self.output_role is None):
            raise ValueError("an input binding names both the dependent's input role and the new job's output role")
        return self


class FollowOnJobs(ContractModel):
    """Jobs created atomically with a completion, e.g. a triggered escalation plus the scorecard's
    new dependency on it, so an assembler cannot publish before the escalation exists. Follow-on
    jobs may list the completing job in ``requires_job_ids``: it is SUCCEEDED before their initial
    status is computed (COMPLETION_TRANSACTION_STEPS)."""

    jobs: List[JobDefinition] = Field(min_length=1, max_length=64)
    add_dependencies: List[NewDependency] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid(self):
        _check_graph(self.jobs)
        by_ref = {job.ref: job for job in self.jobs}
        bound = set()
        for dep in self.add_dependencies:
            if dep.requires_ref not in by_ref:
                raise ValueError("add_dependencies reference refs created in this completion")
            if dep.output_role is not None:
                upstream_type = by_ref[dep.requires_ref].job_type
                if dep.output_role not in JOB_TYPE_RULES[upstream_type].outputs:
                    raise ValueError(f"{upstream_type.value} has no output role {dep.output_role}")
                key = (dep.dependent_job_id, dep.input_role)
                if key in bound:
                    raise ValueError("one binding per dependent input role")
                bound.add(key)
        return self


class ResultPublication(ContractModel):
    """Tells Store which result group the output publishes to and whether it is complete."""

    kind: ResultKind
    state: ResultState = ResultState.AVAILABLE
    partial_reason: Optional[ShortText] = None

    @model_validator(mode="after")
    def _publishable(self):
        if self.state not in (ResultState.AVAILABLE, ResultState.PARTIAL):
            raise ValueError("a completion publishes an available or partial result")
        if self.state is ResultState.PARTIAL and not self.partial_reason:
            raise ValueError("a partial result says what is missing")
        return self


COMPLETION_TRANSACTION_STEPS: List[str] = [
    "look up (job_id, completion_key): same request digest -> return the stored receipt (replayed=true), whatever the lease state; different digest -> completion_key_reused",
    "verify the claim token is the job's active claim (else claim_token_stale) and cancel_requested is false (else job_cancelling)",
    "verify every output: one committed artifact per declared role, kind and checksum match, produced by this job under this claim, and in a draft-test graph every output slot is under draft:<request_id>: (outside one, none is)",
    "verify the usage outcome is allowed for the job type (JobTypeRule.completion_outcomes), a failed outcome only on the claim's final attempt with a PROVIDER_FAILURE_CODES code, and the provenance matches the frozen route and the key's installation",
    "link the outputs: assign versions per slot, supersede the previous linked version of each slot (a draft-test graph's slots are its own, so it never supersedes the call's artifacts)",
    "write the usage row, the attempt's provenance and the completion receipt",
    "mark the job SUCCEEDED and clear its lease",
    "project the result (ResultPublication) unless the graph is a draft test, whose scorecard is recorded on the request instead: no projection, no review-queue items, no change to the call's result groups",
    "insert follow-on jobs and their edges, then add_dependencies edges and their input bindings (dependents must be BLOCKED and directly require this job; a bound input role must be new on the dependent)",
    "compute the initial status of each follow-on job (QUEUED when every upstream is already satisfied, else BLOCKED)",
    "release every BLOCKED dependent whose edges are now all satisfied, resolving and pinning its upstream-output inputs",
    "write the change events and return the receipt with released and created job IDs",
]
"""The order Store applies inside the one completion transaction (plan, "Job lifecycle" step 4)."""


class CompletionRequest(ContractModel):
    """Idempotent completion. The request digest is ``canonical_digest`` of this body; Store looks
    up the completion key before it checks the token (COMPLETION_TRANSACTION_STEPS), so a lost
    response is replayed with the same key and body and returns the original receipt."""

    claim_token: str
    completion_key: IdempotencyKey = Field(description="Stable per claim; complete, fail and release share the key space, one call ends a claim.")
    outputs: List[JobOutput] = Field(min_length=1, description="Committed artifacts only: one per role in the job type's outputs, plus at most one per role in its optional_outputs (1.3.0); Store verifies each ID and checksum.")
    usage: UsageRecordInput
    provenance: AttemptProvenance
    result: Optional[ResultPublication] = Field(default=None, description="Present exactly for the group's publishing job type (JobTypeRule.publishes), except in a draft-test graph, where it must be null (graph_invalid otherwise).")
    follow_on: Optional[FollowOnJobs] = None

    @model_validator(mode="after")
    def _consistent(self):
        roles = [o.role for o in self.outputs]
        if len(roles) != len(set(roles)):
            raise ValueError("one output per role")
        if self.usage.outcome is UsageOutcome.CANCELLED:
            raise ValueError("a cancelled attempt is reported with /fail, code cancelled")
        flagged_failure = self.usage.outcome is UsageOutcome.FAILED
        if flagged_failure and self.usage.error_code not in PROVIDER_FAILURE_CODES:
            raise ValueError("a completion records only a final-attempt provider failure (PROVIDER_FAILURE_CODES) as failed; anything else is reported with /fail")
        route = self.provenance.route
        confidential = route is not None and route.route_class is RouteClass.CALL1_CONFIDENTIAL
        failure = self.provenance.pro1_failure
        if confidential and flagged_failure:
            # Pro1ConfidentialInference.md section 4, "Escalation": pro1_unreachable and
            # pro1_service_error still fire a configured escalation, so the QA assessment completes
            # FLAGGED and the attempt keeps its failure evidence.
            if self.usage.error_code not in PRO1_PROVIDER_FAILURE_CODES:
                raise ValueError("a Pro1 attempt records a provider failure as pro1_unreachable or pro1_service_error")
            if failure is None or failure.error_code is not self.usage.error_code:
                raise ValueError("a FLAGGED Pro1 provider failure carries pro1_failure with the usage row's error code")
            if self.usage.error_code is JobErrorCode.PRO1_SERVICE_ERROR and self.provenance.attestation is None:
                raise ValueError("pro1_service_error comes from a sealed response, so the attempt carries its attestation record")
        elif confidential and (self.provenance.attestation is None or failure is not None):
            raise ValueError("a completed Pro1 attempt carries its attestation record and no failure evidence")
        elif flagged_failure and self.usage.error_code in PRO1_PROVIDER_FAILURE_CODES:
            raise ValueError("pro1_* provider failures belong to the call1_confidential route")
        return self


class CompletionReceipt(ContractModel):
    receipt_id: ResourceId
    job_id: ResourceId
    attempt_number: int = Field(ge=1)
    completion_key: IdempotencyKey
    status: JobStatus = Field(description="SUCCEEDED.")
    result_version: Optional[int] = Field(default=None, ge=1)
    linked_artifact_ids: List[ResourceId] = Field(default_factory=list)
    released_job_ids: List[ResourceId] = Field(default_factory=list, description="Dependents that moved BLOCKED to QUEUED in this transaction.")
    created_job_ids: List[ResourceId] = Field(default_factory=list, description="Follow-on jobs created in this transaction.")
    usage_record_id: ResourceId
    change_cursor: ChangeCursor
    committed_at: Timestamp
    replayed: bool = Field(description="True when this receipt was returned for a repeated completion_key.")


class FailureRequest(ContractModel):
    """Idempotent like completion (same key space and lookup order). ``usage.outcome`` must be the
    outcome ``usage_outcome_for(error_code)`` names and ``usage.error_code`` must equal
    ``error_code``. On the Pro1 route, ``provenance.pro1_failure.error_code`` also equals
    ``error_code``, and a ``pro1_*`` code always carries that evidence."""

    claim_token: str
    completion_key: IdempotencyKey
    error_code: JobErrorCode
    error_detail: Optional[SafeText] = Field(default=None, description="Safe text. No provider bodies, prompts, URLs with credentials or key material.")
    usage: UsageRecordInput
    provenance: Optional[AttemptProvenance] = Field(default=None, description="As much as is known; absent only when the attempt failed before its route was resolved. A Pro1 attempt carries pro1_failure.")

    @model_validator(mode="after")
    def _usage_matches(self):
        if self.error_code is JobErrorCode.LEASE_EXPIRED:
            raise ValueError("lease_expired is Store's code; a worker never reports it")
        if self.usage.error_code is not self.error_code:
            raise ValueError("the usage row carries the same error code as the failure")
        if self.usage.outcome is not usage_outcome_for(self.error_code):
            raise ValueError(f"a {self.error_code.value} failure records usage outcome {usage_outcome_for(self.error_code).value}")
        prov = self.provenance
        failure = prov.pro1_failure if prov is not None else None
        if failure is not None and failure.error_code is not self.error_code:
            raise ValueError("the Pro1 failure evidence carries the failure's error code")
        if self.error_code.value.startswith("pro1_") and failure is None:
            raise ValueError("a pro1_* failure is a call1_confidential attempt and carries its pro1_failure evidence")
        return self


class FailureReceipt(ContractModel):
    receipt_id: ResourceId
    job_id: ResourceId
    attempt_number: int = Field(ge=1)
    completion_key: IdempotencyKey
    status: JobStatus = Field(description="QUEUED when retry is scheduled, FAILED when terminal, CANCELLED when the failure acknowledged a cancel.")
    error_class: JobErrorClass
    next_run_at: Optional[Timestamp] = None
    attempts_remaining: int = Field(ge=0)
    pro1_connection_blocked: bool = Field(default=False, description="True when Store blocked the Pro1 connection in this transaction because the code is a definitive Pro1 code.")
    released_job_ids: List[ResourceId] = Field(default_factory=list, description="Dependents on after edges released because this job became terminal.")
    usage_record_id: ResourceId
    change_cursor: ChangeCursor
    committed_at: Timestamp
    replayed: bool


RELEASE_REQUEUE_CODES = frozenset({JobErrorCode.RESOURCE_UNAVAILABLE, JobErrorCode.INPUT_UNAVAILABLE})
"""Temporary local conditions found before inference started (slot lost, input fetch failed): the
job goes back to QUEUED, attempt refunded. Transient Pro1 conditions (unreachable, revocation
unavailable, release unapproved) are reported with /fail so they retry with backoff and exhaust
attempts (Pro1ConfidentialInference.md section 4)."""

RELEASE_REJECT_CODES = frozenset(code for code, cls in JOB_ERROR_CLASSES.items() if cls is JobErrorClass.CONFIGURATION)
"""Configuration conditions found before inference started: the job FAILS, attempt refunded."""


class JobReleaseRequest(ContractModel):
    """End a claim before any inference started, without consuming an attempt (plan: 'temporary
    resource pressure delays it without consuming an attempt'). No usage row: a released claim is
    not an attempt. Idempotent by completion_key like complete and fail."""

    claim_token: str
    completion_key: IdempotencyKey
    disposition: Literal["requeue", "reject"]
    reason_code: JobErrorCode
    detail: Optional[SafeText] = None
    not_before: Optional[Timestamp] = Field(default=None, description="requeue only: the earliest next claim time (the 'next check time').")

    @model_validator(mode="after")
    def _codes(self):
        allowed = RELEASE_REQUEUE_CODES if self.disposition == "requeue" else RELEASE_REJECT_CODES
        if self.reason_code not in allowed:
            raise ValueError(f"{self.disposition} takes one of {sorted(c.value for c in allowed)}")
        if self.not_before is not None and self.disposition != "requeue":
            raise ValueError("only a requeue sets not_before")
        return self


class JobReleaseReceipt(ContractModel):
    job_id: ResourceId
    attempt_number: int = Field(ge=1)
    completion_key: IdempotencyKey
    status: JobStatus = Field(description="QUEUED for requeue, FAILED for reject.")
    attempt_count: int = Field(ge=0)
    next_run_at: Optional[Timestamp] = None
    change_cursor: ChangeCursor
    committed_at: Timestamp
    replayed: bool


class RetryRequest(ContractModel):
    reason: SafeText


class CancelRequest(ContractModel):
    """Cancelling a RUNNING job sets cancel_requested; the worker learns of it on heartbeat, or as
    409 ``job_cancelling`` if it tries to complete, and then reports failure code ``cancelled``."""

    reason: SafeText
    cascade: bool = Field(default=True, description="Also cancel every job that depends on this one, transitively.")


class CancelResponse(ContractModel):
    job: Job
    cancelled_job_ids: List[ResourceId]


class AttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LEASE_EXPIRED = "lease_expired"
    RELEASED = "released"
    """Ended by a release before inference started; not counted as an attempt, no usage row."""


class Attempt(ContractModel):
    job_id: ResourceId
    attempt_number: int = Field(ge=1)
    status: AttemptStatus
    counts_as_attempt: bool = Field(description="False for a released claim.")
    started_at: Timestamp
    ended_at: Optional[Timestamp] = None
    worker_id: ShortText
    installation_id: ResourceId
    claim_token_hash: Sha256Digest
    error_code: Optional[JobErrorCode] = None
    error_detail: Optional[SafeText] = None
    provenance: Optional[AttemptProvenance] = None
    usage_record_id: Optional[ResourceId] = Field(default=None, description="Present for every attempt that counts once it has ended (synthesized on lease expiry).")
    resource_estimate: ResourceEstimate

    @model_validator(mode="after")
    def _released(self):
        if (self.status is AttemptStatus.RELEASED) == self.counts_as_attempt:
            raise ValueError("a released claim, and only it, does not count as an attempt")
        if self.status is AttemptStatus.RELEASED and self.usage_record_id is not None:
            raise ValueError("a released claim has no usage row")
        return self


# --- Group progress ----------------------------------------------------------------------


class GroupProgress(ContractModel):
    """Counts of one result group's jobs. Jobs of draft-test graphs are never counted."""

    kind: ResultKind
    state: ResultState = Field(description="derive_result_state for this group.")
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    running: int = Field(ge=0)
    queued: int = Field(ge=0)
    blocked: int = Field(ge=0)
    dead_blocked: int = Field(ge=0, description="Of blocked: waiting on a FAILED or CANCELLED upstream.")
    failed: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    waiting_reason: Optional[WaitingReason] = None


class JobGroupProgress(ContractModel):
    conversation_id: ResourceId
    groups: List[GroupProgress]
    supporting_jobs_total: int = Field(ge=0, description="Jobs whose type has no group (embeddings).")
    settled: bool = Field(description="Every job outside draft-test graphs is terminal or dead-blocked.")
    updated_at: Timestamp


# --- Reanalysis requests (Store -> Process) -----------------------------------------------


class ReanalysisKind(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    CONTACT_SIGNALS = "contact_signals"
    SPEAKER_CORRECTION = "speaker_correction"
    FULL = "full"
    QA_DRAFT_TEST = "qa_draft_test"
    """Score one call with an unpublished rubric draft. Created only through
    ``POST /rubrics/{rubric_id}/draft/tests``; the result is readable on the request only."""
    EMBEDDINGS = "embeddings"
    """Added in 1.2.0. Re-embed the call's linked transcript for semantic search (one ``embeddings``
    job), e.g. after the search embedder changed. Affects no result group."""
    CONTACT_SIGNALS_PREVIEW = "contact_signals_preview"
    """Added in 1.3.0. A draft-test kind: run Contact Signals v2 on one call into draft slots, either
    to preview a taxonomy (``createSignalPreview``, priority +5) or to compare v2 with a published v1
    result (a compare backfill or a shadow-mode companion, priority -10). Nothing reaches the call's
    group, projections, queue or metrics; the merge output is recorded on the request as
    ``preview_result_artifact_id``. Never created through ``requestReanalysis``."""


DRAFT_TEST_KINDS: FrozenSet[ReanalysisKind] = frozenset({ReanalysisKind.QA_DRAFT_TEST, ReanalysisKind.CONTACT_SIGNALS_PREVIEW})
"""Added in 1.3.0. Kinds whose graph is a draft-test graph: every output lives in the request's
``draft:<request_id>:`` slots, completions carry ``result: null`` and nothing is projected, and the
graph never counts toward the call's result groups (``calls.result_state_inputs``). Store sets the
graph's draft-test request for any of these kinds; the draft-rubric lookup stays QA-only."""

SIGNAL_REANALYSIS_KINDS: FrozenSet[ReanalysisKind] = frozenset({ReanalysisKind.CONTACT_SIGNALS, ReanalysisKind.CONTACT_SIGNALS_PREVIEW, ReanalysisKind.SPEAKER_CORRECTION, ReanalysisKind.FULL})
"""Added in 1.3.0. Kinds whose graph runs contact signals, so Store resolves ``signal_taxonomy_version``
and ``signal_pipeline`` on them at creation."""

REANALYSIS_PRIORITY_DEFAULT = 0
REANALYSIS_PRIORITY_SIGNAL_PREVIEW = 5
"""An admin is waiting on a taxonomy preview."""
REANALYSIS_PRIORITY_SIGNAL_BACKFILL = -10
"""Backfills and compare requests never delay new calls."""


REANALYSIS_KIND_AFFECTS: Dict[ReanalysisKind, FrozenSet[ResultKind]] = {
    ReanalysisKind.QA: frozenset({ResultKind.QA}),
    ReanalysisKind.SUMMARY: frozenset({ResultKind.SUMMARY}),
    ReanalysisKind.CONTACT_SIGNALS: frozenset({ResultKind.CONTACT_SIGNALS}),
    ReanalysisKind.SPEAKER_CORRECTION: frozenset({ResultKind.TONE, ResultKind.TEXT_SENTIMENT, ResultKind.QA, ResultKind.SUMMARY, ResultKind.CONTACT_SIGNALS}),
    ReanalysisKind.FULL: frozenset(ResultKind),
    ReanalysisKind.QA_DRAFT_TEST: frozenset(),
    ReanalysisKind.EMBEDDINGS: frozenset(),
    ReanalysisKind.CONTACT_SIGNALS_PREVIEW: frozenset(),
}
"""The result groups that go ``stale`` from the moment a request of each kind is created until its
graph publishes (or fails, leaving the old version shown with a failure code).

A kind lists only groups whose publishing job (``JobTypeRule.publishes``) the fulfilling graph
reruns; stale ends through that publisher, so a group without one would drop back to its old
state as soon as the request is fulfilled. A speaker correction therefore does not list the
transcript: ``asr`` is not rerun. Its graph's ``speaker_attribution`` job (with
``parameters.speaker_correction``) writes a new ``reviewer_correction`` attribution, which
``TranscriptView`` shows once it is linked (``speaker_attribution_version``). A draft test lists
nothing: its graph never counts toward the call's result groups (``calls.result_state_inputs``).
Nor does ``embeddings`` (1.2.0): the embeddings job is supporting work with no group. Nor does
``contact_signals_preview`` (1.3.0), the second draft-test kind (``DRAFT_TEST_KINDS``)."""


class ReanalysisRequestCreate(ContractModel):
    kind: ReanalysisKind
    rubric: Optional[RubricVersionRef] = Field(default=None, description="For QA: the immutable rubric version to score with; defaults to the call's current rubric.")
    speaker_correction: Optional[SpeakerCorrection] = None
    note: Optional[SafeText] = None
    rescore_signals: bool = Field(default=False, description="Added in 1.3.0; contact_signals only. False reruns only the stages whose digests differ from the current taxonomy's; true reruns every stage (e.g. after an engine change).")

    @model_validator(mode="after")
    def _kind_fields(self):
        if self.kind is ReanalysisKind.QA_DRAFT_TEST:
            raise ValueError("draft tests are requested through POST /rubrics/{rubric_id}/draft/tests")
        if self.kind is ReanalysisKind.CONTACT_SIGNALS_PREVIEW:
            raise ValueError("signal previews are requested through POST /signals/previews or a compare backfill")
        if (self.kind is ReanalysisKind.SPEAKER_CORRECTION) != (self.speaker_correction is not None):
            raise ValueError("speaker correction requests carry the correction; others do not")
        if self.rescore_signals and self.kind is not ReanalysisKind.CONTACT_SIGNALS:
            raise ValueError("rescore_signals applies to a contact_signals request")
        return self


class DraftTestRequest(ContractModel):
    """Score one call with the rubric's current draft. Store snapshots the stored draft into a
    ``rubric_snapshot`` artifact in the same transaction and creates a qa_draft_test request."""

    call_id: ResourceId
    expected_draft_revision: int = Field(ge=1)
    note: Optional[SafeText] = None


class ReanalysisStatus(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    FULFILLED = "fulfilled"
    REJECTED = "rejected"


class ReanalysisTransition(ContractModel):
    from_status: ReanalysisStatus
    to_status: ReanalysisStatus
    trigger: ShortText
    actor: TransitionActor


REANALYSIS_TRANSITIONS: List[ReanalysisTransition] = [
    ReanalysisTransition(from_status=ReanalysisStatus.PENDING, to_status=ReanalysisStatus.CLAIMED, trigger="claim", actor=TransitionActor.PROCESS),
    ReanalysisTransition(from_status=ReanalysisStatus.CLAIMED, to_status=ReanalysisStatus.PENDING, trigger="claim_lease_expired", actor=TransitionActor.STORE),
    ReanalysisTransition(from_status=ReanalysisStatus.CLAIMED, to_status=ReanalysisStatus.FULFILLED, trigger="graph_created", actor=TransitionActor.PROCESS),
    ReanalysisTransition(from_status=ReanalysisStatus.CLAIMED, to_status=ReanalysisStatus.REJECTED, trigger="reject", actor=TransitionActor.PROCESS),
]
"""The reanalysis request state machine. FULFILLED happens only inside the graph-creation
transaction that carries the active claim token."""


class ReanalysisRequest(ContractModel):
    id: ResourceId
    call_id: ResourceId
    conversation_id: ResourceId
    kind: ReanalysisKind
    rubric: Optional[RubricVersionRef] = None
    draft_rubric: Optional[DraftRubricRef] = Field(default=None, description="qa_draft_test only: the snapshot Store minted from the draft.")
    speaker_correction: Optional[SpeakerCorrection] = None
    note: Optional[SafeText] = None
    status: ReanalysisStatus
    requested_by_account_id: Optional[ResourceId] = Field(default=None, description="Absent for migration-created requests.")
    requested_at: Timestamp
    claimed_by_installation_id: Optional[ResourceId] = None
    claim_expires_at: Optional[Timestamp] = None
    graph_id: Optional[ResourceId] = None
    draft_result_artifact_id: Optional[ResourceId] = Field(default=None, description="qa_draft_test only: the qa_scorecard artifact once the graph succeeds, linked in the request's draft:<request_id>: slot. Never projected as the call's QA and never superseding the call's artifacts.")
    rejected_reason: Optional[SafeText] = None
    idempotency_key: IdempotencyKey
    # --- Contact Signals v2 (1.3.0) ---
    priority: int = Field(default=REANALYSIS_PRIORITY_DEFAULT, description="Added in 1.3.0; read-only. +5 for taxonomy previews, -10 for backfills and compare requests, 0 otherwise. Claims order by priority desc, requested_at, id (reanalysis_claim_order_key); the fulfilling graph's jobs take the same priority.")
    rescore_signals: bool = Field(default=False, description="Added in 1.3.0; contact_signals only. Rerun every stage rather than only the outdated ones.")
    signal_taxonomy_version: Optional[int] = Field(default=None, ge=1, description="Added in 1.3.0. The published signal taxonomy version Store resolved at creation (SIGNAL_REANALYSIS_KINDS). Widening a pending request takes the latest version.")
    signal_pipeline: Optional[Literal["v1", "v2"]] = Field(default=None, description="Added in 1.3.0. The pipeline the fulfilling graph builds, resolved by Store at creation: v2 for every contact_signals_preview request; otherwise the settings pipeline, with shadow building v1 (its compare companion builds v2). Process may still fall back to v1 with pipeline_note when no qualified engine is installed and v1_fallback is set.")
    signal_backfill_id: Optional[ResourceId] = Field(default=None, description="Added in 1.3.0. The backfill that created this request (contact_signals for rescore, contact_signals_preview for compare).")
    signal_preview_id: Optional[ResourceId] = Field(default=None, description="Added in 1.3.0. Set exactly on contact_signals_preview requests: the preview (or compare) the result belongs to.")
    signal_taxonomy_snapshot_artifact_id: Optional[ResourceId] = Field(default=None, description="Added in 1.3.0; contact_signals_preview only. The snapshot Store minted for the request (a preview's unsaved taxonomy, or the published version for a compare).")
    preview_result_artifact_id: Optional[ResourceId] = Field(default=None, description="Added in 1.3.0; contact_signals_preview only. The merge's contact_signals artifact once the graph succeeds, linked in the request's draft:<request_id>: slot. Never projected.")

    @model_validator(mode="after")
    def _kind_fields(self):
        if (self.kind is ReanalysisKind.QA_DRAFT_TEST) != (self.draft_rubric is not None):
            raise ValueError("a draft test, and only a draft test, carries the draft snapshot")
        if self.draft_result_artifact_id is not None and self.kind is not ReanalysisKind.QA_DRAFT_TEST:
            raise ValueError("only a draft test records a result on the request")
        preview = self.kind is ReanalysisKind.CONTACT_SIGNALS_PREVIEW
        if preview != (self.signal_preview_id is not None):
            raise ValueError("a contact_signals_preview request, and only one, names its preview")
        if not preview and (self.signal_taxonomy_snapshot_artifact_id is not None or self.preview_result_artifact_id is not None):
            raise ValueError("only a contact_signals_preview request carries a preview snapshot or result")
        if self.rescore_signals and self.kind is not ReanalysisKind.CONTACT_SIGNALS:
            raise ValueError("rescore_signals applies to a contact_signals request")
        if self.signal_backfill_id is not None and self.kind not in (ReanalysisKind.CONTACT_SIGNALS, ReanalysisKind.CONTACT_SIGNALS_PREVIEW):
            raise ValueError("a backfill creates contact_signals or contact_signals_preview requests only")
        if (self.signal_taxonomy_version is not None or self.signal_pipeline is not None) and self.kind not in SIGNAL_REANALYSIS_KINDS:
            raise ValueError("only a request whose graph runs contact signals resolves a taxonomy version and pipeline")
        if preview and self.signal_pipeline == "v1":
            raise ValueError("a contact_signals_preview request runs v2")
        return self


def reanalysis_claim_order_key(request: "ReanalysisRequest") -> Tuple[int, object, str]:
    """Added in 1.3.0. Claim order: ``priority`` descending, then ``requested_at``, then ``id``."""
    return (-request.priority, request.requested_at, request.id)


class DraftTestResult(ContractModel):
    """What Rubric Studio renders for a draft test (permission manage_rubrics)."""

    request_id: ResourceId
    call_id: ResourceId
    rubric_id: ShortText
    draft_revision: int = Field(ge=1)
    state: Literal["pending", "available", "failed"] = Field(description="failed once the draft graph's qa_scorecard is FAILED, CANCELLED or dead-blocked; the call's own QA state is unaffected either way.")
    failure_code: Optional[JobErrorCode] = None
    scorecard: Optional[QaScorecardContent] = Field(default=None, description="Masked like any reviewer read; present when available.")


class ReanalysisClaimRequest(ContractModel):
    """Claim pending requests. Since 1.3.0 a worker that serves only some kinds names them in
    ``kinds``; absent means every kind. A 1.3.0 Store returns requests with the 1.3.0 fields, which a
    1.2.x Process rejects, so Process and Store upgrade together (README, "Version 1.3.0")."""

    worker_id: ShortText
    max_requests: int = Field(default=1, ge=1, le=64)
    kinds: Optional[List[ReanalysisKind]] = Field(default=None, min_length=1, description="Added in 1.3.0. Claim only these kinds; absent claims every kind.")

    def accepts(self, kind: ReanalysisKind) -> bool:
        """Whether a request of ``kind`` may be returned to this claim."""
        return self.kinds is None or kind in self.kinds


class ClaimedReanalysisRequest(ContractModel):
    request: ReanalysisRequest
    claim_token: str = Field(pattern=r"^[A-Za-z0-9_-]{32,128}$", description="Sent as JobGraphRequest.reanalysis_claim_token, or on reject.")
    claim_expires_at: Timestamp


class ReanalysisClaimResponse(ContractModel):
    requests: List[ClaimedReanalysisRequest]


class ReanalysisReject(ContractModel):
    claim_token: str
    reason: SafeText
