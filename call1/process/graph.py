"""The job-graph builder: Stage 2 job graphs from the contract's job types and rules
(``call1.contracts.jobs.JOB_TYPE_RULES``).

An ingest graph for one registered call conversation and its pinned ``source_audio`` artifact:

====================  ===========================================================================
ref                   job
====================  ===========================================================================
``vad``               ``validation_vad`` (code) on the audio; publishes the media fields
``asr``               ``asr`` on the audio; publishes the transcript; at completion it adds the
                      summary segment and synthesis jobs (their windows need the transcript)
``speaker``           ``speaker_attribution`` (mono recordings only) on audio + transcript
``tone``              ``acoustic_tone`` on audio + transcript (+ attribution)
``sentiment``         ``text_sentiment`` on transcript (+ attribution)
``enrichment``        ``enrichment`` on the transcript (+ attribution): the numeric entities and,
                      since contract 1.2.0, the ``pii_findings`` of the model PII layer (decision
                      19) for this transcript revision; every masked text-model job pins them
``embeddings``        ``embeddings`` with Nemotron-3-Embed-1B (model stage, contract 1.2.0)
``qa-<n>-<id>``       one ``qa_criterion`` per semantic criterion; a triggered escalation is added
                      as a follow-on job at its completion
``qa-det``            ``qa_deterministic`` when the rubric has non-semantic checks
``scorecard``         ``qa_scorecard`` requiring every assessment (input ``assessment:<criterion>``)
``summary``           ``summary_assembly`` requiring ``asr``; segment inputs bound at ASR completion
``cs-lifecycle``      ``contact_signals_lifecycle``
``cs-resolution``     ``contact_signals_resolution``
``cs-merge``          ``contact_signals_merge`` with ``after`` edges and optional pass inputs
====================  ===========================================================================

Contact Signals v2 (contract 1.3.0, docs/ContactSignalsV2.md section 8.1) replaces the two passes
when the pinned ``signal_taxonomy_snapshot``'s settings say ``pipeline: v2`` and this host has usable
``signal_category`` and ``signal_subcategory`` entries (``add_signals_v2``):

====================  ===========================================================================
``cs-categorize``     ``contact_signals_categorize``: stage 1 on every ~7 s segment (primary host)
``cs-subcategorize``  ``contact_signals_subcategorize`` after it: stage 2 on each span
``cs-extract``        ``contact_signals_extract`` after both (LLM route), only when an active node
                      has fields or ``narrow_quote``
``cs-merge``          the same merge, ``stage:*`` inputs on ``after`` edges, ``pii_findings`` required
====================  ===========================================================================

Every v2 job pins the snapshot under input role ``taxonomy``, freezes its digest in
``parameters.signals`` and requires ``pii_findings``: v2 always reads masked text, even with
``mask_model_text`` off (decision 22, Q12). Segments are rows inside one job, not graph windows.

Dual transcription (contract 1.3.0, decision 33, docs/DualAsr.md) changes no ref and no edge: when
Store's ASR vocabulary is ``active``, the ``asr`` job freezes ``parameters.asr_vocabulary`` (the
effective terms, their digest and the catalog's ``asr_vocabulary`` entry, installed or not) and runs
the vocabulary pass and merge inside the job; its estimate is scaled by ``DUAL_ASR_RUNTIME_FACTOR``.

Every model-backed job freezes a ``FrozenSelection`` from the Process catalog when the graph is
built. Every QA job pins the Store-minted ``rubric_snapshot`` under input role ``rubric``.
Idempotency keys are stable per logical job (``<graph key>.<ref>``), so a replayed ingest returns
the same graph. Reanalysis graphs reuse the same stage builders with pinned artifacts in place of
upstream outputs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

from call1.contracts.artifacts import Artifact
from call1.contracts.catalog import ALWAYS_MASKED_PURPOSES, FrozenSelection, ModelPurpose
from call1.contracts.common import ArtifactRef
from call1.contracts.contents import ContactSignalPass, EscalationTrigger, TranscriptContent, TurnWindow
from call1.contracts.jobs import (
    FollowOnJobs,
    GraphReason,
    JobDefinition,
    JobGraph,
    JobGraphRequest,
    JobInput,
    JobParameters,
    JobStatus,
    JobType,
    MemorySlot,
    NewDependency,
    ResourceEstimate,
    SegmentSpec,
    SignalJobParameters,
    SizeClass,
    SpeakerCorrection,
    UpstreamOutput,
)
from call1.contracts.admin import MaskingSettings
from call1.contracts.rubrics import CheckType, DraftRubricRef, RubricCriterion, RubricDefinition, RubricVersionRef
from call1.contracts.signals import SIGNAL_TAXONOMY_INPUT_ROLE, SignalTaxonomySnapshotContent
from call1.contracts.vocabulary import AsrVocabularyParameters, AsrVocabularyRecord
from call1.pipeline.signals_v2 import ALL, DEFAULT_STAGE2_FACTORS, RerunPlan, full_plan

from .audio import AudioInfo
from .catalog import CatalogEntry, CatalogError, ProcessCatalog
from .config import ProcessConfig
from .transcripts import plan_segments
from .vocabulary import DUAL_ASR_RUNTIME_FACTOR, PARAKEET_REAL_TIME_FACTOR, vocabulary_parameters

TEXT_MODEL_PURPOSES = frozenset({ModelPurpose.SEMANTIC_QA, ModelPurpose.SUMMARY, ModelPurpose.CONTACT_SIGNALS,
                                 ModelPurpose.SIGNAL_CATEGORY, ModelPurpose.SIGNAL_SUBCATEGORY, ModelPurpose.SIGNAL_EXTRACTION})
"""The purposes whose prompts carry transcript text to a text model (masked on the appliance route
when appliance text masking is on; the three Contact Signals v2 purposes always are)."""

V2_UNAVAILABLE_NOTE = "v2 selected; no qualified classifier on this host"
"""``pipeline_note`` on a v1 result built because v2 was selected but the host has no usable
``signal_category`` and ``signal_subcategory`` entries (``v1_fallback`` on)."""

PRIORITY: Dict[JobType, int] = {
    JobType.VALIDATION_VAD: 50, JobType.ASR: 50, JobType.SPEAKER_ATTRIBUTION: 40,
    JobType.ACOUSTIC_TONE: 30, JobType.TEXT_SENTIMENT: 30, JobType.ENRICHMENT: 30, JobType.EMBEDDINGS: 20,
    JobType.QA_DETERMINISTIC: 20, JobType.QA_CRITERION: 20, JobType.QA_ESCALATION: 25, JobType.QA_SCORECARD: 20,
    JobType.SUMMARY_SEGMENT: 10, JobType.SUMMARY_SYNTHESIS: 10, JobType.SUMMARY_ASSEMBLY: 10,
    JobType.CONTACT_SIGNALS_LIFECYCLE: 10, JobType.CONTACT_SIGNALS_RESOLUTION: 10, JobType.CONTACT_SIGNALS_MERGE: 10,
    JobType.CONTACT_SIGNALS_CATEGORIZE: 10, JobType.CONTACT_SIGNALS_SUBCATEGORIZE: 10, JobType.CONTACT_SIGNALS_EXTRACT: 10,
}

CODE_SLOT: Dict[JobType, MemorySlot] = {
    JobType.VALIDATION_VAD: MemorySlot.IO,
    JobType.ENRICHMENT: MemorySlot.CPU,
    JobType.QA_DETERMINISTIC: MemorySlot.CPU,
    JobType.QA_SCORECARD: MemorySlot.CPU,
    JobType.SUMMARY_ASSEMBLY: MemorySlot.CPU,
    JobType.CONTACT_SIGNALS_MERGE: MemorySlot.CPU,
    JobType.SPEAKER_ATTRIBUTION: MemorySlot.CPU,  # only when it applies a reviewer correction (code stage)
    JobType.CONTACT_SIGNALS_CATEGORIZE: MemorySlot.CPU,  # only in rederive mode (code stage: no model)
}

PASS_JOB_TYPES = {ContactSignalPass.LIFECYCLE: JobType.CONTACT_SIGNALS_LIFECYCLE, ContactSignalPass.RESOLUTION: JobType.CONTACT_SIGNALS_RESOLUTION}

SCORECARD_REF = "scorecard"
SUMMARY_REF = "summary"
ESCALATION_REF = "esc"


class PlanError(ValueError):
    """The graph cannot be built (unknown or unavailable model, missing inputs). Safe text."""


@dataclass(frozen=True)
class Src:
    """Where an input comes from: a committed artifact pinned by checksum, an upstream job of this
    request by ref, or an existing job by ID, each with the upstream's output role."""

    artifact: Optional[ArtifactRef] = None
    ref: Optional[str] = None
    job_id: Optional[str] = None
    output_role: Optional[str] = None

    @classmethod
    def pinned(cls, artifact: Union[Artifact, ArtifactRef]) -> "Src":
        ref = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef(artifact_id=artifact.id, checksum=artifact.checksum)
        return cls(artifact=ref)

    @classmethod
    def upstream(cls, ref: str, output_role: str) -> "Src":
        return cls(ref=ref, output_role=output_role)

    @classmethod
    def job(cls, job_id: str, output_role: str) -> "Src":
        return cls(job_id=job_id, output_role=output_role)

    def as_input(self, role: str, optional: bool = False) -> JobInput:
        if self.artifact is not None:
            return JobInput(role=role, artifact=self.artifact)
        return JobInput(role=role, upstream=UpstreamOutput(ref=self.ref, job_id=self.job_id, output_role=self.output_role), optional=optional)


In = Tuple[str, Optional[Src], bool]
"""(input role, source, optional). A ``None`` source is skipped (a stage that is not in the graph)."""


def bounded_key(value: str) -> str:
    """An IdempotencyKey (``[A-Za-z0-9._:-]{8,128}``): long values keep a digest suffix."""
    clean = re.sub(r"[^A-Za-z0-9._:-]", "-", value)
    if len(clean) < 8:
        clean = clean + "." + hashlib.sha256(value.encode()).hexdigest()[:8]
    if len(clean) > 128:
        clean = clean[:111] + "." + hashlib.sha256(value.encode()).hexdigest()[:16]
    return clean


def slug(value: str, limit: int = 40) -> str:
    return (re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-") or "x")[:limit]


class _Jobs:
    """Accumulates job definitions; derives edges from the inputs' sources."""

    def __init__(self, key_prefix: str, priority_offset: int = 0) -> None:
        self.key_prefix = key_prefix
        self.priority_offset = priority_offset
        """Added to every job's priority: a reanalysis request's (+5 previews, -10 backfills; CF section 4.1)."""
        self.defs: List[JobDefinition] = []
        self.refs: Dict[str, JobDefinition] = {}

    def add(self, ref: str, job_type: JobType, *, inputs: Sequence[In] = (), estimate: ResourceEstimate,
            selection: Optional[FrozenSelection] = None, parameters: Optional[JobParameters] = None,
            requires: Sequence[Src] = (), priority: Optional[int] = None) -> str:
        requires_refs: List[str] = []
        requires_ids: List[str] = []
        after_refs: List[str] = []
        after_ids: List[str] = []

        def edge(src: Src, optional: bool) -> None:
            if src.ref is not None:
                target = after_refs if optional else requires_refs
                if src.ref not in target:
                    target.append(src.ref)
            elif src.job_id is not None:
                target = after_ids if optional else requires_ids
                if src.job_id not in target:
                    target.append(src.job_id)

        job_inputs = []
        for role, src, optional in inputs:
            if src is None:
                continue
            job_inputs.append(src.as_input(role, optional=optional and src.artifact is None))
            edge(src, optional and src.artifact is None)
        for src in requires:
            edge(src, False)
        definition = JobDefinition(
            ref=ref, job_type=job_type, idempotency_key=bounded_key(f"{self.key_prefix}.{ref}"),
            priority=(PRIORITY[job_type] if priority is None else priority) + self.priority_offset,
            inputs=job_inputs, selection=selection, resource_estimate=estimate, parameters=parameters or JobParameters(),
            requires_refs=requires_refs, requires_job_ids=requires_ids, after_refs=after_refs, after_job_ids=after_ids,
        )
        self.defs.append(definition)
        self.refs[ref] = definition
        return ref


def size_class(audio_seconds: Optional[float]) -> SizeClass:
    if audio_seconds is None:
        return SizeClass.M
    if audio_seconds < 60:
        return SizeClass.XS
    if audio_seconds < 300:
        return SizeClass.S
    if audio_seconds < 900:
        return SizeClass.M
    if audio_seconds < 1800:
        return SizeClass.L
    return SizeClass.XL


def estimate_for(slot: MemorySlot, audio_seconds: Optional[float], *, entry: Optional[CatalogEntry] = None,
                 tokens: Optional[int] = None, uses_audio: bool = False) -> ResourceEstimate:
    return ResourceEstimate(
        size_class=size_class(audio_seconds), memory_slot=slot,
        outbound_connection_ref=None,
        estimated_input_tokens=tokens, context_limit_tokens=entry.context_limit_tokens if entry else None,
        output_token_limit=entry.output_token_limit if entry else None,
        audio_seconds=round(audio_seconds, 3) if (uses_audio and audio_seconds is not None) else None,
    )


def transcript_tokens(audio_seconds: Optional[float]) -> Optional[int]:
    """About 2.3 spoken words a second and 1.3 tokens a word: an admission estimate only."""
    return int(audio_seconds * 3.0) + 64 if audio_seconds is not None else None


@dataclass
class QaSources:
    transcript: Src
    speaker: Optional[Src] = None
    enrichment: Optional[Src] = None
    sentiment: Optional[Src] = None
    pii: Optional[Src] = None
    """The ``enrichment`` job's ``pii_findings`` output (contract 1.2.0), pinned by masked text-model jobs."""
    tone: Optional[Src] = None


@dataclass
class SignalsInput:
    """What a graph's contact-signals stage is planned from: the pinned taxonomy snapshot (Store-minted)
    and its content, the pipeline the request resolved (``ReanalysisRequest.signal_pipeline``; else
    the snapshot's settings), the previous stage artifacts a reanalysis pins, and what to rerun."""

    snapshot: Artifact
    content: SignalTaxonomySnapshotContent
    pipeline: Optional[str] = None
    previous: Dict[str, Artifact] = field(default_factory=dict)
    """``categories``, ``subcategories``, ``extraction``: the call's current v2 stage artifacts."""
    plan: Optional[RerunPlan] = None
    """None runs every stage (ingest, full, speaker correction, previews)."""
    draft_request_id: Optional[str] = None
    """A ``contact_signals_preview`` request: every output lands in its draft slots."""
    preview_id: Optional[str] = None
    require_v2: bool = False
    """A preview or compare request runs v2 or is rejected; it never falls back to v1."""

    @property
    def effective_pipeline(self) -> str:
        return self.pipeline or self.content.settings.pipeline


class GraphPlanner:
    def __init__(self, catalog: ProcessCatalog, config: ProcessConfig, masking: Optional[MaskingSettings] = None) -> None:
        self.catalog = catalog
        self.config = config
        self.masking = masking or MaskingSettings()

    # --- selections --------------------------------------------------------------------------

    def mask_appliance_text(self) -> bool:
        """Whether text-model prompts on the appliance route are masked. By default this follows
        Store's text masking (``MaskingSettings.mask_reviewer_reads``, contract default on), which
        is the legacy ``redaction.text`` toggle: when it was on, the pre-split router and summarizer
        masked QA and summary prompts even for the included local model (team decision 3 keeps
        that). ``mask_model_text`` ``on``/``off`` in the Process config overrides it."""
        mode = getattr(self.config, "mask_model_text", "store")
        if mode == "on":
            return True
        if mode == "off":
            return False
        return bool(self.masking.mask_reviewer_reads)

    def masked(self, entry: CatalogEntry, purpose: Optional[ModelPurpose] = None) -> bool:
        """The frozen ``RouteRecord.masked`` for a selection: every purpose on a route class in
        ``masked_route_classes``; on the appliance route, the text-model purposes (QA, summaries,
        contact signals) when appliance text masking is on. Audio and embedding models never are:
        their outputs (transcript, tone, sentiment) stay raw or derived."""
        if entry.route_class in self.masking.masked_route_classes or purpose in ALWAYS_MASKED_PURPOSES:
            return True
        return purpose in TEXT_MODEL_PURPOSES and self.mask_appliance_text()

    def pins_enrichment(self, entry: CatalogEntry) -> bool:
        """Jobs on a route class in ``masked_route_classes`` pin the ``enrichment`` artifact for
        their masking. Appliance text masking adds no graph edge: the handler runs the same
        deterministic numeric extractor in-process (``handlers/real/masking.py``)."""
        return entry.route_class in self.masking.masked_route_classes

    def pick(self, purpose: ModelPurpose, entry_id: Optional[str] = None) -> CatalogEntry:
        try:
            return self.catalog.select(purpose, entry_id)
        except CatalogError as exc:
            raise PlanError(str(exc)) from None

    def selection(self, entry: CatalogEntry, purpose: ModelPurpose, output_contract: Optional[str] = None) -> FrozenSelection:
        return entry.selection(purpose, output_contract=output_contract, masked=self.masked(entry, purpose))

    def asr_vocabulary(self, record: Optional[AsrVocabularyRecord]) -> Optional[AsrVocabularyParameters]:
        """What a new ``asr`` job freezes for dual transcription (``vocabulary.vocabulary_parameters``):
        None unless the record is active and the catalog has an ``asr_vocabulary`` entry."""
        return vocabulary_parameters(record, self.catalog)

    def _slot(self, entry: Optional[CatalogEntry], job_type: JobType) -> MemorySlot:
        return entry.memory_slot if entry is not None else CODE_SLOT[job_type]

    # --- whole graphs ------------------------------------------------------------------------

    def ingest(self, *, conversation_id: str, source: Artifact, audio: AudioInfo, rubric_ref: RubricVersionRef,
               rubric: RubricDefinition, snapshot: Artifact, reason: GraphReason = GraphReason.INGEST,
               key: Optional[str] = None, reanalysis_request_id: Optional[str] = None,
               reanalysis_claim_token: Optional[str] = None, agent_channel: Optional[int] = None,
               signals: Optional[SignalsInput] = None, priority_offset: int = 0,
               asr_vocabulary: Optional[AsrVocabularyParameters] = None) -> JobGraphRequest:
        key = key or f"ingest.{conversation_id}"
        jobs = _Jobs(key, priority_offset)
        self.add_media_stages(jobs, Src.pinned(source), audio, agent_channel=agent_channel, asr_vocabulary=asr_vocabulary)
        return JobGraphRequest(idempotency_key=bounded_key(key), reason=reason, reanalysis_request_id=reanalysis_request_id,
                               reanalysis_claim_token=reanalysis_claim_token,
                               jobs=self._finish(jobs, audio, rubric_ref, rubric, snapshot, signals))

    def _finish(self, jobs: _Jobs, audio: AudioInfo, rubric_ref: RubricVersionRef, rubric: RubricDefinition, snapshot: Artifact,
                signals: Optional[SignalsInput] = None) -> List[JobDefinition]:
        transcript = Src.upstream("asr", "transcript")
        speaker = Src.upstream("speaker", "speaker_attribution") if "speaker" in jobs.refs else None
        sources = self.add_analysis_stages(jobs, transcript, speaker, audio.duration_seconds, audio_src=_audio_src(jobs))
        self.add_qa(jobs, sources, rubric=rubric, rubric_ref=rubric_ref, snapshot=Src.pinned(snapshot), audio_seconds=audio.duration_seconds)
        if self.config.stages.summary:
            entry = self.pick(ModelPurpose.SUMMARY)
            masked = self.masked(entry, ModelPurpose.SUMMARY)
            jobs.add(SUMMARY_REF, JobType.SUMMARY_ASSEMBLY, inputs=[("transcript", transcript, False), ("pii_findings", sources.pii if masked else None, False)],
                     estimate=estimate_for(MemorySlot.CPU, audio.duration_seconds),
                     parameters=JobParameters(extra=_summary_extra(entry, masked)))
        if self.config.stages.contact_signals:
            self.add_contact_signals(jobs, sources, audio.duration_seconds, signals=signals)
        return jobs.defs

    # --- stage builders ----------------------------------------------------------------------

    def add_media_stages(self, jobs: _Jobs, audio_src: Src, audio: AudioInfo, *, agent_channel: Optional[int] = None,
                         asr_vocabulary: Optional[AsrVocabularyParameters] = None) -> None:
        """``agent_channel`` is the registered call's (``CallMetadata.agent_channel``): the stereo
        channel carrying the agent, which ASR labels AGENT and validation reports. ``asr_vocabulary``
        (``asr_vocabulary(record)``) turns on dual transcription inside the ``asr`` job."""
        seconds = audio.duration_seconds
        media: Dict[str, object] = {"agent_channel": agent_channel} if agent_channel is not None else {}
        jobs.add("vad", JobType.VALIDATION_VAD, inputs=[("audio", audio_src, False)],
                 estimate=estimate_for(MemorySlot.IO, seconds, uses_audio=True), parameters=JobParameters(extra=dict(media)))
        asr = self.pick(ModelPurpose.ASR)
        extra: Dict[str, object] = {"channels": audio.channels or 0, **media}
        if self.config.stages.summary:
            summary = self.pick(ModelPurpose.SUMMARY)
            extra.update({"summary_plan": SUMMARY_REF, "summary_batch_turns": self.config.summary_batch_turns,
                          "summary_entry": summary.entry_id, "summary_entry_version": summary.entry_version})
        estimate = estimate_for(self._slot(asr, JobType.ASR), seconds, entry=asr, uses_audio=True)
        if asr_vocabulary is not None and seconds is not None:
            # Parakeet then prompted Whisper Small in the one slot: about 3.5x Parakeet alone.
            estimate = estimate.model_copy(update={"estimated_runtime_seconds": round(seconds * PARAKEET_REAL_TIME_FACTOR * DUAL_ASR_RUNTIME_FACTOR, 3)})
        jobs.add("asr", JobType.ASR, inputs=[("audio", audio_src, False)], selection=self.selection(asr, ModelPurpose.ASR),
                 estimate=estimate, parameters=JobParameters(extra=extra, asr_vocabulary=asr_vocabulary))
        if audio.mono:
            diar = self.pick(ModelPurpose.SPEAKER_DIARIZATION)
            jobs.add("speaker", JobType.SPEAKER_ATTRIBUTION,
                     inputs=[("audio", audio_src, False), ("transcript", Src.upstream("asr", "transcript"), False)],
                     selection=self.selection(diar, ModelPurpose.SPEAKER_DIARIZATION),
                     estimate=estimate_for(self._slot(diar, JobType.SPEAKER_ATTRIBUTION), seconds, entry=diar, uses_audio=True))

    def add_analysis_stages(self, jobs: _Jobs, transcript: Src, speaker: Optional[Src], seconds: Optional[float], *,
                            audio_src: Optional[Src], ref_prefix: str = "", include_enrichment: bool = True,
                            include_embeddings: bool = True) -> QaSources:
        """Tone, text sentiment, enrichment and embeddings on a transcript (+ attribution)."""
        sources = QaSources(transcript=transcript, speaker=speaker)
        if audio_src is not None:
            tone = self.pick(ModelPurpose.ACOUSTIC_TONE)
            jobs.add(ref_prefix + "tone", JobType.ACOUSTIC_TONE,
                     inputs=[("audio", audio_src, False), ("transcript", transcript, False), ("speaker_attribution", speaker, False)],
                     selection=self.selection(tone, ModelPurpose.ACOUSTIC_TONE),
                     estimate=estimate_for(self._slot(tone, JobType.ACOUSTIC_TONE), seconds, entry=tone, uses_audio=True))
            sources.tone = Src.upstream(ref_prefix + "tone", "tone_blocks")
        sent = self.pick(ModelPurpose.TEXT_SENTIMENT)
        jobs.add(ref_prefix + "sentiment", JobType.TEXT_SENTIMENT,
                 inputs=[("transcript", transcript, False), ("speaker_attribution", speaker, False)],
                 selection=self.selection(sent, ModelPurpose.TEXT_SENTIMENT),
                 estimate=estimate_for(self._slot(sent, JobType.TEXT_SENTIMENT), seconds, entry=sent))
        sources.sentiment = Src.upstream(ref_prefix + "sentiment", "text_sentiment")
        if include_enrichment:
            # The attribution lets the PII findings recognize the agent's self-introduction.
            jobs.add(ref_prefix + "enrichment", JobType.ENRICHMENT, inputs=[("transcript", transcript, False), ("speaker_attribution", speaker, False)],
                     estimate=estimate_for(MemorySlot.CPU, seconds))
            sources.enrichment = Src.upstream(ref_prefix + "enrichment", "enrichment")
            sources.pii = Src.upstream(ref_prefix + "enrichment", "pii_findings")
        if include_embeddings and self.config.stages.embeddings:
            self.add_embeddings(jobs, transcript, speaker, seconds, ref_prefix=ref_prefix)
        return sources

    def add_embeddings(self, jobs: _Jobs, transcript: Src, speaker: Optional[Src], seconds: Optional[float], *, ref_prefix: str = "") -> None:
        """One ``embeddings`` job (the search vectors) on a transcript (+ attribution)."""
        emb = self.pick(ModelPurpose.EMBEDDINGS)
        jobs.add(ref_prefix + "embeddings", JobType.EMBEDDINGS,
                 inputs=[("transcript", transcript, False), ("speaker_attribution", speaker, False)],
                 selection=self.selection(emb, ModelPurpose.EMBEDDINGS),
                 estimate=estimate_for(self._slot(emb, JobType.EMBEDDINGS), seconds, entry=emb))

    def escalation_entry(self, criterion: RubricCriterion, primary: CatalogEntry) -> Optional[CatalogEntry]:
        """The escalation model the pre-split router would use: the criterion's
        ``escalation_model_id`` (``none`` disables), else this installation's default; never the
        primary itself. An unknown or unavailable entry means no escalation."""
        choice = criterion.check.escalation_model_id
        if choice == "none":
            return None
        entry_id = choice or self.config.escalation_entry_id
        if not entry_id:
            return None
        try:
            entry = self.catalog.get(entry_id)
        except CatalogError:
            return None
        if entry.entry_id == primary.entry_id or not self.catalog.usable(entry, ModelPurpose.SEMANTIC_QA):
            return None
        return entry

    def add_qa(self, jobs: _Jobs, sources: QaSources, *, rubric: RubricDefinition, snapshot: Src,
               rubric_ref: Optional[RubricVersionRef] = None, draft: Optional[DraftRubricRef] = None,
               draft_request_id: Optional[str] = None, audio_seconds: Optional[float] = None) -> None:
        if not rubric.criteria:
            raise PlanError(f"rubric {rubric.rubric_id} has no criteria")
        common_extra: Dict[str, object] = {"draft_test_request_id": draft_request_id} if draft_request_id else {}
        semantic = [c for c in rubric.criteria if c.check.check_type is CheckType.SEMANTIC_JUDGEMENT]
        deterministic = [c for c in rubric.criteria if c.check.check_type is not CheckType.SEMANTIC_JUDGEMENT]
        tokens = transcript_tokens(audio_seconds)
        scorecard_inputs: List[In] = [("rubric", snapshot, False)]
        for index, criterion in enumerate(semantic):
            primary = self.pick(ModelPurpose.SEMANTIC_QA, criterion.check.primary_model_id)
            escalation = self.escalation_entry(criterion, primary)
            pins = self.pins_enrichment(primary)
            ref = f"qa-{index}-{slug(criterion.criterion_id)}"
            extra = dict(common_extra, scorecard_ref=SCORECARD_REF, escalation_entry=escalation.entry_id if escalation else "",
                         escalation_entry_version=escalation.entry_version if escalation else 0)
            jobs.add(ref, JobType.QA_CRITERION,
                     inputs=[("transcript", sources.transcript, False), ("speaker_attribution", sources.speaker, False),
                             ("enrichment", sources.enrichment if pins else None, False),
                             ("pii_findings", sources.pii if self.masked(primary, ModelPurpose.SEMANTIC_QA) else None, False), ("rubric", snapshot, False)],
                     selection=self.selection(primary, ModelPurpose.SEMANTIC_QA),
                     estimate=estimate_for(self._slot(primary, JobType.QA_CRITERION), audio_seconds, entry=primary, tokens=tokens),
                     parameters=JobParameters(rubric=rubric_ref, draft_rubric=draft, criterion_id=criterion.criterion_id, extra=extra))
            scorecard_inputs.append((f"assessment:{criterion.criterion_id}", Src.upstream(ref, "assessment"), False))
        if deterministic:
            jobs.add("qa-det", JobType.QA_DETERMINISTIC,
                     inputs=[("transcript", sources.transcript, False), ("speaker_attribution", sources.speaker, False),
                             ("text_sentiment", sources.sentiment, False), ("tone_blocks", sources.tone, False), ("rubric", snapshot, False)],
                     estimate=estimate_for(MemorySlot.CPU, audio_seconds),
                     parameters=JobParameters(rubric=rubric_ref, draft_rubric=draft, extra=dict(common_extra)))
            scorecard_inputs.append(("verdicts", Src.upstream("qa-det", "verdicts"), False))
        jobs.add(SCORECARD_REF, JobType.QA_SCORECARD, inputs=scorecard_inputs, estimate=estimate_for(MemorySlot.CPU, audio_seconds),
                 parameters=JobParameters(rubric=rubric_ref, draft_rubric=draft, extra=dict(common_extra)))

    def add_summary_jobs(self, jobs: _Jobs, windows: Sequence[TurnWindow], transcript: Src, speaker: Optional[Src], entry: CatalogEntry,
                         audio_seconds: Optional[float] = None, pii: Optional[Src] = None) -> List[Tuple[str, str, str]]:
        """Segment jobs, then pairwise synthesis while more than two remain, then a final synthesis
        (the pre-split summarizer's reduction). Returns the assembly's bindings as
        (input role, ref, output role). Masked segments pin ``pii`` (the PII findings)."""
        masked = self.masked(entry, ModelPurpose.SUMMARY)
        bindings: List[Tuple[str, str, str]] = []
        level: List[Tuple[str, str]] = []
        for index, window in enumerate(windows):
            ref = f"sum-seg-{index}"
            jobs.add(ref, JobType.SUMMARY_SEGMENT,
                     inputs=[("transcript", transcript, False), ("speaker_attribution", speaker, False),
                             ("pii_findings", pii if masked else None, False)],
                     selection=self.selection(entry, ModelPurpose.SUMMARY, "summary_segment.v1"),
                     estimate=estimate_for(self._slot(entry, JobType.SUMMARY_SEGMENT), audio_seconds, entry=entry),
                     parameters=JobParameters(segment=SegmentSpec(index=index, window=window),
                                              extra={"masked": masked, "segment_count": len(windows)}))
            bindings.append((f"segment:{index}", ref, "segment"))
            level.append((ref, "segment"))
        if len(level) == 1:
            return bindings
        depth = 0
        while len(level) > 2:
            reduced: List[Tuple[str, str]] = []
            for k in range(0, len(level), 2):
                pair = level[k:k + 2]
                if len(pair) == 1:
                    reduced.append(pair[0])
                    continue
                ref = f"sum-syn-{depth}-{k // 2}"
                jobs.add(ref, JobType.SUMMARY_SYNTHESIS,
                         inputs=[(f"part:{i}", Src.upstream(r, role), False) for i, (r, role) in enumerate(pair)],
                         selection=self.selection(entry, ModelPurpose.SUMMARY, "summary_synthesis.v1"),
                         estimate=estimate_for(self._slot(entry, JobType.SUMMARY_SYNTHESIS), audio_seconds, entry=entry),
                         parameters=JobParameters(extra={"final": False, "level": depth}))
                reduced.append((ref, "synthesis"))
            level = reduced
            depth += 1
        jobs.add("sum-final", JobType.SUMMARY_SYNTHESIS,
                 inputs=[(f"part:{i}", Src.upstream(r, role), False) for i, (r, role) in enumerate(level)],
                 selection=self.selection(entry, ModelPurpose.SUMMARY, "summary_synthesis.v1"),
                 estimate=estimate_for(self._slot(entry, JobType.SUMMARY_SYNTHESIS), audio_seconds, entry=entry),
                 parameters=JobParameters(extra={"final": True, "level": depth}))
        bindings.append(("synthesis", "sum-final", "synthesis"))
        return bindings

    def add_contact_signals(self, jobs: _Jobs, sources: QaSources, audio_seconds: Optional[float], *,
                            signals: Optional[SignalsInput] = None) -> None:
        """The default semantic/Laya/Gemma process requires a v2 taxonomy snapshot.

        Its category and confirmation entries must be usable; no legacy execution or fallback
        is allowed. Historical/fake catalogs retain their older v1/shadow graph behavior for
        compatibility, without substituting a model within a purpose.
        """
        from .system_one import ENTRY_ID

        pipeline = signals.effective_pipeline if signals is not None else "v1"
        if self.catalog.defaults.get(ModelPurpose.SIGNAL_CATEGORY) == ENTRY_ID and pipeline != "v2":
            raise PlanError("The semantic/Laya/Gemma process requires a v2 taxonomy snapshot; legacy pipelines are read-only")
        if pipeline != "v2":
            self.add_signals_v1(jobs, sources, audio_seconds)
            return
        if self.signals_v2_available():
            self.add_signals_v2(jobs, sources, audio_seconds, signals)  # type: ignore[arg-type]
            return
        if self.catalog.defaults.get(ModelPurpose.SIGNAL_CATEGORY) == ENTRY_ID:
            raise PlanError("The default Contact Signals cascade needs qualified local Laya and Gemma entries; it cannot fall back to legacy analysis")
        if signals.require_v2:  # type: ignore[union-attr]
            raise PlanError("Contact Signals v2 needs a qualified signal classifier on this Process host")
        if signals.content.settings.v1_fallback:  # type: ignore[union-attr]
            self.add_signals_v1(jobs, sources, audio_seconds, note=V2_UNAVAILABLE_NOTE)
            return
        jobs.add("cs-merge", JobType.CONTACT_SIGNALS_MERGE,
                 inputs=[("transcript", sources.transcript, False), ("speaker_attribution", sources.speaker, False)],
                 estimate=estimate_for(MemorySlot.CPU, audio_seconds),
                 parameters=JobParameters(extra={"configuration_error": "Contact Signals v2 is selected, v1 fallback is off, and this host has no qualified signal classifier"}))

    def add_signals_v1(self, jobs: _Jobs, sources: QaSources, audio_seconds: Optional[float], *, note: Optional[str] = None) -> None:
        """Today's lifecycle and resolution passes and the merge, unchanged (``note``: the v1 result's
        ``pipeline_note``)."""
        entry = self.pick(ModelPurpose.CONTACT_SIGNALS)
        pins = self.pins_enrichment(entry)
        merge_inputs: List[In] = [("transcript", sources.transcript, False), ("speaker_attribution", sources.speaker, False)]
        for kind, job_type in PASS_JOB_TYPES.items():
            ref = f"cs-{kind.value}"
            jobs.add(ref, job_type,
                     inputs=[("transcript", sources.transcript, False), ("speaker_attribution", sources.speaker, False),
                             ("enrichment", sources.enrichment if pins else None, False),
                             ("pii_findings", sources.pii if self.masked(entry, ModelPurpose.CONTACT_SIGNALS) else None, False)],
                     selection=self.selection(entry, ModelPurpose.CONTACT_SIGNALS),
                     estimate=estimate_for(self._slot(entry, job_type), audio_seconds, entry=entry, tokens=transcript_tokens(audio_seconds)),
                     parameters=JobParameters(pass_kind=kind))
            merge_inputs.append((f"pass:{kind.value}:0", Src.upstream(ref, "pass"), True))
        jobs.add("cs-merge", JobType.CONTACT_SIGNALS_MERGE, inputs=merge_inputs, estimate=estimate_for(MemorySlot.CPU, audio_seconds),
                 parameters=JobParameters(extra={"pipeline_note": note}) if note else None)

    def signals_v2_available(self) -> bool:
        """v2 needs a usable entry for both classifier purposes. The included model (Gemma) serves both
        (decision 24), so a real install with its weights is v2-capable; only one usable purpose is not v2."""
        for purpose in (ModelPurpose.SIGNAL_CATEGORY, ModelPurpose.SIGNAL_SUBCATEGORY):
            try:
                self.catalog.select(purpose)
            except CatalogError:
                return False
        return True

    def stage2_factors(self, entry: CatalogEntry) -> Tuple[str, ...]:
        """The state factors the frozen stage-2 adapter version enables (section 4.2). Tone and
        sentiment are off by default, so stage 2 never waits on the tone model; a later adapter version
        that turns one on adds that ``after`` edge."""
        return DEFAULT_STAGE2_FACTORS

    def extraction_fallback(self, primary: CatalogEntry, settings) -> Optional[CatalogEntry]:
        """The declared in-job stage-3 fallback (section 5.7): the admin's ``fallback_extraction_entry_id``,
        else the ``signal_extraction`` default (Gemma), frozen only when it differs from the primary,
        is usable here and is on the primary's route class. Never a substitution: the job still runs
        its primary first."""
        entry_id = settings.fallback_extraction_entry_id or self.catalog.defaults.get(ModelPurpose.SIGNAL_EXTRACTION)
        if not entry_id:
            return None
        try:
            entry = self.catalog.get(entry_id)
        except CatalogError:
            return None
        if entry.entry_id == primary.entry_id or entry.route_class is not primary.route_class:
            return None
        if ModelPurpose.SIGNAL_EXTRACTION not in entry.purposes or not self.catalog.usable(entry, ModelPurpose.SIGNAL_EXTRACTION):
            return None
        return entry

    def add_signals_v2(self, jobs: _Jobs, sources: QaSources, audio_seconds: Optional[float], signals: SignalsInput) -> None:
        """The v2 cascade (section 8.1): categorize, subcategorize after it, extract after both (only
        when an active node has fields or narrow_quote), and the merge after all three, with
        ``pii_findings`` required on every job. A reanalysis ``plan`` runs only the stages with work
        and pins the previous artifact in place of a stage it does not rerun."""
        if sources.pii is None:
            raise PlanError("Contact Signals v2 reads masked text only, and this call has no PII findings for its transcript")
        content = signals.content
        taxonomy = content.taxonomy
        plan = signals.plan or full_plan(taxonomy)
        previous = signals.previous
        digest = content.taxonomy_ref.digest
        taxonomy_src = Src.pinned(signals.snapshot)
        extra: Dict[str, object] = {"draft_test_request_id": signals.draft_request_id} if signals.draft_request_id else {}

        def params(**kw) -> JobParameters:
            return JobParameters(signals=SignalJobParameters(taxonomy_digest=digest, preview_id=signals.preview_id, **kw), extra=dict(extra))

        def pinned(name: str) -> Optional[Src]:
            found = previous.get(name)
            return Src.pinned(found) if found is not None else None

        base: List[In] = [("transcript", sources.transcript, False), ("speaker_attribution", sources.speaker, False),
                          ("pii_findings", sources.pii, False), (SIGNAL_TAXONOMY_INPUT_ROLE, taxonomy_src, False)]
        pinned_stages: List[str] = []
        mode = plan.categorize
        if mode == "rederive" and previous.get("categories") is None:
            mode = "run"
        if mode is None and previous.get("categories") is None:
            mode = "run"
        if mode:
            entry = self.pick(ModelPurpose.SIGNAL_CATEGORY) if mode == "run" else None
            cat_extra = dict(extra)
            if mode == "run" and plan.scopes is not None:
                cat_extra["scopes"] = ",".join(plan.scopes)
            jobs.add("cs-categorize", JobType.CONTACT_SIGNALS_CATEGORIZE,
                     inputs=base + [("previous_categories", pinned("categories"), False)],
                     selection=self.selection(entry, ModelPurpose.SIGNAL_CATEGORY) if entry else None,
                     estimate=estimate_for(self._slot(entry, JobType.CONTACT_SIGNALS_CATEGORIZE), audio_seconds, entry=entry),
                     parameters=JobParameters(signals=SignalJobParameters(taxonomy_digest=digest, stage1_mode=mode, preview_id=signals.preview_id),
                                              extra=cat_extra))
            categories: Src = Src.upstream("cs-categorize", "categories")
        else:
            categories = pinned("categories")  # type: ignore[assignment]
            pinned_stages.append("categorize")
        cat_optional = categories.artifact is None

        sub = plan.subcategorize
        if sub is None and previous.get("subcategories") is None:
            sub = ALL
        if sub is not None:
            entry = self.pick(ModelPurpose.SIGNAL_SUBCATEGORY)
            factors = self.stage2_factors(entry)
            jobs.add("cs-subcategorize", JobType.CONTACT_SIGNALS_SUBCATEGORIZE,
                     inputs=base + [("categories", categories, cat_optional),
                                    ("tone_blocks", sources.tone if "tone" in factors else None, True),
                                    ("text_sentiment", sources.sentiment if "sentiment" in factors else None, True),
                                    ("previous_subcategories", pinned("subcategories"), False)],
                     selection=self.selection(entry, ModelPurpose.SIGNAL_SUBCATEGORY),
                     estimate=estimate_for(self._slot(entry, JobType.CONTACT_SIGNALS_SUBCATEGORIZE), audio_seconds, entry=entry),
                     parameters=params(span_keys=None if sub == ALL else list(sub)))  # type: ignore[arg-type]
            subcategories: Src = Src.upstream("cs-subcategorize", "subcategories")
        else:
            subcategories = pinned("subcategories")  # type: ignore[assignment]
            pinned_stages.append("subcategorize")

        extraction: Optional[Src] = None
        if plan.extract_planned:
            ext = plan.extract
            if ext is None and previous.get("extraction") is None:
                ext = ALL
            if ext is not None:
                entry = self.pick(ModelPurpose.SIGNAL_EXTRACTION)
                fallback = self.extraction_fallback(entry, content.settings)
                jobs.add("cs-extract", JobType.CONTACT_SIGNALS_EXTRACT,
                         inputs=base + [("categories", categories, cat_optional), ("subcategories", subcategories, subcategories.artifact is None),
                                        ("previous_extraction", pinned("extraction"), False)],
                         selection=self.selection(entry, ModelPurpose.SIGNAL_EXTRACTION),
                         estimate=estimate_for(self._slot(entry, JobType.CONTACT_SIGNALS_EXTRACT), audio_seconds, entry=entry),
                         parameters=params(span_keys=None if ext == ALL else list(ext),  # type: ignore[arg-type]
                                           fallback_entry_id=fallback.entry_id if fallback else None))
                extraction = Src.upstream("cs-extract", "extraction")
            else:
                extraction = pinned("extraction")
                pinned_stages.append("extract")
        planned = ["categorize", "subcategorize"] + (["extract"] if plan.extract_planned else [])
        merge_extra = dict(extra, planned_stages=",".join(planned))
        if pinned_stages:
            merge_extra["pinned_stages"] = ",".join(pinned_stages)
        merge_inputs: List[In] = base + [("stage:categorize", categories, cat_optional),
                                         ("stage:subcategorize", subcategories, subcategories.artifact is None)]
        if extraction is not None:
            merge_inputs.append(("stage:extract", extraction, extraction.artifact is None))
        jobs.add("cs-merge", JobType.CONTACT_SIGNALS_MERGE, inputs=merge_inputs, estimate=estimate_for(MemorySlot.CPU, audio_seconds),
                 parameters=JobParameters(signals=SignalJobParameters(taxonomy_digest=digest, preview_id=signals.preview_id), extra=merge_extra))

    # --- reanalysis graphs -------------------------------------------------------------------

    def qa_graph(self, jobs: _Jobs, sources: QaSources, *, rubric: RubricDefinition, snapshot: Src,
                 rubric_ref: Optional[RubricVersionRef] = None, draft: Optional[DraftRubricRef] = None,
                 draft_request_id: Optional[str] = None, audio_seconds: Optional[float] = None) -> None:
        self.add_qa(jobs, sources, rubric=rubric, snapshot=snapshot, rubric_ref=rubric_ref, draft=draft,
                    draft_request_id=draft_request_id, audio_seconds=audio_seconds)

    def summary_graph(self, jobs: _Jobs, transcript_content: TranscriptContent, transcript: Src, speaker: Optional[Src],
                      audio_seconds: Optional[float] = None, pii: Optional[Src] = None) -> None:
        entry = self.pick(ModelPurpose.SUMMARY)
        masked = self.masked(entry, ModelPurpose.SUMMARY)
        windows = plan_segments(transcript_content, self.config.summary_batch_turns)
        bindings = self.add_summary_jobs(jobs, windows, transcript, speaker, entry, audio_seconds, pii=pii)
        inputs: List[In] = [("transcript", transcript, False), ("pii_findings", pii if masked else None, False)]
        inputs += [(role, Src.upstream(ref, output_role), False) for role, ref, output_role in bindings]
        jobs.add(SUMMARY_REF, JobType.SUMMARY_ASSEMBLY, inputs=inputs, estimate=estimate_for(MemorySlot.CPU, audio_seconds),
                 parameters=JobParameters(extra=_summary_extra(entry, masked)))

    def speaker_correction_jobs(self, jobs: _Jobs, correction: SpeakerCorrection, transcript: Src, current: Optional[Src],
                                audio_seconds: Optional[float]) -> Src:
        """The code-stage ``speaker_attribution`` job applying a reviewer's correction."""
        jobs.add("speaker", JobType.SPEAKER_ATTRIBUTION,
                 inputs=[("transcript", transcript, False), ("speaker_attribution", current, False)],
                 estimate=estimate_for(MemorySlot.CPU, audio_seconds), parameters=JobParameters(speaker_correction=correction))
        return Src.upstream("speaker", "speaker_attribution")

    # --- follow-on jobs at completion ---------------------------------------------------------

    def summary_follow_on(self, asr_job_id: str, extra: Dict[str, object], transcript: TranscriptContent, graph: JobGraph,
                          audio_seconds: Optional[float] = None) -> Optional[FollowOnJobs]:
        """At ASR completion: the segment and synthesis jobs of the graph's summary assembly, whose
        windows need the transcript, bound to the assembly (``segment:<n>``, ``synthesis``)."""
        target = extra.get("summary_plan")
        if not target:
            return None
        assembly = next((j for j in graph.jobs if j.ref == target and j.job_type is JobType.SUMMARY_ASSEMBLY), None)
        if assembly is None or assembly.status is not JobStatus.BLOCKED:
            return None
        try:
            entry = self.catalog.get(str(extra.get("summary_entry") or ""))
        except CatalogError:
            raise PlanError("the summary model this graph froze is no longer in the catalog") from None
        speaker_job = next((j for j in graph.jobs if j.ref == "speaker" and j.job_type is JobType.SPEAKER_ATTRIBUTION), None)
        enrichment_job = next((j for j in graph.jobs if j.ref == "enrichment" and j.job_type is JobType.ENRICHMENT), None)
        batch = int(extra.get("summary_batch_turns") or self.config.summary_batch_turns)
        windows = plan_segments(transcript, batch)
        jobs = _Jobs(f"sum.{asr_job_id}")
        bindings = self.add_summary_jobs(jobs, windows, Src.job(asr_job_id, "transcript"),
                                         Src.job(speaker_job.job_id, "speaker_attribution") if speaker_job else None, entry, audio_seconds,
                                         pii=Src.job(enrichment_job.job_id, "pii_findings") if enrichment_job else None)
        return FollowOnJobs(jobs=jobs.defs, add_dependencies=[
            NewDependency(dependent_job_id=assembly.job_id, requires_ref=ref, input_role=role, output_role=output_role)
            for role, ref, output_role in bindings])

    def escalation_follow_on(self, *, job_id: str, parameters: JobParameters, pinned_inputs: Sequence[Tuple[str, ArtifactRef]],
                             trigger: EscalationTrigger, scorecard_job_id: str, audio_seconds: Optional[float] = None) -> Optional[FollowOnJobs]:
        """At a ``qa_criterion`` completion whose trigger fired: the ``qa_escalation`` job (same
        pinned inputs, the escalation model frozen when the graph was built) and the scorecard's new
        dependency, bound to its ``assessment`` as ``escalation:<criterion>``."""
        entry_id = str(parameters.extra.get("escalation_entry") or "")
        if not entry_id or parameters.criterion_id is None:
            return None
        try:
            entry = self.catalog.get(entry_id)
        except CatalogError:
            return None
        if entry.entry_version != int(parameters.extra.get("escalation_entry_version") or entry.entry_version):
            return None
        if not self.catalog.usable(entry, ModelPurpose.SEMANTIC_QA):
            return None
        extra = {k: v for k, v in parameters.extra.items() if k == "draft_test_request_id"}
        definition = JobDefinition(
            ref=ESCALATION_REF, job_type=JobType.QA_ESCALATION, idempotency_key=bounded_key(f"esc.{job_id}"),
            priority=PRIORITY[JobType.QA_ESCALATION],
            inputs=[JobInput(role=role, artifact=ref) for role, ref in pinned_inputs],
            selection=self.selection(entry, ModelPurpose.SEMANTIC_QA),
            resource_estimate=estimate_for(self._slot(entry, JobType.QA_ESCALATION), audio_seconds, entry=entry,
                                           tokens=transcript_tokens(audio_seconds)),
            parameters=JobParameters(rubric=parameters.rubric, draft_rubric=parameters.draft_rubric, criterion_id=parameters.criterion_id,
                                     escalation_trigger=trigger.value, extra=extra),
            requires_job_ids=[job_id],
        )
        return FollowOnJobs(jobs=[definition], add_dependencies=[NewDependency(
            dependent_job_id=scorecard_job_id, requires_ref=ESCALATION_REF, input_role=f"escalation:{parameters.criterion_id}",
            output_role="assessment")])


def _summary_extra(entry: CatalogEntry, masked: bool) -> Dict[str, object]:
    """The assembly is a code stage with no selection, so it carries the summary route's masking
    flag itself (it checks citations against the text the segments' model saw)."""
    return {"summary_entry": entry.entry_id, "summary_entry_version": entry.entry_version, "route_class": entry.route_class.value,
            "masked": masked}


def _audio_src(jobs: _Jobs) -> Optional[Src]:
    vad = jobs.refs.get("vad")
    if vad is None:
        return None
    audio = next((i for i in vad.inputs if i.role == "audio"), None)
    return Src(artifact=audio.artifact) if audio is not None and audio.artifact is not None else None


def new_jobs(key_prefix: str, priority_offset: int = 0) -> _Jobs:
    return _Jobs(key_prefix, priority_offset)


__all__ = ["GraphPlanner", "PlanError", "Src", "QaSources", "SignalsInput", "new_jobs", "bounded_key", "SCORECARD_REF", "SUMMARY_REF",
           "V2_UNAVAILABLE_NOTE"]
