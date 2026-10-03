"""Conversations, calls, result groups and the read projections Evaluate renders.

A conversation is the ingestion unit (an audio call or a text import). A call conversation has a
``call_id`` and Store projects each linked publication into the call read models below.
Projections are derived from committed, linked artifacts (their content models are in
``contents.py``); Process never writes them directly. Where a view is one artifact, it is that
artifact's content model plus Store's identifiers; ``TranscriptView`` composes several.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Optional, Tuple

from pydantic import Field, StringConstraints, model_validator

from .common import ChangeCursor, ContractModel, PageQuery, ResourceId, SafeText, Sha256Digest, ShortText, Timestamp
from .contents import (  # noqa: F401  (re-exported vocabulary)
    AudioChannelLayout,
    ContactSignalKind,
    ContactSignalsContent,
    ContactSignalView,
    ModelAttemptView,
    QaScorecardContent,
    ResultKind,
    ResultState,
    RubricHighlightView,
    SpeakerRole,
    SummaryContent,
    TextSentimentLabel,
    ToneBlockStatus,
    ToneBlockView,
    VadMetricsView,
    VerdictStatus,
    VerdictView,
    WordTimestampView,
)
from .contents import SignalNodeId, VocabularyCorrectionStatus
from .vocabulary import VocabularyTermSource
from .errors import JobErrorCode
from .signals import SignalAlertMatch, SignalHitFeedback, SignalTaxonomyStatus


# --- Registration -------------------------------------------------------------------------


class IngestionKind(str, Enum):
    CALL_AUDIO = "call_audio"
    TEXT_IMPORT = "text_import"


class SourceKind(str, Enum):
    S3_EVENT = "s3_event"
    LOCAL_IMPORT = "local_import"
    API_UPLOAD = "api_upload"
    MIGRATION = "migration"


class SourceReference(ContractModel):
    """Where the source came from. Its dedup identity makes registration idempotent: duplicate S3
    notifications map to one conversation (the existing ``(bucket, key, etag)`` identity)."""

    kind: SourceKind
    bucket: Optional[ShortText] = None
    object_key: Optional[str] = Field(default=None, max_length=1024, description="The recorder's object key in the customer's bucket. An external identity, not a Store location.")
    etag: Optional[ShortText] = None
    content_digest: Optional[Sha256Digest] = Field(default=None, description="SHA-256 of the source bytes; required for imports and uploads.")
    received_at: Timestamp
    legacy_ingest_job_id: Optional[int] = Field(default=None, description="Set by the migration importer for rows from the pre-split ingest_jobs table.")

    @property
    def dedup_identity(self) -> str:
        if self.kind is SourceKind.S3_EVENT:
            return f"s3:{self.bucket}/{self.object_key}#{self.etag}"
        return f"{self.kind.value}:{self.content_digest}"

    @model_validator(mode="after")
    def _identity_fields(self):
        if self.kind is SourceKind.S3_EVENT and not (self.bucket and self.object_key and self.etag):
            raise ValueError("an S3 event source is identified by bucket, object key and etag")
        if self.kind is not SourceKind.S3_EVENT and not self.content_digest:
            raise ValueError("imports, uploads and migrated sources are identified by content digest")
        return self


AGENT_DISPLAY_NAME_MAX_LENGTH = 100
AGENT_EXTENSION_MAX_LENGTH = 20

AgentDisplayName = Annotated[str, StringConstraints(min_length=1, max_length=AGENT_DISPLAY_NAME_MAX_LENGTH, pattern=r"^\S(?:.*\S)?$")]
"""The agent's name as people know it (``Samantha Reyes``). Trimmed: no leading or trailing
whitespace. It is the agent's work identity, never the caller's."""

AgentExtension = Annotated[str, StringConstraints(pattern=r"^[0-9A-Za-z*#+._-]{1,20}$")]
"""The agent's PBX or phone extension (``104``, ``4410#2``): 1-20 dial-string characters, no spaces."""


class CallMetadata(ContractModel):
    """What the recorder or importer knows about a call. Added in 1.1.0: ``agent_display_name``
    and ``agent_extension``. ``agent_id`` stays the stable identifier that filters, metrics and
    queue rules key on; the two new fields are for people. Clients label the agent with
    ``agent_label()``: ``display_name (extension)`` when a display name is present, otherwise
    ``agent_id``, with `` (extension)`` appended whenever an extension is present.

    Re-registering a call's source identity with different metadata updates it (see
    ``merge_call_metadata`` and the README's "Idempotency" section)."""

    agent_id: ShortText = Field(default="Unknown", description="Stable agent identifier from the recorder, PBX or importer; filters, metrics and queue rules key on it.")
    agent_display_name: Optional[AgentDisplayName] = Field(default=None, description="Added in 1.1.0. The agent's name for people; clients show 'display_name (extension)' when present, else agent_id (agent_label).")
    agent_extension: Optional[AgentExtension] = Field(default=None, description="Added in 1.1.0. The agent's PBX or phone extension (1-20 dial-string characters).")
    agent_channel: Optional[int] = Field(default=None, ge=0, le=1, description="Channel carrying the agent on a stereo recording.")
    recorded_at: Optional[Timestamp] = None
    external_call_ref: Optional[ShortText] = Field(default=None, description="The recorder's or PBX's call identifier.")
    caller_reference: Optional[ShortText] = Field(default=None, description="An already-masked caller identifier. Never a raw phone number.")


def agent_label(agent_id: str, agent_display_name: Optional[str] = None, agent_extension: Optional[str] = None) -> str:
    """The normative agent label every client shows (Evaluate, both consoles, exports).

    ``display_name (extension)`` when both are present, ``display_name`` alone without an
    extension, and ``agent_id`` (plus `` (extension)`` when there is one) without a display name.
    Evaluate implements the same rule in TypeScript; it never parses ``agent_id``."""
    base = agent_display_name or agent_id
    return f"{base} ({agent_extension})" if agent_extension else base


def merge_call_metadata(stored: Optional[CallMetadata], incoming: CallMetadata) -> Tuple[CallMetadata, List[str]]:
    """The normative re-registration rule (1.1.0). Returns the metadata Store keeps and the
    names of the fields that changed, in field order; an empty list means nothing changed.

    Only the fields the incoming registration **sets** (present in the request body, including
    an explicit ``null``) replace the stored ones; absent fields keep their stored values. So a
    re-upload that names no agent never resets a known agent to ``Unknown``, and a client clears
    an optional field by sending ``null`` for it. With nothing stored, the incoming metadata is
    kept as sent (its defaults included)."""
    if stored is None:
        return incoming, [name for name in CallMetadata.model_fields if getattr(incoming, name) is not None]
    merged = stored.model_copy(update={name: getattr(incoming, name) for name in incoming.model_fields_set})
    merged = CallMetadata.model_validate(merged.model_dump())
    changed = [name for name in CallMetadata.model_fields if getattr(merged, name) != getattr(stored, name)]
    return (merged if changed else stored), changed


class ConversationRegistration(ContractModel):
    ingestion_kind: IngestionKind
    source: SourceReference
    call_metadata: Optional[CallMetadata] = None

    @model_validator(mode="after")
    def _call_needs_metadata(self):
        if self.ingestion_kind is IngestionKind.CALL_AUDIO and self.call_metadata is None:
            raise ValueError("a call registration carries call metadata")
        return self


class Conversation(ContractModel):
    id: ResourceId
    ingestion_kind: IngestionKind
    source: SourceReference
    call_id: Optional[ResourceId] = Field(default=None, description="Present for call conversations; unique across Store.")
    call_metadata: Optional[CallMetadata] = None
    created_at: Timestamp
    registered_by_installation_id: Optional[ResourceId] = None


class ConversationRegistered(ContractModel):
    conversation: Conversation = Field(description="The conversation as it stands after this registration, with any metadata update applied.")
    created: bool = Field(description="False when the source identity was already registered; the existing conversation is returned.")
    metadata_updated: bool = Field(default=False, description="Added in 1.1.0. True when the source identity was already registered and this registration changed its call_metadata (merge_call_metadata). Store wrote a call_metadata_updated audit event and a change event in the same transaction; nothing was reprocessed. Always false when created is true.")
    updated_fields: List[str] = Field(default_factory=list, description="Added in 1.1.0. The CallMetadata field names that changed, in field order; empty unless metadata_updated.")

    @model_validator(mode="after")
    def _metadata_update(self):
        if self.metadata_updated and self.created:
            raise ValueError("a newly created conversation has no metadata update")
        if self.metadata_updated != bool(self.updated_fields):
            raise ValueError("updated_fields names the changed fields exactly when metadata_updated")
        unknown = set(self.updated_fields) - set(CallMetadata.model_fields)
        if unknown:
            raise ValueError(f"updated_fields names only CallMetadata fields, not {sorted(unknown)}")
        return self


# --- Result groups and their derived state ------------------------------------------------


class PublisherState(str, Enum):
    """Where the publishing job of a result group stands in one graph."""

    IN_PROGRESS = "in_progress"
    """BLOCKED behind upstream work that can still succeed, QUEUED or RUNNING."""
    SUCCEEDED = "succeeded"
    ENDED_WITHOUT_RESULT = "ended_without_result"
    """FAILED or CANCELLED, or dead-blocked: BLOCKED on a success edge whose upstream is FAILED or
    CANCELLED (transitively), so it cannot run until someone retries that upstream."""


class ResultStateInputs(ContractModel):
    """Everything ``derive_result_state`` looks at for one group of one conversation. Store computes
    it with ``result_state_inputs``, which ignores draft-test graphs entirely."""

    has_work: bool = Field(description="Some graph for this conversation, other than a draft-test graph, contains a job of this group, or a version was once published.")
    published: Optional[ResultState] = Field(default=None, description="State of the latest linked version: available or partial; null when none was ever published.")
    newest_publisher: Optional[PublisherState] = Field(default=None, description="The publishing job of the newest graph, other than a draft-test graph, that contains one for this group.")
    newest_publisher_is_newer_than_published: bool = Field(default=False, description="True when that graph was created after the published version's graph.")
    reanalysis_pending: bool = Field(default=False, description="A reanalysis request whose kind affects this group (REANALYSIS_KIND_AFFECTS) is pending or claimed.")

    @model_validator(mode="after")
    def _published(self):
        if self.published not in (None, ResultState.AVAILABLE, ResultState.PARTIAL):
            raise ValueError("a published version is available or partial")
        return self


def derive_result_state(inputs: ResultStateInputs) -> ResultState:
    """The normative result-group rule. Store evaluates it; Evaluate and Process only render it.

    1. Nothing was ever requested or published for the group: ``disabled``.
    2. A version is published: ``stale`` while a reanalysis affecting the group is pending or its
       newer graph's publisher is in progress; otherwise that version's state (``available`` or
       ``partial``). A newer graph whose publisher ended without a result leaves the published
       version shown, with ``ResultGroup.failure_code`` set.
    3. Nothing published: ``failed`` once the newest publisher ended without a result
       ('Needs attention'), otherwise ``pending`` ('Analyzing').
    """
    if not inputs.has_work and inputs.published is None and not inputs.reanalysis_pending:
        return ResultState.DISABLED
    if inputs.published is not None:
        underway = inputs.reanalysis_pending or (
            inputs.newest_publisher_is_newer_than_published and inputs.newest_publisher is PublisherState.IN_PROGRESS)
        return ResultState.STALE if underway else inputs.published
    if inputs.newest_publisher is PublisherState.ENDED_WITHOUT_RESULT and not inputs.reanalysis_pending:
        return ResultState.FAILED
    return ResultState.PENDING


class GroupGraphStake(ContractModel):
    """What one job graph of a conversation contributes to one of its result groups."""

    graph_id: ResourceId
    graph_created_at: Timestamp
    draft_test: bool = Field(description="The graph fulfils a draft-test request (jobs.DRAFT_TEST_KINDS: qa_draft_test, or contact_signals_preview since 1.3.0). Its jobs never count toward the call's result groups, pending_work or settled state.")
    has_group_jobs: bool = Field(description="The graph contains at least one job whose JobTypeRule.group is this group.")
    publisher: Optional[PublisherState] = Field(default=None, description="Where this graph's publishing job for the group stands; null when the graph has none.")


def result_state_inputs(
    stakes: List[GroupGraphStake],
    published: Optional[ResultState],
    published_graph_created_at: Optional[Timestamp],
    reanalysis_pending: bool,
) -> ResultStateInputs:
    """The normative way Store builds ``ResultStateInputs`` for one group of one conversation.

    ``published`` is the state of the group's latest linked version and
    ``published_graph_created_at`` the creation time of the graph that produced it (null for a
    migrated version, which has no graph). Draft-test graphs are dropped first, so a draft test
    that runs, fails or succeeds never changes the call's state or ``failure_code``.
    """
    live = [s for s in stakes if not s.draft_test]
    publishers = [s for s in live if s.publisher is not None]
    newest = max(publishers, key=lambda s: s.graph_created_at, default=None)
    newer = newest is not None and published is not None and (
        published_graph_created_at is None or newest.graph_created_at > published_graph_created_at)
    return ResultStateInputs(
        has_work=any(s.has_group_jobs for s in live) or published is not None,
        published=published,
        newest_publisher=newest.publisher if newest is not None else None,
        newest_publisher_is_newer_than_published=newer,
        reanalysis_pending=reanalysis_pending,
    )


SETTLED_RESULT_STATES = frozenset({ResultState.AVAILABLE, ResultState.PARTIAL, ResultState.FAILED, ResultState.DISABLED})
"""A conversation is settled (the plan's 'persisted terminal state') when every job is SUCCEEDED,
FAILED, CANCELLED or dead-blocked; then every result group is in one of these states."""


class ResultGroup(ContractModel):
    kind: ResultKind
    state: ResultState
    version: Optional[int] = Field(default=None, ge=1)
    artifact_id: Optional[ResourceId] = None
    produced_by_graph_id: Optional[ResourceId] = None
    committed_at: Optional[Timestamp] = None
    partial_reason: Optional[ShortText] = None
    failure_code: Optional[JobErrorCode] = Field(default=None, description="Set when the newest publisher (never a draft-test graph's) ended without a result: the publisher's error code, or the first failed upstream's.")
    reanalysis_request_id: Optional[ResourceId] = Field(default=None, description="Set while a reanalysis affecting this group is underway.")


class PendingWorkIndicator(ContractModel):
    """A safe indicator Evaluate can read without reaching Process: counts only, no job details.
    Jobs of draft-test graphs are not counted (their progress is ``DraftTestResult.state``)."""

    jobs_total: int = Field(ge=0)
    jobs_succeeded: int = Field(ge=0)
    jobs_running: int = Field(ge=0)
    jobs_queued: int = Field(ge=0)
    jobs_blocked: int = Field(ge=0)
    jobs_failed: int = Field(ge=0)
    jobs_cancelled: int = Field(ge=0)
    settled: bool = Field(description="Every job is terminal or dead-blocked (see SETTLED_RESULT_STATES).")


# --- Read projections ---------------------------------------------------------------------


class TranscriptTurnView(ContractModel):
    turn_id: int = Field(ge=0)
    speaker: SpeakerRole
    speaker_cluster: Optional[str] = None
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    text: str = Field(description="Masked according to the Store masking setting for reviewer reads.")
    channel: Optional[int] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    word_timestamps: Optional[List[WordTimestampView]] = None
    text_sentiment: Optional[float] = None
    text_sentiment_label: Optional[TextSentimentLabel] = None
    sentiment_divergence: Optional[float] = None


class TranscriptReplacementView(ContractModel):
    """Added in 1.3.0 (decision 33). One vocabulary correction as a reviewer sees it. ``char_start``
    and ``char_end`` are offsets into the view's (masked) ``TranscriptTurnView.text``, null when Store
    could not locate the words there; ``word_start``/``word_end`` index the view's word_timestamps.
    ``heard`` is the base engine's words after the call's masking, and null whenever masking changed
    them or they hold a digit (Evaluate then says the original words are masked)."""

    turn_id: int = Field(ge=0)
    word_start: int = Field(ge=0)
    word_end: int = Field(ge=1)
    char_start: Optional[int] = Field(default=None, ge=0)
    char_end: Optional[int] = Field(default=None, ge=1)
    term: ShortText
    source: VocabularyTermSource
    heard: Optional[str] = Field(default=None, max_length=400)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)

    @model_validator(mode="after")
    def _offsets(self):
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end are both set or both null")
        if self.char_start is not None and self.char_end <= self.char_start:
            raise ValueError("a located replacement covers at least one character")
        if self.word_end <= self.word_start:
            raise ValueError("a replacement covers at least one word")
        return self


class VocabularyCorrectionView(ContractModel):
    """Added in 1.3.0 (decision 33). The transcript's vocabulary correction for reviewers: status,
    the safe note, and the replacements Store may show. A replacement whose term overlaps a masked
    span of the view is left out and counted in ``withheld_count``; while the text is withheld every
    replacement is."""

    status: VocabularyCorrectionStatus
    note: Optional[SafeText] = None
    replacement_count: int = Field(ge=0, description="Replacements in the transcript artifact, shown or not.")
    withheld_count: int = Field(default=0, ge=0)
    replacements: List[TranscriptReplacementView] = Field(default_factory=list)


class TranscriptView(ContractModel):
    """Composed by Store from the linked artifacts of the call: ``transcript`` (turns, timing,
    text, masked for reviewer reads), the newest ``speaker_attribution`` (speaker labels),
    ``text_sentiment`` (per-turn score and label), ``tone_blocks``, ``vad_metrics``, and
    ``sentiment_divergence`` computed from tone and sentiment. ``artifact_id`` and ``version`` are
    the transcript artifact's."""

    call_id: ResourceId
    artifact_id: ResourceId
    version: int = Field(ge=1)
    speaker_attribution_version: Optional[int] = Field(default=None, ge=1)
    is_redacted: bool
    text_withheld: bool = Field(default=False, description="Added in 1.2.0. True while reviewer reads are masked and the PII findings (pii_findings) for this transcript revision are not linked yet, or could not be made: every turn's text is empty and word_timestamps are null (fail closed). The transcript result group reads 'partial' with a partial_reason meanwhile.")
    duration_seconds: float = Field(ge=0)
    turns: List[TranscriptTurnView]
    tone_blocks: List[ToneBlockView] = Field(default_factory=list)
    vad_metrics: Optional[VadMetricsView] = None
    avg_agent_tone: Optional[float] = None
    avg_caller_sentiment: Optional[float] = None
    vocabulary_correction: Optional[VocabularyCorrectionView] = Field(default=None, description="Added in 1.3.0 (decision 33): present when the transcript was made with dual transcription.")


class EvaluationView(QaScorecardContent):
    """The automated scorecard for one result version: the ``qa_scorecard`` content plus Store's
    identifiers. Human decisions live in reviews.py and reference ``version`` here; they never
    rewrite this record."""

    call_id: ResourceId
    artifact_id: ResourceId
    version: int = Field(ge=1)


class SummaryView(SummaryContent):
    """The ``summary`` content plus Store's identifiers."""

    call_id: ResourceId
    artifact_id: ResourceId
    version: int = Field(ge=1)


class ContactSignalsView(ContactSignalsContent):
    """The ``contact_signals`` content plus Store's identifiers; ``completeness`` is partial when a
    pass (v1) or stage (v2) failed, and the result group's state is then ``partial``. Store masks
    quotes, field values, surface text and evidence again on every read.

    Since 1.3.0 Store adds read-time context: the result's standing against the current taxonomy,
    the reviewers' hit feedback, the enabled alert rules that match it now, and for admins the
    v2 compare result of a v1 call. None of it is part of the artifact."""

    call_id: ResourceId
    artifact_id: ResourceId
    version: int = Field(ge=1)
    taxonomy_status: Optional[SignalTaxonomyStatus] = Field(default=None, description="Added in 1.3.0. signals.signal_taxonomy_status against the current taxonomy ('Scored with taxonomy v3 (current v5)').")
    feedback: List[SignalHitFeedback] = Field(default_factory=list, description="Added in 1.3.0. Hit feedback on this result's hits. A merged multi-segment hit (decision 25) also lists feedback saved on a part's own earlier hit ID (the anchor ID with the part's t<turn>b<block>): the judgement of an earlier segmentation, shown but not the anchor's own verdict.")
    alerts: List[SignalAlertMatch] = Field(default_factory=list, description="Added in 1.3.0. Enabled alert rules matching this result now (computed at read time).")
    text_withheld: bool = Field(default=False, description="Added in 1.3.0. The 1.2.0 fail-closed rule: while the PII findings for the current transcript revision are missing, quotes and field text read [REDACTED].")
    comparison_preview_id: Optional[ResourceId] = Field(default=None, description="Added in 1.3.0; admins only. A v2 compare result exists for this call (getSignalPreview).")


# --- Call list and detail -----------------------------------------------------------------


class CallRecordView(ContractModel):
    """Media fields come from the ``validation_report`` and ``vad_metrics`` artifacts and stay null
    until ``validation_vad`` publishes them, so an in-progress call is readable ('Analyzing')."""

    call_id: ResourceId
    conversation_id: ResourceId
    agent_id: ShortText
    agent_display_name: Optional[AgentDisplayName] = Field(default=None, description="Added in 1.1.0; from the call's current CallMetadata. Label with agent_label().")
    agent_extension: Optional[AgentExtension] = Field(default=None, description="Added in 1.1.0; from the call's current CallMetadata.")
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    channels: Optional[int] = Field(default=None, ge=1)
    channel_layout: Optional[AudioChannelLayout] = None
    sample_rate: Optional[int] = Field(default=None, ge=1)
    codec: Optional[ShortText] = None
    silence_ratio: Optional[float] = Field(default=None, ge=0, le=1)
    overtalk_duration: Optional[float] = Field(default=None, ge=0)
    avg_agent_tone: Optional[float] = None
    avg_caller_sentiment: Optional[float] = None
    created_at: Timestamp
    recorded_at: Optional[Timestamp] = None
    legacy_call: bool = Field(default=False, description="True for calls migrated from the pre-split database; their results have no job graph.")


class CallListItem(ContractModel):
    call_id: ResourceId
    conversation_id: ResourceId
    agent_id: ShortText
    agent_display_name: Optional[AgentDisplayName] = Field(default=None, description="Added in 1.1.0; from the call's current CallMetadata. Label with agent_label().")
    agent_extension: Optional[AgentExtension] = Field(default=None, description="Added in 1.1.0; from the call's current CallMetadata.")
    duration_seconds: Optional[float] = Field(default=None, ge=0, description="Null until validation publishes.")
    channel_layout: Optional[AudioChannelLayout] = None
    created_at: Timestamp
    transcript_state: ResultState
    qa_state: ResultState
    summary_state: ResultState
    rubric_id: Optional[ShortText] = None
    overall_score: Optional[float] = Field(default=None, description="Absent until scorecard assembly succeeds; Evaluate shows 'Analyzing' rather than a blank badge.")
    passed: Optional[bool] = None
    critical_failure: Optional[bool] = None
    requires_human_review: bool = False
    review_status: Optional[ShortText] = Field(default=None, description="Escalation status from the review record, when one exists.")
    review_version: int = Field(ge=0)
    contact_signals_state: ResultState = Field(default=ResultState.DISABLED, description="Added in 1.3.0. The contact_signals group's derived state; Store always sets it (Evaluate shows 'Analyzing' or 'Refreshing' beside the signal chips).")
    signal_categories: List[SignalNodeId] = Field(default_factory=list, description="Added in 1.3.0. Category IDs with a current hit, on active nodes.")
    caller_needs: List[SignalNodeId] = Field(default_factory=list, description="Added in 1.3.0. Subcategory IDs (or 'other') of the current intent hits, on active nodes.")
    signal_alerts: List[SignalNodeId] = Field(default_factory=list, description="Added in 1.3.0. Enabled alert rules matching the call's current signals.")


class CallListQuery(PageQuery):
    agent_id: Optional[ShortText] = None
    needs_review: Optional[bool] = None
    rubric_id: Optional[ShortText] = None
    created_after: Optional[Timestamp] = None
    created_before: Optional[Timestamp] = None
    text: Optional[str] = Field(default=None, max_length=200, description="Simple text search over agent ID, agent display name, agent extension (1.1.0) and external call reference; not transcript search.")
    signal_category: Optional[SignalNodeId] = Field(default=None, description="Added in 1.3.0. Calls with a current hit of this category.")
    signal_subcategory: Optional[SignalNodeId] = Field(default=None, description="Added in 1.3.0. Calls with a current hit of this subcategory (or 'other'); combine with signal_category.")
    signal_alert: Optional[SignalNodeId] = Field(default=None, description="Added in 1.3.0. Calls an enabled alert rule matches now.")


class CallDetail(ContractModel):
    call: CallRecordView
    results: List[ResultGroup]
    pending_work: PendingWorkIndicator
    review_version: int = Field(ge=0, description="Expected-version token for review writes on this call.")
    evaluation: Optional[EvaluationView] = None
    change_cursor: ChangeCursor = Field(description="Position at which this snapshot was read; pass to the change feed to follow updates.")


class SemanticSearchQuery(ContractModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    speaker_filter: Optional[SpeakerRole] = None
    call_id: Optional[ResourceId] = None
    min_score: float = Field(default=0.0, ge=0, le=1)


class SemanticSearchHit(ContractModel):
    call_id: ResourceId
    turn_id: int
    speaker: SpeakerRole
    start_time: float
    end_time: float
    text: str
    similarity_score: float = Field(ge=0, le=1)


class SemanticSearchResponse(ContractModel):
    query: str
    count: int = Field(ge=0)
    results: List[SemanticSearchHit]
    embedding_scheme: Optional[ShortText] = Field(default=None, description="Added in 1.2.0. The scheme Store embedded the query with; only turn vectors of this scheme were ranked.")
    calls_needing_reembedding: int = Field(default=0, ge=0, description="Added in 1.2.0. Calls in scope (all, or call_id) whose indexed embeddings are of another scheme, so they were not searched. Request reanalysis kind 'embeddings' to re-embed them.")
