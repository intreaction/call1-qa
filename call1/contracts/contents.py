"""Artifact content contracts: the JSON each machine-produced artifact kind carries.

Process produces these payloads and Store parses them to build the read projections in
``calls.py``. A content model never carries a Store-assigned identifier or version (artifact ID,
artifact version, call ID): Store assigns those when it commits and links the artifact, after the
checksum over the content is fixed. Job IDs and attempt numbers may appear, because Store assigns
them before the producing job runs.

The checksum of a JSON artifact is ``canonical_digest(payload)`` (RFC 8785, ``common.py``).
``artifacts.ARTIFACT_CONTENT_CONTRACTS`` names the contract per kind and
``artifacts.ARTIFACT_CONTENT_MODELS`` maps each JSON contract to its model here; the OpenAPI
document carries every one as a component schema.

This module also holds the shared vocabulary (speaker roles, verdict statuses, result kinds and
states) that content and projections both use; ``calls.py`` re-exports it.

Contact Signals v2 (1.3.0) adds the three stage artifacts (``signal_categories``,
``signal_subcategories``, ``signal_extraction``) and the v2 additions to ``ContactSignalView`` and
``ContactSignalsContent``. The low-level signal vocabulary they share with ``signals.py`` (node IDs,
field types, the taxonomy reference, span keys and hit IDs) lives here, because ``signals.py``
builds on this module; ``signals.py`` re-exports it.

Dual transcription (1.3.0, decision 33, docs/DualAsr.md) adds ``TranscriptContent.vocabulary_correction``
(the merge's provenance and every replacement) and ``asr_vocabulary_pass.v1`` (the raw
vocabulary-prompted Whisper pass); the base engine's own transcript reuses ``transcript.v1``.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field, StringConstraints, model_validator

from .common import ArtifactRef, ContractModel, JsonScalar, ResourceId, SafeText, Sha256Digest, ShortText, Timestamp
from .errors import JobErrorCode
from .vocabulary import VocabularyMergeRule, VocabularyTermSource


# --- Shared vocabulary ----------------------------------------------------------------------


class SpeakerRole(str, Enum):
    AGENT = "AGENT"
    CALLER = "CALLER"
    SYSTEM = "SYSTEM"
    UNKNOWN = "UNKNOWN"


class AudioChannelLayout(str, Enum):
    MONO = "MONO"
    STEREO = "STEREO"
    MULTI_CHANNEL = "MULTI_CHANNEL"
    UNKNOWN = "UNKNOWN"


class VerdictStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    FLAGGED = "FLAGGED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class TextSentimentLabel(str, Enum):
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class ToneBlockStatus(str, Enum):
    SCORED = "SCORED"
    NO_SPEECH = "NO_SPEECH"
    INSUFFICIENT_SPEECH = "INSUFFICIENT_SPEECH"
    OVERLAP = "OVERLAP"
    UNATTRIBUTED = "UNATTRIBUTED"
    AMBIGUOUS_SPEAKER = "AMBIGUOUS_SPEAKER"
    DISABLED = "DISABLED"
    MODEL_ERROR = "MODEL_ERROR"


class EscalationTrigger(str, Enum):
    """Why a QA assessment asks for a second opinion (the rubric's ``escalation_when``)."""

    NEEDS_REVIEW = "needs_review"
    INVALID_ANSWER = "invalid_answer"
    PROVIDER_ERROR = "provider_error"
    ALWAYS = "always"


class ContactSignalKind(str, Enum):
    INTENT = "intent"
    ISSUE = "issue"
    FRICTION = "friction"
    FIX_PROPOSED = "fix_proposed"
    AGENT_REPORTS_COMPLETED = "agent_reports_completed"
    CALLER_CONFIRMS_RESOLVED = "caller_confirms_resolved"
    CALLER_REPORTS_UNRESOLVED = "caller_reports_unresolved"
    DEFERRED = "deferred"
    CUSTOM = "custom"
    """Added in 1.3.0: a hit of an admin-defined (custom) category. ``label`` carries the category's
    name and ``category_id`` its ID, so a 1.2.x client that does not know this value still shows the
    admin's name (``CONTACT_SIGNAL_LABEL[kind] ?? label``)."""


class ContactSignalPass(str, Enum):
    LIFECYCLE = "lifecycle"
    RESOLUTION = "resolution"


class ResultKind(str, Enum):
    """A result group Evaluate renders independently."""

    TRANSCRIPT = "transcript"
    TONE = "tone"
    TEXT_SENTIMENT = "text_sentiment"
    QA = "qa"
    SUMMARY = "summary"
    CONTACT_SIGNALS = "contact_signals"


class ResultState(str, Enum):
    """Derived state of a result group. Not a job status. ``calls.derive_result_state`` is the rule."""

    PENDING = "pending"
    """Analyzing: no version published yet and work is in progress."""
    AVAILABLE = "available"
    PARTIAL = "partial"
    """Published with explicitly missing parts (for example one contact-signal pass failed)."""
    STALE = "stale"
    """A reanalysis affecting this group is underway; the published version is still shown, labeled."""
    FAILED = "failed"
    """No version published and the group's work ended without one ('Needs attention')."""
    DISABLED = "disabled"
    """No graph for this conversation contains the group's work (stage off, not requested, legacy call)."""


class TurnWindow(ContractModel):
    """A bounded, inclusive range of original transcript turn IDs."""

    turn_start: int = Field(ge=0)
    turn_end: int = Field(ge=0, description="Inclusive.")

    @model_validator(mode="after")
    def _ordered(self):
        if self.turn_end < self.turn_start:
            raise ValueError("turn_end precedes turn_start")
        return self


class WordTimestampView(ContractModel):
    word: str
    start_time: float
    end_time: float
    probability: float = Field(ge=0, le=1)


class ToneBlockView(ContractModel):
    """One speaker's acoustic affect in a fixed call-time window (pre-split ``ToneBlock``)."""

    block_id: int
    speaker: SpeakerRole
    start_time: float
    end_time: float
    status: ToneBlockStatus
    speech_seconds: float = Field(ge=0)
    valence: Optional[float] = None
    arousal: Optional[float] = None
    dominance: Optional[float] = None
    emotion: Optional[str] = None
    emotion_probabilities: Optional[Dict[str, float]] = None
    turn_ids: List[int] = Field(default_factory=list)
    model: ShortText
    revision: ShortText
    analysis_version: ShortText


class VadMetricsView(ContractModel):
    total_speech_duration: float = Field(ge=0)
    total_silence_duration: float = Field(ge=0)
    silence_ratio: float = Field(ge=0, le=1)
    overtalk_duration: float = Field(ge=0)
    overtalk_ratio: float = Field(ge=0, le=1)


class ModelAttemptView(ContractModel):
    """One model assessment behind a verdict, with provenance but never the prompt or raw body."""

    job_id: Optional[ResourceId] = None
    attempt_number: Optional[int] = Field(default=None, ge=1)
    catalog_entry_id: ShortText
    model_revision: ShortText
    route_class: ShortText
    destination_host: ShortText
    status: VerdictStatus
    reasoning: str
    quoted_evidence: Optional[str] = None
    trigger: Optional[EscalationTrigger] = None
    latency_ms: int = Field(ge=0)
    tokens_input: Optional[int] = Field(default=None, ge=0)
    tokens_output: Optional[int] = Field(default=None, ge=0)
    error_code: Optional[JobErrorCode] = None


class VerdictView(ContractModel):
    criterion_id: ShortText
    criterion_name: ShortText
    status: VerdictStatus
    confidence: float = Field(ge=0, le=1)
    quoted_evidence: Optional[str] = None
    speaker: SpeakerRole = SpeakerRole.AGENT
    timestamp_range: Optional[Tuple[float, float]] = None
    quote_turn_id: Optional[int] = None
    reasoning: str
    hallucination_detected: bool = False
    model_attempts: List[ModelAttemptView] = Field(default_factory=list)


class RubricHighlightView(ContractModel):
    criterion_id: ShortText
    criterion_name: ShortText
    status: VerdictStatus
    note: str


# --- Contact Signals v2 vocabulary (1.3.0; docs/ContactSignalsV2.md sections 6 and 7) ------------

SIGNAL_NODE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}$"
"""Category, subcategory, field and alert-rule IDs: immutable once saved, at most 40 characters.
'.' is not allowed, so it stays free as the separator of span keys and hit IDs; '_' is allowed
because the built-in kinds use it."""

SignalNodeId = Annotated[str, StringConstraints(pattern=SIGNAL_NODE_ID_PATTERN)]

ShortDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{12}$")]
"""The first 12 hex digits of a ``Sha256Digest`` (``short_digest``), as hit IDs and feedback keys carry them."""

SIGNAL_SPAN_KEY_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,39}\.t[0-9]+b[0-9]+$"
SignalSpanKey = Annotated[str, StringConstraints(pattern=SIGNAL_SPAN_KEY_PATTERN)]
"""``<category_id>.t<turn_id>b<block>`` (``signal_span_key``): stable within one transcript revision."""

SignalStage = Literal["categorize", "subcategorize", "extract"]
SIGNAL_STAGES: Tuple[str, ...] = ("categorize", "subcategorize", "extract")
"""The v2 cascade in pipeline order: stage 1 (category), stage 2 (subcategory), stage 3 (fields)."""

SIGNAL_BLOCK_WINDOWS = 4
"""Windows per block: a block is ``window // 4`` (about 28 s). A span never crosses a turn or a block."""

SIGNAL_NONE_OPTION = "none"
"""The stage-1 option that means 'none of these'. Always present in ``SegmentScores.probabilities``."""
SIGNAL_OTHER_OPTION = "other"
"""The stage-2 'Other <category>' option; also the ``subcategory_id`` of a hit no subcategory fit."""
SIGNAL_NOT_OPTION = "not"
"""The stage-2 'Not <category>' option: stage 2 rejects the span, so it is not a hit."""

SIGNAL_SCOPE_KEYS = frozenset({"AGENT", "CALLER", "UNKNOWN"})
"""Speaker scopes that have a stage-1 option set (``stage1_digests`` keys). SYSTEM turns are skipped."""


def short_digest(digest: str) -> str:
    """The first 12 hex digits of ``sha256:<hex>``: what hit IDs and feedback keys carry."""
    hexpart = digest.removeprefix("sha256:")
    if len(hexpart) < 12 or not all(ch in "0123456789abcdef" for ch in hexpart[:12]):
        raise ValueError("not a sha256 digest")
    return hexpart[:12]


def signal_span_key(category_id: str, turn_id: int, block: int) -> str:
    """``<category_id>.t<turn_id>b<block>``: one span per category, turn and block (section 3.5)."""
    return f"{category_id}.t{turn_id}b{block}"


def signal_hit_id(category_id: str, category_digest: str, transcript_checksum: str, turn_id: int, block: int) -> str:
    """The normative hit ID (section 6.3):
    ``<category_id>.<category_digest[:12]>.<transcript_checksum[:8]>.t<turn_id>b<block>``.

    ``category_digest`` is ``signals.category_digest`` (ID, gloss and speaker only), so a threshold
    edit, a name or description edit, or any change to another node keeps the ID and its feedback.
    ``transcript_checksum`` is the checksum of the transcript artifact revision the hit cites."""
    return f"{category_id}.{short_digest(category_digest)}.{short_digest(transcript_checksum)[:8]}.t{turn_id}b{block}"


def signal_preview_hit_id(category_id: str, turn_id: int, block: int) -> str:
    """``<category_id>.preview.t<turn_id>b<block>``: hit IDs inside a preview or compare result."""
    return f"{category_id}.preview.t{turn_id}b{block}"


class SignalFieldType(str, Enum):
    """Type of an admin-defined extraction field (section 5.2). ``signals.SignalField.type``."""

    STRING = "string"
    ENUM = "enum"
    BOOLEAN = "boolean"
    NUMBER = "number"
    AMOUNT = "amount"
    DATE = "date"


SURFACE_FIELD_TYPES = frozenset({SignalFieldType.STRING, SignalFieldType.NUMBER, SignalFieldType.AMOUNT, SignalFieldType.DATE})
"""Types grounded by their surface text: an exact substring of the span (section 5.3)."""
EVIDENCE_FIELD_TYPES = frozenset({SignalFieldType.ENUM, SignalFieldType.BOOLEAN})
"""Types that may carry an ``evidence`` quote (Gemma's ``evidence_quote``; Needle gives none)."""


class SignalTaxonomyRef(ContractModel):
    """Which taxonomy a result, snapshot or preview used: a published version and its digest
    (``signals.taxonomy_digest``), or, in preview output only, a digest with no version."""

    version: Optional[int] = Field(default=None, ge=1, description="Published taxonomy version; null only in preview output (an unsaved taxonomy).")
    digest: Sha256Digest


class QuoteRange(ContractModel):
    """An exact substring of a masked turn: ``text == masked_turn_text[char_start:char_end]``."""

    char_start: int = Field(ge=0)
    char_end: int = Field(ge=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _exact(self):
        if self.char_end <= self.char_start or len(self.text) != self.char_end - self.char_start:
            raise ValueError("a quote range covers exactly its text")
        return self


class SignalSpanView(ContractModel):
    """Where a v2 hit's span sits in the segment grid, and its +/-1 isolation window in seconds."""

    block: int = Field(ge=0)
    first_window: int = Field(ge=0)
    last_window: int = Field(ge=0)
    timing: Literal["words", "interpolated"]
    context_start: float = Field(ge=0, description="Start of the +/-1 isolation window (seconds); stages 2 and 3 read it, evidence is the core span only.")
    context_end: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self):
        if self.last_window < self.first_window or self.context_end < self.context_start:
            raise ValueError("a span's windows and context are ordered")
        return self


ExtractedFieldStatus = Literal["extracted", "absent", "ungrounded", "withheld_pii", "invalid"]


class ExtractedField(ContractModel):
    """One admin-defined field on one span (stage 3). Absence remains absence: a field the engine
    left out is ``absent``, never empty or zero, and nothing is filled from defaults, context or
    other spans. Only ``extracted`` carries a value. Text (``value`` for strings, ``surface``,
    ``evidence``) is masked text; offsets are in the **masked** turn text (section 6.1). Only an
    ``extracted`` or ``invalid`` (grounded but unparseable) field keeps any text."""

    field_id: SignalNodeId
    type: SignalFieldType
    status: ExtractedFieldStatus
    value: Optional[JsonScalar] = Field(default=None, description="Normalized value; set iff status is extracted. str for string, enum and date (ISO 8601); a number for number and amount; bool for boolean.")
    surface: Optional[str] = Field(default=None, description="string, number, amount and date: the exact substring of the span the value was read from.")
    evidence: Optional[str] = Field(default=None, description="enum and boolean only: the verified evidence quote (Gemma), an exact substring of the core span.")
    char_start: Optional[int] = Field(default=None, ge=0, description="Of surface or evidence, in the masked turn text.")
    char_end: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _status(self):
        extracted = self.status == "extracted"
        if extracted != (self.value is not None):
            raise ValueError("a field has a value exactly when it was extracted")
        if extracted:
            t = self.type
            ok = (isinstance(self.value, bool) if t is SignalFieldType.BOOLEAN
                  else isinstance(self.value, (int, float)) and not isinstance(self.value, bool) if t in (SignalFieldType.NUMBER, SignalFieldType.AMOUNT)
                  else isinstance(self.value, str))
            if not ok:
                raise ValueError(f"an extracted {t.value} field carries a normalized {t.value} value")
        if self.surface is not None and self.type not in SURFACE_FIELD_TYPES:
            raise ValueError("only string, number, amount and date fields carry surface text")
        if self.evidence is not None and self.type not in EVIDENCE_FIELD_TYPES:
            raise ValueError("only enum and boolean fields carry an evidence quote")
        text = self.surface is not None or self.evidence is not None
        if text and self.status not in ("extracted", "invalid"):
            raise ValueError("an absent, ungrounded or withheld field keeps no text")
        if (self.char_start is None) != (self.char_end is None) or (self.char_start is not None) != text:
            raise ValueError("offsets locate the surface or evidence text, and only it")
        if self.char_start is not None and self.char_end < self.char_start:
            raise ValueError("offsets are ordered")
        return self


class ExtractedFieldView(ExtractedField):
    """An extracted field as a hit shows it, with the field's name at the scored taxonomy version.
    Store masks ``value``, ``surface`` and ``evidence`` again on every reviewer read."""

    name: ShortText
    turn_id: Optional[int] = Field(default=None, ge=0, description="Set only on a merged multi-segment hit (decision 25) when the field was read from one of its parts: the part's turn, which char_start/char_end index. Absent means the hit's own turn_id.")


_KIND_VALUES = frozenset(k.value for k in ContactSignalKind)


class SignalHitPart(ContractModel):
    """A later span merged into a multi-segment v2 hit (1.3.0, decision 25; docs/ContactSignalsV2.md
    section 6.5). The merge joins consecutive hits of one speaker with the same category and stage-2
    outcome into the first one (the anchor, which keeps its hit ID, quote and feedback); each later
    span becomes a part. ``quote`` was re-verified against the part's own masked turn at
    ``char_start``/``char_end``, exactly like the anchor's, and Store masks it again on every read."""

    turn_id: int = Field(ge=0)
    block: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    quote: str
    char_start: int = Field(ge=0, description="Offset of quote in the part's masked turn text.")
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self):
        if self.end < self.start or self.char_end <= self.char_start:
            raise ValueError("a part's times and offsets are ordered")
        return self


class ContactSignalView(ContractModel):
    """One contact signal (hit). A v1 hit fills the fields up to ``review_status``. A v2 hit (1.3.0)
    also names its ``category_id`` and fills the v1 fields as follows, so a 1.2.x client keeps
    rendering it: ``id`` is the hit ID (``signal_hit_id``, or ``signal_preview_hit_id`` in a
    preview), ``label`` the category name, ``start``/``end`` and ``quote`` the span or the narrowed
    quote, ``confidence`` 1 - p(Not). **On a v2 hit, ``char_start``/``char_end`` are offsets of
    ``quote`` in the masked turn text** (section 6.1).

    **Multi-segment signals** (decision 25, section 6.5). When one speaker's consecutive spans carry
    the same category and stage-2 outcome, the merge folds them into one hit: the first span (the
    anchor) keeps ``id``, ``quote``, ``turn_id``, offsets, ``span`` and feedback identity; each later
    span is a ``SignalHitPart`` in ``parts``, and ``span_end`` is the last part's end. The merged
    hit's ``confidence`` is the maximum over its spans and its ``fields`` the union by field ID (the
    first extracted value wins; a field extracted from a part carries offsets in that part's turn, named by the field's ``turn_id``)."""

    id: ShortText = Field(description="Stable within the artifact; assigned by the producing job, not by Store. On a v2 hit, the hit ID.")
    kind: ContactSignalKind
    label: ShortText
    start: float
    end: float
    speaker: SpeakerRole
    quote: str
    turn_id: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    confidence: float = Field(ge=0, le=1)
    review_status: ShortText = "unreviewed"
    # --- Contact Signals v2 (1.3.0); all optional, absent on v1 hits ---
    category_id: Optional[SignalNodeId] = Field(default=None, description="Added in 1.3.0 (v2 hits). Equals kind for built-in categories; a custom category's ID when kind is custom.")
    category_digest: Optional[ShortDigest] = Field(default=None, description="Added in 1.3.0. The first 12 hex digits of signals.category_digest, as in the hit ID.")
    category_confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Added in 1.3.0. Stage-1 peak probability.")
    subcategory_id: Optional[SignalNodeId] = Field(default=None, description="Added in 1.3.0. A subcategory ID, or 'other'.")
    subcategory_label: Optional[ShortText] = Field(default=None, description="Added in 1.3.0. The subcategory's name at the scored taxonomy version.")
    subcategory_digest: Optional[ShortDigest] = Field(default=None, description="Added in 1.3.0. The first 12 hex digits of signals.subcategory_digest; subcategory feedback keys on it.")
    subcategory_confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Added in 1.3.0. p(assigned subcategory).")
    span: Optional[SignalSpanView] = Field(default=None, description="Added in 1.3.0. The span's place in the segment grid and its isolation window.")
    fields: List[ExtractedFieldView] = Field(default_factory=list, description="Added in 1.3.0. Stage-3 fields of the hit's category and subcategory path.")
    quote_narrowed: bool = Field(default=False, description="Added in 1.3.0. True when stage 3 narrowed the quote inside the span (narrow_quote).")
    parts: List[SignalHitPart] = Field(default_factory=list, description=(
        "Added in 1.3.0 (decision 25); v2 only. A multi-segment signal: the later spans merged into this hit, in call order. "
        "The hit's own fields describe its first span (the anchor); empty for a single-span hit."))
    span_end: Optional[float] = Field(default=None, ge=0, description=(
        "Added in 1.3.0 (decision 25); v2 only. The end time of the last part (seconds); null when there are no parts."))
    why: Optional["SignalHitWhy"] = Field(default=None, description=(
        "Added in 1.4.0; v2 only. Set on every hit of a result in which the rules engine ran: whether rules or Gemma decided the "
        "category and subcategory, the Gemma check, and the rules' outcomes, score and nearest examples. Null otherwise."))

    @model_validator(mode="after")
    def _v2_hit(self):
        custom = self.kind is ContactSignalKind.CUSTOM
        if self.category_id is None:
            if custom:
                raise ValueError("a custom-category hit names its category_id")
            if (self.category_digest or self.category_confidence is not None or self.subcategory_id or self.subcategory_label
                    or self.subcategory_digest or self.subcategory_confidence is not None or self.span or self.fields or self.quote_narrowed
                    or self.parts or self.span_end is not None or self.why is not None):
                raise ValueError("v2 hit fields belong to a hit that names its category_id")
            return self
        if custom and self.category_id in _KIND_VALUES:
            raise ValueError("a custom category never uses a built-in kind's ID")
        if not custom and self.category_id != self.kind.value:
            raise ValueError("a built-in category's hit has category_id equal to its kind")
        if self.span is None or self.category_digest is None or self.turn_id is None or self.char_start is None or self.char_end is None:
            raise ValueError("a v2 hit carries its span, category digest, turn and quote offsets")
        if self.subcategory_id is None and (self.subcategory_label or self.subcategory_digest or self.subcategory_confidence is not None):
            raise ValueError("subcategory details belong to a hit with a subcategory_id")
        if self.subcategory_id == SIGNAL_NOT_OPTION:
            raise ValueError("a span stage 2 rejected is not a hit")
        if not (self.id.startswith(self.category_id + ".") and self.id.endswith(f".t{self.turn_id}b{self.span.block}")):
            raise ValueError("a v2 hit's id is its hit ID: <category_id>. ... .t<turn_id>b<block>")
        ids = [f.field_id for f in self.fields]
        if len(ids) != len(set(ids)):
            raise ValueError("a hit lists each field once")
        if (self.span_end is None) != (not self.parts):
            raise ValueError("span_end is set exactly when the hit has parts")
        if self.parts:
            spans = [(self.turn_id, self.span.block)] + [(p.turn_id, p.block) for p in self.parts]
            if len(spans) != len(set(spans)):
                raise ValueError("a multi-segment hit names each span once, and never its anchor as a part")
            starts = [self.start] + [p.start for p in self.parts]
            if any(later < earlier for earlier, later in zip(starts, starts[1:])):
                raise ValueError("parts are in call order after the anchor")
            if self.span_end < self.end or any(self.span_end < p.end for p in self.parts):  # type: ignore[operator]
                raise ValueError("span_end is at or after the end of the anchor and of every part")
        return self


# --- Audio validation and VAD (job validation_vad) -------------------------------------------


class AudioValidationContent(ContractModel):
    """``validation_report.v1``: what the source audio is. Feeds the call record's media fields."""

    container: ShortText = Field(description="wav, mp3, flac, ogg or m4a.")
    codec: ShortText
    sample_rate: int = Field(ge=1)
    channels: int = Field(ge=1)
    channel_layout: AudioChannelLayout
    duration_seconds: float = Field(ge=0)
    agent_channel: Optional[int] = Field(default=None, ge=0)
    warnings: List[SafeText] = Field(default_factory=list)


class VadSegmentContent(ContractModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    channel: int = Field(default=0, ge=0)


class VadMetricsContent(VadMetricsView):
    """``vad_metrics.v1``."""

    segments: List[VadSegmentContent] = Field(default_factory=list)


# --- Transcript and per-turn analyses --------------------------------------------------------


class TranscriptTurnContent(ContractModel):
    turn_id: int = Field(ge=0)
    speaker: SpeakerRole = Field(description="From the channel on stereo calls; UNKNOWN on mono calls until speaker attribution.")
    speaker_cluster: Optional[str] = None
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    text: str = Field(description="At the artifact's sensitivity (raw or masked).")
    channel: Optional[int] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    word_timestamps: Optional[List[WordTimestampView]] = None


class VocabularyCorrectionStatus(str, Enum):
    """Added in 1.3.0 (decision 33, docs/DualAsr.md)."""

    APPLIED = "applied"
    """The vocabulary pass ran and the rule merge was applied; there may be zero replacements."""
    BASE_ONLY = "base_only"
    """The vocabulary pass failed or could not run (model not installed, no MLX runtime, a runtime
    error): the transcript is the base engine's alone, with ``note`` saying why. Never a failed call."""


class TranscriptReplacement(ContractModel):
    """Added in 1.3.0. One vocabulary replacement: the term written over the base engine's words,
    with both source strings and times. Offsets and word indices are in the **merged** turn (this
    artifact's raw text); the base engine's words it replaced are ``heard``. Raw text: Store never
    shows ``heard`` to a reviewer without masking it (``TranscriptReplacementView``)."""

    turn_id: int = Field(ge=0)
    word_start: int = Field(ge=0, description="Index of the first replaced word in the merged turn's word_timestamps.")
    word_end: int = Field(ge=1, description="Exclusive end index in the merged turn's word_timestamps.")
    char_start: int = Field(ge=0, description="Offset of the term in the merged turn's text.")
    char_end: int = Field(ge=1, description="Exclusive end offset of the term in the merged turn's text.")
    term: ShortText = Field(description="The vocabulary term as written in the vocabulary (no digits).")
    source: VocabularyTermSource
    heard: str = Field(min_length=1, max_length=400, description="What the base engine (Parakeet) wrote for the span, exactly.")
    candidate_text: str = Field(min_length=1, max_length=400, description="What the vocabulary pass (prompted Whisper) wrote, exactly.")
    start_time: float = Field(ge=0, description="Start of the base engine's span; the term's words share the span evenly.")
    end_time: float = Field(ge=0)
    candidate_start_time: float = Field(ge=0)
    candidate_end_time: float = Field(ge=0)
    phonetic_similarity: float = Field(ge=0, le=1)
    character_similarity: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _spans(self):
        if self.word_end <= self.word_start or self.char_end <= self.char_start:
            raise ValueError("a replacement covers at least one word and one character")
        if self.end_time < self.start_time or self.candidate_end_time < self.candidate_start_time:
            raise ValueError("a replacement's times run forward")
        if any(ch.isdigit() for ch in self.term):
            raise ValueError("a vocabulary term never contains a digit")
        return self


class VocabularyCorrection(ContractModel):
    """Added in 1.3.0. ``TranscriptContent.vocabulary_correction``: present exactly when the ``asr``
    job ran with ``parameters.asr_vocabulary``. Provenance of the merge and every replacement."""

    status: VocabularyCorrectionStatus
    note: Optional[SafeText] = Field(default=None, description="Why the pass did not run or was partial; safe text, never transcript content.")
    failure_code: Optional[JobErrorCode] = Field(default=None, description="base_only: the per-attempt code the vocabulary pass would have failed with (model_unavailable, configuration_error, ...). The asr job itself still succeeds.")
    vocabulary_digest: Sha256Digest = Field(description="parameters.asr_vocabulary.digest of the job.")
    term_count: int = Field(ge=1)
    glossary_term_count: int = Field(default=0, ge=0, description="Terms in the per-call glossary shortlist the vocabulary pass was prompted with.")
    base_engine: ShortText = Field(description="e.g. 'parakeet-tdt-0.6b-v3'.")
    candidate_engine: Optional[ShortText] = Field(default=None, description="e.g. 'whisper-small'; null when it never ran.")
    candidate_model_revision: Optional[ShortText] = None
    rule: VocabularyMergeRule
    candidates: int = Field(default=0, ge=0, description="Vocabulary hits the pass produced (candidates before the rule).")
    replacements: List[TranscriptReplacement] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self):
        if self.status is VocabularyCorrectionStatus.BASE_ONLY:
            if self.replacements or not self.note:
                raise ValueError("a base_only transcript has no replacements and says why in note")
        elif self.failure_code is not None:
            raise ValueError("failure_code is set only on base_only")
        ordered = sorted(self.replacements, key=lambda r: (r.turn_id, r.word_start))
        if [(r.turn_id, r.word_start) for r in ordered] != [(r.turn_id, r.word_start) for r in self.replacements]:
            raise ValueError("replacements are in transcript order (turn, then word)")
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.turn_id == prev.turn_id and (cur.word_start < prev.word_end or cur.char_start < prev.char_end):
                raise ValueError("replacements never overlap")
        return self


class TranscriptContent(ContractModel):
    """``transcript.v1`` (job asr). Turn IDs are the original IDs every later citation uses. Since
    1.3.0 (decision 33) the ``asr`` job's transcript may be the vocabulary-merged one; the base
    engine's own transcript is then kept as the optional ``asr_base_transcript`` output."""

    duration_seconds: float = Field(ge=0)
    language: Optional[ShortText] = None
    is_redacted: bool
    turns: List[TranscriptTurnContent]
    vocabulary_correction: Optional[VocabularyCorrection] = Field(default=None, description="Added in 1.3.0: present exactly when the asr job ran with parameters.asr_vocabulary (dual transcription). Null on every transcript made without a vocabulary, and on an asr_base_transcript.")


class AsrPassWord(ContractModel):
    """Added in 1.3.0. One word of the vocabulary pass, with Whisper's own timing and probability."""

    word: str = Field(max_length=400)
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    probability: Optional[float] = Field(default=None, ge=0, le=1)
    channel: Optional[int] = Field(default=None, ge=0, le=1)
    segment: Optional[int] = Field(default=None, ge=0, description="Index into AsrVocabularyPassContent.segments.")


class AsrPassSegment(ContractModel):
    """Added in 1.3.0. One decoded Whisper segment's statistics (no text; the words carry it)."""

    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    channel: Optional[int] = Field(default=None, ge=0, le=1)
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = Field(default=None, ge=0, le=1)
    compression_ratio: Optional[float] = Field(default=None, ge=0)
    temperature: Optional[float] = Field(default=None, ge=0)


class AsrVocabularyPassContent(ContractModel):
    """Added in 1.3.0. ``asr_vocabulary_pass.v1`` (optional output ``vocabulary_pass`` of job asr):
    the raw vocabulary-prompted Whisper Small pass. **A candidate finder only, never a transcript:**
    prompted Whisper can skip whole 30 s windows (docs/DualAsr.md). Raw sensitivity; Store never
    projects it to reviewers."""

    engine: ShortText
    model_revision: Optional[ShortText] = None
    duration_seconds: float = Field(ge=0)
    channels: int = Field(ge=1, le=2)
    glossary_terms: List[ShortText] = Field(default_factory=list, description="The per-call shortlist the pass was prompted with, in prompt order (vocabulary terms only).")
    words: List[AsrPassWord]
    segments: List[AsrPassSegment] = Field(default_factory=list)


class SpeakerAssignment(ContractModel):
    turn_id: int = Field(ge=0)
    speaker: SpeakerRole
    speaker_cluster: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class SpeakerAttributionContent(ContractModel):
    """``speaker_attribution.v1`` (job speaker_attribution, or a reviewer's speaker correction)."""

    method: Literal["channel", "diarization", "reviewer_correction"]
    assignments: List[SpeakerAssignment]
    corrected_turn_ids: List[int] = Field(default_factory=list, description="Turns a reviewer relabelled, for reviewer_correction.")


class ToneBlocksContent(ContractModel):
    """``tone_blocks.v1`` (job acoustic_tone)."""

    blocks: List[ToneBlockView]
    avg_agent_tone: Optional[float] = None


class TurnSentiment(ContractModel):
    turn_id: int = Field(ge=0)
    score: Optional[float] = Field(default=None, ge=-1, le=1, description="Polarity, -1 to +1; absent when not scored.")
    label: Optional[TextSentimentLabel] = None
    probabilities: Dict[str, float] = Field(default_factory=dict)
    status: ShortText = "SCORED"


class TextSentimentContent(ContractModel):
    """``text_sentiment.v1`` (job text_sentiment)."""

    model: ShortText
    revision: ShortText
    turns: List[TurnSentiment]
    avg_caller_sentiment: Optional[float] = None


class TurnEmbedding(ContractModel):
    turn_id: int = Field(ge=0)
    vector: List[float] = Field(min_length=1, max_length=4096)


class EmbeddingsContent(ContractModel):
    """``embeddings.v1`` (job embeddings). ``scheme`` names how the vectors were made, so a query
    can only ever be compared with vectors of the same scheme. Since 1.2.0 (decision 18) the
    ``embeddings`` job is a model stage: ``nemotron-3-embed-1b@<revision[:7]>``
    (nvidia/Nemotron-3-Embed-1B-BF16, one mean-pooled, L2-normalized 2048-dimension vector per turn,
    embedded as ``passage: <text>``), and Store embeds search queries (``query: <text>``) with the
    same model. Store ranks only vectors whose scheme matches its configured embedder; artifacts of an
    older scheme (``hashing-projection-v1`` before 1.2.0) are kept but never ranked until the call is
    re-embedded (reanalysis kind ``embeddings``)."""

    scheme: ShortText = Field(description="The embedder and revision, e.g. 'nemotron-3-embed-1b@c0c9fea' (1.2.0); 'hashing-projection-v1' on artifacts from before 1.2.0.")
    dimensions: int = Field(ge=1, le=4096)
    turn_vectors: List[TurnEmbedding]


class NumericEntityContent(ContractModel):
    raw_text: str
    normalized_value: float | str
    entity_type: Literal["CURRENCY", "PERCENTAGE", "DATE", "PHONE_NUMBER", "ACCOUNT_NUMBER", "DURATION", "GENERIC_NUMBER"]
    start_time: float
    end_time: float
    unit: Optional[str] = None


class TurnEnrichment(ContractModel):
    turn_id: int = Field(ge=0)
    numeric_entities: List[NumericEntityContent] = Field(default_factory=list)


class EnrichmentContent(ContractModel):
    """``enrichment.v1`` (job enrichment): deterministic numeric extraction per turn. Raw customer
    content; masking uses it."""

    turns: List[TurnEnrichment]


PiiCategory = Literal["account_number", "private_address", "private_email", "private_person", "private_phone", "private_url", "secret"]
"""The masked categories of the model PII layer (team decision 19): every ``openai/privacy-filter``
label except ``private_date`` (dates stay visible because rubric criteria depend on them)."""


class PiiSpanContent(ContractModel):
    """One masked finding: a character range of the turn's ``text`` and the text it covers."""

    start: int = Field(ge=0, description="Inclusive character offset into the turn's transcript text.")
    end: int = Field(ge=1, description="Exclusive character offset.")
    category: PiiCategory
    text: str = Field(min_length=1, max_length=2000, description="The matched text (raw customer content). Store masks this span by position (turn_id, start, end) on reviewer reads; it propagates the text by value to other occurrences only when it is a strong identifier (team decision 22).")

    @model_validator(mode="after")
    def _ordered(self):
        if self.end <= self.start:
            raise ValueError("a PII span ends after it starts")
        return self


class TurnPiiFindings(ContractModel):
    turn_id: int = Field(ge=0)
    spans: List[PiiSpanContent] = Field(default_factory=list)


class PiiFindingsContent(ContractModel):
    """``pii_findings.v1`` (job enrichment, output role ``pii_findings``; added in 1.2.0, team
    decision 19). The model PII layer's findings for one transcript revision, already filtered by
    the masking policy: only masked categories, and no span that names only the agent (the call's
    ``agent_display_name`` plus agent self-introductions). Raw customer content.

    Store masks every reviewer read, and mutes audio, over the union of the rule-based values
    (``call1.redaction``) and these spans: each kept span is masked at its position (``turn_id``,
    ``start``, ``end``), and its text is propagated by value only when it is a strong identifier
    (team decision 22; common-word spans are dropped). It uses findings only when ``transcript`` is the
    call's current published transcript; while none match, reviewer text is withheld (fail closed;
    see the contract README, "Version 1.2.0")."""

    transcript: ArtifactRef = Field(description="The transcript artifact (revision) these findings were made from.")
    detector: ShortText = Field(description="'openai/privacy-filter', or 'stub' (the labelled model-free stand-in on fake-handler stacks).")
    detector_revision: ShortText = Field(description="The model revision, or the stub's version.")
    turns: List[TurnPiiFindings]


# --- Prompt inputs and QA --------------------------------------------------------------------


class PromptInputContent(ContractModel):
    """``prompt_input.v1``: what an LLM job sent, by reference and hash only. Never the prompt text,
    the transcript or a provider body."""

    template_id: ShortText
    template_version: ShortText
    prompt_digest: Sha256Digest = Field(description="canonical_digest of the exact messages sent (after masking).")
    masked: bool
    transcript_window: Optional[TurnWindow] = None
    inputs: List[ArtifactRef] = Field(default_factory=list)
    estimated_input_tokens: Optional[int] = Field(default=None, ge=0)


class QaAssessmentContent(ContractModel):
    """``qa_assessment.v1`` (jobs qa_criterion and qa_escalation). One model assessment of one
    criterion. A provider failure on the job's final attempt, or an answer that fails schema or
    quote validation, is recorded here as a FLAGGED assessment with its trigger, not as a job
    failure (see README, "QA assessments that cannot answer")."""

    criterion_id: ShortText
    assessment_kind: Literal["primary", "escalation"]
    status: VerdictStatus
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    quoted_evidence: Optional[str] = None
    quote_turn_id: Optional[int] = None
    timestamp_range: Optional[Tuple[float, float]] = None
    speaker: SpeakerRole = SpeakerRole.AGENT
    hallucination_detected: bool = False
    trigger: Optional[EscalationTrigger] = Field(default=None, description="The trigger this outcome raises: needs_review, invalid_answer, provider_error (or always).")
    escalation_requested: bool = Field(description="True when the trigger is in the criterion's escalation_when and an escalation job was created with this completion.")
    attempt: ModelAttemptView

    @model_validator(mode="after")
    def _failure_is_flagged(self):
        if self.trigger in (EscalationTrigger.INVALID_ANSWER, EscalationTrigger.PROVIDER_ERROR) and self.status is not VerdictStatus.FLAGGED:
            raise ValueError("an invalid answer or a provider failure is recorded as FLAGGED")
        if (self.trigger is EscalationTrigger.PROVIDER_ERROR) != (self.attempt.error_code is not None):
            raise ValueError("a provider-error assessment names its safe error code, and only it does")
        return self


class QaVerdictContent(ContractModel):
    """``qa_verdict.v1`` (job qa_deterministic): the rubric's deterministic checks."""

    verdicts: List[VerdictView]


class ScorecardRubricRef(ContractModel):
    """The rubric a scorecard was scored with: a published version, or a draft under test."""

    rubric_id: ShortText
    rubric_version: Optional[int] = Field(default=None, ge=1, description="Published rubric version; absent for a draft test.")
    draft_revision: Optional[int] = Field(default=None, ge=1, description="Draft revision; present only for a draft test.")
    digest: Sha256Digest = Field(description="canonical_digest of the rubric definition scored with.")

    @model_validator(mode="after")
    def _one_source(self):
        if (self.rubric_version is None) == (self.draft_revision is None):
            raise ValueError("a scorecard names exactly one of a published version or a draft revision")
        return self


class QaScorecardContent(ContractModel):
    """``qa_scorecard.v1`` (job qa_scorecard). ``calls.EvaluationView`` is this plus Store's IDs."""

    rubric: ScorecardRubricRef
    overall_score: float = Field(ge=0, le=100)
    passed: bool
    critical_failure: bool
    requires_human_review: bool
    escalation_reasons: List[str] = Field(default_factory=list)
    verdicts: List[VerdictView]
    evaluated_at: Timestamp


# --- Summary ---------------------------------------------------------------------------------


class SummaryCitation(ContractModel):
    """Ties one claim to the original transcript turns that support it."""

    claim: Literal["narrative", "key_point"]
    index: int = Field(ge=0, description="Key point index; 0 for the narrative.")
    turn_ids: List[int] = Field(min_length=1)


class SummarySegmentContent(ContractModel):
    """``summary_segment.v1`` (job summary_segment)."""

    segment_index: int = Field(ge=0)
    window: TurnWindow
    narrative: str
    key_points: List[str] = Field(default_factory=list)
    citations: List[SummaryCitation] = Field(default_factory=list)


class SummarySynthesisContent(ContractModel):
    """``summary_synthesis.v1`` (job summary_synthesis): a pairwise or final synthesis."""

    segment_indexes: List[int] = Field(min_length=1)
    final: bool
    narrative: str
    key_points: List[str] = Field(default_factory=list)
    citations: List[SummaryCitation] = Field(default_factory=list)


class SummaryContent(ContractModel):
    """``summary.v1`` (job summary_assembly). ``calls.SummaryView`` is this plus Store's IDs."""

    narrative: str
    key_points: List[str]
    rubric_highlights: List[RubricHighlightView] = Field(default_factory=list)
    citations: List[SummaryCitation] = Field(default_factory=list)
    grounding: Dict[str, JsonScalar] = Field(default_factory=dict, description="Grounding flags as the pre-split summarizer records them (for example model_narrative_only, key_points_citations_checked).")
    route_class: ShortText
    catalog_entry_id: ShortText
    generated_at: Timestamp
    segments: int = Field(ge=1, description="How many segment jobs fed this summary.")


# --- Contact signals -------------------------------------------------------------------------


class ContactSignalsPassContent(ContractModel):
    """``contact_signals_pass.v1`` (jobs contact_signals_lifecycle and contact_signals_resolution):
    quote-verified observations of one pass over one window."""

    pass_kind: ContactSignalPass
    window: Optional[TurnWindow] = Field(default=None, description="Present when the pass was split into windows.")
    signals: List[ContactSignalView]


class ContactSignalsPassOutcome(ContractModel):
    pass_kind: ContactSignalPass
    window: Optional[TurnWindow] = None
    included: bool
    failure_code: Optional[JobErrorCode] = None


# --- Contact Signals v2 stage artifacts (1.3.0) ----------------------------------------------


class SignalStageProvenance(ContractModel):
    """How one v2 stage ran (section 6.4). v2 purposes are always masked (decision 22, Q12), so
    ``masked`` is always true."""

    stage: SignalStage
    catalog_entry_id: ShortText
    model_revision: ShortText
    adapter_version: ShortText
    calibration_id: Optional[ShortText] = Field(default=None, description="The fitted temperatures and thresholds (e.g. 'fake-signals-v1' on fake handlers).")
    question_template: Optional[ShortText] = Field(default=None, description="e.g. signals.categorize.v1, signals.subcategorize.v1, signals.extract.v1.")
    key_orders: int = Field(default=1, ge=1, le=2, description="K neutral-key orders averaged per row.")
    device: Literal["mps", "cpu", "fake", "remote"]
    route_class: ShortText
    masked: bool = Field(description="Always true: every v2 stage reads masked text only (section 11.2).")
    rows: int = Field(ge=0, description="Classifier rows (stages 1 and 2) or spans (stage 3) the engine ran.")

    @model_validator(mode="after")
    def _masked(self):
        if not self.masked:
            raise ValueError("v2 stages always run on masked text")
        return self


class SignalSegmentRef(ContractModel):
    """One ~7 s segment of the grid (stage 0). No text: the segment is
    ``masked_turn_text[char_start:char_end]`` and never crosses a turn."""

    index: int = Field(ge=0, description="Call-wide order.")
    turn_id: int = Field(ge=0)
    window: int = Field(ge=0, description="k-th window of the turn, 0-based.")
    block: int = Field(ge=0, description="window // 4.")
    speaker: SpeakerRole = Field(description="AGENT, CALLER or UNKNOWN (unattributed mono); SYSTEM turns are skipped, not segmented.")
    char_start: int = Field(ge=0, description="Offset in the masked turn text.")
    char_end: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    timing: Literal["words", "interpolated"]

    @model_validator(mode="after")
    def _grid(self):
        if self.block != self.window // SIGNAL_BLOCK_WINDOWS:
            raise ValueError("a segment's block is window // 4")
        if self.speaker is SpeakerRole.SYSTEM:
            raise ValueError("SYSTEM turns are skipped, not segmented")
        if self.char_end <= self.char_start or self.end < self.start:
            raise ValueError("a segment's offsets and times are ordered")
        return self


class SegmentScores(ContractModel):
    """Calibrated stage-1 probabilities of one scored segment. Sparse: every option with p >= 0.02
    is kept, and 'none' always is."""

    index: int = Field(ge=0)
    probabilities: Dict[str, float] = Field(description="Option ID (a category ID, or 'none') -> calibrated p.")

    @model_validator(mode="after")
    def _sparse(self):
        if SIGNAL_NONE_OPTION not in self.probabilities:
            raise ValueError("every scored segment keeps p(none)")
        for option, p in self.probabilities.items():
            if not 0 <= p <= 1:
                raise ValueError("probabilities are in [0, 1]")
            if option != SIGNAL_NONE_OPTION and p < 0.02:
                raise ValueError("options under 0.02 are dropped, except 'none'")
        return self


class SignalSpanRef(ContractModel):
    """One stage-1 span: the windows of one turn and block that fired one category, plus the
    +/-1 isolation window (segment indexes) that stages 2 and 3 read."""

    span_key: SignalSpanKey
    category_id: SignalNodeId
    turn_id: int = Field(ge=0)
    block: int = Field(ge=0)
    first_window: int = Field(ge=0)
    last_window: int = Field(ge=0)
    peak_window: int = Field(ge=0)
    peak_probability: float = Field(ge=0, le=1)
    context_first: int = Field(ge=0, description="Segment index of the first segment of the isolation window.")
    context_last: int = Field(ge=0)

    @model_validator(mode="after")
    def _span(self):
        if self.span_key != signal_span_key(self.category_id, self.turn_id, self.block):
            raise ValueError("span_key is <category_id>.t<turn_id>b<block>")
        if not self.first_window <= self.peak_window <= self.last_window:
            raise ValueError("the peak window lies inside the span")
        if self.first_window // SIGNAL_BLOCK_WINDOWS != self.block or self.last_window // SIGNAL_BLOCK_WINDOWS != self.block:
            raise ValueError("a span never crosses a block")
        if self.context_last < self.context_first:
            raise ValueError("the isolation window is ordered")
        return self


def _unique(values: List[str], what: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{what} are unique")


# --- Contact Signals rules engine (1.4.0; docs/SignalsEmbeddings.md sections 2-4) -----------------


class SignalRuleType(str, Enum):
    """The closed set of v1 rule types a category recipe may use (``signals.SignalRecipe``). A new
    type is a contract change (docs/SignalsEmbeddings.md section 3.5 lists the candidates)."""

    SIMILAR_TO_EXAMPLES = "similar_to_examples"
    PHRASE = "phrase"
    SPEAKER = "speaker"
    CALL_POSITION = "call_position"


SIGNAL_RULES_ENTRY_ID = "call1-signal-rules"
"""``SignalStageProvenance.catalog_entry_id`` of a categorize run in which the rules engine decided at
least one category (Gemma may still have answered the categories without a rules recipe)."""

SIGNAL_BANK_ID_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,63}$"


class SignalExampleBankRef(ContractModel):
    """An example bank pinned by a taxonomy (``SignalRulesConfig.bank``): an installed pack of masked,
    labelled example segments (for the retail seed, ``retail-v1``). Process checks the pack's
    ``canonical_digest`` against ``digest`` before it uses it and fails the job otherwise."""

    bank_id: Annotated[str, StringConstraints(pattern=SIGNAL_BANK_ID_PATTERN)]
    digest: Sha256Digest


class SignalKnnSettings(ContractModel):
    """The kNN share vote over the example bank (docs/SignalsEmbeddings.md section 3.1): the top ``k``
    same-speaker neighbours by cosine, weighted ``exp((cos - top) / temperature)``; entries built from
    the taxonomy's own text (names, glosses, examples) count ``taxonomy_example_weight`` times."""

    k: int = Field(default=10, ge=1, le=40)
    temperature: float = Field(default=0.1, gt=0, le=1)
    taxonomy_example_weight: float = Field(default=2.0, ge=0, le=10)


class SignalRuleOutcome(ContractModel):
    """One rule of a recipe on one segment. Numbers and indexes only, never text."""

    rule_id: SignalNodeId
    type: SignalRuleType
    result: Literal["pass", "fail"]
    value: Optional[float] = Field(default=None, description="similar_to_examples: the kNN share; call_position: where the segment starts, as a fraction of the call.")
    phrase_index: Optional[int] = Field(default=None, ge=0, description="phrase: the index of the first lexicon phrase that matched.")
    vetoed: bool = Field(default=False, description="phrase: a negation cue stood within the veto window before a match.")


class SignalNeighbour(ContractModel):
    """One of the nearest example-bank entries behind a rule decision (no text)."""

    entry_id: ShortText = Field(description="The bank entry (``<bank_id>:<n>``) or taxonomy entry (``taxonomy:<category>[/<subcategory>][#<example>]``).")
    cosine: float = Field(ge=-1, le=1)
    carries_category: bool = Field(description="The entry is labelled with the decided category.")
    subcategory_id: Optional[SignalNodeId] = Field(default=None, description="The entry's subcategory of the decided category ('other' when it carries the category without one).")


class SignalRuleDecision(ContractModel):
    """Why the rules engine decided one span (docs/SignalsEmbeddings.md section 2.5): the recipe's rule
    outcomes, the score parts and the nearest bank entries of the span's strongest segment, and the
    subcategory the kNN vote picked. No text."""

    span_key: SignalSpanKey
    category_id: SignalNodeId
    segment_index: int = Field(ge=0, description="The span's strongest segment (largest margin score - threshold).")
    recipe_digest: ShortDigest = Field(description="signals.recipe_digest of the category's recipe, first 12 hex digits.")
    score: float = Field(description="kNN share + lexicon_weight * lexicon match.")
    threshold: float = Field(ge=0, le=1)
    knn_share: float = Field(ge=0, le=1)
    lexicon_weight: float = Field(ge=0, le=1)
    lexicon_match: bool
    lexicon_phrase: Optional[int] = Field(default=None, ge=0)
    outcomes: List[SignalRuleOutcome] = Field(default_factory=list, max_length=16)
    neighbours: List[SignalNeighbour] = Field(default_factory=list, max_length=3)
    subcategory_id: Optional[SignalNodeId] = Field(default=None, description="The kNN vote's subcategory; null means Other.")
    subcategory_share: float = Field(default=0.0, ge=0, le=1, description="The winning subcategory's share of the category's vote.")
    check: bool = Field(default=False, description="The recipe asks Gemma to confirm or reject the span (subcategorize runs today's stage-2 prompt on it).")
    system_one_score: Optional[float] = Field(default=None, ge=0, le=1, description="Optional Laya category score, not a calibrated accuracy probability.")
    system_one_kept: bool = Field(default=False, description="Laya and strong semantic category/subcategory agreement allowed this span to bypass Gemma confirmation.")
    system_one_fallback: Optional[JobErrorCode] = Field(default=None, description="Why Laya could not score the candidate; Gemma must confirm it.")

    @model_validator(mode="after")
    def _decision(self):
        if self.system_one_kept and (self.system_one_score is None or self.check or self.system_one_fallback is not None):
            raise ValueError("a System One shortcut has a score, no fallback, and no Gemma check")
        if not self.span_key.startswith(self.category_id + ".t"):
            raise ValueError("a rule decision's span key is its category's")
        if self.subcategory_id in (SIGNAL_OTHER_OPTION, SIGNAL_NOT_OPTION):
            raise ValueError("a rule decision names a subcategory, or null for Other")
        return self


class SignalRuleCounts(ContractModel):
    """Per category and call, the preview funnel of the rules engine."""

    category_id: SignalNodeId
    segments: int = Field(ge=0, description="Segments in the category's speaker scope.")
    filter_passed: int = Field(ge=0)
    fired: int = Field(ge=0, description="Segments the category fired on, after the two-fires-per-segment cap.")


class SignalRulesProvenance(ContractModel):
    """How the rules engine ran in a categorize job (1.4.0). Present exactly when at least one active
    category was decided by rules (``SignalSettings.detection`` rules and the category's recipe engine
    rules)."""

    engine_version: ShortText
    embedder_scheme: ShortText
    bank: Optional[SignalExampleBankRef] = None
    bank_entries: int = Field(ge=0, description="Entries of the pinned bank pack.")
    taxonomy_entries: int = Field(ge=0, description="Entries built from the taxonomy's own text.")
    knn: SignalKnnSettings
    rules_categories: List[SignalNodeId] = Field(description="Categories the rules engine decided.")
    checked_categories: List[SignalNodeId] = Field(default_factory=list, description="Rules categories whose recipe asks for a Gemma check.")
    gemma_categories: List[SignalNodeId] = Field(default_factory=list, description="Active categories Gemma decided in the same run (no rules recipe).")
    recipe_digests: Dict[str, ShortDigest] = Field(default_factory=dict)
    counts: List[SignalRuleCounts] = Field(default_factory=list)
    embedded_segments: int = Field(ge=0)
    embed_seconds: float = Field(ge=0, description="Embedder load and encode, wall time.")
    rules_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _categories(self):
        if not self.rules_categories:
            raise ValueError("rules provenance names the categories the rules decided")
        if not set(self.checked_categories) <= set(self.rules_categories) or set(self.rules_categories) & set(self.gemma_categories):
            raise ValueError("checked categories are rules categories, and no category is both rules and Gemma")
        return self


class SignalHitWhy(ContractModel):
    """Added in 1.4.0. Where a v2 hit came from (docs/SignalsEmbeddings.md section 2.5), set on every
    hit of a result in which the rules engine ran. No text: Store masks nothing here."""

    category_source: Literal["rules", "gemma"]
    subcategory_source: Optional[Literal["rules", "gemma"]] = Field(default=None, description="Null when stage 2 did not reach the merge.")
    check: Optional[Literal["confirmed"]] = Field(default=None, description="'confirmed' when Gemma checked the rule-decided span; a rejected span is not a hit.")
    rule: Optional[SignalRuleDecision] = Field(default=None, description="Set exactly when category_source is rules.")

    @model_validator(mode="after")
    def _why(self):
        if (self.category_source == "rules") != (self.rule is not None):
            raise ValueError("a rules hit carries its rule decision, and only a rules hit does")
        if self.check is not None and self.category_source != "rules":
            raise ValueError("only a rule-decided hit is checked")
        return self


class SignalCategoriesContent(ContractModel):
    """``signal_categories.v1`` (job contact_signals_categorize; derived, no text). The segment grid,
    sparse stage-1 probabilities, the spans they fire at the resolved thresholds, skip counts and
    provenance. Because it holds scores, a threshold edit re-derives spans with no model run
    (``mode: rederive``, section 7.5)."""

    mode: Literal["run", "rederive"]
    provenance: SignalStageProvenance = Field(description="The run's; a rederive copies the previous run's.")
    segmenter_version: ShortText
    window_seconds: float = Field(gt=0)
    taxonomy_ref: SignalTaxonomyRef
    stage1_digests: Dict[str, Sha256Digest] = Field(description="Speaker scope (AGENT, CALLER, UNKNOWN) -> signals.stage1_digest.")
    thresholds: Dict[str, float] = Field(description="Category ID -> the resolved stage-1 threshold used.")
    transcript: ArtifactRef
    speaker_attribution: Optional[ArtifactRef] = None
    segments: List[SignalSegmentRef]
    scores: List[SegmentScores]
    spans: List[SignalSpanRef]
    skipped_unattributed: int = Field(ge=0, description="UNKNOWN segments skipped by speaker-scoped categories.")
    skipped_system: int = Field(ge=0)
    unscored_no_options: int = Field(ge=0, description="Segments whose only option was 'none', so no row was scored.")
    rules: Optional[SignalRulesProvenance] = Field(default=None, description="Added in 1.4.0. How the rules engine ran; null when no category was decided by rules.")
    rule_decisions: List[SignalRuleDecision] = Field(default_factory=list, description="Added in 1.4.0. One per span of a rules category: why it fired and its kNN subcategory.")

    @model_validator(mode="after")
    def _grid(self):
        if self.provenance.stage != "categorize":
            raise ValueError("signal_categories carries categorize provenance")
        if self.rule_decisions and self.rules is None:
            raise ValueError("rule decisions come with rules provenance")
        _unique([d.span_key for d in self.rule_decisions], "rule decision span keys")
        if not set(self.stage1_digests) <= SIGNAL_SCOPE_KEYS:
            raise ValueError("stage1_digests is keyed by speaker scope: AGENT, CALLER or UNKNOWN")
        if any(not 0 <= t <= 1 for t in self.thresholds.values()):
            raise ValueError("thresholds are probabilities")
        indexes = [s.index for s in self.segments]
        _unique(indexes, "segment indexes")
        scored = [s.index for s in self.scores]
        _unique(scored, "scored segment indexes")
        if not set(scored) <= set(indexes):
            raise ValueError("scores refer to segments of the grid")
        _unique([s.span_key for s in self.spans], "span keys")
        return self


class SpanSubcategoryDecision(ContractModel):
    """Stage 2 on one span: pick a subcategory, 'Other' or reject ('Not'). Rejected spans stay in
    the artifact for evaluation but are not hits."""

    span_key: SignalSpanKey
    stage2_digest: Sha256Digest
    probabilities: Dict[str, float] = Field(description="Subcategory IDs, 'other' and 'not' -> calibrated p.")
    decision: Literal["subcategory", "other", "rejected"]
    subcategory_id: Optional[SignalNodeId] = Field(default=None, description="Set iff decision is subcategory; never 'other' or 'not'.")
    confidence: float = Field(ge=0, le=1, description="1 - p(not).")
    factors: List[ShortText] = Field(description="The state factors the frozen adapter version enabled (section 4.2).")
    truncated: bool = False
    status: Literal["decided", "error"]
    error_code: Optional[JobErrorCode] = None
    source: Literal["engine", "rules"] = Field(default="engine", description="Added in 1.4.0. rules: passed through from the rules engine's kNN vote with no model (signal_categories.rule_decisions).")
    checked: bool = Field(default=False, description="Added in 1.4.0. The engine (Gemma) checked a rule-decided span: confirmed with a subcategory, or rejected ('not').")

    @model_validator(mode="after")
    def _decision(self):
        if (self.status == "error") != (self.error_code is not None):
            raise ValueError("an error decision names its error code, and only it does")
        if self.source == "rules" and self.checked:
            raise ValueError("a checked span's decision is the engine's, not the rules'")
        if (self.decision == "subcategory") != (self.subcategory_id is not None):
            raise ValueError("a subcategory decision names its subcategory, and only it does")
        if self.subcategory_id in (SIGNAL_OTHER_OPTION, SIGNAL_NOT_OPTION):
            raise ValueError("'other' and 'not' are decisions, not subcategory IDs")
        if any(not 0 <= p <= 1 for p in self.probabilities.values()):
            raise ValueError("probabilities are in [0, 1]")
        p_not = self.probabilities.get(SIGNAL_NOT_OPTION)
        if self.status == "decided" and p_not is not None and abs(self.confidence - (1 - p_not)) > 1e-6:
            raise ValueError("confidence is 1 - p(not)")
        return self


class SignalSubcategoriesContent(ContractModel):
    """``signal_subcategories.v1`` (job contact_signals_subcategorize; derived, no text). Every span's
    decision, including those carried forward unchanged from ``previous``."""

    provenance: SignalStageProvenance
    decisions: List[SpanSubcategoryDecision]
    carried_forward: List[SignalSpanKey] = Field(default_factory=list, description="Span keys whose decision was reused from the previous artifact.")

    @model_validator(mode="after")
    def _decisions(self):
        if self.provenance.stage != "subcategorize":
            raise ValueError("signal_subcategories carries subcategorize provenance")
        keys = [d.span_key for d in self.decisions]
        _unique(keys, "span keys")
        if not set(self.carried_forward) <= set(keys):
            raise ValueError("carried-forward spans are listed among the decisions")
        return self


class SpanExtraction(ContractModel):
    """Stage 3 on one span: the path's fields and, with narrow_quote, the narrowed quote."""

    span_key: SignalSpanKey
    stage3_digest: Sha256Digest
    status: Literal["extracted", "error", "over_budget", "upstream_missing"]
    error_code: Optional[JobErrorCode] = None
    fields: List[ExtractedField]
    narrowed_quote: Optional[QuoteRange] = Field(default=None, description="An exact substring of the core span in the masked turn text, at least 3 characters.")
    source: Literal["primary", "fallback"] = Field(default="primary", description="fallback when the span reran on the declared in-job fallback entry (section 5.7).")
    engine_confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Recorded, never used to gate values in v1.")

    @model_validator(mode="after")
    def _span(self):
        if (self.status == "error") != (self.error_code is not None):
            raise ValueError("an error span names its error code, and only it does")
        if self.narrowed_quote is not None and self.status != "extracted":
            raise ValueError("only an extracted span narrows its quote")
        _unique([f.field_id for f in self.fields], "field IDs of a span")
        return self


class SignalExtractionContent(ContractModel):
    """``signal_extraction.v1`` (job contact_signals_extract; masked text)."""

    provenance: SignalStageProvenance
    fallback_provenance: Optional[SignalStageProvenance] = Field(default=None, description="Set iff any span ran on the in-job fallback entry.")
    spans: List[SpanExtraction]
    carried_forward: List[SignalSpanKey] = Field(default_factory=list)

    @model_validator(mode="after")
    def _spans(self):
        if self.provenance.stage != "extract" or (self.fallback_provenance is not None and self.fallback_provenance.stage != "extract"):
            raise ValueError("signal_extraction carries extract provenance")
        if (self.fallback_provenance is not None) != any(s.source == "fallback" for s in self.spans):
            raise ValueError("fallback_provenance is set exactly when a span ran on the fallback")
        keys = [s.span_key for s in self.spans]
        _unique(keys, "span keys")
        if not set(self.carried_forward) <= set(keys):
            raise ValueError("carried-forward spans are listed among the spans")
        return self


class SignalStageOutcome(ContractModel):
    """Whether one planned v2 stage reached the merge (the v2 counterpart of a pass outcome)."""

    stage: SignalStage
    included: bool
    failure_code: Optional[JobErrorCode] = None
    provenance: Optional[SignalStageProvenance] = None
    spans_carried_forward: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _outcome(self):
        if self.included and self.failure_code is not None:
            raise ValueError("an included stage has no failure code")
        if self.provenance is not None and self.provenance.stage != self.stage:
            raise ValueError("a stage outcome carries its own stage's provenance")
        return self


class SegmentationSummary(ContractModel):
    segmenter_version: ShortText
    window_seconds: float = Field(gt=0)
    segments: int = Field(ge=0)
    scored_segments: int = Field(ge=0)
    skipped_unattributed: int = Field(ge=0)
    skipped_system: int = Field(ge=0)
    interpolated_turns: int = Field(ge=0, description="Turns whose cuts were placed by character proportion (no word timestamps).")

    @model_validator(mode="after")
    def _counts(self):
        if self.scored_segments > self.segments:
            raise ValueError("scored segments are segments")
        return self


class ContactSignalsContent(ContractModel):
    """``contact_signals.v1`` (job contact_signals_merge). ``calls.ContactSignalsView`` is this plus
    Store's IDs.

    **v1** (``pipeline: v1``): the lifecycle and resolution passes; partial when a pass or window failed.
    **v2** (``pipeline: v2``, 1.3.0): the three-stage cascade, published into the same group by the
    same merge. ``passes`` is empty and ``stages`` lists the planned stages in order (categorize,
    subcategorize, and extract when an active node has fields or narrow_quote). A v2 result exists
    only when categorize produced its output (otherwise the merge fails), and it is partial when a
    later stage is missing or spans went over the extraction cap, with ``partial_reason`` naming it.
    There is no v1 fallback inside a v2 run; ``pipeline_note`` says when v2 was selected but v1 ran."""

    completeness: Literal["complete", "partial"]
    partial_reason: Optional[ShortText] = None
    signals: List[ContactSignalView]
    passes: List[ContactSignalsPassOutcome]
    transcript_fingerprint: Sha256Digest
    generated_at: Timestamp
    # --- Contact Signals v2 (1.3.0) ---
    pipeline: Literal["v1", "v2"] = Field(default="v1", description="Added in 1.3.0. Which pipeline produced this result.")
    pipeline_note: Optional[ShortText] = Field(default=None, description="Added in 1.3.0. e.g. 'v2 selected; no qualified classifier on this host' on a v1 result.")
    taxonomy: Optional[SignalTaxonomyRef] = Field(default=None, description="Added in 1.3.0; v2 only. The taxonomy version and digest the result was scored with.")
    stages: List[SignalStageOutcome] = Field(default_factory=list, description="Added in 1.3.0; v2 only. The planned stages, in pipeline order.")
    segmentation: Optional[SegmentationSummary] = Field(default=None, description="Added in 1.3.0; v2 only.")
    stage1_digests: Dict[str, Sha256Digest] = Field(default_factory=dict, description="Added in 1.3.0; v2 only. Copied from the categories artifact so Store can label the result outdated at read time.")

    @model_validator(mode="after")
    def _partial(self):
        if (self.completeness == "partial") != bool(self.partial_reason):
            raise ValueError("a partial result says what is missing, and only a partial result does")
        if self.pipeline == "v1":
            if self.completeness == "complete" and not all(p.included for p in self.passes):
                raise ValueError("a complete result includes every pass")
            if self.taxonomy is not None or self.stages or self.segmentation is not None or self.stage1_digests:
                raise ValueError("only a v2 result carries a taxonomy, stages, segmentation or stage-1 digests")
            if any(s.kind is ContactSignalKind.CUSTOM or s.category_id is not None for s in self.signals):
                raise ValueError("a v1 result has built-in kinds only and no v2 hit fields")
            return self
        if self.passes:
            raise ValueError("a v2 result lists stages, not v1 passes")
        if self.taxonomy is None or self.segmentation is None:
            raise ValueError("a v2 result names its taxonomy and segmentation")
        if not set(self.stage1_digests) <= SIGNAL_SCOPE_KEYS:
            raise ValueError("stage1_digests is keyed by speaker scope: AGENT, CALLER or UNKNOWN")
        names = [s.stage for s in self.stages]
        if names[:2] != ["categorize", "subcategorize"] or names != [s for s in SIGNAL_STAGES if s in names]:
            raise ValueError("a v2 result lists categorize, subcategorize and, when planned, extract, once each and in order")
        if not self.stages[0].included:
            raise ValueError("a v2 result is published only when categorize produced its output")
        if self.completeness == "complete" and not all(s.included for s in self.stages):
            raise ValueError("a complete result includes every planned stage")
        if any(s.category_id is None for s in self.signals):
            raise ValueError("every v2 hit names its category")
        return self


# --- Migration -------------------------------------------------------------------------------


class MigrationRecordContent(ContractModel):
    """``migration_record.v1``: the pre-split row an imported record came from, for traceability."""

    source_table: ShortText
    legacy_id: ShortText
    fields: Dict[str, Any] = Field(default_factory=dict, description="The original column values the contract has no field for (e.g. a PRO1_AUTOMATED strategy).")
    imported_at: Timestamp


# ContactSignalView.why refers to SignalHitWhy, defined with the rules-engine models further down.
ContactSignalView.model_rebuild()
ContactSignalsPassContent.model_rebuild()
