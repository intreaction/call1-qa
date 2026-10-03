"""Pydantic schemas and contracts for Call1."""

from __future__ import annotations

import re
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# --- Ingest & S3 Event Models ---

class MinIOBucket(BaseModel):
    name: str
    arn: Optional[str] = None


class MinIOEntity(BaseModel):
    key: str
    size: Optional[int] = None
    eTag: Optional[str] = None
    sequencer: Optional[str] = None
    contentType: Optional[str] = None


class MinIOS3Info(BaseModel):
    s3SchemaVersion: Optional[str] = None
    configurationId: Optional[str] = None
    bucket: MinIOBucket
    object: MinIOEntity


class MinIOEventRecord(BaseModel):
    eventVersion: Optional[str] = None
    eventSource: Optional[str] = None
    awsRegion: Optional[str] = None
    eventTime: Optional[str] = None
    eventName: str
    s3: MinIOS3Info


class MinIOEventPayload(BaseModel):
    """Payload sent by MinIO bucket notification webhooks."""
    EventName: Optional[str] = None
    Key: Optional[str] = None
    Records: List[MinIOEventRecord] = Field(default_factory=list)


# --- Pre-Flight Audio Validation Models ---

class AudioValidationStatus(str, Enum):
    PASSED = "PASSED"
    DISCARD_TOO_SHORT = "DISCARD_TOO_SHORT"
    EXCEEDS_MAX_DURATION = "EXCEEDS_MAX_DURATION"
    CORRUPTED = "CORRUPTED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"


class AudioChannelLayout(str, Enum):
    MONO = "MONO"
    STEREO = "STEREO"
    MULTI_CHANNEL = "MULTI_CHANNEL"
    UNKNOWN = "UNKNOWN"


class ValidationResult(BaseModel):
    """Result of audio validation checks before ASR."""
    is_valid: bool
    status: AudioValidationStatus
    file_path: str
    file_size_bytes: int = 0
    duration_seconds: float = 0.0
    channels: int = 0
    channel_layout: AudioChannelLayout = AudioChannelLayout.UNKNOWN
    sample_rate: int = 0
    codec: str = "unknown"
    error_message: Optional[str] = None
    quarantine_path: Optional[str] = None


# --- VAD & Audio Timing Models ---

class VADSegment(BaseModel):
    """Continuous speech segment detected by VAD."""
    start_time: float
    end_time: float
    duration: float
    channel: int = 0


class VADMetrics(BaseModel):
    """Telephony conversational timing metrics derived from VAD."""
    total_speech_duration: float = 0.0
    total_silence_duration: float = 0.0
    silence_ratio: float = 0.0  # Dead-air percentage (0.0 - 1.0)
    overtalk_duration: float = 0.0  # Simultaneous agent & caller speech
    overtalk_ratio: float = 0.0
    segments: List[VADSegment] = Field(default_factory=list)


# --- ASR & Transcription Models ---

class SpeakerRole(str, Enum):
    AGENT = "AGENT"
    CALLER = "CALLER"
    SYSTEM = "SYSTEM"
    UNKNOWN = "UNKNOWN"


class WordTimestamp(BaseModel):
    """Individual token/word aligned timestamp from Whisper."""
    word: str
    start_time: float
    end_time: float
    probability: float = 1.0


class NumericEntityType(str, Enum):
    CURRENCY = "CURRENCY"
    PERCENTAGE = "PERCENTAGE"
    DATE = "DATE"
    PHONE_NUMBER = "PHONE_NUMBER"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    DURATION = "DURATION"
    GENERIC_NUMBER = "GENERIC_NUMBER"


class NumericEntity(BaseModel):
    """Normalized numeric reference extracted from spoken text."""
    raw_text: str
    normalized_value: float | str
    entity_type: NumericEntityType
    start_time: float
    end_time: float
    unit: Optional[str] = None


class TranscriptTurn(BaseModel):
    """Individual spoken utterance in a conversation."""
    turn_id: int
    speaker: SpeakerRole
    start_time: float
    end_time: float
    text: str
    raw_text: Optional[str] = None
    channel: Optional[int] = None
    speaker_cluster: Optional[str] = None
    confidence: Optional[float] = None
    word_timestamps: Optional[List[WordTimestamp]] = None
    # Acoustic Tone (WavLM: valence & arousal)
    tone_score: Optional[float] = None  # Valence: -1.0 (negative/tense) to +1.0 (positive/calm)
    tone_arousal: Optional[float] = None  # Arousal: 0.0 (calm/subdued) to 1.0 (excited/agitated)
    tone_label: Optional[str] = None
    # Text Sentiment (RoBERTa/DeBERTa)
    text_sentiment: Optional[float] = None  # Polarity: -1.0 to +1.0
    text_sentiment_label: Optional[str] = None  # POSITIVE, NEUTRAL, NEGATIVE
    text_analysis: Optional[Dict[str, Any]] = None  # Model, revision, probabilities, status
    # Multimodal Sentiment Divergence (|tone - sentiment|)
    sentiment_divergence: Optional[float] = None
    # Extracted Numeric References
    numeric_entities: Optional[List[NumericEntity]] = None
    # Semantic embedding vector (e.g. 384-dim BGE-small)
    embedding: Optional[List[float]] = None


class ToneBlock(BaseModel):
    """One speaker's acoustic affect in a fixed seven-second call-time window."""
    analysis_version: str = "seven-second-speaker-v1"
    block_id: int
    speaker: SpeakerRole
    start_time: float
    end_time: float
    channel: Optional[int] = None
    turn_ids: List[int] = Field(default_factory=list)
    speech_seconds: float = 0
    speech_intervals: List[Tuple[float, float]] = Field(default_factory=list)
    status: str = "NO_SPEECH"
    valence: Optional[float] = Field(default=None, ge=0, le=1)
    arousal: Optional[float] = Field(default=None, ge=0, le=1)
    dominance: Optional[float] = Field(default=None, ge=0, le=1)
    emotion: Optional[str] = None
    emotion_probabilities: Optional[Dict[str, float]] = None
    tonal_score: Optional[float] = Field(default=None, ge=0, le=1)
    tonal_label: Optional[str] = None
    model: str = "MERaLiON/MERaLiON-SER-v1"
    revision: str = "7e3ee6fa4534dea8316e8ca43e377e2fbb58496b"

class CallTranscript(BaseModel):
    """Full timestamped transcript for a call interaction."""
    call_id: str
    turns: List[TranscriptTurn] = Field(default_factory=list)
    tone_blocks: List[ToneBlock] = Field(default_factory=list)
    duration_seconds: float
    is_redacted: bool = False
    redaction_log: List[Dict[str, Any]] = Field(default_factory=list)
    vad_metrics: Optional[VADMetrics] = None
    avg_agent_tone: Optional[float] = None
    avg_caller_sentiment: Optional[float] = None
    cohesive_tone_score: Optional[float] = None
    @property
    def full_text(self) -> str:
        return "\n".join(f"{t.speaker.value}: {t.text}" for t in self.turns)

    @property
    def agent_text(self) -> str:
        return "\n".join(t.text for t in self.turns if t.speaker == SpeakerRole.AGENT)


# --- Semantic Search Models ---

class SemanticSearchQuery(BaseModel):
    query: str
    top_k: int = 5
    speaker_filter: Optional[SpeakerRole] = None
    call_id: Optional[str] = None
    min_score: float = 0.3


class SemanticSearchResult(BaseModel):
    turn_id: int
    call_id: str
    speaker: SpeakerRole
    start_time: float
    end_time: float
    text: str
    similarity_score: float
    tone_score: Optional[float] = None
    text_sentiment: Optional[float] = None
# --- Rubric & Scoring Models ---

class VerdictStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    FLAGGED = "FLAGGED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class CheckType(str, Enum):
    """The kind of evaluation a modular rubric check performs."""
    SENTIMENT_METRIC = "sentiment_metric"
    PHRASE_ANY = "phrase_any"
    PHRASE_ALL = "phrase_all"
    PHRASE_NONE = "phrase_none"
    CONDITIONAL_RESPONSE = "conditional_response"
    SEMANTIC_JUDGEMENT = "semantic_judgement"
    CUSTOM_REGEX = "custom_regex"


class RubricCheck(BaseModel):
    """Modular, un-prescribed check definition for a rubric criterion.

    A criterion carries one of these instead of (or alongside) the legacy
    rule_type/parameters pair. The check describes *what to look for* in the
    transcript; the evaluator dispatches on `check_type`. Every field is
    optional so a creator composes only what their check needs.
    """
    check_type: CheckType = CheckType.PHRASE_ANY
    metric: Literal["text_polarity", "valence", "arousal", "dominance"] = "text_polarity"
    aggregation: Literal["mean", "min", "max"] = "mean"
    comparison: Literal["gte", "lte"] = "gte"
    metric_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    min_samples: int = Field(default=2, ge=1, le=10000)
    min_coverage: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_metric_threshold(self):
        if self.check_type == CheckType.SENTIMENT_METRIC and self.metric != "text_polarity" and self.metric_threshold < 0:
            raise ValueError("Tone thresholds must be between 0 and 1.")
        return self

    phrases: List[str] = Field(default_factory=list)
    #: Similarity a phrase must reach to count as a match, 0-100.
    threshold: int = Field(default=80, ge=0, le=100)
    #: >0 = first N seconds of the call, <0 = last N seconds, None = whole call.
    window_seconds: Optional[float] = None
    #: Whose speech the check applies to. None = both speakers.
    speaker: Optional[SpeakerRole] = SpeakerRole.AGENT
    #: Phrases that trigger a conditional response (conditional_response).
    trigger_phrases: List[str] = Field(default_factory=list)
    #: Phrases the agent must say in response (conditional_response).
    response_phrases: List[str] = Field(default_factory=list)
    #: Plain-language guidance for semantic_judgement.
    pass_when: Optional[str] = None
    fail_when: Optional[str] = None
    not_applicable_when: Optional[str] = None
    requires_policy: bool = False
    policy_context: Optional[str] = Field(default=None, max_length=12000)
    primary_model_id: Optional[str] = None
    # None inherits the default; "none" explicitly disables escalation.
    escalation_model_id: Optional[str] = None
    escalation_when: List[Literal["needs_review", "invalid_answer", "provider_error", "always"]] = Field(
        default_factory=lambda: ["needs_review", "invalid_answer"])
    #: Regular expression for custom_regex.
    pattern: Optional[str] = None
    #: Legacy rule_type this check was derived from, if any. Set by the
    #: legacy→modular adapter so the evaluator can preserve legacy reasoning.
    legacy_rule: Optional[str] = None


def legacy_rule_to_check(rule_type: str, parameters: Dict[str, Any]) -> RubricCheck:
    """Convert a legacy rule_type/parameters pair into a modular RubricCheck.

    Legacy criteria were authored with `rule_type` (e.g. "compliance_phrase")
    and a `parameters` dict. This maps each legacy rule onto the equivalent
    modular check so every criterion can evaluate through the single modular
    path. Unknown or "custom" rules fall back to a bare PHRASE_ANY check.
    """
    if rule_type == "timing_greeting":
        max_seconds = float(parameters.get("max_seconds", 15.0))
        return RubricCheck(
            check_type=CheckType.PHRASE_ANY,
            phrases=[
                "hello", "hi", "good morning", "good afternoon", "good evening",
                "thank you for calling", "thanks for calling", "welcome to",
            ],
            threshold=80,
            window_seconds=max_seconds,
            speaker=SpeakerRole.AGENT,
            legacy_rule=rule_type,
        )
    if rule_type == "empathy_sentiment":
        return RubricCheck(
            check_type=CheckType.CONDITIONAL_RESPONSE,
            trigger_phrases=[
                "frustrated", "upset", "angry", "terrible", "awful", "unacceptable",
                "problem", "issue", "charged twice", "overcharged", "dispute",
                "not working", "broken", "ridiculous", "scam", "waste of time",
            ],
            response_phrases=[
                "sorry to hear", "understand", "apologize", "let me help",
                "completely understand", "happy to help", "stressful",
                "take care of this", "get this sorted", "my apologies",
            ],
            threshold=80,
            speaker=SpeakerRole.AGENT,
            legacy_rule=rule_type,
        )
    if rule_type == "compliance_phrase":
        target = parameters.get(
            "target_phrase",
            "This call may be monitored or recorded for quality assurance.",
        )
        threshold = float(parameters.get("fuzzy_threshold", 0.40))
        required_keywords = parameters.get("required_keywords") or [
            "recorded", "monitored", "quality", "training",
        ]
        return RubricCheck(
            check_type=CheckType.PHRASE_ANY,
            phrases=[target] + list(required_keywords),
            threshold=int(threshold * 100),
            speaker=SpeakerRole.AGENT,
            legacy_rule=rule_type,
        )
    if rule_type == "auth_verification":
        return RubricCheck(
            check_type=CheckType.PHRASE_ANY,
            phrases=[
                "verify", "authenticate", "account number", "date of birth", "dob",
                "ssn", "social security", "pin", "security question",
                "billing address", "zip code",
            ],
            threshold=80,
            speaker=SpeakerRole.AGENT,
            legacy_rule=rule_type,
        )
    if rule_type == "hold_etiquette":
        return RubricCheck(
            check_type=CheckType.CONDITIONAL_RESPONSE,
            trigger_phrases=[
                "place you on hold", "put you on hold", "one moment",
                "bear with me", "check on that",
            ],
            response_phrases=[
                "thank you for holding", "thanks for holding",
                "appreciate your patience", "thanks for waiting",
            ],
            threshold=80,
            speaker=SpeakerRole.AGENT,
            legacy_rule=rule_type,
        )
    if rule_type == "closing_etiquette":
        return RubricCheck(
            check_type=CheckType.PHRASE_ANY,
            phrases=[
                "anything else", "help you with", "further assistance", "other questions",
                "thank you for calling", "have a great day", "have a wonderful day",
                "have a good day", "goodbye", "take care", "thanks for reaching out",
            ],
            threshold=80,
            speaker=SpeakerRole.AGENT,
            legacy_rule=rule_type,
        )
    if rule_type == "keyword_regex":
        pattern = parameters.get("pattern", "")
        return RubricCheck(
            check_type=CheckType.CUSTOM_REGEX,
            pattern=pattern or None,
            speaker=SpeakerRole.AGENT,
            legacy_rule=rule_type,
        )
    return RubricCheck(
        check_type=CheckType.PHRASE_ANY,
        phrases=[],
        threshold=80,
        speaker=SpeakerRole.AGENT,
        legacy_rule=rule_type,
    )


class RubricCriterion(BaseModel):
    """Specific compliance or quality requirement to score."""
    criterion_id: str
    name: str
    category: str = "COMPLIANCE"
    description: str = ""
    weight: float = 25.0
    critical: bool = False  # If critical fails, overall call fails compliance
    rule_type: str = "custom"  # timing_greeting, empathy_sentiment, compliance_phrase, auth_verification, closing_etiquette, hold_etiquette, problem_resolution, keyword_regex, custom
    parameters: Dict[str, Any] = Field(default_factory=dict)
    #: Modular check definition. When set, the evaluator dispatches on
    #: `check.check_type`; rule_type/parameters remain for legacy criteria.
    check: Optional[RubricCheck] = None

    @model_validator(mode="after")
    def _ensure_check(self) -> "RubricCriterion":
        """Auto-convert legacy rule_type/parameters into a modular check.

        `check` is the primary definition. When it is absent but legacy
        `rule_type`/`parameters` are provided, derive a RubricCheck so the
        criterion always evaluates through the modular path.
        """
        if self.check is None and (self.rule_type != "custom" or self.parameters):
            self.check = legacy_rule_to_check(self.rule_type, self.parameters)
        return self


class RubricDefinition(BaseModel):
    """Collection of criteria making up a QA scorecard."""
    rubric_id: str
    name: str
    description: str
    category: str = "GENERAL"  # GENERAL, BANKING, CUSTOMER_CARE, COLLECTIONS, HEALTHCARE
    pass_threshold: float = 80.0
    criteria: List[RubricCriterion] = Field(default_factory=list)

class ModelAttempt(BaseModel):
    model_id: str
    model_name: str
    model: str
    source: str
    status: VerdictStatus
    reasoning: str
    quoted_evidence: Optional[str] = None
    trigger: Optional[str] = None
    latency_ms: int = 0
    usage: Dict[str, int] = Field(default_factory=dict)


class RubricVerdict(BaseModel):
    """Evaluated verdict for a single rubric criterion."""
    criterion_id: str
    criterion_name: str
    status: VerdictStatus
    confidence: float = Field(ge=0.0, le=1.0)
    quoted_evidence: Optional[str] = None
    speaker: SpeakerRole = SpeakerRole.AGENT
    timestamp_range: Optional[Tuple[float, float]] = None
    reasoning: str
    hallucination_detected: bool = False
    model_attempts: List[ModelAttempt] = Field(default_factory=list)


class CallEvaluationResult(BaseModel):
    """Overall QA score and itemized verdicts for a call."""
    call_id: str
    rubric_id: str
    overall_score: float = Field(ge=0.0, le=100.0)
    passed: bool
    critical_failure: bool = False
    requires_human_review: bool = False
    escalation_reasons: List[str] = Field(default_factory=list)
    verdicts: List[RubricVerdict] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Persisted App Settings ---

class RedactionSettings(BaseModel):
    """Redaction toggles. Text and audio are independent; enabling text does not
    imply audio masking and vice versa. These mask sensitive content at every
    API boundary (transcript, evidence, reasoning, search, summary, audio)."""
    model_config = ConfigDict(extra="forbid")

    text: bool = True
    audio: bool = True
    pii_patterns: bool = True  # Regex SSN / card / phone / PIN masking in addition to numeric entities


class FeatureSettings(BaseModel):
    """Feature switches. Only switches backed by real behavior are exposed."""
    model_config = ConfigDict(extra="forbid")

    tone: bool = True
    sentiment: bool = True
    numeric: bool = True
    semantic_search: bool = True
    summary: bool = True
    contact_signals: bool = True

class SummaryProviderSettings(BaseModel):
    """Local inference configuration for per-call summaries.

    The endpoint is restricted to loopback addresses (127.0.0.1, ::1,
    localhost) with no credentials, query, or fragment, so call data can never
    be sent off the appliance. provider is a literal; only the local Ollama
    client is implemented. batch_turns is the chunk size for long calls
    (full-call coverage via chunk synthesis), not a truncation limit.
    """
    model_config = ConfigDict(extra="forbid")

    model_id: Optional[str] = None  # Registry reference; None preserves legacy local configuration.
    provider: Literal["ollama", "mlx"] = "ollama"
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "gemma4:e2b"
    batch_turns: int = Field(default=60, ge=1, le=500)
    max_tokens: int = Field(default_factory=lambda: 768 if os.getenv("CALL1_BACKEND") == "mlx" else 400, ge=64, le=2048)

    @field_validator("endpoint")
    @classmethod
    def _validate_loopback_endpoint(cls, v: str) -> str:
        """Reject any endpoint that is not a loopback http(s) URL.

        Delegates to the shared local_http validator, which also rejects
        malformed ports and non-loopback hosts (no DNS rebinding to public
        addresses).
        """
        from call1.local_http import validate_loopback_url

        return validate_loopback_url(v)

    @field_validator("model")
    @classmethod
    def _validate_model_nonblank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model must not be blank")
        return v


class QuestionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    name: str = Field(min_length=1, max_length=120)
    source: Literal["internal", "pro1", "call1", "byok"] = "byok"
    model: str = Field(min_length=1, max_length=200)
    endpoint: Optional[str] = None
    # Only dedicated model credentials may be forwarded to configured endpoints.
    api_key_env: Optional[str] = Field(default=None, pattern=r"^CALL1_MODEL_KEY_[A-Z0-9_]+$")
    enabled: bool = True
    max_tokens: int = Field(default=768, ge=64, le=8192)
    timeout_seconds: int = Field(default=90, ge=5, le=300)
    json_mode: bool = True
    schema_mode: bool = True
    local_model_id: Optional[Literal["gemma4-e4b", "gemma4-12b"]] = None

    @model_validator(mode="before")
    @classmethod
    def migrate_included_profile(cls, values):
        # The stable ID keeps all question/default references intact on upgrade.
        if (isinstance(values, dict) and values.get("id") == "call1-bundled"
                and values.get("source") == "internal"
                and values.get("model") in ("Qwen3-4B-Instruct-2507-4bit", "gemma-4-e4b-it-4bit")):
            values = dict(values, model="gemma-4-e2b-it-4bit")
            if values.get("name") in ("Call1 Included · Qwen3 4B", "Call1 Included · Gemma 4 E4B"):
                values["name"] = "Call1 Included · Gemma 4 E2B"
        return values

    @model_validator(mode="after")
    def validate_connection(self):
        from urllib.parse import urlsplit
        if not self.name.strip() or not self.model.strip():
            raise ValueError("Model name and model ID must not be blank")
        if self.local_model_id:
            from call1.model_catalog import LOCAL_PACKS
            if (self.source != "call1" or self.model != LOCAL_PACKS[self.local_model_id]["model"]
                    or self.endpoint or self.api_key_env or self.id in ("call1-bundled", "none")):
                raise ValueError("Local model packs use their catalog model without an endpoint or credential")
            return self
        if self.source == "internal":
            if (self.id != "call1-bundled" or self.model != "gemma-4-e2b-it-4bit"
                    or self.endpoint or self.api_key_env or not self.enabled):
                raise ValueError("The included model profile cannot be replaced or disabled")
        else:
            if self.id in ("call1-bundled", "none"):
                raise ValueError("Reserved model ID")
            parts = urlsplit(self.endpoint or "")
            if (not parts.hostname or parts.scheme not in ("https", "http")
                    or parts.username is not None or parts.password is not None
                    or parts.query or parts.fragment):
                raise ValueError("Provide an HTTP(S) API base URL without credentials, query, or fragment")
            if parts.port is not None and not 1 <= parts.port <= 65535:
                raise ValueError("Invalid endpoint port")
            if parts.scheme == "http" and parts.hostname not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError("Remote model endpoints require HTTPS")
            self.endpoint = self.endpoint.rstrip("/")
        return self


class QuestionModelsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: List[QuestionModel] = Field(default_factory=lambda: [QuestionModel(
        id="call1-bundled", name="Call1 Included · Gemma 4 E2B", source="internal",
        model="gemma-4-e2b-it-4bit")])
    primary_model_id: str = "call1-bundled"
    escalation_model_id: Optional[str] = None
    allow_external: bool = False

    @model_validator(mode="after")
    def validate_registry(self):
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)) or "call1-bundled" not in ids:
            raise ValueError("Model IDs must be unique and the included model must remain registered")
        enabled = {model.id for model in self.models if model.enabled}
        if self.primary_model_id not in enabled or (self.escalation_model_id and self.escalation_model_id not in enabled):
            raise ValueError("Default models must reference enabled models")
        if self.primary_model_id == self.escalation_model_id:
            raise ValueError("Primary and escalation defaults must be different")
        return self


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    redaction: RedactionSettings = RedactionSettings()
    features: FeatureSettings = FeatureSettings()
    @model_validator(mode="before")
    @classmethod
    def migrate_packaged_summary(cls, values):
        if isinstance(values, dict):
            summary = values.get("summary")
            if (isinstance(summary, dict) and summary.get("provider") == "mlx"
                    and summary.get("model") in ("Qwen3-4B-Instruct-2507-4bit", "gemma-4-e4b-it-4bit")):
                values = dict(values, summary=dict(summary, model="gemma-4-e2b-it-4bit"))
        return values

    summary: SummaryProviderSettings = SummaryProviderSettings()
    question_models: QuestionModelsSettings = Field(default_factory=QuestionModelsSettings)

    @model_validator(mode="after")
    def validate_summary_model(self):
        if self.summary.model_id is not None and not any(
                m.id == self.summary.model_id and m.enabled for m in self.question_models.models):
            raise ValueError("Summary model must reference an enabled model in Models settings")
        return self



class ModelStatus(BaseModel):
    configured: str  # "available" | "unavailable" | "unknown"
    error: Optional[str] = None


class SettingsResponse(BaseModel):
    settings: AppSettings
    available_models: List[str] = Field(default_factory=list)
    model_status: ModelStatus


# --- Per-Call LLM Summary ---

class SummaryStatus(str, Enum):
    MISSING = "MISSING"
    GENERATING = "GENERATING"
    GENERATED = "GENERATED"
    STALE = "STALE"
    ERROR = "ERROR"
    DISABLED = "DISABLED"


class RubricHighlight(BaseModel):
    """Deterministic rubric indication attached to a summary (not model narrative)."""
    criterion_id: str
    criterion_name: str
    status: VerdictStatus
    note: str


class CallSummary(BaseModel):
    """LLM-generated narrative grounded in the (redacted) transcript plus
    deterministic rubric highlights."""
    narrative: str
    key_points: List[str] = Field(default_factory=list)
    rubric_highlights: List[RubricHighlight] = Field(default_factory=list)
    grounding: Dict[str, Any] = Field(default_factory=dict)


class CallSummaryResponse(BaseModel):
    call_id: str
    status: SummaryStatus
    summary: Optional[CallSummary] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    local: bool = True
    generated_at: Optional[datetime] = None
    stale_reason: Optional[str] = None
    error_message: Optional[str] = None
    regenerating: bool = False


# --- Per-Rubric Metrics ---

class CriterionCounts(BaseModel):
    PASS: int = 0
    FAIL: int = 0
    FLAGGED: int = 0
    NOT_APPLICABLE: int = 0
    total: int = 0
    pass_rate_pct: float = 0.0


class CriterionMetric(BaseModel):
    criterion_id: str
    criterion_name: str
    category: str
    counts: CriterionCounts


class DailyMetric(BaseModel):
    date: str  # YYYY-MM-DD
    evaluated: int
    mean_score: float


class RubricMetricsResponse(BaseModel):
    rubric_id: str
    rubric_name: str
    total_calls_evaluated: int
    average_score: float
    pass_rate_pct: float
    criteria: List[CriterionMetric] = Field(default_factory=list)
    daily: List[DailyMetric] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Review Queue & Work Distribution ---

class ReviewStream(str, Enum):
    """Why a call entered the review queue."""
    TRIAGE = "TRIAGE"  # Machine flagged: critical breach or low confidence
    AUDIT_SAMPLE = "AUDIT_SAMPLE"  # Random % of confident passes (false negatives / drift)
    MANDATE = "MANDATE"  # Business rules: VIP/Disputes, high value, campaigns
    CALIBRATION = "CALIBRATION"  # Multi-reviewer blind scoring


class DistributionStrategy(str, Enum):
    """How unassigned queue items are routed to auditors."""
    ROUND_ROBIN = "ROUND_ROBIN"  # Even cyclic distribution
    LEAST_OUTSTANDING = "LEAST_OUTSTANDING"  # Smallest open backlog
    SKILL_MATCHED = "SKILL_MATCHED"  # Auditors with matching skills
    UNASSIGNED_CLAIM = "UNASSIGNED_CLAIM"  # Auditors pull the next call
    PRO1_AUTOMATED = "PRO1_AUTOMATED"  # Automated escalation to Pro1 larger model ($0.10/question)

class QueueStatus(str, Enum):
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    OVERRIDDEN = "OVERRIDDEN"
    RELEASED = "RELEASED"


class Auditor(BaseModel):
    """A human QA auditor who reviews queue items."""
    id: str
    name: str
    email: str
    role: str
    skills: List[str] = Field(default_factory=list)
    capacity_weight: float = Field(default=1.0, ge=0.1)
    active: bool = True
    pending_count: int = 0


class QueueRule(BaseModel):
    """Business rule deciding which calls enter the review queue and how they
    are distributed to auditors. Every queue item carries the rule that put
    it there, so the 'why' is always explainable."""
    id: str
    name: str
    stream_type: ReviewStream
    enabled: bool = True
    priority: int = Field(default=100, ge=0)  # Lower = higher priority
    distribution_strategy: DistributionStrategy = DistributionStrategy.UNASSIGNED_CLAIM
    target_skills: List[str] = Field(default_factory=list)
    critical_failure_only: bool = False
    low_confidence_only: bool = False
    sampling_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    target_domains: List[str] = Field(default_factory=list)
    target_agents: List[str] = Field(default_factory=list)
    description: str = ""


class QueueItem(BaseModel):
    """A single call in the review queue with its stream, rule, and reason."""
    call_id: str
    rule_id: str
    rule_name: str
    stream_type: ReviewStream
    reason: str
    urgency_score: float = Field(default=0.0, ge=0.0)
    assigned_to: Optional[str] = None
    assigned_auditor_name: Optional[str] = None
    status: QueueStatus = QueueStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_at: Optional[datetime] = None
    reviewer_notes: Optional[str] = None
    # Call metadata snapshot for the reviewer workbench
    agent_id: Optional[str] = None
    overall_score: Optional[float] = None
    critical_failure: bool = False
    duration_seconds: Optional[float] = None
    # Pro1 audit resolution details (if resolved or analyzed by Pro1)
    pro1_resolution: Optional[Dict[str, Any]] = None
