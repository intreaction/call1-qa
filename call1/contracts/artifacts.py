"""Artifacts: immutable, versioned, checksummed content owned by Store.

Artifacts are addressed by ID and checksum, never by path. Large content lives in Store-managed
object storage behind short-lived, object-scoped transfer URLs; small JSON travels inline. The
storage reference itself is Store-internal and never appears in the API, so no client can assume
where Store keeps bytes (location neutrality, HostedV2.md 1.4).

Versions and supersession are per logical **slot**: ``(conversation, kind, slot)``. A slot is a
small label the producer sets, for example the criterion ID of a QA assessment, ``segment:3`` of a
summary segment, ``lifecycle:0`` of a contact-signal pass window, or ``""`` for a per-call
singleton such as the transcript. A job's output is **linked** only when the completion that
references it commits; only then does it get a version and supersede the previous linked artifact
of its slot. A committed artifact no completion links is an orphan: it is excluded from listings by
default, supersedes nothing, and Store deletes it after ``ORPHAN_ARTIFACT_RETENTION``.

A draft-test graph writes every artifact under its own slot namespace, ``draft:<request_id>:<slot>``
(``draft_test_slot``), so testing an unpublished rubric on a call never versions or supersedes
that call's live artifacts. Default listings leave those slots out.

JSON content has one canonical form: ``canonical_json`` of the content model's full dump
(``model_dump(mode="json")``, every default and null present). An inline payload or uploaded JSON
body in any other form is rejected, so the producer's checksum and Store's always agree
(``canonical_content``).

Attestation evidence and trust-anchor bundles are **global** artifacts: they belong to no
conversation, carry no customer data, and are addressed by digest.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import Field, model_validator

from . import contents, rubrics, signals
from .common import ArtifactRef, ContractModel, PageQuery, ResourceId, Sha256Digest, ShortText, Timestamp, canonical_json



class ArtifactKind(str, Enum):
    SOURCE_AUDIO = "source_audio"
    REDACTED_AUDIO = "redacted_audio"
    VALIDATION_REPORT = "validation_report"
    VAD_METRICS = "vad_metrics"
    TRANSCRIPT = "transcript"
    SPEAKER_ATTRIBUTION = "speaker_attribution"
    TONE_BLOCKS = "tone_blocks"
    TEXT_SENTIMENT = "text_sentiment"
    EMBEDDINGS = "embeddings"
    ENRICHMENT = "enrichment"
    PII_FINDINGS = "pii_findings"
    """Added in 1.2.0: the model PII layer's findings for one transcript revision (job enrichment)."""
    RUBRIC_SNAPSHOT = "rubric_snapshot"
    PROMPT_INPUT = "prompt_input"
    QA_ASSESSMENT = "qa_assessment"
    QA_VERDICT = "qa_verdict"
    QA_SCORECARD = "qa_scorecard"
    SUMMARY_SEGMENT = "summary_segment"
    SUMMARY_SYNTHESIS = "summary_synthesis"
    SUMMARY = "summary"
    CONTACT_SIGNALS_PASS = "contact_signals_pass"
    CONTACT_SIGNALS = "contact_signals"
    ATTESTATION_EVIDENCE = "attestation_evidence"
    TRUST_ANCHOR_BUNDLE = "trust_anchor_bundle"
    MIGRATION_RECORD = "migration_record"
    SIGNAL_TAXONOMY_SNAPSHOT = "signal_taxonomy_snapshot"
    """Added in 1.3.0: the taxonomy and settings a Contact Signals v2 graph runs with (Store-minted, derived)."""
    SIGNAL_CATEGORIES = "signal_categories"
    """Added in 1.3.0: stage 1's segment grid, sparse probabilities and spans (derived, no text)."""
    SIGNAL_SUBCATEGORIES = "signal_subcategories"
    """Added in 1.3.0: stage 2's per-span decisions (derived, no text)."""
    SIGNAL_EXTRACTION = "signal_extraction"
    """Added in 1.3.0: stage 3's fields and narrowed quotes (masked text)."""
    ASR_BASE_TRANSCRIPT = "asr_base_transcript"
    """Added in 1.3.0 (decision 33): the base engine's (Parakeet's) own transcript when the asr job ran
    dual transcription (optional output ``base_transcript``; ``transcript.v1``, raw)."""
    ASR_VOCABULARY_PASS = "asr_vocabulary_pass"
    """Added in 1.3.0 (decision 33): the raw vocabulary-prompted Whisper pass, a candidate finder only
    (optional output ``vocabulary_pass``; raw)."""


GLOBAL_ARTIFACT_KINDS = frozenset({ArtifactKind.ATTESTATION_EVIDENCE, ArtifactKind.TRUST_ANCHOR_BUNDLE})
"""Kinds that belong to no conversation. Created through their own upload routes, addressed by digest."""

STORE_MINTED_KINDS = frozenset({ArtifactKind.RUBRIC_SNAPSHOT, ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT})
"""Kinds only Store creates. A published rubric version's snapshot is minted by ``mintRubricSnapshot``
(``POST /conversations/{id}/rubric-snapshots``), a draft's by ``testRubricDraft``. Since 1.3.0 a
published signal taxonomy version's snapshot is minted by ``mintSignalTaxonomySnapshot``
(``POST /conversations/{id}/signal-taxonomy-snapshots``) and a preview's by ``createSignalPreview``.
Process reads them like any other input and never uploads one, so every snapshot is Store's copy of
a stored rubric or taxonomy."""


class Sensitivity(str, Enum):
    """Recorded on every artifact so routing policy can enforce what a route may receive."""

    RAW = "raw"
    """Contains unmasked customer content (audio, unredacted text)."""
    MASKED = "masked"
    """Customer text after masking. Masking is not exhaustive; still customer data."""
    DERIVED = "derived"
    """Customer-derived with no transcript text: metrics, verdict statuses, timings."""
    NON_CUSTOMER = "non_customer"
    """No customer data at all, e.g. attestation evidence, collateral and trust anchors."""


ARTIFACT_CONTENT_CONTRACTS: Dict[ArtifactKind, str] = {
    ArtifactKind.SOURCE_AUDIO: "audio.v1",
    ArtifactKind.REDACTED_AUDIO: "audio.v1",
    ArtifactKind.VALIDATION_REPORT: "validation_report.v1",
    ArtifactKind.VAD_METRICS: "vad_metrics.v1",
    ArtifactKind.TRANSCRIPT: "transcript.v1",
    ArtifactKind.SPEAKER_ATTRIBUTION: "speaker_attribution.v1",
    ArtifactKind.TONE_BLOCKS: "tone_blocks.v1",
    ArtifactKind.TEXT_SENTIMENT: "text_sentiment.v1",
    ArtifactKind.EMBEDDINGS: "embeddings.v1",
    ArtifactKind.ENRICHMENT: "enrichment.v1",
    ArtifactKind.PII_FINDINGS: "pii_findings.v1",
    ArtifactKind.RUBRIC_SNAPSHOT: "rubric_snapshot.v1",
    ArtifactKind.PROMPT_INPUT: "prompt_input.v1",
    ArtifactKind.QA_ASSESSMENT: "qa_assessment.v1",
    ArtifactKind.QA_VERDICT: "qa_verdict.v1",
    ArtifactKind.QA_SCORECARD: "qa_scorecard.v1",
    ArtifactKind.SUMMARY_SEGMENT: "summary_segment.v1",
    ArtifactKind.SUMMARY_SYNTHESIS: "summary_synthesis.v1",
    ArtifactKind.SUMMARY: "summary.v1",
    ArtifactKind.CONTACT_SIGNALS_PASS: "contact_signals_pass.v1",
    ArtifactKind.CONTACT_SIGNALS: "contact_signals.v1",
    ArtifactKind.ATTESTATION_EVIDENCE: "attestation_evidence.v1",
    ArtifactKind.TRUST_ANCHOR_BUNDLE: "trust_anchor_bundle.v1",
    ArtifactKind.MIGRATION_RECORD: "migration_record.v1",
    ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT: "signal_taxonomy_snapshot.v1",
    ArtifactKind.SIGNAL_CATEGORIES: "signal_categories.v1",
    ArtifactKind.SIGNAL_SUBCATEGORIES: "signal_subcategories.v1",
    ArtifactKind.SIGNAL_EXTRACTION: "signal_extraction.v1",
    ArtifactKind.ASR_BASE_TRANSCRIPT: "transcript.v1",
    ArtifactKind.ASR_VOCABULARY_PASS: "asr_vocabulary_pass.v1",
}
"""The content contract per kind. JSON contracts have a model in ``ARTIFACT_CONTENT_MODELS``."""

ARTIFACT_CONTENT_MODELS: Dict[str, Optional[Type[ContractModel]]] = {
    "audio.v1": None,
    "validation_report.v1": contents.AudioValidationContent,
    "vad_metrics.v1": contents.VadMetricsContent,
    "transcript.v1": contents.TranscriptContent,
    "speaker_attribution.v1": contents.SpeakerAttributionContent,
    "tone_blocks.v1": contents.ToneBlocksContent,
    "text_sentiment.v1": contents.TextSentimentContent,
    "embeddings.v1": contents.EmbeddingsContent,
    "enrichment.v1": contents.EnrichmentContent,
    "pii_findings.v1": contents.PiiFindingsContent,
    "rubric_snapshot.v1": rubrics.RubricSnapshotContent,
    "prompt_input.v1": contents.PromptInputContent,
    "qa_assessment.v1": contents.QaAssessmentContent,
    "qa_verdict.v1": contents.QaVerdictContent,
    "qa_scorecard.v1": contents.QaScorecardContent,
    "summary_segment.v1": contents.SummarySegmentContent,
    "summary_synthesis.v1": contents.SummarySynthesisContent,
    "summary.v1": contents.SummaryContent,
    "contact_signals_pass.v1": contents.ContactSignalsPassContent,
    "contact_signals.v1": contents.ContactSignalsContent,
    "attestation_evidence.v1": None,
    "trust_anchor_bundle.v1": None,
    "migration_record.v1": contents.MigrationRecordContent,
    "signal_taxonomy_snapshot.v1": signals.SignalTaxonomySnapshotContent,
    "signal_categories.v1": contents.SignalCategoriesContent,
    "signal_subcategories.v1": contents.SignalSubcategoriesContent,
    "signal_extraction.v1": contents.SignalExtractionContent,
    "asr_vocabulary_pass.v1": contents.AsrVocabularyPassContent,
}
"""JSON content contracts map to their model; ``None`` marks an opaque byte format: audio (the
container the source arrived in, WAV, MP3, FLAC, OGG or M4A, per ``content_type``), the Pro1
evidence bundle (Pro1ConfidentialInference.md 2.2; opaque to Store, verified by Process and by
``scripts/verify_attestation.py``) and the trust-anchor bundle (the verifier's format)."""

ALLOWED_SENSITIVITY: Dict[ArtifactKind, frozenset] = {
    ArtifactKind.SOURCE_AUDIO: frozenset({Sensitivity.RAW}),
    ArtifactKind.REDACTED_AUDIO: frozenset({Sensitivity.MASKED}),
    ArtifactKind.ATTESTATION_EVIDENCE: frozenset({Sensitivity.NON_CUSTOMER}),
    ArtifactKind.TRUST_ANCHOR_BUNDLE: frozenset({Sensitivity.NON_CUSTOMER}),
    ArtifactKind.VAD_METRICS: frozenset({Sensitivity.DERIVED}),
    ArtifactKind.VALIDATION_REPORT: frozenset({Sensitivity.DERIVED}),
    ArtifactKind.EMBEDDINGS: frozenset({Sensitivity.DERIVED}),
    ArtifactKind.PII_FINDINGS: frozenset({Sensitivity.RAW}),
    ArtifactKind.RUBRIC_SNAPSHOT: frozenset({Sensitivity.DERIVED}),
    ArtifactKind.PROMPT_INPUT: frozenset({Sensitivity.DERIVED}),
    ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT: frozenset({Sensitivity.DERIVED}),
    ArtifactKind.SIGNAL_CATEGORIES: frozenset({Sensitivity.DERIVED}),
    ArtifactKind.SIGNAL_SUBCATEGORIES: frozenset({Sensitivity.DERIVED}),
    ArtifactKind.SIGNAL_EXTRACTION: frozenset({Sensitivity.MASKED}),
    ArtifactKind.ASR_BASE_TRANSCRIPT: frozenset({Sensitivity.RAW}),
    ArtifactKind.ASR_VOCABULARY_PASS: frozenset({Sensitivity.RAW}),
}
"""Where a kind's sensitivity is fixed. Other kinds carry raw, masked or derived per their content."""


def content_model_for(kind: ArtifactKind) -> Optional[Type[ContractModel]]:
    return ARTIFACT_CONTENT_MODELS[ARTIFACT_CONTENT_CONTRACTS[kind]]


class ArtifactStorage(str, Enum):
    OBJECT = "object"
    INLINE = "inline"


SLOT_MAX_LENGTH = 128
"""Longest live slot. A draft-test slot adds ``draft:<request_id>:`` in front (at most 263 characters)."""

SLOT_PATTERN = r"^[A-Za-z0-9._:-]{0,263}$"

DRAFT_TEST_SLOT_PREFIX = "draft:"
"""Reserved slot prefix of draft-test graphs. A live slot never starts with it."""


def _slot(value: str) -> str:
    if not re.fullmatch(SLOT_PATTERN, value):
        raise ValueError(f"{value!r} is not a valid slot")
    return value


def is_draft_test_slot(slot: str) -> bool:
    return slot.startswith(DRAFT_TEST_SLOT_PREFIX)


def draft_test_slot(request_id: str, slot: str) -> str:
    """``draft:<request_id>:<slot>``: where a draft-test graph (a ``jobs.DRAFT_TEST_KINDS`` request
    ``request_id``: ``qa_draft_test`` or, since 1.3.0, ``contact_signals_preview``) links what would
    otherwise go in ``slot``. Its versions and supersession are
    separate from the call's live artifacts."""
    if is_draft_test_slot(slot) or len(slot) > SLOT_MAX_LENGTH:
        raise ValueError("a draft-test slot wraps one live slot")
    return _slot(f"{DRAFT_TEST_SLOT_PREFIX}{request_id}:{slot}")


def rubric_snapshot_slot(rubric_id: str, version: int) -> str:
    """``rubric:<rubric_id>:v<version>``: the slot of a published version's snapshot, one per
    conversation and version."""
    return _slot(f"rubric:{rubric_id}:v{version}")


def draft_rubric_snapshot_slot(request_id: str, rubric_id: str, draft_revision: int) -> str:
    """The slot of the draft snapshot Store mints for a draft test:
    ``draft:<request_id>:rubric:<rubric_id>:r<draft_revision>``."""
    return draft_test_slot(request_id, f"rubric:{rubric_id}:r{draft_revision}")


def signal_taxonomy_snapshot_slot(version: int) -> str:
    """Added in 1.3.0. ``signals:v<version>``: the slot of a published signal taxonomy version's
    snapshot, one per conversation and version (``mintSignalTaxonomySnapshot``)."""
    return _slot(f"signals:v{version}")


def preview_signal_taxonomy_snapshot_slot(request_id: str) -> str:
    """Added in 1.3.0. ``draft:<request_id>:signals:preview``: where ``createSignalPreview`` mints the
    unsaved taxonomy's snapshot for one preview request (a draft-test kind)."""
    return draft_test_slot(request_id, "signals:preview")


def canonical_content(content_contract: str, payload: Any) -> bytes:
    """The stored bytes of a JSON artifact: ``canonical_json`` of ``payload``, after checking that
    ``payload`` validates against the content model and already is its full dump
    (``model_dump(mode="json")``: every default and null present, timestamps as the model emits
    them). Raises ``ValueError`` otherwise. The artifact checksum is ``sha256`` of these bytes, which
    equals ``canonical_digest(payload)``; Store answers ``checksum_mismatch`` when the declared one
    differs."""
    model = ARTIFACT_CONTENT_MODELS[content_contract]
    if model is None:
        raise ValueError(f"{content_contract} is not JSON content")
    try:
        sent = canonical_json(payload)
        full = canonical_json(model.model_validate(payload).model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not valid {content_contract} content: {exc}") from None
    if sent != full:
        raise ValueError(f"payload is not the canonical {content_contract} document: send model_dump(mode='json') with every default present")
    return sent


class ArtifactDescriptor(ContractModel):
    kind: ArtifactKind
    slot: str = Field(default="", pattern=SLOT_PATTERN, description="Logical slot within (conversation, kind): '' for a per-call singleton, a criterion ID, 'segment:3', 'lifecycle:0', 'rubric:<rubric_id>:v<version>'; at most SLOT_MAX_LENGTH characters. A draft-test graph's artifacts use draft:<request_id>:<slot> (draft_test_slot). Versions and supersession are per slot.")
    content_type: ShortText = Field(description="IANA media type, e.g. audio/wav or application/json.")
    size_bytes: int = Field(ge=0)
    checksum: Sha256Digest = Field(description="SHA-256 of the stored bytes. For JSON content the stored bytes are canonical_content(): canonical_json of the content model's full dump, so the checksum is canonical_digest(payload) of a payload sent exactly in that form.")
    content_contract: ShortText = Field(description="Must equal ARTIFACT_CONTENT_CONTRACTS[kind] for this contract version.")
    sensitivity: Sensitivity
    producing_job_id: Optional[ResourceId] = Field(default=None, description="The job whose attempt produced this artifact; absent for source, Store-minted, migrated and global artifacts.")
    labels: Dict[str, str] = Field(default_factory=dict, description="Small non-secret labels such as segment index or criterion ID. Never content.")

    @model_validator(mode="after")
    def _contract_and_sensitivity(self):
        expected = ARTIFACT_CONTENT_CONTRACTS[self.kind]
        if self.content_contract != expected:
            raise ValueError(f"{self.kind.value} artifacts use content contract {expected}")
        allowed = ALLOWED_SENSITIVITY.get(self.kind)
        if allowed and self.sensitivity not in allowed:
            raise ValueError(f"{self.kind.value} artifacts are {', '.join(s.value for s in allowed)}")
        if len(self.labels) > 16 or any(len(k) > 64 or len(v) > 200 for k, v in self.labels.items()):
            raise ValueError("labels are small identifiers, not content")
        if ARTIFACT_CONTENT_MODELS[expected] is not None and self.content_type != "application/json":
            raise ValueError(f"{expected} is JSON content")
        if self.kind in GLOBAL_ARTIFACT_KINDS and (self.producing_job_id or self.slot):
            raise ValueError("global artifacts belong to no job and no slot")
        if not is_draft_test_slot(self.slot) and len(self.slot) > SLOT_MAX_LENGTH:
            raise ValueError(f"a live slot is at most {SLOT_MAX_LENGTH} characters")
        return self


class Artifact(ArtifactDescriptor):
    """An immutable committed artifact. A new version of the same slot gets a new ID."""

    id: ResourceId
    conversation_id: Optional[ResourceId] = Field(description="Null exactly for global kinds (attestation evidence, trust-anchor bundles).")
    linked: bool = Field(description="True once a completion references it (or at commit for every artifact with no producing job: sources, Store-minted rubric snapshots, migrated and global artifacts). Unlinked artifacts are orphan candidates.")
    linked_by_receipt_id: Optional[ResourceId] = Field(default=None, description="The completion receipt that linked a job output.")
    version: Optional[int] = Field(default=None, ge=1, description="Monotonic per (conversation, kind, slot); assigned when linked, null while unlinked.")
    storage: ArtifactStorage
    committed_at: Timestamp
    superseded_by: Optional[ResourceId] = Field(default=None, description="The next linked version of the same slot, when one exists. The old artifact stays readable.")

    @model_validator(mode="after")
    def _identity(self):
        if (self.kind in GLOBAL_ARTIFACT_KINDS) != (self.conversation_id is None):
            raise ValueError("global kinds have no conversation; every other kind has one")
        if self.linked != (self.version is not None):
            raise ValueError("an artifact has a version exactly when it is linked")
        if self.linked_by_receipt_id is not None and (not self.linked or self.producing_job_id is None):
            raise ValueError("only a linked job output names the receipt that linked it")
        return self


class InlineArtifactCreate(ArtifactDescriptor):
    """Create a small JSON artifact in one call. Bounded by INLINE_ARTIFACT_MAX_BYTES of canonical
    JSON. ``payload`` must be the content model's full dump (``canonical_content``: validated, every
    default present, not normalized by Store), so there is one checksum: Store recomputes
    ``canonical_digest(payload)``, rejects a mismatch with ``checksum_mismatch``, and stores exactly
    those canonical bytes. Store also checks ``size_bytes`` against their length.

    Natural idempotency: for a job output the key is (producing job, the attempt the claim token
    belongs to, kind, slot, checksum); for other artifacts (conversation, kind, slot, checksum). A
    replay returns the existing artifact. A deliberate rerun is a different job, so it always gets a
    new artifact and, once linked, a new version."""

    payload: Dict[str, Any] = Field(description="The JSON document, matching the kind's content model.")
    claim_token: Optional[str] = Field(default=None, description="Required when producing_job_id is set: proves the active claim.")

    @model_validator(mode="after")
    def _json_payload(self):
        model = ARTIFACT_CONTENT_MODELS[self.content_contract]
        if model is None:
            raise ValueError("inline artifacts carry JSON content kinds only")
        if self.kind in GLOBAL_ARTIFACT_KINDS:
            raise ValueError("global artifacts are uploaded through their own routes")
        if self.kind in STORE_MINTED_KINDS:
            raise ValueError(f"{self.kind.value} artifacts are minted by Store only")
        if (self.producing_job_id is None) != (self.claim_token is None):
            raise ValueError("a job output proves its claim with the claim token, and only a job output does")
        canonical_content(self.content_contract, self.payload)
        return self


class UploadGrantRequest(ArtifactDescriptor):
    """Request a grant for a large artifact of a conversation. Idempotency as for inline creation.
    For a JSON kind, the uploaded bytes must be ``canonical_content`` of the payload; the commit
    rejects anything else (checksum_mismatch when the digest differs, validation_failed when the
    bytes are not the canonical document)."""

    claim_token: Optional[str] = Field(default=None, description="Required when producing_job_id is set.")

    @model_validator(mode="after")
    def _conversation_kind(self):
        if self.kind in GLOBAL_ARTIFACT_KINDS:
            raise ValueError("global artifacts are uploaded through their own routes")
        if self.kind in STORE_MINTED_KINDS:
            raise ValueError(f"{self.kind.value} artifacts are minted by Store only")
        if (self.producing_job_id is None) != (self.claim_token is None):
            raise ValueError("a job output proves its claim with the claim token, and only a job output does")
        return self


class GlobalArtifactUploadRequest(ContractModel):
    """Upload attestation evidence (scope key-release:write) or a trust-anchor bundle (scope
    release-trust:write). Natural idempotency by checksum: a bundle already stored returns
    ``existing`` and no grant."""

    content_type: ShortText
    size_bytes: int = Field(ge=1)
    checksum: Sha256Digest
    labels: Dict[str, str] = Field(default_factory=dict)


class UploadGrant(ContractModel):
    """A short-lived, object-scoped upload authorization. The URL is on encrypted transport and
    points at Store (or a Store-managed endpoint), never at raw storage credentials."""

    upload_id: ResourceId
    artifact_id: ResourceId = Field(description="Reserved ID the artifact will have once committed.")
    method: str = Field(default="PUT", pattern=r"^PUT$")
    url: str = Field(pattern=r"^https://", max_length=2048)
    headers: Dict[str, str] = Field(default_factory=dict, description="Headers the client must send with the PUT (content type, length). Never a reusable credential.")
    expires_at: Timestamp
    max_bytes: int = Field(ge=1)


class GlobalArtifactUpload(ContractModel):
    existing: Optional[Artifact] = Field(default=None, description="Present when an artifact with this checksum is already stored; no upload is needed.")
    grant: Optional[UploadGrant] = None

    @model_validator(mode="after")
    def _one(self):
        if (self.existing is None) == (self.grant is None):
            raise ValueError("either the existing artifact or an upload grant")
        return self


class UploadCommit(ContractModel):
    """Store verifies the uploaded bytes against these before the artifact metadata is committed.
    Idempotent by upload ID: a repeated commit returns the committed artifact."""

    checksum: Sha256Digest
    size_bytes: int = Field(ge=0)


class ContentGrant(ContractModel):
    """A short-lived, object-scoped download authorization for one artifact."""

    artifact_id: ResourceId
    url: str = Field(pattern=r"^https://", max_length=2048)
    expires_at: Timestamp
    checksum: Sha256Digest
    content_type: ShortText
    size_bytes: int = Field(ge=0)


class ArtifactListQuery(PageQuery):
    kind: Optional[ArtifactKind] = None
    slot: Optional[str] = Field(default=None, pattern=SLOT_PATTERN)
    include_superseded: bool = False
    include_unlinked: bool = Field(default=False, description="Include orphan candidates (committed, never linked).")
    include_draft_tests: bool = Field(default=False, description="Include artifacts in draft-test slots (draft:<request_id>:...). A slot filter under draft: implies it.")


ArtifactRefs = List[ArtifactRef]
