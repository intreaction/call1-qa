"""Rubrics: editable drafts and immutable published versions.

Process scores against an immutable rubric version (``RubricVersionRef``) frozen on each QA job.
Evaluate's Rubric Studio edits a draft and publishes it as the next version. The field shapes
mirror the pre-split ``RubricDefinition`` so the migration importer is a straight copy and the
existing evaluator contracts carry over unchanged.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from .common import ContractModel, PageQuery, ResourceId, SafeText, Sha256Digest, ShortText, Timestamp, canonical_digest
from .contents import EscalationTrigger, SpeakerRole


class RubricCategory(str, Enum):
    GENERAL = "GENERAL"
    BANKING = "BANKING"
    CUSTOMER_CARE = "CUSTOMER_CARE"
    COLLECTIONS = "COLLECTIONS"
    HEALTHCARE = "HEALTHCARE"
    TECH_SUPPORT = "TECH_SUPPORT"


class CheckType(str, Enum):
    SENTIMENT_METRIC = "sentiment_metric"
    PHRASE_ANY = "phrase_any"
    PHRASE_ALL = "phrase_all"
    PHRASE_NONE = "phrase_none"
    CONDITIONAL_RESPONSE = "conditional_response"
    SEMANTIC_JUDGEMENT = "semantic_judgement"
    CUSTOM_REGEX = "custom_regex"


class RubricCheck(ContractModel):
    """Modular check definition; the evaluator dispatches on ``check_type``.
    Model references name catalog entry IDs (the pre-split QuestionModel IDs are preserved as such)."""

    check_type: CheckType = CheckType.PHRASE_ANY
    metric: Literal["text_polarity", "valence", "arousal", "dominance"] = "text_polarity"
    aggregation: Literal["mean", "min", "max"] = "mean"
    comparison: Literal["gte", "lte"] = "gte"
    metric_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    min_samples: int = Field(default=2, ge=1, le=10000)
    min_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    phrases: List[str] = Field(default_factory=list)
    threshold: int = Field(default=80, ge=0, le=100, description="Similarity a phrase must reach to count as a match, 0-100.")
    window_seconds: Optional[float] = Field(default=None, description=">0 = first N seconds, <0 = last N seconds, null = whole call.")
    speaker: Optional[SpeakerRole] = SpeakerRole.AGENT
    trigger_phrases: List[str] = Field(default_factory=list)
    response_phrases: List[str] = Field(default_factory=list)
    pass_when: Optional[str] = None
    fail_when: Optional[str] = None
    not_applicable_when: Optional[str] = None
    requires_policy: bool = False
    policy_context: Optional[str] = Field(default=None, max_length=12000)
    primary_model_id: Optional[ShortText] = Field(default=None, description="Catalog entry ID for the primary assessment; null inherits the purpose default.")
    escalation_model_id: Optional[ShortText] = Field(default=None, description="Catalog entry ID for escalation; null inherits, 'none' disables.")
    escalation_when: List[EscalationTrigger] = Field(default_factory=lambda: [EscalationTrigger.NEEDS_REVIEW, EscalationTrigger.INVALID_ANSWER])
    pattern: Optional[str] = None
    legacy_rule: Optional[str] = Field(default=None, description="The pre-split rule_type this check was derived from, when migrated.")

    @model_validator(mode="after")
    def _tone_threshold(self):
        if self.check_type is CheckType.SENTIMENT_METRIC and self.metric != "text_polarity" and self.metric_threshold < 0:
            raise ValueError("Tone thresholds must be between 0 and 1.")
        return self


CRITERION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
"""A criterion ID becomes the QA assessment's artifact slot, so it must be slot-safe: no ':' (which
would collide with the ``segment:``, ``rubric:`` and ``draft:`` slot namespaces) and at most
SLOT_MAX_LENGTH characters. Every pre-split seed ID fits; the migration importer rejects, and
reports, a legacy ID that does not."""


class RubricCriterion(ContractModel):
    criterion_id: str = Field(pattern=CRITERION_ID_PATTERN, description="Slot-safe criterion ID; becomes the qa_assessment slot. See CRITERION_ID_PATTERN.")
    name: ShortText
    category: ShortText = "COMPLIANCE"
    description: str = ""
    weight: float = Field(default=25.0, ge=0)
    critical: bool = False
    check: RubricCheck = Field(description="Canonical definition. Legacy rule_type/parameters are carried for migrated criteria only.")
    rule_type: Optional[str] = Field(default=None, description="Legacy rule engine type, migrated criteria only.")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Legacy rule engine parameters, migrated criteria only.")


class RubricDefinition(ContractModel):
    rubric_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    name: ShortText
    description: str = ""
    category: RubricCategory = RubricCategory.GENERAL
    pass_threshold: float = Field(default=80.0, ge=0, le=100)
    criteria: List[RubricCriterion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_criteria(self):
        ids = [c.criterion_id for c in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs must be unique within a rubric")
        return self


class RubricVersionRef(ContractModel):
    """Points at an immutable published rubric version. Frozen on every QA job and scorecard."""

    rubric_id: str
    version: int = Field(ge=1)
    digest: Sha256Digest = Field(description="canonical_digest(definition) of the published version.")


class DraftRubricRef(ContractModel):
    """Points at the snapshot of an unpublished draft that a draft test scores with. Store mints the
    rubric_snapshot artifact from the stored draft when the test is requested, in the request's
    draft slot (``artifacts.draft_rubric_snapshot_slot``)."""

    rubric_id: str
    draft_revision: int = Field(ge=1)
    snapshot_artifact_id: ResourceId
    digest: Sha256Digest = Field(description="canonical_digest(definition) of the draft as snapshotted.")


class RubricVersionStatus(str, Enum):
    ACTIVE = "active"
    RETIRED = "retired"


class RubricVersion(ContractModel):
    ref: RubricVersionRef
    definition: RubricDefinition
    status: RubricVersionStatus
    published_at: Timestamp
    published_by_account_id: Optional[ResourceId] = Field(default=None, description="Absent for versions created by the migration importer.")
    notes: Optional[SafeText] = None

    @model_validator(mode="after")
    def _digest(self):
        if self.ref.rubric_id != self.definition.rubric_id or self.ref.digest != canonical_digest(self.definition):
            raise ValueError("a version's ref names its rubric and canonical_digest(definition)")
        return self


class RubricSnapshotContent(ContractModel):
    """``rubric_snapshot.v1``: the exact rubric definition a QA job scores with, as an input artifact.
    Only Store creates one (``artifacts.STORE_MINTED_KINDS``):

    - **published**: ``POST /conversations/{id}/rubric-snapshots`` (``mintRubricSnapshot``, Process,
      ``jobs:write``) copies a stored published version into the conversation's slot
      ``rubric:<rubric_id>:v<version>``, linked at commit. It is idempotent per conversation and
      version, so every ingest or reanalysis graph pins the same artifact.
    - **draft**: ``POST /rubrics/{id}/draft/tests`` copies the stored draft into the draft-test slot
      in the transaction that creates the request (``DraftRubricRef.snapshot_artifact_id``).

    Every QA job's ``rubric`` input pins one, and at graph creation Store checks that it names the
    same rubric, version or draft revision, and digest as the job's parameters (graph_invalid)."""

    source: Literal["published", "draft"]
    rubric_id: str
    rubric_version: Optional[int] = Field(default=None, ge=1)
    draft_revision: Optional[int] = Field(default=None, ge=1)
    digest: Sha256Digest
    definition: RubricDefinition

    @model_validator(mode="after")
    def _source(self):
        if (self.source == "published") != (self.rubric_version is not None) or (self.source == "draft") != (self.draft_revision is not None):
            raise ValueError("a published snapshot names its version, a draft snapshot its draft revision")
        if self.digest != canonical_digest(self.definition):
            raise ValueError("digest is canonical_digest(definition)")
        return self


class RubricSnapshotRequest(ContractModel):
    """Ask Store for the snapshot of one published version (active or retired) for one conversation.
    A repeat returns the same linked artifact."""

    rubric_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    version: int = Field(ge=1)


RUBRIC_INPUT_ROLE = "rubric"
"""The input role under which every QA job (``JobTypeRule.needs_rubric``) pins its rubric_snapshot."""


class RubricSummary(ContractModel):
    rubric_id: str
    name: ShortText
    description: str
    category: RubricCategory
    pass_threshold: float
    criteria_count: int = Field(ge=0)
    current_version: Optional[int] = Field(default=None, ge=1, description="Absent when only a draft exists.")
    current_digest: Optional[Sha256Digest] = None
    has_draft: bool
    updated_at: Timestamp


class RubricDraft(ContractModel):
    rubric_id: str
    definition: RubricDefinition
    draft_revision: int = Field(ge=1, description="Optimistic-concurrency token for draft saves.")
    based_on_version: Optional[int] = Field(default=None, ge=1)
    updated_at: Timestamp
    updated_by_account_id: ResourceId


class RubricDraftSave(ContractModel):
    definition: RubricDefinition
    expected_draft_revision: Optional[int] = Field(default=None, ge=0, description="0 or null to create; otherwise the revision being replaced.")


class RubricPublishRequest(ContractModel):
    expected_current_version: int = Field(ge=0, description="0 when publishing the first version.")
    expected_draft_revision: int = Field(ge=1)
    notes: Optional[SafeText] = None


class RubricRetireRequest(ContractModel):
    expected_current_version: int = Field(ge=1)
    reason: SafeText


class RubricListQuery(PageQuery):
    category: Optional[RubricCategory] = None
    include_retired: bool = False
