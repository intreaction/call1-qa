"""The human review queue and reviewer decisions.

This is the human queue: who reviews which call and why, and what they decided. It shares no
type or endpoint with the processing job queue in ``jobs.py``. Every decision references the
exact machine-result version it was made against and carries an expected review version, so two
Evaluate clients cannot silently overwrite each other. Reanalysis marks affected decisions stale
until reconsidered or explicitly retained.

The stale-write rule (one rule for every review write): a write names the machine version it
judged, and it must be the call's current evaluation version, or Store answers 409 ``conflict``
with ``details.current_evaluation_version``. When a new machine version commits, Store supersedes
every unresolved queue item of that call (status SUPERSEDED, ``stale: true``), creates the items
the queue rules call for against the new version, marks resolved items ``stale: true``, and marks
the call's review ``stale`` if any decision referenced the older version. A supervisor may then
``retain`` the call-level decisions; nothing is resolved against a non-current version.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from .calls import AgentDisplayName, AgentExtension, VerdictStatus
from .common import ChangeCursor, ContractModel, JsonScalar, PageQuery, ResourceId, SafeText, ShortText, Timestamp
from .contents import SignalNodeId
from .jobs import SpeakerCorrection


# --- Queue rules and items ---------------------------------------------------------------


class ReviewStream(str, Enum):
    TRIAGE = "TRIAGE"
    AUDIT_SAMPLE = "AUDIT_SAMPLE"
    MANDATE = "MANDATE"
    CALIBRATION = "CALIBRATION"
    SIGNAL = "SIGNAL"
    """Added in 1.3.0. Contact-signal alerts drive the rule: its base condition is always true and
    ``target_signal_alerts`` (at least one) decides. Items read 'Signal: <name> (<speaker>, <m:ss>)'."""


class DistributionStrategy(str, Enum):
    """How unassigned items reach reviewers. The pre-split ``PRO1_AUTOMATED`` strategy is not in v1
    (the frontier review offering is future work); the importer maps it to UNASSIGNED_CLAIM and
    records the original value on the migration record."""

    ROUND_ROBIN = "ROUND_ROBIN"
    LEAST_OUTSTANDING = "LEAST_OUTSTANDING"
    SKILL_MATCHED = "SKILL_MATCHED"
    UNASSIGNED_CLAIM = "UNASSIGNED_CLAIM"


class ReviewQueueStatus(str, Enum):
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    OVERRIDDEN = "OVERRIDDEN"
    SUPERSEDED = "SUPERSEDED"
    """A newer machine version committed before this item was resolved; a new item replaces it."""


class ReviewQueueTrigger(str, Enum):
    START = "start"
    RELEASE = "release"
    RESOLVE_APPROVED = "resolve_approved"
    RESOLVE_OVERRIDDEN = "resolve_overridden"
    SUPERSEDE = "supersede"


class ReviewQueueTransition(ContractModel):
    from_status: ReviewQueueStatus
    to_status: ReviewQueueStatus
    trigger: ReviewQueueTrigger
    note: ShortText


REVIEW_QUEUE_TRANSITIONS: List[ReviewQueueTransition] = [
    ReviewQueueTransition(from_status=ReviewQueueStatus.PENDING, to_status=ReviewQueueStatus.IN_REVIEW, trigger=ReviewQueueTrigger.START, note="The assignee opens the item, or a reviewer claims the next unassigned item; item_version + 1."),
    ReviewQueueTransition(from_status=ReviewQueueStatus.IN_REVIEW, to_status=ReviewQueueStatus.PENDING, trigger=ReviewQueueTrigger.RELEASE, note="Released back to the pool; assignment cleared; item_version + 1."),
    ReviewQueueTransition(from_status=ReviewQueueStatus.IN_REVIEW, to_status=ReviewQueueStatus.APPROVED, trigger=ReviewQueueTrigger.RESOLVE_APPROVED, note="The machine result stands; writes a review history entry and bumps the call's review_version."),
    ReviewQueueTransition(from_status=ReviewQueueStatus.IN_REVIEW, to_status=ReviewQueueStatus.OVERRIDDEN, trigger=ReviewQueueTrigger.RESOLVE_OVERRIDDEN, note="The reviewer overrode the machine result; same writes as approved."),
    ReviewQueueTransition(from_status=ReviewQueueStatus.PENDING, to_status=ReviewQueueStatus.SUPERSEDED, trigger=ReviewQueueTrigger.SUPERSEDE, note="Store, when a newer machine version commits; the rules create a replacement item for it."),
    ReviewQueueTransition(from_status=ReviewQueueStatus.IN_REVIEW, to_status=ReviewQueueStatus.SUPERSEDED, trigger=ReviewQueueTrigger.SUPERSEDE, note="Store, when a newer machine version commits mid-review; the reviewer's client sees 409 conflict on resolve and opens the replacement."),
]
"""Assignment (``assigned_to``) changes while PENDING without a status transition."""


class ReviewQueueRule(ContractModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    name: ShortText
    stream: ReviewStream
    enabled: bool = True
    rank: int = Field(default=100, ge=0, description="Lower ranks are evaluated first. Unrelated to processing-job priority.")
    distribution_strategy: DistributionStrategy = DistributionStrategy.UNASSIGNED_CLAIM
    target_skills: List[ShortText] = Field(default_factory=list)
    critical_failure_only: bool = False
    low_confidence_only: bool = False
    sampling_rate: float = Field(default=0.0, ge=0, le=1)
    target_domains: List[ShortText] = Field(default_factory=list)
    target_agents: List[ShortText] = Field(default_factory=list)
    description: str = ""
    target_signal_alerts: List[SignalNodeId] = Field(default_factory=list, description="Added in 1.3.0. When set, at least one listed signal alert rule must match the call's current signals; checked before the stream, like target_domains. A SIGNAL rule lists at least one. '20% of cancellations' is AUDIT_SAMPLE plus an alert.")

    @model_validator(mode="after")
    def _signal_rule(self):
        if self.stream is ReviewStream.SIGNAL and not self.target_signal_alerts:
            raise ValueError("a SIGNAL rule targets at least one signal alert rule")
        if len(self.target_signal_alerts) != len(set(self.target_signal_alerts)):
            raise ValueError("target_signal_alerts lists each rule once")
        return self


class ReviewQueueRuleRecord(ReviewQueueRule):
    rule_version: int = Field(ge=1)
    updated_at: Timestamp
    updated_by_account_id: Optional[ResourceId] = None


class ReviewQueueRuleSave(ContractModel):
    rule: ReviewQueueRule
    expected_rule_version: int = Field(ge=0, description="0 to create.")


class ReviewerProfile(ContractModel):
    """Work-distribution attributes of a reviewer account (the pre-split auditor fields)."""

    account_id: ResourceId
    display_name: ShortText
    skills: List[ShortText] = Field(default_factory=list)
    capacity_weight: float = Field(default=1.0, ge=0.1)
    accepting_assignments: bool = True
    pending_count: int = Field(default=0, ge=0)


class ReviewerProfileUpdate(ContractModel):
    skills: Optional[List[ShortText]] = None
    capacity_weight: Optional[float] = Field(default=None, ge=0.1)
    accepting_assignments: Optional[bool] = None


class ReviewQueueItem(ContractModel):
    id: ResourceId
    call_id: ResourceId
    conversation_id: ResourceId
    rule_id: str
    rule_name: ShortText
    stream: ReviewStream
    reason: SafeText
    urgency_score: float = Field(ge=0, le=100)
    evaluation_version: int = Field(ge=1, description="The machine result version this item was created for.")
    stale: bool = Field(description="True when a newer machine result version exists than evaluation_version.")
    superseded_by_item_id: Optional[ResourceId] = Field(default=None, description="The replacement item, for a SUPERSEDED item.")
    status: ReviewQueueStatus
    item_version: int = Field(ge=1, description="Optimistic-concurrency token for item writes.")
    assigned_to_account_id: Optional[ResourceId] = None
    assigned_display_name: Optional[ShortText] = None
    created_at: Timestamp
    assigned_at: Optional[Timestamp] = None
    started_at: Optional[Timestamp] = None
    resolved_at: Optional[Timestamp] = None
    resolved_by_account_id: Optional[ResourceId] = None
    reviewer_notes: Optional[SafeText] = None
    agent_id: Optional[ShortText] = None
    agent_display_name: Optional[AgentDisplayName] = Field(default=None, description="Added in 1.1.0; from the call's current CallMetadata. Label with calls.agent_label().")
    agent_extension: Optional[AgentExtension] = Field(default=None, description="Added in 1.1.0; from the call's current CallMetadata.")
    overall_score: Optional[float] = None
    critical_failure: bool = False
    duration_seconds: Optional[float] = None
    signals_version: Optional[int] = Field(default=None, ge=1, description="Added in 1.3.0. The contact_signals version the item's alert match was evaluated on.")
    trigger_alert_rule_ids: List[SignalNodeId] = Field(default_factory=list, description="Added in 1.3.0. The signal alert rules that created the item; an item whose alerts stop matching stays, marked 'Signals changed since this item was created'.")

    @model_validator(mode="after")
    def _superseded(self):
        superseded = self.status is ReviewQueueStatus.SUPERSEDED
        if superseded != (self.superseded_by_item_id is not None) or (superseded and not self.stale):
            raise ValueError("a superseded item is stale and names its replacement, and only it does")
        return self


class ReviewQueueQuery(PageQuery):
    status: Optional[ReviewQueueStatus] = None
    stream: Optional[ReviewStream] = None
    assigned_to_account_id: Optional[ResourceId] = None
    unassigned_only: bool = False
    call_id: Optional[ResourceId] = None
    include_stale: bool = True


class ReviewQueueStats(ContractModel):
    total: int = Field(ge=0)
    pending: int = Field(ge=0)
    in_review: int = Field(ge=0)
    unassigned: int = Field(ge=0)
    stale: int = Field(ge=0)
    by_stream: Dict[str, int] = Field(default_factory=dict)
    by_status: Dict[str, int] = Field(default_factory=dict)


class AssignRequest(ContractModel):
    account_id: Optional[ResourceId] = Field(default=None, description="Null unassigns.")
    expected_item_version: int = Field(ge=1)


class ClaimNextResponse(ContractModel):
    item: Optional[ReviewQueueItem] = Field(default=None, description="Null when nothing is claimable.")


class StartReviewRequest(ContractModel):
    expected_item_version: int = Field(ge=1)


class ReleaseRequest(ContractModel):
    expected_item_version: int = Field(ge=1)
    note: Optional[SafeText] = None


class ResolveRequest(ContractModel):
    """Resolving an item assigned to another reviewer needs ``resolve_any_review`` (supervisor)."""

    status: ReviewQueueStatus = Field(description="APPROVED or OVERRIDDEN.")
    reviewer_notes: Optional[SafeText] = None
    expected_item_version: int = Field(ge=1)
    expected_review_version: int = Field(ge=0, description="The call's review_version this decision was made against.")
    evaluation_version: int = Field(ge=1, description="The machine result version reviewed: must equal the item's evaluation_version, which must be the call's current version (a stale or superseded item is 409 conflict).")

    @model_validator(mode="after")
    def _resolution_status(self):
        if self.status not in (ReviewQueueStatus.APPROVED, ReviewQueueStatus.OVERRIDDEN):
            raise ValueError("a resolution is APPROVED or OVERRIDDEN; use release to return an item to the pool")
        return self


# --- Call-level review state --------------------------------------------------------------


class OverrideReasonCode(str, Enum):
    """Structured reason for a per-criterion override (additive after cutover; optional in v1)."""

    MODEL_MISREAD_EVIDENCE = "model_misread_evidence"
    EVIDENCE_NOT_IN_TRANSCRIPT = "evidence_not_in_transcript"
    TRANSCRIPTION_ERROR = "transcription_error"
    SPEAKER_MISATTRIBUTED = "speaker_misattributed"
    POLICY_EXCEPTION = "policy_exception"
    CRITERION_NOT_APPLICABLE = "criterion_not_applicable"
    RUBRIC_AMBIGUOUS = "rubric_ambiguous"
    OTHER = "other"


class VerdictOverride(ContractModel):
    status: VerdictStatus
    reviewer_notes: Optional[SafeText] = None
    reason_code: Optional[OverrideReasonCode] = None
    expected_version: int = Field(ge=0, description="The call's current review_version. Mismatch: 409 review_version_conflict with current_version.")
    evaluation_version: int = Field(ge=1, description="The machine result version being overridden; must be the current version (409 conflict otherwise).")


class VerdictOverrideRecord(ContractModel):
    id: ResourceId
    call_id: ResourceId
    criterion_id: ShortText
    evaluation_version: int = Field(ge=1)
    original_status: VerdictStatus
    status: VerdictStatus
    reason_code: Optional[OverrideReasonCode] = None
    reviewer_notes: Optional[SafeText] = None
    account_id: ResourceId
    review_version: int = Field(ge=1, description="The review_version this write produced.")
    created_at: Timestamp


class EscalationStatus(str, Enum):
    NONE = "NONE"
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    OVERRIDDEN = "OVERRIDDEN"


class EscalationResolution(ContractModel):
    escalation_status: EscalationStatus = Field(description="APPROVED or OVERRIDDEN.")
    reviewer_notes: Optional[SafeText] = None
    expected_version: int = Field(ge=0)
    evaluation_version: int = Field(ge=1, description="Must be the current version (409 conflict otherwise).")

    @model_validator(mode="after")
    def _resolution(self):
        if self.escalation_status not in (EscalationStatus.APPROVED, EscalationStatus.OVERRIDDEN):
            raise ValueError("a resolution is APPROVED or OVERRIDDEN")
        return self


class ReviewStaleness(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    """A newer machine result version exists; the decisions below refer to an older one."""
    RETAINED = "retained"
    """A supervisor explicitly kept the decisions despite a newer machine version."""


class ReviewedScore(ContractModel):
    """Current-version reviewer decisions applied to the pinned rubric; machine artifacts remain unchanged."""

    evaluation_version: int = Field(ge=1)
    overall_score: float = Field(ge=0, le=100)
    passed: bool
    critical_failure: bool
    requires_human_review: bool


class CallReviewState(ContractModel):
    call_id: ResourceId
    review_version: int = Field(ge=0, description="Bumped by every review write; the expected-version token.")
    current_evaluation_version: Optional[int] = Field(default=None, ge=1)
    reviewed_evaluation_version: Optional[int] = Field(default=None, ge=1, description="The machine version the decisions below were made against.")
    staleness: ReviewStaleness
    escalation_status: EscalationStatus
    escalation_resolved_by_account_id: Optional[ResourceId] = None
    escalation_resolved_at: Optional[Timestamp] = None
    reviewer_notes: Optional[SafeText] = None
    overrides: List[VerdictOverrideRecord] = Field(default_factory=list)
    reviewed_score: Optional[ReviewedScore] = Field(default=None, description="Score after the latest override per criterion of the current evaluation. Absent for stale decisions or no overrides; never changes the immutable machine scorecard.")
    retained_by_account_id: Optional[ResourceId] = None
    retained_at: Optional[Timestamp] = None
    updated_at: Timestamp


class RetainReviewRequest(ContractModel):
    expected_version: int = Field(ge=0)
    note: Optional[SafeText] = None


class SpeakerCorrectionRequest(ContractModel):
    correction: SpeakerCorrection
    expected_version: int = Field(ge=0)


class ReviewWriteResult(ContractModel):
    call_id: ResourceId
    review_version: int = Field(ge=1, description="The new review_version.")
    change_cursor: ChangeCursor
    committed_at: Timestamp


class ReviewHistoryKind(str, Enum):
    VERDICT_OVERRIDE = "verdict_override"
    ESCALATION_RESOLUTION = "escalation_resolution"
    QUEUE_RESOLUTION = "queue_resolution"
    SPEAKER_CORRECTION = "speaker_correction"
    REVIEW_RETAINED = "review_retained"
    REANALYSIS_REQUESTED = "reanalysis_requested"
    MARKED_STALE = "marked_stale"
    LEGACY_ACTION = "legacy_action"


class ReviewHistoryEntry(ContractModel):
    id: ResourceId
    call_id: ResourceId
    kind: ReviewHistoryKind
    account_id: Optional[ResourceId] = Field(default=None, description="Null for pre-split shared-token actions, whose actor label is kept in payload.actor_label.")
    review_version: int = Field(ge=0)
    evaluation_version: Optional[int] = Field(default=None, ge=1)
    payload: Dict[str, JsonScalar] = Field(default_factory=dict, description="Safe details only.")
    created_at: Timestamp


class EscalationListItem(ContractModel):
    call_id: ResourceId
    agent_id: ShortText
    agent_display_name: Optional[AgentDisplayName] = Field(default=None, description="Added in 1.1.0; from the call's current CallMetadata. Label with calls.agent_label().")
    agent_extension: Optional[AgentExtension] = Field(default=None, description="Added in 1.1.0; from the call's current CallMetadata.")
    duration_seconds: float
    overall_score: Optional[float] = None
    escalation_status: EscalationStatus
    critical_failure: bool
    evaluation_version: int = Field(ge=1)
    review_version: int = Field(ge=0)
    created_at: Timestamp


class EscalationQuery(PageQuery):
    status: Optional[EscalationStatus] = EscalationStatus.PENDING
