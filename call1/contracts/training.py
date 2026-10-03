"""On-device training labels (contract 1.3.0; docs/OnDeviceTraining.md section 2, team decision 28).

Reviewers already correct the machine in Evaluate: they override QA verdicts, Confirm or Dismiss a
contact signal's category, Confirm or Correct its subcategory, and correct speakers. Store appends
each of those labels to an append-only log (``results_training_labels``; the ``seq`` is the
cursor). Process pages the log with ``listTrainingLabels`` (``training:read``), rebuilds each
labelled prompt with the engine's own prompt builder and masking from the source artifacts, and
trains the customer's LoRA **on the device**. The labels, the datasets and the adapter never leave
the customer's hardware.

The log and this read model hold **IDs, enums and versions only**: no transcript text, no quote, no
reviewer note, no signal note and no reviewer identity. Text is rebuilt by Process from the
artifacts named in ``sources``, masked with the engines' own masking, and never read here.

Open core (Apache 2.0 once the license audit clears): nothing here reads a Pro1 connection or
entitlement, and nothing is gated.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Dict, FrozenSet, List, Literal, Optional, Tuple

from pydantic import Field, StringConstraints, model_validator

from .artifacts import ArtifactKind
from .common import ContractModel, ResourceId, Sha256Digest, ShortText, Timestamp
from .contents import SIGNAL_NOT_OPTION, ShortDigest, SignalNodeId, VerdictStatus
from .reviews import OverrideReasonCode


class TrainingLabelKind(str, Enum):
    QA_VERDICT = "qa_verdict"
    """A QA verdict override (``overrideVerdict``). Subject ``qa:<call_id>:<criterion_id>``."""
    SIGNAL_HIT = "signal_hit"
    """Contact Signals v2 hit feedback (``saveSignalHitFeedback``). Subject ``signal:<call_id>:<hit_id>``."""
    SPEAKER_ROLE = "speaker_role"
    """A speaker correction (``correctSpeaker``). Subject ``speaker:<call_id>:<cluster or t<turn_id>>``."""


TrainingSubject = Annotated[str, StringConstraints(pattern=r"^(qa|signal|speaker):[A-Za-z0-9]", max_length=400)]
"""What a label judges: one criterion of one call, one signal hit, or one speaker cluster (or turn).
The newest label (highest ``seq``) for a subject supersedes the older ones, and a ``withdrawn`` one
removes the subject. Longer than ``ShortText`` because it joins two IDs. Built only by
``training_label_subject``."""

TRAINING_JUDGED_ARTIFACT_KIND: Dict[TrainingLabelKind, ArtifactKind] = {
    TrainingLabelKind.QA_VERDICT: ArtifactKind.QA_SCORECARD,
    TrainingLabelKind.SIGNAL_HIT: ArtifactKind.CONTACT_SIGNALS,
    TrainingLabelKind.SPEAKER_ROLE: ArtifactKind.SPEAKER_ATTRIBUTION,
}
"""The kind of the artifact a label judged (the log's ``source_artifact_id``): the scorecard
publication at ``evaluation_version``, the ``contact_signals`` publication at ``signals_version``,
or the ``speaker_attribution`` artifact current at correction time."""

TRAINING_SOURCE_ROLES: Dict[TrainingLabelKind, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    TrainingLabelKind.QA_VERDICT: (
        frozenset({"transcript", "assessment"}),
        frozenset({"speaker_attribution", "rubric", "enrichment", "prompt_input", "escalation_assessment", "pii_findings"}),
    ),
    TrainingLabelKind.SIGNAL_HIT: (
        frozenset({"transcript", "contact_signals"}),
        frozenset({"speaker_attribution", "pii_findings", "taxonomy", "stage:categorize", "stage:subcategorize", "stage:extract"}),
    ),
    TrainingLabelKind.SPEAKER_ROLE: (
        frozenset({"transcript", "speaker_attribution"}),
        frozenset({"pii_findings", "enrichment"}),
    ),
}
"""(required, optional) ``TrainingSourceRef.role`` values per kind (docs/OnDeviceTraining.md 2.3).
A resolved item (non-empty ``sources``) carries every required role and no role outside the two
sets. ``pii_findings`` is optional because the newest findings for the label's transcript may not
exist; Process then skips the label (``no_pii_findings``) and never trains it unmasked."""

TRAINING_SOURCE_ROLE_KINDS: Dict[str, FrozenSet[ArtifactKind]] = {
    "transcript": frozenset({ArtifactKind.TRANSCRIPT}),
    "speaker_attribution": frozenset({ArtifactKind.SPEAKER_ATTRIBUTION}),
    "pii_findings": frozenset({ArtifactKind.PII_FINDINGS}),
    "rubric": frozenset({ArtifactKind.RUBRIC_SNAPSHOT}),
    "enrichment": frozenset({ArtifactKind.ENRICHMENT}),
    "assessment": frozenset({ArtifactKind.QA_ASSESSMENT}),
    "escalation_assessment": frozenset({ArtifactKind.QA_ASSESSMENT}),
    "prompt_input": frozenset({ArtifactKind.PROMPT_INPUT}),
    "contact_signals": frozenset({ArtifactKind.CONTACT_SIGNALS}),
    "taxonomy": frozenset({ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT}),
    "stage:categorize": frozenset({ArtifactKind.SIGNAL_CATEGORIES}),
    "stage:subcategorize": frozenset({ArtifactKind.SIGNAL_SUBCATEGORIES}),
    "stage:extract": frozenset({ArtifactKind.SIGNAL_EXTRACTION}),
}
"""The artifact kind each source role must have."""

EXCLUDED_QA_REASON_CODES: FrozenSet[OverrideReasonCode] = frozenset({
    OverrideReasonCode.TRANSCRIPTION_ERROR,
    OverrideReasonCode.SPEAKER_MISATTRIBUTED,
    OverrideReasonCode.POLICY_EXCEPTION,
})
"""Override reasons that blame the input, not the model. Store still logs these labels (the log is
a faithful record); Process builds no training example from them."""


def training_label_subject(kind: TrainingLabelKind, call_id: str, key: str) -> str:
    """The normative subject: ``<prefix>:<call_id>:<key>``, where ``key`` is the criterion ID
    (``qa_verdict``), the hit ID (``signal_hit``) or ``speaker_subject_key`` (``speaker_role``)."""
    prefix = {TrainingLabelKind.QA_VERDICT: "qa", TrainingLabelKind.SIGNAL_HIT: "signal", TrainingLabelKind.SPEAKER_ROLE: "speaker"}[TrainingLabelKind(kind)]
    return f"{prefix}:{call_id}:{key}"


def speaker_subject_key(turn_id: int, apply_to_cluster: bool, speaker_cluster: Optional[str]) -> str:
    """The cluster name when the correction applied to a resolved cluster, else ``t<turn_id>``."""
    return speaker_cluster if (apply_to_cluster and speaker_cluster) else f"t{turn_id}"


class TrainingSourceRef(ContractModel):
    """One artifact Process fetches (``getArtifactContent``, existing ``artifacts:read``) to rebuild
    the labelled prompt. Draft-test and preview artifacts are never sources."""

    role: ShortText = Field(description="An input role of the source job (transcript, rubric, taxonomy, stage:categorize, ...) or one of assessment, prompt_input, escalation_assessment, contact_signals, pii_findings (TRAINING_SOURCE_ROLES).")
    artifact_id: ResourceId
    checksum: Sha256Digest
    kind: ArtifactKind


class QaVerdictLabel(ContractModel):
    """A reviewer's override of one criterion verdict. No reviewer notes are copied."""

    override_id: ResourceId
    criterion_id: ShortText
    evaluation_version: int = Field(ge=1, description="The machine result version the override judged.")
    original_status: VerdictStatus
    status: VerdictStatus
    reason_code: Optional[OverrideReasonCode] = Field(default=None, description="Labels whose reason is in EXCLUDED_QA_REASON_CODES are logged but never trained on.")


class SignalSpanAt(ContractModel):
    """One segment a signal hit covers: the anchor or a part of a multi-segment hit (decision 25)."""

    turn_id: int = Field(ge=0)
    block: int = Field(ge=0)


class SignalHitLabel(ContractModel):
    """A reviewer's verdicts on one v2 hit, with the (turn, block) segments read from the judged
    ``contact_signals`` artifact at save time. No signal note is copied. Both verdicts null means
    the reviewer cleared the feedback; that row is ``withdrawn``."""

    hit_id: ShortText
    category_id: SignalNodeId
    signals_version: int = Field(ge=1, description="The contact_signals publication the reviewer judged.")
    spans: List[SignalSpanAt] = Field(min_length=1, max_length=64, description="The anchor first, then the parts in call order.")
    category_verdict: Optional[Literal["confirmed", "dismissed"]] = None
    subcategory_id: Optional[SignalNodeId] = Field(default=None, description="The subcategory the verdict judged (or 'other').")
    subcategory_digest: Optional[ShortDigest] = None
    subcategory_verdict: Optional[Literal["confirmed", "corrected"]] = None
    corrected_subcategory_id: Optional[SignalNodeId] = None
    feedback_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _verdicts(self):
        if not self.hit_id.startswith(self.category_id + "."):
            raise ValueError("a v2 hit ID starts with its category ID")
        if (self.subcategory_verdict == "corrected") != (self.corrected_subcategory_id is not None):
            raise ValueError("a correction names the corrected subcategory, and only a correction does")
        if self.subcategory_verdict is not None and self.subcategory_id is None:
            raise ValueError("a subcategory verdict names the subcategory it judged")
        if SIGNAL_NOT_OPTION in (self.subcategory_id, self.corrected_subcategory_id):
            raise ValueError("'not' is a stage-2 rejection, not a subcategory")
        seen = [(s.turn_id, s.block) for s in self.spans]
        if len(seen) != len(set(seen)):
            raise ValueError("each (turn_id, block) once")
        return self

    @property
    def cleared(self) -> bool:
        return self.category_verdict is None and self.subcategory_verdict is None


class SpeakerRoleLabel(ContractModel):
    """A reviewer's speaker correction. ``speaker_cluster`` is the judged artifact's cluster for the
    turn, resolved into the row at correction time. No notes are copied."""

    turn_id: int = Field(ge=0)
    speaker: Literal["AGENT", "CALLER", "UNKNOWN"]
    apply_to_cluster: bool
    speaker_cluster: Optional[ShortText] = None
    reanalysis_request_id: ResourceId


_LABEL_FIELD = {TrainingLabelKind.QA_VERDICT: "qa", TrainingLabelKind.SIGNAL_HIT: "signal", TrainingLabelKind.SPEAKER_ROLE: "speaker"}


class TrainingLabel(ContractModel):
    """One row of the label log with its sources resolved at read time (docs/OnDeviceTraining.md
    2.2-2.3). Exactly the label field matching ``kind`` is set. A withdrawn row (only a
    ``signal_hit`` whose verdicts were both cleared) resolves nothing: ``source_job_id`` is null and
    ``sources`` is empty. A live row whose source can no longer be resolved (an orphaned artifact)
    also has ``sources: []``; Process skips it as ``source_unavailable``."""

    seq: int = Field(ge=1, description="The log position; the cursor.")
    kind: TrainingLabelKind
    subject: TrainingSubject = Field(description="training_label_subject(kind, call_id, key); the newest seq per subject wins.")
    call_id: ResourceId
    conversation_id: ResourceId
    recorded_at: Timestamp
    withdrawn: bool = False
    source_job_id: Optional[ResourceId] = Field(default=None, description="The job whose prompt Process rebuilds: the qa_criterion job, the contact_signals_merge job, or the speaker_attribution job. Null when withdrawn or unresolvable.")
    sources: List[TrainingSourceRef] = Field(default_factory=list, max_length=16)
    qa: Optional[QaVerdictLabel] = None
    signal: Optional[SignalHitLabel] = None
    speaker: Optional[SpeakerRoleLabel] = None

    @model_validator(mode="after")
    def _shape(self):
        set_fields = {name for name in _LABEL_FIELD.values() if getattr(self, name) is not None}
        if set_fields != {_LABEL_FIELD[self.kind]}:
            raise ValueError(f"a {self.kind.value} label sets exactly the {_LABEL_FIELD[self.kind]} field")
        if self.kind is TrainingLabelKind.QA_VERDICT:
            key = self.qa.criterion_id
        elif self.kind is TrainingLabelKind.SIGNAL_HIT:
            key = self.signal.hit_id
            if self.signal.cleared != self.withdrawn:
                raise ValueError("a signal label is withdrawn exactly when both verdicts are cleared")
        else:
            key = speaker_subject_key(self.speaker.turn_id, self.speaker.apply_to_cluster, self.speaker.speaker_cluster)
        if self.subject != training_label_subject(self.kind, self.call_id, key):
            raise ValueError("subject must be training_label_subject(kind, call_id, key)")
        if self.withdrawn and self.kind is not TrainingLabelKind.SIGNAL_HIT:
            raise ValueError("only a signal_hit label can be withdrawn")
        if self.withdrawn and (self.source_job_id is not None or self.sources):
            raise ValueError("a withdrawn label resolves no source job or sources")
        if self.sources:
            roles = [s.role for s in self.sources]
            if len(roles) != len(set(roles)):
                raise ValueError("each source role once")
            required, optional = TRAINING_SOURCE_ROLES[self.kind]
            if not required <= set(roles):
                raise ValueError(f"a resolved {self.kind.value} label carries the roles {sorted(required)}")
            unknown = set(roles) - required - optional
            if unknown:
                raise ValueError(f"unknown source roles for {self.kind.value}: {sorted(unknown)}")
            for s in self.sources:
                if s.kind not in TRAINING_SOURCE_ROLE_KINDS[s.role]:
                    raise ValueError(f"source role {s.role} must be a {sorted(k.value for k in TRAINING_SOURCE_ROLE_KINDS[s.role])} artifact")
            if self.source_job_id is None:
                raise ValueError("resolved sources name their source job")
        return self


class TrainingLabelQuery(ContractModel):
    after: int = Field(default=0, ge=0, description="Return rows with seq > after; 0 starts from the beginning of the log.")
    limit: int = Field(default=200, ge=0, le=500, description="0 returns the counts only (no items), and is not audited.")
    kinds: Optional[List[TrainingLabelKind]] = Field(default=None, description="Null means every kind.")


class TrainingLabelPage(ContractModel):
    """Items in ``seq`` order. Rows are never rewritten, so paging from ``next_after`` misses
    nothing; a later label for the same subject appears as a new row with a higher ``seq``."""

    items: List[TrainingLabel]
    next_after: int = Field(ge=0, description="The last seq returned, or the request's after when empty. Pass as after on the next call.")
    count_after: int = Field(ge=0, description="Rows with seq > after matching kinds, before paging.")
    high_water: int = Field(ge=0, description="The log's max seq over every kind (0 when empty).")

    @model_validator(mode="after")
    def _order(self):
        seqs = [item.seq for item in self.items]
        if any(b <= a for a, b in zip(seqs, seqs[1:])):
            raise ValueError("items come in strictly increasing seq order")
        if seqs and self.next_after != seqs[-1]:
            raise ValueError("next_after is the last seq returned")
        if self.count_after < len(seqs):
            raise ValueError("count_after counts at least the items returned")
        if self.high_water < self.next_after:
            raise ValueError("high_water is the log's max seq")
        return self


__all__ = [
    "TrainingLabelKind", "TrainingSubject", "TrainingSourceRef", "QaVerdictLabel", "SignalSpanAt", "SignalHitLabel",
    "SpeakerRoleLabel", "TrainingLabel", "TrainingLabelQuery", "TrainingLabelPage", "TRAINING_JUDGED_ARTIFACT_KIND",
    "TRAINING_SOURCE_ROLES", "TRAINING_SOURCE_ROLE_KINDS", "EXCLUDED_QA_REASON_CODES", "training_label_subject",
    "speaker_subject_key",
]
