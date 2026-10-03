"""Audit events and the change feed.

Audit events record who did what to which trust- or identity-relevant thing, with safe details.
The change feed is how Evaluate and Process learn that something changed without polling every
record: it carries IDs, versions and statuses, never content. A queue wakeup carries only a job
ID and means "look in Store's queue".
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from .common import ChangeCursor, ContractModel, JsonScalar, PageQuery, ResourceId, Sha256Digest, ShortText, Timestamp, canonical_digest


class ActorKind(str, Enum):
    REVIEWER = "reviewer"
    PROCESS_SERVICE = "process_service"
    BREAK_GLASS = "break_glass"
    INSTALLER = "installer"
    STORE_SYSTEM = "store_system"
    MIGRATION = "migration"
    LEGACY_SHARED_TOKEN = "legacy_shared_token"
    """Pre-split shared admin/reviewer token actions keep their label; no per-person attribution is invented."""


class Actor(ContractModel):
    kind: ActorKind
    account_id: Optional[ResourceId] = None
    installation_id: Optional[ResourceId] = None
    session_id: Optional[ResourceId] = None
    display: Optional[ShortText] = None


class AuditAction(str, Enum):
    # identity
    SETUP_CODE_ISSUED = "setup_code_issued"
    SETUP_CODE_REDEEMED = "setup_code_redeemed"
    BREAK_GLASS_USED = "break_glass_used"
    INVITATION_ISSUED = "invitation_issued"
    INVITATION_REDEEMED = "invitation_redeemed"
    INVITATION_REVOKED = "invitation_revoked"
    REINVITE_ISSUED = "reinvite_issued"
    ACCOUNT_CREATED = "account_created"
    ACCOUNT_UPDATED = "account_updated"
    ACCOUNT_DISABLED = "account_disabled"
    AUTHENTICATOR_ADDED = "authenticator_added"
    AUTHENTICATOR_REMOVED = "authenticator_removed"
    SESSION_REVOKED = "session_revoked"
    # installations and service keys
    INSTALLATION_REGISTERED = "installation_registered"
    INSTALLATION_RETIRED = "installation_retired"
    SERVICE_KEY_ISSUED = "service_key_issued"
    SERVICE_KEY_ROTATED = "service_key_rotated"
    SERVICE_KEY_REVOKED = "service_key_revoked"
    # trust and routes
    ROUTE_OPT_IN_CHANGED = "route_opt_in_changed"
    ENDPOINT_CHANGED = "endpoint_changed"
    MASKING_CHANGED = "masking_changed"
    RELEASE_APPROVED = "release_approved"
    RELEASE_APPROVAL_WITHDRAWN = "release_approval_withdrawn"
    RELEASE_DECLINED = "release_declined"
    RELEASE_REVOCATION_OBSERVED = "release_revocation_observed"
    ATTESTATION_POLICY_CHANGED = "attestation_policy_changed"
    MINIMUM_RELEASE_SVN_CHANGED = "minimum_release_svn_changed"
    TRUST_ANCHOR_PENDING = "trust_anchor_pending"
    TRUST_ANCHOR_ADOPTED = "trust_anchor_adopted"
    KEY_MANAGER_CHANGED = "key_manager_changed"
    NOTIFICATION_CHANGED = "notification_changed"
    KEY_RELEASED = "key_released"
    KEY_RELEASE_CLOSED = "key_release_closed"
    PRO1_CONNECTION_BLOCKED = "pro1_connection_blocked"
    PRO1_BLOCK_CLEARED = "pro1_block_cleared"
    PRO1_CONNECTION_VERIFIED = "pro1_connection_verified"
    LOG_EQUIVOCATION_DETECTED = "log_equivocation_detected"
    UPDATE_PACKAGE_STAGED = "update_package_staged"
    UPDATER_VERIFICATION = "updater_verification"
    UPDATE_INSTALLED = "update_installed"
    BUILD_APPROVAL_CHECK = "build_approval_check"
    # tls, hostname, backup
    HOSTNAME_SET = "hostname_set"
    CERTIFICATE_ISSUED = "certificate_issued"
    CERTIFICATE_IMPORTED = "certificate_imported"
    CERTIFICATE_RENEWED = "certificate_renewed"
    BACKUP_CREATED = "backup_created"
    RESTORE_PERFORMED = "restore_performed"
    # review and rubric administration
    RUBRIC_PUBLISHED = "rubric_published"
    RUBRIC_RETIRED = "rubric_retired"
    QUEUE_RULE_CHANGED = "queue_rule_changed"
    REVIEW_RESOLVED = "review_resolved"
    VERDICT_OVERRIDDEN = "verdict_overridden"
    ESCALATION_RESOLVED = "escalation_resolved"
    REVIEW_RETAINED = "review_retained"
    REANALYSIS_REQUESTED = "reanalysis_requested"
    # calls
    CALL_METADATA_UPDATED = "call_metadata_updated"
    """Added in 1.1.0. A re-registration of an existing source identity changed the conversation's
    call_metadata (calls.merge_call_metadata). Target ``call`` (``conversation`` when the
    conversation has no call); details carry ``conversation_id`` and ``updated_fields`` (the changed
    field names, comma-separated), never the values."""
    # contact signals v2 (1.3.0). Details carry IDs, versions and digests, never taxonomy text.
    SIGNAL_TAXONOMY_SAVED = "signal_taxonomy_saved"
    """Details: version, digest, and the paths of changed nodes."""
    SIGNAL_SETTINGS_CHANGED = "signal_settings_changed"
    """Details: the old and new pipeline."""
    SIGNAL_TAXONOMY_REDACTED = "signal_taxonomy_redacted"
    """Details: version and digest."""
    SIGNAL_ALERT_RULE_SAVED = "signal_alert_rule_saved"
    """Details: rule_id, record_version, enabled."""
    SIGNAL_BACKFILL_REQUESTED = "signal_backfill_requested"
    """Details: backfill_id, mode, window, requests_created."""
    SIGNAL_PREVIEW_REQUESTED = "signal_preview_requested"
    """Details: preview_id, taxonomy digest, call count."""
    SIGNAL_HIT_REVIEWED = "signal_hit_reviewed"
    """Details: call_id, hit_id, verdicts."""
    # processing
    JOB_CANCELLED = "job_cancelled"
    JOB_RETRIED = "job_retried"
    CATALOG_SNAPSHOT_PUBLISHED = "catalog_snapshot_published"
    USAGE_REPORT_EXPORTED = "usage_report_exported"
    PRICE_TABLE_CHANGED = "price_table_changed"
    MIGRATION_IMPORTED = "migration_imported"
    # on-device training (1.3.0, decision 28)
    TRAINING_LABELS_READ = "training_labels_read"
    """A ``listTrainingLabels`` read with ``limit > 0`` (count-only reads are not audited). Actor: the
    Process installation. Details: ``after``, ``next_after``, the item count and counts per kind;
    never content."""
    # dual transcription (1.3.0, decision 33)
    ASR_VOCABULARY_SAVED = "asr_vocabulary_saved"
    """An admin's ``saveAsrVocabulary``, or a pack install by Store's local ``apply-vocabulary-seed``
    (actor: the installer). Details: record_version, old and new effective digest, enabled, pack_id
    and version, and term counts per source; never the terms."""


class AuditTarget(ContractModel):
    kind: ShortText = Field(description="e.g. account, invitation, service_key, release, admin_state, job, call, rubric.")
    id: ShortText


class AuditEventBody(ContractModel):
    """Every field of an audit event except its digest."""

    id: ResourceId
    sequence: int = Field(ge=1, description="Dense, monotonic per Store.")
    occurred_at: Timestamp
    actor: Actor
    action: AuditAction
    target: AuditTarget
    details: Dict[str, JsonScalar] = Field(default_factory=dict, description="Safe details: destination host, digest, route class, reason. Never content or secrets.")
    previous_event_digest: Optional[Sha256Digest] = Field(default=None, description="Hash chain: digest of the preceding event; null for the first.")


def audit_event_digest(event: AuditEventBody) -> str:
    """canonical_digest of the event body (everything but event_digest, previous digest included)."""
    return canonical_digest(AuditEventBody.model_validate({k: getattr(event, k) for k in AuditEventBody.model_fields}))


class AuditEvent(AuditEventBody):
    event_digest: Sha256Digest = Field(description="audit_event_digest(): canonical_digest of the body, so the log is tamper-evident.")

    @model_validator(mode="after")
    def _digest(self):
        if self.event_digest != audit_event_digest(self):
            raise ValueError("event_digest is canonical_digest of the event body")
        return self


class AuditQuery(PageQuery):
    action: Optional[AuditAction] = None
    actor_kind: Optional[ActorKind] = None
    target_kind: Optional[ShortText] = None
    target_id: Optional[ShortText] = None
    since: Optional[Timestamp] = None
    until: Optional[Timestamp] = None


class ChangeKind(str, Enum):
    CALL = "call"
    RESULT = "result"
    REVIEW = "review"
    REVIEW_QUEUE = "review_queue"
    JOB_GROUP = "job_group"
    JOB = "job"
    RUBRIC = "rubric"
    ADMIN_STATE = "admin_state"
    PRO1_CONNECTION = "pro1_connection"
    REANALYSIS_REQUEST = "reanalysis_request"
    CATALOG = "catalog"
    SIGNAL_TAXONOMY = "signal_taxonomy"
    """Added in 1.3.0. The signal taxonomy or settings changed. resource_id ``signal_taxonomy``;
    status ``saved:v<N>`` (``signal_taxonomy_saved_status``) or ``settings``. Never text."""
    SIGNAL_ALERT_RULE = "signal_alert_rule"
    """Added in 1.3.0. resource_id is the rule ID; status ``saved``, ``enabled`` or ``disabled``."""
    SIGNAL_ALERT = "signal_alert"
    """Added in 1.3.0. A signals publish made an enabled alert rule match a call that did not match
    at its previous signals version. resource_id and call_id are the call's; status
    ``fired:<rule_id>`` (``signal_alert_fired_status``). Rule edits never fire events for older calls."""
    ASR_VOCABULARY = "asr_vocabulary"
    """Added in 1.3.0 (decision 33). The ASR vocabulary document changed (a save or a pack install).
    resource_id ``asr_vocabulary``; version is the record version; status ``saved`` or
    ``pack_installed``. Never terms."""


CHANGE_KINDS_BY_PRINCIPAL = {
    "process_service_key": [ChangeKind.JOB, ChangeKind.JOB_GROUP, ChangeKind.REANALYSIS_REQUEST, ChangeKind.ADMIN_STATE, ChangeKind.PRO1_CONNECTION, ChangeKind.CATALOG, ChangeKind.RUBRIC, ChangeKind.SIGNAL_TAXONOMY, ChangeKind.ASR_VOCABULARY],
    "reviewer": [ChangeKind.CALL, ChangeKind.RESULT, ChangeKind.REVIEW, ChangeKind.REVIEW_QUEUE, ChangeKind.RUBRIC, ChangeKind.REANALYSIS_REQUEST, ChangeKind.JOB_GROUP, ChangeKind.SIGNAL_TAXONOMY, ChangeKind.SIGNAL_ALERT_RULE, ChangeKind.SIGNAL_ALERT],
    "supervisor": [ChangeKind.CALL, ChangeKind.RESULT, ChangeKind.REVIEW, ChangeKind.REVIEW_QUEUE, ChangeKind.RUBRIC, ChangeKind.REANALYSIS_REQUEST, ChangeKind.JOB_GROUP, ChangeKind.JOB, ChangeKind.CATALOG, ChangeKind.SIGNAL_TAXONOMY, ChangeKind.SIGNAL_ALERT_RULE, ChangeKind.SIGNAL_ALERT],
    "admin": list(ChangeKind),
}
"""What each principal's feed contains. Store filters server-side: a ``kinds`` filter narrows it
further, and asking for a kind outside this list returns none of it (never 403). A reviewer sees
job-group progress counts but job events only from supervisor up (read_jobs). Since 1.3.0 Process
also receives ``signal_taxonomy`` (it mints snapshots of the current version), and reviewers and
supervisors receive all three signal kinds. Process also receives ``asr_vocabulary`` (1.3.0,
decision 33) so it can refresh the vocabulary it freezes into new ``asr`` jobs; admins get it with
every other kind."""

ASR_VOCABULARY_ID = "asr_vocabulary"
"""Added in 1.3.0. The resource_id of every ``asr_vocabulary`` change event (the document is a singleton)."""

ASR_VOCABULARY_STATUSES = ("saved", "pack_installed")
"""Added in 1.3.0. ``asr_vocabulary`` event statuses."""


SIGNAL_TAXONOMY_ID = "signal_taxonomy"
"""Added in 1.3.0. The resource_id of every ``signal_taxonomy`` change event (the document is a singleton)."""

SIGNAL_SETTINGS_STATUS = "settings"
"""Added in 1.3.0. ``signal_taxonomy`` event status when only the settings changed."""

SIGNAL_ALERT_RULE_STATUSES = ("saved", "enabled", "disabled")
"""Added in 1.3.0. ``signal_alert_rule`` event statuses."""

SIGNAL_FEEDBACK_STATUS = "signal_feedback"
"""Added in 1.3.0. Status of the ``review`` event Store emits when hit feedback is saved
(resource_id and call_id are the call's)."""


def signal_taxonomy_saved_status(version: int) -> str:
    """Added in 1.3.0. ``saved:v<N>``: the ``signal_taxonomy`` event status for a new published version."""
    return f"saved:v{version}"


def signal_alert_fired_status(rule_id: str) -> str:
    """Added in 1.3.0. ``fired:<rule_id>``: the ``signal_alert`` event status."""
    return f"fired:{rule_id}"


CALL_METADATA_UPDATED_STATUS = "metadata_updated"
"""Added in 1.1.0. ``ChangeEvent.status`` of the ``call`` event Store emits when a re-registration
changes a call's metadata (resource_id and call_id are the call's ID)."""


class ChangeEvent(ContractModel):
    cursor: ChangeCursor
    occurred_at: Timestamp
    kind: ChangeKind
    resource_id: ResourceId
    conversation_id: Optional[ResourceId] = None
    call_id: Optional[ResourceId] = None
    version: Optional[int] = Field(default=None, ge=0)
    status: Optional[ShortText] = None


class ChangeFeedQuery(ContractModel):
    after: Optional[ChangeCursor] = Field(default=None, description="Return events after this cursor; null starts from the oldest retained.")
    limit: int = Field(default=200, ge=1, le=1000)
    kinds: Optional[List[ChangeKind]] = None


class ChangeFeed(ContractModel):
    """Store assigns cursors in commit order: no event ever becomes visible below a cursor already
    served, so a reader that resumes from ``next_cursor`` misses nothing."""

    events: List[ChangeEvent]
    next_cursor: ChangeCursor = Field(description="Pass as 'after' on the next call: the scan position reached, which is at or after the request cursor even when no event matched the filter.")
    latest_cursor: ChangeCursor
    feed_epoch: ResourceId = Field(description="Changes on every restore; a cursor from another epoch is 410 cursor_unknown.")
    retention_until: Timestamp = Field(description="Cursors older than this may be expired (410 cursor_expired with oldest_cursor).")


class QueueWakeup(ContractModel):
    """The only payload a queue transport carries. Losing or duplicating it cannot lose or
    duplicate a job; a worker must find queued work after restart without it."""

    job_id: ResourceId
