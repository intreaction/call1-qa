"""The error model: API error codes with their HTTP status, and the safe per-attempt job error codes.

Every non-2xx Store response is an ``ErrorResponse``. Messages are safe text: they never carry
transcript content, prompts, provider bodies, endpoint URLs with credentials, or key material.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from pydantic import Field

from .common import ContractModel, JsonScalar, SafeText


class ErrorCode(str, Enum):
    """API-level error codes. ``ERROR_HTTP_STATUS`` maps each to exactly one HTTP status."""

    UNAUTHENTICATED = "unauthenticated"
    SESSION_EXPIRED = "session_expired"
    CSRF_FAILED = "csrf_failed"
    ORIGIN_NOT_ALLOWED = "origin_not_allowed"
    FORBIDDEN = "forbidden"
    INSUFFICIENT_SCOPE = "insufficient_scope"
    INSUFFICIENT_ROLE = "insufficient_role"
    NOT_FOUND = "not_found"
    VALIDATION_FAILED = "validation_failed"
    HOSTNAME_INVALID = "hostname_invalid"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    CONFLICT = "conflict"
    REVIEW_VERSION_CONFLICT = "review_version_conflict"
    STATE_VERSION_CONFLICT = "state_version_conflict"
    RUBRIC_VERSION_CONFLICT = "rubric_version_conflict"
    CLAIM_TOKEN_STALE = "claim_token_stale"
    JOB_NOT_CLAIMABLE = "job_not_claimable"
    JOB_CANCELLING = "job_cancelling"
    """A completion arrived after cancel_requested was set; the worker reports failure code cancelled instead."""
    INVALID_TRANSITION = "invalid_transition"
    IDEMPOTENCY_KEY_REUSED = "idempotency_key_reused"
    COMPLETION_KEY_REUSED = "completion_key_reused"
    GRAPH_INVALID = "graph_invalid"
    CURSOR_EXPIRED = "cursor_expired"
    CURSOR_UNKNOWN = "cursor_unknown"
    """A change cursor from another feed epoch (before a restore) or ahead of the latest; re-snapshot."""
    UPLOAD_EXPIRED = "upload_expired"
    INVITATION_INVALID = "invitation_invalid"
    SETUP_CODE_INVALID = "setup_code_invalid"
    WEBAUTHN_VERIFICATION_FAILED = "webauthn_verification_failed"
    ACCOUNT_DISABLED = "account_disabled"
    ROUTE_NOT_PERMITTED = "route_not_permitted"
    RATE_LIMITED = "rate_limited"
    STORE_UNAVAILABLE = "store_unavailable"
    SEARCH_UNAVAILABLE = "search_unavailable"
    """Added in 1.2.0. Semantic search cannot embed the query: Store's search embedder is not
    installed or failed to load. details.reason is ``not_installed`` or ``load_failed``;
    details.embedding_scheme names the scheme Store is configured for. Not retryable."""
    NOT_IMPLEMENTED = "not_implemented"
    SIGNAL_TAXONOMY_CONFLICT = "signal_taxonomy_conflict"
    """Added in 1.3.0. A signal taxonomy or settings save named a stale ``expected_record_version``
    (another admin saved first). details.current_version (the published taxonomy version) and
    details.record_version (the record's concurrency token). Re-read and re-apply."""


ERROR_HTTP_STATUS: Dict[ErrorCode, int] = {
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.SESSION_EXPIRED: 401,
    ErrorCode.WEBAUTHN_VERIFICATION_FAILED: 401,
    ErrorCode.CSRF_FAILED: 403,
    ErrorCode.ORIGIN_NOT_ALLOWED: 403,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.INSUFFICIENT_SCOPE: 403,
    ErrorCode.INSUFFICIENT_ROLE: 403,
    ErrorCode.ACCOUNT_DISABLED: 403,
    ErrorCode.ROUTE_NOT_PERMITTED: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.REVIEW_VERSION_CONFLICT: 409,
    ErrorCode.STATE_VERSION_CONFLICT: 409,
    ErrorCode.RUBRIC_VERSION_CONFLICT: 409,
    ErrorCode.SIGNAL_TAXONOMY_CONFLICT: 409,
    ErrorCode.CLAIM_TOKEN_STALE: 409,
    ErrorCode.JOB_NOT_CLAIMABLE: 409,
    ErrorCode.JOB_CANCELLING: 409,
    ErrorCode.INVALID_TRANSITION: 409,
    ErrorCode.IDEMPOTENCY_KEY_REUSED: 409,
    ErrorCode.COMPLETION_KEY_REUSED: 409,
    ErrorCode.CURSOR_EXPIRED: 410,
    ErrorCode.CURSOR_UNKNOWN: 410,
    ErrorCode.UPLOAD_EXPIRED: 410,
    ErrorCode.INVITATION_INVALID: 410,
    ErrorCode.SETUP_CODE_INVALID: 410,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.VALIDATION_FAILED: 422,
    ErrorCode.HOSTNAME_INVALID: 422,
    ErrorCode.CHECKSUM_MISMATCH: 422,
    ErrorCode.GRAPH_INVALID: 422,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.NOT_IMPLEMENTED: 501,
    ErrorCode.STORE_UNAVAILABLE: 503,
    ErrorCode.SEARCH_UNAVAILABLE: 503,
}


class ErrorResponse(ContractModel):
    """Body of every non-2xx response."""

    code: ErrorCode
    message: SafeText = Field(description="Safe, human-readable summary. Never customer content or secrets.")
    details: Dict[str, JsonScalar] = Field(default_factory=dict, description="Safe structured context, e.g. current_version, oldest_cursor, expected_state_version.")
    retryable: bool = Field(default=False, description="True when the same request may succeed later without change.")
    request_id: Optional[str] = Field(default=None, description="Store-assigned ID for support correlation.")


class JobErrorClass(str, Enum):
    """How Store schedules after a failed attempt (see JOB_ERROR_CLASSES)."""

    TRANSIENT = "transient"
    """Retry on the same route with backoff while attempts remain."""
    CONFIGURATION = "configuration"
    """No automatic retry; needs an operator or admin change, then a manual retry."""
    DEFINITIVE = "definitive"
    """No automatic retry. For Pro1 codes the connection also enters the blocked state."""
    TERMINAL = "terminal"
    """Outcome, not a fault (cancelled, validation rejected). No automatic retry."""


class JobErrorCode(str, Enum):
    """Safe error codes an attempt may end with. The set is closed: unknown codes are rejected."""

    VALIDATION_REJECTED = "validation_rejected"
    PROVIDER_ERROR = "provider_error"
    PROVIDER_TIMEOUT = "provider_timeout"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_UNQUALIFIED = "model_unqualified"
    ROUTE_DISABLED = "route_disabled"
    ROUTE_POLICY_REJECTED = "route_policy_rejected"
    OLLAMA_REMOTE_MODEL_REJECTED = "ollama_remote_model_rejected"
    CREDENTIAL_MISSING = "credential_missing"
    CONFIGURATION_ERROR = "configuration_error"
    INPUT_UNAVAILABLE = "input_unavailable"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    LEASE_EXPIRED = "lease_expired"
    WORKER_CRASHED = "worker_crashed"
    CANCELLED = "cancelled"
    STORE_PUBLICATION_FAILED = "store_publication_failed"
    # Pro1 attested route (Pro1ConfidentialInference.md section 4)
    PRO1_KEY_UNAVAILABLE = "pro1_key_unavailable"
    PRO1_RELEASE_UNAPPROVED = "pro1_release_unapproved"
    PRO1_UNREACHABLE = "pro1_unreachable"
    PRO1_REVOCATION_UNAVAILABLE = "pro1_revocation_unavailable"
    PRO1_ATTESTATION_INVALID = "pro1_attestation_invalid"
    PRO1_PLATFORM_UNENDORSED = "pro1_platform_unendorsed"
    PRO1_ATTESTATION_STALE = "pro1_attestation_stale"
    PRO1_MEASUREMENT_REJECTED = "pro1_measurement_rejected"
    PRO1_TEE_POLICY_REJECTED = "pro1_tee_policy_rejected"
    PRO1_BINDING_FAILED = "pro1_binding_failed"
    PRO1_MODEL_REVISION_MISMATCH = "pro1_model_revision_mismatch"
    PRO1_RESPONSE_REJECTED = "pro1_response_rejected"
    PRO1_SERVICE_ERROR = "pro1_service_error"
    PRO1_CONNECTION_BLOCKED = "pro1_connection_blocked"


JOB_ERROR_CLASSES: Dict[JobErrorCode, JobErrorClass] = {
    JobErrorCode.VALIDATION_REJECTED: JobErrorClass.TERMINAL,
    JobErrorCode.PROVIDER_ERROR: JobErrorClass.TRANSIENT,
    JobErrorCode.PROVIDER_TIMEOUT: JobErrorClass.TRANSIENT,
    JobErrorCode.CONTEXT_LIMIT_EXCEEDED: JobErrorClass.CONFIGURATION,
    JobErrorCode.MODEL_UNAVAILABLE: JobErrorClass.CONFIGURATION,
    JobErrorCode.MODEL_UNQUALIFIED: JobErrorClass.CONFIGURATION,
    JobErrorCode.ROUTE_DISABLED: JobErrorClass.CONFIGURATION,
    JobErrorCode.ROUTE_POLICY_REJECTED: JobErrorClass.CONFIGURATION,
    JobErrorCode.OLLAMA_REMOTE_MODEL_REJECTED: JobErrorClass.CONFIGURATION,
    JobErrorCode.CREDENTIAL_MISSING: JobErrorClass.CONFIGURATION,
    JobErrorCode.CONFIGURATION_ERROR: JobErrorClass.CONFIGURATION,
    JobErrorCode.INPUT_UNAVAILABLE: JobErrorClass.TRANSIENT,
    JobErrorCode.RESOURCE_UNAVAILABLE: JobErrorClass.TRANSIENT,
    JobErrorCode.LEASE_EXPIRED: JobErrorClass.TRANSIENT,
    JobErrorCode.WORKER_CRASHED: JobErrorClass.TRANSIENT,
    JobErrorCode.CANCELLED: JobErrorClass.TERMINAL,
    JobErrorCode.STORE_PUBLICATION_FAILED: JobErrorClass.TRANSIENT,
    JobErrorCode.PRO1_KEY_UNAVAILABLE: JobErrorClass.CONFIGURATION,
    JobErrorCode.PRO1_RELEASE_UNAPPROVED: JobErrorClass.TRANSIENT,
    JobErrorCode.PRO1_UNREACHABLE: JobErrorClass.TRANSIENT,
    JobErrorCode.PRO1_REVOCATION_UNAVAILABLE: JobErrorClass.TRANSIENT,
    JobErrorCode.PRO1_ATTESTATION_INVALID: JobErrorClass.DEFINITIVE,
    JobErrorCode.PRO1_PLATFORM_UNENDORSED: JobErrorClass.DEFINITIVE,
    JobErrorCode.PRO1_ATTESTATION_STALE: JobErrorClass.DEFINITIVE,
    JobErrorCode.PRO1_MEASUREMENT_REJECTED: JobErrorClass.DEFINITIVE,
    JobErrorCode.PRO1_TEE_POLICY_REJECTED: JobErrorClass.DEFINITIVE,
    JobErrorCode.PRO1_BINDING_FAILED: JobErrorClass.DEFINITIVE,
    JobErrorCode.PRO1_MODEL_REVISION_MISMATCH: JobErrorClass.CONFIGURATION,
    JobErrorCode.PRO1_RESPONSE_REJECTED: JobErrorClass.DEFINITIVE,
    JobErrorCode.PRO1_SERVICE_ERROR: JobErrorClass.TRANSIENT,
    JobErrorCode.PRO1_CONNECTION_BLOCKED: JobErrorClass.CONFIGURATION,
}

PROVIDER_FAILURE_CODES = frozenset({
    JobErrorCode.PROVIDER_ERROR, JobErrorCode.PROVIDER_TIMEOUT,
    JobErrorCode.PRO1_UNREACHABLE, JobErrorCode.PRO1_SERVICE_ERROR,
})
"""Transient provider faults. On a QA assessment job's final attempt they are recorded as a FLAGGED
assessment with trigger ``provider_error`` (the pre-split router's behaviour), which may fire the
criterion's configured escalation; on earlier attempts they are reported with ``/fail`` and retried.
A completion whose usage outcome is ``failed`` carries one of these codes and nothing else."""

PRO1_PROVIDER_FAILURE_CODES = frozenset(code for code in PROVIDER_FAILURE_CODES if code.value.startswith("pro1_"))
"""The Pro1 route's provider faults, ``pro1_unreachable`` and ``pro1_service_error``: the only Pro1
codes for which a configured escalation still applies (Pro1ConfidentialInference.md section 4,
"Escalation"). On the ``call1_confidential`` route a FLAGGED provider failure uses one of these,
never ``provider_error`` or ``provider_timeout``, and a completion on another route never uses them."""

PRO1_ATTESTATION_CLASS_CODES = frozenset(
    code for code in JobErrorCode
    if code.value.startswith("pro1_") and JOB_ERROR_CLASSES[code] is not JobErrorClass.TRANSIENT
) | {JobErrorCode.PRO1_REVOCATION_UNAVAILABLE, JobErrorCode.PRO1_RELEASE_UNAPPROVED}
"""Codes that never trigger the question-level provider-failure escalation
(Pro1ConfidentialInference.md section 4, "Escalation")."""
