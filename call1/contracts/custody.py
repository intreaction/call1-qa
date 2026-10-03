"""Data custody: route classes, the per-job route record, Pro1 attestation evidence, key release and
the Pro1 connection state.

The invariant (StrategyAlignment team decision 4): Call1, the company, never sees customer data.
Every LLM route belongs to exactly one route class, frozen on each job. A Call1-operated
destination is reachable only through the attested ``call1_confidential`` route, and a job on that
route records attestation evidence and a key-release reference on every attempt that reached key
release, and its rejected evidence on every attempt that did not.

``call1_operated_host`` is the Stage 0 rule (``call1.question_models.is_call1_operated``) stated as
a pure function of the host and the configured Pro1 endpoint host, with no environment lookups, so
Store, Process and Evaluate evaluate the same thing. ``tests/test_contracts.py`` proves it agrees
with the Stage 0 guard on every case, including IPv6 literals.
"""

from __future__ import annotations

import ipaddress
from enum import Enum
from typing import Dict, Iterable, List, Literal, Optional
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .common import CALL1_DOMAINS, ContractModel, ModelCredentialEnvName, ResourceId, SafeText, Sha256Digest, ShortText, Timestamp
from .errors import ErrorCode, JobErrorClass, JobErrorCode, JOB_ERROR_CLASSES


class RouteClass(str, Enum):
    """Who can read a job's data (plan, "Data custody"). Frozen on every job and attempt."""

    APPLIANCE = "appliance"
    """Included MLX model in-process, or Ollama on loopback serving local weights only."""
    CUSTOMER_LAN = "customer_lan"
    """Ollama on a customer-operated LAN host over HTTPS; audited admin opt-in."""
    CALL1_CONFIDENTIAL = "call1_confidential"
    """Pro1: attested enclave running a customer-approved release; session key released by the
    customer's key-release service. Off by default; audited admin opt-in."""
    CUSTOMER_DIRECTED = "customer_directed"
    """Bring your own (BYOK): a provider under the customer's own contract and credentials,
    reached directly from Process. Off by default; audited admin opt-in; labeled with destination."""


class ProviderType(str, Enum):
    """v1 provider protocols (plan, "Execution backends"). PAIR, LM Studio, Anthropic and
    OpenAI Batch are future work and absent by decision (FutureProviders.md)."""

    MLX = "mlx"
    OLLAMA = "ollama"
    PRO1 = "pro1"
    BYOK = "byok"


class RouteClassRule(ContractModel):
    """The admission rule for a route class, as data so UI and tests read the same thing."""

    route_class: RouteClass
    provider_types: List[ProviderType]
    who_can_read: ShortText
    default_enabled: bool
    opt_in_audited: bool
    requires_attestation: bool
    requires_key_release: bool
    destination_may_be_call1_operated: bool
    plaintext_loopback_allowed: bool


ROUTE_CLASS_RULES: Dict[RouteClass, RouteClassRule] = {
    RouteClass.APPLIANCE: RouteClassRule(
        route_class=RouteClass.APPLIANCE, provider_types=[ProviderType.MLX, ProviderType.OLLAMA],
        who_can_read="The customer (Process host)", default_enabled=True, opt_in_audited=False,
        requires_attestation=False, requires_key_release=False, destination_may_be_call1_operated=False,
        plaintext_loopback_allowed=True),
    RouteClass.CUSTOMER_LAN: RouteClassRule(
        route_class=RouteClass.CUSTOMER_LAN, provider_types=[ProviderType.OLLAMA],
        who_can_read="The customer", default_enabled=False, opt_in_audited=True,
        requires_attestation=False, requires_key_release=False, destination_may_be_call1_operated=False,
        plaintext_loopback_allowed=False),
    RouteClass.CALL1_CONFIDENTIAL: RouteClassRule(
        route_class=RouteClass.CALL1_CONFIDENTIAL, provider_types=[ProviderType.PRO1],
        who_can_read="Call1-authored release code inside an attested enclave the customer's admin approved; Call1 operators see ciphertext only",
        default_enabled=False, opt_in_audited=True, requires_attestation=True, requires_key_release=True,
        destination_may_be_call1_operated=True, plaintext_loopback_allowed=False),
    RouteClass.CUSTOMER_DIRECTED: RouteClassRule(
        route_class=RouteClass.CUSTOMER_DIRECTED, provider_types=[ProviderType.BYOK],
        who_can_read="The customer and the provider the customer contracted", default_enabled=False,
        opt_in_audited=True, requires_attestation=False, requires_key_release=False,
        destination_may_be_call1_operated=False, plaintext_loopback_allowed=False),
}

PRO1_CONNECTION_REF = "pro1"
"""The provider_connection_ref every call1_confidential route uses (the one Pro1 connection)."""

IN_PROCESS_DESTINATION = "in-process"
"""Destination host recorded for the included MLX model, which runs inside Process."""
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")

CALL1_OPERATED_DOMAINS = tuple(CALL1_DOMAINS)
"""Domains the host rule always treats as Call1-operated, in addition to the configured Pro1
endpoint host. Exported in the OpenAPI document (``x-call1.call1_operated_domains``) so Evaluate's
forms apply the same rule."""


def normalize_host(host: str) -> str:
    """Lower-case, strip whitespace, one trailing dot and IPv6 brackets."""
    value = (host or "").strip().lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value.rstrip(".")


def host_of(base_url: Optional[str]) -> str:
    """The normalized host of a URL (IPv6 brackets removed), or '' when it has none."""
    try:
        return normalize_host(urlsplit(base_url or "").hostname or "")
    except ValueError:
        return ""


def call1_operated_host(host: str, pro1_endpoint_hosts: Iterable[str] = ()) -> bool:
    """The Stage 0 host rule, pure and environment-independent.

    True for a host equal to or under a Call1-owned domain (``CALL1_OPERATED_DOMAINS``) or under
    one of ``pro1_endpoint_hosts`` (the Pro1 endpoint host from admin state, which Process's
    ``CALL1_PRO1_ENDPOINT`` also names), compared case-insensitively with a trailing dot stripped;
    for a missing host; and for a public IP literal, IPv4 or IPv6, whose owner cannot be checked.
    Private, loopback and link-local literals are not matched.
    """
    value = normalize_host(host)
    if not value:
        return True
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        domains = {normalize_host(d) for d in (*CALL1_OPERATED_DOMAINS, *pro1_endpoint_hosts) if d and normalize_host(d)}
        return any(value == d or value.endswith("." + d) for d in domains)
    return not (address.is_private or address.is_loopback or address.is_link_local)


is_call1_operated_host = call1_operated_host
"""Alias kept for readers of the first draft."""


class RouteRecord(ContractModel):
    """The route a job is frozen to. Recorded on the job at creation and on every attempt.

    Model validation applies the static part of the host rule. At graph creation Store also
    checks ``destination_host`` against ``call1_operated_host(host, [the admin-state Pro1 endpoint
    host])`` and that ``provider_connection_ref`` names an enabled connection in admin state whose
    host and credential name match; a mismatch is 403 ``route_not_permitted``.
    """

    route_class: RouteClass
    provider_type: ProviderType
    destination_host: ShortText = Field(description="Hostname the request goes to; 'in-process' for the included MLX model. Shown wherever the route is selectable.")
    masked: bool = Field(description="Whether transcript masking was applied for this route. Masking is data minimization, never what permits a route.")
    masking_override_audit_event_id: Optional[ResourceId] = Field(default=None, description="Required when masked is false on a non-appliance route: the audited admin action that allowed it.")
    provider_connection_ref: Optional[ResourceId] = Field(default=None, description="The admin-state connection this route uses (CustomerLanHost, ByokProvider, or 'pro1'). Required for every non-appliance route. Not a credential.")
    credential_env_name: Optional[ModelCredentialEnvName] = Field(default=None, description="Name of the Process environment variable holding the provider credential (BYOK, customer-LAN proxy, Pro1 account). Never the value.")
    attestation_policy_version: Optional[Sha256Digest] = Field(default=None, description="Digest of the customer's attestation policy the job was created under. Required for call1_confidential.")

    @model_validator(mode="after")
    def _enforce_route_rules(self):
        rule = ROUTE_CLASS_RULES[self.route_class]
        if self.provider_type not in rule.provider_types:
            raise ValueError(f"provider {self.provider_type.value} is not allowed on route class {self.route_class.value}")
        host = normalize_host(self.destination_host)
        if self.route_class is RouteClass.APPLIANCE:
            if self.provider_type is ProviderType.MLX and host != IN_PROCESS_DESTINATION:
                raise ValueError("the included MLX model runs in-process")
            if self.provider_type is ProviderType.OLLAMA and host not in LOOPBACK_HOSTS:
                raise ValueError("appliance Ollama must be on loopback; a LAN host is customer_lan")
        else:
            if host == IN_PROCESS_DESTINATION or host in LOOPBACK_HOSTS:
                raise ValueError("only the appliance route class may name loopback or in-process destinations")
            if self.provider_connection_ref is None:
                raise ValueError("a non-appliance route names its admin-state connection")
            if (self.route_class is RouteClass.CALL1_CONFIDENTIAL) != (self.provider_connection_ref == PRO1_CONNECTION_REF):
                raise ValueError("the call1_confidential route, and only it, uses the 'pro1' connection")
        if not rule.destination_may_be_call1_operated and host != IN_PROCESS_DESTINATION and call1_operated_host(host):
            raise ValueError("a Call1-operated host is reachable only through the attested call1_confidential route")
        if self.route_class is RouteClass.CALL1_CONFIDENTIAL and self.attestation_policy_version is None:
            raise ValueError("call1_confidential jobs record the attestation policy version")
        if self.route_class is not RouteClass.APPLIANCE and not self.masked and not self.masking_override_audit_event_id:
            raise ValueError("unmasked non-appliance routes require the audited admin action that allowed it")
        return self


# --- Pro1 attestation evidence -------------------------------------------------------------


class Pro1Platform(str, Enum):
    AZURE_SNP_PARAVISOR = "azure-snp-paravisor"
    SEV_SNP = "sev-snp"
    TDX = "tdx"


class GpuClaims(ContractModel):
    model: ShortText
    driver_version: ShortText
    vbios_version: ShortText
    confidential_mode: bool
    devtools_mode: bool = False


class Pro1AttestationRecord(ContractModel):
    """What a call1_confidential attempt that passed verification records
    (Pro1ConfidentialInference.md 3.2 and 3.4).

    The evidence bundle itself is stored once, conversation-independent, as an
    ``attestation_evidence`` artifact named by its digest (``POST /attestation-evidence/uploads``);
    this record points at it. It never contains customer data or key material.
    """

    evidence_digest: Sha256Digest = Field(description="SHA-256 of the evidence bundle bytes; equals the attestation_evidence artifact's checksum.")
    evidence_artifact_id: ResourceId = Field(description="The stored attestation_evidence artifact (bundle plus fetched collateral, CRLs, OCSP responses and checkpoint).")
    policy_version: Sha256Digest = Field(description="AdminState.attestation_policy_digest at verification time: canonical_digest of the AttestationPolicy used.")
    trust_anchor_bundle_digest: Sha256Digest = Field(description="The adopted trust-anchor bundle the verifier used.")
    approval_id: ResourceId = Field(description="The admin approval that made this release eligible.")
    release_id: ShortText
    manifest_digest: Sha256Digest = Field(description="Equals the approved manifest digest; a matching release ID alone is never enough.")
    release_svn: int = Field(ge=0, description="Release security version; at or above the policy minimum.")
    log_id: ShortText
    log_index: int = Field(ge=0)
    checkpoint_digest: Sha256Digest = Field(description="Digest of the witness-cosigned checkpoint used.")
    revocation_list_sequence: int = Field(ge=0)
    platform: Pro1Platform
    gpus: List[GpuClaims] = Field(min_length=1)
    session_id: ResourceId
    key_id: ShortText = Field(description="One-way identifier derived from the session key. Never the key.")
    key_release_ref: ResourceId = Field(description="The key-release record for this session.")
    model_id: ShortText = Field(description="Model ID echoed in the sealed response.")
    weights_digest: Sha256Digest = Field(description="Weights digest echoed in the sealed response; equals the frozen selection.")
    verified_at: Timestamp
    expires_at: Timestamp


class Pro1FailedCheck(str, Enum):
    """Where a Pro1 attempt stopped (Pro1ConfidentialInference.md 2.2 and section 4)."""

    ADMISSION = "admission"
    EVIDENCE_REQUEST = "evidence_request"
    CPU_TEE_AUTHENTICITY = "cpu_tee_authenticity"
    TEE_POLICY = "tee_policy"
    FLEET_EVIDENCE = "fleet_evidence"
    MEASUREMENTS = "measurements"
    MANIFEST_APPROVAL = "manifest_approval"
    TRANSPARENCY_AND_REVOCATION = "transparency_and_revocation"
    GPU = "gpu"
    BINDING = "binding"
    RELEASE_KEY = "release_key"
    KEY_RELEASE = "key_release"
    MODEL_REVISION = "model_revision"
    RESPONSE = "response"
    SERVICE = "service"
    INTERRUPTED = "interrupted"
    """Not a Pro1 check: the attempt stopped before verification for a reason outside the Pro1 route
    (``PRO1_INTERRUPTION_CODES``: cancelled, worker crash, input, resource or publication failure)."""


PRO1_ADMISSION_CODES = frozenset({JobErrorCode.ROUTE_DISABLED, JobErrorCode.MODEL_UNQUALIFIED})
"""Admission codes a Pro1 attempt may record with a Pro1 check (Pro1ConfidentialInference.md section 4)."""

PRO1_INTERRUPTION_CODES = frozenset({
    JobErrorCode.CANCELLED, JobErrorCode.WORKER_CRASHED, JobErrorCode.INPUT_UNAVAILABLE,
    JobErrorCode.RESOURCE_UNAVAILABLE, JobErrorCode.STORE_PUBLICATION_FAILED,
})
"""Non-Pro1 codes a call1_confidential attempt can end with before it reached verification. Its
failure evidence then names ``failed_check: interrupted``. After verification the attempt carries
its attestation record instead and needs no failure evidence for these codes."""


class Pro1AttemptFailureEvidence(ContractModel):
    """What a call1_confidential attempt that failed records (Pro1ConfidentialInference.md 3.4:
    provenance is recorded 'whether it succeeded or failed'). A failure before key release has no
    key-release reference; one after (response rejected, service error) also carries the full
    ``Pro1AttestationRecord`` on the provenance.

    ``error_code`` is the code the attempt ended with: the ``/fail`` request's ``error_code``, or
    the usage row's code on a FLAGGED completion (``pro1_unreachable``, ``pro1_service_error``).
    A Pro1 or admission code names the Pro1 check that failed; a ``PRO1_INTERRUPTION_CODES`` code
    names ``interrupted``."""

    failed_check: Pro1FailedCheck
    error_code: JobErrorCode
    policy_version: Sha256Digest
    evidence_digest: Optional[Sha256Digest] = Field(default=None, description="The rejected bundle, when one was received; stored as an attestation_evidence artifact like any other.")
    evidence_artifact_id: Optional[ResourceId] = None
    release_id: Optional[ShortText] = Field(default=None, description="As parsed from the bundle, when it could be.")
    manifest_digest: Optional[Sha256Digest] = None
    session_id: Optional[ResourceId] = None

    @model_validator(mode="after")
    def _pro1_code(self):
        pro1_check = self.error_code.value.startswith("pro1_") or self.error_code in PRO1_ADMISSION_CODES
        if not pro1_check and self.error_code not in PRO1_INTERRUPTION_CODES:
            raise ValueError("a Pro1 failure carries a Pro1, admission or interruption error code")
        if pro1_check == (self.failed_check is Pro1FailedCheck.INTERRUPTED):
            raise ValueError("a Pro1 or admission code names the Pro1 check that failed; an interruption code names 'interrupted'")
        if (self.evidence_artifact_id is None) != (self.evidence_digest is None):
            raise ValueError("rejected evidence is named by digest and artifact together")
        return self


# --- Key source and key release ------------------------------------------------------------


class KeySourceKind(str, Enum):
    """The key-release service's key source (Pro1ConfidentialInference.md 2.4)."""

    LOCAL = "local"
    """v1 backend: 32 bytes from the OS CSPRNG on the appliance."""
    CUSTOMER_ANCHOR = "customer_anchor"
    """Optional: generated or unwrapped by the customer's own KMS or HSM under a scoped credential."""


class KeyAnchorProvider(str, Enum):
    AWS_KMS = "aws_kms"
    AZURE_KEY_VAULT = "azure_key_vault"
    GOOGLE_CLOUD_KMS = "google_cloud_kms"
    PKCS11_HSM = "pkcs11_hsm"


class KeySourceRef(ContractModel):
    kind: KeySourceKind
    anchor_provider: Optional[KeyAnchorProvider] = None
    anchor_key_ref: Optional[ShortText] = Field(default=None, description="The customer's key identifier at the anchor (an ARN, key URI or PKCS#11 label). Not a secret.")

    @model_validator(mode="after")
    def _anchor_fields(self):
        if self.kind is KeySourceKind.CUSTOMER_ANCHOR and not (self.anchor_provider and self.anchor_key_ref):
            raise ValueError("a customer anchor names its provider and key reference")
        if self.kind is KeySourceKind.LOCAL and (self.anchor_provider or self.anchor_key_ref):
            raise ValueError("the local key source has no anchor")
        return self


class KeyReleaseInput(ContractModel):
    """Written by Process (scope key-release:write) at step 3 of the attested flow, before the
    wrapped key leaves the appliance. Store checks ``KEY_RELEASE_PRECONDITIONS``, records it and
    writes a ``key_released`` audit event. Process enforces the same conditions itself before it
    wraps the key; Store's checks are defense in depth and the customer's audit record."""

    session_id: ResourceId
    key_id: ShortText = Field(description="One-way identifier derived from the session key; never the key.")
    key_source: KeySourceRef
    evidence_digest: Sha256Digest
    release_id: ShortText
    manifest_digest: Sha256Digest
    approval_id: ResourceId
    policy_version: Sha256Digest
    released_at: Timestamp
    expires_at: Timestamp


class KeyReleasePrecondition(ContractModel):
    check: ShortText
    error: ErrorCode
    reason: ShortText = Field(description="The value Store puts in ErrorResponse.details.reason.")


KEY_RELEASE_PRECONDITIONS: List[KeyReleasePrecondition] = [
    KeyReleasePrecondition(check="The call1_confidential route opt-in is enabled in admin state", error=ErrorCode.ROUTE_NOT_PERMITTED, reason="route_disabled"),
    KeyReleasePrecondition(check="The Pro1 connection is not blocked", error=ErrorCode.ROUTE_NOT_PERMITTED, reason="pro1_connection_blocked"),
    KeyReleasePrecondition(check="approval_id names a pro1_release approval whose status is approved", error=ErrorCode.CONFLICT, reason="approval_not_active"),
    KeyReleasePrecondition(check="That approval's release_id and manifest_sha256 equal release_id and manifest_digest", error=ErrorCode.CONFLICT, reason="approval_mismatch"),
    KeyReleasePrecondition(check="policy_version equals the current AdminState.attestation_policy_digest", error=ErrorCode.CONFLICT, reason="policy_version_stale"),
    KeyReleasePrecondition(check="key_source equals AdminState.key_manager.key_source", error=ErrorCode.CONFLICT, reason="key_source_mismatch"),
    KeyReleasePrecondition(check="evidence_digest names a committed attestation_evidence artifact", error=ErrorCode.NOT_FOUND, reason="evidence_not_stored"),
    KeyReleasePrecondition(check="session_id has no other key-release record (natural idempotency returns the same record for an identical replay)", error=ErrorCode.CONFLICT, reason="session_already_released"),
]
"""What Store verifies before recording a key release, in order, each with the error it returns."""


class KeyReleaseCloseReason(str, Enum):
    LIFETIME = "lifetime"
    POLICY_CHANGED = "policy_changed"
    RELEASE_REVOKED = "release_revoked"
    ROUTE_DISABLED = "route_disabled"
    RESPONSE_REJECTED = "response_rejected"
    PROCESS_RESTART = "process_restart"
    CONNECTION_BLOCKED = "connection_blocked"


class KeyReleaseRecord(KeyReleaseInput):
    id: ResourceId
    installation_id: ResourceId = Field(description="The Process installation whose service key wrote the record.")
    closed_at: Optional[Timestamp] = None
    close_reason: Optional[KeyReleaseCloseReason] = None
    audit_event_id: ResourceId


class KeyReleaseClose(ContractModel):
    closed_at: Timestamp
    close_reason: KeyReleaseCloseReason


# --- Pro1 connection state -----------------------------------------------------------------


class Pro1ConnectionStatus(str, Enum):
    PENDING_VERIFICATION = "pending_verification"
    """No passing verification since the route was enabled or a block was cleared. Pro1 jobs wait."""
    READY = "ready"
    BLOCKED = "blocked"
    """A definitive verification failure. Pro1 jobs fail at admission until an admin clears it and a
    fresh verification passes (Pro1ConfidentialInference.md section 4)."""


class Pro1ConnectionTrigger(str, Enum):
    VERIFICATION_PASSED = "verification_passed"
    DEFINITIVE_FAILURE = "definitive_failure"
    BLOCK_CLEARED = "block_cleared"
    ROUTE_DISABLED = "route_disabled"


class Pro1ConnectionTransition(ContractModel):
    from_status: Pro1ConnectionStatus
    to_status: Pro1ConnectionStatus
    trigger: Pro1ConnectionTrigger
    by: ShortText
    audited: bool


PRO1_CONNECTION_TRANSITIONS: List[Pro1ConnectionTransition] = [
    Pro1ConnectionTransition(from_status=Pro1ConnectionStatus.PENDING_VERIFICATION, to_status=Pro1ConnectionStatus.READY, trigger=Pro1ConnectionTrigger.VERIFICATION_PASSED, by="Store, on a passing Pro1VerificationReport from Process", audited=True),
    Pro1ConnectionTransition(from_status=Pro1ConnectionStatus.PENDING_VERIFICATION, to_status=Pro1ConnectionStatus.BLOCKED, trigger=Pro1ConnectionTrigger.DEFINITIVE_FAILURE, by="Store, on a Pro1VerificationReport with a definitive code", audited=True),
    Pro1ConnectionTransition(from_status=Pro1ConnectionStatus.READY, to_status=Pro1ConnectionStatus.BLOCKED, trigger=Pro1ConnectionTrigger.DEFINITIVE_FAILURE, by="Store, inside POST /jobs/{id}/fail with a definitive Pro1 code, or on a Pro1VerificationReport with one", audited=True),
    Pro1ConnectionTransition(from_status=Pro1ConnectionStatus.BLOCKED, to_status=Pro1ConnectionStatus.PENDING_VERIFICATION, trigger=Pro1ConnectionTrigger.BLOCK_CLEARED, by="An admin, POST /admin/pro1/clear-block", audited=True),
    Pro1ConnectionTransition(from_status=Pro1ConnectionStatus.READY, to_status=Pro1ConnectionStatus.PENDING_VERIFICATION, trigger=Pro1ConnectionTrigger.ROUTE_DISABLED, by="Store, when an admin disables the call1_confidential opt-in", audited=True),
]
"""The only Pro1 connection transitions. BLOCKED leaves only through an admin clear followed by a
passing verification; enabling or disabling the route never clears a block. Store is the only
writer of this record; Process reports verification outcomes as events and never sets a status."""


class Pro1Connection(ContractModel):
    """Store-held state of the Pro1 connection. Not part of AdminState: it has its own version, its
    own transitions, and no admin change request can write it. A fresh install starts at
    pending_verification with connection_version 0."""

    status: Pro1ConnectionStatus
    connection_version: int = Field(ge=0, description="Bumped by every transition; POST /admin/pro1/clear-block carries the expected value.")
    blocked_reason: Optional[JobErrorCode] = Field(default=None, description="The definitive code, when blocked.")
    blocked_since: Optional[Timestamp] = None
    blocked_by_job_id: Optional[ResourceId] = Field(default=None, description="The failed job whose attempt blocked it, when a job did.")
    blocked_by_audit_event_id: Optional[ResourceId] = None
    cleared_at: Optional[Timestamp] = None
    cleared_by_account_id: Optional[ResourceId] = None
    last_verified_at: Optional[Timestamp] = None
    last_verification_outcome: Optional[Literal["passed", "failed"]] = None
    active_release_id: Optional[ShortText] = None
    active_manifest_digest: Optional[Sha256Digest] = None
    updated_at: Timestamp

    @model_validator(mode="after")
    def _blocked_fields(self):
        blocked = self.status is Pro1ConnectionStatus.BLOCKED
        if blocked != (self.blocked_reason is not None and self.blocked_since is not None):
            raise ValueError("a blocked connection names its reason and time, and only a blocked one does")
        if self.blocked_reason is not None and JOB_ERROR_CLASSES[self.blocked_reason] is not JobErrorClass.DEFINITIVE:
            raise ValueError("only a definitive Pro1 code blocks the connection")
        return self


class Pro1VerificationReport(ContractModel):
    """Process reports the outcome of an attested verification that is not part of a job attempt:
    the attested connection test after an approval, and the fresh verification after a block is
    cleared. It never carries customer data. Store applies ``PRO1_CONNECTION_TRANSITIONS``; a
    transient failure changes nothing."""

    outcome: Literal["passed", "failed"]
    error_code: Optional[JobErrorCode] = None
    policy_version: Sha256Digest
    evidence_digest: Optional[Sha256Digest] = None
    release_id: Optional[ShortText] = None
    manifest_digest: Optional[Sha256Digest] = None
    verified_at: Timestamp

    @model_validator(mode="after")
    def _outcome(self):
        if (self.outcome == "failed") != (self.error_code is not None):
            raise ValueError("a failed verification names its code, and only a failed one does")
        if self.error_code is not None and not self.error_code.value.startswith("pro1_"):
            raise ValueError("verification failures use Pro1 codes")
        if self.outcome == "passed" and not (self.evidence_digest and self.release_id and self.manifest_digest):
            raise ValueError("a passing verification names the evidence and the release it verified")
        return self


class Pro1BlockClear(ContractModel):
    """Audited admin acknowledgement that moves a blocked Pro1 connection to pending_verification; a
    fresh verification must then pass before Pro1 jobs are admitted again."""

    acknowledged_reason: SafeText = Field(min_length=1)
    expected_connection_version: int = Field(ge=0)


DestinationLabel = Literal["in-process", "loopback", "customer-lan", "pro1", "byok"]
