"""Release trust: approvals of appliance builds and Pro1 releases, transparency-log scan state,
the updater's verification contract, the attestation policy, trust anchors and the egress allowlist.

Team decision 7: every appliance update and every Pro1 release that could receive customer data
waits for explicit approval by the customer admin. A Call1 signature, a log entry or a waiting
period is never sufficient. Trust settings change only through audited admin actions and never
through an update.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from .common import ContractModel, HexDigest, JsonScalar, ResourceId, SafeText, Sha256Digest, ShortText, Timestamp
from .custody import Pro1Platform


class ReleaseKind(str, Enum):
    APPLIANCE_BUILD = "appliance_build"
    PRO1_RELEASE = "pro1_release"


class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    WITHDRAWN = "withdrawn"
    """The admin withdrew approval (an audited admin action)."""
    REVOKED = "revoked"
    """Call1 revoked the release in the log; applies at once and needs no approval."""


class ReleaseEvidence(ContractModel):
    """What the admin reviewed. Recorded verbatim on the approval."""

    manifest_digest: Sha256Digest
    log_id: ShortText
    log_index: int = Field(ge=0)
    checkpoint_digest: Sha256Digest
    witness_cosignatures: int = Field(ge=0)
    witness_quorum: int = Field(ge=1)
    rebuild_statements: int = Field(ge=0, description="Independent rebuilder statements found; zero is a warning, not a block.")
    source_commit: ShortText
    sbom_digest: Sha256Digest
    release_svn: int = Field(ge=0)
    changelog_digest: Optional[Sha256Digest] = None
    log_age_seconds: int = Field(ge=0, description="Age of the log entry when reviewed; a minimum age may delay approval but never approves.")


class ReleaseApprovalRequest(ContractModel):
    """Approve one pending release that Process's log scan recorded. The request names it; it
    carries no evidence of its own. Store copies the evidence from that PendingRelease and rejects
    the request with 409 ``conflict`` (details.reason ``not_pending``) unless (kind, release_id,
    manifest_digest) equals a pending, undeclined, unrevoked release."""

    kind: ReleaseKind
    release_id: ShortText
    manifest_digest: Sha256Digest
    raise_minimum_svn_to: Optional[int] = Field(default=None, ge=0, description="Approving a security release may raise the minimum release security version of its kind (AttestationPolicy.minimum_release_svn for Pro1, AdminState.minimum_appliance_build_svn for builds) in the same audited transaction. It never lowers it.")
    expected_state_version: int = Field(ge=0)
    reason: SafeText


class ReleaseApproval(ContractModel):
    id: ResourceId
    kind: ReleaseKind
    release_id: ShortText
    manifest_sha256: Sha256Digest = Field(description="The verifier and updater accept only an offered manifest whose digest equals this.")
    log_id: ShortText
    log_index: int = Field(ge=0)
    checkpoint_digest: Sha256Digest
    release_svn: int = Field(ge=0)
    evidence: ReleaseEvidence
    status: ApprovalStatus
    approved_by_account_id: ResourceId
    approved_at: Timestamp
    withdrawn_at: Optional[Timestamp] = None
    withdrawn_by_account_id: Optional[ResourceId] = None
    revoked_at: Optional[Timestamp] = None
    revocation_sequence: Optional[int] = Field(default=None, ge=0)
    audit_event_id: ResourceId


class ApprovalWithdrawRequest(ContractModel):
    expected_state_version: int = Field(ge=0)
    reason: SafeText


class PendingReleaseDecision(str, Enum):
    PENDING = "pending"
    DECLINED = "declined"


class PendingRelease(ContractModel):
    """A logged release Process's scan has seen that no admin has approved. Keyed by (kind,
    manifest_digest). Declining is sticky for that exact manifest; a new manifest under the same
    release ID is a different pending release."""

    kind: ReleaseKind
    release_id: ShortText
    manifest_digest: Sha256Digest
    log_index: int = Field(ge=0)
    release_svn: int = Field(ge=0)
    first_seen_at: Timestamp
    evidence: ReleaseEvidence
    revoked: bool = False
    decision: PendingReleaseDecision = PendingReleaseDecision.PENDING
    declined_at: Optional[Timestamp] = None
    declined_by_account_id: Optional[ResourceId] = None
    decline_reason: Optional[SafeText] = None

    @model_validator(mode="after")
    def _declined(self):
        if (self.decision is PendingReleaseDecision.DECLINED) != (self.declined_at is not None):
            raise ValueError("a declined release records when it was declined, and only it does")
        return self


class PendingReleaseQuery(ContractModel):
    kind: Optional[ReleaseKind] = None
    include_declined: bool = False


class ReleaseDeclineRequest(ContractModel):
    """Decline a pending release (audited). It leaves the pending list and banner; it never
    revokes anything and can be followed by an approval later if the admin changes their mind."""

    kind: ReleaseKind
    release_id: ShortText
    manifest_digest: Sha256Digest
    expected_state_version: int = Field(ge=0)
    reason: SafeText


class RevocationEntry(ContractModel):
    kind: ReleaseKind
    release_id: ShortText
    manifest_digest: Optional[Sha256Digest] = None
    revocation_sequence: int = Field(ge=0)
    issued_at: Timestamp


class LogScanStateInput(ContractModel):
    """Process reports the transparency-log client's state after each full scan
    (Pro1ConfidentialInference.md 2.2 check 6)."""

    log_id: ShortText
    last_verified_tree_size: int = Field(ge=0)
    last_checkpoint_digest: Sha256Digest
    last_checkpoint_cosigned_at: Timestamp = Field(description="Newest witness cosignature timestamp among the quorum; checkpoint age is measured from it.")
    cosignature_count: int = Field(ge=0)
    last_revocation_sequence: int = Field(ge=0)
    last_revocation_issued_at: Timestamp
    scanned_at: Timestamp
    equivocation_detected: bool = Field(default=False, description="Two manifests under one release ID, or a revocation sequence lower than one already seen.")
    equivocation_detail: Optional[SafeText] = None
    pending_releases: List[PendingRelease] = Field(default_factory=list)
    revocations: List[RevocationEntry] = Field(default_factory=list)


class LogScanState(LogScanStateInput):
    """Keyed by (installation_id, log_id): each Process installation scans for itself."""

    installation_id: ResourceId
    updated_at: Timestamp


class UpdaterVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


class UpdaterRejectReason(str, Enum):
    DIGEST_NOT_IN_LOG = "digest_not_in_log"
    NO_APPROVAL_RECORD = "no_approval_record"
    MANIFEST_DIGEST_MISMATCH = "manifest_digest_mismatch"
    INSTALL_SCRIPTS_PRESENT = "install_scripts_present"
    SIGNATURE_INVALID = "signature_invalid"
    WITNESS_QUORUM_MISSING = "witness_quorum_missing"
    LOG_UNAVAILABLE = "log_unavailable"
    RELEASE_REVOKED = "release_revoked"
    BELOW_MINIMUM_SVN = "below_minimum_svn"


class UpdaterChecks(ContractModel):
    signature_valid: bool
    digest_in_witnessed_log: bool
    approval_matches_digest: bool
    no_install_scripts: bool
    not_revoked: bool
    svn_at_or_above_minimum: bool


class UpdaterVerification(ContractModel):
    """The verdict of the installed build's updater, which runs on the Store host inside Store's
    own process, before any code from the package runs. Checks performed by the new package on
    itself never count. No route accepts this model as input: Store writes it when a staged
    package finishes uploading (``POST /admin/updates/packages/{id}/commit``). It is evidence only;
    installing needs a separate audited admin action, allowed only while the package's latest
    verification is ``accept`` and its approval is still ``approved``."""

    package_digest: Sha256Digest
    release_id: ShortText
    manifest_digest: Sha256Digest
    log_index: Optional[int] = Field(default=None, ge=0)
    approval_id: Optional[ResourceId] = None
    checks: UpdaterChecks
    verdict: UpdaterVerdict
    reject_reason: Optional[UpdaterRejectReason] = None
    verified_by_build_digest: Sha256Digest = Field(description="Manifest digest of the build whose verifier ran.")
    verified_at: Timestamp

    @model_validator(mode="after")
    def _verdict_consistent(self):
        all_ok = all(vars(self.checks).values())
        if self.verdict is UpdaterVerdict.ACCEPT and (not all_ok or self.approval_id is None or self.reject_reason):
            raise ValueError("accept requires every check to pass and an approval record")
        if self.verdict is UpdaterVerdict.REJECT and self.reject_reason is None:
            raise ValueError("a rejection names its reason")
        return self


class UpdaterVerificationRecord(UpdaterVerification):
    id: ResourceId
    package_id: ResourceId
    audit_event_id: ResourceId


class UpdatePackageStatus(str, Enum):
    AWAITING_UPLOAD = "awaiting_upload"
    VERIFIED = "verified"
    REJECTED = "rejected"
    INSTALLING = "installing"
    INSTALLED = "installed"


class UpdatePackageCreate(ContractModel):
    """The admin selects a downloaded package in Store's admin view. Store returns an upload grant
    for its bytes; the digest the admin typed is only a claim until the verifier checks it."""

    package_digest: Sha256Digest
    size_bytes: int = Field(ge=1)
    release_id: ShortText
    manifest_digest: Sha256Digest
    content_type: ShortText = "application/octet-stream"


class UpdatePackage(ContractModel):
    id: ResourceId
    package_digest: Sha256Digest
    release_id: ShortText
    manifest_digest: Sha256Digest
    status: UpdatePackageStatus
    upload_url: Optional[str] = Field(default=None, pattern=r"^https://", max_length=2048, description="Present while awaiting_upload: a short-lived, object-scoped PUT URL on the Store origin.")
    upload_expires_at: Optional[Timestamp] = None
    verification: Optional[UpdaterVerificationRecord] = None
    created_by_account_id: ResourceId
    created_at: Timestamp
    installed_at: Optional[Timestamp] = None


class UpdateInstallRequest(ContractModel):
    expected_state_version: int = Field(ge=0)
    reason: SafeText


class RunningBuild(ContractModel):
    """Recorded by Store at startup; a build no admin approved is flagged, not blocked
    (the check runs in the new build and catches an accidental or hand-copied install)."""

    manifest_digest: Sha256Digest
    release_id: Optional[ShortText] = None
    approval_id: Optional[ResourceId] = None
    approved: bool
    started_at: Timestamp


class WitnessRef(ContractModel):
    name: ShortText
    key_id: ShortText = Field(description="Fingerprint of the witness public key from the adopted trust-anchor bundle.")


class GpuVerificationMode(str, Enum):
    NRAS = "nras"
    LOCAL = "local"


class TdxPinnedValues(ContractModel):
    """Host-chosen TDX fields MRTD does not cover; each must equal its pin (or be all zero)."""

    mrconfigid: Optional[HexDigest] = Field(default=None, description="Null pins zero.")
    mrowner: Optional[HexDigest] = None
    mrownerconfig: Optional[HexDigest] = None


class PlatformIdentityPins(ContractModel):
    """Expected TEE-policy fields and platform identity (Pro1ConfidentialInference.md 2.2 checks 2-3)."""

    expected_id_key_digests: List[HexDigest] = Field(default_factory=list, description="SEV-SNP ID_KEY_DIGEST values accepted (on Azure, Microsoft's paravisor signing key).")
    expected_author_key_digests: List[HexDigest] = Field(default_factory=list)
    expected_vm_configuration: Dict[str, JsonScalar] = Field(default_factory=dict, description="Azure vm-configuration runtime claims that must match exactly.")
    tdx: TdxPinnedValues = Field(default_factory=TdxPinnedValues)
    cvm_firmware_pcrs: Dict[str, HexDigest] = Field(default_factory=dict, description="PCR index ('0'..'7') to the confidential VM firmware reference value, when pinned here rather than in the trust-anchor bundle.")
    cvm_firmware_pcrs_source: Literal["policy", "trust_anchor_bundle"] = "trust_anchor_bundle"

    @model_validator(mode="after")
    def _pcrs(self):
        if any(k not in {str(i) for i in range(8)} for k in self.cvm_firmware_pcrs):
            raise ValueError("firmware PCR pins are PCRs 0-7")
        if (self.cvm_firmware_pcrs_source == "policy") != bool(self.cvm_firmware_pcrs):
            raise ValueError("pin firmware PCRs here exactly when the policy is their source")
        return self


class AttestationPolicySettings(ContractModel):
    """The customer's attestation policy, without its minimum release SVN. This is the payload of
    the ``attestation_policy`` admin-state section; the minimum SVN changes only through the
    ``minimum_release_svn`` section or an approval that raises it."""

    max_revocation_list_age_seconds: int = Field(default=86400, ge=3600)
    witnesses: List[WitnessRef] = Field(min_length=1)
    witness_quorum: int = Field(default=2, ge=1)
    max_checkpoint_age_seconds: int = Field(default=86400, ge=3600)
    max_collateral_age_seconds: int = Field(default=86400, ge=3600)
    minimum_tcb: Dict[str, int] = Field(default_factory=dict, description="Component name (boot_loader, tee, snp, microcode) to minimum version.")
    allowed_platforms: List[Pro1Platform] = Field(min_length=1)
    allowed_product_lines: List[ShortText] = Field(default_factory=lambda: ["genoa"], description="Milan is rejected outright (5.3).")
    expected_vmpl: int = Field(default=0, ge=0)
    smt_allowed: bool = True
    require_confidential_vm_evidence: bool = Field(default=True, description="Reject AKs that cannot be shown to belong to a confidential VM once a CVM-specific property is confirmed (open decision).")
    gpu_verification_mode: GpuVerificationMode = GpuVerificationMode.NRAS
    allowed_gpu_models: List[ShortText] = Field(default_factory=lambda: ["H100"])
    platform_identity: PlatformIdentityPins = Field(default_factory=PlatformIdentityPins)
    session_lifetime_seconds: int = Field(default=600, ge=60, description="Attested session lifetime (Pro1ConfidentialInference.md 2.2 step 4). The only place this value lives.")
    minimum_log_age_seconds_before_approval: int = Field(default=0, ge=0, description="Delays the approve action; never approves anything.")

    @model_validator(mode="after")
    def _quorum(self):
        if self.witness_quorum > len(self.witnesses):
            raise ValueError("witness quorum cannot exceed the number of witnesses")
        return self


class AttestationPolicy(AttestationPolicySettings):
    """The full policy (Pro1ConfidentialInference.md 3.1). ``canonical_digest`` of it is
    ``AdminState.attestation_policy_digest``, the ``policy_version`` recorded on every Pro1 attempt,
    so every value that admits an enclave, including the minimum release SVN and the platform pins,
    is covered. Approved releases are the ``ReleaseApproval`` records; trust anchors are
    ``TrustAnchorState``; neither is duplicated here."""

    minimum_release_svn: int = Field(ge=0, description="The Pro1 release floor. The only copy.")


class TrustAnchorBundleRef(ContractModel):
    artifact_id: ResourceId = Field(description="The trust_anchor_bundle artifact; Process fetches it by ID and checksum and refuses any other bytes.")
    digest: Sha256Digest = Field(description="The bundle's checksum. Covers AMD and Intel roots, NVIDIA CA, Azure vTPM root, CVM reference values, Call1 release keys, the log checkpoint key and URL, and witness keys.")
    description: ShortText
    shipped_with_build_digest: Optional[Sha256Digest] = None
    received_at: Timestamp


class TrustAnchorBundleSubmit(ContractModel):
    """Process registers the bundle its build shipped (after uploading it as a trust_anchor_bundle
    artifact). Store sets it as ``pending`` (audited ``trust_anchor_pending``) unless it equals the
    adopted bundle or the current pending one; it never adopts it."""

    artifact_id: ResourceId
    digest: Sha256Digest
    description: ShortText
    shipped_with_build_digest: Sha256Digest


class TrustAnchorState(ContractModel):
    adopted: TrustAnchorBundleRef
    adopted_at: Timestamp
    adopted_by_account_id: Optional[ResourceId] = Field(default=None, description="Null for the bundle the installer seeded at first install.")
    pending: Optional[TrustAnchorBundleRef] = Field(default=None, description="A bundle shipped with an update waits here, inert, until an admin adopts it. Until then the verifier keeps using the adopted bundle, including its log key and URL.")


class EgressPurpose(str, Enum):
    PRO1_FRONTEND = "pro1_frontend"
    AMD_KDS = "amd_kds"
    INTEL_PCS = "intel_pcs"
    NVIDIA_OCSP = "nvidia_ocsp"
    NVIDIA_RIM = "nvidia_rim"
    NVIDIA_NRAS = "nvidia_nras"
    TRANSPARENCY_LOG = "transparency_log"
    WITNESS = "witness"
    KEY_ANCHOR = "key_anchor"
    BYOK_PROVIDER = "byok_provider"
    CUSTOMER_LAN_MODEL_HOST = "customer_lan_model_host"
    UPDATE_HOST = "update_host"
    MODEL_PROVISIONING = "model_provisioning"
    CUSTOMER_SMTP_RELAY = "customer_smtp_relay"


CUSTOMER_OPERATED_DATA_PURPOSES = frozenset({EgressPurpose.BYOK_PROVIDER, EgressPurpose.CUSTOMER_LAN_MODEL_HOST, EgressPurpose.CUSTOMER_SMTP_RELAY})
"""Destinations that receive customer data in plaintext and are the customer's own or contracted:
model hosts receive text, the SMTP relay receives reviewer emails and invitation links."""


class EgressEntry(ContractModel):
    destination: ShortText = Field(description="Hostname or documented hostname pattern.")
    port: int = Field(default=443, ge=1, le=65535)
    protocol: Literal["https", "smtps", "smtp+starttls"] = "https"
    purpose: EgressPurpose
    required_when: ShortText = Field(description="e.g. 'Pro1 enabled', 'updater verifying a package', 'installing models'.")
    carries_customer_data: bool = Field(description="True for ciphertext to the Pro1 front end and for the customer-operated or customer-contracted destinations in CUSTOMER_OPERATED_DATA_PURPOSES.")
    ciphertext_only: bool = Field(default=False, description="True for the Pro1 front end: sealed bodies and wrapped keys only.")

    @model_validator(mode="after")
    def _custody(self):
        if self.purpose is EgressPurpose.PRO1_FRONTEND and not (self.carries_customer_data and self.ciphertext_only):
            raise ValueError("the Pro1 front end receives customer data as ciphertext only")
        if self.purpose in CUSTOMER_OPERATED_DATA_PURPOSES and not self.carries_customer_data:
            raise ValueError("model hosts and the customer's SMTP relay receive customer data; say so")
        if self.purpose not in CUSTOMER_OPERATED_DATA_PURPOSES | {EgressPurpose.PRO1_FRONTEND} and self.carries_customer_data:
            raise ValueError("control-plane destinations never carry customer data")
        if (self.protocol != "https") != (self.purpose is EgressPurpose.CUSTOMER_SMTP_RELAY):
            raise ValueError("only the SMTP relay uses an SMTP protocol")
        return self


class EgressAllowlist(ContractModel):
    """Produced by the installer for customer IT to enforce at the firewall. With all egress blocked,
    the single-computer layout still processes and reviews calls."""

    generated_at: Timestamp
    installer_build_digest: Sha256Digest
    entries: List[EgressEntry]
