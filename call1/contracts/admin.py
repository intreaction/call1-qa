"""Store admin state and administration: route opt-ins and provider endpoints, masking,
key-manager configuration, attestation policy, trust anchors, TLS and hostname state,
backup/restore, status.

Every section of ``AdminState`` changes only through an audited admin action carrying the expected
state version; an update never writes any of it, and no Process scope can. A new default applies
only to a fresh install. Section payloads are settings only: who changed a section, when, and
under which audit event is kept in ``AdminState.section_changes``, written by Store. The Pro1
connection state is not admin state (``custody.Pro1Connection``).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .common import ChangeCursor, ContractInfo, ContractModel, EnvVarName, ModelCredentialEnvName, ResourceId, SafeText, Sha256Digest, ShortText, Timestamp, canonical_digest, validate_store_hostname
from .custody import LOOPBACK_HOSTS, PRO1_CONNECTION_REF, KeySourceKind, KeySourceRef, RouteClass, call1_operated_host, host_of
from .release_trust import AttestationPolicy, AttestationPolicySettings, EgressAllowlist, ReleaseKind, RunningBuild, TrustAnchorState


# --- Route opt-ins, provider endpoints, masking, key manager -------------------------------


def _https_base_url(value: str, what: str) -> str:
    value = value.strip()
    if not value.startswith("https://"):
        raise ValueError(f"a {what} is reached over HTTPS")
    host = host_of(value)
    if not host:
        raise ValueError(f"a {what} names a host")
    if host in LOOPBACK_HOSTS:
        raise ValueError(f"a {what} is not on loopback; loopback Ollama is the appliance route")
    if "@" in value.split("://", 1)[1].split("/", 1)[0] or "?" in value or "#" in value:
        raise ValueError(f"a {what} carries no credentials, query or fragment")
    return value.rstrip("/")


class CustomerLanHost(ContractModel):
    """A customer-operated Ollama host. Process connects only to ``base_url`` with the credential
    named here; its own configuration never supplies another endpoint."""

    connection_ref: ResourceId = Field(description="Referenced by RouteRecord.provider_connection_ref and ResourceEstimate.outbound_connection_ref.")
    base_url: str = Field(max_length=2048, description="Full non-secret endpoint, e.g. https://ollama.corp.example:8443/v1 (the OpenAI-compatible base). Resolves to a private address; certificate trusted by Process.")
    label: ShortText
    pinned_model_digests: Dict[str, str] = Field(default_factory=dict, description="Model ID to the digest the host reported at qualification.")
    credential_env_name: Optional[ModelCredentialEnvName] = None

    @field_validator("base_url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = _https_base_url(value, "customer-LAN host")
        if call1_operated_host(host_of(value)):
            raise ValueError("a customer-LAN host cannot be Call1-operated, a public IP literal or hostless")
        return value

    @property
    def host(self) -> str:
        return host_of(self.base_url)


class CustomerLanOptIn(ContractModel):
    enabled: bool = False
    hosts: List[CustomerLanHost] = Field(default_factory=list)


class Pro1OptIn(ContractModel):
    enabled: bool = False
    endpoint_base_url: Optional[str] = Field(default=None, max_length=2048, description="The Pro1 front end, e.g. https://pro1.call1.cc. Required to enable; its host joins the Call1-operated host set.")
    credential_env_name: ModelCredentialEnvName = Field(default="CALL1_MODEL_KEY_PRO1", description="The Pro1 account credential stays in Process under this name.")

    @field_validator("endpoint_base_url")
    @classmethod
    def _url(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _https_base_url(value, "Pro1 endpoint")

    @model_validator(mode="after")
    def _enabled_needs_endpoint(self):
        if self.enabled and not self.endpoint_base_url:
            raise ValueError("enabling Pro1 names its endpoint")
        return self

    @property
    def endpoint_host(self) -> Optional[str]:
        return host_of(self.endpoint_base_url) if self.endpoint_base_url else None


class ByokProvider(ContractModel):
    """A customer-directed provider. Process sends ``<base_url>/chat/completions`` there and
    nowhere else."""

    connection_ref: ResourceId
    label: ShortText
    base_url: str = Field(max_length=2048, description="Full non-secret endpoint, e.g. https://api.openai.com/v1. Its host is shown on every selection and job that uses it.")
    credential_env_name: ModelCredentialEnvName
    legacy_question_model_id: Optional[ShortText] = None

    @field_validator("base_url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = _https_base_url(value, "BYOK provider")
        if call1_operated_host(host_of(value)):
            raise ValueError("a BYOK provider cannot be Call1-operated; only the attested Pro1 route may name a Call1 host")
        return value

    @property
    def destination_host(self) -> str:
        return host_of(self.base_url)


class ByokOptIn(ContractModel):
    enabled: bool = False
    providers: List[ByokProvider] = Field(default_factory=list)


class RouteOptIns(ContractModel):
    """The appliance route needs no opt-in and is always available. Endpoint changes here are
    audited as ``endpoint_changed`` in addition to ``route_opt_in_changed``."""

    customer_lan: CustomerLanOptIn = Field(default_factory=CustomerLanOptIn)
    call1_confidential: Pro1OptIn = Field(default_factory=Pro1OptIn)
    customer_directed: ByokOptIn = Field(default_factory=ByokOptIn)

    @model_validator(mode="after")
    def _hosts(self):
        pro1_hosts = [self.call1_confidential.endpoint_host] if self.call1_confidential.endpoint_host else []
        for provider in self.customer_directed.providers:
            if call1_operated_host(provider.destination_host, pro1_hosts):
                raise ValueError("a BYOK provider cannot share the Pro1 endpoint's domain")
        for host in self.customer_lan.hosts:
            if call1_operated_host(host.host, pro1_hosts):
                raise ValueError("a customer-LAN host cannot share the Pro1 endpoint's domain")
        refs = [h.connection_ref for h in self.customer_lan.hosts] + [p.connection_ref for p in self.customer_directed.providers] + [PRO1_CONNECTION_REF]
        if len(refs) != len(set(refs)):
            raise ValueError("connection refs are unique across routes ('pro1' is reserved)")
        return self


class MaskingSettings(ContractModel):
    """Masking is forced for every non-appliance route at install. Turning it off for a route
    class is an audited admin action and never what permits the route."""

    masked_route_classes: List[RouteClass] = Field(default_factory=lambda: [RouteClass.CUSTOMER_LAN, RouteClass.CALL1_CONFIDENTIAL, RouteClass.CUSTOMER_DIRECTED])
    mask_reviewer_reads: bool = Field(default=True, description="Whether Evaluate reads (transcript, evidence, audio) are masked.")


class KeyManagerConfig(ContractModel):
    key_source: KeySourceRef = Field(default_factory=lambda: KeySourceRef(kind=KeySourceKind.LOCAL))
    anchor_credential_env_name: Optional[EnvVarName] = Field(default=None, description="Process environment variable naming the scoped anchor credential. Never a value.")

    @model_validator(mode="after")
    def _anchor_credential(self):
        if (self.key_source.kind is KeySourceKind.CUSTOMER_ANCHOR) != (self.anchor_credential_env_name is not None):
            raise ValueError("a customer anchor names its credential variable; the local source has none")
        return self


class SmtpRelayConfig(ContractModel):
    """The customer's own relay for invitation delivery. No Call1 mail service exists."""

    host: ShortText
    port: int = Field(default=587, ge=1, le=65535)
    security: Literal["starttls", "implicit_tls"] = Field(default="starttls", description="starttls on 587, implicit TLS (smtps) on 465. Never plaintext.")
    from_address: str = Field(pattern=r"^[^@\s]{1,64}@[^@\s]{1,190}$")
    credential_env_name: Optional[EnvVarName] = None


class AdminStateSection(str, Enum):
    ATTESTATION_POLICY = "attestation_policy"
    MINIMUM_RELEASE_SVN = "minimum_release_svn"
    TRUST_ANCHORS = "trust_anchors"
    KEY_MANAGER = "key_manager"
    ROUTE_OPT_INS = "route_opt_ins"
    MASKING = "masking"
    NOTIFICATION = "notification"


class SectionChange(ContractModel):
    """Written by Store for the last audited change of a section. Never part of a request."""

    changed_at: Timestamp
    changed_by_account_id: Optional[ResourceId] = Field(default=None, description="Null for installer defaults.")
    audit_event_id: ResourceId
    state_version: int = Field(ge=1)


class AdminState(ContractModel):
    state_version: int = Field(ge=0, description="Bumped by every audited change to any section and by approvals, withdrawals and declines; every such request carries the expected value.")
    attestation_policy: AttestationPolicy
    attestation_policy_digest: Sha256Digest = Field(description="canonical_digest(attestation_policy): the policy_version recorded on Pro1 attempts and key releases.")
    minimum_appliance_build_svn: int = Field(ge=0, description="The appliance build floor. The Pro1 floor is attestation_policy.minimum_release_svn.")
    trust_anchors: TrustAnchorState
    key_manager: KeyManagerConfig
    route_opt_ins: RouteOptIns
    masking: MaskingSettings
    notification: Optional[SmtpRelayConfig] = None
    section_changes: Dict[AdminStateSection, SectionChange] = Field(default_factory=dict)
    running_build: RunningBuild
    changed_only_by: Literal["audited_admin_action"] = "audited_admin_action"
    updated_at: Timestamp

    @model_validator(mode="after")
    def _policy_digest(self):
        if self.attestation_policy_digest != canonical_digest(self.attestation_policy):
            raise ValueError("attestation_policy_digest is canonical_digest(attestation_policy)")
        return self


class MinimumSvnChange(ContractModel):
    """The dedicated action for a release floor. Raising needs nothing more; lowering needs
    ``confirm_lower`` and is audited as such. A Pro1 change rewrites
    AttestationPolicy.minimum_release_svn, so the policy digest changes with it."""

    kind: ReleaseKind
    value: int = Field(ge=0)
    confirm_lower: bool = False


class TrustAnchorAdoption(ContractModel):
    """Adopt the pending bundle the admin reviewed (the UI shows the diff first). Store rejects it
    with 409 ``conflict`` unless ``pending_digest`` is the current pending bundle's digest."""

    pending_digest: Sha256Digest


class AdminStateChange(ContractModel):
    """One audited change to exactly one section. Payloads are settings only."""

    section: AdminStateSection
    expected_state_version: int = Field(ge=0)
    reason: SafeText
    attestation_policy: Optional[AttestationPolicySettings] = Field(default=None, description="Everything but the minimum release SVN, which has its own section.")
    minimum_release_svn: Optional[MinimumSvnChange] = None
    trust_anchor_adoption: Optional[TrustAnchorAdoption] = None
    key_manager: Optional[KeyManagerConfig] = None
    route_opt_ins: Optional[RouteOptIns] = None
    masking: Optional[MaskingSettings] = None
    notification: Optional[SmtpRelayConfig] = None

    @model_validator(mode="after")
    def _exactly_one_section(self):
        payloads = {
            AdminStateSection.ATTESTATION_POLICY: self.attestation_policy,
            AdminStateSection.MINIMUM_RELEASE_SVN: self.minimum_release_svn,
            AdminStateSection.TRUST_ANCHORS: self.trust_anchor_adoption,
            AdminStateSection.KEY_MANAGER: self.key_manager,
            AdminStateSection.ROUTE_OPT_INS: self.route_opt_ins,
            AdminStateSection.MASKING: self.masking,
            AdminStateSection.NOTIFICATION: self.notification,
        }
        present = [s for s, v in payloads.items() if v is not None]
        if present != [self.section]:
            raise ValueError("a change carries the payload of its section and nothing else")
        return self


class AdminStateChanged(ContractModel):
    state: AdminState
    audit_event_id: ResourceId


# --- TLS, hostname, relying party ---------------------------------------------------------


class TlsTrustPath(str, Enum):
    PATH_A_CALL1_CA = "path_a_call1_ca"
    """Installer-generated Call1 CA, name-constrained to the Store hostname; server cert issued from it."""
    PATH_B_CUSTOMER_CERT = "path_b_customer_cert"
    """Customer-issued certificate and key for the Store hostname, imported."""


class CertificateInfo(ContractModel):
    subject: ShortText
    issuer: ShortText
    not_before: Timestamp
    not_after: Timestamp
    fingerprint_sha256: Sha256Digest
    san_dns: List[ShortText] = Field(default_factory=list)


class TlsHealth(str, Enum):
    OK = "ok"
    EXPIRING = "expiring"
    EXPIRED = "expired"
    HOSTNAME_MISMATCH = "hostname_mismatch"
    UNTRUSTED = "untrusted"


class TlsState(ContractModel):
    store_hostname: str
    trust_path: TlsTrustPath
    server_certificate: CertificateInfo
    ca_certificate: Optional[CertificateInfo] = Field(default=None, description="Path A only.")
    ca_name_constraint: Optional[str] = Field(default=None, description="Path A: the permitted dNSName, which equals the Store hostname.")
    ca_key_separate_from_server_key: Literal[True] = True
    renewal_owner: ShortText
    health: TlsHealth
    days_until_expiry: int

    @field_validator("store_hostname")
    @classmethod
    def _hostname(cls, value: str) -> str:
        return validate_store_hostname(value)

    @model_validator(mode="after")
    def _paths(self):
        if self.trust_path is TlsTrustPath.PATH_A_CALL1_CA:
            if self.ca_certificate is None or self.ca_name_constraint != self.store_hostname:
                raise ValueError("Path A has a CA whose name constraint is exactly the Store hostname")
        elif self.ca_certificate is not None or self.ca_name_constraint is not None:
            raise ValueError("Path B has no Call1 CA")
        if self.store_hostname not in [s.lower() for s in self.server_certificate.san_dns]:
            raise ValueError("the server certificate names the Store hostname")
        return self


class WebAuthnRelyingParty(ContractModel):
    """The RP ID is the Store hostname and the allowed origins are that hostname only."""

    rp_id: str
    rp_name: ShortText = "Call1 Store"
    allowed_origins: List[str] = Field(min_length=1)

    @field_validator("rp_id")
    @classmethod
    def _hostname(cls, value: str) -> str:
        return validate_store_hostname(value)

    @model_validator(mode="after")
    def _origins(self):
        for origin in self.allowed_origins:
            scheme, _, rest = origin.partition("://")
            host, _, port = rest.partition(":")
            if scheme != "https" or host.lower() != self.rp_id or (port and not port.isdigit()) or "/" in rest:
                raise ValueError("every allowed origin is https://<store-hostname>[:port]")
        return self


class ObjectStoreKind(str, Enum):
    LOCAL_DIRECTORY = "local_directory"
    S3_COMPATIBLE = "s3_compatible"


class StoreHealth(ContractModel):
    """Anonymous health and compatibility view: only what a client needs before sign-in."""

    contract: ContractInfo
    store_hostname: str
    relying_party: WebAuthnRelyingParty
    tls_health: TlsHealth
    server_time: Timestamp


class SearchEmbedderState(str, Enum):
    """Added in 1.2.0. Whether Store can embed semantic-search queries."""

    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    """Weights are present; the model loads on the first search."""
    LOADED = "loaded"
    FAILED = "failed"
    FAKE = "fake"
    """The deterministic test embedder (fake-handler stacks); not a model."""


class SearchEmbedderStatus(ContractModel):
    """Added in 1.2.0 (decision 18). The one model Store runs: the local search embedder, which
    embeds queries into the same space as the turn vectors Process writes. Never a remote model."""

    scheme: ShortText = Field(description="The embeddings scheme Store ranks, e.g. 'nemotron-3-embed-1b@c0c9fea'.")
    model: ShortText
    revision: ShortText
    dimensions: int = Field(ge=1, le=4096)
    state: SearchEmbedderState
    detail: Optional[SafeText] = None


class StoreStatus(ContractModel):
    """Detailed status for admins and Process. Carries no location: clients reach Store only
    through the configured Store URL, and nothing here names where Store keeps its data."""

    contract: ContractInfo
    store_hostname: str
    relying_party: WebAuthnRelyingParty
    tls: TlsState
    running_build: RunningBuild
    schema_version: int = Field(ge=1)
    object_store_kind: ObjectStoreKind
    admin_state_version: int = Field(ge=0)
    feed_epoch: ResourceId = Field(description="Changes on every restore. A client holding cursors or receipts from another epoch re-snapshots and re-validates them.")
    latest_change_cursor: ChangeCursor
    server_time: Timestamp
    search_embedder: Optional[SearchEmbedderStatus] = Field(default=None, description="Added in 1.2.0. The local search embedder's state; semantic search answers 503 search_unavailable unless it is installed, loaded or fake.")


# --- Backup and restore --------------------------------------------------------------------


class BackupManifest(ContractModel):
    """The metadata database and object storage are one logical dataset. A backup includes
    WebAuthn credential records, audit events, usage records, service-key hashes and, on Path A,
    the CA key, so it is more sensitive than the data alone."""

    dataset_id: ResourceId
    feed_epoch: ResourceId = Field(description="The epoch the backup was taken in; a restore always starts a new one.")
    created_at: Timestamp
    store_hostname: str
    rp_id: str
    contract_version: str
    schema_version: int = Field(ge=1)
    database_digest: Sha256Digest
    object_manifest_digest: Sha256Digest = Field(description="Digest of the list of (artifact ID, checksum, size) the backup carries.")
    object_count: int = Field(ge=0)
    object_bytes: int = Field(ge=0)
    includes_ca_key: bool
    includes_credential_records: Literal[True] = True
    includes_audit_events: Literal[True] = True
    includes_usage_records: Literal[True] = True
    trust_path: TlsTrustPath


class RestorePreflightRequest(ContractModel):
    manifest: BackupManifest
    target_hostname: str

    @field_validator("target_hostname")
    @classmethod
    def _hostname(cls, value: str) -> str:
        return validate_store_hostname(value)


class RestorePreflightResult(ContractModel):
    """The rule: restore database and objects together and verify every object checksum against
    the database before serving. Same hostname keeps enrollments; a different hostname revokes every
    credential and requires re-inviting every reviewer, and setup says so before proceeding. Every
    restore starts a new feed epoch: old change cursors answer 410 ``cursor_unknown``, and leases
    and claim tokens from after the backup are gone, so Process re-checks its claims and spooled
    receipts before publishing."""

    hostname_matches: bool
    enrollments_preserved: bool
    requires_reinvite_all: bool
    sessions_invalidated: Literal[True] = True
    object_checksum_verification_required: Literal[True] = True
    new_feed_epoch: Literal[True] = True
    warnings: List[SafeText] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rule(self):
        if self.hostname_matches != self.enrollments_preserved or self.hostname_matches == self.requires_reinvite_all:
            raise ValueError("enrollments survive exactly when the hostname matches")
        return self


class EgressAllowlistView(EgressAllowlist):
    pass
