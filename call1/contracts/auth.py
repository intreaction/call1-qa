"""Authentication and identity: reviewer accounts (passkey-only), invitations, setup codes,
break-glass, server-side sessions, roles and permissions, Process service keys, and the
loopback console credential.

Store keeps no password verifier and offers no password sign-in, reset or fallback. It stores
only hashes of invitation tokens, setup codes, service keys and session cookies. The few fields
that carry a secret exactly once (an issued key, an invitation link, a code being redeemed) are
listed in ``ONE_TIME_SECRET_FIELDS``; the session's CSRF token is session-bound and readable again
from ``GET /auth/session`` (``SESSION_BOUND_SECRET_FIELDS``). Neither kind appears on a stored
record.

Authorization is evaluated on every request from the account's current role and status, never
from a snapshot taken at sign-in. Disabling an account, re-inviting it and restoring under a new
hostname revoke all of its sessions in the same transaction; a demotion takes effect on the next
request.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, FrozenSet, List, Literal, Optional

from pydantic import Field, StringConstraints, model_validator

from .common import ContractModel, EnvVarName, PageQuery, ResourceId, ReviewerRole, SafeText, ServiceScope, Sha256Digest, ShortText, Timestamp

Email = str
_EMAIL_PATTERN = r"^[^@\s]{1,64}@[^@\s]{1,190}$"
Base64Url = str
_B64URL = r"^[A-Za-z0-9_-]+$"


# --- Permissions --------------------------------------------------------------------------


class Permission(str, Enum):
    READ_CALLS = "read_calls"
    PLAY_AUDIO = "play_audio"
    SEARCH = "search"
    READ_METRICS = "read_metrics"
    CLAIM_REVIEW = "claim_review"
    RESOLVE_OWN_REVIEW = "resolve_own_review"
    OVERRIDE_VERDICT = "override_verdict"
    REQUEST_REANALYSIS = "request_reanalysis"
    MANAGE_OWN_AUTHENTICATORS = "manage_own_authenticators"
    ASSIGN_REVIEW = "assign_review"
    RESOLVE_ANY_REVIEW = "resolve_any_review"
    RESOLVE_ESCALATION = "resolve_escalation"
    RETAIN_REVIEW = "retain_review"
    MANAGE_RUBRICS = "manage_rubrics"
    MANAGE_QUEUE_RULES = "manage_queue_rules"
    READ_JOBS = "read_jobs"
    MANAGE_ACCOUNTS = "manage_accounts"
    MANAGE_INVITATIONS = "manage_invitations"
    MANAGE_SESSIONS = "manage_sessions"
    MANAGE_SERVICE_KEYS = "manage_service_keys"
    MANAGE_ADMIN_STATE = "manage_admin_state"
    APPROVE_RELEASES = "approve_releases"
    READ_USAGE_REPORT = "read_usage_report"
    READ_AUDIT = "read_audit"
    MANAGE_BACKUP = "manage_backup"
    MANAGE_SIGNALS = "manage_signals"
    """Added in 1.3.0; admin only (decision 22, Q1). The signal taxonomy, signal settings, alert
    rules, previews, backfills and taxonomy-text redaction. Reading the taxonomy and alert rules
    needs only read_calls; hit feedback needs override_verdict."""
    MANAGE_VOCABULARY = "manage_vocabulary"
    """Added in 1.3.0 (decision 33); admin only. Read and edit the ASR vocabulary (customer terms,
    disabled pack terms, on/off). Reviewers see corrections on transcripts without it."""


_REVIEWER = frozenset({
    Permission.READ_CALLS, Permission.PLAY_AUDIO, Permission.SEARCH, Permission.READ_METRICS,
    Permission.CLAIM_REVIEW, Permission.RESOLVE_OWN_REVIEW, Permission.OVERRIDE_VERDICT,
    Permission.REQUEST_REANALYSIS, Permission.MANAGE_OWN_AUTHENTICATORS,
})
_SUPERVISOR = _REVIEWER | frozenset({
    Permission.ASSIGN_REVIEW, Permission.RESOLVE_ANY_REVIEW, Permission.RESOLVE_ESCALATION,
    Permission.RETAIN_REVIEW, Permission.MANAGE_RUBRICS, Permission.MANAGE_QUEUE_RULES, Permission.READ_JOBS,
})
_ADMIN = _SUPERVISOR | frozenset({
    Permission.MANAGE_ACCOUNTS, Permission.MANAGE_INVITATIONS, Permission.MANAGE_SESSIONS,
    Permission.MANAGE_SERVICE_KEYS, Permission.MANAGE_ADMIN_STATE, Permission.APPROVE_RELEASES,
    Permission.READ_USAGE_REPORT, Permission.READ_AUDIT, Permission.MANAGE_BACKUP,
    Permission.MANAGE_SIGNALS, Permission.MANAGE_VOCABULARY,
})

ROLE_PERMISSIONS: Dict[ReviewerRole, FrozenSet[Permission]] = {
    ReviewerRole.REVIEWER: _REVIEWER,
    ReviewerRole.SUPERVISOR: _SUPERVISOR,
    ReviewerRole.ADMIN: _ADMIN,
}


# --- Accounts and credentials -------------------------------------------------------------


class AccountStatus(str, Enum):
    PENDING_ENROLLMENT = "pending_enrollment"
    ACTIVE = "active"
    DISABLED = "disabled"
    REINVITE_REQUIRED = "reinvite_required"
    """Every credential was revoked (re-invite or restore under a new hostname)."""


class ReviewerAccount(ContractModel):
    id: ResourceId
    email: str = Field(pattern=_EMAIL_PATTERN, description="Account identifier, not a secret.")
    display_name: ShortText
    role: ReviewerRole
    status: AccountStatus
    created_at: Timestamp
    last_sign_in_at: Optional[Timestamp] = None
    authenticator_count: int = Field(ge=0)
    legacy_auditor_id: Optional[ShortText] = Field(default=None, description="The pre-split auditors.id this account continues, when migrated.")


class AccountUpdate(ContractModel):
    display_name: Optional[ShortText] = None
    role: Optional[ReviewerRole] = None
    status: Optional[Literal["active", "disabled"]] = None
    reason: SafeText


class AccountListQuery(PageQuery):
    role: Optional[ReviewerRole] = None
    status: Optional[AccountStatus] = None


class AuthenticatorTransport(str, Enum):
    """Transports Store records. Browsers may report others (the list is open-ended in WebAuthn L3);
    Store drops values it does not know when it stores a credential."""

    USB = "usb"
    NFC = "nfc"
    BLE = "ble"
    INTERNAL = "internal"
    HYBRID = "hybrid"
    CABLE = "cable"
    SMART_CARD = "smart-card"


BrowserTransport = Annotated[str, StringConstraints(min_length=1, max_length=32)]
"""A transport string as a browser reports it (``@simplewebauthn/browser`` passes ``string[]``)."""


class WebAuthnCredentialRecord(ContractModel):
    """A registered authenticator. The public key is not secret; no private material exists on Store.
    ``id`` (the ``authenticator_id`` in paths) is Store's handle; ``credential_id`` is the WebAuthn
    credential ID and never appears in a path."""

    id: ResourceId
    account_id: ResourceId
    credential_id: str = Field(pattern=_B64URL, description="WebAuthn credential ID, base64url.")
    public_key_cose: str = Field(pattern=_B64URL, description="COSE public key, base64url.")
    sign_count: int = Field(ge=0, description="Last signature counter seen; a lower value on assertion is rejected where the authenticator provides one.")
    transports: List[AuthenticatorTransport] = Field(default_factory=list)
    aaguid: Optional[str] = Field(default=None, pattern=r"^[0-9a-f-]{36}$")
    nickname: Optional[ShortText] = None
    backup_eligible: Optional[bool] = None
    backup_state: Optional[bool] = None
    attestation_format: Literal["none"] = "none"
    created_at: Timestamp
    last_used_at: Optional[Timestamp] = None


class CredentialUpdate(ContractModel):
    nickname: ShortText


# --- Invitations, setup codes, break-glass -------------------------------------------------


class InvitationDelivery(str, Enum):
    OUT_OF_BAND = "out_of_band"
    """The link is shown once to the issuing admin, who delivers it themselves."""
    SMTP_RELAY = "smtp_relay"
    """Sent through the customer's own SMTP relay (admin state: notification). Never a Call1 service."""


class InvitationStatus(str, Enum):
    PENDING = "pending"
    REDEEMED = "redeemed"
    REVOKED = "revoked"
    EXPIRED = "expired"


class InvitationListQuery(PageQuery):
    status: Optional[InvitationStatus] = None


class InvitationCreate(ContractModel):
    email: str = Field(pattern=_EMAIL_PATTERN)
    display_name: ShortText
    role: ReviewerRole
    delivery: InvitationDelivery = InvitationDelivery.OUT_OF_BAND
    reinvite_of_account_id: Optional[ResourceId] = Field(default=None, description="Recovery: revokes every credential and session of that account when the invitation is issued.")


class Invitation(ContractModel):
    id: ResourceId
    email: str = Field(pattern=_EMAIL_PATTERN)
    display_name: ShortText
    role: ReviewerRole
    status: InvitationStatus
    delivery: InvitationDelivery
    token_hash: Sha256Digest = Field(description="SHA-256 of the single-use token. The token is never stored.")
    issued_by_account_id: Optional[ResourceId] = Field(default=None, description="Null when issued by the installer or break-glass.")
    issued_at: Timestamp
    expires_at: Timestamp
    redeemed_at: Optional[Timestamp] = None
    account_id: Optional[ResourceId] = Field(default=None, description="The account created or re-enrolled by this invitation.")
    reinvite_of_account_id: Optional[ResourceId] = None


class InvitationIssued(ContractModel):
    invitation: Invitation
    invitation_url: Optional[str] = Field(default=None, max_length=2048, description="ONE-TIME: https://<store-hostname>/enroll#<token>. Present only for out_of_band delivery and only in this response.")


class SetupCodePurpose(str, Enum):
    FIRST_ADMIN = "first_admin"
    BREAK_GLASS = "break_glass"


class SetupCodeIssueRequest(ContractModel):
    """What the host command (run as the Store OS user on the Store host; never an HTTP route)
    takes to issue a first-admin or break-glass code. The code is printed once on the host.

    * ``first_admin`` creates a new admin account with this email and display name; allowed only
      while no active admin exists.
    * ``break_glass`` without ``target_account_id`` creates a new admin account; with it,
      re-enrolls that account: at redemption Store revokes the account's credentials and sessions,
      sets its role to admin and enrolls the new authenticator. The email must match the target's.
      Both write ``break_glass_used``; no other account is touched.
    """

    purpose: SetupCodePurpose
    email: str = Field(pattern=_EMAIL_PATTERN)
    display_name: ShortText
    target_account_id: Optional[ResourceId] = None
    os_user: ShortText = Field(description="The OS user that ran the command, as the host reports it.")

    @model_validator(mode="after")
    def _target(self):
        if self.target_account_id is not None and self.purpose is not SetupCodePurpose.BREAK_GLASS:
            raise ValueError("only break-glass re-enrolls an existing account")
        return self


class SetupCodeRecord(ContractModel):
    """Generated on the Store host by a command run as the Store OS user; never by an HTTP route.
    Carries the identity the code enrolls, so enrollment needs nothing but the code."""

    id: ResourceId
    purpose: SetupCodePurpose
    email: str = Field(pattern=_EMAIL_PATTERN)
    display_name: ShortText
    role: Literal["admin"] = "admin"
    target_account_id: Optional[ResourceId] = Field(default=None, description="break_glass only: the account re-enrolled; null creates a new admin account.")
    code_hash: Sha256Digest
    issued_at: Timestamp
    expires_at: Timestamp
    used_at: Optional[Timestamp] = None
    enrolled_account_id: Optional[ResourceId] = None
    audit_event_id: ResourceId


class BreakGlassRecord(ContractModel):
    """Local break-glass: OS-level access on the Store host issues a one-time admin enrollment."""

    setup_code_id: ResourceId
    os_user: ShortText
    target_account_id: Optional[ResourceId] = None
    issued_at: Timestamp
    used_at: Optional[Timestamp] = None
    enrolled_account_id: Optional[ResourceId] = None
    revoked_credential_count: int = Field(default=0, ge=0, description="Credentials of the target account revoked at redemption.")
    audit_event_id: ResourceId


# --- WebAuthn ceremonies -----------------------------------------------------------------


class RelyingParty(ContractModel):
    id: str = Field(description="Equals the fixed Store hostname.")
    name: ShortText


class UserEntity(ContractModel):
    id: str = Field(pattern=_B64URL, description="Opaque user handle, base64url; never the email.")
    name: str = Field(pattern=_EMAIL_PATTERN)
    displayName: ShortText


class PubKeyCredParam(ContractModel):
    type: Literal["public-key"] = "public-key"
    alg: int


class CredentialDescriptor(ContractModel):
    type: Literal["public-key"] = "public-key"
    id: str = Field(pattern=_B64URL)
    transports: List[AuthenticatorTransport] = Field(default_factory=list)


class AuthenticatorSelection(ContractModel):
    authenticatorAttachment: Optional[Literal["platform", "cross-platform"]] = Field(default=None, description="Null: both platform passkeys and roaming FIDO2 keys.")
    residentKey: Literal["preferred", "required", "discouraged"] = "preferred"
    requireResidentKey: bool = False
    userVerification: Literal["required"] = "required"


class CredentialCreationOptions(ContractModel):
    """PublicKeyCredentialCreationOptions, JSON-encoded for @simplewebauthn/browser."""

    rp: RelyingParty
    user: UserEntity
    challenge: str = Field(pattern=_B64URL, description="Single-use; expires after CHALLENGE_LIFETIME.")
    pubKeyCredParams: List[PubKeyCredParam] = Field(default_factory=lambda: [PubKeyCredParam(alg=-7), PubKeyCredParam(alg=-257)])
    timeout: int = Field(default=120000, ge=30000)
    excludeCredentials: List[CredentialDescriptor] = Field(default_factory=list)
    authenticatorSelection: AuthenticatorSelection = Field(default_factory=AuthenticatorSelection)
    attestation: Literal["none"] = "none"
    extensions: Dict[str, Any] = Field(default_factory=dict)


class CredentialRequestOptions(ContractModel):
    """PublicKeyCredentialRequestOptions, JSON-encoded. Sign-in is account-first so
    non-discoverable credentials on security keys work.

    Anti-enumeration: for an unknown, disabled or re-invite-pending email Store returns the same
    shape and timing as for a real account, with deterministic decoy descriptors (IDs derived as
    HMAC(server secret, email), one or two of them, plausible transports), so the response never
    reveals whether the account exists. Sign-in finish accepts only an assertion from a credential
    registered to the account bound to the ceremony; any other credential, including a
    discoverable one of a different account, is ``webauthn_verification_failed``."""

    challenge: str = Field(pattern=_B64URL)
    rpId: str
    allowCredentials: List[CredentialDescriptor] = Field(min_length=1, description="The account's credentials, or decoys (never empty).")
    userVerification: Literal["required"] = "required"
    timeout: int = Field(default=120000, ge=30000)
    extensions: Dict[str, Any] = Field(default_factory=dict)


class EnrollmentBeginRequest(ContractModel):
    """Anonymous: start enrollment with exactly one single-use secret."""

    invitation_token: Optional[str] = Field(default=None, min_length=16, max_length=256, description="ONE-TIME: the token from the invitation link.")
    setup_code: Optional[str] = Field(default=None, min_length=8, max_length=128, description="ONE-TIME: the first-admin or break-glass code from the Store host.")

    @model_validator(mode="after")
    def _exactly_one(self):
        if bool(self.invitation_token) == bool(self.setup_code):
            raise ValueError("exactly one of invitation_token or setup_code")
        return self


class AddAuthenticatorBeginRequest(ContractModel):
    """Signed-in reviewer adds another authenticator. It is a step-up ceremony: the response
    carries an assertion challenge for the account's existing credentials as well as the
    registration options, and finish needs both (plan: 'after a fresh user-verified ceremony')."""

    nickname: Optional[ShortText] = None


class RegistrationBeginResponse(ContractModel):
    ceremony_id: ResourceId
    options: CredentialCreationOptions
    expires_at: Timestamp


class AddAuthenticatorBeginResponse(ContractModel):
    ceremony_id: ResourceId
    reauthentication: CredentialRequestOptions = Field(description="Run first (startAuthentication): allowCredentials are the account's existing credentials; userVerification required.")
    options: CredentialCreationOptions = Field(description="Run second (startRegistration); excludeCredentials lists the existing credentials.")
    expires_at: Timestamp = Field(description="Both challenges expire together after CHALLENGE_LIFETIME.")


class AuthenticatorAttestationResponseJSON(ContractModel):
    """As ``@simplewebauthn/browser`` ``startRegistration`` returns it (v13 and v14)."""

    clientDataJSON: str = Field(pattern=_B64URL)
    attestationObject: str = Field(pattern=_B64URL)
    authenticatorData: Optional[str] = Field(default=None, pattern=_B64URL)
    transports: List[BrowserTransport] = Field(default_factory=list, max_length=16)
    publicKeyAlgorithm: Optional[int] = None
    publicKey: Optional[str] = Field(default=None, pattern=_B64URL)


class RegistrationCredentialJSON(ContractModel):
    id: str = Field(pattern=_B64URL)
    rawId: str = Field(pattern=_B64URL)
    type: Literal["public-key"] = "public-key"
    response: AuthenticatorAttestationResponseJSON
    clientExtensionResults: Dict[str, Any] = Field(default_factory=dict)
    authenticatorAttachment: Optional[Literal["platform", "cross-platform"]] = None


class RegistrationFinishRequest(ContractModel):
    ceremony_id: ResourceId
    credential: RegistrationCredentialJSON
    nickname: Optional[ShortText] = None


class SessionInfo(ContractModel):
    """The signed-in reviewer as of this request. ``role`` and ``permissions`` are the account's
    current ones (re-read on every request), so a client re-reads this after a 403."""

    session_id: ResourceId = Field(description="A non-secret handle for listing and revoking sessions. Never the cookie value.")
    account_id: ResourceId
    email: str = Field(pattern=_EMAIL_PATTERN)
    display_name: ShortText
    role: ReviewerRole
    permissions: List[Permission]
    created_at: Timestamp
    last_seen_at: Timestamp
    idle_expires_at: Timestamp
    absolute_expires_at: Timestamp
    authenticator_id_used: ResourceId = Field(description="WebAuthnCredentialRecord.id of the authenticator that signed in.")
    prompt_second_authenticator: bool = Field(description="True while the account has one authenticator; setup prompts every reviewer to enroll a second.")
    csrf_token: str = Field(pattern=_B64URL, min_length=22, description="Session-bound, not one-time: send as X-Call1-CSRF on every state-changing request. Readable again from GET /auth/session (same origin; SameSite=Strict and CORS keep other sites from reading it), so a reload or a new tab recovers it. Keep it in memory, never in web storage.")


class SignInResponse(ContractModel):
    """Also sets the session cookie (SESSION_COOKIE)."""

    session: SessionInfo


class RegistrationFinishResponse(ContractModel):
    account: ReviewerAccount
    credential: WebAuthnCredentialRecord
    signed_in: Optional[SignInResponse] = Field(default=None, description="Present for enrollment (the new reviewer is signed in); absent for add-authenticator.")


class AuthenticationBeginRequest(ContractModel):
    email: str = Field(pattern=_EMAIL_PATTERN)


class AuthenticationBeginResponse(ContractModel):
    ceremony_id: ResourceId
    options: CredentialRequestOptions
    expires_at: Timestamp


class AuthenticatorAssertionResponseJSON(ContractModel):
    clientDataJSON: str = Field(pattern=_B64URL)
    authenticatorData: str = Field(pattern=_B64URL)
    signature: str = Field(pattern=_B64URL)
    userHandle: Optional[str] = Field(default=None, pattern=_B64URL)


class AuthenticationCredentialJSON(ContractModel):
    id: str = Field(pattern=_B64URL)
    rawId: str = Field(pattern=_B64URL)
    type: Literal["public-key"] = "public-key"
    response: AuthenticatorAssertionResponseJSON
    clientExtensionResults: Dict[str, Any] = Field(default_factory=dict)
    authenticatorAttachment: Optional[Literal["platform", "cross-platform"]] = None


class AuthenticationFinishRequest(ContractModel):
    ceremony_id: ResourceId
    credential: AuthenticationCredentialJSON


class AddAuthenticatorFinishRequest(ContractModel):
    """Store verifies ``reauthentication`` first: an assertion over the step-up challenge, with the
    user-verified flag, from a credential already registered to the signed-in account, within
    CHALLENGE_LIFETIME of begin. Only then does it verify and store ``credential``."""

    ceremony_id: ResourceId
    reauthentication: AuthenticationCredentialJSON
    credential: RegistrationCredentialJSON
    nickname: Optional[ShortText] = None


class SessionCookieSpec(ContractModel):
    """The cookie carries a random session secret that Store keeps only as a hash; ``session_id``
    is a separate, non-secret handle. The ``__Host-`` prefix pins it to the Store hostname (Secure,
    Path=/, no Domain), so a sibling host in the customer's zone cannot set or shadow it."""

    name: Literal["__Host-call1_session"] = "__Host-call1_session"
    secure: Literal[True] = True
    http_only: Literal[True] = True
    same_site: Literal["Strict"] = "Strict"
    path: Literal["/"] = "/"
    domain: None = None
    csrf_header: Literal["X-Call1-CSRF"] = "X-Call1-CSRF"
    stored_as: Literal["sha256"] = "sha256"


SESSION_COOKIE = SessionCookieSpec()


class SessionListItem(ContractModel):
    session_id: ResourceId
    account_id: ResourceId
    created_at: Timestamp
    last_seen_at: Timestamp
    idle_expires_at: Timestamp
    absolute_expires_at: Timestamp
    current: bool


class SignOutResponse(ContractModel):
    signed_out: Literal[True] = True


# --- Process installations and service keys ---------------------------------------------


class InstallationCreate(ContractModel):
    label: ShortText
    primary_host: bool = Field(default=False, description="The designated primary Process host (runs ML stages). At most one active installation holds it.")


class InstallationRetire(ContractModel):
    """Retiring an installation revokes every key it holds at once; its active leases expire normally."""

    reason: SafeText


class ProcessInstallation(ContractModel):
    """A registered Process installation. Every Process request acts as the installation its key
    belongs to: Store derives ``installation_id`` from the key, and any path or body field naming a
    different installation is 403 ``forbidden``."""

    id: ResourceId
    label: ShortText
    primary_host: bool
    created_at: Timestamp
    created_by_account_id: Optional[ResourceId] = Field(default=None, description="Null when the installer registered it on the Store host.")
    retired_at: Optional[Timestamp] = None


class ServiceKeyRecord(ContractModel):
    """Store keeps the hash only. Claims and completions are tied to ``installation_id``."""

    id: ResourceId
    installation_id: ResourceId = Field(description="A registered ProcessInstallation.")
    label: ShortText
    scopes: List[ServiceScope] = Field(min_length=1)
    key_prefix: str = Field(pattern=r"^c1sk_[A-Za-z0-9]{6}$", description="Non-secret prefix for identifying a key in logs and UI.")
    key_hash: Sha256Digest
    hash_algorithm: Literal["sha256"] = "sha256"
    created_at: Timestamp
    created_by_account_id: Optional[ResourceId] = Field(default=None, description="Null when issued by the installer on the Store host.")
    expires_at: Optional[Timestamp] = None
    revoked_at: Optional[Timestamp] = None
    rotated_from_key_id: Optional[ResourceId] = Field(default=None, description="On a replacement key: the key it replaced.")
    superseded_by_key_id: Optional[ResourceId] = Field(default=None, description="On a rotated-out key: its replacement.")
    grace_until: Optional[Timestamp] = Field(default=None, description="On a rotated-out key: it keeps working until this time (SERVICE_KEY_ROTATION_GRACE), then it is rejected.")
    last_used_at: Optional[Timestamp] = None

    @model_validator(mode="after")
    def _rotation(self):
        if (self.superseded_by_key_id is None) != (self.grace_until is None):
            raise ValueError("a rotated-out key names its replacement and its grace end, and only it does")
        if self.rotated_from_key_id is not None and self.rotated_from_key_id == self.superseded_by_key_id:
            raise ValueError("a key cannot replace its own replacement")
        return self


class ServiceKeyCreate(ContractModel):
    installation_id: ResourceId = Field(description="Must name a registered, non-retired ProcessInstallation (404 otherwise).")
    label: ShortText
    scopes: List[ServiceScope] = Field(min_length=1)
    expires_at: Optional[Timestamp] = None


class ServiceKeyIssued(ContractModel):
    key: ServiceKeyRecord
    token: str = Field(pattern=r"^c1sk_[A-Za-z0-9]{6}_[A-Za-z0-9_-]{43,}$", description="ONE-TIME: shown exactly once. Store keeps only key_hash.")
    recommended_env_name: EnvVarName = Field(default="CALL1_STORE_SERVICE_KEY", description="Where Process configuration is expected to reference the token by name.")


class ServiceKeyRotate(ContractModel):
    """Issue a replacement with the same installation and scopes. The old key gets
    ``superseded_by_key_id`` and ``grace_until``; both work until then. Leases, claim tokens and
    job IDs are unaffected (claims are tied to the installation, not the key), reviewer sessions
    never change, and a rotation never adds scopes."""

    grace_seconds: Optional[int] = Field(default=None, ge=0, le=7 * 24 * 3600, description="Defaults to SERVICE_KEY_ROTATION_GRACE; 0 revokes the old key at once.")


class ServiceKeyRevoke(ContractModel):
    reason: SafeText


# --- Loopback console credential (Process side; documented here, never a Store principal) --


class ConsoleOperation(str, Enum):
    """Everything the native console may do, all against Process's loopback UI. Provider endpoints,
    credential variable names, route opt-ins, masking, the attestation policy, approvals, trust
    anchors and the key manager are Store admin state: the console displays them and never changes
    them, and Process uses a provider connection only when its URL and credential name match the
    current admin state."""

    IMPORT_RECORDING = "import_recording"
    VIEW_PIPELINE = "view_pipeline"
    RETRY_JOB = "retry_job"
    CANCEL_JOB = "cancel_job"
    VIEW_MODELS = "view_models"
    SET_MODEL_DEFAULTS = "set_model_defaults"
    """Default catalog entry per purpose and per-run overrides, among entries admin state permits."""
    SET_LOCAL_RUNTIME = "set_local_runtime"
    """Process-local tuning only: pool sizes, timeouts, local model installation and qualification."""
    VIEW_STORE_ADMIN_STATE = "view_store_admin_state"
    VIEW_USAGE = "view_usage"
    OPEN_EVALUATE_IN_BROWSER = "open_evaluate_in_browser"


class ConsoleCredentialDescriptor(ContractModel):
    """The console credential is issued at install, stored hashed in Process's protected
    configuration (readable only by the OS user running Process), accepted only on loopback, and
    scoped to Process operations. It has no Store review or admin rights and is not a Store
    principal: no route in the Store table accepts it."""

    installation_id: ResourceId
    credential_hash: Sha256Digest
    loopback_only: Literal[True] = True
    allowed_operations: List[ConsoleOperation] = Field(default_factory=lambda: list(ConsoleOperation))
    store_rights: Literal["none"] = "none"
    created_at: Timestamp
    rotated_at: Optional[Timestamp] = None


ONE_TIME_SECRET_FIELDS: FrozenSet[tuple] = frozenset({
    ("InvitationIssued", "invitation_url"),
    ("EnrollmentBeginRequest", "invitation_token"),
    ("EnrollmentBeginRequest", "setup_code"),
    ("ServiceKeyIssued", "token"),
})
"""The (model, field) pairs that carry a secret value exactly once and never on a stored record."""

SESSION_BOUND_SECRET_FIELDS: FrozenSet[tuple] = frozenset({
    ("SessionInfo", "csrf_token"),
})
"""Secrets bound to the current session and returned only to it (sign-in, enrollment,
GET /auth/session). Never on a stored record and never in a list of sessions."""

ONE_TIME_CARRIER_MODELS: FrozenSet[str] = frozenset({
    "InvitationIssued", "EnrollmentBeginRequest", "ServiceKeyIssued",
    "SessionInfo", "SignInResponse", "RegistrationFinishResponse",
})
"""Models that may contain a one-time or session-bound secret, directly or by nesting
``SessionInfo``. Each is used only as a top-level request or response of the routes that own it
and is never nested in any other model."""
