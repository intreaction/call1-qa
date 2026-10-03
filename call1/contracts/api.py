"""The versioned Store HTTP surface as a declarative route table.

Every route declares its method, path, request and response models, the principals that may
call it (Process service key with a scope, reviewer session with a minimum role and permission,
or anonymous), its idempotency semantics, whether it writes an audit event, and the error codes
it can return. ``build_app()`` turns the table into a FastAPI app with stub handlers whose only
purpose is to emit OpenAPI; it contains no business logic and opens no database.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Type

from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from pydantic.json_schema import models_json_schema

from . import admin, artifacts, auth, calls, catalog, contents, custody, events, jobs, metrics, release_trust, reviews, rubrics, signals, training, usage, vocabulary
from .common import CANONICAL_JSON, CONTRACT_PARAMETERS, CONTRACT_VERSION, STORE_API_PREFIX, ContractInfo, ContractModel, Page, PrincipalKind, ReviewerRole, ServiceScope
from .errors import ERROR_HTTP_STATUS, JOB_ERROR_CLASSES, PRO1_ATTESTATION_CLASS_CODES, PRO1_PROVIDER_FAILURE_CODES, PROVIDER_FAILURE_CODES, ErrorCode, ErrorResponse, JobErrorCode

# Sentinels for routes whose 200 body is not JSON.
BINARY = "binary"
CSV = "csv"

GENERATOR_PINS = {"fastapi": "0.141", "pydantic": "2.13", "openapi-typescript": "7.13.0"}
"""The generator versions the committed openapi.json and store-v1.ts were produced with (minor
versions for the Python side). ``python -m call1.contracts.generate`` refuses to run, and
``--check`` fails, under different versions, because schema output differs between them."""


class Idempotency(str, Enum):
    NONE = "none"
    """A read, or a write whose repetition has no additional effect."""
    NATURAL = "natural"
    """The resource's own identity deduplicates (source identity, fingerprint, session ID, checksum)."""
    BODY_KEY = "body_key"
    """The body carries idempotency_key or completion_key; a replay returns the original result, a
    reuse with a different payload is 409 idempotency_key_reused / completion_key_reused."""
    HEADER = "header"
    """The Idempotency-Key header is required; same semantics as body_key, scoped to the session."""
    EXPECTED_VERSION = "expected_version"
    """The body carries an expected version; a mismatch is a 409 *_version_conflict with current_version."""


class PrincipalRule(ContractModel):
    kind: PrincipalKind
    scope: Optional[ServiceScope] = None
    min_role: Optional[ReviewerRole] = None
    permission: Optional[auth.Permission] = None


def process(scope: ServiceScope) -> PrincipalRule:
    return PrincipalRule(kind=PrincipalKind.PROCESS_SERVICE_KEY, scope=scope)


def session(min_role: ReviewerRole, permission: auth.Permission) -> PrincipalRule:
    if permission not in auth.ROLE_PERMISSIONS[min_role]:
        raise ValueError(f"role {min_role.value} does not hold {permission.value}")
    return PrincipalRule(kind=PrincipalKind.REVIEWER_SESSION, min_role=min_role, permission=permission)


ANONYMOUS = PrincipalRule(kind=PrincipalKind.ANONYMOUS)
P = auth.Permission
R = ReviewerRole
S = ServiceScope
E = ErrorCode

COMMON_ERRORS = [E.UNAUTHENTICATED, E.FORBIDDEN, E.VALIDATION_FAILED, E.RATE_LIMITED, E.STORE_UNAVAILABLE]


@dataclass(frozen=True)
class Route:
    method: str
    path: str
    operation_id: str
    summary: str
    tag: str
    principals: List[PrincipalRule]
    response: Any = None
    request: Optional[Type[BaseModel]] = None
    query: Optional[Type[BaseModel]] = None
    idempotency: Idempotency = Idempotency.NONE
    audited: bool = False
    status_code: int = 200
    errors: List[ErrorCode] = field(default_factory=list)
    description: str = ""
    object_rule: str = ""
    """A per-object authorization rule Store applies on top of the principal rules."""
    object_permissions: List[auth.Permission] = field(default_factory=list)
    """Permissions the object rule may additionally require."""
    stage: int = 0
    """Delivery stage in docs/AsyncJobPipelinePlan.md that first implements this route (DELIVERY_STAGES)."""

    def all_errors(self) -> List[ErrorCode]:
        base = [] if self.principals == [ANONYMOUS] else COMMON_ERRORS
        if self.principals == [ANONYMOUS]:
            base = [E.VALIDATION_FAILED, E.RATE_LIMITED, E.STORE_UNAVAILABLE]
        if any(p.kind is PrincipalKind.REVIEWER_SESSION for p in self.principals):
            base = base + [E.SESSION_EXPIRED, E.ACCOUNT_DISABLED, E.INSUFFICIENT_ROLE]
            if self.method != "GET":
                base = base + [E.CSRF_FAILED, E.ORIGIN_NOT_ALLOWED]
        if any(p.kind is PrincipalKind.PROCESS_SERVICE_KEY for p in self.principals):
            base = base + [E.INSUFFICIENT_SCOPE]
        seen: List[ErrorCode] = []
        for code in base + list(self.errors):
            if code not in seen:
                seen.append(code)
        return seen


def _r(method: str, path: str, operation_id: str, summary: str, tag: str, principals: List[PrincipalRule], **kw) -> Route:
    if not path.startswith(STORE_API_PREFIX + "/"):
        raise ValueError(f"{path} is not under {STORE_API_PREFIX}")
    return Route(method=method, path=path, operation_id=operation_id, summary=summary, tag=tag, principals=principals, **kw)


_V = STORE_API_PREFIX
NOT_FOUND = [E.NOT_FOUND]
CLAIM_ERRORS = [E.NOT_FOUND, E.CLAIM_TOKEN_STALE, E.JOB_NOT_CLAIMABLE, E.INVALID_TRANSITION]
OWN_INSTALLATION = "Every installation_id in the path or body must equal the calling key's installation (403 forbidden otherwise)."

ROUTES: List[Route] = [
    # --- status --------------------------------------------------------------------------
    _r("GET", f"{_V}/status", "getStatus", "Health, contract version, hostname, relying party and TLS health", "status", [ANONYMOUS], response=admin.StoreHealth),
    _r("GET", f"{_V}/status/detail", "getStatusDetail", "Build, schema, admin-state version, feed epoch and latest cursor", "status", [session(R.ADMIN, P.MANAGE_ADMIN_STATE), process(S.ADMIN_STATE_READ)], response=admin.StoreStatus),
    _r("GET", f"{_V}/contract", "getContract", "Contract version and named parameters", "status", [ANONYMOUS], response=ContractInfo),

    # --- reviewer auth (passkey-only) ------------------------------------------------------
    _r("POST", f"{_V}/auth/enroll/begin", "enrollBegin", "Start enrollment with an invitation token or a setup code", "auth", [ANONYMOUS], request=auth.EnrollmentBeginRequest, response=auth.RegistrationBeginResponse, errors=[E.INVITATION_INVALID, E.SETUP_CODE_INVALID], description="The account's email, display name and role come from the invitation or the setup-code record; the request carries only the code."),
    _r("POST", f"{_V}/auth/enroll/finish", "enrollFinish", "Finish enrollment: verify the registration and sign the new reviewer in", "auth", [ANONYMOUS], request=auth.RegistrationFinishRequest, response=auth.RegistrationFinishResponse, audited=True, errors=[E.WEBAUTHN_VERIFICATION_FAILED, E.ORIGIN_NOT_ALLOWED, E.INVITATION_INVALID, E.SETUP_CODE_INVALID]),
    _r("POST", f"{_V}/auth/sign-in/begin", "signInBegin", "Account-first sign-in: email, then allowCredentials (decoys for unknown accounts)", "auth", [ANONYMOUS], request=auth.AuthenticationBeginRequest, response=auth.AuthenticationBeginResponse),
    _r("POST", f"{_V}/auth/sign-in/finish", "signInFinish", "Verify the assertion against the account bound to the ceremony and create a server-side session", "auth", [ANONYMOUS], request=auth.AuthenticationFinishRequest, response=auth.SignInResponse, errors=[E.WEBAUTHN_VERIFICATION_FAILED, E.ORIGIN_NOT_ALLOWED, E.ACCOUNT_DISABLED]),
    _r("GET", f"{_V}/auth/session", "getSession", "The signed-in reviewer, current role and permissions, and the session's CSRF token", "auth", [session(R.REVIEWER, P.READ_CALLS)], response=auth.SessionInfo),
    _r("POST", f"{_V}/auth/sign-out", "signOut", "End this session", "auth", [session(R.REVIEWER, P.READ_CALLS)], response=auth.SignOutResponse),
    _r("POST", f"{_V}/auth/authenticators/begin", "addAuthenticatorBegin", "Add another authenticator: a step-up assertion challenge plus registration options", "auth", [session(R.REVIEWER, P.MANAGE_OWN_AUTHENTICATORS)], request=auth.AddAuthenticatorBeginRequest, response=auth.AddAuthenticatorBeginResponse),
    _r("POST", f"{_V}/auth/authenticators/finish", "addAuthenticatorFinish", "Verify the fresh user-verified assertion from an existing credential, then the new registration", "auth", [session(R.REVIEWER, P.MANAGE_OWN_AUTHENTICATORS)], request=auth.AddAuthenticatorFinishRequest, response=auth.RegistrationFinishResponse, audited=True, errors=[E.WEBAUTHN_VERIFICATION_FAILED, E.ORIGIN_NOT_ALLOWED]),
    _r("GET", f"{_V}/auth/authenticators", "listOwnAuthenticators", "The signed-in reviewer's authenticators", "auth", [session(R.REVIEWER, P.MANAGE_OWN_AUTHENTICATORS)], response=Page[auth.WebAuthnCredentialRecord]),
    _r("PATCH", f"{_V}/auth/authenticators/{{authenticator_id}}", "renameOwnAuthenticator", "Rename one of the signed-in reviewer's authenticators (authenticator_id = WebAuthnCredentialRecord.id)", "auth", [session(R.REVIEWER, P.MANAGE_OWN_AUTHENTICATORS)], request=auth.CredentialUpdate, response=auth.WebAuthnCredentialRecord, errors=NOT_FOUND),
    _r("DELETE", f"{_V}/auth/authenticators/{{authenticator_id}}", "removeOwnAuthenticator", "Remove an authenticator; the last one cannot be removed", "auth", [session(R.REVIEWER, P.MANAGE_OWN_AUTHENTICATORS)], status_code=204, audited=True, errors=[E.NOT_FOUND, E.CONFLICT]),
    _r("GET", f"{_V}/auth/sessions", "listOwnSessions", "The signed-in reviewer's sessions (handles only, never cookie values)", "auth", [session(R.REVIEWER, P.READ_CALLS)], response=Page[auth.SessionListItem]),
    _r("DELETE", f"{_V}/auth/sessions/{{session_id}}", "revokeOwnSession", "Revoke one of the signed-in reviewer's sessions", "auth", [session(R.REVIEWER, P.READ_CALLS)], status_code=204, audited=True, errors=NOT_FOUND),

    # --- admin: identity -------------------------------------------------------------------
    _r("GET", f"{_V}/admin/accounts", "listAccounts", "Reviewer accounts", "admin-identity", [session(R.ADMIN, P.MANAGE_ACCOUNTS)], query=auth.AccountListQuery, response=Page[auth.ReviewerAccount]),
    _r("GET", f"{_V}/admin/accounts/{{account_id}}", "getAccount", "One reviewer account", "admin-identity", [session(R.ADMIN, P.MANAGE_ACCOUNTS)], response=auth.ReviewerAccount, errors=NOT_FOUND),
    _r("PATCH", f"{_V}/admin/accounts/{{account_id}}", "updateAccount", "Change role, display name or active/disabled status", "admin-identity", [session(R.ADMIN, P.MANAGE_ACCOUNTS)], request=auth.AccountUpdate, response=auth.ReviewerAccount, audited=True, errors=[E.NOT_FOUND, E.CONFLICT], description="Disabling revokes every session of the account in the same transaction; a role change applies from the next request."),
    _r("GET", f"{_V}/admin/accounts/{{account_id}}/authenticators", "listAccountAuthenticators", "An account's authenticators", "admin-identity", [session(R.ADMIN, P.MANAGE_ACCOUNTS)], response=Page[auth.WebAuthnCredentialRecord], errors=NOT_FOUND),
    _r("DELETE", f"{_V}/admin/accounts/{{account_id}}/authenticators/{{authenticator_id}}", "revokeAccountAuthenticator", "Revoke one credential of an account (a lost key); revoking the last one makes the account reinvite_required", "admin-identity", [session(R.ADMIN, P.MANAGE_ACCOUNTS)], status_code=204, audited=True, errors=NOT_FOUND),
    _r("DELETE", f"{_V}/admin/accounts/{{account_id}}/sessions", "revokeAccountSessions", "Revoke every session of an account", "admin-identity", [session(R.ADMIN, P.MANAGE_SESSIONS)], status_code=204, audited=True, errors=NOT_FOUND),
    _r("POST", f"{_V}/admin/invitations", "createInvitation", "Issue a single-use, expiring invitation (or re-invite, revoking the account's credentials and sessions)", "admin-identity", [session(R.ADMIN, P.MANAGE_INVITATIONS)], request=auth.InvitationCreate, response=auth.InvitationIssued, audited=True, errors=[E.CONFLICT]),
    _r("GET", f"{_V}/admin/invitations", "listInvitations", "Invitations", "admin-identity", [session(R.ADMIN, P.MANAGE_INVITATIONS)], query=auth.InvitationListQuery, response=Page[auth.Invitation]),
    _r("DELETE", f"{_V}/admin/invitations/{{invitation_id}}", "revokeInvitation", "Revoke a pending invitation", "admin-identity", [session(R.ADMIN, P.MANAGE_INVITATIONS)], status_code=204, audited=True, errors=[E.NOT_FOUND, E.INVITATION_INVALID]),
    _r("GET", f"{_V}/admin/setup-codes", "listSetupCodes", "Setup codes issued on the Store host (issuance is host-only, never an HTTP route)", "admin-identity", [session(R.ADMIN, P.READ_AUDIT)], response=Page[auth.SetupCodeRecord]),
    _r("GET", f"{_V}/admin/break-glass", "listBreakGlass", "Break-glass enrollments", "admin-identity", [session(R.ADMIN, P.READ_AUDIT)], response=Page[auth.BreakGlassRecord]),
    _r("POST", f"{_V}/admin/installations", "registerInstallation", "Register a Process installation (keys are issued to it)", "admin-identity", [session(R.ADMIN, P.MANAGE_SERVICE_KEYS)], request=auth.InstallationCreate, response=auth.ProcessInstallation, status_code=201, audited=True, errors=[E.CONFLICT]),
    _r("GET", f"{_V}/admin/installations", "listInstallations", "Registered Process installations", "admin-identity", [session(R.ADMIN, P.MANAGE_SERVICE_KEYS)], response=Page[auth.ProcessInstallation]),
    _r("POST", f"{_V}/admin/installations/{{installation_id}}/retire", "retireInstallation", "Retire an installation and revoke all its keys", "admin-identity", [session(R.ADMIN, P.MANAGE_SERVICE_KEYS)], request=auth.InstallationRetire, response=auth.ProcessInstallation, audited=True, errors=[E.NOT_FOUND, E.INVALID_TRANSITION]),
    _r("POST", f"{_V}/admin/service-keys", "createServiceKey", "Issue a scoped Process service key for a registered installation; the token is shown once", "admin-identity", [session(R.ADMIN, P.MANAGE_SERVICE_KEYS)], request=auth.ServiceKeyCreate, response=auth.ServiceKeyIssued, status_code=201, audited=True, errors=NOT_FOUND),
    _r("GET", f"{_V}/admin/service-keys", "listServiceKeys", "Service keys (hashes and metadata only)", "admin-identity", [session(R.ADMIN, P.MANAGE_SERVICE_KEYS)], response=Page[auth.ServiceKeyRecord]),
    _r("POST", f"{_V}/admin/service-keys/{{key_id}}/rotate", "rotateServiceKey", "Issue a replacement key with the same installation and scopes; the old key works until grace_until", "admin-identity", [session(R.ADMIN, P.MANAGE_SERVICE_KEYS)], request=auth.ServiceKeyRotate, response=auth.ServiceKeyIssued, audited=True, errors=[E.NOT_FOUND, E.CONFLICT]),
    _r("POST", f"{_V}/admin/service-keys/{{key_id}}/revoke", "revokeServiceKey", "Revoke a key immediately; active leases held under it expire normally", "admin-identity", [session(R.ADMIN, P.MANAGE_SERVICE_KEYS)], request=auth.ServiceKeyRevoke, response=auth.ServiceKeyRecord, audited=True, errors=NOT_FOUND),
    _r("GET", f"{_V}/admin/reviewer-profiles", "listReviewerProfiles", "Work-distribution profiles of reviewer accounts", "admin-identity", [session(R.SUPERVISOR, P.ASSIGN_REVIEW)], response=Page[reviews.ReviewerProfile]),
    _r("PATCH", f"{_V}/admin/reviewer-profiles/{{account_id}}", "updateReviewerProfile", "Skills, capacity and assignment availability", "admin-identity", [session(R.SUPERVISOR, P.ASSIGN_REVIEW)], request=reviews.ReviewerProfileUpdate, response=reviews.ReviewerProfile, errors=NOT_FOUND),

    # --- admin: trust state ----------------------------------------------------------------
    _r("GET", f"{_V}/admin/state", "getAdminState", "Attestation policy, release floors, trust anchors, key manager, route opt-ins and endpoints, masking", "admin-state", [session(R.ADMIN, P.MANAGE_ADMIN_STATE), process(S.ADMIN_STATE_READ)], response=admin.AdminState),
    _r("POST", f"{_V}/admin/state/changes", "changeAdminState", "One audited change to one section, with the expected state version", "admin-state", [session(R.ADMIN, P.MANAGE_ADMIN_STATE)], request=admin.AdminStateChange, response=admin.AdminStateChanged, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.STATE_VERSION_CONFLICT, E.ROUTE_NOT_PERMITTED, E.CONFLICT], description="Lowering a release floor needs confirm_lower; adopting anchors needs the pending digest the admin reviewed (409 conflict otherwise)."),
    _r("GET", f"{_V}/release-trust/pro1-connection", "getPro1Connection", "The Pro1 connection state (ready, pending verification or blocked)", "release-trust", [session(R.ADMIN, P.MANAGE_ADMIN_STATE), process(S.ADMIN_STATE_READ)], response=custody.Pro1Connection),
    _r("POST", f"{_V}/admin/pro1/clear-block", "clearPro1Block", "Acknowledge a blocked Pro1 connection; it moves to pending_verification until a fresh verification passes", "admin-state", [session(R.ADMIN, P.MANAGE_ADMIN_STATE)], request=custody.Pro1BlockClear, response=custody.Pro1Connection, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.STATE_VERSION_CONFLICT, E.INVALID_TRANSITION]),
    _r("POST", f"{_V}/release-trust/pro1-connection/verifications", "reportPro1Verification", "Process reports an attested verification outside a job (connection test, post-clear check); Store applies the connection transitions", "release-trust", [process(S.RELEASE_TRUST_WRITE)], request=custody.Pro1VerificationReport, response=custody.Pro1Connection, audited=True, description="A passing report moves pending_verification to ready; a definitive failure blocks; a transient failure changes nothing. Process never sets a status directly."),
    _r("GET", f"{_V}/admin/releases/pending", "listPendingReleases", "Logged releases awaiting an admin decision, with their evidence", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES), process(S.ADMIN_STATE_READ)], query=release_trust.PendingReleaseQuery, response=Page[release_trust.PendingRelease]),
    _r("POST", f"{_V}/admin/releases/pending/decline", "declineRelease", "Decline one pending release (it leaves the pending list and banner)", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES)], request=release_trust.ReleaseDeclineRequest, response=release_trust.PendingRelease, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.STATE_VERSION_CONFLICT, E.CONFLICT]),
    _r("GET", f"{_V}/admin/releases/approvals", "listReleaseApprovals", "Release approvals (approved manifest digest and log index)", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES), process(S.ADMIN_STATE_READ)], response=Page[release_trust.ReleaseApproval]),
    _r("POST", f"{_V}/admin/releases/approvals", "approveRelease", "Approve one pending release by its exact manifest digest; Store copies the evidence the log scan recorded", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES)], request=release_trust.ReleaseApprovalRequest, response=release_trust.ReleaseApproval, status_code=201, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.STATE_VERSION_CONFLICT, E.CONFLICT]),
    _r("POST", f"{_V}/admin/releases/approvals/{{approval_id}}/withdraw", "withdrawReleaseApproval", "Withdraw an approval", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES)], request=release_trust.ApprovalWithdrawRequest, response=release_trust.ReleaseApproval, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.STATE_VERSION_CONFLICT, E.INVALID_TRANSITION]),
    _r("GET", f"{_V}/release-trust/log-scans", "listLogScanStates", "Transparency-log scan state per installation and log", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES), process(S.ADMIN_STATE_READ)], response=Page[release_trust.LogScanState]),
    _r("PUT", f"{_V}/release-trust/log-scans/{{log_id}}", "putLogScanState", "Process records the result of its own full log scan (audited when equivocation is detected)", "release-trust", [process(S.RELEASE_TRUST_WRITE)], request=release_trust.LogScanStateInput, response=release_trust.LogScanState, idempotency=Idempotency.NATURAL, audited=True, errors=[E.CONFLICT], object_rule="Keyed by (the key's installation, log_id); a lower tree size or revocation sequence than recorded is 409 conflict and audited as possible equivocation."),
    _r("POST", f"{_V}/release-trust/trust-anchor-bundles/uploads", "uploadTrustAnchorBundle", "Upload the trust-anchor bundle a Process build shipped (global artifact, by digest)", "release-trust", [process(S.RELEASE_TRUST_WRITE)], request=artifacts.GlobalArtifactUploadRequest, response=artifacts.GlobalArtifactUpload, status_code=201, idempotency=Idempotency.NATURAL),
    _r("POST", f"{_V}/release-trust/trust-anchor-bundles", "submitTrustAnchorBundle", "Register an uploaded bundle as the pending bundle; only an admin adopts it", "release-trust", [process(S.RELEASE_TRUST_WRITE)], request=release_trust.TrustAnchorBundleSubmit, response=release_trust.TrustAnchorState, idempotency=Idempotency.NATURAL, audited=True, errors=[E.NOT_FOUND, E.CHECKSUM_MISMATCH], description="Registering a pending bundle does not bump state_version and changes nothing the verifier uses."),
    _r("GET", f"{_V}/release-trust/trust-anchor-bundles/{{artifact_id}}/content", "getTrustAnchorBundle", "Bytes of a stored bundle (Process fetches the adopted one by ID and checksum)", "release-trust", [session(R.ADMIN, P.MANAGE_ADMIN_STATE), process(S.ADMIN_STATE_READ)], response=BINARY, errors=NOT_FOUND),
    _r("POST", f"{_V}/admin/updates/packages", "stageUpdatePackage", "Select a downloaded appliance package; returns an upload URL for its bytes", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES)], request=release_trust.UpdatePackageCreate, response=release_trust.UpdatePackage, status_code=201, audited=True),
    _r("GET", f"{_V}/admin/updates/packages", "listUpdatePackages", "Staged packages with their verification", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES)], response=Page[release_trust.UpdatePackage]),
    _r("GET", f"{_V}/admin/updates/packages/{{package_id}}", "getUpdatePackage", "One staged package with its verification", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES)], response=release_trust.UpdatePackage, errors=NOT_FOUND),
    _r("POST", f"{_V}/admin/updates/packages/{{package_id}}/commit", "commitUpdatePackage", "Finish the upload; Store's installed updater verifies the package and records its verdict", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES)], request=artifacts.UploadCommit, response=release_trust.UpdatePackage, idempotency=Idempotency.NATURAL, audited=True, errors=[E.NOT_FOUND, E.CHECKSUM_MISMATCH, E.UPLOAD_EXPIRED]),
    _r("POST", f"{_V}/admin/updates/packages/{{package_id}}/install", "installUpdatePackage", "Install a verified package whose approval is still active", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES)], request=release_trust.UpdateInstallRequest, response=release_trust.UpdatePackage, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.STATE_VERSION_CONFLICT, E.INVALID_TRANSITION, E.CONFLICT]),
    _r("GET", f"{_V}/release-trust/updater-verifications", "listUpdaterVerifications", "Updater verification records (written only by Store's installed updater)", "release-trust", [session(R.ADMIN, P.APPROVE_RELEASES)], response=Page[release_trust.UpdaterVerificationRecord]),
    _r("GET", f"{_V}/admin/egress-allowlist", "getEgressAllowlist", "The installer's egress allowlist for customer IT", "admin-state", [session(R.ADMIN, P.MANAGE_ADMIN_STATE)], response=release_trust.EgressAllowlist),
    _r("GET", f"{_V}/admin/tls", "getTlsState", "Certificate hostname, trust path, expiry and renewal owner", "admin-state", [session(R.ADMIN, P.MANAGE_ADMIN_STATE)], response=admin.TlsState),
    _r("GET", f"{_V}/admin/backup/manifest", "getBackupManifest", "Describe the current dataset for backup verification (backup itself runs offline on the Store host)", "admin-state", [session(R.ADMIN, P.MANAGE_BACKUP)], response=admin.BackupManifest),
    _r("POST", f"{_V}/admin/backup/restore-preflight", "restorePreflight", "Check a backup manifest against the target hostname before restoring", "admin-state", [session(R.ADMIN, P.MANAGE_BACKUP)], request=admin.RestorePreflightRequest, response=admin.RestorePreflightResult, errors=[E.HOSTNAME_INVALID]),
    _r("GET", f"{_V}/admin/audit", "listAuditEvents", "Audit events", "admin-state", [session(R.ADMIN, P.READ_AUDIT)], query=events.AuditQuery, response=Page[events.AuditEvent]),

    # --- usage -----------------------------------------------------------------------------
    _r("POST", f"{_V}/admin/usage/report", "usageReport", "Usage rollup by route class, model, purpose and hardware profile, optionally with estimates from the price table (audited export)", "usage", [session(R.ADMIN, P.READ_USAGE_REPORT)], request=usage.UsageReportQuery, response=usage.UsageReport, audited=True),
    _r("GET", f"{_V}/admin/usage/records", "listUsageRecords", "Raw usage records", "usage", [session(R.ADMIN, P.READ_USAGE_REPORT)], query=usage.UsageRecordQuery, response=Page[usage.UsageRecord]),
    _r("POST", f"{_V}/admin/usage/records.csv", "exportUsageRecordsCsv", "CSV export of the raw usage records in the range (USAGE_CSV_COLUMNS; audited export)", "usage", [session(R.ADMIN, P.READ_USAGE_REPORT)], request=usage.UsageReportQuery, response=CSV, audited=True),
    _r("GET", f"{_V}/admin/usage/price-table", "getPriceTable", "The admin-entered, dated price table", "usage", [session(R.ADMIN, P.READ_USAGE_REPORT)], response=usage.PriceTable),
    _r("PUT", f"{_V}/admin/usage/price-table", "savePriceTable", "Replace the price table", "usage", [session(R.ADMIN, P.MANAGE_ADMIN_STATE)], request=usage.PriceTableSave, response=usage.PriceTable, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.CONFLICT]),
    _r("GET", f"{_V}/usage/medians", "getUsageMedians", "Measured medians per catalog entry, route and hardware profile (aggregates only)", "usage", [session(R.ADMIN, P.READ_USAGE_REPORT), process(S.USAGE_READ)], query=usage.UsageMediansQuery, response=usage.UsageMedians),

    # --- key release and attestation evidence ----------------------------------------------
    _r("POST", f"{_V}/key-releases", "recordKeyRelease", "Process records a session-key release before the wrapped key leaves the appliance", "custody", [process(S.KEY_RELEASE_WRITE)], request=custody.KeyReleaseInput, response=custody.KeyReleaseRecord, status_code=201, idempotency=Idempotency.NATURAL, audited=True, errors=[E.CONFLICT, E.ROUTE_NOT_PERMITTED, E.NOT_FOUND], object_rule="Store checks custody.KEY_RELEASE_PRECONDITIONS in order."),
    _r("POST", f"{_V}/key-releases/{{key_release_id}}/close", "closeKeyRelease", "Record the end of the session", "custody", [process(S.KEY_RELEASE_WRITE)], request=custody.KeyReleaseClose, response=custody.KeyReleaseRecord, audited=True, errors=[E.NOT_FOUND, E.INVALID_TRANSITION]),
    _r("GET", f"{_V}/key-releases", "listKeyReleases", "Key-release records (the key-release log)", "custody", [session(R.ADMIN, P.READ_AUDIT), process(S.ADMIN_STATE_READ)], response=Page[custody.KeyReleaseRecord]),
    _r("POST", f"{_V}/attestation-evidence/uploads", "uploadAttestationEvidence", "Store an evidence bundle once, conversation-independent, by digest", "custody", [process(S.KEY_RELEASE_WRITE)], request=artifacts.GlobalArtifactUploadRequest, response=artifacts.GlobalArtifactUpload, status_code=201, idempotency=Idempotency.NATURAL),
    _r("GET", f"{_V}/attestation-evidence/{{digest}}", "getAttestationEvidence", "Evidence bundle metadata, for the evidence viewer and offline re-verification", "custody", [session(R.ADMIN, P.READ_AUDIT), process(S.ARTIFACTS_READ)], response=artifacts.Artifact, errors=NOT_FOUND),
    _r("GET", f"{_V}/attestation-evidence/{{digest}}/content", "getAttestationEvidenceContent", "Evidence bundle bytes", "custody", [session(R.ADMIN, P.READ_AUDIT), process(S.ARTIFACTS_READ)], response=BINARY, errors=NOT_FOUND),

    # --- hardware and catalog --------------------------------------------------------------
    _r("PUT", f"{_V}/hardware-profiles", "upsertHardwareProfile", "Upsert a hardware profile by fingerprint", "usage", [process(S.HARDWARE_WRITE)], request=usage.HardwareProfileInput, response=usage.HardwareProfile, idempotency=Idempotency.NATURAL),
    _r("GET", f"{_V}/hardware-profiles", "listHardwareProfiles", "Hardware profiles", "usage", [session(R.ADMIN, P.READ_USAGE_REPORT), process(S.HARDWARE_WRITE)], response=Page[usage.HardwareProfile]),
    _r("PUT", f"{_V}/catalog-snapshot", "publishCatalogSnapshot", "Process publishes its own installation's read-only catalog snapshot", "catalog", [process(S.CATALOG_PUBLISH)], request=catalog.CatalogSnapshot, response=catalog.CatalogSnapshot, idempotency=Idempotency.NATURAL, audited=True, errors=[E.ROUTE_NOT_PERMITTED], object_rule=OWN_INSTALLATION),
    _r("GET", f"{_V}/catalog-snapshots", "listCatalogSnapshots", "Catalog snapshots of every Process installation", "catalog", [session(R.REVIEWER, P.READ_CALLS)], response=Page[catalog.CatalogSnapshot]),

    # --- conversations and artifacts -------------------------------------------------------
    _r("POST", f"{_V}/conversations", "registerConversation", "Register a source; a duplicate source identity returns the existing conversation, with its call metadata updated when the registration changes it (audited when it does)", "conversations", [process(S.CALLS_WRITE)], request=calls.ConversationRegistration, response=calls.ConversationRegistered, idempotency=Idempotency.NATURAL, audited=True, description="Natural idempotency by the source's dedup identity. A new identity creates the conversation (created: true). A known identity returns the existing conversation (created: false); since 1.1.0, if the registration's call_metadata differs, Store applies calls.merge_call_metadata (only the fields the body sets replace the stored ones; absent fields are kept, an explicit null clears), and in one transaction saves it, refreshes every read projection that shows it (call list and detail, review-queue items, escalations), writes a call_metadata_updated audit event (details: conversation_id and updated_fields, never values) and a call change event with status metadata_updated. The response then has metadata_updated: true and updated_fields. Nothing is reprocessed: no graph, no reanalysis, no stale result, and the review version is unchanged. Identical metadata is a pure replay (metadata_updated: false, no event). Concurrent re-registrations apply in commit order; the last one wins field by field."),
    _r("GET", f"{_V}/conversations/{{conversation_id}}", "getConversation", "One conversation", "conversations", [process(S.ARTIFACTS_READ), session(R.SUPERVISOR, P.READ_JOBS)], response=calls.Conversation, errors=NOT_FOUND),
    _r("POST", f"{_V}/conversations/{{conversation_id}}/artifacts", "createInlineArtifact", "Commit a small JSON artifact in one call", "artifacts", [process(S.ARTIFACTS_WRITE)], request=artifacts.InlineArtifactCreate, response=artifacts.Artifact, status_code=201, idempotency=Idempotency.NATURAL, errors=[E.NOT_FOUND, E.CHECKSUM_MISMATCH, E.PAYLOAD_TOO_LARGE, E.CLAIM_TOKEN_STALE], description="The payload is the content model's full dump (artifacts.canonical_content); Store stores exactly its canonical bytes and never normalizes. rubric_snapshot and signal_taxonomy_snapshot artifacts are refused (Store mints them). A job output of a draft-test graph uses its request's draft:<request_id>: slot, and no other artifact created here does (validation_failed)."),
    _r("POST", f"{_V}/conversations/{{conversation_id}}/rubric-snapshots", "mintRubricSnapshot", "Store copies one published rubric version into the conversation as a linked rubric_snapshot input artifact", "artifacts", [process(S.JOBS_WRITE)], request=rubrics.RubricSnapshotRequest, response=artifacts.Artifact, status_code=201, idempotency=Idempotency.NATURAL, errors=[E.NOT_FOUND], description="Idempotent per (conversation, rubric_id, version): slot rubric:<rubric_id>:v<version>, no producing job, linked at commit. Active and retired versions both mint, so a reanalysis can name an older version."),
    _r("POST", f"{_V}/conversations/{{conversation_id}}/signal-taxonomy-snapshots", "mintSignalTaxonomySnapshot", "Store copies one published signal taxonomy version (with the current settings) into the conversation as a linked signal_taxonomy_snapshot input artifact", "artifacts", [process(S.JOBS_WRITE)], request=signals.SignalTaxonomySnapshotRequest, response=artifacts.Artifact, status_code=201, idempotency=Idempotency.NATURAL, errors=[E.NOT_FOUND, E.CONFLICT], description="Added in 1.3.0. Idempotent per (conversation, version, current settings): slot signals:v<version>, no producing job, linked at commit, content signals.SignalTaxonomySnapshotContent with source published. Store returns the snapshot linked in the slot only when it matches the version and the current settings (signals.signal_taxonomy_snapshot_current); after a settings change it links a new version in the same slot (per-slot supersession), so a reanalysis never runs with stale settings. Every v2 job pins it under input role 'taxonomy' and freezes its digest in parameters.signals.taxonomy_digest; Store checks the two at graph creation (graph_invalid). A version whose text was redacted cannot be minted (409 conflict, details.reason redacted). Store reads taxonomy versions through the results area's API (results_api), never its tables."),
    _r("POST", f"{_V}/conversations/{{conversation_id}}/artifacts/uploads", "createUploadGrant", "Get a short-lived, object-scoped upload URL for a large artifact", "artifacts", [process(S.ARTIFACTS_WRITE)], request=artifacts.UploadGrantRequest, response=artifacts.UploadGrant, status_code=201, idempotency=Idempotency.NATURAL, errors=[E.NOT_FOUND, E.CLAIM_TOKEN_STALE]),
    _r("POST", f"{_V}/artifact-uploads/{{upload_id}}/commit", "commitUpload", "Verify the uploaded bytes and commit the artifact metadata", "artifacts", [process(S.ARTIFACTS_WRITE), process(S.KEY_RELEASE_WRITE), process(S.RELEASE_TRUST_WRITE)], request=artifacts.UploadCommit, response=artifacts.Artifact, status_code=201, idempotency=Idempotency.NATURAL, errors=[E.NOT_FOUND, E.CHECKSUM_MISMATCH, E.UPLOAD_EXPIRED], object_rule="The key must hold the scope that created the grant: artifacts:write for conversation artifacts, key-release:write for attestation evidence, release-trust:write for trust-anchor bundles."),
    _r("GET", f"{_V}/conversations/{{conversation_id}}/artifacts", "listArtifacts", "Artifacts of a conversation (linked only, unless asked)", "artifacts", [process(S.ARTIFACTS_READ), session(R.SUPERVISOR, P.READ_JOBS)], query=artifacts.ArtifactListQuery, response=Page[artifacts.Artifact], errors=NOT_FOUND),
    _r("GET", f"{_V}/artifacts/{{artifact_id}}", "getArtifact", "Artifact metadata", "artifacts", [process(S.ARTIFACTS_READ), session(R.SUPERVISOR, P.READ_JOBS)], response=artifacts.Artifact, errors=NOT_FOUND),
    _r("GET", f"{_V}/artifacts/{{artifact_id}}/content-grant", "getContentGrant", "Short-lived download URL for an artifact's bytes", "artifacts", [process(S.ARTIFACTS_READ)], response=artifacts.ContentGrant, errors=NOT_FOUND),
    _r("GET", f"{_V}/artifacts/{{artifact_id}}/content", "getArtifactContent", "Stream an artifact's bytes", "artifacts", [process(S.ARTIFACTS_READ)], response=BINARY, errors=NOT_FOUND),

    # --- processing jobs (never the review queue) -------------------------------------------
    _r("POST", f"{_V}/conversations/{{conversation_id}}/job-graphs", "createJobGraph", "Create an idempotent job graph; dependents start BLOCKED, ready jobs QUEUED (a reanalysis graph also fulfils its request)", "jobs", [process(S.JOBS_WRITE)], request=jobs.JobGraphRequest, response=jobs.JobGraph, status_code=201, idempotency=Idempotency.BODY_KEY, errors=[E.NOT_FOUND, E.GRAPH_INVALID, E.IDEMPOTENCY_KEY_REUSED, E.CHECKSUM_MISMATCH, E.ROUTE_NOT_PERMITTED, E.CLAIM_TOKEN_STALE, E.INVALID_TRANSITION]),
    _r("GET", f"{_V}/job-graphs/{{graph_id}}", "getJobGraph", "One job graph with its edges", "jobs", [process(S.JOBS_WRITE), session(R.SUPERVISOR, P.READ_JOBS)], response=jobs.JobGraph, errors=NOT_FOUND),
    _r("POST", f"{_V}/jobs/claim", "claimJobs", "Atomically claim eligible jobs against the offered slots: one lease and claim token each", "jobs", [process(S.JOBS_CLAIM)], request=jobs.ClaimRequest, response=jobs.ClaimResponse, object_rule=OWN_INSTALLATION),
    _r("GET", f"{_V}/jobs", "listJobs", "Jobs", "jobs", [process(S.JOBS_WRITE), session(R.SUPERVISOR, P.READ_JOBS)], query=jobs.JobListQuery, response=Page[jobs.Job]),
    _r("GET", f"{_V}/jobs/{{job_id}}", "getJob", "One job with its lease, edges, blocking reasons and outputs", "jobs", [process(S.JOBS_WRITE), session(R.SUPERVISOR, P.READ_JOBS)], response=jobs.Job, errors=NOT_FOUND),
    _r("GET", f"{_V}/jobs/{{job_id}}/attempts", "listAttempts", "Attempts (and released claims) of a job with provenance", "jobs", [process(S.JOBS_WRITE), session(R.SUPERVISOR, P.READ_JOBS)], response=Page[jobs.Attempt], errors=NOT_FOUND),
    _r("POST", f"{_V}/jobs/{{job_id}}/heartbeat", "heartbeat", "Renew the lease; learn whether cancellation was requested", "jobs", [process(S.JOBS_WRITE)], request=jobs.HeartbeatRequest, response=jobs.HeartbeatResponse, errors=CLAIM_ERRORS),
    _r("POST", f"{_V}/jobs/{{job_id}}/complete", "completeJob", "Idempotent completion: outputs, usage, receipt, result projection, follow-on jobs and dependent release in one transaction", "jobs", [process(S.JOBS_WRITE)], request=jobs.CompletionRequest, response=jobs.CompletionReceipt, idempotency=Idempotency.BODY_KEY, errors=CLAIM_ERRORS + [E.COMPLETION_KEY_REUSED, E.JOB_CANCELLING, E.CHECKSUM_MISMATCH, E.GRAPH_INVALID, E.ROUTE_NOT_PERMITTED], object_rule="The completion key is looked up before the claim token (jobs.COMPLETION_TRANSACTION_STEPS). claim_token_stale details: current_attempt_number, status, and your_attempt_outcome (succeeded, failed, lease_expired, released or cancelled) for the attempt the token belonged to. " + OWN_INSTALLATION),
    _r("POST", f"{_V}/jobs/{{job_id}}/fail", "failJob", "Report a failed attempt with its safe error code and usage; Store schedules backoff or terminal failure, and blocks the Pro1 connection on a definitive Pro1 code", "jobs", [process(S.JOBS_WRITE)], request=jobs.FailureRequest, response=jobs.FailureReceipt, idempotency=Idempotency.BODY_KEY, audited=True, errors=CLAIM_ERRORS + [E.COMPLETION_KEY_REUSED], object_rule="Same key lookup order as completion. " + OWN_INSTALLATION),
    _r("POST", f"{_V}/jobs/{{job_id}}/release", "releaseJob", "End a claim before inference started without consuming an attempt: requeue (temporary) or reject (configuration)", "jobs", [process(S.JOBS_WRITE)], request=jobs.JobReleaseRequest, response=jobs.JobReleaseReceipt, idempotency=Idempotency.BODY_KEY, errors=CLAIM_ERRORS + [E.COMPLETION_KEY_REUSED]),
    _r("POST", f"{_V}/jobs/{{job_id}}/attempts/{{attempt_number}}/usage", "attachLateUsage", "Attach a late worker's measurements to its lease-expired attempt's synthesized usage row (once)", "jobs", [process(S.JOBS_WRITE)], request=usage.LateUsageReport, response=usage.UsageRecord, idempotency=Idempotency.NATURAL, errors=[E.NOT_FOUND, E.CLAIM_TOKEN_STALE, E.CONFLICT], object_rule="The token must be the one that attempt was claimed with; the attempt must have ended by lease expiry; accepted once."),
    _r("POST", f"{_V}/jobs/{{job_id}}/retry", "retryJob", "Manual retry of one failed job", "jobs", [process(S.JOBS_CONTROL)], request=jobs.RetryRequest, response=jobs.Job, audited=True, errors=[E.NOT_FOUND, E.INVALID_TRANSITION]),
    _r("POST", f"{_V}/jobs/{{job_id}}/cancel", "cancelJob", "Cancel a job and, by default, everything that depends on it", "jobs", [process(S.JOBS_CONTROL)], request=jobs.CancelRequest, response=jobs.CancelResponse, audited=True, errors=[E.NOT_FOUND, E.INVALID_TRANSITION]),
    _r("GET", f"{_V}/conversations/{{conversation_id}}/progress", "getGroupProgress", "Compact per-result-group progress with completed/total counts and derived states", "jobs", [process(S.JOBS_WRITE), session(R.REVIEWER, P.READ_CALLS)], response=jobs.JobGroupProgress, errors=NOT_FOUND),
    _r("GET", f"{_V}/conversations/{{conversation_id}}/usage", "listConversationUsage", "Usage records of a conversation", "usage", [process(S.JOBS_WRITE), session(R.ADMIN, P.READ_USAGE_REPORT)], response=Page[usage.UsageRecord], errors=NOT_FOUND),

    # --- reanalysis (Evaluate -> Store -> Process) ----------------------------------------
    _r("POST", f"{_V}/calls/{{call_id}}/reanalysis-requests", "requestReanalysis", "Durable, idempotent request for new machine results; repeated clicks do not duplicate work", "reanalysis", [session(R.REVIEWER, P.REQUEST_REANALYSIS)], request=jobs.ReanalysisRequestCreate, response=jobs.ReanalysisRequest, status_code=201, idempotency=Idempotency.HEADER, audited=True, errors=[E.NOT_FOUND, E.IDEMPOTENCY_KEY_REUSED, E.CONFLICT], description="qa_draft_test and contact_signals_preview are refused here (their own routes create them). Since 1.3.0 Store resolves signal_taxonomy_version and signal_pipeline for kinds that run contact signals, and at most one pending contact_signals request exists per call: a new one widens a pending, unclaimed request (latest taxonomy version, higher priority, rescore_signals or-ed) and returns it instead of 409. 'Update signals' on one call is kind contact_signals: it reruns only the stages whose digests differ unless rescore_signals is set."),
    _r("POST", f"{_V}/rubrics/{{rubric_id}}/draft/tests", "testRubricDraft", "Score one call with the current draft: Store snapshots the draft and creates a qa_draft_test request", "reanalysis", [session(R.SUPERVISOR, P.MANAGE_RUBRICS)], request=jobs.DraftTestRequest, response=jobs.ReanalysisRequest, status_code=201, idempotency=Idempotency.HEADER, audited=True, errors=[E.NOT_FOUND, E.IDEMPOTENCY_KEY_REUSED, E.RUBRIC_VERSION_CONFLICT]),
    _r("GET", f"{_V}/reanalysis-requests/{{request_id}}/draft-result", "getDraftTestResult", "The scorecard of a draft test (never the call's QA)", "reanalysis", [session(R.SUPERVISOR, P.MANAGE_RUBRICS)], response=jobs.DraftTestResult, errors=[E.NOT_FOUND, E.INVALID_TRANSITION]),
    _r("GET", f"{_V}/calls/{{call_id}}/reanalysis-requests", "listReanalysisRequests", "Reanalysis requests of a call", "reanalysis", [session(R.REVIEWER, P.READ_CALLS)], response=Page[jobs.ReanalysisRequest], errors=NOT_FOUND),
    _r("GET", f"{_V}/reanalysis-requests/{{request_id}}", "getReanalysisRequest", "One reanalysis request", "reanalysis", [session(R.REVIEWER, P.READ_CALLS), process(S.REANALYSIS_CLAIM)], response=jobs.ReanalysisRequest, errors=NOT_FOUND),
    _r("POST", f"{_V}/reanalysis-requests/claim", "claimReanalysisRequests", "Process claims pending requests under a lease; the graph that fulfils one carries its claim token", "reanalysis", [process(S.REANALYSIS_CLAIM)], request=jobs.ReanalysisClaimRequest, response=jobs.ReanalysisClaimResponse, description="Since 1.3.0: requests are claimed in order of priority (desc), requested_at, id (jobs.reanalysis_claim_order_key), and kinds, when present, limits the claim to those kinds (absent means every kind). A 1.3.0 Store and a 1.2.x Process do not interoperate: Process and Store upgrade together."),
    _r("POST", f"{_V}/reanalysis-requests/{{request_id}}/reject", "rejectReanalysisRequest", "Reject a claimed request with a safe reason", "reanalysis", [process(S.REANALYSIS_CLAIM)], request=jobs.ReanalysisReject, response=jobs.ReanalysisRequest, errors=[E.NOT_FOUND, E.CLAIM_TOKEN_STALE, E.INVALID_TRANSITION]),

    # --- calls (Evaluate reads) -----------------------------------------------------------
    _r("GET", f"{_V}/calls", "listCalls", "Paged call list with derived result states", "calls", [session(R.REVIEWER, P.READ_CALLS)], query=calls.CallListQuery, response=Page[calls.CallListItem]),
    _r("GET", f"{_V}/calls/{{call_id}}", "getCall", "Call detail: result groups, pending work, review version, current evaluation", "calls", [session(R.REVIEWER, P.READ_CALLS)], response=calls.CallDetail, errors=NOT_FOUND),
    _r("GET", f"{_V}/calls/{{call_id}}/transcript", "getTranscript", "Current transcript projection (masked per Store settings)", "calls", [session(R.REVIEWER, P.READ_CALLS)], response=calls.TranscriptView, errors=NOT_FOUND, description="Since 1.2.0 (decision 19): with reviewer reads masked, text is masked over the union of the rule-based values and the model PII findings (`pii_findings`) made for this transcript revision. Until those findings are linked (or when they could not be made), the view fails closed: `text_withheld` is true, every turn's text is empty and word timestamps are null, and the transcript result group reads `partial`. Evaluation, summary and contact-signal text read meanwhile is fully redacted, and semantic search returns no hits from the call."),
    _r("GET", f"{_V}/calls/{{call_id}}/evaluation", "getEvaluation", "Current automated scorecard", "calls", [session(R.REVIEWER, P.READ_CALLS)], response=calls.EvaluationView, errors=NOT_FOUND),
    _r("GET", f"{_V}/calls/{{call_id}}/evaluations/{{version}}", "getEvaluationVersion", "A specific scorecard version (history stays readable)", "calls", [session(R.REVIEWER, P.READ_CALLS)], response=calls.EvaluationView, errors=NOT_FOUND),
    _r("GET", f"{_V}/calls/{{call_id}}/summary", "getSummary", "Current summary", "calls", [session(R.REVIEWER, P.READ_CALLS)], response=calls.SummaryView, errors=NOT_FOUND),
    _r("GET", f"{_V}/calls/{{call_id}}/contact-signals", "getContactSignals", "Current contact signals (may be partial)", "calls", [session(R.REVIEWER, P.READ_CALLS)], response=calls.ContactSignalsView, errors=NOT_FOUND, description="Since 1.3.0 the view may be a v2 result (pipeline v2: categories, subcategories, fields, spans) and carries read-time context: taxonomy_status (outdated stages against the current taxonomy), feedback, alerts (enabled rules matching now), text_withheld and, for admins, comparison_preview_id. Quotes, field values, surface text and evidence are masked on read, and read [REDACTED] while the PII findings are pending."),
    _r("GET", f"{_V}/calls/{{call_id}}/audio", "getCallAudio", "Authorized audio playback (redacted per masking settings), supports Range", "calls", [session(R.REVIEWER, P.PLAY_AUDIO)], response=BINARY, errors=NOT_FOUND, description="Muted over the same values as text reads, including the model PII findings (since 1.2.0). While reviewer reads are masked and the findings for the current transcript revision are not linked, Store serves no audio: 503 `store_unavailable`, retryable, `details.reason` `pii_findings_pending`. Never the unmuted original."),
    _r("POST", f"{_V}/search/semantic", "semanticSearch", "Semantic search over published transcripts (Nemotron-3-Embed-1B)", "calls", [session(R.REVIEWER, P.SEARCH)], request=calls.SemanticSearchQuery, response=calls.SemanticSearchResponse, errors=[E.SEARCH_UNAVAILABLE], description="Since 1.2.0 (decision 18, reversing open question 12 option a): Store embeds the query locally with the same model Process embeds turns with, `nvidia/Nemotron-3-Embed-1B-BF16` at a pinned revision (scheme `nemotron-3-embed-1b@<revision[:7]>`; queries embedded as `query: <text>`, turns as `passage: <text>`), and ranks only turn vectors whose scheme matches (response `embedding_scheme`). This embedder is the only model Store runs, never a remote one. Calls indexed under another scheme (e.g. `hashing-projection-v1`) are not searched; `calls_needing_reembedding` counts them, and reanalysis kind `embeddings` re-embeds one. When the embedder is not installed or fails to load, 503 `search_unavailable` (not retryable; details.reason `not_installed` or `load_failed`)."),

    # --- reviews (human decisions) -------------------------------------------------------
    _r("GET", f"{_V}/calls/{{call_id}}/review", "getReviewState", "Review state: version, escalation, overrides, staleness", "reviews", [session(R.REVIEWER, P.READ_CALLS)], response=reviews.CallReviewState, errors=NOT_FOUND),
    _r("GET", f"{_V}/calls/{{call_id}}/review/history", "listReviewHistory", "Review history", "reviews", [session(R.REVIEWER, P.READ_CALLS)], response=Page[reviews.ReviewHistoryEntry], errors=NOT_FOUND),
    _r("POST", f"{_V}/calls/{{call_id}}/verdicts/{{criterion_id}}", "overrideVerdict", "Override one criterion verdict against the current machine version", "reviews", [session(R.REVIEWER, P.OVERRIDE_VERDICT)], request=reviews.VerdictOverride, response=reviews.ReviewWriteResult, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.REVIEW_VERSION_CONFLICT, E.CONFLICT]),
    _r("POST", f"{_V}/calls/{{call_id}}/escalation", "resolveEscalation", "Supervisor resolution of a pending escalation", "reviews", [session(R.SUPERVISOR, P.RESOLVE_ESCALATION)], request=reviews.EscalationResolution, response=reviews.ReviewWriteResult, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.REVIEW_VERSION_CONFLICT, E.INVALID_TRANSITION, E.CONFLICT]),
    _r("POST", f"{_V}/calls/{{call_id}}/review/retain", "retainReview", "Keep existing decisions despite a newer machine version", "reviews", [session(R.SUPERVISOR, P.RETAIN_REVIEW)], request=reviews.RetainReviewRequest, response=reviews.ReviewWriteResult, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.REVIEW_VERSION_CONFLICT, E.INVALID_TRANSITION]),
    _r("POST", f"{_V}/calls/{{call_id}}/speaker-corrections", "correctSpeaker", "Relabel a turn's speaker; creates a speaker_correction reanalysis request and marks reviews stale", "reviews", [session(R.REVIEWER, P.REQUEST_REANALYSIS)], request=reviews.SpeakerCorrectionRequest, response=jobs.ReanalysisRequest, status_code=201, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.REVIEW_VERSION_CONFLICT]),
    _r("GET", f"{_V}/escalations", "listEscalations", "Calls with pending (or filtered) escalations", "reviews", [session(R.REVIEWER, P.READ_CALLS)], query=reviews.EscalationQuery, response=Page[reviews.EscalationListItem]),

    # --- review queue (human work distribution; never the job queue) ---------------------
    _r("GET", f"{_V}/review-queue", "listReviewQueue", "Review queue items", "review-queue", [session(R.REVIEWER, P.READ_CALLS)], query=reviews.ReviewQueueQuery, response=Page[reviews.ReviewQueueItem]),
    _r("GET", f"{_V}/review-queue/stats", "getReviewQueueStats", "Queue counts by stream and status", "review-queue", [session(R.REVIEWER, P.READ_CALLS)], response=reviews.ReviewQueueStats),
    _r("POST", f"{_V}/review-queue/claim-next", "claimNextReview", "Claim the next unassigned item for the signed-in reviewer", "review-queue", [session(R.REVIEWER, P.CLAIM_REVIEW)], response=reviews.ClaimNextResponse),
    _r("GET", f"{_V}/review-queue/items/{{item_id}}", "getReviewQueueItem", "One queue item", "review-queue", [session(R.REVIEWER, P.READ_CALLS)], response=reviews.ReviewQueueItem, errors=NOT_FOUND),
    _r("POST", f"{_V}/review-queue/items/{{item_id}}/assign", "assignReviewQueueItem", "Assign or unassign a pending item", "review-queue", [session(R.SUPERVISOR, P.ASSIGN_REVIEW)], request=reviews.AssignRequest, response=reviews.ReviewQueueItem, idempotency=Idempotency.EXPECTED_VERSION, errors=[E.NOT_FOUND, E.CONFLICT, E.INVALID_TRANSITION]),
    _r("POST", f"{_V}/review-queue/items/{{item_id}}/start", "startReview", "Move an item to IN_REVIEW for the signed-in reviewer", "review-queue", [session(R.REVIEWER, P.CLAIM_REVIEW)], request=reviews.StartReviewRequest, response=reviews.ReviewQueueItem, idempotency=Idempotency.EXPECTED_VERSION, errors=[E.NOT_FOUND, E.CONFLICT, E.INVALID_TRANSITION], object_rule="An item assigned to someone else can be started only by its assignee."),
    _r("POST", f"{_V}/review-queue/items/{{item_id}}/release", "releaseReview", "Return an item to the pool", "review-queue", [session(R.REVIEWER, P.CLAIM_REVIEW)], request=reviews.ReleaseRequest, response=reviews.ReviewQueueItem, idempotency=Idempotency.EXPECTED_VERSION, errors=[E.NOT_FOUND, E.CONFLICT, E.INVALID_TRANSITION], object_rule="Only the reviewer holding the item, or a holder of resolve_any_review, may release it.", object_permissions=[P.RESOLVE_ANY_REVIEW]),
    _r("POST", f"{_V}/review-queue/items/{{item_id}}/resolve", "resolveReview", "Resolve an item as APPROVED or OVERRIDDEN against the current machine version", "review-queue", [session(R.REVIEWER, P.RESOLVE_OWN_REVIEW)], request=reviews.ResolveRequest, response=reviews.ReviewWriteResult, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.CONFLICT, E.REVIEW_VERSION_CONFLICT, E.INVALID_TRANSITION], object_rule="Resolving an item assigned to another reviewer needs resolve_any_review.", object_permissions=[P.RESOLVE_ANY_REVIEW]),
    _r("GET", f"{_V}/review-queue/rules", "listReviewQueueRules", "Queue rules", "review-queue", [session(R.REVIEWER, P.READ_CALLS)], response=Page[reviews.ReviewQueueRuleRecord]),
    _r("PUT", f"{_V}/review-queue/rules/{{rule_id}}", "saveReviewQueueRule", "Create or replace a queue rule", "review-queue", [session(R.SUPERVISOR, P.MANAGE_QUEUE_RULES)], request=reviews.ReviewQueueRuleSave, response=reviews.ReviewQueueRuleRecord, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.CONFLICT]),
    _r("DELETE", f"{_V}/review-queue/rules/{{rule_id}}", "deleteReviewQueueRule", "Delete a queue rule; existing items keep their rule name", "review-queue", [session(R.SUPERVISOR, P.MANAGE_QUEUE_RULES)], status_code=204, audited=True, errors=NOT_FOUND),

    # --- rubrics ---------------------------------------------------------------------------
    _r("GET", f"{_V}/rubrics", "listRubrics", "Rubrics with their current version", "rubrics", [session(R.REVIEWER, P.READ_CALLS), process(S.JOBS_WRITE)], query=rubrics.RubricListQuery, response=Page[rubrics.RubricSummary]),
    _r("GET", f"{_V}/rubrics/{{rubric_id}}", "getRubric", "The current published version", "rubrics", [session(R.REVIEWER, P.READ_CALLS), process(S.JOBS_WRITE)], response=rubrics.RubricVersion, errors=NOT_FOUND),
    _r("GET", f"{_V}/rubrics/{{rubric_id}}/versions", "listRubricVersions", "Published versions", "rubrics", [session(R.REVIEWER, P.READ_CALLS), process(S.JOBS_WRITE)], response=Page[rubrics.RubricVersion], errors=NOT_FOUND),
    _r("GET", f"{_V}/rubrics/{{rubric_id}}/versions/{{version}}", "getRubricVersion", "One immutable published version, as frozen on QA jobs", "rubrics", [session(R.REVIEWER, P.READ_CALLS), process(S.JOBS_WRITE)], response=rubrics.RubricVersion, errors=NOT_FOUND),
    _r("GET", f"{_V}/rubrics/{{rubric_id}}/draft", "getRubricDraft", "The unpublished draft", "rubrics", [session(R.SUPERVISOR, P.MANAGE_RUBRICS)], response=rubrics.RubricDraft, errors=NOT_FOUND),
    _r("PUT", f"{_V}/rubrics/{{rubric_id}}/draft", "saveRubricDraft", "Create or replace the draft", "rubrics", [session(R.SUPERVISOR, P.MANAGE_RUBRICS)], request=rubrics.RubricDraftSave, response=rubrics.RubricDraft, idempotency=Idempotency.EXPECTED_VERSION, errors=[E.RUBRIC_VERSION_CONFLICT]),
    _r("DELETE", f"{_V}/rubrics/{{rubric_id}}/draft", "discardRubricDraft", "Discard the draft", "rubrics", [session(R.SUPERVISOR, P.MANAGE_RUBRICS)], status_code=204, errors=NOT_FOUND),
    _r("POST", f"{_V}/rubrics/{{rubric_id}}/publish", "publishRubric", "Publish the draft as the next immutable version", "rubrics", [session(R.SUPERVISOR, P.MANAGE_RUBRICS)], request=rubrics.RubricPublishRequest, response=rubrics.RubricVersion, status_code=201, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.RUBRIC_VERSION_CONFLICT, E.VALIDATION_FAILED]),
    _r("POST", f"{_V}/rubrics/{{rubric_id}}/retire", "retireRubric", "Retire the current version; frozen jobs and history keep referencing it", "rubrics", [session(R.SUPERVISOR, P.MANAGE_RUBRICS)], request=rubrics.RubricRetireRequest, response=rubrics.RubricVersion, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.RUBRIC_VERSION_CONFLICT]),

    # --- contact signals v2 (1.3.0; docs/ContactSignalsV2.md section 7.7) ------------------------
    _r("GET", f"{_V}/signals/taxonomy", "getSignalTaxonomy", "The current signal taxonomy version, the signal settings and the record version", "signals", [session(R.REVIEWER, P.READ_CALLS), process(S.JOBS_WRITE)], response=signals.SignalTaxonomyRecord, description="Process reads it at ingest and mints a snapshot of the current version (mintSignalTaxonomySnapshot)."),
    _r("GET", f"{_V}/signals/taxonomy/versions", "listSignalTaxonomyVersions", "Published signal taxonomy versions, newest first", "signals", [session(R.REVIEWER, P.READ_CALLS), process(S.JOBS_WRITE)], response=Page[signals.SignalTaxonomyVersion]),
    _r("GET", f"{_V}/signals/taxonomy/versions/{{version}}", "getSignalTaxonomyVersion", "One immutable published signal taxonomy version", "signals", [session(R.REVIEWER, P.READ_CALLS), process(S.JOBS_WRITE)], response=signals.SignalTaxonomyVersion, errors=NOT_FOUND),
    _r("PUT", f"{_V}/signals/taxonomy", "saveSignalTaxonomy", "Save the whole taxonomy: publish it as the next immutable version (a save that changes nothing returns the record unchanged)", "signals", [session(R.ADMIN, P.MANAGE_SIGNALS)], request=signals.SignalTaxonomySave, response=signals.SignalTaxonomyRecord, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.SIGNAL_TAXONOMY_CONFLICT], object_rule="Store's save validator applies the section 9.6 caps from its effective ContractParameters (signals.signal_taxonomy_cap_violations), the built-in rules, and its rule-based PII detectors over every text path (signals.signal_taxonomy_text_paths); a refusal is validation_failed with details.field naming the path, never the value. A stale expected_record_version is 409 signal_taxonomy_conflict with details.current_version and details.record_version.", description="One document, versioned whole: a save whose taxonomy_digest equals the current version's returns the record unchanged; any other save publishes version N+1. There are no drafts (the preview takes an unsaved taxonomy) and no delete (active false retires a custom node). On a built-in category an admin changes only threshold, subcategory_threshold, subcategories, fields, narrow_quote and examples. Audit signal_taxonomy_saved (version, digest, changed node paths); change event signal_taxonomy saved:v<N>."),
    _r("PUT", f"{_V}/signals/settings", "saveSignalSettings", "Change the signal settings (pipeline v1, shadow or v2; v1 fallback; stage-3 fallback entry)", "signals", [session(R.ADMIN, P.MANAGE_SIGNALS)], request=signals.SignalSettingsSave, response=signals.SignalTaxonomyRecord, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.SIGNAL_TAXONOMY_CONFLICT], description="Shares record_version with the taxonomy. Audit signal_settings_changed (old and new pipeline); change event signal_taxonomy settings. Rolling back to v1 keeps published v2 results."),
    _r("POST", f"{_V}/signals/taxonomy/versions/{{version}}/redaction", "redactSignalTaxonomyText", "Tombstone a non-current version's custom text to numbered [REDACTED <n>] placeholders, keeping its digest", "signals", [session(R.ADMIN, P.MANAGE_SIGNALS)], request=signals.SignalTaxonomyRedaction, response=signals.SignalTaxonomyVersion, idempotency=Idempotency.NATURAL, audited=True, errors=[E.NOT_FOUND, E.CONFLICT], description="Taxonomy text is customer data (section 9.4). Store stores signals.redact_signal_taxonomy_text(taxonomy): each custom text path becomes '[REDACTED <n>]', numbered from 1 in the document order of signals.signal_taxonomy_text_paths, built-in constants kept, so the redacted version still validates. The current version cannot be redacted (409 conflict, details.reason redact_current); the digest in the body must be the version's (409 conflict, details.reason digest_mismatch); a redacted version cannot be minted into a snapshot. Redacting a redacted version returns it. Audit signal_taxonomy_redacted (version, digest)."),
    _r("GET", f"{_V}/signals/alert-rules", "listSignalAlertRules", "Signal alert rules with node_active", "signals", [session(R.REVIEWER, P.READ_CALLS)], response=Page[signals.SignalAlertRuleRecord]),
    _r("PUT", f"{_V}/signals/alert-rules/{{rule_id}}", "saveSignalAlertRule", "Create or replace a signal alert rule (evaluated at read time; editing one runs no model)", "signals", [session(R.ADMIN, P.MANAGE_SIGNALS)], request=signals.SignalAlertRuleSave, response=signals.SignalAlertRuleRecord, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.CONFLICT], object_rule="rule.rule_id equals the path's rule_id. The condition must fit the current taxonomy (signals.signal_alert_condition_problem; validation_failed with details.reason), at most ContractParameters.max_signal_alert_rules rules exist, and the name passes the definition-text detectors. A stale expected_record_version is 409 conflict with details.current_version.", description="Audit signal_alert_rule_saved (rule_id, record_version, enabled); change event signal_alert_rule saved, enabled or disabled. No outbound delivery: alerts feed queue rules, metrics, filters and the change feed only."),
    _r("PUT", f"{_V}/calls/{{call_id}}/signal-hits/{{hit_id}}/feedback", "saveSignalHitFeedback", "Confirm or dismiss a hit's category, and confirm or correct its subcategory", "signals", [session(R.REVIEWER, P.OVERRIDE_VERDICT)], request=signals.SignalHitFeedbackSave, response=signals.SignalHitFeedback, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.NOT_FOUND, E.CONFLICT], object_rule="hit_id must be a hit of the call's current contact_signals version. Store records the subcategory the verdict judged (subcategory_id and its digest) from that hit. A stale expected_feedback_version is 409 conflict with details.current_version.", description="Keyed on (call_id, hit_id), so it survives threshold edits and sibling additions (hit IDs are stable). Unscored: feedback never changes a scorecard or the review version. Audit signal_hit_reviewed (call_id, hit_id, verdicts); change event review with status signal_feedback."),
    _r("POST", f"{_V}/signals/previews", "createSignalPreview", "Test a taxonomy (usually unsaved) on up to 10 calls, into draft slots", "signals", [session(R.ADMIN, P.MANAGE_SIGNALS)], request=signals.SignalPreviewCreate, response=signals.SignalPreview, status_code=201, idempotency=Idempotency.HEADER, audited=True, errors=[E.NOT_FOUND, E.IDEMPOTENCY_KEY_REUSED], object_rule="The taxonomy passes the same caps and text checks as a save (validation_failed with details.field); at most ContractParameters.signal_preview_max_calls calls.", description="Store mints a preview snapshot per call (slot draft:<request_id>:signals:preview) and creates one contact_signals_preview request each (priority +5). Nothing reaches the calls' groups, projections, queue or metrics. Audit signal_preview_requested (preview_id, taxonomy digest, call count)."),
    _r("GET", f"{_V}/signals/previews/{{preview_id}}", "getSignalPreview", "A preview or compare: per-call state, masked result and diff against the published signals", "signals", [session(R.ADMIN, P.MANAGE_SIGNALS)], response=signals.SignalPreview, errors=NOT_FOUND),
    _r("POST", f"{_V}/signals/backfills", "createSignalBackfill", "Update existing calls to the current taxonomy (rescore) or run v2 beside v1 (compare)", "signals", [session(R.ADMIN, P.MANAGE_SIGNALS)], request=signals.SignalBackfillCreate, response=signals.SignalBackfill, status_code=201, idempotency=Idempotency.HEADER, audited=True, errors=[E.IDEMPOTENCY_KEY_REUSED], object_rule="max_calls is at most ContractParameters.signal_backfill_max_calls.", description="rescore creates contact_signals requests (priority -10) that rerun only the outdated stages unless rescore_signals is set; a call with a pending request is widened, not duplicated. compare creates contact_signals_preview requests collected by one preview. Audit signal_backfill_requested (backfill_id, mode, window, requests_created)."),

    # --- metrics ---------------------------------------------------------------------------
    _r("GET", f"{_V}/metrics/executive", "getExecutiveMetrics", "Executive metrics", "metrics", [session(R.REVIEWER, P.READ_METRICS)], query=metrics.MetricsQuery, response=metrics.ExecutiveMetrics),
    _r("GET", f"{_V}/metrics/rubrics/{{rubric_id}}", "getRubricMetrics", "Per-rubric and per-criterion metrics", "metrics", [session(R.REVIEWER, P.READ_METRICS)], query=metrics.MetricsQuery, response=metrics.RubricMetrics, errors=NOT_FOUND),
    _r("GET", f"{_V}/metrics/review-agreement", "getReviewAgreement", "Reviewer agreement, per-criterion precision/recall and automatic-decision coverage", "metrics", [session(R.SUPERVISOR, P.READ_METRICS)], query=metrics.MetricsQuery, response=metrics.ReviewAgreementMetrics),
    _r("GET", f"{_V}/metrics/signals", "getSignalMetrics", "Contact-signal metrics: top caller needs, per-category hit rates and precision, field distributions, alert match rates", "metrics", [session(R.REVIEWER, P.READ_METRICS)], query=metrics.SignalMetricsQuery, response=metrics.SignalMetrics, description="Added in 1.3.0. Aggregate SQL over the projection tables (no text), joined to each call's current signals version; precision is null under 5 judged hits."),

    # --- ASR vocabulary (1.3.0; decision 33, docs/DualAsr.md) ------------------------------------
    _r("GET", f"{_V}/vocabulary", "getAsrVocabulary", "The ASR vocabulary: the installed industry pack, the customer's terms and switches, and the effective terms dual transcription runs with", "vocabulary", [session(R.ADMIN, P.MANAGE_VOCABULARY), process(S.JOBS_WRITE)], response=vocabulary.AsrVocabularyRecord, description="Added in 1.3.0. A singleton document (record_version 0 and an empty vocabulary before the first save or pack install). Process reads it at ingest and at full reanalysis, refreshing on the asr_vocabulary change event, and freezes the effective terms into the asr job's parameters.asr_vocabulary when active is true. Reviewers never need it: corrections reach them on TranscriptView.vocabulary_correction."),
    _r("PUT", f"{_V}/vocabulary", "saveAsrVocabulary", "Replace the vocabulary settings: on/off, the customer's own terms, and the pack terms switched off", "vocabulary", [session(R.ADMIN, P.MANAGE_VOCABULARY)], request=vocabulary.AsrVocabularySave, response=vocabulary.AsrVocabularyRecord, idempotency=Idempotency.EXPECTED_VERSION, audited=True, errors=[E.CONFLICT], object_rule="Every term passes vocabulary.vocabulary_term_problem (Pydantic already refuses a digit), customer_terms holds at most ContractParameters.max_vocabulary_terms terms, every disabled_pack_terms entry is a term of the installed pack (details.reason unknown_pack_term), and Store's rule-based PII detectors find nothing in any term (details.reason pii_detected, details.index the term's position); each refusal is validation_failed with details.field and details.reason, never the term. A stale expected_record_version is 409 conflict with details.current_version.", description="Added in 1.3.0. A save that changes nothing returns the record unchanged. Changes apply to graphs created afterwards; existing transcripts are not re-run (a full reanalysis uses the current vocabulary). Audit asr_vocabulary_saved (record_version, old and new effective digest, enabled, term counts per source; never terms); change event asr_vocabulary saved."),

    # --- on-device training (1.3.0; decision 28, docs/OnDeviceTraining.md section 2) ---------
    _r("GET", f"{_V}/training/labels", "listTrainingLabels", "Reviewer labels (QA overrides, signal feedback, speaker corrections) after a seq cursor, with the source job and artifacts each judged, for on-device training", "training", [process(S.TRAINING_READ)], query=training.TrainingLabelQuery, response=training.TrainingLabelPage, audited=True, description="Added in 1.3.0. Store's append-only label log (results area) holds IDs, enums and versions only: no transcript text, quote, reviewer note, signal note or reviewer identity. Items come in seq order; the newest seq per subject wins and a withdrawn row removes its subject. Store resolves source_job_id and sources at read time (training.TRAINING_SOURCE_ROLES) through the queue area's API; an unresolvable source leaves sources empty. pii_findings is the newest pii_findings artifact for the label's transcript, omitted when there is none; draft-test and preview artifacts are never sources. limit 0 returns count_after and high_water only and is not audited (the Process console polls it); every read with limit > 0 appends training_labels_read (after, next_after, item count and counts per kind; never content). Process fetches the sources with artifacts:read and the source job with jobs:write (getJob). No reviewer-session access."),

    # --- change feed -----------------------------------------------------------------------
    _r("GET", f"{_V}/changes", "listChanges", "Change feed: IDs, versions and statuses after a cursor", "changes", [session(R.REVIEWER, P.READ_CALLS), process(S.CHANGES_READ)], query=events.ChangeFeedQuery, response=events.ChangeFeed, errors=[E.CURSOR_EXPIRED, E.CURSOR_UNKNOWN], object_rule="Store filters kinds by principal (events.CHANGE_KINDS_BY_PRINCIPAL)."),
]


DELIVERY_STAGES: Dict[int, str] = {
    2: "Substrate, real handlers, and reviewer passkey sessions on /store/v1 (localhost secure context)",
    4: "Identity and trust across devices: Call1 CA or customer certificate, fixed hostname, same-hostname restore",
    5: "Separate processes and packaging: updater, egress allowlist",
}
"""Stages of docs/AsyncJobPipelinePlan.md in which Store routes first ship. Stage 3 (cutover) adds no routes."""

STAGE_BY_OPERATION: Dict[str, int] = {
    "getTlsState": 4, "getBackupManifest": 4, "restorePreflight": 4,
    "stageUpdatePackage": 5, "listUpdatePackages": 5, "getUpdatePackage": 5, "commitUpdatePackage": 5,
    "installUpdatePackage": 5, "listUpdaterVerifications": 5, "getEgressAllowlist": 5,
}
"""Routes that ship after Stage 2. Every other route is Stage 2."""

ROUTES = [dataclasses.replace(r, stage=STAGE_BY_OPERATION.get(r.operation_id, 2)) for r in ROUTES]


def routes_by_operation() -> Dict[str, Route]:
    return {route.operation_id: route for route in ROUTES}


# --- FastAPI app (stubs only) -------------------------------------------------------------

_PATH_PARAM = re.compile(r"{(\w+)}")


def _stub(route: Route):
    params: List[inspect.Parameter] = []
    for name in _PATH_PARAM.findall(route.path):
        annotation = int if name in ("version", "attempt_number") else str
        params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation))
    if route.query is not None:
        params.append(inspect.Parameter("query", inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[route.query, Query()]))
    if route.request is not None:
        params.append(inspect.Parameter("body", inspect.Parameter.KEYWORD_ONLY, annotation=route.request))
    if route.idempotency is Idempotency.HEADER:
        params.append(inspect.Parameter("idempotency_key", inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[str, Header(alias="Idempotency-Key")]))

    async def endpoint(**kwargs):  # pragma: no cover - never called; the app only emits OpenAPI
        raise HTTPException(status_code=501, detail=ErrorResponse(code=ErrorCode.NOT_IMPLEMENTED, message="contract stub").model_dump())

    endpoint.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    endpoint.__name__ = route.operation_id
    return endpoint


def _principal_doc(rule: PrincipalRule) -> Dict[str, Any]:
    doc: Dict[str, Any] = {"kind": rule.kind.value}
    if rule.scope:
        doc["scope"] = rule.scope.value
    if rule.min_role:
        doc["min_role"] = rule.min_role.value
    if rule.permission:
        doc["permission"] = rule.permission.value
    return doc


def _error_responses(route: Route) -> Dict[int, Dict[str, Any]]:
    by_status: Dict[int, List[str]] = {}
    for code in route.all_errors():
        by_status.setdefault(ERROR_HTTP_STATUS[code], []).append(code.value)
    return {status: {"model": ErrorResponse, "description": "codes: " + ", ".join(sorted(codes))} for status, codes in sorted(by_status.items())}


def build_app() -> FastAPI:
    app = FastAPI(title="Call1 Store API", version=CONTRACT_VERSION, description="The frozen Store contract shared by Process, Store and Evaluate. Stub handlers only.")
    for route in ROUTES:
        responses = _error_responses(route)
        response_model: Any = None
        if route.response is BINARY:
            responses[route.status_code] = {"content": {"application/octet-stream": {}}, "description": "Byte stream"}
        elif route.response is CSV:
            responses[route.status_code] = {"content": {"text/csv": {}}, "description": "RFC 4180 CSV, UTF-8, header line (usage.USAGE_CSV_COLUMNS)"}
        elif route.response is not None:
            response_model = route.response
        app.add_api_route(
            route.path,
            _stub(route),
            methods=[route.method],
            response_model=response_model,
            status_code=route.status_code,
            operation_id=route.operation_id,
            summary=route.summary,
            description=route.description or None,
            tags=[route.tag],
            responses=responses,
            openapi_extra={
                "x-call1-principals": [_principal_doc(p) for p in route.principals],
                "x-call1-idempotency": route.idempotency.value,
                "x-call1-audited": route.audited,
                "x-call1-errors": [c.value for c in route.all_errors()],
                "x-call1-stage": route.stage,
                **({"x-call1-object-rule": {"rule": route.object_rule, "permissions": [p.value for p in route.object_permissions]}} if route.object_rule else {}),
            },
        )
    return app


def contract_extensions() -> Dict[str, Any]:
    """Contract-level data embedded in the OpenAPI document under ``x-call1``."""
    return {
        "contract_version": CONTRACT_VERSION,
        "api_prefix": STORE_API_PREFIX,
        "delivery_stages": {str(k): v for k, v in DELIVERY_STAGES.items()},
        "generator": dict(GENERATOR_PINS),
        "canonical_json": CANONICAL_JSON,
        "parameters": CONTRACT_PARAMETERS.model_dump(),
        "job_statuses": [s.value for s in jobs.JobStatus],
        "reserved_job_statuses": sorted(s.value for s in jobs.RESERVED_STATUSES),
        "job_transitions": [t.model_dump(mode="json") for t in jobs.JOB_TRANSITIONS],
        "completion_transaction_steps": list(jobs.COMPLETION_TRANSACTION_STEPS),
        "release_requeue_codes": sorted(c.value for c in jobs.RELEASE_REQUEUE_CODES),
        "release_reject_codes": sorted(c.value for c in jobs.RELEASE_REJECT_CODES),
        "reanalysis_transitions": [t.model_dump(mode="json") for t in jobs.REANALYSIS_TRANSITIONS],
        "reanalysis_kind_affects": {k.value: sorted(g.value for g in v) for k, v in jobs.REANALYSIS_KIND_AFFECTS.items()},
        "review_queue_transitions": [t.model_dump(mode="json") for t in reviews.REVIEW_QUEUE_TRANSITIONS],
        "pro1_connection_transitions": [t.model_dump(mode="json") for t in custody.PRO1_CONNECTION_TRANSITIONS],
        "key_release_preconditions": [p.model_dump(mode="json") for p in custody.KEY_RELEASE_PRECONDITIONS],
        "job_error_classes": {code.value: cls.value for code, cls in JOB_ERROR_CLASSES.items()},
        "provider_failure_codes": sorted(c.value for c in PROVIDER_FAILURE_CODES),
        "pro1_attestation_class_codes": sorted(c.value for c in PRO1_ATTESTATION_CLASS_CODES),
        "pro1_provider_failure_codes": sorted(c.value for c in PRO1_PROVIDER_FAILURE_CODES),
        "pro1_interruption_codes": sorted(c.value for c in custody.PRO1_INTERRUPTION_CODES),
        "route_class_rules": {rc.value: rule.model_dump(mode="json") for rc, rule in custody.ROUTE_CLASS_RULES.items()},
        "call1_operated_domains": list(custody.CALL1_OPERATED_DOMAINS),
        "role_permissions": {role.value: sorted(p.value for p in perms) for role, perms in auth.ROLE_PERMISSIONS.items()},
        "service_scopes": [s.value for s in ServiceScope],
        "session_cookie": auth.SESSION_COOKIE.model_dump(),
        "one_time_secret_fields": sorted(f"{m}.{f}" for m, f in auth.ONE_TIME_SECRET_FIELDS),
        "session_bound_secret_fields": sorted(f"{m}.{f}" for m, f in auth.SESSION_BOUND_SECRET_FIELDS),
        "console_credential": {"store_principal": False, "loopback_only": True, "operations": [o.value for o in auth.ConsoleOperation]},
        "change_kinds_by_principal": {k: [c.value for c in v] for k, v in events.CHANGE_KINDS_BY_PRINCIPAL.items()},
        "artifact_content_contracts": {k.value: v for k, v in artifacts.ARTIFACT_CONTENT_CONTRACTS.items()},
        "artifact_content_schemas": {contract: (model.__name__ if model else None) for contract, model in artifacts.ARTIFACT_CONTENT_MODELS.items()},
        "global_artifact_kinds": sorted(k.value for k in artifacts.GLOBAL_ARTIFACT_KINDS),
        "store_minted_artifact_kinds": sorted(k.value for k in artifacts.STORE_MINTED_KINDS),
        "draft_test_slot_prefix": artifacts.DRAFT_TEST_SLOT_PREFIX,
        "slot_max_length": artifacts.SLOT_MAX_LENGTH,
        "rubric_input_role": rubrics.RUBRIC_INPUT_ROLE,
        "signal_taxonomy_input_role": signals.SIGNAL_TAXONOMY_INPUT_ROLE,
        "draft_test_kinds": sorted(k.value for k in jobs.DRAFT_TEST_KINDS),
        "signal_reanalysis_kinds": sorted(k.value for k in jobs.SIGNAL_REANALYSIS_KINDS),
        "signal_stages": list(signals.SIGNAL_STAGES),
        "signal_node_id_pattern": signals.SIGNAL_NODE_ID_PATTERN,
        "asr_vocabulary": {
            "term_max_chars": vocabulary.VOCABULARY_TERM_MAX_CHARS,
            "term_max_words": vocabulary.VOCABULARY_TERM_MAX_WORDS,
            "term_punctuation": "".join(sorted(vocabulary.VOCABULARY_TERM_PUNCTUATION)),
            "pack_id_pattern": vocabulary.ASR_VOCABULARY_PACK_ID_PATTERN,
            "merge_rule": vocabulary.VocabularyMergeRule().model_dump(mode="json"),
            "asr_optional_output_roles": [jobs.ASR_BASE_TRANSCRIPT_ROLE, jobs.ASR_VOCABULARY_PASS_ROLE],
        },
        "training_source_roles": {k.value: {"required": sorted(req), "optional": sorted(opt)} for k, (req, opt) in training.TRAINING_SOURCE_ROLES.items()},
        "training_judged_artifact_kinds": {k.value: v.value for k, v in training.TRAINING_JUDGED_ARTIFACT_KIND.items()},
        "excluded_qa_reason_codes": sorted(c.value for c in training.EXCLUDED_QA_REASON_CODES),
        "builtin_signal_categories": {cid: {"name": b.name, "gloss": b.gloss, "speaker": b.speaker.value} for cid, b in signals.BUILTIN_SIGNAL_CATEGORIES.items()},
        "builtin_signal_editable_fields": list(signals.BUILTIN_EDITABLE_FIELDS),
        "reserved_signal_subcategory_ids": sorted(signals.RESERVED_SUBCATEGORY_IDS),
        "reserved_signal_field_ids": sorted(signals.RESERVED_FIELD_IDS),
        "forbidden_field_pii_classes": sorted(c.value for c in signals.FORBIDDEN_FIELD_PII_CLASSES),
        "always_masked_purposes": sorted(p.value for p in catalog.ALWAYS_MASKED_PURPOSES),
        "job_type_rules": {jt.value: rule.model_dump(mode="json") for jt, rule in jobs.JOB_TYPE_RULES.items()},
        "usage_csv_columns": list(usage.USAGE_CSV_COLUMNS),
        "error_http_status": {code.value: status for code, status in ERROR_HTTP_STATUS.items()},
        "job_error_codes": [c.value for c in JobErrorCode],
    }


def content_schemas() -> Dict[str, Any]:
    """JSON Schemas of every artifact content model (serialization mode, like responses)."""
    models = sorted({m for m in artifacts.ARTIFACT_CONTENT_MODELS.values() if m is not None}, key=lambda m: m.__name__)
    _, top = models_json_schema([(m, "serialization") for m in models], ref_template="#/components/schemas/{model}")
    return top.get("$defs", {})


def _without_null_defaults(value: Any) -> Any:
    """FastAPI drops ``"default": null`` from route schemas; do the same before comparing."""
    if isinstance(value, dict):
        return {k: _without_null_defaults(v) for k, v in value.items() if not (k == "default" and v is None)}
    if isinstance(value, list):
        return [_without_null_defaults(v) for v in value]
    return value


def build_openapi() -> Dict[str, Any]:
    document = build_app().openapi()
    schemas = document.setdefault("components", {}).setdefault("schemas", {})
    for name, schema in content_schemas().items():
        schema = _without_null_defaults(schema)
        if name in schemas and _without_null_defaults(schemas[name]) != schema:
            raise RuntimeError(f"content schema {name} differs from the route schema of the same name")
        schemas.setdefault(name, schema)
    document["components"]["schemas"] = dict(sorted(schemas.items()))
    document["x-call1"] = contract_extensions()
    return document
