"""Shared building blocks of the Store contract.

Everything here is used by more than one contract area: the strict base model, identifier
and digest types, timestamps, pagination, change cursors, the principal vocabulary and the
named timing parameters that lease, session and retry semantics refer to.

Nothing in this module (or anywhere in ``call1.contracts``) may carry a credential value,
a filesystem path, or an assumption that Store and Process share a host. Tests enforce it.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Generic, List, Optional, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, field_validator

CONTRACT_VERSION = "1.4.0"
"""Semantic version of the Store contract. See README "Versioning and change rules"."""

STORE_API_PREFIX = "/store/v1"
"""Every Store route lives under this prefix. The major segment moves only with a breaking change."""


class ContractModel(BaseModel):
    """Base for every contract model: unknown fields are rejected so drift is loud, not silent.

    ``json_schema_serialization_defaults_required`` makes every field Store always sends appear as
    required in response schemas, so generated client types need no casts (for example the
    WebAuthn options handed to ``@simplewebauthn/browser``). A model used both in requests and in
    responses therefore appears twice in OpenAPI, as ``Name-Input`` and ``Name-Output``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, validate_assignment=True, json_schema_serialization_defaults_required=True)


# --- Scalar types -------------------------------------------------------------------------

ResourceId = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"),
]
"""Opaque identifier minted by Store (``conv_``, ``call_``, ``art_``, ``job_``, ``att_``, ``acct_``,
``cred_``, ``inv_``, ``sess_``, ``key_``, ``evt_``, ``apr_``, ``kr_``, ``rq_``, ``rvw_`` prefixes are
conventional, never parsed). Clients treat every ID as opaque text."""

Sha256Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
"""Lower-case hex SHA-256 with an explicit algorithm prefix. The only checksum form in the contract."""

EnvVarName = Annotated[str, StringConstraints(pattern=r"^CALL1_[A-Z0-9_]+$")]
"""The name of a server environment variable. Credentials are referenced by name, never by value."""

ModelCredentialEnvName = Annotated[str, StringConstraints(pattern=r"^CALL1_MODEL_KEY_[A-Z0-9_]+$")]
"""Process-side model credential reference (BYOK or customer-LAN proxy). Never stored in Store as a value."""

SafeText = Annotated[str, StringConstraints(max_length=2000)]
"""Bounded human-readable text that must never contain transcript content, prompts, or secrets."""

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=200)]

IdempotencyKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._:-]{8,128}$")]
"""Client-chosen key that makes a create/complete request replay-safe. Scoped per principal and route."""

ChangeCursor = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
"""Opaque position in Store's change feed, totally ordered within one feed epoch. It encodes the
epoch, so a cursor from before a restore is recognisably foreign (410 ``cursor_unknown``). Compare
only through Store. Every cursor-valued field in the contract uses this type."""

HexDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64,128}$")]
"""Lower-case hex of a platform measurement or key digest whose algorithm the platform fixes
(SHA-256 or SHA-384), e.g. an SEV-SNP ID-key digest or a TDX MRCONFIGID. Not a checksum."""

PageToken = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,256}$")]
"""Opaque continuation token for list endpoints."""

Timestamp = AwareDatetime
"""RFC 3339 timestamps with an explicit offset; Store emits UTC (``Z``)."""

JsonScalar = str | int | float | bool | None

_DNS_LABEL = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")
CALL1_DOMAINS = ("call1.cc",)
"""Domains Call1 owns. A Store hostname is never under one (HostedV2.md 3.8), and only the attested
Pro1 route may name one as a destination (Stage 0 guard, ``is_call1_operated``)."""


def validate_store_hostname(value: str) -> str:
    """The fixed-hostname rule (plan, "Transport and hostname").

    The Store hostname is a name in the customer's own DNS zone, fixed at install. It is never
    an IP address, never ``localhost``, never a single label, never an mDNS ``.local`` name and
    never on a Call1-owned domain. It is the WebAuthn relying-party ID, so it cannot change
    without re-inviting every reviewer.
    """
    host = value.strip().rstrip(".").lower()
    if not host:
        raise ValueError("Store hostname is required")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("Store hostname must be a DNS name, not an IP address")
    labels = host.split(".")
    if len(labels) < 2:
        raise ValueError("Store hostname must be a fully qualified name in the customer's DNS zone")
    if host == "localhost" or labels[-1] in ("localhost", "local", "localdomain", "home", "arpa"):
        raise ValueError("Store hostname cannot be localhost, an mDNS .local name or a special-use domain")
    if any(host == d or host.endswith("." + d) for d in CALL1_DOMAINS):
        raise ValueError("Store hostname cannot be on a Call1-owned domain")
    if len(host) > 253 or not all(_DNS_LABEL.match(label) for label in labels):
        raise ValueError("Store hostname is not a valid DNS name")
    return host


StoreHostname = Annotated[str, Field(description="Fixed Store hostname in the customer's DNS zone (see validate_store_hostname).")]


# --- Principals ---------------------------------------------------------------------------


class PrincipalKind(str, Enum):
    """Who can call a Store route. The route table names one or more per route."""

    PROCESS_SERVICE_KEY = "process_service_key"
    """A Process installation, authenticated by a scoped service key (``Authorization: Bearer``)."""
    REVIEWER_SESSION = "reviewer_session"
    """A signed-in reviewer, authenticated by a server-side Store session cookie plus CSRF header."""
    CONSOLE_CREDENTIAL = "console_credential"
    """The native console's loopback-only credential. It authenticates to Process, never to Store;
    no Store route accepts it (tests enforce this)."""
    ANONYMOUS = "anonymous"
    """No credential. Only health and the first step of a code- or invitation-bearing ceremony."""


class ServiceScope(str, Enum):
    """Scopes a Process service key may hold. A key holds a subset; routes require one scope each."""

    CALLS_WRITE = "calls:write"
    ARTIFACTS_READ = "artifacts:read"
    ARTIFACTS_WRITE = "artifacts:write"
    JOBS_CLAIM = "jobs:claim"
    JOBS_WRITE = "jobs:write"
    JOBS_CONTROL = "jobs:control"
    USAGE_READ = "usage:read"
    """Aggregates only (per-route medians for Process Models settings); never raw rows of other data."""
    HARDWARE_WRITE = "hardware:write"
    REANALYSIS_CLAIM = "reanalysis:claim"
    KEY_RELEASE_WRITE = "key-release:write"
    RELEASE_TRUST_WRITE = "release-trust:write"
    ADMIN_STATE_READ = "admin-state:read"
    CATALOG_PUBLISH = "catalog:publish"
    CHANGES_READ = "changes:read"
    TRAINING_READ = "training:read"
    """Added in 1.3.0 (decision 28): reviewer labels and the artifact IDs they judged, for on-device
    training; IDs, enums and versions only (``listTrainingLabels``)."""


class ReviewerRole(str, Enum):
    """Roles a reviewer account holds. Higher roles include every permission of lower ones."""

    REVIEWER = "reviewer"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"


ROLE_ORDER = {ReviewerRole.REVIEWER: 0, ReviewerRole.SUPERVISOR: 1, ReviewerRole.ADMIN: 2}


def role_satisfies(role: ReviewerRole, minimum: ReviewerRole) -> bool:
    return ROLE_ORDER[role] >= ROLE_ORDER[minimum]


# --- Pagination and cursors ---------------------------------------------------------------

T = TypeVar("T")


class PageQuery(ContractModel):
    """Query parameters shared by list endpoints."""

    limit: int = Field(default=50, ge=1, le=200, description="Maximum items to return.")
    page_token: Optional[PageToken] = Field(default=None, description="Continuation token from a previous page.")


class Page(ContractModel, Generic[T]):
    items: List[T]
    next_page_token: Optional[PageToken] = Field(default=None, description="Absent on the last page.")


# --- Named timing parameters ---------------------------------------------------------------


class ContractParameters(ContractModel):
    """Named parameters the contract's semantics refer to. Values are seconds unless stated.

    Store owns the effective values and reports them at ``GET /store/v1/status``. Process and
    Evaluate read them from there rather than hardcoding. Defaults below are the v1 defaults.
    """

    lease_duration_seconds: int = Field(default=300, ge=30, description="LEASE_DURATION: how long a claim is valid without a heartbeat.")
    heartbeat_interval_seconds: int = Field(default=100, ge=5, description="HEARTBEAT_INTERVAL: recommended renewal cadence (about LEASE_DURATION / 3).")
    lease_grace_seconds: int = Field(default=30, ge=0, description="LEASE_GRACE: slack Store allows after expiry before it requeues the job.")
    max_claim_batch: int = Field(default=16, ge=1, le=64, description="MAX_CLAIM_BATCH: most jobs one claim call may return.")
    default_max_attempts: int = Field(default=3, ge=1, description="DEFAULT_MAX_ATTEMPTS: inference executions per job before terminal failure.")
    retry_attempt_grant: int = Field(default=1, ge=1, description="RETRY_ATTEMPT_GRANT: attempts added by one manual retry.")
    retry_backoff_base_seconds: int = Field(default=30, ge=1, description="RETRY_BACKOFF_BASE: first automatic retry delay.")
    retry_backoff_factor: float = Field(default=2.0, ge=1.0, description="RETRY_BACKOFF_FACTOR: multiplier per retry.")
    retry_backoff_max_seconds: int = Field(default=3600, ge=1, description="RETRY_BACKOFF_MAX: ceiling on the automatic delay.")
    reanalysis_claim_lease_seconds: int = Field(default=300, ge=30, description="REANALYSIS_CLAIM_LEASE: how long Process has to turn a claimed reanalysis request into a graph.")
    upload_grant_lifetime_seconds: int = Field(default=900, ge=60, description="UPLOAD_GRANT_LIFETIME: validity of a Store-issued object upload URL.")
    orphan_artifact_retention_seconds: int = Field(default=24 * 3600, ge=900, description="ORPHAN_ARTIFACT_RETENTION: how long a committed artifact that no completion ever linked is kept before Store deletes it.")
    download_grant_lifetime_seconds: int = Field(default=300, ge=30, description="DOWNLOAD_GRANT_LIFETIME: validity of a Store-issued object download URL.")
    inline_artifact_max_bytes: int = Field(default=1_048_576, ge=1024, description="INLINE_ARTIFACT_MAX_BYTES: largest JSON artifact carried in the API body instead of object storage.")
    change_feed_retention_seconds: int = Field(default=7 * 24 * 3600, ge=3600, description="CHANGE_FEED_RETENTION: how long a change cursor stays resumable.")
    session_idle_lifetime_seconds: int = Field(default=8 * 3600, ge=300, description="SESSION_IDLE_LIFETIME: reviewer session idle timeout.")
    session_absolute_lifetime_seconds: int = Field(default=12 * 3600, ge=600, description="SESSION_ABSOLUTE_LIFETIME: reviewer session hard cap.")
    webauthn_challenge_lifetime_seconds: int = Field(default=120, ge=30, le=600, description="CHALLENGE_LIFETIME: WebAuthn challenge validity; single use.")
    invitation_lifetime_seconds: int = Field(default=7 * 24 * 3600, ge=3600, description="INVITATION_LIFETIME: reviewer invitation validity; single use.")
    setup_code_lifetime_seconds: int = Field(default=1800, ge=300, description="SETUP_CODE_LIFETIME: first-admin and break-glass code validity; single use.")
    service_key_rotation_grace_seconds: int = Field(default=24 * 3600, ge=0, description="SERVICE_KEY_ROTATION_GRACE: how long the previous key keeps working after rotation.")
    # The Pro1 session lifetime is not a contract parameter: it is AttestationPolicy.session_lifetime_seconds,
    # customer admin state covered by the policy digest.

    # Contact Signals v2 caps (1.3.0; docs/ContactSignalsV2.md section 9.6). Store's taxonomy save
    # validator reads them (signals.signal_taxonomy_cap_violations), so S0 can tighten a cap by
    # changing a value with no request-shape change. The Pydantic bounds on the taxonomy models are
    # fixed outer ceilings, which is why each `le` below stops at the ceiling.
    max_custom_signal_categories: int = Field(default=8, ge=0, le=16, description="Active custom top-level signal categories (at most 16 custom in total, active or not). A segment then sees at most 14 stage-1 options (5 built-ins + 8 + none).")
    max_active_subcategories: int = Field(default=12, ge=0, le=12, description="Active subcategories per category: stage 2 then has at most 14 options (12 + Other + Not).")
    max_fields_per_path: int = Field(default=12, ge=0, le=16, description="Extraction fields on one category + subcategory path (Needle's schema budget and the Gemma batcher's output budget).")
    max_option_gloss_chars: int = Field(default=40, ge=8, le=80, description="Option text (gloss) of custom categories and of every subcategory; the contract field allows 80 as an outer ceiling. Built-in glosses are Call1's.")
    max_signal_alert_rules: int = Field(default=50, ge=0, description="Signal alert rules (SQL evaluation cost on read).")
    signal_preview_max_calls: int = Field(default=10, ge=1, le=10, description="Calls in one taxonomy preview (priority +5).")
    signal_backfill_max_calls: int = Field(default=500, ge=1, le=500, description="Calls one signal backfill may request (priority -10).")
    max_extraction_spans_per_call: int = Field(default=24, ge=1, description="Spans stage 3 extracts per call; spans past the cap stay categorized without fields and the result is partial (extraction_cap).")
    # Contact Signals rules-engine caps (1.4.0; docs/SignalsEmbeddings.md section 9.3). Store's
    # taxonomy save validator reads them; the Pydantic bounds on the recipe models are the ceilings.
    max_signal_recipe_rules: int = Field(default=8, ge=1, le=16, description="Rules in one category recipe's filter (1.4.0).")
    max_signal_lexicon_phrases: int = Field(default=24, ge=1, le=64, description="Phrases in one lexicon or phrase rule (1.4.0).")

    # ASR vocabulary caps (1.3.0, decision 33; docs/DualAsr.md). Store's vocabulary save and pack
    # install read them; the Pydantic list bound (vocabulary.VOCABULARY_LIST_CEILING) is the ceiling.
    max_vocabulary_terms: int = Field(default=500, ge=0, le=2000, description="The customer's own vocabulary terms (AsrVocabularySettings.customer_terms).")
    max_vocabulary_pack_terms: int = Field(default=2000, ge=0, le=2000, description="Terms in an installed industry vocabulary pack.")


CONTRACT_PARAMETERS = ContractParameters()


class ContractInfo(ContractModel):
    """Reported by ``GET /store/v1/status`` so clients can check compatibility before doing work."""

    contract_version: str = Field(description="Semver of the contract this Store implements.")
    parameters: ContractParameters


class TimeRange(ContractModel):
    start: Timestamp
    end: Timestamp

    @field_validator("end")
    @classmethod
    def _ordered(cls, end: datetime, info):
        start = info.data.get("start")
        if start is not None and end < start:
            raise ValueError("end must not precede start")
        return end


class ArtifactRef(ContractModel):
    """How one record points at an artifact: its ID plus the checksum the referrer saw."""

    artifact_id: ResourceId
    checksum: Sha256Digest


# --- Canonical JSON and digests ------------------------------------------------------------

CANONICAL_JSON = "RFC 8785 (JCS)"
"""The one canonical JSON form in the contract. Every ``Sha256Digest`` computed over JSON (inline
artifact checksums, rubric digests, the attestation-policy digest recorded on every Pro1 attempt,
hardware fingerprints, audit-event digests, request payload digests for replay) is
``canonical_digest`` of that JSON. For a contract model the JSON is ``model_dump(mode="json")``
with defaults and explicit nulls included. TypeScript clients use any RFC 8785 implementation."""


def _jcs_number(value: Any) -> str:
    if isinstance(value, bool):  # pragma: no cover - handled by the caller
        raise TypeError("bool is not a number")
    if isinstance(value, int):
        if abs(value) > 2 ** 53:
            raise ValueError("integers beyond 2^53 have no exact canonical JSON form; send them as strings")
        return str(value)
    if not math.isfinite(value):
        raise ValueError("NaN and infinities have no JSON form")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    text = repr(abs(value))  # shortest round-trip digits, as ECMAScript Number::toString uses
    mantissa, _, exp = text.partition("e")
    integer, _, fraction = mantissa.partition(".")
    digits = (integer + fraction).lstrip("0")
    exponent = (int(exp) if exp else 0) - len(fraction)
    stripped = digits.rstrip("0")
    exponent += len(digits) - len(stripped)
    digits = stripped
    k = len(digits)
    n = k + exponent  # value = 0.digits x 10^n
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        body = (digits[0] + ("." + digits[1:] if k > 1 else "")) + "e" + ("+" if e >= 0 else "-") + str(abs(e))
    return sign + body


def _jcs(value: Any, out: List[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, (int, float)):
        out.append(_jcs_number(value))
    elif isinstance(value, str):
        value.encode("utf-8")  # rejects lone surrogates
        out.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _jcs(item, out)
        out.append("]")
    elif isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("canonical JSON object keys are strings")
        out.append("{")
        for i, key in enumerate(sorted(value, key=lambda k: k.encode("utf-16-be"))):
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _jcs(value[key], out)
        out.append("}")
    else:
        raise TypeError(f"{type(value).__name__} has no canonical JSON form; dump the model with mode='json' first")


def canonical_json(value: Any) -> bytes:
    """RFC 8785 canonical UTF-8 bytes of a JSON value or a contract model."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    out: List[str] = []
    _jcs(value, out)
    return "".join(out).encode("utf-8")


def canonical_digest(value: Any) -> str:
    """``sha256:<hex>`` of ``canonical_json(value)``. The only way the contract hashes JSON."""
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()

