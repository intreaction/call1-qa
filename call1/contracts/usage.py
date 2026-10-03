"""Usage records, hardware profiles, the usage report and the admin price table
(plan, "Usage and cost capture").

Every attempt (a claim that was not released before inference started), ML or LLM, successful or
not, has exactly one usage record: written through the completion or failure call that ends it, or
synthesized by Store with outcome ``abandoned`` when the attempt's lease expires. A late worker may
replace a synthesized row's measurements once (``POST /jobs/{id}/attempts/{n}/usage``). Records
hold measurements only: no prompts, responses, transcript text, credentials or dollar amounts. They
are customer data in the customer's Store and are never sent to Call1.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from .catalog import CatalogEntryRef, ModelPurpose
from .common import ContractModel, PageQuery, ResourceId, SafeText, Sha256Digest, ShortText, Timestamp, canonical_digest
from .custody import RouteClass
from .errors import JobErrorCode


class HardwareProfileKind(str, Enum):
    PROCESS_HOST = "process_host"
    CUSTOMER_LAN_HOST = "customer_lan_host"
    ATTESTED_ENVIRONMENT = "attested_environment"


class HardwareProfileSource(str, Enum):
    MEASURED = "measured"
    DECLARED = "declared"
    ATTESTED = "attested"


HARDWARE_SOURCE_FOR_KIND = {
    HardwareProfileKind.PROCESS_HOST: HardwareProfileSource.MEASURED,
    HardwareProfileKind.CUSTOMER_LAN_HOST: HardwareProfileSource.DECLARED,
    HardwareProfileKind.ATTESTED_ENVIRONMENT: HardwareProfileSource.ATTESTED,
}


class HardwareProfileFields(ContractModel):
    """The descriptive fields of a hardware profile; its fingerprint is canonical_digest of these."""

    kind: HardwareProfileKind
    source: HardwareProfileSource
    chip: ShortText = Field(description="CPU or SoC, e.g. 'Apple M4 Pro' or 'AMD EPYC 9V84 (SEV-SNP)'.")
    accelerator: Optional[ShortText] = Field(default=None, description="GPU or NPU when separate from the chip, e.g. 'NVIDIA H100 (CC mode)'.")
    memory_bytes: int = Field(ge=0)
    os_name: ShortText
    os_version: ShortText
    runtime_versions: Dict[str, str] = Field(default_factory=dict, description="Runtime name to version, e.g. mlx, ollama, torch, driver.")

    @model_validator(mode="after")
    def _source_matches_kind(self):
        if HARDWARE_SOURCE_FOR_KIND[self.kind] is not self.source:
            raise ValueError(f"a {self.kind.value} profile is {HARDWARE_SOURCE_FOR_KIND[self.kind].value}")
        return self


def hardware_fingerprint(fields: HardwareProfileFields) -> str:
    """canonical_digest of a profile's descriptive fields."""
    return canonical_digest(HardwareProfileFields.model_validate({k: getattr(fields, k) for k in HardwareProfileFields.model_fields}))


class HardwareProfileInput(HardwareProfileFields):
    fingerprint: Sha256Digest = Field(description="hardware_fingerprint(): canonical_digest of the HardwareProfileFields; Store upserts by fingerprint.")

    @model_validator(mode="after")
    def _fingerprint(self):
        if self.fingerprint != hardware_fingerprint(self):
            raise ValueError("fingerprint is canonical_digest of the profile fields")
        return self


class HardwareProfile(HardwareProfileInput):
    id: ResourceId
    first_seen_at: Timestamp
    last_seen_at: Timestamp


class TokenSource(str, Enum):
    PROVIDER_REPORTED = "provider_reported"
    LOCAL_TOKENIZER = "local_tokenizer"
    UNAVAILABLE = "unavailable"


class TokenCount(ContractModel):
    count: Optional[int] = Field(default=None, ge=0)
    source: TokenSource

    @model_validator(mode="after")
    def _unavailable_has_no_count(self):
        if (self.source is TokenSource.UNAVAILABLE) != (self.count is None):
            raise ValueError("count is present exactly when the source is not 'unavailable'; never estimate silently")
        return self


class UsageOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    VALIDATION_REJECTED = "validation_rejected"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"
    """Store-synthesized: the attempt's lease expired with no completion or failure call."""


class UsageRecordedBy(str, Enum):
    PROCESS = "process"
    STORE_SYNTHESIZED = "store_synthesized"
    """Lease expiry: tokens unavailable, durations from the lease, hardware from the claim."""
    PROCESS_LATE = "process_late"
    """A late worker replaced a synthesized row with its measurements; the outcome stays abandoned."""


class BillingUnit(ContractModel):
    """Units a provider returned with its response, if any. Never a dollar amount."""

    unit: ShortText
    quantity: float = Field(ge=0)


class UsageRecordInput(ContractModel):
    """What Process supplies with completion or failure. Store adds the identity fields."""

    tokens_input: TokenCount
    tokens_output: TokenCount
    queue_wait_seconds: Optional[float] = Field(default=None, ge=0, description="Release to claim; Store fills it when Process omits it.")
    slot_wait_seconds: float = Field(default=0, ge=0, description="Local slot wait before execution.")
    model_load_seconds: Optional[float] = Field(default=None, ge=0, description="Where the runtime reports it.")
    inference_seconds: float = Field(ge=0)
    total_seconds: float = Field(ge=0)
    session_setup_seconds: Optional[float] = Field(default=None, ge=0, description="Pro1: evidence, verification and key release, charged to the first attempt of the session.")
    session_id: Optional[ResourceId] = Field(default=None, description="Pro1 attested session, so setup cost can be amortized.")
    audio_seconds_processed: Optional[float] = Field(default=None, ge=0, description="ML stages record audio seconds processed.")
    peak_memory_bytes: Optional[int] = Field(default=None, ge=0)
    hardware_profile_id: ResourceId
    outcome: UsageOutcome
    error_code: Optional[JobErrorCode] = None
    billing_units: List[BillingUnit] = Field(default_factory=list)
    provider_reported_model_id: Optional[ShortText] = None

    @model_validator(mode="after")
    def _outcome_error_consistency(self):
        if self.outcome is UsageOutcome.SUCCEEDED and self.error_code is not None:
            raise ValueError("a succeeded attempt has no error code")
        if self.outcome in (UsageOutcome.FAILED, UsageOutcome.VALIDATION_REJECTED, UsageOutcome.CANCELLED) and self.error_code is None:
            raise ValueError("a failed, rejected or cancelled attempt names its safe error code")
        if self.outcome is UsageOutcome.VALIDATION_REJECTED and self.error_code is not JobErrorCode.VALIDATION_REJECTED:
            raise ValueError("a validation_rejected outcome carries the validation_rejected code")
        if self.outcome is UsageOutcome.CANCELLED and self.error_code is not JobErrorCode.CANCELLED:
            raise ValueError("a cancelled outcome carries the cancelled code")
        if self.outcome is UsageOutcome.FAILED and self.error_code in (JobErrorCode.VALIDATION_REJECTED, JobErrorCode.CANCELLED):
            raise ValueError("validation_rejected and cancelled have their own outcomes")
        if self.outcome is UsageOutcome.ABANDONED:
            raise ValueError("abandoned rows are synthesized by Store, never submitted")
        return self


def usage_outcome_for(error_code: Optional[JobErrorCode]) -> UsageOutcome:
    """The usage outcome a failure with this code records."""
    if error_code is None:
        return UsageOutcome.SUCCEEDED
    if error_code is JobErrorCode.VALIDATION_REJECTED:
        return UsageOutcome.VALIDATION_REJECTED
    if error_code is JobErrorCode.CANCELLED:
        return UsageOutcome.CANCELLED
    return UsageOutcome.FAILED


class UsageRecord(ContractModel):
    """One row per attempt. Identity fields come from the job and attempt, never from Process input.
    Carries the same measurement fields as UsageRecordInput; ``outcome`` may also be ``abandoned``."""

    tokens_input: TokenCount
    tokens_output: TokenCount
    queue_wait_seconds: Optional[float] = Field(default=None, ge=0)
    slot_wait_seconds: float = Field(default=0, ge=0)
    model_load_seconds: Optional[float] = Field(default=None, ge=0)
    inference_seconds: float = Field(ge=0)
    total_seconds: float = Field(ge=0)
    session_setup_seconds: Optional[float] = Field(default=None, ge=0)
    session_id: Optional[ResourceId] = None
    audio_seconds_processed: Optional[float] = Field(default=None, ge=0)
    peak_memory_bytes: Optional[int] = Field(default=None, ge=0)
    hardware_profile_id: ResourceId
    outcome: UsageOutcome
    error_code: Optional[JobErrorCode] = None
    billing_units: List[BillingUnit] = Field(default_factory=list)
    provider_reported_model_id: Optional[ShortText] = None
    recorded_by: UsageRecordedBy = UsageRecordedBy.PROCESS

    id: ResourceId
    job_id: ResourceId
    attempt_number: int = Field(ge=1)
    conversation_id: ResourceId
    job_type: ShortText
    purpose: Optional[ModelPurpose] = Field(default=None, description="Absent for code stages.")
    catalog_entry: Optional[CatalogEntryRef] = None
    model_revision: Optional[ShortText] = None
    route_class: Optional[RouteClass] = Field(default=None, description="Absent for stages that use no model route.")
    provider_connection_ref: Optional[ResourceId] = None
    destination_host: Optional[ShortText] = None
    recorded_at: Timestamp

    @model_validator(mode="after")
    def _synthesized(self):
        if (self.outcome is UsageOutcome.ABANDONED) != (self.recorded_by is not UsageRecordedBy.PROCESS):
            raise ValueError("abandoned rows, and only they, are Store-synthesized (or late replacements of one)")
        return self


class LateUsageReport(ContractModel):
    """A worker whose attempt ended by lease expiry attaches its measurements to that attempt's
    synthesized row. Accepted once per attempt, with that attempt's own (now stale) claim token; the
    outcome stays ``abandoned`` and the job's state does not change."""

    claim_token: str
    usage: UsageRecordInput


class UsageGroupBy(str, Enum):
    ROUTE_CLASS = "route_class"
    MODEL = "model"
    PURPOSE = "purpose"
    HARDWARE_PROFILE = "hardware_profile"
    OUTCOME = "outcome"


class UsageReportQuery(ContractModel):
    start: Timestamp
    end: Timestamp
    group_by: List[UsageGroupBy] = Field(default_factory=lambda: [UsageGroupBy.ROUTE_CLASS])
    route_class: Optional[RouteClass] = None
    purpose: Optional[ModelPurpose] = None
    include_estimates: bool = Field(default=False, description="Add cost estimates from the current price table, labeled as estimates.")


class UsageRecordQuery(PageQuery):
    """Raw usage rows over a time range, for the admin view and the CSV export."""

    start: Optional[Timestamp] = None
    end: Optional[Timestamp] = None
    conversation_id: Optional[ResourceId] = None
    route_class: Optional[RouteClass] = None
    purpose: Optional[ModelPurpose] = None
    outcome: Optional[UsageOutcome] = None


USAGE_CSV_COLUMNS: List[str] = [
    "id", "recorded_at", "conversation_id", "job_id", "attempt_number", "job_type", "purpose",
    "catalog_entry_id", "catalog_entry_version", "model_revision", "provider_reported_model_id",
    "route_class", "provider_connection_ref", "destination_host", "hardware_profile_id", "outcome",
    "error_code", "recorded_by", "tokens_input", "tokens_input_source", "tokens_output",
    "tokens_output_source", "queue_wait_seconds", "slot_wait_seconds", "model_load_seconds",
    "inference_seconds", "total_seconds", "session_setup_seconds", "session_id",
    "audio_seconds_processed", "peak_memory_bytes", "billing_units",
]
"""The CSV export (``POST /admin/usage/records.csv``): one line per UsageRecord matching the query,
in this column order, RFC 4180 quoting, UTF-8, a header line, empty cells for nulls, billing_units as
``unit=quantity`` pairs joined by ``;``. It is the underlying rows, so a rollup recomputed from it
equals the JSON report for the same range."""


class UsageRollupRow(ContractModel):
    keys: Dict[str, str] = Field(description="Group key values, keyed by the UsageGroupBy names requested.")
    attempts: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    validation_rejected: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    tokens_input_total: int = Field(ge=0)
    tokens_output_total: int = Field(ge=0)
    attempts_with_unavailable_tokens: int = Field(ge=0)
    inference_seconds_total: float = Field(ge=0)
    audio_seconds_total: float = Field(ge=0)
    billing_units: List[BillingUnit] = Field(default_factory=list)


class PerScoredCallRollup(ContractModel):
    scored_calls: int = Field(ge=0)
    attempts_per_scored_call: float = Field(ge=0)
    escalations_per_scored_call: float = Field(ge=0)
    validation_reject_rate: float = Field(ge=0, le=1)
    tokens_by_route_class: Dict[str, int] = Field(default_factory=dict)
    inference_seconds_by_hardware_profile: Dict[str, float] = Field(default_factory=dict)


class CostEstimate(ContractModel):
    """A figure computed from the admin's own price table. Always an estimate, never billing."""

    amount: float = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    price_table_version: int = Field(ge=1)
    label: Literal["estimate"] = "estimate"
    unpriced_attempts: int = Field(ge=0, description="Attempts in the row that no price entry covered.")


class UsageReport(ContractModel):
    """Admin-only. Measurements, plus optional estimates from the admin-entered, dated price table
    in Store. Never sent to Call1."""

    query: UsageReportQuery
    rows: List[UsageRollupRow]
    per_scored_call: PerScoredCallRollup
    estimates_by_row: Optional[List[Optional[CostEstimate]]] = Field(default=None, description="Aligned with rows when include_estimates is true.")
    generated_at: Timestamp


class PriceBasis(str, Enum):
    PER_1K_INPUT_TOKENS = "per_1k_input_tokens"
    PER_1K_OUTPUT_TOKENS = "per_1k_output_tokens"
    PER_INFERENCE_HOUR = "per_inference_hour"
    """Hardware amortization and power for local routes, per hour of inference time on a profile."""
    PER_BILLING_UNIT = "per_billing_unit"
    """A provider-returned unit (the published Pro1 price or the customer's BYOK price)."""


class PriceEntry(ContractModel):
    """One dated price the admin entered. The most specific matching entry in effect at the usage
    row's time applies: catalog entry, then hardware profile, then route class."""

    route_class: RouteClass
    catalog_entry_id: Optional[ShortText] = None
    hardware_profile_id: Optional[ResourceId] = None
    basis: PriceBasis
    billing_unit: Optional[ShortText] = Field(default=None, description="Required for per_billing_unit.")
    unit_price: float = Field(ge=0)
    effective_from: Timestamp
    note: Optional[SafeText] = None

    @model_validator(mode="after")
    def _unit(self):
        if (self.basis is PriceBasis.PER_BILLING_UNIT) != (self.billing_unit is not None):
            raise ValueError("per_billing_unit prices name their unit, and only they do")
        return self


class PriceTableSave(ContractModel):
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    entries: List[PriceEntry] = Field(max_length=1000)
    expected_version: int = Field(ge=0, description="0 to create.")


class PriceTable(ContractModel):
    """The admin's own dated prices, kept in Store (customer data). Dollar figures anywhere in the
    product come only from here and are labeled as estimates; nothing is hardcoded."""

    version: int = Field(ge=0, description="0 when no table has been saved.")
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    entries: List[PriceEntry]
    updated_at: Optional[Timestamp] = None
    updated_by_account_id: Optional[ResourceId] = None


class UsageMediansQuery(ContractModel):
    since: Optional[Timestamp] = Field(default=None, description="Defaults to the last 30 days.")
    purpose: Optional[ModelPurpose] = None


class UsageMedianRow(ContractModel):
    catalog_entry: CatalogEntryRef
    purpose: ModelPurpose
    route_class: RouteClass
    destination_host: Optional[ShortText] = None
    hardware_profile_id: ResourceId
    attempts: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    median_inference_seconds: Optional[float] = Field(default=None, ge=0)
    median_total_seconds: Optional[float] = Field(default=None, ge=0)
    median_tokens_input: Optional[float] = Field(default=None, ge=0)
    median_tokens_output: Optional[float] = Field(default=None, ge=0)
    validation_reject_rate: Optional[float] = Field(default=None, ge=0, le=1)


class UsageMedians(ContractModel):
    """Aggregates only, for Process Models settings (``usage:read``) and the admin view: measured
    medians per catalog entry, route and hardware profile. Never raw rows."""

    since: Timestamp
    rows: List[UsageMedianRow]
    generated_at: Timestamp
