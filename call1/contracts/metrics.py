"""Metrics read models for Evaluate. Aggregates only; never transcript content."""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Union

from pydantic import Field, model_validator

from .calls import AgentDisplayName, AgentExtension
from .common import ContractModel, ShortText, Timestamp
from .contents import SignalNodeId


class MetricsQuery(ContractModel):
    rubric_id: Optional[ShortText] = None
    start: Optional[Timestamp] = None
    end: Optional[Timestamp] = None


class ExecutiveMetrics(ContractModel):
    total_audited_calls: int = Field(ge=0)
    calls_pending_analysis: int = Field(ge=0, description="Calls whose scorecard has not been assembled yet; never counted as failed.")
    pass_rate_pct: float = Field(ge=0, le=100)
    critical_compliance_breaches: int = Field(ge=0)
    supervisor_escalations: int = Field(ge=0)
    average_score: float = Field(ge=0, le=100)
    total_hours_audited: float = Field(ge=0)
    generated_at: Timestamp


class CriterionCounts(ContractModel):
    PASS: int = Field(ge=0)
    FAIL: int = Field(ge=0)
    FLAGGED: int = Field(ge=0)
    NOT_APPLICABLE: int = Field(ge=0)
    total: int = Field(ge=0)
    pass_rate_pct: float = Field(ge=0, le=100)


class CriterionMetric(ContractModel):
    criterion_id: ShortText
    criterion_name: ShortText
    category: ShortText
    counts: CriterionCounts


class DailyMetric(ContractModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    evaluated: int = Field(ge=0)
    mean_score: float = Field(ge=0, le=100)


class RubricMetrics(ContractModel):
    rubric_id: ShortText
    rubric_name: ShortText
    rubric_version: Optional[int] = Field(default=None, ge=1, description="Null aggregates across versions.")
    total_calls_evaluated: int = Field(ge=0)
    average_score: float = Field(ge=0, le=100)
    pass_rate_pct: float = Field(ge=0, le=100)
    criteria: List[CriterionMetric]
    daily: List[DailyMetric]
    generated_at: Timestamp


class CriterionAgreement(ContractModel):
    criterion_id: ShortText
    criterion_name: ShortText
    machine_decisions: int = Field(ge=0)
    human_reviewed: int = Field(ge=0)
    overrides: int = Field(ge=0)
    agreement_rate: Optional[float] = Field(default=None, ge=0, le=1)
    precision: Optional[float] = Field(default=None, ge=0, le=1)
    recall: Optional[float] = Field(default=None, ge=0, le=1)
    automatic_decision_coverage: Optional[float] = Field(default=None, ge=0, le=1, description="Share of decisions no human changed.")


class ReviewAgreementMetrics(ContractModel):
    rubric_id: Optional[ShortText] = None
    calls_reviewed: int = Field(ge=0)
    overall_agreement_rate: Optional[float] = Field(default=None, ge=0, le=1)
    per_criterion: List[CriterionAgreement]
    generated_at: Timestamp


# --- Contact Signals v2 (1.3.0; docs/ContactSignalsV2.md sections 7.4 and 9.5) -------------------
#
# Aggregate SQL over Store's signal projection tables (IDs, enum and boolean values only; never
# text), joined to each call's current contact_signals version. The denominator is calls whose
# current result is v2, or v1 for built-in categories. Signals never touch the scorecard metrics.

SIGNAL_PRECISION_MIN_JUDGED = 5
"""Precision is shown only once at least this many hits have been judged (confirmed + dismissed)."""


def signal_precision_pct(confirmed: int, dismissed: int) -> Optional[float]:
    """``100 * confirmed / judged``, or None under ``SIGNAL_PRECISION_MIN_JUDGED`` judged hits."""
    judged = confirmed + dismissed
    return None if judged < SIGNAL_PRECISION_MIN_JUDGED else 100.0 * confirmed / judged


def _precision_needs_judgements(confirmed: int, dismissed: int, precision_pct: Optional[float]) -> None:
    if precision_pct is not None and confirmed + dismissed < SIGNAL_PRECISION_MIN_JUDGED:
        raise ValueError(f"precision is null under {SIGNAL_PRECISION_MIN_JUDGED} judged hits")


class SignalMetricsQuery(MetricsQuery):
    category_id: Optional[SignalNodeId] = Field(default=None, description="Drill into one category.")
    include_inactive: bool = Field(default=False, description="Include retired categories, subcategories and rules.")


class SignalCount(ContractModel):
    """One ranked row: a subcategory (or 'other') of a category, e.g. a top caller need."""

    id: SignalNodeId
    name: ShortText
    calls_with_hit: int = Field(ge=0)
    hits: int = Field(ge=0)
    share_pct: float = Field(ge=0, le=100)
    confirmed: int = Field(ge=0)
    dismissed: int = Field(ge=0)
    precision_pct: Optional[float] = Field(default=None, ge=0, le=100, description="Null under 5 judged hits.")

    @model_validator(mode="after")
    def _precision(self):
        _precision_needs_judgements(self.confirmed, self.dismissed, self.precision_pct)
        return self


class SignalDayCount(ContractModel):
    day: date
    calls_scored: int = Field(ge=0)
    calls_with_hit: int = Field(ge=0)


class SignalAgentCount(ContractModel):
    agent_id: ShortText
    agent_display_name: Optional[AgentDisplayName] = None
    agent_extension: Optional[AgentExtension] = None
    calls_with_hit: int = Field(ge=0)


class SignalValueCount(ContractModel):
    value: Union[str, bool]
    count: int = Field(ge=0)


class SignalFieldDistribution(ContractModel):
    """Values of one enum or boolean field on a node (never string, number, amount or date values)."""

    node_id: SignalNodeId = Field(description="The category or subcategory the field belongs to.")
    field_id: SignalNodeId
    name: ShortText
    values: List[SignalValueCount]


class SignalCategoryMetric(ContractModel):
    category_id: SignalNodeId
    name: ShortText
    builtin: bool
    active: bool
    calls_scored: int = Field(ge=0)
    calls_with_hit: int = Field(ge=0)
    hit_rate_pct: float = Field(ge=0, le=100)
    hits_total: int = Field(ge=0)
    confirmed: int = Field(ge=0)
    dismissed: int = Field(ge=0)
    precision_pct: Optional[float] = Field(default=None, ge=0, le=100)
    subcategories: List[SignalCount] = Field(description="Ranked; includes 'other'.")
    subcategory_precision_pct: Optional[float] = Field(default=None, ge=0, le=100)
    by_day: List[SignalDayCount]
    top_agents: List[SignalAgentCount] = Field(max_length=10)
    fields: List[SignalFieldDistribution]

    @model_validator(mode="after")
    def _precision(self):
        _precision_needs_judgements(self.confirmed, self.dismissed, self.precision_pct)
        if self.calls_with_hit > self.calls_scored:
            raise ValueError("calls with a hit are calls scored")
        return self


class SignalAlertMetric(ContractModel):
    rule_id: SignalNodeId
    name: str = Field(min_length=1, max_length=60)
    enabled: bool
    calls_matched: int = Field(ge=0)
    match_rate_pct: float = Field(ge=0, le=100)
    by_day: List[SignalDayCount]


class SignalMetrics(ContractModel):
    """``GET /metrics/signals``. "Top caller needs" is intent's subcategory ranking."""

    start: Optional[Timestamp] = None
    end: Optional[Timestamp] = None
    calls_scored: int = Field(ge=0)
    calls_with_signal: int = Field(ge=0)
    top_caller_needs: List[SignalCount] = Field(description="intent's subcategories ranked by calls, with 'other'; empty until intent has any.")
    categories: List[SignalCategoryMetric]
    alerts: List[SignalAlertMetric]
    calls_by_pipeline: Dict[str, int] = Field(description="Calls per current pipeline: 'v1', 'v2'.")

    @model_validator(mode="after")
    def _pipelines(self):
        if not set(self.calls_by_pipeline) <= {"v1", "v2"} or any(n < 0 for n in self.calls_by_pipeline.values()):
            raise ValueError("calls_by_pipeline counts v1 and v2 calls")
        return self
