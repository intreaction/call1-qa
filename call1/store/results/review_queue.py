"""The human review queue: rules, items, work distribution and the stale-write rule.

This is **not** the processing job queue (``call1/store/queue/``): separate tables, types, statuses
and routes (architecture rule 5, ``reviews.REVIEW_QUEUE_TRANSITIONS``).

Items are created when a QA result version commits (``projections.apply_completion``), one per
enabled rule that matches the new scorecard (the pre-split ``QueueManager`` rules). The same
transaction applies the stale-write rule to the call's older items:

* every unresolved item (PENDING or IN_REVIEW) becomes SUPERSEDED, ``stale``, and names its
  replacement: the new version's item for the same rule. When that rule no longer matches (or was
  disabled or deleted), Store still carries the item forward for the new version, so work a human
  was given never disappears silently;
* resolved items of older versions become ``stale``.

Contract 1.3.0 (Contact Signals v2, docs/ContactSignalsV2.md section 9.3): a rule may target signal
alert rules (``target_signal_alerts``, checked before the stream like ``target_domains``), and the
``SIGNAL`` stream's base condition is always true. ``on_new_evaluation`` reads the call's current
alerts into ``ScoreFacts``; ``on_new_signals`` (called when a contact_signals version is projected)
creates the missing items of alert-targeting rules at the call's current evaluation version, with a
pre-check per (call, rule, evaluation version) in the same transaction, and never supersedes
anything. A call without an evaluation gets no signal item until QA publishes (decision 22, Q2).

Distribution follows the rule's strategy at creation: UNASSIGNED_CLAIM leaves the item in the pool,
ROUND_ROBIN cycles through accepting reviewers by name, LEAST_OUTSTANDING picks the smallest
open backlog per capacity weight, SKILL_MATCHED cycles through reviewers holding a target skill.
Reviewers are the active accounts (``auth.api.list_accounts``) that accept assignments.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional

from call1.contracts.auth import AccountStatus
from call1.contracts.contents import QaScorecardContent, TranscriptContent, VerdictStatus
from call1.contracts.events import ChangeKind
from call1.contracts.reviews import (
    DistributionStrategy,
    ReviewerProfile,
    ReviewQueueItem,
    ReviewQueueRule,
    ReviewQueueRuleRecord,
    ReviewQueueStatus,
    ReviewStream,
)

from .. import db, feed
from ..auth import api as auth_api
from ..db import StoreConnection
from ..ids import new_id

log = logging.getLogger("call1.store.results")

LOW_CONFIDENCE = 0.70
"""A verdict below this confidence (or FLAGGED) makes the scorecard low-confidence (pre-split rule)."""

UNRESOLVED = (ReviewQueueStatus.PENDING.value, ReviewQueueStatus.IN_REVIEW.value)
RESOLVED = (ReviewQueueStatus.APPROVED.value, ReviewQueueStatus.OVERRIDDEN.value)

_DISPUTE_TERMS = ("dispute", "disputed", "charge", "charged", "overcharge", "overcharged", "billing error", "escalat",
                  "fraud", "unauthorized", "refund", "complaint")


# --- rules -----------------------------------------------------------------------------------


def rule_record(row) -> ReviewQueueRuleRecord:
    rule = ReviewQueueRule.model_validate(db.loads(row["rule_json"]))
    return ReviewQueueRuleRecord(**rule.model_dump(), rule_version=int(row["rule_version"]), updated_at=db.parse_ts(row["updated_at"]),
                                 updated_by_account_id=row["updated_by_account_id"])


def list_rules(conn: StoreConnection, *, enabled_only: bool = False) -> List[ReviewQueueRuleRecord]:
    sql = "SELECT * FROM results_queue_rules" + (" WHERE enabled = 1" if enabled_only else "") + " ORDER BY rank, id"
    return [rule_record(r) for r in conn.execute(sql).fetchall()]


def get_rule(conn: StoreConnection, rule_id: str) -> Optional[ReviewQueueRuleRecord]:
    row = conn.execute("SELECT * FROM results_queue_rules WHERE id = ?", (rule_id,)).fetchone()
    return None if row is None else rule_record(row)


# --- items -----------------------------------------------------------------------------------


def item_from_row(row) -> ReviewQueueItem:
    return ReviewQueueItem(
        id=row["id"],
        call_id=row["call_id"],
        conversation_id=row["conversation_id"],
        rule_id=row["rule_id"],
        rule_name=row["rule_name"],
        stream=ReviewStream(row["stream"]),
        reason=row["reason"],
        urgency_score=row["urgency_score"],
        evaluation_version=row["evaluation_version"],
        stale=bool(row["stale"]),
        superseded_by_item_id=row["superseded_by_item_id"],
        status=ReviewQueueStatus(row["status"]),
        item_version=row["item_version"],
        assigned_to_account_id=row["assigned_to_account_id"],
        assigned_display_name=row["assigned_display_name"],
        created_at=db.parse_ts(row["created_at"]),
        assigned_at=db.parse_ts(row["assigned_at"]),
        started_at=db.parse_ts(row["started_at"]),
        resolved_at=db.parse_ts(row["resolved_at"]),
        resolved_by_account_id=row["resolved_by_account_id"],
        reviewer_notes=row["reviewer_notes"],
        agent_id=row["agent_id"],
        agent_display_name=row["agent_display_name"],
        agent_extension=row["agent_extension"],
        overall_score=row["overall_score"],
        critical_failure=bool(row["critical_failure"]),
        duration_seconds=row["duration_seconds"],
        signals_version=row["signals_version"],
        trigger_alert_rule_ids=db.loads(row["trigger_alert_rule_ids_json"]) or [],
    )


def get_item_row(conn: StoreConnection, item_id: str):
    return conn.execute("SELECT * FROM results_review_items WHERE id = ?", (item_id,)).fetchone()


def item_changed(conn: StoreConnection, row_or_id) -> str:
    """Append the review_queue change event for an item (inside the writing transaction)."""
    row = get_item_row(conn, row_or_id) if isinstance(row_or_id, str) else row_or_id
    return feed.append(conn, ChangeKind.REVIEW_QUEUE, row["id"], int(row["item_version"]), row["status"],
                       conversation_id=row["conversation_id"], call_id=row["call_id"])


# --- reviewers and profiles ------------------------------------------------------------------


@dataclass
class Reviewer:
    account_id: str
    display_name: str
    skills: List[str]
    capacity_weight: float
    accepting: bool
    pending_count: int


def pending_counts(conn: StoreConnection) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT assigned_to_account_id AS a, COUNT(*) AS n FROM results_review_items WHERE status IN (?, ?) "
        "AND assigned_to_account_id IS NOT NULL GROUP BY assigned_to_account_id", UNRESOLVED).fetchall()
    return {r["a"]: int(r["n"]) for r in rows}


def profile_row(conn: StoreConnection, account_id: str):
    return conn.execute("SELECT * FROM results_reviewer_profiles WHERE account_id = ?", (account_id,)).fetchone()


def profile_for(conn: StoreConnection, account_id: str, display_name: str, counts: Optional[Dict[str, int]] = None) -> ReviewerProfile:
    row = profile_row(conn, account_id)
    counts = pending_counts(conn) if counts is None else counts
    return ReviewerProfile(
        account_id=account_id,
        display_name=display_name,
        skills=db.loads(row["skills_json"]) if row else [],
        capacity_weight=float(row["capacity_weight"]) if row else 1.0,
        accepting_assignments=bool(row["accepting_assignments"]) if row else True,
        pending_count=counts.get(account_id, 0),
    )


def active_reviewers(conn: StoreConnection) -> List[Reviewer]:
    """Active accounts with their distribution profile. Empty (items stay in the claim pool) while
    the auth area's ``list_accounts`` is not built yet."""
    try:
        accounts = auth_api.list_accounts(conn, status=AccountStatus.ACTIVE)
    except NotImplementedError:
        log.warning("auth.api.list_accounts is not implemented; review items stay unassigned")
        return []
    counts = pending_counts(conn)
    out: List[Reviewer] = []
    for account in accounts:
        profile = profile_for(conn, account.id, account.display_name, counts)
        out.append(Reviewer(account.id, account.display_name, list(profile.skills), profile.capacity_weight,
                            profile.accepting_assignments, profile.pending_count))
    return out


def _pick_assignee(conn: StoreConnection, rule: ReviewQueueRule, reviewers: List[Reviewer]) -> Optional[Reviewer]:
    strategy = rule.distribution_strategy
    pool = [r for r in reviewers if r.accepting]
    if strategy is DistributionStrategy.UNASSIGNED_CLAIM or not pool:
        return None
    if strategy is DistributionStrategy.SKILL_MATCHED:
        wanted = set(rule.target_skills)
        if wanted:
            pool = [r for r in pool if wanted.intersection(r.skills)]
        if not pool:
            return None
    if strategy is DistributionStrategy.LEAST_OUTSTANDING:
        return min(pool, key=lambda r: (r.pending_count / max(r.capacity_weight, 0.1), r.display_name, r.account_id))
    # ROUND_ROBIN and SKILL_MATCHED: a stable cycle by name over the rule's assignment count.
    pool.sort(key=lambda r: (r.display_name, r.account_id))
    assigned_so_far = conn.execute(
        "SELECT COUNT(*) AS n FROM results_review_items WHERE rule_id = ? AND assigned_at IS NOT NULL", (rule.id,)).fetchone()["n"]
    return pool[int(assigned_so_far) % len(pool)]


# --- rule matching (the pre-split QueueManager rules) ----------------------------------------


@dataclass(frozen=True)
class ScoreFacts:
    call_id: str
    conversation_id: str
    evaluation_version: int
    agent_id: str
    overall_score: Optional[float]
    critical_failure: bool
    low_confidence: bool
    dispute_signal: bool
    duration_seconds: Optional[float]
    rubric_category: Optional[str]
    agent_display_name: Optional[str] = None
    agent_extension: Optional[str] = None
    signals_version: Optional[int] = None
    signal_alerts: FrozenSet[str] = frozenset()
    signal_alert_reasons: Mapping[str, str] = None  # rule_id -> "Cancel account (caller, 0:42)"


def is_low_confidence(scorecard: QaScorecardContent) -> bool:
    return any(v.confidence < LOW_CONFIDENCE or v.status is VerdictStatus.FLAGGED for v in scorecard.verdicts)


def dispute_signal(transcript: Optional[TranscriptContent]) -> bool:
    if transcript is None:
        return False
    return any(term in (t.text or "").lower() for t in transcript.turns for term in _DISPUTE_TERMS)


def _sampled(facts: ScoreFacts, rule: ReviewQueueRule) -> bool:
    """A deterministic draw per (call, rule, version) so a replayed projection samples the same."""
    digest = hashlib.sha256(f"{facts.call_id}:{rule.id}:{facts.evaluation_version}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < rule.sampling_rate


def triggered_alerts(rule: ReviewQueueRule, facts: ScoreFacts) -> List[str]:
    """The rule's target signal alerts that match the call's current signals, in the rule's order."""
    return [a for a in rule.target_signal_alerts if a in facts.signal_alerts]


def rule_matches(rule: ReviewQueueRule, facts: ScoreFacts) -> bool:
    if rule.target_domains and (facts.rubric_category or "") not in rule.target_domains:
        return False
    if rule.target_signal_alerts and not triggered_alerts(rule, facts):
        return False
    stream = rule.stream
    if stream is ReviewStream.SIGNAL:
        return True
    if stream is ReviewStream.TRIAGE:
        if rule.critical_failure_only and rule.low_confidence_only:
            return facts.critical_failure or facts.low_confidence
        if rule.critical_failure_only:
            return facts.critical_failure
        if rule.low_confidence_only:
            return facts.low_confidence
        return facts.critical_failure or facts.low_confidence
    if stream is ReviewStream.AUDIT_SAMPLE:
        if facts.critical_failure or facts.low_confidence or facts.overall_score is None or rule.sampling_rate <= 0:
            return False
        return _sampled(facts, rule)
    if stream is ReviewStream.MANDATE:
        if rule.target_agents:
            return facts.agent_id in rule.target_agents
        return facts.dispute_signal
    if stream is ReviewStream.CALIBRATION:
        return True
    return False


def reason_for(rule: ReviewQueueRule, facts: ScoreFacts) -> str:
    if rule.stream is ReviewStream.TRIAGE:
        parts = []
        if facts.critical_failure and (rule.critical_failure_only or not rule.low_confidence_only):
            parts.append("critical compliance breach")
        if facts.low_confidence and (rule.low_confidence_only or not rule.critical_failure_only):
            parts.append("low-confidence verdicts")
        return f"Triage: {' and '.join(parts) or 'machine flagged'}."
    if rule.stream is ReviewStream.AUDIT_SAMPLE:
        return f"Audit sample: random {int(rule.sampling_rate * 100)}% of confident passes (false-negative / drift check)."
    if rule.stream is ReviewStream.MANDATE:
        return f"Mandate: {rule.name}."
    if rule.stream is ReviewStream.CALIBRATION:
        return "Calibration: blind multi-reviewer scoring."
    if rule.stream is ReviewStream.SIGNAL:
        triggered = triggered_alerts(rule, facts)
        reasons = facts.signal_alert_reasons or {}
        return "Signal: " + "; ".join(reasons.get(a, a) for a in triggered) if triggered else f"Signal: {rule.name}."
    return rule.name


def urgency_for(rule: ReviewQueueRule, facts: ScoreFacts) -> float:
    if rule.stream is ReviewStream.TRIAGE:
        score = 0.0
        if facts.critical_failure:
            score += 60.0
        if facts.low_confidence:
            score += 30.0
        if facts.overall_score is not None:
            score += max(0.0, 100.0 - facts.overall_score) * 0.1
        return round(min(100.0, score), 1)
    if rule.stream in (ReviewStream.MANDATE, ReviewStream.SIGNAL):
        return 50.0
    if rule.stream is ReviewStream.CALIBRATION:
        return 40.0
    return 10.0


# --- the stale-write rule on a new machine version --------------------------------------------


def _insert_item(conn: StoreConnection, facts: ScoreFacts, *, rule_id: str, rule_name: str, stream: str, reason: str, urgency: float,
                 assignee: Optional[Reviewer], carried_assignee: Optional[tuple] = None, signals_version: Optional[int] = None,
                 trigger_alert_rule_ids: Optional[List[str]] = None) -> str:
    now = db.ts(conn.now())
    item_id = new_id("rvw")
    assigned_to, assigned_name = (assignee.account_id, assignee.display_name) if assignee else (carried_assignee or (None, None))
    conn.execute(
        "INSERT INTO results_review_items (id, call_id, conversation_id, rule_id, rule_name, stream, reason, urgency_score, evaluation_version, "
        "stale, superseded_by_item_id, status, item_version, assigned_to_account_id, assigned_display_name, created_at, assigned_at, "
        "agent_id, agent_display_name, agent_extension, overall_score, critical_failure, duration_seconds, signals_version, trigger_alert_rule_ids_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 'PENDING', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (item_id, facts.call_id, facts.conversation_id, rule_id, rule_name[:200], stream, reason[:2000], min(100.0, max(0.0, urgency)),
         facts.evaluation_version, assigned_to, assigned_name, now, now if assigned_to else None, facts.agent_id, facts.agent_display_name,
         facts.agent_extension, facts.overall_score, 1 if facts.critical_failure else 0, facts.duration_seconds, signals_version,
         db.dumps(list(trigger_alert_rule_ids or []))),
    )
    if assignee is not None:
        assignee.pending_count += 1
    return item_id


def on_new_evaluation(conn: StoreConnection, facts: ScoreFacts) -> List[str]:
    """Create the new version's items and supersede/stale the older ones. Returns new item IDs."""
    existing = conn.execute(
        "SELECT * FROM results_review_items WHERE call_id = ? AND evaluation_version = ?", (facts.call_id, facts.evaluation_version)).fetchall()
    if existing:
        return []  # this version was already projected (replay safety)
    reviewers = active_reviewers(conn)
    created: Dict[str, str] = {}
    for rule in list_rules(conn, enabled_only=True):
        if rule_matches(rule, facts):
            created[rule.id] = _create_for_rule(conn, rule, facts, reviewers)

    older = conn.execute(
        "SELECT * FROM results_review_items WHERE call_id = ? AND evaluation_version < ? AND status IN (?, ?) ORDER BY created_at, id",
        (facts.call_id, facts.evaluation_version, *UNRESOLVED)).fetchall()
    for row in older:
        replacement = created.get(row["rule_id"])
        if replacement is None:
            replacement = _insert_item(
                conn, facts, rule_id=row["rule_id"], rule_name=row["rule_name"], stream=row["stream"],
                reason=f"Carried forward for evaluation version {facts.evaluation_version}: {row['reason']}"[:2000],
                urgency=float(row["urgency_score"]), assignee=None,
                carried_assignee=(row["assigned_to_account_id"], row["assigned_display_name"]),
                signals_version=row["signals_version"], trigger_alert_rule_ids=db.loads(row["trigger_alert_rule_ids_json"]) or [])
            created[row["rule_id"]] = replacement
        conn.execute(
            "UPDATE results_review_items SET status = 'SUPERSEDED', stale = 1, superseded_by_item_id = ?, item_version = item_version + 1 WHERE id = ?",
            (replacement, row["id"]))
        item_changed(conn, row["id"])
    conn.execute("UPDATE results_review_items SET stale = 1 WHERE call_id = ? AND evaluation_version < ? AND stale = 0",
                 (facts.call_id, facts.evaluation_version))
    for item_id in created.values():
        item_changed(conn, item_id)
    return list(created.values())


def _create_for_rule(conn: StoreConnection, rule: ReviewQueueRule, facts: ScoreFacts, reviewers: List[Reviewer]) -> str:
    targeted = bool(rule.target_signal_alerts)
    return _insert_item(conn, facts, rule_id=rule.id, rule_name=rule.name, stream=rule.stream.value, reason=reason_for(rule, facts),
                        urgency=urgency_for(rule, facts), assignee=_pick_assignee(conn, rule, reviewers),
                        signals_version=facts.signals_version if targeted else None,
                        trigger_alert_rule_ids=triggered_alerts(rule, facts) if targeted else [])


def on_new_signals(conn: StoreConnection, facts: ScoreFacts) -> List[str]:
    """A contact_signals version was projected. For the call's current evaluation version, create
    the item of every enabled alert-targeting rule that matches and has none yet. The pre-check per
    (call, rule, evaluation version) runs in this transaction (``_insert_item`` also bumps the
    assignee's pending count, so a constraint violation is never relied on); nothing is superseded."""
    if facts.evaluation_version is None or facts.evaluation_version < 1:
        return []
    reviewers: Optional[List[Reviewer]] = None
    created: List[str] = []
    for rule in list_rules(conn, enabled_only=True):
        if not rule.target_signal_alerts or not rule_matches(rule, facts):
            continue
        exists = conn.execute("SELECT 1 FROM results_review_items WHERE call_id = ? AND rule_id = ? AND evaluation_version = ?",
                              (facts.call_id, rule.id, facts.evaluation_version)).fetchone()
        if exists is not None:
            continue
        if reviewers is None:
            reviewers = active_reviewers(conn)
        created.append(_create_for_rule(conn, rule, facts, reviewers))
    for item_id in created:
        item_changed(conn, item_id)
    return created


def list_profiles(conn: StoreConnection) -> List[ReviewerProfile]:
    accounts = auth_api.list_accounts(conn, status=AccountStatus.ACTIVE)
    counts = pending_counts(conn)
    profiles = [profile_for(conn, a.id, a.display_name, counts) for a in accounts]
    profiles.sort(key=lambda p: (p.display_name, p.account_id))
    return profiles

