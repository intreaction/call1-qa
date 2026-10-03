"""Metrics read models (aggregates only, never transcript content).

Every metric counts each call once, at its current machine evaluation version, filtered by the
call's ``created_at`` (``MetricsQuery.start`` inclusive, ``end`` exclusive) and optionally by the
current evaluation's rubric. A call without a scorecard is ``calls_pending_analysis`` and never
counts as failed, unless its work stopped without a result ('Needs attention'), which is counted in
neither.

Review agreement compares the machine verdict of the current evaluation with the human outcome:
a criterion the reviewer overrode (the newest override of that version) counts as an override; any
call with a human decision on its current version (an override, an escalation resolution or a
resolved queue item) counts as human-reviewed for every criterion. FAIL is the positive class for
precision and recall.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from call1.contracts.contents import ResultState, VerdictStatus
from call1.contracts.metrics import (
    CriterionAgreement,
    CriterionCounts,
    CriterionMetric,
    DailyMetric,
    ExecutiveMetrics,
    MetricsQuery,
    ReviewAgreementMetrics,
    RubricMetrics,
)

from .. import db
from ..db import StoreConnection, read_snapshot
from ..errors import not_found
from . import rubric_store


def _window(query: MetricsQuery, alias: str = "c") -> Tuple[str, list]:
    where, args = [], []
    if query.start is not None:
        where.append(f"{alias}.created_at >= ?")
        args.append(db.ts(query.start))
    if query.end is not None:
        where.append(f"{alias}.created_at < ?")
        args.append(db.ts(query.end))
    if query.rubric_id is not None:
        where.append(f"{alias}.rubric_id = ?")
        args.append(query.rubric_id)
    return (" AND ".join(where) if where else "1=1"), args


HOURS_AUDITED_DECIMALS = 6
"""``total_hours_audited`` keeps 6 decimals (0.0036 s of audio): a short call is never rounded away
(one 44 s call is 0.012222 h, not 0.01 h) and totals stay within a second of the true sum. The
contract gives only a float in hours; how many digits to show is the client's choice."""


def _pct(part: int, total: int) -> float:
    return round(part / total * 100.0, 1) if total else 0.0


def _pending_analysis(conn: StoreConnection, query: MetricsQuery) -> int:
    """Calls without a scorecard whose analysis is still under way.

    A call whose work stopped without a result (a result group reads ``failed``, 'Needs
    attention', and nothing is still ``pending``) is not pending: it will not get a scorecard
    until someone reanalyses it. The contract has no field for those, so they are simply not
    counted here."""
    from .records import result_groups  # records imports the queue area; keep metrics import-light

    where, args = _window(MetricsQuery(start=query.start, end=query.end))
    rows = conn.execute(f"SELECT c.conversation_id FROM results_calls c WHERE c.evaluation_version IS NULL AND {where}", args).fetchall()
    count = 0
    for row in rows:
        states = {g.state for g in result_groups(conn, row["conversation_id"])}
        if ResultState.FAILED in states and ResultState.PENDING not in states:
            continue
        count += 1
    return count


def executive(conn: StoreConnection, query: MetricsQuery) -> ExecutiveMetrics:
    with read_snapshot(conn):
        where, args = _window(query)
        row = conn.execute(
            f"SELECT COUNT(*) AS audited, SUM(passed) AS passed, SUM(critical_failure) AS critical, SUM(requires_human_review) AS review, "
            f"AVG(overall_score) AS avg_score, SUM(COALESCE(duration_seconds, 0)) AS seconds FROM results_calls c "
            f"WHERE c.evaluation_version IS NOT NULL AND {where}", args).fetchone()
        pending = _pending_analysis(conn, query) if query.rubric_id is None else 0
        audited = int(row["audited"] or 0)
        return ExecutiveMetrics(
            total_audited_calls=audited,
            calls_pending_analysis=int(pending or 0),
            pass_rate_pct=_pct(int(row["passed"] or 0), audited),
            critical_compliance_breaches=int(row["critical"] or 0),
            supervisor_escalations=int(row["review"] or 0),
            average_score=round(float(row["avg_score"] or 0.0), 1),
            total_hours_audited=round(float(row["seconds"] or 0.0) / 3600.0, HOURS_AUDITED_DECIMALS),
            generated_at=conn.now(),
        )


def rubric(conn: StoreConnection, rubric_id: str, query: MetricsQuery) -> RubricMetrics:
    with read_snapshot(conn):
        current = rubric_store.current_version(conn, rubric_id)
        if current is None:
            raise not_found("Rubric", rubric_id=rubric_id)
        where, args = _window(MetricsQuery(rubric_id=rubric_id, start=query.start, end=query.end))
        calls = conn.execute(
            f"SELECT call_id, evaluation_version, overall_score, passed, evaluated_at FROM results_calls c "
            f"WHERE c.evaluation_version IS NOT NULL AND {where}", args).fetchall()
        counts: Dict[str, Dict[str, int]] = {}
        names: Dict[str, Tuple[str, str]] = {c.criterion_id: (c.name, c.category) for c in current.definition.criteria}
        order: List[str] = [c.criterion_id for c in current.definition.criteria]
        for call in calls:
            for v in conn.execute("SELECT criterion_id, criterion_name, status FROM results_verdicts WHERE call_id = ? AND evaluation_version = ?",
                                  (call["call_id"], call["evaluation_version"])).fetchall():
                cid = v["criterion_id"]
                if cid not in names:
                    names[cid] = (v["criterion_name"], "UNKNOWN")
                    order.append(cid)
                bucket = counts.setdefault(cid, {s.value: 0 for s in VerdictStatus})
                bucket[v["status"]] = bucket.get(v["status"], 0) + 1
        criteria = []
        for cid in order:
            bucket = counts.get(cid, {s.value: 0 for s in VerdictStatus})
            total = sum(bucket.values())
            criteria.append(CriterionMetric(
                criterion_id=cid, criterion_name=names[cid][0], category=names[cid][1],
                counts=CriterionCounts(PASS=bucket["PASS"], FAIL=bucket["FAIL"], FLAGGED=bucket["FLAGGED"], NOT_APPLICABLE=bucket["NOT_APPLICABLE"],
                                       total=total, pass_rate_pct=_pct(bucket["PASS"], total)),
            ))
        daily: Dict[str, List[float]] = {}
        for call in calls:
            day = (call["evaluated_at"] or "")[:10]
            if day:
                daily.setdefault(day, []).append(float(call["overall_score"] or 0.0))
        total = len(calls)
        return RubricMetrics(
            rubric_id=rubric_id,
            rubric_name=current.definition.name,
            rubric_version=None,
            total_calls_evaluated=total,
            average_score=round(sum(float(c["overall_score"] or 0.0) for c in calls) / total, 1) if total else 0.0,
            pass_rate_pct=_pct(sum(1 for c in calls if c["passed"]), total),
            criteria=criteria,
            daily=[DailyMetric(date=d, evaluated=len(s), mean_score=round(sum(s) / len(s), 1)) for d, s in sorted(daily.items())],
            generated_at=conn.now(),
        )


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def review_agreement(conn: StoreConnection, query: MetricsQuery) -> ReviewAgreementMetrics:
    with read_snapshot(conn):
        where, args = _window(query)
        calls = conn.execute(
            f"SELECT c.call_id, c.evaluation_version FROM results_calls c WHERE c.evaluation_version IS NOT NULL AND {where}", args).fetchall()
        stats: Dict[str, Dict[str, int]] = {}
        names: Dict[str, str] = {}
        reviewed_calls = 0
        for call in calls:
            call_id, version = call["call_id"], int(call["evaluation_version"])
            state = conn.execute("SELECT reviewed_evaluation_version, escalation_status, escalation_evaluation_version FROM results_review_state "
                                 "WHERE call_id = ?", (call_id,)).fetchone()
            resolved_item = conn.execute("SELECT 1 FROM results_review_items WHERE call_id = ? AND evaluation_version = ? AND status IN "
                                         "('APPROVED', 'OVERRIDDEN') LIMIT 1", (call_id, version)).fetchone()
            reviewed = bool(resolved_item) or (state is not None and (
                state["reviewed_evaluation_version"] == version or state["escalation_evaluation_version"] == version))
            reviewed_calls += 1 if reviewed else 0
            final: Dict[str, str] = {}
            for o in conn.execute("SELECT criterion_id, status FROM results_verdict_overrides WHERE call_id = ? AND evaluation_version = ? "
                                  "ORDER BY created_at, id", (call_id, version)).fetchall():
                final[o["criterion_id"]] = o["status"]
            for v in conn.execute("SELECT criterion_id, criterion_name, status FROM results_verdicts WHERE call_id = ? AND evaluation_version = ?",
                                  (call_id, version)).fetchall():
                cid = v["criterion_id"]
                names.setdefault(cid, v["criterion_name"])
                s = stats.setdefault(cid, {"machine": 0, "human": 0, "overrides": 0, "tp": 0, "fp": 0, "fn": 0})
                s["machine"] += 1
                if not reviewed and cid not in final:
                    continue
                s["human"] += 1
                human = final.get(cid, v["status"])
                if human != v["status"]:
                    s["overrides"] += 1
                machine_fail, human_fail = v["status"] == "FAIL", human == "FAIL"
                s["tp"] += 1 if machine_fail and human_fail else 0
                s["fp"] += 1 if machine_fail and not human_fail else 0
                s["fn"] += 1 if human_fail and not machine_fail else 0
        per = []
        agree_total = human_total = 0
        for cid in sorted(stats):
            s = stats[cid]
            agree_total += s["human"] - s["overrides"]
            human_total += s["human"]
            per.append(CriterionAgreement(
                criterion_id=cid, criterion_name=names[cid], machine_decisions=s["machine"], human_reviewed=s["human"], overrides=s["overrides"],
                agreement_rate=_ratio(s["human"] - s["overrides"], s["human"]),
                precision=_ratio(s["tp"], s["tp"] + s["fp"]), recall=_ratio(s["tp"], s["tp"] + s["fn"]),
                automatic_decision_coverage=_ratio(s["machine"] - s["overrides"], s["machine"]),
            ))
        return ReviewAgreementMetrics(rubric_id=query.rubric_id, calls_reviewed=reviewed_calls,
                                      overall_agreement_rate=_ratio(agree_total, human_total), per_criterion=per, generated_at=conn.now())


# --- Contact Signals v2 (contract 1.3.0; docs/ContactSignalsV2.md sections 7.4 and 9.5) ---------


def signal_metrics(conn: StoreConnection, query):
    """``getSignalMetrics``: aggregates over the signal projection tables (IDs, enum and boolean
    values only, never content), each call at its current ``signals_version``, filtered by
    ``_window()``. The denominator of a built-in category is every call with projected signals (v1
    or v2); of a custom category, calls whose current result is v2. Precision is null under 5 judged
    hits (``metrics.signal_precision_pct``). A subcategory verdict counts only while it judged the
    hit's current subcategory (ID and digest)."""
    from call1.contracts.contents import SIGNAL_OTHER_OPTION, SignalFieldType
    from call1.contracts.metrics import (
        SignalAgentCount,
        SignalAlertMetric,
        SignalCategoryMetric,
        SignalCount,
        SignalDayCount,
        SignalFieldDistribution,
        SignalMetrics,
        SignalValueCount,
        signal_precision_pct,
    )

    from . import signal_store, signals

    with read_snapshot(conn):
        where, args = _window(query)
        calls = {r["call_id"]: r for r in conn.execute(
            "SELECT c.call_id, c.created_at, c.agent_id, c.agent_display_name, c.agent_extension, o.pipeline FROM results_calls c "
            "JOIN results_signal_outcomes o ON o.call_id = c.call_id AND o.signals_version = c.signals_version "
            f"WHERE c.signals_version IS NOT NULL AND {where}", args).fetchall()}
        hits = [h for h in conn.execute(
            "SELECT h.* FROM results_signal_hits h JOIN results_calls c ON c.call_id = h.call_id AND c.signals_version = h.signals_version "
            f"WHERE {where}", args).fetchall() if h["call_id"] in calls]
        fields = conn.execute(
            "SELECT f.* FROM results_signal_hit_fields f JOIN results_calls c ON c.call_id = f.call_id AND c.signals_version = f.signals_version "
            f"WHERE f.status = 'extracted' AND {where}", args).fetchall()
        feedback = {(r["call_id"], r["hit_id"]): r for r in conn.execute("SELECT * FROM results_signal_feedback").fetchall()}
        taxonomy = signal_store.current_version(conn).taxonomy
        rules = signal_store.list_alert_rules(conn) if query.include_inactive else signal_store.active_alert_rules(conn)
        matched_by_rule = {}
        for rule in rules:
            if not rule.enabled or not rule.node_active:
                matched_by_rule[rule.rule_id] = set()
                continue
            cond, cond_args = signals.alert_condition_sql(rule)
            rows = conn.execute(
                "SELECT DISTINCT h.call_id FROM results_signal_hits h JOIN results_calls c ON c.call_id = h.call_id "
                f"AND c.signals_version = h.signals_version WHERE {cond} AND {where}", (*cond_args, *args)).fetchall()
            matched_by_rule[rule.rule_id] = {r["call_id"] for r in rows} & set(calls)

    def day(call_id):
        return calls[call_id]["created_at"][:10]

    def by_day(scored, with_hit):
        days = {}
        for call_id in scored:
            d = days.setdefault(day(call_id), [0, 0])
            d[0] += 1
            d[1] += 1 if call_id in with_hit else 0
        return [SignalDayCount(day=d, calls_scored=n, calls_with_hit=w) for d, (n, w) in sorted(days.items())]

    def verdicts(rows):
        confirmed = dismissed = 0
        for h in rows:
            fb = feedback.get((h["call_id"], h["hit_id"]))
            if fb is not None:
                confirmed += fb["category_verdict"] == "confirmed"
                dismissed += fb["category_verdict"] == "dismissed"
        return confirmed, dismissed

    def sub_verdicts(rows):
        confirmed = corrected = 0
        for h in rows:
            fb = feedback.get((h["call_id"], h["hit_id"]))
            if fb is None or fb["subcategory_verdict"] is None:
                continue
            if fb["subcategory_id"] != h["subcategory_id"] or fb["subcategory_digest"] != h["subcategory_digest"]:
                continue  # judged an earlier subcategory
            confirmed += fb["subcategory_verdict"] == "confirmed"
            corrected += fb["subcategory_verdict"] == "corrected"
        return confirmed, corrected

    fields_by_hit = {}
    for f in fields:
        fields_by_hit.setdefault((f["call_id"], f["hit_id"]), []).append(f)

    def category_metric(c):
        scored = [cid for cid, r in calls.items() if c.builtin or r["pipeline"] == "v2"]
        scored_set = set(scored)
        rows = [h for h in hits if h["category_id"] == c.category_id and h["call_id"] in scored_set]
        with_hit = {h["call_id"] for h in rows}
        confirmed, dismissed = verdicts(rows)
        subs = {s.subcategory_id: s for s in c.subcategories}
        grouped = {}
        for h in rows:
            sid = h["subcategory_id"]
            if sid is None:
                continue
            if sid != SIGNAL_OTHER_OPTION and not query.include_inactive and not (sid in subs and subs[sid].active):
                continue
            grouped.setdefault(sid, []).append(h)
        total_sub = sum(len(v) for v in grouped.values())
        counts = []
        for sid, group in grouped.items():
            sc, sd = sub_verdicts(group)
            name = "Other" if sid == SIGNAL_OTHER_OPTION else (subs[sid].name if sid in subs else sid)
            counts.append(SignalCount(id=sid, name=name, calls_with_hit=len({h["call_id"] for h in group}), hits=len(group),
                                      share_pct=_pct(len(group), total_sub), confirmed=sc, dismissed=sd, precision_pct=signal_precision_pct(sc, sd)))
        counts.sort(key=lambda s: (-s.calls_with_hit, -s.hits, s.id))
        sc_all, sd_all = sub_verdicts([h for g in grouped.values() for h in g])
        agents = {}
        for cid in with_hit:
            r = calls[cid]
            a = agents.setdefault(r["agent_id"], [r, 0])
            a[1] += 1
        top = sorted(agents.values(), key=lambda a: (-a[1], a[0]["agent_id"]))[:10]
        distributions = []
        nodes = [(c.category_id, None, c.fields)] + [(s.subcategory_id, s.subcategory_id, s.fields) for s in c.subcategories
                                                     if s.active or query.include_inactive]
        for node_id, sub_id, node_fields in nodes:
            for fd in node_fields:
                if fd.type not in (SignalFieldType.ENUM, SignalFieldType.BOOLEAN):
                    continue
                tally = {}
                for h in rows:
                    if sub_id is not None and h["subcategory_id"] != sub_id:
                        continue
                    for f in fields_by_hit.get((h["call_id"], h["hit_id"]), []):
                        if f["field_id"] != fd.field_id:
                            continue
                        value = f["value_enum"] if fd.type is SignalFieldType.ENUM else bool(f["value_bool"])
                        tally[value] = tally.get(value, 0) + 1
                values = list(fd.enum_values) if fd.type is SignalFieldType.ENUM else [True, False]
                values += [v for v in tally if v not in values]
                distributions.append(SignalFieldDistribution(node_id=node_id, field_id=fd.field_id, name=fd.name,
                                                             values=[SignalValueCount(value=v, count=tally.get(v, 0)) for v in values]))
        return SignalCategoryMetric(
            category_id=c.category_id, name=c.name, builtin=c.builtin, active=c.active, calls_scored=len(scored),
            calls_with_hit=len(with_hit), hit_rate_pct=_pct(len(with_hit), len(scored)), hits_total=len(rows),
            confirmed=confirmed, dismissed=dismissed, precision_pct=signal_precision_pct(confirmed, dismissed),
            subcategories=counts, subcategory_precision_pct=signal_precision_pct(sc_all, sd_all), by_day=by_day(scored, with_hit),
            top_agents=[SignalAgentCount(agent_id=a[0]["agent_id"], agent_display_name=a[0]["agent_display_name"],
                                         agent_extension=a[0]["agent_extension"], calls_with_hit=a[1]) for a in top],
            fields=distributions,
        )

    included = [c for c in taxonomy.categories if (c.active or query.include_inactive)
                and (query.category_id is None or c.category_id == query.category_id)]
    categories = [category_metric(c) for c in included]
    intent = taxonomy.category("intent")
    intent_metric = next((m for m in categories if m.category_id == "intent"), None) or category_metric(intent)
    active_ids = {c.category_id for c in taxonomy.categories if c.active or query.include_inactive}
    with_signal = {h["call_id"] for h in hits if h["category_id"] in active_ids}
    alert_metrics = [SignalAlertMetric(rule_id=r.rule_id, name=r.name, enabled=r.enabled, calls_matched=len(matched_by_rule[r.rule_id]),
                                       match_rate_pct=_pct(len(matched_by_rule[r.rule_id]), len(calls)),
                                       by_day=by_day(list(calls), matched_by_rule[r.rule_id])) for r in rules]
    pipelines = {}
    for r in calls.values():
        pipelines[r["pipeline"]] = pipelines.get(r["pipeline"], 0) + 1
    return SignalMetrics(start=query.start, end=query.end, calls_scored=len(calls), calls_with_signal=len(with_signal),
                         top_caller_needs=list(intent_metric.subcategories), categories=categories, alerts=alert_metrics,
                         calls_by_pipeline=pipelines)
