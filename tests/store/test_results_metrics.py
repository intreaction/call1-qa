"""Metrics: executive, per-rubric and review agreement (aggregates over current evaluations)."""

from __future__ import annotations

from call1.contracts.contents import VerdictStatus
from call1.contracts.jobs import JobType

from .test_results_harness import accounts, fq, validation_report  # noqa: F401


def _call(fq, verdicts, *, duration=1800.0, **scorecard):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.VALIDATION_VAD])
    from .test_results_harness import vad_metrics
    fq.complete(conv, graph, JobType.VALIDATION_VAD, {"validation_report": validation_report(duration), "vad_metrics": vad_metrics()})
    fq.ingest_qa(conv, verdicts, **scorecard)
    return conv


def test_executive_metrics(fq, client, reviewer_session):
    _call(fq, [("REG-01", VerdictStatus.PASS, 0.9)], overall_score=90, passed=True)
    _call(fq, [("REG-01", VerdictStatus.FAIL, 0.9)], overall_score=40, passed=False, critical_failure=True, requires_human_review=True)
    fq.register()  # no scorecard yet: pending, never failed
    body = client.get("/store/v1/metrics/executive", headers=reviewer_session.read_headers).json()
    assert body["total_audited_calls"] == 2 and body["calls_pending_analysis"] == 1
    assert body["pass_rate_pct"] == 50.0 and body["average_score"] == 65.0 and body["critical_compliance_breaches"] == 1
    assert body["supervisor_escalations"] == 1 and body["total_hours_audited"] == 1.0
    other = client.get("/store/v1/metrics/executive", params={"rubric_id": "acme"}, headers=reviewer_session.read_headers).json()
    assert other["total_audited_calls"] == 0 and other["pass_rate_pct"] == 0.0
    later = client.get("/store/v1/metrics/executive", params={"start": "2030-01-01T00:00:00Z"}, headers=reviewer_session.read_headers).json()
    assert later["total_audited_calls"] == 0


def test_failed_call_is_not_pending_analysis(fq, client, reviewer_session):
    """A call whose scorecard job ended without a result ('Needs attention') is not pending: it
    will not get a scorecard until someone reanalyses it. A call still in flight stays pending."""
    failed = fq.register()
    fq.fail(failed, fq.graph(failed, [JobType.QA_SCORECARD]), JobType.QA_SCORECARD)
    in_flight = fq.register()
    fq.graph(in_flight, [JobType.QA_SCORECARD])
    body = client.get("/store/v1/metrics/executive", headers=reviewer_session.read_headers).json()
    assert body["calls_pending_analysis"] == 1 and body["total_audited_calls"] == 0


def test_hours_audited_keeps_short_calls(fq, client, reviewer_session):
    """q15: one 44 s audited call is 0.012222 h, not a 2-decimal 0.01 (and never 0.0); totals stay
    within a second of the true sum of durations."""
    _call(fq, [("REG-01", VerdictStatus.PASS, 0.9)], duration=44.0, overall_score=90, passed=True)
    hours = client.get("/store/v1/metrics/executive", headers=reviewer_session.read_headers).json()["total_hours_audited"]
    assert hours == round(44.0 / 3600.0, 6) == 0.012222
    _call(fq, [("REG-01", VerdictStatus.PASS, 0.9)], duration=1.5, overall_score=90, passed=True)
    hours = client.get("/store/v1/metrics/executive", headers=reviewer_session.read_headers).json()["total_hours_audited"]
    assert abs(hours * 3600.0 - 45.5) < 0.01, hours


def test_rubric_metrics_use_the_current_version_of_each_call(fq, client, reviewer_session):
    conv = _call(fq, [("REG-01", VerdictStatus.FAIL, 0.9), ("SEC-01", VerdictStatus.PASS, 0.9)], overall_score=50, passed=False)
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.PASS, 0.9), ("SEC-01", VerdictStatus.PASS, 0.9)], overall_score=100)
    _call(fq, [("REG-01", VerdictStatus.FLAGGED, 0.3), ("SEC-01", VerdictStatus.NOT_APPLICABLE, 0.9)], overall_score=80)
    body = client.get("/store/v1/metrics/rubrics/call1_standard_v2", headers=reviewer_session.read_headers).json()
    assert body["rubric_name"].startswith("Call1 Standard") and body["total_calls_evaluated"] == 2 and body["average_score"] == 90.0
    by_id = {c["criterion_id"]: c for c in body["criteria"]}
    assert [c["criterion_id"] for c in body["criteria"]][:4] == ["REG-01", "SEC-01", "COMP-01", "ETIQ-01"]
    assert by_id["REG-01"]["counts"] == {"PASS": 1, "FAIL": 0, "FLAGGED": 1, "NOT_APPLICABLE": 0, "total": 2, "pass_rate_pct": 50.0}
    assert by_id["COMP-01"]["counts"]["total"] == 0 and by_id["REG-01"]["category"] == "COMPLIANCE"
    assert body["daily"] == [{"date": "2026-09-25", "evaluated": 2, "mean_score": 90.0}]
    assert client.get("/store/v1/metrics/rubrics/unknown", headers=reviewer_session.read_headers).status_code == 404


def test_review_agreement(fq, client, reviewer_session, supervisor_session):
    reviewed = _call(fq, [("REG-01", VerdictStatus.FAIL, 0.9), ("SEC-01", VerdictStatus.PASS, 0.9)])
    _call(fq, [("REG-01", VerdictStatus.PASS, 0.9), ("SEC-01", VerdictStatus.PASS, 0.9)])
    client.post(f"/store/v1/calls/{reviewed.call_id}/verdicts/REG-01", json={"status": "PASS", "expected_version": 0, "evaluation_version": 1},
                headers=reviewer_session.headers)
    assert client.get("/store/v1/metrics/review-agreement", headers=reviewer_session.read_headers).json()["code"] == "insufficient_role"
    body = client.get("/store/v1/metrics/review-agreement", headers=supervisor_session.read_headers).json()
    assert body["calls_reviewed"] == 1 and body["overall_agreement_rate"] == 0.5
    per = {c["criterion_id"]: c for c in body["per_criterion"]}
    assert per["REG-01"] == {"criterion_id": "REG-01", "criterion_name": "Criterion REG-01", "machine_decisions": 2, "human_reviewed": 1,
                             "overrides": 1, "agreement_rate": 0.0, "precision": 0.0, "recall": None, "automatic_decision_coverage": 0.5}
    assert per["SEC-01"]["agreement_rate"] == 1.0 and per["SEC-01"]["overrides"] == 0
