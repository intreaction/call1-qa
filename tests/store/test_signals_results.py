"""Contact Signals v2 in the results area: projection, read-time alerts, reads, feedback, metrics and
the review queue (contract 1.3.0; docs/ContactSignalsV2.md sections 7.4, 9.2, 9.3 and 9.5; F2
acceptance "Projection", "Review queue", "Alerts", "Feedback", "Metrics", "Reads", "Areas").

The queue is the results tests' ``FakeQueue``: completions drive the real projection hooks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from call1.contracts.contents import VerdictStatus
from call1.contracts.jobs import JobType
from call1.store import db
from call1.store.results import projections, records, review_queue

from .test_results_harness import accounts, fq, transcript  # noqa: F401
from .test_signals_harness import (
    BILLING,
    CANCEL,
    TAXONOMY,
    UPSELL,
    V,
    hit,
    put_alert,
    put_taxonomy,
    reason_field,
    taxonomy_with,
    v1_content,
    v2_content,
)

CALLER_LINE = "I want to cancel my account, my card is 4111 1111 1111 1111"


@pytest.fixture
def taxonomy(client, admin_session):
    put_taxonomy(client, admin_session, TAXONOMY)
    return TAXONOMY


def _call(fq, *, text: bool = True):
    conv = fq.register()
    graph = fq.graph(conv, [JobType.ASR, JobType.CONTACT_SIGNALS_MERGE])
    if text:
        fq.complete(conv, graph, JobType.ASR, {"transcript": transcript([("AGENT", "Thanks for calling."), ("CALLER", CALLER_LINE)])})
    return conv


def _publish(fq, conv, content):
    graph = fq.graph(conv, [JobType.CONTACT_SIGNALS_MERGE])
    return fq.complete(conv, graph, JobType.CONTACT_SIGNALS_MERGE, {"contact_signals": content})


def _cancel_hit(sub="cancel_account", reason="price", **kw):
    return hit(TAXONOMY, "intent", 1, sub=sub, quote=CALLER_LINE, fields=[reason_field(reason)] if sub == "cancel_account" else [], **kw)


def _view(client, session, conv):
    response = client.get(f"{V}/calls/{conv.call_id}/contact-signals", headers=session.read_headers)
    assert response.status_code == 200, response.text
    return response.json()


def _events(client, session, kind):
    return [e for e in client.get(f"{V}/changes", headers=session.read_headers).json()["events"] if e["kind"] == kind]


# --- projection -------------------------------------------------------------------------------------


def test_v1_and_v2_results_both_project_without_text(fq, store, taxonomy):
    v1_conv = _call(fq)
    _publish(fq, v1_conv, v1_content([("intent", 1, CALLER_LINE), ("issue", 1, "fee on my card 4111 1111 1111 1111")]))
    v2_conv = _call(fq)
    _publish(fq, v2_conv, v2_content(taxonomy, 2, [_cancel_hit(), hit(taxonomy, "upsell", 0, quote="Would you like premium?", speaker="AGENT")]))
    with store.connection() as conn:
        v1_hits = conn.execute("SELECT category_id, subcategory_id FROM results_signal_hits WHERE call_id = ? ORDER BY category_id", (v1_conv.call_id,)).fetchall()
        assert [(r["category_id"], r["subcategory_id"]) for r in v1_hits] == [("intent", None), ("issue", None)]
        v2_hits = conn.execute("SELECT category_id, subcategory_id FROM results_signal_hits WHERE call_id = ? ORDER BY category_id", (v2_conv.call_id,)).fetchall()
        assert [(r["category_id"], r["subcategory_id"]) for r in v2_hits] == [("intent", "cancel_account"), ("upsell", None)]
        [field] = conn.execute("SELECT * FROM results_signal_hit_fields WHERE call_id = ?", (v2_conv.call_id,)).fetchall()
        assert (field["field_id"], field["status"], field["value_enum"]) == ("reason", "extracted", "price")
        outcomes = {r["call_id"]: (r["pipeline"], r["taxonomy_version"], r["hit_count"]) for r in conn.execute("SELECT * FROM results_signal_outcomes")}
        assert outcomes == {v1_conv.call_id: ("v1", None, 2), v2_conv.call_id: ("v2", 2, 2)}
        dump = json.dumps([dict(r) for table in ("results_signal_hits", "results_signal_hit_fields", "results_signal_outcomes")
                           for r in conn.execute(f"SELECT * FROM {table}")])
    for text in ("cancel my account", "4111", "premium", "fee on"):
        assert text not in dump


def test_projection_is_replay_safe_and_keeps_the_newest_version(fq, store, taxonomy):
    conv = _call(fq)
    first = _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit()]))
    second = _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit(), hit(taxonomy, "upsell", 0, speaker="AGENT", quote="Premium?")]))
    with store.connection() as conn, db.transaction(conn):
        pub = records.publication(conn, conv.id, records.ResultKind.CONTACT_SIGNALS, first)
        projections._publish_signals(conn, conv.call_id, first, pub.checksum)  # an out-of-order replay of v1
        assert conn.execute("SELECT signals_version FROM results_calls WHERE call_id = ?", (conv.call_id,)).fetchone()[0] == second
        assert conn.execute("SELECT COUNT(*) FROM results_signal_outcomes WHERE call_id = ?", (conv.call_id,)).fetchone()[0] == 2
        rows = conn.execute("SELECT COUNT(*) FROM results_signal_hits WHERE call_id = ? AND signals_version = ?", (conv.call_id, second)).fetchone()[0]
        assert rows == 2


def test_project_signals_projects_results_published_before_1_3_0(fq, store, taxonomy):
    conv = _call(fq)
    _publish(fq, conv, v1_content())
    with store.connection() as conn, db.transaction(conn):
        conn.execute("DELETE FROM results_signal_hits")
        conn.execute("UPDATE results_calls SET signals_version = NULL")
    with store.connection() as conn:
        assert projections.project_existing_signals(conn) == 1
        assert projections.project_existing_signals(conn) == 0
        assert conn.execute("SELECT COUNT(*) FROM results_signal_hits").fetchone()[0] == 1


# --- reads -------------------------------------------------------------------------------------------


def test_reads_are_masked_and_carry_the_read_time_context(fq, client, taxonomy, reviewer_session, admin_session):
    conv = _call(fq)
    string_field = {"field_id": "reason", "type": "enum", "status": "extracted", "value": "price", "name": "Reason",
                    "evidence": CALLER_LINE, "char_start": 0, "char_end": len(CALLER_LINE)}
    _publish(fq, conv, v2_content(taxonomy, 2, [hit(taxonomy, "intent", 1, sub="cancel_account", quote=CALLER_LINE, fields=[string_field])]))
    view = _view(client, reviewer_session, conv)
    [signal] = view["signals"]
    assert "4111" not in signal["quote"] and "[REDACTED]" in signal["quote"] and "cancel my account" in signal["quote"]
    assert "4111" not in signal["fields"][0]["evidence"] and signal["fields"][0]["value"] == "price"
    assert view["text_withheld"] is False and view["pipeline"] == "v2"
    assert view["taxonomy_status"] == {"scored_version": 2, "current_version": 2, "outdated_stages": [], "thresholds_changed": False}
    assert view["feedback"] == [] and view["alerts"] == [] and view["comparison_preview_id"] is None
    # A new subcategory outdates stage 2 only; a threshold edit only re-derives.
    put_taxonomy(client, admin_session, taxonomy_with(intent_subs=[CANCEL, BILLING, BILLING.model_copy(update={
        "subcategory_id": "store_hours", "name": "Store hours", "gloss": "Asks when a store is open"})]))
    assert _view(client, reviewer_session, conv)["taxonomy_status"] == {"scored_version": 2, "current_version": 3,
                                                                          "outdated_stages": ["subcategorize"], "thresholds_changed": False}
    put_taxonomy(client, admin_session, taxonomy_with(intent_subs=[CANCEL, BILLING, BILLING.model_copy(update={
        "subcategory_id": "store_hours", "name": "Store hours", "gloss": "Asks when a store is open"})], intent_threshold=0.3))
    status = _view(client, reviewer_session, conv)["taxonomy_status"]
    assert status["thresholds_changed"] is True and status["current_version"] == 4


def test_text_is_withheld_while_pii_findings_are_pending(fq, client, taxonomy, reviewer_session):
    fq.auto_pii = False
    conv = _call(fq)
    _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit()]))
    view = _view(client, reviewer_session, conv)
    assert view["text_withheld"] is True and view["signals"][0]["quote"] == "[REDACTED]"
    assert view["signals"][0]["subcategory_label"] == "Cancel account" and view["signals"][0]["fields"][0]["value"] == "price"
    v1_conv = _call(fq)
    _publish(fq, v1_conv, v1_content())
    assert _view(client, reviewer_session, v1_conv)["signals"][0]["quote"] == "[REDACTED]"


def test_a_multi_segment_hit_counts_once_and_its_part_quotes_are_masked_on_read(fq, client, store, taxonomy, reviewer_session, admin_session):
    """Decision 25 (section 6.5): the anchor's row carries the whole span; part quotes are masked too."""
    from call1.contracts.contents import SignalHitPart

    conv = _call(fq)
    anchor = hit(taxonomy, "upsell", 0, quote="Would you like premium?", speaker="AGENT", start=0.5, end=3.0, confidence=0.6)
    part = SignalHitPart(turn_id=2, block=0, start=9.0, end=12.5, quote=CALLER_LINE, char_start=0, char_end=len(CALLER_LINE))
    merged = anchor.model_copy(update={"parts": [part], "span_end": 12.5, "confidence": 0.8})
    _publish(fq, conv, v2_content(taxonomy, 2, [merged, _cancel_hit()]))
    with store.connection() as conn:
        rows = conn.execute("SELECT hit_id, start, \"end\", confidence FROM results_signal_hits WHERE call_id = ? AND category_id = 'upsell'",
                            (conv.call_id,)).fetchall()
        assert [(r["hit_id"], r["start"], r["end"], r["confidence"]) for r in rows] == [(anchor.id, 0.5, 12.5, 0.8)]
        assert conn.execute("SELECT hit_count FROM results_signal_outcomes WHERE call_id = ?", (conv.call_id,)).fetchone()[0] == 2
    view = _view(client, reviewer_session, conv)
    upsell = next(s for s in view["signals"] if s["category_id"] == "upsell")
    assert upsell["id"] == anchor.id and upsell["span_end"] == 12.5 and len(upsell["parts"]) == 1
    [masked] = upsell["parts"]
    assert "4111" not in masked["quote"] and "[REDACTED]" in masked["quote"] and "cancel my account" in masked["quote"]
    assert (masked["turn_id"], masked["char_start"], masked["char_end"]) == (2, 0, len(CALLER_LINE))
    # While the findings are pending, part quotes are withheld like the anchor's.
    fq.auto_pii = False
    pending = _call(fq)
    _publish(fq, pending, v2_content(taxonomy, 2, [merged]))
    [withheld] = _view(client, reviewer_session, pending)["signals"]
    assert withheld["quote"] == "[REDACTED]" and withheld["parts"][0]["quote"] == "[REDACTED]"


def test_feedback_on_a_hit_that_later_becomes_a_part_stays_readable(fq, client, taxonomy, reviewer_session):
    """Decision 25: a hit judged while separate, then merged into another hit's parts on a rescore,
    keeps its feedback in the view under its own (earlier) ID, beside the anchor's."""
    from call1.contracts.contents import SignalHitPart

    conv = _call(fq)
    first = hit(taxonomy, "upsell", 0, quote="Would you like premium?", speaker="AGENT", start=0.5, end=3.0)
    second = hit(taxonomy, "upsell", 2, quote="Premium adds priority support.", speaker="AGENT", start=9.0, end=12.5)
    _publish(fq, conv, v2_content(taxonomy, 2, [first, second]))
    saved = client.put(f"{V}/calls/{conv.call_id}/signal-hits/{second.id}/feedback",
                       json={"category_verdict": "dismissed", "expected_feedback_version": 0}, headers=reviewer_session.headers)
    assert saved.status_code == 200, saved.text
    part = SignalHitPart(turn_id=2, block=0, start=9.0, end=12.5, quote=second.quote, char_start=0, char_end=len(second.quote))
    merged = first.model_copy(update={"parts": [part], "span_end": 12.5})
    _publish(fq, conv, v2_content(taxonomy, 2, [merged]))
    view = _view(client, reviewer_session, conv)
    assert [s["id"] for s in view["signals"]] == [first.id]
    assert second.id == re.sub(r"\.t\d+b\d+$", ".t2b0", first.id)
    assert [(f["hit_id"], f["category_verdict"]) for f in view["feedback"]] == [(second.id, "dismissed")]


# --- alerts ------------------------------------------------------------------------------------------


def test_alerts_are_evaluated_at_read_time_and_fire_only_on_a_new_match(fq, client, taxonomy, admin_session, reviewer_session):
    put_alert(client, admin_session, "cancel-price", {"category_id": "intent", "subcategory_id": "cancel_account", "field_id": "reason",
                                                      "field_equals": "price"}, name="Cancel for price")
    conv = _call(fq)
    _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit()]))
    view = _view(client, reviewer_session, conv)
    assert view["alerts"] == [{"rule_id": "cancel-price", "name": "Cancel for price", "hit_ids": [view["signals"][0]["id"]]}]
    fired = _events(client, admin_session, "signal_alert")
    assert [(e["status"], e["resource_id"], e["call_id"]) for e in fired] == [("fired:cancel-price", conv.call_id, conv.call_id)]
    # A republish that still matches fires nothing new.
    _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit()]))
    assert len(_events(client, admin_session, "signal_alert")) == 1
    # A rule edit applies at once to reads, runs no model and fires no event for older calls.
    other = _call(fq)
    _publish(fq, other, v2_content(taxonomy, 2, [_cancel_hit(reason="moving")]))
    assert _view(client, reviewer_session, other)["alerts"] == []
    put_alert(client, admin_session, "cancel-price", {"category_id": "intent", "subcategory_id": "cancel_account"}, name="Cancel")
    assert [a["rule_id"] for a in _view(client, reviewer_session, other)["alerts"]] == ["cancel-price"]
    assert len(_events(client, admin_session, "signal_alert")) == 1
    # A later publish that newly matches fires.
    _publish(fq, other, v2_content(taxonomy, 2, [_cancel_hit(reason="moving")]))
    assert len(_events(client, admin_session, "signal_alert")) == 1  # it already matched at the previous version
    # min_confidence and a disabled rule.
    put_alert(client, admin_session, "sure", {"category_id": "intent", "min_confidence": 0.95})
    assert [a["rule_id"] for a in _view(client, reviewer_session, conv)["alerts"]] == ["cancel-price"]
    put_alert(client, admin_session, "cancel-price", {"category_id": "intent", "subcategory_id": "cancel_account"}, enabled=False)
    assert _view(client, reviewer_session, conv)["alerts"] == []


def test_call_list_fields_and_filters(fq, client, taxonomy, admin_session, reviewer_session):
    put_alert(client, admin_session, "any-upsell", {"category_id": "upsell"})
    a = _call(fq)
    _publish(fq, a, v2_content(taxonomy, 2, [_cancel_hit(), hit(taxonomy, "upsell", 0, speaker="AGENT", quote="Premium?")]))
    b = _call(fq)
    _publish(fq, b, v1_content([("issue", 1, "a fee")]))
    c = _call(fq)  # nothing published yet
    items = {i["call_id"]: i for i in client.get(f"{V}/calls", headers=reviewer_session.read_headers).json()["items"]}
    assert items[a.call_id]["signal_categories"] == ["intent", "upsell"] and items[a.call_id]["caller_needs"] == ["cancel_account"]
    assert items[a.call_id]["signal_alerts"] == ["any-upsell"] and items[a.call_id]["contact_signals_state"] == "available"
    assert items[b.call_id]["signal_categories"] == ["issue"] and items[b.call_id]["caller_needs"] == []
    assert items[c.call_id]["signal_categories"] == [] and items[c.call_id]["contact_signals_state"] == "pending"

    def ids(**params):
        return {i["call_id"] for i in client.get(f"{V}/calls", params=params, headers=reviewer_session.read_headers).json()["items"]}

    assert ids(signal_category="intent") == {a.call_id}
    assert ids(signal_category="issue") == {b.call_id}
    assert ids(signal_category="intent", signal_subcategory="cancel_account") == {a.call_id}
    assert ids(signal_subcategory="billing_question") == set()
    assert ids(signal_alert="any-upsell") == {a.call_id}
    assert ids(signal_alert="no-such-rule") == set()
    # Retiring a custom category hides its chips and stops its rule matching.
    put_taxonomy(client, admin_session, taxonomy_with(custom=[UPSELL.model_copy(update={"active": False})]))
    items = {i["call_id"]: i for i in client.get(f"{V}/calls", headers=reviewer_session.read_headers).json()["items"]}
    assert items[a.call_id]["signal_categories"] == ["intent"] and items[a.call_id]["signal_alerts"] == []


# --- feedback ------------------------------------------------------------------------------------------


def test_feedback_survives_threshold_edits_and_sibling_additions_and_the_subcategory_verdict_detaches(fq, client, taxonomy, admin_session,
                                                                                                     reviewer_session):
    conv = _call(fq)
    _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit()]))
    hit_id = _view(client, reviewer_session, conv)["signals"][0]["id"]
    url = f"{V}/calls/{conv.call_id}/signal-hits/{hit_id}/feedback"
    saved = client.put(url, json={"category_verdict": "confirmed", "subcategory_verdict": "confirmed", "expected_feedback_version": 0},
                       headers=reviewer_session.headers)
    assert saved.status_code == 200, saved.text
    assert saved.json()["subcategory_id"] == "cancel_account" and saved.json()["feedback_version"] == 1
    stale = client.put(url, json={"category_verdict": "dismissed", "expected_feedback_version": 0}, headers=reviewer_session.headers)
    assert stale.status_code == 409 and stale.json()["details"]["current_version"] == 1
    [event] = _events(client, admin_session, "review")[-1:]
    assert event["status"] == "signal_feedback" and event["call_id"] == conv.call_id
    # A threshold edit and a new sibling, then a republish: same hit ID, the feedback still attaches.
    edited = taxonomy_with(intent_subs=[CANCEL, BILLING, BILLING.model_copy(update={"subcategory_id": "store_hours", "name": "Store hours",
                                                                                   "gloss": "Asks when a store is open"})], intent_threshold=0.3)
    put_taxonomy(client, admin_session, edited)
    _publish(fq, conv, v2_content(edited, 3, [hit(edited, "intent", 1, sub="cancel_account", quote=CALLER_LINE, fields=[reason_field()])]))
    view = _view(client, reviewer_session, conv)
    assert view["signals"][0]["id"] == hit_id and [f["hit_id"] for f in view["feedback"]] == [hit_id]
    # Stage 2 now says billing: the old subcategory verdict names cancel_account, which the client shows as detached.
    _publish(fq, conv, v2_content(edited, 3, [hit(edited, "intent", 1, sub="billing_question", quote=CALLER_LINE)]))
    view = _view(client, reviewer_session, conv)
    assert view["signals"][0]["subcategory_id"] == "billing_question" and view["feedback"][0]["subcategory_id"] == "cancel_account"
    metric = client.get(f"{V}/metrics/signals", headers=reviewer_session.read_headers).json()["categories"][0]
    billing = next(s for s in metric["subcategories"] if s["id"] == "billing_question")
    assert (billing["confirmed"], billing["dismissed"]) == (0, 0) and metric["confirmed"] == 1
    # Corrections name a subcategory of the category; a v1 hit has none to judge.
    bad = client.put(url, json={"subcategory_verdict": "corrected", "corrected_subcategory_id": "nope", "expected_feedback_version": 1},
                     headers=reviewer_session.headers)
    assert bad.status_code == 422
    ok = client.put(url, json={"subcategory_verdict": "corrected", "corrected_subcategory_id": "other", "expected_feedback_version": 1},
                    headers=reviewer_session.headers)
    assert ok.status_code == 200 and ok.json()["subcategory_id"] == "billing_question"
    assert client.put(f"{V}/calls/{conv.call_id}/signal-hits/intent.nope/feedback", json={"expected_feedback_version": 0},
                      headers=reviewer_session.headers).status_code == 404


# --- metrics -----------------------------------------------------------------------------------------


def test_signal_metrics_match_a_hand_computed_fixture(fq, client, taxonomy, admin_session, reviewer_session):
    put_alert(client, admin_session, "cancel-price", {"category_id": "intent", "subcategory_id": "cancel_account", "field_id": "reason",
                                                      "field_equals": "price"}, name="Cancel for price")
    a, b, c, d = (_call(fq) for _ in range(4))
    _publish(fq, a, v2_content(taxonomy, 2, [_cancel_hit(), hit(taxonomy, "intent", 1, block=1, sub="billing_question", quote="bill")]))
    _publish(fq, b, v2_content(taxonomy, 2, [_cancel_hit(reason="moving"), hit(taxonomy, "upsell", 0, speaker="AGENT", quote="Premium?")]))
    _publish(fq, c, v1_content([("intent", 1, "cancel"), ("issue", 1, "fee")]))
    _publish(fq, d, v2_content(taxonomy, 2, [hit(taxonomy, "intent", 1, sub="other", quote="courier")]))
    intent_hits = [(conv, h["id"]) for conv in (a, b, c, d) for h in _view(client, reviewer_session, conv)["signals"] if h["category_id"] in (None, "intent")
                   and (h["category_id"] or h["kind"]) == "intent"]
    assert len(intent_hits) == 5
    for n, (conv, hit_id) in enumerate(intent_hits):
        verdict = "dismissed" if n == 0 else "confirmed"
        saved = client.put(f"{V}/calls/{conv.call_id}/signal-hits/{hit_id}/feedback", json={"category_verdict": verdict, "expected_feedback_version": 0},
                           headers=reviewer_session.headers)
        assert saved.status_code == 200, saved.text
    body = client.get(f"{V}/metrics/signals", headers=reviewer_session.read_headers)
    assert body.status_code == 200, body.text
    m = body.json()
    assert (m["calls_scored"], m["calls_with_signal"], m["calls_by_pipeline"]) == (4, 4, {"v1": 1, "v2": 3})
    intent = next(x for x in m["categories"] if x["category_id"] == "intent")
    assert (intent["calls_scored"], intent["calls_with_hit"], intent["hit_rate_pct"], intent["hits_total"]) == (4, 4, 100.0, 5)
    assert (intent["confirmed"], intent["dismissed"], intent["precision_pct"]) == (4, 1, 80.0)
    assert [(s["id"], s["calls_with_hit"], s["hits"], s["share_pct"]) for s in intent["subcategories"]] == [
        ("cancel_account", 2, 2, 50.0), ("billing_question", 1, 1, 25.0), ("other", 1, 1, 25.0)]
    assert intent["subcategory_precision_pct"] is None  # no subcategory verdicts
    assert m["top_caller_needs"] == intent["subcategories"]
    [reason] = intent["fields"]
    assert reason["node_id"] == "cancel_account" and {v["value"]: v["count"] for v in reason["values"]} == {
        "price": 1, "service quality": 0, "moving": 1, "other": 0}
    upsell = next(x for x in m["categories"] if x["category_id"] == "upsell")
    assert (upsell["calls_scored"], upsell["calls_with_hit"], upsell["hit_rate_pct"], upsell["precision_pct"]) == (3, 1, 33.3, None)
    issue = next(x for x in m["categories"] if x["category_id"] == "issue")
    assert (issue["calls_with_hit"], issue["hit_rate_pct"]) == (1, 25.0)
    [alert] = m["alerts"]
    assert (alert["rule_id"], alert["calls_matched"], alert["match_rate_pct"]) == ("cancel-price", 1, 25.0)
    assert sum(day["calls_scored"] for day in intent["by_day"]) == 4
    assert intent["top_agents"] == [{"agent_id": "agent-7", "agent_display_name": None, "agent_extension": None, "calls_with_hit": 4}]
    only = client.get(f"{V}/metrics/signals", params={"category_id": "upsell"}, headers=reviewer_session.read_headers).json()
    assert [x["category_id"] for x in only["categories"]] == ["upsell"] and only["top_caller_needs"] == intent["subcategories"]


def test_precision_is_null_under_five_judged_hits(fq, client, taxonomy, reviewer_session):
    conv = _call(fq)
    _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit()]))
    hit_id = _view(client, reviewer_session, conv)["signals"][0]["id"]
    client.put(f"{V}/calls/{conv.call_id}/signal-hits/{hit_id}/feedback", json={"category_verdict": "confirmed", "subcategory_verdict": "confirmed",
                                                                             "expected_feedback_version": 0}, headers=reviewer_session.headers)
    intent = client.get(f"{V}/metrics/signals", headers=reviewer_session.read_headers).json()["categories"][0]
    assert intent["confirmed"] == 1 and intent["precision_pct"] is None and intent["subcategories"][0]["precision_pct"] is None


# --- the review queue ---------------------------------------------------------------------------------


def _signal_rule(client, supervisor_session, *, alerts=("cancel",), stream="SIGNAL", rule_id="rule-signal-cancel", **extra):
    rule = {"id": rule_id, "name": "Cancellations", "stream": stream, "rank": 5, "target_signal_alerts": list(alerts), **extra}
    saved = client.put(f"{V}/review-queue/rules/{rule_id}", json={"rule": rule, "expected_rule_version": 0}, headers=supervisor_session.headers)
    assert saved.status_code == 200, saved.text


def _items(store, call_id):
    with store.connection() as conn:
        return [(r["rule_id"], r["evaluation_version"], r["status"], r["signals_version"], json.loads(r["trigger_alert_rule_ids_json"]))
                for r in conn.execute("SELECT * FROM results_review_items WHERE call_id = ? AND rule_id = 'rule-signal-cancel' "
                                      "ORDER BY evaluation_version, created_at", (call_id,))]


def test_signal_items_need_an_evaluation_and_every_publish_order_yields_one_item_per_version(fq, client, store, taxonomy, admin_session,
                                                                                            supervisor_session):
    put_alert(client, admin_session, "cancel", {"category_id": "intent", "subcategory_id": "cancel_account"}, name="Cancel account")
    _signal_rule(client, supervisor_session)
    # Order 1: signals before QA. No evaluation: no item yet (decision 22, Q2); QA's publish creates it.
    first = _call(fq)
    signals_version = _publish(fq, first, v2_content(taxonomy, 2, [_cancel_hit()]))
    assert _items(store, first.call_id) == []
    _, qa1 = fq.ingest_qa(first, [("REG-01", VerdictStatus.PASS, 0.95)])
    assert _items(store, first.call_id) == [("rule-signal-cancel", qa1, "PENDING", signals_version, ["cancel"])]
    _publish(fq, first, v2_content(taxonomy, 2, [_cancel_hit()]))  # signals republish at the same evaluation: no duplicate
    assert len(_items(store, first.call_id)) == 1
    # Order 2: QA N, then signals (item at N), then a replayed QA N, then QA N+1.
    second = _call(fq)
    _, n = fq.ingest_qa(second, [("REG-01", VerdictStatus.PASS, 0.95)])
    assert _items(store, second.call_id) == []
    _publish(fq, second, v2_content(taxonomy, 2, [_cancel_hit()]))
    [item] = _items(store, second.call_id)
    assert item[:3] == ("rule-signal-cancel", n, "PENDING")
    with store.connection() as conn, db.transaction(conn):
        row = records.call_row(conn, second.call_id)
        pub = records.publication(conn, second.id, records.ResultKind.QA, n)
        projections._publish_qa(conn, row, n, pub.checksum, db.ts(conn.now()))  # replayed QA N: nothing new
    assert len(_items(store, second.call_id)) == 1
    _, n1 = fq.ingest_qa(second, [("REG-01", VerdictStatus.PASS, 0.95)])
    items = _items(store, second.call_id)
    assert [(i[1], i[2]) for i in items] == [(n, "SUPERSEDED"), (n1, "PENDING")]
    with store.connection() as conn:
        counts = conn.execute("SELECT evaluation_version, COUNT(*) AS c FROM results_review_items WHERE call_id = ? AND rule_id = ? "
                              "GROUP BY evaluation_version", (second.call_id, "rule-signal-cancel")).fetchall()
        assert {r["evaluation_version"]: r["c"] for r in counts} == {n: 1, n1: 1}
    # The item reads its trigger and the call's signals version. Select it by rule: call ids are random, so the
    # seeded 10% AUDIT_SAMPLE rule (040_results.sql) can also draw this confident pass and list an item at n1.
    listed = client.get(f"{V}/review-queue", params={"call_id": second.call_id}, headers=supervisor_session.read_headers).json()["items"]
    [current] = [i for i in listed if i["evaluation_version"] == n1 and i["rule_id"] == "rule-signal-cancel"]
    assert current["stream"] == "SIGNAL" and current["trigger_alert_rule_ids"] == ["cancel"] and current["signals_version"] is not None
    assert current["reason"] == "Signal: Cancel account (caller, 0:02)" and current["urgency_score"] == 50.0


def test_an_item_whose_alert_stops_matching_stays(fq, client, store, taxonomy, admin_session, supervisor_session):
    put_alert(client, admin_session, "cancel", {"category_id": "intent", "subcategory_id": "cancel_account"})
    _signal_rule(client, supervisor_session)
    conv = _call(fq)
    fq.ingest_qa(conv, [("REG-01", VerdictStatus.PASS, 0.95)])
    _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit()]))
    _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit(sub="billing_question")]))
    [item] = _items(store, conv.call_id)
    assert item[2] == "PENDING"  # "Signals changed since this item was created" is the client's comparison of signals_version


def test_target_signal_alerts_narrow_other_streams(fq, client, store, taxonomy, admin_session, supervisor_session):
    put_alert(client, admin_session, "cancel", {"category_id": "intent", "subcategory_id": "cancel_account"})
    _signal_rule(client, supervisor_session, stream="CALIBRATION", rule_id="rule-signal-cancel")
    matched, unmatched = _call(fq), _call(fq)
    for conv, sub in ((matched, "cancel_account"), (unmatched, "billing_question")):
        _publish(fq, conv, v2_content(taxonomy, 2, [_cancel_hit(sub=sub)]))
        fq.ingest_qa(conv, [("REG-01", VerdictStatus.PASS, 0.95)])
    assert len(_items(store, matched.call_id)) == 1 and _items(store, unmatched.call_id) == []


def test_rule_matching_is_pure():
    facts = review_queue.ScoreFacts(call_id="c", conversation_id="v", evaluation_version=1, agent_id="a", overall_score=90, critical_failure=False,
                                    low_confidence=False, dispute_signal=False, duration_seconds=60, rubric_category=None, signal_alerts=frozenset({"x"}))
    rule = review_queue.ReviewQueueRule(id="r", name="R", stream="SIGNAL", target_signal_alerts=["x", "y"])
    assert review_queue.rule_matches(rule, facts) and review_queue.triggered_alerts(rule, facts) == ["x"]
    assert not review_queue.rule_matches(rule.model_copy(update={"target_signal_alerts": ["y"]}), facts)


# --- areas -------------------------------------------------------------------------------------------


def test_queue_code_reads_the_taxonomy_only_through_results_api():
    queue = Path(__file__).resolve().parents[2] / "call1" / "store" / "queue"
    for path in queue.glob("*.py"):
        if " 2." in path.name:
            continue
        text = path.read_text(encoding="utf-8")
        assert "results_signal" not in text, path.name
        assert not re.search(r"from \.\.results(\.| import )(?!api\b|projections\b)", text), path.name
        assert "signal_store" not in text, path.name
