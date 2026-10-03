"""The change feed and the audit log through the real Store server (inventory q13, q14)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from call1.contracts.events import AuditEvent

from .store_api_support import all_pages, error_code, poll, settled_call, worker_harness

pytestmark = pytest.mark.e2e


def _seq(cursor: str) -> int:
    return int(cursor.rsplit("-", 1)[1])


# --- q13: change feed ---------------------------------------------------------------------------


def test_q13_cursor_semantics_and_resync(stack, reviewer_session):
    start = reviewer_session.get("/changes", params={"limit": 1}).json()
    epoch, latest = start["feed_epoch"], start["latest_cursor"]
    assert latest.startswith(epoch + "-")

    receipt = settled_call(stack, "call_01_compliant", agent_id="agent-q13")
    after = reviewer_session.get("/changes", params={"after": latest, "limit": 500})
    assert after.status_code == 200, after.text
    feed = after.json()
    events = feed["events"]
    assert events and all(_seq(e["cursor"]) > _seq(latest) for e in events), "events after a cursor are strictly newer"
    assert [_seq(e["cursor"]) for e in events] == sorted(_seq(e["cursor"]) for e in events)
    mine = [e for e in events if e["call_id"] == receipt["call_id"]]
    assert {e["kind"] for e in mine} >= {"call", "result"}, sorted({e["kind"] for e in mine})
    assert "job" not in {e["kind"] for e in events}, "reviewers are not sent processing-job events"

    # Paging with limit=1 walks the same events one at a time.
    walked, cursor = [], latest
    for _ in range(len(events)):
        page = reviewer_session.get("/changes", params={"after": cursor, "limit": 1}).json()
        if not page["events"]:
            break
        walked.append(page["events"][0]["cursor"])
        cursor = page["next_cursor"]
    assert walked == [e["cursor"] for e in events][:len(walked)] and len(walked) == len(events), (walked, [e["cursor"] for e in events])

    # Process's key sees job events; a kinds filter narrows the feed.
    service = stack.store_get("/changes", session="service", params={"after": latest, "limit": 500, "kinds": "job"}).json()
    assert service["events"] and {e["kind"] for e in service["events"]} == {"job"}, {e["kind"] for e in service["events"]}

    # Unknown cursors are 410 cursor_unknown: another epoch, or ahead of the feed. The client resyncs
    # by reading without a cursor.
    foreign = reviewer_session.get("/changes", params={"after": "ep000000000000-000000000001"})
    assert foreign.status_code == 410 and error_code(foreign) == "cursor_unknown", (foreign.status_code, foreign.text)
    ahead = reviewer_session.get("/changes", params={"after": f"{epoch}-{_seq(feed['latest_cursor']) + 10_000:012d}"})
    assert ahead.status_code == 410 and error_code(ahead) == "cursor_unknown", (ahead.status_code, ahead.text)
    resync = reviewer_session.get("/changes", params={"limit": 5})
    assert resync.status_code == 200 and resync.json()["feed_epoch"] == epoch

    retention = stack.store_get("/contract").json()["parameters"]["change_feed_retention_seconds"]
    until = datetime.fromisoformat(feed["retention_until"].replace("Z", "+00:00"))
    assert abs((datetime.now(timezone.utc) - timedelta(seconds=retention) - until).total_seconds()) < 60, feed["retention_until"]


def test_q13_maintenance_prunes_old_events_and_old_cursors_expire(stack_factory):
    """The running Store's maintenance sweep prunes events past change_feed_retention_seconds (min
    3600 s, so the test ages this stack's events in store.db instead of waiting an hour); a cursor
    below the pruned horizon is then 410 cursor_expired with the oldest resumable cursor."""
    private = stack_factory(name="prune", with_process=False, store_env={"CALL1_STORE_MAINTENANCE_SECONDS": "1"})
    admin = private.admin()
    q = worker_harness(private)
    for _ in range(3):
        q.register()  # one 'call' change event each
    feed = admin.get("/changes", params={"limit": 500}).json()
    assert len(feed["events"]) >= 3, feed
    first_cursor = feed["events"][0]["cursor"]
    horizon = _seq(feed["latest_cursor"])
    oldest_before = admin.get("/changes", params={"after": first_cursor}).status_code
    assert oldest_before == 200

    db = sqlite3.connect(str(private.store_data / "store.db"), timeout=30)
    try:
        with db:
            db.execute("UPDATE change_events SET occurred_at = ? WHERE seq <= ?", ("2000-01-01T00:00:00.000000Z", horizon))
    finally:
        db.close()

    expired = poll(lambda: admin.get("/changes", params={"after": first_cursor}), lambda r: r.status_code != 200, timeout=15,
                   what="the maintenance sweep to prune aged events")
    assert expired.status_code == 410 and error_code(expired) == "cursor_expired", (expired.status_code, expired.text)
    oldest = expired.json()["details"]["oldest_cursor"]
    assert _seq(oldest) == horizon, (oldest, horizon)
    resumed = admin.get("/changes", params={"after": oldest})
    assert resumed.status_code == 200, resumed.text
    fresh = admin.get("/changes", params={"limit": 500}).json()
    assert all(_seq(e["cursor"]) > horizon for e in fresh["events"]), "pruned events are gone from a fresh read"


# --- q14: audit log -----------------------------------------------------------------------------


def test_q14_audit_events_are_hash_chained_and_cover_audited_actions(stack, admin_session, reviewer_session):
    receipt = settled_call(stack, "call_01_compliant", agent_id="agent-q14")
    request = reviewer_session.post(f"/calls/{receipt['call_id']}/reanalysis-requests", json={"kind": "summary"}, idempotency_key=True)
    assert request.status_code == 201, request.text
    request_id = request.json()["id"]

    found = admin_session.get("/admin/audit", params={"action": "reanalysis_requested", "target_id": receipt["call_id"]}).json()["items"]
    found = [e for e in found if e["details"].get("request_id") == request_id]
    assert len(found) == 1, found
    event = found[0]
    assert event["target"] == {"kind": "call", "id": receipt["call_id"]} and event["details"]["kind"] == "summary", event
    assert event["actor"]["kind"] == "reviewer" and event["actor"]["account_id"] == reviewer_session.account_id, event["actor"]

    events = sorted(all_pages(admin_session.get, "/admin/audit"), key=lambda e: e["sequence"])
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1)), "sequence is dense from 1"
    for previous, current in zip(events, events[1:]):
        assert current["previous_event_digest"] == previous["event_digest"], f"chain broken at sequence {current['sequence']}"
    assert events[0]["previous_event_digest"] is None
    for e in events:
        AuditEvent.model_validate(e)  # recomputes event_digest from the body

    assert reviewer_session.get("/admin/audit").status_code == 403
    assert stack.store_get("/admin/audit", session="service").status_code in (401, 403)
