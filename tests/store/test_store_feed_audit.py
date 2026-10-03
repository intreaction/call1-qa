"""The change feed (cursors, epochs, principal filtering) and the hash-chained audit log."""

from __future__ import annotations

import pytest

from call1.contracts.common import ServiceScope
from call1.contracts.events import Actor, ActorKind, AuditAction, AuditQuery, ChangeFeed, ChangeKind
from call1.store import audit, db, feed
from call1.store.errors import StoreError

K = ChangeKind


def _append(conn, *events):
    cursors = []
    with db.transaction(conn):
        for kind, resource in events:
            cursors.append(feed.append(conn, kind, resource, version=1, status="queued", conversation_id="conv_1"))
    return cursors


def test_append_needs_the_changing_transaction(conn):
    with pytest.raises(RuntimeError, match="inside the transaction"):
        feed.append(conn, K.JOB, "job_1")


def test_cursors_are_ordered_and_resumable(conn):
    cursors = _append(conn, (K.JOB, "job_1"), (K.CALL, "call_1"), (K.JOB, "job_2"))
    assert cursors == sorted(cursors) and len(set(cursors)) == 3
    everything = feed.read(conn, audience="admin", after=None, limit=200, kinds=None, retention_seconds=3600)
    assert [e.resource_id for e in everything.events] == ["job_1", "call_1", "job_2"]
    assert everything.next_cursor == everything.latest_cursor == cursors[-1]
    first = feed.read(conn, audience="admin", after=None, limit=2, kinds=None, retention_seconds=3600)
    assert first.next_cursor == cursors[1]
    rest = feed.read(conn, audience="admin", after=first.next_cursor, limit=2, kinds=None, retention_seconds=3600)
    assert [e.resource_id for e in rest.events] == ["job_2"]
    filtered = feed.read(conn, audience="admin", after=None, limit=200, kinds=[K.RUBRIC], retention_seconds=3600)
    assert filtered.events == [] and filtered.next_cursor == cursors[-1]  # a filtered reader never stalls


def test_feed_is_filtered_by_principal(conn):
    _append(conn, (K.JOB, "job_1"), (K.CALL, "call_1"), (K.ADMIN_STATE, "state"))
    seen = lambda audience: [e.kind for e in feed.read(conn, audience=audience, after=None, limit=200, kinds=None, retention_seconds=3600).events]  # noqa: E731
    assert seen("reviewer") == [K.CALL]
    assert seen("supervisor") == [K.JOB, K.CALL]
    assert seen("process_service_key") == [K.JOB, K.ADMIN_STATE]
    assert seen("admin") == [K.JOB, K.CALL, K.ADMIN_STATE]
    asked = feed.read(conn, audience="reviewer", after=None, limit=200, kinds=[K.JOB], retention_seconds=3600)
    assert asked.events == []  # asking for a kind outside the principal's list returns none of it, never 403


def test_foreign_ahead_and_expired_cursors(conn, clock):
    cursors = _append(conn, (K.JOB, "job_1"))
    epoch = feed.feed_epoch(conn)
    for cursor in (feed.format_cursor("ep000000000000", 1), feed.format_cursor(epoch, 99), "garbage"):
        with pytest.raises(StoreError) as unknown:
            feed.read(conn, audience="admin", after=cursor, limit=10, kinds=None, retention_seconds=3600)
        assert unknown.value.code.value == "cursor_unknown" and unknown.value.details == {"feed_epoch": epoch}
    clock.advance(seconds=7200)
    _append(conn, (K.JOB, "job_2"))
    assert feed.prune(conn, retention_seconds=3600) == 1
    with pytest.raises(StoreError) as expired:
        feed.read(conn, audience="admin", after=feed.format_cursor(epoch, 0), limit=10, kinds=None, retention_seconds=3600)
    assert expired.value.code.value == "cursor_expired" and expired.value.details["oldest_cursor"] == cursors[0]
    assert [e.resource_id for e in feed.read(conn, audience="admin", after=None, limit=10, kinds=None, retention_seconds=3600).events] == ["job_2"]
    new_epoch = feed.start_new_epoch(conn)
    with pytest.raises(StoreError):
        feed.read(conn, audience="admin", after=cursors[0], limit=10, kinds=None, retention_seconds=3600)
    assert feed.latest_cursor(conn).startswith(new_epoch)


def test_changes_route_filters_by_the_callers_principal(client, conn, reviewer_session, mint_service_key):
    _append(conn, (K.JOB, "job_1"), (K.CALL, "call_1"))
    as_reviewer = ChangeFeed.model_validate(client.get("/store/v1/changes", headers=reviewer_session.read_headers).json())
    assert [e.resource_id for e in as_reviewer.events] == ["call_1"]
    key = mint_service_key([ServiceScope.CHANGES_READ])
    as_process = client.get("/store/v1/changes", params={"kinds": ["job", "call"]}, headers=key.headers).json()
    assert [e["resource_id"] for e in as_process["events"]] == ["job_1"]
    stale = client.get("/store/v1/changes", params={"after": "ep000000000000-000000000001"}, headers=key.headers)
    assert stale.status_code == 410 and stale.json()["code"] == "cursor_unknown"
    assert client.get("/store/v1/changes").status_code == 401


def _event(conn, action=AuditAction.JOB_CANCELLED, target="job_1", **details):
    with db.transaction(conn):
        return audit.append(conn, actor=Actor(kind=ActorKind.PROCESS_SERVICE, installation_id="inst_1"), action=action,
                            target_kind="job", target_id=target, details=details)


def test_audit_log_is_hash_chained(conn):
    with pytest.raises(RuntimeError):
        audit.append(conn, actor=audit.SYSTEM_ACTOR, action=AuditAction.JOB_RETRIED, target_kind="job", target_id="j")
    first = _event(conn, reason="operator")
    second = _event(conn, action=AuditAction.JOB_RETRIED)
    assert (first.sequence, second.sequence) == (1, 2)
    assert first.previous_event_digest is None and second.previous_event_digest == first.event_digest
    assert audit.verify_chain(conn)
    with db.transaction(conn):
        conn.execute("UPDATE audit_events SET details_json = ? WHERE sequence = 1", ('{"reason":"forged"}',))
    assert not audit.verify_chain(conn)


def test_audit_listing_and_route(client, conn, admin_session, supervisor_session):
    for n in range(3):
        _event(conn, target=f"job_{n}")
    _event(conn, action=AuditAction.JOB_RETRIED, target="job_x")
    page = audit.list_events(conn, AuditQuery(limit=2))
    assert [e.sequence for e in page.items] == [4, 3] and page.next_page_token
    rest = audit.list_events(conn, AuditQuery(limit=2, page_token=page.next_page_token))
    assert [e.sequence for e in rest.items] == [2, 1] and rest.next_page_token is None
    assert [e.target.id for e in audit.list_events(conn, AuditQuery(action=AuditAction.JOB_RETRIED)).items] == ["job_x"]
    body = client.get("/store/v1/admin/audit", params={"target_id": "job_1"}, headers=admin_session.read_headers).json()
    assert [e["target"]["id"] for e in body["items"]] == ["job_1"]
    denied = client.get("/store/v1/admin/audit", headers=supervisor_session.read_headers)
    assert denied.status_code == 403 and denied.json()["code"] == "insufficient_role"
    bad = client.get("/store/v1/admin/audit", params={"page_token": "!!"}, headers=admin_session.read_headers)
    assert bad.status_code == 422 and bad.json()["code"] == "validation_failed"


def test_audit_actor_for_principals(reviewer_session, service_key):
    reviewer = audit.actor_for(reviewer_session.principal)
    assert reviewer.kind is ActorKind.REVIEWER and reviewer.account_id == reviewer_session.account_id
    process = audit.actor_for(service_key.principal)
    assert process.kind is ActorKind.PROCESS_SERVICE and process.installation_id == service_key.installation_id
