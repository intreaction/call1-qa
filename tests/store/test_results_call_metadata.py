"""Contract 1.1.0 in Store: agent identity (``agent_display_name``, ``agent_extension``) on every
view that shows an agent, and re-registration of a known source identity updating call metadata
(``calls.merge_call_metadata``) with an audit event and a change event, without reprocessing.

Driven through the real queue, results and auth areas over HTTP (``Process`` from
``test_results_end_to_end``)."""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone

from call1.contracts.calls import agent_label
from call1.contracts.contents import VerdictStatus
from call1.store import db
from call1.store.db import MIGRATIONS_DIR, Database

from .test_results_end_to_end import Process, _scorecard_job, _key
from .test_results_harness import qa_scorecard

V = "/store/v1"


def _source(digest_seed: str) -> dict:
    return {"kind": "api_upload", "content_digest": "sha256:" + hashlib.sha256(digest_seed.encode()).hexdigest(),
            "received_at": datetime.now(timezone.utc).isoformat()}


def _register(client, process: Process, source: dict, call_metadata: dict):
    body = {"ingestion_kind": "call_audio", "source": source, "call_metadata": call_metadata}
    response = client.post(f"{V}/conversations", json=body, headers=process.h)
    assert response.status_code == 200, response.text
    return response.json()


def _changes(client, headers, after=None):
    params = {"kinds": ["call"], "limit": 1000}
    if after:
        params["after"] = after
    feed = client.get(f"{V}/changes", params=params, headers=headers).json()
    return feed["events"], feed["next_cursor"]


def _audit(client, admin, call_id):
    return client.get(f"{V}/admin/audit", params={"action": "call_metadata_updated", "target_id": call_id, "limit": 50},
                      headers=admin.read_headers).json()["items"]


def _evaluate(client, process: Process, conversation_id: str, *, critical: bool = True) -> None:
    """One QA version that requires review (so the call has review-queue items and an escalation)."""
    rubric = client.get(f"{V}/rubrics/call1_standard_v2", headers=process.h).json()
    snapshot = process.ok(client.post(f"{V}/conversations/{conversation_id}/rubric-snapshots",
                                      json={"rubric_id": "call1_standard_v2", "version": 1}, headers=process.h))
    process.graph(conversation_id, [_scorecard_job(snapshot, rubric)])
    claimed = process.claim("qa_scorecard")
    card = qa_scorecard([("REG-01", VerdictStatus.FAIL, 0.95)], digest=rubric["ref"]["digest"], overall_score=40, passed=False,
                        critical_failure=critical, requires_human_review=True)
    process.complete(claimed, {"scorecard": process.output(conversation_id, claimed, "qa_scorecard", card)}, result={"kind": "qa", "state": "available"})


def test_new_registration_projects_display_name_and_extension_everywhere(client, clock, service_key, reviewer_session, supervisor_session):
    process = Process(client, service_key, clock)
    meta = {"agent_id": "agt-104", "agent_display_name": "Samantha Reyes", "agent_extension": "104", "external_call_ref": "pbx-9"}
    registered = _register(client, process, _source(_key("src")), meta)
    assert registered["created"] is True and registered["metadata_updated"] is False and registered["updated_fields"] == []
    call_id = registered["conversation"]["call_id"]
    _evaluate(client, process, registered["conversation"]["id"])
    r = reviewer_session.read_headers

    row = client.get(f"{V}/calls", headers=r).json()["items"][0]
    assert (row["agent_id"], row["agent_display_name"], row["agent_extension"]) == ("agt-104", "Samantha Reyes", "104")
    assert agent_label(row["agent_id"], row["agent_display_name"], row["agent_extension"]) == "Samantha Reyes (104)"
    call = client.get(f"{V}/calls/{call_id}", headers=r).json()["call"]
    assert (call["agent_display_name"], call["agent_extension"]) == ("Samantha Reyes", "104")
    items = client.get(f"{V}/review-queue", params={"call_id": call_id}, headers=r).json()["items"]
    assert items and all((i["agent_display_name"], i["agent_extension"]) == ("Samantha Reyes", "104") for i in items)
    escalations = client.get(f"{V}/escalations", headers=supervisor_session.read_headers).json()["items"]
    assert [(e["call_id"], e["agent_display_name"], e["agent_extension"]) for e in escalations] == [(call_id, "Samantha Reyes", "104")]

    # Text search matches the display name and the extension, as well as agent ID and call reference.
    for text in ("Samantha", "reyes", "104", "agt-1", "pbx-9"):
        found = client.get(f"{V}/calls", params={"text": text}, headers=r).json()["items"]
        assert [i["call_id"] for i in found] == [call_id], text
    assert client.get(f"{V}/calls", params={"text": "Bob"}, headers=r).json()["items"] == []
    # The agent_id filter stays exact on agent_id, never the display name.
    assert client.get(f"{V}/calls", params={"agent_id": "Samantha Reyes"}, headers=r).json()["items"] == []


def test_views_send_the_new_fields_as_null_when_unknown(client, clock, service_key, reviewer_session):
    process = Process(client, service_key, clock)
    registered = _register(client, process, _source(_key("src")), {"agent_id": "agent-7"})
    call_id = registered["conversation"]["call_id"]
    row = client.get(f"{V}/calls", headers=reviewer_session.read_headers).json()["items"][0]
    assert "agent_display_name" in row and row["agent_display_name"] is None and row["agent_extension"] is None
    call = client.get(f"{V}/calls/{call_id}", headers=reviewer_session.read_headers).json()["call"]
    assert call["agent_display_name"] is None and call["agent_extension"] is None


def test_reregistration_updates_metadata_audits_and_emits_without_reprocessing(client, clock, service_key, reviewer_session,
                                                                               supervisor_session, admin_session):
    process = Process(client, service_key, clock)
    source = _source(_key("src"))
    first = _register(client, process, source, {"external_call_ref": "pbx-1"})  # no agent: the call says "Unknown"
    conversation_id, call_id = first["conversation"]["id"], first["conversation"]["call_id"]
    _evaluate(client, process, conversation_id)
    r = reviewer_session.read_headers
    before = client.get(f"{V}/calls/{call_id}", headers=r).json()
    assert before["call"]["agent_id"] == "Unknown"
    jobs_before = client.get(f"{V}/jobs", params={"conversation_id": conversation_id, "limit": 100}, headers=process.h).json()["items"]
    _, cursor = _changes(client, r)

    # A re-upload naming the agent updates the call; fields it does not send keep their values.
    again = _register(client, process, dict(source, received_at=datetime.now(timezone.utc).isoformat()),
                      {"agent_id": "bob-202", "agent_display_name": "Bob", "agent_extension": "202"})
    assert again["created"] is False and again["metadata_updated"] is True
    assert again["updated_fields"] == ["agent_id", "agent_display_name", "agent_extension"]
    assert again["conversation"]["id"] == conversation_id and again["conversation"]["call_id"] == call_id
    stored = again["conversation"]["call_metadata"]
    assert (stored["agent_id"], stored["agent_display_name"], stored["agent_extension"], stored["external_call_ref"]) == ("bob-202", "Bob", "202", "pbx-1")
    assert client.get(f"{V}/conversations/{conversation_id}", headers=process.h).json()["call_metadata"] == stored

    after = client.get(f"{V}/calls/{call_id}", headers=r).json()
    assert (after["call"]["agent_id"], after["call"]["agent_display_name"], after["call"]["agent_extension"]) == ("bob-202", "Bob", "202")
    row = client.get(f"{V}/calls", params={"agent_id": "bob-202"}, headers=r).json()["items"]
    assert [(i["call_id"], i["agent_display_name"], i["agent_extension"]) for i in row] == [(call_id, "Bob", "202")]
    items = client.get(f"{V}/review-queue", params={"call_id": call_id}, headers=r).json()["items"]
    assert items and all((i["agent_id"], i["agent_display_name"], i["agent_extension"]) == ("bob-202", "Bob", "202") for i in items)
    escalation = [e for e in client.get(f"{V}/escalations", headers=supervisor_session.read_headers).json()["items"] if e["call_id"] == call_id]
    assert [(e["agent_id"], e["agent_display_name"]) for e in escalation] == [("bob-202", "Bob")]

    # Nothing reprocessed: same jobs, same results, same review version, same queue item versions.
    assert client.get(f"{V}/jobs", params={"conversation_id": conversation_id, "limit": 100}, headers=process.h).json()["items"] == jobs_before
    assert after["results"] == before["results"] and after["review_version"] == before["review_version"]
    assert after["evaluation"] == before["evaluation"] and after["pending_work"] == before["pending_work"]

    # One call change event with status metadata_updated, one audit event naming fields, never values.
    events, cursor = _changes(client, r, cursor)
    assert [(e["resource_id"], e["call_id"], e["conversation_id"], e["status"]) for e in events] == [(call_id, call_id, conversation_id, "metadata_updated")]
    audits = _audit(client, admin_session, call_id)
    assert len(audits) == 1
    event = audits[0]
    assert event["actor"]["kind"] == "process_service" and event["actor"]["installation_id"] == process.installation_id
    assert event["target"] == {"kind": "call", "id": call_id}
    assert event["details"] == {"conversation_id": conversation_id, "updated_fields": "agent_id,agent_display_name,agent_extension"}
    assert "Bob" not in str(event) and "202" not in str(event["details"])

    # Replaying the same metadata, a subset of it, or no fields at all is a pure replay: no event of either kind.
    for metadata in ({"agent_id": "bob-202", "agent_display_name": "Bob", "agent_extension": "202"}, {"agent_id": "bob-202"}, {}):
        replay = _register(client, process, source, metadata)
        assert replay["created"] is False and replay["metadata_updated"] is False and replay["updated_fields"] == [], metadata
        assert replay["conversation"]["call_metadata"] == stored
    assert _changes(client, r, cursor)[0] == [] and len(_audit(client, admin_session, call_id)) == 1

    # An explicit null clears an optional field; the list follows at once.
    cleared = _register(client, process, source, {"agent_display_name": None})
    assert cleared["metadata_updated"] is True and cleared["updated_fields"] == ["agent_display_name"]
    row = client.get(f"{V}/calls/{call_id}", headers=r).json()["call"]
    assert (row["agent_id"], row["agent_display_name"], row["agent_extension"]) == ("bob-202", None, "202")
    assert agent_label(row["agent_id"], row["agent_display_name"], row["agent_extension"]) == "bob-202 (202)"
    assert len(_audit(client, admin_session, call_id)) == 2


def test_reregistration_with_invalid_agent_fields_is_refused_and_changes_nothing(client, clock, service_key, reviewer_session):
    process = Process(client, service_key, clock)
    source = _source(_key("src"))
    first = _register(client, process, source, {"agent_id": "agent-1"})
    for bad in ({"agent_display_name": " padded "}, {"agent_display_name": ""}, {"agent_extension": "10 4"}, {"agent_extension": "x" * 21}):
        response = client.post(f"{V}/conversations", json={"ingestion_kind": "call_audio", "source": source, "call_metadata": bad}, headers=process.h)
        assert response.status_code == 422 and response.json()["code"] == "validation_failed", (bad, response.text)
    call = client.get(f"{V}/calls/{first['conversation']['call_id']}", headers=reviewer_session.read_headers).json()["call"]
    assert (call["agent_id"], call["agent_display_name"], call["agent_extension"]) == ("agent-1", None, None)


def test_text_conversation_reregistration_audits_the_conversation(client, clock, service_key, admin_session):
    process = Process(client, service_key, clock)
    source = {"kind": "local_import", "content_digest": "sha256:" + hashlib.sha256(_key("txt").encode()).hexdigest(),
              "received_at": datetime.now(timezone.utc).isoformat()}
    first = client.post(f"{V}/conversations", json={"ingestion_kind": "text_import", "source": source}, headers=process.h).json()
    assert first["created"] is True and first["conversation"]["call_id"] is None
    again = client.post(f"{V}/conversations", json={"ingestion_kind": "text_import", "source": source, "call_metadata": {"agent_id": "a-9"}},
                        headers=process.h).json()
    assert again["metadata_updated"] is True and again["updated_fields"] == ["agent_id"]
    audits = client.get(f"{V}/admin/audit", params={"action": "call_metadata_updated", "target_id": first["conversation"]["id"]},
                        headers=admin_session.read_headers).json()["items"]
    assert [a["target"] for a in audits] == [{"kind": "conversation", "id": first["conversation"]["id"]}]


def test_migration_041_adds_the_columns_to_an_existing_040_database(tmp_path):
    """A Store created before 1.1.0 (migrations up to 040) keeps its rows; 041 adds the agent columns
    as NULL, which the views then send as null."""
    old = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, old, ignore=shutil.ignore_patterns("* 2.*", "041_*"))
    database = Database(tmp_path / "store.db")
    database.initialize(old)
    with database.connection() as conn:
        with db.transaction(conn):
            conn.execute("INSERT INTO results_calls (call_id, conversation_id, agent_id, created_at, updated_at) VALUES "
                         "('call_old', 'conv_old', 'agent-old', '2026-09-01T00:00:00.000000Z', '2026-09-01T00:00:00.000000Z')")
        assert db.migrate(conn) == [41]
        row = conn.execute("SELECT agent_id, agent_display_name, agent_extension FROM results_calls WHERE call_id = 'call_old'").fetchone()
        assert tuple(row) == ("agent-old", None, None)
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(results_review_items)")}
        assert {"agent_display_name", "agent_extension"} <= columns
