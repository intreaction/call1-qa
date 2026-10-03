"""Store's own maintenance: leases expire and the change feed is pruned without Process claiming.

``maintenance.run_once`` is what the app's background task runs every
``StoreConfig.maintenance_interval_seconds`` (``CALL1_STORE_MAINTENANCE_SECONDS``)."""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from call1.store import maintenance
from call1.store.app import create_app
from call1.store.config import ConfigError, StoreConfig

from .conftest import STORE_BASE_URL
from .test_queue_harness import hooks, q, rubrics  # noqa: F401


def _job_status(store, job_id: str) -> str:
    with store.connection() as conn:
        return conn.execute("SELECT status FROM q_jobs WHERE id = ?", (job_id,)).fetchone()["status"]


def test_run_once_expires_leases_with_no_claim_arriving(q, store):
    ingest = q.ingest()
    claimed = q.claim_one(ingest["ids"]["vad"])
    assert _job_status(store, claimed["job"]["id"]) == "RUNNING"
    q.clock.advance(24 * 3600)  # far past any lease; Process is gone and never claims again
    counts = maintenance.run_once(store)
    assert counts["leases_expired"] == 1
    assert _job_status(store, claimed["job"]["id"]) == "QUEUED"
    assert q.heartbeat(claimed, expect=409).json()["code"] == "claim_token_stale"


def test_run_once_prunes_the_change_feed_so_old_cursors_expire(q, store):
    q.ingest()  # job events: what Process's feed audience sees
    before = q.get("/changes").json()
    assert before["events"]
    start = before["events"][0]["cursor"]  # a reader parked at the first event
    q.clock.advance(store.config.parameters.change_feed_retention_seconds + 60)
    q.ingest()  # newer events that survive the prune
    assert q.get("/changes", params={"after": start}).status_code == 200  # nothing pruned yet (no claim has come in)
    assert maintenance.run_once(store)["change_events_pruned"] >= len(before["events"])
    expired = q.get("/changes", params={"after": start}, expect=410).json()
    assert expired["code"] == "cursor_expired" and expired["details"]["oldest_cursor"] == before["latest_cursor"]
    resumed = q.get("/changes", params={"after": before["latest_cursor"]}).json()
    assert resumed["events"] and all(e["cursor"] > before["latest_cursor"] for e in resumed["events"])


def test_the_app_runs_maintenance_in_the_background(tmp_path, monkeypatch):
    ran = threading.Event()
    seen = []

    def fake_run_once(store):
        seen.append(store)
        ran.set()
        return {"leases_expired": 0}

    monkeypatch.setattr(maintenance, "run_once", fake_run_once)
    app = create_app(StoreConfig.for_tests(tmp_path / "store", maintenance_interval_seconds=0.01))
    with TestClient(app, base_url=STORE_BASE_URL):
        assert ran.wait(5), "the maintenance task never ran"
    assert seen[0] is app.state.store


def test_maintenance_is_off_in_tests_and_on_for_serve(tmp_path):
    assert StoreConfig.for_tests(tmp_path).maintenance_interval_seconds == 0
    assert StoreConfig.from_env({"CALL1_STORE_DATA": str(tmp_path)}).maintenance_interval_seconds == 60
    assert StoreConfig.from_env({"CALL1_STORE_DATA": str(tmp_path), "CALL1_STORE_MAINTENANCE_SECONDS": "5"}).maintenance_interval_seconds == 5
    with pytest.raises(ConfigError):
        StoreConfig.from_env({"CALL1_STORE_DATA": str(tmp_path), "CALL1_STORE_MAINTENANCE_SECONDS": "soon"})
    with pytest.raises(ConfigError):
        StoreConfig.for_tests(tmp_path, maintenance_interval_seconds=-1)
