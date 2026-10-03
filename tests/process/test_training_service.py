"""On-device training: the start rule and the claim pause (P3) against a fake Store client and fake
worker pools, the real worker's pause, and the console routes (P12), docs/OnDeviceTraining.md
sections 3.3-3.4 and 6.1. Fake trainer and fake generator only; no MLX, torch or real model runs."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient

from call1.pipeline.signals_v2 import estimate_tokens
from call1.process import console
from call1.process.app import create_app
from call1.process.training import scheduler as scheduler_mod
from call1.process.training.scheduler import TrainingBusy, TrainingService, TrainingUnavailable

from .training_support import INSTALLATION, FakeStore, fake_base, loose_settings, registry_for, world

LOCAL = "http://127.0.0.1:8020"


class FakeWorker:
    """The pause surface of ``Worker``: pools that are busy or not, and a log of pause/resume calls."""

    def __init__(self) -> None:
        self.busy: List[str] = []
        self.running_jobs = 0
        self.calls: List[tuple] = []
        self._paused: Optional[dict] = None
        self.primary_host = True
        self.on_pause = None

    def pause_claims(self, run_id, until=None):
        self.calls.append(("pause", run_id))
        self._paused = {"run_id": run_id, "since": "now", "until": until.isoformat() if until else None}
        if self.on_pause is not None:
            self.on_pause()

    def resume_claims(self, run_id=None):
        self.calls.append(("resume", run_id))
        self._paused = None

    @property
    def claims_paused(self):
        return dict(self._paused) if self._paused else None

    def busy_pools(self, names=None):
        return [p for p in self.busy if names is None or p in names]

    def idle(self):
        return not self.busy and not self.running_jobs


@pytest.fixture
def svc(tmp_path):
    store = FakeStore()
    world(store, tmp_path, train_calls=1, eval_calls=1)
    base = fake_base(tmp_path)
    worker = FakeWorker()
    config = SimpleNamespace(data_dir=tmp_path / "process", config_path=tmp_path / "process" / "config.json", configured=True, handlers="fake",
                             installation_id="inst_training_tests", training=loose_settings().to_dict())
    ledger = SimpleNamespace(at=None, latest_at=lambda: ledger.at)
    service = TrainingService(config, client=store, registry=registry_for(tmp_path, base), base_model=base, worker=lambda: worker,
                              ledger=ledger, count_tokens=estimate_tokens, recheck_seconds=0.05, drain_timeout=0.3, kill_grace=1.0)
    return SimpleNamespace(service=service, store=store, worker=worker, ledger=ledger, config=config)


def wait_for(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_queued_or_running_local_memory_jobs_block_a_run_until_they_clear(svc):
    svc.store.queued.append({"status": "QUEUED", "memory_slot": "local_memory"})
    run = svc.service.request_run()
    assert run["trigger"] == "manual"
    assert wait_for(lambda: svc.service.describe()["status"]["detail"] == "Waiting: model jobs queued")
    assert svc.worker.calls == []  # never paused while the rule fails
    svc.store.queued[0] = {"status": "RUNNING", "memory_slot": "local_memory"}
    assert wait_for(lambda: svc.service.describe()["status"]["detail"] == "Waiting: model jobs running")
    svc.store.queued.clear()
    record = svc.service.wait()
    assert record["status"] == "promoted"
    assert svc.worker.calls == [("pause", run["run_id"]), ("resume", run["run_id"])]
    assert svc.worker.claims_paused is None


def test_a_busy_mlx_or_torch_pool_blocks_but_blocked_and_cpu_work_do_not(svc):
    svc.worker.busy = ["torch"]
    svc.service.request_run()
    assert wait_for(lambda: "a model job is running" in svc.service.describe()["status"]["detail"])
    svc.store.queued.append({"status": "BLOCKED", "memory_slot": "local_memory"})  # a job behind a failed upstream
    svc.store.queued.append({"status": "QUEUED", "memory_slot": "cpu"})  # manual runs ignore other slots
    svc.worker.busy = []
    assert svc.service.wait()["status"] == "promoted"


def test_an_empty_claim_in_flight_does_not_hold_back_train_now(svc):
    """A pool loop reserves its slots for each claim round-trip, so an empty claim in flight reads as a
    busy mlx pool. The run pauses claims first; once the round-trip ends the pool is free and the run
    starts at once, with no recheck interval and a single pause."""
    svc.service.recheck_seconds = 30.0  # a backoff would outlast the wait below
    svc.worker.busy = ["mlx"]

    def claim_returns_empty():
        svc.worker.busy = []

    svc.worker.on_pause = lambda: threading.Timer(0.05, claim_returns_empty).start()
    run = svc.service.request_run()
    record = svc.service.wait(timeout=10)
    assert record is not None and record["status"] == "promoted"
    assert svc.worker.calls == [("pause", run["run_id"]), ("resume", run["run_id"])]


def test_work_that_arrives_during_the_pause_resumes_claims_and_waits(svc):
    arrived = {"n": 0}

    def arrive():
        if arrived["n"] == 0:
            svc.store.queued.append({"status": "QUEUED", "memory_slot": "local_memory"})
        arrived["n"] += 1

    svc.worker.on_pause = arrive
    run = svc.service.request_run()
    assert wait_for(lambda: len(svc.worker.calls) >= 2)
    assert svc.worker.calls[:2] == [("pause", run["run_id"]), ("resume", run["run_id"])]
    assert svc.worker.claims_paused is None
    svc.store.queued.clear()
    assert svc.service.wait()["status"] == "promoted"
    assert svc.worker.calls == [("pause", run["run_id"]), ("resume", run["run_id"])] * 2


def test_running_jobs_that_outlast_the_drain_timeout_resume_claims(svc):
    svc.worker.running_jobs = 1

    def finish_later():
        time.sleep(0.8)
        svc.worker.running_jobs = 0

    run = svc.service.request_run()
    threading.Thread(target=finish_later, daemon=True).start()
    record = svc.service.wait()
    assert record["status"] == "promoted"
    pauses = [c for c in svc.worker.calls if c[0] == "pause"]
    assert len(pauses) >= 2 and svc.worker.calls[-1] == ("resume", run["run_id"]) and svc.worker.claims_paused is None


@pytest.mark.parametrize("env, status", [({}, "promoted"), ({"CALL1_FAKE_TRAINER_EXIT": "9"}, "failed"),
                                         ({"CALL1_FAKE_TRAINING_OUTCOMES": "reject"}, "rejected"),
                                         ({"CALL1_FAKE_TRAINING_OUTCOMES": "crash"}, "failed")])
def test_claims_resume_in_every_outcome(svc, monkeypatch, env, status):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    run = svc.service.request_run()
    record = svc.service.wait()
    assert record["status"] == status
    assert svc.worker.calls[-1] == ("resume", run["run_id"]) and svc.worker.claims_paused is None
    assert svc.service.history()[0]["run_id"] == run["run_id"]


def test_claims_resume_after_a_cancel_and_an_exception(svc, monkeypatch):
    monkeypatch.setenv("CALL1_FAKE_TRAINER_SECONDS", "20")
    run = svc.service.request_run()
    assert wait_for(lambda: svc.service.describe()["status"]["phase"] == "training")
    assert svc.worker.claims_paused["run_id"] == run["run_id"]
    assert svc.service.describe()["status"]["claims_paused"]["run_id"] == run["run_id"]
    cancelled = svc.service.cancel(run["run_id"])
    assert cancelled["run_id"] == run["run_id"]
    record = svc.service.wait()
    assert record["status"] == "cancelled" and svc.worker.claims_paused is None
    assert svc.service.cancel(run["run_id"])["status"] == "cancelled"  # idempotent
    assert svc.service.registry.active() is None

    def boom(*a, **k):
        raise RuntimeError("builder exploded")

    monkeypatch.setattr("call1.process.training.runner.collect", boom)
    second = svc.service.request_run()
    record = svc.service.wait()
    assert record["status"] == "failed" and record["reason"] == "error: RuntimeError"
    assert svc.worker.calls[-1] == ("resume", second["run_id"]) and svc.worker.claims_paused is None


def test_a_waiting_run_is_dropped_on_cancel_and_train_now_is_exclusive(svc):
    svc.store.queued.append({"status": "QUEUED", "memory_slot": "local_memory"})
    run = svc.service.request_run()
    with pytest.raises(TrainingBusy):
        svc.service.request_run()
    with pytest.raises(TrainingBusy):
        svc.service.activate(None)
    svc.service.cancel(run["run_id"])
    record = svc.service.wait()
    assert record["status"] == "cancelled" and svc.worker.calls == []


def test_the_start_window_closing_skips_the_run_as_busy(svc):
    svc.store.queued.append({"status": "QUEUED", "memory_slot": "local_memory"})
    clock = {"now": datetime(2026, 9, 26, 2, 0, tzinfo=timezone.utc)}
    svc.service.now = lambda: clock["now"]
    svc.service.request_run()
    assert wait_for(lambda: "Waiting" in svc.service.describe()["status"]["detail"])
    clock["now"] += timedelta(hours=4, seconds=1)
    record = svc.service.wait()
    assert record["status"] == "skipped" and record["reason"] == "busy: model jobs queued"


def test_only_when_idle_applies_to_scheduled_runs_only(svc):
    svc.store.queued.append({"status": "QUEUED", "memory_slot": "cpu"})
    assert svc.service.start_rule(idle_mode=True) == (False, "jobs queued")
    assert svc.service.start_rule(idle_mode=False) == (True, "")
    svc.store.queued.clear()
    svc.ledger.at = svc.service.now() - timedelta(minutes=5)
    assert svc.service.start_rule(idle_mode=True) == (False, "a recording was ingested in the last 15 minutes")
    svc.ledger.at = svc.service.now() - timedelta(minutes=16)
    assert svc.service.start_rule(idle_mode=True) == (True, "")
    svc.worker.busy = ["cpu_io"]
    assert svc.service.start_rule(idle_mode=True) == (False, "jobs are running")
    assert svc.service.start_rule(idle_mode=False) == (True, "")


def test_training_is_unavailable_without_mlx_unless_the_trainer_is_fake(svc, monkeypatch):
    svc.config.handlers = "real"
    monkeypatch.delenv("CALL1_PROCESS_TRAINER", raising=False)
    svc.service.settings = loose_settings(trainer="mlx_lm")
    monkeypatch.setattr(scheduler_mod, "mlx_available", lambda: False)
    assert svc.service.unavailable_reason().startswith("MLX is not available")
    with pytest.raises(TrainingUnavailable):
        svc.service.request_run()
    with pytest.raises(TrainingUnavailable):
        svc.service.update_settings({"enabled": True})
    assert svc.service.update_settings({"min_new_labels": 5}).min_new_labels == 5  # settings stay editable
    svc.worker.primary_host = False
    assert svc.service.unavailable_reason() == "This Process is not the primary host"


def test_the_fake_outcomes_are_consumed_one_per_run(svc, monkeypatch):
    monkeypatch.setenv("CALL1_FAKE_TRAINING_OUTCOMES", "reject,promote")
    svc.service.request_run()
    assert svc.service.wait()["status"] == "rejected"
    svc.service.request_run()
    assert svc.service.wait()["status"] == "promoted"
    svc.service.request_run()
    assert svc.service.wait()["status"] == "promoted"  # used up: promote
    runs = svc.service.runs()
    assert [r["status"] for r in runs] == ["promoted", "promoted", "rejected"]
    assert svc.service.state()["cursor"] == len(svc.store.labels)


# --- the real worker's pause -----------------------------------------------------------------------


def test_the_worker_claims_nothing_while_paused_and_reports_the_pause(make_runtime):
    from .conftest import SAMPLE

    runtime = make_runtime()
    worker = runtime.connect()
    runtime.ingestor.ingest_file(SAMPLE)
    worker.pause_claims("tr-x", until=datetime.now(timezone.utc) + timedelta(hours=1))
    assert worker.drain() == 0 and worker.claims_paused["run_id"] == "tr-x"
    assert worker.describe()["claims_paused"]["run_id"] == "tr-x"
    assert runtime.overview()["training"]["claims_paused"]["run_id"] == "tr-x"
    worker.resume_claims("tr-other")  # only the pausing run lifts it
    assert worker.claims_paused is not None
    worker.resume_claims("tr-x")
    assert worker.claims_paused is None and worker.idle()
    assert worker.drain() > 0


# --- P12: the console routes -----------------------------------------------------------------------


@pytest.fixture
def api(make_runtime, tmp_path):
    runtime = make_runtime()
    runtime.connect()
    store = FakeStore()
    world(store, tmp_path, train_calls=1, eval_calls=1)
    base = fake_base(tmp_path)
    runtime.training.client = store
    runtime.training.config = runtime.config.with_overrides(installation_id=INSTALLATION)  # the fixture world's split
    runtime.training.base_model = base
    runtime.training._count_tokens = estimate_tokens
    runtime.training.recheck_seconds = 0.05
    runtime.adapters.base = base
    runtime.training.settings = loose_settings()
    token, _ = console.issue(runtime.config)
    app = create_app(runtime, start_background=False, loopback_only=True, static_root=tmp_path / "static")
    with TestClient(app, base_url=LOCAL, client=("127.0.0.1", 50000)) as client:
        client.runtime = runtime  # type: ignore[attr-defined]
        client.headers.update({"Origin": LOCAL})
        client.token = token  # type: ignore[attr-defined]
        yield client


def auth(api):
    return {console.HEADER: api.token}


def test_the_training_view(api):
    body = api.get("/process/api/training").json()
    assert body["available"] is True and body["unavailable_reason"] is None and body["trainer"] == "fake"
    assert body["settings"]["enabled"] is False and body["next_run_at"] is None and body["timezone"]
    assert body["labels"] == {"total": len(api.runtime.training.client.labels), "new_since_last_run": len(api.runtime.training.client.labels),
                              "error": None}
    assert body["status"]["phase"] == "idle" and body["active"] is None and body["versions"] == [] and body["runs"] == []


def test_writes_need_the_console_token(api):
    assert api.put("/process/api/training/settings", json={"enabled": True}).status_code == 401
    assert api.post("/process/api/training/runs").status_code == 401
    assert api.post("/process/api/training/active", json={"version": None}).status_code == 401
    assert api.post("/process/api/training/runs/tr-x/cancel").status_code == 401
    cross = api.post("/process/api/training/runs", headers={**auth(api), "Origin": "https://evil.example"})
    assert cross.status_code == 403


def test_settings_round_trip_and_validation(api):
    response = api.put("/process/api/training/settings", headers=auth(api),
                       json={"enabled": True, "schedule": {"frequency": "weekly", "weekday": 2, "time": "03:15"}, "min_new_labels": 10,
                             "max_duration_minutes": 90, "only_when_idle": False})
    assert response.status_code == 200, response.text
    settings = response.json()["settings"]
    assert settings["enabled"] is True and settings["schedule"] == {"frequency": "weekly", "weekday": 2, "time": "03:15"}
    saved = json.loads(api.runtime.config.config_path.read_text())["training"]
    assert saved["schedule"]["time"] == "03:15" and saved["min_new_labels"] == 10
    view = api.get("/process/api/training").json()
    assert view["settings"]["schedule"]["weekday"] == 2 and view["next_run_at"] is not None
    bad = api.put("/process/api/training/settings", headers=auth(api), json={"schedule": {"time": "25:00"}})
    assert bad.status_code == 422 and bad.json()["code"] == "validation_failed" and bad.json()["details"]["field"] == "schedule.time"
    assert api.put("/process/api/training/settings", headers=auth(api), json={"keep_versions": 3}).status_code == 422
    assert api.put("/process/api/training/settings", headers=auth(api), json=[1]).status_code == 422


def test_enabling_on_an_unavailable_host_is_refused(api, monkeypatch):
    monkeypatch.setattr(api.runtime.training, "unavailable_reason", lambda: "MLX is not available on this host")
    response = api.put("/process/api/training/settings", headers=auth(api), json={"enabled": True})
    assert response.status_code == 409 and response.json()["code"] == "training_unavailable"
    assert api.post("/process/api/training/runs", headers=auth(api)).json()["code"] == "training_unavailable"
    assert api.get("/process/api/training").json()["notices"][0]["code"] == "training_unavailable"


def test_train_now_is_202_then_409_and_promotes(api):
    api.runtime.training.client.queued.append({"status": "QUEUED", "memory_slot": "local_memory"})
    first = api.post("/process/api/training/runs", headers=auth(api))
    assert first.status_code == 202 and first.json()["run"]["trigger"] == "manual"
    busy = api.post("/process/api/training/runs", headers=auth(api))
    assert busy.status_code == 409 and busy.json()["code"] == "training_busy"
    assert api.post("/process/api/training/active", headers=auth(api), json={"version": None}).json()["code"] == "training_busy"
    api.runtime.training.client.queued.clear()
    record = api.runtime.training.wait()
    assert record["status"] == "promoted"
    view = api.get("/process/api/training").json()
    assert view["active"]["version"] == record["candidate_version"] and view["runs"][0]["status"] == "promoted"
    assert view["versions"][0]["version"] == record["candidate_version"]
    assert api.get("/process/api/training/runs?limit=5").json()["items"][0]["run_id"] == record["run_id"]


def test_cancel_and_activate_and_rollback(api, monkeypatch):
    monkeypatch.setenv("CALL1_FAKE_TRAINER_SECONDS", "20")
    run = api.post("/process/api/training/runs", headers=auth(api)).json()["run"]
    assert wait_for(lambda: api.get("/process/api/training").json()["status"]["phase"] == "training")
    cancelled = api.post(f"/process/api/training/runs/{run['run_id']}/cancel", headers=auth(api))
    assert cancelled.status_code == 200
    assert api.runtime.training.wait()["status"] == "cancelled"
    again = api.post(f"/process/api/training/runs/{run['run_id']}/cancel", headers=auth(api))
    assert again.status_code == 200 and again.json()["run"]["status"] == "cancelled"
    assert api.post("/process/api/training/runs/tr-nope/cancel", headers=auth(api)).status_code == 404

    monkeypatch.delenv("CALL1_FAKE_TRAINER_SECONDS")
    api.post("/process/api/training/runs", headers=auth(api))
    version = api.runtime.training.wait()["candidate_version"]
    base = api.post("/process/api/training/active", headers=auth(api), json={"version": None})
    assert base.status_code == 200 and base.json()["active"]["version"] is None
    assert api.get("/process/api/training").json()["active"] is None
    back = api.post("/process/api/training/active", headers=auth(api), json={"version": version})
    assert back.status_code == 200 and back.json()["active"]["version"] == version
    missing = api.post("/process/api/training/active", headers=auth(api), json={"version": "ft-20000101T000000Z"})
    assert missing.status_code == 404
    (api.runtime.adapters.base / "model.safetensors").write_bytes(b"an updated base model")
    from call1.process.training import registry as registry_mod

    registry_mod._fingerprints.clear()
    stale = api.post("/process/api/training/active", headers=auth(api), json={"version": version})
    assert stale.status_code == 409 and stale.json()["code"] == "stale_base"
    assert any(n["code"] == "base_changed" for n in api.get("/process/api/training").json()["notices"])


def test_a_key_without_training_read_shows_the_fix(api):
    from call1.process.store_client import StoreError

    api.runtime.training.client.fail_labels = StoreError("insufficient_scope", "missing training:read", status=403)
    api.runtime.training._labels_cache = (0.0, {})
    view = api.get("/process/api/training").json()
    assert view["labels"]["error"]["code"] == "insufficient_scope" and "--scope training:read" in view["labels"]["error"]["message"]
    assert any(n["code"] == "insufficient_scope" for n in view["notices"])


def test_the_training_routes_are_loopback_only(make_runtime):
    runtime = make_runtime()
    app = create_app(runtime, start_background=False, loopback_only=True)
    with TestClient(app, base_url=LOCAL, client=("10.0.0.5", 50000)) as remote:
        assert remote.get("/process/api/training").status_code == 403
    with TestClient(app, base_url="http://attacker.example", client=("127.0.0.1", 50000)) as rebound:
        assert rebound.get("/process/api/training").status_code == 403
