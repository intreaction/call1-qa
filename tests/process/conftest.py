"""Fixtures for Process tests: a real Store app in-process, reached through Starlette's TestClient
as the Process Store client's httpx transport. No network ports are bound.

    clock            the Store's ManualClock, started at the real current time
    store_app, store the in-process Store (dev mode, data in tmp_path)
    store_http       TestClient on the Store app, base URL http://localhost:8010
    mint_key(...)    TEST-ONLY Process service key (default: the Stage 2 scopes plus jobs:control)
    session(role)    TEST-ONLY reviewer session headers (read / write)
    make_runtime     a ProcessRuntime (fake handlers) on its own data dir, talking to that Store
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pytest
from fastapi.testclient import TestClient

from call1.contracts.common import ReviewerRole, ServiceScope
from call1.process.config import ProcessConfig, SlotSizes
from call1.process.handlers.fake import FakeBehavior
from call1.process.runtime import ProcessRuntime
from call1.process.store_client import StoreClient
from call1.store.app import create_app as create_store_app
from call1.store.clock import ManualClock
from call1.store.config import StoreConfig

STORE_URL = "http://localhost:8010"
REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "sample_audio" / "call_01_compliant.wav"

PROCESS_SCOPES = [
    ServiceScope.CALLS_WRITE, ServiceScope.ARTIFACTS_READ, ServiceScope.ARTIFACTS_WRITE, ServiceScope.JOBS_WRITE, ServiceScope.JOBS_CLAIM,
    ServiceScope.REANALYSIS_CLAIM, ServiceScope.CHANGES_READ, ServiceScope.HARDWARE_WRITE, ServiceScope.CATALOG_PUBLISH, ServiceScope.USAGE_READ,
    ServiceScope.ADMIN_STATE_READ, ServiceScope.TRAINING_READ,
]


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(start=datetime.now(timezone.utc).replace(microsecond=0))


@pytest.fixture
def store_app(tmp_path, clock):
    return create_store_app(StoreConfig.for_tests(tmp_path / "store"), clock=clock)


@pytest.fixture
def store(store_app):
    return store_app.state.store


@pytest.fixture
def store_http(store_app):
    with TestClient(store_app, base_url=STORE_URL) as client:
        yield client


@pytest.fixture
def mint_key(store):
    def mint(scopes: Optional[Iterable[ServiceScope]] = None, *, installation_id: Optional[str] = None, primary_host: bool = True,
             control: bool = True):
        chosen = list(scopes) if scopes is not None else list(PROCESS_SCOPES) + ([ServiceScope.JOBS_CONTROL] if control else [])
        with store.connection() as conn:
            return store.auth.mint_service_key_for_tests(conn, scopes=chosen, installation_id=installation_id, primary_host=primary_host,
                                                         label="process test key")
    return mint


@pytest.fixture
def session(store):
    def mint(role: ReviewerRole = ReviewerRole.REVIEWER):
        with store.connection() as conn:
            minted = store.auth.mint_session_for_tests(conn, role=ReviewerRole(role))
        return minted
    return mint


def write_headers(minted) -> dict:
    return {**minted.headers, "Origin": STORE_URL}


@pytest.fixture
def make_runtime(tmp_path, store_http, mint_key):
    runtimes = []

    def make(name: str = "p1", *, key=None, behavior: Optional[FakeBehavior] = None, **overrides) -> ProcessRuntime:
        key = key or mint_key()
        data = tmp_path / name
        data.mkdir(parents=True, exist_ok=True)
        config_path = data / "config.json"
        raw = {"store_url": STORE_URL, "installation_id": key.installation_id, "service_key_id": "key_test", "service_key": key.token}
        config_path.write_text(json.dumps(raw))
        settings = dict(config_path=config_path, store_url=STORE_URL, installation_id=key.installation_id, service_key_id="key_test",
                        service_key=key.token, data_dir=data, handlers="fake", worker_id=f"{name}-worker", poll_interval_seconds=0.05,
                        shutdown_grace_seconds=2.0, slots=SlotSizes(cpu_io=4, torch=1, mlx=1, outbound=0))
        settings.update(overrides)
        config = ProcessConfig(**settings)
        client = StoreClient(STORE_URL, key.token, http=store_http, sleep=lambda _s: None)
        runtime = ProcessRuntime(config, client=client, fake_behavior=behavior or FakeBehavior())
        runtime.test_key = key  # type: ignore[attr-defined]
        runtimes.append(runtime)
        return runtime

    yield make
    for runtime in runtimes:
        if runtime.worker is not None and runtime.worker._threads:
            runtime.stop()
