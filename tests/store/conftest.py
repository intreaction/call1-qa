"""Fixtures for Store tests. No network ports: every request goes through FastAPI's TestClient.

    app, store, client, clock, conn                 a fresh dev-mode Store in tmp_path
    mint_service_key(scopes=None, ...)              TEST-ONLY Process key (default: every scope)
    service_key, service_key_headers                one key with every scope, primary host
    mint_session(role)                              TEST-ONLY reviewer session
    reviewer_session, supervisor_session, admin_session

A minted session carries ``read_headers`` (cookie) and ``headers`` (cookie + X-Call1-CSRF); pass
them per request, so several roles can act in one test. ``clock`` is a ManualClock: advance it to
expire leases, sessions and grants.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pytest
from fastapi.testclient import TestClient

from call1.contracts.common import ReviewerRole, ServiceScope
from call1.store.app import create_app
from call1.store.clock import ManualClock
from call1.store.config import StoreConfig
from call1.store.principals import MintedServiceKey, MintedSession

STORE_BASE_URL = "http://localhost:8010"


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def store_config(tmp_path) -> StoreConfig:
    return StoreConfig.for_tests(tmp_path / "store")


@pytest.fixture
def app(store_config, clock):
    return create_app(store_config, clock=clock)


@pytest.fixture
def store(app):
    return app.state.store


@pytest.fixture
def client(app):
    with TestClient(app, base_url=STORE_BASE_URL) as test_client:
        yield test_client


@pytest.fixture
def conn(store):
    with store.connection() as connection:
        yield connection


@pytest.fixture
def mint_service_key(store):
    def mint(scopes: Optional[Iterable[ServiceScope]] = None, *, installation_id: Optional[str] = None,
             primary_host: bool = True, label: str = "test key") -> MintedServiceKey:
        with store.connection() as connection:
            return store.auth.mint_service_key_for_tests(
                connection, scopes=list(scopes) if scopes is not None else list(ServiceScope),
                installation_id=installation_id, primary_host=primary_host, label=label,
            )

    return mint


@pytest.fixture
def service_key(mint_service_key) -> MintedServiceKey:
    return mint_service_key()


@pytest.fixture
def service_key_headers(service_key) -> dict:
    return service_key.headers


@pytest.fixture
def mint_session(store):
    def mint(role: ReviewerRole = ReviewerRole.REVIEWER, *, email: Optional[str] = None, display_name: Optional[str] = None) -> MintedSession:
        with store.connection() as connection:
            return store.auth.mint_session_for_tests(connection, role=ReviewerRole(role), email=email, display_name=display_name)

    return mint


@pytest.fixture
def reviewer_session(mint_session) -> MintedSession:
    return mint_session(ReviewerRole.REVIEWER)


@pytest.fixture
def supervisor_session(mint_session) -> MintedSession:
    return mint_session(ReviewerRole.SUPERVISOR)


@pytest.fixture
def admin_session(mint_session) -> MintedSession:
    return mint_session(ReviewerRole.ADMIN)
