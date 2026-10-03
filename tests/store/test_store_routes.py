"""Every contract route is registered, owned or deferred, and answers with the contract envelope."""

from __future__ import annotations

import os
import re
from collections import Counter

import pytest
from fastapi import Depends

from call1.contracts.api import ROUTES, routes_by_operation
from call1.contracts.calls import ConversationRegistration, ConversationRegistered
from call1.contracts.catalog import CatalogSnapshot
from call1.contracts.common import Page
from call1.contracts.errors import ErrorResponse
from call1.store.app import HandlerContractError, create_app, route_status
from call1.store.deps import current_principal
from call1.store.principals import Principal
from call1.store.queue.routes import router as queue_router
from call1.store.routing import DEFERRED, OWNER_BY_OPERATION, Area, AreaRouter

_PARAM = re.compile(r"{(\w+)}")


def _concrete(path: str) -> str:
    return _PARAM.sub(lambda m: "1" if m.group(1) in ("version", "attempt_number") else "x1", path)


def _contract_routes(app):
    return [r for r in app.routes if getattr(r, "path", "").startswith("/store/v1/")]


def test_every_contract_route_is_registered_once_with_its_method_and_path(app):
    registered = Counter((method, route.path) for route in _contract_routes(app) for method in route.methods)
    expected = Counter((route.method, route.path) for route in ROUTES)
    assert registered == expected
    by_op = {route.operation_id: route for route in _contract_routes(app)}
    for contract_route in ROUTES:
        served = by_op[contract_route.operation_id]
        assert served.path == contract_route.path and served.methods == {contract_route.method}


def test_ownership_and_deferral_partition_the_route_table():
    operations = {route.operation_id for route in ROUTES}
    owned, deferred = set(OWNER_BY_OPERATION), set(DEFERRED)
    assert not owned & deferred
    assert owned | deferred == operations
    assert len(OWNER_BY_OPERATION) + len(DEFERRED) == len(ROUTES)


def test_deferred_set_follows_the_split_brief():
    """docs/SplitBuild.md "Deferred": custody, release trust and admin/pro1, Stage 4 and 5 routes,
    the price table. Admin state is outside the brief's Stage 2 area list, so it is deferred too."""
    expected = {
        r.operation_id for r in ROUTES
        if r.tag in ("custody", "release-trust")
        or r.path.startswith("/store/v1/admin/pro1/")
        or r.stage in (4, 5)
        or r.operation_id in ("getPriceTable", "savePriceTable", "getAdminState", "changeAdminState")
    }
    assert set(DEFERRED) == expected


def test_every_owned_operation_is_stage_2():
    routes = routes_by_operation()
    assert {routes[op].stage for op in OWNER_BY_OPERATION} == {2}


def test_route_status_matches_the_tables(app):
    status = route_status(app)
    assert set(status) == {r.operation_id for r in ROUTES}
    assert {op for op, s in status.items() if s == "deferred"} == set(DEFERRED)
    assert {"getStatus", "getStatusDetail", "getContract", "listChanges", "listAuditEvents"} <= {op for op, s in status.items() if s == "handled"}


def test_deferred_routes_answer_501_with_the_envelope(client):
    for op in DEFERRED:
        route = routes_by_operation()[op]
        response = client.request(route.method, _concrete(route.path))
        assert response.status_code == 501, op
        body = ErrorResponse.model_validate(response.json())
        assert body.code.value == "not_implemented"
        assert body.details["operation_id"] == op and body.details["pending"] is False
        assert body.request_id == response.headers["X-Request-ID"]


def test_pending_routes_answer_501_and_name_their_owner(app, client):
    pending = [op for op, s in route_status(app).items() if s == "pending"]
    for op in pending:
        route = routes_by_operation()[op]
        response = client.request(route.method, _concrete(route.path))
        assert response.status_code == 501, op
        details = response.json()["details"]
        assert details["pending"] is True and details["owner"] == OWNER_BY_OPERATION[op].value


def test_in_scope_routes_have_handlers(app):
    """Skips (listing what is left) until every in-scope route is built; the integration phase
    sets CALL1_STORE_REQUIRE_COMPLETE=1 to turn the list into a failure."""
    pending = sorted(op for op, s in route_status(app).items() if s == "pending")
    if pending and os.environ.get("CALL1_STORE_REQUIRE_COMPLETE") != "1":
        pytest.skip(f"{len(pending)} in-scope routes still pending: " + ", ".join(pending))
    assert pending == []


def test_area_router_refuses_foreign_deferred_and_unknown_operations():
    router = AreaRouter(Area.QUEUE)
    with pytest.raises(ValueError, match="belongs to the auth area"):
        router.operation("signInBegin")
    with pytest.raises(ValueError, match="deferred"):
        router.operation("recordKeyRelease")
    with pytest.raises(KeyError):
        router.operation("noSuchOperation")


def test_a_handler_is_wired_with_its_guard_and_response_model(monkeypatch, store_config, clock, mint_session):
    def list_catalog_snapshots(principal: Principal = Depends(current_principal)) -> Page[CatalogSnapshot]:
        return Page[CatalogSnapshot](items=[])

    monkeypatch.setitem(queue_router.handlers, "listCatalogSnapshots", list_catalog_snapshots)
    from fastapi.testclient import TestClient

    app = create_app(store_config, clock=clock)
    assert route_status(app)["listCatalogSnapshots"] == "handled"
    store = app.state.store
    with store.connection() as conn:
        from call1.contracts.common import ReviewerRole

        session = store.auth.mint_session_for_tests(conn, role=ReviewerRole.REVIEWER)
    client = TestClient(app, base_url="http://localhost:8010")
    assert client.get("/store/v1/catalog-snapshots").status_code == 401
    ok = client.get("/store/v1/catalog-snapshots", headers=session.read_headers)
    assert ok.status_code == 200 and ok.json() == {"items": [], "next_page_token": None}


def test_handler_signatures_are_checked_against_the_contract(monkeypatch, store_config, clock):
    def wrong_body(body: ConversationRegistered) -> ConversationRegistered:  # contract body is ConversationRegistration
        raise AssertionError

    monkeypatch.setitem(queue_router.handlers, "registerConversation", wrong_body)
    with pytest.raises(HandlerContractError, match="registerConversation"):
        create_app(store_config, clock=clock)

    def wrong_path(conversation: str) -> None:
        raise AssertionError

    monkeypatch.delitem(queue_router.handlers, "registerConversation")
    monkeypatch.setitem(queue_router.handlers, "getConversation", wrong_path)
    with pytest.raises(HandlerContractError, match="getConversation"):
        create_app(store_config, clock=clock)


def test_a_correct_handler_signature_is_accepted(monkeypatch, store_config, clock):
    def register_conversation(body: ConversationRegistration) -> ConversationRegistered:
        raise AssertionError

    monkeypatch.setitem(queue_router.handlers, "registerConversation", register_conversation)
    assert route_status(create_app(store_config, clock=clock))["registerConversation"] == "handled"
