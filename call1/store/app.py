"""``create_app()``: the Store HTTP application.

It registers **every** route of the frozen contract (``call1.contracts.api.ROUTES``) at its exact
method and path:

* an operation whose owning area has a handler gets that handler, the route's status code and
  response model, and the principal guard (``deps.guard_for``);
* an in-scope operation without a handler yet answers 501 ``not_implemented`` with
  ``details.pending = true`` (``route_status(app)`` lists them);
* a deferred operation (``routing.DEFERRED``) answers 501 ``not_implemented`` with the reason.

Beside the contract it serves the byte-transfer endpoints behind upload and download grants
(``/store/transfer/...``), the demo-mode routes (``/demo/...``, 404 unless ``CALL1_STORE_DEMO``;
``auth/demo.py``), the Store console at ``/console/`` and Evaluate at ``/``.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import secrets
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.routing import APIRoute
from starlette.datastructures import MutableHeaders

from call1.contracts.api import BINARY, CSV, ROUTES, Idempotency, Route
from call1.contracts.common import CONTRACT_VERSION, STORE_API_PREFIX
from call1.contracts.errors import ErrorCode

from .principals import AuthBackend
from .clock import Clock
from .config import StoreConfig
from .context import Store
from .deps import guard_for
from .errors import StoreError, install_error_handlers
from .routing import DEFERRED, OWNER_BY_OPERATION, Area, AreaRouter
from .static import STATIC_ROOT, mount_frontends

_PATH_PARAM = re.compile(r"{(\w+)}")
_INT_PATH_PARAMS = ("version", "attempt_number")


class HandlerContractError(RuntimeError):
    """An area's handler does not match its contract route (raised at startup)."""


def area_routers() -> Dict[Area, AreaRouter]:
    from .auth.routes import router as auth_router
    from .queue.routes import router as queue_router
    from .results.routes import router as results_router
    from .routes import router as core_router

    return {Area.CORE: core_router, Area.QUEUE: queue_router, Area.AUTH: auth_router, Area.RESULTS: results_router}


class RequestContextMiddleware:
    """Assigns ``request.state.request_id`` (echoed as ``X-Request-ID`` and in error envelopes)
    and adds conservative headers to API responses."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = "req_" + secrets.token_hex(8)
        scope.setdefault("state", {})["request_id"] = request_id
        api = scope.get("path", "").startswith("/store/")

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers.setdefault("X-Content-Type-Options", "nosniff")
                headers.setdefault("Referrer-Policy", "no-referrer")
                if api:
                    headers.setdefault("Cache-Control", "no-store")
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _not_implemented(route: Route, *, reason: str, pending: bool) -> Callable:
    owner = OWNER_BY_OPERATION.get(route.operation_id)
    details: Dict[str, Any] = {"operation_id": route.operation_id, "stage": route.stage, "pending": pending}
    if owner is not None:
        details["owner"] = owner.value

    async def not_implemented(request: Request):
        raise StoreError(ErrorCode.NOT_IMPLEMENTED, reason, details=details)

    not_implemented.__name__ = f"not_implemented_{route.operation_id}"
    return not_implemented


def _collect_params(dependant) -> Dict[str, list]:
    """Path, query, header and body parameters of a handler and all its dependencies."""
    found: Dict[str, list] = {"path": [], "query": [], "header": [], "body": []}
    seen = set()
    stack = [dependant]
    while stack:
        current = stack.pop()
        key = id(current.call) if current.call is not None else id(current)
        if key in seen:
            continue
        seen.add(key)
        found["path"].extend(current.path_params)
        found["query"].extend(current.query_params)
        found["header"].extend(current.header_params)
        found["body"].extend(current.body_params)
        stack.extend(current.dependencies)
    return found


def _annotation(field) -> Any:
    info = getattr(field, "field_info", None)
    annotation = getattr(info, "annotation", None)
    return annotation if annotation is not None else getattr(field, "type_", None)


def verify_handler(route: Route, api_route: APIRoute) -> None:
    """Refuse a handler whose path parameters, body, query model or Idempotency-Key header differ
    from the contract route. Extra header, cookie and dependency parameters are fine."""
    flat = _collect_params(api_route.dependant)
    problems = []
    expected_path = _PATH_PARAM.findall(route.path)
    got_path = {p.name: _annotation(p) for p in flat["path"]}
    if set(got_path) != set(expected_path):
        problems.append(f"path parameters {sorted(got_path)} != contract {sorted(expected_path)}")
    for name, annotation in got_path.items():
        wanted = int if name in _INT_PATH_PARAMS else str
        if annotation is not wanted:
            problems.append(f"path parameter {name} must be {wanted.__name__}")
    bodies = [_annotation(b) for b in flat["body"]]
    if route.request is None and bodies:
        problems.append("the contract route has no request body")
    if route.request is not None and bodies != [route.request]:
        problems.append(f"body must be exactly one {route.request.__name__} (got {bodies})")
    queries = [_annotation(q) for q in flat["query"]]
    if route.query is None and queries:
        problems.append("the contract route has no query parameters")
    if route.query is not None and queries != [route.query]:
        problems.append(f"query must be Annotated[{route.query.__name__}, Query()] (got {queries})")
    if route.idempotency is Idempotency.HEADER and not any(h.alias.lower() == "idempotency-key" for h in flat["header"]):
        problems.append("header idempotency: declare Annotated[str, Header(alias='Idempotency-Key')]")
    if problems:
        raise HandlerContractError(f"{route.operation_id} ({route.method} {route.path}): " + "; ".join(problems))


def _response_model(route: Route) -> Any:
    if route.response in (BINARY, CSV) or route.response is None:
        return None
    return route.response


def register_contract_routes(app: FastAPI, routers: Mapping[Area, AreaRouter]) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    unknown = {op for r in routers.values() for op in r.handlers} - {route.operation_id for route in ROUTES}
    if unknown:
        raise HandlerContractError(f"handlers for unknown operations: {sorted(unknown)}")
    for route in ROUTES:
        op = route.operation_id
        common = dict(methods=[route.method], operation_id=op, name=op, tags=[route.tag], summary=route.summary)
        if op in DEFERRED:
            app.add_api_route(route.path, _not_implemented(route, reason=f"Not implemented: {DEFERRED[op]}", pending=False), status_code=501, **common)
            statuses[op] = "deferred"
            continue
        owner = OWNER_BY_OPERATION[op]
        handler = routers[owner].handlers.get(op)
        if handler is None:
            reason = f"Not implemented yet: the {owner.value} area has not built {op} (Stage {route.stage})"
            app.add_api_route(route.path, _not_implemented(route, reason=reason, pending=True), status_code=501, **common)
            statuses[op] = "pending"
            continue
        app.add_api_route(
            route.path,
            handler,
            status_code=route.status_code,
            response_model=_response_model(route),
            dependencies=[Depends(guard_for(route))],
            **common,
        )
        verify_handler(route, app.router.routes[-1])
        statuses[op] = "handled"
    return statuses


def route_status(app: FastAPI) -> Dict[str, str]:
    """operation_id -> handled | pending | deferred."""
    return dict(app.state.route_status)


def create_app(config: Optional[StoreConfig] = None, *, clock: Optional[Clock] = None,
               auth_backend: Optional[AuthBackend] = None, static_root: Path = STATIC_ROOT) -> FastAPI:
    config = config or StoreConfig.from_env()
    store = Store.open(config, clock=clock, auth_backend=auth_backend)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Lease expiry and feed pruning must not wait for Process to come back and claim.
        task = None
        if config.maintenance_interval_seconds > 0:
            from .maintenance import run_forever

            task = asyncio.create_task(run_forever(store, config.maintenance_interval_seconds), name="call1-store-maintenance")
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="Call1 Store", version=CONTRACT_VERSION, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store = store
    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware)
    app.state.route_status = register_contract_routes(app, area_routers())
    from .routes.transfer import transfer_router

    app.include_router(transfer_router)
    from .auth.demo import router as demo_router

    app.include_router(demo_router)  # /demo/*: 404 unless demo mode; before the Evaluate catch-all
    mount_frontends(app, static_root)
    return app


__all__ = ["create_app", "route_status", "register_contract_routes", "verify_handler", "HandlerContractError", "STORE_API_PREFIX"]
