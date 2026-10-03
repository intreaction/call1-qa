"""``GET /status``, ``GET /status/detail`` and ``GET /contract``.

Stage 2 gaps (raised with the orchestrator; the contract is not patched):

* TLS state is Stage 4, so ``StoreHealth.tls_health`` is ``untrusted`` (no certificate Store has
  verified) and ``StoreStatus.tls`` is ``null`` although the contract types it as required.
* Admin state is not in the Stage 2 scope, so ``admin_state_version`` is 0 (installer defaults,
  never changed).
* Release trust is deferred, so ``running_build.manifest_digest`` is a fingerprint of the running
  Store and contract sources (``context.build_manifest_digest``) and ``approved`` is false.
* In dev mode the relying party is ``localhost`` with an ``http://`` origin (``devmode``).

``StoreStatus.search_embedder`` (1.2.0) reports the local search embedder (``results.search``)
without loading it: ``not_installed``, ``installed`` (loads on the first search), ``loaded``,
``failed`` or ``fake``.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Depends
from pydantic import TypeAdapter
from starlette.responses import JSONResponse

from call1.contracts.admin import ObjectStoreKind, StoreHealth, TlsHealth, WebAuthnRelyingParty
from call1.contracts.common import CONTRACT_VERSION, ChangeCursor, ContractInfo, ResourceId, Timestamp
from call1.contracts.release_trust import RunningBuild

from .. import devmode, feed
from ..context import Store, build_manifest_digest
from ..db import StoreConnection, read_snapshot, schema_version
from ..deps import get_conn, get_store
from ..results.search import embedder_status
from . import router

ADMIN_STATE_VERSION = 0
"""Admin state is deferred in Stage 2: the installer defaults, never changed."""


def contract_info(store: Store) -> ContractInfo:
    return ContractInfo(contract_version=CONTRACT_VERSION, parameters=store.config.parameters)


def relying_party(store: Store) -> Dict[str, Any]:
    return devmode.validate(
        WebAuthnRelyingParty,
        {"rp_id": store.config.rp_id, "rp_name": "Call1 Store", "allowed_origins": list(store.config.allowed_origins)},
        store.config,
    )


def tls_health(store: Store) -> TlsHealth:
    return TlsHealth.UNTRUSTED  # Stage 4 supplies TlsState; until then Store has verified no certificate


def running_build(store: Store) -> Dict[str, Any]:
    return RunningBuild(manifest_digest=build_manifest_digest(), approved=False, started_at=store.started_at).model_dump(mode="json")


@router.operation("getStatus")
def get_status(store: Store = Depends(get_store)) -> JSONResponse:
    payload = {
        "contract": contract_info(store),
        "store_hostname": store.config.hostname,
        "relying_party": relying_party(store),
        "tls_health": tls_health(store),
        "server_time": store.clock.now(),
    }
    return devmode.respond(StoreHealth, payload, store.config)


@router.operation("getStatusDetail")
def get_status_detail(store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn)) -> JSONResponse:
    with read_snapshot(conn):
        version = schema_version(conn)
        epoch = feed.feed_epoch(conn)
        latest = feed.latest_cursor(conn)
    body = {
        "contract": contract_info(store).model_dump(mode="json"),
        "store_hostname": store.config.hostname,
        "relying_party": relying_party(store),
        "tls": None,  # Stage 4 (see module docstring)
        "running_build": running_build(store),
        "schema_version": max(version, 1),
        "object_store_kind": ObjectStoreKind.LOCAL_DIRECTORY.value,
        "admin_state_version": ADMIN_STATE_VERSION,
        "feed_epoch": TypeAdapter(ResourceId).validate_python(epoch),
        "latest_change_cursor": TypeAdapter(ChangeCursor).validate_python(latest),
        "server_time": TypeAdapter(Timestamp).dump_python(store.clock.now(), mode="json"),
        "search_embedder": embedder_status(),
    }
    return JSONResponse(body)


@router.operation("getContract")
def get_contract(store: Store = Depends(get_store)) -> ContractInfo:
    return contract_info(store)
