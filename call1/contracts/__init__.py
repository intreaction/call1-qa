"""The frozen Store contract: the only thing Process, Store and Evaluate share.

Public surface:
- ``CONTRACT_VERSION`` and ``STORE_API_PREFIX``
- the area modules (``calls``, ``contents``, ``artifacts``, ``jobs``, ``reviews``, ``rubrics``,
  ``signals`` (1.3.0), ``training`` (1.3.0), ``vocabulary`` (1.3.0), ``auth``, ``custody``, ``release_trust``, ``usage``,
  ``admin``, ``events``, ``catalog``, ``metrics``, ``errors``, ``common``), each exporting strict
  Pydantic v2 models and explicit enums
- ``common.canonical_json`` / ``canonical_digest`` (RFC 8785), the only way the contract hashes JSON
- ``api.ROUTES`` and ``api.build_app()`` for the HTTP surface and its OpenAPI document
- ``openapi.json`` (committed) and ``frontend/src/contracts/store-v1.ts`` (generated types)

Change rule: this package changes only in dedicated contract commits, which every track then
picks up. See README.md in this directory.
"""

from __future__ import annotations

import inspect
from typing import Dict, Type

from . import admin, api, artifacts, auth, calls, catalog, common, contents, custody, errors, events, jobs, metrics, release_trust, reviews, rubrics, signals, training, usage, vocabulary
from .api import ROUTES, build_app, build_openapi
from .common import CONTRACT_PARAMETERS, CONTRACT_VERSION, STORE_API_PREFIX, ContractModel, canonical_digest, canonical_json

AREA_MODULES = (common, errors, catalog, vocabulary, contents, custody, usage, rubrics, signals, artifacts, calls, jobs, reviews, training, auth, release_trust, admin, events, metrics, api)


def all_models() -> Dict[str, Type[ContractModel]]:
    """Every contract model, keyed by class name (names are unique across the package)."""
    found: Dict[str, Type[ContractModel]] = {}
    for module in AREA_MODULES:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if "[" in name:  # parametrized generics such as Page[Artifact] are not separate contract models
                continue
            if issubclass(obj, ContractModel) and obj is not ContractModel and obj.__module__ == module.__name__:
                if name in found and found[name] is not obj:
                    raise RuntimeError(f"duplicate contract model name {name}")
                found[name] = obj
    return found


__all__ = [
    "CONTRACT_VERSION", "STORE_API_PREFIX", "CONTRACT_PARAMETERS", "ContractModel", "ROUTES", "build_app",
    "build_openapi", "all_models", "AREA_MODULES", "canonical_json", "canonical_digest",
    "admin", "api", "artifacts", "auth", "calls", "catalog", "common", "contents", "custody", "errors", "events", "jobs",
    "metrics", "release_trust", "reviews", "rubrics", "signals", "training", "usage", "vocabulary",
]
