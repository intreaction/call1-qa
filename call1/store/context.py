"""The running Store: configuration, database, object store, clock and auth backend.

``create_app`` builds one ``Store`` and keeps it on ``app.state.store``. Handlers reach it with
``Depends(deps.get_store)`` and their per-request connection with ``Depends(deps.get_conn)``.
"""

from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from call1.contracts.common import canonical_digest

from .clock import Clock, SystemClock
from .config import StoreConfig
from .db import Database, StoreConnection
from .objects import ObjectStore
from .principals import AuthBackend

_PACKAGE_ROOT = Path(__file__).resolve().parent
_CONTRACTS_ROOT = _PACKAGE_ROOT.parent / "contracts"


@functools.lru_cache(maxsize=1)
def build_manifest_digest() -> str:
    """A digest of the running Store build: every Store and contract source file and migration.

    Stage 2 has no signed release manifest yet (release trust is deferred), so
    ``RunningBuild.manifest_digest`` is this fingerprint and ``approved`` is false.
    """
    entries = []
    for root in (_PACKAGE_ROOT, _CONTRACTS_ROOT):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in (".py", ".sql", ".json") or "__pycache__" in path.parts or " 2." in path.name:
                continue
            entries.append({"path": path.relative_to(_PACKAGE_ROOT.parent).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return canonical_digest(entries)


@dataclass
class Store:
    config: StoreConfig
    db: Database
    objects: ObjectStore
    clock: Clock
    auth: AuthBackend = field(default=None)  # type: ignore[assignment]
    started_at: Optional[datetime] = None

    @classmethod
    def open(cls, config: StoreConfig, *, clock: Optional[Clock] = None, auth_backend: Optional[AuthBackend] = None) -> "Store":
        clock = clock or SystemClock()
        config.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(config.data_dir, 0o700)
        except OSError:
            pass
        database = Database(config.db_path, clock)
        database.initialize()
        objects = ObjectStore(config.objects_dir, base_url=config.public_base_url, parameters=config.parameters, max_upload_bytes=config.max_upload_bytes)
        # On-device training (1.3.0): import labels written before the label log existed, once.
        from .results.training_labels import backfill as backfill_training_labels

        with database.connection() as conn:
            backfill_training_labels(conn)
        store = cls(config=config, db=database, objects=objects, clock=clock, started_at=clock.now())
        if auth_backend is None:
            from .auth.backend import build_auth_backend

            auth_backend = build_auth_backend(store)
        store.auth = auth_backend
        return store

    def connection(self) -> Iterator[StoreConnection]:
        """``with store.connection() as conn:`` for work outside a request (CLI, maintenance)."""
        return self.db.connection()
