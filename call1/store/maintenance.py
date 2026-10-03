"""Store's own housekeeping, so nothing depends on Process being alive.

``run_once`` expires leases, reanalysis claims and upload sessions, applies admission, drops old
orphans (``queue.lifecycle.sweep``) and prunes the change feed past
``ContractParameters.change_feed_retention_seconds`` (older cursors then get 410
``cursor_expired``). ``create_app`` runs it every ``StoreConfig.maintenance_interval_seconds``
while the app is up (``CALL1_STORE_MAINTENANCE_SECONDS``, default 60 for ``serve``, 0 = off).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict

from . import feed

log = logging.getLogger("call1.store.maintenance")


def run_once(store) -> Dict[str, int]:
    from .queue import lifecycle

    counts = dict(lifecycle.sweep(store))
    with store.connection() as conn:
        counts["change_events_pruned"] = feed.prune(conn, retention_seconds=store.config.parameters.change_feed_retention_seconds)
    return counts


async def run_forever(store, interval_seconds: float) -> None:
    """Run ``run_once`` in a worker thread every ``interval_seconds`` until cancelled. A failed pass
    is logged and retried on the next tick; it never stops the loop."""
    while True:
        try:
            counts = await asyncio.to_thread(run_once, store)
            if any(counts.values()):
                log.info("maintenance: %s", counts)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - keep sweeping; the next pass retries
            log.exception("maintenance pass failed")
        await asyncio.sleep(interval_seconds)


__all__ = ["run_once", "run_forever"]
