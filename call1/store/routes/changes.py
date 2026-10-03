"""``GET /store/v1/changes`` (listChanges): the change feed, filtered by principal."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query

from call1.contracts.events import ChangeFeed, ChangeFeedQuery

from .. import feed
from ..context import Store
from ..db import StoreConnection
from ..deps import current_principal, get_conn, get_store
from ..principals import Principal
from . import router


@router.operation("listChanges")
def list_changes(
    query: Annotated[ChangeFeedQuery, Query()],
    store: Store = Depends(get_store),
    conn: StoreConnection = Depends(get_conn),
    principal: Principal = Depends(current_principal),
) -> ChangeFeed:
    return feed.read(
        conn,
        audience=principal.feed_audience,
        after=query.after,
        limit=query.limit,
        kinds=query.kinds,
        retention_seconds=store.config.parameters.change_feed_retention_seconds,
    )
