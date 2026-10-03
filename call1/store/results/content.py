"""Reading committed artifact bytes from inside the results area.

The projection hooks receive only the caller's connection, not the ``Store``, so this module finds
Store's object directory from the connection's database file. ``StoreConfig`` fixes the layout:
``<CALL1_STORE_DATA>/store.db`` and ``<CALL1_STORE_DATA>/objects``. Nothing outside Store ever
sees a path (architecture rule 1); results only reads objects by checksum.

JSON artifacts are stored in their canonical form (``artifacts.canonical_content``), so parsing
them with their content model cannot fail for anything Store accepted.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from call1.contracts.common import ContractParameters

from ..db import StoreConnection, conn_path
from ..objects import ObjectStore

M = TypeVar("M", bound=BaseModel)

_lock = threading.Lock()
_stores: Dict[str, ObjectStore] = {}


def objects_for(conn: StoreConnection) -> ObjectStore:
    """The object store beside this connection's database (read-only use)."""
    db_file = Path(conn_path(conn)).resolve()
    key = str(db_file)
    with _lock:
        found = _stores.get(key)
        if found is None:
            found = ObjectStore(db_file.parent / "objects", base_url="", parameters=ContractParameters(), max_upload_bytes=0)
            _stores[key] = found
        return found


def read_bytes(conn: StoreConnection, checksum: str) -> bytes:
    return objects_for(conn).read_bytes(checksum)


def read_model(conn: StoreConnection, checksum: str, model: Type[M]) -> M:
    return model.model_validate(json.loads(read_bytes(conn, checksum)))


def read_optional(conn: StoreConnection, checksum: Optional[str], model: Type[M]) -> Optional[M]:
    return None if checksum is None else read_model(conn, checksum, model)
