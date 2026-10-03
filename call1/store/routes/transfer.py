"""Byte transfer behind upload and download grants (not contract routes).

``PUT /store/transfer/uploads/{upload_id}?token=...`` receives the bytes of an upload session
(``objects.ObjectStore.create_upload``); the one-time token in the grant URL is the authorization,
and the bytes must match the checksum and size declared when the grant was issued.
``GET /store/transfer/downloads/{token}`` streams an object named by a signed, short-lived
``content_grant`` token. Both answer with the contract error envelope.
"""

from __future__ import annotations

import hashlib
import os
import secrets

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response

from call1.contracts.errors import ErrorCode

from ..context import Store
from ..db import StoreConnection
from ..deps import get_conn, get_store
from ..errors import StoreError
from ..objects import TRANSFER_DOWNLOADS, TRANSFER_UPLOADS

transfer_router = APIRouter()


@transfer_router.put(TRANSFER_UPLOADS + "/{upload_id}")
async def put_upload(upload_id: str, request: Request, token: str = Query(min_length=16, max_length=128)) -> Response:
    store: Store = request.app.state.store

    def authorize():
        with store.db.connection() as conn:
            return store.objects.authorize_transfer(conn, upload_id, token)

    record = await run_in_threadpool(authorize)
    staging = store.objects.staging_path(upload_id)
    partial = staging.with_name(f"{staging.name}.{secrets.token_hex(4)}")
    digest = hashlib.sha256()
    size = 0
    try:
        with partial.open("wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > record.expected_size:
                    raise StoreError(ErrorCode.PAYLOAD_TOO_LARGE, "Upload is larger than its grant", details={"max_bytes": record.expected_size})
                digest.update(chunk)
                handle.write(chunk)
        checksum = "sha256:" + digest.hexdigest()
        if checksum != record.expected_checksum or size != record.expected_size:
            raise StoreError(ErrorCode.CHECKSUM_MISMATCH, "Uploaded bytes do not match the grant's checksum and size",
                             details={"reason": "uploaded_bytes_differ", "received_size": size})
        os.replace(partial, staging)
    finally:
        partial.unlink(missing_ok=True)

    def record_received():
        with store.db.connection() as conn:
            store.objects.record_received(conn, upload_id, checksum, size)

    await run_in_threadpool(record_received)
    return JSONResponse({"upload_id": upload_id, "size_bytes": size, "checksum": checksum}, headers={"ETag": f'"{checksum[7:]}"'})


@transfer_router.get(TRANSFER_DOWNLOADS + "/{token}")
def get_download(token: str, store: Store = Depends(get_store), conn: StoreConnection = Depends(get_conn)) -> Response:
    _artifact_id, checksum, content_type = store.objects.verify_download_token(conn, token)
    return store.objects.file_response(checksum, content_type)
