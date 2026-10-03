"""The content-addressed object store, verified uploads and signed download grants."""

from __future__ import annotations

import hashlib
from datetime import datetime
from urllib.parse import urlsplit

import pytest

from call1.contracts.artifacts import ContentGrant, UploadGrant
from call1.contracts.common import ServiceScope
from call1.contracts.errors import ErrorCode
from call1.store import db, devmode
from call1.store.config import StoreConfig
from call1.store.errors import StoreError
from call1.store.objects import sha256_checksum

AUDIO = b"RIFF" + bytes(range(256)) * 64


def _path(url: str) -> str:
    parts = urlsplit(url)
    return parts.path + ("?" + parts.query if parts.query else "")


def _grant(store, conn, data: bytes = AUDIO, **kw):
    with db.transaction(conn):
        return store.objects.create_upload(conn, purpose="conversation_artifact", expected_checksum=sha256_checksum(data),
                                           expected_size=len(data), content_type="audio/wav", required_scope=ServiceScope.ARTIFACTS_WRITE,
                                           installation_id="inst_test", metadata={"artifact_id": "art_reserved"}, **kw)


def test_objects_are_content_addressed(store):
    stored = store.objects.put_bytes(b"hello")
    assert stored.checksum == "sha256:" + hashlib.sha256(b"hello").hexdigest() and stored.size_bytes == 5
    assert store.objects.put_bytes(b"hello") == stored
    assert store.objects.exists(stored.checksum) and store.objects.verify(stored.checksum)
    assert store.objects.read_bytes(stored.checksum) == b"hello"
    assert store.objects.delete(stored.checksum) and not store.objects.exists(stored.checksum)
    with pytest.raises(StoreError) as missing:
        store.objects.read_bytes(stored.checksum)
    assert missing.value.code is ErrorCode.NOT_FOUND


def test_upload_put_then_commit(store, conn, client):
    ticket = _grant(store, conn)
    assert ticket.url.startswith("http://localhost:8010/store/transfer/uploads/")
    assert ticket.headers == {"Content-Type": "audio/wav", "Content-Length": str(len(AUDIO))}
    put = client.put(_path(ticket.url), content=AUDIO, headers={"Content-Type": "audio/wav"})
    assert put.status_code == 200 and put.json()["checksum"] == sha256_checksum(AUDIO)
    seen = []
    with db.transaction(conn):
        record = store.objects.commit_upload(conn, ticket.upload_id, checksum=sha256_checksum(AUDIO), size_bytes=len(AUDIO), validate=seen.append)
    assert record.status == "committed" and record.metadata == {"artifact_id": "art_reserved"}
    assert record.required_scope is ServiceScope.ARTIFACTS_WRITE and record.installation_id == "inst_test"
    assert seen == [AUDIO] and store.objects.read_bytes(sha256_checksum(AUDIO)) == AUDIO
    assert not store.objects.staging_path(ticket.upload_id).exists()
    with db.transaction(conn):  # a repeated commit is a replay
        again = store.objects.commit_upload(conn, ticket.upload_id, checksum=sha256_checksum(AUDIO), size_bytes=len(AUDIO))
    assert again.committed_at == record.committed_at
    assert client.put(_path(ticket.url), content=AUDIO).json()["code"] == "conflict"


def test_commit_checks(store, conn, client):
    ticket = _grant(store, conn)

    def commit(checksum=sha256_checksum(AUDIO), size=len(AUDIO), **kw):
        with pytest.raises(StoreError) as error, db.transaction(conn):
            store.objects.commit_upload(conn, ticket.upload_id, checksum=checksum, size_bytes=size, **kw)
        return error.value

    nothing = commit()
    assert nothing.code is ErrorCode.CHECKSUM_MISMATCH and nothing.details["reason"] == "no_bytes_received"
    client.put(_path(ticket.url), content=AUDIO)
    assert commit(checksum=sha256_checksum(b"other")).details["reason"] == "commit_differs_from_grant"

    def reject(_: bytes) -> None:
        raise StoreError(ErrorCode.VALIDATION_FAILED, "not canonical")

    assert commit(validate=reject).code is ErrorCode.VALIDATION_FAILED
    assert store.objects.get_upload(conn, ticket.upload_id).status == "received"
    with pytest.raises(StoreError) as unknown, db.transaction(conn):
        store.objects.commit_upload(conn, "upl_missing", checksum=sha256_checksum(AUDIO), size_bytes=len(AUDIO))
    assert unknown.value.code is ErrorCode.NOT_FOUND


def test_put_is_checked_against_the_grant(store, conn, client, clock):
    ticket = _grant(store, conn)
    wrong = client.put(_path(ticket.url), content=AUDIO[:-1] + b"x")
    assert wrong.status_code == 422 and wrong.json()["code"] == "checksum_mismatch"
    too_big = client.put(_path(ticket.url), content=AUDIO + b"extra")
    assert too_big.status_code == 413 and too_big.json()["code"] == "payload_too_large"
    bad_token = client.put(f"/store/transfer/uploads/{ticket.upload_id}?token={'A' * 43}", content=AUDIO)
    assert bad_token.status_code == 404
    assert store.objects.get_upload(conn, ticket.upload_id).status == "pending"
    clock.advance(seconds=store.config.parameters.upload_grant_lifetime_seconds + 1)
    expired = client.put(_path(ticket.url), content=AUDIO)
    assert expired.status_code == 410 and expired.json()["code"] == "upload_expired"
    assert store.objects.sweep_expired_uploads(conn) == 1 and store.objects.get_upload(conn, ticket.upload_id) is None


def test_download_grants(store, conn, client, clock):
    stored = store.objects.put_bytes(AUDIO)
    ticket = store.objects.content_grant(conn, artifact_id="art_1", checksum=stored.checksum, content_type="audio/wav")
    assert ticket.url.startswith("http://localhost:8010/store/transfer/downloads/")
    response = client.get(_path(ticket.url))
    assert response.status_code == 200 and response.content == AUDIO and response.headers["content-type"] == "audio/wav"
    ranged = client.get(_path(ticket.url), headers={"Range": "bytes=0-3"})
    assert ranged.status_code == 206 and ranged.content == b"RIFF"
    tampered = _path(ticket.url)[:-2] + ("AA" if not _path(ticket.url).endswith("AA") else "BB")
    assert client.get(tampered).json()["code"] == "forbidden"
    clock.advance(seconds=store.config.parameters.download_grant_lifetime_seconds + 1)
    assert client.get(_path(ticket.url)).json()["details"]["reason"] == "grant_expired"


def test_devmode_emits_dev_urls_but_validates_everything_else(store, tmp_path):
    grant = {"upload_id": "upl_1", "artifact_id": "art_1", "url": "http://localhost:8010/store/transfer/uploads/upl_1?token=t",
             "headers": {}, "expires_at": datetime.fromisoformat("2026-09-25T12:15:00+00:00"), "max_bytes": 10}
    body = devmode.validate(UploadGrant, grant, store.config)
    assert body["url"] == grant["url"] and body["method"] == "PUT" and body["expires_at"] == "2026-09-25T12:15:00Z"
    with pytest.raises(ValueError):
        devmode.validate(UploadGrant, {**grant, "max_bytes": 0}, store.config)
    production = StoreConfig.for_tests(tmp_path / "p", hostname="qa.example.com")
    with pytest.raises(ValueError):
        devmode.validate(UploadGrant, grant, production)
    https = {**grant, "url": "https://qa.example.com:8010/store/transfer/uploads/upl_1?token=t"}
    assert devmode.validate(UploadGrant, https, production)["url"] == https["url"]
    content = {"artifact_id": "art_1", "url": "http://localhost:8010/store/transfer/downloads/x", "expires_at": grant["expires_at"],
               "checksum": sha256_checksum(b"x"), "content_type": "audio/wav", "size_bytes": 1}
    assert devmode.validate(ContentGrant, content, store.config)["url"] == content["url"]
