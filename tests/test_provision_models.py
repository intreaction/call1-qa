"""scripts/provision_models.py licence step: a pinned licence already in place needs no network."""
import hashlib
import urllib.request

import pytest

from scripts import provision_models


def _spec(body: bytes) -> dict:
    return {
        'license': {'url': 'https://example.invalid/LICENSE', 'sha256': hashlib.sha256(body).hexdigest(), 'filename': 'LICENSE.txt'},
        'notice': 'Example model notice.',
    }


def _offline(*_args, **_kwargs):
    raise AssertionError('network access attempted')


def test_licence_already_pinned_is_kept_offline(tmp_path, monkeypatch):
    body = b'Pinned licence text\n'
    (tmp_path / 'LICENSE.txt').write_bytes(body)
    monkeypatch.setattr(urllib.request, 'urlopen', _offline)

    provision_models.install_license(_spec(body), tmp_path)

    assert (tmp_path / 'LICENSE.txt').read_bytes() == body
    assert (tmp_path / 'Notice').read_text() == 'Example model notice.\n'


def test_licence_with_wrong_checksum_is_downloaded_again(tmp_path, monkeypatch):
    body = b'Pinned licence text\n'
    (tmp_path / 'LICENSE.txt').write_bytes(b'tampered\n')
    calls = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return body

    def _fetch(url, timeout):
        calls.append(url)
        return _Response()

    monkeypatch.setattr(urllib.request, 'urlopen', _fetch)
    provision_models.install_license(_spec(body), tmp_path)

    assert calls == ['https://example.invalid/LICENSE']
    assert (tmp_path / 'LICENSE.txt').read_bytes() == body


def test_missing_licence_offline_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, 'urlopen', _offline)
    with pytest.raises(AssertionError):
        provision_models.install_license(_spec(b'x'), tmp_path)
