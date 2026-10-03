"""The host commands through ``python -m call1.store``: ``setup-code`` prints a code that enrolls
the first admin, ``issue-service-key`` registers the installation, issues a working key and writes
Process's config (mode 0600). Both run on the Store's own database with the system clock."""

from __future__ import annotations

import json
import os
import stat

import pytest
from fastapi.testclient import TestClient

from call1.contracts.common import ServiceScope
from call1.contracts.errors import ErrorCode
from call1.store import __main__ as cli
from call1.store import audit
from call1.store.app import create_app
from call1.store.auth import cli as auth_cli
from call1.store.config import StoreConfig
from call1.store.context import Store

from .auth_flows import API, enroll
from .auth_softauthn import ORIGIN, SoftAuthenticator


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    path = tmp_path / "data"
    monkeypatch.setenv("CALL1_STORE_DATA", str(path))
    for name in ("CALL1_STORE_HOSTNAME", "CALL1_STORE_DEV", "CALL1_STORE_PARAMETERS", "CALL1_PROCESS_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    return path


def _app(data_dir):
    return create_app(StoreConfig.for_tests(data_dir))  # system clock, like the host command


def test_setup_code_command_prints_a_code_that_enrolls_the_first_admin(data_dir, capsys):
    assert cli.main(["setup-code", "--email", "it@example.com", "--display-name", "IT Admin"]) == 0
    out = capsys.readouterr()
    code = out.out.strip().splitlines()[-1]
    assert "it@example.com (first_admin)" in out.out and "works once" in out.err
    client = TestClient(_app(data_dir), base_url=ORIGIN)
    enrolled = enroll(client, SoftAuthenticator(), setup_code=code)
    assert enrolled.status_code == 200, enrolled.text
    assert enrolled.json()["account"]["role"] == "admin" and enrolled.json()["account"]["email"] == "it@example.com"
    refused = cli.main(["setup-code", "--email", "other@example.com", "--display-name", "Other"])
    assert refused == 2 and "active administrator exists" in capsys.readouterr().err
    store = Store.open(StoreConfig.for_tests(data_dir))
    with store.connection() as conn:
        issued = conn.execute("SELECT actor_json FROM audit_events WHERE action = 'setup_code_issued'").fetchone()
        assert json.loads(issued["actor_json"])["kind"] == "installer" and audit.verify_chain(conn)
        assert conn.execute("SELECT COUNT(*) FROM auth_setup_codes WHERE code_hash LIKE 'sha256:%'").fetchone()[0] == 1


def test_break_glass_command_targets_an_existing_account(data_dir, capsys):
    assert cli.main(["setup-code", "--email", "it@example.com", "--display-name", "IT"]) == 0
    code = capsys.readouterr().out.strip().splitlines()[-1]
    app = _app(data_dir)
    account_id = enroll(TestClient(app, base_url=ORIGIN), SoftAuthenticator(), setup_code=code).json()["account"]["id"]
    assert cli.main(["setup-code", "--purpose", "break_glass", "--email", "nobody@example.com", "--display-name", "X",
                     "--target-account-id", account_id]) == 2
    assert "must match" in capsys.readouterr().err
    assert cli.main(["setup-code", "--purpose", "break_glass", "--email", "it@example.com", "--display-name", "IT",
                     "--target-account-id", account_id]) == 0
    code = capsys.readouterr().out.strip().splitlines()[-1]
    recovered = enroll(TestClient(app, base_url=ORIGIN), SoftAuthenticator(), setup_code=code)
    assert recovered.status_code == 200 and recovered.json()["account"]["id"] == account_id


def test_issue_service_key_writes_process_config_and_the_key_works(data_dir, tmp_path, capsys):
    config_path = tmp_path / "process" / "config.json"
    assert cli.main(["issue-service-key", "--installation", "mac-mini", "--config", str(config_path)]) == 0
    written = json.loads(config_path.read_text())
    assert set(written) == {"store_url", "installation_id", "service_key_id", "service_key"}
    assert written["store_url"] == "http://localhost:8010" and stat.S_IMODE(os.stat(config_path).st_mode) == 0o600
    client = TestClient(_app(data_dir), base_url=ORIGIN)
    headers = {"Authorization": f"Bearer {written['service_key']}"}
    assert client.get(f"{API}/status/detail", headers=headers).status_code == 200
    store = client.app.state.store
    with store.connection() as conn:
        principal = store.auth.authenticate_service_key(conn, written["service_key"], store.clock.now())
        assert principal.installation_id == written["installation_id"] and principal.primary_host is True
        assert principal.key_id == written["service_key_id"] and principal.scopes == frozenset(cli.PROCESS_DEFAULT_SCOPES)
        assert not principal.scopes & {ServiceScope.JOBS_CONTROL, ServiceScope.KEY_RELEASE_WRITE, ServiceScope.RELEASE_TRUST_WRITE}
        stored = conn.execute("SELECT key_hash FROM auth_service_keys").fetchone()["key_hash"]
        assert written["service_key"] not in stored
        actors = [json.loads(r["actor_json"])["kind"] for r in conn.execute("SELECT actor_json FROM audit_events ORDER BY sequence")]
        assert actors == ["installer", "installer"]
    # the same label reuses the installation; another installation is not primary unless asked (and it cannot be while one is)
    assert cli.main(["issue-service-key", "--installation", "mac-mini", "--scope", "jobs:claim", "--config", str(config_path)]) == 0
    again = json.loads(config_path.read_text())
    assert again["installation_id"] == written["installation_id"] and again["service_key"] != written["service_key"]
    assert cli.main(["issue-service-key", "--installation", "studio", "--print-token"]) == 0
    token = capsys.readouterr().out.strip().splitlines()[-1]
    with store.connection() as conn:
        assert store.auth.authenticate_service_key(conn, token, store.clock.now()).primary_host is False
    assert cli.main(["issue-service-key", "--installation", "garage", "--primary-host", "--print-token"]) == 2
    assert "primary host" in capsys.readouterr().err


def test_default_process_config_path_comes_from_the_environment(data_dir, tmp_path, monkeypatch):
    target = tmp_path / "elsewhere" / "process.json"
    monkeypatch.setenv("CALL1_PROCESS_CONFIG", str(target))
    assert cli.main(["issue-service-key", "--installation", "mac-mini", "--no-primary-host"]) == 0
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    store = Store.open(StoreConfig.for_tests(data_dir))
    with store.connection() as conn:
        assert conn.execute("SELECT primary_host FROM auth_installations").fetchone()["primary_host"] == 0


def test_refusals_are_store_errors_for_programmatic_callers(data_dir):
    store = Store.open(StoreConfig.for_tests(data_dir))
    auth_cli.issue_service_key(store, installation_label="a", scopes=["jobs:claim"], primary_host=True, os_user="root")
    with pytest.raises(auth_cli.HostCommandRefused) as refused:
        auth_cli.issue_service_key(store, installation_label="b", scopes=["jobs:claim"], primary_host=True, os_user="root")
    assert refused.value.code is ErrorCode.CONFLICT and refused.value.details["reason"] == "primary_host_taken"
