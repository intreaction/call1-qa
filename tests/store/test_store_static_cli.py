"""Evaluate and console hosting, request IDs, and the ``python -m call1.store`` commands."""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from call1.contracts.auth import ServiceKeyIssued, ServiceKeyRecord
from call1.store import __main__ as cli
from call1.store.app import create_app
from call1.store.config import StoreConfig


def test_placeholders_until_the_apps_are_built(tmp_path, clock):
    app = create_app(StoreConfig.for_tests(tmp_path / "s"), clock=clock, static_root=tmp_path / "static-missing")
    client = TestClient(app)
    for path, title in (("/", "Call1 Evaluate"), ("/workbench/calls/c1", "Call1 Evaluate"), ("/console/", "Call1 Store"), ("/console", "Call1 Store")):
        page = client.get(path)
        assert page.status_code == 200 and title in page.text and "Not built yet" in page.text, path
    assert client.get("/assets/app.js").status_code == 404


def test_built_apps_are_served_with_spa_fallback(tmp_path, clock):
    static = tmp_path / "static"
    (static / "evaluate" / "assets").mkdir(parents=True)
    (static / "evaluate" / "index.html").write_text("<html>evaluate</html>")
    (static / "evaluate" / "assets" / "app.js").write_text("console.log(1)")
    (static / "store-console.html").write_text("<html>console</html>")
    (tmp_path / "secret.txt").write_text("outside")
    client = TestClient(create_app(StoreConfig.for_tests(tmp_path / "s"), clock=clock, static_root=static))
    assert client.get("/").text == "<html>evaluate</html>"
    assert client.get("/review-queue").text == "<html>evaluate</html>"
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/console/").text == "<html>console</html>"
    assert client.get("/missing.js").status_code == 404
    assert client.get("/..%2Fsecret.txt").status_code == 404


def test_unknown_api_paths_are_envelopes_not_pages(client):
    response = client.get("/store/v1/no-such-thing")
    assert response.status_code == 404 and response.json()["code"] == "not_found"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert client.get("/store").json()["code"] == "not_found"


def test_migrate_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CALL1_STORE_DATA", str(tmp_path / "data"))
    assert cli.main(["migrate"]) == 0
    assert "010" in capsys.readouterr().out
    assert cli.main(["migrate"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_serve_binds_loopback_in_dev_and_refuses_plaintext_production(tmp_path, monkeypatch):
    import uvicorn

    calls = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.update(kw))
    monkeypatch.setenv("CALL1_STORE_DATA", str(tmp_path / "data"))
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert calls["host"] == "127.0.0.1" and calls["port"] == 8123 and "ssl_certfile" not in calls
    monkeypatch.setenv("CALL1_STORE_HOSTNAME", "qa.example.com")
    assert cli.main(["serve"]) == 2  # no TLS certificate configured


def test_host_commands_report_the_pending_auth_area(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CALL1_STORE_DATA", str(tmp_path / "data"))
    from call1.store.auth import cli as auth_cli

    def pending(*args, **kwargs):
        raise NotImplementedError("auth pending")

    monkeypatch.setattr(auth_cli, "issue_setup_code", pending)
    monkeypatch.setattr(auth_cli, "issue_service_key", pending)
    assert cli.main(["setup-code", "--email", "admin@example.com", "--display-name", "Admin"]) == 2
    assert cli.main(["issue-service-key", "--installation", "mac-mini"]) == 2
    assert "auth pending" in capsys.readouterr().err


def test_issue_service_key_writes_process_config_0600(tmp_path, monkeypatch):
    monkeypatch.setenv("CALL1_STORE_DATA", str(tmp_path / "data"))
    from call1.store.auth import cli as auth_cli

    token = "c1sk_Abc123_" + "x" * 43
    seen = {}

    def issue(store, *, installation_label, scopes, primary_host):
        seen.update(label=installation_label, scopes=list(scopes), primary_host=primary_host)
        record = ServiceKeyRecord(id="key_1", installation_id="inst_1", label=installation_label, scopes=list(scopes), key_prefix="c1sk_Abc123",
                                  key_hash="sha256:" + "0" * 64, created_at=datetime(2026, 9, 25, tzinfo=timezone.utc))
        return ServiceKeyIssued(key=record, token=token)

    monkeypatch.setattr(auth_cli, "issue_service_key", issue)
    config_path = tmp_path / "process" / "config.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"worker_slots": 2}))
    assert cli.main(["issue-service-key", "--installation", "mac-mini", "--config", str(config_path)]) == 0
    assert seen["label"] == "mac-mini" and seen["primary_host"] is None \
        and {s.value for s in seen["scopes"]} == {"calls:write", "artifacts:read", "artifacts:write", "jobs:write", "jobs:claim", "reanalysis:claim",
                                                  "changes:read", "hardware:write", "catalog:publish", "usage:read",
                                                  "admin-state:read", "training:read"}
    written = json.loads(config_path.read_text())
    assert written == {"worker_slots": 2, "store_url": "http://localhost:8010", "installation_id": "inst_1", "service_key_id": "key_1", "service_key": token}
    assert stat.S_IMODE(os.stat(config_path).st_mode) == 0o600
    assert cli.main(["issue-service-key", "--installation", "b", "--scope", "jobs:claim", "--no-primary-host", "--print-token"]) == 0
    assert seen["scopes"] == ["jobs:claim"] and seen["primary_host"] is False


@pytest.mark.parametrize("argv", [["setup-code", "--email", "a@example.com"], ["issue-service-key"], ["bogus"]])
def test_cli_rejects_incomplete_commands(argv):
    with pytest.raises(SystemExit):
        cli.main(argv)
