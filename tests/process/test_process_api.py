"""The Process operator API and console: loopback only, console credential on writes, the views the
console renders, recording upload, and the static console with its placeholder."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from call1.contracts.common import CONTRACT_VERSION
from call1.process import console
from call1.process.app import create_app

from .conftest import SAMPLE, STORE_URL

LOCAL = "http://127.0.0.1:8020"


@pytest.fixture
def api(make_runtime, tmp_path):
    runtime = make_runtime()
    runtime.connect()
    token, _ = console.issue(runtime.config)
    static = tmp_path / "static"
    static.mkdir()
    app = create_app(runtime, start_background=False, loopback_only=True, static_root=static)
    with TestClient(app, base_url=LOCAL, client=("127.0.0.1", 50000)) as client:
        client.runtime = runtime  # type: ignore[attr-defined]
        client.token = token  # type: ignore[attr-defined]
        client.static = static  # type: ignore[attr-defined]
        yield client


def test_only_loopback_clients_and_hosts_are_served(make_runtime):
    runtime = make_runtime()
    app = create_app(runtime, start_background=False, loopback_only=True)
    with TestClient(app, base_url=LOCAL, client=("10.0.0.5", 50000)) as remote:
        assert remote.get("/process/api/health").status_code == 403
    with TestClient(app, base_url="http://attacker.example", client=("127.0.0.1", 50000)) as rebound:
        response = rebound.get("/process/api/overview")
        assert response.status_code == 403 and response.json()["code"] == "loopback_only"
    with TestClient(app, base_url=LOCAL, client=("127.0.0.1", 50000)) as local:
        assert local.get("/process/api/health").json()["ok"] is True


def test_session_marks_only_demo_launches(api, monkeypatch):
    monkeypatch.delenv("CALL1_STORE_DEMO", raising=False)
    assert api.get("/process/api/session").json()["demo"] is False
    monkeypatch.setenv("CALL1_STORE_DEMO", "1")
    assert api.get("/process/api/session").json()["demo"] is True


def test_demo_recording_is_credential_and_demo_gated(api, monkeypatch):
    monkeypatch.delenv("CALL1_STORE_DEMO", raising=False)
    assert api.post("/process/api/demo/recordings").status_code == 401
    headers = {console.HEADER: api.token}
    assert api.post("/process/api/demo/recordings", headers=headers).status_code == 404
    monkeypatch.setenv("CALL1_STORE_DEMO", "1")
    first = api.post("/process/api/demo/recordings", headers=headers)
    second = api.post("/process/api/demo/recordings", headers=headers)
    assert first.status_code == second.status_code == 201
    assert first.json()["conversation_id"] != second.json()["conversation_id"]
    assert first.json()["graph_created"] and second.json()["graph_created"]


def test_writes_need_the_console_credential_and_a_loopback_origin(api):
    session = api.get("/process/api/session").json()
    assert session["console_credential_configured"] is True and session["token_valid"] is False
    assert api.get("/process/api/session", headers={console.HEADER: api.token}).json()["token_valid"] is True
    files = {"file": ("call.wav", SAMPLE.read_bytes(), "audio/wav")}
    assert api.post("/process/api/recordings", files=files).status_code == 401
    assert api.post("/process/api/recordings", files=files, headers={console.HEADER: "c1con_wrong"}).status_code == 401
    cross = api.post("/process/api/recordings", files=files, headers={console.HEADER: api.token, "Origin": "https://evil.example"})
    assert cross.status_code == 403 and cross.json()["code"] == "origin_not_allowed"


def test_upload_ingests_and_the_views_follow_progress(api):
    response = api.post("/process/api/recordings", files={"file": ("call_01.wav", SAMPLE.read_bytes(), "audio/wav")},
                        data={"agent_id": "agent-9"}, headers={console.HEADER: api.token, "Origin": LOCAL})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["evaluate_url"] == f"{STORE_URL}/#/calls/{body['call_id']}" and body["jobs"] >= 10

    listed = api.get("/process/api/conversations").json()
    [item] = listed["items"]
    assert item["conversation_id"] == body["conversation_id"] and item["label"] == "call_01.wav"
    assert item["progress"]["settled"] is False and item["progress_line"].startswith("Transcript")

    api.runtime.worker.drain()
    detail = api.get(f"/process/api/conversations/{body['conversation_id']}").json()
    assert detail["progress"]["settled"] is True and {j["status"] for j in detail["jobs"]} == {"SUCCEEDED"}
    assert "Transcript ready" in detail["progress_line"] and "QA ready" in detail["progress_line"]
    asr = next(j for j in detail["jobs"] if j["job_type"] == "asr")
    assert asr["catalog_entry"] == "parakeet-tdt-0.6b-v3" and asr["route_class"] == "appliance" and asr["destination_host"] == "in-process"
    job = api.get(f"/process/api/jobs/{asr['id']}").json()
    assert job["job"]["status"] == "SUCCEEDED" and job["attempts"][0]["provenance"]["adapter_id"] == "fake.asr"
    assert job["evaluate_url"].endswith(body["call_id"])


def test_conversation_jobs_carry_dependencies_and_run_times(api):
    """The Pipeline view orders jobs by their upstreams and shows how long each ran."""
    body = _upload(api).json()
    pending = api.get(f"/process/api/conversations/{body['conversation_id']}").json()
    assert all(j["started_at"] is None and j["ended_at"] is None for j in pending["jobs"] if j["status"] in ("QUEUED", "BLOCKED"))
    api.runtime.worker.drain()
    jobs = api.get(f"/process/api/conversations/{body['conversation_id']}").json()["jobs"]
    by_id = {j["id"]: j for j in jobs}
    for j in jobs:
        assert set(j["requires_job_ids"]) | set(j["after_job_ids"]) <= set(by_id)
        assert j["started_at"] is not None and j["ended_at"] is not None and j["started_at"] <= j["ended_at"]
    asr = next(j for j in jobs if j["job_type"] == "asr")
    assert any(asr["id"] in j["requires_job_ids"] for j in jobs)  # something waits on the transcript
    calls = []
    original = api.runtime.client.list_attempts
    api.runtime.client.list_attempts = lambda job_id: calls.append(job_id) or original(job_id)
    try:
        again = api.get(f"/process/api/conversations/{body['conversation_id']}").json()["jobs"]
    finally:
        api.runtime.client.list_attempts = original
    assert calls == [] and [j["started_at"] for j in again] == [j["started_at"] for j in jobs]  # finished jobs are cached


def _upload(api, data=None, name="call_01.wav"):
    return api.post("/process/api/recordings", files={"file": (name, SAMPLE.read_bytes(), "audio/wav")}, data=data or {},
                    headers={console.HEADER: api.token, "Origin": LOCAL})


def test_upload_carries_agent_identity_and_a_changed_reupload_updates_the_call(api, session):
    first = _upload(api, {"agent_id": "a-104", "agent_display_name": " Samantha ", "agent_extension": "104", "external_call_ref": "pbx-1"})
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["conversation_created"] and body["metadata_updated"] is False and body["updated_fields"] == []
    assert body["agent_label"] == "Samantha (104)"
    stored = api.runtime.client.get_conversation(body["conversation_id"]).call_metadata
    assert (stored.agent_id, stored.agent_display_name, stored.agent_extension, stored.external_call_ref) == ("a-104", "Samantha", "104", "pbx-1")

    # the same recording with nothing new is a pure replay; absent fields are not reset
    same = _upload(api).json()
    assert (same["conversation_id"], same["graph_id"]) == (body["conversation_id"], body["graph_id"])
    assert same["metadata_updated"] is False and same["agent_label"] == "Samantha (104)"
    stored = api.runtime.client.get_conversation(body["conversation_id"]).call_metadata
    assert (stored.agent_id, stored.agent_extension, stored.external_call_ref) == ("a-104", "104", "pbx-1")

    # different metadata updates the call, and nothing is reprocessed
    changed = _upload(api, {"agent_display_name": "Bob", "agent_extension": "202", "agent_channel": "1"}).json()
    assert changed["metadata_updated"] is True and changed["conversation_created"] is False and changed["graph_created"] is False
    assert changed["updated_fields"] == ["agent_display_name", "agent_extension", "agent_channel"]
    assert changed["graph_id"] == body["graph_id"] and changed["agent_label"] == "Bob (202)"
    stored = api.runtime.client.get_conversation(body["conversation_id"]).call_metadata
    assert (stored.agent_id, stored.agent_display_name, stored.agent_extension, stored.agent_channel) == ("a-104", "Bob", "202", 1)
    graphs = {j.graph_id for j in api.runtime.client.list_jobs(conversation_id=body["conversation_id"])}
    assert graphs == {body["graph_id"]}


def test_upload_refuses_agent_metadata_the_contract_rejects(api):
    bad = _upload(api, {"agent_extension": "10 4"})
    assert bad.status_code == 422 and bad.json()["code"] == "validation_failed" and "agent_extension" in bad.json()["message"]
    long = _upload(api, {"agent_display_name": "x" * 101})
    assert long.status_code == 422 and "agent_display_name" in long.json()["message"]
    assert api.get("/process/api/conversations").json()["items"] == []  # nothing was registered


def test_unsupported_recordings_are_refused(api):
    response = api.post("/process/api/recordings", files={"file": ("notes.txt", b"hello there", "text/plain")},
                        headers={console.HEADER: api.token})
    assert response.status_code == 415 and response.json()["code"] == "unsupported_audio"


def test_overview_and_catalog_describe_the_installation(api):
    overview = api.get("/process/api/overview").json()
    assert overview["app"] == "Call1 Process" and overview["state"] == "connected"
    assert overview["store"]["contract_version"] == CONTRACT_VERSION and overview["store"]["compatible"] is True and overview["store"]["dev_mode"]
    assert {p["pool"] for p in overview["worker"]["pools"]} == {"mlx", "torch", "cpu_io"}
    assert next(p for p in overview["worker"]["pools"] if p["pool"] == "mlx")["size"] == 1
    assert overview["handlers"]["mode"] == "fake" and overview["handlers"]["missing_job_types"] == []
    assert overview["catalog"]["published"]["catalog_version"] == overview["catalog"]["version"]
    catalog = api.get("/process/api/catalog").json()
    assert catalog["defaults"]["asr"] == "parakeet-tdt-0.6b-v3" and catalog["defaults"]["semantic_qa"] == "call1-bundled"
    entry = next(e for e in catalog["entries"] if e["entry_id"] == "roberta-sentiment")
    assert entry["runtime"] == "torch" and entry["status"] == "available" and entry["route_class"] == "appliance"


def test_the_console_is_served_with_a_placeholder_until_built(api):
    page = api.get("/")
    assert page.status_code == 200 and "Not built yet" in page.text and "Call1 Process" in page.text
    (api.static / "process.html").write_text("<!doctype html><title>Call1 Process</title><div id=root></div>")
    (api.static / "assets").mkdir()
    (api.static / "assets" / "app.js").write_text("console.log(1)")
    assert "id=root" in api.get("/").text and "id=root" in api.get("/some/client/route").text
    assert api.get("/assets/app.js").text == "console.log(1)"
    assert api.get("/assets/missing.js").status_code == 404
    unknown = api.get("/process/api/nope")
    assert unknown.status_code == 404 and unknown.json()["code"] == "not_found"


def test_console_credential_is_stored_hashed(make_runtime):
    runtime = make_runtime()
    token, record = console.issue(runtime.config)
    stored = runtime.config.config_path.read_text()
    assert token not in stored and record["credential_hash"] in stored and token.startswith("c1con_")
    with pytest.raises(FileExistsError):
        console.issue(runtime.config)
    rotated, _ = console.issue(runtime.config, rotate=True)
    auth = console.ConsoleAuth(runtime.config.config_path)
    assert auth.check(rotated) and not auth.check(token)


def test_the_pipeline_listing_answers_at_once_during_a_store_outage(api):
    import httpx

    for name in ("call_01_compliant.wav", "call_02_critical_breach.wav", "call_03_dispute_escalation.wav"):
        data = (SAMPLE.parent / name).read_bytes()
        response = api.post("/process/api/recordings", files={"file": (name, data, "audio/wav")}, headers={console.HEADER: api.token})
        assert response.status_code == 201, response.text
    listed = api.get("/process/api/conversations").json()
    assert listed["store_unavailable"] is False and all(i["progress_line"] and i["progress_stale"] is False for i in listed["items"])

    store = api.runtime.client
    calls = []

    def down(method, url, **kwargs):
        calls.append((url, kwargs.get("timeout")))
        raise httpx.ConnectError("Store stopped")

    real_request = store.http.request
    store.http.request = down  # type: ignore[method-assign]
    try:
        during = api.get("/process/api/conversations").json()
    finally:
        store.http.request = real_request  # type: ignore[method-assign]
    # one short try for the whole listing, no retry backoff; every row keeps its last known progress
    assert len(calls) == 1 and calls[0][1] is not None and calls[0][1] <= 5
    assert during["store_unavailable"] is True and len(during["items"]) == 3
    for before, item in zip(listed["items"], during["items"]):
        assert item["progress_error"] == "store_unavailable" and item["progress_stale"] is True
        assert item["progress_line"] == before["progress_line"]

    after = api.get("/process/api/conversations").json()
    assert after["store_unavailable"] is False and all(i["progress_error"] is None for i in after["items"])
