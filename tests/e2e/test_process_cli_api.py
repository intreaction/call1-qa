"""Process features p9-p12 (inventory area "process"), end to end: the one-computer launcher, the
Process CLIs, the operator API (including a real Store outage) and the loopback and console rules.

Each test is named after its inventory feature ID.
"""

from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import time

import httpx
import pytest

from .process_helpers import clean_env, free_port, jobs_of, one, port_open, python_executable, run_python
from .stack import E2E_ROOT, REPO, sample_path, unique_wav_copy

pytestmark = pytest.mark.e2e


# --- p9: the launcher ----------------------------------------------------------------------------


def test_p9_launcher_starts_both_apps_on_a_clean_data_dir(tmp_path_factory):
    E2E_ROOT.mkdir(parents=True, exist_ok=True)
    root = E2E_ROOT / f"p9-launch-{time.strftime('%H%M%S')}-{os.getpid()}"
    root.mkdir()
    store_port, process_port = free_port(), free_port()
    while process_port == store_port:
        process_port = free_port()
    config = root / "data" / "process" / "config.json"
    env = clean_env(CALL1_STORE_DATA=str(root / "data" / "store"), CALL1_STORE_PORT=str(store_port), CALL1_STORE_HOSTNAME="localhost",
                    CALL1_STORE_BIND="127.0.0.1", CALL1_PROCESS_CONFIG=str(config), CALL1_PROCESS_PORT=str(process_port),
                    CALL1_PROCESS_DATA=str(root / "data" / "process"))
    log_path = root / "data" / "logs" / "launch.log"
    terminal = open(root / "terminal.log", "w")
    launcher = subprocess.Popen([python_executable(), "-m", "call1.launch", "--handlers", "fake", "--installation", "e2e-launch", "--log", str(log_path)],
                                env=env, cwd=str(root), stdout=terminal, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        state = None
        while time.monotonic() < deadline:
            try:
                store_ok = httpx.get(f"http://127.0.0.1:{store_port}/store/v1/status", timeout=1).status_code == 200
                state = httpx.get(f"http://127.0.0.1:{process_port}/process/api/health", timeout=1).json().get("state")
                if store_ok and state == "running":
                    break
            except (httpx.HTTPError, ValueError):
                pass
            assert launcher.poll() is None, (root / "terminal.log").read_text()
            time.sleep(0.2)
        assert state == "running", f"Process state {state!r}:\n{(root / 'terminal.log').read_text()[-3000:]}"

        written = json.loads(config.read_text())
        assert written["store_url"] == f"http://localhost:{store_port}" and written["service_key"].startswith("c1sk_"), sorted(written)
        assert stat.S_IMODE(config.stat().st_mode) == 0o600

        out = (root / "terminal.log").read_text()
        command = re.search(r"python -m call1\.store setup-code --email \S+ --display-name .+", out)
        assert command, f"the launcher prints the setup-code command:\n{out[-2000:]}"
        token = re.search(r"console_token=(c1con_[A-Za-z0-9_-]+)", out)
        assert token, "the terminal shows the one-time console URL"

        # The printed command works against this data dir and mints a first-admin code.
        code = run_python(["-m", "call1.store", "setup-code", "--email", "first@e2e.test", "--display-name", "First Admin"], env, root)
        assert code.returncode == 0 and code.stdout.strip(), code.stderr

        # The issued key holds jobs:control: a console retry reaches Store's state machine, not a 403.
        headers = {"X-Call1-Console-Token": token.group(1)}
        path = unique_wav_copy(sample_path("call_01_compliant"), root / "uploads", 1)
        with open(path, "rb") as handle:
            receipt = httpx.post(f"http://127.0.0.1:{process_port}/process/api/recordings", headers=headers,
                                 files={"file": (path.name, handle, "audio/wav")}, timeout=30).json()
        assert receipt.get("conversation_created") is True, receipt
        jobs = httpx.get(f"http://127.0.0.1:{store_port}/store/v1/jobs", params={"conversation_id": receipt["conversation_id"], "limit": 200},
                         headers={"Authorization": f"Bearer {written['service_key']}"}).json()["items"]
        retry = httpx.post(f"http://127.0.0.1:{process_port}/process/api/jobs/{jobs[0]['id']}/retry", json={"reason": "scope probe"}, headers=headers)
        assert retry.status_code != 403 and retry.json().get("code") != "insufficient_scope", retry.text

        # The shared log is 0600 and redacts credentials.
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
        shared = log_path.read_text()
        assert "[store]" in shared and "[process]" in shared
        assert token.group(1) not in shared and written["service_key"] not in shared, "launch.log leaks a credential"
    finally:
        launcher.send_signal(signal.SIGTERM)
        try:
            code = launcher.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(launcher.pid, signal.SIGKILL)
            raise
        finally:
            terminal.close()
    time.sleep(0.5)
    assert not port_open(store_port) and not port_open(process_port), "SIGTERM to the launcher stops both apps"
    if os.environ.get("CALL1_E2E_KEEP") != "1":
        import shutil

        shutil.rmtree(root, ignore_errors=True)


# --- p10: CLIs -----------------------------------------------------------------------------------


def test_p10_status_reports_store_reachability_and_contract_as_json(stack_factory):
    private = stack_factory(name="p10-status", with_process=False)
    ok = private.process_cli("status", check=False)
    assert ok.returncode == 0, ok.stderr
    body = json.loads(ok.stdout)
    contract = private.store_get("/contract").json()
    assert body["state"] == "store_reachable" and body["store"]["compatible"] is True, body["store"]
    assert body["store"]["contract_version"] == contract["contract_version"]
    private.stop_store()
    down = private.process_cli("status", check=False)
    assert down.returncode == 1, down.stdout
    assert json.loads(down.stdout)["state"] == "store_unreachable"


def test_p10_drain_runs_every_ready_job_then_exits_without_http(stack_factory):
    private = stack_factory(name="p10-drain", with_process=False)
    path = unique_wav_copy(sample_path("call_01_compliant"), private.uploads_dir, 7)
    receipt = json.loads(private.process_cli("ingest", str(path), "--agent-id", "agent-p10").stdout)
    port = json.loads(private.process_config_path.read_text())["port"]
    started = time.monotonic()
    drained = private.process_cli("drain", check=False)
    assert drained.returncode == 0, drained.stderr[-2000:]
    assert time.monotonic() - started < 60
    result = json.loads(drained.stdout)
    progress = private.progress(receipt["conversation_id"])
    assert progress["settled"] is True, progress
    jobs = jobs_of(private, receipt["conversation_id"])
    assert all(j["status"] == "SUCCEEDED" for j in jobs), [(j["job_type"], j["status"]) for j in jobs]
    assert result["jobs_run"] == len(jobs), f"drain ran {result['jobs_run']} jobs for {len(jobs)}"
    assert "Uvicorn running" not in drained.stderr and not port_open(port), "drain serves no HTTP"


def test_p10_console_token_rotate_invalidates_the_old_token(stack_factory):
    private = stack_factory(name="p10-token")
    old = private.console_token
    assert private.process_get("/session", token=old).json()["token_valid"] is True
    refused = private.process_cli("console-token", check=False)
    assert refused.returncode == 2 and "--rotate" in refused.stderr, (refused.returncode, refused.stderr)
    rotated = private.process_cli("console-token", "--rotate")
    new = re.search(r"console_token=(c1con_[A-Za-z0-9_-]+)", rotated.stdout).group(1)
    assert new != old
    assert private.process_get("/session", token=new).json()["token_valid"] is True, "the running console accepts the new token"
    assert private.process_get("/session", token=old).json()["token_valid"] is False, "the old token still works after --rotate"
    write = private.process_post("/jobs/job_does_not_exist/retry", {"reason": "old token"}, token=old)
    assert write.status_code == 401, write.text


# --- p1/q16: agent identity at ingest (contract 1.1.0) -------------------------------------------


def _audited_updates(admin_session, call_id):
    return admin_session.get("/admin/audit", params={"action": "call_metadata_updated", "target_id": call_id}).json()["items"]


def test_p1_agent_identity_through_the_api_and_a_changed_reupload_updates_the_call(stack, admin_session):
    """Team decisions 2026-09-25 (1) and (2): the upload form carries the agent's display name and
    extension; the same recording again with different details updates the call's metadata
    (audited), without reprocessing; details left out keep their stored values."""
    path = unique_wav_copy(sample_path("call_01_compliant"), stack.uploads_dir, 911)
    agent = f"agent-p1-api-{int(time.time() * 1000)}"

    def upload(data):
        with open(path, "rb") as handle:
            return stack.process_request("POST", "/recordings", files={"file": (path.name, handle, "audio/wav")}, data=data)

    first = upload({"agent_id": agent, "agent_display_name": "Samantha", "agent_extension": "104"})
    assert first.status_code == 201, first.text
    receipt = first.json()
    assert receipt["conversation_created"] and receipt["metadata_updated"] is False and receipt["agent_label"] == "Samantha (104)"
    call = admin_session.get(f"/calls/{receipt['call_id']}").json()["call"]
    assert (call["agent_id"], call["agent_display_name"], call["agent_extension"]) == (agent, "Samantha", "104")

    again = upload({"agent_display_name": "Bob", "agent_extension": "202"}).json()  # no agent_id this time
    assert (again["conversation_id"], again["graph_id"]) == (receipt["conversation_id"], receipt["graph_id"])
    assert again["metadata_updated"] is True and again["conversation_created"] is False and again["graph_created"] is False
    assert again["updated_fields"] == ["agent_display_name", "agent_extension"] and again["agent_label"] == "Bob (202)"
    call = admin_session.get(f"/calls/{receipt['call_id']}").json()["call"]
    assert (call["agent_id"], call["agent_display_name"], call["agent_extension"]) == (agent, "Bob", "202")  # agent ID kept
    assert {j["graph_id"] for j in jobs_of(stack, receipt["conversation_id"])} == {receipt["graph_id"]}  # nothing reprocessed
    audit = _audited_updates(admin_session, receipt["call_id"])
    assert len(audit) == 1 and "Bob" not in json.dumps(audit), audit  # changed field names only, never values

    replay = upload({}).json()  # naming nothing new: a pure replay
    assert replay["metadata_updated"] is False and replay["agent_label"] == "Bob (202)"
    assert len(_audited_updates(admin_session, receipt["call_id"])) == 1

    refused = upload({"agent_display_name": "x" * 101})
    assert refused.status_code == 422 and refused.json()["code"] == "validation_failed", refused.text


def test_p10_cli_ingest_takes_agent_name_and_extension_and_a_rerun_updates_the_call(stack, admin_session):
    # A CLI import (source kind local_import) is its own source identity, separate from API uploads.
    path = unique_wav_copy(sample_path("call_01_compliant"), stack.uploads_dir, 912)
    agent = f"agent-p10-cli-{int(time.time() * 1000)}"
    first = json.loads(stack.process_cli("ingest", str(path), "--agent-id", agent, "--agent-name", "Ana", "--agent-extension", "7#1").stdout)
    assert first["conversation_created"] and first["agent_label"] == "Ana (7#1)"
    call = admin_session.get(f"/calls/{first['call_id']}").json()["call"]
    assert (call["agent_id"], call["agent_display_name"], call["agent_extension"]) == (agent, "Ana", "7#1")

    proc = stack.process_cli("ingest", str(path), "--agent-name", "Ana Ruiz")
    again = json.loads(proc.stdout)
    assert again["conversation_id"] == first["conversation_id"] and again["graph_id"] == first["graph_id"]
    assert again["metadata_updated"] is True and again["updated_fields"] == ["agent_display_name"]
    assert again["agent_label"] == "Ana Ruiz (7#1)"
    assert "Metadata updated" in proc.stderr and "nothing was reprocessed" in proc.stderr
    assert len(_audited_updates(admin_session, first["call_id"])) == 1

    replay = json.loads(stack.process_cli("ingest", str(path)).stdout)
    assert replay["metadata_updated"] is False

    bad = stack.process_cli("ingest", str(path), "--agent-extension", "ext 104", check=False)
    assert bad.returncode == 2 and "agent_extension" in bad.stderr, (bad.returncode, bad.stderr)


# --- p11: the operator API ---------------------------------------------------------------------


def test_p11_operator_api_reads(stack):
    receipt = stack.ingest("call_01_compliant", agent_id="agent-p11")
    stack.wait_until_settled(receipt["call_id"])

    assert stack.process_get("/health", token=False).json() == {"ok": True, "state": "running"}
    overview = stack.process_get("/overview", token=False).json()
    assert overview["state"] == "running" and overview["store"]["url"] == stack.store_url
    assert overview["store"]["compatible"] is True and overview["store"]["contract_version"]
    assert set(overview["store"]["parameters"]) >= {"lease_duration_seconds", "heartbeat_interval_seconds", "inline_artifact_max_bytes"}
    assert overview["handlers"]["mode"] == "fake" and isinstance(overview["handlers"]["missing_job_types"], list)
    worker = overview["worker"]
    assert worker["state"] == "running" and {p["pool"] for p in worker["pools"]} >= {"mlx", "cpu_io"}
    assert isinstance(worker["running"], list) and isinstance(worker["spooled"], int)
    assert overview["reanalysis"] is not None and overview["catalog"]["published"]["catalog_version"] == overview["catalog"]["version"]
    assert isinstance(overview["scratch_bytes"], int) and overview["evaluate_url"] == stack.store_url + "/"
    snapshots = stack.admin().get("/catalog-snapshots")  # a reviewer-session route (service keys are refused)
    assert snapshots.status_code == 200, snapshots.text
    assert overview["catalog"]["version"] in json.dumps(snapshots.json()), "Store serves back the catalog snapshot Process published"
    profiles = stack.store_get("/hardware-profiles", session="service").json()["items"]
    assert overview["hardware_profile_id"] in {p["id"] for p in profiles}

    listing = stack.process_get("/conversations", params={"limit": 200}, token=False).json()
    item = next(i for i in listing["items"] if i["conversation_id"] == receipt["conversation_id"])
    assert item["progress"]["settled"] is True and item["progress_error"] is None and item["progress_stale"] is False
    assert item["progress_line"].startswith("Transcript ready") and "QA ready" in item["progress_line"], item["progress_line"]
    assert item["evaluate_url"] == f"{stack.store_url}/#/calls/{receipt['call_id']}" and listing["store_unavailable"] is False

    detail = stack.process_get(f"/conversations/{receipt['conversation_id']}", token=False).json()
    assert detail["conversation"]["call_id"] == receipt["call_id"] and detail["evaluate_url"] == item["evaluate_url"]
    store_jobs = {j["id"] for j in jobs_of(stack, receipt["conversation_id"])}
    assert {j["id"] for j in detail["jobs"]} == store_jobs
    asr = next(j for j in detail["jobs"] if j["job_type"] == "asr")
    assert asr["status"] == "SUCCEEDED" and asr["attempt_count"] == 1 and asr["catalog_entry"] and asr["route_class"], asr

    job = stack.process_get(f"/jobs/{asr['id']}", token=False).json()
    assert job["job"]["id"] == asr["id"] and len(job["attempts"]) == 1 and job["attempts"][0]["provenance"]["adapter_id"] == "fake.asr"
    assert not {"outputs", "content", "transcript"} & set(job["attempts"][0]), "attempts carry provenance, never content"
    assert job["evaluate_url"] == item["evaluate_url"]

    catalog = stack.process_get("/catalog", token=False).json()
    assert catalog["version"] == overview["catalog"]["version"] and catalog["entries"] and catalog["defaults"]
    assert catalog["handlers"]["mode"] == "fake" and "masking" in catalog
    missing = stack.process_get("/jobs/job_does_not_exist", token=False)
    assert missing.status_code == 404 and missing.json()["code"], missing.text


def test_p11_conversations_degrade_during_a_store_outage(stack_factory):
    private = stack_factory(name="p11-outage")
    receipts = [private.ingest("call_01_compliant", agent_id=f"agent-p11o-{n}") for n in range(3)]
    for receipt in receipts:
        private.wait_until_settled(receipt["call_id"])
    before = {i["conversation_id"]: i for i in private.process_get("/conversations").json()["items"]}
    private.stop_store()
    started = time.monotonic()
    during = private.process_get("/conversations")
    elapsed = time.monotonic() - started
    assert during.status_code == 200, during.text
    body = during.json()
    assert elapsed < 5, f"the listing took {elapsed:.1f}s during the outage"
    assert body["store_unavailable"] is True
    for item in body["items"]:
        assert item["progress_stale"] is True and item["progress_error"] == "store_unavailable", item
        assert item["progress_line"] == before[item["conversation_id"]]["progress_line"], item
    overview = private.process_get("/overview")
    assert overview.status_code == 200
    private.start_store()
    after = private.wait_for(lambda: (lambda b: b if not b["store_unavailable"] else None)(private.process_get("/conversations").json()),
                             timeout=20, what="fresh listing")
    assert all(i["progress_stale"] is False for i in after["items"])


# --- p12: loopback and console rules --------------------------------------------------------------


def test_p12_a_non_loopback_bind_is_refused_at_startup(stack):
    env = stack.process_env()
    env.update(CALL1_PROCESS_BIND="0.0.0.0", CALL1_PROCESS_PORT=str(free_port()))
    proc = subprocess.run([python_executable(), "-m", "call1.process", "serve"], env=env, cwd=str(stack.dir), capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0, "serve started on 0.0.0.0"
    assert "loopback" in (proc.stdout + proc.stderr).lower(), proc.stderr[-1000:]


def test_p12_host_origin_and_console_token_rules(stack):
    url = stack.process_url
    rebinding = httpx.get(f"{url}/process/api/overview", headers={"Host": "call1.attacker.example"})
    assert rebinding.status_code == 403 and rebinding.json()["code"] == "loopback_only", rebinding.text
    assert httpx.get(f"{url}/process/api/overview", headers={"Host": f"localhost:{stack.process_port}"}).status_code == 200

    target = f"{url}/process/api/jobs/job_does_not_exist/retry"
    token = {"X-Call1-Console-Token": stack.console_token}
    none = httpx.post(target, json={"reason": "x"})
    assert none.status_code == 401 and none.json()["code"] == "console_credential_invalid", none.text
    wrong = httpx.post(target, json={"reason": "x"}, headers={"X-Call1-Console-Token": "c1con_wrong"})
    assert wrong.status_code == 401
    cross = httpx.post(target, json={"reason": "x"}, headers={**token, "Origin": "https://evil.example"})
    assert cross.status_code == 403 and cross.json()["code"] == "origin_not_allowed", cross.text
    same = httpx.post(target, json={"reason": "x"}, headers={**token, "Origin": url})
    assert same.status_code not in (401, 403), same.text
    bearer = httpx.post(target, json={"reason": "x"}, headers={"Authorization": f"Bearer {stack.console_token}"})
    assert bearer.status_code not in (401, 403), bearer.text
    upload = httpx.post(f"{url}/process/api/recordings", files={"file": ("x.wav", b"RIFF", "audio/wav")})
    assert upload.status_code == 401, "an upload without the console token is refused"
    # Reads need no token.
    assert httpx.get(f"{url}/process/api/conversations").status_code == 200
