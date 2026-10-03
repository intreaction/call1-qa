"""The one-computer launcher: its first run (real Store host commands in subprocesses, no ports), its
shared log's redaction, and ``--demo`` (unit pieces, then the real launcher with fake handlers on
spare ports and a temporary data root under /private/tmp/call1-e2e/)."""

from __future__ import annotations

import io
import json
import os
import queue
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from call1 import launch
from call1.launch import PROCESS_SCOPES, Demo, DemoPlan, Launcher, choose_demo_handlers, demo_env, demo_recordings, redact, reset_demo_root

REPO = Path(__file__).resolve().parents[2]
PYTHON = str(REPO / ".venv-local" / "bin" / "python") if (REPO / ".venv-local" / "bin" / "python").exists() else sys.executable
RESERVED_PORTS = {8000, 8010, 8020}  # the user's live apps


def test_first_run_migrates_issues_a_key_once_and_prints_the_setup_command(tmp_path):
    env = dict(os.environ, CALL1_STORE_DATA=str(tmp_path / "store"), CALL1_PROCESS_CONFIG=str(tmp_path / "process" / "config.json"))
    out = io.StringIO()
    launcher = Launcher(python=sys.executable, env=env, log_path=tmp_path / "launch.log", out=out)
    assert launcher.needs_service_key()
    assert launcher.first_run("test-host") == 0
    config = json.loads((tmp_path / "process" / "config.json").read_text())
    assert config["service_key"].startswith("c1sk_") and config["store_url"] == "http://localhost:8010" and config["installation_id"]
    assert oct((tmp_path / "process" / "config.json").stat().st_mode & 0o777) == "0o600"
    text = out.getvalue()
    assert "applied migrations" in text and "python -m call1.store setup-code" in text and "jobs:control" in text
    log = (tmp_path / "launch.log").read_text()
    assert config["service_key"] not in log and oct((tmp_path / "launch.log").stat().st_mode & 0o777) == "0o600"

    again = Launcher(python=sys.executable, env=env, log_path=tmp_path / "launch.log", out=io.StringIO())
    assert not again.needs_service_key()
    assert again.first_run("test-host") == 0
    assert json.loads((tmp_path / "process" / "config.json").read_text())["service_key"] == config["service_key"]
    assert "jobs:control" in PROCESS_SCOPES and "calls:write" in PROCESS_SCOPES


def test_the_shared_log_never_keeps_credentials():
    line = "[process] Open: http://127.0.0.1:8020/#console_token=c1con_abc-DEF_123 key c1sk_AbCdEf_S3cr3t-value"
    cleaned = redact(line)
    assert "c1con_abc" not in cleaned and "S3cr3t" not in cleaned and cleaned.count("[redacted]") == 2


# --- demo mode ---------------------------------------------------------------------------------


def test_demo_recordings_name_the_agents_from_the_manifest_or_the_legacy_seed():
    items = {i["name"]: i["fields"] for i in demo_recordings()}
    assert sorted(items) == ["call_01_compliant", "call_02_critical_breach", "call_03_dispute_escalation", "call_04_dropped_too_short",
                             "call_05_pii_heavy"]
    for name in ("call_01_compliant", "call_03_dispute_escalation", "call_05_pii_heavy"):
        assert (items[name]["agent_display_name"], items[name]["agent_extension"], items[name]["agent_id"]) == ("Samantha", "104", "agent-104")
    for name in ("call_02_critical_breach", "call_04_dropped_too_short"):
        assert (items[name]["agent_display_name"], items[name]["agent_extension"]) == ("Bob", "202")
    assert items["call_01_compliant"]["external_call_ref"] == "call_01_compliant" and items["call_01_compliant"]["agent_channel"] == 0
    assert "agent_channel" not in items["call_04_dropped_too_short"]  # mono


def test_demo_recordings_prefer_names_the_manifest_gives(tmp_path):
    for name in ("call_01_a.wav", "call_02_b.wav"):
        (tmp_path / name).write_bytes(b"RIFF")
    (tmp_path / "manifest.json").write_text(json.dumps({"call_01_a": {"call_id": "ext-1", "agent_display_name": "Ana", "agent_extension": "301",
                                                                      "agent_channel": 1}}))
    items = {i["name"]: i["fields"] for i in demo_recordings(tmp_path)}
    assert items["call_01_a"] == {"agent_id": "agent-104", "agent_display_name": "Ana", "agent_extension": "301", "external_call_ref": "ext-1",
                                  "agent_channel": 1}
    assert items["call_02_b"]["agent_display_name"] == "Bob" and items["call_02_b"]["external_call_ref"] == "call_02_b"


def test_demo_handlers_are_real_only_on_apple_silicon_with_weights(tmp_path, monkeypatch):
    models = tmp_path / "models"
    monkeypatch.setattr(launch, "apple_silicon", lambda: True)
    assert choose_demo_handlers({"CALL1_MODELS_DIR": str(models)}, None)[0] == "fake"  # no weights
    (models / "parakeet-tdt-0.6b-v3").mkdir(parents=True)
    handlers, note = choose_demo_handlers({"CALL1_MODELS_DIR": str(models)}, None)
    assert handlers == "real" and "mlx" in note.lower()
    monkeypatch.setattr(launch, "apple_silicon", lambda: False)
    handlers, note = choose_demo_handlers({"CALL1_MODELS_DIR": str(models)}, None)
    assert handlers == "fake" and "not an Apple-Silicon Mac" in note
    assert choose_demo_handlers({}, "real")[0] == "real" and choose_demo_handlers({"CALL1_MODELS_DIR": str(models)}, "fake")[0] == "fake"


def test_demo_env_isolates_the_data_root_and_stays_on_localhost(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "apple_silicon", lambda: True)
    plan = DemoPlan(root=tmp_path / "demo", store_port=18001, process_port=18002, handlers="real", handlers_note="")
    env = demo_env({"CALL1_STORE_DATA": "data/store", "CALL1_STORE_PUBLIC_URL": "https://qa.example.com", "CALL1_STORE_HOSTNAME": "qa.example.com",
                    "CALL1_PROCESS_CONFIG": "data/process/config.json", "PATH": "/bin"}, plan)
    root = str((tmp_path / "demo").resolve())
    assert env["CALL1_STORE_DATA"] == root + "/store" and env["CALL1_PROCESS_CONFIG"] == root + "/process/config.json"
    assert env["CALL1_PROCESS_DATA"] == root + "/process" and env["CALL1_STORE_DEMO"] == "1" and env["CALL1_STORE_HOSTNAME"] == "localhost"
    assert env["CALL1_PROCESS_STORE_URL"] == "http://localhost:18001" and env["CALL1_PROCESS_PORT"] == "18002"
    assert env["CALL1_PROCESS_HANDLERS"] == "real" and env["CALL1_BACKEND"] == "mlx" and "CALL1_STORE_PUBLIC_URL" not in env
    assert env["PATH"] == "/bin"


def test_demo_runs_contact_signals_v2_and_gives_the_fifth_call_the_cancel_script_on_fakes(tmp_path, monkeypatch):
    """docs/ContactSignalsV2.md section 15 / F5: the demo starts on pipeline v2 (overridable for a
    rehearsal), and on fake handlers call_05 plays the ``cancel`` fake script, keyed on the checksum
    Process records for the upload (the SHA-256 of the file's bytes)."""
    import hashlib

    assert launch.demo_signals_pipeline({}) == "v2"
    assert launch.demo_signals_pipeline({"CALL1_SIGNALS_PIPELINE": "shadow"}) == "shadow"
    with pytest.raises(SystemExit):
        launch.demo_signals_pipeline({"CALL1_SIGNALS_PIPELINE": "v3"})

    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / f"{launch.DEMO_CANCEL_CALL}.wav").write_bytes(b"RIFF-demo")
    checksum = "sha256:" + hashlib.sha256(b"RIFF-demo").hexdigest()
    fake = DemoPlan(root=tmp_path / "demo", store_port=18001, process_port=18002, handlers="fake", handlers_note="", sample_dir=samples)
    env = demo_env({"CALL1_FAKE_SCRIPTS": json.dumps({"sha256:other": "competitor"})}, fake)
    assert json.loads(env["CALL1_FAKE_SCRIPTS"]) == {"sha256:other": "competitor", checksum: "cancel"}  # the operator's mapping is kept
    assert launch.demo_fake_scripts(None, tmp_path / "absent") == {}  # no sample, no mapping
    monkeypatch.setattr(launch, "apple_silicon", lambda: True)
    real = DemoPlan(root=tmp_path / "demo", store_port=18001, process_port=18002, handlers="real", handlers_note="", sample_dir=samples)
    assert "CALL1_FAKE_SCRIPTS" not in demo_env({}, real)  # real handlers transcribe the real audio
    assert "v2" in launch.signals_line("v2", "real") and "Gemma" in launch.signals_line("v2", "real")
    assert launch.DEMO_CANCEL_CALL in launch.signals_line("v2", "fake")
    # the real sample exists, so the shipped demo really maps it
    assert any(v == "cancel" for v in launch.demo_fake_scripts(None).values())


def test_demo_applies_the_retail_seed_once_per_root_then_only_sets_the_pipeline(tmp_path):
    """Decision 22 / section 15: ``--demo`` publishes the retail seed taxonomy with the pipeline, on the
    real Store host commands, before any call is ingested. A restart keeps a taxonomy edited on stage
    (only the pipeline is re-asserted), and a seed that cannot apply falls back to the pipeline alone."""
    import sqlite3

    root = tmp_path / "demo"
    root.mkdir()
    env = dict(os.environ, CALL1_STORE_DATA=str(root / "store"), CALL1_PROCESS_CONFIG=str(root / "process" / "config.json"))
    launcher = Launcher(python=PYTHON, env=env, log_path=tmp_path / "launch.log", out=io.StringIO())
    assert launcher.run_step("store", ["-m", "call1.store", "migrate"]) == 0
    plan = DemoPlan(root=root, store_port=18001, process_port=18002, handlers="fake", handlers_note="")

    def state():
        with sqlite3.connect(f"file:{root / 'store' / 'store.db'}?mode=ro", uri=True) as conn:
            version, settings = conn.execute("SELECT current_version, settings_json FROM results_signal_taxonomy WHERE id = 1").fetchone()
            taxonomy = json.loads(conn.execute("SELECT taxonomy_json FROM results_signal_taxonomy_versions WHERE version = ?",
                                               (version,)).fetchone()[0])
        return version, json.loads(settings)["pipeline"], {c["category_id"] for c in taxonomy["categories"]}

    assert launch.seed_demo_signals(launcher, plan, "v2")
    version, pipeline, categories = state()
    assert version == 2 and pipeline == "v2" and {"intent", "upsell_attempt", "agent_conduct_concern"} <= categories
    assert (root / launch.SIGNALS_SEEDED_FILE).exists()
    # a restart (the marker is there) re-asserts only the pipeline: no new taxonomy version
    assert launch.seed_demo_signals(launcher, plan, "shadow")
    assert state()[:2] == (2, "shadow")
    # no marker and a seed that no longer applies (the taxonomy moved on): reported, pipeline still set
    (root / launch.SIGNALS_SEEDED_FILE).unlink()
    stale = tmp_path / "stale.json"
    body = json.loads(launch.DEMO_SIGNALS_SEED.read_text())
    body["taxonomy"]["categories"][0]["gloss"] = "Caller says why they called today"
    stale.write_text(json.dumps(body))
    out = io.StringIO()
    launcher.out = out
    assert launch.seed_demo_signals(launcher, plan, "v2", seed=stale)
    assert state()[:2] == (2, "v2") and "was not applied" in out.getvalue()
    assert not (root / launch.SIGNALS_SEEDED_FILE).exists()


def test_reset_wipes_only_a_demo_root(tmp_path):
    real = tmp_path / "store"
    real.mkdir()
    (real / "store.db").write_text("x")
    with pytest.raises(SystemExit):
        reset_demo_root(real)
    assert (real / "store.db").exists()
    marked = tmp_path / "classroom"
    marked.mkdir()
    (marked / launch.DEMO_MARKER).touch()
    reset_demo_root(marked)
    assert not marked.exists()
    reset_demo_root(tmp_path / "absent")  # nothing to do


def test_the_demo_opens_evaluate_the_store_console_and_the_process_console_with_its_token(tmp_path):
    opened = []
    out = io.StringIO()
    plan = DemoPlan(root=tmp_path, store_port=18001, process_port=18002, handlers="fake", handlers_note="")
    demo = Demo(Launcher(log_path=tmp_path / "l.log", out=out), plan, "http://127.0.0.1:18002/#console_token=c1con_abc", opener=opened.append)
    demo.open_browser()
    assert opened == ["http://localhost:18001/", "http://localhost:18001/console/", "http://127.0.0.1:18002/#console_token=c1con_abc"]
    assert demo.token == "c1con_abc"
    plan.open_browser = False
    demo.open_browser()
    assert len(opened) == 3 and "--no-open" in out.getvalue()


# --- the launcher itself, end to end on spare ports --------------------------------------------


def _free_port() -> int:
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in RESERVED_PORTS:
            return port


class _Run:
    """``python -m call1.launch --demo ...`` in a subprocess, its output read line by line."""

    def __init__(self, root: Path, cwd: Path, ports, *extra: str) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("CALL1_")}  # never the user's live data
        env.update(PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1", CALL1_STORE_PORT=str(ports[0]), CALL1_PROCESS_PORT=str(ports[1]))
        self.proc = subprocess.Popen([PYTHON, "-m", "call1.launch", "--demo", "--demo-root", str(root), "--handlers", "fake", "--no-open", *extra],
                                     cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.seen = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in iter(self.proc.stdout.readline, ""):
            self.lines.put(line.rstrip("\n"))

    def until(self, text: str, timeout: float = 120.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                if self.proc.poll() is not None and self.lines.empty():
                    break
                continue
            self.seen.append(line)
            if text in line:
                return line
        raise AssertionError(f"{text!r} not printed; output:\n" + "\n".join(self.seen[-60:]))

    def stop(self, *signals: int) -> int:
        """SIGTERM (or the given signals in quick succession, as a process group or ``timeout``
        sends them), then the exit status; the rest of the output lands in ``seen``."""
        if self.proc.poll() is None:
            for sig in signals or (signal.SIGTERM,):
                self.proc.send_signal(sig)
                time.sleep(0.05)
        try:
            code = self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            code = self.proc.wait()
        time.sleep(0.2)
        while not self.lines.empty():
            self.seen.append(self.lines.get())
        return code


@pytest.fixture
def demo_dir():
    base = Path(os.environ.get("CALL1_E2E_ROOT", str(Path(tempfile.gettempdir()) / "call1-e2e")))
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"launch-demo-{int(time.time())}-{os.getpid()}-{secrets.token_hex(3)}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_launch_demo_seeds_the_samples_and_follows_them_until_settled(demo_dir):
    root, cwd = demo_dir / "demo", demo_dir / "cwd"
    cwd.mkdir()
    ports = (_free_port(), _free_port())
    run = _Run(root, cwd, ports)
    try:
        run.until("CALL1 DEMO MODE")
        run.until("fake handlers (--handlers fake)")
        run.until("Demo history: added 560 synthetic sessions")
        for agent in ("call_01_compliant: Samantha (104)", "call_02_critical_breach: Bob (202)", "call_05_pii_heavy: Samantha (104)"):
            assert "(new)" in run.until(agent)
        console = run.until("Process console   http://127.0.0.1:")
        url = console.split("Process console", 1)[1].strip()
        assert url.startswith(f"http://127.0.0.1:{ports[1]}/#console_token=c1con_")
        run.until("All 5 demo calls have settled", timeout=180)
        assert not any('"GET /' in line for line in run.seen)  # access lines stay in the log file

        token = url.split("console_token=", 1)[1]
        session = httpx.get(f"http://127.0.0.1:{ports[1]}/process/api/session", headers={"X-Call1-Console-Token": token}).json()
        assert session["token_valid"] is True
        demo = httpx.get(f"http://localhost:{ports[0]}/demo/status")
        assert demo.status_code == 200 and demo.json()["demo"] is True  # Store runs with CALL1_STORE_DEMO=1
        assert httpx.get(f"http://localhost:{ports[0]}/").status_code == 200
        items = httpx.get(f"http://127.0.0.1:{ports[1]}/process/api/conversations").json()["items"]
        assert len(items) >= 5 and all(i["progress"]["settled"] for i in items)
        overview = httpx.get(f"http://127.0.0.1:{ports[1]}/process/api/overview").json()
        assert overview["handlers"]["mode"] == "fake"
    finally:
        # repeated signals during the graceful stop must not abort it
        assert run.stop(signal.SIGTERM, signal.SIGTERM, signal.SIGINT) == 0
    assert not any("Traceback" in line for line in run.seen), "\n".join(run.seen[-30:])
    # everything under the demo root, nothing relative to the working directory
    assert (root / "store" / "store.db").exists() and (root / "process" / "config.json").exists() and not (cwd / "data").exists()
    seeded = json.loads((root / launch.SEEDED_FILE).read_text())
    assert len(seeded["calls"]) == 5
    assert "Contact Signals: pipeline v2" in "\n".join(run.seen)
    # the demo's retail policy (BUG D1), alert rule and queue rule, applied before the first ingest
    setup = json.loads((root / launch.DEMO_SETUP_FILE).read_text())
    assert setup["rubric_version"] == 2 and setup["alert_rule"] == 1 and setup["queue_rule"] == 1
    output = "\n".join(run.seen)
    assert "published version 2 with the retail verification and disclosure policy" in output
    assert output.index("Alert rule stock-check") < output.index("call_01_compliant: Samantha (104)")
    _assert_demo_signals_v2(root / "store" / "store.db", {c["name"]: c["call_id"] for c in seeded["calls"]})
    log = (root / "logs" / "launch.log").read_text()
    config = json.loads((root / "process" / "config.json").read_text())
    assert config["service_key"] not in log and token not in log and "console_token=[redacted]" in log
    assert oct((root / "logs" / "launch.log").stat().st_mode & 0o777) == "0o600"
    for port in ports:  # both apps stopped
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))

    # a second start keeps the seeded demo; --reset starts over
    run = _Run(root, cwd, ports)
    try:
        run.until("Demo history: added 0 synthetic sessions")
        run.until("Already seeded (5 calls")
        assert not any("Rubric call1_standard_v2" in line for line in run.seen)  # applied once per demo root
        run.until("All 5 demo calls have settled", timeout=60)
    finally:
        assert run.stop() == 0
    run = _Run(root, cwd, ports, "--reset")
    try:
        assert "(new)" in run.until("call_01_compliant: Samantha (104)")
        run.until("All 5 demo calls have settled", timeout=180)
    finally:
        assert run.stop() == 0
    assert (root / launch.DEMO_MARKER).exists()


def _assert_demo_signals_v2(db_path: Path, calls: dict) -> None:
    """Section 15 / F5: the demo Store runs pipeline v2 and every seeded call published v2 hits
    (span-keyed IDs) on the retail seed taxonomy; call_05 played the cancel script (its caller opens on the cancellation, with no
    fee), the others the default script (whose caller asks about a fee)."""
    import re
    import sqlite3

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        version, settings_json = conn.execute("SELECT current_version, settings_json FROM results_signal_taxonomy WHERE id = 1").fetchone()
        assert json.loads(settings_json)["pipeline"] == "v2"
        taxonomy = json.loads(conn.execute("SELECT taxonomy_json FROM results_signal_taxonomy_versions WHERE version = ?",
                                           (version,)).fetchone()[0])
        assert version == 2 and "upsell_attempt" in {c["category_id"] for c in taxonomy["categories"]}  # the retail seed (decision 22)
        hits = {name: conn.execute("SELECT hit_id, category_id FROM results_signal_hits WHERE call_id = ?", (call_id,)).fetchall()
                for name, call_id in calls.items()}
    for name, rows in hits.items():
        assert rows and all(re.search(r"\.t\d+b\d+$", hit_id) for hit_id, _ in rows), (name, rows)
    cancel = {c for _, c in hits[launch.DEMO_CANCEL_CALL]}
    assert "intent" in cancel and "issue" not in cancel, hits[launch.DEMO_CANCEL_CALL]
    assert "issue" in {c for _, c in hits["call_01_compliant"]}, hits["call_01_compliant"]


def test_launch_demo_refuses_a_port_in_use(demo_dir):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        env = {k: v for k, v in os.environ.items() if not k.startswith("CALL1_")}
        env.update(PYTHONPATH=str(REPO), CALL1_STORE_PORT=str(port), CALL1_PROCESS_PORT=str(_free_port()))
        proc = subprocess.run([PYTHON, "-m", "call1.launch", "--demo", "--demo-root", str(demo_dir / "demo"), "--handlers", "fake", "--no-open"],
                              cwd=demo_dir, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and f"Port {port} (Store) is already in use" in proc.stderr
    assert not (demo_dir / "demo" / "store").exists()


def test_demo_flags_need_demo():
    with pytest.raises(SystemExit):
        launch.main(["--reset"])
