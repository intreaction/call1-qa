"""``python -m call1.launch``: the one-computer launcher. Starts Store and Process together.

1. ``python -m call1.store migrate`` (first run creates Store's database).
2. When Process's config (``CALL1_PROCESS_CONFIG``, default ``data/process/config.json``) has no
   service key, ``python -m call1.store issue-service-key`` issues one for this computer's
   installation, with the default Process scopes plus ``jobs:control`` so the console can retry
   and cancel jobs, and the command to create the first admin is printed.
3. ``python -m call1.store serve`` and ``python -m call1.process serve`` run as subprocesses. Their
   output is interleaved with a ``[store]``/``[process]`` prefix on this terminal and appended to
   ``data/logs/launch.log`` (mode 0600, with credentials redacted).
4. Ctrl-C or SIGTERM (or either app exiting) stops both, gracefully (SIGTERM to each app).

``python -m call1.launch --demo`` is the class-demo variant (DEMO MODE, localhost only):

* Everything lives under a separate data root, ``data/demo/`` (``store/``, ``process/``,
  ``logs/``), so ``data/store`` and ``data/process`` are never touched. ``--reset`` wipes the demo
  root first.
* Store runs with ``CALL1_STORE_DEMO=1`` (persona sign-in without a passkey, dev mode on
  ``localhost`` only; the passkey ceremonies are unchanged beside it).
* Handlers: the real ones with ``CALL1_BACKEND=mlx`` on Apple Silicon when ``data/models`` (or
  ``CALL1_MODELS_DIR``) holds weights, else the fake ones (a note says so). ``--handlers`` wins.
* A fresh console credential is issued at every start, so the Process console URL can carry it.
* The demo also imports 560 labeled synthetic history sessions across eight weeks through
  Store's host-only seed-demo-history command (no audio or inference; idempotent).
* Once both apps answer, the five ``sample_audio/call_0*.wav`` recordings are ingested through the
  Process API (agent names, extensions and channels from ``sample_audio/manifest.json``, falling
  back to the legacy seed's Samantha 104 / Bob 202), unless the demo root says it is already
  seeded. The three URLs open in the default browser (``--no-open`` skips that), and progress is
  printed until every call has settled.
* Contact Signals v2 is on from the first call (docs/ContactSignalsV2.md section 15): right after
  the migration, before either app starts and so before any call is ingested,
  ``python -m call1.store apply-signals-seed call1/store/seeds/signals_retail_v1.json --pipeline v2``
  publishes the retail starter taxonomy (decision 22: only ``--demo`` applies it) and sets the Store
  setting, both audited and idempotent (``CALL1_SIGNALS_PIPELINE=v1|shadow`` overrides the pipeline
  for a rehearsal). The seed is applied once per demo root (``signals-seeded.json`` records it), so a
  taxonomy edited on stage survives a restart; later starts only re-assert the pipeline with
  ``python -m call1.store signals-pipeline``. On fake handlers the
  fifth sample call, ``call_05_pii_heavy``, is given the ``cancel`` fake script
  (``CALL1_FAKE_SCRIPTS``, keyed on the recording's checksum), so the on-stage "Cancel account"
  step has a call to light up. Real handlers transcribe the real audio and run all three v2 stages
  on the included model (team decision 24).
* Dual transcription (team decision 33, docs/DualAsr.md section 10): right after the signals seed,
  ``python -m call1.store apply-vocabulary-seed call1/store/seeds/asr_vocabulary_retail_v1.json``
  installs the retail vocabulary as the industry pack (only ``--demo`` applies it). It is audited
  and idempotent, so it runs at every start: the pack already installed is a no-op, and terms an
  admin switched off or added on stage are kept. With the vocabulary on (the default for a
  non-empty one), every seeded call's ``asr`` job runs the vocabulary pass. A pack that cannot be
  installed is reported and the demo goes on without it.
* Demo policy and alert (``call1.demo_setup``): once Store answers and before the sample calls are
  ingested, the launcher signs in as the demo admin and publishes the next ``call1_standard_v2``
  version with a retail verification and disclosure policy for SEC-01 and COMP-01 (without one they
  always need review and no call passes), and creates the ``stock-check`` Contact Signals alert
  rule with a SIGNAL review-queue rule on it. Once per demo root (``demo-setup.json``);
  ``scripts/apply_demo_policy.py`` applies the same to a demo that is already running.

Ports come from ``CALL1_STORE_PORT`` / ``CALL1_PROCESS_PORT`` (8010 / 8020).

The launcher imports neither app: each runs in its own interpreter, exactly as it would alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROCESS_SCOPES = (
    "calls:write", "artifacts:read", "artifacts:write", "jobs:write", "jobs:claim", "reanalysis:claim", "changes:read",
    "hardware:write", "catalog:publish", "usage:read", "admin-state:read", "training:read", "jobs:control",
)
"""The Stage 2 Process worker scopes (Store's default set, with ``training:read`` since 1.3.0) plus ``jobs:control`` for the console."""

_SECRET = re.compile(r"(console_token=|c1con_|c1sk_[A-Za-z0-9]{6}_)[A-Za-z0-9_-]+")

DEFAULT_STORE_PORT = 8010
DEFAULT_PROCESS_PORT = 8020
DEFAULT_DEMO_ROOT = Path("data/demo")
DEMO_MARKER = ".call1-demo"
"""Written into a demo root when it is created; ``--reset`` refuses a directory without it (unless
it is named ``demo``), so a mistyped ``--demo-root`` can never wipe real data."""
SEEDED_FILE = "demo-seeded.json"
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_audio"
DEMO_SIGNALS_PIPELINE = "v2"
"""The Contact Signals pipeline a demo runs (section 15); ``CALL1_SIGNALS_PIPELINE`` overrides it."""
SIGNALS_PIPELINES = ("v1", "shadow", "v2")
DEMO_SIGNALS_SEED = Path(__file__).resolve().parent / "store" / "seeds" / "signals_retail_v1.json"
"""The retail starter taxonomy ``--demo`` applies (decision 22); installs start with the built-ins only."""
SIGNALS_SEEDED_FILE = "signals-seeded.json"
"""Written in the demo root once the seed is published, so a restart keeps on-stage taxonomy edits."""
DEMO_SETUP_FILE = "demo-setup.json"
"""Written in the demo root once ``call1.demo_setup`` (retail rubric policy, alert and queue rule)
has been applied, so a restart keeps on-stage rubric and alert edits."""
DEMO_VOCABULARY_SEED = Path(__file__).resolve().parent / "store" / "seeds" / "asr_vocabulary_retail_v1.json"
"""The retail ASR vocabulary ``--demo`` installs as the industry pack (decision 33); installs start with none."""
DEMO_CANCEL_CALL = "call_05_pii_heavy"
"""On fake handlers this sample gets the ``cancel`` fake script (section 15), in place of ``SCRIPT``."""
CONSOLE_HEADER = "X-Call1-Console-Token"

# The legacy seed (scripts/seed_database.py): calls 01, 03 and 05 are Samantha's, the rest Bob's.
DEMO_AGENTS = {
    "samantha": {"agent_id": "agent-104", "agent_display_name": "Samantha", "agent_extension": "104"},
    "bob": {"agent_id": "agent-202", "agent_display_name": "Bob", "agent_extension": "202"},
}


def redact(line: str) -> str:
    """The shared log never keeps a console credential or service-key secret (the terminal shows
    the one-time console URL; the file does not)."""
    return _SECRET.sub(lambda m: m.group(1) + "[redacted]", line)


_ACCESS_LOG = re.compile(r'^INFO:\s+\S+:\d+ - "')
"""A uvicorn access-log line (one per HTTP request)."""


class Launcher:
    def __init__(self, *, python: str = sys.executable, env: Optional[Dict[str, str]] = None, log_path: Path = Path("data/logs/launch.log"),
                 out=None, quiet_access: bool = False) -> None:
        self.python = python
        self.quiet_access = quiet_access
        """Keep the apps' per-request access lines out of the terminal (they still go to the log);
        the demo uses it so the progress lines stay readable."""
        self.env = dict(os.environ if env is None else env)
        self.log_path = log_path
        self.out = out or sys.stdout
        self.procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._log = None

    # --- output ------------------------------------------------------------------------------

    def emit(self, name: str, line: str) -> None:
        text = f"[{name}] {line.rstrip()}"
        with self._lock:
            if not (self.quiet_access and _ACCESS_LOG.match(line)):
                print(text, file=self.out, flush=True)
            if self._log is None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                self._log = os.fdopen(fd, "a", encoding="utf-8")
            self._log.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + redact(text) + "\n")
            self._log.flush()

    def _pump(self, name: str, stream) -> None:
        for raw in iter(stream.readline, ""):
            self.emit(name, raw)
        stream.close()

    # --- steps -------------------------------------------------------------------------------

    def run_step(self, name: str, args: List[str]) -> int:
        proc = subprocess.run([self.python, *args], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in (proc.stdout or "").splitlines():
            self.emit(name, line)
        return proc.returncode

    def process_config_path(self) -> Path:
        return Path(self.env.get("CALL1_PROCESS_CONFIG") or "data/process/config.json")

    def needs_service_key(self) -> bool:
        path = self.process_config_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except (OSError, ValueError):
            return True
        return not (isinstance(data, dict) and data.get("service_key") and data.get("store_url"))

    def first_run(self, installation: str, *, setup_hint: bool = True) -> int:
        if self.run_step("store", ["-m", "call1.store", "migrate"]) != 0:
            self.emit("launch", "Store migration failed; not starting.")
            return 1
        if self.needs_service_key():
            args = ["-m", "call1.store", "issue-service-key", "--installation", installation]
            for scope in PROCESS_SCOPES:
                args += ["--scope", scope]
            if self.run_step("store", args) != 0:
                self.emit("launch", "Could not issue the Process service key; not starting.")
                return 1
            self.emit("launch", f"Issued a Process service key for installation {installation!r} (with jobs:control for console retry/cancel).")
            if setup_hint:
                self.emit("launch", "Create the first admin, then enroll a passkey in Evaluate with the code it prints:")
                self.emit("launch", "  python -m call1.store setup-code --email you@example.com --display-name \"Your Name\"")
        return 0

    def issue_console_token(self) -> Optional[str]:
        """A fresh console credential (``console-token --rotate``); returns the console URL with the
        token in its fragment. Only the credential's hash is stored; the URL is never logged."""
        proc = subprocess.run([self.python, "-m", "call1.process", "console-token", "--rotate"], env=self.env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        url = next((line.strip() for line in (proc.stdout or "").splitlines() if "console_token=" in line), None)
        if proc.returncode != 0 or url is None:
            for line in (proc.stderr or "").splitlines():
                self.emit("process", line)
            return None
        return url

    def start(self) -> None:
        for name, module in (("store", "call1.store"), ("process", "call1.process")):
            proc = subprocess.Popen([self.python, "-m", module, "serve"], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            self.procs[name] = proc
            threading.Thread(target=self._pump, args=(name, proc.stdout), daemon=True).start()
            if name == "store":
                time.sleep(1.0)  # let Store bind before Process's first contract check

    def exited(self) -> Optional[str]:
        """The name of an app that has exited, if any."""
        return next((name for name, proc in self.procs.items() if proc.poll() is not None), None)

    def stop(self, timeout: float = 15.0) -> None:
        # SIGTERM: both apps run uvicorn, which shuts down gracefully on it (Process then stops its
        # worker: running jobs get the shutdown grace, and interrupted ones are reported to Store).
        for proc in self.procs.values():
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except OSError:
                    pass
        deadline = time.monotonic() + timeout
        for name, proc in self.procs.items():
            remaining = max(0.1, deadline - time.monotonic())
            try:
                proc.wait(remaining)
            except subprocess.TimeoutExpired:
                self.emit("launch", f"{name} did not stop in time; killing it")
                proc.kill()
                proc.wait()

    def wait(self) -> int:
        """Block until either app exits (then stop the other) or Ctrl-C."""
        try:
            while True:
                for name, proc in self.procs.items():
                    code = proc.poll()
                    if code is not None:
                        self.emit("launch", f"{name} exited with status {code}; stopping the other app")
                        return code or 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.emit("launch", "stopping Store and Process")
            return 0
        finally:
            _ignore_further_interrupts()  # a second Ctrl-C or SIGTERM must not abort the graceful stop
            self.stop()
            if self._log is not None:
                self._log.close()
                self._log = None


# --- demo mode -----------------------------------------------------------------------------------


@dataclass
class DemoPlan:
    root: Path
    store_port: int
    process_port: int
    handlers: str
    handlers_note: str
    open_browser: bool = True
    sample_dir: Path = SAMPLE_DIR

    @property
    def store_url(self) -> str:
        return f"http://localhost:{self.store_port}"

    @property
    def process_url(self) -> str:
        return f"http://127.0.0.1:{self.process_port}"

    @property
    def seeded_path(self) -> Path:
        return self.root / SEEDED_FILE


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def models_present(env: Dict[str, str]) -> bool:
    root = Path(env.get("CALL1_MODELS_DIR") or "data/models")
    try:
        return root.is_dir() and any(p.is_dir() for p in root.iterdir())
    except OSError:
        return False


def choose_demo_handlers(env: Dict[str, str], requested: Optional[str]) -> tuple:
    """(handlers, note). Real handlers only where the real appliance stack can run."""
    if requested == "real":
        return "real", "real handlers (--handlers real)"
    if requested == "fake":
        return "fake", "fake handlers (--handlers fake): scripted results, no models"
    if apple_silicon() and models_present(env):
        return "real", "real handlers on Apple Silicon (CALL1_BACKEND=mlx, weights from data/models)"
    why = "this is not an Apple-Silicon Mac" if not apple_silicon() else "data/models has no model weights"
    return "fake", f"fake handlers, because {why}: every stage runs, but results are scripted, not model output"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def reset_demo_root(root: Path) -> None:
    """Wipe a demo root, refusing anything that does not look like one."""
    if not root.exists():
        return
    resolved = root.resolve()
    if not (resolved.name == "demo" or (resolved / DEMO_MARKER).exists()):
        raise SystemExit(f"--reset refuses {resolved}: it is not a demo root (no {DEMO_MARKER} marker)")
    for protected in (Path("data/store"), Path("data/process"), Path("data"), Path.cwd(), Path.home()):
        if resolved == protected.resolve():
            raise SystemExit(f"--reset refuses {resolved}")
    shutil.rmtree(resolved)


def demo_env(base: Dict[str, str], plan: DemoPlan) -> Dict[str, str]:
    """The demo's environment: its own data root, dev-mode localhost Store with demo sign-in, and
    the chosen handlers. Settings that could point either app at real data or off loopback are
    dropped."""
    env = {k: v for k, v in base.items() if k not in (
        "CALL1_STORE_PUBLIC_URL", "CALL1_STORE_TLS_CERT", "CALL1_STORE_TLS_KEY", "CALL1_STORE_DEV", "CALL1_STORE_DEV_ORIGINS",
        "CALL1_PROCESS_WORKER_ID")}
    root = plan.root.resolve()
    env.update({
        "CALL1_STORE_DATA": str(root / "store"),
        "CALL1_STORE_HOSTNAME": "localhost",
        "CALL1_STORE_BIND": "127.0.0.1",
        "CALL1_STORE_PORT": str(plan.store_port),
        "CALL1_STORE_DEMO": "1",
        "CALL1_PROCESS_CONFIG": str(root / "process" / "config.json"),
        "CALL1_PROCESS_DATA": str(root / "process"),
        "CALL1_PROCESS_PORT": str(plan.process_port),
        "CALL1_PROCESS_BIND": "127.0.0.1",
        "CALL1_PROCESS_STORE_URL": plan.store_url,
        "CALL1_PROCESS_HANDLERS": plan.handlers,
    })
    if plan.handlers == "real" and apple_silicon():
        env["CALL1_BACKEND"] = "mlx"
    if plan.handlers == "fake":
        env["CALL1_FAKE_SCRIPTS"] = json.dumps(demo_fake_scripts(base.get("CALL1_FAKE_SCRIPTS"), plan.sample_dir))
    return env


def demo_signals_pipeline(env: Dict[str, str]) -> str:
    """The pipeline setting a demo starts with: ``v2`` unless ``CALL1_SIGNALS_PIPELINE`` names another."""
    wanted = (env.get("CALL1_SIGNALS_PIPELINE") or DEMO_SIGNALS_PIPELINE).strip()
    if wanted not in SIGNALS_PIPELINES:
        raise SystemExit(f"CALL1_SIGNALS_PIPELINE is v1, shadow or v2, not {wanted!r}")
    return wanted


def demo_fake_scripts(existing: Optional[str], sample_dir: Path = SAMPLE_DIR) -> Dict[str, str]:
    """``CALL1_FAKE_SCRIPTS`` for a fake-handler demo: the operator's own mapping (JSON), plus the
    ``cancel`` script on ``DEMO_CANCEL_CALL``, keyed on the checksum Process records for it (the
    SHA-256 of the file's bytes). A missing sample file just leaves the mapping as it was."""
    mapping: Dict[str, str] = {}
    if existing:
        try:
            parsed = json.loads(existing)
            if isinstance(parsed, dict):
                mapping.update({str(k): str(v) for k, v in parsed.items()})
        except ValueError:
            pass
    path = sample_dir / f"{DEMO_CANCEL_CALL}.wav"
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return mapping
    mapping.setdefault(f"sha256:{digest}", "cancel")
    return mapping


def demo_recordings(sample_dir: Path = SAMPLE_DIR) -> List[Dict[str, Any]]:
    """The five sample calls with their ingest metadata: agent identity and channel from
    ``manifest.json`` when it names them, else the legacy seed's agents (Samantha 104 on calls 01,
    03 and 05, Bob 202 on 02 and 04); ``external_call_ref`` is the manifest's call ID."""
    try:
        manifest = json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    items = []
    for path in sorted(sample_dir.glob("call_0*.wav")):
        entry = manifest.get(path.stem) or {}
        agent = dict(DEMO_AGENTS["samantha" if any(f"_{n}_" in path.name for n in ("01", "03", "05")) else "bob"])
        for key in ("agent_id", "agent_display_name", "agent_extension"):
            if entry.get(key):
                agent[key] = str(entry[key])
        if entry.get("agent_name") and not entry.get("agent_display_name"):
            agent["agent_display_name"] = str(entry["agent_name"])
        fields: Dict[str, Any] = dict(agent, external_call_ref=str(entry.get("call_id") or path.stem))
        if entry.get("agent_channel") in (0, 1):
            fields["agent_channel"] = entry["agent_channel"]
        elif int(entry.get("channels") or 0) == 2:
            channel = next((t.get("channel") for t in entry.get("turns") or [] if t.get("speaker") == "AGENT"), 0)
            fields["agent_channel"] = channel if channel in (0, 1) else 0
        items.append({"path": path, "name": path.stem, "fields": fields})
    return items


class Demo:
    """What ``--demo`` does once both apps run: wait for them, seed, open the browser, follow
    progress. HTTP only (``httpx``); it imports neither app."""

    def __init__(self, launcher: Launcher, plan: DemoPlan, console_url: str, *, opener: Optional[Callable[[str], None]] = None,
                 sample_dir: Path = SAMPLE_DIR, poll_seconds: float = 2.0) -> None:
        self.launcher = launcher
        self.plan = plan
        self.console_url = console_url
        self.token = console_url.split("console_token=", 1)[1] if "console_token=" in console_url else ""
        self.opener = opener or open_url
        self.sample_dir = sample_dir
        self.poll_seconds = poll_seconds

    def say(self, line: str) -> None:
        self.launcher.emit("demo", line)

    def _alive(self) -> bool:
        return self.launcher.exited() is None

    def wait_ready(self, timeout: float = 90.0) -> bool:
        import httpx

        deadline = time.monotonic() + timeout
        store_ok = process_ok = False
        state = None
        while time.monotonic() < deadline and self._alive():
            try:
                if not store_ok:
                    store_ok = httpx.get(f"{self.plan.store_url}/store/v1/status", timeout=3).status_code == 200
                if store_ok:
                    state = httpx.get(f"{self.plan.process_url}/process/api/health", timeout=3).json().get("state")
                    process_ok = state == "running"
            except (httpx.HTTPError, ValueError):
                pass
            if store_ok and process_ok:
                return True
            time.sleep(0.5)
        self.say(f"The apps did not come up (Store ready: {store_ok}, Process state: {state}); see the log above.")
        return False

    def seeded(self) -> Optional[List[Dict[str, Any]]]:
        try:
            data = json.loads(self.plan.seeded_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        calls = data.get("calls") if isinstance(data, dict) else None
        return calls if isinstance(calls, list) and calls else None

    def seed(self) -> List[Dict[str, Any]]:
        """Ingest every sample through the Process API. Ingest is idempotent (same recording, same
        conversation), so a partial seed is simply repeated at the next start."""
        import httpx

        recordings = demo_recordings(self.sample_dir)
        if self.plan.handlers == "real" and (self.sample_dir / ".test-audio.json").exists():
            self.say("Skipping generated test tones in real-model mode; use the AppTek demo studio or your own recording.")
            recordings = []
        calls: List[Dict[str, Any]] = []
        with httpx.Client(base_url=self.plan.process_url, timeout=120, headers={CONSOLE_HEADER: self.token}) as client:
            for item in recordings:
                fields = {k: str(v) for k, v in item["fields"].items()}
                with open(item["path"], "rb") as handle:
                    response = client.post("/process/api/recordings", data=fields, files={"file": (item["path"].name, handle, "audio/wav")})
                label = f"{fields['agent_display_name']} ({fields['agent_extension']})"
                if response.status_code != 201:
                    detail = response.text[:200]
                    self.say(f"  {item['name']}: not ingested ({response.status_code} {detail})")
                    continue
                body = response.json()
                calls.append({"name": item["name"], "agent": body.get("agent_label") or label, "conversation_id": body["conversation_id"],
                              "call_id": body["call_id"]})
                jobs = body.get("jobs")
                count = len(jobs) if isinstance(jobs, list) else jobs
                self.say(f"  {item['name']}: {body.get('agent_label') or label}, {count} jobs "
                         f"({'new' if body.get('conversation_created') else 'already registered'})")
        if calls and len(calls) == len(recordings):
            self.plan.seeded_path.write_text(json.dumps({"seeded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "calls": calls}, indent=2) + "\n",
                                             encoding="utf-8")
        return calls

    def open_browser(self) -> None:
        urls = [("Evaluate", f"{self.plan.store_url}/"), ("Store console", f"{self.plan.store_url}/console/"),
                ("Process console", self.console_url)]
        if not self.plan.open_browser:
            self.say("Not opening the browser (--no-open).")
            return
        for name, url in urls:
            try:
                self.opener(url)
            except Exception as exc:  # a demo without a browser still runs
                self.say(f"Could not open {name} ({type(exc).__name__}); open it by hand.")

    def follow(self, calls: List[Dict[str, Any]], timeout: float) -> bool:
        """Print each call's progress line when it changes, until every call has settled."""
        import httpx

        wanted = {c["conversation_id"]: c for c in calls}
        last: Dict[str, str] = {}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self._alive():
            try:
                items = httpx.get(f"{self.plan.process_url}/process/api/conversations", params={"limit": 200}, timeout=10).json()["items"]
            except (httpx.HTTPError, ValueError, KeyError):
                time.sleep(self.poll_seconds)
                continue
            settled = set()
            for item in items:
                call = wanted.get(item.get("conversation_id"))
                if call is None:
                    continue
                progress = item.get("progress") or {}
                line = item.get("progress_line") or "waiting"
                if progress.get("settled"):
                    settled.add(item["conversation_id"])
                    line = f"settled · {line}"
                if last.get(item["conversation_id"]) != line:
                    last[item["conversation_id"]] = line
                    self.say(f"  {call['name']} [{call['agent']}]: {line}")
            if settled >= set(wanted):
                self.say(f"All {len(wanted)} demo calls have settled. Evaluate: {self.plan.store_url}/")
                return True
            time.sleep(self.poll_seconds)
        if self._alive():
            self.say("Still processing; the calls keep going in the background (the Process console shows each job).")
        return False

    def setup(self) -> bool:
        """The demo's rubric policy, alert rule and queue rule (``call1.demo_setup``), once per demo
        root and before any call is ingested, so the seeded calls are scored with the policy. A
        failure is reported and the demo goes on (calls then show SEC-01/COMP-01 as needing policy)."""
        import httpx

        from call1.demo_setup import apply_demo_setup

        marker = self.plan.root / DEMO_SETUP_FILE
        if marker.exists():
            return True
        try:
            with httpx.Client(base_url=self.plan.store_url, timeout=30) as client:
                summary = apply_demo_setup(client, say=lambda line: self.say(f"  {line}"))
        except Exception as exc:
            self.say(f"Demo policy and alert setup not applied ({type(exc).__name__}: {exc}); scripts/apply_demo_policy.py retries it.")
            return False
        marker.write_text(json.dumps({"applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary}) + "\n", encoding="utf-8")
        return True

    def run(self, settle_timeout: Optional[float] = None) -> bool:
        if not self.wait_ready():
            return False
        self.setup()
        if self.launcher.run_step("store", ["-m", "call1.store", "seed-demo-history"]) != 0:
            self.say("Synthetic history was not seeded; the five audio demos can still run.")
        calls = self.seeded()
        if calls:
            self.say(f"Already seeded ({len(calls)} calls in {self.plan.root}); --reset starts over.")
        else:
            self.say(f"Seeding the demo: {len(demo_recordings(self.sample_dir))} sample calls from sample_audio/")
            calls = self.seed()
        self.say("Open:")
        self.say(f"  Evaluate          {self.plan.store_url}/   (use the labelled demo sign-in)")
        self.say(f"  Store console     {self.plan.store_url}/console/")
        self.say(f"  Process console   {self.console_url}")
        self.open_browser()
        if not calls:
            return False
        timeout = settle_timeout if settle_timeout is not None else (1800.0 if self.plan.handlers == "real" else 300.0)
        return self.follow(calls, timeout)


def open_url(url: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", url], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        import webbrowser

        webbrowser.open(url)


def run_demo(args: argparse.Namespace, base_env: Dict[str, str]) -> int:
    try:
        store_port = int(base_env.get("CALL1_STORE_PORT") or DEFAULT_STORE_PORT)
        process_port = int(base_env.get("CALL1_PROCESS_PORT") or DEFAULT_PROCESS_PORT)
    except ValueError:
        print("CALL1_STORE_PORT and CALL1_PROCESS_PORT must be port numbers", file=sys.stderr)
        return 2
    root = Path(args.demo_root or base_env.get("CALL1_DEMO_ROOT") or DEFAULT_DEMO_ROOT)
    if args.reset:
        reset_demo_root(root)
    for port, name, var in ((store_port, "Store", "CALL1_STORE_PORT"), (process_port, "Process", "CALL1_PROCESS_PORT")):
        if not port_free(port):
            print(f"Port {port} ({name}) is already in use; is another Call1 running? Stop it, or pick another port with {var}=...",
                  file=sys.stderr)
            return 2
    handlers, note = choose_demo_handlers(base_env, args.handlers)
    pipeline = demo_signals_pipeline(base_env)
    plan = DemoPlan(root=root, store_port=store_port, process_port=process_port, handlers=handlers, handlers_note=note,
                    open_browser=not args.no_open)
    (root / "process").mkdir(parents=True, exist_ok=True)
    (root / DEMO_MARKER).touch()
    env = demo_env(base_env, plan)
    env.setdefault("PYTHONUNBUFFERED", "1")
    launcher = Launcher(env=env, log_path=Path(args.log) if args.log else root / "logs" / "launch.log", quiet_access=True)
    bar = "=" * 78
    for line in (bar, "CALL1 DEMO MODE: localhost only. For a class demo, never for real calls.",
                 f"Demo data: {root.resolve()} (data/store and data/process are not touched).",
                 "Store demo sign-in is on (CALL1_STORE_DEMO=1): labelled personas, no passkey. Passkeys still work.",
                 f"Handlers: {note}.", signals_line(pipeline, handlers),
                 f"Per-request access lines go to {launcher.log_path} only.", bar):
        launcher.emit("launch", line)
    code = launcher.first_run(f"{args.installation}-demo", setup_hint=False)
    if code != 0:
        return code
    # Before either app starts, so every seeded call is planned on this taxonomy and pipeline (section 15).
    if not seed_demo_signals(launcher, plan, pipeline):
        launcher.emit("launch", f"Could not set the Contact Signals pipeline to {pipeline}; not starting.")
        return 1
    seed_demo_vocabulary(launcher)
    console_url = launcher.issue_console_token()
    if console_url is None:
        launcher.emit("launch", "Could not issue the Process console credential; not starting.")
        return 1
    launcher.start()
    demo = Demo(launcher, plan, console_url)
    threading.Thread(target=_guarded, args=(demo,), daemon=True, name="call1-demo").start()
    return launcher.wait()


def seed_demo_signals(launcher: Launcher, plan: DemoPlan, pipeline: str, seed: Path = DEMO_SIGNALS_SEED) -> bool:
    """Publish the retail seed taxonomy and set the pipeline, once per demo root; afterwards only set
    the pipeline, so a taxonomy edited on stage is kept across restarts. A seed that cannot be applied
    (say, a demo root from before the seed whose taxonomy was already edited) is reported and the demo
    goes on with the taxonomy it has. False only when the pipeline could not be set."""
    marker = plan.root / SIGNALS_SEEDED_FILE
    if not marker.exists():
        if launcher.run_step("store", ["-m", "call1.store", "apply-signals-seed", str(seed), "--pipeline", pipeline]) == 0:
            marker.write_text(json.dumps({"seed": seed.name, "seeded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n", encoding="utf-8")
            return True
        launcher.emit("launch", f"The retail signal taxonomy seed ({seed.name}) was not applied; keeping this demo root's taxonomy.")
    return launcher.run_step("store", ["-m", "call1.store", "signals-pipeline", pipeline]) == 0


def seed_demo_vocabulary(launcher: Launcher, seed: Path = DEMO_VOCABULARY_SEED) -> bool:
    """Install the retail ASR vocabulary as the industry pack (decision 33), before any call is
    ingested, so the seeded calls plan dual transcription. Idempotent on Store's side, so it runs at
    every start and keeps on-stage edits. A pack that cannot be installed is reported; the demo goes on."""
    if launcher.run_step("store", ["-m", "call1.store", "apply-vocabulary-seed", str(seed)]) == 0:
        return True
    launcher.emit("launch", f"The retail vocabulary seed ({seed.name}) was not installed; transcripts are not vocabulary-corrected.")
    return False


def signals_line(pipeline: str, handlers: str) -> str:
    what = {"v2": "category, subcategory and field stages", "shadow": "v2 beside v1 for comparison", "v1": "the v1 passes"}[pipeline]
    engine = ("the included model (Gemma 4 E2B)" if handlers == "real"
              else f"fake engines; {DEMO_CANCEL_CALL} plays the cancel script")
    return f"Contact Signals: pipeline {pipeline} ({what}), retail seed taxonomy, on {engine}."


def _guarded(demo: Demo) -> None:
    try:
        demo.run()
    except Exception as exc:  # the apps keep running; the operator can seed from the console
        demo.say(f"Demo setup stopped: {type(exc).__name__}: {exc}")


def _interrupt(signum, frame):
    # The first Ctrl-C or SIGTERM stops everything; ignore the rest at once, so one that lands while
    # the stop is being announced (a process group, or ``timeout`` forwarding its own) cannot abort it.
    _ignore_further_interrupts()
    raise KeyboardInterrupt


def _ignore_further_interrupts() -> None:
    """While both apps stop (each gets SIGTERM and a bounded wait, then a kill), further SIGINT or
    SIGTERM, e.g. from ``timeout`` or a process group, is ignored instead of aborting the stop."""
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m call1.launch", description="Run Call1 Store and Process together on this computer")
    parser.add_argument("--installation", default=socket.gethostname().split(".")[0] or "this-computer",
                        help="Process installation label for a first-run service key (default: this computer's name)")
    parser.add_argument("--handlers", choices=["fake", "real"], help="Process handlers (sets CALL1_PROCESS_HANDLERS)")
    parser.add_argument("--log", help="shared log file (default data/logs/launch.log; with --demo, <demo root>/logs/launch.log)")
    demo = parser.add_argument_group("demo mode (localhost only, for a class demo)")
    demo.add_argument("--demo", action="store_true", help="separate data root, Store demo sign-in, the five sample calls seeded")
    demo.add_argument("--reset", action="store_true", help="with --demo: wipe the demo data root first")
    demo.add_argument("--no-open", action="store_true", help="with --demo: do not open the three URLs in the browser")
    demo.add_argument("--demo-root", help="with --demo: the demo data root (default CALL1_DEMO_ROOT or data/demo)")
    args = parser.parse_args(argv)
    if (args.reset or args.no_open or args.demo_root) and not args.demo:
        parser.error("--reset, --no-open and --demo-root go with --demo")
    # Ctrl-C and SIGTERM both stop the pair, even when SIGINT was inherited as ignored.
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    env = dict(os.environ)
    if args.demo:
        return run_demo(args, env)
    if args.handlers:
        env["CALL1_PROCESS_HANDLERS"] = args.handlers
    env.setdefault("PYTHONUNBUFFERED", "1")
    launcher = Launcher(env=env, log_path=Path(args.log or "data/logs/launch.log"))
    code = launcher.first_run(args.installation)
    if code != 0:
        return code
    launcher.start()
    store_port = env.get("CALL1_STORE_PORT") or str(DEFAULT_STORE_PORT)
    process_port = env.get("CALL1_PROCESS_PORT") or str(DEFAULT_PROCESS_PORT)
    launcher.emit("launch", f"Evaluate: http://localhost:{store_port}/   Process console: http://127.0.0.1:{process_port}/   Ctrl-C stops both.")
    return launcher.wait()


if __name__ == "__main__":
    sys.exit(main())
