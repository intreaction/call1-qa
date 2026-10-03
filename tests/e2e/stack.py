"""A real Store + Process pair for end-to-end tests, and HTTP helpers to drive it.

``Stack`` starts ``python -m call1.store serve`` and ``python -m call1.process serve`` as separate
server processes on free loopback ports, exactly as an operator would run them:

1. ``python -m call1.store migrate`` on a fresh data directory;
2. ``python -m call1.store issue-service-key`` (the Stage 2 worker scopes plus ``jobs:control``),
   which writes Process's config file (mode 0600);
3. ``python -m call1.process console-token``, whose one-time token the stack keeps;
4. both servers, then a wait until Store answers ``/store/v1/status`` and Process reports its
   worker ``running``.

Everything a stack writes lives in ``/private/tmp/call1-e2e/<name>-<unique>/`` (never in the repo,
which is in iCloud Drive), and ``close()`` deletes it unless ``CALL1_E2E_KEEP=1``. Ports are chosen at
runtime and are never 8000, 8010 or 8020 (the user's live apps).

Reviewer sessions go through the real passkey ceremonies (``/auth/enroll/*``, ``/auth/sign-in/*``)
with the Store tests' software authenticator; nothing is minted.

This module has no pytest dependency: ``conftest.py`` wraps it in fixtures and ``serve_stack.py``
runs it for the Playwright suite.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import httpx

from .softauthn import SoftAuthenticator

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "sample_audio"
E2E_ROOT = Path(os.environ.get("CALL1_E2E_ROOT", "/private/tmp/call1-e2e"))
RESERVED_PORTS = frozenset({8000, 8010, 8020})
"""The user's live legacy app, Store and Process. A stack never binds them."""

API = "/store/v1"
PROCESS_API = "/process/api"
CONSOLE_HEADER = "X-Call1-Console-Token"
CSRF_HEADER = "X-Call1-CSRF"

PROCESS_SCOPES = (
    "calls:write", "artifacts:read", "artifacts:write", "jobs:write", "jobs:claim", "reanalysis:claim", "changes:read",
    "hardware:write", "catalog:publish", "usage:read", "admin-state:read", "training:read", "jobs:control",
)
"""What ``python -m call1.launch`` issues: Store's default Process scopes (``training:read`` since
1.3.0) plus ``jobs:control``."""

_PASSTHROUGH_CALL1_ENV = ("CALL1_REAL_MODELS",)
"""Inherited ``CALL1_*`` variables that may reach the servers. Every other one is dropped, so a
developer shell pointed at live data (``CALL1_STORE_DATA``, ``CALL1_PROCESS_CONFIG``...) can
never leak into a test stack."""

_unique = itertools.count(1)


def python_executable() -> str:
    """The interpreter the servers run under: ``CALL1_E2E_PYTHON``, else ``.venv-local``, else this one."""
    override = os.environ.get("CALL1_E2E_PYTHON")
    if override:
        return override
    local = REPO / ".venv-local" / "bin" / "python"
    return str(local) if local.exists() else sys.executable


def free_port() -> int:
    """A free loopback TCP port that is not one of the user's live-app ports."""
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in RESERVED_PORTS:
            return port
    raise RuntimeError("no free port")  # pragma: no cover


def _port_is_free(port: int) -> bool:
    """No listener on the port. SO_REUSEADDR, as uvicorn binds: TIME_WAIT connections left by a
    stopped server do not count."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def sample_path(name: Union[str, Path]) -> Path:
    """``"call_01_compliant"``, ``"call_01_compliant.wav"`` or any path to a recording."""
    candidate = Path(name)
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    for option in (SAMPLES / str(name), SAMPLES / f"{name}.wav"):
        if option.is_file():
            return option
    raise FileNotFoundError(f"no sample {name!r} in {SAMPLES}")


def unique_wav_copy(source: Path, target_dir: Path, marker: int) -> Path:
    """A copy of a PCM WAV whose content digest is new (Process identifies a recording by its
    SHA-256, so ingesting the same bytes twice returns the same conversation). The first four
    samples become small random values (|x| <= 255 for 16-bit audio): inaudible, and the header,
    duration and channels are unchanged. The Playwright helper does the same."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}-{marker:06d}-{secrets.token_hex(3)}{source.suffix}"
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = bytearray(reader.readframes(reader.getnframes()))
    width = params.sampwidth
    for index in range(min(4, len(frames) // width)):
        if width == 1:  # 8-bit WAV is unsigned around 128
            stamp = bytes([128 + secrets.randbelow(33) - 16])
        else:
            stamp = (secrets.randbelow(511) - 255).to_bytes(width, "little", signed=True)
        frames[index * width:(index + 1) * width] = stamp
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(bytes(frames))
    return target


def api_path(path: str, prefix: str = API) -> str:
    """Accepts ``/calls``, ``calls`` or ``/store/v1/calls`` (and the same for ``/process/api``)."""
    if path.startswith(("http://", "https://")):
        return path
    if path.startswith(prefix + "/") or path == prefix:
        return path
    return prefix + (path if path.startswith("/") else "/" + path)


def new_idempotency_key() -> str:
    return "e2e-" + uuid.uuid4().hex


class StackError(RuntimeError):
    """A server did not start, a CLI step failed, or a wait timed out. The message carries the
    relevant log tail."""


class _Server:
    def __init__(self, name: str, args: List[str], env: Mapping[str, str], cwd: Path, log_path: Path) -> None:
        self.name = name
        self.args = args
        self.env = dict(env)
        self.cwd = cwd
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self._log = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab", buffering=0)
        self._log.write(f"\n==== {self.name} start {time.strftime('%H:%M:%S')}: {' '.join(self.args)}\n".encode())
        self.proc = subprocess.Popen(self.args, env=self.env, cwd=str(self.cwd), stdin=subprocess.DEVNULL, stdout=self._log,
                                     stderr=subprocess.STDOUT, start_new_session=True)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: float = 15.0) -> Optional[int]:
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except OSError:
                    self.proc.kill()
                self.proc.wait(5)
        code = self.proc.returncode
        self.proc = None
        if self._log is not None:
            self._log.close()
            self._log = None
        return code


class Stack:
    """A Store and a Process server with temp data dirs. See the module docstring and README.md.

    ``handlers`` is ``"fake"`` (default; fast, no models) or ``"real"``. ``real_models=True`` also
    sets ``CALL1_BACKEND=mlx`` and points ``data/models`` at the repo's weights (Apple Silicon).
    ``fake_behavior`` is the ``CALL1_FAKE_BEHAVIOR`` script (``{"asr": ["fail:provider_error"]}``).
    ``store_parameters`` overrides ``ContractParameters`` (``CALL1_STORE_PARAMETERS``).
    ``process_config`` is merged into Process's config file (``{"slots": {...}}``).
    ``store_env`` / ``process_env`` add environment variables to one server.
    ``spread_clients`` (default on; ``CALL1_E2E_SPREAD_CLIENTS=0`` turns it off) gives every
    ``StoreSession`` its own client address for the anonymous auth steps; see ``StoreSession``.
    ``signals_pipeline`` (``v1``, ``shadow`` or ``v2``; default ``CALL1_SIGNALS_PIPELINE``, else
    Store's install default ``v1``) sets the Contact Signals pipeline setting with ``python -m
    call1.store signals-pipeline`` before either server starts (docs/ContactSignalsV2.md §8.6).
    ``signals_seed`` (a ``SignalTaxonomySave`` JSON path, e.g. the retail seed) is published first with
    ``python -m call1.store apply-signals-seed``, as ``python -m call1.launch --demo`` does (§15).
    ``vocabulary_seed`` (an ASR vocabulary seed JSON path, e.g. ``asr_vocabulary_retail_v1.json``) is
    installed as the industry pack with ``python -m call1.store apply-vocabulary-seed`` before either
    server starts, as ``--demo`` does (docs/DualAsr.md), so every ingest plans dual transcription.
    """

    def __init__(self, *, name: str = "stack", handlers: str = "fake", real_models: bool = False,
                 fake_behavior: Optional[Mapping[str, Any]] = None, store_parameters: Optional[Mapping[str, Any]] = None,
                 process_config: Optional[Mapping[str, Any]] = None, store_env: Optional[Mapping[str, str]] = None,
                 process_env: Optional[Mapping[str, str]] = None, with_process: bool = True, keep: Optional[bool] = None,
                 root: Optional[Path] = None, log_level: str = "info", start_timeout: Optional[float] = None,
                 spread_clients: Optional[bool] = None, signals_pipeline: Optional[str] = None,
                 signals_seed: Optional[str] = None, vocabulary_seed: Optional[str] = None) -> None:
        if handlers not in ("fake", "real"):
            raise ValueError("handlers is 'fake' or 'real'")
        self.name = re.sub(r"[^A-Za-z0-9_-]+", "-", name)[:40] or "stack"
        self.handlers = "real" if real_models else handlers
        self.real_models = real_models
        self.fake_behavior = dict(fake_behavior) if fake_behavior else None
        self.store_parameters = dict(store_parameters) if store_parameters else None
        self.process_config_extra = dict(process_config or {})
        self.store_env_extra = dict(store_env or {})
        self.process_env_extra = dict(process_env or {})
        self.with_process = with_process
        self.keep = (os.environ.get("CALL1_E2E_KEEP", "") == "1") if keep is None else keep
        self.log_level = log_level
        self.start_timeout = start_timeout or (180.0 if real_models else 45.0)
        self.python = python_executable()
        self.spread_clients = (os.environ.get("CALL1_E2E_SPREAD_CLIENTS", "1") != "0") if spread_clients is None else spread_clients
        pipeline = signals_pipeline if signals_pipeline is not None else (os.environ.get("CALL1_SIGNALS_PIPELINE") or None)
        if pipeline is not None and pipeline not in ("v1", "shadow", "v2"):
            raise ValueError(f"signals_pipeline (CALL1_SIGNALS_PIPELINE) is v1, shadow or v2, not {pipeline!r}")
        self.signals_pipeline = pipeline
        self.signals_seed = str(Path(signals_seed).resolve()) if signals_seed else None
        self.vocabulary_seed = str(Path(vocabulary_seed).resolve()) if vocabulary_seed else None

        root = Path(root or E2E_ROOT).resolve()
        if REPO in root.parents or root == REPO:
            raise StackError(f"e2e data must live outside the repo (iCloud Drive), not in {root}")
        self.dir = root / f"{self.name}-{time.strftime('%H%M%S')}-{os.getpid()}-{secrets.token_hex(3)}"
        self.store_data = self.dir / "store"
        self.process_dir = self.dir / "process"
        self.process_config_path = self.process_dir / "config.json"
        self.logs_dir = self.dir / "logs"
        self.uploads_dir = self.dir / "uploads"

        self.store_port: Optional[int] = None
        self.process_port: Optional[int] = None
        self.console_token: Optional[str] = None
        self.service_key: Optional[str] = None
        self.service_key_id: Optional[str] = None
        self.installation_id: Optional[str] = None

        self._store: Optional[_Server] = None
        self._process: Optional[_Server] = None
        self._admin: Optional[StoreSession] = None
        self._users: Dict[str, StoreSession] = {}
        self._conversations: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._sessions: List[StoreSession] = []
        self._http: Optional[httpx.Client] = None
        self._process_http: Optional[httpx.Client] = None
        self.started = False

    # --- URLs -------------------------------------------------------------------------------

    @property
    def store_url(self) -> str:
        """``http://localhost:<port>``: the WebAuthn origin (RP ID ``localhost``). Not 127.0.0.1."""
        return f"http://localhost:{self.store_port}"

    @property
    def process_url(self) -> str:
        return f"http://127.0.0.1:{self.process_port}"

    @property
    def evaluate_url(self) -> str:
        return self.store_url + "/"

    @property
    def process_console_url(self) -> str:
        """The console with its credential in the fragment, as ``serve`` prints it."""
        return f"{self.process_url}/#console_token={self.console_token}"

    # --- environment ------------------------------------------------------------------------

    def _base_env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("CALL1_") or k in _PASSTHROUGH_CALL1_ENV}
        env["PYTHONPATH"] = str(REPO)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CALL1_PROCESS_CONFIG"] = str(self.process_config_path)
        return env

    def store_env(self) -> Dict[str, str]:
        env = self._base_env()
        env.update({
            "CALL1_STORE_DATA": str(self.store_data),
            "CALL1_STORE_HOSTNAME": "localhost",
            "CALL1_STORE_BIND": "127.0.0.1",
            "CALL1_STORE_PORT": str(self.store_port or 0),
        })
        # Store embeds search queries with the same backend Process embeds turns with: the fake
        # embedder unless the stack runs the real models (call1.embedding).
        env["CALL1_EMBEDDING_BACKEND"] = "nemotron" if self.real_models else "fake"
        if self.store_parameters:
            env["CALL1_STORE_PARAMETERS"] = json.dumps(self.store_parameters)
        env.update(self.store_env_extra)
        return env

    def process_env(self) -> Dict[str, str]:
        env = self._base_env()
        env.update({
            "CALL1_PROCESS_HANDLERS": self.handlers,
            "CALL1_EMBEDDING_BACKEND": "nemotron" if self.real_models else "fake",
            "CALL1_PROCESS_PORT": str(self.process_port or 0),
            "CALL1_PROCESS_DATA": str(self.process_dir / "data"),
            "CALL1_PROCESS_WORKER_ID": f"e2e-{self.name}"[:60],
        })
        if self.fake_behavior:
            env["CALL1_FAKE_BEHAVIOR"] = json.dumps(self.fake_behavior)
        if self.real_models:
            env["CALL1_BACKEND"] = "mlx"
            env["CALL1_MODELS_DIR"] = str(self.dir / "data" / "models")
        env.update(self.process_env_extra)
        return env

    # --- CLI steps --------------------------------------------------------------------------

    def _run(self, name: str, args: List[str], env: Mapping[str, str], check: bool = True, timeout: float = 120.0) -> subprocess.CompletedProcess:
        proc = subprocess.run([self.python, *args], env=dict(env), cwd=str(self.dir), capture_output=True, text=True, timeout=timeout)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.logs_dir / "cli.log", "a", encoding="utf-8") as log:
            shown = re.sub(r"(c1sk_[A-Za-z0-9]{6}_)[A-Za-z0-9_-]+", r"\1[redacted]", proc.stdout + proc.stderr)
            log.write(f"==== {name}: {' '.join(args)} -> {proc.returncode}\n{shown}\n")
        if check and proc.returncode != 0:
            raise StackError(f"{name} failed ({proc.returncode}): {' '.join(args)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        return proc

    def store_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """``python -m call1.store <args>`` against this stack's data (as the Store host user)."""
        return self._run("store-cli", ["-m", "call1.store", *args], self.store_env(), check=check)

    def process_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """``python -m call1.process <args>`` with this stack's Process config."""
        return self._run("process-cli", ["-m", "call1.process", *args], self.process_env(), check=check)

    def setup_code(self, email: str, display_name: str = "E2E Admin", *, purpose: str = "first_admin",
                   target_account_id: Optional[str] = None) -> str:
        """A one-time enrollment code from ``python -m call1.store setup-code`` (the only way to get one)."""
        args = ["setup-code", "--email", email, "--display-name", display_name, "--purpose", purpose]
        if target_account_id:
            args += ["--target-account-id", target_account_id]
        proc = self.store_cli(*args)
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            raise StackError(f"setup-code printed no code:\n{proc.stdout}\n{proc.stderr}")
        return lines[-1]

    def _read_process_config(self) -> Dict[str, Any]:
        return json.loads(self.process_config_path.read_text(encoding="utf-8"))

    def _merge_process_config(self, values: Mapping[str, Any]) -> None:
        current = self._read_process_config() if self.process_config_path.exists() else {}
        current.update(values)
        tmp = self.process_config_path.with_name("config.json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2, sort_keys=True)
        os.replace(tmp, self.process_config_path)

    # --- lifecycle --------------------------------------------------------------------------

    def start(self) -> "Stack":
        if self.started:
            return self
        self.dir.mkdir(parents=True, exist_ok=False)
        for path in (self.store_data, self.process_dir, self.logs_dir, self.uploads_dir):
            path.mkdir(parents=True, exist_ok=True)
        models = REPO / "data" / "models"
        if models.is_dir():
            # Pre-split code resolves "data/models/..." against the working directory, which is
            # this stack's directory: point it at the repo's weights (read-only use).
            (self.dir / "data").mkdir(exist_ok=True)
            (self.dir / "data" / "models").symlink_to(models, target_is_directory=True)
        self.store_port = free_port()
        self.process_port = free_port()
        while self.process_port == self.store_port:
            self.process_port = free_port()
        try:
            self.store_cli("migrate")
            args = ["issue-service-key", "--installation", f"e2e-{self.name}"[:60], "--primary-host", "--config", str(self.process_config_path)]
            for scope in PROCESS_SCOPES:
                args += ["--scope", scope]
            self.store_cli(*args)
            if self.signals_seed:
                self.store_cli("apply-signals-seed", self.signals_seed)
            if self.vocabulary_seed:
                self.store_cli("apply-vocabulary-seed", self.vocabulary_seed)
            if self.signals_pipeline:
                self.store_cli("signals-pipeline", self.signals_pipeline)
            config = self._read_process_config()
            self.service_key = config["service_key"]
            self.service_key_id = config["service_key_id"]
            self.installation_id = config["installation_id"]
            extra = {"port": self.process_port, "data_dir": str(self.process_dir / "data"), "handlers": self.handlers,
                     "poll_interval_seconds": 0.2, "shutdown_grace_seconds": 2.0}
            extra.update(self.process_config_extra)
            self._merge_process_config(extra)
            if self.with_process:
                proc = self.process_cli("console-token")
                match = re.search(r"console_token=(c1con_[A-Za-z0-9_-]+)", proc.stdout)
                if not match:
                    raise StackError(f"console-token printed no token:\n{proc.stdout}\n{proc.stderr}")
                self.console_token = match.group(1)
            self._first_start("store")
            if self.with_process:
                self._first_start("process")
        except BaseException:
            self.stop()
            if not self.keep:
                self._remove_dir()
            raise
        self.started = True
        return self

    def _first_start(self, name: str, attempts: int = 3) -> None:
        """Start one server; if another process took its port between ``free_port()`` and the bind
        (parallel test runs), move to a new port and try again."""
        def move() -> None:
            if name == "store":
                self.store_port = free_port()
                self._merge_process_config({"store_url": self.store_url})
            else:
                self.process_port = free_port()
                self._merge_process_config({"port": self.process_port})

        for attempt in range(attempts):
            if not _port_is_free(self.store_port if name == "store" else self.process_port):
                move()
            try:
                self.start_store() if name == "store" else self.start_process()
                return
            except StackError:
                if attempt == attempts - 1 or "address already in use" not in self.log_tail(name, 30).lower():
                    raise
            move()

    def start_store(self) -> None:
        """Start (or restart after ``stop_store``) Store on the same port and data."""
        if self._store is not None and self._store.running:
            return
        self._wait_port_free(self.store_port)
        self._store = _Server("store", [self.python, "-m", "call1.store", "serve", "--log-level", self.log_level], self.store_env(),
                              self.dir, self.logs_dir / "store.log")
        self._store.start()
        self._wait(lambda: self._store_ready(), "Store to answer /store/v1/status", self._store, timeout=self.start_timeout)

    def stop_store(self) -> None:
        """Stop Store (SIGTERM). Process keeps running and sees Store unreachable."""
        if self._store is not None:
            self._store.stop()

    def start_process(self, *, wait_running: bool = True) -> None:
        """Start (or restart after ``stop_process``) Process; waits for its worker to be ``running``."""
        if self._process is not None and self._process.running:
            return
        self._wait_port_free(self.process_port)
        self._process = _Server("process", [self.python, "-m", "call1.process", "serve", "--log-level", self.log_level], self.process_env(),
                                self.dir, self.logs_dir / "process.log")
        self._process.start()
        if wait_running:
            self._wait(lambda: self._process_state() == "running", "Process worker state 'running'", self._process, timeout=self.start_timeout)
        else:
            self._wait(lambda: self._process_state() is not None, "Process to answer /process/api/health", self._process,
                       timeout=self.start_timeout)

    def stop_process(self) -> None:
        if self._process is not None:
            self._process.stop()

    def ensure_running(self) -> None:
        """Restart a server a previous test stopped (or that died); the ``stack`` fixture calls
        this before every test so one test's outage cannot leak into the next."""
        if self._store is not None and not self._store.running:
            self.start_store()
        if self.with_process and self._process is not None and not self._process.running:
            self.start_process()

    def stop(self) -> None:
        for session in list(self._sessions):
            session.close()
        self._sessions.clear()
        for client in (self._http, self._process_http):
            if client is not None:
                client.close()
        self._http = self._process_http = None
        # Process first: its graceful stop reports interrupted jobs to Store.
        for server in (self._process, self._store):
            if server is not None:
                server.stop()

    def close(self) -> None:
        """Stop both servers and delete the stack's directory (kept when ``CALL1_E2E_KEEP=1``)."""
        self.stop()
        self.started = False
        if not self.keep:
            self._remove_dir()

    def _remove_dir(self) -> None:
        if self.dir.exists() and E2E_ROOT.resolve() in self.dir.resolve().parents:
            link = self.dir / "data" / "models"
            if link.is_symlink():
                link.unlink()  # never follow the link into the repo's weights
            shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self) -> "Stack":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # --- waits ------------------------------------------------------------------------------

    @staticmethod
    def _wait_port_free(port: Optional[int], timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while port and not _port_is_free(port):
            if time.monotonic() > deadline:
                raise StackError(f"port {port} is still in use")
            time.sleep(0.1)

    def _store_ready(self) -> bool:
        try:
            return httpx.get(f"http://127.0.0.1:{self.store_port}{API}/status", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            return False

    def _process_state(self) -> Optional[str]:
        try:
            response = httpx.get(f"{self.process_url}{PROCESS_API}/health", timeout=1.0)
        except httpx.HTTPError:
            return None
        return response.json().get("state") if response.status_code == 200 else None

    def _wait(self, ready, what: str, server: Optional[_Server], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if ready():
                return
            if server is not None and not server.running:
                raise StackError(f"{server.name} exited while waiting for {what}:\n{self.log_tail(server.name)}")
            time.sleep(0.1)
        raise StackError(f"timed out after {timeout:.0f}s waiting for {what}:\n{self.log_tail(server.name if server else 'store')}")

    def log_tail(self, name: str, lines: int = 80) -> str:
        """The last lines of ``logs/<name>.log`` (``store``, ``process`` or ``cli``)."""
        path = self.logs_dir / f"{name}.log"
        if not path.exists():
            return f"(no {path})"
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(text[-lines:])

    def wait_for(self, predicate, *, timeout: float = 20.0, interval: float = 0.2, what: str = "condition"):
        """Poll ``predicate()`` until it returns a truthy value, which is returned."""
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = predicate()
            if last:
                return last
            time.sleep(interval)
        raise StackError(f"timed out after {timeout:.0f}s waiting for {what} (last value: {last!r})")

    # --- raw HTTP ---------------------------------------------------------------------------

    @property
    def http(self) -> httpx.Client:
        """An anonymous client on Store (no cookies are kept between tests: do not sign in with it)."""
        if self._http is None:
            self._http = httpx.Client(base_url=self.store_url, timeout=30.0)
        return self._http

    @property
    def process_http(self) -> httpx.Client:
        if self._process_http is None:
            self._process_http = httpx.Client(base_url=self.process_url, timeout=60.0)
        return self._process_http

    def service_headers(self) -> Dict[str, str]:
        """``Authorization: Bearer c1sk_...``: Process's own service key (the 12 scopes above)."""
        return {"Authorization": f"Bearer {self.service_key}"}

    def store_request(self, method: str, path: str, *, session: Union["StoreSession", str, None] = None, json: Any = None,
                      idempotency_key: Union[str, bool, None] = None, headers: Optional[Mapping[str, str]] = None, **kw) -> httpx.Response:
        """One request to ``/store/v1``. ``session``: a ``StoreSession`` (cookie + CSRF on writes),
        ``"service"`` (Process's service key) or ``None`` (anonymous)."""
        if isinstance(session, StoreSession):
            return session.request(method, path, json=json, idempotency_key=idempotency_key, headers=headers, **kw)
        merged = dict(headers or {})
        if session == "service":
            merged.update(self.service_headers())
        elif session is not None:
            raise ValueError("session is a StoreSession, 'service' or None")
        if idempotency_key:
            merged["Idempotency-Key"] = new_idempotency_key() if idempotency_key is True else str(idempotency_key)
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            merged.setdefault("Origin", self.store_url)
        return self.http.request(method, api_path(path), json=json, headers=merged, **kw)

    def store_get(self, path: str, *, session: Union["StoreSession", str, None] = None, **kw) -> httpx.Response:
        return self.store_request("GET", path, session=session, **kw)

    def store_post(self, path: str, json: Any = None, *, session: Union["StoreSession", str, None] = None,
                   idempotency_key: Union[str, bool, None] = None, **kw) -> httpx.Response:
        return self.store_request("POST", path, session=session, json=json, idempotency_key=idempotency_key, **kw)

    def process_request(self, method: str, path: str, *, token: Union[bool, str] = True, headers: Optional[Mapping[str, str]] = None,
                        **kw) -> httpx.Response:
        """One request to Process's loopback API (``/process/api``). ``token=True`` sends the console
        credential (writes need it; reads do not); a string sends that token; ``False`` none."""
        merged = dict(headers or {})
        if token:
            merged[CONSOLE_HEADER] = self.console_token if token is True else str(token)
        return self.process_http.request(method, api_path(path, PROCESS_API), headers=merged, **kw)

    def process_get(self, path: str, **kw) -> httpx.Response:
        return self.process_request("GET", path, **kw)

    def process_post(self, path: str, json: Any = None, **kw) -> httpx.Response:
        return self.process_request("POST", path, json=json, **kw)

    # --- recordings -------------------------------------------------------------------------

    def ingest(self, source: Union[str, Path] = "call_01_compliant", *, unique: bool = True, agent_id: Optional[str] = None,
               agent_channel: Optional[int] = None, external_call_ref: Optional[str] = None, filename: Optional[str] = None,
               content_type: str = "audio/wav", expect_status: Optional[int] = 201) -> Dict[str, Any]:
        """Upload a recording through Process's API (``POST /process/api/recordings``), as the
        console's Import view does. ``source`` is a sample name (``call_03_dispute_escalation``)
        or a path. ``unique=True`` (default) ingests a copy with a new content digest, so every
        call is a new conversation even on the shared stack; ``unique=False`` sends the file as is
        (a repeat returns the same conversation). Returns Process's receipt:
        ``{conversation_id, call_id, graph_id, conversation_created, graph_created, jobs, evaluate_url}``."""
        path = sample_path(source)
        if unique and path.suffix.lower() == ".wav":
            path = unique_wav_copy(path, self.uploads_dir, next(_unique))
        data: Dict[str, Any] = {}
        if agent_id is not None:
            data["agent_id"] = agent_id
        if agent_channel is not None:
            data["agent_channel"] = str(agent_channel)
        if external_call_ref is not None:
            data["external_call_ref"] = external_call_ref
        with open(path, "rb") as handle:
            response = self.process_request("POST", "/recordings", files={"file": (filename or path.name, handle, content_type)}, data=data)
        if expect_status is not None and response.status_code != expect_status:
            raise StackError(f"ingest of {path.name} answered {response.status_code}: {response.text}\n{self.log_tail('process', 40)}")
        body = response.json()
        if response.status_code < 300 and body.get("call_id"):
            self._conversations[body["call_id"]] = body["conversation_id"]
        return body

    def conversation_id(self, call_or_conversation_id: str) -> str:
        """The conversation of a call ingested through this stack (or of any call Process's ledger lists)."""
        value = call_or_conversation_id
        if value in self._conversations:
            return self._conversations[value]
        if value.startswith("conv_"):
            return value
        response = self.process_get("/conversations", params={"limit": 200}, token=False)
        for item in response.json().get("items", []):
            if item.get("call_id") == value:
                self._conversations[value] = item["conversation_id"]
                return item["conversation_id"]
        raise StackError(f"{value} is neither a conversation ID nor a call Process's ledger lists")

    def progress(self, call_or_conversation_id: str) -> Dict[str, Any]:
        """Store's ``JobGroupProgress`` for the call's conversation (read with the service key)."""
        conversation = self.conversation_id(call_or_conversation_id)
        response = self.store_get(f"/conversations/{conversation}/progress", session="service")
        if response.status_code != 200:
            raise StackError(f"progress of {conversation} answered {response.status_code}: {response.text}")
        return response.json()

    def wait_until_settled(self, call_or_conversation_id: str, *, timeout: Optional[float] = None, interval: float = 0.25) -> Dict[str, Any]:
        """Poll until every job of the call's conversation is terminal or dead-blocked
        (``JobGroupProgress.settled``); returns that progress. Raises ``StackError`` with the Process
        log tail on timeout."""
        timeout = timeout if timeout is not None else (900.0 if self.real_models else 60.0)
        deadline = time.monotonic() + timeout
        last: Dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.progress(call_or_conversation_id)
            if last.get("settled"):
                return last
            time.sleep(interval)
        raise StackError(f"conversation of {call_or_conversation_id} not settled after {timeout:.0f}s: "
                         f"{json.dumps(last)[:2000]}\n{self.log_tail('process', 60)}")

    # --- reviewer sessions --------------------------------------------------------------------

    def new_session(self, email: Optional[str] = None, *, authenticator: Optional[SoftAuthenticator] = None) -> "StoreSession":
        """A fresh browser-like client (own cookie jar, own software authenticator), not signed in."""
        session = StoreSession(self, email or unique_email("user"), authenticator=authenticator)
        self._sessions.append(session)
        return session

    def admin(self) -> "StoreSession":
        """The stack's first admin, enrolled once with a CLI setup code through the real ceremony."""
        with self._lock:
            if self._admin is None:
                admin = self.new_session("admin@e2e.test")
                admin.enroll(setup_code=self.setup_code(admin.email, "E2E Admin"), nickname="E2E admin key")
                self._admin = admin
            return self._admin

    def invite(self, email: str, role: str = "reviewer", *, display_name: Optional[str] = None, **extra) -> str:
        """The admin issues an out-of-band invitation; returns the invitation token (the URL fragment)."""
        response = self.admin().post("/admin/invitations", json={"email": email, "display_name": display_name or email.split("@")[0],
                                                                  "role": role, **extra})
        if response.status_code != 200:
            raise StackError(f"createInvitation answered {response.status_code}: {response.text}")
        return response.json()["invitation_url"].split("#", 1)[1]

    def user(self, role: str = "reviewer", *, email: Optional[str] = None, display_name: Optional[str] = None,
             cached: bool = True) -> "StoreSession":
        """A signed-in account of ``role`` (reviewer, supervisor or admin): invited by the admin and
        enrolled through the real ceremony. ``cached=True`` returns the same account per role on
        this stack; ``cached=False`` (or an ``email``) always creates a new one."""
        with self._lock:
            key = role
            if cached and email is None and key in self._users:
                return self._users[key]
            person = self.new_session(email or unique_email(role))
            person.enroll(invitation_token=self.invite(person.email, role, display_name=display_name))
            if cached and email is None:
                self._users[key] = person
            return person

    # --- description ------------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """JSON-serializable facts about the running stack (what ``serve_stack.py`` prints)."""
        return {
            "dir": str(self.dir),
            "handlers": self.handlers,
            "real_models": self.real_models,
            "signals_pipeline": self.signals_pipeline or "v1",
            "python": self.python,
            "repo": str(REPO),
            "store_url": self.store_url,
            "store_port": self.store_port,
            "store_data": str(self.store_data),
            "process_url": self.process_url,
            "process_port": self.process_port,
            "process_config": str(self.process_config_path),
            "process_console_url": self.process_console_url if self.console_token else None,
            "console_token": self.console_token,
            "installation_id": self.installation_id,
            "service_key_id": self.service_key_id,
            "service_key": self.service_key,
            "logs_dir": str(self.logs_dir),
            "store_env": {k: v for k, v in self.store_env().items() if k.startswith("CALL1_") or k in ("PYTHONPATH",)},
        }


def unique_email(prefix: str = "user") -> str:
    """``<prefix>-<n>-<hex>@e2e.test``: unique on a shared stack."""
    safe = re.sub(r"[^a-z0-9]+", "-", prefix.lower()).strip("-") or "user"
    return f"{safe}-{next(_unique)}-{secrets.token_hex(3)}@e2e.test"


_client_counter = itertools.count(1)


def synthetic_client_address() -> str:
    """A unique ``10.x.y.z`` address for one simulated browser (see ``StoreSession``)."""
    n = next(_client_counter)
    return f"10.{secrets.randbelow(200) + 20}.{(n >> 8) & 0xFF}.{n & 0xFF or 1}"


class StoreSession:
    """One reviewer's browser, over HTTP: its own cookie jar and software authenticator.

    ``enroll`` / ``sign_in`` run the real WebAuthn ceremonies. Writes send the session's
    ``X-Call1-CSRF`` and the Store ``Origin``; ``idempotency_key=True`` adds a fresh
    ``Idempotency-Key`` (or pass your own string to repeat a request under the same key).

    **Auth rate limits.** Store limits the anonymous steps per client address (enrollment begin 10
    a minute, sign-in begin 30 a minute) and sign-in per email (10 per 5 minutes); every test
    process connects from 127.0.0.1. So that unrelated tests on the shared stack do not starve each
    other, each session sends its own ``X-Forwarded-For`` (``client_address``) on the two ``begin``
    calls only; uvicorn trusts that header from 127.0.0.1 by default, so Store keys its limiter on
    it. Nothing else is sent with it, so session records keep the real address. A 429 on a begin
    call is waited out once (``Retry-After``) unless ``retry_rate_limited=False``. Tests of the
    limits themselves pass ``client_address=None`` (or use ``Stack(spread_clients=False)``).
    """

    def __init__(self, stack: Stack, email: str, *, authenticator: Optional[SoftAuthenticator] = None,
                 client_address: Union[str, None, bool] = True) -> None:
        self.stack = stack
        self.email = email
        self.authenticator = authenticator or SoftAuthenticator()
        self.http = httpx.Client(base_url=stack.store_url, timeout=30.0)
        self.session: Dict[str, Any] = {}
        self.account: Dict[str, Any] = {}
        if client_address is True:
            client_address = synthetic_client_address() if stack.spread_clients else None
        self.client_address: Optional[str] = client_address or None
        self.retry_rate_limited = True

    def _begin(self, path: str, body: Mapping[str, Any]) -> httpx.Response:
        headers = {"Origin": self.stack.store_url}
        if self.client_address:
            headers["X-Forwarded-For"] = self.client_address
        response = self.http.post(api_path(path), json=dict(body), headers=headers)
        if response.status_code == 429 and self.retry_rate_limited:
            wait = min(float(response.headers.get("Retry-After") or 5) + 0.5, 65.0)
            time.sleep(wait)
            response = self.http.post(api_path(path), json=dict(body), headers=headers)
        return response

    # identity
    @property
    def account_id(self) -> str:
        return self.session["account_id"]

    @property
    def role(self) -> str:
        return self.session["role"]

    @property
    def csrf_token(self) -> Optional[str]:
        return self.session.get("csrf_token")

    @property
    def permissions(self) -> List[str]:
        return list(self.session.get("permissions", []))

    @property
    def cookies(self) -> Dict[str, str]:
        return dict(self.http.cookies)

    # ceremonies
    def enroll(self, *, setup_code: Optional[str] = None, invitation_token: Optional[str] = None, nickname: Optional[str] = None,
               check: bool = True, **register_kw) -> httpx.Response:
        """``/auth/enroll/begin`` + the authenticator + ``/auth/enroll/finish``. On success the
        session is signed in. ``register_kw`` goes to ``SoftAuthenticator.register`` (``origin=``,
        ``rp_id=``, ``user_verified=`` ... to misbehave on purpose)."""
        body = {"setup_code": setup_code} if setup_code else {"invitation_token": invitation_token}
        begin = self._begin("/auth/enroll/begin", body)
        if begin.status_code != 200:
            if check:
                raise StackError(f"enroll/begin answered {begin.status_code}: {begin.text}")
            return begin
        options = begin.json()
        register_kw.setdefault("origin", self.stack.store_url)
        credential = self.authenticator.register(options["options"], **register_kw)
        finish: Dict[str, Any] = {"ceremony_id": options["ceremony_id"], "credential": credential}
        if nickname:
            finish["nickname"] = nickname
        response = self.http.post(api_path("/auth/enroll/finish"), json=finish, headers={"Origin": self.stack.store_url})
        if response.status_code == 200:
            body = response.json()
            self.account = body.get("account") or {}
            if body.get("signed_in"):
                self.session = body["signed_in"]["session"]
        elif check:
            raise StackError(f"enroll/finish answered {response.status_code}: {response.text}")
        return response

    def sign_in(self, *, check: bool = True, email: Optional[str] = None, **assert_kw) -> httpx.Response:
        """Account-first sign-in (``/auth/sign-in/begin`` with the email, then the assertion)."""
        begin = self._begin("/auth/sign-in/begin", {"email": email or self.email})
        if begin.status_code != 200:
            if check:
                raise StackError(f"sign-in/begin answered {begin.status_code}: {begin.text}")
            return begin
        options = begin.json()
        assert_kw.setdefault("origin", self.stack.store_url)
        credential = self.authenticator.assert_(options["options"], **assert_kw)
        response = self.http.post(api_path("/auth/sign-in/finish"), json={"ceremony_id": options["ceremony_id"], "credential": credential},
                                  headers={"Origin": self.stack.store_url})
        if response.status_code == 200:
            self.session = response.json()["session"]
        elif check:
            raise StackError(f"sign-in/finish answered {response.status_code}: {response.text}")
        return response

    def sign_out(self) -> httpx.Response:
        response = self.post("/auth/sign-out")
        if response.status_code < 300:
            self.session = {}
        return response

    def refresh(self) -> Dict[str, Any]:
        """Re-read ``GET /auth/session`` (role and permissions are re-read on every request)."""
        response = self.get("/auth/session")
        if response.status_code == 200:
            body = response.json()
            self.session = body.get("session", body)
        return self.session

    # requests
    def request(self, method: str, path: str, *, json: Any = None, idempotency_key: Union[str, bool, None] = None,
                headers: Optional[Mapping[str, str]] = None, csrf: bool = True, **kw) -> httpx.Response:
        merged = dict(headers or {})
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            merged.setdefault("Origin", self.stack.store_url)
            if csrf and self.csrf_token:
                merged.setdefault(CSRF_HEADER, self.csrf_token)
        if idempotency_key:
            merged["Idempotency-Key"] = new_idempotency_key() if idempotency_key is True else str(idempotency_key)
        return self.http.request(method, api_path(path), json=json, headers=merged, **kw)

    def get(self, path: str, **kw) -> httpx.Response:
        return self.request("GET", path, **kw)

    def post(self, path: str, json: Any = None, **kw) -> httpx.Response:
        return self.request("POST", path, json=json, **kw)

    def put(self, path: str, json: Any = None, **kw) -> httpx.Response:
        return self.request("PUT", path, json=json, **kw)

    def patch(self, path: str, json: Any = None, **kw) -> httpx.Response:
        return self.request("PATCH", path, json=json, **kw)

    def delete(self, path: str, **kw) -> httpx.Response:
        return self.request("DELETE", path, **kw)

    def close(self) -> None:
        self.http.close()

    def __repr__(self) -> str:
        return f"StoreSession({self.email!r}, role={self.session.get('role')!r})"
