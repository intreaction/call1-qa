"""Helpers for the Process e2e tests (``test_process_*.py``). Test infrastructure only: nothing
here changes application code. Everything talks to the real servers a ``Stack`` started.

- ``jobs_of`` / ``attempts_of`` / ``artifact_content``: Store reads with Process's service key.
- ``wait_job``: poll one job until a predicate holds.
- ``fixed_copy`` / ``mono_copy``: recordings with a known digest (for idempotency) or one channel.
- ``validate_outputs``: every output artifact of a job, parsed with the contract content model.
- ``SecondProcess``: another ``python -m call1.process serve`` on the same Store and installation
  (a second worker racing the stack's own), on its own free port and data dir.
- ``free_port`` re-export, ``run_python`` for one-off host commands with a stack-like environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import httpx

from .stack import REPO, Stack, StackError, _Server, free_port, python_executable, sample_path, unique_wav_copy  # noqa: F401

TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


# --- Store reads (service key) ----------------------------------------------------------------


def jobs_of(stack: Stack, conversation_id: str, **params) -> List[Dict[str, Any]]:
    """Every job of a conversation (``GET /store/v1/jobs?conversation_id=``), all pages."""
    items: List[Dict[str, Any]] = []
    token: Optional[str] = None
    while True:
        query = {"conversation_id": conversation_id, "limit": 200, **params}
        if token:
            query["page_token"] = token
        response = stack.store_get("/jobs", session="service", params=query)
        assert response.status_code == 200, response.text
        body = response.json()
        items.extend(body["items"])
        token = body.get("next_page_token")
        if not token:
            return items


def job(stack: Stack, job_id: str) -> Dict[str, Any]:
    response = stack.store_get(f"/jobs/{job_id}", session="service")
    assert response.status_code == 200, response.text
    return response.json()


def attempts_of(stack: Stack, job_id: str) -> List[Dict[str, Any]]:
    response = stack.store_get(f"/jobs/{job_id}/attempts", session="service", params={"limit": 200})
    assert response.status_code == 200, response.text
    return response.json()["items"]


def artifact(stack: Stack, artifact_id: str) -> Dict[str, Any]:
    response = stack.store_get(f"/artifacts/{artifact_id}", session="service")
    assert response.status_code == 200, response.text
    return response.json()


def artifact_content(stack: Stack, artifact_id: str) -> bytes:
    response = stack.store_get(f"/artifacts/{artifact_id}/content", session="service", follow_redirects=True)
    assert response.status_code == 200, f"content of {artifact_id}: {response.status_code} {response.text[:300]}"
    return response.content


def graph(stack: Stack, graph_id: str) -> Dict[str, Any]:
    response = stack.store_get(f"/job-graphs/{graph_id}", session="service")
    assert response.status_code == 200, response.text
    return response.json()


def by_type(jobs: Iterable[Mapping[str, Any]], job_type: str) -> List[Mapping[str, Any]]:
    return [j for j in jobs if j["job_type"] == job_type]


def one(jobs: Iterable[Mapping[str, Any]], job_type: str) -> Mapping[str, Any]:
    found = by_type(jobs, job_type)
    assert len(found) == 1, f"expected one {job_type} job, got {len(found)}: {[j['id'] for j in found]}"
    return found[0]


def wait_job(stack: Stack, job_id: str, predicate: Callable[[Dict[str, Any]], bool], *, timeout: float = 30.0,
             what: str = "job condition") -> Dict[str, Any]:
    return stack.wait_for(lambda: (lambda j: j if predicate(j) else None)(job(stack, job_id)), timeout=timeout, interval=0.2,
                          what=f"{what} on {job_id}")


def wait_job_of_type(stack: Stack, conversation_id: str, job_type: str, predicate: Callable[[Dict[str, Any]], bool], *,
                     timeout: float = 30.0) -> Dict[str, Any]:
    def probe():
        for j in jobs_of(stack, conversation_id):
            if j["job_type"] == job_type and predicate(j):
                return j
        return None
    return stack.wait_for(probe, timeout=timeout, interval=0.2, what=f"{job_type} of {conversation_id}")


def rubric(stack: Stack, rubric_id: str = "call1_standard_v2") -> Dict[str, Any]:
    response = stack.store_get(f"/rubrics/{rubric_id}", session="service")
    assert response.status_code == 200, response.text
    return response.json()


# --- recordings -----------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_copy(stack: Stack, sample: str = "call_01_compliant") -> Path:
    """A new recording (new digest) saved once, so the test can upload the SAME bytes twice."""
    return unique_wav_copy(sample_path(sample), stack.uploads_dir, secrets.randbelow(900_000) + 100_000)


def mono_copy(stack: Stack, sample: str = "call_01_compliant") -> Path:
    """A one-channel downmix of a stereo sample (new digest), for the mono-only speaker stage."""
    source = sample_path(sample)
    target = stack.uploads_dir / f"{source.stem}-mono-{secrets.token_hex(4)}.wav"
    stack.uploads_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    assert params.sampwidth == 2, "the samples are 16-bit PCM"
    import array

    data = array.array("h")
    data.frombytes(frames)
    if params.nchannels == 2:
        mono = array.array("h", ((data[i] + data[i + 1]) // 2 for i in range(0, len(data) - 1, 2)))
    else:
        mono = data
    mono[0] = secrets.randbelow(200) - 100  # a new digest even for the same downmix
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(params.framerate)
        writer.writeframes(mono.tobytes())
    return target


# --- contract validation --------------------------------------------------------------------------


def validate_outputs(stack: Stack, job_record: Mapping[str, Any]) -> Dict[str, Any]:
    """Every output of a SUCCEEDED job: the role matches the job type's contract rule, the artifact
    is committed and linked with the declared kind, its bytes hash to the recorded checksum, and a
    JSON kind parses with its contract content model. Returns ``{role: parsed content or bytes}``."""
    from call1.contracts.artifacts import ArtifactKind, content_model_for
    from call1.contracts.jobs import JOB_TYPE_RULES, JobType

    rule = JOB_TYPE_RULES[JobType(job_record["job_type"])]
    roles = {o["role"]: o for o in job_record["outputs"]}
    assert set(roles) == set(rule.outputs), f"{job_record['job_type']} {job_record['id']}: outputs {sorted(roles)} != contract {sorted(rule.outputs)}"
    parsed: Dict[str, Any] = {}
    for role, output in roles.items():
        meta = artifact(stack, output["artifact_id"])
        assert meta["kind"] == rule.outputs[role].value, (role, meta["kind"])
        assert meta["linked"] is True, meta
        assert meta["checksum"] == output["checksum"], (meta["checksum"], output["checksum"])
        raw = artifact_content(stack, output["artifact_id"])
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == output["checksum"], f"{role}: content does not hash to its checksum"
        model = content_model_for(ArtifactKind(meta["kind"]))
        parsed[role] = model.model_validate(json.loads(raw)) if model is not None else raw
    return parsed


# --- a second Process worker on the same Store ----------------------------------------------------


class SecondProcess:
    """``python -m call1.process serve`` with a copy of the stack's Process config (same Store, same
    installation and key), its own worker ID, port and data dir: a second worker racing the first."""

    def __init__(self, stack: Stack, name: str = "second", extra_env: Optional[Mapping[str, str]] = None) -> None:
        self.stack = stack
        self.name = name
        self.dir = stack.dir / f"process-{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.port = free_port()
        config = json.loads(stack.process_config_path.read_text(encoding="utf-8"))
        config.update(port=self.port, data_dir=str(self.dir / "data"))
        fd = os.open(self.config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        env = stack.process_env()
        env.update({"CALL1_PROCESS_CONFIG": str(self.config_path), "CALL1_PROCESS_PORT": str(self.port),
                    "CALL1_PROCESS_DATA": str(self.dir / "data"), "CALL1_PROCESS_WORKER_ID": f"e2e-{name}"})
        env.update(extra_env or {})
        self.worker_id = env["CALL1_PROCESS_WORKER_ID"]
        self.server = _Server(f"process-{name}", [python_executable(), "-m", "call1.process", "serve", "--log-level", "info"], env,
                              stack.dir, stack.logs_dir / f"process-{name}.log")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "SecondProcess":
        self.server.start()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                r = httpx.get(self.url + "/process/api/health", timeout=1.0)
                if r.status_code == 200 and r.json().get("state") == "running":
                    return self
            except httpx.HTTPError:
                pass
            if not self.server.running:
                break
            time.sleep(0.1)
        raise StackError(f"second Process did not reach running:\n{self.stack.log_tail(f'process-{self.name}')}")

    def stop(self) -> None:
        self.server.stop()


# --- host commands -------------------------------------------------------------------------------


def clean_env(**extra: str) -> Dict[str, str]:
    """The inherited environment without any CALL1_* variable (as the harness does), plus ``extra``."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("CALL1_")}
    env.update({"PYTHONPATH": str(REPO), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(extra)
    return env


def run_python(args: List[str], env: Mapping[str, str], cwd: Path, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run([python_executable(), *args], env=dict(env), cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def port_open(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0
