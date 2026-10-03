"""The pluggable trainer (docs/OnDeviceTraining.md section 4.3).

A trainer is only a subprocess command and a progress parser, so the runner owns the process: the
minimal offline environment, the deadline, cancel (``SIGTERM``, then ``SIGKILL`` after 10 s) and
the parsing of progress lines (numbers only).

* ``MlxLoraTrainer`` runs the proven recipe through ``python -m call1.process.training.mlx_lora`` (``mlx_lm
  lora`` with the chat template rendered thinking-off, as production renders it; check L2): ``--model <base> --train --data
  <work>/data --fine-tune-type lora --mask-prompt --num-layers 16 --batch-size 1 --learning-rate 1e-4
  --max-seq-length 2600 --grad-checkpoint --adapter-path <work>/candidate --iters N --seed S
  --val-batches 20``. Apple Silicon only; qualification L1-L4 is local-only.
* ``FakeTrainer`` runs ``python -m call1.process.training.fake_trainer``, which prints mlx_lm-shaped
  progress and writes a small adapter. It never imports MLX or torch; every test and fake-mode e2e
  run uses it. ``CALL1_FAKE_TRAINER_SECONDS`` and ``CALL1_FAKE_TRAINER_EXIT`` exercise slowness,
  cancel, timeout and crashes.
"""

from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
KILL_GRACE_SECONDS = 10.0
PASS_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER")
FAKE_ENV = ("CALL1_FAKE_TRAINER_SECONDS", "CALL1_FAKE_TRAINER_EXIT")


@dataclass(frozen=True)
class TrainSpec:
    base_model: Path
    data_dir: Path
    adapter_dir: Path
    iters: int
    seed: int
    max_seq_length: int = 2600
    num_layers: int = 16
    batch_size: int = 1
    learning_rate: float = 1e-4

    def record(self) -> Dict[str, object]:
        """The recipe as the manifest records it (no local paths beyond the base's name)."""
        data = asdict(self)
        data["base_model"] = Path(self.base_model).name
        data.pop("data_dir")
        data.pop("adapter_dir")
        return data


@dataclass
class TrainProgress:
    iteration: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    it_per_s: Optional[float] = None
    peak_memory_gb: Optional[float] = None


class Trainer(Protocol):
    name: str

    def argv(self, spec: TrainSpec) -> List[str]: ...

    def parse_progress(self, line: str) -> Optional[TrainProgress]: ...


_ITER = re.compile(r"Iter\s+(\d+):")
_TRAIN = re.compile(r"Train loss\s+([0-9.]+(?:e[-+]?\d+)?)", re.I)
_VAL = re.compile(r"Val loss\s+([0-9.]+(?:e[-+]?\d+)?)", re.I)
_ITS = re.compile(r"It/sec\s+([0-9.]+(?:e[-+]?\d+)?)", re.I)
_PEAK = re.compile(r"Peak mem\s+([0-9.]+)\s*GB", re.I)


def parse_mlx_progress(line: str) -> Optional[TrainProgress]:
    """``Iter N: Train loss ..., It/sec ..., Peak mem ... GB`` and ``Iter N: Val loss ...``: numbers
    only, never text."""
    found = _ITER.search(line)
    if found is None:
        return None
    progress = TrainProgress(iteration=int(found.group(1)))
    for pattern, name in ((_TRAIN, "train_loss"), (_VAL, "val_loss"), (_ITS, "it_per_s"), (_PEAK, "peak_memory_gb")):
        match = pattern.search(line)
        if match:
            try:
                setattr(progress, name, float(match.group(1)))
            except ValueError:
                pass
    return progress


class MlxLoraTrainer:
    name = "mlx_lm"

    def argv(self, spec: TrainSpec) -> List[str]:
        # the subprocess runs in the work directory, so every path is absolute
        return [sys.executable, "-m", "call1.process.training.mlx_lora", "--model", str(Path(spec.base_model).absolute()), "--train", "--data", str(spec.data_dir),
                "--fine-tune-type", "lora", "--mask-prompt", "--num-layers", str(spec.num_layers), "--batch-size", str(spec.batch_size),
                "--learning-rate", f"{spec.learning_rate:g}", "--max-seq-length", str(spec.max_seq_length), "--grad-checkpoint",
                "--adapter-path", str(spec.adapter_dir), "--iters", str(spec.iters), "--seed", str(spec.seed), "--val-batches", "20"]

    def parse_progress(self, line: str) -> Optional[TrainProgress]:
        return parse_mlx_progress(line)


class FakeTrainer:
    name = "fake"

    def argv(self, spec: TrainSpec) -> List[str]:
        return [sys.executable, "-m", "call1.process.training.fake_trainer", "--data", str(spec.data_dir), "--adapter-path", str(spec.adapter_dir),
                "--iters", str(spec.iters), "--seed", str(spec.seed)]

    def parse_progress(self, line: str) -> Optional[TrainProgress]:
        return parse_mlx_progress(line)


TRAINERS = {"mlx_lm": MlxLoraTrainer, "fake": FakeTrainer}


def trainer_for(name: str) -> Trainer:
    return TRAINERS[name]()


def seed_for(run_id: str) -> int:
    """A stable seed per run."""
    return int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8], 16) % 2_000_000_000


def iterations(n_train: int, *, budget_seconds: float, it_per_s: float) -> int:
    """``min(clamp(2 x n_train, 100, 1000), floor(budget_s x it_per_s))``, at least 1."""
    wanted = min(1000, max(100, 2 * n_train))
    return max(1, min(wanted, int(max(0.0, budget_seconds) * max(0.0, it_per_s))))


def subprocess_env(extra_names=()) -> Dict[str, str]:
    """The minimal offline environment: no Store credential and no ``CALL1_TEXT_ADAPTER``."""
    env = {name: os.environ[name] for name in PASS_ENV if name in os.environ}
    for name in extra_names:
        if name in os.environ:
            env[name] = os.environ[name]
    env.update(PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(REPO_ROOT))
    return env


@dataclass
class ProcessOutcome:
    exit_code: Optional[int]
    ended: str
    """``exited``, ``cancelled`` or ``timed_out``."""
    seconds: float


def run_process(argv: List[str], *, env: Dict[str, str], cwd: Path, on_line: Callable[[str], None], cancel: threading.Event,
                deadline: Optional[float], kill_grace: float = KILL_GRACE_SECONDS, started: Optional[Callable[[subprocess.Popen], None]] = None
                ) -> ProcessOutcome:
    """Run ``argv`` to its end, feeding each stdout/stderr line to ``on_line``. On ``cancel`` or at
    ``deadline`` (monotonic) the process gets ``SIGTERM``, then ``SIGKILL`` after ``kill_grace``."""
    began = time.monotonic()
    process = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               text=True, bufsize=1, start_new_session=True)
    if started is not None:
        started(process)

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                on_line(line.rstrip("\n"))
            except Exception:  # pragma: no cover - a parser bug never kills the run
                pass

    reader = threading.Thread(target=pump, name="call1-training-output", daemon=True)
    reader.start()
    ended = "exited"
    while process.poll() is None:
        if cancel.is_set():
            ended = "cancelled"
            break
        if deadline is not None and time.monotonic() >= deadline:
            ended = "timed_out"
            break
        cancel.wait(0.1)
    if process.poll() is None:
        _terminate(process, kill_grace)
    reader.join(timeout=2.0)
    return ProcessOutcome(exit_code=process.returncode, ended=ended, seconds=time.monotonic() - began)


def _terminate(process: subprocess.Popen, grace: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=5)


__all__ = ["FAKE_ENV", "FakeTrainer", "MlxLoraTrainer", "ProcessOutcome", "TrainProgress", "TrainSpec", "Trainer", "iterations",
           "parse_mlx_progress", "run_process", "seed_for", "subprocess_env", "trainer_for"]
