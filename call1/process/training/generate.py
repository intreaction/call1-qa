"""Answering the held-out prompts, once with the active adapter and once with the candidate
(docs/OnDeviceTraining.md section 4.4).

**The generation worker** is a subprocess: ``python -m call1.process.training.generate --base <path>
--adapter <path|none> --prompts eval-prompts.jsonl --out answers.jsonl``. It loads the model once
(``mlx_lm.load(base, adapter_path=...)``) and answers each ``{id, system, user, schema,
max_tokens}`` through ``call1.adapters.mlx.generate_loaded``, the helper ``MLXAdapter.generate``
uses, so the chat template (thinking off), the outlines JSON constraint and Gemma's final-answer
extraction are the same code as production. Apple Silicon only.

**``FakeGenerator``** answers in-process from the expected labels, with a per-adapter error rate.
``CALL1_FAKE_TRAINING_OUTCOMES`` (``promote,reject,invalid,crash``, one consumed per run) chooses
the result for tests and fake-mode e2e runs; it never imports MLX or torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .examples import EvalItem, EvalPrompt, answer_json
from .trainer import run_process, subprocess_env

FAKE_OUTCOMES = ("promote", "reject", "invalid", "crash")


class EvaluatorFailed(Exception):
    """The generation worker crashed or exited non-zero (``evaluator_exit``)."""


class EvaluationStopped(Exception):
    """Cancelled, or the run's deadline passed, while answering."""

    def __init__(self, ended: str) -> None:
        super().__init__(ended)
        self.ended = ended


class Generator(Protocol):
    name: str

    def answer(self, which: str, adapter: Optional[Path], prompts_path: Path, out_path: Path, *, cancel: threading.Event,
               deadline: Optional[float]) -> Dict[str, Optional[str]]: ...


class SubprocessGenerator:
    """The real generation worker, one subprocess per adapter."""

    name = "mlx_lm"

    def __init__(self, base: Path, cwd: Path, *, on_start=None) -> None:
        self.base = Path(base).absolute()  # the worker runs in the work directory
        self.cwd = Path(cwd)
        self.on_start = on_start

    def argv(self, adapter: Optional[Path], prompts_path: Path, out_path: Path) -> List[str]:
        return [sys.executable, "-m", "call1.process.training.generate", "--base", str(self.base), "--adapter",
                str(adapter) if adapter is not None else "none", "--prompts", str(prompts_path), "--out", str(out_path)]

    def answer(self, which: str, adapter: Optional[Path], prompts_path: Path, out_path: Path, *, cancel: threading.Event,
               deadline: Optional[float]) -> Dict[str, Optional[str]]:
        outcome = run_process(self.argv(adapter, prompts_path, out_path), env=subprocess_env(), cwd=self.cwd, on_line=lambda line: None,
                              cancel=cancel, deadline=deadline, started=self.on_start)
        if outcome.ended != "exited":
            raise EvaluationStopped(outcome.ended)
        if outcome.exit_code != 0:
            raise EvaluatorFailed(f"the generation worker exited {outcome.exit_code}")
        return read_answers(out_path)


def read_answers(path: Path) -> Dict[str, Optional[str]]:
    answers: Dict[str, Optional[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answers[str(row["id"])] = row.get("raw")
    return answers


class FakeGenerator:
    """Scripted answers built from the expected labels. ``error_rates`` maps ``active`` and
    ``candidate`` to the fraction of held-out items answered wrong (chosen deterministically);
    ``invalid`` makes the candidate's first prompt unparseable; ``crash`` fails the candidate."""

    name = "fake"

    def __init__(self, prompts: Dict[str, EvalPrompt], items: List[EvalItem], outcome: str = "promote") -> None:
        if outcome not in FAKE_OUTCOMES:
            raise ValueError(f"fake training outcome must be one of {', '.join(FAKE_OUTCOMES)}")
        self.prompts = prompts
        self.items = items
        self.outcome = outcome
        self.error_rates = {"promote": {"active": 0.25, "candidate": 0.0}, "reject": {"active": 0.0, "candidate": 0.5},
                            "invalid": {"active": 0.0, "candidate": 0.0}, "crash": {"active": 0.0, "candidate": 0.0}}[outcome]

    def answer(self, which: str, adapter: Optional[Path], prompts_path: Path, out_path: Path, *, cancel: threading.Event,
               deadline: Optional[float]) -> Dict[str, Optional[str]]:
        if cancel.is_set():
            raise EvaluationStopped("cancelled")
        if self.outcome == "crash" and which == "candidate":
            raise EvaluatorFailed("the fake generation worker crashed as asked")
        rate = self.error_rates.get(which, 0.0)
        ranked = sorted(self.items, key=lambda item: hashlib.sha256(f"{which}:{item.task}:{item.seq}:{item.prompt_ids}".encode()).hexdigest())
        wrong_count = 0 if rate <= 0 else max(1, int(round(rate * len(ranked))))
        wrong = {id(item) for item in ranked[:wrong_count]}
        drafts: Dict[str, Any] = {pid: prompt.draft() for pid, prompt in self.prompts.items()}
        for item in self.items:
            item.script(drafts, id(item) not in wrong)
        answers: Dict[str, Optional[str]] = {pid: answer_json(value) for pid, value in drafts.items()}
        if self.outcome == "invalid" and which == "candidate" and answers:
            first = sorted(answers)[0]
            answers[first] = "not json"
        Path(out_path).write_text("".join(json.dumps({"id": pid, "raw": raw}) + "\n" for pid, raw in answers.items()), encoding="utf-8")
        return answers


# --- the worker process (Apple Silicon only) -------------------------------------------------------


def main(argv=None) -> int:  # pragma: no cover - needs MLX and the model weights (local qualification L1)
    parser = argparse.ArgumentParser(prog="call1.process.training.generate")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    import gc

    import mlx.core as mx
    from mlx_lm import load

    from call1.adapters.mlx import generate_loaded

    adapter = None if args.adapter == "none" else args.adapter
    model, tokenizer = load(args.base, adapter_path=adapter) if adapter else load(args.base)
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as handle:
        for line in Path(args.prompts).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prompt = json.loads(line)
            try:
                raw: Optional[str] = generate_loaded(model, tokenizer, prompt["system"], prompt["user"], int(prompt["max_tokens"]), prompt.get("schema"))
            except Exception as exc:
                print(f"prompt {prompt['id']}: {type(exc).__name__}", flush=True)
                raw = None
            handle.write(json.dumps({"id": prompt["id"], "raw": raw}) + "\n")
            handle.flush()
            print(f"answered {prompt['id']}", flush=True)
    del model, tokenizer
    gc.collect()
    mx.clear_cache()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
