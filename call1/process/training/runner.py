"""One on-device training run, from collecting labels to the decision (docs/OnDeviceTraining.md
section 4).

The scheduler (``scheduler.TrainingService``) owns ``waiting`` and ``pausing``: the start rule and
the claim pause. It then calls ``Runner.execute`` with claims paused and ``inference_lock`` held;
the runner owns the rest:

``collecting`` -> ``building`` -> ``training`` -> ``evaluating`` -> ``deciding`` -> ``promoted`` |
``rejected`` | ``skipped`` (never touches the GPU) | ``failed`` | ``cancelled`` | ``timed_out``.

``work/<run_id>/`` (0700) holds the downloaded inputs (deleted as soon as the masked examples are
written), the dataset, the eval prompts and answers, and the candidate. It is deleted at the end of
every run, whatever the outcome; a promoted candidate is moved into the registry first. The runner
never writes to Store: Store sees only the label reads (audited), the job and artifact reads, and
later the adapter version in attempt provenance.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple

from . import dataset as datasets
from .examples import BuilderBug, ExampleBuilder
from .generate import EvaluationStopped, EvaluatorFailed, Generator
from .labels import collect
from .evaluate import adapter_scope, decide, metrics, score
from .registry import AdapterRegistry, adapter_complete, base_fingerprint, private_dir, utcnow
from .replay import Sources
from .settings import TrainingSettings
from .trainer import FAKE_ENV, TrainSpec, iterations, run_process, seed_for, subprocess_env, trainer_for

log = logging.getLogger("call1.process.training.runner")

MIN_FREE_BYTES = 2 * 1024 ** 3
EVAL_RESERVE_SECONDS = 300.0
DEFAULT_IT_PER_S = 0.15
"""Measured (qualification L4, 2026-09-26, two real runs): Gemma 4 E2B on an M3 Pro 18 GB, max_seq_length
2600, 324 iterations in 2136 s and 2194 s (0.152 and 0.148 it/s), mlx_lm's validation passes included."""
DEFAULT_S_PER_PROMPT = 23.0
"""Measured (L4): 20.9 and 24.5 s per production-shaped held-out prompt (stage-1 packs of about 7,500
tokens), averaged over both passes with the model loads."""
_EXCEPTION_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt))(?::|$)")
"""The last line of a Python traceback; only the class name is kept, so a run record never holds text."""
TERMINAL = ("promoted", "rejected", "skipped", "failed", "cancelled", "timed_out", "interrupted")


class RunStopped(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class RunContext:
    """What a run needs from the Process around it."""

    client: Any
    registry: AdapterRegistry
    settings: TrainingSettings
    trainer: str
    base_model: Path
    installation_id: str
    work_root: Path
    count_tokens: Callable[[str], int]
    measurements: Dict[str, float] = field(default_factory=dict)
    """``it_per_s`` and ``s_per_prompt`` from the last run (``state.json``)."""
    generator_factory: Optional[Callable[[datasets.Dataset, "Runner"], Generator]] = None
    catalog: Any = None
    kill_grace: float = 10.0
    min_free_bytes: int = MIN_FREE_BYTES


def new_record(run_id: str, trigger: str) -> Dict[str, Any]:
    """A run record (section 6.1): counts, metrics and reasons only, never text."""
    return {"run_id": run_id, "trigger": trigger, "requested_at": utcnow().isoformat(), "started_at": None, "ended_at": None,
            "status": "waiting", "reason": None, "label_cursor": {"from": None, "to": None},
            "labels": {"qa_verdict": 0, "signal_hit": 0, "speaker_role": 0, "withdrawn": 0},
            "examples": {"train": 0, "valid": 0, "eval_items": 0, "by_task": {}}, "skipped": {},
            "trainer": {"iters": None, "it_per_s": None, "train_loss": None, "val_loss": None, "peak_memory_gb": None, "seconds": None},
            "eval": {"tasks": {}, "overall": {"n": 0, "active": None, "candidate": None}}, "candidate_version": None, "active_before": None,
            "active_after": None, "dataset_digest": None, "notes": {}}


class Runner:
    def __init__(self, record: Dict[str, Any], ctx: RunContext, cancel: threading.Event, *, label_cursor_from: int = 0) -> None:
        self.record = record
        self.ctx = ctx
        self.cancel = cancel
        self.run_id = record["run_id"]
        self.work = (Path(ctx.work_root) / self.run_id).absolute()  # subprocesses run with this as cwd
        self.phase = "collecting"
        self.detail = ""
        self.progress: Dict[str, Any] = {"iteration": None, "iterations": None, "train_loss": None, "val_loss": None}
        self.record["label_cursor"]["from"] = label_cursor_from
        self.process = None  # the running subprocess, for the console and tests

    # --- the phases --------------------------------------------------------------------------

    def execute(self, deadline: float) -> Dict[str, Any]:
        """Run the phases until a terminal status. ``deadline`` is ``time.monotonic()`` at which the
        run is ``timed_out`` (``max_duration`` after the claim pause began)."""
        self.record["started_at"] = self.record["started_at"] or utcnow().isoformat()
        try:
            self._execute(deadline)
        except RunStopped as stop:
            self._end(stop.status, stop.reason)
        except BuilderBug as exc:
            log.exception("training run %s: example builder bug", self.run_id)
            self._end("failed", f"builder_bug: {str(exc)[:200]}")
        except Exception as exc:  # never let a run take Process down
            log.exception("training run %s failed", self.run_id)
            self._end("failed", f"error: {type(exc).__name__}")
        finally:
            shutil.rmtree(self.work, ignore_errors=True)
        return self.record

    def _check(self, deadline: float) -> None:
        if self.cancel.is_set():
            raise RunStopped("cancelled", "cancelled")
        if time.monotonic() >= deadline:
            raise RunStopped("timed_out", "the maximum duration was reached")

    def _set(self, phase: str, detail: str = "") -> None:
        self.phase = phase
        self.detail = detail
        self.record["status"] = phase

    def _execute(self, deadline: float) -> None:
        from ..store_client import StoreError, StoreUnavailable

        ctx, settings = self.ctx, self.ctx.settings
        private_dir(self.work)
        free = shutil.disk_usage(self.work).free
        if free < ctx.min_free_bytes:
            raise RunStopped("skipped", "disk: less than 2 GB free")
        active = ctx.registry.active()
        self.record["active_before"] = active.get("version") if active else None
        self.record["active_after"] = self.record["active_before"]

        # 1. collect
        self._set("collecting", "Reading reviewer labels")
        try:
            collected = collect(ctx.client, cancelled=self.cancel.is_set)
        except StoreUnavailable:
            raise RunStopped("failed", "store_unreachable") from None
        except StoreError as exc:
            raise RunStopped("failed", "insufficient_scope" if exc.code == "insufficient_scope" else f"store_error: {exc.code}") from None
        self._check(deadline)
        self.record["label_cursor"]["to"] = collected.cursor
        self.record["labels"] = collected.counts

        # 2. build
        self._set("building", f"Rebuilding {len(collected.current)} labelled prompts")
        taxonomy = None
        if any(label.kind.value == "signal_hit" for label in collected.current):
            try:
                taxonomy = ctx.client.get_signal_taxonomy().current.taxonomy
            except StoreUnavailable:
                raise RunStopped("failed", "store_unreachable") from None
            except StoreError as exc:
                log.warning("training run %s: taxonomy not readable (%s); signal labels count as outdated", self.run_id, exc.code)
        sources = Sources(ctx.client, self.work / "inputs", catalog=ctx.catalog)
        builder = ExampleBuilder(sources, installation_id=ctx.installation_id, count_tokens=ctx.count_tokens,
                                 max_seq_length=settings.max_seq_length, taxonomy=taxonomy)
        try:
            built = builder.build(collected.current, cancelled=self.cancel.is_set)
        except StoreUnavailable:
            raise RunStopped("failed", "store_unreachable") from None
        except StoreError as exc:
            raise RunStopped("failed", "insufficient_scope" if exc.code == "insufficient_scope" else f"store_error: {exc.code}") from None
        finally:
            shutil.rmtree(self.work / "inputs", ignore_errors=True)  # raw inputs never outlive the masked examples
        self._check(deadline)
        data = datasets.assemble(built, max_train_examples=settings.max_train_examples)
        counts = data.counts()
        self.record["examples"] = {"train": counts["train"], "valid": counts["valid"], "eval_items": counts["eval_items"],
                                   "by_task": counts["by_task"], "eval_by_task": counts["eval_by_task"], "labeled_calls": counts["labeled_calls"],
                                   "duplicates": counts["duplicates"]}
        self.record["skipped"] = dict(data.skipped)
        self.record["notes"] = dict(data.notes)
        self.record["dataset_digest"] = data.digest
        unmet = datasets.unmet_minimum(data, min_labeled_calls=settings.min_labeled_calls, min_train_examples=settings.min_train_examples,
                                       min_eval_items=settings.min_eval_items)
        if unmet:
            raise RunStopped("skipped", unmet)
        data_dir = self.work / "data"
        datasets.write(data, data_dir)
        prompts_path = datasets.write_eval_prompts(data, self.work / "eval-prompts.jsonl")

        # 3. train
        self._check(deadline)
        it_per_s = float(ctx.measurements.get("it_per_s") or DEFAULT_IT_PER_S)
        s_per_prompt = float(ctx.measurements.get("s_per_prompt") or DEFAULT_S_PER_PROMPT)
        budget = deadline - time.monotonic() - 2 * len(data.prompts) * s_per_prompt - EVAL_RESERVE_SECONDS
        iters = iterations(len(data.train), budget_seconds=budget, it_per_s=it_per_s)
        candidate = self.work / "candidate"
        spec = TrainSpec(base_model=ctx.base_model, data_dir=data_dir, adapter_dir=candidate, iters=iters, seed=seed_for(self.run_id),
                         max_seq_length=settings.max_seq_length)
        trainer = trainer_for(ctx.trainer)
        self.progress.update(iteration=0, iterations=iters)
        self.record["trainer"].update(iters=iters, name=trainer.name)
        self._set("training", f"Training: iteration 0 of {iters}")

        failure: Dict[str, str] = {}

        def on_line(line: str) -> None:
            error = _EXCEPTION_LINE.match(line)
            if error:
                failure["error"] = error.group(1).rsplit(".", 1)[-1]  # the exception's class name only, never its message
            progress = trainer.parse_progress(line)
            if progress is None:
                return
            if progress.iteration is not None:
                self.progress["iteration"] = progress.iteration
            if progress.train_loss is not None:
                self.progress["train_loss"] = progress.train_loss
                self.record["trainer"]["train_loss"] = progress.train_loss
            if progress.val_loss is not None:
                self.progress["val_loss"] = progress.val_loss
                self.record["trainer"]["val_loss"] = progress.val_loss
            if progress.peak_memory_gb is not None:
                self.record["trainer"]["peak_memory_gb"] = progress.peak_memory_gb
            loss = self.progress.get("train_loss")
            self.detail = f"Training: iteration {self.progress['iteration']} of {iters}" + (f", loss {loss:.2f}" if loss is not None else "")

        outcome = run_process(trainer.argv(spec), env=subprocess_env(FAKE_ENV if trainer.name == "fake" else ()), cwd=self.work,
                              on_line=on_line, cancel=self.cancel, deadline=deadline, kill_grace=ctx.kill_grace, started=self._started)
        self.process = None
        self.record["trainer"]["seconds"] = round(outcome.seconds, 3)
        if outcome.ended == "cancelled":
            raise RunStopped("cancelled", "cancelled")
        if outcome.ended == "timed_out":
            raise RunStopped("timed_out", "the maximum duration was reached while training")
        if outcome.exit_code != 0:
            peak = self.record["trainer"].get("peak_memory_gb")
            detail = [f"peak memory {peak:.1f} GB"] if peak else []
            if failure.get("error"):
                detail.append(failure["error"])
            raise RunStopped("failed", f"trainer_exit {outcome.exit_code}" + (f" ({', '.join(detail)})" if detail else ""))
        if not adapter_complete(candidate):
            raise RunStopped("failed", "trainer_output: no adapter was written")
        if outcome.seconds > 0:
            measured = max(0.01, min(50.0, iters / outcome.seconds))
            self.record["trainer"]["it_per_s"] = round(measured, 4)
            ctx.measurements["it_per_s"] = measured

        # 4. evaluate like production
        self._check(deadline)
        self._set("evaluating", f"Evaluating on {len(data.items)} held-out items")
        generator = ctx.generator_factory(data, self) if ctx.generator_factory is not None else self._subprocess_generator()
        active_path, active_tasks = self._active_adapter()
        started = time.monotonic()
        try:
            active_answers = self._answer_like_production(generator, data, prompts_path, active_path, active_tasks, deadline)
            self._check(deadline)
            candidate_answers = generator.answer("candidate", candidate, prompts_path, self.work / "answers-candidate.jsonl", cancel=self.cancel,
                                                 deadline=deadline)
        except EvaluatorFailed:
            raise RunStopped("failed", "evaluator_exit") from None
        except EvaluationStopped as stop:
            raise RunStopped(stop.ended, "cancelled" if stop.ended == "cancelled" else "the maximum duration was reached while evaluating") from None
        finally:
            self.process = None
        elapsed = time.monotonic() - started
        if data.prompts and elapsed > 0:
            ctx.measurements["s_per_prompt"] = max(0.01, elapsed / (2 * len(data.prompts)))
        self._check(deadline)

        # 5. decide
        self._set("deciding", "Comparing the candidate with the active adapter")
        active_scores = score(data.items, active_answers)
        candidate_scores = score(data.items, candidate_answers)
        self.record["eval"] = metrics(active_scores, candidate_scores)
        promote, reason = decide(active_scores, candidate_scores, min_eval_items=settings.min_eval_items)
        version = ctx.registry.new_version()
        self.record["candidate_version"] = version
        if not promote:
            shutil.rmtree(candidate, ignore_errors=True)  # only the metrics of a rejected candidate are kept
            raise RunStopped("rejected", reason)
        tasks, held_back = adapter_scope(active_scores, candidate_scores)
        if not tasks:
            shutil.rmtree(candidate, ignore_errors=True)
            raise RunStopped("rejected", "rejected: the candidate is worse than the active adapter on every task")
        if held_back:
            reason += "; on the base for " + ", ".join(
                f"{task} ({candidate_scores.tasks[task].accuracy:.2f} < {active_scores.tasks[task].accuracy:.2f}, n={candidate_scores.tasks[task].n})"
                for task in held_back)
        manifest = {"version": version, "created_at": utcnow().isoformat(), "base": Path(ctx.base_model).name,
                    "base_fingerprint": base_fingerprint(ctx.base_model), "recipe": spec.record(), "trainer": trainer.name, "tasks": tasks,
                    "label_cursor": collected.cursor, "dataset": {"counts": self.record["examples"], "digest": data.digest},
                    "eval": self.record["eval"], "decision": reason, "run_id": self.run_id}
        pointer = ctx.registry.promote(candidate, manifest, keep_versions=settings.keep_versions)
        self.record["active_after"] = pointer["version"]
        self._end("promoted", reason)

    def _started(self, process) -> None:
        self.process = process

    def _subprocess_generator(self) -> Generator:
        from .generate import SubprocessGenerator

        return SubprocessGenerator(self.ctx.base_model, self.work, on_start=self._started)

    def _active_adapter(self) -> Tuple[Optional[Path], Set[str]]:
        """(the active adapter's directory, the tasks production runs on it); (None, empty) when
        there is none or it is set aside, so the candidate is compared with the base."""
        active = self.ctx.registry.active()
        if active is None:
            return None, set()
        path = self.ctx.registry.root / str(active["version"])
        current = base_fingerprint(self.ctx.base_model)
        if not adapter_complete(path) or (current is not None and active.get("base_fingerprint") != current):
            return None, set()
        return path, set(active.get("tasks") or [])

    def _answer_like_production(self, generator: Generator, data: datasets.Dataset, prompts_path: Path, active_path: Optional[Path],
                                active_tasks: Set[str], deadline: float) -> Dict[str, Optional[str]]:
        """The baseline pass: each held-out prompt answered by what production uses for its task
        (``AdapterRegistry.resolve``): the active adapter for its ``tasks``, the base for the rest.
        A task that gets held-out items for the first time is therefore compared with the base."""
        out_path = self.work / "answers-active.jsonl"
        tasks = {prompt.task for prompt in data.prompts.values()}
        on_adapter = tasks & active_tasks if active_path is not None else set()
        if not on_adapter or on_adapter == tasks:
            return generator.answer("active", active_path if on_adapter else None, prompts_path, out_path, cancel=self.cancel, deadline=deadline)
        answers: Dict[str, Optional[str]] = {}
        for label, adapter, subset in (("adapter", active_path, on_adapter), ("base", None, tasks - on_adapter)):
            ids = {pid for pid, prompt in data.prompts.items() if prompt.task in subset}
            part_prompts = datasets.write_eval_prompts(data, self.work / f"eval-prompts-active-{label}.jsonl", tasks=subset)
            part = generator.answer("active", adapter, part_prompts, self.work / f"answers-active-{label}.jsonl", cancel=self.cancel,
                                    deadline=deadline)
            answers.update({pid: raw for pid, raw in part.items() if pid in ids})
            self._check(deadline)
        return answers

    def _end(self, status: str, reason: str) -> None:
        self.phase = status
        self.detail = reason
        self.record["status"] = status
        self.record["reason"] = reason
        self.record["ended_at"] = utcnow().isoformat()


__all__ = ["DEFAULT_IT_PER_S", "DEFAULT_S_PER_PROMPT", "MIN_FREE_BYTES", "RunContext", "RunStopped", "Runner", "TERMINAL", "new_record"]
