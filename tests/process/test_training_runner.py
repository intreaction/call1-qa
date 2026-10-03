"""On-device training: the runner end to end with the ``FakeTrainer`` subprocess and ``FakeGenerator``
(P8), no GPU in fake mode (P13) and privacy (P14), docs/OnDeviceTraining.md section 4.

Every outcome: promoted, rejected, timed out, cancelled, a trainer crash, an evaluator crash, the
minimums, Store failures and interrupted-run recovery. No MLX, torch or real model runs."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from call1.pipeline.signals_v2 import estimate_tokens
from call1.process.store_client import StoreError, StoreUnavailable
from call1.process.training import runner as runner_mod
from call1.process.training.generate import FakeGenerator
from call1.process.training.runner import RunContext, Runner, new_record
from call1.process.training.trainer import FakeTrainer, MlxLoraTrainer, TrainSpec, parse_mlx_progress

from .training_support import INSTALLATION, FakeStore, fake_base, loose_settings, registry_for, world

HEAVY = ("mlx", "mlx_lm", "torch")


@pytest.fixture
def setup(tmp_path):
    store = FakeStore()
    world(store, tmp_path)
    base = fake_base(tmp_path)
    return SimpleNamespace(store=store, base=base, registry=registry_for(tmp_path, base), work=tmp_path / "process" / "training" / "work")


def run(setup, *, outcome="promote", settings=None, deadline=600.0, cancel=None, hook=None, run_id="tr-test"):
    def factory(data, runner):
        if hook is not None:
            hook(data, runner)
        return FakeGenerator(data.prompts, data.items, outcome)

    ctx = RunContext(client=setup.store, registry=setup.registry, settings=settings or loose_settings(), trainer="fake", base_model=setup.base,
                     installation_id=INSTALLATION, work_root=setup.work, count_tokens=estimate_tokens, generator_factory=factory, kill_grace=1.0)
    runner = Runner(new_record(run_id, "manual"), ctx, cancel or threading.Event())
    return runner.execute(time.monotonic() + deadline), runner


def test_a_promoted_run_moves_the_candidate_writes_the_manifest_and_the_pointer(setup):
    record, runner = run(setup)
    assert record["status"] == "promoted" and record["reason"].startswith("promoted: overall 1.000 ≥ 0.")
    version = record["candidate_version"]
    assert record["active_before"] is None and record["active_after"] == version
    pointer = setup.registry.active()
    assert pointer["version"] == version and pointer["previous"] is None
    assert pointer["tasks"] == ["qa_verdict", "signal_stage1", "signal_stage2", "speaker_roles"]
    manifest = setup.registry.manifest(version)
    assert manifest["base"] == "gemma-4-e2b-it" and manifest["base_fingerprint"].startswith("sha256:")
    assert manifest["recipe"]["max_seq_length"] == 2600 and manifest["recipe"]["num_layers"] == 16 and manifest["recipe"]["base_model"] == "gemma-4-e2b-it"
    assert manifest["label_cursor"] == record["label_cursor"]["to"] == len(setup.store.labels)
    assert manifest["eval"]["overall"]["candidate"] == 1.0 and manifest["decision"] == record["reason"]
    adapter = setup.registry.root / version
    assert (adapter / "adapters.safetensors").is_file() and (adapter / "adapter_config.json").is_file()
    assert stat.S_IMODE(os.stat(adapter).st_mode) == 0o700 and stat.S_IMODE(os.stat(setup.registry.root).st_mode) == 0o700
    assert not (setup.work / "tr-test").exists()
    assert record["labels"] == {"qa_verdict": 4, "signal_hit": 20, "speaker_role": 4, "withdrawn": 0}
    assert record["examples"]["train"] > 0 and record["examples"]["eval_items"] > 0
    assert record["trainer"]["iters"] >= 1 and record["trainer"]["train_loss"] is not None and record["trainer"]["peak_memory_gb"] == 1.25
    assert record["dataset_digest"].startswith("sha256:")


def test_a_rejected_run_keeps_the_pointer_deletes_the_weights_and_records_why(setup):
    first, _ = run(setup)
    record, _ = run(setup, outcome="reject", run_id="tr-reject")
    assert record["status"] == "rejected" and record["reason"].startswith("rejected: ")
    assert "fell" in record["reason"] or "overall" in record["reason"]
    assert setup.registry.active()["version"] == first["candidate_version"] == record["active_after"]
    assert [m["version"] for m in setup.registry.versions()] == [first["candidate_version"]]
    assert record["eval"]["overall"]["candidate"] < record["eval"]["overall"]["active"]
    assert not (setup.work / "tr-reject").exists()


def test_invalid_candidate_answers_reject(setup):
    record, _ = run(setup, outcome="invalid")
    assert record["status"] == "rejected" and record["reason"].startswith("rejected: ")
    assert setup.registry.active() is None


def test_the_baseline_pass_answers_each_task_on_what_production_uses(setup):
    """The active adapter answers only its own tasks' held-out prompts; the rest run on the base,
    as ``AdapterRegistry.resolve`` routes production jobs."""
    first, _ = run(setup, run_id="tr-1")
    version = first["candidate_version"]
    pointer = json.loads(setup.registry.active_path.read_text())
    pointer["tasks"] = ["qa_verdict"]  # an adapter measured on QA only
    setup.registry.active_path.write_text(json.dumps(pointer))
    calls = []

    class Recording(FakeGenerator):
        def answer(self, which, adapter, prompts_path, out_path, *, cancel, deadline):
            ids = [json.loads(line)["id"] for line in Path(prompts_path).read_text().splitlines() if line.strip()]
            calls.append((which, adapter.name if adapter is not None else None, {self.prompts[i].task for i in ids}))
            return super().answer(which, adapter, prompts_path, out_path, cancel=cancel, deadline=deadline)

    ctx = RunContext(client=setup.store, registry=setup.registry, settings=loose_settings(), trainer="fake", base_model=setup.base,
                     installation_id=INSTALLATION, work_root=setup.work, count_tokens=estimate_tokens,
                     generator_factory=lambda data, runner: Recording(data.prompts, data.items, "promote"), kill_grace=1.0)
    record = Runner(new_record("tr-2", "manual"), ctx, threading.Event()).execute(time.monotonic() + 600)
    assert record["status"] == "promoted"
    active_calls = [c for c in calls if c[0] == "active"]
    assert ("active", version, {"qa_verdict"}) in active_calls
    base_call = next(c for c in active_calls if c[1] is None)
    assert "qa_verdict" not in base_call[2] and base_call[2]
    assert [c[0] for c in calls].count("candidate") == 1


def test_a_second_promotion_records_the_previous_version(setup):
    first, _ = run(setup, run_id="tr-1")
    second, _ = run(setup, run_id="tr-2")
    assert second["status"] == "promoted" and second["candidate_version"] != first["candidate_version"]
    assert setup.registry.active()["previous"] == first["candidate_version"]


def test_the_maximum_duration_kills_the_trainer(setup, monkeypatch):
    monkeypatch.setenv("CALL1_FAKE_TRAINER_SECONDS", "20")
    started = time.monotonic()
    record, _ = run(setup, deadline=2.0)
    assert record["status"] == "timed_out" and time.monotonic() - started < 10
    assert setup.registry.active() is None and not (setup.work / "tr-test").exists()


def test_cancel_terminates_the_trainer_and_leaves_the_pointer(setup, monkeypatch):
    monkeypatch.setenv("CALL1_FAKE_TRAINER_SECONDS", "20")
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    started = time.monotonic()
    record, _ = run(setup, cancel=cancel)
    assert record["status"] == "cancelled" and time.monotonic() - started < 10
    assert setup.registry.active() is None and not (setup.work / "tr-test").exists()


def test_a_trainer_crash_fails_with_its_exit_code_and_peak_memory(setup, monkeypatch):
    monkeypatch.setenv("CALL1_FAKE_TRAINER_EXIT", "3")
    record, _ = run(setup)
    assert record["status"] == "failed" and record["reason"] == "trainer_exit 3 (peak memory 1.2 GB)"
    assert setup.registry.active() is None and setup.registry.versions() == []


def test_an_evaluator_crash_fails_and_discards_the_candidate(setup):
    record, _ = run(setup, outcome="crash")
    assert record["status"] == "failed" and record["reason"] == "evaluator_exit"
    assert setup.registry.active() is None and setup.registry.versions() == [] and not (setup.work / "tr-test").exists()


def test_unmet_minimums_skip_without_touching_the_gpu(setup, monkeypatch):
    launched = []
    monkeypatch.setattr(runner_mod, "run_process", lambda *a, **k: launched.append(a) or pytest.fail("no subprocess"))
    record, _ = run(setup, settings=loose_settings(min_train_examples=1000))
    assert record["status"] == "skipped" and "training examples; at least 1000" in record["reason"]
    assert launched == [] and record["trainer"]["iters"] is None
    record, _ = run(setup, settings=loose_settings(min_eval_items=1000), run_id="tr-2")
    assert record["status"] == "skipped" and "held-out items" in record["reason"]
    record, _ = run(setup, settings=loose_settings(min_labeled_calls=1000), run_id="tr-3")
    assert record["status"] == "skipped" and "labelled calls" in record["reason"]


def test_too_little_disk_skips(setup):
    ctx_bytes = 10 ** 18

    def factory(data, runner):  # pragma: no cover - never reached
        raise AssertionError

    ctx = RunContext(client=setup.store, registry=setup.registry, settings=loose_settings(), trainer="fake", base_model=setup.base,
                     installation_id=INSTALLATION, work_root=setup.work, count_tokens=estimate_tokens, generator_factory=factory,
                     min_free_bytes=ctx_bytes)
    record = Runner(new_record("tr-disk", "manual"), ctx, threading.Event()).execute(time.monotonic() + 60)
    assert record["status"] == "skipped" and record["reason"].startswith("disk")


@pytest.mark.parametrize("error, reason", [(StoreUnavailable("down"), "store_unreachable"),
                                           (StoreError("insufficient_scope", "no", status=403), "insufficient_scope")])
def test_store_failures_fail_the_run(setup, error, reason):
    setup.store.fail_labels = error
    record, _ = run(setup)
    assert record["status"] == "failed" and record["reason"] == reason
    assert setup.registry.active() is None


def test_interrupted_runs_are_recorded_and_their_work_deleted(tmp_path):
    from call1.process.training.scheduler import TrainingService

    base = fake_base(tmp_path)
    config = SimpleNamespace(data_dir=tmp_path / "process", config_path=tmp_path / "process" / "config.json", configured=True, handlers="fake",
                             installation_id="inst", training={})
    root = tmp_path / "process" / "training"
    (root / "work" / "tr-old" / "candidate").mkdir(parents=True)
    (root / "work" / "tr-old" / "candidate" / "adapters.safetensors").write_bytes(b"x")
    root.joinpath("state.json").write_text(json.dumps({"active_run": new_record("tr-old", "schedule"), "cursor": 4}))
    service = TrainingService(config, client=FakeStore(), registry=registry_for(tmp_path, base), base_model=base)
    assert service.history() == [] and (root / "work").exists()  # constructing never recovers
    assert service.claim_ownership()
    history = service.history()
    assert history[0]["run_id"] == "tr-old" and history[0]["status"] == "interrupted"
    assert not (root / "work").exists() and service.state()["active_run"] is None and service.state()["cursor"] == 4
    assert service.registry.active() is None
    service.release_ownership()


def test_a_second_service_never_recovers_or_runs_while_the_owner_holds_the_lock(tmp_path):
    """A CLI command (status, drain, training status/run) beside a running serve: the serve's run in
    progress, its state and its work directory are untouched, and the CLI cannot start a run."""
    from call1.process.training.scheduler import TrainingService, TrainingUnavailable

    base = fake_base(tmp_path)
    config = SimpleNamespace(data_dir=tmp_path / "process", config_path=tmp_path / "process" / "config.json", configured=True, handlers="fake",
                             installation_id="inst", training={})
    root = tmp_path / "process" / "training"
    serve = TrainingService(config, client=FakeStore(), registry=registry_for(tmp_path, base), base_model=base)
    assert serve.claim_ownership()
    (root / "work" / "tr-live" / "candidate").mkdir(parents=True)
    root.joinpath("state.json").write_text(json.dumps({"active_run": new_record("tr-live", "manual"), "cursor": 2}))

    cli = TrainingService(config, client=FakeStore(), registry=registry_for(tmp_path, base), base_model=base)
    assert not cli.claim_ownership() and not cli.owned
    assert cli.recover() is None
    assert (root / "work" / "tr-live" / "candidate").is_dir()
    assert cli.state()["active_run"]["run_id"] == "tr-live" and cli.history() == []
    status = cli.describe()["status"]
    assert status["run_id"] == "tr-live" and "another Process" in status["detail"]
    with pytest.raises(TrainingUnavailable, match="owns on-device training"):
        cli.request_run("manual")

    serve.release_ownership()  # the serve stopped: now the CLI may own it (and recovers the stale run)
    assert cli.claim_ownership()
    assert cli.history()[0]["status"] == "interrupted" and not (root / "work").exists()
    cli.release_ownership()


# --- P13: no GPU in fake mode ------------------------------------------------------------------


def test_a_full_fake_run_imports_no_mlx_or_torch_and_runs_no_mlx_lm(setup, monkeypatch):
    argvs = []
    original = runner_mod.run_process

    def recording(argv, **kwargs):
        argvs.append(list(argv))
        return original(argv, **kwargs)

    monkeypatch.setattr(runner_mod, "run_process", recording)
    before = {name for name in sys.modules if name.split(".")[0] in HEAVY}
    record, _ = run(setup)
    after = {name for name in sys.modules if name.split(".")[0] in HEAVY}
    assert record["status"] == "promoted"
    assert after == before  # nothing new was imported by the run
    assert argvs and all("mlx_lm" not in " ".join(a) for a in argvs)
    assert argvs[0][1:3] == ["-m", "call1.process.training.fake_trainer"]


def test_the_trainers_argv_and_progress_parsing(tmp_path):
    spec = TrainSpec(base_model=tmp_path / "gemma-4-e2b-it", data_dir=tmp_path / "data", adapter_dir=tmp_path / "candidate", iters=320, seed=7)
    argv = MlxLoraTrainer().argv(spec)
    assert argv[1:3] == ["-m", "call1.process.training.mlx_lora"]
    joined = " ".join(argv)
    for flag in ("--train", "--fine-tune-type lora", "--mask-prompt", "--num-layers 16", "--batch-size 1", "--learning-rate 0.0001",
                 "--max-seq-length 2600", "--grad-checkpoint", "--iters 320", "--seed 7", "--val-batches 20"):
        assert flag in joined
    assert "mlx_lm" not in " ".join(FakeTrainer().argv(spec))
    line = "Iter 320: Train loss 0.312, Learning Rate 1.000e-04, It/sec 0.215, Tokens/sec 480.1, Trained Tokens 99, Peak mem 9.874 GB"
    progress = parse_mlx_progress(line)
    assert (progress.iteration, progress.train_loss, progress.it_per_s, progress.peak_memory_gb) == (320, 0.312, 0.215, 9.874)
    assert parse_mlx_progress("Iter 1: Val loss 2.890, Val took 12.3s").val_loss == 2.89
    assert parse_mlx_progress("Loading pretrained model") is None


def test_the_trainer_subprocess_gets_a_minimal_offline_environment(setup, monkeypatch):
    monkeypatch.setenv("CALL1_TEXT_ADAPTER", "/somewhere")
    monkeypatch.setenv("CALL1_PROCESS_SERVICE_KEY", "c1sk_secret")
    envs = []
    original = runner_mod.run_process

    def recording(argv, **kwargs):
        envs.append(dict(kwargs["env"]))
        return original(argv, **kwargs)

    monkeypatch.setattr(runner_mod, "run_process", recording)
    run(setup)
    env = envs[0]
    assert env["PYTHONUNBUFFERED"] == "1" and env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert "CALL1_TEXT_ADAPTER" not in env and not any("c1sk_" in value for value in env.values())


# --- P14: privacy ------------------------------------------------------------------------------


def test_work_is_private_raw_inputs_go_first_and_the_runner_writes_nothing_to_store(setup):
    seen = {}

    def hook(data, runner):
        work = runner.work
        seen["mode"] = stat.S_IMODE(os.stat(work).st_mode)
        seen["inputs"] = (work / "inputs").exists()
        seen["files"] = sorted(p.name for p in (work / "data").iterdir())
        seen["train"] = (work / "data" / "train.jsonl").read_text()
        seen["index"] = [json.loads(line) for line in (work / "data" / "index.jsonl").read_text().splitlines()]

    record, _ = run(setup, hook=hook)
    assert record["status"] == "promoted"
    assert seen["mode"] == 0o700 and seen["inputs"] is False
    assert seen["files"] == ["index.jsonl", "train.jsonl", "valid.jsonl"]
    for line in seen["train"].splitlines():
        row = json.loads(line)
        assert list(row) == ["messages"] and [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert {"split", "task", "label_seqs", "call", "digest", "tokens"} == set(seen["index"][0])
    assert "Maria" not in seen["train"] and "call_0" not in json.dumps(seen["index"])
    assert setup.store.writes == []
    assert not setup.work.joinpath("tr-test").exists()
    text = json.dumps(record)
    assert "Maria" not in text and "[REDACTED]" not in text  # the record keeps counts, never text


def test_the_adapter_scope_holds_back_a_small_task_the_candidate_got_worse_on():
    """Qualification L1: QA fell 6/6 -> 1/6 on six held-out items, under the per-task drop rule's
    10-item floor, while the signal stages improved. The candidate is promoted, but QA stays on the
    base; a task joins the scope only when the candidate is not worse on it."""
    from call1.process.training.evaluate import Score, Scores, adapter_scope, decide

    active = Scores(tasks={"qa_verdict": Score(n=6, correct=6), "signal_stage1": Score(n=85, correct=28), "signal_stage2": Score(n=120, correct=55)})
    candidate = Scores(tasks={"qa_verdict": Score(n=6, correct=1), "signal_stage1": Score(n=85, correct=65), "signal_stage2": Score(n=120, correct=70)})
    assert decide(active, candidate, min_eval_items=20)[0] is True
    assert adapter_scope(active, candidate) == (["signal_stage1", "signal_stage2"], ["qa_verdict"])
    tie = Scores(tasks={"qa_verdict": Score(n=6, correct=6, invalid=0)})
    assert adapter_scope(tie, tie) == (["qa_verdict"], [])  # ties join, as in the promotion rule
    more_invalid = Scores(tasks={"qa_verdict": Score(n=6, correct=6, invalid=1)})
    assert adapter_scope(tie, more_invalid) == ([], ["qa_verdict"])


def test_the_real_trainer_argv_uses_absolute_paths_because_it_runs_in_the_work_directory(tmp_path, monkeypatch):
    """Qualification L1: a relative data/models base made mlx_lm treat the path as a Hub repo ID
    (HFValidationError) once the subprocess ran with the work directory as its cwd."""
    monkeypatch.chdir(tmp_path)
    spec = TrainSpec(base_model=Path("data/models/gemma-4-e2b-it"), data_dir=tmp_path / "data", adapter_dir=tmp_path / "candidate", iters=1, seed=1)
    argv = MlxLoraTrainer().argv(spec)
    assert argv[argv.index("--model") + 1] == str(tmp_path / "data" / "models" / "gemma-4-e2b-it")
    from call1.process.training.generate import SubprocessGenerator

    assert Path(SubprocessGenerator(Path("data/models/gemma-4-e2b-it"), tmp_path).argv(None, tmp_path / "p", tmp_path / "o")[4]).is_absolute()


def test_a_failed_trainer_reason_names_the_exception_class_only():
    pattern = runner_mod._EXCEPTION_LINE
    line = "huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name': 'data/models/x'"
    assert pattern.match(line).group(1).rsplit(".", 1)[-1] == "HFValidationError"
    assert pattern.match("KeyboardInterrupt").group(1) == "KeyboardInterrupt"
    assert pattern.match("Iter 3: Train loss 0.5") is None
    assert pattern.match("  File \"x.py\", line 1, in <module>") is None
