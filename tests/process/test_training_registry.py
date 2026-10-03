"""On-device training: the adapter registry (P9), ``text_adapter_path`` precedence (P10) and
per-job activation with provenance (P11), docs/OnDeviceTraining.md sections 4.5 and 5.1. No MLX,
torch or real model runs: generation is scripted."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from call1.adapters import mlx
from call1.contracts.catalog import ModelPurpose
from call1.contracts.jobs import JobParameters, JobType
from call1.process.catalog import seeded_catalog
from call1.process.handlers.real.llm import LlmTransport
from call1.process.handlers.signal_stages import carry_forwardable
from call1.process.training import registry as registry_mod
from call1.process.training.registry import AdapterRegistry, RegistryError, lora_suffix, task_for

from call1.process.handlers.fake import CANCEL_SCRIPT

from .test_signals_support import cancel_taxonomy, findings_for, make_job, run_v2, script_transcript
from .training_support import fake_base, qa_primary, script_for


def candidate(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "work" / name
    path.mkdir(parents=True)
    (path / "adapters.safetensors").write_bytes(b"fake")
    (path / "adapter_config.json").write_text("{}")
    return path


def promote(registry: AdapterRegistry, tmp_path: Path, version: str, tasks=("qa_verdict", "signal_stage1"), keep: int = 5) -> dict:
    manifest = {"version": version, "base": "gemma-4-e2b-it", "base_fingerprint": registry.fingerprint(), "tasks": list(tasks)}
    return registry.promote(candidate(tmp_path, version), manifest, keep_versions=keep)


@pytest.fixture
def registry(tmp_path):
    return AdapterRegistry(tmp_path / "adapters", base=fake_base(tmp_path))


@pytest.fixture(autouse=True)
def _reset_current():
    yield
    registry_mod.set_current(None)


# --- P9: the registry --------------------------------------------------------------------------


def test_promotion_replaces_the_pointer_atomically_and_records_the_previous(registry, tmp_path):
    first = promote(registry, tmp_path, "ft-20260927T090012Z")
    assert first["version"] == "ft-20260927T090012Z" and first["previous"] is None and first["tasks"] == ["qa_verdict", "signal_stage1"]
    second = promote(registry, tmp_path, "ft-20260928T090012Z")
    assert second["previous"] == "ft-20260927T090012Z"
    assert json.loads((registry.root / "active.json").read_text()) == second
    assert not list(registry.root.glob("*.tmp"))
    assert registry.manifest("ft-20260928T090012Z")["tasks"] == ["qa_verdict", "signal_stage1"]
    with pytest.raises(RegistryError):
        promote(registry, tmp_path / "again", "ft-20260928T090012Z")
    assert registry.new_version(registry_mod.datetime(2026, 9, 28, 9, 0, 12, tzinfo=registry_mod.timezone.utc)) == "ft-20260928T090013Z"


def test_rollback_to_the_previous_version_and_to_the_base_then_reactivate(registry, tmp_path):
    promote(registry, tmp_path, "ft-20260927T090012Z")
    promote(registry, tmp_path, "ft-20260928T090012Z")
    back = registry.activate("ft-20260927T090012Z")
    assert back["version"] == "ft-20260927T090012Z" and back["previous"] == "ft-20260928T090012Z"
    base = registry.activate(None)
    assert base["version"] is None and registry.active() is None and base["previous"] == "ft-20260927T090012Z"
    assert registry.resolve("qa_verdict", str(registry.base)) is None
    again = registry.activate("ft-20260928T090012Z")
    assert registry.active()["version"] == "ft-20260928T090012Z" and again["previous"] is None
    with pytest.raises(RegistryError) as err:
        registry.activate("ft-20990101T000000Z")
    assert err.value.code == "not_found"


def test_retention_keeps_the_newest_and_never_the_active_or_its_previous(registry, tmp_path):
    for day in range(1, 7):
        promote(registry, tmp_path, f"ft-202609{day:02d}T000000Z", keep=3)
    kept = [m["version"] for m in registry.versions()]
    assert kept == ["ft-20260906T000000Z", "ft-20260905T000000Z", "ft-20260904T000000Z"]
    registry.activate("ft-20260904T000000Z")
    promote(registry, tmp_path, "ft-20260907T000000Z", keep=2)  # previous is ft-...04
    kept = [m["version"] for m in registry.versions()]
    assert kept == ["ft-20260907T000000Z", "ft-20260906T000000Z", "ft-20260904T000000Z"]  # the newest two, plus the protected previous


def test_a_changed_base_sets_the_adapter_aside_and_refuses_activation(registry, tmp_path):
    promote(registry, tmp_path, "ft-20260927T090012Z")
    assert registry.resolve("qa_verdict", str(registry.base)) == (str(registry.root / "ft-20260927T090012Z"), "ft-20260927T090012Z")
    (registry.base / "model.safetensors").write_bytes(b"new weights, another size")
    registry_mod._fingerprints.clear()
    assert registry.stale()
    assert registry.resolve("qa_verdict", str(registry.base)) is None
    assert registry.set_aside_reason == "Adapter set aside: the base model changed; retrain"
    with pytest.raises(RegistryError) as err:
        registry.activate("ft-20260927T090012Z")
    assert err.value.code == "stale_base"
    assert registry.activate(None)["version"] is None  # the base is always allowed


def test_resolution_is_per_task_included_model_only_and_never_fails_a_job(registry, tmp_path):
    promote(registry, tmp_path, "ft-20260927T090012Z", tasks=("signal_stage1",))
    base = str(registry.base)
    assert registry.resolve("signal_stage1", base)[1] == "ft-20260927T090012Z"
    assert registry.resolve("qa_verdict", base) is None  # no held-out items for it: runs on the base
    assert registry.resolve(None, base) is None
    assert registry.resolve("signal_stage1", str(tmp_path / "models" / "gemma4-e4b")) is None  # a local pack
    (registry.root / "ft-20260927T090012Z" / "adapters.safetensors").unlink()
    assert registry.resolve("signal_stage1", base) is None  # missing: logged once, the base runs
    assert [task_for(t) for t in (JobType.CONTACT_SIGNALS_CATEGORIZE, JobType.CONTACT_SIGNALS_SUBCATEGORIZE, JobType.QA_CRITERION,
                                  JobType.QA_ESCALATION, JobType.SPEAKER_ATTRIBUTION, JobType.SUMMARY_SEGMENT, JobType.CONTACT_SIGNALS_EXTRACT)] == [
        "signal_stage1", "signal_stage2", "qa_verdict", "qa_verdict", "speaker_roles", None, None]


# --- P10: text_adapter_path ----------------------------------------------------------------------


def test_text_adapter_path_env_over_context_over_none(tmp_path, monkeypatch):
    monkeypatch.delenv("CALL1_TEXT_ADAPTER", raising=False)
    base = str(tmp_path / "models" / "gemma-4-e2b-it")
    assert mlx.text_adapter_path(base) is None  # legacy callers never set the context: unchanged
    with mlx.use_text_adapter("/adapters/ft-1"):
        assert mlx.text_adapter_path(base) == "/adapters/ft-1"
        assert mlx.text_adapter_path(str(tmp_path / "models" / "gemma4-e4b")) is None  # included model only
        assert mlx.text_adapter_path(None) is None
        env = tmp_path / "manual"
        env.mkdir()
        (env / "adapters.safetensors").write_bytes(b"x")
        monkeypatch.setenv("CALL1_TEXT_ADAPTER", str(env))
        assert mlx.text_adapter_path(base) == str(env)
        monkeypatch.setenv("CALL1_TEXT_ADAPTER", str(tmp_path / "empty"))
        with pytest.raises(RuntimeError):
            mlx.text_adapter_path(base)
        monkeypatch.delenv("CALL1_TEXT_ADAPTER")
    assert mlx.text_adapter_path(base) is None
    with mlx.use_text_adapter(None):
        assert mlx.text_adapter_path(base) is None


# --- P11: per-job activation and provenance ------------------------------------------------------


@pytest.fixture
def mlx_host(tmp_path, monkeypatch):
    """CALL1_BACKEND=mlx with a stand-in included model, so the transport resolves adapters."""
    base = fake_base(tmp_path)
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    monkeypatch.setenv("CALL1_MLX_TEXT_PATH", str(base))
    monkeypatch.delenv("CALL1_TEXT_ADAPTER", raising=False)
    registry = AdapterRegistry(tmp_path / "adapters", base=base)
    registry_mod.set_current(registry)
    return registry


def _qa_job(tmp_path):
    transcript = script_transcript(script_for(0))
    return transcript, findings_for(tmp_path, transcript)


def test_qa_provenance_carries_the_adapter_version(tmp_path, mlx_host):
    promote(mlx_host, tmp_path, "ft-20260927T090012Z", tasks=("qa_verdict",))
    seen = []
    real = mlx.text_adapter_path

    def capture(model, system, prompt, *a, text_model_path=None, **k):
        seen.append((text_model_path, real(text_model_path)))
        return json.dumps({"assessment": "ok", "verdict": "pass", "quote": "This call may be recorded for quality assurance."}), {}

    transcript, findings = _qa_job(tmp_path)
    with patch("call1.question_models.generate_text", capture):
        from call1.process.handlers.real.qa import RealQaCriterion
        job, _result, _prompts = qa_primary(tmp_path, transcript, findings)
        result = RealQaCriterion().run(job)
    assert seen[-1] == (str(mlx_host.base.resolve()), str(mlx_host.root / "ft-20260927T090012Z"))
    revision = result.model_revision
    assert revision is not None and revision.endswith("+lora.ft-20260927T090012Z")
    assert result.outputs["assessment"].content.attempt.model_revision == revision
    assert lora_suffix(revision) == "+lora.ft-20260927T090012Z"


def test_a_task_the_adapter_was_not_measured_on_runs_the_base_with_no_suffix(tmp_path, mlx_host):
    promote(mlx_host, tmp_path, "ft-20260927T090012Z", tasks=("signal_stage1",))
    transcript, findings = _qa_job(tmp_path)
    job, result, _ = qa_primary(tmp_path, transcript, findings)
    assert result.model_revision is None


def test_one_adapter_per_job_even_when_the_pointer_changes_mid_job(tmp_path, mlx_host):
    promote(mlx_host, tmp_path, "ft-20260927T090012Z", tasks=("qa_verdict",))
    transcript, findings = _qa_job(tmp_path)
    job, _result, _ = qa_primary(tmp_path, transcript, findings)
    transport = LlmTransport(job)
    used = []

    def capture(model, system, prompt, *a, text_model_path=None, **k):
        used.append(mlx.text_adapter_path(text_model_path))
        return "{}", {}

    with patch("call1.question_models.generate_text", capture):
        transport.generate("s", "p")
        mlx_host.activate(None)  # rolled back mid-job
        transport.generate("s", "p")
    assert used == [str(mlx_host.root / "ft-20260927T090012Z")] * 2
    assert LlmTransport(job).adapter_version is None  # the next job runs the base
    assert transport.model_revision().endswith("+lora.ft-20260927T090012Z")


def test_the_manual_env_override_records_lora_env(tmp_path, mlx_host, monkeypatch):
    manual = tmp_path / "manual"
    manual.mkdir()
    (manual / "adapters.safetensors").write_bytes(b"x")
    monkeypatch.setenv("CALL1_TEXT_ADAPTER", str(manual))
    transcript, findings = _qa_job(tmp_path)
    job, result, _ = qa_primary(tmp_path, transcript, findings)
    assert result.model_revision.endswith("+lora.env")


def test_the_signal_classifier_and_speaker_roles_record_their_adapter(tmp_path, mlx_host):
    from call1.process.handlers.real import media
    from call1.process.handlers.real.signals_v2 import GemmaSegmentClassifier

    promote(mlx_host, tmp_path, "ft-20260927T090012Z", tasks=("signal_stage1", "speaker_roles"))
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, {}, entry_id="call1-bundled", purpose=ModelPurpose.SIGNAL_CATEGORY,
                   catalog=seeded_catalog(mode="real"))
    assert GemmaSegmentClassifier(job).model_revision.endswith("+lora.ft-20260927T090012Z")
    job2 = make_job(tmp_path, JobType.CONTACT_SIGNALS_SUBCATEGORIZE, {}, entry_id="call1-bundled", purpose=ModelPurpose.SIGNAL_SUBCATEGORY,
                    catalog=seeded_catalog(mode="real"))
    assert GemmaSegmentClassifier(job2).model_revision is None  # stage 2 not measured: the base

    script = script_for(0)
    transcript = script_transcript(script, stereo=False)
    from call1.contracts.artifacts import ArtifactKind

    speakers = make_job(tmp_path, JobType.SPEAKER_ATTRIBUTION, {"transcript": (ArtifactKind.TRANSCRIPT, transcript),
                                                               "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript))},
                        entry_id=None, parameters=JobParameters())
    clusters = {i: ("spk_0" if role.value == "AGENT" else "spk_1") for i, (role, _) in enumerate(script)}
    adapters = []
    with patch("call1.question_models.generate_text", lambda *a, **k: (json.dumps({"assessment": "a", "roles": {"S1": "agent", "S2": "caller"}}), {})):
        roles = media.infer_roles(speakers, transcript, clusters, adapters)
    assert roles == {"spk_0": "agent", "spk_1": "caller"} and adapters == ["ft-20260927T090012Z"]


def test_carry_forward_is_refused_across_adapter_versions(tmp_path):
    from call1.process.handlers.fake import FakeSignalClassifier

    previous = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy())
    prov = previous.subcategories.provenance
    engine = FakeSignalClassifier(cancel_taxonomy(), 2)
    kw = {"template": prov.question_template, "adapter_version": prov.adapter_version}
    assert carry_forwardable(prov, engine, **kw)  # base and base
    with_a = prov.model_copy(update={"model_revision": prov.model_revision + "+lora.ft-a"})
    assert not carry_forwardable(with_a, engine, **kw)  # an adapter's answers are not the base's
    engine.model_revision = prov.model_revision + "+lora.ft-a"  # type: ignore[attr-defined]
    assert carry_forwardable(with_a, engine, **kw)
    engine.model_revision = prov.model_revision + "+lora.ft-b"  # type: ignore[attr-defined]
    assert not carry_forwardable(with_a, engine, **kw)
    assert not carry_forwardable(prov, engine, **kw)
