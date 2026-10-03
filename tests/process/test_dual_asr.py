"""Dual transcription on the Process side (contract 1.3.0, team decision 33, docs/DualAsr.md sections 5,
7 and 9): reading and freezing the vocabulary, the catalog entry and its bundled tokenizer, the ``asr``
handler's vocabulary pass and every failure row with fakes (no GPU, no model), the fake ASR's scripted
replacements, the worker's optional outputs, and the whole path through an in-process Store.

The Store tests at the end skip while that Store has not built ``getAsrVocabulary`` (501).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.catalog import CatalogEntryStatus, ModelPurpose
from call1.contracts.common import ReviewerRole
from call1.contracts.contents import (
    SpeakerRole,
    TranscriptContent,
    TranscriptTurnContent,
    VocabularyCorrectionStatus,
    WordTimestampView,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import ASR_BASE_TRANSCRIPT_ROLE, ASR_VOCABULARY_PASS_ROLE, JOB_TYPE_RULES, JobParameters, JobStatus, JobType
from call1.contracts.vocabulary import (
    AsrVocabularyPack,
    AsrVocabularyRecord,
    AsrVocabularySettings,
    effective_vocabulary,
    vocabulary_active,
    vocabulary_digest,
)
from call1.pipeline import vocabulary_asr
from call1.pipeline.vocabulary_asr import PassSegment, PassWord
from call1.process.audio import AudioInfo
from call1.process.catalog import (
    WHISPER_VOCAB_ENTRY_ID,
    CatalogEntry,
    ProcessCatalog,
    installed_status,
    required_file_problem,
    seeded_catalog,
)
from call1.process.config import ProcessConfig
from call1.process.graph import GraphPlanner
from call1.process.handlers.base import JobCancelled
from call1.process.handlers.fake import FakeAsr, FakeBehavior
from call1.process.handlers.real import paths
from call1.process.handlers.real.media import RealAsr
from call1.process.handlers.real.vocabulary import NOTES, correct
from call1.process.store_client import StoreError
from call1.process.vocabulary import read_vocabulary, vocabulary_parameters
from call1.process.worker import output_kinds, output_roles_ok

from .conftest import SAMPLE, write_headers
from .test_process_units import _artifact_model, _rubric
from .test_real_handlers import FFMPEG, make_job, wav

V = "/store/v1"
CATALOG = seeded_catalog(mode="fake")
TERMS = ["Stanley cup", "Chadstone", "Afterpay"]


def record(terms=TERMS, *, enabled: bool = True, pack: Optional[AsrVocabularyPack] = None) -> AsrVocabularyRecord:
    settings = AsrVocabularySettings(enabled=enabled, customer_terms=list(terms))
    effective = effective_vocabulary(pack, settings)
    return AsrVocabularyRecord(record_version=1, settings=settings, pack=pack, effective_terms=effective,
                               effective_digest=vocabulary_digest(effective) if effective else None,
                               active=vocabulary_active(settings, effective))


PARAMS = vocabulary_parameters(record(), CATALOG)


def base_transcript(text: str = "I bought a standy cup today", channel: Optional[int] = None, words: bool = True) -> TranscriptContent:
    tokens = text.split()
    timed = [WordTimestampView(word=t, start_time=1.0 + i * 0.4, end_time=1.35 + i * 0.4, probability=0.9) for i, t in enumerate(tokens)]
    turn = TranscriptTurnContent(turn_id=0, speaker=SpeakerRole.UNKNOWN, start_time=1.0, end_time=timed[-1].end_time, text=text, channel=channel,
                                 word_timestamps=timed if words else None)
    return TranscriptContent(duration_seconds=6.0, is_redacted=False, turns=[turn])


def pass_result(base: TranscriptContent, replace: Optional[dict] = None) -> SimpleNamespace:
    """A ``vocabulary_asr.PassResult`` stand-in: the base words, with ``replace`` (base word -> pass word)."""
    replace = replace or {"standy": "Stanley"}
    words = [PassWord(word=replace.get(w.word, w.word), start=w.start_time + 0.02, end=w.end_time, probability=0.7, channel=None, segment=0)
             for t in base.turns for w in t.word_timestamps or []]
    return SimpleNamespace(glossary="Glossary: Stanley cup, Chadstone.", glossary_terms=["Stanley cup", "Chadstone"], words=words,
                           segments=[PassSegment(start=0.0, end=6.0, channel=None, avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.3,
                                                 temperature=0.0)],
                           inference_seconds=2.0, load_seconds=0.5, peak_memory_bytes=900)


def asr_job(tmp_path, params=PARAMS, channels: int = 1):
    return make_job(tmp_path, JobType.ASR, {}, parameters=JobParameters(asr_vocabulary=params, extra={"channels": channels}),
                    entry_id="parakeet-tdt-0.6b-v3")


@pytest.fixture
def installed(monkeypatch):
    """MLX on, and the Whisper Small entry installed (its install check passes)."""
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    monkeypatch.setattr(vocabulary_asr, "install_problem", lambda model_dir: None)


# --- reading and freezing the vocabulary -----------------------------------------------------------


def test_only_an_active_vocabulary_is_frozen_with_the_catalogs_whisper_entry():
    assert vocabulary_parameters(None, CATALOG) is None
    assert vocabulary_parameters(record(enabled=False), CATALOG) is None
    assert vocabulary_parameters(record(terms=[]), CATALOG) is None
    params = vocabulary_parameters(record(), CATALOG)
    assert [t.term for t in params.terms] == TERMS and params.digest == record().effective_digest
    assert params.candidate_entry.entry_id == WHISPER_VOCAB_ENTRY_ID and params.glossary_prompt_limit == 120
    # no asr_vocabulary entry in the catalog at all: Parakeet only
    bare = ProcessCatalog([e for e in CATALOG.entries.values() if e.entry_id != WHISPER_VOCAB_ENTRY_ID],
                          {p: e for p, e in CATALOG.defaults.items() if p is not ModelPurpose.ASR_VOCABULARY})
    assert vocabulary_parameters(record(), bare) is None


def test_the_entry_is_frozen_even_when_it_is_not_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("CALL1_MODELS_DIR", str(tmp_path / "empty"))
    real = seeded_catalog(mode="real")
    assert real.status(real.get(WHISPER_VOCAB_ENTRY_ID))[0] is CatalogEntryStatus.NOT_INSTALLED
    assert vocabulary_parameters(record(), real).candidate_entry.entry_id == WHISPER_VOCAB_ENTRY_ID


def test_a_store_without_the_vocabulary_route_means_no_vocabulary():
    class Client:
        def __init__(self, exc=None, value=None):
            self.exc, self.value = exc, value

        def get_asr_vocabulary(self):
            if self.exc:
                raise self.exc
            return self.value

    assert read_vocabulary(Client(StoreError("not_implemented", "later", status=501))) is None
    assert read_vocabulary(Client(value=record())) == record()
    with pytest.raises(StoreError):
        read_vocabulary(Client(StoreError("forbidden", "no", status=403)))


def _graph(asr_vocabulary=None, channels: int = 2):
    planner = GraphPlanner(CATALOG, ProcessConfig())
    definition, ref = _rubric()
    audio = AudioInfo(container="wav", content_type="audio/wav", size_bytes=10, channels=channels, duration_seconds=600)
    graph = planner.ingest(conversation_id="conv_1", source=_artifact_model(), audio=audio, rubric_ref=ref, rubric=definition,
                           snapshot=_artifact_model("rubric_snapshot", "art_snap"), asr_vocabulary=asr_vocabulary)
    return {j.ref: j for j in graph.jobs}


def test_the_planner_freezes_the_vocabulary_on_the_asr_job_only_and_keeps_the_graph_shape():
    plain = _graph()
    dual = _graph(PARAMS)
    assert plain["asr"].parameters.asr_vocabulary is None and plain["asr"].resource_estimate.estimated_runtime_seconds is None
    assert dual["asr"].parameters.asr_vocabulary == PARAMS
    assert dual["asr"].resource_estimate.estimated_runtime_seconds == pytest.approx(600 * 0.012 * 3.5)
    assert set(plain) == set(dual)
    for ref, job in dual.items():
        assert job.inputs == plain[ref].inputs
        assert (job.requires_refs, job.after_refs) == (plain[ref].requires_refs, plain[ref].after_refs)
        if ref != "asr":
            assert job.parameters.asr_vocabulary is None
    # every consumer, enrichment (and so PII masking) included, reads the asr job's merged transcript
    enrichment = next(i for i in dual["enrichment"].inputs if i.role == "transcript")
    assert (enrichment.upstream.ref, enrichment.upstream.output_role) == ("asr", "transcript")
    roles = {i.upstream.output_role for j in dual.values() for i in j.inputs if i.upstream is not None}
    assert not roles & {ASR_BASE_TRANSCRIPT_ROLE, ASR_VOCABULARY_PASS_ROLE}


def test_planner_asr_vocabulary_reads_the_record():
    planner = GraphPlanner(CATALOG, ProcessConfig())
    assert planner.asr_vocabulary(record()) == PARAMS and planner.asr_vocabulary(record(enabled=False)) is None


# --- the catalog entry and its bundled tokenizer ------------------------------------------------------


def _whisper_dir(root: Path, tokenizer: Optional[bytes] = b"tokens") -> Path:
    directory = root / "whisper-small"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}")
    (directory / "weights.npz").write_bytes(b"x")
    if tokenizer is not None:
        (directory / "multilingual.tiktoken").write_bytes(tokenizer)
    return directory


def test_the_whisper_entry_is_installed_only_with_the_pinned_tokenizer(tmp_path, monkeypatch):
    entry = CATALOG.get(WHISPER_VOCAB_ENTRY_ID)
    assert entry.purposes == (ModelPurpose.ASR_VOCABULARY,) and entry.runtime == "mlx" and entry.model_directory == "whisper-small"
    assert entry.model_revision == "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e" and "MIT" in entry.license_notice
    directory = _whisper_dir(tmp_path / "models", tokenizer=None)
    assert required_file_problem(entry, directory) == "multilingual.tiktoken is missing from the model directory"
    assert installed_status(entry, tmp_path / "models")[0] is CatalogEntryStatus.NOT_INSTALLED
    (directory / "multilingual.tiktoken").write_bytes(b"not the file")
    assert required_file_problem(entry, directory) == "multilingual.tiktoken does not match its pinned checksum"
    good = CatalogEntry(**{**entry.__dict__, "required_files": (("multilingual.tiktoken", hashlib.sha256(b"not the file").hexdigest()),)})
    assert required_file_problem(good, directory) is None and installed_status(good, tmp_path / "models")[0] is CatalogEntryStatus.AVAILABLE
    # the real-handler status: MLX only, and the console says which file is missing
    monkeypatch.setenv("CALL1_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("CALL1_BACKEND", raising=False)
    assert paths.entry_status(entry)[0] is CatalogEntryStatus.INCOMPATIBLE
    assert paths.entry_detail(entry) == "needs CALL1_BACKEND=mlx (Apple Silicon)"
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    # Runtime availability is independent of tokenizer/file validation. Exercise both
    # hosts explicitly so this unit test needs no Apple-only packages on Linux.
    monkeypatch.setattr(paths, "_importable", lambda module: False)
    assert paths.entry_status(entry)[0] is CatalogEntryStatus.INCOMPATIBLE
    monkeypatch.setattr(paths, "_importable", lambda module: True)
    assert paths.entry_status(entry)[0] is CatalogEntryStatus.NOT_INSTALLED
    assert "multilingual.tiktoken does not match its pinned checksum" in paths.entry_detail(entry)
    assert "--models asr_vocabulary" in paths.entry_detail(entry)
    monkeypatch.setenv("CALL1_WHISPER_VOCAB_PATH", str(tmp_path / "elsewhere"))
    assert paths.weights_path(entry) == tmp_path / "elsewhere"
    catalog = CATALOG.with_status(paths.entry_status, paths.entry_detail)
    row = next(r for r in catalog.describe() if r["entry_id"] == WHISPER_VOCAB_ENTRY_ID)
    assert row["status"] == "not_installed" and row["detail"] == "config.json is missing from the model directory"


def test_install_problem_and_bundling_the_tokenizer(tmp_path, monkeypatch):
    directory = _whisper_dir(tmp_path, tokenizer=None)
    assert vocabulary_asr.install_problem(None) == "the model directory is missing"
    assert vocabulary_asr.install_problem(directory) == "multilingual.tiktoken is missing from the model directory"
    source = tmp_path / "cache" / "multilingual.tiktoken"
    source.parent.mkdir()
    source.write_bytes(b"cached tokenizer")
    with pytest.raises(RuntimeError, match="pinned sha256"):
        vocabulary_asr.install_tokenizer(directory, source=source)
    assert not (directory / "multilingual.tiktoken").exists()
    monkeypatch.setattr(vocabulary_asr, "TOKENIZER_SHA256", hashlib.sha256(b"cached tokenizer").hexdigest())
    target = vocabulary_asr.install_tokenizer(directory, source=source)
    assert target.read_bytes() == b"cached tokenizer" and vocabulary_asr.install_problem(directory) is None
    (directory / "weights.npz").unlink()
    assert vocabulary_asr.install_problem(directory) == "the weights are missing from the model directory"


def test_the_manifest_pins_the_whisper_weights_tokenizer_and_licence():
    manifest = json.loads((Path(__file__).resolve().parents[2] / "model-manifest.json").read_text())
    spec = manifest["asr_vocabulary"]
    assert spec["repository"] == "mlx-community/whisper-small-mlx" and spec["directory"] == "whisper-small"
    assert spec["tokenizer"]["sha256"] == vocabulary_asr.TOKENIZER_SHA256 and spec["tokenizer"]["url"] == vocabulary_asr.TOKENIZER_URL
    assert spec["tokenizer"]["filename"] == vocabulary_asr.TOKENIZER_FILE


def test_every_window_is_prompted_with_the_glossary_and_the_previous_tail():
    seen: List[list] = []

    class Model:
        def decode(self, segment, options):
            seen.append(list(options.prompt))
            return segment

    model = Model()
    cancelled = []
    original = vocabulary_asr.prompt_every_window(model, [1, 2, 3], check_cancelled=lambda: cancelled.append(1))
    Options = __import__("dataclasses").make_dataclass("Options", [("prompt", list), ("temperature", float)])
    model.decode("w1", Options(prompt=[], temperature=0.0))
    model.decode("w2", Options(prompt=list(range(100, 400)), temperature=0.0))
    assert seen[0] == [1, 2, 3]
    assert seen[1][:3] == [1, 2, 3] and len(seen[1]) == vocabulary_asr.PROMPT_LIMIT and seen[1][-1] == 399
    assert len(cancelled) == 2 and original.__self__ is model


# --- the asr handler's vocabulary pass, every failure row with fakes -----------------------------------


def test_the_pass_and_merge_return_three_outputs(tmp_path, installed):
    base = base_transcript()
    result = correct(asr_job(tmp_path), base, tmp_path / "a.wav", 1, base_engine="parakeet-tdt-0.6b-v3", catalog=CATALOG,
                     runner=lambda *a, **k: pass_result(base))
    assert set(result.outputs) == {"transcript", ASR_BASE_TRANSCRIPT_ROLE, ASR_VOCABULARY_PASS_ROLE}
    merged = result.outputs["transcript"].content
    assert merged.turns[0].text == "I bought a Stanley cup today"
    correction = merged.vocabulary_correction
    assert correction.status is VocabularyCorrectionStatus.APPLIED and correction.failure_code is None
    assert (correction.vocabulary_digest, correction.term_count, correction.glossary_term_count) == (PARAMS.digest, 3, 2)
    assert (correction.base_engine, correction.candidate_engine) == ("parakeet-tdt-0.6b-v3", "whisper-small")
    assert correction.candidate_model_revision == CATALOG.get(WHISPER_VOCAB_ENTRY_ID).model_revision and correction.candidates == 1
    assert [(r.heard, r.candidate_text, r.term) for r in correction.replacements] == [("standy cup", "Stanley cup", "Stanley cup")]
    assert result.outputs[ASR_BASE_TRANSCRIPT_ROLE].content == base and base.vocabulary_correction is None
    raw = result.outputs[ASR_VOCABULARY_PASS_ROLE].content
    assert raw.engine == "whisper-small" and raw.glossary_terms == ["Stanley cup", "Chadstone"] and raw.segments[0].avg_logprob == -0.2
    assert (result.inference_seconds, result.load_seconds, result.peak_memory_bytes) == (2.0, 0.5, 900)


def test_a_pass_that_finds_nothing_is_applied_with_no_replacements(tmp_path, installed):
    base = base_transcript("hello there")
    result = correct(asr_job(tmp_path), base, tmp_path / "a.wav", 1, base_engine="p", catalog=CATALOG, runner=lambda *a, **k: pass_result(base))
    correction = result.outputs["transcript"].content.vocabulary_correction
    assert correction.status is VocabularyCorrectionStatus.APPLIED and correction.replacements == [] and correction.candidates == 0
    assert ASR_VOCABULARY_PASS_ROLE in result.outputs


def _base_only(result, code: JobErrorCode, note: str):
    transcript = result.outputs["transcript"].content
    correction = transcript.vocabulary_correction
    assert correction.status is VocabularyCorrectionStatus.BASE_ONLY and correction.failure_code is code and correction.note == note
    assert correction.replacements == [] and correction.vocabulary_digest == PARAMS.digest
    assert set(result.outputs) == {"transcript", ASR_BASE_TRANSCRIPT_ROLE}
    assert transcript.turns == result.outputs[ASR_BASE_TRANSCRIPT_ROLE].content.turns
    return correction


def _explode(exc):
    def runner(*args, **kwargs):
        raise exc
    return runner


@pytest.mark.parametrize("exc, code, note", [
    (ImportError("mlx_audio"), JobErrorCode.MODEL_UNAVAILABLE, NOTES["runtime_missing"]),
    (subprocess.CalledProcessError(1, "ffmpeg"), JobErrorCode.VALIDATION_REJECTED, NOTES["decode"]),
    (MemoryError(), JobErrorCode.RESOURCE_UNAVAILABLE, NOTES["memory"]),
    (RuntimeError("[metal] out of memory"), JobErrorCode.RESOURCE_UNAVAILABLE, NOTES["memory"]),
    (RuntimeError("decode failed"), JobErrorCode.PROVIDER_ERROR, NOTES["runtime"]),
    (ValueError("bad segment"), JobErrorCode.PROVIDER_ERROR, NOTES["runtime"]),
])
def test_a_runtime_failure_keeps_the_parakeet_transcript(tmp_path, installed, caplog, exc, code, note):
    base = base_transcript()
    result = correct(asr_job(tmp_path), base, tmp_path / "a.wav", 1, base_engine="p", catalog=CATALOG, runner=_explode(exc))
    _base_only(result, code, note)
    assert "job_1" in caplog.text and "standy" not in caplog.text


def test_not_installed_no_mlx_no_timings_and_no_entry_are_base_only(tmp_path, monkeypatch):
    base = base_transcript()
    never = _explode(AssertionError("the pass must not run"))
    monkeypatch.delenv("CALL1_BACKEND", raising=False)
    _base_only(correct(asr_job(tmp_path), base, tmp_path / "a.wav", 1, base_engine="p", catalog=CATALOG, runner=never),
               JobErrorCode.CONFIGURATION_ERROR, NOTES["needs_mlx"])
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    monkeypatch.setenv("CALL1_MODELS_DIR", str(tmp_path / "models"))
    _whisper_dir(tmp_path / "models", tokenizer=None)
    correction = _base_only(correct(asr_job(tmp_path), base, tmp_path / "a.wav", 1, base_engine="p", catalog=CATALOG, runner=never),
                            JobErrorCode.MODEL_UNAVAILABLE,
                            NOTES["not_installed"].format(problem="multilingual.tiktoken is missing from the model directory"))
    assert correction.candidate_engine == "whisper-small"
    monkeypatch.setattr(vocabulary_asr, "install_problem", lambda model_dir: None)
    _base_only(correct(asr_job(tmp_path), base_transcript(words=False), tmp_path / "a.wav", 1, base_engine="p", catalog=CATALOG, runner=never),
               JobErrorCode.CONFIGURATION_ERROR, NOTES["no_word_timings"])
    other = ProcessCatalog([e for e in CATALOG.entries.values() if e.entry_id != WHISPER_VOCAB_ENTRY_ID], CATALOG.defaults)
    correction = _base_only(correct(asr_job(tmp_path), base, tmp_path / "a.wav", 1, base_engine="p", catalog=other, runner=never),
                            JobErrorCode.MODEL_UNAVAILABLE, NOTES["not_in_catalog"])
    assert correction.candidate_engine is None


def test_cancellation_during_the_pass_cancels_the_job(tmp_path, installed):
    with pytest.raises(JobCancelled):
        correct(asr_job(tmp_path), base_transcript(), tmp_path / "a.wav", 1, base_engine="p", catalog=CATALOG, runner=_explode(JobCancelled()))


def test_stereo_runs_the_merge_per_channel(tmp_path, installed):
    agent = base_transcript("welcome to Chad Stone", channel=0).turns[0]
    caller = base_transcript("I paid with after pay", channel=1).turns[0].model_copy(update={"turn_id": 1, "speaker": SpeakerRole.CALLER})
    base = TranscriptContent(duration_seconds=6.0, is_redacted=False, turns=[agent.model_copy(update={"speaker": SpeakerRole.AGENT}), caller])

    def runner(model_dir, audio, channels, terms, words_by_channel, limit, check_cancelled=None):
        assert channels == 2 and set(words_by_channel) == {0, 1} and terms == TERMS and limit == 120
        words = []
        for turn in base.turns:
            t = turn.word_timestamps
            if turn.channel == 0:
                words += [PassWord("welcome", t[0].start_time, t[0].end_time, 0.9, 0, 0), PassWord("to", t[1].start_time, t[1].end_time, 0.9, 0, 0),
                          PassWord("Chadstone", t[2].start_time, t[3].end_time, 0.8, 0, 0)]
            else:
                words += [PassWord(w.word, w.start_time, w.end_time, 0.9, 1, 1) for w in t[:3]]
                words.append(PassWord("Afterpay", t[3].start_time, t[4].end_time, 0.8, 1, 1))
        return SimpleNamespace(glossary_terms=terms, words=words, segments=[], inference_seconds=1.0, load_seconds=0.1, peak_memory_bytes=None)

    result = correct(asr_job(tmp_path, channels=2), base, tmp_path / "a.wav", 2, base_engine="p", catalog=CATALOG, runner=runner)
    merged = result.outputs["transcript"].content
    assert [t.text for t in merged.turns] == ["welcome to Chadstone", "I paid with Afterpay"]
    assert {w.channel for w in result.outputs[ASR_VOCABULARY_PASS_ROLE].content.words} == {0, 1}


@FFMPEG
def test_real_asr_runs_parakeet_then_the_vocabulary_pass(tmp_path, monkeypatch, installed):
    from call1.models.schemas import SpeakerRole as LegacyRole, TranscriptTurn, WordTimestamp

    root = tmp_path / "models" / "parakeet-tdt-0.6b-v3"
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}")
    (root / "weights.npz").write_bytes(b"x")
    monkeypatch.setenv("CALL1_MODELS_DIR", str(tmp_path / "models"))
    spoken = [("I", 0.5, 0.7), ("bought", 0.7, 1.0), ("a", 1.0, 1.1), ("standy", 1.2, 1.6), ("cup.", 1.6, 2.0)]

    def transcribe(self, path, channels, agent_channel=0):
        return [TranscriptTurn(turn_id=0, speaker=LegacyRole.UNKNOWN, start_time=0.5, end_time=2.0, text="I bought a standy cup.",
                               raw_text="I bought a standy cup.", channel=0,
                               word_timestamps=[WordTimestamp(word=w, start_time=s, end_time=e, probability=0.9) for w, s, e in spoken])]

    def run_pass(model_dir, audio, channels, terms, words_by_channel, limit, check_cancelled=None):
        assert model_dir == tmp_path / "models" / "whisper-small" and channels == 1 and Path(audio).is_file()
        check_cancelled()
        return SimpleNamespace(glossary_terms=["Stanley cup"], segments=[], inference_seconds=3.0, load_seconds=1.0, peak_memory_bytes=5_000,
                               words=[PassWord(w if w != "standy" else "Stanley", s, e, 0.8, None, 0) for w, s, e in spoken])

    monkeypatch.setattr("call1.adapters.mlx.MLXAdapter.transcribe", transcribe)
    monkeypatch.setattr(vocabulary_asr, "run_pass", run_pass)
    audio = wav(tmp_path / "mono.wav", seconds=6.0, channels=1)
    job = make_job(tmp_path, JobType.ASR, {"audio": (ArtifactKind.SOURCE_AUDIO, audio)},
                   parameters=JobParameters(asr_vocabulary=PARAMS, extra={"channels": 1}), entry_id="parakeet-tdt-0.6b-v3")
    handler = RealAsr()
    handler.catalog = CATALOG
    handler.ready(job)
    result = handler.run(job)
    assert set(result.outputs) == {"transcript", ASR_BASE_TRANSCRIPT_ROLE, ASR_VOCABULARY_PASS_ROLE}
    assert result.outputs["transcript"].content.turns[0].text == "I bought a Stanley cup."
    assert result.outputs[ASR_BASE_TRANSCRIPT_ROLE].content.turns[0].text == "I bought a standy cup."
    assert result.usage.peak_memory_bytes == 5_000 and result.usage.model_load_seconds == 1.0
    # without asr_vocabulary the job is exactly as before 1.3.0
    plain = handler.run(make_job(tmp_path, JobType.ASR, {"audio": (ArtifactKind.SOURCE_AUDIO, audio)},
                                 parameters=JobParameters(extra={"channels": 1}), entry_id="parakeet-tdt-0.6b-v3"))
    assert set(plain.outputs) == {"transcript"} and plain.outputs["transcript"].content.vocabulary_correction is None


def test_the_worker_accepts_the_optional_outputs_and_nothing_else():
    rule = JOB_TYPE_RULES[JobType.ASR]
    assert output_roles_ok(rule, {"transcript": 1})
    assert output_roles_ok(rule, {"transcript": 1, ASR_BASE_TRANSCRIPT_ROLE: 1})
    assert output_roles_ok(rule, {"transcript": 1, ASR_BASE_TRANSCRIPT_ROLE: 1, ASR_VOCABULARY_PASS_ROLE: 1})
    assert not output_roles_ok(rule, {ASR_BASE_TRANSCRIPT_ROLE: 1})
    assert not output_roles_ok(rule, {"transcript": 1, "whisper": 1})
    assert [r for r, _ in output_kinds(rule, {"transcript": 1, ASR_VOCABULARY_PASS_ROLE: 1})] == ["transcript", ASR_VOCABULARY_PASS_ROLE]
    assert dict(output_kinds(rule, {"transcript": 1, ASR_BASE_TRANSCRIPT_ROLE: 1}))[ASR_BASE_TRANSCRIPT_ROLE] is ArtifactKind.ASR_BASE_TRANSCRIPT


# --- the fake ASR --------------------------------------------------------------------------------------


def _fake_run(tmp_path, actions, params=PARAMS, seconds: float = 40.0, channels: int = 1):
    audio = wav(tmp_path / f"fake-{channels}.wav", seconds=seconds, channels=channels)
    job = make_job(tmp_path, JobType.ASR, {"audio": (ArtifactKind.SOURCE_AUDIO, audio)},
                   parameters=JobParameters(asr_vocabulary=params, extra={"channels": channels}))
    return FakeAsr(FakeBehavior({"asr": actions})).run(job)


def test_the_fake_vocabulary_script_shows_three_real_replacements(tmp_path):
    result = _fake_run(tmp_path, ["script:vocabulary"])
    merged = result.outputs["transcript"].content
    correction = merged.vocabulary_correction
    assert correction.status is VocabularyCorrectionStatus.APPLIED and correction.candidate_engine == "fake-whisper-small"
    assert [(r.turn_id, r.heard, r.term) for r in correction.replacements] == [
        (1, "standy cup", "Stanley cup"), (1, "Chad Stone", "Chadstone"), (4, "after pay", "Afterpay")]
    assert "a Stanley cup for click and collect at the Chadstone store" in merged.turns[1].text
    assert "to your Afterpay account" in merged.turns[4].text
    for r in correction.replacements:
        turn = next(t for t in merged.turns if t.turn_id == r.turn_id)
        assert turn.text[r.char_start:r.char_end] == r.term
        assert " ".join(w.word for w in turn.word_timestamps[r.word_start:r.word_end]).strip(".,") == r.term
    base = result.outputs[ASR_BASE_TRANSCRIPT_ROLE].content
    assert "standy cup" in base.turns[1].text and base.vocabulary_correction is None and base.turns[1].word_timestamps
    raw = result.outputs[ASR_VOCABULARY_PASS_ROLE].content
    assert raw.engine == "fake-whisper-small" and set(raw.glossary_terms) == set(TERMS)


def test_the_fake_leaves_terms_outside_the_vocabulary_and_scripts_without_mishearings(tmp_path):
    only_cups = vocabulary_parameters(record(["Stanley cup"]), CATALOG)
    replaced = _fake_run(tmp_path, ["script:vocabulary"], params=only_cups).outputs["transcript"].content.vocabulary_correction.replacements
    assert [r.term for r in replaced] == ["Stanley cup"]
    plain = _fake_run(tmp_path, ["ok"]).outputs["transcript"].content
    assert plain.vocabulary_correction.status is VocabularyCorrectionStatus.APPLIED and plain.vocabulary_correction.replacements == []


def test_the_fake_vocabulary_failure_is_base_only(tmp_path):
    result = _fake_run(tmp_path, ["script:vocabulary,vocabulary_fail"])
    assert set(result.outputs) == {"transcript", ASR_BASE_TRANSCRIPT_ROLE}
    correction = result.outputs["transcript"].content.vocabulary_correction
    assert correction.status is VocabularyCorrectionStatus.BASE_ONLY and correction.failure_code is JobErrorCode.MODEL_UNAVAILABLE
    assert "standy cup" in result.outputs["transcript"].content.turns[1].text
    coded = _fake_run(tmp_path, ["vocabulary_fail:resource_unavailable"]).outputs["transcript"].content.vocabulary_correction
    assert coded.failure_code is JobErrorCode.RESOURCE_UNAVAILABLE


def test_the_fake_without_a_vocabulary_is_unchanged(tmp_path):
    result = _fake_run(tmp_path, ["script:vocabulary"], params=None)
    transcript = result.outputs["transcript"].content
    assert set(result.outputs) == {"transcript"} and transcript.vocabulary_correction is None
    assert all(t.word_timestamps is None for t in transcript.turns)


def test_the_fake_stereo_pass_is_per_channel(tmp_path):
    result = _fake_run(tmp_path, ["script:vocabulary"], channels=2)
    raw = result.outputs[ASR_VOCABULARY_PASS_ROLE].content
    assert raw.channels == 2 and {w.channel for w in raw.words} == {0, 1}
    assert len(result.outputs["transcript"].content.vocabulary_correction.replacements) == 3


# --- through an in-process Store (skips until Store has the vocabulary routes) ---------------------------


def _admin_vocabulary(store_http, session, terms=TERMS, enabled=True):
    admin = session(ReviewerRole.ADMIN)
    current = store_http.get(f"{V}/vocabulary", headers=admin.read_headers)
    if current.status_code == 501:
        pytest.skip("this Store has not built getAsrVocabulary yet (docs/DualAsr.md section 4)")
    assert current.status_code == 200, current.text
    body = {"expected_record_version": current.json()["record_version"],
            "settings": {"enabled": enabled, "customer_terms": list(terms), "disabled_pack_terms": []}}
    saved = store_http.put(f"{V}/vocabulary", json=body, headers=write_headers(admin))
    assert saved.status_code == 200, saved.text
    return AsrVocabularyRecord.model_validate(saved.json())


def _linked(runtime, conversation_id, kind):
    return [a for a in runtime.client.list_artifacts(conversation_id, kind=kind) if a.linked and a.superseded_by is None]


def test_ingest_runs_dual_transcription_end_to_end(make_runtime, store_http, session):
    saved = _admin_vocabulary(store_http, session)
    runtime = make_runtime(behavior=FakeBehavior({"asr": ["script:vocabulary"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    jobs = runtime.client.list_jobs(conversation_id=result.conversation_id)
    assert {j.status for j in jobs} == {JobStatus.SUCCEEDED}
    asr = next(j for j in jobs if j.job_type is JobType.ASR)
    assert asr.parameters.asr_vocabulary.digest == saved.effective_digest
    assert asr.parameters.asr_vocabulary.candidate_entry.entry_id == WHISPER_VOCAB_ENTRY_ID

    transcript = _linked(runtime, result.conversation_id, ArtifactKind.TRANSCRIPT)
    assert len(transcript) == 1
    content = TranscriptContent.model_validate(json.loads(runtime.client.download(transcript[0])))
    assert [r.term for r in content.vocabulary_correction.replacements] == ["Stanley cup", "Chadstone", "Afterpay"]
    assert len(_linked(runtime, result.conversation_id, ArtifactKind.ASR_BASE_TRANSCRIPT)) == 1
    assert len(_linked(runtime, result.conversation_id, ArtifactKind.ASR_VOCABULARY_PASS)) == 1
    # enrichment (and so the PII findings every masked read uses) ran on the merged transcript
    enrichment = next(j for j in jobs if j.job_type is JobType.ENRICHMENT)
    resolved = runtime.client.get_job(enrichment.id).resolved_inputs
    assert [i.artifact_id for i in resolved if i.role == "transcript"] == [transcript[0].id]

    reviewer = session(ReviewerRole.REVIEWER)
    view = store_http.get(f"{V}/calls/{result.call_id}/transcript", headers=reviewer.read_headers).json()
    assert "Stanley cup" in " ".join(t["text"] for t in view["turns"])

    # a later vocabulary change does not break a repeated ingest of the same recording
    _admin_vocabulary(store_http, session, terms=["Stanley cup"])
    again = runtime.ingestor.ingest_file(SAMPLE)
    assert (again.graph_id, again.graph_created) == (result.graph_id, False)


def test_a_vocabulary_failure_keeps_the_call_and_links_no_pass(make_runtime, store_http, session):
    _admin_vocabulary(store_http, session)
    runtime = make_runtime(behavior=FakeBehavior({"asr": ["script:vocabulary,vocabulary_fail"]}))
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    assert {j.status for j in runtime.client.list_jobs(conversation_id=result.conversation_id)} == {JobStatus.SUCCEEDED}
    transcript = _linked(runtime, result.conversation_id, ArtifactKind.TRANSCRIPT)[0]
    correction = TranscriptContent.model_validate(json.loads(runtime.client.download(transcript))).vocabulary_correction
    assert correction.status is VocabularyCorrectionStatus.BASE_ONLY
    assert len(_linked(runtime, result.conversation_id, ArtifactKind.ASR_BASE_TRANSCRIPT)) == 1
    assert _linked(runtime, result.conversation_id, ArtifactKind.ASR_VOCABULARY_PASS) == []


def test_a_vocabulary_that_is_off_plans_parakeet_alone(make_runtime, store_http, session):
    _admin_vocabulary(store_http, session, enabled=False)
    runtime = make_runtime()
    worker = runtime.connect()
    result = runtime.ingestor.ingest_file(SAMPLE)
    worker.drain()
    asr = next(j for j in runtime.client.list_jobs(conversation_id=result.conversation_id) if j.job_type is JobType.ASR)
    assert asr.parameters.asr_vocabulary is None and asr.status is JobStatus.SUCCEEDED
    assert _linked(runtime, result.conversation_id, ArtifactKind.ASR_BASE_TRANSCRIPT) == []
