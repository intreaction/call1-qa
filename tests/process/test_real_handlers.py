"""The real stage handlers (``call1/process/handlers/real``) with their model runtimes replaced by
fakes: each handler's contract output, its error mapping, and parity with the pre-split pipeline
functions it wraps. No model weights, no MLX, no network, no ports.

The end-to-end run on the real models is ``tests/process/real/test_real_models.py`` (opt-in).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.catalog import CatalogEntryStatus, ModelPurpose
from call1.contracts.common import canonical_digest, canonical_json
from call1.contracts.contents import (
    ContactSignalPass,
    EnrichmentContent,
    EscalationTrigger,
    QaAssessmentContent,
    SpeakerAttributionContent,
    SpeakerRole,
    SummarySegmentContent,
    SummarySynthesisContent,
    TextSentimentContent,
    ToneBlocksContent,
    TranscriptContent,
    TranscriptTurnContent,
    TurnWindow,
    VerdictStatus,
    WordTimestampView,
)
from call1.contracts.custody import RouteClass
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import ClaimedJob, Job, JobParameters, JobType, SegmentSpec, SpeakerCorrection
from call1.contracts.rubrics import CheckType, RubricCheck, RubricCriterion, RubricDefinition, RubricSnapshotContent, RubricVersionRef
from call1.process.catalog import seeded_catalog
from call1.process.handlers import build_registry
from call1.process.handlers.base import HandlerError, HandlerJob, InputArtifact, JobCancelled, ReleaseJob
from call1.process.handlers.real import paths, real_handlers
from call1.process.handlers.real.analysis import RealAcousticTone, RealEnrichment, RealTextSentiment
from call1.process.handlers.real.llm import check_route
from call1.process.handlers.real.media import RealAsr, RealSpeakerAttribution, RealValidationVad
from call1.process.handlers.real.qa import RealQaCriterion, RealQaDeterministic, RealQaEscalation
from call1.process.handlers.real.signals import RealLifecyclePass, RealResolutionPass
from call1.process.handlers.real.summary import RealSummaryAssembly, RealSummarySegment, RealSummarySynthesis
from call1.question_models import generate_text as REAL_GENERATE_TEXT

FFMPEG = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is needed to decode audio")
CATALOG = seeded_catalog(mode="fake")


# --- building a claimed job without Store ------------------------------------------------------


def _content_bytes(content: Any) -> bytes:
    return canonical_json(content) if not isinstance(content, (bytes, bytearray)) else bytes(content)


def _artifact(role: str, kind: ArtifactKind, data: bytes, content_type: str) -> Artifact:
    return Artifact.model_construct(id=f"art_{role.replace(':', '_')}", kind=kind, slot="", content_type=content_type, size_bytes=len(data),
                                    checksum="sha256:" + hashlib.sha256(data).hexdigest(), content_contract="", sensitivity="raw",
                                    conversation_id="conv_1", linked=True, version=1)


def make_job(tmp_path: Path, job_type: JobType, inputs: Dict[str, Tuple[ArtifactKind, Any]], *, parameters: Optional[JobParameters] = None,
             entry_id: Optional[str] = None, purpose: Optional[ModelPurpose] = None, masked: bool = False, final_attempt: bool = False,
             audio_type: str = "audio/wav") -> HandlerJob:
    entry = CATALOG.get(entry_id) if entry_id else None
    selection = entry.selection(purpose or entry.purposes[0], masked=masked) if entry else None
    job = Job.model_construct(id="job_1", conversation_id="conv_1", graph_id="graph_1", job_type=job_type, parameters=parameters or JobParameters(),
                              selection=selection)
    claimed = ClaimedJob.model_construct(job=job, attempt_number=1, final_attempt=final_attempt, upstream=[], inputs=[])
    scratch = tmp_path / f"scratch-{job_type.value}"
    scratch.mkdir(parents=True, exist_ok=True)
    resolved: Dict[str, Optional[InputArtifact]] = {}
    for role, (kind, content) in inputs.items():
        if isinstance(content, Path):
            data = content.read_bytes()
            content_type = audio_type
        else:
            data = _content_bytes(content)
            content_type = "application/json"
        artifact = _artifact(role, kind, data, content_type)

        def fetch(_artifact, dest, data=data):
            if dest is None:
                return data
            dest.write_bytes(data)
            return dest

        resolved[role] = InputArtifact(role, artifact, fetch, scratch)
    return HandlerJob(claimed, resolved, scratch, catalog_entry=entry)


def wav(path: Path, seconds: float = 8.0, channels: int = 2, rate: int = 16000, silent: bool = False) -> Path:
    """Tone bursts: channel c sounds for 1.5 s out of every 2 s, offset by 0.5 s per channel, so the
    energy VAD finds speech on both channels and some overtalk."""
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        data = bytearray()
        for i in range(frames):
            t = i / rate
            for c in range(channels):
                on = not silent and ((t + 0.5 * c) % 2.0) < 1.5
                value = int(8000 * math.sin(2 * math.pi * (220 + 110 * c) * t)) if on else 0
                data += struct.pack("<h", value)
        w.writeframes(bytes(data))
    return path


def turn(i: int, speaker: str, start: float, end: float, text: str, channel: Optional[int] = None, words: bool = True) -> TranscriptTurnContent:
    timestamps = None
    if words:
        tokens = text.split()
        step = (end - start) / max(1, len(tokens))
        timestamps = [WordTimestampView(word=(" " if k else "") + w, start_time=round(start + k * step, 3), end_time=round(start + (k + 1) * step, 3),
                                        probability=0.9) for k, w in enumerate(tokens)]
    return TranscriptTurnContent(turn_id=i, speaker=SpeakerRole(speaker), start_time=start, end_time=end, text=text, channel=channel,
                                 confidence=0.9, word_timestamps=timestamps)


TRANSCRIPT = TranscriptContent(duration_seconds=30.0, language=None, is_redacted=False, turns=[
    turn(0, "AGENT", 0.0, 4.0, "Thank you for calling. This call may be recorded for quality assurance.", 0),
    turn(1, "CALLER", 4.5, 8.0, "Hi, I was charged a fee on account 1234-5678-9012 and I want it removed.", 1),
    turn(2, "AGENT", 8.5, 14.0, "I can help with that. I have reversed the monthly fee of $5.00 today.", 0),
    turn(3, "CALLER", 14.5, 18.0, "Great, that fixes it. Thank you so much.", 1),
    turn(4, "AGENT", 18.5, 24.0, "Is there anything else I can help you with today? Have a great day.", 0),
])


def criterion(cid: str, check: RubricCheck, name: Optional[str] = None, **kw) -> RubricCriterion:
    return RubricCriterion(criterion_id=cid, name=name or cid, check=check, **kw)


SEMANTIC = RubricCheck(check_type=CheckType.SEMANTIC_JUDGEMENT, pass_when="The agent discloses the recording.", fail_when="It does not.",
                       not_applicable_when="Never.")
RUBRIC = RubricDefinition(rubric_id="test_rubric", name="Test", criteria=[
    criterion("REG-01", SEMANTIC),
    criterion("SEC-01", RubricCheck(check_type=CheckType.SEMANTIC_JUDGEMENT, pass_when="Verified.", requires_policy=True), category="SECURITY"),
    criterion("GREET-01", RubricCheck(check_type=CheckType.PHRASE_ANY, phrases=["thank you for calling"], window_seconds=15)),
    criterion("REGEX-01", RubricCheck(check_type=CheckType.CUSTOM_REGEX, pattern=r"reversed the .* fee")),
    criterion("NONE-01", RubricCheck(check_type=CheckType.PHRASE_NONE, phrases=["that is not my problem"])),
    criterion("SPAN-01", RubricCheck(check_type=CheckType.PHRASE_ANY, phrases=["quality assurance hi i was"], speaker=None, threshold=90)),
    criterion("TEXT-01", RubricCheck(check_type=CheckType.SENTIMENT_METRIC, metric="text_polarity", metric_threshold=0.0, min_samples=2)),
])


def snapshot(rubric: RubricDefinition = RUBRIC) -> RubricSnapshotContent:
    return RubricSnapshotContent(source="published", rubric_id=rubric.rubric_id, rubric_version=1, digest=canonical_digest(rubric), definition=rubric)


def ref(rubric: RubricDefinition = RUBRIC) -> RubricVersionRef:
    return RubricVersionRef(rubric_id=rubric.rubric_id, version=1, digest=canonical_digest(rubric))


def qa_job(tmp_path, cid: str, *, job_type=JobType.QA_CRITERION, masked=False, transcript=TRANSCRIPT, rubric=RUBRIC, **kw) -> HandlerJob:
    inputs: Dict[str, Tuple[ArtifactKind, Any]] = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "rubric": (ArtifactKind.RUBRIC_SNAPSHOT, snapshot(rubric))}
    if masked:
        inputs["enrichment"] = (ArtifactKind.ENRICHMENT, _enrich(transcript))
    return make_job(tmp_path, job_type, inputs, parameters=JobParameters(rubric=ref(rubric), criterion_id=cid), entry_id="call1-bundled",
                    purpose=ModelPurpose.SEMANTIC_QA, masked=masked, **kw)


def _enrich(transcript: TranscriptContent) -> EnrichmentContent:
    job = make_job(_TMP[0], JobType.ENRICHMENT, {"transcript": (ArtifactKind.TRANSCRIPT, transcript)})
    return RealEnrichment().run(job).outputs["enrichment"].content  # type: ignore[return-value]


_TMP: List[Path] = []


@pytest.fixture(autouse=True)
def _scratch(tmp_path, monkeypatch):
    _TMP[:] = [tmp_path]
    monkeypatch.delenv("CALL1_BACKEND", raising=False)
    monkeypatch.setenv("CALL1_SENTIMENT_MODELS", "1")
    for env in paths.LEGACY_PATH_ENV.values():
        monkeypatch.delenv(env, raising=False)
    yield


def answer(verdict: str, quote: str = "", assessment: str = "Checked the call.") -> str:
    return json.dumps({"assessment": assessment, "verdict": verdict, "quote": quote})


class FakeGenerate:
    """Stands in for ``call1.question_models.generate_text``: scripted replies, every call recorded."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: List[dict] = []

    def __call__(self, model, system, prompt, allow_external=False, response_schema=None, schema_name="answer", max_tokens=None,
                 synthetic=False, text_model_path=None):
        self.calls.append({"model": model, "system": system, "prompt": prompt, "schema": response_schema, "schema_name": schema_name,
                           "max_tokens": max_tokens, "text_model_path": text_model_path, "allow_external": allow_external})
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply, {}


# --- registry and entry status ---------------------------------------------------------------


def test_real_mode_registers_every_job_type_and_reports_this_hosts_models(tmp_path, monkeypatch):
    registry = build_registry("real", config=SimpleNamespace(summary_batch_turns=60), catalog=CATALOG)
    assert registry.missing() == []
    assert registry.get(JobType.SUMMARY_ASSEMBLY).adapter_id == "call1.summarizer.assembly"  # replaces the code stage
    assert registry.get(JobType.QA_SCORECARD).adapter_id == "call1.code.scorecard"  # the code stages stay
    assert registry.get(JobType.EMBEDDINGS).adapter_id == "call1.fake.embeddings"  # CALL1_EMBEDDING_BACKEND=fake (tests/conftest.py)
    monkeypatch.setenv("CALL1_EMBEDDING_BACKEND", "nemotron")
    assert build_registry("real", config=SimpleNamespace(summary_batch_turns=60), catalog=CATALOG).get(JobType.EMBEDDINGS).adapter_id == "call1.torch.nemotron_embed"
    assert registry.entry_status is paths.entry_status
    assert {h.job_type for h in real_handlers()} == set(JobType) - {JobType.EMBEDDINGS, JobType.QA_SCORECARD, JobType.CONTACT_SIGNALS_MERGE}

    monkeypatch.setenv("CALL1_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    monkeypatch.setattr(paths, "_importable", lambda module: True)
    status = {e.entry_id: paths.entry_status(e)[0] for e in CATALOG.entries.values()}
    assert status["nemotron-3-embed-1b"] is CatalogEntryStatus.NOT_INSTALLED
    assert status["parakeet-tdt-0.6b-v3"] is CatalogEntryStatus.NOT_INSTALLED
    assert status["gemma4-12b"] is CatalogEntryStatus.UNQUALIFIED  # not supported by this app version
    asr = tmp_path / "models" / "parakeet-tdt-0.6b-v3"
    asr.mkdir(parents=True)
    (asr / "config.json").write_text("{}")
    assert paths.entry_status(CATALOG.get("parakeet-tdt-0.6b-v3"))[0] is CatalogEntryStatus.NOT_INSTALLED  # config without weights
    (asr / "model.safetensors").write_bytes(b"x")
    assert paths.entry_status(CATALOG.get("parakeet-tdt-0.6b-v3")) == (CatalogEntryStatus.AVAILABLE, [ModelPurpose.ASR])
    # the pre-split per-model variable still wins
    legacy = tmp_path / "legacy-asr"
    monkeypatch.setenv("CALL1_MLX_ASR_PATH", str(legacy))
    assert paths.weights_path(CATALOG.get("parakeet-tdt-0.6b-v3")) == legacy
    # without MLX, ASR is faster-whisper and diarization is skipped (not refused)
    monkeypatch.delenv("CALL1_BACKEND")
    assert paths.entry_status(CATALOG.get("nemotron-3-diarization"))[0] is CatalogEntryStatus.AVAILABLE


# --- media -----------------------------------------------------------------------------------


@FFMPEG
def test_validation_and_vad_report_the_recording(tmp_path):
    audio = wav(tmp_path / "call.wav", seconds=8.0, channels=2)
    job = make_job(tmp_path, JobType.VALIDATION_VAD, {"audio": (ArtifactKind.SOURCE_AUDIO, audio)},
                   parameters=JobParameters(extra={"agent_channel": 1}))
    handler = RealValidationVad()
    handler.ready(job)
    result = handler.run(job)
    report = result.outputs["validation_report"].content
    assert (report.container, report.codec, report.channels, report.channel_layout.value, report.agent_channel) == ("wav", "pcm_s16le", 2, "STEREO", 1)
    assert report.duration_seconds == pytest.approx(8.0, abs=0.01) and report.sample_rate == 16000
    vad = result.outputs["vad_metrics"].content
    assert vad.total_speech_duration > 5 and vad.overtalk_duration > 1 and {s.channel for s in vad.segments} == {0, 1}
    assert result.usage.audio_seconds_processed == report.duration_seconds


@FFMPEG
def test_validation_rejects_what_the_pre_split_gates_rejected(tmp_path):
    short = make_job(tmp_path, JobType.VALIDATION_VAD, {"audio": (ArtifactKind.SOURCE_AUDIO, wav(tmp_path / "short.wav", seconds=2.0, channels=1))})
    with pytest.raises(HandlerError) as err:
        RealValidationVad().run(short)
    assert err.value.code is JobErrorCode.VALIDATION_REJECTED and "DISCARD_TOO_SHORT" in err.value.detail
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio at all")
    with pytest.raises(HandlerError) as err:
        RealValidationVad().run(make_job(tmp_path, JobType.VALIDATION_VAD, {"audio": (ArtifactKind.SOURCE_AUDIO, junk)}))
    assert err.value.code is JobErrorCode.VALIDATION_REJECTED


def _legacy_turns(channels: int = 2):
    from call1.models.schemas import SpeakerRole as LegacyRole, TranscriptTurn, WordTimestamp

    words = [WordTimestamp(word=" Hello", start_time=0.0, end_time=0.5, probability=1.2), WordTimestamp(word=" there.", start_time=0.5, end_time=1.0, probability=0.8)]
    role = LegacyRole.AGENT if channels == 2 else LegacyRole.UNKNOWN
    return [TranscriptTurn(turn_id=0, speaker=role, start_time=0.0, end_time=1.0, text="Hello there.", raw_text="Hello there.", channel=0,
                           word_timestamps=words, confidence=0.9),
            TranscriptTurn(turn_id=1, speaker=LegacyRole.CALLER if channels == 2 else LegacyRole.UNKNOWN, start_time=1.2, end_time=3.0,
                           text="Hi, I need help.", raw_text="Hi, I need help.", channel=1 if channels == 2 else 0)]


def _fake_models_dir(tmp_path, monkeypatch, *dirs: Tuple[str, str]) -> None:
    root = tmp_path / "models"
    for directory, weights in dirs:
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / "config.json").write_text("{}")
        (root / directory / weights).write_bytes(b"x")
    monkeypatch.setenv("CALL1_MODELS_DIR", str(root))


@FFMPEG
def test_asr_on_mlx_wraps_the_pre_split_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    _fake_models_dir(tmp_path, monkeypatch, ("parakeet-tdt-0.6b-v3", "weights.npz"))
    seen = {}

    def transcribe(self, path, channels, agent_channel=0):
        seen.update(path=path, channels=channels, agent_channel=agent_channel, asr=self._path("ASR"))
        return _legacy_turns()

    monkeypatch.setattr("call1.adapters.mlx.MLXAdapter.transcribe", transcribe)
    audio = wav(tmp_path / "call.wav", seconds=6.0, channels=2)
    job = make_job(tmp_path, JobType.ASR, {"audio": (ArtifactKind.SOURCE_AUDIO, audio)}, parameters=JobParameters(extra={"channels": 2, "agent_channel": 1}),
                   entry_id="parakeet-tdt-0.6b-v3")
    handler = RealAsr()
    handler.ready(job)
    result = handler.run(job)
    transcript = result.outputs["transcript"].content
    assert seen["channels"] == 2 and seen["agent_channel"] == 1 and seen["asr"] == str((tmp_path / "models" / "parakeet-tdt-0.6b-v3").resolve())
    assert [t.speaker.value for t in transcript.turns] == ["AGENT", "CALLER"] and transcript.duration_seconds == pytest.approx(6.0)
    assert transcript.turns[0].word_timestamps[0].probability == 1.0  # clamped into the contract's range
    assert handler.adapter_id == "call1.mlx.parakeet" and result.model_revision is None and not transcript.is_redacted


@FFMPEG
def test_asr_without_mlx_is_faster_whisper_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr("call1.pipeline.transcriber.LocalTranscriber.transcribe", lambda self, path, channels, agent_channel=0: _legacy_turns(1))
    audio = wav(tmp_path / "mono.wav", seconds=6.0, channels=1)
    job = make_job(tmp_path, JobType.ASR, {"audio": (ArtifactKind.SOURCE_AUDIO, audio)}, entry_id="parakeet-tdt-0.6b-v3")
    result = RealAsr().run(job)
    assert {t.speaker.value for t in result.outputs["transcript"].content.turns} == {"UNKNOWN"}
    assert RealAsr().adapter_id == "call1.faster_whisper" and result.model_revision == "faster-whisper:small"

    def silent(self, path, channels, agent_channel=0):
        raise RuntimeError("No speech was transcribed; recording needs manual review.")

    monkeypatch.setattr("call1.pipeline.transcriber.LocalTranscriber.transcribe", silent)
    with pytest.raises(HandlerError) as err:
        RealAsr().run(make_job(tmp_path, JobType.ASR, {"audio": (ArtifactKind.SOURCE_AUDIO, audio)}, entry_id="parakeet-tdt-0.6b-v3"))
    assert err.value.code is JobErrorCode.VALIDATION_REJECTED


def test_asr_refuses_missing_weights_before_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    monkeypatch.setenv("CALL1_MODELS_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    job = make_job(tmp_path, JobType.ASR, {}, entry_id="parakeet-tdt-0.6b-v3")
    with pytest.raises(ReleaseJob) as err:
        RealAsr().ready(job)
    assert err.value.disposition == "reject" and err.value.code is JobErrorCode.MODEL_UNAVAILABLE


MONO = TranscriptContent(duration_seconds=10.0, is_redacted=False, turns=[
    turn(0, "UNKNOWN", 0.0, 3.0, "Thanks for calling, how can I help?", 0),
    turn(1, "UNKNOWN", 3.2, 6.0, "My card was declined twice today.", 0),
    turn(2, "UNKNOWN", 6.0, 9.0, "Let me check that.", 0),
])


@FFMPEG
def test_speaker_attribution_labels_whole_turns_with_anonymous_clusters(tmp_path, monkeypatch):
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    _fake_models_dir(tmp_path, monkeypatch, ("nemotron-3-diarization", "model.safetensors"))
    segments = [{"start": 0.0, "end": 3.1, "speaker": 0}, {"start": 3.1, "end": 6.0, "speaker": 1}, {"start": 6.0, "end": 7.0, "speaker": 0},
                {"start": 7.0, "end": 9.0, "speaker": 1}]
    calls = []
    monkeypatch.setattr("call1.adapters.mlx_diarization.diarize", lambda wav_path, model_path: calls.append(model_path) or segments)
    job = make_job(tmp_path, JobType.SPEAKER_ATTRIBUTION, {"audio": (ArtifactKind.SOURCE_AUDIO, wav(tmp_path / "m.wav", 10.0, 1)),
                                                           "transcript": (ArtifactKind.TRANSCRIPT, MONO)}, entry_id="nemotron-3-diarization")
    handler = RealSpeakerAttribution()
    handler.ready(job)
    content: SpeakerAttributionContent = handler.run(job).outputs["speaker_attribution"].content
    assert content.method == "diarization" and calls == [str((tmp_path / "models" / "nemotron-3-diarization").resolve())]
    # Decision 26: no model answer here, so the first speaker is the agent (0.6); turn 2 spans two
    # speakers, so diarization abstains and the fill rules give it the nearest neighbour's role (0.5).
    assert [(a.turn_id, a.speaker.value, a.speaker_cluster, a.confidence) for a in content.assignments] == [
        (0, "AGENT", "speaker_1", 0.6), (1, "CALLER", "speaker_2", 0.6), (2, "CALLER", None, 0.5)]


def test_speaker_attribution_without_mlx_and_reviewer_corrections(tmp_path):
    job = make_job(tmp_path, JobType.SPEAKER_ATTRIBUTION, {"transcript": (ArtifactKind.TRANSCRIPT, MONO)}, entry_id="nemotron-3-diarization")
    result = RealSpeakerAttribution().run(job)
    assert {a.speaker_cluster for a in result.outputs["speaker_attribution"].content.assignments} == {None}
    assert result.model_revision == "not-run:needs-mlx"
    corrected = make_job(tmp_path, JobType.SPEAKER_ATTRIBUTION, {"transcript": (ArtifactKind.TRANSCRIPT, MONO)},
                         parameters=JobParameters(speaker_correction=SpeakerCorrection(turn_id=1, speaker="CALLER")))
    content = RealSpeakerAttribution().run(corrected).outputs["speaker_attribution"].content
    assert content.method == "reviewer_correction" and content.corrected_turn_ids == [1]


# --- tone, sentiment, enrichment -------------------------------------------------------------


class FakeModels:
    def text(self, text):
        return {"POSITIVE": 0.7, "NEGATIVE": 0.1, "NEUTRAL": 0.2} if "Thank" in text or "Great" in text else {"POSITIVE": 0.1, "NEGATIVE": 0.6, "NEUTRAL": 0.3}

    def tone(self, samples):
        return [0.7, 0.4, 0.6], {"HAPPY": 0.8, "NEUTRAL": 0.2}


def _patch_models(monkeypatch):
    monkeypatch.setattr("call1.pipeline.sentiment.LocalSentimentModels.text", lambda self, text: FakeModels().text(text))
    monkeypatch.setattr("call1.pipeline.sentiment.LocalSentimentModels.tone", lambda self, samples: FakeModels().tone(samples))


@FFMPEG
def test_tone_blocks_match_the_pre_split_enrichment(tmp_path, monkeypatch):
    from call1.pipeline.sentiment import enrich_sentiment

    from call1.process.handlers.real.convert import legacy_turns

    _patch_models(monkeypatch)
    _fake_models_dir(tmp_path, monkeypatch, ("meralion-ser-v1", "model.safetensors"))
    audio = wav(tmp_path / "call.wav", seconds=24.0, channels=2)
    job = make_job(tmp_path, JobType.ACOUSTIC_TONE, {"audio": (ArtifactKind.SOURCE_AUDIO, audio), "transcript": (ArtifactKind.TRANSCRIPT, TRANSCRIPT)},
                   entry_id="meralion-ser-v1")
    content: ToneBlocksContent = RealAcousticTone().run(job).outputs["tone_blocks"].content
    legacy = enrich_sentiment(legacy_turns(TRANSCRIPT), audio, 2, 0, models=FakeModels())
    assert [(b.block_id, b.speaker.value, b.status.value, b.valence, b.speech_seconds) for b in content.blocks] == [
        (b.block_id, b.speaker.value, b.status, b.valence, b.speech_seconds) for b in legacy]
    assert content.blocks[0].analysis_version == "seven-second-speaker-v1" and content.avg_agent_tone == pytest.approx(0.4)


def test_text_sentiment_matches_the_pre_split_scoring(tmp_path, monkeypatch):
    _patch_models(monkeypatch)
    _fake_models_dir(tmp_path, monkeypatch, ("roberta-sentiment", "pytorch_model.bin"))
    job = make_job(tmp_path, JobType.TEXT_SENTIMENT, {"transcript": (ArtifactKind.TRANSCRIPT, TRANSCRIPT)}, entry_id="roberta-sentiment")
    RealTextSentiment().ready(job)
    content: TextSentimentContent = RealTextSentiment().run(job).outputs["text_sentiment"].content
    assert content.model == "cardiffnlp/twitter-roberta-base-sentiment-latest"
    assert [(r.turn_id, r.score, r.label.value, r.status) for r in content.turns][:2] == [(0, 0.6, "POSITIVE", "SCORED"), (1, -0.5, "NEGATIVE", "SCORED")]
    assert content.avg_caller_sentiment == pytest.approx(round((-0.5 + 0.6) / 2, 2))
    assert set(content.turns[0].probabilities) == {"POSITIVE", "NEUTRAL", "NEGATIVE"}


def test_enrichment_is_the_numeric_extractor(tmp_path):
    content = _enrich(TRANSCRIPT)
    kinds = {(row.turn_id, e.entity_type) for row in content.turns for e in row.numeric_entities}
    assert (1, "ACCOUNT_NUMBER") in kinds and (2, "CURRENCY") in kinds
    currency = next(e for row in content.turns for e in row.numeric_entities if e.entity_type == "CURRENCY")
    assert currency.normalized_value == 5.0 and isinstance(currency.normalized_value, float)


# --- QA --------------------------------------------------------------------------------------


def test_deterministic_checks_match_the_pre_split_evaluator(tmp_path, monkeypatch):
    from call1.pipeline.evaluator import RubricEvaluator

    from call1.process.handlers.real.convert import legacy_rubric, legacy_transcript

    _patch_models(monkeypatch)
    sentiment = RealTextSentiment().run(make_job(tmp_path, JobType.TEXT_SENTIMENT, {"transcript": (ArtifactKind.TRANSCRIPT, TRANSCRIPT)},
                                                 entry_id="roberta-sentiment")).outputs["text_sentiment"].content
    job = make_job(tmp_path, JobType.QA_DETERMINISTIC, {"transcript": (ArtifactKind.TRANSCRIPT, TRANSCRIPT), "text_sentiment": (ArtifactKind.TEXT_SENTIMENT, sentiment),
                                                        "rubric": (ArtifactKind.RUBRIC_SNAPSHOT, snapshot())}, parameters=JobParameters(rubric=ref()))
    verdicts = {v.criterion_id: v for v in RealQaDeterministic().run(job).outputs["verdicts"].content.verdicts}
    assert set(verdicts) == {"GREET-01", "REGEX-01", "NONE-01", "SPAN-01", "TEXT-01"}
    assert verdicts["GREET-01"].status is VerdictStatus.PASS and verdicts["GREET-01"].quote_turn_id == 0
    assert verdicts["REGEX-01"].status is VerdictStatus.PASS and verdicts["REGEX-01"].quote_turn_id == 2
    assert verdicts["NONE-01"].status is VerdictStatus.PASS
    # a phrase matched across two turns is not a verbatim quote: the grounding guardrail flags it
    assert verdicts["SPAN-01"].status is VerdictStatus.FLAGGED and verdicts["SPAN-01"].hallucination_detected
    assert verdicts["TEXT-01"].status is VerdictStatus.FAIL  # agent turns average below zero with these fake scores

    legacy = RubricEvaluator(legacy_rubric(RUBRIC)).evaluate_deterministic(legacy_transcript(TRANSCRIPT, sentiment=sentiment))
    for v in legacy.verdicts:
        if v.criterion_id in verdicts:
            ours = verdicts[v.criterion_id]
            assert (ours.status.value, ours.quoted_evidence, ours.reasoning) == (v.status.value, v.quoted_evidence, v.reasoning)

    # the scorecard names the guardrail rejection, as evaluate_deterministic did
    from call1.process.handlers.code import score

    _, _, _, _, review, reasons = score([c for c in RUBRIC.criteria if c.criterion_id in verdicts], verdicts, 80.0)
    assert review and any(r.startswith("Guardrail rejection on SPAN-01") for r in reasons)


def test_qa_pass_is_quote_verified_against_the_text_the_model_saw(tmp_path, monkeypatch):
    fake = FakeGenerate(answer("pass", "This call may be recorded for quality assurance."))
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    job = qa_job(tmp_path, "REG-01")
    RealQaCriterion().ready(job)
    result = RealQaCriterion().run(job)
    assessment = result.outputs["assessment"].content
    assert assessment.status is VerdictStatus.PASS and assessment.trigger is None and assessment.assessment_kind == "primary"
    assert assessment.quote_turn_id == 0 and assessment.timestamp_range == (0.0, 4.0) and not assessment.hallucination_detected
    assert assessment.attempt.catalog_entry_id == "call1-bundled" and assessment.attempt.tokens_input is None  # MLX reports no tokens
    call = fake.calls[0]
    assert call["model"].id == "call1-bundled" and call["schema_name"] == "qa_answer" and call["allow_external"] is False
    assert '"pass_when": "The agent discloses the recording."' in call["prompt"]  # RubricEvaluator._semantic_prompt
    prompt = result.outputs["prompt_input"].content
    assert prompt.masked is False and prompt.template_id == "call1.qa.semantic_judgement"
    assert prompt.prompt_digest == canonical_digest([[{"role": "system", "content": call["system"]}, {"role": "user", "content": call["prompt"]}]])
    assert result.usage.tokens_input is None


@pytest.mark.parametrize("reply, trigger, hallucination", [
    (answer("needs_review"), EscalationTrigger.NEEDS_REVIEW, False),
    (answer("pass", "We never record anything."), EscalationTrigger.INVALID_ANSWER, True),
    ("I think it passes", EscalationTrigger.INVALID_ANSWER, False),
    (answer("fail", ""), EscalationTrigger.INVALID_ANSWER, False),
])
def test_qa_answers_that_cannot_be_trusted_are_flagged_outcomes(tmp_path, monkeypatch, reply, trigger, hallucination):
    monkeypatch.setattr("call1.question_models.generate_text", FakeGenerate(reply))
    assessment = RealQaCriterion().run(qa_job(tmp_path, "REG-01")).outputs["assessment"].content
    assert assessment.status is VerdictStatus.FLAGGED and assessment.trigger is trigger
    assert assessment.hallucination_detected is hallucination and assessment.attempt.trigger is trigger


def test_qa_gates_flag_without_calling_a_model(tmp_path, monkeypatch):
    fake = FakeGenerate(answer("pass", "x"))
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    result = RealQaCriterion().run(qa_job(tmp_path, "SEC-01"))
    assessment = result.outputs["assessment"].content
    assert fake.calls == [] and assessment.status is VerdictStatus.FLAGGED and assessment.trigger is None
    assert "requires a configured business policy" in assessment.reasoning
    assert result.outputs["prompt_input"].content.prompt_digest == canonical_digest([])
    unknown = TranscriptContent(duration_seconds=5, is_redacted=False, turns=[turn(0, "UNKNOWN", 0, 4, "This call may be recorded.")])
    assessment = RealQaCriterion().run(qa_job(tmp_path, "REG-01", transcript=unknown)).outputs["assessment"].content
    assert fake.calls == [] and "Speaker identity is unknown" in assessment.reasoning


def test_qa_provider_failures_and_the_pro1_closure_are_errors_not_verdicts(tmp_path, monkeypatch):
    from call1.question_models import PRO1_UNAVAILABLE

    monkeypatch.setattr("call1.question_models.generate_text", FakeGenerate(RuntimeError("Model request failed; check credentials")))
    with pytest.raises(HandlerError) as err:
        RealQaCriterion().run(qa_job(tmp_path, "REG-01"))
    assert err.value.code is JobErrorCode.PROVIDER_ERROR and "prompt_input" in err.value.outputs
    monkeypatch.setattr("call1.question_models.generate_text", FakeGenerate(RuntimeError(PRO1_UNAVAILABLE)))
    with pytest.raises(HandlerError) as err:
        RealQaCriterion().run(qa_job(tmp_path, "REG-01"))
    assert err.value.code is JobErrorCode.ROUTE_POLICY_REJECTED
    # Pro1 stays closed before inference, whatever the selection says
    confidential = SimpleNamespace(selection=SimpleNamespace(route=SimpleNamespace(route_class=RouteClass.CALL1_CONFIDENTIAL)))
    with pytest.raises(ReleaseJob) as release:
        check_route(confidential)  # type: ignore[arg-type]
    assert release.value.code is JobErrorCode.ROUTE_POLICY_REJECTED
    # the real Pro1 guard itself is the one generate_text applies
    from call1.models.schemas import QuestionModel

    pro1 = QuestionModel(id="pro1", name="Pro1", source="pro1", model="m", endpoint="https://api.call1.cc")
    with pytest.raises(RuntimeError, match="Pro1 inference is unavailable"):
        REAL_GENERATE_TEXT(pro1, "s", "customer transcript", True)


def test_qa_context_overflow_is_a_flagged_assessment_as_before_the_split(tmp_path, monkeypatch):
    """Judge condition 7: the pre-split router FLAGGED a criterion whose prompt overflowed the
    model's context (reason provider_error) instead of failing, so the scorecard still publishes."""
    monkeypatch.setattr("call1.question_models.generate_text",
                        FakeGenerate(RuntimeError("Input exceeds the appliance context budget; split into smaller chunks.")))
    for handler, job_type, kind in ((RealQaCriterion(), JobType.QA_CRITERION, "primary"), (RealQaEscalation(), JobType.QA_ESCALATION, "escalation")):
        result = handler.run(qa_job(tmp_path, "REG-01", job_type=job_type))
        assessment = result.outputs["assessment"].content
        assert assessment.status is VerdictStatus.FLAGGED and assessment.assessment_kind == kind
        assert assessment.trigger is EscalationTrigger.PROVIDER_ERROR and assessment.confidence == 0.0
        assert assessment.attempt.error_code is JobErrorCode.CONTEXT_LIMIT_EXCEEDED and assessment.attempt.status is VerdictStatus.FLAGGED
        assert "context budget" in assessment.reasoning and assessment.quoted_evidence is None
        assert result.outputs["prompt_input"].content.prompt_digest.startswith("sha256:")
        QaAssessmentContent.model_validate(assessment.model_dump(mode="json"))  # the contract accepts it


def test_qa_cancellation_is_not_a_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr("call1.question_models.generate_text", FakeGenerate(answer("pass", "x")))
    job = qa_job(tmp_path, "REG-01")
    job._cancel.set()
    with pytest.raises(JobCancelled):
        RealQaCriterion().run(job)


def test_qa_on_a_masked_route_masks_the_prompt_and_verifies_against_it(tmp_path, monkeypatch):
    # the extractor's account entity includes the word "account", so the whole phrase is masked
    fake = FakeGenerate(answer("pass", "I was charged a fee on [REDACTED] and I want it removed.", "ok"))
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    rubric = RubricDefinition(rubric_id="masked", name="Masked", criteria=[criterion("ISSUE-01", SEMANTIC.model_copy(update={"speaker": SpeakerRole.CALLER}))])
    result = RealQaEscalation().run(qa_job(tmp_path, "ISSUE-01", job_type=JobType.QA_ESCALATION, masked=True, rubric=rubric))
    assert "1234-5678-9012" not in fake.calls[0]["prompt"] and "[REDACTED]" in fake.calls[0]["prompt"]
    assessment = result.outputs["assessment"].content
    assert assessment.status is VerdictStatus.PASS and assessment.quote_turn_id == 1 and assessment.assessment_kind == "escalation"
    assert assessment.speaker is SpeakerRole.CALLER and result.outputs["prompt_input"].content.masked is True


def test_masked_qa_without_a_pinned_enrichment_extracts_the_same_entities_in_process(tmp_path, monkeypatch):
    """Appliance text masking adds no graph edge: with no ``enrichment`` input the handler runs the
    enrichment stage's own extractor, and the model sees exactly the same masked prompt."""
    prompts = []
    for pinned in (True, False):
        fake = FakeGenerate(answer("pass", "Thank you for calling. This call may be recorded for quality assurance.", "ok"))
        monkeypatch.setattr("call1.question_models.generate_text", fake)
        job = qa_job(tmp_path, "REG-01", masked=True)
        if not pinned:
            job.inputs.pop("enrichment")
        assert (job.input("enrichment") is not None) is pinned
        result = RealQaCriterion().run(job)
        assert result.outputs["assessment"].content.status is VerdictStatus.PASS
        prompts.append(fake.calls[0]["prompt"])
    assert prompts[0] == prompts[1] and "1234-5678-9012" not in prompts[1] and "[REDACTED]" in prompts[1]
    enrichment = make_job(tmp_path, JobType.ENRICHMENT, {"transcript": (ArtifactKind.TRANSCRIPT, TRANSCRIPT)})
    assert RealEnrichment().run(enrichment).outputs["enrichment"].content == _enrich(TRANSCRIPT)


# --- summary ---------------------------------------------------------------------------------


def summary(narrative: str, *points: str) -> str:
    return json.dumps({"narrative": narrative, "key_points": list(points)})


def seg_job(tmp_path, index: int, window: TurnWindow, count: int, **kw) -> HandlerJob:
    return make_job(tmp_path, JobType.SUMMARY_SEGMENT, {"transcript": (ArtifactKind.TRANSCRIPT, TRANSCRIPT)},
                    parameters=JobParameters(segment=SegmentSpec(index=index, window=window), extra={"segment_count": count}),
                    entry_id="call1-bundled", purpose=ModelPurpose.SUMMARY, **kw)


def test_a_single_segment_is_the_full_call_prompt_with_checked_citations(tmp_path, monkeypatch):
    fake = FakeGenerate("{not json", summary("The caller disputed a fee and the agent reversed it.",
                                             "Agent reversed the monthly fee today (turn 2)", "Caller confirmed the fix worked (turn 3)",
                                             "Caller asked about the weather (turn 4)", "No citation here"))
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    result = RealSummarySegment().run(seg_job(tmp_path, 0, TurnWindow(turn_start=0, turn_end=4), 1))
    content: SummarySegmentContent = result.outputs["segment"].content
    assert len(fake.calls) == 2 and fake.calls[0]["prompt"].startswith("This is the full call transcript.")  # malformed once, retried
    assert "[2] AGENT: I can help with that." in fake.calls[0]["prompt"] and fake.calls[0]["schema_name"] == "call_summary"
    assert content.key_points == ["Agent reversed the monthly fee today (turn 2)"]  # "fixes it" shares <2 words; weather is ungrounded
    assert [(c.claim, c.index, c.turn_ids) for c in content.citations] == [("key_point", 0, [2])]
    prompt = result.outputs["prompt_input"].content
    assert prompt.transcript_window == TurnWindow(turn_start=0, turn_end=4)
    assert prompt.prompt_digest == canonical_digest([[{"role": "system", "content": c["system"]}, {"role": "user", "content": c["prompt"]}]
                                                     for c in fake.calls])  # both requests of the attempt
    monkeypatch.setattr("call1.question_models.generate_text", FakeGenerate(summary("n", "Nothing grounded (turn 9)")))
    with pytest.raises(HandlerError) as err:
        RealSummarySegment().run(seg_job(tmp_path, 0, TurnWindow(turn_start=0, turn_end=4), 1))
    assert err.value.code is JobErrorCode.VALIDATION_REJECTED and err.value.detail == "No key points survived citation checking."


def test_segments_synthesis_and_assembly_follow_the_pre_split_reduction(tmp_path, monkeypatch):
    fake = FakeGenerate(summary("Opening and dispute.", "Agent said the call may be recorded for quality (turn 0)"),
                        summary("Resolution.", "Agent reversed the monthly fee of $5.00 (turn 2)"))
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    first = RealSummarySegment().run(seg_job(tmp_path, 0, TurnWindow(turn_start=0, turn_end=1), 2)).outputs["segment"].content
    second = RealSummarySegment().run(seg_job(tmp_path, 1, TurnWindow(turn_start=2, turn_end=4), 2)).outputs["segment"].content
    assert fake.calls[0]["prompt"].startswith("This is segment 1 of 2 of the call transcript.")
    # the final synthesis loses the citations: the assembly falls back to the checked source points
    fake = FakeGenerate(summary("The caller disputed a fee; the agent reversed it.", "Fee reversed"))
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    parts = {"part:0": (ArtifactKind.SUMMARY_SEGMENT, first), "part:1": (ArtifactKind.SUMMARY_SEGMENT, second)}
    job = make_job(tmp_path, JobType.SUMMARY_SYNTHESIS, parts, parameters=JobParameters(extra={"final": True}), entry_id="call1-bundled",
                   purpose=ModelPurpose.SUMMARY)
    synthesis: SummarySynthesisContent = RealSummarySynthesis().run(job).outputs["synthesis"].content
    assert fake.calls[0]["prompt"].startswith("SEGMENT 1:\n{") and "\n\nSEGMENT 2:\n" in fake.calls[0]["prompt"]
    assert synthesis.final and synthesis.segment_indexes == [0, 1] and synthesis.citations == []
    pairwise = make_job(tmp_path, JobType.SUMMARY_SYNTHESIS, parts, parameters=JobParameters(extra={"final": False}), entry_id="call1-bundled",
                        purpose=ModelPurpose.SUMMARY)
    RealSummarySynthesis().run(pairwise)
    assert json.loads(fake.calls[1]["prompt"])[1]["narrative"] == "Resolution."

    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, TRANSCRIPT), "segment:0": (ArtifactKind.SUMMARY_SEGMENT, first),
              "segment:1": (ArtifactKind.SUMMARY_SEGMENT, second), "synthesis": (ArtifactKind.SUMMARY_SYNTHESIS, synthesis)}
    assembly = make_job(tmp_path, JobType.SUMMARY_ASSEMBLY, inputs, parameters=JobParameters(extra={"summary_entry": "call1-bundled", "route_class": "appliance"}))
    content = RealSummaryAssembly().run(assembly).outputs["summary"].content
    assert content.narrative == "The caller disputed a fee; the agent reversed it." and content.segments == 2
    assert content.key_points == ["Agent said the call may be recorded for quality (turn 0)", "Agent reversed the monthly fee of $5.00 (turn 2)"]
    assert content.grounding["key_points_from_source_segments"] is True and content.grounding["chunked"] is True
    assert [c.turn_ids for c in content.citations] == [[0], [2]] and content.rubric_highlights == []


def test_an_assembly_with_nothing_grounded_fails(tmp_path):
    empty = SummarySegmentContent(segment_index=0, window=TurnWindow(turn_start=0, turn_end=4), narrative="n", key_points=["Unrelated (turn 3)"])
    job = make_job(tmp_path, JobType.SUMMARY_ASSEMBLY, {"transcript": (ArtifactKind.TRANSCRIPT, TRANSCRIPT), "segment:0": (ArtifactKind.SUMMARY_SEGMENT, empty)})
    with pytest.raises(HandlerError) as err:
        RealSummaryAssembly().run(job)
    assert err.value.code is JobErrorCode.VALIDATION_REJECTED


# --- contact signals -------------------------------------------------------------------------


def _pass_job(tmp_path, job_type, kind):
    return make_job(tmp_path, job_type, {"transcript": (ArtifactKind.TRANSCRIPT, TRANSCRIPT)}, parameters=JobParameters(pass_kind=kind),
                    entry_id="call1-bundled", purpose=ModelPurpose.CONTACT_SIGNALS)


def test_contact_signal_passes_keep_only_quote_verified_observations(tmp_path, monkeypatch):
    from call1.pipeline.contact_signals import LIFECYCLE_SCHEMA, build_transcript_block, pass_prompt

    from call1.process.handlers.real.convert import legacy_turns

    observations = {"observations": [
        {"kind": "issue", "confidence": 0.9, "evidence_turn": 1, "evidence_quote": "I was charged a fee"},
        {"kind": "issue", "confidence": 0.9, "evidence_turn": 1, "evidence_quote": "I was charged a fee"},  # duplicate span
        {"kind": "intent", "confidence": 1.4, "evidence_turn": 1, "evidence_quote": "I want it removed"},
        {"kind": "intent", "confidence": 0.9, "evidence_turn": 0, "evidence_quote": "Thank you for calling"},  # agent turn: wrong speaker
        {"kind": "friction", "confidence": 0.9, "evidence_turn": 3, "evidence_quote": "this is still broken"},  # not in the turn
    ]}
    fake = FakeGenerate(json.dumps(observations))
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    result = RealLifecyclePass().run(_pass_job(tmp_path, JobType.CONTACT_SIGNALS_LIFECYCLE, ContactSignalPass.LIFECYCLE))
    content = result.outputs["pass"].content
    assert [(s.kind.value, s.turn_id, s.quote, s.speaker.value, s.confidence) for s in content.signals] == [
        ("issue", 1, "I was charged a fee", "CALLER", 0.9), ("intent", 1, "I want it removed", "CALLER", 1.0)]
    assert content.signals[0].id == "issue-t1-c4" and content.pass_kind is ContactSignalPass.LIFECYCLE
    _, block = build_transcript_block(legacy_turns(TRANSCRIPT))
    system, user, schema = pass_prompt("lifecycle", 5, block)
    assert (fake.calls[0]["system"], fake.calls[0]["prompt"], fake.calls[0]["schema"], fake.calls[0]["max_tokens"]) == (system, user, LIFECYCLE_SCHEMA, 512)
    assert schema is LIFECYCLE_SCHEMA

    monkeypatch.setattr("call1.question_models.generate_text", FakeGenerate("not json"))
    with pytest.raises(HandlerError) as err:
        RealResolutionPass().run(_pass_job(tmp_path, JobType.CONTACT_SIGNALS_RESOLUTION, ContactSignalPass.RESOLUTION))
    assert err.value.code is JobErrorCode.VALIDATION_REJECTED
    monkeypatch.setattr("call1.question_models.generate_text", FakeGenerate(RuntimeError("Model request failed")))
    with pytest.raises(HandlerError) as err:
        RealResolutionPass().run(_pass_job(tmp_path, JobType.CONTACT_SIGNALS_RESOLUTION, ContactSignalPass.RESOLUTION))
    assert err.value.code is JobErrorCode.PROVIDER_ERROR


def test_llm_jobs_on_mlx_pass_the_catalog_weights_to_the_legacy_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("CALL1_BACKEND", "mlx")
    _fake_models_dir(tmp_path, monkeypatch, ("gemma-4-e2b-it", "model.safetensors"))
    fake = FakeGenerate(json.dumps({"observations": []}))
    monkeypatch.setattr("call1.question_models.generate_text", fake)
    RealResolutionPass().run(_pass_job(tmp_path, JobType.CONTACT_SIGNALS_RESOLUTION, ContactSignalPass.RESOLUTION))
    assert fake.calls[0]["text_model_path"] == str((tmp_path / "models" / "gemma-4-e2b-it").resolve())

    # and generate_text hands that path to the MLX adapter for the included model
    from call1.models.schemas import QuestionModelsSettings

    bundled = QuestionModelsSettings().models[0]
    with patch("call1.adapters.get_adapter") as adapter:
        adapter.return_value.generate.return_value = "{}"
        REAL_GENERATE_TEXT(bundled, "s", "p", False, None, "x", 64, text_model_path="/weights")
    assert adapter.return_value.generate.call_args.kwargs["text_model_path"] == "/weights"


def test_a_broken_real_handler_module_never_stops_process(monkeypatch):
    import call1.process.handlers.real as real

    def broken(config=None):
        raise ImportError("no runtime")

    monkeypatch.setattr(real, "real_handlers", broken)
    registry = build_registry("real", config=None, catalog=CATALOG)
    assert JobType.ASR in registry.missing() and JobType.QA_SCORECARD not in registry.missing()
    assert any("failed to load (ImportError)" in note for note in registry.notes)
