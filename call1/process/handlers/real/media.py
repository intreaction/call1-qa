"""Media stages: audio validation and VAD, ASR, and mono speaker attribution.

They wrap the pre-split modules unchanged: ``call1.pipeline.audio_validation.AudioValidator`` and
``call1.pipeline.audio_processor.VoiceActivityDetector`` (validation and VAD),
``call1.adapters.mlx.MLXAdapter.transcribe`` (Parakeet TDT 0.6B v3) or
``call1.pipeline.transcriber.LocalTranscriber`` (faster-whisper) for ASR, and
``call1.adapters.mlx_diarization.diarize`` (Nemotron-3-Diarization) with the pre-split cluster rule for speaker
attribution. Model runtimes are imported inside ``run``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from call1.contracts.contents import (
    AudioChannelLayout,
    AudioValidationContent,
    SpeakerAssignment,
    SpeakerAttributionContent,
    SpeakerRole,
    TranscriptContent,
    VadMetricsContent,
    VadSegmentContent,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType

from call1.process.audio import UnsupportedAudio, probe
from call1.process.handlers.base import Handler, HandlerError, HandlerJob, HandlerResult, Output, ReleaseJob, Usage
from call1.process.transcripts import apply_speaker_correction

from call1.pipeline.speaker_roles import ROLE_AGENT, ROLE_CALLER, ROLE_CONFIDENCE

from .convert import contract_turn
from .paths import mlx_backend, peak_memory, reset_peak_memory, weights_installed, weights_path

log = logging.getLogger("call1.process.handlers.real")

ADAPTER_VERSION = "1"


def _agent_channel(job: HandlerJob) -> int:
    value = job.parameters.extra.get("agent_channel")
    return int(value) if value in (0, 1) else 0


def _validate(path: Path):
    """The pre-split validation gates; a rejected recording fails the attempt definitively."""
    from call1.pipeline.audio_validation import AudioValidator
    from call1.models.schemas import AudioValidationStatus

    result = AudioValidator().validate(path)
    if result.is_valid:
        return result
    if result.status is AudioValidationStatus.FILE_NOT_FOUND:
        raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, "the recording was not available in scratch")
    detail = {
        AudioValidationStatus.DISCARD_TOO_SHORT: f"The recording is shorter than the minimum ({result.duration_seconds:.2f}s).",
        AudioValidationStatus.EXCEEDS_MAX_DURATION: f"The recording is longer than the maximum ({result.duration_seconds:.0f}s).",
        AudioValidationStatus.UNSUPPORTED_FORMAT: f"Unsupported codec '{result.codec}'.",
        AudioValidationStatus.CORRUPTED: "The audio container is corrupted or its header is unreadable.",
    }.get(result.status, "The recording failed validation.")
    raise HandlerError(JobErrorCode.VALIDATION_REJECTED, f"{result.status.value}: {detail}")


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise ReleaseJob("reject", JobErrorCode.CONFIGURATION_ERROR, "ffmpeg is not installed on this Process host")


class RealValidationVad(Handler):
    """``validation_vad``: the pre-split validation gates (ffprobe, codec whitelist, 5 s to 90 min)
    and the energy VAD over the decoded channels (speech, silence and overtalk)."""

    job_type = JobType.VALIDATION_VAD
    adapter_id = "call1.pipeline.validation_vad"
    adapter_version = ADAPTER_VERSION

    def ready(self, job: HandlerJob) -> None:
        _require_ffmpeg()

    def run(self, job: HandlerJob) -> HandlerResult:
        from call1.pipeline.audio_processor import VoiceActivityDetector

        audio = job.require("audio")
        path = audio.path()
        try:
            info = probe(path, content_type=audio.artifact.content_type)
        except UnsupportedAudio as exc:
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, str(exc)) from None
        result = _validate(path)
        job.check_cancelled()
        try:
            metrics = VoiceActivityDetector().analyze(path)
        except subprocess.CalledProcessError:
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "The recording could not be decoded for VAD.") from None
        except ValueError:
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "VAD needs nonempty mono or stereo audio.") from None
        layout = AudioChannelLayout(result.channel_layout.value)
        report = AudioValidationContent(
            container=info.container, codec=(result.codec or "unknown")[:200], sample_rate=max(1, int(result.sample_rate)),
            channels=max(1, int(result.channels)), channel_layout=layout, duration_seconds=round(float(result.duration_seconds), 3),
            agent_channel=_agent_channel(job) if result.channels == 2 else None, warnings=[],
        )
        vad = VadMetricsContent(
            total_speech_duration=max(0.0, metrics.total_speech_duration), total_silence_duration=max(0.0, metrics.total_silence_duration),
            silence_ratio=min(1.0, max(0.0, metrics.silence_ratio)), overtalk_duration=max(0.0, metrics.overtalk_duration),
            overtalk_ratio=min(1.0, max(0.0, metrics.overtalk_ratio)),
            segments=[VadSegmentContent(start_time=max(0.0, s.start_time), end_time=max(0.0, s.end_time), channel=max(0, int(s.channel)))
                      for s in metrics.segments],
        )
        return HandlerResult(outputs={"validation_report": Output(report), "vad_metrics": Output(vad)},
                             usage=Usage(audio_seconds_processed=report.duration_seconds))


def _pathed_adapter(paths: Dict[str, Optional[Path]]):
    """The pre-split MLX adapter reading its weights from this Process's catalog paths."""
    from call1.adapters.mlx import MLXAdapter

    class CatalogMLXAdapter(MLXAdapter):
        def _path(self, kind: str) -> str:
            path = paths.get(kind)
            if path is None:
                return super()._path(kind)
            if not (Path(path) / "config.json").is_file():
                raise RuntimeError(f"Provision the {kind} model before processing recordings.")
            return str(Path(path).resolve())

    return CatalogMLXAdapter()


class RealAsr(Handler):
    """``asr``: Parakeet TDT 0.6B v3 on Apple Silicon (``CALL1_BACKEND=mlx``), else faster-whisper, one
    channel at a time; stereo channels are labelled AGENT/CALLER by the registered agent channel,
    mono turns stay UNKNOWN until speaker attribution. Word timestamps are kept.

    With ``parameters.asr_vocabulary`` (dual transcription, decision 33) Parakeet's transcript then
    goes through the vocabulary pass and rule merge (``handlers/real/vocabulary.py``): the job returns
    the merged ``transcript``, Parakeet's own as ``base_transcript``, and the raw Whisper pass as
    ``vocabulary_pass`` when it ran. A vocabulary problem never fails the job. ``ready`` still rejects
    only on missing Parakeet weights."""

    job_type = JobType.ASR
    adapter_version = ADAPTER_VERSION
    catalog = None
    """The Process catalog (set by ``register``): resolves the frozen vocabulary-pass entry."""

    @property
    def adapter_id(self) -> str:  # type: ignore[override]
        return "call1.mlx.parakeet" if mlx_backend() else "call1.faster_whisper"

    def ready(self, job: HandlerJob) -> None:
        _require_ffmpeg()
        if mlx_backend() and job.catalog_entry is not None and not weights_installed(job.catalog_entry):
            raise ReleaseJob("reject", JobErrorCode.MODEL_UNAVAILABLE, "the ASR weights are not installed on this host")

    def run(self, job: HandlerJob) -> HandlerResult:
        audio = job.require("audio")
        path = audio.path()
        validation = _validate(path)
        channels = int(validation.channels)
        if channels not in (1, 2):
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "Only mono or dual-channel recordings are supported.")
        agent_channel = _agent_channel(job)
        job.check_cancelled()
        job.progress(0.05, "transcribing")
        started = time.monotonic()
        model_revision = None
        try:
            if mlx_backend():
                entry = job.catalog_entry
                adapter = _pathed_adapter({"ASR": weights_path(entry) if entry is not None else None})
                reset_peak_memory()
                turns = adapter.transcribe(str(path), channels, agent_channel)
            else:
                from call1.pipeline.transcriber import LocalTranscriber

                turns = LocalTranscriber().transcribe(str(path), channels, agent_channel)
                model_revision = f"faster-whisper:{os.getenv('CALL1_ASR_MODEL', 'small')}"[:200]
        except subprocess.CalledProcessError:
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "The recording could not be decoded for transcription.") from None
        except ImportError:
            raise HandlerError(JobErrorCode.MODEL_UNAVAILABLE, "the ASR runtime is not installed on this host") from None
        except RuntimeError as exc:
            message = str(exc)
            if "No speech" in message:
                raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "No speech was transcribed; the recording needs manual review.") from None
            if "Provision" in message or "Install" in message:
                raise HandlerError(JobErrorCode.MODEL_UNAVAILABLE, "the ASR model is not installed on this host") from None
            if "decode" in message:
                raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "The recording could not be decoded for transcription.") from None
            raise
        except ValueError as exc:
            raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, f"ASR configuration: {str(exc)[:200]}") from None
        elapsed = time.monotonic() - started
        # Parakeet picks the language itself and does not report it, so the MLX transcript names none;
        # CALL1_ASR_LANGUAGE applies only to faster-whisper.
        language = None if mlx_backend() else (os.getenv("CALL1_ASR_LANGUAGE") or None)
        transcript = TranscriptContent(duration_seconds=round(float(validation.duration_seconds), 3),
                                       language=language, is_redacted=False,
                                       turns=[contract_turn(t) for t in turns])
        outputs = {"transcript": Output(transcript)}
        peak = peak_memory()
        load_seconds = None
        if job.parameters.asr_vocabulary is not None:
            from .vocabulary import correct

            job.check_cancelled()
            entry = job.catalog_entry
            base_engine = (entry.entry_id if entry is not None else "parakeet-tdt-0.6b-v3") if mlx_backend() else (model_revision or "faster-whisper")
            extra = correct(job, transcript, path, channels, base_engine=base_engine, catalog=self.catalog)
            outputs = extra.outputs
            elapsed = max(elapsed, time.monotonic() - started - extra.load_seconds)  # both passes (usage covers both)
            load_seconds = round(extra.load_seconds, 6) if extra.load_seconds else None
            if extra.peak_memory_bytes is not None:
                peak = max(peak or 0, extra.peak_memory_bytes)
        return HandlerResult(outputs=outputs, model_revision=model_revision,
                             usage=Usage(audio_seconds_processed=transcript.duration_seconds, inference_seconds=round(elapsed, 6),
                                         model_load_seconds=load_seconds, peak_memory_bytes=peak))


ROLE_SPEAKER = {ROLE_AGENT: SpeakerRole.AGENT, ROLE_CALLER: SpeakerRole.CALLER}

def roles_prompt(job: HandlerJob, transcript: TranscriptContent, clusters: Dict[int, Optional[str]]):
    """(system, user, alias -> cluster, response schema): the speaker-roles prompt ``infer_roles``
    sends (decision 26), over the masked turns: the number rules plus the PII model's findings
    (``masking.sensitive_values`` with the job: the pinned ``pii_findings`` when the job has them for
    this transcript revision, else the model runs here). On-device training rebuilds its examples
    with this."""
    from call1.pipeline.speaker_roles import ROLES_SYSTEM, render_prompt

    from .masking import mask, sensitive_values

    values = sensitive_values(transcript.turns, job=job)
    turns = [(t.turn_id, mask(t.text, values)) for t in transcript.turns]
    prompt, cluster_of, schema = render_prompt(turns, clusters)
    return ROLES_SYSTEM, prompt, cluster_of, schema


def infer_roles(job: HandlerJob, transcript: TranscriptContent, clusters: Dict[int, Optional[str]],
                adapters: Optional[List[str]] = None) -> Optional[Dict[str, str]]:
    """Cluster -> agent/caller/neither from the included model (decision 26), or None to leave every
    turn UNKNOWN. The model sees only masked text: the number rules plus the PII model's findings
    (``masking.sensitive_values`` with the job, which runs the PII model here because enrichment has
    not run yet). One constrained prompt; any failure is logged and leaves the roles unassigned.
    ``adapters`` collects the on-device adapter version that answered, for provenance."""
    from call1.pipeline.speaker_roles import OUTPUT_TOKENS, parse_roles
    from call1.process.catalog import BUNDLED_LLM_ENTRY_ID, seeded_catalog

    from .llm import LlmTransport
    from .signals_v2 import _EntryJob

    if len({c for c in clusters.values() if c is not None}) < 2:
        return None
    try:
        entry = seeded_catalog(mode="real").get(BUNDLED_LLM_ENTRY_ID)
        system, prompt, cluster_of, schema = roles_prompt(job, transcript, clusters)
        job.check_cancelled()
        transport = LlmTransport(_EntryJob(job, entry))
        answer = transport.generate(system, prompt, response_schema=schema, schema_name="speaker_roles", max_tokens=OUTPUT_TOKENS)
        if adapters is not None and transport.adapter_version:
            adapters.append(transport.adapter_version)
        roles = parse_roles(answer.raw, cluster_of)
    except (HandlerError, ReleaseJob, KeyError, ValueError, RuntimeError, OSError) as exc:
        log.warning("job %s: speaker roles not inferred (%s); turns stay unattributed", job.job.id, type(exc).__name__)
        return None
    if roles is None:
        log.warning("job %s: the model's speaker roles were unusable; turns stay unattributed", job.job.id)
    return roles


def attribute_turns(job: HandlerJob, transcript: TranscriptContent, clusters: Dict[int, Optional[str]],
                    adapters: Optional[List[str]] = None) -> List[SpeakerAssignment]:
    """Every turn gets agent or caller (decision 26): the model's cluster roles (confidence 0.8), else
    the first speaker as the agent (0.6), and the fill rules for turns with no cluster or a
    "neither" cluster (0.5; ``speaker_roles.fill_roles``). A reviewer's correction still overrides."""
    from call1.pipeline.speaker_roles import FILLED_CONFIDENCE, FIRST_SPEAKER_CONFIDENCE, fill_roles, first_speaker_roles

    turn_ids = [t.turn_id for t in transcript.turns]
    roles = infer_roles(job, transcript, clusters, adapters) if clusters else None
    confidence = ROLE_CONFIDENCE
    if roles is None:
        roles, confidence = first_speaker_roles(clusters, turn_ids), FIRST_SPEAKER_CONFIDENCE
    known = {t: roles[c] for t, c in clusters.items() if c is not None and roles.get(c) in ROLE_SPEAKER}
    filled = fill_roles([(t.turn_id, t.text or "", t.start_time, t.end_time) for t in transcript.turns], known)
    out = []
    for t in transcript.turns:
        role = known.get(t.turn_id) or filled.get(t.turn_id)
        out.append(SpeakerAssignment(turn_id=t.turn_id, speaker=ROLE_SPEAKER[role], speaker_cluster=clusters.get(t.turn_id),
                                     confidence=confidence if t.turn_id in known else FILLED_CONFIDENCE))
    return out


class RealSpeakerAttribution(Handler):
    """``speaker_attribution`` on a mono call: Nemotron-3-Diarization clusters (up to eight) on the normalized 16 kHz audio,
    one anonymous cluster per whole turn (``call1.pipeline.diarization.turn_clusters``). The included
    model then names each cluster agent, caller or neither from the masked opening of the call
    (decision 26, ``infer_roles``). Every turn then gets a role (``attribute_turns``): the first
    speaker is the agent when the model gives no usable answer, and turns with no cluster are filled
    from what they say and their neighbours. A speaker-correction job applies the reviewer's correction instead (no model)."""

    job_type = JobType.SPEAKER_ATTRIBUTION
    adapter_version = ADAPTER_VERSION

    @property
    def adapter_id(self) -> str:  # type: ignore[override]
        return "call1.mlx.nemotron_diarization" if mlx_backend() else "call1.speakers.unattributed"

    def ready(self, job: HandlerJob) -> None:
        if job.parameters.speaker_correction is not None:
            return
        if mlx_backend():
            _require_ffmpeg()
            if job.catalog_entry is not None and not weights_installed(job.catalog_entry):
                raise ReleaseJob("reject", JobErrorCode.MODEL_UNAVAILABLE, "the diarization weights are not installed on this host")

    def run(self, job: HandlerJob) -> HandlerResult:
        transcript: TranscriptContent = job.require("transcript").content()  # type: ignore[assignment]
        correction = job.parameters.speaker_correction
        if correction is not None:
            try:
                content = apply_speaker_correction(transcript, job.attribution(), correction)
            except ValueError as exc:
                raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, str(exc)[:200]) from None
            return HandlerResult(outputs={"speaker_attribution": Output(content)})
        clusters: Dict[int, Optional[str]] = {}
        model_revision = None
        started = time.monotonic()
        if mlx_backend():
            from call1.adapters.mlx_diarization import diarize
            from call1.pipeline.diarization import turn_clusters

            from .convert import legacy_turns

            entry = job.catalog_entry
            model_path = weights_path(entry) if entry is not None else None
            audio = job.require("audio")
            wav = job.scratch_dir / "diarization-mono.wav"
            try:
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(audio.path()), "-ac", "1", "-ar", "16000",
                                "-c:a", "pcm_s16le", str(wav)], check=True, capture_output=True, timeout=180)
            except subprocess.CalledProcessError:
                raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "The recording could not be decoded for diarization.") from None
            job.check_cancelled()
            reset_peak_memory()
            try:
                segments = diarize(str(wav), str(Path(model_path).resolve()) if model_path else "data/models/nemotron-3-diarization")
            except ImportError:
                raise HandlerError(JobErrorCode.MODEL_UNAVAILABLE, "the diarization runtime is not installed on this host") from None
            clusters = turn_clusters(legacy_turns(transcript), segments)
        else:
            log.warning("job %s: diarization needs CALL1_BACKEND=mlx; mono turns stay unattributed", job.job.id)
            model_revision = "not-run:needs-mlx"
        adapters: List[str] = []
        assignments = attribute_turns(job, transcript, clusters, adapters) if mlx_backend() else [
            SpeakerAssignment(turn_id=t.turn_id, speaker=SpeakerRole.UNKNOWN, speaker_cluster=None) for t in transcript.turns]
        if adapters:
            from .llm import lora_revision

            model_revision = lora_revision(job, adapters[0])
        return HandlerResult(outputs={"speaker_attribution": Output(SpeakerAttributionContent(method="diarization", assignments=assignments))},
                             model_revision=model_revision,
                             usage=Usage(audio_seconds_processed=transcript.duration_seconds, inference_seconds=round(time.monotonic() - started, 6),
                                         peak_memory_bytes=peak_memory()))


__all__ = ["RealAsr", "RealSpeakerAttribution", "RealValidationVad", "attribute_turns", "infer_roles", "roles_prompt"]
