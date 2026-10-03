"""Per-turn and per-block analyses: MERaLiON acoustic tone, Cardiff RoBERTa text sentiment, and
the deterministic numeric enrichment.

They wrap ``call1.pipeline.sentiment`` (``analyze_tone_blocks`` / ``analyze_text_sentiment``, the
two halves of the pre-split ``enrich_sentiment``, under the same inference lock and with the same
statuses: SCORED, NO_SPEECH, INSUFFICIENT_SPEECH, OVERLAP, UNATTRIBUTED, AMBIGUOUS_SPEAKER,
DISABLED, MODEL_ERROR) and ``call1.pipeline.numeric_extractor.extract_numeric_references``.
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

from call1.contracts.contents import (
    SpeakerRole,
    TextSentimentContent,
    TextSentimentLabel,
    ToneBlocksContent,
    TranscriptContent,
    TurnSentiment,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType

from call1.process.handlers.base import Handler, HandlerError, HandlerJob, HandlerResult, Output, ReleaseJob, Usage

from .convert import contract_tone_block, legacy_turns
from .paths import weights_installed, weights_path

ADAPTER_VERSION = "1"


def _require_weights(job: HandlerJob, what: str) -> None:
    if job.catalog_entry is not None and not weights_installed(job.catalog_entry):
        raise ReleaseJob("reject", JobErrorCode.MODEL_UNAVAILABLE, f"the {what} weights are not installed on this host")


def _agent_channel(transcript: TranscriptContent, job: HandlerJob) -> int:
    """The stereo channel carrying the agent: the registered one when the job names it, else the
    channel ASR labelled AGENT (turn channels survive reviewer role corrections), else 0."""
    value = job.parameters.extra.get("agent_channel")
    if value in (0, 1):
        return int(value)
    channels = {t.channel for t in transcript.turns if t.speaker is SpeakerRole.AGENT and t.channel in (0, 1)}
    return channels.pop() if len(channels) == 1 else 0


class RealAcousticTone(Handler):
    """``acoustic_tone``: MERaLiON SER valence/arousal/dominance and emotion per speaker in fixed
    seven-second call-time blocks, on the speaker's own attributed speech only."""

    job_type = JobType.ACOUSTIC_TONE
    adapter_id = "call1.torch.meralion"
    adapter_version = ADAPTER_VERSION

    def ready(self, job: HandlerJob) -> None:
        _require_weights(job, "tone")

    def run(self, job: HandlerJob) -> HandlerResult:
        from call1.pipeline.sentiment import LocalSentimentModels, analyze_tone_blocks, average_agent_tone

        transcript = job.transcript()
        audio = job.require("audio")
        turns = legacy_turns(transcript)
        entry = job.catalog_entry
        models = LocalSentimentModels(tone_path=str(weights_path(entry)) if entry is not None and weights_path(entry) else None)
        started = time.monotonic()
        job.check_cancelled()
        try:
            blocks = analyze_tone_blocks(turns, audio.path(), 1, _agent_channel(transcript, job), models)
        except ValueError:
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "Tone analysis needs mono or stereo audio.") from None
        except subprocess.CalledProcessError:
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "The recording could not be decoded for tone analysis.") from None
        avg = average_agent_tone(blocks)
        content = ToneBlocksContent(blocks=[contract_tone_block(b) for b in blocks],
                                    avg_agent_tone=round(float(avg), 6) if avg is not None else None)
        return HandlerResult(outputs={"tone_blocks": Output(content)},
                             usage=Usage(audio_seconds_processed=transcript.duration_seconds, inference_seconds=round(time.monotonic() - started, 6)))


class RealTextSentiment(Handler):
    """``text_sentiment``: Cardiff RoBERTa negative/neutral/positive per turn, token-weighted over
    512-token windows so a long turn is never truncated; score = P(positive) - P(negative)."""

    job_type = JobType.TEXT_SENTIMENT
    adapter_id = "call1.torch.roberta_sentiment"
    adapter_version = ADAPTER_VERSION

    def ready(self, job: HandlerJob) -> None:
        _require_weights(job, "sentiment")

    def run(self, job: HandlerJob) -> HandlerResult:
        from call1.pipeline.sentiment import TEXT_MODEL, TEXT_REVISION, LocalSentimentModels, analyze_text_sentiment

        transcript = job.transcript()
        turns = legacy_turns(transcript)
        entry = job.catalog_entry
        models = LocalSentimentModels(text_path=str(weights_path(entry)) if entry is not None and weights_path(entry) else None)
        started = time.monotonic()
        job.check_cancelled()
        analyze_text_sentiment(turns, models)
        rows = []
        for turn in turns:
            info = turn.text_analysis or {}
            score = turn.text_sentiment
            label: Optional[TextSentimentLabel] = TextSentimentLabel(turn.text_sentiment_label) if turn.text_sentiment_label else None
            probabilities = {str(k): float(v) for k, v in (info.get("probabilities") or {}).items()}
            rows.append(TurnSentiment(turn_id=turn.turn_id, score=None if score is None else max(-1.0, min(1.0, float(score))), label=label,
                                      probabilities=probabilities, status=str(info.get("status") or "EMPTY_TEXT")[:200]))
        caller = [t.text_sentiment for t in turns if t.speaker.value == SpeakerRole.CALLER.value and t.text_sentiment is not None]
        content = TextSentimentContent(model=TEXT_MODEL, revision=TEXT_REVISION, turns=rows,
                                       avg_caller_sentiment=round(sum(caller) / len(caller), 2) if caller else None)
        return HandlerResult(outputs={"text_sentiment": Output(content)}, usage=Usage(inference_seconds=round(time.monotonic() - started, 6)))


class RealEnrichment(Handler):
    """``enrichment``: currency, percentages, account and phone numbers, dates, durations and
    generic numbers per turn with interpolated timestamps (deterministic; masking reads it), plus
    (contract 1.2.0, decision 19) the ``pii_findings`` of the model PII layer for this transcript
    revision (``masking.pii_findings``: ``openai/privacy-filter``, or the labelled stub). Missing
    weights fail the job closed (``model_unavailable``), which keeps Store's reviewer text withheld."""

    job_type = JobType.ENRICHMENT
    adapter_id = "call1.pipeline.numeric_extractor"
    adapter_version = ADAPTER_VERSION

    def run(self, job: HandlerJob) -> HandlerResult:
        from .masking import extract_enrichment, pii_findings

        transcript: TranscriptContent = job.require("transcript").content()  # type: ignore[assignment]
        started = time.monotonic()
        findings = pii_findings(job)
        return HandlerResult(outputs={"enrichment": Output(extract_enrichment(transcript)), "pii_findings": Output(findings)},
                             usage=Usage(inference_seconds=round(time.monotonic() - started, 6)))


__all__ = ["RealAcousticTone", "RealEnrichment", "RealTextSentiment"]
