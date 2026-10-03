"""Fake handlers (``CALL1_PROCESS_HANDLERS=fake``): contract-valid content for every job type the
code handlers do not cover, with no model and no Apple Silicon.

Everything they produce is **synthetic test content**: the transcript is a fixed script spread
over the recording's real duration, and the analyses are simple deterministic functions of it.
Every attempt records adapter ``fake.<job_type>``, so a fake result is never mistaken for a real
one. Validation reads the real WAV header when it can.

``FakeBehavior`` scripts outcomes for tests (or ``CALL1_FAKE_BEHAVIOR``, JSON): a queue of actions
per key, consumed one per attempt, where a key is a job type (``"asr"``) or a job type and
criterion (``"qa_criterion:REG-01"``). Actions: ``ok``, ``fail:<job error code>``,
``release:<code>``, ``crash``, ``hold:<seconds>`` (runs slowly, honouring cancel), and for QA
``pass``, ``fail``, ``not_applicable``, ``needs_review``, ``invalid_answer``, ``provider_error``.

Contact Signals v2 (contract 1.3.0, docs/ContactSignalsV2.md section 8.6): ``FakeSignalsCategorize``,
``FakeSignalsSubcategorize`` and ``FakeSignalsExtract`` run the real segmenter, masking, span, grounding
and carry-forward code (``handlers/signal_stages.py``) with substring-matching fake engines, labelled
``fake``. Their keys (``contact_signals_categorize`` and so on) also accept ``invalid_answer``,
``provider_error`` and ``low_confidence`` (scores at the threshold minus 0.05).

Named fake scripts: ``FakeAsr`` emits ``SCRIPT`` by default, or a named script (``cancel``,
``competitor``, ``caller_name``, ``returns``, ``vocabulary``) chosen by the ``asr`` action
``script:<name>`` or by the source recording through ``FAKE_SCRIPTS_BY_SOURCE`` (the source audio's
checksum; extend it with ``CALL1_FAKE_SCRIPTS``, a JSON object of checksum to script name).

Dual transcription (contract 1.3.0, docs/DualAsr.md section 5): with ``parameters.asr_vocabulary``,
``FakeAsr`` gives the scripted transcript word timings and plays both engines. The script is the
"Parakeet" transcript; the "Whisper" pass is the same words with every ``FAKE_MISHEARINGS`` phrase
whose term is in the job's vocabulary written as the term ("standy cup" becomes "Stanley cup"). The
real glossary shortlist and rule merge then run, so the ``vocabulary`` script shows three real
replacements with no model. The ``asr`` action ``vocabulary_fail`` (or ``vocabulary_fail:<code>``)
yields a ``base_only`` transcript; actions combine with commas (``script:vocabulary,vocabulary_fail``).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from call1.contracts.common import canonical_digest
from call1.contracts.contents import (
    AsrPassSegment,
    AsrPassWord,
    AsrVocabularyPassContent,
    AudioChannelLayout,
    AudioValidationContent,
    ContactSignalKind,
    ContactSignalPass,
    ContactSignalsPassContent,
    ContactSignalView,
    EnrichmentContent,
    EscalationTrigger,
    ModelAttemptView,
    NumericEntityContent,
    PromptInputContent,
    QaAssessmentContent,
    QaVerdictContent,
    SpeakerAssignment,
    SpeakerAttributionContent,
    SpeakerRole,
    SummaryCitation,
    SummarySegmentContent,
    SummarySynthesisContent,
    TextSentimentContent,
    TextSentimentLabel,
    ToneBlocksContent,
    ToneBlockStatus,
    ToneBlockView,
    TranscriptContent,
    TranscriptTurnContent,
    TurnEnrichment,
    TurnSentiment,
    TurnWindow,
    VadMetricsContent,
    VadSegmentContent,
    VerdictStatus,
    VerdictView,
    WordTimestampView,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import ASR_BASE_TRANSCRIPT_ROLE, ASR_VOCABULARY_PASS_ROLE, RELEASE_REQUEUE_CODES, JobType
from call1.contracts.vocabulary import AsrVocabularyParameters, vocabulary_term_key
from call1.contracts.rubrics import CheckType, RubricCriterion
from call1.contracts.usage import TokenCount, TokenSource

from call1.contracts.contents import SIGNAL_NONE_OPTION, SIGNAL_NOT_OPTION, SIGNAL_OTHER_OPTION, SignalFieldType
from call1.contracts.signals import SignalTaxonomy, SignalTaxonomySnapshotContent, stage2_options
from call1.pipeline.signals_v2 import ChoiceRow, EngineError, ExtractionSpan, RawExtraction, RowBudget, resolve_thresholds

from ..audio import UnsupportedAudio, probe
from ..transcripts import apply_speaker_correction, estimate_tokens, turns_in
from .base import Handler, HandlerError, HandlerJob, HandlerResult, Output, ReleaseJob, Usage
from .signal_stages import FAKE_ENGINE_DEFAULTS, ExtractEngine, run_categorize, run_extract, run_subcategorize

FAKE_ADAPTER_VERSION = "fake-1"

SCRIPT: List[Tuple[SpeakerRole, str]] = [
    (SpeakerRole.AGENT, "Thank you for calling. This call may be recorded for quality assurance. How can I help you today?"),
    (SpeakerRole.CALLER, "Hi, I have a question about a fee on my account."),
    (SpeakerRole.AGENT, "I can help with that. For security, please verify your date of birth and the last four digits of the account."),
    (SpeakerRole.CALLER, "Sure, the last four digits are 1234."),
    (SpeakerRole.AGENT, "Thank you, you are verified. The monthly service fee is $5.00 and you can dispute it within 60 days."),
    (SpeakerRole.CALLER, "Okay, that makes sense. Thanks for explaining."),
    (SpeakerRole.AGENT, "Is there anything else I can help you with today?"),
    (SpeakerRole.CALLER, "No, that is all. Thank you."),
    (SpeakerRole.AGENT, "Thank you for calling, and have a great day."),
]
"""The fake transcript. It is spread over the recording's duration, one turn every four seconds or so."""

CALLER_NAME_SCRIPT: List[Tuple[SpeakerRole, str]] = [
    (SpeakerRole.AGENT, "Thank you for calling, this is Sam. This call may be recorded for quality assurance. How can I help you today?"),
    (SpeakerRole.CALLER, "Hi, my name is Maria Lopez and I have a question about a fee on my account."),
    *SCRIPT[2:],
]
"""The fake transcript for the scripted ASR action ``caller_name``: the agent introduces themself
(Sam, kept visible) and the caller gives a name (Maria Lopez, which the stub PII detector flags)."""

CANCEL_SCRIPT: List[Tuple[SpeakerRole, str]] = [
    SCRIPT[0],
    (SpeakerRole.CALLER, "Hi, I want to cancel my account because the price went up again."),
    (SpeakerRole.AGENT, "I'm sorry to hear that. For security, please verify your date of birth and the last four digits of the account."),
    (SpeakerRole.CALLER, "Sure, the last four digits are 1234."),
    (SpeakerRole.AGENT, "Thank you, you are verified. I can offer you a lower plan for $20.00 a month, or cancel the account today."),
    (SpeakerRole.CALLER, "Please cancel it. That makes sense, thanks for explaining."),
    (SpeakerRole.AGENT, "Is there anything else I can help you with today?"),
    (SpeakerRole.CALLER, "No, that is all. Thank you."),
    (SpeakerRole.AGENT, "Thank you for calling, and have a great day."),
]
"""The ``cancel`` fake script (section 8.6): caller turn 1 asks to cancel over a price rise. It keeps
``SCRIPT``'s speaker order turn by turn (the fake attribution assigns roles from ``SCRIPT`` by index)."""

COMPETITOR_SCRIPT: List[Tuple[SpeakerRole, str]] = [
    SCRIPT[0],
    (SpeakerRole.CALLER, "Hi, I am calling about my internet bill and a better deal."),
    (SpeakerRole.AGENT, "I see you're with a competitor for your mobile line. I can offer you a bundle discount."),
    (SpeakerRole.CALLER, "Sure, the last four digits are 1234."),
    *SCRIPT[4:],
]
"""The ``competitor`` fake script (CustomFlags section 4.9): an agent turn within the first three says
"you're with a competitor"."""

RETURN_SCRIPT: List[Tuple[SpeakerRole, str]] = [
    SCRIPT[0],
    (SpeakerRole.CALLER, "Hi, I'd like to return a blender that arrived cracked."),
    (SpeakerRole.AGENT, "I'm sorry about that. I can offer you a prepaid return label by email today."),
    (SpeakerRole.CALLER, "Okay, go on."),
    (SpeakerRole.AGENT, "Once the carrier scans it, we can offer a replacement or a refund to the original card."),
    (SpeakerRole.CALLER, "That works, thanks for explaining."),
    (SpeakerRole.AGENT, "Is there anything else I can help you with today?"),
    (SpeakerRole.CALLER, "No, that is all. Thank you."),
    (SpeakerRole.AGENT, "Thank you for calling, and have a great day."),
]
"""The ``returns`` fake script (decision 25, ContactSignalsV2 section 6.5): the agent proposes the
fix over two turns (2 and 4) with only the caller's "Okay, go on." between them, so the v2 merge
shows one multi-segment ``fix_proposed`` signal with one part. It keeps ``SCRIPT``'s speaker order."""

VOCABULARY_SCRIPT: List[Tuple[SpeakerRole, str]] = [
    SCRIPT[0],
    (SpeakerRole.CALLER, "Hi, I ordered a standy cup for click and collect at the Chad Stone store and it never arrived."),
    SCRIPT[2],
    SCRIPT[3],
    (SpeakerRole.AGENT, "Thank you, you are verified. I can refund it to your after pay account or send a new one by express shipping."),
    *SCRIPT[5:],
]
"""The ``vocabulary`` fake script (dual transcription, docs/DualAsr.md): "Parakeet" mishears three
retail seed terms ("standy cup", "Chad Stone", "after pay"); with the retail vocabulary active the
merge writes "Stanley cup", "Chadstone" and "Afterpay" over them. It keeps ``SCRIPT``'s speaker order."""

FAKE_MISHEARINGS: Dict[str, str] = {"Stanley cup": "standy cup", "Chadstone": "Chad Stone", "Afterpay": "after pay"}
"""Vocabulary term -> how the fake "Parakeet" writes it. The fake "Whisper" pass hears the term."""

FAKE_SCRIPTS: Dict[str, List[Tuple[SpeakerRole, str]]] = {
    "default": SCRIPT, "caller_name": CALLER_NAME_SCRIPT, "cancel": CANCEL_SCRIPT, "competitor": COMPETITOR_SCRIPT,
    "returns": RETURN_SCRIPT, "vocabulary": VOCABULARY_SCRIPT,
}

FAKE_SCRIPTS_BY_SOURCE: Dict[str, str] = {}
"""Source audio checksum (``sha256:...``) to fake script name. ``CALL1_FAKE_SCRIPTS`` (JSON) extends it,
so a demo can give one recording the ``cancel`` script without touching the others."""


def script_for(action: str, source_checksum: Optional[str] = None) -> List[Tuple[SpeakerRole, str]]:
    """The script ``FakeAsr`` emits: the action ``script:<name>`` (or the older ``caller_name``), else
    the source recording's mapped script, else ``SCRIPT``."""
    if action.startswith("script:"):
        name = action.split(":", 1)[1]
        if name not in FAKE_SCRIPTS:
            raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, f"unknown fake script {name!r}")
        return FAKE_SCRIPTS[name]
    if action == "caller_name":
        return CALLER_NAME_SCRIPT
    mapping = dict(FAKE_SCRIPTS_BY_SOURCE)
    raw = os.environ.get("CALL1_FAKE_SCRIPTS")
    if raw:
        try:
            extra = json.loads(raw)
            if isinstance(extra, dict):
                mapping.update({str(k): str(v) for k, v in extra.items()})
        except json.JSONDecodeError:
            pass
    name = mapping.get(source_checksum or "")
    return FAKE_SCRIPTS.get(name or "default", SCRIPT)


_POSITIVE = ("thank", "thanks", "great", "help", "sure", "okay", "appreciate", "verified", "sense")
_NEGATIVE = ("problem", "unfortunately", "dispute", "not", "cannot", "complaint", "wrong")


class FakeBehavior:
    """Scripted outcomes per key, consumed one per attempt (thread-safe)."""

    def __init__(self, actions: Optional[Dict[str, Iterable[str]]] = None) -> None:
        self._queues: Dict[str, Deque[str]] = defaultdict(deque)
        self._lock = threading.Lock()
        for key, values in (actions or {}).items():
            self._queues[key].extend(values)

    @classmethod
    def from_env(cls) -> "FakeBehavior":
        raw = os.environ.get("CALL1_FAKE_BEHAVIOR")
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls()
        return cls({str(k): [str(v) for v in (vs if isinstance(vs, list) else [vs])] for k, vs in data.items()}) if isinstance(data, dict) else cls()

    def push(self, key: str, *actions: str) -> None:
        with self._lock:
            self._queues[key].extend(actions)

    def next(self, *keys: str) -> str:
        with self._lock:
            for key in keys:
                queue = self._queues.get(key)
                if queue:
                    return queue.popleft()
        return "ok"


def _usage_tokens(count: Optional[int]) -> Optional[TokenCount]:
    return TokenCount(count=count, source=TokenSource.LOCAL_TOKENIZER) if count is not None else None


class _Fake(Handler):
    adapter_version = FAKE_ADAPTER_VERSION

    def __init__(self, behavior: FakeBehavior) -> None:
        self.behavior = behavior

    @property
    def adapter_id(self) -> str:  # type: ignore[override]
        return f"fake.{self.job_type.value}"

    def keys(self, job: HandlerJob) -> List[str]:
        return [self.job_type.value]

    def run(self, job: HandlerJob) -> HandlerResult:
        action = self.behavior.next(*self.keys(job))
        if action.startswith("fail:"):
            raise HandlerError(JobErrorCode(action.split(":", 1)[1]), "scripted fake failure")
        if action.startswith("release:"):
            code = JobErrorCode(action.split(":", 1)[1])
            raise ReleaseJob("requeue" if code in RELEASE_REQUEUE_CODES else "reject", code, "scripted fake release")
        if action == "crash":
            raise RuntimeError("scripted fake crash")
        if action.startswith("hold:"):
            deadline = time.monotonic() + float(action.split(":", 1)[1])
            while time.monotonic() < deadline:
                job.check_cancelled()
                time.sleep(0.02)
            job.check_cancelled()
            action = "ok"
        return self.produce(job, action)

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:  # pragma: no cover - interface
        raise NotImplementedError


# --- media -------------------------------------------------------------------------------------


def _duration(job: HandlerJob) -> float:
    audio = job.input("audio")
    if audio is None:
        return 36.0
    try:
        info = probe(audio.path(), content_type=audio.artifact.content_type)
    except UnsupportedAudio:
        return 36.0
    return info.duration_seconds if info.duration_seconds is not None else 36.0


class FakeValidationVad(_Fake):
    job_type = JobType.VALIDATION_VAD

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        audio = job.require("audio")
        try:
            info = probe(audio.path(), content_type=audio.artifact.content_type)
        except UnsupportedAudio as exc:
            raise HandlerError(JobErrorCode.VALIDATION_REJECTED, str(exc)) from None
        channels = info.channels or 1
        duration = round(info.duration_seconds or 0.0, 3)
        warnings = [] if info.duration_seconds is not None else ["Fake validation could not read this container's duration."]
        layout = AudioChannelLayout.MONO if channels == 1 else AudioChannelLayout.STEREO if channels == 2 else AudioChannelLayout.MULTI_CHANNEL
        report = AudioValidationContent(container=info.container, codec="pcm_s16le" if info.container == "wav" else "unknown",
                                        sample_rate=info.sample_rate or 16000, channels=channels, channel_layout=layout, duration_seconds=duration,
                                        agent_channel=0 if channels == 2 else None, warnings=warnings)
        segments = []
        start = 0.0
        while start < duration:
            end = min(duration, start + 4.0)
            segments.append(VadSegmentContent(start_time=round(start, 3), end_time=round(end, 3)))
            start += 5.0
        speech = round(sum(s.end_time - s.start_time for s in segments), 3)
        silence = round(max(0.0, duration - speech), 3)
        metrics = VadMetricsContent(total_speech_duration=speech, total_silence_duration=silence,
                                    silence_ratio=round(silence / duration, 4) if duration else 0.0, overtalk_duration=0.0, overtalk_ratio=0.0,
                                    segments=segments)
        return HandlerResult(outputs={"validation_report": Output(report), "vad_metrics": Output(metrics)},
                             usage=Usage(audio_seconds_processed=duration))


def _fake_word_timings(turn: TranscriptTurnContent) -> TranscriptTurnContent:
    words = turn.text.split()
    if not words:
        return turn
    step = (turn.end_time - turn.start_time) / len(words)
    timed = [WordTimestampView(word=w, start_time=round(turn.start_time + i * step, 3),
                               end_time=round(max(turn.start_time + i * step, turn.start_time + (i + 1) * step - 0.02), 3), probability=0.9)
             for i, w in enumerate(words)]
    return turn.model_copy(update={"word_timestamps": timed})


def _bare(word: str) -> str:
    return re.sub(r"[^\w']", "", word).casefold()


def fake_vocabulary_pass(transcript: TranscriptContent, params: AsrVocabularyParameters, channels: int) -> AsrVocabularyPassContent:
    """The fake "Whisper" pass: the transcript's words, with each ``FAKE_MISHEARINGS`` phrase whose term
    is in the job's vocabulary written as the term over the phrase's time."""
    from call1.pipeline.vocabulary_asr import glossary_for

    keys = {vocabulary_term_key(t.term) for t in params.terms}
    phrases = [(term, [_bare(w) for w in heard.split()]) for term, heard in FAKE_MISHEARINGS.items() if vocabulary_term_key(term) in keys]
    words: List[AsrPassWord] = []
    segments: List[AsrPassSegment] = []
    base_words = []
    for turn in transcript.turns:
        channel = turn.channel if channels >= 2 else None
        timed = list(turn.word_timestamps or [])
        segments.append(AsrPassSegment(start_time=turn.start_time, end_time=turn.end_time, channel=channel, avg_logprob=-0.2,
                                       no_speech_prob=0.01, compression_ratio=1.4, temperature=0.0))
        i = 0
        while i < len(timed):
            match = next(((term, n) for term, bare in phrases for n in [len(bare)]
                          if [_bare(w.word) for w in timed[i:i + n]] == bare), None)
            if match is not None:
                term, n = match
                start, end = timed[i].start_time, timed[i + n - 1].end_time
                parts = term.split(" ")
                step = (end - start) / len(parts)
                for k, part in enumerate(parts):
                    words.append(AsrPassWord(word=part, start_time=round(start + k * step, 3), end_time=round(start + (k + 1) * step, 3),
                                             probability=0.8, channel=channel, segment=len(segments) - 1))
                i += n
                continue
            w = timed[i]
            words.append(AsrPassWord(word=w.word, start_time=w.start_time, end_time=w.end_time, probability=0.9, channel=channel,
                                     segment=len(segments) - 1))
            i += 1
        base_words.extend({"word": w.word, "start": w.start_time, "end": w.end_time} for w in timed)
    _, glossary = glossary_for([t.term for t in params.terms], base_words, lambda text: 2 * len(text.split()), params.glossary_prompt_limit)
    return AsrVocabularyPassContent(engine="fake-whisper-small", model_revision=FAKE_ADAPTER_VERSION, duration_seconds=transcript.duration_seconds,
                                    channels=max(1, min(2, channels)), glossary_terms=glossary, words=words, segments=segments)


class FakeAsr(_Fake):
    job_type = JobType.ASR

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        duration = _duration(job)
        channels = int(job.parameters.extra.get("channels") or 0) or 1
        stereo = channels >= 2
        audio = job.input("audio")
        parts = [part.strip() for part in action.split(",") if part.strip()] or ["ok"]
        vocabulary_fail = next((part for part in parts if part.startswith("vocabulary_fail")), None)
        script_action = next((part for part in parts if part.startswith("script:") or part == "caller_name"), "ok")
        script = script_for(script_action, audio.artifact.checksum if audio is not None else None)
        count = min(len(script), max(2, int(duration // 4)))
        step = duration / count if count else duration
        turns = []
        for index in range(count):
            role, text = script[index]
            start = round(index * step, 3)
            end = round(max(start, (index + 1) * step - 0.2), 3)
            turns.append(TranscriptTurnContent(
                turn_id=index, speaker=role if stereo else SpeakerRole.UNKNOWN, start_time=start, end_time=end, text=text,
                channel=(0 if role is SpeakerRole.AGENT else 1) if stereo else None, confidence=0.9))
        transcript = TranscriptContent(duration_seconds=round(duration, 3), language="en", is_redacted=False, turns=turns)
        usage = Usage(audio_seconds_processed=round(duration, 3))
        params = job.parameters.asr_vocabulary
        if params is None:
            return HandlerResult(outputs={"transcript": Output(transcript)}, usage=usage)
        from call1.process.vocabulary import applied_correction, base_only_correction, corrected, merge

        base = transcript.model_copy(update={"turns": [_fake_word_timings(t) for t in transcript.turns]})
        outputs = {ASR_BASE_TRANSCRIPT_ROLE: Output(base)}
        if vocabulary_fail is not None:
            code = JobErrorCode(vocabulary_fail.split(":", 1)[1]) if ":" in vocabulary_fail else JobErrorCode.MODEL_UNAVAILABLE
            correction = base_only_correction(params, base_engine="fake-asr", note="Scripted fake failure: the vocabulary pass did not run.",
                                              code=code, candidate_engine="fake-whisper-small", candidate_model_revision=FAKE_ADAPTER_VERSION)
            outputs["transcript"] = Output(corrected(base, correction))
            return HandlerResult(outputs=outputs, usage=usage)
        content = fake_vocabulary_pass(base, params, channels)
        outcome = merge(base, content.words, params, channels=channels)
        correction = applied_correction(params, outcome, base_engine="fake-asr", candidate_engine="fake-whisper-small",
                                        candidate_model_revision=FAKE_ADAPTER_VERSION, glossary_term_count=len(content.glossary_terms))
        outputs["transcript"] = Output(corrected(base, correction, outcome))
        outputs[ASR_VOCABULARY_PASS_ROLE] = Output(content)
        return HandlerResult(outputs=outputs, usage=usage)


class FakeSpeakerAttribution(_Fake):
    job_type = JobType.SPEAKER_ATTRIBUTION

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        transcript: TranscriptContent = job.require("transcript").content()  # type: ignore[assignment]
        correction = job.parameters.speaker_correction
        if correction is not None:
            try:
                content = apply_speaker_correction(transcript, job.attribution(), correction)
            except ValueError as exc:
                raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, str(exc)) from None
            return HandlerResult(outputs={"speaker_attribution": Output(content)})
        assignments = []
        for index, turn in enumerate(transcript.turns):
            role = SCRIPT[index % len(SCRIPT)][0]
            assignments.append(SpeakerAssignment(turn_id=turn.turn_id, speaker=role,
                                                 speaker_cluster="speaker_0" if role is SpeakerRole.AGENT else "speaker_1", confidence=0.9))
        return HandlerResult(outputs={"speaker_attribution": Output(SpeakerAttributionContent(method="diarization", assignments=assignments))},
                             usage=Usage(audio_seconds_processed=transcript.duration_seconds))


class FakeAcousticTone(_Fake):
    job_type = JobType.ACOUSTIC_TONE

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        transcript = job.transcript()
        selection = job.selection
        blocks: List[ToneBlockView] = []
        window = 15.0
        duration = max(transcript.duration_seconds, max((t.end_time for t in transcript.turns), default=0.0))
        for speaker in (SpeakerRole.AGENT, SpeakerRole.CALLER):
            start = 0.0
            while start < duration:
                end = min(duration, start + window)
                turns = [t for t in transcript.turns if t.speaker is speaker and t.start_time < end and t.end_time > start]
                speech = round(sum(min(end, t.end_time) - max(start, t.start_time) for t in turns), 3)
                scored = speech >= 1.0
                blocks.append(ToneBlockView(
                    block_id=len(blocks), speaker=speaker, start_time=round(start, 3), end_time=round(end, 3),
                    status=ToneBlockStatus.SCORED if scored else (ToneBlockStatus.NO_SPEECH if speech == 0 else ToneBlockStatus.INSUFFICIENT_SPEECH),
                    speech_seconds=max(0.0, speech), valence=0.62 if scored and speaker is SpeakerRole.AGENT else 0.55 if scored else None,
                    arousal=0.45 if scored else None, dominance=0.5 if scored else None, emotion="neutral" if scored else None,
                    emotion_probabilities={"neutral": 0.7, "happy": 0.2, "sad": 0.1} if scored else None,
                    turn_ids=[t.turn_id for t in turns], model=selection.model_family if selection else "fake",
                    revision=selection.model_revision if selection else "fake", analysis_version="fake-tone-1"))
                start = end
        agent = [b.valence for b in blocks if b.speaker is SpeakerRole.AGENT and b.valence is not None]
        content = ToneBlocksContent(blocks=blocks, avg_agent_tone=round(sum(agent) / len(agent), 4) if agent else None)
        return HandlerResult(outputs={"tone_blocks": Output(content)}, usage=Usage(audio_seconds_processed=transcript.duration_seconds))


def _polarity(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    score = 0.15 * sum(w.startswith(_POSITIVE) for w in words) - 0.2 * sum(w in _NEGATIVE for w in words)
    return round(max(-1.0, min(1.0, score)), 4)


class FakeTextSentiment(_Fake):
    job_type = JobType.TEXT_SENTIMENT

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        transcript = job.transcript()
        selection = job.selection
        turns = []
        for turn in transcript.turns:
            score = _polarity(turn.text)
            label = TextSentimentLabel.POSITIVE if score > 0.05 else TextSentimentLabel.NEGATIVE if score < -0.05 else TextSentimentLabel.NEUTRAL
            positive = round(max(0.0, score), 4)
            negative = round(max(0.0, -score), 4)
            turns.append(TurnSentiment(turn_id=turn.turn_id, score=score, label=label,
                                       probabilities={"positive": positive, "negative": negative, "neutral": round(1 - positive - negative, 4)}))
        caller = [s.score for s, t in zip(turns, transcript.turns) if t.speaker is SpeakerRole.CALLER and s.score is not None]
        content = TextSentimentContent(model=selection.model_family if selection else "fake", revision=selection.model_revision if selection else "fake",
                                       turns=turns, avg_caller_sentiment=round(sum(caller) / len(caller), 4) if caller else None)
        return HandlerResult(outputs={"text_sentiment": Output(content)})


_CURRENCY = re.compile(r"\$\s?(\d+(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_NUMBER = re.compile(r"(?<![\d$.])(\d{2,})(?![\d.%])")


class FakeEnrichment(_Fake):
    job_type = JobType.ENRICHMENT

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        transcript: TranscriptContent = job.require("transcript").content()  # type: ignore[assignment]
        turns = []
        for turn in transcript.turns:
            entities = []
            for match in _CURRENCY.finditer(turn.text):
                entities.append(NumericEntityContent(raw_text=match.group(0), normalized_value=float(match.group(1)), entity_type="CURRENCY",
                                                     start_time=turn.start_time, end_time=turn.end_time, unit="USD"))
            for match in _PERCENT.finditer(turn.text):
                entities.append(NumericEntityContent(raw_text=match.group(0), normalized_value=float(match.group(1)), entity_type="PERCENTAGE",
                                                     start_time=turn.start_time, end_time=turn.end_time))
            for match in _NUMBER.finditer(turn.text):
                entities.append(NumericEntityContent(raw_text=match.group(0), normalized_value=float(match.group(1)), entity_type="GENERIC_NUMBER",
                                                     start_time=turn.start_time, end_time=turn.end_time))
            turns.append(TurnEnrichment(turn_id=turn.turn_id, numeric_entities=entities))
        # The PII findings come from the labelled stub detector (``call1.pii_model``, decision 19),
        # the same code path the real handler takes, so Store's masking is exercised end to end.
        from call1 import pii_model

        from .real.masking import pii_findings

        try:
            backend = pii_model.configured_backend("fake")
        except pii_model.PiiConfigError as exc:
            raise HandlerError(JobErrorCode.MODEL_UNAVAILABLE, str(exc)) from None
        return HandlerResult(outputs={"enrichment": Output(EnrichmentContent(turns=turns)),
                                      "pii_findings": Output(pii_findings(job, backend=backend))})


# --- QA ----------------------------------------------------------------------------------------


def _speaker_turns(transcript: TranscriptContent, speaker: Optional[SpeakerRole], window_seconds: Optional[float]) -> List[TranscriptTurnContent]:
    turns = [t for t in transcript.turns if speaker is None or t.speaker is speaker]
    if window_seconds is not None and window_seconds > 0:
        turns = [t for t in turns if t.start_time <= window_seconds]
    elif window_seconds is not None and window_seconds < 0:
        edge = transcript.duration_seconds + window_seconds
        turns = [t for t in turns if t.end_time >= edge]
    return turns


def _has(text: str, phrase: str) -> bool:
    return phrase.strip().lower() in text.lower()


def deterministic_verdict(criterion: RubricCriterion, transcript: TranscriptContent, sentiment: Optional[TextSentimentContent],
                          tone: Optional[ToneBlocksContent]) -> VerdictView:
    check = criterion.check
    turns = _speaker_turns(transcript, check.speaker, check.window_seconds)
    status, reasoning, evidence = VerdictStatus.NOT_APPLICABLE, "Not evaluated by the fake handler.", None
    if check.check_type in (CheckType.PHRASE_ANY, CheckType.PHRASE_ALL, CheckType.PHRASE_NONE):
        hits = [(p, t) for p in check.phrases for t in turns if _has(t.text, p)]
        found = {p for p, _ in hits}
        if check.check_type is CheckType.PHRASE_ANY:
            status = VerdictStatus.PASS if found else VerdictStatus.FAIL
        elif check.check_type is CheckType.PHRASE_ALL:
            status = VerdictStatus.PASS if check.phrases and found == set(check.phrases) else VerdictStatus.FAIL
        else:
            status = VerdictStatus.FAIL if found else VerdictStatus.PASS
        evidence = hits[0][1] if hits else None
        reasoning = f"{len(found)} of {len(check.phrases)} phrases found."
    elif check.check_type is CheckType.CUSTOM_REGEX and check.pattern:
        match = next((t for t in turns if re.search(check.pattern, t.text, flags=re.IGNORECASE)), None)
        status, evidence = (VerdictStatus.PASS if match else VerdictStatus.FAIL), match
        reasoning = "Pattern found." if match else "Pattern not found."
    elif check.check_type is CheckType.CONDITIONAL_RESPONSE:
        trigger = next((t for t in transcript.turns if any(_has(t.text, p) for p in check.trigger_phrases)), None)
        if trigger is None:
            reasoning = "The trigger did not occur."
        else:
            response = next((t for t in turns if t.turn_id > trigger.turn_id and any(_has(t.text, p) for p in check.response_phrases)), None)
            status, evidence = (VerdictStatus.PASS if response else VerdictStatus.FAIL), response
            reasoning = "The required response followed the trigger." if response else "No required response followed the trigger."
    elif check.check_type is CheckType.SENTIMENT_METRIC:
        values: List[float] = []
        if check.metric == "text_polarity" and sentiment is not None:
            ids = {t.turn_id for t in turns}
            values = [s.score for s in sentiment.turns if s.turn_id in ids and s.score is not None]
        elif tone is not None:
            values = [getattr(b, check.metric) for b in tone.blocks if b.speaker is (check.speaker or SpeakerRole.AGENT) and getattr(b, check.metric) is not None]
        if len(values) >= check.min_samples:
            agg = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}[check.aggregation]
            ok = agg >= check.metric_threshold if check.comparison == "gte" else agg <= check.metric_threshold
            status, reasoning = (VerdictStatus.PASS if ok else VerdictStatus.FAIL), f"{check.aggregation} {check.metric} {agg:.3f}."
        else:
            reasoning = "Not enough samples."
    return VerdictView(criterion_id=criterion.criterion_id, criterion_name=criterion.name, status=status, confidence=0.95,
                       quoted_evidence=evidence.text if evidence else None, speaker=evidence.speaker if evidence else (check.speaker or SpeakerRole.AGENT),
                       timestamp_range=(evidence.start_time, evidence.end_time) if evidence else None,
                       quote_turn_id=evidence.turn_id if evidence else None, reasoning=reasoning)


class FakeQaDeterministic(_Fake):
    job_type = JobType.QA_DETERMINISTIC

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        transcript = job.transcript()
        sentiment = job.input("text_sentiment")
        tone = job.input("tone_blocks")
        verdicts = [deterministic_verdict(c, transcript, sentiment.content() if sentiment else None, tone.content() if tone else None)  # type: ignore[arg-type]
                    for c in job.rubric().criteria if c.check.check_type is not CheckType.SEMANTIC_JUDGEMENT]
        return HandlerResult(outputs={"verdicts": Output(QaVerdictContent(verdicts=verdicts))})


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())}


def prompt_input_for(job: HandlerJob, template_id: str, transcript: TranscriptContent, window=None) -> PromptInputContent:
    refs = [item.ref for role, item in sorted(job.inputs.items()) if item is not None]
    text = " ".join(t.text for t in turns_in(transcript, window))
    return PromptInputContent(
        template_id=template_id, template_version=FAKE_ADAPTER_VERSION,
        prompt_digest=canonical_digest({"template": template_id, "inputs": [r.model_dump(mode="json") for r in refs],
                                        "parameters": job.parameters.model_dump(mode="json")}),
        masked=bool(job.selection and job.selection.route.masked), transcript_window=window, inputs=refs,
        estimated_input_tokens=estimate_tokens(text))


class FakeQaAssessment(_Fake):
    """``qa_criterion`` and ``qa_escalation``: finds the turn that best matches the criterion text
    and answers from the scripted action (``pass`` by default)."""

    kind = "primary"

    def keys(self, job: HandlerJob) -> List[str]:
        return [f"{self.job_type.value}:{job.parameters.criterion_id}", self.job_type.value]

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        criterion = job.criterion()
        transcript = job.transcript()
        prompt = prompt_input_for(job, f"fake.{self.job_type.value}", transcript)
        if action == "provider_error":
            raise HandlerError(JobErrorCode.PROVIDER_ERROR, "scripted fake provider error", outputs={"prompt_input": Output(prompt)})
        wanted = _words(" ".join(filter(None, [criterion.name, criterion.description, criterion.check.pass_when])))
        candidates = [t for t in transcript.turns if criterion.check.speaker is None or t.speaker is criterion.check.speaker] or list(transcript.turns)
        best = max(candidates, key=lambda t: (len(_words(t.text) & wanted), -t.turn_id), default=None)
        status, confidence, trigger = {
            "ok": (VerdictStatus.PASS, 0.9, None), "pass": (VerdictStatus.PASS, 0.9, None), "fail": (VerdictStatus.FAIL, 0.9, None),
            "not_applicable": (VerdictStatus.NOT_APPLICABLE, 0.9, None), "needs_review": (VerdictStatus.FLAGGED, 0.5, EscalationTrigger.NEEDS_REVIEW),
            "invalid_answer": (VerdictStatus.FLAGGED, 0.0, EscalationTrigger.INVALID_ANSWER),
        }.get(action, (VerdictStatus.PASS, 0.9, None))
        quote = best if status in (VerdictStatus.PASS, VerdictStatus.FAIL) and best is not None else None
        reasoning = {
            VerdictStatus.PASS: "Fake assessment: the quoted turn satisfies the criterion.",
            VerdictStatus.FAIL: "Fake assessment: the quoted turn contradicts the criterion.",
            VerdictStatus.NOT_APPLICABLE: "Fake assessment: the criterion does not apply.",
            VerdictStatus.FLAGGED: ("Fake assessment: the answer failed schema or quote validation." if trigger is EscalationTrigger.INVALID_ANSWER
                                    else "Fake assessment: the evidence is ambiguous and needs review."),
        }[status]
        selection = job.selection
        tokens_in = prompt.estimated_input_tokens
        attempt = ModelAttemptView(
            job_id=job.job.id, attempt_number=job.attempt_number, catalog_entry_id=selection.catalog_entry.entry_id if selection else "fake",
            model_revision=selection.model_revision if selection else "fake", route_class=selection.route.route_class.value if selection else "appliance",
            destination_host=selection.route.destination_host if selection else "in-process", status=status, reasoning=reasoning,
            quoted_evidence=quote.text if quote else None, trigger=trigger, latency_ms=5, tokens_input=tokens_in, tokens_output=40)
        assessment = QaAssessmentContent(
            criterion_id=criterion.criterion_id, assessment_kind=self.kind, status=status, confidence=confidence, reasoning=reasoning,
            quoted_evidence=quote.text if quote else None, quote_turn_id=quote.turn_id if quote else None,
            timestamp_range=(quote.start_time, quote.end_time) if quote else None, speaker=quote.speaker if quote else (criterion.check.speaker or SpeakerRole.AGENT),
            trigger=trigger, escalation_requested=False, attempt=attempt)
        return HandlerResult(outputs={"assessment": Output(assessment), "prompt_input": Output(prompt)},
                             usage=Usage(tokens_input=_usage_tokens(tokens_in), tokens_output=_usage_tokens(40)))


class FakeQaCriterion(FakeQaAssessment):
    job_type = JobType.QA_CRITERION
    kind = "primary"


class FakeQaEscalation(FakeQaAssessment):
    job_type = JobType.QA_ESCALATION
    kind = "escalation"


# --- summary -----------------------------------------------------------------------------------


def _short(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


class FakeSummarySegment(_Fake):
    job_type = JobType.SUMMARY_SEGMENT

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        spec = job.parameters.segment
        transcript = job.transcript()
        window = spec.window if spec else None
        turns = turns_in(transcript, window)
        index = spec.index if spec else 0
        prompt = prompt_input_for(job, "fake.summary_segment", transcript, window)
        if not turns:
            content = SummarySegmentContent(segment_index=index, window=window or _window0(), narrative="No speech was transcribed in this part of the call.")
        else:
            first_agent = next((t for t in turns if t.speaker is SpeakerRole.AGENT), turns[0])
            first_caller = next((t for t in turns if t.speaker is SpeakerRole.CALLER), turns[-1])
            points = [f'The agent said "{_short(first_agent.text)}" (turn {first_agent.turn_id}).',
                      f'The caller said "{_short(first_caller.text)}" (turn {first_caller.turn_id}).']
            content = SummarySegmentContent(
                segment_index=index, window=window or _window0(),
                narrative=f"Turns {turns[0].turn_id} to {turns[-1].turn_id}: {len(turns)} turns between the agent and the caller (fake summary).",
                key_points=points,
                citations=[SummaryCitation(claim="narrative", index=0, turn_ids=[turns[0].turn_id]),
                           SummaryCitation(claim="key_point", index=0, turn_ids=[first_agent.turn_id]),
                           SummaryCitation(claim="key_point", index=1, turn_ids=[first_caller.turn_id])])
        return HandlerResult(outputs={"segment": Output(content), "prompt_input": Output(prompt)},
                             usage=Usage(tokens_input=_usage_tokens(prompt.estimated_input_tokens), tokens_output=_usage_tokens(60)))


def _window0() -> TurnWindow:
    return TurnWindow(turn_start=0, turn_end=0)


class FakeSummarySynthesis(_Fake):
    job_type = JobType.SUMMARY_SYNTHESIS

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        parts = [item.content() for _, item in sorted(job.inputs_with_prefix("part:").items(), key=lambda kv: int(kv[0].split(":")[1]))]
        if not parts:
            raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, "a synthesis needs its parts")
        indexes: List[int] = []
        narratives: List[str] = []
        key_points: List[str] = []
        citations: List[SummaryCitation] = []
        for part in parts:
            indexes += [part.segment_index] if isinstance(part, SummarySegmentContent) else list(part.segment_indexes)  # type: ignore[union-attr]
            narratives.append(part.narrative)  # type: ignore[union-attr]
            offset = len(key_points)
            key_points += list(part.key_points)  # type: ignore[union-attr]
            for citation in part.citations:  # type: ignore[union-attr]
                if citation.claim == "key_point":
                    citations.append(citation.model_copy(update={"index": citation.index + offset}))
        narrative_cites = sorted({t for p in parts for c in p.citations if c.claim == "narrative" for t in c.turn_ids})  # type: ignore[union-attr]
        if narrative_cites:
            citations.insert(0, SummaryCitation(claim="narrative", index=0, turn_ids=narrative_cites))
        prompt = PromptInputContent(template_id="fake.summary_synthesis", template_version=FAKE_ADAPTER_VERSION,
                                    prompt_digest=canonical_digest({"parts": [i.ref.model_dump(mode="json") for i in job.inputs.values() if i]}),
                                    masked=bool(job.selection and job.selection.route.masked),
                                    inputs=[i.ref for _, i in sorted(job.inputs.items()) if i is not None],
                                    estimated_input_tokens=estimate_tokens(" ".join(narratives + key_points)))
        content = SummarySynthesisContent(segment_indexes=sorted(set(indexes)), final=bool(job.parameters.extra.get("final")),
                                          narrative=" ".join(narratives), key_points=key_points[:12],
                                          citations=[c for c in citations if c.claim == "narrative" or c.index < 12])
        return HandlerResult(outputs={"synthesis": Output(content), "prompt_input": Output(prompt)},
                             usage=Usage(tokens_input=_usage_tokens(prompt.estimated_input_tokens), tokens_output=_usage_tokens(80)))


# --- contact signals ---------------------------------------------------------------------------


def _signal(n: int, kind: ContactSignalKind, turn: TranscriptTurnContent) -> ContactSignalView:
    quote = _short(turn.text, 80).rstrip(".")
    if quote not in turn.text:
        quote = turn.text[:80]
    start = turn.text.find(quote)
    return ContactSignalView(id=f"s{n}", kind=kind, label=kind.value.replace("_", " ").capitalize(), start=turn.start_time, end=turn.end_time,
                             speaker=turn.speaker, quote=quote, turn_id=turn.turn_id, char_start=start, char_end=start + len(quote), confidence=0.8)


class _FakePass(_Fake):
    pass_kind: ContactSignalPass

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        transcript = job.transcript()
        window = job.parameters.window
        turns = turns_in(transcript, window)
        agents = [t for t in turns if t.speaker is SpeakerRole.AGENT]
        callers = [t for t in turns if t.speaker is SpeakerRole.CALLER]
        signals: List[ContactSignalView] = []
        if self.pass_kind is ContactSignalPass.LIFECYCLE:
            if callers:
                signals.append(_signal(1, ContactSignalKind.INTENT, callers[0]))
            if len(agents) > 1:
                signals.append(_signal(2, ContactSignalKind.FIX_PROPOSED, agents[1]))
        else:
            if agents:
                signals.append(_signal(1, ContactSignalKind.AGENT_REPORTS_COMPLETED, agents[-1]))
            if callers:
                signals.append(_signal(2, ContactSignalKind.CALLER_CONFIRMS_RESOLVED, callers[-1]))
        prompt = prompt_input_for(job, f"fake.{self.job_type.value}", transcript, window)
        content = ContactSignalsPassContent(pass_kind=self.pass_kind, window=window, signals=signals)
        return HandlerResult(outputs={"pass": Output(content), "prompt_input": Output(prompt)},
                             usage=Usage(tokens_input=_usage_tokens(prompt.estimated_input_tokens), tokens_output=_usage_tokens(50)))


class FakeLifecyclePass(_FakePass):
    job_type = JobType.CONTACT_SIGNALS_LIFECYCLE
    pass_kind = ContactSignalPass.LIFECYCLE


class FakeResolutionPass(_FakePass):
    job_type = JobType.CONTACT_SIGNALS_RESOLUTION
    pass_kind = ContactSignalPass.RESOLUTION


# --- Contact Signals v2 fakes (section 8.6) ----------------------------------------------------

FAKE_SIGNAL_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "intent": ("question about", "calling about", "i want to", "i'd like to", "i would like to"),
    "issue": ("fee", "charged", "problem with"),
    "friction": ("frustrat", "keeps happening", "third time"),
    "fix_proposed": ("you can dispute", "i can offer", "we can offer", "let me reset"),
    "agent_reports_completed": ("you are verified", "i have cancelled", "i've cancelled", "has been processed", "is done"),
    "caller_confirms_resolved": ("makes sense", "that works", "that fixed"),
    "caller_reports_unresolved": ("still not working", "doesn't work", "didn't help"),
    "deferred": ("call you back", "get back to you", "follow up", "escalate"),
}
"""The fixed fake keywords of the built-in categories (case-insensitive substrings). A custom category
fires on its own ``examples``. On the default ``SCRIPT``, caller turn 1 ("a question about a fee")
fires both ``intent`` and ``issue``: co-occurrence."""

FAKE_CALIBRATION_ID = "fake-signals-v1"
FAKE_NOT_PHRASE = "not really"


class FakeSignalClassifier:
    """Stages 1 and 2 without a model. Stage 1 scores 0.9 on every option whose category's keyword or
    example the segment contains, 0.02 on every other option, and 'none' takes what remains (never
    below 0). The thresholded pick then fires at most two per segment. Stage 2 picks the first subcategory whose examples appear in the span, else 'Other', and
    'Not' when the span says "not really"."""

    entry_id = "fake-signal-classifier"
    calibration_id = FAKE_CALIBRATION_ID
    key_orders = 1
    budget = RowBudget()

    def __init__(self, taxonomy: SignalTaxonomy, stage: int, action: str = "ok", entry_id: Optional[str] = None) -> None:
        self.taxonomy = taxonomy
        self.stage = stage
        self.action = action
        if entry_id:
            self.entry_id = entry_id
        self.loads = 0
        self.rows: List[ChoiceRow] = []
        self.thresholds = resolve_thresholds(taxonomy, FAKE_ENGINE_DEFAULTS.stage1, FAKE_ENGINE_DEFAULTS.stage1_default)

    def load(self) -> None:
        self.loads += 1

    def release(self) -> None:
        return None

    def keywords(self, category_id: str) -> List[str]:
        category = self.taxonomy.category(category_id)
        examples = list(category.examples) if category is not None else []
        return [k.lower() for k in examples + list(FAKE_SIGNAL_KEYWORDS.get(category_id, ()))]

    def choose(self, rows: Sequence[ChoiceRow]) -> List[Dict[str, float]]:
        self.rows.extend(rows)
        if self.action == "invalid_answer":
            raise EngineError("validation_rejected", "scripted fake invalid answer")
        if self.action == "provider_error":
            raise EngineError("provider_error", "scripted fake provider error")
        return [self._stage1(row) if self.stage == 1 else self._stage2(row) for row in rows]

    def _stage1(self, row: ChoiceRow) -> Dict[str, float]:
        text = str(row.state.get("turn") or "").lower()
        options = [o for o, _ in row.options if o != SIGNAL_NONE_OPTION]
        matched = [o for o in options if any(k in text for k in self.keywords(o))]
        p = {o: 0.02 for o in options}
        for o in matched:
            p[o] = max(0.02, round(self.thresholds.get(o, 0.3) - 0.05, 6)) if self.action == "low_confidence" else 0.9
        p[SIGNAL_NONE_OPTION] = round(max(0.0, 1.0 - sum(p.values())), 6)
        return p

    def _stage2(self, row: ChoiceRow) -> Dict[str, float]:
        text = str(row.state.get("turn") or "").lower()
        category = self.taxonomy.category(row.key.rsplit(".t", 1)[0])
        options = [o for o, _ in row.options]
        pick = SIGNAL_OTHER_OPTION
        if FAKE_NOT_PHRASE in text:
            pick = SIGNAL_NOT_OPTION
        elif category is not None:
            for sid, _ in stage2_options(category):
                sub = next(s for s in category.subcategories if s.subcategory_id == sid)
                if any(e.lower() in text for e in sub.examples):
                    pick = sid
                    break
        top = 0.8
        if self.action == "low_confidence" and pick not in (SIGNAL_OTHER_OPTION, SIGNAL_NOT_OPTION):
            tau = category.subcategory_threshold if category is not None and category.subcategory_threshold is not None else FAKE_ENGINE_DEFAULTS.subcategory
            top = max(0.0, tau - 0.05)
        p = {o: 0.0 for o in options}
        p[pick] = top
        others = [o for o in options if o != pick]
        for o in others:
            p[o] = round((1.0 - top) / len(others), 6) if others else 0.0
        if pick != SIGNAL_NOT_OPTION and SIGNAL_NOT_OPTION in p:
            p[SIGNAL_NOT_OPTION] = min(p[SIGNAL_NOT_OPTION], 0.05)
        return p


_SENTENCE = re.compile(r"[^.?!]+[.?!]?")


class FakeSignalExtractor:
    """Stage 3 without a model (section 8.6): an enum field takes the first of its values found in the
    span (with that text as its evidence), a string field the span's first sentence, number and amount
    fields the first number the pre-split numeric extractor finds; dates and booleans stay absent.
    ``narrow_quote`` quotes the first example found in the span, else its first sentence. Grounding
    then runs for real."""

    requires_evidence = True

    def __init__(self, entry_id: str = "fake-signal-extractor", action: str = "ok") -> None:
        self.entry_id = entry_id
        self.action = action
        self.calls: List[List[str]] = []
        self.released = 0

    def release(self) -> None:
        self.released += 1

    def extract(self, spans: Sequence[ExtractionSpan]) -> List[RawExtraction]:
        self.calls.append([s.span_key for s in spans])
        if self.action in ("invalid_answer", "provider_error"):
            code = "validation_rejected" if self.action == "invalid_answer" else "provider_error"
            return [RawExtraction(span_key=s.span_key, status="error", error_code=code) for s in spans]
        return [self._one(s) for s in spans]

    def _one(self, span: ExtractionSpan) -> RawExtraction:
        from call1.pipeline.numeric_extractor import extract_numeric_references

        text = span.text
        lowered = text.lower()
        first = _SENTENCE.match(text.strip())
        sentence = first.group(0).strip() if first else text.strip()
        values: Dict[str, object] = {}
        evidence: Dict[str, str] = {}
        for f in span.fields:
            if f.type is SignalFieldType.ENUM:
                for value in f.enum_values:
                    at = lowered.find(value.lower())
                    if at >= 0:
                        values[f.field_id] = value
                        evidence[f.field_id] = text[at:at + len(value)]
                        break
            elif f.type is SignalFieldType.STRING:
                if sentence and sentence in text:
                    values[f.field_id] = sentence
            elif f.type in (SignalFieldType.NUMBER, SignalFieldType.AMOUNT):
                found = [e.raw_text for e in extract_numeric_references(text) if e.raw_text in text]
                if found:
                    values[f.field_id] = found[0]
        if span.narrow_quote:
            quote = next((text[lowered.find(e.lower()):lowered.find(e.lower()) + len(e)] for e in span.examples if e.lower() in lowered), sentence)
            values["quote"] = quote
        return RawExtraction(span_key=span.span_key, values=values, evidence=evidence)


def _snapshot(job: HandlerJob) -> SignalTaxonomySnapshotContent:
    return job.require("taxonomy").content()  # type: ignore[return-value]


class FakeSignalsCategorize(_Fake):
    job_type = JobType.CONTACT_SIGNALS_CATEGORIZE

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        params = job.parameters.signals
        engine = None
        if params is None or params.stage1_mode != "rederive":
            entry = job.selection.catalog_entry.entry_id if job.selection else None
            engine = FakeSignalClassifier(_snapshot(job).taxonomy, 1, action, entry_id=entry)
        content = run_categorize(job, engine, adapter_version=FAKE_ADAPTER_VERSION, device="fake")
        return HandlerResult(outputs={"categories": Output(content)})


class FakeSignalsSubcategorize(_Fake):
    job_type = JobType.CONTACT_SIGNALS_SUBCATEGORIZE

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        entry = job.selection.catalog_entry.entry_id if job.selection else None
        engine = FakeSignalClassifier(_snapshot(job).taxonomy, 2, action, entry_id=entry)
        content = run_subcategorize(job, engine, adapter_version=FAKE_ADAPTER_VERSION, device="fake")
        return HandlerResult(outputs={"subcategories": Output(content)})


class FakeSignalsExtract(_Fake):
    """The primary is a fake extractor answering with the scripted action; the declared in-job
    fallback (``parameters.signals.fallback_entry_id``) is another fake extractor that answers ``ok``."""

    job_type = JobType.CONTACT_SIGNALS_EXTRACT

    def produce(self, job: HandlerJob, action: str) -> HandlerResult:
        entry = job.selection.catalog_entry.entry_id if job.selection else "fake-signal-extractor"
        primary = ExtractEngine(FakeSignalExtractor(entry, action), calibration_id=FAKE_CALIBRATION_ID, device="fake",
                                adapter_version=FAKE_ADAPTER_VERSION, model_revision="fake")
        params = job.parameters.signals
        fallback_id = params.fallback_entry_id if params is not None else None

        def fallback() -> Optional[ExtractEngine]:
            if not fallback_id:
                return None
            return ExtractEngine(FakeSignalExtractor(fallback_id, "ok"), calibration_id=FAKE_CALIBRATION_ID, device="fake",
                                 adapter_version=FAKE_ADAPTER_VERSION, model_revision="fake")

        content = run_extract(job, primary, fallback)
        prompt = prompt_input_for(job, "fake.contact_signals_extract", job.transcript())
        return HandlerResult(outputs={"extraction": Output(content), "prompt_input": Output(prompt)},
                             usage=Usage(tokens_input=_usage_tokens(prompt.estimated_input_tokens), tokens_output=_usage_tokens(40)))


FAKE_HANDLER_CLASSES = (
    FakeValidationVad, FakeAsr, FakeSpeakerAttribution, FakeAcousticTone, FakeTextSentiment, FakeEnrichment, FakeQaDeterministic,
    FakeQaCriterion, FakeQaEscalation, FakeSummarySegment, FakeSummarySynthesis, FakeLifecyclePass, FakeResolutionPass,
    FakeSignalsCategorize, FakeSignalsSubcategorize, FakeSignalsExtract,
)


def fake_handlers(behavior: Optional[FakeBehavior] = None) -> List[Handler]:
    behavior = behavior or FakeBehavior()
    return [cls(behavior) for cls in FAKE_HANDLER_CLASSES]
