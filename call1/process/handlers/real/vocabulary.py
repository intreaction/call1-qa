"""Dual transcription inside the ``asr`` job (team decision 33, docs/DualAsr.md sections 2, 5 and 7).

After Parakeet, when the job carries ``parameters.asr_vocabulary``, ``correct`` runs the
vocabulary-prompted Whisper Small pass (``call1.pipeline.vocabulary_asr``) and the deterministic rule
merge (``call1.pipeline.vocabulary_merge``), and returns the ``asr`` outputs:

* ``transcript``: the merged transcript with ``vocabulary_correction`` (``applied``), or Parakeet's
  turns unchanged with ``vocabulary_correction`` ``base_only`` and a note;
* ``base_transcript`` (``asr_base_transcript``): Parakeet's own transcript, always;
* ``vocabulary_pass`` (``asr_vocabulary_pass``): the raw Whisper pass, only when it produced output.

**A vocabulary problem never fails the call** (section 7): a missing catalog entry or install
(``model_unavailable``), no MLX backend or no word timings (``configuration_error``), an import error
(``model_unavailable``), an audio decode error (``validation_rejected``), memory exhaustion
(``resource_unavailable``) or any other runtime error (``provider_error``) keeps Parakeet's transcript
as ``base_only``. Each is logged with the job ID and the exception type, never text. Cancellation
during the pass cancels the job, as today.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from call1.contracts.contents import AsrPassSegment, AsrPassWord, AsrVocabularyPassContent, TranscriptContent
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import ASR_BASE_TRANSCRIPT_ROLE, ASR_VOCABULARY_PASS_ROLE
from call1.pipeline import vocabulary_asr

from call1.process.catalog import CatalogEntry, ProcessCatalog, seeded_catalog
from call1.process.handlers.base import HandlerJob, JobCancelled, Output
from call1.process.vocabulary import applied_correction, base_only_correction, corrected, has_word_timings, merge

from .paths import mlx_backend, weights_path

log = logging.getLogger("call1.process.handlers.real.vocabulary")

CANDIDATE_ENGINE = "whisper-small"

NOTES = {
    "not_in_catalog": "The vocabulary model is not in this Process catalog, so the transcript was not corrected from the vocabulary.",
    "needs_mlx": "Vocabulary correction needs CALL1_BACKEND=mlx (Apple Silicon); the Parakeet transcript is kept.",
    "not_installed": "The vocabulary model is not installed ({problem}); provision it with scripts/provision_models.py --models asr_vocabulary.",
    "no_word_timings": "The base transcript has no word timings, so vocabulary corrections could not be aligned.",
    "runtime_missing": "The vocabulary runtime (mlx-audio, tiktoken) is not installed on this host.",
    "decode": "The recording could not be decoded for the vocabulary pass.",
    "memory": "The vocabulary pass ran out of memory; the Parakeet transcript is kept.",
    "runtime": "The vocabulary pass failed at runtime; the Parakeet transcript is kept.",
}
"""Safe notes for ``base_only`` (never transcript content)."""


@dataclass
class VocabularyResult:
    outputs: Dict[str, Output]
    inference_seconds: float = 0.0
    load_seconds: float = 0.0
    peak_memory_bytes: Optional[int] = None


class _BaseOnly(Exception):
    def __init__(self, code: JobErrorCode, note: str) -> None:
        super().__init__(note)
        self.code = code
        self.note = note


def resolve_entry(catalog: Optional[ProcessCatalog], job: HandlerJob) -> Optional[CatalogEntry]:
    """The frozen ``candidate_entry`` in this Process's catalog (same entry version), or None."""
    params = job.parameters.asr_vocabulary
    if params is None:
        return None
    catalog = catalog or seeded_catalog(mode="real")
    return catalog.by_ref(params.candidate_entry)


def base_words_by_channel(base: TranscriptContent, channels: int) -> Dict[Optional[int], List[Dict]]:
    out: Dict[Optional[int], List[Dict]] = {}
    for turn in base.turns:
        key = turn.channel if channels >= 2 else None
        for w in turn.word_timestamps or []:
            out.setdefault(key, []).append({"word": w.word, "start": w.start_time, "end": w.end_time})
    for words in out.values():
        words.sort(key=lambda w: w["start"])
    return out


def pass_content(result: Any, *, entry: CatalogEntry, duration: float, channels: int) -> AsrVocabularyPassContent:
    """The raw ``asr_vocabulary_pass.v1`` artifact from a ``vocabulary_asr.PassResult``."""
    return AsrVocabularyPassContent(
        engine=CANDIDATE_ENGINE, model_revision=entry.model_revision[:200], duration_seconds=max(0.0, float(duration)),
        channels=max(1, min(2, int(channels))), glossary_terms=[t[:200] for t in result.glossary_terms],
        words=[AsrPassWord(word=w.word, start_time=w.start, end_time=max(w.start, w.end), probability=w.probability,
                           channel=w.channel, segment=w.segment) for w in result.words],
        segments=[AsrPassSegment(start_time=s.start, end_time=max(s.start, s.end), channel=s.channel, avg_logprob=s.avg_logprob,
                                 no_speech_prob=s.no_speech_prob, compression_ratio=s.compression_ratio, temperature=s.temperature)
                  for s in result.segments],
    )


def correct(job: HandlerJob, base: TranscriptContent, audio_path: Path, channels: int, *, base_engine: str,
            catalog: Optional[ProcessCatalog] = None, runner: Any = None) -> VocabularyResult:
    """The vocabulary pass and merge for an ``asr`` job with ``parameters.asr_vocabulary``.
    ``runner`` stands in for ``vocabulary_asr.run_pass`` in tests (same signature); every check before
    it still applies."""
    params = job.parameters.asr_vocabulary
    assert params is not None
    entry = resolve_entry(catalog, job)
    candidate_engine = CANDIDATE_ENGINE if entry is not None else None
    revision = entry.model_revision if entry is not None else None
    outputs: Dict[str, Output] = {ASR_BASE_TRANSCRIPT_ROLE: Output(base)}
    try:
        if entry is None:
            raise _BaseOnly(JobErrorCode.MODEL_UNAVAILABLE, NOTES["not_in_catalog"])
        if not mlx_backend():
            raise _BaseOnly(JobErrorCode.CONFIGURATION_ERROR, NOTES["needs_mlx"])
        model_dir = weights_path(entry)
        problem = vocabulary_asr.install_problem(model_dir)
        if problem is not None:
            raise _BaseOnly(JobErrorCode.MODEL_UNAVAILABLE, NOTES["not_installed"].format(problem=problem))
        if not has_word_timings(base):
            raise _BaseOnly(JobErrorCode.CONFIGURATION_ERROR, NOTES["no_word_timings"])
        job.check_cancelled()
        job.progress(0.6, "vocabulary pass")
        result = (runner or vocabulary_asr.run_pass)(model_dir, audio_path, channels, [t.term for t in params.terms], base_words_by_channel(base, channels),
                        params.glossary_prompt_limit, check_cancelled=job.check_cancelled)
        job.check_cancelled()
        content = pass_content(result, entry=entry, duration=base.duration_seconds, channels=channels)
        outcome = merge(base, content.words, params, channels=channels)
        correction = applied_correction(params, outcome, base_engine=base_engine, candidate_engine=CANDIDATE_ENGINE,
                                        candidate_model_revision=revision, glossary_term_count=len(content.glossary_terms))
        outputs["transcript"] = Output(corrected(base, correction, outcome))
        outputs[ASR_VOCABULARY_PASS_ROLE] = Output(content)
        return VocabularyResult(outputs=outputs, inference_seconds=float(result.inference_seconds or 0.0),
                                load_seconds=float(result.load_seconds or 0.0), peak_memory_bytes=result.peak_memory_bytes)
    except JobCancelled:
        raise
    except _BaseOnly as skip:
        code, note, kind = skip.code, skip.note, "skipped"
    except ImportError:
        code, note, kind = JobErrorCode.MODEL_UNAVAILABLE, NOTES["runtime_missing"], "ImportError"
    except subprocess.CalledProcessError:
        code, note, kind = JobErrorCode.VALIDATION_REJECTED, NOTES["decode"], "CalledProcessError"
    except MemoryError:
        code, note, kind = JobErrorCode.RESOURCE_UNAVAILABLE, NOTES["memory"], "MemoryError"
    except Exception as exc:  # a vocabulary problem never fails the call (docs/DualAsr.md section 7)
        memory = "out of memory" in str(exc).lower() or "outofmemory" in type(exc).__name__.lower()
        code, note = (JobErrorCode.RESOURCE_UNAVAILABLE, NOTES["memory"]) if memory else (JobErrorCode.PROVIDER_ERROR, NOTES["runtime"])
        kind = type(exc).__name__
    log.warning("job %s: vocabulary correction did not run (%s, %s); the Parakeet transcript is kept", job.job.id, code.value, kind)
    correction = base_only_correction(params, base_engine=base_engine, note=note, code=code, candidate_engine=candidate_engine,
                                      candidate_model_revision=revision)
    outputs["transcript"] = Output(corrected(base, correction))
    return VocabularyResult(outputs=outputs)


__all__ = ["CANDIDATE_ENGINE", "NOTES", "VocabularyResult", "base_words_by_channel", "correct", "pass_content", "resolve_entry"]
