"""Dual transcription on the Process side (team decision 33, docs/DualAsr.md section 5).

* **Reading the vocabulary.** ``read_vocabulary`` reads ``GET /store/v1/vocabulary``
  (``getAsrVocabulary``) at ingest and at full reanalysis. A Store that has not built the route yet
  (501 ``not_implemented``, 404) means no vocabulary: the graph is planned with Parakeet alone, as
  before 1.3.0.
* **Freezing it.** ``vocabulary_parameters`` turns an ``active`` record into the ``asr`` job's
  ``parameters.asr_vocabulary``: the effective terms, their digest and the catalog's
  ``asr_vocabulary`` entry (Whisper Small), frozen even when that entry is not installed here, so the
  transcript says why nothing was corrected. With no ``asr_vocabulary`` entry in the catalog at all,
  nothing is frozen.
* **The transcript's provenance.** ``applied_correction`` and ``base_only_correction`` build
  ``TranscriptContent.vocabulary_correction`` for the real and the fake ``asr`` handlers alike.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from call1.contracts.catalog import ModelPurpose
from call1.contracts.contents import (
    AsrPassWord,
    TranscriptContent,
    VocabularyCorrection,
    VocabularyCorrectionStatus,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.vocabulary import AsrVocabularyParameters, AsrVocabularyRecord
from call1.pipeline.vocabulary_merge import MergeOutcome, merge_transcript

from .catalog import CatalogEntry, ProcessCatalog
from .store_client import StoreClient, StoreError

log = logging.getLogger("call1.process.vocabulary")

WHISPER_VOCAB_ENTRY_ID = "whisper-small-vocab"

DUAL_ASR_RUNTIME_FACTOR = 3.5
"""The ``asr`` job with a vocabulary pass takes about 3.5x Parakeet alone (Parakeet 7 s, prompted
Whisper Small 18 s per 10-minute call; benchmarks/2026-09-26-dual-asr-merge.md)."""

PARAKEET_REAL_TIME_FACTOR = 0.012
"""Parakeet's measured real-time factor on an M3 Pro, including its per-call model load."""


def _unsupported(exc: StoreError) -> bool:
    return exc.status in (404, 501) and exc.code in ("not_implemented", "not_found", "unexpected_response")


def read_vocabulary(client: StoreClient) -> Optional[AsrVocabularyRecord]:
    """The Store's vocabulary record, or None when this Store has no vocabulary route yet."""
    try:
        return client.get_asr_vocabulary()
    except StoreError as exc:
        if _unsupported(exc):
            log.info("Store has no ASR vocabulary (%s); planning ASR without the vocabulary pass", exc.code)
            return None
        raise


def vocabulary_entry(catalog: ProcessCatalog) -> Optional[CatalogEntry]:
    """The catalog's ``asr_vocabulary`` entry (its default for the purpose), installed or not."""
    entry_id = catalog.defaults.get(ModelPurpose.ASR_VOCABULARY)
    entry = catalog.entries.get(entry_id) if entry_id else None
    if entry is None or ModelPurpose.ASR_VOCABULARY not in entry.purposes:
        return None
    return entry


def vocabulary_parameters(record: Optional[AsrVocabularyRecord], catalog: ProcessCatalog) -> Optional[AsrVocabularyParameters]:
    """What a new ``asr`` job freezes: None unless the record is ``active`` and the catalog has an
    ``asr_vocabulary`` entry."""
    if record is None or not record.active or not record.effective_terms or record.effective_digest is None:
        return None
    entry = vocabulary_entry(catalog)
    if entry is None:
        log.warning("the vocabulary is on but this Process catalog has no asr_vocabulary entry; planning ASR without it")
        return None
    return AsrVocabularyParameters(terms=list(record.effective_terms), digest=record.effective_digest, candidate_entry=entry.ref)


def base_only_correction(params: AsrVocabularyParameters, *, base_engine: str, note: str, code: JobErrorCode,
                         candidate_engine: Optional[str] = None, candidate_model_revision: Optional[str] = None,
                         glossary_term_count: int = 0) -> VocabularyCorrection:
    """The transcript's provenance when the vocabulary pass did not run or failed: Parakeet alone, with
    the reason. Never a failed call."""
    return VocabularyCorrection(
        status=VocabularyCorrectionStatus.BASE_ONLY, note=note[:2000], failure_code=code, vocabulary_digest=params.digest,
        term_count=len(params.terms), glossary_term_count=glossary_term_count, base_engine=base_engine[:200],
        candidate_engine=candidate_engine[:200] if candidate_engine else None,
        candidate_model_revision=candidate_model_revision[:200] if candidate_model_revision else None, rule=params.rule,
    )


def merge(base: TranscriptContent, pass_words: Iterable[AsrPassWord], params: AsrVocabularyParameters, *, channels: int) -> MergeOutcome:
    return merge_transcript(base.turns, list(pass_words), params.terms, params.rule, channels=channels)


def applied_correction(params: AsrVocabularyParameters, outcome: MergeOutcome, *, base_engine: str, candidate_engine: str,
                       candidate_model_revision: Optional[str], glossary_term_count: int, note: Optional[str] = None) -> VocabularyCorrection:
    return VocabularyCorrection(
        status=VocabularyCorrectionStatus.APPLIED, note=note[:2000] if note else None, vocabulary_digest=params.digest,
        term_count=len(params.terms), glossary_term_count=glossary_term_count, base_engine=base_engine[:200],
        candidate_engine=candidate_engine[:200], candidate_model_revision=candidate_model_revision[:200] if candidate_model_revision else None,
        rule=params.rule, candidates=outcome.candidates, replacements=list(outcome.replacements),
    )


def corrected(base: TranscriptContent, correction: VocabularyCorrection, outcome: Optional[MergeOutcome] = None) -> TranscriptContent:
    """The ``asr`` job's ``transcript``: the merged turns (or the base turns on ``base_only``) with
    ``vocabulary_correction`` set."""
    turns: List = list(outcome.turns) if outcome is not None else list(base.turns)
    return base.model_copy(update={"turns": turns, "vocabulary_correction": correction})


def has_word_timings(base: TranscriptContent) -> bool:
    return any(turn.word_timestamps for turn in base.turns)


__all__ = [
    "DUAL_ASR_RUNTIME_FACTOR", "PARAKEET_REAL_TIME_FACTOR", "WHISPER_VOCAB_ENTRY_ID", "applied_correction", "base_only_correction",
    "corrected", "has_word_timings", "merge", "read_vocabulary", "vocabulary_entry", "vocabulary_parameters",
]
