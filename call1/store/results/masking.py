"""Masking of reviewer reads (``admin.MaskingSettings.mask_reviewer_reads``).

Admin state is deferred in the Stage 2 split (``routing.DEFERRED``: getAdminState/changeAdminState),
so Store applies the contract default: ``mask_reviewer_reads = true``. Transcript text and word
timestamps, verdict evidence and reasoning, summary text, contact-signal quotes and search hits are
masked; audio is muted over the same values (``audio.py``).

The value rules are the pre-split ones, reused from ``call1.redaction`` (pure functions over dicts;
importing it loads no model): privacy-sensitive numeric entities (from the ``enrichment``
artifact) plus the PII regexes (SSN, card, phone, account number, PIN), every occurrence masked.

Since contract 1.2.0 (team decision 19) the values also include the model PII layer's findings:
the ``pii_findings`` artifact Process's ``enrichment`` job writes per transcript revision (names,
addresses, emails, URLs, secrets; ``openai/privacy-filter`` or the labelled stub). Store runs no
model for this; it only reads the artifact. Findings count only when they were made from the call's
current published transcript (``PiiFindingsContent.transcript.checksum``). Until then a masked read
**fails closed** (``Masker.withheld``): transcript turns are served with empty text and no word
timestamps (``TranscriptView.text_withheld``), every other text field reads ``[REDACTED]``, search
skips the call and audio is refused. This is approximate masking, not certified PII removal.

By position (team decision 22, privacy-gap fixes). ``call_masks`` builds, from the transcript Store
already has, the spans masked *in place* in each turn (``call1.redaction.TurnPositions``):

- the digits spoken inside a strong card read-out window (``call1.redaction.read_out_digit_spans``),
  so a CVV the ASR split into "seven" / "Two four" is masked there without hiding every "seven";
- each PII finding by its ``turn_id`` and ``start``/``end`` (the contract fields). A finding made
  only of common words is dropped (``call1.pii_model.keep_span``); only strong identifiers
  (``call1.pii_model.strong_identifier``: emails, URLs, number runs, multi-word names, capitalized
  names of 3+ characters) are also masked by value across the call. A finding whose offsets do not
  fit its turn text is masked by value instead (fail closed).

Turn text and word timestamps get their own turn's spans; derived text (quotes, reasoning,
summaries) gets the spans of the turn it repeats verbatim plus the anchored neighbourhoods
(``TurnPositions.spans``). Audio mutes the same spans and every read-out window whole
(``audio.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from call1 import pii_model
from call1.contracts.admin import MaskingSettings
from call1.contracts.contents import EnrichmentContent, PiiFindingsContent, TranscriptContent
from call1.redaction import (REDACTED, TurnPositions, extract_sensitive_values, mask_text, mask_word_timestamps,
                             read_out_digit_spans)
from call1.redaction import _all_value_spans, _merge  # the span arithmetic mask_word_timestamps uses

PII_PATTERNS = True
"""The pre-split default (``RedactionSettings.pii_patterns``)."""


def masking_enabled() -> bool:
    """Whether reviewer reads are masked: the contract default while admin state is deferred."""
    return MaskingSettings().mask_reviewer_reads


def turn_dicts(transcript: TranscriptContent, enrichment: Optional[EnrichmentContent]) -> List[Dict[str, Any]]:
    """Turns in the shape ``call1.redaction`` reads: text, numeric entities, word timestamps."""
    entities: Dict[int, List[Dict[str, Any]]] = {}
    if enrichment is not None:
        for turn in enrichment.turns:
            entities[turn.turn_id] = [e.model_dump(mode="json") for e in turn.numeric_entities]
    return [
        {
            "turn_id": t.turn_id,
            "text": t.text,
            "start_time": t.start_time,
            "end_time": t.end_time,
            "numeric_entities": entities.get(t.turn_id, []),
            "word_timestamps": [w.model_dump(mode="json") for w in t.word_timestamps] if t.word_timestamps else [],
        }
        for t in transcript.turns
    ]


def _finding_spans(findings: Optional[PiiFindingsContent]) -> Dict[int, List[pii_model.PiiSpan]]:
    """The kept findings (common-word spans dropped) by turn id."""
    if findings is None:
        return {}
    out: Dict[int, List[pii_model.PiiSpan]] = {}
    for turn in findings.turns:
        spans = [pii_model.PiiSpan(str(s.category), s.start, s.end, s.text) for s in turn.spans]
        out.setdefault(turn.turn_id, []).extend(span for span in spans if pii_model.keep_span(span))
    return out


def call_masks(turns: List[Dict[str, Any]], findings: Optional[PiiFindingsContent]) -> Tuple[Set[str], List[List[Tuple[int, int]]]]:
    """(values the findings add *by value*, spans masked *by position* per turn, aligned with
    ``turns`` from ``turn_dicts``): the read-out window digits plus every kept finding that fits its
    turn text. Strong identifiers, and findings whose offsets do not fit their text, are values."""
    texts = [str(t.get("text") or "") for t in turns]
    cased = pii_model.transcript_cased(texts)
    positions = read_out_digit_spans(turns)
    by_turn = _finding_spans(findings)
    values: Set[str] = set()
    for index, turn in enumerate(turns):
        text = texts[index]
        for span in by_turn.get(turn.get("turn_id"), []):
            if 0 <= span.start < span.end <= len(text) and text[span.start:span.end] == span.text:
                positions[index].append((span.start, span.end))
            else:
                values.add(span.text)
            if pii_model.strong_identifier(span, cased):
                values.add(span.text)
    return values, positions


def findings_values(findings: Optional[PiiFindingsContent], transcript: Optional[TranscriptContent] = None) -> Set[str]:
    """The span texts the findings mask *by value* across the call: their strong identifiers
    (casing judged from ``transcript`` when given). Every kept finding is also masked by position
    (``call_masks``)."""
    if findings is None:
        return set()
    cased = pii_model.transcript_cased(t.text for t in transcript.turns) if transcript is not None else True
    return {span.text for spans in _finding_spans(findings).values() for span in spans if pii_model.strong_identifier(span, cased)}


def findings_match(findings: Optional[PiiFindingsContent], transcript_checksum: Optional[str]) -> bool:
    """Whether the findings were made from this transcript revision."""
    return findings is not None and transcript_checksum is not None and findings.transcript.checksum == transcript_checksum


class Masker:
    """The sensitive values of one call, applied consistently to every surface of a read.

    ``withheld`` (masking on, but no PII findings for the current transcript revision): nothing
    derived from the call's text may be shown, so ``text`` answers ``[REDACTED]`` for any non-empty
    value and ``words`` answers None."""

    def __init__(self, values: Set[str], enabled: bool, withheld: bool = False, positions: Optional[TurnPositions] = None) -> None:
        self.values = values
        self.enabled = enabled
        self.withheld = enabled and withheld
        self.positions = positions if positions is not None else TurnPositions.empty()

    @classmethod
    def disabled(cls) -> "Masker":
        return cls(set(), False)

    @classmethod
    def for_call(cls, transcript: Optional[TranscriptContent], enrichment: Optional[EnrichmentContent],
                 findings: Optional[PiiFindingsContent] = None, transcript_checksum: Optional[str] = None) -> "Masker":
        """Rule values over ``transcript`` (+ ``enrichment``) united with ``findings``' span texts.
        Withheld when the findings are not for ``transcript_checksum`` (or there is no transcript)."""
        if not masking_enabled():
            return cls.disabled()
        if transcript is None:
            return cls(set(), True, withheld=True)
        turns = turn_dicts(transcript, enrichment)
        values = extract_sensitive_values(turns, PII_PATTERNS)
        if not findings_match(findings, transcript_checksum):
            return cls(values, True, withheld=True)
        extra, spans = call_masks(turns, findings)
        return cls(values | extra, True, positions=TurnPositions([t["text"] for t in turns], spans))

    def text(self, value: Optional[str]) -> Optional[str]:
        if value is None or not self.enabled:
            return value
        if self.withheld:
            return REDACTED if value else value
        return mask_text(value, self.values, self.positions)

    def words(self, words: Optional[List[Dict[str, Any]]], text: str) -> Optional[List[Dict[str, Any]]]:
        if self.withheld:
            return None
        if not words or not self.enabled:
            return words
        return mask_word_timestamps(words, text, self.values, self.positions)

    def spans(self, text: Optional[str]) -> List[Tuple[int, int]]:
        """The merged character spans of ``text`` (a turn text) that a masked read replaces: the value
        occurrences plus the positional spans, as ``words`` computes them. Empty when masking is off.
        Callers that map offsets into the masked text must still check the result against it
        (``records.vocabulary_correction_view``)."""
        if not self.enabled or not text:
            return []
        return _merge(_all_value_spans(text, self.values) + self.positions.spans(text))


def merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ordered = sorted(intervals)
    merged: List[Tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


__all__ = ["REDACTED", "Masker", "call_masks", "findings_match", "findings_values", "masking_enabled", "turn_dicts", "merge_intervals"]
