"""Conversions between the contract's artifact content and the pre-split pipeline's models
(``call1.models.schemas``), so the real handlers call the legacy analysis code unchanged.

The legacy models carry a whole call in one ``CallTranscript`` (turns with their per-turn analyses,
tone blocks, VAD). The contract splits that into separate artifacts; these helpers put the pieces a
stage needs back together and turn the legacy results into contract content.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

from call1.contracts.contents import (
    EnrichmentContent,
    NumericEntityContent,
    SpeakerRole,
    TextSentimentContent,
    ToneBlocksContent,
    ToneBlockStatus,
    ToneBlockView,
    TranscriptContent,
    TranscriptTurnContent,
    VerdictStatus,
    VerdictView,
    WordTimestampView,
)
from call1.contracts.rubrics import RubricCriterion, RubricDefinition
from call1.models import schemas as legacy


def _finite(value: Optional[float], low: Optional[float] = None, high: Optional[float] = None) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


# --- transcript ------------------------------------------------------------------------------


def contract_turn(turn: legacy.TranscriptTurn) -> TranscriptTurnContent:
    """A legacy ASR turn as ``transcript.v1`` content (``raw_text`` and analyses are not part of it)."""
    start = max(0.0, float(turn.start_time))
    end = max(start, float(turn.end_time))
    words = None
    if turn.word_timestamps is not None:
        words = []
        for w in turn.word_timestamps:
            w_start = max(0.0, float(w.start_time))
            words.append(WordTimestampView(word=w.word, start_time=w_start, end_time=max(w_start, float(w.end_time)),
                                           probability=_finite(w.probability, 0.0, 1.0) or 0.0))
    return TranscriptTurnContent(
        turn_id=int(turn.turn_id), speaker=SpeakerRole(turn.speaker.value), speaker_cluster=turn.speaker_cluster, start_time=start,
        end_time=end, text=turn.text, channel=turn.channel, confidence=_finite(turn.confidence, 0.0, 1.0), word_timestamps=words,
    )


def legacy_turns(transcript: TranscriptContent, enrichment: Optional[EnrichmentContent] = None) -> List[legacy.TranscriptTurn]:
    """Legacy turns (speaker labels as the contract transcript carries them), with the numeric
    entities of ``enrichment`` attached when given (masking reads them)."""
    entities: Dict[int, list] = {}
    if enrichment is not None:
        for row in enrichment.turns:
            entities[row.turn_id] = [legacy.NumericEntity(raw_text=e.raw_text, normalized_value=e.normalized_value,
                                                          entity_type=legacy.NumericEntityType(e.entity_type), start_time=e.start_time,
                                                          end_time=e.end_time, unit=e.unit) for e in row.numeric_entities]
    turns = []
    for turn in transcript.turns:
        words = None
        if turn.word_timestamps is not None:
            words = [legacy.WordTimestamp(word=w.word, start_time=w.start_time, end_time=w.end_time, probability=w.probability)
                     for w in turn.word_timestamps]
        turns.append(legacy.TranscriptTurn(
            turn_id=turn.turn_id, speaker=legacy.SpeakerRole(turn.speaker.value), start_time=turn.start_time, end_time=turn.end_time,
            text=turn.text, raw_text=turn.text, channel=turn.channel, speaker_cluster=turn.speaker_cluster, confidence=turn.confidence,
            word_timestamps=words, numeric_entities=entities.get(turn.turn_id),
        ))
    return turns


def legacy_transcript(transcript: TranscriptContent, *, call_id: str = "split", enrichment: Optional[EnrichmentContent] = None,
                      sentiment: Optional[TextSentimentContent] = None, tone: Optional[ToneBlocksContent] = None) -> legacy.CallTranscript:
    """A legacy ``CallTranscript`` with the per-turn text sentiment and the tone blocks the
    deterministic checks read (``call1.pipeline.sentiment_rules``)."""
    turns = legacy_turns(transcript, enrichment)
    if sentiment is not None:
        by_turn = {s.turn_id: s for s in sentiment.turns}
        for turn in turns:
            row = by_turn.get(turn.turn_id)
            if row is None:
                continue
            turn.text_sentiment = row.score
            turn.text_sentiment_label = row.label.value if row.label is not None else None
            turn.text_analysis = {"model": sentiment.model, "revision": sentiment.revision, "status": row.status,
                                  "probabilities": dict(row.probabilities)}
    blocks = legacy_tone_blocks(tone, turns) if tone is not None else []
    return legacy.CallTranscript(call_id=call_id, turns=turns, tone_blocks=blocks, duration_seconds=transcript.duration_seconds,
                                 avg_caller_sentiment=sentiment.avg_caller_sentiment if sentiment else None,
                                 avg_agent_tone=tone.avg_agent_tone if tone else None)


def speech_intervals(block: ToneBlockView, turns: Sequence[legacy.TranscriptTurn], stereo: bool) -> List[tuple]:
    """The block's attributed speech intervals as ``call1.pipeline.sentiment._tone_blocks`` computed
    them (the speaker's own turns inside the block, minus other speakers' turns on the same audio);
    the tone artifact does not carry them, and the tone-metric checks weight by them."""
    from call1.pipeline.sentiment import _subtract, _union

    start, end = block.start_time, block.end_time
    active = [t for t in turns if t.start_time < end and t.end_time > start]
    own = [t for t in active if t.speaker.value == block.speaker.value]
    intervals = [(max(start, t.start_time), min(end, t.end_time)) for t in own]
    if stereo:
        channels = {t.channel for t in own if t.channel in (0, 1)}
        if len(channels) > 1:
            return []
        channel = next(iter(channels)) if channels else None
        conflicts = [(max(start, t.start_time), min(end, t.end_time)) for t in active
                     if t.speaker.value != block.speaker.value and channel is not None and t.channel == channel]
    else:
        conflicts = [(max(start, t.start_time), min(end, t.end_time)) for t in active if t.speaker.value != block.speaker.value]
    return _subtract(_union(intervals), conflicts)


def legacy_tone_blocks(tone: ToneBlocksContent, turns: Sequence[legacy.TranscriptTurn]) -> List[legacy.ToneBlock]:
    stereo = any(t.channel == 1 for t in turns)
    blocks = []
    for block in tone.blocks:
        blocks.append(legacy.ToneBlock(
            analysis_version=block.analysis_version, block_id=block.block_id, speaker=legacy.SpeakerRole(block.speaker.value),
            start_time=block.start_time, end_time=block.end_time, turn_ids=list(block.turn_ids), speech_seconds=block.speech_seconds,
            speech_intervals=speech_intervals(block, turns, stereo), status=block.status.value,
            valence=block.valence, arousal=block.arousal, dominance=block.dominance, emotion=block.emotion,
            emotion_probabilities=block.emotion_probabilities, model=block.model, revision=block.revision,
        ))
    return blocks


def contract_tone_block(block: legacy.ToneBlock) -> ToneBlockView:
    def unit(value):
        return _finite(value, 0.0, 1.0)

    probabilities = None
    if block.emotion_probabilities is not None:
        probabilities = {str(k): float(v) for k, v in block.emotion_probabilities.items() if math.isfinite(float(v))}
    return ToneBlockView(
        block_id=int(block.block_id), speaker=SpeakerRole(block.speaker.value), start_time=float(block.start_time), end_time=float(block.end_time),
        status=ToneBlockStatus(block.status), speech_seconds=max(0.0, float(block.speech_seconds)), valence=unit(block.valence),
        arousal=unit(block.arousal), dominance=unit(block.dominance), emotion=block.emotion, emotion_probabilities=probabilities,
        turn_ids=[int(t) for t in block.turn_ids], model=block.model[:200], revision=block.revision[:200],
        analysis_version=block.analysis_version[:200],
    )


def contract_entities(entities: Iterable[legacy.NumericEntity]) -> List[NumericEntityContent]:
    out = []
    for entity in entities:
        value = entity.normalized_value
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            value = str(value)
        elif isinstance(value, int):
            value = float(value)
        out.append(NumericEntityContent(raw_text=entity.raw_text, normalized_value=value, entity_type=entity.entity_type.value,
                                        start_time=float(entity.start_time), end_time=float(entity.end_time), unit=entity.unit))
    return out


# --- rubric and verdicts ---------------------------------------------------------------------


def legacy_criterion(criterion: RubricCriterion) -> legacy.RubricCriterion:
    """The contract criterion in the pre-split shape (the field names are the same by design)."""
    data = criterion.model_dump(mode="json")
    if data.get("rule_type") is None:
        data.pop("rule_type", None)
    if data.get("parameters") is None:
        data.pop("parameters", None)
    return legacy.RubricCriterion.model_validate(data)


def legacy_rubric(definition: RubricDefinition) -> legacy.RubricDefinition:
    return legacy.RubricDefinition(rubric_id=definition.rubric_id, name=definition.name, description=definition.description,
                                   category=definition.category.value, pass_threshold=definition.pass_threshold,
                                   criteria=[legacy_criterion(c) for c in definition.criteria])


def contract_verdict(verdict: legacy.RubricVerdict, criterion: RubricCriterion, *, quote_turn_id: Optional[int] = None) -> VerdictView:
    timestamp = tuple(float(x) for x in verdict.timestamp_range) if verdict.timestamp_range else None
    return VerdictView(
        criterion_id=criterion.criterion_id, criterion_name=criterion.name, status=VerdictStatus(verdict.status.value),
        confidence=_finite(verdict.confidence, 0.0, 1.0) or 0.0, quoted_evidence=verdict.quoted_evidence,
        speaker=SpeakerRole(verdict.speaker.value), timestamp_range=timestamp, quote_turn_id=quote_turn_id, reasoning=verdict.reasoning,
        hallucination_detected=verdict.hallucination_detected,
    )


__all__ = [
    "contract_entities", "contract_tone_block", "contract_turn", "contract_verdict", "legacy_criterion", "legacy_rubric",
    "legacy_tone_blocks", "legacy_transcript", "legacy_turns", "speech_intervals",
]
