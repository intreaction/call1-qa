"""Transcript helpers shared by the graph planner, the fake handlers and the real handlers.

Pure functions over contract content models: apply a speaker attribution, apply a reviewer's
speaker correction, plan summary segments (the pre-split summarizer's chunking by turn count and
bytes), and the contact-signal transcript fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

from call1.contracts.contents import (
    SpeakerAssignment,
    SpeakerAttributionContent,
    SpeakerRole,
    TranscriptContent,
    TranscriptTurnContent,
    TurnWindow,
)
from call1.contracts.jobs import SpeakerCorrection

SUMMARY_SEGMENT_MAX_BYTES = 3500
"""The pre-split summarizer's per-chunk byte bound (``call1/summarizer.py``)."""


def apply_attribution(transcript: TranscriptContent, attribution: Optional[SpeakerAttributionContent]) -> TranscriptContent:
    """The transcript with speaker labels from ``attribution`` (turns it does not name keep theirs)."""
    if attribution is None:
        return transcript
    by_turn: Dict[int, SpeakerAssignment] = {a.turn_id: a for a in attribution.assignments}
    turns = []
    for turn in transcript.turns:
        assignment = by_turn.get(turn.turn_id)
        if assignment is None:
            turns.append(turn)
            continue
        turns.append(turn.model_copy(update={"speaker": assignment.speaker, "speaker_cluster": assignment.speaker_cluster or turn.speaker_cluster}))
    return transcript.model_copy(update={"turns": turns})


def apply_speaker_correction(transcript: TranscriptContent, current: Optional[SpeakerAttributionContent],
                             correction: SpeakerCorrection) -> SpeakerAttributionContent:
    """A reviewer's correction as a ``reviewer_correction`` attribution: the current labels with the
    corrected turn (or its whole speaker cluster) relabelled. A code stage; no model runs."""
    effective = apply_attribution(transcript, current)
    target = next((t for t in effective.turns if t.turn_id == correction.turn_id), None)
    if target is None:
        raise ValueError(f"turn {correction.turn_id} is not in the transcript")
    speaker = SpeakerRole(correction.speaker)
    corrected: List[int] = []
    assignments: List[SpeakerAssignment] = []
    for turn in effective.turns:
        relabel = turn.turn_id == correction.turn_id or (
            correction.apply_to_cluster and target.speaker_cluster is not None and turn.speaker_cluster == target.speaker_cluster)
        if relabel:
            corrected.append(turn.turn_id)
        assignments.append(SpeakerAssignment(turn_id=turn.turn_id, speaker=speaker if relabel else turn.speaker,
                                             speaker_cluster=turn.speaker_cluster, confidence=1.0 if relabel else None))
    return SpeakerAttributionContent(method="reviewer_correction", assignments=assignments, corrected_turn_ids=corrected)


def _line(turn: TranscriptTurnContent) -> str:
    return f"[{turn.turn_id}] {turn.speaker.value}: {turn.text}"


def plan_segments(transcript: TranscriptContent, batch_turns: int = 60, max_bytes: int = SUMMARY_SEGMENT_MAX_BYTES) -> List[TurnWindow]:
    """Summary segment windows over original turn IDs: at most ``batch_turns`` turns and about
    ``max_bytes`` of numbered lines per segment, as the pre-split summarizer chunked. A turn is never
    split across segments. Always at least one window, so an empty transcript still gets a summary
    job that says so."""
    turns = sorted(transcript.turns, key=lambda t: t.turn_id)
    if not turns:
        return [TurnWindow(turn_start=0, turn_end=0)]
    windows: List[TurnWindow] = []
    chunk: List[TranscriptTurnContent] = []
    size = 0
    for turn in turns:
        length = len(_line(turn).encode("utf-8"))
        if chunk and (len(chunk) >= batch_turns or size + length > max_bytes):
            windows.append(TurnWindow(turn_start=chunk[0].turn_id, turn_end=chunk[-1].turn_id))
            chunk, size = [], 0
        chunk.append(turn)
        size += length
    if chunk:
        windows.append(TurnWindow(turn_start=chunk[0].turn_id, turn_end=chunk[-1].turn_id))
    return windows


def turns_in(transcript: TranscriptContent, window: Optional[TurnWindow]) -> List[TranscriptTurnContent]:
    if window is None:
        return list(transcript.turns)
    return [t for t in transcript.turns if window.turn_start <= t.turn_id <= window.turn_end]


def transcript_fingerprint(transcript: TranscriptContent) -> str:
    """``sha256:`` of the turn sequence, as ``call1.pipeline.contact_signals.compute_transcript_fingerprint``
    computes it (turn ID, lower-case speaker, stripped text)."""
    payload = [(t.turn_id, t.speaker.value.lower(), t.text.strip()) for t in transcript.turns]
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """A rough token estimate (about 1.3 tokens per word) for admission, never billing."""
    return int(len(text.split()) * 1.3) + 1
