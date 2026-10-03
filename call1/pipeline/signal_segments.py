"""Contact Signals v2, stage 0: the segmenter (docs/ContactSignalsV2.md section 2).

Pure code, no model. Process, the fake handlers and the bake-off harness all import it; it imports
neither ``call1.db`` nor Store. It cuts every speaker turn into windows of about 7 s:

* **Segments never cross a turn.** One speaker per segment (the target-speaker rule), evidence stays
  an exact substring of one turn, and a classifier's state is turn-shaped.
* **Partition.** A turn of duration d gets ``n = max(1, ceil(d / 7))`` windows with target cuts at
  ``k * d / n``. Each cut snaps to an inter-word gap within +/-1.5 s of its target, preferring
  sentence-final punctuation, then the longest gap. A cut never falls inside a masked value (an
  occurrence of a job sensitive value or a ``pii_findings`` span of the turn). A window with fewer
  than 3 words merges into its neighbour in the same turn.
* **No word timestamps** (``FakeAsr`` emits none, some imports lack them): cuts are placed by
  character proportion, snapped to whitespace, and the segment records ``timing: interpolated``.
* **Masked text only** (decisions 15 and 19). Cuts are made on the raw turn, where the word times
  live, and the offsets are mapped into the masked turn through the ordered list of replacements.
  Cuts never fall inside a masked value, so the mapping is exact and every segment's text is an exact
  substring of the masked turn (``masked_turn_text[char_start:char_end]``). Engines receive
  ``Segment.text`` only.
* **Speakers.** Stereo turns carry their channel's role; mono turns the attribution's, else
  ``UNKNOWN``. ``SYSTEM`` turns are skipped and counted. Stage 1 decides what ``UNKNOWN`` segments
  are scored against (only categories with no speaker scope; decision 22, Q4).

The hop equals the window: the partition does not overlap.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

from call1.contracts.contents import SIGNAL_BLOCK_WINDOWS, SignalSegmentRef, SpeakerRole
from call1.redaction import REDACTED, mask_text

SEGMENTER_VERSION = "seg-v1"
WINDOW_SECONDS = 7.0
SNAP_SECONDS = 1.5
MIN_WORDS = 3

Timing = Literal["words", "interpolated"]

_WORD = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")


@dataclass(frozen=True)
class Segment:
    index: int
    """Call-wide order."""
    turn_id: int
    window: int
    """k-th window of the turn, 0-based."""
    block: int
    """``window // 4`` (section 3.5): a span never crosses a block."""
    speaker: SpeakerRole
    """AGENT, CALLER or UNKNOWN: one speaker per segment, always."""
    text: str
    """Masked: ``masked_turn_text[char_start:char_end]``."""
    char_start: int
    char_end: int
    start: float
    end: float
    timing: Timing

    def ref(self) -> SignalSegmentRef:
        """The segment as the ``signal_categories`` artifact records it (no text)."""
        return SignalSegmentRef(index=self.index, turn_id=self.turn_id, window=self.window, block=self.block, speaker=self.speaker,
                                char_start=self.char_start, char_end=self.char_end, start=self.start, end=self.end, timing=self.timing)


@dataclass(frozen=True)
class MaskedTurn:
    """One turn as every v2 stage sees it: the masked text the segments and quotes are offsets into."""

    turn_id: int
    speaker: SpeakerRole
    start: float
    end: float
    text: str
    timing: Timing


@dataclass
class SegmentationResult:
    segments: List[Segment]
    turns: Dict[int, MaskedTurn]
    """Every turn's masked text, SYSTEM turns included (they are skipped, not hidden)."""
    skipped_system: int = 0
    """SYSTEM turns skipped."""
    interpolated_turns: int = 0
    """Segmented turns whose cuts were placed by character proportion (no usable word timestamps)."""
    windows_planned: int = 0
    """Windows before the fewer-than-3-words merge: ``sum(max(1, ceil(d / 7)))`` over segmented turns."""
    window_seconds: float = WINDOW_SECONDS
    version: str = SEGMENTER_VERSION

    def by_index(self) -> Dict[int, Segment]:
        return {s.index: s for s in self.segments}

    def masked_text(self, turn_id: int) -> Optional[str]:
        turn = self.turns.get(turn_id)
        return turn.text if turn is not None else None


def window_count(duration: float, window: float = WINDOW_SECONDS) -> int:
    """``n = max(1, ceil(d / window))``, measured to the millisecond so a 7.0 s turn is one window."""
    d = max(0.0, round(float(duration), 3))
    return max(1, math.ceil(d / window - 1e-9))


# --- masking alignment -----------------------------------------------------------------------


def align_replacements(raw: str, masked: str, token: str = REDACTED) -> Optional[List[Tuple[int, int, int, int]]]:
    """The replacements that turn ``raw`` into ``masked``: ``(raw_start, raw_end, masked_start,
    masked_end)`` per placeholder, in order. ``[]`` when nothing was masked; ``None`` when ``masked``
    is not ``raw`` with spans replaced by ``token`` (the caller then segments the masked text itself)."""
    if raw == masked:
        return []
    parts = masked.split(token)
    if len(parts) == 1:
        return None
    pattern = "".join(re.escape(p) + ("(.+?)" if i < len(parts) - 1 else "") for i, p in enumerate(parts))
    match = re.fullmatch(pattern, raw, re.DOTALL)
    if match is None:
        return None
    out: List[Tuple[int, int, int, int]] = []
    cursor = 0
    for i in range(len(parts) - 1):
        cursor += len(parts[i])
        rs, re_ = match.span(i + 1)
        out.append((rs, re_, cursor, cursor + len(token)))
        cursor += len(token)
    return out


def _mapper(replacements: Sequence[Tuple[int, int, int, int]]) -> Tuple[Callable[[int], int], Callable[[int], int]]:
    """Raw offset -> masked offset, for a segment start and a segment end. An offset strictly inside
    a replaced span maps to the placeholder's start (for a start) or end (for an end)."""

    def shift(c: int, inside_start: bool) -> int:
        delta = 0
        for rs, re_, ms, me in replacements:
            if c <= rs:
                break
            if c < re_:
                return ms if inside_start else me
            delta = me - re_
        return c + delta

    return (lambda c: shift(c, True)), (lambda c: shift(c, False))


# --- timing ----------------------------------------------------------------------------------


def _field(turn, name: str, default=None):
    return turn.get(name, default) if isinstance(turn, dict) else getattr(turn, name, default)


def _word_times(text: str, words: Sequence[Tuple[int, int]], stamps) -> Optional[List[Tuple[float, float]]]:
    """Per whitespace word of ``text``: (start, end) seconds from the ASR word timestamps, located in
    order. None when the timestamps do not cover the words well enough to cut on."""
    if not stamps:
        return None
    located: List[Tuple[int, int, float, float]] = []
    cursor = 0
    for stamp in stamps:
        word = str(_field(stamp, "word") or "").strip()
        if not word:
            continue
        at = text.find(word, cursor)
        if at < 0:
            continue
        located.append((at, at + len(word), float(_field(stamp, "start_time")), float(_field(stamp, "end_time"))))
        cursor = at + len(word)
    if not located:
        return None
    times: List[Optional[Tuple[float, float]]] = []
    for ws, we in words:
        hits = [(t0, t1) for s, e, t0, t1 in located if s < we and e > ws]
        times.append((min(t for t, _ in hits), max(t for _, t in hits)) if hits else None)
    known = sum(1 for t in times if t is not None)
    if known * 2 < len(words):
        return None
    # Words the timestamps missed take their neighbours' boundary.
    out: List[Tuple[float, float]] = []
    for i, value in enumerate(times):
        if value is not None:
            out.append(value)
            continue
        before = next((times[j][1] for j in range(i - 1, -1, -1) if times[j] is not None), None)  # type: ignore[index]
        after = next((times[j][0] for j in range(i + 1, len(times)) if times[j] is not None), None)  # type: ignore[index]
        t0 = before if before is not None else after
        t1 = after if after is not None else before
        out.append((float(t0), float(max(t0, t1))))  # type: ignore[arg-type]
    return out


@dataclass
class _Gap:
    idx: int
    """The gap follows word ``idx``."""
    time: float
    length: float
    sentence_end: bool
    valid: bool


def _cuts(gaps: List[_Gap], start: float, duration: float, n: int, snap: float) -> List[int]:
    chosen: List[int] = []
    last = -1
    for k in range(1, n):
        target = start + k * duration / n
        usable = [g for g in gaps if g.valid and g.idx > last]
        if not usable:
            break
        near = [g for g in usable if abs(g.time - target) <= snap]
        if near:
            pick = min(near, key=lambda g: (not g.sentence_end, -round(g.length, 3), abs(g.time - target), g.idx))
        else:
            pick = min(usable, key=lambda g: (abs(g.time - target), g.idx))
        chosen.append(pick.idx)
        last = pick.idx
    return chosen


def _merge_short(pieces: List[List[int]], min_words: int) -> List[List[int]]:
    pieces = [p for p in pieces if p]
    while len(pieces) > 1:
        short = next((i for i, p in enumerate(pieces) if len(p) < min_words), None)
        if short is None:
            break
        if short == 0:
            target = 1
        elif short == len(pieces) - 1:
            target = short - 1
        else:
            target = short - 1 if len(pieces[short - 1]) <= len(pieces[short + 1]) else short + 1
        lo, hi = sorted((short, target))
        pieces[lo:hi + 1] = [pieces[lo] + pieces[hi]]
    return pieces


# --- the segmenter ---------------------------------------------------------------------------


def _speaker(turn, attribution) -> SpeakerRole:
    value = _field(turn, "speaker")
    role = SpeakerRole(getattr(value, "value", value) or "UNKNOWN")
    if attribution is not None:
        for assignment in _field(attribution, "assignments", []) or []:
            if _field(assignment, "turn_id") == _field(turn, "turn_id"):
                assigned = _field(assignment, "speaker")
                return SpeakerRole(getattr(assigned, "value", assigned))
    return role


def segment_call(turns: Sequence, attribution=None, values=None, *, window: float = WINDOW_SECONDS, snap: float = SNAP_SECONDS,
                 min_words: int = MIN_WORDS, mask: Optional[Callable[[str], str]] = None) -> SegmentationResult:
    """Cut a call's turns into masked, speaker-tagged ~7 s segments.

    ``turns`` are transcript turns (contract ``TranscriptTurnContent`` or dicts with ``turn_id``,
    ``speaker``, ``start_time``, ``end_time``, ``text`` and optional ``word_timestamps``), in
    transcript order. ``attribution`` (a ``SpeakerAttributionContent``) relabels mono turns.
    ``values`` is the job's sensitive-value set (``handlers/real/masking.sensitive_values``, which
    also carries the positional spans); ``mask`` overrides how a turn text is masked."""
    positions = getattr(values, "positions", None)
    masker = mask or (lambda text: mask_text(text, set(values or ()), positions))
    segments: List[Segment] = []
    masked_turns: Dict[int, MaskedTurn] = {}
    skipped_system = 0
    interpolated = 0
    planned = 0
    for turn in turns:
        turn_id = int(_field(turn, "turn_id"))
        speaker = _speaker(turn, attribution)
        start = float(_field(turn, "start_time") or 0.0)
        end = max(start, float(_field(turn, "end_time") or start))
        raw = str(_field(turn, "text") or "")
        masked = masker(raw) if raw else raw
        replacements = align_replacements(raw, masked) if raw else []
        base = raw if replacements is not None else masked
        words = [(m.start(), m.end()) for m in _WORD.finditer(base)]
        stamps = _field(turn, "word_timestamps") if replacements is not None else None
        word_times = _word_times(base, words, stamps) if words else None
        timing: Timing = "words" if word_times is not None else "interpolated"
        masked_turns[turn_id] = MaskedTurn(turn_id=turn_id, speaker=speaker, start=start, end=end, text=masked, timing=timing)
        if speaker is SpeakerRole.SYSTEM:
            skipped_system += 1
            continue
        if not words:
            continue
        duration = end - start
        n = window_count(duration, window)
        planned += n
        if timing == "interpolated":
            interpolated += 1
            span = max(1, len(base))
            word_times = [(start + duration * ws / span, start + duration * we / span) for ws, we in words]
        blocked = [(rs, re_) for rs, re_, _, _ in (replacements or [])]
        gaps = []
        for i in range(len(words) - 1):
            prev_end, next_start = words[i][1], words[i + 1][0]
            t_prev, t_next = word_times[i][1], word_times[i + 1][0]  # type: ignore[index]
            gaps.append(_Gap(
                idx=i, time=(t_prev + t_next) / 2, length=max(0.0, t_next - t_prev) if timing == "words" else 0.0,
                sentence_end=bool(_SENTENCE_END.search(base[words[i][0]:prev_end])),
                valid=not any(bs < next_start and be > prev_end for bs, be in blocked)))
        cut_after = _cuts(gaps, start, duration, n, snap) if n > 1 else []
        pieces: List[List[int]] = []
        current: List[int] = []
        for i in range(len(words)):
            current.append(i)
            if i in cut_after:
                pieces.append(current)
                current = []
        pieces.append(current)
        pieces = _merge_short(pieces, min_words)
        to_start, to_end = _mapper(replacements or [])
        for k, piece in enumerate(pieces):
            first, last = piece[0], piece[-1]
            rs, re_ = words[first][0], words[last][1]
            cs, ce = (to_start(rs), to_end(re_)) if replacements is not None else (rs, re_)
            t0 = start if k == 0 and timing == "interpolated" else word_times[first][0]  # type: ignore[index]
            t1 = end if k == len(pieces) - 1 and timing == "interpolated" else word_times[last][1]  # type: ignore[index]
            t0 = min(max(start, t0), end)
            t1 = min(max(t0, t1), end)
            segments.append(Segment(index=len(segments), turn_id=turn_id, window=k, block=k // SIGNAL_BLOCK_WINDOWS, speaker=speaker,
                                    text=masked[cs:ce], char_start=cs, char_end=ce, start=round(t0, 3), end=round(t1, 3), timing=timing))
    return SegmentationResult(segments=segments, turns=masked_turns, skipped_system=skipped_system, interpolated_turns=interpolated,
                              windows_planned=planned, window_seconds=window)


def time_at(segments: Sequence[Segment], turn_id: int, char: int) -> Optional[float]:
    """The time of a masked-turn offset, by character position inside the segment that holds it
    (a segment's boundaries are exact word times, or interpolated)."""
    for seg in segments:
        if seg.turn_id == turn_id and seg.char_start <= char <= seg.char_end:
            width = max(1, seg.char_end - seg.char_start)
            return round(seg.start + (seg.end - seg.start) * (char - seg.char_start) / width, 3)
    return None


__all__ = ["MIN_WORDS", "SEGMENTER_VERSION", "SNAP_SECONDS", "WINDOW_SECONDS", "MaskedTurn", "Segment", "SegmentationResult",
           "align_replacements", "segment_call", "time_at", "window_count"]
