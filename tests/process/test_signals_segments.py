"""Contact Signals v2 stage 0, the segmenter (docs/ContactSignalsV2.md section 2; F3 acceptance)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from call1.contracts.contents import SpeakerAssignment, SpeakerAttributionContent, SpeakerRole, TranscriptTurnContent, WordTimestampView
from call1.pipeline.signal_segments import (
    MIN_WORDS,
    SEGMENTER_VERSION,
    WINDOW_SECONDS,
    align_replacements,
    segment_call,
    time_at,
    window_count,
)
from call1.redaction import REDACTED

REPO = Path(__file__).resolve().parents[2]
TELEPHONY = REPO / "sample_audio" / "telephony_training_set" / "manifest.json"
RETAIL = REPO / "sample_audio" / "apptek_retail" / "manifest.json"


def turn(i: int, speaker: str, start: float, end: float, text: str, *, words: bool = False) -> TranscriptTurnContent:
    stamps = None
    if words:
        tokens = text.split()
        step = (end - start) / len(tokens)
        stamps = [WordTimestampView(word=(" " if k else "") + w, start_time=round(start + k * step, 3), end_time=round(start + (k + 0.8) * step, 3),
                                    probability=0.9) for k, w in enumerate(tokens)]
    return TranscriptTurnContent(turn_id=i, speaker=SpeakerRole(speaker), start_time=start, end_time=end, text=text, word_timestamps=stamps)


LONG = ("I have been a customer for years and the price keeps going up every single month. Last month it went up again by ten "
        "dollars and nobody told me about it. I called twice already and each time I was told it would be fixed. It was not "
        "fixed, so now I would like to cancel the whole account today please.")


def _check_grid(result, turns):
    by_id = {t.turn_id: t for t in turns}
    for seg in result.segments:
        masked = result.turns[seg.turn_id].text
        assert masked[seg.char_start:seg.char_end] == seg.text  # an exact substring of the masked turn
        assert seg.text == seg.text.strip() and seg.text
        assert seg.block == seg.window // 4
        source = by_id[seg.turn_id]
        assert source.start_time <= seg.start <= seg.end <= source.end_time  # never crosses its turn
        assert seg.speaker is not SpeakerRole.SYSTEM
    indexes = [s.index for s in result.segments]
    assert indexes == list(range(len(indexes)))


def test_no_segment_crosses_a_turn_and_every_text_is_an_exact_substring():
    turns = [turn(0, "AGENT", 0.0, 3.0, "Thank you for calling, how can I help?"),
             turn(1, "CALLER", 3.2, 30.0, LONG),
             turn(2, "AGENT", 30.5, 33.0, "I can help with that today.")]
    result = segment_call(turns)
    _check_grid(result, turns)
    long = [s for s in result.segments if s.turn_id == 1]
    assert len(long) == window_count(26.8) == 4 and [s.window for s in long] == [0, 1, 2, 3]
    assert " ".join(s.text for s in long) == LONG  # the partition covers the whole turn
    assert result.version == SEGMENTER_VERSION == "seg-v1" and result.window_seconds == WINDOW_SECONDS
    # A turn over four windows spills into block 1.
    monologue = segment_call([turn(0, "CALLER", 0.0, 40.0, LONG + " " + LONG)])
    assert max(s.window for s in monologue.segments) >= 4 and {s.block for s in monologue.segments} == {0, 1}


def test_cuts_never_fall_inside_a_masked_value_and_offsets_map_into_the_masked_turn():
    # The value straddles the natural cut near 7 s; a cut there would split "Maria Lopez".
    text = "Hello there my friend it is me calling again and my name is Maria Lopez and I would like to talk about my bill today please."
    words = text.split()
    at = words.index("Maria")
    t = turn(0, "CALLER", 0.0, 14.0, text)
    result = segment_call([t], values={"Maria Lopez"})
    masked = result.turns[0].text
    assert "Maria" not in masked and REDACTED in masked
    _check_grid(result, [t])
    assert any(REDACTED in s.text for s in result.segments)
    assert all("Maria" not in s.text and "Lopez" not in s.text for s in result.segments)
    # Every cut sits outside the placeholder: the placeholder appears whole in exactly one segment.
    assert sum(s.text.count(REDACTED) for s in result.segments) == masked.count(REDACTED) == 1
    assert at > 0
    # Positional spans (the PII model's findings, decision 22) block cuts too.
    class Values(set):
        pass

    from call1.redaction import TurnPositions

    values = Values()
    start = text.index("Maria")
    values.positions = TurnPositions([text], [[(start, start + len("Maria Lopez"))]])
    positional = segment_call([t], values=values)
    assert positional.turns[0].text == masked
    assert [s.text for s in positional.segments] == [s.text for s in result.segments]


def test_alignment_maps_raw_replacements_to_placeholders():
    raw = "my card is 4111 1111 and my name is Sam Smith ok"
    masked = f"my card is {REDACTED} and my name is {REDACTED} ok"
    reps = align_replacements(raw, masked)
    assert [raw[a:b] for a, b, _, _ in reps] == ["4111 1111", "Sam Smith"]
    assert all(masked[c:d] == REDACTED for _, _, c, d in reps)
    assert align_replacements(raw, raw) == []
    assert align_replacements(raw, "something else entirely") is None


def test_interpolated_timing_without_word_timestamps_and_word_timing_with_them():
    plain = segment_call([turn(0, "CALLER", 10.0, 30.0, LONG)])
    assert {s.timing for s in plain.segments} == {"interpolated"} and plain.interpolated_turns == 1
    assert plain.segments[0].start == 10.0 and plain.segments[-1].end == 30.0
    timed = segment_call([turn(0, "CALLER", 10.0, 30.0, LONG, words=True)])
    assert {s.timing for s in timed.segments} == {"words"} and timed.interpolated_turns == 0
    # Cuts snap to a gap within 1.5 s of the target and prefer a sentence end.
    targets = [10.0 + k * 20.0 / 3 for k in (1, 2)]
    boundaries = [s.end for s in timed.segments[:-1]]
    assert len(boundaries) == 2 and all(abs(b - t) <= 1.5 + 0.8 for b, t in zip(boundaries, targets))
    assert timed.segments[0].text.endswith(".") or timed.segments[1].text.endswith(".")
    # Times inside a segment are interpolated by character position.
    seg = timed.segments[0]
    assert time_at(timed.segments, 0, seg.char_start) == seg.start and time_at(timed.segments, 0, seg.char_end) == seg.end


def test_windows_with_fewer_than_three_words_merge_into_a_neighbour():
    # 15 s of four words would give three windows of one or two words: they merge.
    result = segment_call([turn(0, "AGENT", 0.0, 15.0, "Okay. Right. Sure thing.")])
    assert len(result.segments) == 1 and result.segments[0].text == "Okay. Right. Sure thing."
    assert result.windows_planned == 3
    assert all(len(s.text.split()) >= MIN_WORDS for s in segment_call([turn(0, "CALLER", 0.0, 29.0, LONG)]).segments)


def test_unknown_and_system_turns_are_counted_and_attribution_relabels_mono_turns():
    turns = [turn(0, "UNKNOWN", 0.0, 3.0, "Thank you for calling, how can I help?"),
             turn(1, "SYSTEM", 3.0, 5.0, "This call is recorded."),
             turn(2, "UNKNOWN", 5.0, 8.0, "I have a question about a fee.")]
    result = segment_call(turns)
    assert result.skipped_system == 1 and 1 in result.turns and all(s.turn_id != 1 for s in result.segments)
    assert [s.speaker for s in result.segments] == [SpeakerRole.UNKNOWN, SpeakerRole.UNKNOWN]
    attribution = SpeakerAttributionContent(method="diarization", assignments=[
        SpeakerAssignment(turn_id=0, speaker=SpeakerRole.AGENT), SpeakerAssignment(turn_id=2, speaker=SpeakerRole.CALLER)])
    relabelled = segment_call(turns, attribution)
    assert [s.speaker for s in relabelled.segments] == [SpeakerRole.AGENT, SpeakerRole.CALLER]


def test_window_count_is_ceil_of_duration_over_seven_to_the_millisecond():
    assert window_count(0) == 1 and window_count(7.0) == 1 and window_count(7.0000000001) == 1
    assert window_count(7.001) == 2 and window_count(28.0) == 4 and window_count(29.95) == 5


def _manifest(path: Path):
    if not path.is_file():
        pytest.skip(f"{path.name} is not present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_apptek_reference_counts_reproduce():
    """Section 2's measured basis: windows are ceil(d / 7) per turn, before the fewer-than-3-words merge."""
    counts = {}
    for name, path in (("telephony", TELEPHONY), ("retail", RETAIL)):
        calls = _manifest(path)
        total = agent = caller = 0
        per_call = []
        two_block = []
        for call_id, call in calls.items():
            result = segment_call(call["turns"], min_words=0)
            assert result.windows_planned == len(result.segments)  # every planned window found words to cut on
            per_call.append(len(result.segments))
            total += len(result.segments)
            agent += sum(1 for s in result.segments if s.speaker is SpeakerRole.AGENT)
            caller += sum(1 for s in result.segments if s.speaker is SpeakerRole.CALLER)
            two_block += [(call_id, s.turn_id) for s in result.segments if s.block == 1]
            # The shipped segmenter (with the merge) keeps every text an exact substring of its turn.
            merged = segment_call(call["turns"])
            assert all(merged.turns[s.turn_id].text[s.char_start:s.char_end] == s.text for s in merged.segments)
            assert merged.windows_planned == result.windows_planned
            assert sum(window_count(t["end_time"] - t["start_time"]) for t in call["turns"] if t["text"].strip()) == result.windows_planned
        counts[name] = (len(calls), total, agent, caller, min(per_call), max(per_call), sorted(two_block))
    assert counts["telephony"][:6] == (10, 988, 655, 333, 70, 145)
    assert counts["telephony"][6] == []
    assert counts["retail"][:6] == (30, 3802, 2229, 1573, 62, 183)
    # The two retail turns over 28 s (five windows each) are the only 2-block turns.
    assert len(counts["retail"][6]) == 2 and len({call for call, _ in counts["retail"][6]}) >= 1
    longest = _manifest(RETAIL)["en_US_Aave_Retail_1591484"]
    assert len(segment_call(longest["turns"], min_words=0).segments) == 183 and math.isclose(max(counts["retail"][4:6]), 183)
