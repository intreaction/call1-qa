"""Multi-segment signals (decision 25; docs/ContactSignalsV2.md section 6.5): the v2 merge folds one
speaker's consecutive hits with the same category and stage-2 outcome into one hit. The first span
(the anchor) keeps its hit ID, quote, offsets and span; each later span becomes a verified part.

The rule itself is ``signals_v2.merge_multi_segment`` (pure; tested on hand-built hits here), and
``signal_stages.merge_v2`` applies it after stage-2 rejection and quote re-verification (tested
through the fake cascade on the ``returns`` script)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from call1.contracts.contents import (
    ContactSignalKind,
    ContactSignalView,
    ExtractedFieldView,
    QuoteRange,
    SignalSpanView,
    SpeakerRole,
)
from call1.pipeline.signals_v2 import MERGE_GAP_SECONDS, hit_id, merge_multi_segment
from call1.process.handlers.fake import RETURN_SCRIPT
from call1.redaction import REDACTED

from .test_signals_merge import _merge, _transcript_checksum
from .test_signals_support import run_v2, with_categories

A, C = SpeakerRole.AGENT, SpeakerRole.CALLER


def _hit(turn_id: int, start: float, end: float, *, speaker: SpeakerRole = A, category: str = "fix_proposed", sub: Optional[str] = None,
         block: int = 0, confidence: float = 0.8, quote: Optional[str] = None, fields: Sequence[Dict] = ()) -> ContactSignalView:
    quote = quote or f"quote of turn {turn_id} block {block}"
    return ContactSignalView(
        id=f"{category}.abcdef012345.12345678.t{turn_id}b{block}", kind=ContactSignalKind(category), label=category, start=start, end=end,
        speaker=speaker, quote=quote, turn_id=turn_id, char_start=0, char_end=len(quote), confidence=confidence, category_id=category,
        category_digest="abcdef012345", category_confidence=confidence, subcategory_id=sub,
        span=SignalSpanView(block=block, first_window=block * 4, last_window=block * 4, timing="interpolated", context_start=start, context_end=end),
        fields=[ExtractedFieldView.model_validate(f) for f in fields])


def _turns(*speakers: SpeakerRole) -> List[Tuple[int, SpeakerRole]]:
    return list(enumerate(speakers))


ALTERNATING = _turns(A, C, A, C, A, C, A, C, A)


# --- the rule ------------------------------------------------------------------------------------


def test_consecutive_same_speaker_hits_join_across_the_other_speakers_turns():
    anchor = _hit(2, 8.0, 11.8, confidence=0.7)
    later = _hit(4, 16.0, 19.8, confidence=0.9)
    last = _hit(6, 24.0, 27.8, confidence=0.6)
    other = _hit(3, 12.0, 13.0, speaker=C, category="caller_confirms_resolved")
    merged = merge_multi_segment([last, other, later, anchor], ALTERNATING)
    assert [h.id for h in merged] == [anchor.id, other.id]
    hit = merged[0]
    assert (hit.quote, hit.turn_id, hit.start, hit.end, hit.span, hit.char_start, hit.char_end) == (
        anchor.quote, 2, 8.0, 11.8, anchor.span, anchor.char_start, anchor.char_end)
    assert [(p.turn_id, p.block, p.start, p.end, p.quote) for p in hit.parts] == [(4, 0, 16.0, 19.8, later.quote), (6, 0, 24.0, 27.8, last.quote)]
    assert hit.span_end == 27.8 and hit.confidence == 0.9  # the max over the run
    assert merged[1].parts == [] and merged[1].span_end is None


def test_blocks_of_one_turn_join():
    first, second = _hit(4, 16.0, 23.0, block=0), _hit(4, 30.0, 37.0, block=1)
    [hit] = merge_multi_segment([first, second], ALTERNATING)
    assert hit.id == first.id and [(p.turn_id, p.block) for p in hit.parts] == [(4, 1)]


def test_a_same_speaker_turn_without_the_category_breaks_the_run():
    # Agent turn 4 lies between the two agent hits and carries no fix_proposed hit.
    hits = [_hit(2, 8.0, 11.8), _hit(6, 24.0, 27.8)]
    assert [len(h.parts) for h in merge_multi_segment(hits, ALTERNATING)] == [0, 0]
    # A caller's turns in between never break it; a SYSTEM turn is another speaker too.
    hits = [_hit(0, 0.0, 3.0), _hit(3, 12.0, 15.0)]
    [hit] = merge_multi_segment(hits, _turns(A, C, SpeakerRole.SYSTEM, A))
    assert [p.turn_id for p in hit.parts] == [3]


def test_a_different_subcategory_or_speaker_breaks_and_both_other_join():
    a, b = _hit(2, 8.0, 11.8, category="intent", sub="cancel_account"), _hit(4, 16.0, 19.8, category="intent", sub="fee_question")
    assert len(merge_multi_segment([a, b], ALTERNATING)) == 2
    a, b = _hit(2, 8.0, 11.8, category="intent", sub="other"), _hit(4, 16.0, 19.8, category="intent", sub="other")
    [hit] = merge_multi_segment([a, b], ALTERNATING)
    assert hit.subcategory_id == "other" and len(hit.parts) == 1
    # A classified span and an "other" one are different outcomes.
    a, b = _hit(2, 8.0, 11.8, category="intent", sub="cancel_account"), _hit(4, 16.0, 19.8, category="intent", sub="other")
    assert len(merge_multi_segment([a, b], ALTERNATING)) == 2
    # Different categories never join.
    assert len(merge_multi_segment([_hit(2, 8.0, 11.8), _hit(4, 16.0, 19.8, category="deferred")], ALTERNATING)) == 2
    # The same category from the other speaker is a different signal.
    assert len(merge_multi_segment([_hit(2, 8.0, 11.8), _hit(3, 12.0, 15.8, speaker=C)], ALTERNATING)) == 2


def test_the_gap_limit_is_inclusive_and_measured_from_the_runs_end():
    assert MERGE_GAP_SECONDS == 20.0
    at_limit = [_hit(2, 8.0, 10.0), _hit(4, 30.0, 32.0)]
    assert len(merge_multi_segment(at_limit, ALTERNATING)) == 1
    past = [_hit(2, 8.0, 10.0), _hit(4, 30.5, 32.0)]
    assert len(merge_multi_segment(past, ALTERNATING)) == 2
    # The run's end moves with each part: 10 → 29, and the third hit starts 19 s after that.
    chain = [_hit(2, 8.0, 10.0), _hit(4, 25.0, 29.0), _hit(6, 48.0, 50.0)]
    [hit] = merge_multi_segment(chain, ALTERNATING)
    assert [p.turn_id for p in hit.parts] == [4, 6] and hit.span_end == 50.0


def test_fields_union_by_id_and_the_first_extracted_value_wins():
    absent = {"field_id": "reason", "type": "enum", "status": "absent", "name": "Reason"}
    price = {"field_id": "reason", "type": "enum", "status": "extracted", "value": "price", "name": "Reason"}
    service = {"field_id": "reason", "type": "enum", "status": "extracted", "value": "service", "name": "Reason"}
    note_absent = {"field_id": "note", "type": "string", "status": "absent", "name": "Note"}
    hits = [_hit(2, 8.0, 11.8, fields=[absent, note_absent]), _hit(4, 16.0, 19.8, fields=[price]), _hit(6, 24.0, 27.8, fields=[service])]
    [hit] = merge_multi_segment(hits, ALTERNATING)
    assert [(f.field_id, f.status, f.value) for f in hit.fields] == [("reason", "extracted", "price"), ("note", "absent", None)]
    assert all(f.turn_id is None for f in hit.fields)  # no offsets, so no turn to name


def test_a_field_read_from_a_part_names_the_turn_its_offsets_index():
    """A merged hit's field offsets taken from a part index that part's turn, so the field says
    which (``ExtractedFieldView.turn_id``); the anchor's own fields keep the hit's turn implicit."""
    own = {"field_id": "reason", "type": "enum", "status": "extracted", "value": "price", "name": "Reason",
           "evidence": "price", "char_start": 3, "char_end": 8}
    from_part = {"field_id": "plan", "type": "string", "status": "extracted", "value": "premium", "name": "Plan",
                 "surface": "premium", "char_start": 10, "char_end": 17}
    [hit] = merge_multi_segment([_hit(2, 8.0, 11.8, fields=[own]), _hit(4, 16.0, 19.8, fields=[from_part])], ALTERNATING)
    assert [(f.field_id, f.turn_id, f.char_start) for f in hit.fields] == [("reason", None, 3), ("plan", 4, 10)]
    # Two blocks of one turn share it: nothing to name.
    [same_turn] = merge_multi_segment([_hit(2, 8.0, 9.0), _hit(2, 9.5, 11.0, block=1, fields=[from_part])], ALTERNATING)
    assert same_turn.fields[0].turn_id is None


def test_single_hits_pass_through_unchanged():
    hits = [_hit(2, 8.0, 11.8), _hit(3, 12.0, 15.0, speaker=C, category="intent")]
    assert merge_multi_segment(hits, ALTERNATING) == sorted(hits, key=lambda h: (h.start, h.id))
    assert merge_multi_segment([], ALTERNATING) == []


# --- through the cascade ---------------------------------------------------------------------------


def test_the_returns_script_yields_one_multi_segment_signal_with_a_verified_part(tmp_path):
    run = run_v2(tmp_path, RETURN_SCRIPT)
    fixes = [h for h in run.result.signals if h.category_id == "fix_proposed"]
    assert len(fixes) == 1
    [hit] = fixes
    # Two stage-1 spans, one signal: the anchor is turn 2's span and keeps that span's hit ID.
    assert sorted(s.span_key for s in run.categories.spans if s.category_id == "fix_proposed") == ["fix_proposed.t2b0", "fix_proposed.t4b0"]
    assert hit.id.endswith(".t2b0") and hit.turn_id == 2 and hit.span.block == 0
    assert [(p.turn_id, p.block) for p in hit.parts] == [(4, 0)]
    from call1.process.handlers.signal_stages import SignalContext

    ctx = SignalContext(run.jobs["merge"])
    assert ctx.masked_turns[2][hit.char_start:hit.char_end] == hit.quote
    for part in hit.parts:
        assert ctx.masked_turns[part.turn_id][part.char_start:part.char_end] == part.quote
        assert part.start >= hit.end
    assert hit.span_end == hit.parts[-1].end
    # Other categories are untouched, and the result still validates as a contract.
    assert {h.category_id for h in run.result.signals} == {"intent", "fix_proposed", "caller_confirms_resolved"}
    assert all(h.parts == [] for h in run.result.signals if h.category_id != "fix_proposed")


def test_the_anchor_keeps_its_hit_id_across_threshold_edits(tmp_path):
    base = run_v2(tmp_path, RETURN_SCRIPT)
    lowered = run_v2(tmp_path, RETURN_SCRIPT, with_categories(fix_proposed={"threshold": 0.2}))
    first = next(h for h in base.result.signals if h.category_id == "fix_proposed")
    second = next(h for h in lowered.result.signals if h.category_id == "fix_proposed")
    assert first.id == second.id and first.parts == second.parts
    # The ID is exactly the anchor span's own hit ID (section 6.3), as if it stood alone.
    category = base.jobs["merge"].input("taxonomy").content().taxonomy.category("fix_proposed")
    assert first.id == hit_id(category, _transcript_checksum(RETURN_SCRIPT), 2, 0)


def test_part_quotes_are_masked_and_verified_against_their_own_turn(tmp_path):
    script = list(RETURN_SCRIPT)
    script[4] = (A, "Once the carrier scans it, we can offer a refund to the card ending 4111 1111 1111 1111 or email maria.lopez@example.com.")
    run = run_v2(tmp_path, script)
    [hit] = [h for h in run.result.signals if h.category_id == "fix_proposed"]
    [part] = hit.parts
    assert REDACTED in part.quote and "4111" not in part.quote and "maria.lopez" not in part.quote
    from call1.process.handlers.signal_stages import SignalContext

    ctx = SignalContext(run.jobs["merge"])
    assert ctx.masked_turns[4][part.char_start:part.char_end] == part.quote


def test_a_later_span_that_fails_re_verification_is_not_a_part(tmp_path):
    taxonomy = with_categories(fix_proposed={"narrow_quote": True})
    run = run_v2(tmp_path, RETURN_SCRIPT, taxonomy)
    [hit] = [h for h in run.result.signals if h.category_id == "fix_proposed"]
    assert len(hit.parts) == 1 and hit.quote_narrowed
    # Tamper with the part's narrowed quote (same offsets, different text): that span is dropped, so
    # the anchor stands alone.
    spans = []
    for span in run.extraction.spans:
        if span.span_key == "fix_proposed.t4b0" and span.narrowed_quote is not None:
            q = span.narrowed_quote
            span = span.model_copy(update={"narrowed_quote": QuoteRange(char_start=q.char_start, char_end=q.char_end, text="x" * len(q.text))})
        spans.append(span)
    merged = _merge(tmp_path, run, extraction=run.extraction.model_copy(update={"spans": spans}), taxonomy=taxonomy, script=RETURN_SCRIPT)
    [alone] = [h for h in merged.signals if h.category_id == "fix_proposed"]
    assert alone.id == hit.id and alone.parts == [] and alone.span_end is None


def test_a_stage_two_rejection_removes_the_span_before_the_merge(tmp_path):
    script = list(RETURN_SCRIPT)
    script[4] = (A, "Not really, but we can offer a replacement or a refund to the original card.")
    run = run_v2(tmp_path, script)
    [hit] = [h for h in run.result.signals if h.category_id == "fix_proposed"]
    assert hit.id.endswith(".t2b0") and hit.parts == []


@pytest.mark.parametrize("preview", [False, True])
def test_previews_merge_the_same_way(tmp_path, preview):
    run = run_v2(tmp_path, RETURN_SCRIPT, preview_id="spv_1" if preview else None)
    [hit] = [h for h in run.result.signals if h.category_id == "fix_proposed"]
    assert len(hit.parts) == 1 and (".preview.t2b0" in hit.id) == preview
