"""The preview diff and the feedback IDs against multi-segment signals (decision 25,
docs/ContactSignalsV2.md sections 6.5 and 7.5): hits pair by any segment they cover, a span that
became a part is not "removed", a change only to a hit's segments is reported, 'Other' and no
subcategory read the same, and feedback saved on a part's earlier hit ID stays readable."""

from __future__ import annotations

from types import SimpleNamespace as NS

from call1.store.results.signals import feedback_hit_ids, preview_diff

DIGEST = "intent.abcdef012345.1234abcd"


def _hit(turn, block=0, *, category="intent", sub=None, parts=(), fields=(), prefix=None):
    hit_id = f"{prefix or category + '.abcdef012345.1234abcd'}.t{turn}b{block}"
    return NS(id=hit_id, category_id=category, kind=NS(value=category), turn_id=turn, span=NS(block=block), subcategory_id=sub,
              parts=[NS(turn_id=t, block=b) for t, b in parts], fields=list(fields))


def _content(*hits, pipeline="v2"):
    return NS(pipeline=pipeline, signals=list(hits))


def test_a_span_that_became_a_part_pairs_with_the_merged_hit_and_reports_the_segment_change():
    published = _content(_hit(1), _hit(3))
    preview = _content(_hit(1, parts=[(3, 0)]))
    diff = preview_diff(published, preview)
    assert diff.removed == [] and diff.added == [] and diff.relabelled == [] and diff.fields_changed == []
    assert diff.segments_changed == [preview.signals[0].id]
    assert diff.builtin_changed == []


def test_a_merged_hit_that_splits_again_reports_the_new_hit_once_and_nothing_removed():
    published = _content(_hit(1, parts=[(3, 0)]))
    preview = _content(_hit(1), _hit(3))
    diff = preview_diff(published, preview)
    assert diff.removed == [] and diff.added == []
    assert sorted(diff.segments_changed) == sorted(h.id for h in preview.signals)


def test_a_change_only_to_the_parts_is_reported_and_unchanged_hits_are_not():
    published = _content(_hit(1, parts=[(3, 0)]), _hit(7, category="deferred"))
    preview = _content(_hit(1, parts=[(3, 0), (5, 0)]), _hit(7, category="deferred"))
    diff = preview_diff(published, preview)
    assert diff.segments_changed == [preview.signals[0].id]
    assert not (diff.added or diff.removed or diff.relabelled or diff.fields_changed)


def test_other_and_no_subcategory_read_the_same_but_a_real_subcategory_is_a_relabel():
    published = _content(_hit(1), _hit(5, category="issue"))
    preview = _content(_hit(1, sub="cancel_account"), _hit(5, category="issue", sub="other"))
    diff = preview_diff(published, preview)
    assert diff.relabelled == [preview.signals[0].id]
    assert not (diff.added or diff.removed or diff.fields_changed or diff.segments_changed)


def test_label_and_field_changes_win_over_segment_changes_and_new_and_gone_hits_are_builtin_changes():
    published = _content(_hit(1, sub="a"), _hit(9, category="deferred"))
    preview = _content(_hit(1, sub="b", parts=[(3, 0)]), _hit(11, category="friction"))
    diff = preview_diff(published, preview)
    assert diff.relabelled == [preview.signals[0].id] and diff.segments_changed == []
    assert diff.added == [preview.signals[1].id] and diff.removed == [published.signals[1].id]
    assert sorted(diff.builtin_changed) == sorted([preview.signals[1].id, published.signals[1].id])


def test_against_a_v1_result_hits_pair_by_category_and_turn():
    published = _content(NS(id="intent-1", category_id=None, kind=NS(value="intent"), turn_id=1, span=None, subcategory_id=None, parts=[], fields=[]),
                         pipeline="v1")
    preview = _content(_hit(1, block=2))
    diff = preview_diff(published, preview)
    assert not (diff.added or diff.removed or diff.relabelled or diff.fields_changed or diff.segments_changed)


def test_feedback_ids_include_each_parts_earlier_hit_id():
    merged = _hit(1, parts=[(3, 0), (3, 1)])
    plain = _hit(9, category="deferred")
    assert feedback_hit_ids([merged, plain]) == [
        merged.id, f"{DIGEST}.t3b0", f"{DIGEST}.t3b1", plain.id,
    ]
    preview_ids = NS(id="intent.preview.t1b0", parts=[NS(turn_id=2, block=0)])
    assert feedback_hit_ids([preview_ids]) == ["intent.preview.t1b0", "intent.preview.t2b0"]
    assert feedback_hit_ids([NS(id="legacy-v1-id", parts=None)]) == ["legacy-v1-id"]
