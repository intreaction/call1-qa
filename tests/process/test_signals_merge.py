"""The merge's v2 branch (docs/ContactSignalsV2.md sections 6, 7.3 and 8.1; F3 acceptance "Merge")."""

from __future__ import annotations

import hashlib
import re

import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.common import CONTRACT_PARAMETERS, canonical_json
from call1.contracts.contents import ContactSignalKind, ContactSignalsContent, QuoteRange, SpeakerRole
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType
from call1.contracts.signals import SignalField, category_digest
from call1.process.handlers import build_registry
from call1.process.handlers.base import HandlerError
from call1.process.handlers.fake import CALLER_NAME_SCRIPT, CANCEL_SCRIPT, SCRIPT
from call1.redaction import REDACTED

from .test_signals_support import (
    cancel_taxonomy,
    custom_category,
    findings_for,
    make_job,
    run_v2,
    script_transcript,
    signal_params,
    snapshot,
    with_categories,
)

HIT_ID = re.compile(r"^[a-z0-9_-]+\.[0-9a-f]{12}\.[0-9a-f]{8}\.t\d+b\d+$")


def _transcript_checksum(script) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(script_transcript(script))).hexdigest()


def test_hit_ids_follow_section_6_3_and_fill_the_v1_fields(tmp_path):
    run = run_v2(tmp_path, SCRIPT)
    result = run.result
    ContactSignalsContent.model_validate(result.model_dump(mode="json"))
    assert result.pipeline == "v2" and result.passes == [] and result.completeness == "complete"
    assert [s.stage for s in result.stages] == ["categorize", "subcategorize"] and all(s.included for s in result.stages)
    assert result.taxonomy == snapshot().taxonomy_ref and result.stage1_digests == run.categories.stage1_digests
    checksum = _transcript_checksum(SCRIPT)
    for hit in result.signals:
        assert HIT_ID.match(hit.id), hit.id
        category = snapshot().taxonomy.category(hit.category_id)
        assert hit.id == f"{hit.category_id}.{category_digest(category)[7:19]}.{checksum[7:15]}.t{hit.turn_id}b{hit.span.block}"
        assert hit.kind is ContactSignalKind(hit.category_id) and hit.label == category.name and hit.review_status == "unreviewed"
        assert hit.category_digest == category_digest(category)[7:19]
        turn = next(t for t in script_transcript(SCRIPT).turns if t.turn_id == hit.turn_id)
        assert turn.text[hit.char_start:hit.char_end] == hit.quote  # nothing masked in this script
        assert turn.start_time <= hit.start <= hit.end <= turn.end_time and hit.speaker is turn.speaker
        assert hit.confidence == pytest.approx(1 - next(d for d in run.subcategories.decisions if d.span_key.endswith(f".t{hit.turn_id}b0")
                                                        and d.span_key.startswith(hit.category_id)).probabilities["not"])
    # A preview result uses preview IDs.
    preview = run_v2(tmp_path, SCRIPT, preview_id="spv_1")
    assert all(".preview.t" in h.id for h in preview.result.signals) and preview.result.taxonomy.version is None


def test_threshold_edits_keep_hit_ids_and_a_gloss_edit_changes_them(tmp_path):
    base = with_categories([custom_category()])
    from call1.process.handlers.fake import COMPETITOR_SCRIPT

    first = {h.category_id: h.id for h in run_v2(tmp_path, COMPETITOR_SCRIPT, base).result.signals}
    lowered = with_categories([custom_category(threshold=0.2)], intent={"threshold": 0.25})
    second = {h.category_id: h.id for h in run_v2(tmp_path, COMPETITOR_SCRIPT, lowered).result.signals}
    assert first["competitor"] == second["competitor"] and first["intent"] == second["intent"]
    reglossed = with_categories([custom_category(gloss="Someone names a rival provider")])
    third = {h.category_id: h.id for h in run_v2(tmp_path, COMPETITOR_SCRIPT, reglossed).result.signals}
    assert third["competitor"] != first["competitor"] and third["intent"] == first["intent"]
    custom = next(h for h in run_v2(tmp_path, COMPETITOR_SCRIPT, base).result.signals if h.category_id == "competitor")
    assert custom.kind is ContactSignalKind.CUSTOM and custom.label == "Mentions a competitor" and custom.speaker is SpeakerRole.AGENT


def test_a_caller_segment_with_a_masked_name_survives_the_merge(tmp_path):
    run = run_v2(tmp_path, CALLER_NAME_SCRIPT)
    hits = [h for h in run.result.signals if h.turn_id == 1]
    assert {h.category_id for h in hits} == {"intent", "issue"}  # co-occurrence on the masked segment
    for hit in hits:
        assert REDACTED in hit.quote and "Maria" not in hit.quote and "Lopez" not in hit.quote
        masked_turn = run.jobs["merge"].input("transcript").content().turns[1].text  # raw; the hit's offsets are in the masked turn
        assert "Maria Lopez" in masked_turn
    # Offsets index the masked turn the stages saw (section 6.1): re-derive it the way the merge does.
    from call1.process.handlers.signal_stages import SignalContext

    ctx = SignalContext(run.jobs["merge"])
    for hit in hits:
        assert ctx.masked_turns[1][hit.char_start:hit.char_end] == hit.quote


def test_a_quote_not_in_the_masked_turn_at_its_offsets_is_dropped(tmp_path):
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy())
    hit = next(h for h in run.result.signals if h.category_id == "intent")
    assert hit.quote_narrowed and hit.quote == "cancel my account"
    # Tamper with the narrowed quote: same offsets, different text.
    spans = []
    for span in run.extraction.spans:
        if span.narrowed_quote is not None:
            q = span.narrowed_quote
            span = span.model_copy(update={"narrowed_quote": QuoteRange(char_start=q.char_start, char_end=q.char_end, text="x" * len(q.text))})
        spans.append(span)
    tampered = run.extraction.model_copy(update={"spans": spans})
    merged = _merge(tmp_path, run, extraction=tampered, taxonomy=cancel_taxonomy(), script=CANCEL_SCRIPT)
    assert "intent" not in {h.category_id for h in merged.signals}
    assert {h.category_id for h in merged.signals} == {h.category_id for h in run.result.signals} - {"intent"}
    # A field whose surface is not at its offsets is dropped; the hit stays.
    spans = []
    for span in run.extraction.spans:
        fields = [f.model_copy(update={"evidence": "PRICE"}) if f.evidence else f for f in span.fields]
        spans.append(span.model_copy(update={"fields": fields}))
    merged = _merge(tmp_path, run, extraction=run.extraction.model_copy(update={"spans": spans}), taxonomy=cancel_taxonomy(), script=CANCEL_SCRIPT)
    intent = next(h for h in merged.signals if h.category_id == "intent")
    assert intent.fields == [] and intent.quote_narrowed


def _merge(tmp_path, run, *, categories=None, subcategories=None, extraction=None, taxonomy=None, script=SCRIPT, planned=None, upstream=()):
    transcript = script_transcript(script)
    snap = snapshot(taxonomy)
    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript)),
              "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap),
              "stage:categorize": (ArtifactKind.SIGNAL_CATEGORIES, categories if categories is not None else run.categories),
              "stage:subcategorize": (ArtifactKind.SIGNAL_SUBCATEGORIES, subcategories if subcategories is not None else run.subcategories)}
    if extraction is not None or run.extraction is not None:
        inputs["stage:extract"] = (ArtifactKind.SIGNAL_EXTRACTION, extraction if extraction is not None else run.extraction)
    planned = planned or ("categorize,subcategorize,extract" if "stage:extract" in inputs else "categorize,subcategorize")
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_MERGE, inputs, entry_id=None, upstream=upstream,
                   parameters=signal_params(snap, extra={"planned_stages": planned}))
    return build_registry("fake").get(JobType.CONTACT_SIGNALS_MERGE).run(job).outputs["contact_signals"].content


def test_absence_remains_absence(tmp_path):
    quiet = [(SpeakerRole.AGENT, "Good morning, you have reached the store."), (SpeakerRole.CALLER, "Good morning to you as well.")]
    run = run_v2(tmp_path, quiet)
    assert run.result.signals == [] and run.result.completeness == "complete"
    assert run.result.segmentation.scored_segments == 2  # "no signals found" only when stage 1 ran over every scorable segment
    # A field the engine left out is absent, never empty or zero.
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy())
    extraction = next(s for s in run.extraction.spans if s.span_key == "intent.t1b0")
    assert [f.status for f in extraction.fields] == ["extracted"]
    blank = run.extraction.model_copy(update={"spans": [s.model_copy(update={"fields": [f.model_copy(update={"status": "absent", "value": None, "evidence": None,
                                                                                                                "char_start": None, "char_end": None})
                                                                                          for f in s.fields]}) for s in run.extraction.spans]})
    merged = _merge(tmp_path, run, extraction=blank, taxonomy=cancel_taxonomy(), script=CANCEL_SCRIPT)
    intent = next(h for h in merged.signals if h.category_id == "intent")
    assert [(f.field_id, f.status, f.value) for f in intent.fields] == [("reason", "absent", None)]


def test_a_missing_categorize_output_fails_the_merge(tmp_path):
    run = run_v2(tmp_path, SCRIPT)
    transcript = script_transcript(SCRIPT)
    snap = snapshot()
    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript)),
              "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap), "stage:categorize": (ArtifactKind.SIGNAL_CATEGORIES, None),
              "stage:subcategorize": (ArtifactKind.SIGNAL_SUBCATEGORIES, run.subcategories)}
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_MERGE, inputs, entry_id=None, parameters=signal_params(snap))
    with pytest.raises(HandlerError) as err:
        build_registry("fake").get(JobType.CONTACT_SIGNALS_MERGE).run(job)
    assert err.value.code is JobErrorCode.INPUT_UNAVAILABLE


def test_a_missing_stage_two_or_three_output_is_partial_and_names_the_stage(tmp_path):
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), skip=["subcategorize"])
    result = run.result
    assert result.completeness == "partial" and "Subcategories unavailable (provider_error)" in result.partial_reason
    stages = {s.stage: s for s in result.stages}
    assert not stages["subcategorize"].included and stages["subcategorize"].failure_code is JobErrorCode.PROVIDER_ERROR
    assert result.signals and all(h.subcategory_id is None and h.confidence == h.category_confidence for h in result.signals)
    # The extract job saw no stage-2 output: every span is upstream_missing, and no model ran.
    assert {s.status for s in run.extraction.spans} == {"upstream_missing"}
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), skip=["extract"])
    assert run.result.completeness == "partial" and "Fields unavailable (provider_error)" in run.result.partial_reason
    assert next(h for h in run.result.signals if h.category_id == "intent").subcategory_id == "cancel_account"


def test_long_call_extracts_fields_beyond_the_old_twenty_four_span_limit(tmp_path):
    item = SignalField(field_id="item", name="Item number", type="number", description="The item number being discussed", pii_class="none")
    taxonomy = with_categories([custom_category("thanks", name="Thanks", gloss="Someone thanks the other", examples=["thank"], fields=[item])])
    script = [(SpeakerRole.CALLER, f"Thank you for checking item {i}.") for i in range(32)]
    run = run_v2(tmp_path, script, taxonomy)
    thanks = [hit for hit in run.result.signals if hit.category_id == "thanks"]
    assert len(run.extraction.spans) == 32 and sum(1 + len(hit.parts) for hit in thanks) == 32
    assert [span.fields[0].value for span in run.extraction.spans] == list(range(32))
    assert all(span.fields[0].status == "extracted" for span in run.extraction.spans)
    assert run.result.completeness == "complete" and run.result.partial_reason is None


def test_spans_past_the_extraction_cap_stay_categorized_and_the_result_says_so(tmp_path, monkeypatch):
    from call1.process.handlers import signal_stages

    monkeypatch.setattr(signal_stages, "CONTRACT_PARAMETERS", CONTRACT_PARAMETERS.model_copy(update={"max_extraction_spans_per_call": 1}))
    taxonomy = with_categories([custom_category("thanks", name="Thanks", gloss="Someone thanks the other", examples=["thank"], narrow_quote=True)])
    run = run_v2(tmp_path, SCRIPT, taxonomy)
    thanks = [h for h in run.result.signals if h.category_id == "thanks"]
    spans = sum(1 + len(h.parts) for h in thanks)  # the cap counts spans; a multi-segment hit is several (decision 25)
    assert len(thanks) > 1 and len(run.extraction.spans) == 1
    assert run.result.completeness == "partial" and f"extraction_cap: {spans - 1} spans" in run.result.partial_reason
    assert sum(1 for h in thanks if h.quote_narrowed) == 1


def test_a_subcategory_threshold_edit_is_re_derived_in_the_merge(tmp_path):
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy())
    taxonomy = cancel_taxonomy()
    data = taxonomy.model_dump()
    next(c for c in data["categories"] if c["category_id"] == "intent")["subcategory_threshold"] = 0.9
    from call1.contracts.signals import SignalTaxonomy

    merged = _merge(tmp_path, run, taxonomy=SignalTaxonomy.model_validate(data), script=CANCEL_SCRIPT)
    intent = next(h for h in merged.signals if h.category_id == "intent")
    assert intent.subcategory_id == "other" and intent.fields == [] and not intent.quote_narrowed


def test_the_v1_branch_is_unchanged(tmp_path):
    from call1.contracts.contents import ContactSignalPass, ContactSignalsPassContent, ContactSignalView

    transcript = script_transcript(SCRIPT)
    view = ContactSignalView(id="s1", kind=ContactSignalKind.INTENT, label="Intent", start=4.0, end=7.8, speaker=SpeakerRole.CALLER,
                             quote="a question about a fee", turn_id=1, char_start=0, char_end=0, confidence=0.8)
    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript),
              "pass:lifecycle:0": (ArtifactKind.CONTACT_SIGNALS_PASS, ContactSignalsPassContent(pass_kind=ContactSignalPass.LIFECYCLE, signals=[view]))}
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_MERGE, inputs, entry_id=None)
    content = build_registry("fake").get(JobType.CONTACT_SIGNALS_MERGE).run(job).outputs["contact_signals"].content
    assert content.pipeline == "v1" and [s.id for s in content.signals] == ["lifecycle-s1"] and content.stages == []
