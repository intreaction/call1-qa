"""The v2 fakes, scripts and masking (docs/ContactSignalsV2.md sections 3.3, 8.6, 9.4 and 11.2;
F3 acceptance "Fakes" and "Masking")."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.catalog import ModelPurpose
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobParameters, JobType
from call1.contracts.signals import SignalSubcategory
from call1.pipeline.signals_v2 import ChoiceRow, render_extraction_prompt
from call1.process.catalog import FAKE_SIGNAL_CLASSIFIER_ID, FAKE_SIGNAL_EXTRACTOR_ID, seeded_catalog
from call1.process.handlers import build_registry
from call1.process.handlers.base import HandlerError, ReleaseJob
from call1.process.handlers.fake import (
    CALLER_NAME_SCRIPT,
    CANCEL_SCRIPT,
    COMPETITOR_SCRIPT,
    SCRIPT,
    FakeBehavior,
    FakeSignalClassifier,
    FakeSignalExtractor,
)
from call1.redaction import REDACTED

from .test_signals_support import cancel_taxonomy, custom_category, make_job, run_v2, with_categories

SAMPLE = Path(__file__).resolve().parents[2] / "sample_audio" / "call_01_compliant.wav"


def test_co_occurrence_on_the_default_script(tmp_path):
    run = run_v2(tmp_path, SCRIPT)
    turn1 = {h.category_id for h in run.result.signals if h.turn_id == 1}
    assert turn1 == {"intent", "issue"}  # "a question about a fee": both fire on one segment
    row = next(s for s in run.categories.scores if s.index == 1)
    assert row.probabilities["intent"] >= run.categories.thresholds["intent"] and row.probabilities["issue"] >= run.categories.thresholds["issue"]
    assert run.categories.provenance.calibration_id == "fake-signals-v1" and run.categories.provenance.device == "fake"
    assert run.categories.provenance.catalog_entry_id == FAKE_SIGNAL_CLASSIFIER_ID


def _asr(tmp_path, behavior: FakeBehavior):
    job = make_job(tmp_path, JobType.ASR, {}, entry_id="parakeet-tdt-0.6b-v3", purpose=ModelPurpose.ASR,
                   parameters=JobParameters(extra={"channels": 2}))
    from call1.process.handlers.base import InputArtifact
    from .test_signals_support import artifact

    data = SAMPLE.read_bytes()
    record, _ = artifact("audio", ArtifactKind.SOURCE_AUDIO, data)
    record = record.model_copy(update={"content_type": "audio/wav"})

    def fetch(_artifact, dest):
        if dest is None:
            return data
        dest.write_bytes(data)
        return dest

    job.inputs["audio"] = InputArtifact("audio", record, fetch, job.scratch_dir)
    return build_registry("fake", fake_behavior=behavior).get(JobType.ASR).run(job).outputs["transcript"].content, record.checksum


def test_the_cancel_and_competitor_scripts_by_action_and_by_source(tmp_path, monkeypatch):
    transcript, checksum = _asr(tmp_path, FakeBehavior({"asr": ["script:cancel"]}))
    assert transcript.turns[1].text == CANCEL_SCRIPT[1][1] == "Hi, I want to cancel my account because the price went up again."
    assert [t.speaker for t in transcript.turns] == [role for role, _ in SCRIPT[:len(transcript.turns)]]  # SCRIPT's speaker order
    transcript, _ = _asr(tmp_path, FakeBehavior({"asr": ["script:competitor"]}))
    assert any("you're with a competitor" in t.text for t in transcript.turns[:3])
    default, _ = _asr(tmp_path, FakeBehavior())
    assert default.turns[1].text == SCRIPT[1][1]  # SCRIPT stays the default
    monkeypatch.setenv("CALL1_FAKE_SCRIPTS", json.dumps({checksum: "cancel"}))
    by_source, _ = _asr(tmp_path, FakeBehavior())
    assert by_source.turns[1].text == CANCEL_SCRIPT[1][1]
    with pytest.raises(HandlerError):
        _asr(tmp_path, FakeBehavior({"asr": ["script:nope"]}))
    assert len(COMPETITOR_SCRIPT) == len(CANCEL_SCRIPT) == len(SCRIPT)


def test_the_cancel_script_yields_the_demo_hit(tmp_path):
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy())
    intent = next(h for h in run.result.signals if h.category_id == "intent")
    assert intent.subcategory_id == "cancel_account" and intent.subcategory_label == "Cancel account"
    assert [(f.name, f.value) for f in intent.fields] == [("Reason", "price")] and intent.quote == "cancel my account" and intent.quote_narrowed
    assert run.extraction.provenance.catalog_entry_id == FAKE_SIGNAL_EXTRACTOR_ID


def test_competitor_on_either_speaker(tmp_path):
    run = run_v2(tmp_path, COMPETITOR_SCRIPT, with_categories([custom_category()]))
    hit = next(h for h in run.result.signals if h.category_id == "competitor")
    assert hit.turn_id == 2 and "competitor" in hit.quote and hit.kind.value == "custom"


@pytest.mark.parametrize("job_type", [JobType.CONTACT_SIGNALS_CATEGORIZE, JobType.CONTACT_SIGNALS_SUBCATEGORIZE, JobType.CONTACT_SIGNALS_EXTRACT])
def test_the_fake_behavior_keys(tmp_path, job_type):
    key = job_type.value
    with pytest.raises(HandlerError) as err:
        run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), behavior=FakeBehavior({key: ["fail:provider_timeout"]}))
    assert err.value.code is JobErrorCode.PROVIDER_TIMEOUT
    with pytest.raises(ReleaseJob) as release:
        run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), behavior=FakeBehavior({key: ["release:resource_unavailable"]}))
    assert release.value.disposition == "requeue"
    with pytest.raises(RuntimeError):
        run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), behavior=FakeBehavior({key: ["crash"]}))
    held = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), behavior=FakeBehavior({key: ["hold:0.05"]}))
    assert held.result.completeness == "complete"
    if job_type is JobType.CONTACT_SIGNALS_EXTRACT:
        return  # invalid_answer and provider_error on stage 3 fail spans, not the job (test_signals_stage3)
    for action, code in (("invalid_answer", JobErrorCode.VALIDATION_REJECTED), ("provider_error", JobErrorCode.PROVIDER_ERROR)):
        with pytest.raises(HandlerError) as err:
            run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), behavior=FakeBehavior({key: [action]}))
        assert err.value.code is code


def test_low_confidence_scores_just_under_the_threshold(tmp_path):
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), behavior=FakeBehavior({"contact_signals_categorize": ["low_confidence"]}))
    assert run.categories.spans == [] and run.result.signals == []
    intent = next(s for s in run.categories.scores if s.index == 1).probabilities["intent"]
    assert intent == pytest.approx(run.categories.thresholds["intent"] - 0.05)
    run = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy(), behavior=FakeBehavior({"contact_signals_subcategorize": ["low_confidence"]}))
    assert next(h for h in run.result.signals if h.category_id == "intent").subcategory_id == "other"


def test_stage_two_says_not_on_not_really(tmp_path):
    script = [SCRIPT[0], (SCRIPT[1][0], "Well, not really a question about a fee, just saying hi."), *SCRIPT[2:]]
    run = run_v2(tmp_path, script)
    decisions = {d.span_key: d for d in run.subcategories.decisions}
    assert decisions["intent.t1b0"].decision == "rejected" and decisions["intent.t1b0"].confidence == pytest.approx(0.2)
    assert not any(h.turn_id == 1 for h in run.result.signals)  # rejected spans are kept for evaluation, but are not hits


# --- masking ---------------------------------------------------------------------------------


class Recorder(FakeSignalClassifier):
    rows_seen: List[ChoiceRow] = []

    def choose(self, rows):
        Recorder.rows_seen.extend(rows)
        return super().choose(rows)


def test_no_raw_sensitive_value_reaches_any_row_option_or_prompt(tmp_path, monkeypatch):
    """The caller's name is masked in every segment, context entry and option, including a gloss,
    a name and an example an admin pasted it into (section 9.4 item 2)."""
    planted = custom_category("maria", name="Maria Lopez calls", gloss="Maria Lopez asks for help", speaker="CALLER",
                              examples=["last four digits"], narrow_quote=True, description="Anything Maria Lopez says")
    sub = SignalSubcategory(subcategory_id="maria_sub", name="About Maria Lopez", gloss="Maria Lopez reads digits", examples=["digits"],
                            description="Maria Lopez details")
    planted = planted.model_copy(update={"subcategories": [sub]})
    taxonomy = with_categories([planted])
    Recorder.rows_seen = []
    monkeypatch.setattr("call1.process.handlers.fake.FakeSignalClassifier", Recorder)
    extracted: List = []
    original = FakeSignalExtractor.extract

    def spy(self, spans):
        extracted.extend(spans)
        return original(self, spans)

    monkeypatch.setattr(FakeSignalExtractor, "extract", spy)
    run = run_v2(tmp_path, CALLER_NAME_SCRIPT, taxonomy)
    assert Recorder.rows_seen and extracted
    blob = json.dumps([[r.question, list(r.options), dict(r.state)] for r in Recorder.rows_seen])
    assert "Maria" not in blob and "Lopez" not in blob and REDACTED in blob
    system, user = render_extraction_prompt(extracted)
    assert "Maria" not in user and "Lopez" not in user and REDACTED in user
    assert all("Maria" not in f.description for s in extracted for f in s.fields)
    assert all("Maria" not in h.quote for h in run.result.signals)
    # The fake engine found the planted category by its example on the masked segment.
    assert any(h.category_id == "maria" for h in run.result.signals)


def test_the_placeholder_is_never_the_classifier_mask_token():
    assert REDACTED == "[REDACTED]" and REDACTED != "[MASK]"


def test_fake_catalog_entries_are_labelled_fake_and_real_mode_uses_gemma():
    fake = seeded_catalog(mode="fake")
    for entry_id in (FAKE_SIGNAL_CLASSIFIER_ID, FAKE_SIGNAL_EXTRACTOR_ID):
        entry = fake.get(entry_id)
        assert "(fake)" in entry.display_name and entry.model_family == "fake"
    assert fake.select(ModelPurpose.SIGNAL_CATEGORY).entry_id == fake.select(ModelPurpose.SIGNAL_SUBCATEGORY).entry_id == FAKE_SIGNAL_CLASSIFIER_ID
    real = seeded_catalog(mode="real")
    assert FAKE_SIGNAL_CLASSIFIER_ID not in real.entries and FAKE_SIGNAL_EXTRACTOR_ID not in real.entries
    assert real.defaults[ModelPurpose.SIGNAL_CATEGORY] == real.defaults[ModelPurpose.SIGNAL_EXTRACTION] == "call1-bundled"


def test_a_segment_whose_only_option_is_none_is_not_scored(tmp_path):
    run = run_v2(tmp_path, SCRIPT, stereo=False)  # mono with attribution: speakers known
    assert run.categories.skipped_unattributed == 0 and len(run.categories.scores) == len(run.categories.segments)
    from .test_signals_support import script_transcript

    unattributed = script_transcript(SCRIPT, stereo=False)
    run = run_v2(tmp_path, SCRIPT, transcript=unattributed)  # UNKNOWN speakers, no either-speaker category
    assert run.categories.skipped_unattributed == run.categories.unscored_no_options == len(run.categories.segments)
    assert run.categories.scores == [] and run.result.signals == [] and run.result.segmentation.scored_segments == 0
    either = run_v2(tmp_path, COMPETITOR_SCRIPT, with_categories([custom_category()]), transcript=script_transcript(COMPETITOR_SCRIPT, stereo=False))
    assert {h.category_id for h in either.result.signals} == {"competitor"}  # only either-speaker categories score UNKNOWN segments
    assert either.categories.unscored_no_options == 0 and either.categories.skipped_unattributed == len(either.categories.segments)
