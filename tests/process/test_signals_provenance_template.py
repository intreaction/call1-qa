"""Stage-1/2 provenance names the prompt that produced it, and carry-forward only reuses answers from
the same engine, adapter, calibration and prompt/schema version (section 6.4, 7.5)."""

from __future__ import annotations

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.jobs import JobType
from call1.pipeline.signals_v2 import STAGE1_TEMPLATE, STAGE2_TEMPLATE
from call1.process.handlers import build_registry
from call1.process.handlers.fake import CANCEL_SCRIPT, FakeSignalClassifier
from call1.process.handlers.real import signals_v2 as real
from call1.process.handlers.real.llm import template_version
from call1.process.handlers.signal_stages import carry_forwardable, classifier_template

from .test_signals_support import (
    cancel_taxonomy,
    findings_for,
    make_job,
    run_v2,
    script_transcript,
    signal_params,
    snapshot,
)


def test_the_gemma_classifier_records_a_digest_of_each_stages_prompt():
    engine = real.GemmaSegmentClassifier
    assert classifier_template(engine, "categorize") == real.STAGE1_TEMPLATE_VERSION
    assert classifier_template(engine, "subcategorize") == real.STAGE2_TEMPLATE_VERSION
    assert real.STAGE1_TEMPLATE_VERSION.startswith("t-") and real.STAGE1_TEMPLATE_VERSION != real.STAGE2_TEMPLATE_VERSION
    # The digest moves with the prompt: an older stage-2 prompt (no fits gate) gets another version.
    older = template_version(STAGE2_TEMPLATE, "an older stage-2 prompt", "assessment+choice", real.STAGE2_PREVIOUS_ENTRIES,
                             real.PICK_SCORES[0], real.PICKED_NONE)
    assert older != real.STAGE2_TEMPLATE_VERSION


def test_an_engine_without_a_prompt_records_the_stage_name():
    engine = FakeSignalClassifier(cancel_taxonomy(), 1)
    assert classifier_template(engine, "categorize") == STAGE1_TEMPLATE
    assert classifier_template(engine, "subcategorize") == STAGE2_TEMPLATE


def test_carry_forward_needs_the_same_engine_adapter_calibration_and_template(tmp_path):
    previous = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy())
    prov = previous.subcategories.provenance
    engine = FakeSignalClassifier(cancel_taxonomy(), 2)
    kw = {"template": prov.question_template, "adapter_version": prov.adapter_version}
    assert carry_forwardable(prov, engine, **kw)
    assert not carry_forwardable(None, engine, **kw)
    assert not carry_forwardable(prov.model_copy(update={"question_template": "t-older-prompt"}), engine, **kw)
    assert not carry_forwardable(prov.model_copy(update={"adapter_version": "0"}), engine, **kw)
    assert not carry_forwardable(prov.model_copy(update={"calibration_id": "another-fit"}), engine, **kw)
    assert not carry_forwardable(prov.model_copy(update={"catalog_entry_id": "another-engine"}), engine, **kw)


def _rerun_stage2(tmp_path, previous_subcategories):
    taxonomy = cancel_taxonomy()
    snap = snapshot(taxonomy)
    transcript = script_transcript(CANCEL_SCRIPT)
    first = run_v2(tmp_path, CANCEL_SCRIPT, taxonomy)
    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript)),
              "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap), "categories": (ArtifactKind.SIGNAL_CATEGORIES, first.categories),
              "previous_subcategories": (ArtifactKind.SIGNAL_SUBCATEGORIES, previous_subcategories(first.subcategories))}
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_SUBCATEGORIZE, inputs, parameters=signal_params(snap))
    return first, build_registry("fake").get(JobType.CONTACT_SIGNALS_SUBCATEGORIZE).run(job).outputs["subcategories"].content


def test_stage_two_carries_forward_decisions_from_the_same_prompt(tmp_path):
    first, content = _rerun_stage2(tmp_path, lambda subs: subs)
    assert first.subcategories.decisions and set(content.carried_forward) == {d.span_key for d in first.subcategories.decisions}


def test_stage_two_reruns_decisions_an_older_prompt_made(tmp_path):
    def older(subs):
        return subs.model_copy(update={"provenance": subs.provenance.model_copy(update={"question_template": "t-older-prompt"})})

    first, content = _rerun_stage2(tmp_path, older)
    assert content.carried_forward == []
    assert {d.span_key for d in content.decisions} == {d.span_key for d in first.subcategories.decisions}
    assert content.provenance.question_template == STAGE2_TEMPLATE
