"""Contact Signals v2 graph planning (docs/ContactSignalsV2.md sections 5.7, 8.1, 11.2; F3 acceptance "Graph")."""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import pytest

from call1.contracts.artifacts import Artifact
from call1.contracts.catalog import ModelPurpose
from call1.contracts.contents import ContactSignalPass, ContactSignalsPassContent
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JOB_TYPE_RULES, JobDefinition, JobGraphRequest, JobParameters, JobType, MemorySlot
from call1.contracts.signals import SignalSettings
from call1.pipeline.signals_v2 import ALL, RerunPlan
from call1.process.audio import AudioInfo
from call1.process.catalog import ProcessCatalog, fake_status, seeded_catalog
from call1.process.config import ProcessConfig
from call1.process.graph import V2_UNAVAILABLE_NOTE, GraphPlanner, PlanError, SignalsInput
from call1.process.handlers import build_registry
from call1.process.handlers.base import HandlerError
from call1.process.worker import TORCH_TYPES

from .test_process_units import _artifact_model, _rubric
from .test_signals_support import cancel_taxonomy, make_job, script_transcript, snapshot

V2_TYPES = (JobType.CONTACT_SIGNALS_CATEGORIZE, JobType.CONTACT_SIGNALS_SUBCATEGORIZE, JobType.CONTACT_SIGNALS_EXTRACT)


def _art(kind: str, aid: str) -> Artifact:
    from call1.contracts.artifacts import ArtifactKind

    base = _artifact_model("transcript", aid)
    return base.model_copy(update={"kind": ArtifactKind(kind), "checksum": "sha256:" + hashlib.sha256(aid.encode()).hexdigest()})


def signals_input(taxonomy=None, settings: Optional[SignalSettings] = None, **kw) -> SignalsInput:
    content = snapshot(taxonomy, settings)
    return SignalsInput(snapshot=_art("signal_taxonomy_snapshot", "art_tax"), content=content, **kw)


def ingest(signals: Optional[SignalsInput], *, channels: int = 2, config: Optional[ProcessConfig] = None, catalog=None,
           planner: Optional[GraphPlanner] = None, priority_offset: int = 0):
    planner = planner or GraphPlanner(catalog or seeded_catalog(mode="fake"), config or ProcessConfig())
    definition, ref = _rubric()
    audio = AudioInfo(container="wav", content_type="audio/wav", size_bytes=10, channels=channels, duration_seconds=90)
    graph = planner.ingest(conversation_id="conv_1", source=_artifact_model(), audio=audio, rubric_ref=ref, rubric=definition,
                           snapshot=_artifact_model("rubric_snapshot", "art_snap"), signals=signals, priority_offset=priority_offset)
    JobGraphRequest.model_validate(graph.model_dump(mode="json"))  # the contract's graph check
    return graph, {j.ref: j for j in graph.jobs}


def _inputs(job) -> Dict[str, object]:
    return {i.role: i for i in job.inputs}


def test_the_v2_graph_shape_uses_after_edges_and_pins_the_taxonomy_snapshot():
    signals = signals_input(cancel_taxonomy())
    graph, by_ref = ingest(signals)
    assert {"cs-categorize", "cs-subcategorize", "cs-extract", "cs-merge"} <= set(by_ref)
    assert not {"cs-lifecycle", "cs-resolution"} & set(by_ref)  # v1 and v2 never mix in one graph
    digest = signals.content.taxonomy_ref.digest
    cat, sub, ext, merge = (by_ref[r] for r in ("cs-categorize", "cs-subcategorize", "cs-extract", "cs-merge"))
    for job in (cat, sub, ext):
        JobDefinition.model_validate(job.model_dump(mode="json"))
        rule = JOB_TYPE_RULES[job.job_type]
        assert rule.needs_signal_taxonomy and job.selection is not None and job.selection.purpose is rule.purpose
        assert job.parameters.signals.taxonomy_digest == digest and job.parameters.window is None and job.parameters.pass_kind is None
        taxonomy = _inputs(job)["taxonomy"]
        assert taxonomy.artifact.artifact_id == "art_tax" and taxonomy.upstream is None
        # Always masked; pii_findings on a success edge (so a failed enrichment dead-blocks the chain).
        assert job.selection.route.masked is True
        assert _inputs(job)["pii_findings"].upstream.ref == "enrichment" and "enrichment" in job.requires_refs
        assert "asr" in job.requires_refs
    assert cat.parameters.signals.stage1_mode == "run" and cat.after_refs == []
    assert sub.after_refs == ["cs-categorize"] and _inputs(sub)["categories"].optional
    assert ext.after_refs == ["cs-categorize", "cs-subcategorize"] and _inputs(ext)["subcategories"].optional
    assert merge.after_refs == ["cs-categorize", "cs-subcategorize", "cs-extract"]
    assert {r for r in _inputs(merge) if r.startswith("stage:")} == {"stage:categorize", "stage:subcategorize", "stage:extract"}
    assert all(_inputs(merge)[r].optional for r in ("stage:categorize", "stage:subcategorize", "stage:extract"))
    assert "enrichment" in merge.requires_refs and merge.parameters.signals.taxonomy_digest == digest
    assert merge.parameters.extra["planned_stages"] == "categorize,subcategorize,extract" and merge.selection is None
    # Stages 1 and 2 run in the torch pool (they share the GPU under inference_lock); stage 3 is an LLM-route job.
    assert JobType.CONTACT_SIGNALS_CATEGORIZE in TORCH_TYPES and JobType.CONTACT_SIGNALS_SUBCATEGORIZE in TORCH_TYPES
    assert cat.resource_estimate.memory_slot is MemorySlot.CPU
    assert cat.selection.catalog_entry.entry_id == "fake-signal-classifier" and ext.selection.catalog_entry.entry_id == "fake-signal-extractor"
    # mono: the categorize job requires the attribution too
    _, mono = ingest(signals_input(cancel_taxonomy()), channels=1)
    assert "speaker" in mono["cs-categorize"].requires_refs and "speaker_attribution" in _inputs(mono["cs-categorize"])


def test_extract_is_planned_only_when_an_active_node_has_fields_or_narrow_quote():
    _, builtins = ingest(signals_input())
    assert "cs-extract" not in builtins and builtins["cs-merge"].parameters.extra["planned_stages"] == "categorize,subcategorize"
    _, narrow_only = ingest(signals_input(cancel_taxonomy(fields=False, narrow=True)))
    assert "cs-extract" in narrow_only
    _, neither = ingest(signals_input(cancel_taxonomy(fields=False, narrow=False)))
    assert "cs-extract" not in neither


def test_v2_purposes_are_masked_even_with_mask_model_text_off():
    _, by_ref = ingest(signals_input(cancel_taxonomy()), config=ProcessConfig(mask_model_text="off"))
    for ref in ("cs-categorize", "cs-subcategorize", "cs-extract"):
        assert by_ref[ref].selection.route.masked is True, ref
    assert by_ref["qa-0-reg-01"].selection.route.masked is False  # QA follows the switch; v2 never does
    _, v1 = ingest(None, config=ProcessConfig(mask_model_text="off"))
    assert v1["cs-lifecycle"].selection.route.masked is False  # the v1 passes are unchanged


def test_stage_two_waits_on_tone_and_sentiment_only_when_its_adapter_enables_them(monkeypatch):
    _, by_ref = ingest(signals_input())
    assert by_ref["cs-subcategorize"].after_refs == ["cs-categorize"]
    assert not {"tone_blocks", "text_sentiment"} & set(_inputs(by_ref["cs-subcategorize"]))
    monkeypatch.setattr(GraphPlanner, "stage2_factors", lambda self, entry: ("span", "previous", "next", "speaker", "tone", "sentiment"))
    _, enabled = ingest(signals_input())
    sub = enabled["cs-subcategorize"]
    assert sub.after_refs == ["cs-categorize", "tone", "sentiment"]
    assert _inputs(sub)["tone_blocks"].optional and _inputs(sub)["text_sentiment"].optional


def _without(purpose: ModelPurpose) -> ProcessCatalog:
    catalog = seeded_catalog(mode="fake")
    defaults = {p: e for p, e in catalog.defaults.items() if p is not purpose}
    return ProcessCatalog(catalog.entries.values(), defaults, fake_status)


def test_v2_needs_both_classifier_entries_else_v1_with_a_note_or_a_configuration_error(tmp_path):
    for catalog in (_without(ModelPurpose.SIGNAL_CATEGORY), _without(ModelPurpose.SIGNAL_SUBCATEGORY)):
        _, by_ref = ingest(signals_input(settings=SignalSettings(pipeline="v2", v1_fallback=True)), catalog=catalog)
        assert {"cs-lifecycle", "cs-resolution", "cs-merge"} <= set(by_ref) and "cs-categorize" not in by_ref
        assert by_ref["cs-merge"].parameters.extra["pipeline_note"] == V2_UNAVAILABLE_NOTE
    # v1 fallback off: only a merge that records configuration_error
    _, by_ref = ingest(signals_input(settings=SignalSettings(pipeline="v2", v1_fallback=False)), catalog=_without(ModelPurpose.SIGNAL_CATEGORY))
    assert not {"cs-lifecycle", "cs-resolution", "cs-categorize"} & set(by_ref)
    merge = by_ref["cs-merge"]
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_MERGE, {}, entry_id=None, parameters=merge.parameters)
    with pytest.raises(HandlerError) as err:
        build_registry("fake").get(JobType.CONTACT_SIGNALS_MERGE).run(job)
    assert err.value.code is JobErrorCode.CONFIGURATION_ERROR
    # a preview or compare request never falls back
    with pytest.raises(PlanError):
        ingest(signals_input(require_v2=True), catalog=_without(ModelPurpose.SIGNAL_SUBCATEGORY))
    # v1 and shadow build today's passes with no note; no snapshot means v1
    for signals in (None, signals_input(settings=SignalSettings(pipeline="v1")), signals_input(settings=SignalSettings(pipeline="shadow"))):
        _, by_ref = ingest(signals)
        assert {"cs-lifecycle", "cs-resolution"} <= set(by_ref) and "pipeline_note" not in by_ref["cs-merge"].parameters.extra


def test_the_v1_merge_records_the_pipeline_note(tmp_path):
    transcript = script_transcript()
    passes = {f"pass:{kind.value}:0": ("contact_signals_pass", ContactSignalsPassContent(pass_kind=kind, signals=[])) for kind in ContactSignalPass}
    from call1.contracts.artifacts import ArtifactKind

    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), **{r: (ArtifactKind.CONTACT_SIGNALS_PASS, c) for r, (_, c) in passes.items()}}
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_MERGE, inputs, entry_id=None, parameters=JobParameters(extra={"pipeline_note": V2_UNAVAILABLE_NOTE}))
    content = build_registry("fake").get(JobType.CONTACT_SIGNALS_MERGE).run(job).outputs["contact_signals"].content
    assert content.pipeline == "v1" and content.pipeline_note == V2_UNAVAILABLE_NOTE and content.completeness == "complete"


def test_the_stage_three_fallback_is_frozen_only_when_it_differs_from_the_primary():
    _, by_ref = ingest(signals_input(cancel_taxonomy()))
    assert by_ref["cs-extract"].parameters.signals.fallback_entry_id is None  # fake default == primary
    settings = SignalSettings(pipeline="v2", fallback_extraction_entry_id="call1-bundled")
    _, by_ref = ingest(signals_input(cancel_taxonomy(), settings))
    assert by_ref["cs-extract"].parameters.signals.fallback_entry_id == "call1-bundled"
    settings = SignalSettings(pipeline="v2", fallback_extraction_entry_id="not-in-this-catalog")
    _, by_ref = ingest(signals_input(cancel_taxonomy(), settings))
    assert by_ref["cs-extract"].parameters.signals.fallback_entry_id is None
    # Decision 24: Gemma serves every v2 stage and is the real-mode default for each.
    real = seeded_catalog(mode="real", status_fn=fake_status)
    bundled = real.get("call1-bundled")
    purposes = {ModelPurpose.SIGNAL_CATEGORY, ModelPurpose.SIGNAL_SUBCATEGORY, ModelPurpose.SIGNAL_EXTRACTION}
    assert purposes <= set(bundled.purposes)
    assert all(real.defaults[p] == "call1-bundled" for p in purposes)


def test_the_real_catalog_builds_v2_on_gemma():
    _, by_ref = ingest(signals_input(), catalog=seeded_catalog(mode="real", status_fn=fake_status))
    assert {"cs-categorize", "cs-subcategorize", "cs-merge"} <= set(by_ref) and "cs-lifecycle" not in by_ref
    assert "call1-bundled" in by_ref["cs-categorize"].selection.model_dump_json()


def test_preview_graphs_write_draft_slots_and_take_the_request_priority():
    planner = GraphPlanner(seeded_catalog(mode="fake"), ProcessConfig())
    signals = signals_input(cancel_taxonomy(), draft_request_id="req_preview1", preview_id="spv_1", require_v2=True, pipeline="v2")
    _, by_ref = ingest(signals, planner=planner, priority_offset=5)
    v2 = [by_ref[r] for r in ("cs-categorize", "cs-subcategorize", "cs-extract", "cs-merge")]
    for job in v2:
        assert job.parameters.extra["draft_test_request_id"] == "req_preview1" and job.parameters.signals.preview_id == "spv_1"
        assert job.priority == 10 + 5
    assert by_ref["asr"].priority == 50 + 5


def _previous() -> Dict[str, Artifact]:
    return {"categories": _art("signal_categories", "art_prev_cat"), "subcategories": _art("signal_subcategories", "art_prev_sub"),
            "extraction": _art("signal_extraction", "art_prev_ext")}


def test_a_reanalysis_plan_runs_only_its_stages_and_pins_the_rest():
    from call1.process.graph import QaSources, Src, new_jobs

    planner = GraphPlanner(seeded_catalog(mode="fake"), ProcessConfig())
    sources = QaSources(transcript=Src.pinned(_art("transcript", "art_tr")), pii=Src.pinned(_art("pii_findings", "art_pii")))

    def plan(p: RerunPlan):
        jobs = new_jobs("reanalysis.req_1", -10)
        planner.add_contact_signals(jobs, sources, 60.0, signals=signals_input(cancel_taxonomy(), previous=_previous(), plan=p, pipeline="v2"))
        JobGraphRequest.model_validate({"idempotency_key": "reanalysis.req_1", "reason": "reanalysis", "reanalysis_request_id": "req_1",
                                        "reanalysis_claim_token": "t" * 40, "jobs": [j.model_dump(mode="json") for j in jobs.defs]})
        return {j.ref: j for j in jobs.defs}

    # A subcategory edit: stage 2 for the listed spans, stage 3 reuses nothing new, stage 1 pinned.
    by_ref = plan(RerunPlan(subcategorize=["intent.t1b0"], extract=["intent.t1b0"], extract_planned=True))
    assert set(by_ref) == {"cs-subcategorize", "cs-extract", "cs-merge"}
    assert by_ref["cs-subcategorize"].parameters.signals.span_keys == ["intent.t1b0"]
    assert _inputs(by_ref["cs-subcategorize"])["categories"].artifact.artifact_id == "art_prev_cat"
    assert _inputs(by_ref["cs-subcategorize"])["previous_subcategories"].artifact.artifact_id == "art_prev_sub"
    merge = by_ref["cs-merge"]
    assert _inputs(merge)["stage:categorize"].artifact.artifact_id == "art_prev_cat" and merge.parameters.extra["pinned_stages"] == "categorize"
    assert all(j.priority == 10 - 10 for j in by_ref.values())
    # A threshold-only edit: categorize re-derives (a code stage, no selection) and the rest follow.
    by_ref = plan(RerunPlan(categorize="rederive", subcategorize=ALL, extract=ALL, extract_planned=True))
    cat = by_ref["cs-categorize"]
    assert cat.selection is None and cat.parameters.signals.stage1_mode == "rederive" and not cat.model_backed
    assert _inputs(cat)["previous_categories"].artifact.artifact_id == "art_prev_cat"
    assert cat.resource_estimate.memory_slot is MemorySlot.CPU
    # Nothing to rerun: just the merge over the previous artifacts.
    by_ref = plan(RerunPlan(extract_planned=True))
    assert set(by_ref) == {"cs-merge"} and by_ref["cs-merge"].parameters.extra["pinned_stages"] == "categorize,subcategorize,extract"
    # A field edit: stage 3 only.
    by_ref = plan(RerunPlan(extract=["intent.t1b0"], extract_planned=True))
    assert set(by_ref) == {"cs-extract", "cs-merge"} and by_ref["cs-extract"].parameters.signals.span_keys == ["intent.t1b0"]
    # Scopes whose stage-1 digest changed are named on the categorize job.
    by_ref = plan(RerunPlan(categorize="run", scopes=["CALLER"], subcategorize=ALL, extract=ALL, extract_planned=True))
    assert by_ref["cs-categorize"].parameters.extra["scopes"] == "CALLER" and by_ref["cs-categorize"].selection is not None


def test_v2_without_pii_findings_is_refused():
    from call1.process.graph import QaSources, Src, new_jobs

    planner = GraphPlanner(seeded_catalog(mode="fake"), ProcessConfig())
    with pytest.raises(PlanError):
        planner.add_contact_signals(new_jobs("k.x"), QaSources(transcript=Src.pinned(_art("transcript", "art_tr"))), 60.0,
                                    signals=signals_input())


def test_default_cascade_does_not_silently_fall_back_to_legacy_when_laya_is_missing():
    from call1.process.system_one import ENTRY_ID
    catalog = seeded_catalog(mode="fake")
    catalog.defaults[ModelPurpose.SIGNAL_CATEGORY] = ENTRY_ID  # absent on this host
    with pytest.raises(PlanError, match="cannot fall back to legacy"):
        ingest(signals_input(cancel_taxonomy(), SignalSettings(pipeline="v2", v1_fallback=True)), catalog=catalog)


@pytest.mark.parametrize("legacy", [None, "v1", "shadow"])
def test_default_cascade_cannot_execute_an_older_pipeline(legacy):
    catalog = seeded_catalog(mode="fake")
    catalog.defaults[ModelPurpose.SIGNAL_CATEGORY] = "laya-system-one"
    inputs = None if legacy is None else signals_input(settings=SignalSettings(pipeline=legacy))
    with pytest.raises(PlanError, match="requires a v2 taxonomy snapshot"):
        ingest(inputs, catalog=catalog)
