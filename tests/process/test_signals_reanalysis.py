"""What a taxonomy change reruns (docs/ContactSignalsV2.md section 7.5; F3 acceptance "Reanalysis"):
every row of the table produces exactly its stages, a threshold-only edit re-derives with no model,
and the reanalysis consumer turns ``contact_signals`` and ``contact_signals_preview`` requests into
exactly those graphs."""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pytest

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.common import canonical_json
from call1.contracts.jobs import JobType, ReanalysisKind, ReanalysisRequest, ReanalysisStatus
from call1.contracts.signals import SignalField, SignalSettings, SignalSubcategory, SignalTaxonomy
from call1.pipeline.signals_v2 import ALL, plan_rerun
from call1.process.catalog import seeded_catalog
from call1.process.config import ProcessConfig
from call1.process.graph import GraphPlanner
from call1.process.handlers import build_registry
from call1.process.handlers.fake import CANCEL_SCRIPT, COMPETITOR_SCRIPT, FakeSignalClassifier
from call1.process.reanalysis import ReanalysisConsumer

from .test_signals_support import (
    REASON,
    cancel_taxonomy,
    custom_category,
    make_job,
    run_v2,
    script_transcript,
    signal_params,
    snapshot,
    with_categories,
)


@pytest.fixture
def previous(tmp_path):
    return run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy())


def _plan(prev, taxonomy: SignalTaxonomy, **kw):
    transcript = script_transcript(CANCEL_SCRIPT)
    checksum = "sha256:" + hashlib.sha256(canonical_json(transcript)).hexdigest()
    kw.setdefault("transcript_checksum", checksum)
    return plan_rerun(taxonomy, previous_categories=prev.categories, previous_subcategories=prev.subcategories,
                      previous_extraction=prev.extraction, engine_thresholds={}, default_threshold=0.3, **kw)


def _edit(taxonomy: SignalTaxonomy, category_id: str, **update) -> SignalTaxonomy:
    data = taxonomy.model_dump()
    for c in data["categories"]:
        if c["category_id"] == category_id:
            c.update(update)
    return SignalTaxonomy.model_validate(data)


def _edit_sub(taxonomy: SignalTaxonomy, category_id: str, subcategory_id: str, **update) -> SignalTaxonomy:
    data = taxonomy.model_dump()
    for c in data["categories"]:
        if c["category_id"] == category_id:
            for s in c["subcategories"]:
                if s["subcategory_id"] == subcategory_id:
                    s.update(update)
    return SignalTaxonomy.model_validate(data)


def test_the_previous_run_has_the_spans_the_rows_edit(previous):
    keys = [s.span_key for s in previous.categories.spans]
    assert "intent.t1b0" in keys and previous.extraction is not None
    decision = next(d for d in previous.subcategories.decisions if d.span_key == "intent.t1b0")
    assert decision.subcategory_id == "cancel_account"


def test_an_unchanged_taxonomy_or_an_alert_rule_edit_reruns_nothing(previous):
    plan = _plan(previous, cancel_taxonomy())
    assert plan.stages() == [] and not plan.full


def test_a_stage_one_threshold_edit_rederives_and_stages_two_and_three_follow_only_for_new_spans(previous):
    # Raising a threshold only removes spans: the re-derive, then the merge (stages 2 and 3 pinned).
    plan = _plan(previous, _edit(cancel_taxonomy(), "fix_proposed", threshold=0.95))
    assert plan.categorize == "rederive" and plan.stages() == ["categorize"]
    # Lowering one adds spans: stage 2 runs on exactly those, and stage 3 only where the node has fields.
    caller = [seg for seg in previous.categories.segments if seg.speaker.value == "CALLER" and seg.turn_id != 1]
    target = caller[-1]
    scores = [sc.model_copy(update={"probabilities": {**sc.probabilities, "issue": 0.2, "intent": 0.2}}) if sc.index == target.index else sc
              for sc in previous.categories.scores]
    prev = copy.copy(previous)
    prev.categories = previous.categories.model_copy(update={"scores": scores})
    before = {s.span_key for s in previous.categories.spans}
    plan = _plan(prev, _edit(cancel_taxonomy(), "issue", threshold=0.1))
    assert plan.categorize == "rederive" and plan.stages() == ["categorize", "subcategorize"] and plan.extract is None
    assert plan.subcategorize == [f"issue.t{target.turn_id}b0"] and plan.subcategorize[0] not in before  # type: ignore[index]
    # A new span on a node with fields also reruns stage 3 for it.
    plan = _plan(prev, _edit(cancel_taxonomy(), "intent", threshold=0.1))
    assert plan.categorize == "rederive" and plan.stages() == ["categorize", "subcategorize", "extract"]
    assert plan.subcategorize == plan.extract == [f"intent.t{target.turn_id}b0"]


def test_a_subcategory_threshold_edit_reruns_no_model(previous):
    plan = _plan(previous, _edit(cancel_taxonomy(), "intent", subcategory_threshold=0.9))
    assert plan.stages() == []  # the merge re-derives the decision from the stored probabilities


def test_adding_a_custom_category_reruns_stage_one_for_its_scopes_only(previous):
    caller = with_categories([custom_category("cancel_ask", name="Cancel request", gloss="Caller asks to cancel", speaker="CALLER",
                                              examples=["cancel"])], intent=cancel_taxonomy().category("intent").model_dump())
    plan = _plan(previous, caller)
    assert plan.categorize == "run" and plan.scopes == ["CALLER"] and plan.subcategorize == ALL
    either = with_categories([custom_category()], intent=cancel_taxonomy().category("intent").model_dump())
    plan = _plan(previous, either)
    assert plan.categorize == "run" and plan.scopes == ["AGENT", "CALLER"]


def test_a_subcategory_edit_reruns_stage_two_for_that_categorys_spans(previous):
    extra = SignalSubcategory(subcategory_id="billing", name="Billing", gloss="Caller asks about billing", examples=["bill"])
    taxonomy = cancel_taxonomy()
    subs = [s.model_dump() for s in taxonomy.category("intent").subcategories] + [extra.model_dump()]
    plan = _plan(previous, _edit(taxonomy, "intent", subcategories=subs))
    assert plan.categorize is None and plan.subcategorize == ["intent.t1b0"]
    assert plan.extract == ["intent.t1b0"]  # the extract job carries it forward when its path did not change


def test_a_field_edit_or_narrow_quote_toggle_reruns_stage_three_only(previous):
    changed = REASON.model_copy(update={"description": "Why the caller is leaving"})
    plan = _plan(previous, _edit_sub(cancel_taxonomy(), "intent", "cancel_account", fields=[changed.model_dump()]))
    assert plan.stages() == ["extract"] and plan.extract == ["intent.t1b0"]
    plan = _plan(previous, _edit_sub(cancel_taxonomy(), "intent", "cancel_account", narrow_quote=False))
    assert plan.stages() == ["extract"] and plan.extract == ["intent.t1b0"]
    added = SignalField(field_id="plan_name", name="Plan", type="string", description="The plan the caller has", pii_class="product")
    plan = _plan(previous, _edit_sub(cancel_taxonomy(), "intent", "cancel_account", fields=[REASON.model_dump(), added.model_dump()]))
    assert plan.stages() == ["extract"]


def test_a_description_edit_reruns_stage_three_only_where_the_extractor_reads_it(previous):
    plan = _plan(previous, _edit_sub(cancel_taxonomy(), "intent", "cancel_account", description="Closing the account for good"))
    assert plan.stages() == ["extract"]
    # A node with no fields and no narrowing: nothing reruns.
    plan = _plan(previous, _edit_sub(cancel_taxonomy(), "intent", "fee_question", description="Any fee question"))
    assert plan.stages() == []


def test_a_category_name_edit_reruns_stage_two_and_keeps_hit_ids(tmp_path):
    taxonomy = with_categories([custom_category()])
    prev = run_v2(tmp_path, COMPETITOR_SCRIPT, taxonomy)
    key = next(s.span_key for s in prev.categories.spans if s.category_id == "competitor")
    renamed = _edit(taxonomy, "competitor", name="Competitor mentioned")
    transcript = script_transcript(COMPETITOR_SCRIPT)
    checksum = "sha256:" + hashlib.sha256(canonical_json(transcript)).hexdigest()
    plan = plan_rerun(renamed, previous_categories=prev.categories, previous_subcategories=prev.subcategories, previous_extraction=None,
                      transcript_checksum=checksum, default_threshold=0.3)
    assert plan.stages() == ["subcategorize"] and plan.subcategorize == [key]
    again = run_v2(tmp_path, COMPETITOR_SCRIPT, renamed)
    ids = lambda run: sorted(h.id for h in run.result.signals if h.category_id == "competitor")  # noqa: E731
    assert ids(again) == ids(prev) and ids(prev)  # the name is not part of the hit ID


def test_rescore_and_new_revisions_rerun_everything(previous):
    assert _plan(previous, cancel_taxonomy(), rescore=True).full
    assert _plan(previous, cancel_taxonomy(), transcript_checksum="sha256:" + "0" * 64).full
    assert _plan(previous, cancel_taxonomy(), attribution_checksum="sha256:" + "1" * 64).full
    assert plan_rerun(cancel_taxonomy()).full  # no previous v2 result (a v1 result, or none)


def test_a_rederive_loads_no_model_and_rebuilds_spans_from_stored_scores(tmp_path, previous, monkeypatch):
    def refuse(self):
        raise AssertionError("a re-derive must not load a model")

    monkeypatch.setattr(FakeSignalClassifier, "load", refuse)
    taxonomy = _edit(cancel_taxonomy(), "fix_proposed", threshold=0.95)
    snap = snapshot(taxonomy)
    transcript = script_transcript(CANCEL_SCRIPT)
    from .test_signals_support import findings_for

    inputs = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "pii_findings": (ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript)),
              "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap), "previous_categories": (ArtifactKind.SIGNAL_CATEGORIES, previous.categories)}
    job = make_job(tmp_path, JobType.CONTACT_SIGNALS_CATEGORIZE, inputs, entry_id=None, parameters=signal_params(snap, stage1_mode="rederive"))
    content = build_registry("fake").get(JobType.CONTACT_SIGNALS_CATEGORIZE).run(job).outputs["categories"].content
    assert content.mode == "rederive" and content.provenance == previous.categories.provenance
    assert content.scores == previous.categories.scores and content.segments == previous.categories.segments
    assert content.thresholds["fix_proposed"] == 0.95
    before = {s.span_key for s in previous.categories.spans}
    after = {s.span_key for s in content.spans}
    assert "fix_proposed.t4b0" in before and after == before - {"fix_proposed.t4b0"}


# --- the consumer ------------------------------------------------------------------------------


def _record(aid: str, kind: ArtifactKind, data: bytes, slot: str = "") -> Artifact:
    return Artifact.model_construct(id=aid, kind=kind, slot=slot, content_type="application/json", size_bytes=len(data),
                                    checksum="sha256:" + hashlib.sha256(data).hexdigest(), content_contract="", sensitivity="derived",
                                    conversation_id="conv_1", linked=True, version=1, superseded_by=None, labels={})


class FakeStore:
    """Just enough of ``StoreClient`` for ``ReanalysisConsumer.build``."""

    def __init__(self, taxonomy: SignalTaxonomy, settings: Optional[SignalSettings] = None) -> None:
        self.data: Dict[str, bytes] = {}
        self.records: Dict[str, Artifact] = {}
        self.taxonomy = taxonomy
        self.settings = settings or SignalSettings(pipeline="v2")
        self.minted: List[int] = []
        self.claimed_kinds = None

    def add(self, aid: str, kind: ArtifactKind, content, slot: str = "") -> Artifact:
        data = canonical_json(content)
        record = _record(aid, kind, data, slot)
        self.data[aid], self.records[aid] = data, record
        return record

    def list_artifacts(self, conversation_id, **kw):
        return [r for r in self.records.values() if not r.slot.startswith("draft:")]

    def download(self, artifact, dest=None):
        return self.data[artifact.id]

    def get_artifact(self, aid):
        return self.records[aid]

    def mint_signal_taxonomy_snapshot(self, conversation_id, version):
        self.minted.append(version)
        return self.add(f"art_snap_v{version}", ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snapshot(self.taxonomy, self.settings, version=version),
                        slot=f"signals:v{version}")

    def get_signal_taxonomy(self):  # pragma: no cover - requests carry their version
        raise AssertionError("the request names its taxonomy version")

    def claim_reanalysis(self, worker_id, max_requests=1, kinds=None):
        from call1.contracts.jobs import ReanalysisClaimResponse

        self.claimed_kinds = kinds
        return ReanalysisClaimResponse.model_construct(requests=[])


def _request(kind: ReanalysisKind, **kw) -> ReanalysisRequest:
    now = datetime.now(timezone.utc)
    base = dict(id="req_1", conversation_id="conv_1", call_id="call_1", kind=kind, status=ReanalysisStatus.CLAIMED, rubric=None,
                speaker_correction=None, draft_rubric=None, requested_at=now, priority=0, rescore_signals=False, signal_taxonomy_version=2,
                signal_pipeline="v2", signal_preview_id=None, signal_taxonomy_snapshot_artifact_id=None, idempotency_key="reanalysis-req-1")
    base.update(kw)
    return ReanalysisRequest.model_construct(**base)


def _store_with_previous(tmp_path, taxonomy: SignalTaxonomy) -> FakeStore:
    prev = run_v2(tmp_path, CANCEL_SCRIPT, cancel_taxonomy())
    store = FakeStore(taxonomy)
    transcript = script_transcript(CANCEL_SCRIPT)
    store.add("art_tr", ArtifactKind.TRANSCRIPT, transcript)
    from .test_signals_support import findings_for

    store.add("art_pii", ArtifactKind.PII_FINDINGS, findings_for(tmp_path, transcript))
    store.add("art_cat", ArtifactKind.SIGNAL_CATEGORIES, prev.categories)
    store.add("art_sub", ArtifactKind.SIGNAL_SUBCATEGORIES, prev.subcategories)
    store.add("art_ext", ArtifactKind.SIGNAL_EXTRACTION, prev.extraction)
    store.add("art_cs", ArtifactKind.CONTACT_SIGNALS, prev.result)
    return store


def _consumer(store: FakeStore, kinds=None) -> ReanalysisConsumer:
    return ReanalysisConsumer(ProcessConfig(), store, GraphPlanner(seeded_catalog(mode="fake"), ProcessConfig()), ledger=None,  # type: ignore[arg-type]
                              worker_id="w1", kinds=kinds)


def test_a_contact_signals_request_builds_only_the_outdated_stages(tmp_path):
    changed = REASON.model_copy(update={"description": "Why the caller is leaving"})
    store = _store_with_previous(tmp_path, _edit_sub(cancel_taxonomy(), "intent", "cancel_account", fields=[changed.model_dump()]))
    graph = _consumer(store).build(_request(ReanalysisKind.CONTACT_SIGNALS), "t" * 40)
    by_ref = {j.ref: j for j in graph.jobs}
    assert set(by_ref) == {"cs-extract", "cs-merge"} and store.minted == [2]
    ext = by_ref["cs-extract"]
    assert ext.parameters.signals.span_keys == ["intent.t1b0"]
    roles = {i.role: i for i in ext.inputs}
    assert roles["categories"].artifact.artifact_id == "art_cat" and roles["previous_extraction"].artifact.artifact_id == "art_ext"
    assert roles["taxonomy"].artifact.artifact_id == "art_snap_v2"
    # rescore reruns every stage and pins nothing previous
    graph = _consumer(store).build(_request(ReanalysisKind.CONTACT_SIGNALS, rescore_signals=True), "t" * 40)
    by_ref = {j.ref: j for j in graph.jobs}
    assert {"cs-categorize", "cs-subcategorize", "cs-extract", "cs-merge"} == set(by_ref)
    assert not any(i.role.startswith("previous_") for j in graph.jobs for i in j.inputs)
    # a request Store resolved to v1 builds the v1 passes
    graph = _consumer(store).build(_request(ReanalysisKind.CONTACT_SIGNALS, signal_pipeline="v1"), "t" * 40)
    assert {"cs-lifecycle", "cs-resolution", "cs-merge"} == {j.ref for j in graph.jobs}


def test_a_preview_request_runs_every_stage_into_draft_slots_at_its_priority(tmp_path):
    store = _store_with_previous(tmp_path, cancel_taxonomy())
    snap = store.add("art_preview_snap", ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snapshot(cancel_taxonomy(), preview_id="spv_1"),
                     slot="draft:req_1:signals:preview")
    request = _request(ReanalysisKind.CONTACT_SIGNALS_PREVIEW, priority=5, signal_preview_id="spv_1", signal_taxonomy_version=None,
                       signal_taxonomy_snapshot_artifact_id=snap.id)
    graph = _consumer(store).build(request, "t" * 40)
    by_ref = {j.ref: j for j in graph.jobs}
    assert {"cs-categorize", "cs-subcategorize", "cs-extract", "cs-merge"} == set(by_ref) and store.minted == []
    for job in graph.jobs:
        assert job.parameters.extra["draft_test_request_id"] == "req_1" and job.parameters.signals.preview_id == "spv_1"
        assert job.priority == 10 + 5
        assert {i.role: i for i in job.inputs}["taxonomy"].artifact.artifact_id == "art_preview_snap"


def test_claims_name_the_kinds_this_consumer_serves(tmp_path):
    store = FakeStore(cancel_taxonomy())
    _consumer(store).poll_once()
    assert store.claimed_kinds is None  # absent means every kind
    _consumer(store, kinds=[ReanalysisKind.CONTACT_SIGNALS, ReanalysisKind.CONTACT_SIGNALS_PREVIEW]).poll_once()
    assert store.claimed_kinds == [ReanalysisKind.CONTACT_SIGNALS, ReanalysisKind.CONTACT_SIGNALS_PREVIEW]
    from call1.contracts.jobs import ReanalysisClaimRequest

    body = ReanalysisClaimRequest(worker_id="w1", kinds=store.claimed_kinds)
    assert body.accepts(ReanalysisKind.CONTACT_SIGNALS) and not body.accepts(ReanalysisKind.QA)



def test_a_host_without_v2_classifiers_leaves_preview_requests_pending(tmp_path):
    """No usable signal_category/signal_subcategory engine: the claim leaves out the v2-only kinds, so
    Store keeps preview and compare requests pending for a host that can run them (sections 7.1, 7.5)."""
    from call1.process.reanalysis import V2_ONLY_KINDS

    store = FakeStore(cancel_taxonomy())
    consumer = _consumer(store)
    consumer.planner.signals_v2_available = lambda: False  # type: ignore[method-assign]
    consumer.poll_once()
    assert store.claimed_kinds is not None
    assert ReanalysisKind.CONTACT_SIGNALS_PREVIEW not in store.claimed_kinds
    assert set(store.claimed_kinds) == set(ReanalysisKind) - V2_ONLY_KINDS
    # an explicit kinds list still wins
    explicit = _consumer(store, kinds=[ReanalysisKind.QA])
    explicit.planner.signals_v2_available = lambda: False  # type: ignore[method-assign]
    explicit.poll_once()
    assert store.claimed_kinds == [ReanalysisKind.QA]