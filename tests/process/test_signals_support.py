"""Shared builders for the Contact Signals v2 Process tests (``test_signals_*.py``): claimed jobs
with in-memory input artifacts (no Store), fake-script transcripts with their stub PII findings,
taxonomy snapshots, and a runner that takes one call through categorize, subcategorize, extract and
the merge exactly as the worker would hand the artifacts along. No tests live here."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.catalog import ModelPurpose
from call1.contracts.common import canonical_json
from call1.contracts.contents import (
    ContactSignalsContent,
    PiiFindingsContent,
    SignalCategoriesContent,
    SignalExtractionContent,
    SignalSubcategoriesContent,
    SignalTaxonomyRef,
    SpeakerAssignment,
    SpeakerAttributionContent,
    SpeakerRole,
    TranscriptContent,
    TranscriptTurnContent,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import ClaimedJob, EdgeKind, Job, JobParameters, JobStatus, JobType, SignalJobParameters, UpstreamOutcome
from call1.contracts.signals import (
    SignalCategory,
    SignalField,
    SignalSettings,
    SignalSubcategory,
    SignalTaxonomy,
    SignalTaxonomySnapshotContent,
    builtin_signal_taxonomy,
    taxonomy_digest,
)
from call1.process.catalog import seeded_catalog
from call1.process.handlers import build_registry
from call1.process.handlers.base import HandlerJob, HandlerResult, InputArtifact
from call1.process.handlers.fake import SCRIPT, FakeBehavior, FakeEnrichment

CATALOG = seeded_catalog(mode="fake")

STAGE_PURPOSE = {JobType.CONTACT_SIGNALS_CATEGORIZE: ModelPurpose.SIGNAL_CATEGORY,
                 JobType.CONTACT_SIGNALS_SUBCATEGORIZE: ModelPurpose.SIGNAL_SUBCATEGORY,
                 JobType.CONTACT_SIGNALS_EXTRACT: ModelPurpose.SIGNAL_EXTRACTION}
STAGE_ENTRY = {JobType.CONTACT_SIGNALS_CATEGORIZE: "fake-signal-classifier", JobType.CONTACT_SIGNALS_SUBCATEGORIZE: "fake-signal-classifier",
               JobType.CONTACT_SIGNALS_EXTRACT: "fake-signal-extractor"}


def artifact(role: str, kind: ArtifactKind, content: Any) -> Tuple[Artifact, bytes]:
    data = canonical_json(content) if not isinstance(content, (bytes, bytearray)) else bytes(content)
    record = Artifact.model_construct(id=f"art_{role.replace(':', '_')}_{hashlib.sha256(data).hexdigest()[:8]}", kind=kind, slot="",
                                      content_type="application/json", size_bytes=len(data), checksum="sha256:" + hashlib.sha256(data).hexdigest(),
                                      content_contract="", sensitivity="derived", conversation_id="conv_1", linked=True, version=1, labels={})
    return record, data


def make_job(tmp_path: Path, job_type: JobType, inputs: Dict[str, Tuple[ArtifactKind, Any]], *, parameters: Optional[JobParameters] = None,
             entry_id: Optional[str] = "default", purpose: Optional[ModelPurpose] = None, masked: bool = True,
             upstream: Sequence[UpstreamOutcome] = (), catalog=CATALOG, route=None) -> HandlerJob:
    if entry_id == "default":
        entry_id = STAGE_ENTRY.get(job_type)
    entry = catalog.get(entry_id) if entry_id else None
    selection = None
    if entry is not None:
        selection = entry.selection(purpose or STAGE_PURPOSE.get(job_type) or entry.purposes[0], masked=masked)
        if route is not None:
            selection = selection.model_copy(update={"route": selection.route.model_copy(update={"route_class": route})})
    job = Job.model_construct(id=f"job_{job_type.value}", conversation_id="conv_1", graph_id="graph_1", job_type=job_type,
                              parameters=parameters or JobParameters(), selection=selection)
    claimed = ClaimedJob.model_construct(job=job, attempt_number=1, final_attempt=False, upstream=list(upstream), inputs=[])
    scratch = tmp_path / f"scratch-{job_type.value}-{len(list(tmp_path.iterdir())) if tmp_path.exists() else 0}"
    scratch.mkdir(parents=True, exist_ok=True)
    resolved: Dict[str, Optional[InputArtifact]] = {}
    for role, (kind, content) in inputs.items():
        if content is None:
            resolved[role] = None
            continue
        record, data = artifact(role, kind, content)

        def fetch(_artifact, dest, data=data):
            if dest is None:
                return data
            dest.write_bytes(data)
            return dest

        resolved[role] = InputArtifact(role, record, fetch, scratch)
    return HandlerJob(claimed, resolved, scratch, catalog_entry=entry)


def script_transcript(script: Sequence[Tuple[SpeakerRole, str]] = SCRIPT, *, seconds: float = 4.0, stereo: bool = True,
                      words: bool = False) -> TranscriptContent:
    turns = []
    for i, (role, text) in enumerate(script):
        start, end = round(i * seconds, 3), round((i + 1) * seconds - 0.2, 3)
        turns.append(TranscriptTurnContent(turn_id=i, speaker=role if stereo else SpeakerRole.UNKNOWN, start_time=start, end_time=end, text=text,
                                           channel=(0 if role is SpeakerRole.AGENT else 1) if stereo else None, confidence=0.9))
    return TranscriptContent(duration_seconds=round(len(script) * seconds, 3), language="en", is_redacted=False, turns=turns)


def attribution_for(script: Sequence[Tuple[SpeakerRole, str]]) -> SpeakerAttributionContent:
    return SpeakerAttributionContent(method="diarization", assignments=[SpeakerAssignment(turn_id=i, speaker=role) for i, (role, _) in enumerate(script)])


def findings_for(tmp_path: Path, transcript: TranscriptContent, attribution: Optional[SpeakerAttributionContent] = None) -> PiiFindingsContent:
    """The stub PII findings the fake enrichment job writes for this exact transcript artifact."""
    inputs: Dict[str, Tuple[ArtifactKind, Any]] = {"transcript": (ArtifactKind.TRANSCRIPT, transcript)}
    if attribution is not None:
        inputs["speaker_attribution"] = (ArtifactKind.SPEAKER_ATTRIBUTION, attribution)
    job = make_job(tmp_path, JobType.ENRICHMENT, inputs, entry_id=None)
    return FakeEnrichment(FakeBehavior()).run(job).outputs["pii_findings"].content  # type: ignore[return-value]


def snapshot(taxonomy: Optional[SignalTaxonomy] = None, settings: Optional[SignalSettings] = None, *, version: Optional[int] = 1,
             preview_id: Optional[str] = None) -> SignalTaxonomySnapshotContent:
    taxonomy = taxonomy or builtin_signal_taxonomy()
    return SignalTaxonomySnapshotContent(source="preview" if preview_id else "published",
                                         taxonomy_ref=SignalTaxonomyRef(version=None if preview_id else version, digest=taxonomy_digest(taxonomy)),
                                         taxonomy=taxonomy, settings=settings or SignalSettings(pipeline="v2"), preview_id=preview_id)


REASON = SignalField(field_id="reason", name="Reason", type="enum", description="Why the caller wants to cancel",
                     enum_values=["price", "service", "moving", "other"], pii_class="none")
PLAN_PRICE = SignalField(field_id="new_price", name="Offered price", type="amount", description="The price the agent offered", pii_class="amount")


def with_categories(extra: Sequence[SignalCategory] = (), **builtin_edits: Dict[str, Any]) -> SignalTaxonomy:
    """The built-ins, edited per category ID (``intent={"subcategories": [...]}``), plus custom categories."""
    base = builtin_signal_taxonomy()
    cats = [c.model_copy(update=builtin_edits.get(c.category_id, {})) for c in base.categories]
    return SignalTaxonomy.model_validate({"categories": [c.model_dump() for c in cats] + [c.model_dump() for c in extra]})


def cancel_taxonomy(*, narrow: bool = True, fields: bool = True) -> SignalTaxonomy:
    """intent > "Cancel account" (examples "cancel my account"), with a ``reason`` enum and quote narrowing."""
    cancel = SignalSubcategory(subcategory_id="cancel_account", name="Cancel account", gloss="Caller wants to cancel their account",
                               examples=["cancel my account", "cancel it"], fields=[REASON] if fields else [], narrow_quote=narrow)
    question = SignalSubcategory(subcategory_id="fee_question", name="Fee question", gloss="Caller asks about a fee", examples=["question about a fee"])
    return with_categories(intent={"subcategories": [cancel, question]})


def custom_category(category_id: str = "competitor", *, name: str = "Mentions a competitor", gloss: str = "Someone names another provider",
                    speaker: Optional[str] = None, examples: Sequence[str] = ("competitor",), **kw) -> SignalCategory:
    return SignalCategory(category_id=category_id, builtin=False, name=name, gloss=gloss, speaker=speaker, examples=list(examples), **kw)


# --- one call through the cascade --------------------------------------------------------------


class Run:
    """The artifacts of one v2 run, and the jobs that made them."""

    def __init__(self) -> None:
        self.categories: Optional[SignalCategoriesContent] = None
        self.subcategories: Optional[SignalSubcategoriesContent] = None
        self.extraction: Optional[SignalExtractionContent] = None
        self.result: Optional[ContactSignalsContent] = None
        self.merge_result: Optional[HandlerResult] = None
        self.jobs: Dict[str, HandlerJob] = {}
        self.results: Dict[str, HandlerResult] = {}


def signal_params(snap: SignalTaxonomySnapshotContent, **kw) -> JobParameters:
    extra = kw.pop("extra", {})
    return JobParameters(signals=SignalJobParameters(taxonomy_digest=snap.taxonomy_ref.digest, **kw), extra=extra)


def run_v2(tmp_path: Path, script: Sequence[Tuple[SpeakerRole, str]] = SCRIPT, taxonomy: Optional[SignalTaxonomy] = None, *,
           settings: Optional[SignalSettings] = None, behavior: Optional[FakeBehavior] = None, stereo: bool = True,
           transcript: Optional[TranscriptContent] = None, extract: Optional[bool] = None, fallback_entry_id: Optional[str] = None,
           skip: Sequence[str] = (), preview_id: Optional[str] = None, registry=None) -> Run:
    """Categorize, subcategorize, extract (when planned) and merge one call on the fake handlers.
    ``skip`` leaves a stage's output missing downstream, as a failed upstream on an ``after`` edge does."""
    from call1.pipeline.signals_v2 import taxonomy_extract_planned

    registry = registry or build_registry("fake", fake_behavior=behavior or FakeBehavior())
    transcript = transcript or script_transcript(script, stereo=stereo)
    attribution = None if stereo else attribution_for(script)
    findings = findings_for(tmp_path, transcript, attribution)
    snap = snapshot(taxonomy, settings, preview_id=preview_id)
    extract = taxonomy_extract_planned(snap.taxonomy) if extract is None else extract
    base: Dict[str, Tuple[ArtifactKind, Any]] = {"transcript": (ArtifactKind.TRANSCRIPT, transcript), "pii_findings": (ArtifactKind.PII_FINDINGS, findings),
                                                 "taxonomy": (ArtifactKind.SIGNAL_TAXONOMY_SNAPSHOT, snap)}
    if attribution is not None:
        base["speaker_attribution"] = (ArtifactKind.SPEAKER_ATTRIBUTION, attribution)
    run = Run()
    upstream: List[UpstreamOutcome] = []

    def go(job_type: JobType, name: str, role: str, inputs: Dict[str, Tuple[ArtifactKind, Any]], params: JobParameters):
        job = make_job(tmp_path, job_type, {**base, **inputs}, parameters=params)
        run.jobs[name] = job
        if name in skip:
            upstream.append(UpstreamOutcome.model_construct(job_id=f"job_{name}", job_type=job_type, edge=EdgeKind.AFTER,
                                                            status=JobStatus.FAILED, error_code=JobErrorCode.PROVIDER_ERROR))
            return None
        result = registry.get(job_type).run(job)
        run.results[name] = result
        return result.outputs[role].content

    run.categories = go(JobType.CONTACT_SIGNALS_CATEGORIZE, "categorize", "categories", {}, signal_params(snap, preview_id=preview_id))
    run.subcategories = go(JobType.CONTACT_SIGNALS_SUBCATEGORIZE, "subcategorize", "subcategories",
                           {"categories": (ArtifactKind.SIGNAL_CATEGORIES, run.categories)}, signal_params(snap, preview_id=preview_id))
    planned = ["categorize", "subcategorize"]
    if extract:
        planned.append("extract")
        run.extraction = go(JobType.CONTACT_SIGNALS_EXTRACT, "extract", "extraction",
                            {"categories": (ArtifactKind.SIGNAL_CATEGORIES, run.categories),
                             "subcategories": (ArtifactKind.SIGNAL_SUBCATEGORIES, run.subcategories)},
                            signal_params(snap, fallback_entry_id=fallback_entry_id, preview_id=preview_id))
    merge_inputs: Dict[str, Tuple[ArtifactKind, Any]] = {"stage:categorize": (ArtifactKind.SIGNAL_CATEGORIES, run.categories),
                                                         "stage:subcategorize": (ArtifactKind.SIGNAL_SUBCATEGORIES, run.subcategories)}
    if extract:
        merge_inputs["stage:extract"] = (ArtifactKind.SIGNAL_EXTRACTION, run.extraction)
    merge = make_job(tmp_path, JobType.CONTACT_SIGNALS_MERGE, {**base, **merge_inputs}, entry_id=None, upstream=upstream,
                     parameters=signal_params(snap, preview_id=preview_id, extra={"planned_stages": ",".join(planned)}))
    run.jobs["merge"] = merge
    run.merge_result = registry.get(JobType.CONTACT_SIGNALS_MERGE).run(merge)
    run.result = run.merge_result.outputs["contact_signals"].content  # type: ignore[assignment]
    return run


__all__ = ["CATALOG", "REASON", "PLAN_PRICE", "Run", "artifact", "attribution_for", "cancel_taxonomy", "custom_category", "findings_for",
           "make_job", "run_v2", "script_transcript", "signal_params", "snapshot", "with_categories"]
