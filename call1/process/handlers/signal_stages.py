"""Contact Signals v2 stage runners (docs/ContactSignalsV2.md sections 3-6 and 8), shared by the fake
handlers (``handlers/fake.py``), the real handlers (``handlers/real/signals_v2.py``) and the merge's
v2 branch (``handlers/code.py``). Engines plug in through ``call1.pipeline.signals_v2``'s
``SegmentClassifier`` and ``SpanExtractor`` protocols; everything else here is the same in every
mode, so the fakes exercise the real segmentation, grounding, carry-forward and merge rules.

**Always masked** (decisions 15, 19 and 22, Q12). Every job rebuilds the call's masked turns the
same way: the number rules over the transcript plus the pinned ``pii_findings`` of this transcript
revision (``handlers/real/masking.sensitive_values``), applied with ``masking.mask``. Taxonomy text
(option glosses, names, descriptions, examples, enum values) is masked with the same values before
any engine sees it. The merge rebuilds the same masked turns and verifies every quote at its offsets.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from call1.contracts.common import CONTRACT_PARAMETERS, ArtifactRef
from call1.contracts.contents import (
    SIGNAL_NOT_OPTION,
    SIGNAL_OTHER_OPTION,
    SIGNAL_RULES_ENTRY_ID,
    ContactSignalKind,
    ContactSignalsContent,
    ContactSignalView,
    ExtractedFieldView,
    SegmentationSummary,
    SegmentScores,
    SignalCategoriesContent,
    SignalExtractionContent,
    SignalHitWhy,
    SignalRuleDecision,
    SignalSpanView,
    SignalStageOutcome,
    SignalStageProvenance,
    SignalSubcategoriesContent,
    SpanExtraction,
    SpanSubcategoryDecision,
    SpeakerRole,
    short_digest,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobStatus, JobType
from call1.contracts.signals import (
    SignalCategory,
    SignalSubcategory,
    SignalTaxonomy,
    SignalTaxonomySnapshotContent,
    category_digest,
    path_fields,
    stage1_digest,
    stage2_digest,
    stage3_digest,
    stage3_planned,
    subcategory_digest,
)
from call1.pipeline import signal_segments
from call1.pipeline.signals_v2 import (
    DEFAULT_REJECT_THRESHOLD,
    DEFAULT_STAGE2_FACTORS,
    STAGE1_TEMPLATE,
    STAGE2_TEMPLATE,
    STAGE3_TEMPLATE,
    ChoiceRow,
    EngineError,
    ExtractionSpan,
    RawExtraction,
    SegmentClassifier,
    SpanExtractor,
    build_spans,
    decide_subcategory,
    extraction_span,
    ground_fields,
    hit_id,
    merge_multi_segment,
    resolve_thresholds,
    span_text,
    sparse_probabilities,
    stage1_rows,
    stage2_option_list,
    stage2_probabilities,
    stage2_rows,
)

from ..transcripts import transcript_fingerprint
from .base import HandlerError, HandlerJob

log = logging.getLogger("call1.process.handlers.signal_stages")

STAGE_OF_JOB_TYPE = {JobType.CONTACT_SIGNALS_CATEGORIZE: "categorize", JobType.CONTACT_SIGNALS_SUBCATEGORIZE: "subcategorize",
                     JobType.CONTACT_SIGNALS_EXTRACT: "extract"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --- engine defaults (the S0-fitted scalars an engine ships with) -----------------------------


@dataclass(frozen=True)
class EngineDefaults:
    """An engine's fitted recall-first thresholds (section 3.4) and its stage-2 thresholds. Stored
    with the catalog entry beside its ``calibration_id``; a re-derive (no model) reads them by the
    previous run's entry."""

    stage1: Mapping[str, float] = field(default_factory=dict)
    stage1_default: float = 0.5
    subcategory: float = 0.5
    reject: float = DEFAULT_REJECT_THRESHOLD


FAKE_ENGINE_DEFAULTS = EngineDefaults(stage1_default=0.3, subcategory=0.5, reject=0.5)
GEMMA_ENGINE_DEFAULTS = EngineDefaults(stage1_default=0.5, subcategory=0.5, reject=0.5)
"""Gemma's constrained pick scores 0.9/0.7 when picked and 0-0.05 when not (``real.signals_v2``), so
0.5 fires exactly the picks. An admin threshold above 0.7 drops second picks; above 0.9, all."""
ENGINE_DEFAULTS: Dict[str, EngineDefaults] = {"fake-signal-classifier": FAKE_ENGINE_DEFAULTS, "call1-bundled": GEMMA_ENGINE_DEFAULTS,
                                              SIGNAL_RULES_ENTRY_ID: GEMMA_ENGINE_DEFAULTS}
"""Per classifier entry. A later "system one" engine registers its fitted values here. The rules
engine (1.4.0, ``handlers/signals_rules.py``) answers with Gemma's pick scores, so it shares Gemma's."""


def engine_defaults(entry_id: Optional[str]) -> EngineDefaults:
    return ENGINE_DEFAULTS.get(entry_id or "", EngineDefaults())


# --- the masked view every v2 job shares ------------------------------------------------------


class SignalContext:
    """One job's masked view of the call: the taxonomy snapshot, the masked turns, the ~7 s segment
    grid over them, and a ``mask`` for taxonomy text. Built the same way by every v2 job and the merge,
    so offsets agree."""

    def __init__(self, job: HandlerJob) -> None:
        from .real.convert import legacy_turns
        from .real.masking import enrichment, mask, sensitive_values

        self.job = job
        item = job.require("taxonomy")
        self.snapshot: SignalTaxonomySnapshotContent = item.content()  # type: ignore[assignment]
        self.taxonomy: SignalTaxonomy = self.snapshot.taxonomy
        self.transcript = job.transcript()
        self.transcript_ref: ArtifactRef = job.require("transcript").ref
        attribution = job.input("speaker_attribution")
        self.attribution_ref: Optional[ArtifactRef] = attribution.ref if attribution is not None else None
        # Always masked (section 11.2), whatever the route says: numeric entities computed in-process,
        # the PII model's findings from the pinned pii_findings input of this transcript revision.
        turns = legacy_turns(self.transcript, enrichment(job, compute=True))
        self.values = sensitive_values(turns, job=job)
        self._mask = mask
        self.segmentation = signal_segments.segment_call(self.transcript.turns, values=self.values, mask=self.mask)
        self.masked_turns: Dict[int, str] = {tid: t.text for tid, t in self.segmentation.turns.items()}

    def mask(self, text: Optional[str]) -> str:
        return self._mask(text or "", self.values)

    def sensitive(self, text: str) -> bool:
        """``withheld_pii`` (section 5.2): the placeholder, a call sensitive value, or a detector match."""
        from call1.pipeline.signals_v2 import default_sensitive

        if default_sensitive(text):
            return True
        folded = text.casefold()
        return any(v and v.casefold() in folded for v in self.values)

    @property
    def preview(self) -> bool:
        signals = self.job.parameters.signals
        return bool(signals and signals.preview_id)


def _category_order(taxonomy: SignalTaxonomy) -> List[str]:
    return [c.category_id for c in taxonomy.categories]


def provenance(job: HandlerJob, stage: str, engine_id: str, *, calibration_id: Optional[str], template: str, key_orders: int, device: str,
               rows: int, adapter_version: str, model_revision: Optional[str] = None) -> SignalStageProvenance:
    selection = job.selection
    return SignalStageProvenance(
        stage=stage, catalog_entry_id=engine_id, model_revision=(model_revision or (selection.model_revision if selection else "none"))[:200],
        adapter_version=adapter_version, calibration_id=calibration_id, question_template=template, key_orders=key_orders, device=device,
        route_class=selection.route.route_class.value if selection else "appliance", masked=True, rows=rows)


def classifier_template(engine: SegmentClassifier, stage: str) -> str:
    """The prompt/schema version a stage-1/2 engine records as ``question_template``. An engine with a
    prompt (Gemma) exposes a digest of it as ``stage1_template``/``stage2_template``, so a prompt or
    schema change is visible in provenance; an engine without one records the stage's name."""
    default = STAGE1_TEMPLATE if stage == "categorize" else STAGE2_TEMPLATE
    return str(getattr(engine, f"stage{1 if stage == 'categorize' else 2}_template", None) or default)


def carry_forwardable(previous: Optional[SignalStageProvenance], engine: SegmentClassifier, *, template: str, adapter_version: str) -> bool:
    """Whether a previous artifact's scores or decisions may be reused by this run: only when the
    same engine, adapter, calibration and prompt/schema version produced them, with the same on-device
    customer adapter (the ``+lora.<version>`` suffix of ``model_revision``). Otherwise an engine,
    prompt, schema or adapter change would keep stale answers as current (section 7.5)."""
    if previous is None:
        return False
    from call1.process.training.registry import lora_suffix

    # An answer from another on-device adapter (or from the base, when an adapter now answers) is
    # rerun, never kept as current (docs/OnDeviceTraining.md section 5.1).
    if lora_suffix(previous.model_revision) != lora_suffix(getattr(engine, "model_revision", None)):
        return False
    return (previous.catalog_entry_id, previous.adapter_version, previous.calibration_id, previous.question_template) == (
        engine.entry_id, adapter_version, engine.calibration_id, template)


def run_classifier(engine: SegmentClassifier, rows: Sequence[ChoiceRow], job: HandlerJob, *, torch_release: bool = False) -> List[Dict[str, float]]:
    """Load, choose, release, all inside ``inference_lock`` (section 8.5): MLX and torch MPS never
    overlap, and the model is never resident between jobs. No rows, no load."""
    if not rows:
        return []
    from call1.pipeline.inference import inference_lock

    job.check_cancelled()
    with inference_lock:
        try:
            engine.load()
            answers = engine.choose(rows)
        except EngineError as exc:
            raise HandlerError(JobErrorCode(exc.code), exc.detail or "the signal classifier failed") from None
        finally:
            engine.release()
            if torch_release:
                gc.collect()
                try:  # pragma: no cover - only with torch on MPS
                    import torch

                    if hasattr(torch, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
    if len(answers) != len(rows):
        raise HandlerError(JobErrorCode.VALIDATION_REJECTED, "the classifier did not answer every row")
    return answers


# --- stage 1 ---------------------------------------------------------------------------------


def run_categorize(job: HandlerJob, engine: Optional[SegmentClassifier], *, adapter_version: str, device: str,
                   torch_release: bool = False) -> SignalCategoriesContent:
    """``contact_signals_categorize``: run mode scores every scorable segment (carrying forward the
    previous scores of speaker scopes whose option set did not change); rederive mode rebuilds spans
    from the previous artifact's stored scores with the current thresholds and loads no model."""
    ctx = SignalContext(job)
    params = job.parameters.signals
    taxonomy = ctx.taxonomy
    order = _category_order(taxonomy)
    previous_item = job.input("previous_categories")
    previous: Optional[SignalCategoriesContent] = previous_item.content() if previous_item is not None else None  # type: ignore[assignment]
    if params is not None and params.stage1_mode == "rederive":
        if previous is None:
            raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, "a re-derive needs the previous categories artifact")
        defaults = engine_defaults(previous.provenance.catalog_entry_id)
        thresholds = resolve_thresholds(taxonomy, defaults.stage1, defaults.stage1_default)
        scores = {s.index: s.probabilities for s in previous.scores}
        spans = build_spans(previous.segments, scores, thresholds, order)
        keys = {sp.span_key for sp in spans}
        return previous.model_copy(update={
            "mode": "rederive", "taxonomy_ref": ctx.snapshot.taxonomy_ref, "thresholds": thresholds, "spans": spans,
            "stage1_digests": {scope: stage1_digest(taxonomy, SpeakerRole(scope)) for scope in previous.stage1_digests},
            "rule_decisions": [d for d in previous.rule_decisions if d.span_key in keys]})
    # Rules detection (1.4.0): the rules engine decides its categories and wraps the engine for the rest.
    from .signals_rules import maybe_rules_engine

    engine = maybe_rules_engine(job, ctx, engine, device=device)
    if engine is None:
        raise HandlerError(JobErrorCode.MODEL_UNAVAILABLE, "no signal classifier for this entry")
    defaults = engine_defaults(engine.entry_id)
    thresholds = resolve_thresholds(taxonomy, defaults.stage1, defaults.stage1_default)
    segments = ctx.segmentation.segments
    scopes_extra = str(job.parameters.extra.get("scopes") or "")
    only = [s for s in scopes_extra.split(",") if s] if scopes_extra else None
    template = classifier_template(engine, "categorize")
    carried: Dict[int, Dict[str, float]] = {}
    if (only is not None and previous is not None and _same_grid(previous, segments)
            and carry_forwardable(previous.provenance, engine, template=template, adapter_version=adapter_version)):
        keep = {s.index for s in previous.segments if s.speaker.value not in only}
        carried = {s.index: dict(s.probabilities) for s in previous.scores if s.index in keep}
    else:
        only = None
    plan = stage1_rows(segments, taxonomy, budget=engine.budget, mask=ctx.mask, only_scopes=only)
    answers = run_classifier(engine, plan.rows, job, torch_release=torch_release)
    scores: Dict[int, Dict[str, float]] = dict(carried)
    for row, answer in zip(plan.rows, answers):
        scores[int(row.key)] = sparse_probabilities(answer, [o for o, _ in row.options])
    spans = build_spans(segments, scores, thresholds, order)
    scopes = sorted({s.speaker.value for s in segments})
    content = SignalCategoriesContent(
        mode="run",
        provenance=provenance(job, "categorize", engine.entry_id, calibration_id=engine.calibration_id, template=template,
                              key_orders=engine.key_orders, device=device, rows=len(plan.rows) * engine.key_orders, adapter_version=adapter_version,
                              model_revision=getattr(engine, "model_revision", None)),
        segmenter_version=signal_segments.SEGMENTER_VERSION, window_seconds=signal_segments.WINDOW_SECONDS, taxonomy_ref=ctx.snapshot.taxonomy_ref,
        stage1_digests={scope: stage1_digest(taxonomy, SpeakerRole(scope)) for scope in scopes}, thresholds=thresholds,
        transcript=ctx.transcript_ref, speaker_attribution=ctx.attribution_ref, segments=[s.ref() for s in segments],
        scores=[SegmentScores(index=i, probabilities=p) for i, p in sorted(scores.items())], spans=spans,
        skipped_unattributed=plan.skipped_unattributed, skipped_system=ctx.segmentation.skipped_system,
        unscored_no_options=plan.unscored_no_options)
    finish = getattr(engine, "finish", None)
    return finish(content) if finish is not None else content


def _same_grid(previous: SignalCategoriesContent, segments: Sequence) -> bool:
    if previous.segmenter_version != signal_segments.SEGMENTER_VERSION or len(previous.segments) != len(segments):
        return False
    return all((a.index, a.turn_id, a.window, a.char_start, a.char_end, a.speaker) == (b.index, b.turn_id, b.window, b.char_start, b.char_end, b.speaker)
               for a, b in zip(previous.segments, segments))


# --- stage 2 ---------------------------------------------------------------------------------


def effective_decision(decision: SpanSubcategoryDecision, category: SignalCategory, defaults: EngineDefaults) -> Tuple[str, Optional[str], float]:
    """A stored stage-2 decision re-derived with the current thresholds (a ``subcategory_threshold``
    edit reruns no model, section 7.5). An error decision stays undecided."""
    if decision.status != "decided":
        return "error", None, decision.confidence
    tau = category.subcategory_threshold if category.subcategory_threshold is not None else defaults.subcategory
    return decide_subcategory(decision.probabilities, category, subcategory_threshold=tau, reject_threshold=defaults.reject)


def _subcategory(category: SignalCategory, subcategory_id: Optional[str]) -> Optional[SignalSubcategory]:
    if subcategory_id is None:
        return None
    return next((s for s in category.subcategories if s.subcategory_id == subcategory_id and s.active), None)


def rule_passthrough(decision: SignalRuleDecision, category: SignalCategory, defaults: EngineDefaults) -> SpanSubcategoryDecision:
    """A rule-decided span's stage-2 decision with no model (1.4.0): the kNN vote's subcategory (or
    Other) at Gemma's pick score, 'not' at the unpicked score, decided with the usual thresholds."""
    options = [o for o, _ in stage2_option_list(category)]
    pick = decision.subcategory_id if decision.subcategory_id in options else SIGNAL_OTHER_OPTION
    answer = {o: 0.0 for o in options}
    answer[pick] = 0.9
    answer[SIGNAL_NOT_OPTION] = 0.05
    probabilities = stage2_probabilities(answer, category)
    tau = category.subcategory_threshold if category.subcategory_threshold is not None else defaults.subcategory
    verdict, sub_id, _ = decide_subcategory(probabilities, category, subcategory_threshold=tau, reject_threshold=defaults.reject)
    return SpanSubcategoryDecision(span_key=decision.span_key, stage2_digest=stage2_digest(category), probabilities=probabilities,
                                   decision=verdict, subcategory_id=sub_id, confidence=round(1 - probabilities.get(SIGNAL_NOT_OPTION, 0.0), 6),
                                   factors=[], status="decided", source="rules")


def run_subcategorize(job: HandlerJob, engine: SegmentClassifier, *, adapter_version: str, device: str,
                      torch_release: bool = False) -> SignalSubcategoriesContent:
    """``contact_signals_subcategorize``: one decision per span. Decisions of the previous artifact
    whose ``stage2_digest`` still matches are carried forward (unless ``span_keys`` names the span).
    With no categories input (stage 1 failed) it completes at once with no decision and no model."""
    categories_item = job.input("categories")
    template = classifier_template(engine, "subcategorize")
    if categories_item is None:
        return SignalSubcategoriesContent(provenance=provenance(job, "subcategorize", engine.entry_id, calibration_id=engine.calibration_id,
                                                                template=template, key_orders=engine.key_orders, device=device, rows=0,
                                                                adapter_version=adapter_version), decisions=[])
    ctx = SignalContext(job)
    categories: SignalCategoriesContent = categories_item.content()  # type: ignore[assignment]
    params = job.parameters.signals
    wanted = set(params.span_keys) if params is not None and params.span_keys is not None else None
    previous_item = job.input("previous_subcategories")
    previous: Dict[str, SpanSubcategoryDecision] = {}
    if previous_item is not None:
        previous_content = previous_item.content()
        if carry_forwardable(previous_content.provenance, engine, template=template, adapter_version=adapter_version):  # type: ignore[attr-defined]
            previous = {d.span_key: d for d in previous_content.decisions}  # type: ignore[attr-defined]
    defaults = engine_defaults(engine.entry_id)
    decisions: Dict[str, SpanSubcategoryDecision] = {}
    carried: List[str] = []
    todo = []
    # Rules detection (1.4.0): a rule-decided span passes its kNN subcategory through with no model,
    # unless its recipe asks for a check; then today's prompt confirms or rejects it ("checked").
    rule_decisions = {d.span_key: d for d in categories.rule_decisions}
    checked: set = set()
    for span in categories.spans:
        category = ctx.taxonomy.category(span.category_id)
        if category is None or not category.active:
            continue
        ruled = rule_decisions.get(span.span_key)
        if ruled is not None and not ruled.check:
            decisions[span.span_key] = rule_passthrough(ruled, category, defaults)
            continue
        if ruled is not None:
            checked.add(span.span_key)
        old = previous.get(span.span_key)
        rerun = wanted is not None and span.span_key in wanted
        if (old is not None and not rerun and old.status == "decided" and old.stage2_digest == stage2_digest(category)
                and old.source == "engine" and old.checked == (span.span_key in checked)):
            decisions[span.span_key] = old
            carried.append(span.span_key)
            continue
        item = span_text(span, categories.segments, ctx.masked_turns)
        if item is None:
            continue
        todo.append(item)
    rows = stage2_rows(todo, ctx.taxonomy, categories.segments, ctx.masked_turns, budget=engine.budget, mask=ctx.mask)
    answers = run_classifier(engine, rows, job, torch_release=torch_release)
    for row, answer in zip(rows, answers):
        category = ctx.taxonomy.category(row.key.rsplit(".t", 1)[0])
        probabilities = stage2_probabilities(answer, category)  # type: ignore[arg-type]
        tau = category.subcategory_threshold if category.subcategory_threshold is not None else defaults.subcategory  # type: ignore[union-attr]
        decision, sub_id, confidence = decide_subcategory(probabilities, category, subcategory_threshold=tau, reject_threshold=defaults.reject)  # type: ignore[arg-type]
        decisions[row.key] = SpanSubcategoryDecision(span_key=row.key, stage2_digest=stage2_digest(category), probabilities=probabilities,  # type: ignore[arg-type]
                                                     decision=decision, subcategory_id=sub_id, confidence=round(1 - probabilities.get(SIGNAL_NOT_OPTION, 0.0), 6),
                                                     factors=list(DEFAULT_STAGE2_FACTORS), truncated=row.truncated, status="decided",
                                                     checked=row.key in checked)
    ordered = [decisions[s.span_key] for s in categories.spans if s.span_key in decisions]
    return SignalSubcategoriesContent(
        provenance=provenance(job, "subcategorize", engine.entry_id, calibration_id=engine.calibration_id, template=template,
                              key_orders=engine.key_orders, device=device, rows=len(rows) * engine.key_orders, adapter_version=adapter_version,
                              model_revision=getattr(engine, "model_revision", None)),
        decisions=ordered, carried_forward=[k for k in carried if k in {d.span_key for d in ordered}])


# --- stage 3 ---------------------------------------------------------------------------------


@dataclass
class ExtractEngine:
    """One stage-3 engine as a job runs it: the extractor and its provenance fields."""

    extractor: SpanExtractor
    calibration_id: Optional[str] = None
    device: str = "mps"
    adapter_version: str = "1"
    model_revision: Optional[str] = None


@dataclass
class ExtractionTarget:
    span_key: str
    category: SignalCategory
    subcategory: Optional[SignalSubcategory]
    span: ExtractionSpan


def extraction_targets(ctx: SignalContext, categories: SignalCategoriesContent, subcategories: Optional[SignalSubcategoriesContent],
                       defaults: EngineDefaults) -> Tuple[List[ExtractionTarget], List[str], int]:
    """The spans stage 3 runs on (section 5.1): not rejected, whose node has fields or narrow_quote,
    in call order, at most ``max_extraction_spans_per_call``. Returns (targets, spans missing their
    stage-2 decision, spans over the cap)."""
    decisions = {d.span_key: d for d in subcategories.decisions} if subcategories is not None else {}
    targets: List[ExtractionTarget] = []
    upstream_missing: List[str] = []
    for span in categories.spans:
        category = ctx.taxonomy.category(span.category_id)
        if category is None or not category.active:
            continue
        if subcategories is None:
            upstream_missing.append(span.span_key)
            continue
        decision = decisions.get(span.span_key)
        if decision is None:
            upstream_missing.append(span.span_key)
            continue
        verdict, sub_id, _ = effective_decision(decision, category, defaults)
        if verdict == "rejected":
            continue
        sub = _subcategory(category, sub_id) if verdict == "subcategory" else None
        if not stage3_planned(category, sub):
            continue
        item = span_text(span, categories.segments, ctx.masked_turns)
        if item is None:
            continue
        targets.append(ExtractionTarget(span.span_key, category, sub, extraction_span(item, category, sub, ctx.masked_turns, ctx.mask)))
    cap = CONTRACT_PARAMETERS.max_extraction_spans_per_call
    over = len(targets) - cap if len(targets) > cap else 0
    return targets[:cap], upstream_missing, over


def _extract(engine: SpanExtractor, spans: Sequence[ExtractionSpan]) -> Dict[str, RawExtraction]:
    if not spans:
        return {}
    try:
        answers = engine.extract(spans)
    except EngineError as exc:
        return {s.span_key: RawExtraction(span_key=s.span_key, status="error", error_code=exc.code) for s in spans}
    except HandlerError as exc:
        return {s.span_key: RawExtraction(span_key=s.span_key, status="error", error_code=exc.code.value) for s in spans}
    by_key = {a.span_key: a for a in answers}
    return {s.span_key: by_key.get(s.span_key) or RawExtraction(span_key=s.span_key, status="error", error_code="validation_rejected")
            for s in spans}


def _error_code(value: Optional[str]) -> JobErrorCode:
    try:
        return JobErrorCode(value or "validation_rejected")
    except ValueError:
        return JobErrorCode.PROVIDER_ERROR


def run_extract(job: HandlerJob, primary: ExtractEngine, fallback: Optional[Callable[[], Optional[ExtractEngine]]] = None, *,
                defaults: Optional[EngineDefaults] = None) -> SignalExtractionContent:
    """``contact_signals_extract`` (section 5): the path's fields on each span, grounded in the core
    span. A previous extraction whose ``stage3_digest`` still matches is carried forward. Spans that
    fail on the primary (crash, invalid output, over budget) rerun on the declared in-job fallback
    entry, after the primary is released, with ``source: fallback`` (section 5.7). The caller holds
    ``inference_lock`` around this for GPU engines."""
    categories_item = job.input("categories")
    subcategories_item = job.input("subcategories")
    prov = lambda engine, rows: provenance(job, "extract", engine.extractor.entry_id, calibration_id=engine.calibration_id,  # noqa: E731
                                           template=STAGE3_TEMPLATE, key_orders=1, device=engine.device, rows=rows,
                                           adapter_version=engine.adapter_version, model_revision=engine.model_revision)
    if categories_item is None:
        return SignalExtractionContent(provenance=prov(primary, 0), spans=[])
    ctx = SignalContext(job)
    categories: SignalCategoriesContent = categories_item.content()  # type: ignore[assignment]
    subcategories: Optional[SignalSubcategoriesContent] = subcategories_item.content() if subcategories_item is not None else None  # type: ignore[assignment]
    sub_entry = subcategories.provenance.catalog_entry_id if subcategories is not None else None
    defaults = defaults or engine_defaults(sub_entry)
    targets, upstream_missing, _over = extraction_targets(ctx, categories, subcategories, defaults)
    params = job.parameters.signals
    wanted = set(params.span_keys) if params is not None and params.span_keys is not None else None
    previous_item = job.input("previous_extraction")
    previous = {s.span_key: s for s in previous_item.content().spans} if previous_item is not None else {}  # type: ignore[attr-defined]
    results: Dict[str, SpanExtraction] = {}
    carried: List[str] = []
    todo: List[ExtractionTarget] = []
    for target in targets:
        old = previous.get(target.span_key)
        rerun = wanted is not None and target.span_key in wanted
        if old is not None and not rerun and old.status == "extracted" and old.stage3_digest == target.span.stage3_digest:
            results[target.span_key] = old
            carried.append(target.span_key)
        else:
            todo.append(target)
    job.check_cancelled()
    raw = _extract(primary.extractor, [t.span for t in todo])
    primary_rows = len(todo)
    try:
        primary.extractor.release()
    except Exception:  # pragma: no cover - a release failure never loses the answers
        log.warning("stage-3 primary engine release failed")
    failed = [t for t in todo if raw[t.span_key].status != "ok"]
    fallback_engine = fallback() if (failed and fallback is not None) else None
    sources: Dict[str, str] = {}
    fallback_rows = 0
    if fallback_engine is not None:
        job.check_cancelled()
        again = _extract(fallback_engine.extractor, [t.span for t in failed])
        fallback_rows = len(failed)
        try:
            fallback_engine.extractor.release()
        except Exception:  # pragma: no cover
            log.warning("stage-3 fallback engine release failed")
        for t in failed:
            raw[t.span_key] = again[t.span_key]
            sources[t.span_key] = "fallback"
    for target in todo:
        answer = raw[target.span_key]
        source = sources.get(target.span_key, "primary")
        engine = fallback_engine if source == "fallback" and fallback_engine is not None else primary
        if answer.status == "ok":
            fields, narrowed = ground_fields(target.span, answer, path_fields(target.category, target.subcategory),
                                             requires_evidence=engine.extractor.requires_evidence, sensitive=ctx.sensitive)
            confidence = answer.engine_confidence
            results[target.span_key] = SpanExtraction(span_key=target.span_key, stage3_digest=target.span.stage3_digest, status="extracted",
                                                      fields=fields, narrowed_quote=narrowed, source=source,
                                                      engine_confidence=min(1.0, max(0.0, confidence)) if confidence is not None else None)
        elif answer.status == "over_budget":
            results[target.span_key] = SpanExtraction(span_key=target.span_key, stage3_digest=target.span.stage3_digest, status="over_budget",
                                                      fields=[], source=source)
        else:
            results[target.span_key] = SpanExtraction(span_key=target.span_key, stage3_digest=target.span.stage3_digest, status="error",
                                                      error_code=_error_code(answer.error_code), fields=[], source=source)
    spans = [results[t.span_key] for t in targets if t.span_key in results]
    for key in upstream_missing:
        span = next(s for s in categories.spans if s.span_key == key)
        category = ctx.taxonomy.category(span.category_id)
        spans.append(SpanExtraction(span_key=key, stage3_digest=stage3_digest(category), status="upstream_missing", fields=[]))  # type: ignore[arg-type]
    used_fallback = any(s.source == "fallback" for s in spans)
    return SignalExtractionContent(provenance=prov(primary, primary_rows),
                                   fallback_provenance=prov(fallback_engine, fallback_rows) if used_fallback and fallback_engine is not None else None,
                                   spans=spans, carried_forward=carried)


# --- the merge's v2 branch (section 6, 7.3) ---------------------------------------------------


STAGE_LABEL = {"subcategorize": "Subcategories", "extract": "Fields"}


def _upstream_failures(job: HandlerJob) -> Dict[str, Optional[JobErrorCode]]:
    out: Dict[str, Optional[JobErrorCode]] = {}
    for outcome in job.upstream:
        stage = STAGE_OF_JOB_TYPE.get(outcome.job_type)
        if stage is None:
            continue
        out[stage] = outcome.error_code or (JobErrorCode.CANCELLED if outcome.status is JobStatus.CANCELLED else None)
    return out


def merge_v2(job: HandlerJob) -> Tuple[ContactSignalsContent, Optional[str]]:
    """The v2 contact-signals result. Every quote is re-verified against the masked turn the stages
    saw (rebuilt from the pinned ``pii_findings``), at its recorded offsets, with no lower-casing;
    whatever fails is dropped and counted. A missing categorize output fails the merge
    (``input_unavailable``); a missing subcategorize or extract output, or spans past the extraction
    cap, make it partial, naming the stage. Verified hits then merge into multi-segment signals
    (decision 25, ``signals_v2.merge_multi_segment``); counts in ``partial_reason`` stay per span."""
    categories_item = job.input("stage:categorize")
    if categories_item is None:
        raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, "the categorize stage did not reach the merge")
    ctx = SignalContext(job)
    categories: SignalCategoriesContent = categories_item.content()  # type: ignore[assignment]
    sub_item = job.input("stage:subcategorize")
    ext_item = job.input("stage:extract")
    subcategories: Optional[SignalSubcategoriesContent] = sub_item.content() if sub_item is not None else None  # type: ignore[assignment]
    extraction: Optional[SignalExtractionContent] = ext_item.content() if ext_item is not None else None  # type: ignore[assignment]
    extra = job.parameters.extra
    planned = [s for s in str(extra.get("planned_stages") or "categorize,subcategorize").split(",") if s]
    pinned = {s for s in str(extra.get("pinned_stages") or "").split(",") if s}
    failures = _upstream_failures(job)
    taxonomy = ctx.taxonomy
    defaults = engine_defaults(subcategories.provenance.catalog_entry_id if subcategories is not None else None)
    decisions = {d.span_key: d for d in subcategories.decisions} if subcategories is not None else {}
    extractions = {s.span_key: s for s in extraction.spans} if extraction is not None else {}
    checksum = ctx.transcript_ref.checksum
    same_revision = categories.transcript.checksum == checksum
    rules_ran = categories.rules is not None
    rule_decisions = {d.span_key: d for d in categories.rule_decisions}
    hits: List[ContactSignalView] = []
    dropped = 0
    stage2_errors = 0
    fields_missing = 0
    over_cap = 0
    stage3_seen = 0
    cap = CONTRACT_PARAMETERS.max_extraction_spans_per_call
    for span in categories.spans:
        category = taxonomy.category(span.category_id)
        if category is None or not category.active:
            continue
        item = span_text(span, categories.segments, ctx.masked_turns) if same_revision else None
        if item is None:
            dropped += 1
            continue
        decision = decisions.get(span.span_key)
        verdict, sub_id, confidence = ("error", None, span.peak_probability)
        if decision is not None:
            verdict, sub_id, confidence = effective_decision(decision, category, defaults)
        if verdict == "rejected":
            continue
        if subcategories is not None and verdict == "error":
            stage2_errors += 1
        if verdict == "error":
            confidence = span.peak_probability
        sub = _subcategory(category, sub_id) if verdict == "subcategory" else None
        quote, cs, ce, start, end, narrowed_flag = item.text, item.char_start, item.char_end, item.start, item.end, False
        fields: List[ExtractedFieldView] = []
        if subcategories is not None and verdict != "error" and stage3_planned(category, sub) and "extract" in planned:
            stage3_seen += 1
            found = extractions.get(span.span_key)
            if found is None and extraction is not None and stage3_seen > cap:
                over_cap += 1
            elif found is not None and found.status == "extracted" and found.stage3_digest == stage3_digest(category, sub):
                turn_text = ctx.masked_turns.get(span.turn_id, "")
                if found.narrowed_quote is not None:
                    q = found.narrowed_quote
                    if not (item.char_start <= q.char_start and q.char_end <= item.char_end and turn_text[q.char_start:q.char_end] == q.text):
                        dropped += 1
                        continue
                    quote, cs, ce, narrowed_flag = q.text, q.char_start, q.char_end, True
                    start = signal_segments.time_at(item.core, span.turn_id, cs) or item.start
                    end = max(start, signal_segments.time_at(item.core, span.turn_id, ce) or item.end)
                names = {f.field_id: f.name for f in path_fields(category, sub)}
                for f in found.fields:
                    text = f.surface if f.surface is not None else f.evidence
                    if text is not None and turn_text[f.char_start:f.char_end] != text:  # type: ignore[index]
                        continue
                    if f.field_id in names:
                        fields.append(ExtractedFieldView(**f.model_dump(), name=names[f.field_id]))
            elif extraction is not None:
                fields_missing += 1
        custom = not category.builtin
        hits.append(ContactSignalView(
            id=hit_id(category, checksum, span.turn_id, span.block, preview=ctx.preview), kind=ContactSignalKind.CUSTOM if custom else ContactSignalKind(category.category_id),
            label=category.name, start=round(start, 3), end=round(max(start, end), 3), speaker=item.speaker, quote=quote, turn_id=span.turn_id,
            char_start=cs, char_end=ce, confidence=min(1.0, max(0.0, confidence)), category_id=category.category_id,
            category_digest=short_digest(category_digest(category)),
            category_confidence=span.peak_probability, subcategory_id=sub.subcategory_id if sub else ("other" if verdict == "other" and category.subcategories else None),
            subcategory_label=sub.name if sub else None, subcategory_digest=short_digest(subcategory_digest(sub)) if sub else None,
            subcategory_confidence=round(decision.probabilities.get(sub.subcategory_id, 0.0), 6) if (sub and decision) else None,
            span=SignalSpanView(block=span.block, first_window=span.first_window, last_window=span.last_window, timing=item.timing,  # type: ignore[arg-type]
                                context_start=_seg_time(categories, span.context_first, "start"), context_end=_seg_time(categories, span.context_last, "end")),
            fields=fields, quote_narrowed=narrowed_flag,
            why=_why(rule_decisions.get(span.span_key), decision, verdict) if rules_ran else None))
    # Decision 25 (section 6.5): one speaker's consecutive hits with the same category and stage-2
    # outcome are one signal that crossed several segments. Merge them; the first keeps its hit ID.
    turns = [(tid, t.speaker) for tid, t in ctx.segmentation.turns.items() if t.text.strip()]
    hits = merge_multi_segment(hits, turns)
    reasons: List[str] = []
    stages: List[SignalStageOutcome] = []
    for stage in planned:
        content = {"categorize": categories, "subcategorize": subcategories, "extract": extraction}.get(stage)
        included = content is not None
        carried = 0
        if content is not None:
            if stage in pinned:
                carried = len(getattr(content, "spans", None) or getattr(content, "decisions", []))
            else:
                carried = len(getattr(content, "carried_forward", []) or [])
        code = None if included else (failures.get(stage) or JobErrorCode.INPUT_UNAVAILABLE)
        stages.append(SignalStageOutcome(stage=stage, included=included, failure_code=code,
                                         provenance=content.provenance if content is not None else None, spans_carried_forward=carried))
        if not included:
            reasons.append(f"{STAGE_LABEL.get(stage, stage)} unavailable ({code.value})")  # type: ignore[union-attr]
    if stage2_errors:
        reasons.append(f"Subcategories unavailable for {stage2_errors} spans (engine error)")
    if fields_missing:
        reasons.append(f"Fields unavailable for {fields_missing} spans")
    if over_cap:
        reasons.append(f"extraction_cap: {over_cap} spans over the extraction cap")
    if dropped:
        log.info("contact signals merge: %d spans failed quote re-verification and were dropped", dropped)
    partial_reason = "; ".join(reasons)[:200] if reasons else None
    scored = len(categories.scores)
    summary = SegmentationSummary(segmenter_version=categories.segmenter_version, window_seconds=categories.window_seconds,
                                  segments=len(categories.segments), scored_segments=min(scored, len(categories.segments)),
                                  skipped_unattributed=categories.skipped_unattributed, skipped_system=categories.skipped_system,
                                  interpolated_turns=len({s.turn_id for s in categories.segments if s.timing == "interpolated"}))
    content = ContactSignalsContent(
        completeness="partial" if partial_reason else "complete", partial_reason=partial_reason, signals=hits, passes=[],
        transcript_fingerprint=transcript_fingerprint(ctx.transcript), generated_at=utcnow(), pipeline="v2",
        taxonomy=ctx.snapshot.taxonomy_ref, stages=stages, segmentation=summary, stage1_digests=dict(categories.stage1_digests))
    return content, partial_reason


def _why(ruled: Optional[SignalRuleDecision], decision: Optional[SpanSubcategoryDecision], verdict: str) -> SignalHitWhy:
    """A hit's provenance in a result where the rules engine ran (1.4.0, section 2.5)."""
    sub_source = None
    if decision is not None and verdict != "error":
        sub_source = "rules" if decision.source == "rules" else "gemma"
    return SignalHitWhy(category_source="rules" if ruled is not None else "gemma", subcategory_source=sub_source,
                        check="confirmed" if ruled is not None and decision is not None and decision.checked and verdict != "error" else None,
                        rule=ruled)


def _seg_time(categories: SignalCategoriesContent, index: int, which: str) -> float:
    seg = next((s for s in categories.segments if s.index == index), None)
    return float(getattr(seg, which)) if seg is not None else 0.0


__all__ = ["ENGINE_DEFAULTS", "EngineDefaults", "ExtractEngine", "FAKE_ENGINE_DEFAULTS", "SignalContext", "engine_defaults", "merge_v2",
           "rule_passthrough", "run_categorize", "run_classifier", "run_extract", "run_subcategorize"]
