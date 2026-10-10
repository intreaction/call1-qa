"""Code handlers: deterministic stages that run the same in every handler mode. (``embeddings`` is a
model stage since contract 1.2.0: ``handlers/embeddings.py``.)

* ``qa_scorecard``: assembles the per-criterion assessments (and escalations) and the deterministic
  verdicts into the scorecard with the pre-split ``RubricEvaluator`` scoring rules.
* ``summary_assembly``: the final summary from the final synthesis (or the single segment), with
  citations checked against the original turn IDs.
* ``contact_signals_merge``: quote-verified, de-duplicated signals from the passes that succeeded;
  partial, and saying which pass is missing, when one did not. Its v2 branch (contract 1.3.0,
  ``stage:*`` inputs, ``parameters.signals`` set) builds hits from the Contact Signals v2 stage
  artifacts (``handlers/signal_stages.merge_v2``), re-verifying every quote against the masked turn.
  The two branches never mix in one graph.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from call1.contracts.contents import (
    ContactSignalPass,
    ContactSignalsContent,
    ContactSignalsPassContent,
    ContactSignalsPassOutcome,
    ContactSignalView,
    QaAssessmentContent,
    QaScorecardContent,
    QaVerdictContent,
    SpeakerRole,
    SummaryCitation,
    SummaryContent,
    SummarySegmentContent,
    SummarySynthesisContent,
    VerdictStatus,
    VerdictView,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobStatus, JobType
from call1.contracts.rubrics import CheckType, RubricCriterion

from ..transcripts import transcript_fingerprint
from .base import Handler, HandlerError, HandlerJob, HandlerResult, Output

LOW_CONFIDENCE = 0.70
DISAGREEMENT = ("Primary and escalation assessments differ or escalation is unresolved. Human review is required.")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _verdict_from_assessment(criterion: RubricCriterion, assessment: QaAssessmentContent) -> VerdictView:
    return VerdictView(
        criterion_id=criterion.criterion_id, criterion_name=criterion.name, status=assessment.status, confidence=assessment.confidence,
        quoted_evidence=assessment.quoted_evidence, speaker=assessment.speaker, timestamp_range=assessment.timestamp_range,
        quote_turn_id=assessment.quote_turn_id, reasoning=assessment.reasoning, hallucination_detected=assessment.hallucination_detected,
        model_attempts=[assessment.attempt],
    )


def semantic_verdict(criterion: RubricCriterion, primary: Optional[QaAssessmentContent], escalation: Optional[QaAssessmentContent]) -> VerdictView:
    """The pre-split ``QuestionRouter`` outcome: the last assessment decides, except that an
    escalation disagreeing with a grounded (non-FLAGGED) primary is FLAGGED for a human."""
    if primary is None:
        return VerdictView(criterion_id=criterion.criterion_id, criterion_name=criterion.name, status=VerdictStatus.FLAGGED, confidence=0,
                           speaker=criterion.check.speaker or SpeakerRole.AGENT, reasoning="No assessment was produced for this criterion.")
    if escalation is None:
        return _verdict_from_assessment(criterion, primary)
    verdict = _verdict_from_assessment(criterion, escalation)
    attempts = [primary.attempt, escalation.attempt]
    if primary.status is not VerdictStatus.FLAGGED and primary.status is not escalation.status:
        return VerdictView(criterion_id=criterion.criterion_id, criterion_name=criterion.name, status=VerdictStatus.FLAGGED, confidence=0,
                           speaker=criterion.check.speaker or SpeakerRole.AGENT, reasoning=DISAGREEMENT, model_attempts=attempts)
    return verdict.model_copy(update={"model_attempts": attempts})


def score(criteria: List[RubricCriterion], verdicts: Dict[str, VerdictView], pass_threshold: float) -> Tuple[List[VerdictView], float, bool, bool, bool, List[str]]:
    """``RubricEvaluator.evaluate_deterministic``'s scoring: low confidence is FLAGGED, NOT_APPLICABLE
    and FLAGGED leave the weight out. FLAGGED makes the score provisional; it is not a failure."""
    total = sum(c.weight for c in criteria)
    earned = 0.0
    critical = False
    review = False
    reasons: List[str] = []
    out: List[VerdictView] = []
    for criterion in criteria:
        verdict = verdicts[criterion.criterion_id]
        if verdict.hallucination_detected and criterion.check.check_type is not CheckType.SEMANTIC_JUDGEMENT:
            # The deterministic stage's grounding guardrail already FLAGGED it (a model's quote is
            # verified by its own assessment job instead, as the pre-split router did).
            reasons.append(f"Guardrail rejection on {criterion.criterion_id}: Cited quote was not found verbatim in transcript.")
        if verdict.confidence < LOW_CONFIDENCE and verdict.status is not VerdictStatus.FLAGGED:
            reasons.append(f"Low confidence ({verdict.confidence:.2f}) on criterion {criterion.criterion_id} ({criterion.name}).")
            verdict = verdict.model_copy(update={"status": VerdictStatus.FLAGGED})
        elif verdict.confidence < LOW_CONFIDENCE:
            reasons.append(f"Low confidence ({verdict.confidence:.2f}) on criterion {criterion.criterion_id} ({criterion.name}).")
        if verdict.status is VerdictStatus.PASS:
            earned += criterion.weight
        elif criterion.critical and verdict.status is VerdictStatus.FAIL:
            critical = True
            reasons.append(f"Critical compliance failure on {criterion.criterion_id}: {criterion.name}.")
        if verdict.status in (VerdictStatus.NOT_APPLICABLE, VerdictStatus.FLAGGED):
            total -= criterion.weight
        if verdict.status is VerdictStatus.FLAGGED:
            review = True
        out.append(verdict)
    overall = round((earned / total) * 100.0, 1) if total > 0 else 0.0
    passed = overall >= (pass_threshold or 80.0) and not critical and not review and total > 0
    if critical:
        review = True
    return out, max(0.0, min(100.0, overall)), passed, critical, review, reasons


class ScorecardHandler(Handler):
    job_type = JobType.QA_SCORECARD
    adapter_id = "call1.code.scorecard"
    adapter_version = "2"

    def run(self, job: HandlerJob) -> HandlerResult:
        rubric = job.rubric()
        deterministic: Dict[str, VerdictView] = {}
        det_input = job.input("verdicts")
        if det_input is not None:
            content: QaVerdictContent = det_input.content()  # type: ignore[assignment]
            deterministic = {v.criterion_id: v for v in content.verdicts}
        verdicts: Dict[str, VerdictView] = {}
        for criterion in rubric.criteria:
            if criterion.check.check_type is CheckType.SEMANTIC_JUDGEMENT:
                primary = job.input(f"assessment:{criterion.criterion_id}")
                escalation = job.input(f"escalation:{criterion.criterion_id}")
                verdicts[criterion.criterion_id] = semantic_verdict(
                    criterion, primary.content() if primary else None, escalation.content() if escalation else None)  # type: ignore[arg-type]
            else:
                verdicts[criterion.criterion_id] = deterministic.get(criterion.criterion_id) or VerdictView(
                    criterion_id=criterion.criterion_id, criterion_name=criterion.name, status=VerdictStatus.FLAGGED, confidence=0,
                    speaker=criterion.check.speaker or SpeakerRole.AGENT, reasoning="The deterministic check did not produce a verdict.")
        ordered, overall, passed, critical, review, reasons = score(list(rubric.criteria), verdicts, rubric.pass_threshold)
        card = QaScorecardContent(rubric=job.scorecard_rubric_ref(), overall_score=overall, passed=passed, critical_failure=critical,
                                  requires_human_review=review, escalation_reasons=reasons, verdicts=ordered, evaluated_at=utcnow())
        return HandlerResult(outputs={"scorecard": Output(card)})


def _check_citations(citations: List[SummaryCitation], turn_ids: Set[int]) -> Tuple[List[SummaryCitation], int]:
    kept: List[SummaryCitation] = []
    dropped = 0
    for citation in citations:
        ids = [t for t in citation.turn_ids if t in turn_ids]
        dropped += len(citation.turn_ids) - len(ids)
        if ids:
            kept.append(citation.model_copy(update={"turn_ids": ids}))
    return kept, dropped


class SummaryAssemblyHandler(Handler):
    job_type = JobType.SUMMARY_ASSEMBLY
    adapter_id = "call1.code.summary_assembly"
    adapter_version = "1"

    def run(self, job: HandlerJob) -> HandlerResult:
        transcript = job.transcript()
        turn_ids = {t.turn_id for t in transcript.turns}
        segments: List[SummarySegmentContent] = [item.content() for _, item in sorted(  # type: ignore[misc]
            job.inputs_with_prefix("segment:").items(), key=lambda kv: int(kv[0].split(":", 1)[1]))]
        if not segments:
            raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, "no summary segments were planned for this call")
        synthesis_input = job.input("synthesis")
        synthesis: Optional[SummarySynthesisContent] = synthesis_input.content() if synthesis_input else None  # type: ignore[assignment]
        source = synthesis if synthesis is not None else segments[0]
        citations, dropped = _check_citations(list(source.citations), turn_ids)
        key_points = list(source.key_points)
        from_segments = False
        if not key_points and len(segments) > 1:
            from_segments = True
            citations = [c for c in citations if c.claim == "narrative"]
            for seg in segments:
                seg_cites, seg_dropped = _check_citations(list(seg.citations), turn_ids)
                dropped += seg_dropped
                for index, point in enumerate(seg.key_points):
                    citations += [c.model_copy(update={"index": len(key_points)}) for c in seg_cites if c.claim == "key_point" and c.index == index]
                    key_points.append(point)
        extra = job.parameters.extra
        content = SummaryContent(
            narrative=source.narrative, key_points=key_points, rubric_highlights=[], citations=citations,
            grounding={"key_points_citations_checked": True, "citations_dropped": dropped, "chunked": len(segments) > 1,
                       "key_points_from_source_segments": from_segments},
            route_class=str(extra.get("route_class") or "appliance"), catalog_entry_id=str(extra.get("summary_entry") or "unknown"),
            generated_at=utcnow(), segments=len(segments),
        )
        return HandlerResult(outputs={"summary": Output(content)})


PASS_OF_JOB_TYPE = {JobType.CONTACT_SIGNALS_LIFECYCLE: ContactSignalPass.LIFECYCLE, JobType.CONTACT_SIGNALS_RESOLUTION: ContactSignalPass.RESOLUTION}


class ContactSignalsMergeHandler(Handler):
    job_type = JobType.CONTACT_SIGNALS_MERGE
    adapter_id = "call1.code.contact_signals_merge"
    adapter_version = "1"

    def run(self, job: HandlerJob) -> HandlerResult:
        problem = job.parameters.extra.get("configuration_error")
        if problem:
            raise HandlerError(JobErrorCode.CONFIGURATION_ERROR, str(problem)[:500])
        if job.parameters.signals is not None:
            from .signal_stages import merge_v2

            content, partial = merge_v2(job)
            return HandlerResult(outputs={"contact_signals": Output(content)}, partial_reason=partial)
        transcript = job.transcript()
        text_by_turn = {t.turn_id: t.text.lower() for t in transcript.turns}
        passes: Dict[Tuple[str, str], ContactSignalsPassOutcome] = {}
        signals: List[ContactSignalView] = []
        seen = set()
        for role, item in sorted(job.inputs_with_prefix("pass:").items()):
            content: ContactSignalsPassContent = item.content()  # type: ignore[assignment]
            key = (content.pass_kind.value, role)
            passes[key] = ContactSignalsPassOutcome(pass_kind=content.pass_kind, window=content.window, included=True)
            for signal in content.signals:
                if signal.turn_id is not None and signal.quote.strip().lower() not in text_by_turn.get(signal.turn_id, ""):
                    continue  # only quote-verified observations are merged
                identity = (signal.kind.value, signal.turn_id, signal.quote.strip().lower())
                if identity in seen:
                    continue
                seen.add(identity)
                signals.append(signal.model_copy(update={"id": f"{content.pass_kind.value}-{signal.id}"[:200]}))
        included_kinds = {k for k, _ in passes}
        missing: List[str] = []
        for outcome in job.upstream:
            kind = PASS_OF_JOB_TYPE.get(outcome.job_type)
            if kind is None or kind.value in included_kinds:
                continue
            code = outcome.error_code or (JobErrorCode.CANCELLED if outcome.status is JobStatus.CANCELLED else None)
            passes[(kind.value, outcome.job_id)] = ContactSignalsPassOutcome(pass_kind=kind, included=False, failure_code=code)
            missing.append(f"{kind.value} pass {'cancelled' if outcome.status is JobStatus.CANCELLED else 'failed'}")
        if not passes:
            raise HandlerError(JobErrorCode.INPUT_UNAVAILABLE, "no contact-signal pass reached the merge")
        signals.sort(key=lambda s: (s.start, s.id))
        partial_reason = "; ".join(sorted(missing)) if missing else None
        content = ContactSignalsContent(
            completeness="partial" if partial_reason else "complete", partial_reason=partial_reason, signals=signals,
            passes=[passes[k] for k in sorted(passes)], transcript_fingerprint=transcript_fingerprint(transcript), generated_at=utcnow(),
            pipeline_note=str(job.parameters.extra.get("pipeline_note"))[:200] if job.parameters.extra.get("pipeline_note") else None,
        )
        return HandlerResult(outputs={"contact_signals": Output(content)}, partial_reason=partial_reason)


def code_handlers() -> List[Handler]:
    return [ScorecardHandler(), SummaryAssemblyHandler(), ContactSignalsMergeHandler()]
