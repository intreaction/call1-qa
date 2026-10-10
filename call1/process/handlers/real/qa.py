"""QA: the rubric's deterministic checks, and one model assessment per semantic criterion.

``qa_deterministic`` runs ``RubricEvaluator._evaluate_criterion`` for every non-semantic criterion
(phrase, regex, conditional-response and sentiment-metric checks), then the pre-split grounding
guardrail: a quote that is not verbatim in the transcript is FLAGGED with
``hallucination_detected``.

``qa_criterion`` and ``qa_escalation`` are the pre-split ``QuestionRouter`` for one model: the
criterion's policy and applicability gates, the speaker-scope gate, the JSON-schema prompt
(``RubricEvaluator._semantic_prompt``, ``call1.qa_output``), strict answer parsing, and quote
verification against exactly the (masked, when the route is masked) text the model saw
(``verify_quoted_evidence``). Generation goes through ``call1.question_models.generate`` (via
``generate_text``), so the Stage 0 Pro1 closure applies. Outcomes map to the contract:

* a grounded PASS/FAIL/NOT_APPLICABLE: trigger none;
* the model says ``needs_review``: FLAGGED, trigger ``needs_review``;
* unparseable output, a missing or unverified quote: FLAGGED, trigger ``invalid_answer`` (an
  outcome, not an error; the core records usage ``validation_rejected``);
* a provider failure: ``HandlerError(provider_error)`` with the attempt's ``prompt_input``; the core
  retries, and on the final attempt records a FLAGGED ``provider_error`` assessment;
* long transcripts use lossless compact rows, then complete section evidence review and bounded
  whole-call synthesis when still oversized. Only an irreducible policy/turn context error
  produces a FLAGGED assessment (trigger
  ``provider_error``, attempt ``error_code: context_limit_exceeded``), as the pre-split router
  recorded it. Retrying cannot help, and failing the job would dead-block the scorecard;
* the policy, speaker-identity and speaker-scope gates FLAG without calling a model, as before the
  split (trigger none, so no escalation fires unless the criterion escalates ``always``).

The escalation decision, the scorecard and the primary/escalation disagreement rule stay in the
worker core and the ``qa_scorecard`` code stage.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from call1.contracts.contents import (
    EscalationTrigger,
    ModelAttemptView,
    QaAssessmentContent,
    QaVerdictContent,
    SpeakerRole,
    VerdictStatus,
)
from call1.contracts.errors import JobErrorCode
from call1.contracts.jobs import JobType
from call1.contracts.rubrics import CheckType

from call1.process.handlers.base import Handler, HandlerError, HandlerJob, HandlerResult, JobCancelled, Output, Usage

from .convert import contract_verdict, legacy_criterion, legacy_rubric, legacy_transcript
from .llm import LlmTransport, check_route, template_version
from .masking import enrichment, mask, route_masked, sensitive_values

ADAPTER_VERSION = "2"
QA_TEMPLATE_ID = "call1.qa.semantic_judgement"
GUIDANCE_FIELDS = ("pass_when", "fail_when", "not_applicable_when", "policy_context")
CONTEXT_OVERFLOW_REASONING = ("The call transcript is longer than the model's context budget, so the model could not assess this "
                              "criterion. Human review is required.")


def _content(job: HandlerJob, role: str):
    item = job.input(role)
    return item.content() if item is not None else None


class RealQaDeterministic(Handler):
    job_type = JobType.QA_DETERMINISTIC
    adapter_id = "call1.pipeline.rubric_evaluator"
    adapter_version = ADAPTER_VERSION

    def run(self, job: HandlerJob) -> HandlerResult:
        from call1.models.schemas import VerdictStatus as LegacyStatus
        from call1.pipeline.evaluator import RubricEvaluator, verify_quoted_evidence

        rubric = job.rubric()
        transcript = legacy_transcript(job.transcript(), call_id=job.job.conversation_id, sentiment=_content(job, "text_sentiment"),
                                       tone=_content(job, "tone_blocks"))
        evaluator = RubricEvaluator(legacy_rubric(rubric))
        verdicts = []
        for criterion in rubric.criteria:
            if criterion.check.check_type is CheckType.SEMANTIC_JUDGEMENT:
                continue
            verdict = evaluator._evaluate_criterion(legacy_criterion(criterion), transcript)
            quote_turn: Optional[int] = None
            if verdict.quoted_evidence:
                # The pre-split grounding guardrail (RubricEvaluator.evaluate_deterministic).
                verified, turn_id, span = verify_quoted_evidence(verdict.quoted_evidence, transcript, expected_speaker=verdict.speaker)
                if verified:
                    verdict.timestamp_range = span
                    verdict.hallucination_detected = False
                    quote_turn = turn_id
                else:
                    verdict.hallucination_detected = True
                    verdict.status = LegacyStatus.FLAGGED
            verdicts.append(contract_verdict(verdict, criterion, quote_turn_id=quote_turn))
        return HandlerResult(outputs={"verdicts": Output(QaVerdictContent(verdicts=verdicts))})


def qa_inputs(job: HandlerJob, *, masked: bool):
    """(legacy criterion, check, transcript, sensitive values) as ``RealQaAssessment`` prompts the
    model: the job's criterion and its transcript (with the ``speaker_attribution`` input applied).
    ``masked`` masks the transcript and the criterion guidance with one value set, as the pre-split
    router did: the number rules plus the pinned ``pii_findings`` of this transcript revision
    (``masking.sensitive_values`` with the job). ``values`` is None when not masked."""
    from call1.models.schemas import CallTranscript

    criterion = job.criterion()
    legacy = legacy_criterion(criterion)
    check = legacy.check
    transcript = legacy_transcript(job.transcript(), call_id=job.job.conversation_id, enrichment=enrichment(job, compute=masked or None))
    values = None
    if masked:
        guidance = [getattr(check, f) or "" for f in GUIDANCE_FIELDS]
        values = sensitive_values(transcript.turns, guidance, job=job)
        transcript = CallTranscript(call_id=transcript.call_id, duration_seconds=transcript.duration_seconds,
                                    turns=[t.model_copy(update={"text": mask(t.text, values)}) for t in transcript.turns])
        check = check.model_copy(update={f: mask(getattr(check, f), values) for f in GUIDANCE_FIELDS if getattr(check, f)})
        legacy = legacy.model_copy(update={"check": check})
    return legacy, check, transcript, values


def qa_prompt(job: HandlerJob, *, masked: bool, compact: Optional[bool] = None):
    """(system, user, masked transcript, check): the exact messages ``RealQaAssessment`` sends for
    the job's semantic criterion (``QA_SYSTEM`` and ``RubricEvaluator._semantic_prompt``). On-device
    training rebuilds its QA examples with this, always ``masked=True``, so their digest equals the
    primary's ``prompt_input.prompt_digest`` when the template is unchanged and the call ran masked."""
    from call1.pipeline.evaluator import RubricEvaluator
    from call1.question_models import SYSTEM

    _legacy, check, transcript, _values = qa_inputs(job, masked=masked)
    from .qa_context import choose_prompt, compact_prompt

    original = RubricEvaluator()._semantic_prompt(check, transcript.turns)
    # Training replay has no live model selection. Rebuild the recorded format rather
    # than using today's model budget to change yesterday's training messages.
    if compact is not None or job.catalog_entry is None:
        prompt = compact_prompt(check, transcript) if compact else original
    else:
        prompt, _compact = choose_prompt(job, SYSTEM, original, check, transcript)
    return SYSTEM, prompt, transcript, check


class _Failure(Exception):
    """Carries the transport's HandlerError through RubricEvaluator, which catches every exception
    from its model backend and FLAGs the criterion."""


class RealQaAssessment(Handler):
    kind = "primary"
    adapter_id = "call1.qa.semantic_judgement"
    adapter_version = ADAPTER_VERSION

    def ready(self, job: HandlerJob) -> None:
        check_route(job)

    def run(self, job: HandlerJob) -> HandlerResult:
        from call1.pipeline.evaluator import RubricEvaluator, verify_quoted_evidence
        from call1.qa_output import QA_SCHEMA
        from call1.question_models import SYSTEM  # QA_SYSTEM, as generate() sends it

        criterion = job.criterion()
        legacy, check, transcript, _values = qa_inputs(job, masked=route_masked(job))

        transport = LlmTransport(job)
        state: Dict[str, object] = {"raw": None, "failure": None, "started": None, "cancelled": False, "compact": False, "sectioned": False}

        def backend(prompt: str) -> str:
            state["started"] = time.monotonic()
            try:
                from .qa_context import Budget, choose_prompt, sectioned_answer

                budget = Budget(job)
                planned, compact = choose_prompt(job, SYSTEM, prompt, check, transcript, budget)
                state["compact"] = compact
                if budget.fits(SYSTEM, planned):
                    try:
                        raw = transport.generate(SYSTEM, planned, response_schema=QA_SCHEMA, schema_name="qa_answer").raw
                    except HandlerError as exc:
                        if exc.code is not JobErrorCode.CONTEXT_LIMIT_EXCEEDED:
                            raise
                        state["sectioned"] = True
                        raw = sectioned_answer(transport, check, transcript, budget)
                else:
                    state["sectioned"] = True
                    raw = sectioned_answer(transport, check, transcript, budget)
            except JobCancelled:
                state["cancelled"] = True
                raise _Failure() from None
            except HandlerError as exc:
                if exc.code is JobErrorCode.VALIDATION_REJECTED:
                    import json
                    raw = json.dumps({"assessment": "Section evidence could not be verified. Human review is required.",
                                      "verdict": "needs_review", "quote": ""})
                else:
                    state["failure"] = exc
                    raise _Failure() from None
            state["raw"] = raw
            return raw

        job.check_cancelled()
        evaluator = RubricEvaluator(model_backend=backend)
        verdict = evaluator._evaluate_criterion(legacy, transcript)
        if state["cancelled"]:
            raise JobCancelled()
        version = template_version(SYSTEM, QA_SCHEMA)
        template_id = QA_TEMPLATE_ID
        if state["compact"] or state["sectioned"]:
            from .qa_context import VERSION
            version = template_version(SYSTEM, QA_SCHEMA, VERSION)
            template_id += ".sectioned" if state["sectioned"] else ".compact"
        prompt_input = transport.prompt_input(template_id, version)
        failure: Optional[HandlerError] = state["failure"]  # type: ignore[assignment]
        if failure is not None and failure.code is JobErrorCode.CONTEXT_LIMIT_EXCEEDED:
            return self._context_overflow(job, criterion, transport, prompt_input)
        if failure is not None:
            raise HandlerError(failure.code, failure.detail, outputs={"prompt_input": Output(prompt_input)}, usage=transport.usage())

        called = state["started"] is not None
        quote_turn: Optional[int] = None
        if verdict.quoted_evidence and called:
            # The router's check: the quote must be verbatim, from the permitted speaker, in exactly
            # the text the model saw.
            verified, turn_id, span = verify_quoted_evidence(verdict.quoted_evidence, transcript, check.speaker)
            verdict.hallucination_detected = not verified
            if verified:
                verdict.timestamp_range = span
                quote_turn = turn_id
                verdict.speaker = next(t.speaker for t in transcript.turns if t.turn_id == turn_id)
        trigger = self._trigger(evaluator, verdict, called, state["raw"])  # type: ignore[arg-type]
        return self._result(job, criterion, verdict, trigger, quote_turn, transport, prompt_input, called)

    @staticmethod
    def _trigger(evaluator, verdict, called: bool, raw: Optional[str]) -> Optional[EscalationTrigger]:
        """The router's escalation reason for this outcome (``always`` is left to the core)."""
        if not called:
            return None
        parsed = evaluator._parse_semantic_answer(raw) if raw is not None else None
        if parsed and parsed[0] == "needs_review":
            return EscalationTrigger.NEEDS_REVIEW
        if verdict.status.value == VerdictStatus.FLAGGED.value:
            return EscalationTrigger.INVALID_ANSWER
        return None

    def _context_overflow(self, job: HandlerJob, criterion, transport: LlmTransport, prompt_input) -> HandlerResult:
        """The pre-split router caught every model exception and FLAGGED the criterion (reason
        ``provider_error``), so a context overflow is an assessment, not a job failure: the
        scorecard still publishes, and a criterion that escalates on ``provider_error`` escalates."""
        selection = job.selection
        attempt = ModelAttemptView(
            job_id=job.job.id, attempt_number=job.attempt_number,
            catalog_entry_id=selection.catalog_entry.entry_id if selection else "unknown",
            model_revision=transport.model_revision() or (selection.model_revision if selection else "unknown"),
            route_class=selection.route.route_class.value if selection else "appliance",
            destination_host=selection.route.destination_host if selection else "in-process",
            status=VerdictStatus.FLAGGED, reasoning=CONTEXT_OVERFLOW_REASONING, trigger=EscalationTrigger.PROVIDER_ERROR,
            latency_ms=int(round(transport.seconds * 1000)), error_code=JobErrorCode.CONTEXT_LIMIT_EXCEEDED,
        )
        assessment = QaAssessmentContent(
            criterion_id=criterion.criterion_id, assessment_kind=self.kind, status=VerdictStatus.FLAGGED, confidence=0.0,
            reasoning=CONTEXT_OVERFLOW_REASONING, speaker=SpeakerRole((criterion.check.speaker or SpeakerRole.AGENT).value),
            trigger=EscalationTrigger.PROVIDER_ERROR, escalation_requested=False, attempt=attempt,
        )
        return HandlerResult(outputs={"assessment": Output(assessment), "prompt_input": Output(prompt_input)}, usage=transport.usage(),
                             model_revision=transport.model_revision())

    def _result(self, job: HandlerJob, criterion, verdict, trigger: Optional[EscalationTrigger], quote_turn: Optional[int],
                transport: LlmTransport, prompt_input, called: bool) -> HandlerResult:
        selection = job.selection
        status = VerdictStatus(verdict.status.value)
        timestamp: Optional[Tuple[float, float]] = tuple(float(x) for x in verdict.timestamp_range) if verdict.timestamp_range else None  # type: ignore[assignment]
        tokens_in, tokens_out = transport.tokens()
        attempt = ModelAttemptView(
            job_id=job.job.id, attempt_number=job.attempt_number,
            catalog_entry_id=selection.catalog_entry.entry_id if selection else "unknown",
            model_revision=(transport.model_revision() if called else None) or (selection.model_revision if selection else "unknown"),
            route_class=selection.route.route_class.value if selection else "appliance",
            destination_host=selection.route.destination_host if selection else "in-process",
            status=status, reasoning=verdict.reasoning, quoted_evidence=verdict.quoted_evidence, trigger=trigger,
            latency_ms=int(round(transport.seconds * 1000)), tokens_input=tokens_in, tokens_output=tokens_out,
        )
        assessment = QaAssessmentContent(
            criterion_id=criterion.criterion_id, assessment_kind=self.kind, status=status,
            confidence=max(0.0, min(1.0, float(verdict.confidence))), reasoning=verdict.reasoning, quoted_evidence=verdict.quoted_evidence,
            quote_turn_id=quote_turn, timestamp_range=timestamp, speaker=SpeakerRole(verdict.speaker.value),
            hallucination_detected=bool(verdict.hallucination_detected), trigger=trigger, escalation_requested=False, attempt=attempt,
        )
        usage = transport.usage() if called else Usage(inference_seconds=0.0)
        return HandlerResult(outputs={"assessment": Output(assessment), "prompt_input": Output(prompt_input)}, usage=usage,
                             model_revision=transport.model_revision() if called else None)


class RealQaCriterion(RealQaAssessment):
    job_type = JobType.QA_CRITERION
    kind = "primary"


class RealQaEscalation(RealQaAssessment):
    job_type = JobType.QA_ESCALATION
    kind = "escalation"


__all__ = ["RealQaCriterion", "RealQaDeterministic", "RealQaEscalation", "qa_inputs", "qa_prompt"]
