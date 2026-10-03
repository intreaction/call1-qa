"""Unit tests for rubric evaluation and quote verification guardrails."""

import unittest

from call1.models.schemas import (
    CallTranscript,
    SpeakerRole,
    TranscriptTurn,
    VerdictStatus,
)
from call1.pipeline.evaluator import (
    DEFAULT_RUBRIC, LEGACY_PRESET_RUBRICS,
    RubricEvaluator,
    verify_quoted_evidence,
)


class TestRubricEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = RubricEvaluator(LEGACY_PRESET_RUBRICS[0])

    def test_quote_verification_success(self):
        turns = [
            TranscriptTurn(
                turn_id=0,
                speaker=SpeakerRole.AGENT,
                start_time=0.0,
                end_time=4.0,
                text="Hello, thank you for calling. This call is being recorded for quality assurance.",
            )
        ]
        transcript = CallTranscript(call_id="call_test_1", turns=turns, duration_seconds=10.0)

        verified, turn_id, ts = verify_quoted_evidence(
            "This call is being recorded for quality assurance.",
            transcript,
            expected_speaker=SpeakerRole.AGENT,
        )
        self.assertTrue(verified)
        self.assertEqual(turn_id, 0)
        self.assertEqual(ts, (0.0, 4.0))

    def test_quote_verification_failure_hallucination(self):
        turns = [
            TranscriptTurn(
                turn_id=0,
                speaker=SpeakerRole.AGENT,
                start_time=0.0,
                end_time=4.0,
                text="Hello, thank you for calling customer support.",
            )
        ]
        transcript = CallTranscript(call_id="call_test_2", turns=turns, duration_seconds=10.0)

        # Non-existent quote
        verified, turn_id, ts = verify_quoted_evidence(
            "Our terms of service apply to this transaction.",
            transcript,
        )
        self.assertFalse(verified)
        self.assertIsNone(turn_id)
        self.assertIsNone(ts)

    def test_compliant_call_evaluation(self):
        turns = [
            TranscriptTurn(
                turn_id=0,
                speaker=SpeakerRole.AGENT,
                start_time=0.0,
                end_time=3.5,
                text="Thank you for calling Apex Bank. This call is recorded. My name is Alex.",
            ),
            TranscriptTurn(
                turn_id=1,
                speaker=SpeakerRole.AGENT,
                start_time=4.0,
                end_time=7.0,
                text="Before we review your balance, may I verify your account number and date of birth?",
            ),
            TranscriptTurn(
                turn_id=2,
                speaker=SpeakerRole.CALLER,
                start_time=7.2,
                end_time=10.0,
                text="Yes, account number 12345, date of birth July 4th 1980.",
            ),
            TranscriptTurn(
                turn_id=3,
                speaker=SpeakerRole.AGENT,
                start_time=10.5,
                end_time=14.0,
                text="Is there anything else I can help you with today? Thank you for calling and have a great day!",
            ),
        ]
        transcript = CallTranscript(call_id="call_compliant", turns=turns, duration_seconds=15.0)

        result = self.evaluator.evaluate_deterministic(transcript)
        self.assertTrue(result.passed)
        self.assertEqual(result.overall_score, 100.0)
        self.assertFalse(result.critical_failure)
        self.assertFalse(result.requires_human_review)

    def test_missing_disclosure_critical_failure(self):
        # Omit recording disclosure and ID verification
        turns = [
            TranscriptTurn(
                turn_id=0,
                speaker=SpeakerRole.AGENT,
                start_time=0.0,
                end_time=3.0,
                text="Apex Bank, how can I help you?",
            ),
            TranscriptTurn(
                turn_id=1,
                speaker=SpeakerRole.CALLER,
                start_time=3.2,
                end_time=6.0,
                text="What is my account balance?",
            ),
            TranscriptTurn(
                turn_id=2,
                speaker=SpeakerRole.AGENT,
                start_time=6.5,
                end_time=8.0,
                text="Your balance is $4,200.",
            ),
        ]
        transcript = CallTranscript(call_id="call_breach", turns=turns, duration_seconds=10.0)

        result = self.evaluator.evaluate_deterministic(transcript)
        self.assertFalse(result.passed)
        self.assertTrue(result.critical_failure)
        self.assertTrue(result.requires_human_review)
        self.assertIn("Critical compliance failure", result.escalation_reasons[0])

    def test_grounding_guardrail_catches_hallucinated_quote(self):
        turns = [
            TranscriptTurn(
                turn_id=0,
                speaker=SpeakerRole.AGENT,
                start_time=0.0,
                end_time=3.0,
                text="Thank you for calling. This call may be recorded.",
            )
        ]
        transcript = CallTranscript(call_id="call_hallucination", turns=turns, duration_seconds=10.0)

        # Trigger simulated hallucination (quote not in transcript)
        result = self.evaluator.evaluate_deterministic(transcript, simulated_hallucination=True)
        reg_verdict = next(v for v in result.verdicts if v.criterion_id == "REG-01")

        self.assertTrue(reg_verdict.hallucination_detected)
        self.assertEqual(reg_verdict.status, VerdictStatus.FLAGGED)
        self.assertTrue(result.requires_human_review)
        self.assertTrue(any("Guardrail rejection" in r for r in result.escalation_reasons))


    def test_timing_greeting_rule(self):
        from call1.pipeline.evaluator import RubricCriterion, RubricDefinition
        greet_rubric = RubricDefinition(
            rubric_id="test_greet",
            name="Greeting Test",
            description="",
            criteria=[
                RubricCriterion(
                    criterion_id="GREET-01",
                    name="Speed to Greet",
                    rule_type="timing_greeting",
                    parameters={"max_seconds": 15.0},
                )
            ],
        )
        evaluator = RubricEvaluator(greet_rubric)

        # 1. Compliant: Greeted at 3.0s (< 15.0s)
        turns_fast = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=2.5, end_time=5.0, text="Good morning, thank you for calling.")
        ]
        res_fast = evaluator.evaluate_deterministic(CallTranscript(call_id="c1", turns=turns_fast, duration_seconds=10.0))
        self.assertEqual(res_fast.verdicts[0].status, VerdictStatus.PASS)
        self.assertIn("within the 15s threshold", res_fast.verdicts[0].reasoning)

        # 2. Delayed: Greeted at 22.0s (> 15.0s)
        turns_slow = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=22.0, end_time=25.0, text="Hello, thank you for calling.")
        ]
        res_slow = evaluator.evaluate_deterministic(CallTranscript(call_id="c2", turns=turns_slow, duration_seconds=30.0))
        self.assertEqual(res_slow.verdicts[0].status, VerdictStatus.FAIL)
        self.assertIn("exceeded the 15s threshold", res_slow.verdicts[0].reasoning)

    def test_empathy_sentiment_rule(self):
        from call1.pipeline.evaluator import RubricCriterion, RubricDefinition
        emp_rubric = RubricDefinition(
            rubric_id="test_emp",
            name="Empathy Test",
            description="",
            criteria=[
                RubricCriterion(
                    criterion_id="EMP-01",
                    name="Empathy Response",
                    rule_type="empathy_sentiment",
                    parameters={"sensitivity": 0.5},
                )
            ],
        )
        evaluator = RubricEvaluator(emp_rubric)

        # 1. Caller upset, agent responds with empathy -> PASS
        turns_empathic = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.CALLER, start_time=2.0, end_time=6.0, text="I am really frustrated! You charged me twice and it's ridiculous."),
            TranscriptTurn(turn_id=1, speaker=SpeakerRole.AGENT, start_time=6.5, end_time=10.0, text="I am so sorry to hear that. I completely understand your frustration, let me help you resolve this right away."),
        ]
        res_emp = evaluator.evaluate_deterministic(CallTranscript(call_id="c3", turns=turns_empathic, duration_seconds=15.0))
        self.assertEqual(res_emp.verdicts[0].status, VerdictStatus.PASS)
        self.assertIn("Agent offered empathic response", res_emp.verdicts[0].reasoning)

        # 2. Caller upset, agent is cold/dismissive -> FAIL
        turns_cold = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.CALLER, start_time=2.0, end_time=6.0, text="I am extremely upset! This is broken and unacceptable."),
            TranscriptTurn(turn_id=1, speaker=SpeakerRole.AGENT, start_time=6.5, end_time=8.0, text="Give me your account number."),
        ]
        res_cold = evaluator.evaluate_deterministic(CallTranscript(call_id="c4", turns=turns_cold, duration_seconds=15.0))
        self.assertEqual(res_cold.verdicts[0].status, VerdictStatus.FAIL)
        self.assertIn("failed to offer an empathic", res_cold.verdicts[0].reasoning)

    def test_compliance_phrase_fuzzy_tolerance(self):
        from call1.pipeline.evaluator import RubricCriterion, RubricDefinition
        comp_rubric = RubricDefinition(
            rubric_id="test_fuzzy",
            name="Fuzzy Compliance Test",
            description="",
            criteria=[
                RubricCriterion(
                    criterion_id="FDCPA-01",
                    name="Mini-Miranda",
                    rule_type="compliance_phrase",
                    parameters={
                        "target_phrase": "This communication is from a debt collector and any information will be used for that purpose.",
                        "fuzzy_threshold": 0.40,
                        "required_keywords": ["debt collector", "information obtained will be used"],
                    },
                )
            ],
        )
        evaluator = RubricEvaluator(comp_rubric)

        # Allowed variation still matches key terms
        turns = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=1.0, end_time=5.0, text="Please note this call is from an authorized debt collector regarding your account.")
        ]
        res = evaluator.evaluate_deterministic(CallTranscript(call_id="c5", turns=turns, duration_seconds=10.0))
        self.assertEqual(res.verdicts[0].status, VerdictStatus.PASS)
        self.assertIn("debt collector", res.verdicts[0].quoted_evidence)


class TestSemanticJudgementGrounding(unittest.TestCase):
    """Semantic questions must never be guessed by the deterministic tier.

    A semantic judgement is a claim about the relationship between turns —
    which way round a sentence points, whether one thing came before another.
    Keyword presence has no opinion about negation or order, so the
    deterministic evaluator must FLAG these checks for human review rather
    than guess PASS/FAIL. When a model backend is wired in, its verdict is
    only published if the supporting quote verifies verbatim.
    """

    def _semantic_rubric(self, pass_when, fail_when, not_applicable_when=None):
        from call1.models.schemas import CheckType, RubricCheck, RubricCriterion, RubricDefinition

        return RubricDefinition(
            rubric_id="test_semantic",
            name="Semantic Test",
            description="",
            criteria=[
                RubricCriterion(
                    criterion_id="SEM-01",
                    name="Semantic Judgement",
                    rule_type="custom",
                    parameters={},
                    check=RubricCheck(
                        check_type=CheckType.SEMANTIC_JUDGEMENT,
                        pass_when=pass_when,
                        fail_when=fail_when,
                        not_applicable_when=not_applicable_when,
                        speaker=SpeakerRole.AGENT,
                    ),
                )
            ],
        )

    def test_negation_does_not_falsely_pass(self):
        """'I can hear you' must not pass a check written around 'I cannot hear you'.

        The two sentences share every content word; only negation separates
        them. A keyword matcher would credit the pass clause and publish a
        PASS. The deterministic tier must not — it flags for review.
        """
        turns = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=0.0, end_time=4.0, text="I can hear you clearly."),
        ]
        transcript = CallTranscript(call_id="sem_negation", turns=turns, duration_seconds=10.0)

        rubric = self._semantic_rubric(
            pass_when="The customer says they cannot hear the agent.",
            fail_when="The customer says they can hear the agent.",
        )
        res = RubricEvaluator(rubric).evaluate_deterministic(transcript)
        verdict = res.verdicts[0]

        self.assertEqual(verdict.status, VerdictStatus.FLAGGED)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIsNone(verdict.quoted_evidence)
        self.assertIn("does not guess", verdict.reasoning)

    def test_order_does_not_falsely_pass(self):
        """'Asked after' must not pass a check written around 'asked before'.

        The same words in a different order mean the opposite thing. Keyword
        presence cannot tell them apart, so the deterministic tier flags.
        """
        turns = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=0.0, end_time=4.0, text="The agent asked for the account number after the disclosure."),
        ]
        transcript = CallTranscript(call_id="sem_order", turns=turns, duration_seconds=10.0)

        rubric = self._semantic_rubric(
            pass_when="The agent asked for the account number before the disclosure.",
            fail_when="The agent asked for the account number after the disclosure.",
        )
        res = RubricEvaluator(rubric).evaluate_deterministic(transcript)
        verdict = res.verdicts[0]

        self.assertEqual(verdict.status, VerdictStatus.FLAGGED)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIsNone(verdict.quoted_evidence)

    def test_deterministic_evaluator_flags_with_zero_confidence(self):
        """The audit trail stays honest: an unanswered semantic question is a
        review gap, never a guessed verdict."""
        turns = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=0.0, end_time=4.0, text="Thank you for calling."),
        ]
        transcript = CallTranscript(call_id="sem_flag", turns=turns, duration_seconds=10.0)

        rubric = self._semantic_rubric(
            pass_when="The agent offers further assistance before closing.",
            fail_when="The agent hangs up without offering help.",
        )
        res = RubricEvaluator(rubric).evaluate_deterministic(transcript)
        verdict = res.verdicts[0]

        self.assertEqual(verdict.status, VerdictStatus.FLAGGED)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIsNone(verdict.quoted_evidence)
        self.assertTrue(res.requires_human_review)

    def test_model_backend_verified_quote_publishes_verdict(self):
        """With a model backend, a verdict whose quote verifies verbatim is
        published with the timestamp range."""
        turns = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=0.0, end_time=4.0, text="Is there anything else I can help you with today?"),
        ]
        transcript = CallTranscript(call_id="sem_model_pass", turns=turns, duration_seconds=10.0)

        rubric = self._semantic_rubric(
            pass_when="The agent offers further assistance before closing.",
            fail_when="The agent hangs up without offering help.",
        )

        def fake_backend(prompt):
            return '{"assessment":"Test assessment.","verdict": "pass", "quote": "Is there anything else I can help you with today?"}'

        res = RubricEvaluator(rubric, model_backend=fake_backend).evaluate_deterministic(transcript)
        verdict = res.verdicts[0]

        self.assertEqual(verdict.status, VerdictStatus.PASS)
        self.assertEqual(verdict.quoted_evidence, "Is there anything else I can help you with today?")
        self.assertEqual(verdict.timestamp_range, (0.0, 4.0))
        self.assertFalse(verdict.hallucination_detected)

    def test_model_backend_unverified_quote_flags_hallucination(self):
        """A model verdict citing a quote the transcript does not contain is a
        hallucination: the verdict is withheld and the check is flagged."""
        turns = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=0.0, end_time=4.0, text="Is there anything else I can help you with today?"),
        ]
        transcript = CallTranscript(call_id="sem_model_halluc", turns=turns, duration_seconds=10.0)

        rubric = self._semantic_rubric(
            pass_when="The agent offers further assistance before closing.",
            fail_when="The agent hangs up without offering help.",
        )

        def fake_backend(prompt):
            return '{"assessment":"Test assessment.","verdict": "pass", "quote": "The agent offered a discount on the account."}'

        res = RubricEvaluator(rubric, model_backend=fake_backend).evaluate_deterministic(transcript)
        verdict = res.verdicts[0]

        self.assertEqual(verdict.status, VerdictStatus.FLAGGED)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIn("does not appear verbatim", verdict.reasoning)
        self.assertTrue(res.requires_human_review)

    def test_model_backend_failure_flags_for_review(self):
        """A failing model backend must not produce a guessed verdict."""
        turns = [
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=0.0, end_time=4.0, text="Thank you for calling."),
        ]
        transcript = CallTranscript(call_id="sem_model_fail", turns=turns, duration_seconds=10.0)

        rubric = self._semantic_rubric(
            pass_when="The agent offers further assistance before closing.",
            fail_when="The agent hangs up without offering help.",
        )

        def broken_backend(prompt):
            raise RuntimeError("model down")

        res = RubricEvaluator(rubric, model_backend=broken_backend).evaluate_deterministic(transcript)
        verdict = res.verdicts[0]

        self.assertEqual(verdict.status, VerdictStatus.FLAGGED)
        self.assertEqual(verdict.confidence, 0.0)
        self.assertIn("model backend failed", verdict.reasoning)


if __name__ == "__main__":
    unittest.main()
