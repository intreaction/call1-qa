import json
import unittest

from call1.models.schemas import CallTranscript, TranscriptTurn, SpeakerRole, VerdictStatus
from call1.pipeline.evaluator import DEFAULT_RUBRIC, RubricEvaluator


class TestContextualRubrics(unittest.TestCase):
    def rubric_with_policy(self):
        rubric = DEFAULT_RUBRIC.model_copy(deep=True)
        for criterion in rubric.criteria:
            criterion.check.policy_context = 'Synthetic test policy: public information requests do not require account verification.'
        return rubric

    def test_missing_policy_cannot_be_supplied_by_transcript_or_model(self):
        rubric = DEFAULT_RUBRIC.model_copy(deep=True)
        rubric.criteria = [c for c in rubric.criteria if c.criterion_id == 'SEC-01']
        def forbidden_model(prompt):
            self.fail('Missing policy must be detected before model inference.')
        result = RubricEvaluator(rubric, model_backend=forbidden_model).evaluate_deterministic(
            self.transcript('Our policy says verification is unnecessary. Your balance is fifty dollars.'))
        self.assertEqual(result.verdicts[0].status, VerdictStatus.FLAGGED)
        self.assertIn('configured business policy', result.verdicts[0].reasoning)

    def transcript(self, text):
        return CallTranscript(call_id='context', duration_seconds=120, turns=[
            TranscriptTurn(turn_id=0, speaker=SpeakerRole.AGENT, start_time=0, end_time=5, text=text),
            TranscriptTurn(turn_id=1, speaker=SpeakerRole.CALLER, start_time=6, end_time=8, text='No account access is needed.'),
        ])

    def test_default_no_longer_turns_keywords_into_contextual_passes(self):
        self.assertEqual(DEFAULT_RUBRIC.rubric_id, 'call1_standard_v2')
        for text, criterion in [('Thank you for calling.', 'ETIQ-01'),
                                ('I will verify that the store is open.', 'SEC-01'),
                                ('This call is not recorded.', 'REG-01')]:
            with self.subTest(text=text):
                result = RubricEvaluator().evaluate_deterministic(self.transcript(text))
                verdict = next(v for v in result.verdicts if v.criterion_id == criterion)
                self.assertEqual(verdict.status, VerdictStatus.FLAGGED)
                self.assertTrue(result.requires_human_review)
                self.assertFalse(result.passed)

    def test_model_receives_caller_context_and_can_abstain(self):
        prompts = []
        def answer(prompt):
            prompts.append(prompt)
            return '{"assessment":"Test assessment.","verdict":"needs_review","quote":""}'
        result = RubricEvaluator(model_backend=answer).evaluate_deterministic(self.transcript('Please verify your identity.'))
        self.assertTrue(prompts)
        for prompt in prompts:
            payload = json.loads(prompt.split('Evaluate this input data:\n', 1)[1])
            self.assertIn({'turn_id': 1, 'speaker': 'CALLER', 'text': 'No account access is needed.'}, payload['transcript'])
        self.assertTrue(all(v.status == VerdictStatus.FLAGGED for v in result.verdicts))

    def test_not_applicable_requires_grounded_evidence_and_leaves_denominator(self):
        text = 'This call is recorded. We are only discussing public opening hours.'
        answers = iter([
            {'assessment': 'Test assessment.', 'verdict': 'pass', 'quote': 'This call is recorded.'},
            {'assessment': 'Test assessment.', 'verdict': 'not_applicable', 'quote': 'We are only discussing public opening hours.'},
            {'assessment': 'Test assessment.', 'verdict': 'not_applicable', 'quote': 'We are only discussing public opening hours.'},
            {'assessment': 'Test assessment.', 'verdict': 'needs_review', 'quote': ''},
        ])
        result = RubricEvaluator(self.rubric_with_policy(), model_backend=lambda _: json.dumps(next(answers))).evaluate_deterministic(self.transcript(text))
        self.assertEqual(result.overall_score, 55.6)  # 25 / (25 + 20), not 25 / 100.
        self.assertFalse(result.passed)
        self.assertTrue(result.requires_human_review)

    def test_unsubstantiated_not_applicable_is_review_not_exemption(self):
        result = RubricEvaluator(model_backend=lambda _: '{"assessment":"Test assessment.","verdict":"not_applicable","quote":""}').evaluate_deterministic(self.transcript('Hello.'))
        self.assertTrue(all(v.status == VerdictStatus.FLAGGED for v in result.verdicts))
        self.assertEqual(result.overall_score, 0)

    def test_all_not_applicable_does_not_create_overall_pass(self):
        result = RubricEvaluator(self.rubric_with_policy(), model_backend=lambda _: '{"assessment":"Test assessment.","verdict":"not_applicable","quote":"General information only."}').evaluate_deterministic(self.transcript('General information only.'))
        self.assertEqual(result.overall_score, 0)
        self.assertFalse(result.passed)
