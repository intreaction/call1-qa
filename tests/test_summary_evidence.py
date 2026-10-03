"""Long-summary synthesis must retain valid original evidence or fail explicitly."""
import unittest
from unittest.mock import patch
from call1.models.schemas import AppSettings, SpeakerRole
from call1.summarizer import generate_summary


class SummaryEvidenceTests(unittest.TestCase):
    def test_lost_synthesis_citations_keep_checked_beginning_and_end(self):
        settings = AppSettings()
        settings.summary.batch_turns = 1
        turns = [(10,SpeakerRole.AGENT,'Account balance is fifty dollars.'),
                 (90,SpeakerRole.CALLER,'Payment transfer is confirmed today.')]
        replies = [
            {'narrative':'Account balance discussed.','key_points':['Account balance is fifty dollars (turn 10)']},
            {'narrative':'Payment transfer confirmed.','key_points':['Payment transfer is confirmed (turn 90)']},
            {'narrative':'Account balance and payment transfer discussed.','key_points':['Unsupported statement (turn 999)']},
        ]
        with patch('call1.summarizer._generate_one',side_effect=replies):
            summary = generate_summary(settings,turns,[],False)
        self.assertEqual(summary.key_points,[replies[0]['key_points'][0],replies[1]['key_points'][0]])
        self.assertTrue(summary.grounding['key_points_from_source_segments'])

    def test_no_source_evidence_still_fails(self):
        settings = AppSettings()
        settings.summary.batch_turns = 1
        turns = [(10,SpeakerRole.AGENT,'Account balance is fifty dollars.'),
                 (90,SpeakerRole.CALLER,'Payment transfer is confirmed today.')]
        reply = {'narrative':'Unverified narrative.','key_points':['No support (turn 999)']}
        with patch('call1.summarizer._generate_one',side_effect=lambda *a:dict(reply)):
            with self.assertRaisesRegex(RuntimeError,'No key points survived'):
                generate_summary(settings,turns,[],False)

    def test_segment_cannot_borrow_another_segments_citations(self):
        settings = AppSettings()
        settings.summary.batch_turns = 1
        turns = [(10,SpeakerRole.AGENT,'Account balance is fifty dollars.'),
                 (90,SpeakerRole.CALLER,'Payment transfer is confirmed today.')]
        replies = [
            {'narrative':'Wrong segment.','key_points':['Payment transfer is confirmed (turn 90)']},
            {'narrative':'Wrong segment.','key_points':['Account balance is fifty dollars (turn 10)']},
            {'narrative':'No source evidence.','key_points':['No support (turn 999)']},
        ]
        with patch('call1.summarizer._generate_one',side_effect=replies):
            with self.assertRaisesRegex(RuntimeError,'No key points survived'):
                generate_summary(settings,turns,[],False)

    def test_truncated_json_retries_once_without_truncating_source_or_raising_budget(self):
        from call1.summarizer import _generate_one
        settings = AppSettings()
        source = 'Full transcript including the final outcome.'
        good = '{"narrative":"Payment completed.","key_points":["Payment completed (turn 90)"]}'
        with patch('call1.summarizer._call_ollama',side_effect=['{"narrative":"unfinished',good]) as generate:
            result = _generate_one(settings,'Summarize.',source)
        self.assertEqual(result['narrative'],'Payment completed.')
        self.assertEqual(generate.call_count,2)
        for call in generate.call_args_list:
            self.assertEqual(call.args[3],source)
            self.assertEqual(call.args[4],settings.summary.max_tokens)
        self.assertIn('at most three original turn numbers',generate.call_args_list[1].args[2])

    def test_persistent_invalid_output_remains_a_failure(self):
        from call1.summarizer import _generate_one
        with patch('call1.summarizer._call_ollama',return_value='unfinished') as generate:
            with self.assertRaisesRegex(RuntimeError,'unparseable'):
                _generate_one(AppSettings(),'Summarize.','Entire source')
        self.assertEqual(generate.call_count,2)

    def test_model_failure_is_not_retried_as_formatting(self):
        from call1.summarizer import _generate_one
        with patch('call1.summarizer._call_ollama',side_effect=RuntimeError('model unavailable')) as generate:
            with self.assertRaisesRegex(RuntimeError,'model unavailable'):
                _generate_one(AppSettings(),'Summarize.','Entire source')
        self.assertEqual(generate.call_count,1)


def test_a_last_key_point_missing_its_closing_quote_is_repaired_and_truncation_still_rejected():
    import pytest
    from call1.summarizer import parse_summary_answer

    raw = ('{ "narrative": "The call involved a stock inquiry.", "key_points": [ "Three units were in stock (turn 21)", '
           '"Caller confirmed online completion (66) ] }')
    assert parse_summary_answer(raw)["key_points"][-1] == "Caller confirmed online completion (66)"
    with pytest.raises(RuntimeError):
        parse_summary_answer('{"narrative": "x", "key_points": ["a (1)"')
