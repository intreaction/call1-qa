import unittest
from scripts.evaluate_transcripts import errors, words


class TranscriptMetricTests(unittest.TestCase):
    def test_substitution_deletion_and_insertion(self):
        self.assertEqual(errors(['a', 'b'], ['a', 'c'])['substitutions'], 1)
        self.assertEqual(errors(['a', 'b'], ['a'])['deletions'], 1)
        self.assertEqual(errors(['a'], ['a', 'b'])['insertions'], 1)
        self.assertEqual(errors(['a', 'b'], ['a', 'b'])['wer'], 0)

    def test_empty_reference_has_no_defined_rate(self):
        self.assertIsNone(errors([], ['a'])['wer'])
        self.assertEqual(errors([], ['a'])['insertions'], 1)
        self.assertEqual(errors(['a'], [])['wer'], 1)

    def test_normalization_retains_numbers_and_apostrophes(self):
        self.assertEqual(words("I'm HERE, and I’m 42."), ["i'm", 'here', 'and', "i'm", '42'])
