import unittest

from call1.models.schemas import TranscriptTurn, SpeakerRole, WordTimestamp
from call1.pipeline.diarization import attach_clusters


class TestClusterAlignment(unittest.TestCase):
    def turn(self, words=None):
        return TranscriptTurn(turn_id=0, speaker=SpeakerRole.UNKNOWN, start_time=0, end_time=3,
                              text='Hello there. Yes.', raw_text='Hello there. Yes.', word_timestamps=words)

    def test_splits_on_voice_change_without_inventing_roles_or_losing_words(self):
        words = [WordTimestamp(word=text, start_time=i, end_time=i+1, probability=.9)
                 for i, text in enumerate(['Hello', ' there.', ' Yes.'])]
        output = attach_clusters([self.turn(words)], [
            {'start': 0, 'end': 2, 'speaker': 0}, {'start': 2, 'end': 3, 'speaker': 1}])
        self.assertEqual([t.speaker_cluster for t in output], ['speaker_1', 'speaker_2'])
        self.assertEqual([t.text for t in output], ['Hello there.', 'Yes.'])
        self.assertEqual([w for t in output for w in t.word_timestamps], words)
        self.assertTrue(all(t.speaker == SpeakerRole.UNKNOWN for t in output))
        self.assertEqual([t.turn_id for t in output], [0, 1])

    def test_overlapping_voices_and_weak_coverage_remain_unassigned(self):
        for spans in [[{'start': 0, 'end': 3, 'speaker': 0}, {'start': 0, 'end': 3, 'speaker': 1}],
                      [{'start': 0, 'end': .5, 'speaker': 0}], []]:
            output = attach_clusters([self.turn()], spans)
            self.assertIsNone(output[0].speaker_cluster)
            self.assertEqual(output[0].text, 'Hello there. Yes.')

    def test_incomplete_word_alignment_never_discards_transcript_text(self):
        words = [WordTimestamp(word='Yes.', start_time=2, end_time=3, probability=.9)]
        output = attach_clusters([self.turn(words)], [{'start': 0, 'end': 3, 'speaker': 0}])
        self.assertEqual(output[0].text, 'Hello there. Yes.')
        self.assertEqual(output[0].word_timestamps, words)
        self.assertEqual(output[0].speaker_cluster, 'speaker_1')
