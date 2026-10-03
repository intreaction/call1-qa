import math
import unittest

from scripts.evaluate_diarization import diarization_errors


def span(start, end, speaker):
    return {'start': start, 'end': end, 'speaker': speaker}


def test_global_mapping_ignores_anonymous_label_order():
    result = diarization_errors([span(0, 5, 'A'), span(5, 10, 'B')],
                                [span(0, 5, 1), span(5, 10, 0)], 10)
    assert result['der'] == 0
    assert result['hypothesis_to_reference_mapping'] == {'0': 'B', '1': 'A'}


def test_mid_call_identity_switch_counts_as_confusion():
    result = diarization_errors([span(0, 10, 'A'), span(10, 20, 'B')],
                                [span(0, 5, 0), span(5, 15, 1), span(15, 20, 0)], 20)
    assert result['confused_speaker_seconds'] == 10
    assert result['der'] == .5


def test_overlap_miss_and_false_alarm_are_counted_separately():
    result = diarization_errors([span(0, 10, 'A'), span(5, 10, 'B')],
                                [span(0, 12, 0)], 12)
    assert result['reference_speaker_seconds'] == 15
    assert result['missed_speaker_seconds'] == 5
    assert result['false_alarm_speaker_seconds'] == 2
    assert result['confused_speaker_seconds'] == 0
    assert math.isclose(result['der'], 7 / 15)


def test_duplicate_same_speaker_segments_do_not_double_count():
    result = diarization_errors([span(0, 10, 'A')],
                                [span(0, 8, 0), span(5, 10, 0)], 10)
    assert result['der'] == 0


def test_silent_reference_has_undefined_der_and_records_false_alarm():
    result = diarization_errors([], [span(0, 5, 0)], 10)
    assert result['der'] is None
    assert result['false_alarm_speaker_seconds'] == 5


def test_missing_hypothesis_and_invalid_timestamps():
    assert diarization_errors([span(0, 10, 'A')], [], 10)['der'] == 1
    with unittest.TestCase().assertRaises(ValueError):
        diarization_errors([span(5, 2, 'A')], [], 10)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, function in globals().copy().items():
        if name.startswith("test_") and callable(function):
            suite.addTest(unittest.FunctionTestCase(function))
    return suite
