"""The Parakeet ASR adapter's pure helpers (no MLX needed): word grouping, number read-outs,
silence-trimmed windows and the packed-stream clock."""
import struct
import wave
from types import SimpleNamespace

import pytest

from call1.adapters.mlx import (_PackedClock, _code_values, _digit_readouts, _parakeet_segments,
                                _parakeet_words, _pause_windows)
from call1.models.schemas import WordTimestamp


def token(text, start, duration):
    return SimpleNamespace(text=text, start=start, duration=duration, end=start + duration)


def words(text):
    return [WordTimestamp(word=" " + w, start_time=i, end_time=i + .5) for i, w in enumerate(text.split())]


def joined(ws):
    return "".join(w.word for w in ws).strip()


def test_subword_tokens_become_words_and_late_punctuation_keeps_the_word_end():
    tokens = [token(" My", 3.76, .16), token(" Sam", 4.4, .32), token("ant", 4.72, .32), token("ha", 5.04, .32),
              token(".", 16.88, .24)]
    out = _parakeet_words(tokens)
    assert [w.word for w in out] == [" My", " Samantha."]
    assert out[1].start_time == pytest.approx(4.4) and out[1].end_time == pytest.approx(5.36)


def test_sentences_and_long_pauses_close_turns():
    sentence = SimpleNamespace(tokens=[token(" Hi", 0, .2), token(" there", .3, .2), token(" again", 2.0, .2), token(".", 2.2, .1)])
    result = SimpleNamespace(sentences=[sentence, SimpleNamespace(tokens=[token(" Bye", 2.5, .2)])])
    assert [joined(t) for t in _parakeet_segments(result)] == ["Hi there", "again.", "Bye"]


@pytest.mark.parametrize("text, expected", [
    ("SSN is four four two eight nine one zero nine nine, and", "SSN is 442-89-1099, and"),
    ("phone four eight zero, five five five, zero one nine nine.", "phone 480-555-0199."),
    ("it is 4111, 2222, 3333, 4444.", "it is 4111-2222-3333-4444."),
    ("account 902, 114, 883.", "account 902-114-883."),
    ("SSN 442-891099 ok", "SSN 442-89-1099 ok"),
    ("account 884-210-993.", "account 884-210-993."),
    ("born May 12, 1984.", "born May 12, 1984."),
    ("pay $14,250 or 5,000 now", "pay $14,250 or 5,000 now"),
    ("oh one two three", "oh 123"),
    ("two two apples", "two two apples"),
    ("in 2024 2025 2026 we grew", "in 2024 2025 2026 we grew"),
    ("scores of 10, 20, 30 today", "scores of 10, 20, 30 today"),
    ("card 4111 2222 3333 4444 ok", "card 4111-2222-3333-4444 ok"),
])
def test_number_read_outs_become_whisper_style_digits(text, expected):
    assert joined(_digit_readouts(words(text))) == expected


def test_a_merged_read_out_spans_its_words():
    out = _digit_readouts(words("is four four two eight nine one zero nine nine"))
    assert out[1].word == " 442-89-1099" and (out[1].start_time, out[1].end_time) == (1, 9.5)


def test_a_code_written_as_currency_is_plain_digits():
    assert joined(_code_values(words("pin is $4,492. costs $4,492."))) == "pin is 4492. costs $4,492."


def _wav(path, pieces, rate=16000):
    samples = []
    for seconds, amplitude in pieces:
        n = int(seconds * rate)
        samples += [amplitude if i % 2 else -amplitude for i in range(n)]
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return str(path)


def test_windows_cut_long_silence_and_trim_silent_edges(tmp_path):
    path = _wav(tmp_path / "c.wav", [(3, 0), (2, 3000), (6, 0), (1, 3000), (0.5, 0)])
    windows = [(round(a / 16000, 2), round(b / 16000, 2)) for a, b in _pause_windows(path)]
    assert windows == [(2.49, 5.52), (10.47, 12.5)]  # ~0.5 s of padding (30 ms frames) around each utterance
    assert _pause_windows(_wav(tmp_path / "silent.wav", [(4, 0)])) == []


def test_quiet_but_audible_audio_is_never_trimmed(tmp_path):
    path = _wav(tmp_path / "q.wav", [(2, 3000), (3, 100), (2, 3000)])  # 100/32768 ~ -50 dBFS: cut, not dropped
    windows = _pause_windows(path)
    assert windows[0][0] == 0 and windows[-1][1] == 7 * 16000 and len(windows) == 2 and windows[0][1] == windows[1][0]


def test_the_packed_clock_maps_back_to_the_recording():
    clock = _PackedClock([(16000, 48000), (160000, 176000)], 16000)  # 1-3 s and 10-11 s
    assert clock(0.5) == pytest.approx(1.5) and clock(2.5) == pytest.approx(10.5) and clock(9) == pytest.approx(11)
