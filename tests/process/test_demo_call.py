import struct
import wave
from call1.process.demo_call import SAMPLE, fresh_sample


def test_fresh_takes_keep_pcm_and_change_container_identity():
    first, second = fresh_sample(), fresh_sample()
    assert first.getvalue() != second.getvalue()
    assert struct.unpack('<I', first.getvalue()[4:8])[0] == len(first.getvalue()) - 8
    with wave.open(str(SAMPLE), 'rb') as source, wave.open(first, 'rb') as take:
        assert source.getparams() == take.getparams()
        assert take.getnframes() / take.getframerate() == 15.5
        assert source.readframes(source.getnframes()) == take.readframes(take.getnframes())
