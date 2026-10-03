import tempfile
import unittest
import wave
from pathlib import Path
from call1.adapters.mlx import _has_pcm_signal


class PCMSilenceTests(unittest.TestCase):
    def check(self, samples):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'audio.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(samples)
            return _has_pcm_signal(str(path))

    def test_digital_silence_is_rejected(self):
        self.assertFalse(self.check(b'\0\0'*160000))

    def test_signal_after_silent_opening_is_kept(self):
        self.assertTrue(self.check(b'\0\0'*160000+b'\x01\0'))

    def test_negative_quiet_sample_is_kept(self):
        self.assertTrue(self.check(b'\xff\xff'))
