"""Actual codec decoding must retain timing metrics, including silent intervals."""
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from call1.pipeline.audio_processor import VoiceActivityDetector


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for codec integration checks')
class VADFormatTests(unittest.TestCase):
    def test_flac_and_mp3_retain_pcm_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pcm = root/'source.wav'
            samples = [int(12000*math.sin(2*math.pi*440*i/16000)) if 16000<=i<48000 else 0 for i in range(80000)]
            with wave.open(str(pcm),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
                wav.writeframes(struct.pack('<'+'h'*len(samples),*samples))
            detector = VoiceActivityDetector()
            reference = detector.analyze(pcm)
            self.assertGreater(reference.total_speech_duration,1.5)
            self.assertGreater(reference.total_silence_duration,2.5)
            for extension,codec in [('flac','flac'),('mp3','libmp3lame')]:
                encoded = root/f'audio.{extension}'
                subprocess.run(['ffmpeg','-v','error','-i',str(pcm),'-c:a',codec,str(encoded)],check=True)
                result = detector.analyze(encoded)
                self.assertAlmostEqual(result.silence_ratio,reference.silence_ratio,delta=.03)
                self.assertGreater(len(result.segments),0)

    def test_decode_failure_does_not_publish_zero_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'broken.wav';path.write_bytes(b'invalid recording')
            with self.assertRaises(subprocess.CalledProcessError):
                VoiceActivityDetector().analyze(path)
