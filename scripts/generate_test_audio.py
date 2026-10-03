"""Generate non-speech PCM fixtures for fake-handler tests, never ASR evaluation."""
from __future__ import annotations

import array
import hashlib
import json
import math
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    directory = ROOT / 'sample_audio'
    manifest = json.loads((directory / 'manifest.json').read_text())
    marker = directory / '.test-audio.json'
    if marker.exists():
        hashes = json.loads(marker.read_text())
        if all((directory / name).exists() and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items()):
            print('Generated test tones already present; unchanged.')
            return
        raise SystemExit('Test fixtures changed; inspect them before regenerating.')
    paths = [(name, spec, directory / spec['file_name']) for name, spec in manifest.items()]
    if any(path.exists() for _, _, path in paths):
        raise SystemExit('Refusing to overwrite existing recordings. Use a clean source checkout.')
    hashes = {}
    for index, (_, spec, path) in enumerate(paths):
        rate, channels = 16000, spec['channels']
        samples = array.array('h')
        for frame in range(round(spec['duration_seconds'] * rate)):
            t = frame / rate
            amplitude = 3600 if t % 1.2 < 0.9 else 0
            for channel in range(channels):
                samples.append(round(amplitude * math.sin(2 * math.pi * (220 + 35 * index + 90 * channel) * t)))
        if sys.byteorder != 'little':
            samples.byteswap()
        with wave.open(str(path), 'wb') as out:
            out.setnchannels(channels)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(samples.tobytes())
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    marker.write_text(json.dumps(hashes, indent=2) + '\n')
    print('Created five non-speech fixtures for fake-handler tests only.')


if __name__ == '__main__':
    main()
