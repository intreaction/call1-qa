"""What Process can tell about a recording before any model runs: container, content type, and
for WAV the channel count, sample rate and duration (``wave`` from the standard library).

The graph planner uses the channel count to decide whether the call needs mono speaker
attribution; the ``validation_vad`` job is still what publishes the authoritative media fields.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
}
"""The containers the contract's ``audio.v1`` accepts."""

_ALIASES = {"audio/x-wav": "wav", "audio/wave": "wav", "audio/vnd.wave": "wav", "audio/wav": "wav", "audio/mpeg": "mp3",
            "audio/mp3": "mp3", "audio/flac": "flac", "audio/x-flac": "flac", "audio/ogg": "ogg", "audio/mp4": "m4a",
            "audio/x-m4a": "m4a", "audio/m4a": "m4a"}


class UnsupportedAudio(ValueError):
    pass


@dataclass(frozen=True)
class AudioInfo:
    container: str
    content_type: str
    size_bytes: int
    channels: Optional[int] = None
    sample_rate: Optional[int] = None
    duration_seconds: Optional[float] = None

    @property
    def mono(self) -> bool:
        """True unless the recording is known to have more than one channel. An unknown layout
        gets speaker attribution, which is harmless on stereo audio."""
        return self.channels is None or self.channels == 1


def _sniff(head: bytes) -> Optional[str]:
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "mp3"
    if head[4:8] == b"ftyp":
        return "m4a"
    return None


def probe(path: Path, *, filename: Optional[str] = None, content_type: Optional[str] = None) -> AudioInfo:
    path = Path(path)
    with path.open("rb") as handle:
        head = handle.read(64)
    container = _sniff(head)
    if container is None and content_type:
        container = _ALIASES.get(content_type.split(";")[0].strip().lower())
    if container is None and filename:
        ext = Path(filename).suffix.lower().lstrip(".")
        container = ext if ext in CONTENT_TYPES else None
    if container is None:
        raise UnsupportedAudio("Unsupported recording: send WAV, MP3, FLAC, OGG or M4A audio")
    size = path.stat().st_size
    if size == 0:
        raise UnsupportedAudio("The recording is empty")
    channels = sample_rate = None
    duration = None
    if container == "wav":
        try:
            with wave.open(str(path), "rb") as w:
                channels = w.getnchannels()
                sample_rate = w.getframerate()
                duration = w.getnframes() / float(sample_rate) if sample_rate else None
        except (wave.Error, EOFError):
            pass  # compressed WAV (not PCM): validation_vad reports it
    return AudioInfo(container=container, content_type=CONTENT_TYPES[container], size_bytes=size, channels=channels,
                     sample_rate=sample_rate, duration_seconds=duration)
