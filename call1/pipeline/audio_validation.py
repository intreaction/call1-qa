"""Audio file validator for pre-flight ingestion checks in Call1.

Shared by the legacy ingest path (``call1.ingest.validator`` re-exports it unchanged) and the split
Process app's ``validation_vad`` stage, which must not import ``call1.ingest``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional, Set

from call1.models.schemas import (
    AudioChannelLayout,
    AudioValidationStatus,
    ValidationResult,
)

# Supported audio codecs for contact center recordings
SUPPORTED_CODECS: Set[str] = {
    "pcm_s16le",
    "pcm_s16be",
    "pcm_u8",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_mulaw",
    "pcm_alaw",
    "mp3",
    "aac",
    "opus",
    "flac",
    "vorbis",
}


class AudioValidator:
    """Pre-flight validator checking format, codec, duration, and channel integrity."""

    def __init__(
        self,
        min_duration_seconds: float = 5.0,
        max_duration_seconds: float = 5400.0,  # 90 minutes
        allowed_codecs: Optional[Set[str]] = None,
        ffprobe_bin: Optional[str] = None,
    ):
        self.min_duration = min_duration_seconds
        self.max_duration = max_duration_seconds
        self.allowed_codecs = allowed_codecs or SUPPORTED_CODECS
        self.ffprobe_bin = ffprobe_bin or shutil.which("ffprobe")

    def validate(self, file_path: str | Path) -> ValidationResult:
        """Run deterministic validation gates on an audio file."""
        path = Path(file_path)

        # Gate 1: Existence and non-zero size check
        if not path.is_file():
            return ValidationResult(
                is_valid=False,
                status=AudioValidationStatus.FILE_NOT_FOUND,
                file_path=str(path),
                error_message=f"File not found: {path}",
            )

        file_size = path.stat().st_size
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                status=AudioValidationStatus.CORRUPTED,
                file_path=str(path),
                file_size_bytes=0,
                error_message="File is 0 bytes (empty upload).",
            )

        # Gate 2: Header and stream probe (ffprobe preferred, wave fallback)
        probe_meta = self._probe_audio(path)
        if not probe_meta:
            return ValidationResult(
                is_valid=False,
                status=AudioValidationStatus.CORRUPTED,
                file_path=str(path),
                file_size_bytes=file_size,
                error_message="Audio container corrupted or unreadable header.",
            )

        codec = probe_meta.get("codec", "unknown").lower()
        duration = float(probe_meta.get("duration", 0.0))
        channels = int(probe_meta.get("channels", 0))
        sample_rate = int(probe_meta.get("sample_rate", 0))

        # Determine channel layout
        if channels == 1:
            channel_layout = AudioChannelLayout.MONO
        elif channels == 2:
            channel_layout = AudioChannelLayout.STEREO
        elif channels > 2:
            channel_layout = AudioChannelLayout.MULTI_CHANNEL
        else:
            channel_layout = AudioChannelLayout.UNKNOWN

        # Gate 3: Codec / Format whitelist check
        if codec not in self.allowed_codecs:
            return ValidationResult(
                is_valid=False,
                status=AudioValidationStatus.UNSUPPORTED_FORMAT,
                file_path=str(path),
                file_size_bytes=file_size,
                duration_seconds=duration,
                channels=channels,
                channel_layout=channel_layout,
                sample_rate=sample_rate,
                codec=codec,
                error_message=f"Unsupported codec '{codec}'. Allowed: {sorted(self.allowed_codecs)}",
            )

        # Gate 4: Duration bounds
        if duration < self.min_duration:
            return ValidationResult(
                is_valid=False,
                status=AudioValidationStatus.DISCARD_TOO_SHORT,
                file_path=str(path),
                file_size_bytes=file_size,
                duration_seconds=duration,
                channels=channels,
                channel_layout=channel_layout,
                sample_rate=sample_rate,
                codec=codec,
                error_message=f"Call duration ({duration:.2f}s) < minimum threshold ({self.min_duration}s).",
            )

        if duration > self.max_duration:
            return ValidationResult(
                is_valid=False,
                status=AudioValidationStatus.EXCEEDS_MAX_DURATION,
                file_path=str(path),
                file_size_bytes=file_size,
                duration_seconds=duration,
                channels=channels,
                channel_layout=channel_layout,
                sample_rate=sample_rate,
                codec=codec,
                error_message=f"Call duration ({duration:.2f}s) exceeds maximum threshold ({self.max_duration}s).",
            )

        # All validation gates passed
        return ValidationResult(
            is_valid=True,
            status=AudioValidationStatus.PASSED,
            file_path=str(path),
            file_size_bytes=file_size,
            duration_seconds=duration,
            channels=channels,
            channel_layout=channel_layout,
            sample_rate=sample_rate,
            codec=codec,
        )

    def _probe_audio(self, path: Path) -> Optional[dict]:
        """Probe audio file metadata using ffprobe or standard wave module."""
        if self.ffprobe_bin:
            try:
                cmd = [
                    self.ffprobe_bin,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration,size,format_name:stream=codec_name,channels,sample_rate",
                    "-of",
                    "json",
                    str(path),
                ]
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout:
                    data = json.loads(proc.stdout)
                    streams = data.get("streams", [])
                    fmt = data.get("format", {})

                    audio_stream = next(
                        (s for s in streams if "codec_name" in s and "channels" in s),
                        None,
                    )

                    duration = fmt.get("duration")
                    if not duration and audio_stream:
                        duration = audio_stream.get("duration")

                    if audio_stream and duration:
                        return {
                            "codec": audio_stream.get("codec_name"),
                            "channels": int(audio_stream.get("channels", 1)),
                            "sample_rate": int(audio_stream.get("sample_rate", 8000)),
                            "duration": float(duration),
                        }
            except Exception:
                pass  # Fallback to wave module

        # Fallback for standard PCM WAV files
        try:
            with wave.open(str(path), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / float(sample_rate) if sample_rate > 0 else 0.0
                sampwidth = wf.getsampwidth()
                codec_map = {1: "pcm_u8", 2: "pcm_s16le", 3: "pcm_s24le", 4: "pcm_s32le"}
                codec = codec_map.get(sampwidth, "pcm_s16le")
                return {
                    "codec": codec,
                    "channels": channels,
                    "sample_rate": sample_rate,
                    "duration": duration,
                }
        except Exception:
            return None


def validate_audio_file(file_path: str | Path, **kwargs) -> ValidationResult:
    """Convenience functional interface for AudioValidator."""
    validator = AudioValidator(**kwargs)
    return validator.validate(file_path)
