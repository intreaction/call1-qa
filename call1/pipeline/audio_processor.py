"""Audio processor providing Voice Activity Detection (VAD) and timing metrics."""

from __future__ import annotations

import math
import struct
import wave
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from call1.models.schemas import VADMetrics, VADSegment


class VoiceActivityDetector:
    """
    Voice Activity Detection and conversational timing analysis.
    Supports energy-based frame analysis and adapts to dual-channel (stereo) telephony.
    """

    def __init__(
        self,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.015,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
    ):
        self.frame_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.min_speech_ms = min_speech_duration_ms
        self.min_silence_ms = min_silence_duration_ms

    def analyze(self, audio_path: str | Path) -> VADMetrics:
        """Decode supported formats consistently before energy/timing analysis.

        Keep channels separate for overlap metrics. Decode failures propagate to
        the preprocessing stage instead of reporting invented zero metrics.
        """
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory(prefix="call1-vad-") as directory:
            decoded = Path(directory) / "normalized.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                            "-vn", "-ar", "16000", "-c:a", "pcm_s16le", str(decoded)],
                           check=True, capture_output=True, timeout=180)
            return self._analyze_pcm(decoded)

    def _analyze_pcm(self, path: Path) -> VADMetrics:
        with wave.open(str(path), "rb") as wf:
            num_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            num_frames = wf.getnframes()
            total_duration = num_frames / float(sample_rate) if sample_rate > 0 else 0.0
            if total_duration <= 0 or sample_width != 2 or num_channels not in (1, 2):
                raise ValueError("Expected nonempty mono/stereo signed 16-bit PCM.")
            frame_size = max(1, int(sample_rate * self.frame_ms / 1000))
            energies = [[] for _ in range(num_channels)]
            while raw := wf.readframes(frame_size):
                samples = struct.unpack(f"<{len(raw) // 2}h", raw)
                for channel in range(num_channels):
                    frame = samples[channel::num_channels]
                    energies[channel].append(math.sqrt(sum(s*s for s in frame) / len(frame)) / 32767)
        channels = [self._segments_from_energies(values, channel) for channel, values in enumerate(energies)]
        all_segments = sorted([segment for channel in channels for segment in channel], key=lambda s: s.start_time)
        overtalk_dur = self._compute_overtalk(*channels) if num_channels == 2 else 0.0

        # Compute total active speech duration (union of segments)
        total_speech_dur = self._compute_active_speech_union(all_segments)
        total_silence_dur = max(0.0, total_duration - total_speech_dur)
        silence_ratio = total_silence_dur / total_duration if total_duration > 0 else 0.0
        overtalk_ratio = overtalk_dur / total_duration if total_duration > 0 else 0.0

        return VADMetrics(
            total_speech_duration=round(total_speech_dur, 2),
            total_silence_duration=round(total_silence_dur, 2),
            silence_ratio=round(silence_ratio, 4),
            overtalk_duration=round(overtalk_dur, 2),
            overtalk_ratio=round(overtalk_ratio, 4),
            segments=all_segments,
        )

    def _detect_speech_segments(
        self, samples: Tuple[int, ...], sample_rate: int, channel: int
    ) -> List[VADSegment]:
        """Detect speech intervals using short-time RMS energy."""
        frame_size = int(sample_rate * (self.frame_ms / 1000.0))
        if frame_size <= 0:
            return []

        num_frames = len(samples) // frame_size
        if num_frames == 0:
            return []

        # Calculate normalized RMS energy per frame
        energies: List[float] = []
        max_int16 = 32767.0
        for i in range(num_frames):
            frame = samples[i * frame_size : (i + 1) * frame_size]
            sum_sq = sum(s * s for s in frame)
            rms = math.sqrt(sum_sq / float(frame_size)) / max_int16
            energies.append(rms)

        return self._segments_from_energies(energies, channel)

    def _segments_from_energies(self, energies: List[float], channel: int) -> List[VADSegment]:
        if not energies:
            return []
        # Dynamic threshold based on background noise floor
        sorted_energies = sorted(energies)
        noise_floor = sorted_energies[int(len(sorted_energies) * 0.15)]
        adaptive_thresh = max(self.energy_threshold, noise_floor * 2.5)

        is_speech = [e >= adaptive_thresh for e in energies]

        # Aggregate frames into continuous segments
        frame_sec = self.frame_ms / 1000.0
        min_speech_frames = int((self.min_speech_ms / 1000.0) / frame_sec)
        hangover_frames = int((self.min_silence_ms / 1000.0) / frame_sec)

        segments: List[VADSegment] = []
        in_segment = False
        seg_start = 0
        silence_count = 0

        for idx, active in enumerate(is_speech):
            if active:
                if not in_segment:
                    in_segment = True
                    seg_start = idx
                silence_count = 0
            else:
                if in_segment:
                    silence_count += 1
                    if silence_count >= hangover_frames or idx == len(is_speech) - 1:
                        seg_end = idx - silence_count + 1
                        if seg_end - seg_start >= min_speech_frames:
                            start_t = round(seg_start * frame_sec, 2)
                            end_t = round(seg_end * frame_sec, 2)
                            segments.append(
                                VADSegment(
                                    start_time=start_t,
                                    end_time=end_t,
                                    duration=round(end_t - start_t, 2),
                                    channel=channel,
                                )
                            )
                        in_segment = False
                        silence_count = 0

        if in_segment:
            seg_end = len(is_speech)
            if seg_end - seg_start >= min_speech_frames:
                start_t = round(seg_start * frame_sec, 2)
                end_t = round(seg_end * frame_sec, 2)
                segments.append(
                    VADSegment(
                        start_time=start_t,
                        end_time=end_t,
                        duration=round(end_t - start_t, 2),
                        channel=channel,
                    )
                )

        return segments

    def _compute_active_speech_union(self, segments: List[VADSegment]) -> float:
        """Merge overlapping speech intervals and compute total active speech seconds."""
        if not segments:
            return 0.0

        intervals = sorted([(s.start_time, s.end_time) for s in segments])
        merged: List[List[float]] = []
        for start, end in intervals:
            if not merged or merged[-1][1] < start:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        return sum(end - start for start, end in merged)

    def _compute_overtalk(
        self, ch0_segs: List[VADSegment], ch1_segs: List[VADSegment]
    ) -> float:
        """Compute seconds where both channels speak simultaneously."""
        overtalk = 0.0
        for s0 in ch0_segs:
            for s1 in ch1_segs:
                overlap_start = max(s0.start_time, s1.start_time)
                overlap_end = min(s0.end_time, s1.end_time)
                if overlap_end > overlap_start:
                    overtalk += (overlap_end - overlap_start)
        return overtalk

    def _fallback_metrics(self, duration: float) -> VADMetrics:
        """Fallback metrics when detailed PCM inspection is unavailable."""
        speech_est = round(duration * 0.75, 2) if duration > 0 else 0.0
        silence_est = round(duration * 0.25, 2) if duration > 0 else 0.0
        return VADMetrics(
            total_speech_duration=speech_est,
            total_silence_duration=silence_est,
            silence_ratio=0.25 if duration > 0 else 0.0,
            overtalk_duration=0.0,
            overtalk_ratio=0.0,
            segments=[],
        )
