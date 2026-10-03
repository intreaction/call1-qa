"""Offline speech recognition with real word alignment and explicit speaker roles."""

from __future__ import annotations

import gc
import os
import subprocess
import tempfile
from pathlib import Path

from call1.models.schemas import SpeakerRole, TranscriptTurn, WordTimestamp
from call1.pipeline.inference import inference_lock


class LocalTranscriber:
    """Small INT8 Whisper, one recording/channel at a time.

    Models must be provisioned before processing; inference never downloads a
    model. Mono speakers remain UNKNOWN; no invented diarization or timestamps.
    Stereo channels require the recorder's agent channel mapping (default 0).
    """

    def transcribe(self, path: str, channels: int, agent_channel: int = 0) -> list[TranscriptTurn]:
        if os.getenv("CALL1_BACKEND") == "mlx":
            from call1.adapters import get_adapter
            return get_adapter().transcribe(path, channels, agent_channel)
        if channels not in (1, 2):
            raise ValueError("Only mono or dual-channel recordings are supported.")
        if agent_channel not in (0, 1):
            raise ValueError("Agent channel must be 0 or 1.")
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to enable local transcription.") from exc

        device = os.getenv("CALL1_ASR_DEVICE", "cpu")
        if device not in ("cpu", "cuda"):
            raise ValueError("CALL1_ASR_DEVICE must be cpu or cuda.")
        model_name = os.getenv("CALL1_ASR_MODEL", "small")
        turns: list[TranscriptTurn] = []
        with inference_lock:
            model = WhisperModel(
                model_name, device=device,
                compute_type="int8_float16" if device == "cuda" else "int8",
                cpu_threads=4, num_workers=1, local_files_only=True,
            )
            try:
                with tempfile.TemporaryDirectory(prefix="call1-asr-") as temp:
                    for channel in range(channels):
                        normalized = Path(temp) / f"channel-{channel}.wav"
                        proc = subprocess.run(
                            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                             "-af", f"pan=mono|c0=c{channel}", "-ar", "16000",
                             "-c:a", "pcm_s16le", str(normalized)],
                            capture_output=True, timeout=180,
                        )
                        if proc.returncode:
                            raise RuntimeError("Could not decode the recording for transcription.")
                        segments, _ = model.transcribe(
                            str(normalized), beam_size=5, word_timestamps=True,
                            vad_filter=True, condition_on_previous_text=False,
                            language=os.getenv("CALL1_ASR_LANGUAGE") or None,
                        )
                        role = (SpeakerRole.UNKNOWN if channels == 1 else
                                SpeakerRole.AGENT if channel == agent_channel else SpeakerRole.CALLER)
                        for segment in segments:
                            if not segment.text.strip():
                                continue
                            words = [WordTimestamp(
                                word=w.word, start_time=w.start, end_time=w.end,
                                probability=w.probability,
                            ) for w in (segment.words or [])]
                            turns.append(TranscriptTurn(
                                turn_id=0, speaker=role, channel=channel,
                                start_time=segment.start, end_time=segment.end,
                                text=segment.text.strip(), raw_text=segment.text.strip(),
                                word_timestamps=words,
                                confidence=(sum(w.probability for w in words) / len(words)) if words else None,
                            ))
            finally:
                # Release ASR before the next local LLM request on small GPUs.
                del model
                gc.collect()
        turns.sort(key=lambda t: (t.start_time, t.channel or 0))
        for index, turn in enumerate(turns):
            turn.turn_id = index
        if not turns:
            raise RuntimeError("No speech was transcribed; recording needs manual review.")
        return turns
