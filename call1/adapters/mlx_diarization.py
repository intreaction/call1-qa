"""Bounded MLX Nemotron-3-Diarization stage, run only after releasing the ASR model.

Nemotron-3-Diarization (mlx-audio ``nemotron_diarization``) labels up to eight anonymous speakers
on 10 ms frames. Its streaming state counts native 10 ms PCM frames, so segment times are already
on the source clock: unlike the retired Sortformer checkpoint (80 ms frames plus one padded frame
per fed chunk), no clock correction is needed. The recording is streamed from the WAV file in
model-sized windows; ``generate`` keeps a bounded speaker cache (AOSC) and merges segments across
window boundaries, so memory does not grow with call length beyond the per-frame probabilities.
"""
import gc
import wave

from call1.pipeline.inference import inference_lock

SAMPLE_RATE = 16000


def _pcm_windows(wav_path, samples):
    import numpy as np

    with wave.open(str(wav_path), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError('Diarization requires normalized mono 16 kHz PCM16.')
        while block := audio.readframes(samples):
            yield np.frombuffer(block, dtype='<i2').astype(np.float32) / 32768


def diarize(wav_path, model_path):
    """Anonymous speaker segments ``[{'start', 'end', 'speaker'}]`` in source seconds."""
    import mlx.core as mx
    from mlx_audio.vad import load

    with inference_lock:
        model = result = None
        try:
            model = load(model_path, strict=True)
            cfg = model.config
            # One configured streaming window per read; any partition gives the same output.
            window = (cfg.modules_config.chunk_len * cfg.encoder_config.subsampling_factor
                      * model._processor_config.hop_length)
            result = model.generate(_pcm_windows(wav_path, window), sample_rate=SAMPLE_RATE,
                                    threshold=.5, min_duration=0, merge_gap=0)
            return [{'start': float(s.start), 'end': float(s.end), 'speaker': int(s.speaker)}
                    for s in result.segments if s.end > s.start]
        finally:
            del model, result
            gc.collect()
            mx.clear_cache()
