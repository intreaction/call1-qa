"""Local learned text sentiment per turn and acoustic VAD per seven-second block."""
from __future__ import annotations

import gc
import logging
import math
import os
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np

from call1.models.schemas import SpeakerRole, ToneBlock, TranscriptTurn
from call1.pipeline.inference import inference_lock

log = logging.getLogger(__name__)
TEXT_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
TEXT_REVISION = '3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7'
BLOCK_SECONDS = 7
MIN_SPEECH_SECONDS = 1.0  # Operational gate, not a claim of model accuracy.


def default_tone_device():
    """MERaLiON's device when ``CALL1_SENTIMENT_DEVICE`` is unset: ``mps`` where torch has Metal
    (Apple Silicon), else ``cpu``. MPS matched CPU on the six benchmark calls (539 scored blocks:
    max |dV/A/D| 3.4e-6, every emotion argmax equal) at 0.37 s a block against 1.40 s
    (benchmarks/2026-09-27-throughput.md)."""
    try:
        import torch
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    except Exception:
        return 'cpu'


class LocalSentimentModels:
    """Lazy, offline models. One instance per call, released before LLM analysis.

    ``CALL1_SENTIMENT_DEVICE`` sets both models' device. Unset, text sentiment runs on ``cpu`` and
    acoustic tone on ``default_tone_device()`` (MPS on Apple Silicon)."""
    def __init__(self, text_path=None, tone_path=None):
        self.text_model = self.tokenizer = self.tone_model = self.extractor = None
        self._tone_weights = None
        explicit = os.getenv('CALL1_SENTIMENT_DEVICE')
        self.device = explicit or 'cpu'
        self._tone_device = explicit
        # Explicit weight directories (the split Process app passes its catalog's); None keeps the
        # environment variable, then the default, resolved when the model first loads.
        self.text_path = text_path
        self.tone_path = tone_path

    def text(self, text):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self.text_model is None:
            path = Path(self.text_path or os.getenv('CALL1_SENTIMENT_PATH', 'data/models/roberta-sentiment')).resolve()
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval().to(self.device)
        # Cover the entire turn; do not silently truncate long ASR turns.
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        window = min(self.tokenizer.model_max_length, 512) - self.tokenizer.num_special_tokens_to_add()
        total = np.zeros(3, dtype=np.float64)
        for offset in range(0, len(tokens), window):
            chunk = tokens[offset:offset + window]
            ids = [self.tokenizer.bos_token_id, *chunk, self.tokenizer.eos_token_id]
            inputs = torch.tensor([ids], device=self.device)
            with torch.inference_mode():
                probs = self.text_model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits.softmax(-1)[0].cpu().numpy()
            total += probs * len(chunk)
        total /= len(tokens)
        labels = [self.text_model.config.id2label[i].upper() for i in range(3)]
        return dict(zip(labels, map(float, total)))

    @property
    def tone_device(self):
        if self._tone_device is None:
            self._tone_device = default_tone_device()
        return self._tone_device

    def prepare_tone(self):
        """Read MERaLiON's weights into CPU memory (no Metal); ``tone`` then moves them to the device."""
        import torch
        from safetensors import safe_open
        from transformers import WhisperFeatureExtractor
        from call1.adapters.meralion.network import MeralionNetwork
        if self.tone_model is not None or self._tone_weights is not None:
            return
        path = Path(self.tone_path or os.getenv('CALL1_TONE_PATH', 'data/models/meralion-ser-v1')).resolve()
        extractor = WhisperFeatureExtractor.from_pretrained(path, local_files_only=True)
        # Do not construct or load the unused Whisper decoder (~460M weights).
        with torch.device('meta'):
            model = MeralionNetwork()
        with safe_open(path / 'model.safetensors', framework='pt', device='cpu') as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys() if not key.startswith('whisper.decoder.')}
        model.load_state_dict(state, strict=True, assign=True)
        self.extractor, self._tone_weights = extractor, model.eval()

    def tone(self, samples):
        import torch
        if self.tone_model is None:
            self.prepare_tone()
            self.tone_model, self._tone_weights = self._tone_weights.to(self.tone_device), None
        device = self.tone_device
        inputs = self.extractor(samples, sampling_rate=16000, return_tensors='pt')
        with torch.inference_mode():
            dims, logits = self.tone_model(inputs.input_features.to(device))
        values = dims[0].cpu().tolist()
        probabilities = logits.softmax(-1)[0].cpu().tolist()
        if len(values) != 3 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError('Invalid VAD model output')
        return values, dict(zip(['NEUTRAL', 'HAPPY', 'SAD', 'ANGRY', 'FEARFUL', 'DISGUSTED', 'SURPRISED'], probabilities))

    def close(self):
        self.text_model = self.tokenizer = self.tone_model = self.extractor = self._tone_weights = None
        gc.collect()
        if 'mps' in (self.device, self._tone_device):
            import torch
            torch.mps.empty_cache()


def _union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def _subtract(intervals, excluded):
    for a, b in _union(excluded):
        intervals = [(x, min(y, a)) for x, y in intervals if x < a] + [(max(x, b), y) for x, y in intervals if y > b]
    return _union(intervals)


def compute_cohesive_tone(valence: float, arousal: float, dominance: float) -> tuple[float, str]:
    """Combine VAD (Valence, Arousal, Dominance) into a cohesive tonal score (0.0–1.0) and label.

    Blends:
      - Valence (55%): pleasantness / positive affect vs negativity.
      - Dominance (25%): vocal control, confidence, and assertiveness.
      - Arousal equilibrium (20%): optimal composure (sweet spot around 0.50; penalizes agitation/lethargy).
    """
    v = max(0.0, min(1.0, float(valence)))
    a = max(0.0, min(1.0, float(arousal)))
    d = max(0.0, min(1.0, float(dominance)))

    equilibrium = 1.0 - min(1.0, abs(a - 0.5) * 2.0)
    score = round(max(0.0, min(1.0, 0.55 * v + 0.25 * d + 0.20 * equilibrium)), 3)

    if a >= 0.65 and v <= 0.40:
        label = "Agitated & Strained"
    elif score >= 0.68:
        label = "Warm & Engaging"
    elif score >= 0.56:
        label = "Composed & Professional"
    elif score >= 0.45:
        label = "Neutral & Steady"
    elif score >= 0.35:
        label = "Subdued & Hesitant"
    else:
        label = "Tense & Negative"

    return score, label


def average_agent_cohesive_tone(blocks) -> tuple[float | None, str | None]:
    """Compute duration-weighted cohesive tonal score and label across agent blocks."""
    scored = [b for b in blocks if b.speaker == SpeakerRole.AGENT and b.status == 'SCORED' and getattr(b, 'tonal_score', None) is not None]
    duration = sum(b.speech_seconds for b in scored)
    if not duration:
        return None, None
    avg_score = round(sum(b.tonal_score * b.speech_seconds for b in scored) / duration, 3)
    # Derive label from average score
    if avg_score >= 0.68:
        label = "Warm & Engaging"
    elif avg_score >= 0.56:
        label = "Composed & Professional"
    elif avg_score >= 0.45:
        label = "Neutral & Steady"
    elif avg_score >= 0.35:
        label = "Subdued & Hesitant"
    else:
        label = "Tense & Negative"
    return avg_score, label

def _decode_tone_audio(audio_path, temp):
    """The recording as 16 kHz PCM WAV in ``temp`` (ffmpeg; no model, no Metal)."""
    path = Path(temp) / 'audio.wav'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(audio_path),
                    '-vn', '-ar', '16000', '-c:a', 'pcm_s16le', str(path)],
                   capture_output=True, check=True, timeout=180)
    return path


def _plan_tone_blocks(path, turns, agent_channel, disabled=False):
    """Every block with its status, and the (block, isolated samples) pairs the model must score, in
    block order. Pure numpy over the decoded WAV: no model, so it runs outside ``inference_lock``."""
    blocks, pending = [], []
    with wave.open(str(path), 'rb') as wav:
        channels = wav.getnchannels()
        if channels not in (1, 2):
            raise ValueError('Tone analysis requires mono or stereo audio')
        duration = wav.getnframes() / 16000
        for block_id, offset in enumerate(range(0, wav.getnframes(), BLOCK_SECONDS * 16000)):
            samples = np.frombuffer(wav.readframes(BLOCK_SECONDS * 16000), dtype='<i2').reshape(-1, channels).astype(np.float32) / 32768
            start, end = offset / 16000, min(duration, offset / 16000 + BLOCK_SECONDS)
            active = [t for t in turns if t.start_time < end and t.end_time > start]
            for speaker in (SpeakerRole.AGENT, SpeakerRole.CALLER):
                own = [t for t in active if t.speaker == speaker]
                block = ToneBlock(block_id=block_id, speaker=speaker, start_time=start, end_time=end,
                                  turn_ids=[t.turn_id for t in own])
                blocks.append(block)
                if channels == 2:
                    # Turn channels survive manual role corrections. Avoid mixing two channels.
                    selected = {t.channel for t in own if t.channel in (0, 1)}
                    if len(selected) > 1:
                        block.status = 'AMBIGUOUS_SPEAKER'
                        continue
                    channel = next(iter(selected)) if selected else (agent_channel if speaker == SpeakerRole.AGENT else 1 - agent_channel)
                    block.channel = channel
                    intervals = [(max(start, t.start_time), min(end, t.end_time)) for t in own]
                    # If attribution on a channel is unresolved, do not score the other voice.
                    conflicts = [(max(start, t.start_time), min(end, t.end_time)) for t in active
                                 if t.speaker != speaker and t.channel == channel]
                else:
                    channel = 0
                    block.channel = 0
                    intervals = [(max(start, t.start_time), min(end, t.end_time)) for t in own]
                    conflicts = [(max(start, t.start_time), min(end, t.end_time)) for t in active if t.speaker != speaker]
                intervals = _subtract(_union(intervals), conflicts)
                mask = np.zeros(len(samples), dtype=bool)
                for a, b in intervals:
                    mask[max(0, round((a-start)*16000)):min(len(samples), round((b-start)*16000))] = True
                isolated = samples[:, channel].copy()
                isolated[~mask] = 0
                # Energy gate removes silence within ASR intervals. No emotion inferred from energy.
                speech = 0
                for i in range(0, len(isolated), 320):
                    frame = isolated[i:i+320]
                    if np.sqrt(np.mean(frame**2)) >= .005:
                        speech += np.count_nonzero(mask[i:i+320])
                block.speech_seconds = round(speech / 16000, 4)
                block.speech_intervals = intervals
                if not own:
                    unresolved = any(t.speaker == SpeakerRole.UNKNOWN and
                                     (channels == 1 or t.channel == channel) for t in active)
                    block.status = 'UNATTRIBUTED' if unresolved else 'NO_SPEECH'
                elif speech == 0:
                    block.status = 'OVERLAP' if conflicts and not intervals else 'NO_SPEECH'
                elif block.speech_seconds < MIN_SPEECH_SECONDS:
                    block.status = 'INSUFFICIENT_SPEECH'
                elif disabled:
                    block.status = 'DISABLED'
                else:
                    pending.append((block, isolated))
    return blocks, pending


def _score_tone_blocks(pending, models):
    """Score the planned blocks in order; after the first model failure the rest are MODEL_ERROR."""
    failure = None
    for block, isolated in pending:
        if failure:
            block.status = failure
            continue
        try:
            values, probabilities = models.tone(isolated)
            block.valence, block.arousal, block.dominance = values
            block.emotion_probabilities = probabilities
            block.emotion = max(probabilities, key=probabilities.get)
            block.tonal_score, block.tonal_label = compute_cohesive_tone(block.valence, block.arousal, block.dominance)
            block.status = 'SCORED'
        except Exception:
            log.exception('Acoustic sentiment unavailable')
            failure = block.status = 'MODEL_ERROR'


def _tone_blocks(audio_path, turns, channels, agent_channel, models, disabled=False):
    """Read bounded PCM windows, preserving time and separating speakers."""
    with tempfile.TemporaryDirectory(prefix='call1-tone-') as temp:
        blocks, pending = _plan_tone_blocks(_decode_tone_audio(audio_path, temp), turns, agent_channel, disabled)
    _score_tone_blocks(pending, models)
    return blocks


def _score_text(turns, models, disabled):
    """Learned text sentiment per turn, in place. The caller holds the inference lock."""
    failure = False
    for turn in turns:
        # Block tone must not masquerade as an independent per-turn measurement.
        turn.tone_score = turn.tone_arousal = turn.tone_label = turn.sentiment_divergence = None
        turn.text_sentiment = turn.text_sentiment_label = None
        info = {'model': TEXT_MODEL, 'revision': TEXT_REVISION, 'aggregation': 'token-weighted-probabilities-v1', 'status': 'EMPTY_TEXT'}
        turn.text_analysis = info
        if not turn.text.strip():
            continue
        if disabled:
            info['status'] = 'DISABLED'
            continue
        if failure:
            info['status'] = 'MODEL_ERROR'
            continue
        try:
            probabilities = models.text(turn.text)
            if set(probabilities) != {'POSITIVE', 'NEUTRAL', 'NEGATIVE'} or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
                raise ValueError('Invalid text probabilities')
            turn.text_sentiment = round(probabilities['POSITIVE'] - probabilities['NEGATIVE'], 6)
            turn.text_sentiment_label = max(probabilities, key=probabilities.get)
            info.update(status='SCORED', probabilities=probabilities, confidence=max(probabilities.values()))
        except Exception:
            log.exception('Text sentiment unavailable')
            failure = True
            info['status'] = 'MODEL_ERROR'


def _models_disabled():
    return os.getenv('CALL1_SENTIMENT_MODELS', '1') == '0'


def enrich_sentiment(turns, audio_path, channels=1, agent_channel=0, models=None):
    """Both modalities share the inference lock with ASR/LLM GPU work."""
    owned = models is None
    models = models or LocalSentimentModels()
    disabled = _models_disabled()
    with inference_lock:
        try:
            _score_text(turns, models, disabled)
            return _tone_blocks(audio_path, turns, channels, agent_channel, models, disabled)
        finally:
            if owned:
                models.close()


def analyze_text_sentiment(turns, models=None):
    """Text sentiment alone (the split Process app's ``text_sentiment`` stage): the same per-turn
    scoring as ``enrich_sentiment``, under the same inference lock."""
    owned = models is None
    models = models or LocalSentimentModels()
    with inference_lock:
        try:
            _score_text(turns, models, _models_disabled())
            return turns
        finally:
            if owned:
                models.close()


def analyze_tone_blocks(turns, audio_path, channels=1, agent_channel=0, models=None):
    """Acoustic tone blocks alone (the split Process app's ``acoustic_tone`` stage): the same
    seven-second speaker blocks as ``enrich_sentiment``. The ffmpeg decode and the block plan (numpy,
    no model, no Metal) run before the inference lock, so they overlap other jobs' model runs; the
    model loads, scores and is released inside it (the weights are ~2 GB, so they are not read
    early and held while the job waits for the lock). ``models`` is always released here."""
    models = models or LocalSentimentModels()
    with tempfile.TemporaryDirectory(prefix='call1-tone-') as temp:
        blocks, pending = _plan_tone_blocks(_decode_tone_audio(audio_path, temp), turns, agent_channel, _models_disabled())
    with inference_lock:
        try:
            _score_tone_blocks(pending, models)
        finally:
            close = getattr(models, 'close', None)
            if close is not None:
                close()  # frees the MPS cache before the next MLX job takes the lock
    return blocks


def average_agent_tone(blocks):
    scored = [b for b in blocks if b.speaker == SpeakerRole.AGENT and b.status == 'SCORED']
    duration = sum(b.speech_seconds for b in scored)
    return sum((2*b.valence-1)*b.speech_seconds for b in scored) / duration if duration else None
