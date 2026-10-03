"""The vocabulary-prompted Whisper Small pass of dual transcription (team decision 33,
docs/DualAsr.md section 5). **A candidate finder only, never a transcript:** prompted Whisper can skip
whole 30 s windows.

* The model is Whisper Small on MLX (``mlx-audio``, pinned), loaded from the catalog entry's model
  directory, with openai-whisper's tokenizer (vendored, ``_vendor/whisper_tokenizer.py``) reading the
  ``multilingual.tiktoken`` file that install bundles in that directory, and the "small" alignment
  heads (``SMALL_ALIGNMENT_HEADS``) for word timestamps. Nothing is fetched at inference time: a
  missing or wrong file is reported by ``install_problem`` and the pass does not run.
* The glossary is the vocabulary ranked by relevance to the base (Parakeet) transcript
  (``vocabulary_merge.rank_terms``) and packed into ``glossary_prompt_limit`` tokens
  (``Glossary: <terms>.``).
* mlx-audio, like openai-whisper, prompts only the first 30 s window. The pass wraps ``model.decode``
  so **every** window's prompt is the glossary plus the tail of the previous text, within Whisper's
  223-token prompt limit (the research's ``run_asr.py``).
* ``word_timestamps=True``, ``language="en"``; stereo runs one pass per channel on that channel's
  audio; afterwards ``gc.collect()`` and ``mx.clear_cache()`` release the model.

Model runtimes are imported inside the functions that need them.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import logging
import math
import shutil
import subprocess
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .vocabulary_merge import pack_glossary, rank_terms

log = logging.getLogger("call1.pipeline.vocabulary_asr")

SMALL_ALIGNMENT_HEADS = b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000"
"""openai-whisper's ``_ALIGNMENT_HEADS["small"]`` (the cross-attention heads word timestamps use)."""

PROMPT_LIMIT = 223
"""Whisper keeps ``prompt_tokens[-(n_text_ctx // 2 - 1):]``: 223 tokens for Whisper Small."""

TOKENIZER_FILE = "multilingual.tiktoken"
TOKENIZER_SHA256 = "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"
TOKENIZER_URL = "https://raw.githubusercontent.com/openai/whisper/v20250625/whisper/assets/multilingual.tiktoken"
"""Pinned source of the tokenizer file install bundles (openai-whisper v20250625, MIT)."""

SAMPLE_RATE = 16000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_problem(model_dir: Optional[Path]) -> Optional[str]:
    """Why Whisper Small cannot run from ``model_dir`` (safe text naming the missing file), or None:
    ``config.json``, the weights (``*.npz`` or ``*.safetensors``) and the bundled, checksum-matching
    ``multilingual.tiktoken``."""
    if model_dir is None or not Path(model_dir).is_dir():
        return "the model directory is missing"
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").is_file():
        return "config.json is missing from the model directory"
    if not (any(model_dir.glob("*.npz")) or any(model_dir.glob("*.safetensors"))):
        return "the weights are missing from the model directory"
    tokenizer = model_dir / TOKENIZER_FILE
    if not tokenizer.is_file():
        return f"{TOKENIZER_FILE} is missing from the model directory"
    if _sha256(tokenizer) != TOKENIZER_SHA256:
        return f"{TOKENIZER_FILE} does not match its pinned checksum"
    return None


def install_tokenizer(model_dir: Path, *, source: Optional[Path] = None, url: str = TOKENIZER_URL, timeout: float = 60.0) -> Path:
    """Bundle ``multilingual.tiktoken`` into ``model_dir`` at install (``scripts/provision_models.py``):
    copied from ``source`` (for example openai-whisper's ``whisper/assets/multilingual.tiktoken`` in a
    local package cache, no network) or downloaded from the pinned ``url``. The sha256 must match
    ``TOKENIZER_SHA256``; nothing is written otherwise."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if source is not None:
        data = Path(source).read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != TOKENIZER_SHA256:
        raise RuntimeError(f"{TOKENIZER_FILE} does not match its pinned sha256; review the source before updating the pin")
    target = model_dir / TOKENIZER_FILE
    partial = target.with_suffix(".partial")
    partial.write_bytes(data)
    partial.replace(target)
    return target


# --- the glossary -------------------------------------------------------------------------------


def glossary_for(terms: Sequence[str], base_words: Sequence[Dict], count_tokens: Callable[[str], int], limit: int) -> Tuple[str, List[str]]:
    """The per-call glossary: ``terms`` ranked against the base transcript's words and packed into
    ``limit`` prompt tokens. Returns (``Glossary: <terms>.``, the chosen terms in prompt order)."""
    ranked = [term for term, _ in rank_terms(terms, base_words)]
    return pack_glossary(ranked, count_tokens, limit)


# --- the pass -----------------------------------------------------------------------------------


@dataclass
class PassWord:
    word: str
    start: float
    end: float
    probability: Optional[float]
    channel: Optional[int]
    segment: int


@dataclass
class PassSegment:
    start: float
    end: float
    channel: Optional[int]
    avg_logprob: Optional[float]
    no_speech_prob: Optional[float]
    compression_ratio: Optional[float]
    temperature: Optional[float]


@dataclass
class PassResult:
    glossary: str
    glossary_terms: List[str]
    words: List[PassWord] = field(default_factory=list)
    segments: List[PassSegment] = field(default_factory=list)
    inference_seconds: float = 0.0
    load_seconds: float = 0.0
    peak_memory_bytes: Optional[int] = None


def _finite(value, low: Optional[float] = None, high: Optional[float] = None) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def decode_channel(audio_path: Path, channel: int, channels: int):
    """One channel of the recording as 16 kHz mono float32 samples (ffmpeg; raises
    ``subprocess.CalledProcessError`` when the audio cannot be decoded)."""
    import numpy as np

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed on this Process host")
    mix = ["-af", f"pan=mono|c0=c{channel}"] if channels >= 2 else ["-ac", "1"]
    completed = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(audio_path), *mix, "-ar", str(SAMPLE_RATE),
                                "-f", "s16le", "-acodec", "pcm_s16le", "-"], check=True, capture_output=True, timeout=300)
    return np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32) / 32768


def load_model(model_dir: Path):
    """Whisper Small from ``model_dir`` with the vendored tokenizer and the "small" alignment heads."""
    from mlx_audio.stt import load

    from ._vendor.whisper_tokenizer import get_tokenizer

    vocab_path = str((Path(model_dir) / TOKENIZER_FILE).resolve())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load(str(Path(model_dir).resolve()))
    model.set_alignment_heads(SMALL_ALIGNMENT_HEADS)
    model.get_tokenizer = lambda language=None, task="transcribe": get_tokenizer(  # type: ignore[method-assign]
        True, num_languages=model.num_languages, language=language, task=task, vocab_path=vocab_path)
    return model


def prompt_every_window(model, glossary_tokens: List[int], check_cancelled: Optional[Callable[[], None]] = None):
    """Wrap ``model.decode`` so each 30 s window's prompt is the glossary plus the tail of the previous
    text within ``PROMPT_LIMIT`` tokens. Returns the original ``decode`` (restore it afterwards).
    ``check_cancelled`` runs before every window, so a cancel stops the pass between windows."""
    original = model.decode
    room = max(0, PROMPT_LIMIT - len(glossary_tokens))

    def decode(segment, options):
        if check_cancelled is not None:
            check_cancelled()
        previous = list(options.prompt or [])
        tail = previous[-room:] if (previous and room) else []
        return original(segment, dataclasses.replace(options, prompt=list(glossary_tokens) + tail))

    model.decode = decode
    return original


def run_pass(model_dir: Path, audio_path: Path, channels: int, terms: Sequence[str], base_words_by_channel: Dict[Optional[int], List[Dict]],
             glossary_limit: int, *, check_cancelled: Optional[Callable[[], None]] = None) -> PassResult:
    """The whole vocabulary pass: load Whisper Small, build the glossary from the base words (all
    channels together), transcribe each channel with the glossary in every window, release the model.
    Exceptions propagate (the handler maps them to a safe ``base_only`` note)."""
    import mlx.core as mx

    from .inference import inference_lock

    from ._vendor.whisper_tokenizer import get_tokenizer

    # The glossary needs only the tokenizer, so it is built before the model is loaded (and outside
    # the inference lock). Whisper Small is multilingual (99 languages); encoding ignores the language.
    tokenizer = get_tokenizer(True, num_languages=99, language="en", task="transcribe",
                              vocab_path=str((Path(model_dir) / TOKENIZER_FILE).resolve()))
    all_words = [w for words in base_words_by_channel.values() for w in words]
    all_words.sort(key=lambda w: w["start"])
    glossary, chosen = glossary_for(terms, all_words, lambda text: len(tokenizer.encode(text)), glossary_limit)
    glossary_tokens = tokenizer.encode(" " + glossary)
    result = PassResult(glossary=glossary, glossary_terms=chosen)
    if check_cancelled is not None:
        check_cancelled()
    started = time.monotonic()
    with inference_lock:
        model = None
        try:
            reset = getattr(mx, "reset_peak_memory", None)
            if reset is not None:
                reset()
            model = load_model(model_dir)
            loaded = time.monotonic()
            result.load_seconds = round(loaded - started, 6)
            original = prompt_every_window(model, glossary_tokens, check_cancelled)
            try:
                for channel in range(max(1, channels)):
                    if check_cancelled is not None:
                        check_cancelled()
                    samples = decode_channel(audio_path, channel, channels)
                    if samples.size == 0:
                        continue
                    label = channel if channels >= 2 else None
                    output = model.generate(samples, language="en", word_timestamps=True, verbose=None)
                    for seg in output.segments:
                        result.segments.append(PassSegment(
                            start=max(0.0, _finite(seg.get("start"), 0.0) or 0.0), end=max(0.0, _finite(seg.get("end"), 0.0) or 0.0),
                            channel=label, avg_logprob=_finite(seg.get("avg_logprob")), no_speech_prob=_finite(seg.get("no_speech_prob"), 0.0, 1.0),
                            compression_ratio=_finite(seg.get("compression_ratio"), 0.0), temperature=_finite(seg.get("temperature"), 0.0)))
                        for w in seg.get("words", []) or []:
                            start = max(0.0, _finite(w.get("start"), 0.0) or 0.0)
                            result.words.append(PassWord(word=str(w.get("word", "")).strip()[:400], start=start,
                                                         end=max(start, _finite(w.get("end"), 0.0) or start),
                                                         probability=_finite(w.get("probability"), 0.0, 1.0), channel=label,
                                                         segment=len(result.segments) - 1))
                    del output, samples
            finally:
                model.decode = original
            result.inference_seconds = round(time.monotonic() - loaded, 6)
            try:
                peak = mx.get_peak_memory()
                result.peak_memory_bytes = int(peak) if peak else None
            except Exception:  # pragma: no cover - measurement only
                pass
        finally:
            del model
            gc.collect()
            mx.clear_cache()
    return result


__all__ = [
    "PROMPT_LIMIT", "PassResult", "PassSegment", "PassWord", "SMALL_ALIGNMENT_HEADS", "TOKENIZER_FILE", "TOKENIZER_SHA256", "TOKENIZER_URL",
    "decode_channel", "glossary_for", "install_problem", "install_tokenizer", "load_model", "prompt_every_window", "run_pass",
]
