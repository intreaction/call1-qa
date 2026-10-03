"""Apple Silicon inference, serialized with explicit model release between stages."""
from __future__ import annotations
import gc
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import subprocess
import tempfile
import wave
from call1.models.schemas import SpeakerRole, TranscriptTurn, WordTimestamp
from call1.pipeline.inference import inference_lock


def _has_pcm_signal(path: str) -> bool:
    """Reject digital silence before ASR can hallucinate speech from it.

    Input is the normalized signed 16-bit PCM channel. This is deliberately
    not a speech detector: quiet speech and background noise still reach ASR.
    """
    with wave.open(path, "rb") as audio:
        if audio.getsampwidth() != 2:
            raise ValueError("Expected normalized 16-bit PCM audio.")
        while frames := audio.readframes(65536):
            if any(frames):
                return True
    return False


# --- Parakeet TDT 0.6B v3 (ASR) ---------------------------------------------------------------
#
# Parakeet's full-attention encoder grows quadratically with input length (a 9-minute channel
# peaks near 7 GB in bf16), so audio longer than PARAKEET_CHUNK_SECONDS is decoded in
# overlapping windows that mlx-audio merges on token identity and time. 120 s with 15 s overlap
# bounds peak memory near 2 GB and measured no worse than whole-file decoding.
PARAKEET_CHUNK_SECONDS = 120.0
PARAKEET_OVERLAP_SECONDS = 15.0
# Parakeet also misbehaves around long silence: on the stereo sample calls (each channel ~60%
# digital silence) whole-channel decoding dropped an utterance that followed a long gap, and
# windows with seconds of silent padding repeated digits ("two two two two ..."). Decoding each
# utterance alone instead loses language context (a one-second name came back in Cyrillic). So
# each channel is cut mid-way through every quiet stretch of at least PAUSE_CUT_SECONDS (RMS
# below the VAD stage's speech-energy threshold), each piece's near-silent edges (below
# SILENCE_RMS, about -60 dBFS: digital silence or a dead line, never speech) are trimmed to
# EDGE_PAD_SECONDS, and the pieces are packed back together and decoded as one stream whose
# timestamps are mapped back to the recording clock. Quiet but audible audio is always decoded.
PAUSE_CUT_SECONDS = 1.0
PAUSE_FRAME_SECONDS = 0.03
PAUSE_RMS = 0.015
SILENCE_RMS = 0.001
EDGE_PAD_SECONDS = 0.5
# A turn also ends at a pause this long, so one channel's sentence never spans the other
# speaker's reply and mono speaker attribution sees short, single-speaker spans.
TURN_PAUSE_SECONDS = 1.0


def _pause_windows(path: str) -> list[tuple[int, int]]:
    """Sample ranges to decode: the channel cut mid-way through long quiet stretches, with each
    piece's near-silent edges trimmed to EDGE_PAD_SECONDS. Pieces silent throughout are dropped,
    like a digitally silent channel."""
    import numpy as np
    with wave.open(path, "rb") as audio:
        total = audio.getnframes()
        frame = max(1, int(audio.getframerate() * PAUSE_FRAME_SECONDS))
        levels = []
        while block := audio.readframes(frame * 4096):
            samples = np.frombuffer(block, dtype="<i2").astype(np.float32) / 32768
            samples = np.pad(samples, (0, -len(samples) % frame))
            levels.append(np.sqrt((samples.reshape(-1, frame) ** 2).mean(axis=1)))
    levels = np.concatenate(levels) if levels else np.zeros(0)
    quiet = (levels < PAUSE_RMS).tolist()
    cuts, run_start = [], None
    minimum = int(round(PAUSE_CUT_SECONDS / PAUSE_FRAME_SECONDS))
    for index, is_quiet in enumerate(quiet + [False]):
        if is_quiet and run_start is None:
            run_start = index
        elif not is_quiet and run_start is not None:
            if index - run_start >= minimum and run_start > 0 and index < len(quiet):
                cuts.append((run_start + index) // 2)
            run_start = None
    pad = int(round(EDGE_PAD_SECONDS / PAUSE_FRAME_SECONDS))
    bounds = [0, *cuts, len(levels)]
    windows = []
    for first, last in zip(bounds, bounds[1:]):
        audible = np.nonzero(levels[first:last] >= SILENCE_RMS)[0]
        if len(audible):
            start = max(first, first + int(audible[0]) - pad)
            end = min(last, first + int(audible[-1]) + 1 + pad)
            windows.append((start * frame, min(total, end * frame)))
    return [(a, b) for a, b in windows if b > a]


class _PackedClock:
    """Maps a time in the packed (silence-trimmed) stream back to the recording clock."""

    def __init__(self, windows: list[tuple[int, int]], rate: int):
        self.windows, self.rate, self.packed_starts = windows, rate, []
        at = 0
        for start, end in windows:
            self.packed_starts.append(at)
            at += end - start

    def __call__(self, seconds: float) -> float:
        from bisect import bisect_right
        sample = seconds * self.rate
        index = max(0, bisect_right(self.packed_starts, sample) - 1)
        start, end = self.windows[index]
        return min(end, start + sample - self.packed_starts[index]) / self.rate


def _parakeet_words(tokens, clock=lambda t: t) -> list[WordTimestamp]:
    """Group Parakeet subword tokens into words: a token whose text starts with a space (the
    SentencePiece word marker) begins a new word. A punctuation-only token adds its text but not
    its time: TDT often emits the closing period after the following silence, which would
    otherwise stretch the word (and the turn) across the pause. Greedy TDT decoding reports no
    token probabilities, so ``probability`` keeps the schema default and turn confidence is None."""
    words: list[dict] = []
    for token in tokens:
        if not token.text:
            continue
        timed = any(c.isalnum() for c in token.text)
        if not words or token.text.startswith(" "):
            words.append({"text": token.text, "start": float(token.start), "end": float(token.end)})
        else:
            words[-1]["text"] += token.text
            if timed:
                words[-1]["end"] = max(words[-1]["end"], float(token.end))
    output = []
    for w in words:
        start = clock(w["start"])
        # A word never crosses a packing seam: its end is measured from its own start.
        output.append(WordTimestamp(word=w["text"], start_time=start, end_time=start + max(0.0, w["end"] - w["start"])))
    return _code_values(_digit_readouts(output))


_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
# Whisper-style layouts by length, so the redaction patterns (written for Whisper's digits)
# recognize an SSN, a phone number or a card number however Parakeet chose to write it.
_DIGIT_LAYOUTS = {9: (3, 2, 4), 10: (3, 3, 4), 16: (4, 4, 4, 4)}
_STOPS = ",.?!;:"


def _layout(groups: list[str]) -> str:
    digits = "".join(groups)
    if len(groups) > 1 and all(len(g) <= 4 for g in groups):
        return "-".join(groups)  # the speaker's own grouping: 902-114-883
    if len(digits) in _DIGIT_LAYOUTS:
        parts, at = [], 0
        for size in _DIGIT_LAYOUTS[len(digits)]:
            parts.append(digits[at:at + size])
            at += size
        return "-".join(parts)
    return digits


def _digit_readouts(words: list[WordTimestamp]) -> list[WordTimestamp]:
    """Rewrite a number read out digit by digit as one Whisper-style digit word.

    Parakeet writes the same read-out inconsistently: as words ("four four two eight ..."), as
    comma-separated groups ("4111, 2222, 3333, 4444") or mis-grouped ("442-891099"). PII redaction
    and numeric extraction match Whisper-style digits (442-89-1099, 4111-2222-3333-4444), so an
    unnormalized SSN or card number would escape masking. Rewritten: a run of at least three
    spoken digits; a run of at least three 2-4 digit groups totalling 9, 10 or 16 digits (an
    SSN, phone or card; "2024 2025 2026" stays as spoken); one hyphenated number with a group
    longer than four digits. The merged word spans the run, so audio muting still covers it.
    Currency, decimals, thousands separators and ordinary numbers are left alone."""
    output: list[WordTimestamp] = []
    i = 0
    while i < len(words):
        spelled = words[i].word.strip().lower().rstrip(_STOPS) in _DIGIT_WORDS
        groups: list[str] = [""]
        j = i
        while j < len(words):
            core = words[j].word.strip()
            bare = core.rstrip(_STOPS)
            if spelled:
                digit = _DIGIT_WORDS.get(bare.lower())
                if digit is None or (j == i and bare.lower() == "oh"):
                    break
                groups[-1] += digit
            else:
                if not (bare.isdigit() and 2 <= len(bare) <= 4):
                    break
                groups[-1] += bare
                groups.append("")
            j += 1
            if core.endswith(","):
                groups.append("")
            elif bare != core:
                break  # a sentence stop ends the read-out
        groups = [g for g in groups if g]
        digits = "".join(groups)
        count = j - i
        # Numeral groups are joined only when they add up to a known identifier length: three
        # or more adjacent short numbers ("2024 2025 2026", "10 20 30") are ordinary numbers.
        if (spelled and len(digits) >= 3) or (not spelled and count >= 3 and len(digits) in _DIGIT_LAYOUTS):
            text = _layout(groups)
        else:
            core = words[i].word.strip()
            bare = core.rstrip(_STOPS)
            parts = bare.split("-")
            if (len(parts) > 1 and all(p.isdigit() for p in parts) and any(len(p) > 4 for p in parts)
                    and len("".join(parts)) in _DIGIT_LAYOUTS):
                lead = words[i].word[:len(words[i].word) - len(words[i].word.lstrip())]
                output.append(words[i].model_copy(update={"word": f"{lead}{_layout([''.join(parts)])}{core[len(bare):]}"}))
            else:
                output.append(words[i])
            i += 1
            continue
        last = words[j - 1].word.strip()
        trailing = last[len(last.rstrip(_STOPS)):]
        lead = words[i].word[:len(words[i].word) - len(words[i].word.lstrip())]
        output.append(WordTimestamp(word=f"{lead}{text}{trailing}", start_time=words[i].start_time,
                                    end_time=words[j - 1].end_time))
        i = j
    return output


_CODE_CONTEXT = ("pin", "code", "passcode", "cvv")


def _code_values(words: list[WordTimestamp]) -> list[WordTimestamp]:
    """Undo number formatting on a code: Parakeet may write "PIN is 4492" as "$4,492", which
    the PIN redaction pattern (digits after pin/security code) would miss."""
    output = list(words)
    for index, word in enumerate(words):
        core = word.word.strip()
        bare = core.rstrip(_STOPS)
        context = [w.word.strip().lower().rstrip(_STOPS) for w in words[max(0, index - 3):index]]
        plain = bare.lstrip("$").replace(",", "")
        if bare != plain and plain.isdigit() and 3 <= len(plain) <= 8 and any(c in _CODE_CONTEXT for c in context):
            lead = word.word[:len(word.word) - len(word.word.lstrip())]
            output[index] = word.model_copy(update={"word": f"{lead}{plain}{core[len(bare):]}"})
    return output


def _parakeet_segments(result, clock=lambda t: t) -> list[list[WordTimestamp]]:
    """Segment-level turns: Parakeet sentences, further split at pauses of TURN_PAUSE_SECONDS
    (measured on the recording clock)."""
    segments = []
    for sentence in result.sentences:
        for word in _parakeet_words(sentence.tokens, clock):
            if segments and segments[-1] and word.start_time - segments[-1][-1].end_time < TURN_PAUSE_SECONDS:
                segments[-1].append(word)
            else:
                segments.append([word])
        segments.append([])  # a sentence end always closes the turn
    return [words for words in segments if "".join(w.word for w in words).strip()]


def _gemma_final_answer(text: str) -> str:
    """Keep Gemma's final answer, never interpret JSON inside its thought channel."""
    text = text.lstrip()
    if text.startswith("<|channel>thought"):
        _, separator, final = text.partition("<channel|>")
        return final.strip() if separator else ""
    return text


INCLUDED_TEXT_MODEL = "gemma-4-e2b-it"
"""The included model's directory name: the only text model a LoRA adapter loads over."""

_active_text_adapter: ContextVar[str | None] = ContextVar("call1_text_adapter", default=None)


@contextmanager
def use_text_adapter(path: str | None):
    """Load ``path`` (a directory with ``adapters.safetensors``) over the included model for every
    ``MLXAdapter.generate`` call inside this block, in this thread (a context variable). Process's
    LLM transport resolves the active on-device adapter once per job (decision 28,
    docs/OnDeviceTraining.md section 5.1) and wraps each generation in this. ``None`` means the base.
    This module never reads Process's data directory."""
    token = _active_text_adapter.set(path)
    try:
        yield path
    finally:
        _active_text_adapter.reset(token)


_resident_text_models: ContextVar[dict | None] = ContextVar("call1_resident_text_models", default=None)


@contextmanager
def keep_text_model_loaded():
    """Hold ``inference_lock`` for the whole block and keep each text model ``MLXAdapter.generate``
    loads inside it resident for the block's later prompts (in this thread), then free it on exit.
    A multi-prompt job (Contact Signals stages 1 and 2) loads the model once instead of once per
    prompt; the model is still never resident between jobs. Generation is unchanged: every prompt
    starts from a fresh KV cache."""
    with inference_lock:
        models: dict = {}
        token = _resident_text_models.set(models)
        try:
            yield
        finally:
            _resident_text_models.reset(token)
            loaded = bool(models)
            models.clear()
            if loaded:
                import mlx.core as mx
                gc.collect()
                mx.clear_cache()


def text_adapter_path(base: str | None) -> str | None:
    """The LoRA adapter to load over the text model ``base``, or None. Only the included model
    (``data/models/gemma-4-e2b-it``) takes one; local packs never do. In order:

    1. ``CALL1_TEXT_ADAPTER`` (a directory holding ``adapters.safetensors`` and
       ``adapter_config.json``): the manual override, unchanged (provenance ``+lora.env``);
    2. the adapter set by ``use_text_adapter`` for this call (the Process's active adapter);
    3. None: the base. Callers that never set the context variable are unchanged."""
    if not base or Path(base).name != INCLUDED_TEXT_MODEL:
        return None
    adapter = os.getenv("CALL1_TEXT_ADAPTER")
    if adapter:
        if not (Path(adapter) / "adapters.safetensors").is_file():
            raise RuntimeError("CALL1_TEXT_ADAPTER does not hold adapters.safetensors")
        return adapter
    return _active_text_adapter.get()


# --- outlines JSON constraint: vocabulary and schema index caches --------------------------------
#
# Building the outlines Vocabulary walks the whole tokenizer vocabulary (262k tokens for Gemma 4) and
# took seconds per prompt when rebuilt for every generation; compiling a schema's Index takes longer
# for the wide Contact Signals schemas. Neither depends on the model weights, so the Vocabulary is
# built once per tokenizer and kept, and ``MLXAdapter.generate`` compiles the schema's Index before
# it takes ``inference_lock`` (CPU work that never touches Metal). Each generation still gets its own
# logits processor (its own Guide state); the constraint and the greedy output are unchanged.
_OUTLINES_INDEX_CACHE = 16
_outlines_vocabularies: dict = {}
_outlines_indexes: "OrderedDict[tuple, object]" = OrderedDict()
_outlines_lock = threading.Lock()


def _tokenizer_key(tokenizer) -> tuple:
    return ("tokenizer", str(getattr(tokenizer, "name_or_path", "") or id(tokenizer)), tokenizer.eos_token_id,
            getattr(tokenizer, "vocab_size", None))


def _outlines_vocabulary(tokenizer):
    """The outlines Vocabulary of an mlx_lm tokenizer, exactly as ``OutlinesCoreBackend`` builds it
    for an ``MLXLM`` model."""
    from outlines.backends.outlines_core import OutlinesCoreBackend
    return OutlinesCoreBackend.create_outlines_core_vocabulary(
        tokenizer.get_vocab(), tokenizer.eos_token_id, tokenizer.eos_token,
        lambda token: tokenizer.convert_tokens_to_string([token]))


def schema_index(vocabulary_key: tuple, response_schema: dict, tokenizer=None):
    """The outlines Index constraining output to ``response_schema`` over the vocabulary cached under
    ``vocabulary_key``. The vocabulary is built from ``tokenizer`` the first time; without one and
    before that, None (the caller builds it once the tokenizer is loaded). Same regex and Index as
    ``OutlinesCoreBackend.get_json_schema_logits_processor``."""
    import json
    schema = json.dumps(response_schema)
    with _outlines_lock:
        index = _outlines_indexes.get((vocabulary_key, schema))
        if index is not None:
            _outlines_indexes.move_to_end((vocabulary_key, schema))
            return index
        vocabulary = _outlines_vocabularies.get(vocabulary_key)
    if vocabulary is None:
        if tokenizer is None:
            return None
        vocabulary = _outlines_vocabulary(tokenizer)
        with _outlines_lock:
            vocabulary = _outlines_vocabularies.setdefault(vocabulary_key, vocabulary)
    from outlines_core import Index
    from outlines_core.json_schema import build_regex_from_schema
    index = Index(build_regex_from_schema(schema, None), vocabulary)
    with _outlines_lock:
        _outlines_indexes[(vocabulary_key, schema)] = index
        _outlines_indexes.move_to_end((vocabulary_key, schema))
        while len(_outlines_indexes) > _OUTLINES_INDEX_CACHE:
            _outlines_indexes.popitem(last=False)
    return index


def clear_outlines_cache() -> None:
    with _outlines_lock:
        _outlines_vocabularies.clear()
        _outlines_indexes.clear()


def generate_loaded(model, tokenizer, system: str, prompt: str, max_tokens: int, response_schema: dict | None = None,
                    index=None) -> str:
    """One generation on an already-loaded model: the chat template with thinking off, the context
    check, the outlines JSON constraint and Gemma's final-answer extraction. ``MLXAdapter.generate``
    and the on-device training evaluator (``call1.process.training.generate``) share it, so the
    evaluator scores exactly what production would send and parse. ``index`` is the schema's
    precompiled outlines Index (``schema_index``); without one it is compiled here (cached)."""
    from mlx_lm import stream_generate
    processors = None
    try:
        tokens = tokenizer.apply_chat_template([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False)
        if len(tokens) + max_tokens > 8192:
            raise RuntimeError("Input exceeds the appliance context budget; split into smaller chunks.")
        if response_schema:
            from outlines.backends.outlines_core import OutlinesCoreLogitsProcessor
            if index is None:
                index = schema_index(_tokenizer_key(tokenizer), response_schema, tokenizer)
            # Each request owns its decoding state (a fresh Guide); do not retain model weights.
            processors = [OutlinesCoreLogitsProcessor(index, "mlx")]
        output = "".join(part.text for part in stream_generate(
            model, tokenizer, tokens, max_tokens=max_tokens, logits_processors=processors))
        return _gemma_final_answer(output) if model.model_type == "gemma4" else output
    finally:
        del processors


class MLXAdapter:
    def _path(self, kind: str) -> str:
        default = {"ASR": "parakeet-tdt-0.6b-v3", "TEXT": "gemma-4-e2b-it",
                   "DIARIZATION": "nemotron-3-diarization"}[kind]
        path = Path(os.getenv(f"CALL1_MLX_{kind}_PATH", f"data/models/{default}")).resolve()
        if not (path / "config.json").is_file():
            raise RuntimeError(f"Provision the {kind} model before processing recordings.")
        return str(path)

    def health(self) -> dict:
        import mlx.core as mx
        return {"adapter": "mlx", "device": mx.device_info(),
                "peak_memory_bytes": mx.get_peak_memory(),
                "asr_path": self._path("ASR"), "text_path": self._path("TEXT"),
                "diarization_path": self._path("DIARIZATION")}

    def attribute_speakers(self, path: str, turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
        from call1.adapters.mlx_diarization import diarize
        from call1.pipeline.diarization import attach_clusters
        model_path = self._path("DIARIZATION")
        with tempfile.TemporaryDirectory(prefix="call1-speakers-") as temp:
            wav = str(Path(temp) / "mono.wav")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path,
                            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                           check=True, capture_output=True, timeout=180)
            return attach_clusters(turns, diarize(wav, model_path))

    def generate(self, system: str, prompt: str, max_tokens: int, response_schema: dict | None = None, text_model_path: str | None = None) -> str:
        """One prompt. The model is loaded and freed inside ``inference_lock``, unless a
        ``keep_text_model_loaded`` block is open in this thread (then the block's first prompt
        loads it and later prompts reuse it). The schema's outlines Index is compiled before the
        lock once this process has seen the tokenizer."""
        import mlx.core as mx
        from mlx_lm import load
        base = text_model_path or self._path("TEXT")
        adapter = text_adapter_path(base)
        vocabulary_key = ("model", base)
        index = schema_index(vocabulary_key, response_schema) if response_schema else None
        resident = _resident_text_models.get()
        with inference_lock:
            model = tokenizer = None
            try:
                key = (base, adapter)
                if resident is not None and key in resident:
                    model, tokenizer = resident[key]
                else:
                    model, tokenizer = load(base, adapter_path=adapter) if adapter else load(base)
                    if resident is not None:
                        resident[key] = (model, tokenizer)
                if response_schema and index is None:
                    index = schema_index(vocabulary_key, response_schema, tokenizer)
                return generate_loaded(model, tokenizer, system, prompt, max_tokens, response_schema, index=index)
            finally:
                del model, tokenizer
                gc.collect()
                mx.clear_cache()

    def transcribe(self, path: str, channels: int, agent_channel: int = 0) -> list[TranscriptTurn]:
        """Parakeet TDT 0.6B v3, one channel at a time. Parakeet detects the language itself
        (25 European languages), so ``CALL1_ASR_LANGUAGE`` does not apply on this backend."""
        import mlx.core as mx
        import numpy as np
        from mlx_audio.stt import load
        if channels not in (1, 2) or agent_channel not in (0, 1):
            raise ValueError("Expected mono or stereo recording and agent channel 0 or 1.")
        turns = []
        with inference_lock:
            model = result = samples = None
            try:
                with tempfile.TemporaryDirectory(prefix="call1-mlx-") as temp:
                    for channel in range(channels):
                        wav = str(Path(temp) / f"{channel}.wav")
                        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path,
                            "-af", f"pan=mono|c0=c{channel}", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                            check=True, capture_output=True, timeout=180)
                        if not _has_pcm_signal(wav):
                            continue
                        if model is None:
                            # The published weights are float32. Loading lazily and casting to
                            # bf16 before evaluation keeps ~1.2 GB resident (peak ~1.7 GB), not 2.4.
                            model = load(self._path("ASR"), lazy=True, strict=True)
                            model.set_dtype(mx.bfloat16)
                            mx.eval(model.parameters())
                            gc.collect()
                            mx.clear_cache()
                        role = SpeakerRole.UNKNOWN if channels == 1 else (
                            SpeakerRole.AGENT if channel == agent_channel else SpeakerRole.CALLER)
                        windows = _pause_windows(wav)
                        if not windows:
                            continue
                        with wave.open(wav, "rb") as audio:
                            rate, pieces = audio.getframerate(), []
                            for start, end in windows:
                                audio.setpos(start)
                                pieces.append(np.frombuffer(audio.readframes(end - start), dtype="<i2"))
                        samples = mx.array(np.concatenate(pieces).astype(np.float32) / 32768).astype(mx.bfloat16)
                        del pieces
                        result = model.generate(samples, dtype=mx.bfloat16, chunk_duration=PARAKEET_CHUNK_SECONDS,
                                                overlap_duration=PARAKEET_OVERLAP_SECONDS)
                        for words in _parakeet_segments(result, _PackedClock(windows, rate)):
                            text = "".join(w.word for w in words).strip()
                            turns.append(TranscriptTurn(turn_id=0, speaker=role, channel=channel,
                                start_time=words[0].start_time, end_time=words[-1].end_time,
                                text=text, raw_text=text, word_timestamps=words, confidence=None))
                        result = samples = None
                        mx.clear_cache()
            finally:
                del model, result, samples
                gc.collect()
                mx.clear_cache()
        turns.sort(key=lambda t: (t.start_time, t.channel or 0))
        for index, turn in enumerate(turns):
            turn.turn_id = index
        if not turns:
            raise RuntimeError("No speech found; recording requires manual review.")
        return turns
