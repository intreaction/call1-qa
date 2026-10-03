"""The model-based PII layer (team decision 19): OpenAI Privacy Filter beside the number rules.

``call1.redaction`` finds sensitive *numbers* with rules. This module adds what rules never caught:
names, addresses, emails, URLs and secrets, found by ``openai/privacy-filter`` (Apache-2.0), a
bidirectional token classifier (about 1.5B parameters, 50M active) with BIOES span labels over
``account_number``, ``private_address``, ``private_date``, ``private_email``, ``private_person``,
``private_phone``, ``private_url`` and ``secret``. Process's masking step (the real handlers'
``handlers/real/masking.py``) takes the union of these spans with the rule-based values.

Masked: every label in ``MASKED_LABELS`` (all but ``private_date``: dates stay visible because
rubric criteria depend on them, e.g. dispute windows), and ``private_person`` except the agent's
own name (greeting by name is a rubric criterion). The agent's name comes from the call's
``agent_display_name`` plus self-introductions in agent turns ("my name is X", "this is X"),
compared case-insensitively token by token.

Findings are masked by position (their turn and offsets), and a finding made only of common words
is dropped (``filter_spans``). Only strong identifiers (``strong_identifier``: emails, URLs,
number runs, multi-word names, capitalized names of 3+ characters) are also masked by value across
the call (``sensitive_values``), so a filter hit on "so" or "lovely" never hides that word
everywhere (team decision 22).

Backends
    ``privacy-filter`` (default)
        ``openai/privacy-filter`` at a pinned revision, loaded with transformers'
        ``OpenAIPrivacyFilterForTokenClassification`` (native in transformers 5.x; no remote code)
        from ``CALL1_PII_MODEL_PATH`` (default ``$CALL1_MODELS_DIR/openai-privacy-filter``). Only the
        root ``model.safetensors``, config, tokenizer and ``viterbi_calibration.json`` are needed
        (``python -m call1.pii_model download``). Spans are decoded with the model card's
        constrained BIOES Viterbi and the calibration file's ``default`` operating point. Runs on
        MPS when available, else CPU, in bfloat16. Process loads it per job and releases it after.
    ``stub``
        A deterministic, model-free **stub** for fake-handler mode and CI: simple patterns for
        emails, URLs, "my name is X" names, street addresses and dates. It is never a fallback: a
        real install whose weights are missing fails the masking job closed (``model_unavailable``).

Selection (``configured_backend``): ``CALL1_PII_MODEL_BACKEND`` (``privacy-filter`` or ``stub``)
wins; else ``stub`` when the handlers are fake (``CALL1_PROCESS_HANDLERS=fake``); else
``privacy-filter``.

This module imports nothing from ``call1.store`` and imports torch and transformers only on load.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

MODEL_REPOSITORY = "openai/privacy-filter"
MODEL_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
MODEL_DIRECTORY = "openai-privacy-filter"
MODEL_LICENSE = "Apache-2.0"
MODEL_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "viterbi_calibration.json",
               "README.md")
"""The root files only: never ``onnx/`` or ``original/``."""

LABELS = ("account_number", "private_address", "private_date", "private_email", "private_person", "private_phone",
          "private_url", "secret")
MASKED_LABELS = frozenset(LABELS) - {"private_date"}

DEFAULT_BACKEND = "privacy-filter"
STUB_BACKEND = "stub"
STUB_REVISION = "stub-v1"
"""``PiiFindingsContent.detector_revision`` for the stub (bump it when its patterns change)."""
BACKENDS = (DEFAULT_BACKEND, STUB_BACKEND)

CHUNK_CHARS = 12_000
"""Turns are classified together (context helps), in documents of about this many characters."""


class PiiModelUnavailable(RuntimeError):
    """The configured PII model cannot run here (weights missing, runtime missing, load failed)."""


class PiiConfigError(ValueError):
    """``CALL1_PII_MODEL_BACKEND`` names no known backend."""


@dataclass(frozen=True)
class PiiSpan:
    label: str
    start: int
    end: int
    text: str


# --- configuration --------------------------------------------------------------------------------


def configured_backend(handlers_mode: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    explicit = (env.get("CALL1_PII_MODEL_BACKEND") or "").strip().lower()
    if explicit:
        if explicit not in BACKENDS:
            raise PiiConfigError(f"CALL1_PII_MODEL_BACKEND={explicit!r}: expected one of {', '.join(BACKENDS)}")
        return explicit
    mode = (handlers_mode or env.get("CALL1_PROCESS_HANDLERS") or "").strip().lower()
    return STUB_BACKEND if mode == "fake" else DEFAULT_BACKEND


def weights_path(env: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    explicit = env.get("CALL1_PII_MODEL_PATH")
    if explicit:
        return Path(explicit)
    return Path(env.get("CALL1_MODELS_DIR") or "data/models") / MODEL_DIRECTORY


def weights_installed(path: Optional[Path] = None) -> bool:
    path = weights_path() if path is None else path
    return all((path / name).is_file() for name in ("config.json", "tokenizer.json", "model.safetensors"))


def download(path: Optional[Path] = None) -> Path:  # pragma: no cover - network
    """Fetch the pinned root files (an explicit setup step; inference never downloads)."""
    from huggingface_hub import snapshot_download

    target = weights_path() if path is None else path
    snapshot_download(MODEL_REPOSITORY, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES), local_dir=str(target))
    return target


# --- BIOES decoding -------------------------------------------------------------------------------

_BIAS_NAMES = ("transition_bias_background_stay", "transition_bias_background_to_start", "transition_bias_end_to_background",
               "transition_bias_end_to_start", "transition_bias_inside_to_continue", "transition_bias_inside_to_end")


def read_calibration(path: Path, operating_point: str = "default") -> Dict[str, float]:
    """The ``viterbi_calibration.json`` biases for an operating point (all zero when absent)."""
    biases = {name: 0.0 for name in _BIAS_NAMES}
    try:
        data = json.loads((path / "viterbi_calibration.json").read_text())
        biases.update({k: float(v) for k, v in data["operating_points"][operating_point]["biases"].items() if k in biases})
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return biases


def _tag(label: str) -> Tuple[str, Optional[str]]:
    if label == "O":
        return "O", None
    boundary, _, category = label.partition("-")
    return boundary, category


def transition_scores(id2label: Mapping[int, str], biases: Mapping[str, float]):
    """(start, transition, end) score arrays for the constrained BIOES decoder: disallowed moves
    are -inf; allowed ones carry the calibration bias for their kind."""
    import numpy as np

    n = len(id2label)
    tags = [_tag(id2label[i]) for i in range(n)]
    neg = -np.inf
    start = np.array([0.0 if b in ("O", "B", "S") else neg for b, _ in tags])
    end = np.array([0.0 if b in ("O", "E", "S") else neg for b, _ in tags])
    trans = np.full((n, n), neg)
    for i, (bi, ci) in enumerate(tags):
        for j, (bj, cj) in enumerate(tags):
            if bi == "O":
                if bj == "O":
                    trans[i, j] = biases["transition_bias_background_stay"]
                elif bj in ("B", "S"):
                    trans[i, j] = biases["transition_bias_background_to_start"]
            elif bi in ("B", "I"):
                if cj == ci and bj == "I":
                    trans[i, j] = biases["transition_bias_inside_to_continue"]
                elif cj == ci and bj == "E":
                    trans[i, j] = biases["transition_bias_inside_to_end"]
            else:  # E or S closes a span
                if bj == "O":
                    trans[i, j] = biases["transition_bias_end_to_background"]
                elif bj in ("B", "S"):
                    trans[i, j] = biases["transition_bias_end_to_start"]
    return start, trans, end


def viterbi(log_probs, start, trans, end) -> List[int]:
    """The best label path under the BIOES constraints. ``log_probs`` is [T, labels]."""
    import numpy as np

    steps = log_probs.shape[0]
    if steps == 0:
        return []
    score = start + log_probs[0]
    back = np.zeros((steps, log_probs.shape[1]), dtype=np.int64)
    for t in range(1, steps):
        candidates = score[:, None] + trans
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_probs[t]
    score = score + end
    path = [int(score.argmax())]
    for t in range(steps - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def decode_spans(text: str, offsets: Sequence[Tuple[int, int]], path: Sequence[int], id2label: Mapping[int, str]) -> List[PiiSpan]:
    """Character spans from a decoded label path, widened to whole words and trimmed of spaces."""
    spans: List[PiiSpan] = []
    open_at: Optional[int] = None
    for index, label_id in enumerate(path):
        boundary, category = _tag(id2label[label_id])
        if boundary in ("B", "S"):
            open_at = index
        if boundary in ("E", "S") and open_at is not None and category:
            start, end = offsets[open_at][0], offsets[index][1]
            # Tokens carry their leading space: trim it first, then widen to whole words.
            while start < end and text[start].isspace():
                start += 1
            while start > 0 and text[start - 1].isalnum():
                start -= 1
            while end < len(text) and text[end].isalnum():
                end += 1
            while end > start and text[end - 1] in " \t\n.,;:!?":
                end -= 1
            if end - start >= 2:
                spans.append(PiiSpan(category, start, end, text[start:end]))
            open_at = None
        if boundary == "O":
            open_at = None
    return spans


# --- detectors ------------------------------------------------------------------------------------


class Detector:
    backend: str = ""

    def detect(self, texts: Sequence[str]) -> List[List[PiiSpan]]:  # pragma: no cover - interface
        raise NotImplementedError

    def release(self) -> None:
        pass


class PrivacyFilter(Detector):
    """``openai/privacy-filter``, loaded on construction and released by ``release``."""

    backend = DEFAULT_BACKEND

    def __init__(self, path: Optional[Path] = None, device: str = "auto") -> None:
        self.path = weights_path() if path is None else Path(path)
        if not weights_installed(self.path):
            raise PiiModelUnavailable(f"the PII model ({MODEL_REPOSITORY}) is not installed at {self.path}")
        started = time.monotonic()
        try:
            import torch
            from transformers import AutoTokenizer, OpenAIPrivacyFilterForTokenClassification

            self.device = ("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
            model = OpenAIPrivacyFilterForTokenClassification.from_pretrained(str(self.path), dtype=torch.bfloat16)
            model.to(self.device)
            model.eval()
            self.model = model
        except Exception as exc:
            raise PiiModelUnavailable(f"the PII model failed to load ({type(exc).__name__})") from exc
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.scores = transition_scores(self.id2label, read_calibration(self.path))
        self.load_seconds = round(time.monotonic() - started, 3)

    def _classify(self, text: str) -> List[PiiSpan]:
        import torch

        encoded = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        offsets = [tuple(o) for o in encoded.pop("offset_mapping")[0].tolist()]
        inputs = {k: v.to(self.device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].float()
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        return decode_spans(text, offsets, viterbi(log_probs, *self.scores), self.id2label)

    def detect(self, texts: Sequence[str]) -> List[List[PiiSpan]]:
        """Spans per text. Texts are joined into documents of about ``CHUNK_CHARS`` so each turn is
        read with its neighbours, then spans are mapped back to their own text."""
        out: List[List[PiiSpan]] = [[] for _ in texts]
        batch: List[int] = []

        def flush() -> None:
            if not batch:
                return
            doc, starts = "", []
            for i in batch:
                starts.append(len(doc))
                doc += texts[i] + "\n"
            for span in self._classify(doc):
                # Each text gets its own part of a span (one crossing a turn break is split).
                for i, base in zip(batch, starts):
                    s, e = max(0, span.start - base), min(len(texts[i]), span.end - base)
                    while s < e and texts[i][s].isspace():
                        s += 1
                    while e > s and texts[i][e - 1] in " \t\n.,;:!?":
                        e -= 1
                    if e - s >= 2:
                        out[i].append(PiiSpan(span.label, s, e, texts[i][s:e]))
            batch.clear()

        size = 0
        for index, text in enumerate(texts):
            if batch and size + len(text) > CHUNK_CHARS:
                flush()
                size = 0
            batch.append(index)
            size += len(text) + 1
        flush()
        return out

    def release(self) -> None:
        device = getattr(self, "device", "cpu")
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if device == "mps":
                torch.mps.empty_cache()
            elif device.startswith("cuda"):  # pragma: no cover
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - cache release is best effort
            pass


class StubPrivacyFilter(Detector):
    """STUB, not a model: deterministic patterns standing in for ``openai/privacy-filter`` in
    fake-handler mode and CI. Emails, URLs, names after "my name is"/"this is", street addresses
    and "Month D, YYYY" dates (so date handling is exercised)."""

    backend = STUB_BACKEND
    _PATTERNS = (
        ("private_email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
        ("private_url", re.compile(r"\b(?:https?://|www\.)\S+[^\s.,;:!?]")),
        ("private_address", re.compile(
            r"\b\d{1,6}(?:\s+[A-Z][a-z]+)+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way)\b"
            r"(?:,\s+[A-Z][a-z]+)?")),
        ("private_date", re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"
            r"(?:st|nd|rd|th)?,?\s+\d{4}\b")),
        ("private_person", re.compile(r"(?i:\b(?:my name is|this is|i am|i'm)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")),
    )

    def detect(self, texts: Sequence[str]) -> List[List[PiiSpan]]:
        out = []
        for text in texts:
            spans = []
            for label, pattern in self._PATTERNS:
                for m in pattern.finditer(text):
                    group = m.lastindex or 0
                    spans.append(PiiSpan(label, m.start(group), m.end(group), m.group(group)))
            out.append(spans)
        return out


def detector(backend: Optional[str] = None) -> Detector:
    """A loaded detector for ``backend`` (default: ``configured_backend``). Release it after use."""
    backend = backend or configured_backend()
    if backend == STUB_BACKEND:
        return StubPrivacyFilter()
    return PrivacyFilter()


# --- the masking policy ---------------------------------------------------------------------------

_INTRODUCTION = re.compile(r"(?i:\b(?:my name is|this is|name's)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def name_tokens(text: str) -> Set[str]:
    return {t.lower() for t in _NAME_TOKEN.findall(text or "")}


def agent_name_tokens(agent_display_name: Optional[str], agent_turn_texts: Iterable[str]) -> Set[str]:
    """The agent's own name, lower-cased by token: the call's display name plus every
    self-introduction in an agent turn."""
    tokens = name_tokens(agent_display_name or "")
    for text in agent_turn_texts:
        for m in _INTRODUCTION.finditer(text or ""):
            tokens |= name_tokens(m.group(1))
    return tokens


def maskable(span: PiiSpan, agent_tokens: Set[str]) -> bool:
    """Whether a detected span is masked: a masked label, and for a person, not only the agent's name."""
    if span.label not in MASKED_LABELS:
        return False
    if span.label == "private_person":
        tokens = name_tokens(span.text)
        return bool(tokens) and not tokens <= agent_tokens
    return True


# --- span filtering: common words dropped, strong identifiers propagated (decision 22) -------------
#
# The filter flags single common words ("so", "and", "Okay", "lovely") often enough on ASR text
# that masking each span's text *by value* hid those words across the whole call. So a finding is:
#   - dropped when every word in it is a stopword or a common conversational word (COMMON_WORDS);
#   - masked by position (its turn and offsets) when it is kept;
#   - also propagated by value to the rest of the call only when it is a *strong identifier*: an
#     email or URL, a number run (3+ digits, numerals or digit words), a multi-word name or address,
#     a single capitalized name word of 3+ characters (any single word of 3+ characters when the
#     ASR writes no casing), or a secret of 4+ characters.
# Store applies the positions (``call1.store.results.masking``); Process masks model input the same
# way (``call1.process.handlers.real.masking``).

COMMON_WORDS = frozenset("""
a about above after again all also am an and any are aren't around as at away back be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
else even ever every few for from further get gets getting go goes going gone got had hadn't has hasn't have haven't
having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is
isn't it it's its itself just let let's like me more most much must mustn't my myself no nor not now of off on once
only or other ought our ours ourselves out over own really same she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this
those though through to too under until up upon us very was wasn't we we'd we'll we're we've were weren't what what's
when when's where where's which while who who's whom why why's with won't would wouldn't yet you you'd you'll you're
you've your yours yourself yourselves
yes yeah yep yup ya yah nope nah okay ok okey alright right sure oh ohh ooh uh um umm uhm ah ahh hmm hm mm mhm er erm
eh huh wow whoa hey hi hello bye goodbye thanks thank thankyou please sorry pardon excuse cheers welcome
great good fine lovely brilliant perfect perfectly awesome amazing wonderful excellent nice cool fantastic super
absolutely definitely certainly exactly totally actually basically literally probably maybe perhaps anyway anyways
well so gonna wanna gotta kinda sorta mate dear love sir madam ma'am miss folks guys guy buddy pal
lot lots bit thing things stuff something anything nothing everything someone anyone everyone somebody anybody nobody
one ones today tomorrow yesterday tonight morning afternoon evening day week month year time moment second minute
order orders item items account card number phone email address name call customer caller agent service help
""".split())
"""Words that are never an identifier on their own. Deliberately *not* here: words that are also
common given names or surnames ("will", "may", "mark", "bill", "grace", "hope", "joy", "rose",
"frank", "faith", "jack", "sue", "pat", "art", "bob", "summer", "dawn", "june", "april"). Dates are
never masked (``MASKED_LABELS``), so month names need no entry."""

_SPAN_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*")
_DIGIT_WORDS = {"zero": 1, "oh": 1, "one": 1, "two": 1, "three": 1, "four": 1, "five": 1, "six": 1, "seven": 1, "eight": 1,
                "nine": 1, "double": 1, "triple": 2}
_PRONOUN_I = {"i", "i'm", "i'll", "i've", "i'd"}


def span_words(text: str) -> List[str]:
    return [w.replace("’", "'") for w in _SPAN_WORD.findall(text or "")]


def is_common(text: str) -> bool:
    """Whether a span holds nothing but stopwords and common conversational words (or no word)."""
    words = span_words(text)
    return all(w.lower() in COMMON_WORDS for w in words)


def transcript_cased(texts: Iterable[str]) -> bool:
    """Whether the ASR writes casing: any capital letter outside the pronoun "I"."""
    for text in texts:
        for word in span_words(text):
            if word.lower() not in _PRONOUN_I and any(c.isupper() for c in word):
                return True
    return False


def _digit_count(text: str) -> int:
    count = 0
    for word in span_words(text):
        lower = word.lower()
        if lower.isdigit():
            count += len(lower)
        else:
            count += _DIGIT_WORDS.get(lower, 0)
    return count


def keep_span(span: PiiSpan) -> bool:
    """Whether a finding is kept at all: dropped when it is only common words."""
    return not is_common(span.text)


def strong_identifier(span: PiiSpan, cased: bool = True) -> bool:
    """Whether a kept finding is strong enough to mask by value everywhere in the call (see the
    notes above ``COMMON_WORDS``). ``cased``: the call's ASR writes casing (``transcript_cased``),
    so a single-word name must be capitalized to count."""
    if not keep_span(span):
        return False
    if span.label in ("private_email", "private_url"):
        return True
    if _digit_count(span.text) >= 3:
        return True
    words = [w for w in span_words(span.text) if w.lower() not in COMMON_WORDS and not w.isdigit()
             and w.lower() not in _DIGIT_WORDS]
    if span.label == "secret":
        return len(span.text.strip()) >= 4
    if span.label in ("private_person", "private_address"):
        if len(words) >= 2:
            return True
        return len(words) == 1 and len(words[0]) >= 3 and (not cased or words[0][0].isupper())
    return False


def filter_spans(spans_per_text: Sequence[Sequence[PiiSpan]], agent_tokens: Set[str]) -> List[List[PiiSpan]]:
    """The findings that are masked: a masked label, not only the agent's name, not only common words."""
    return [[span for span in spans if maskable(span, agent_tokens) and keep_span(span)] for spans in spans_per_text]


def sensitive_values(spans_per_text: Sequence[Sequence[PiiSpan]], agent_tokens: Set[str], cased: bool = True) -> Set[str]:
    """The span texts to mask *by value* across the call (``call1.redaction`` masks every
    occurrence of a value, maps it to word timestamps and mutes its audio): the strong identifiers
    among the masked findings. The other masked findings are masked by position only."""
    return {span.text for spans in filter_spans(spans_per_text, agent_tokens) for span in spans if strong_identifier(span, cased)}


def _main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse

    parser = argparse.ArgumentParser(prog="python -m call1.pii_model", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("download", "status"))
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download())
    else:
        print(json.dumps({"backend": configured_backend(), "path": str(weights_path()), "installed": weights_installed()}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
