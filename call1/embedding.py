"""The search embedder shared by Process and Store (team decision 18, contract 1.2.0).

Process embeds every transcript turn in the ``embeddings`` job, and Store embeds each semantic-search
query, with this one module, so a query and the turn vectors it is ranked against always come from
the same model, revision, tokenizer, pooling and normalization. Store runs **only** this embedder:
never ASR, diarization or text generation, and never a remote model.

Backends
    ``nemotron`` (default)
        ``nvidia/Nemotron-3-Embed-1B-BF16`` (OpenMDW-1.1) at a pinned revision: a bidirectional
        Ministral-3 encoder (``is_causal: false``), loaded with transformers ``AutoModel`` and
        ``AutoTokenizer`` on torch (no remote code). Following the model card: attention-mask mean
        pooling over every token, prompt included, then L2 normalization, 2048 dimensions. Turns are
        embedded as ``"passage: " + text`` and queries as ``"query: " + text``. CPU by default and
        the same dtype on both sides, so Process and Store compute identical vectors for identical
        input. Weights live at ``CALL1_EMBEDDING_PATH`` (default
        ``$CALL1_MODELS_DIR/nemotron-3-embed-1b``, i.e. ``data/models/...``);
        ``python -m call1.embedding download`` fetches them at the pinned revision.
        ``CALL1_EMBEDDING_DTYPE`` (``bfloat16`` default, the checkpoint's own dtype, about 2.3 GB
        resident; ``float32`` is faster on CPU but about 4.8 GB) and ``CALL1_EMBEDDING_DEVICE``
        (``cpu`` default, or ``mps``). The defaults give Process and Store bit-identical vectors for
        identical text; other dtype/device choices agree to cosine 0.9999+ (measured), which does not
        change rankings, but keep both apps on the same setting.
    ``fake``
        A deterministic, model-free token-hashing embedder with its own scheme
        (``fake-embedding-v1``), for tests and machines without the weights. It is never a
        fallback: a missing model is an error, not silently the fake.

Selection (``configured_backend``): ``CALL1_EMBEDDING_BACKEND`` (``nemotron`` or ``fake``) wins;
else ``fake`` when the handlers are fake (``CALL1_PROCESS_HANDLERS=fake`` or Process's configured
mode); else ``nemotron``. Store and Process must resolve the same backend, or search finds nothing:
an ``embeddings`` artifact records its ``scheme`` and Store ranks only vectors of its own scheme.

This module imports nothing from ``call1.store``, ``call1.db`` or ``call1.ingest`` (Process's import
boundary), and imports torch and transformers only when the model is first loaded.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

MODEL_REPOSITORY = "nvidia/Nemotron-3-Embed-1B-BF16"
MODEL_REVISION = "c0c9fea93ea424587517f2c59e20db9f1d6bf615"
MODEL_DIRECTORY = "nemotron-3-embed-1b"
MODEL_SCHEME = f"nemotron-3-embed-1b@{MODEL_REVISION[:7]}"
MODEL_DIMENSIONS = 2048
MODEL_LICENSE = "OpenMDW-1.1"
MODEL_MAX_TOKENS = 1024
"""Truncation length per text. Turns and queries are short; this only bounds a pathological turn."""

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

FAKE_SCHEME = "fake-embedding-v1"
FAKE_DIMENSIONS = 256

DEFAULT_BACKEND = "nemotron"
BACKENDS = (DEFAULT_BACKEND, "fake")
LEGACY_SCHEMES = frozenset({"hashing-projection-v1"})
"""Schemes older Process builds wrote. Store never ranks them; reanalysis re-embeds those calls."""

ROUND_DECIMALS = 5


class EmbedderUnavailable(RuntimeError):
    """The configured embedder cannot run here (weights missing, runtime missing, load failed)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class EmbeddingConfigError(ValueError):
    """``CALL1_EMBEDDING_BACKEND`` names no known backend."""


# --- configuration --------------------------------------------------------------------------------


def configured_backend(handlers_mode: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    explicit = (env.get("CALL1_EMBEDDING_BACKEND") or "").strip().lower()
    if explicit:
        if explicit not in BACKENDS:
            raise EmbeddingConfigError(f"CALL1_EMBEDDING_BACKEND={explicit!r}: expected one of {', '.join(BACKENDS)}")
        return explicit
    mode = (handlers_mode or env.get("CALL1_PROCESS_HANDLERS") or "").strip().lower()
    return "fake" if mode == "fake" else DEFAULT_BACKEND


def weights_path(env: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    explicit = env.get("CALL1_EMBEDDING_PATH")
    if explicit:
        return Path(explicit)
    return Path(env.get("CALL1_MODELS_DIR") or "data/models") / MODEL_DIRECTORY


def weights_installed(path: Optional[Path] = None) -> bool:
    path = weights_path() if path is None else path
    return (path / "config.json").is_file() and (path / "tokenizer.json").is_file() and any(path.glob("*.safetensors"))


def scheme_for(backend: str) -> str:
    return FAKE_SCHEME if backend == "fake" else MODEL_SCHEME


# --- backends -------------------------------------------------------------------------------------


def _unit(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        out = [0.0] * len(vec)
        out[0] = 1.0
        return out
    return [round(x / norm, ROUND_DECIMALS) for x in vec]


class Embedder:
    backend: str
    scheme: str
    dimensions: int
    model: str
    revision: str

    def embed_documents(self, texts: Sequence[str], *, batch_size: int = 16,
                        cancelled: Optional[Callable[[], None]] = None) -> List[List[float]]:  # pragma: no cover
        raise NotImplementedError

    def embed_query(self, query: str) -> List[float]:  # pragma: no cover
        raise NotImplementedError

    def status(self) -> Dict[str, object]:  # pragma: no cover
        raise NotImplementedError


_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIX = re.compile(r"(?:ing|ed|es|s|ly)$")


class FakeEmbedder(Embedder):
    """Deterministic token hashing: each (lightly stemmed) token and adjacent pair adds weight to a
    hashed slot. Queries and documents are embedded the same way. Test content only."""

    backend = "fake"
    scheme = FAKE_SCHEME
    dimensions = FAKE_DIMENSIONS
    model = "call1-fake-embedder"
    revision = "v1"

    def _embed(self, text: str) -> List[float]:
        tokens = [_SUFFIX.sub("", t) if len(t) > 4 else t for t in _TOKEN.findall(text.lower())]
        vec = [0.0] * self.dimensions
        for token in tokens:
            vec[int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big") % self.dimensions] += 1.0
        for a, b in zip(tokens, tokens[1:]):
            vec[int.from_bytes(hashlib.blake2b(f"{a}_{b}".encode(), digest_size=4).digest(), "big") % self.dimensions] += 0.5
        return _unit(vec)

    def embed_documents(self, texts: Sequence[str], *, batch_size: int = 16,
                        cancelled: Optional[Callable[[], None]] = None) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed(query)

    def status(self) -> Dict[str, object]:
        return {"backend": self.backend, "scheme": self.scheme, "model": self.model, "revision": self.revision,
                "dimensions": self.dimensions, "state": "fake", "detail": "Fake embedder: deterministic test vectors, not a model."}


class NemotronEmbedder(Embedder):
    """``nvidia/Nemotron-3-Embed-1B-BF16``, lazily loaded once and kept resident. Loading and
    inference are serialized by a lock, so one instance is safe to share between request threads."""

    backend = DEFAULT_BACKEND
    scheme = MODEL_SCHEME
    dimensions = MODEL_DIMENSIONS
    model = MODEL_REPOSITORY
    revision = MODEL_REVISION

    def __init__(self, path: Path, device: str = "cpu", dtype: str = "bfloat16") -> None:
        self.path = Path(path)
        self.device = device
        self.dtype = dtype
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._error: Optional[str] = None
        self.load_seconds: Optional[float] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def installed(self) -> bool:
        return weights_installed(self.path)

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        if not self.installed():
            raise EmbedderUnavailable("not_installed", f"The search embedding model ({MODEL_REPOSITORY}) is not installed")
        started = time.monotonic()
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(self.path), padding_side="right")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(str(self.path), dtype=getattr(torch, self.dtype))
            if getattr(model.config, "is_causal", False):
                raise RuntimeError("the embedding encoder must be bidirectional (config is_causal is true)")
            if self.device == "auto":  # Process: the Apple GPU when present (~2 s vs ~26 s per 100 turns)
                self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(self.device)
            model.eval()
        except Exception as exc:  # a broken install must surface as "unavailable", not crash the caller
            self._error = type(exc).__name__
            raise EmbedderUnavailable("load_failed", f"The search embedding model failed to load ({type(exc).__name__})") from exc
        self._tokenizer, self._model = tokenizer, model
        self._error = None
        self.load_seconds = round(time.monotonic() - started, 3)

    def load(self) -> None:
        with self._lock:
            self._load_locked()

    def unload(self) -> None:
        """Drop the model (Process does after each job; Store keeps it resident)."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            import gc

            gc.collect()
            if self.device == "mps":
                import torch

                torch.mps.empty_cache()

    def _encode_locked(self, texts: Sequence[str]) -> List[List[float]]:
        import torch
        import torch.nn.functional as F

        batch = self._tokenizer(list(texts), padding=True, truncation=True, max_length=MODEL_MAX_TOKENS, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            hidden = self._model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = F.normalize(pooled, p=2, dim=1)
        return [[round(float(x), ROUND_DECIMALS) for x in row] for row in pooled.cpu().tolist()]

    def embed_documents(self, texts: Sequence[str], *, batch_size: int = 16,
                        cancelled: Optional[Callable[[], None]] = None) -> List[List[float]]:
        out: List[List[float]] = []
        prefixed = [PASSAGE_PREFIX + t for t in texts]
        with self._lock:
            self._load_locked()
            for start in range(0, len(prefixed), batch_size):
                if cancelled is not None:
                    cancelled()
                out.extend(self._encode_locked(prefixed[start:start + batch_size]))
        return out

    def embed_query(self, query: str) -> List[float]:
        with self._lock:
            self._load_locked()
            return self._encode_locked([QUERY_PREFIX + query])[0]

    def status(self) -> Dict[str, object]:
        if self._model is not None:
            state, detail = "loaded", f"Loaded on {self.device} ({self.dtype})" + (f" in {self.load_seconds} s" if self.load_seconds else "")
        elif self._error:
            state, detail = "failed", f"The model failed to load ({self._error})"
        elif self.installed():
            state, detail = "installed", "Installed; loads on the first search"
        else:
            state, detail = "not_installed", "Not installed: run `python -m call1.embedding download`"
        return {"backend": self.backend, "scheme": self.scheme, "model": self.model, "revision": self.revision,
                "dimensions": self.dimensions, "state": state, "detail": detail}


# --- the shared instance --------------------------------------------------------------------------

_instances: Dict[Tuple, Embedder] = {}
_instances_lock = threading.Lock()


def get_embedder(backend: Optional[str] = None, *, handlers_mode: Optional[str] = None, path: Optional[Path] = None,
                 env: Optional[Mapping[str, str]] = None, device: Optional[str] = None) -> Embedder:
    """The process-wide embedder for ``backend`` (default: ``configured_backend``). Never loads the
    model: that happens on the first ``embed_*`` call."""
    env = os.environ if env is None else env
    backend = backend or configured_backend(handlers_mode, env)
    if backend == "fake":
        key: Tuple = ("fake",)
    else:
        key = (DEFAULT_BACKEND, str(path or weights_path(env)), env.get("CALL1_EMBEDDING_DEVICE") or device or "cpu",
               env.get("CALL1_EMBEDDING_DTYPE") or "bfloat16")
    with _instances_lock:
        found = _instances.get(key)
        if found is None:
            found = FakeEmbedder() if backend == "fake" else NemotronEmbedder(Path(key[1]), device=key[2], dtype=key[3])
            _instances[key] = found
        return found


def download(path: Optional[Path] = None) -> Path:
    """Fetch the pinned weights into ``path`` (default ``weights_path()``)."""
    from huggingface_hub import snapshot_download

    target = path or weights_path()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_REPOSITORY, revision=MODEL_REVISION, local_dir=str(target))
    return target


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m call1.embedding", description=f"The Call1 search embedder ({MODEL_REPOSITORY})")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help=f"download {MODEL_REPOSITORY} at revision {MODEL_REVISION[:7]}")
    dl.add_argument("--path", help=f"target directory (default: CALL1_EMBEDDING_PATH or data/models/{MODEL_DIRECTORY})")
    sub.add_parser("status", help="show the configured embedder's status")
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download(Path(args.path) if args.path else None))
        return 0
    print(json.dumps(get_embedder().status(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BACKENDS", "DEFAULT_BACKEND", "Embedder", "EmbedderUnavailable", "EmbeddingConfigError", "FakeEmbedder", "NemotronEmbedder",
    "FAKE_SCHEME", "FAKE_DIMENSIONS", "LEGACY_SCHEMES", "MODEL_DIMENSIONS", "MODEL_DIRECTORY", "MODEL_LICENSE", "MODEL_REPOSITORY",
    "MODEL_REVISION", "MODEL_SCHEME", "PASSAGE_PREFIX", "QUERY_PREFIX", "configured_backend", "download", "get_embedder",
    "scheme_for", "weights_installed", "weights_path",
]
