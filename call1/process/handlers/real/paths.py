"""Where the real handlers find their weights, which runtime they use, and what the catalog reports
as installed on this host (``HandlerRegistry.entry_status``).

A catalog entry's weights live at ``CALL1_MODELS_DIR/<model_directory>`` (default ``data/models``).
The pre-split pipeline's per-model environment variables still win when they are set, so an
appliance configured for the legacy app keeps working unchanged:

=========================  ==============================  ====================================
entry                      override                        default
=========================  ==============================  ====================================
``parakeet-tdt-0.6b-v3``   ``CALL1_MLX_ASR_PATH``          ``<models>/parakeet-tdt-0.6b-v3``
``nemotron-3-diarization`` ``CALL1_MLX_DIARIZATION_PATH``  ``<models>/nemotron-3-diarization``
``meralion-ser-v1``        ``CALL1_TONE_PATH``             ``<models>/meralion-ser-v1``
``roberta-sentiment``      ``CALL1_SENTIMENT_PATH``        ``<models>/roberta-sentiment``
``nemotron-3-embed-1b``    ``CALL1_EMBEDDING_PATH``        ``<models>/nemotron-3-embed-1b``
``call1-bundled``          ``CALL1_MLX_TEXT_PATH``         ``<models>/gemma-4-e2b-it``
``whisper-small-vocab``    ``CALL1_WHISPER_VOCAB_PATH``    ``<models>/whisper-small``
local packs                ``CALL1_OPTIONAL_MODELS_DIR``   ``<models>/<pack directory>``
=========================  ==============================  ====================================

``CALL1_BACKEND=mlx`` selects Apple-Silicon inference: ASR on Parakeet TDT 0.6B v3 and diarization
on Nemotron-3-Diarization (both through mlx-audio), the included LLM on MLX. Otherwise ASR runs on
faster-whisper (``CALL1_ASR_MODEL``, default ``small``, from the local Hugging Face cache), the
included LLM goes to the loopback Ollama the legacy app used, and diarization does not run (mono
turns stay UNKNOWN, as before the split). Tone and text sentiment always run on torch.

``whisper-small-vocab`` (dual transcription's vocabulary pass, docs/DualAsr.md) needs MLX and counts
as installed only when its model directory also holds the bundled ``multilingual.tiktoken`` with the
pinned checksum (``CatalogEntry.required_files``); ``entry_detail`` names what is missing.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import List, Optional, Tuple

from call1.contracts.catalog import CatalogEntryStatus, ModelPurpose

from call1.process.catalog import BUNDLED_LLM_ENTRY_ID, WHISPER_VOCAB_ENTRY_ID, CatalogEntry, models_root, required_file_problem

LEGACY_PATH_ENV = {
    "parakeet-tdt-0.6b-v3": "CALL1_MLX_ASR_PATH",
    "nemotron-3-diarization": "CALL1_MLX_DIARIZATION_PATH",
    "meralion-ser-v1": "CALL1_TONE_PATH",
    "roberta-sentiment": "CALL1_SENTIMENT_PATH",
    "nemotron-3-embed-1b": "CALL1_EMBEDDING_PATH",
    BUNDLED_LLM_ENTRY_ID: "CALL1_MLX_TEXT_PATH",
    WHISPER_VOCAB_ENTRY_ID: "CALL1_WHISPER_VOCAB_PATH",
}
"""The pre-split environment variable that pins each seeded entry's weights."""

LLM_PURPOSES = frozenset({ModelPurpose.SEMANTIC_QA, ModelPurpose.SUMMARY, ModelPurpose.CONTACT_SIGNALS})


def mlx_backend() -> bool:
    """True when ``CALL1_BACKEND=mlx`` (the legacy pipeline's switch for Apple-Silicon inference)."""
    return os.getenv("CALL1_BACKEND", "") == "mlx"


def local_pack_ids() -> frozenset:
    try:
        from call1.model_catalog import LOCAL_PACKS
    except Exception:  # pragma: no cover - plain data
        return frozenset()
    return frozenset(LOCAL_PACKS)


def weights_path(entry: CatalogEntry) -> Optional[Path]:
    """The directory holding ``entry``'s weights on this host, or ``None`` for a code stage."""
    if entry.model_directory is None:
        return None
    env = LEGACY_PATH_ENV.get(entry.entry_id)
    if env and os.getenv(env):
        return Path(os.environ[env])
    if entry.entry_id in local_pack_ids() and os.getenv("CALL1_OPTIONAL_MODELS_DIR"):
        return Path(os.environ["CALL1_OPTIONAL_MODELS_DIR"]) / entry.model_directory
    return models_root() / entry.model_directory


def _has(path: Optional[Path], *names: str) -> bool:
    return path is not None and all((path / name).is_file() for name in names)


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def weights_installed(entry: CatalogEntry) -> bool:
    """The check each runtime's loader will make: ``config.json`` plus the weights file."""
    path = weights_path(entry)
    if path is None:
        return True
    if not _has(path, "config.json"):
        return False
    if entry.required_files and required_file_problem(entry, path) is not None:
        return False
    if entry.entry_id in local_pack_ids() or entry.runtime == "mlx" and LLM_PURPOSES & set(entry.purposes):
        return any(path.glob("*.safetensors"))
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or any(path.glob("*.npz"))


def entry_status(entry: CatalogEntry) -> Tuple[CatalogEntryStatus, List[ModelPurpose]]:
    """What this host can serve, by runtime. The worker releases a claim whose frozen entry is not
    AVAILABLE here (``model_unavailable`` / ``model_unqualified``) before any inference starts."""
    purposes = list(entry.purposes)
    if not entry.supported:
        return CatalogEntryStatus.UNQUALIFIED, []
    if entry.endpoint_url is not None:
        from call1.process.system_one import entry_status as system_one_status

        return system_one_status(entry)
    if entry.runtime == "code" or entry.model_directory is None:
        return CatalogEntryStatus.AVAILABLE, purposes
    if entry.runtime == "torch":
        if not (_importable("torch") and _importable("transformers")):
            return CatalogEntryStatus.INCOMPATIBLE, []
        return (CatalogEntryStatus.AVAILABLE, purposes) if weights_installed(entry) else (CatalogEntryStatus.NOT_INSTALLED, [])
    if entry.entry_id in local_pack_ids():
        # Local packs always run on MLX (call1.question_models routes them through the adapter).
        if not _importable("mlx_lm"):
            return CatalogEntryStatus.INCOMPATIBLE, []
        return (CatalogEntryStatus.AVAILABLE, purposes) if weights_installed(entry) else (CatalogEntryStatus.NOT_INSTALLED, [])
    if ModelPurpose.ASR_VOCABULARY in entry.purposes:
        # Whisper Small's vocabulary pass runs only on MLX; without it the asr job keeps Parakeet's
        # transcript (base_only, configuration_error).
        if not (mlx_backend() and _importable("mlx") and _importable("mlx_audio") and _importable("tiktoken")):
            return CatalogEntryStatus.INCOMPATIBLE, []
        return (CatalogEntryStatus.AVAILABLE, purposes) if weights_installed(entry) else (CatalogEntryStatus.NOT_INSTALLED, [])
    if not mlx_backend():
        if ModelPurpose.ASR in entry.purposes:
            # faster-whisper resolves CALL1_ASR_MODEL from the local cache when the job runs.
            return (CatalogEntryStatus.AVAILABLE, purposes) if _importable("faster_whisper") else (CatalogEntryStatus.INCOMPATIBLE, [])
        # Diarization is skipped without MLX (mono turns stay UNKNOWN, as the legacy pipeline left
        # them); the included LLM is served by the loopback Ollama, checked when a job runs.
        return CatalogEntryStatus.AVAILABLE, purposes
    module = {ModelPurpose.ASR: "mlx_audio", ModelPurpose.SPEAKER_DIARIZATION: "mlx_audio"}.get(entry.purposes[0], "mlx_lm")
    if not (_importable("mlx") and _importable(module)):
        return CatalogEntryStatus.INCOMPATIBLE, []
    return (CatalogEntryStatus.AVAILABLE, purposes) if weights_installed(entry) else (CatalogEntryStatus.NOT_INSTALLED, [])


def entry_detail(entry: CatalogEntry) -> Optional[str]:
    """Why an entry is not installed here, naming the missing file (the console's ``detail``), or None."""
    if entry.endpoint_url is not None:
        from call1.process.system_one import entry_problem

        return entry_problem(entry)
    path = weights_path(entry)
    if path is None:
        return None
    if ModelPurpose.ASR_VOCABULARY in entry.purposes and not mlx_backend():
        return "needs CALL1_BACKEND=mlx (Apple Silicon)"
    if not _has(path, "config.json"):
        return "config.json is missing from the model directory"
    problem = required_file_problem(entry, path) if entry.required_files else None
    if problem is not None:
        return f"{problem}; provision it with scripts/provision_models.py" if ModelPurpose.ASR_VOCABULARY not in entry.purposes else \
            f"{problem}; provision it with scripts/provision_models.py --models asr_vocabulary"
    if not weights_installed(entry):
        return "the weights are missing from the model directory"
    return None


def reset_peak_memory() -> None:
    """Reset MLX's peak-memory counter before a measured model call (MLX backend only)."""
    if not mlx_backend():
        return
    try:
        import mlx.core as mx

        reset = getattr(mx, "reset_peak_memory", None)
        if reset is not None:
            reset()
    except Exception:  # pragma: no cover - measurement only
        pass


def peak_memory() -> Optional[int]:
    """MLX's peak memory since the last reset, in bytes (MLX backend only)."""
    if not mlx_backend():
        return None
    try:
        import mlx.core as mx

        value = mx.get_peak_memory()
        return int(value) if value else None
    except Exception:  # pragma: no cover - measurement only
        return None


def backend_label() -> str:
    return "mlx" if mlx_backend() else "cpu (faster-whisper, torch, loopback Ollama)"


__all__ = ["LEGACY_PATH_ENV", "backend_label", "entry_detail", "entry_status", "mlx_backend", "peak_memory", "reset_peak_memory", "weights_installed",
           "weights_path"]
