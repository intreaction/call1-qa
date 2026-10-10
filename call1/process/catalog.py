"""The Process model catalog: purpose-tagged entries, per-purpose defaults, frozen selections and
the read-only snapshot Process publishes to Store (``PUT /catalog-snapshot``).

Process owns the catalog (weights, adapters, routes); Store only stores references and the
snapshot. The seeded entries are the models the appliance ships today:

* ``model-manifest.json`` pins the included weights: Parakeet TDT 0.6B v3 (ASR), Nemotron-3-Diarization
  (diarization), MERaLiON SER (acoustic tone), Cardiff RoBERTa (text sentiment) and Gemma 4 E2B
  (the included LLM, ``call1-bundled`` in ``call1.question_models``).
* ``call1.model_catalog.LOCAL_PACKS`` adds the optional local LLM packs.
* ``nemotron-3-embed-1b`` is the search embedder (``call1.embedding``: Nemotron-3-Embed-1B at a
  pinned revision, the same model Store embeds queries with; contract 1.2.0).
* ``call1-bundled`` also serves all three Contact Signals v2 purposes (contract 1.3.0):
  ``signal_category``, ``signal_subcategory`` and ``signal_extraction``, and is their real-mode
  default (team decision 24, which replaces decision 21's Laya/Needle engines and Q15).
* ``laya-system-one`` is the optional experimental local Ollama decision engine, for
  ``signal_category`` only. Confirmation and extraction keep their separate selections.
* In fake mode only, ``fake-signal-classifier`` (both classifier purposes) and
  ``fake-signal-extractor`` (``signal_extraction``) register, like ``fake-embedding-v1``, and are the
  defaults for their purposes. They are labelled "fake" wherever an engine name shows.
* ``whisper-small-vocab`` is Whisper Small (MLX, MIT) for dual transcription's vocabulary pass
  (contract 1.3.0, team decision 33, docs/DualAsr.md): purpose ``asr_vocabulary``, never a job's own
  selection. The ``asr`` job freezes it in ``parameters.asr_vocabulary.candidate_entry``. Installed
  means ``config.json``, the weights and the bundled ``multilingual.tiktoken`` (checksum-pinned,
  ``WHISPER_TOKENIZER_SHA256``) are all present (``CatalogEntry.required_files``).
* ``openai-privacy-filter`` is the PII masking model (``call1.pii_model``, team decision 19). It
  serves no contract ``ModelPurpose`` (the contract has no masking purpose, so no job selects it):
  the masking step inside the masked text-model jobs runs it. The entry lists it for the console and
  the published snapshot, with its install status and license.

Curated weights run in-process on the appliance route; Laya runs on loopback Ollama. The contract's ``ProviderType`` has no
value for an in-process runtime other than MLX, so the torch-based tone, sentiment and embedding
models are recorded as provider ``mlx`` with destination ``in-process``; the Process
entry's ``runtime`` field says what really runs (raised as a contract gap in the README).

A job freezes a ``FrozenSelection`` (entry, revision, adapter, output contract, route) when it is
created; a later catalog change affects only new jobs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from call1 import embedding, pii_model
from call1.contracts.catalog import (
    CatalogEntryRef,
    CatalogEntrySnapshot,
    CatalogEntryStatus,
    CatalogSnapshot,
    FrozenSelection,
    ModelPurpose,
    ResourceProfile,
)
from call1.contracts.common import canonical_digest
from call1.contracts.custody import IN_PROCESS_DESTINATION, ProviderType, RouteClass, RouteRecord
from call1.contracts.jobs import MemorySlot

EMBEDDING_ENTRY_ID = embedding.MODEL_DIRECTORY
PII_MODEL_ENTRY_ID = pii_model.MODEL_DIRECTORY
BUNDLED_LLM_ENTRY_ID = "call1-bundled"
FAKE_SIGNAL_CLASSIFIER_ID = "fake-signal-classifier"
FAKE_SIGNAL_EXTRACTOR_ID = "fake-signal-extractor"
WHISPER_VOCAB_ENTRY_ID = "whisper-small-vocab"

WHISPER_TOKENIZER_FILE = "multilingual.tiktoken"
WHISPER_TOKENIZER_SHA256 = "b34b360dbb493e781e479794586d661700670d65564001f23024971d1f2fa126"
"""openai-whisper's multilingual tiktoken vocabulary (v20250625), which install bundles in the Whisper
Small model directory (the MLX weights ship no tokenizer). Never fetched at inference time."""

PURPOSE_OUTPUT_CONTRACTS: Dict[ModelPurpose, str] = {
    ModelPurpose.ASR: "transcript.v1",
    ModelPurpose.SPEAKER_DIARIZATION: "speaker_attribution.v1",
    ModelPurpose.ACOUSTIC_TONE: "tone_blocks.v1",
    ModelPurpose.TEXT_SENTIMENT: "text_sentiment.v1",
    ModelPurpose.EMBEDDINGS: "embeddings.v1",
    ModelPurpose.SEMANTIC_QA: "qa_assessment.v1",
    ModelPurpose.SUMMARY: "summary_segment.v1",
    ModelPurpose.CONTACT_SIGNALS: "contact_signals_pass.v1",
    ModelPurpose.SIGNAL_CATEGORY: "signal_categories.v1",
    ModelPurpose.SIGNAL_SUBCATEGORY: "signal_subcategories.v1",
    ModelPurpose.SIGNAL_EXTRACTION: "signal_extraction.v1",
    ModelPurpose.ASR_VOCABULARY: "asr_vocabulary_pass.v1",
}
"""The default output contract per purpose; summary synthesis jobs freeze ``summary_synthesis.v1``."""


class CatalogError(ValueError):
    """A selection the catalog cannot make: unknown entry, wrong purpose, not available here."""


@dataclass(frozen=True)
class CatalogEntry:
    entry_id: str
    display_name: str
    purposes: Tuple[ModelPurpose, ...]
    model_family: str
    model_revision: str
    provider_model_id: str
    adapter_id: str
    runtime: str
    """What actually runs: ``mlx``, ``torch``, ``code`` or ``ollama``."""
    entry_version: int = 1
    adapter_version: str = "1"
    provider_type: ProviderType = ProviderType.MLX
    route_class: RouteClass = RouteClass.APPLIANCE
    destination_host: str = IN_PROCESS_DESTINATION
    model_directory: Optional[str] = None
    """Directory under the models root holding the weights; ``None`` for code stages."""
    supported: bool = True
    license_notice: Optional[str] = None
    legacy_question_model_id: Optional[str] = None
    memory_bytes: Optional[int] = None
    context_limit_tokens: Optional[int] = None
    output_token_limit: Optional[int] = None
    weights_digest: Optional[str] = None
    required_files: Tuple[Tuple[str, Optional[str]], ...] = ()
    """Files besides ``config.json`` and the weights that must be in the model directory for the entry
    to count as installed, each with its pinned sha256 (hex) or None."""
    endpoint_url: Optional[str] = None
    """Local System One base URL. Validated as loopback; never a credential."""
    mutable_alias: bool = False

    @property
    def ref(self) -> CatalogEntryRef:
        return CatalogEntryRef(entry_id=self.entry_id, entry_version=self.entry_version)

    @property
    def memory_slot(self) -> MemorySlot:
        """The contract memory slot a job on this entry needs."""
        if self.route_class is not RouteClass.APPLIANCE:
            return MemorySlot.OUTBOUND
        if self.runtime in ("mlx", "ollama"):
            return MemorySlot.LOCAL_MEMORY
        return MemorySlot.CPU

    def route(self, masked: bool = False) -> RouteRecord:
        return RouteRecord(route_class=self.route_class, provider_type=self.provider_type, destination_host=self.destination_host,
                           masked=masked)

    def selection(self, purpose: ModelPurpose, *, output_contract: Optional[str] = None, masked: bool = False) -> FrozenSelection:
        if purpose not in self.purposes:
            raise CatalogError(f"{self.entry_id} does not serve {purpose.value}")
        return FrozenSelection(
            catalog_entry=self.ref, purpose=purpose, model_family=self.model_family, model_revision=self.model_revision,
            weights_digest=self.weights_digest, adapter_id=self.adapter_id, adapter_version=self.adapter_version,
            output_contract=output_contract or PURPOSE_OUTPUT_CONTRACTS[purpose], provider_model_id=self.provider_model_id,
            route=self.route(masked=masked), mutable_alias=self.mutable_alias,
        )


EntryStatusFn = Callable[[CatalogEntry], Tuple[CatalogEntryStatus, List[ModelPurpose]]]


def models_root() -> Path:
    return Path(os.getenv("CALL1_MODELS_DIR", "data/models"))


def installed_status(entry: CatalogEntry, root: Optional[Path] = None) -> Tuple[CatalogEntryStatus, List[ModelPurpose]]:
    """Real-mode status: code stages are always available; weights must be installed under the
    models root; unsupported packs are unqualified. Installed curated entries count as qualified
    for their purposes (the appliance qualification gates run locally, not here)."""
    if not entry.supported:
        return CatalogEntryStatus.UNQUALIFIED, []
    if entry.endpoint_url is not None:
        from .system_one import entry_status

        return entry_status(entry)
    if entry.runtime == "code" or entry.model_directory is None:
        return CatalogEntryStatus.AVAILABLE, list(entry.purposes)
    path = (root or models_root()) / entry.model_directory
    if not path.is_dir() or not any(path.iterdir()):
        return CatalogEntryStatus.NOT_INSTALLED, []
    if required_file_problem(entry, path) is not None:
        return CatalogEntryStatus.NOT_INSTALLED, []
    return CatalogEntryStatus.AVAILABLE, list(entry.purposes)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_file_problem(entry: CatalogEntry, path: Optional[Path]) -> Optional[str]:
    """Why ``entry.required_files`` are not satisfied in ``path`` (safe text naming the file), or None."""
    for name, sha256 in entry.required_files:
        target = path / name if path is not None else None
        if target is None or not target.is_file():
            return f"{name} is missing from the model directory"
        if sha256 is not None and _sha256(target) != sha256:
            return f"{name} does not match its pinned checksum"
    return None


def fake_status(entry: CatalogEntry) -> Tuple[CatalogEntryStatus, List[ModelPurpose]]:
    """Fake-handler mode: every supported entry is served by a fake handler."""
    if not entry.supported:
        return CatalogEntryStatus.UNQUALIFIED, []
    return CatalogEntryStatus.AVAILABLE, list(entry.purposes)


DEFAULT_PURPOSE_ENTRIES: Dict[ModelPurpose, str] = {
    ModelPurpose.ASR: "parakeet-tdt-0.6b-v3",
    ModelPurpose.SPEAKER_DIARIZATION: "nemotron-3-diarization",
    ModelPurpose.ACOUSTIC_TONE: "meralion-ser-v1",
    ModelPurpose.TEXT_SENTIMENT: "roberta-sentiment",
    ModelPurpose.EMBEDDINGS: EMBEDDING_ENTRY_ID,
    ModelPurpose.SEMANTIC_QA: BUNDLED_LLM_ENTRY_ID,
    ModelPurpose.SUMMARY: BUNDLED_LLM_ENTRY_ID,
    ModelPurpose.CONTACT_SIGNALS: BUNDLED_LLM_ENTRY_ID,
    ModelPurpose.SIGNAL_CATEGORY: BUNDLED_LLM_ENTRY_ID,
    ModelPurpose.SIGNAL_SUBCATEGORY: BUNDLED_LLM_ENTRY_ID,
    ModelPurpose.SIGNAL_EXTRACTION: BUNDLED_LLM_ENTRY_ID,
    ModelPurpose.ASR_VOCABULARY: WHISPER_VOCAB_ENTRY_ID,
}
"""Real-mode defaults. The included model serves every Contact Signals v2 stage (decision 24)."""

FAKE_PURPOSE_ENTRIES: Dict[ModelPurpose, str] = {
    ModelPurpose.SIGNAL_CATEGORY: FAKE_SIGNAL_CLASSIFIER_ID,
    ModelPurpose.SIGNAL_SUBCATEGORY: FAKE_SIGNAL_CLASSIFIER_ID,
    ModelPurpose.SIGNAL_EXTRACTION: FAKE_SIGNAL_EXTRACTOR_ID,
}
"""Fake mode's Contact Signals v2 defaults (section 8.6)."""


class ProcessCatalog:
    def __init__(self, entries: Iterable[CatalogEntry], defaults: Mapping[ModelPurpose, str], status_fn: EntryStatusFn = fake_status,
                 detail_fn: Optional[Callable[[CatalogEntry], Optional[str]]] = None) -> None:
        self.entries: Dict[str, CatalogEntry] = {}
        for entry in entries:
            if entry.entry_id in self.entries:
                raise CatalogError(f"duplicate catalog entry {entry.entry_id}")
            self.entries[entry.entry_id] = entry
        self.defaults: Dict[ModelPurpose, str] = dict(defaults)
        self.status_fn = status_fn
        self.detail_fn = detail_fn
        """Optional: why an entry is not installed here (the console's ``detail``; e.g. a missing file)."""

    # --- lookup ------------------------------------------------------------------------------

    def get(self, entry_id: str) -> CatalogEntry:
        entry = self.entries.get(entry_id)
        if entry is None:
            entry = next((e for e in self.entries.values() if e.legacy_question_model_id == entry_id), None)
        if entry is None:
            raise CatalogError(f"catalog entry {entry_id!r} is not in this Process catalog")
        return entry

    def by_ref(self, ref: CatalogEntryRef) -> Optional[CatalogEntry]:
        entry = self.entries.get(ref.entry_id)
        return entry if entry is not None and entry.entry_version == ref.entry_version else None

    def status(self, entry: CatalogEntry) -> Tuple[CatalogEntryStatus, List[ModelPurpose]]:
        return self.status_fn(entry)

    def usable(self, entry: CatalogEntry, purpose: ModelPurpose) -> bool:
        status, qualified = self.status(entry)
        return status is CatalogEntryStatus.AVAILABLE and purpose in qualified

    def select(self, purpose: ModelPurpose, entry_id: Optional[str] = None) -> CatalogEntry:
        """The entry a new job of ``purpose`` freezes: ``entry_id`` when given (a rubric's
        ``primary_model_id``), else this installation's default. It must be available and qualified."""
        chosen = entry_id or self.defaults.get(purpose)
        if not chosen:
            raise CatalogError(f"no default catalog entry for {purpose.value}")
        entry = self.get(chosen)
        if purpose not in entry.purposes:
            raise CatalogError(f"{entry.entry_id} does not serve {purpose.value}")
        if not self.usable(entry, purpose):
            status, _ = self.status(entry)
            raise CatalogError(f"{entry.entry_id} is {status.value} on this installation for {purpose.value}")
        return entry

    def qualified_entries(self) -> List[CatalogEntryRef]:
        refs = []
        for entry in self.entries.values():
            status, qualified = self.status(entry)
            if status is CatalogEntryStatus.AVAILABLE and qualified:
                refs.append(entry.ref)
        return refs

    # --- snapshot ----------------------------------------------------------------------------

    def entry_snapshots(self) -> List[CatalogEntrySnapshot]:
        items = []
        for entry in self.entries.values():
            status, qualified = self.status(entry)
            items.append(CatalogEntrySnapshot(
                entry=entry.ref, display_name=entry.display_name, purposes=list(entry.purposes), provider_type=entry.provider_type,
                route_class=entry.route_class, destination_host=entry.destination_host, model_family=entry.model_family,
                model_revision=entry.model_revision, weights_digest=entry.weights_digest, status=status, qualified_for=qualified,
                resource_profile=ResourceProfile(memory_bytes=entry.memory_bytes, context_limit_tokens=entry.context_limit_tokens,
                                                 output_token_limit=entry.output_token_limit),
                license_notice=(entry.license_notice[:197] + "...") if entry.license_notice and len(entry.license_notice) > 200 else entry.license_notice,
                mutable_alias=entry.mutable_alias, legacy_question_model_id=entry.legacy_question_model_id,
            ))
        return items

    def version(self) -> str:
        """A short content digest of the entries, their status and the defaults."""
        body = {"entries": [e.model_dump(mode="json") for e in self.entry_snapshots()],
                "defaults": {p.value: e for p, e in sorted(self.defaults.items(), key=lambda kv: kv[0].value)}}
        return "c-" + canonical_digest(body)[len("sha256:"):][:16]

    def snapshot(self, installation_id: str, now: datetime) -> CatalogSnapshot:
        defaults = {}
        for purpose, entry_id in self.defaults.items():
            entry = self.entries.get(entry_id)
            if entry is not None:
                defaults[purpose] = entry.ref
        return CatalogSnapshot(installation_id=installation_id, catalog_version=self.version(), published_at=now,
                               entries=self.entry_snapshots(), defaults=defaults)

    def describe(self) -> List[dict]:
        """The console's Models view: entries with status, runtime and which purposes default to them."""
        out = []
        for entry in self.entries.values():
            status, qualified = self.status(entry)
            out.append({
                "entry_id": entry.entry_id, "entry_version": entry.entry_version, "display_name": entry.display_name,
                "purposes": [p.value for p in entry.purposes], "default_for": sorted(p.value for p, e in self.defaults.items() if e == entry.entry_id),
                "status": status.value, "qualified_for": [p.value for p in qualified], "runtime": entry.runtime,
                "provider_type": entry.provider_type.value, "route_class": entry.route_class.value, "destination_host": entry.destination_host,
                "model_family": entry.model_family, "model_revision": entry.model_revision, "adapter_id": entry.adapter_id,
                "adapter_version": entry.adapter_version, "license_notice": entry.license_notice,
                "legacy_question_model_id": entry.legacy_question_model_id,
                "detail": self.detail_fn(entry) if (self.detail_fn is not None and status is not CatalogEntryStatus.AVAILABLE) else None,
            })
        return out

    def with_status(self, status_fn: EntryStatusFn, detail_fn: Optional[Callable[[CatalogEntry], Optional[str]]] = None) -> "ProcessCatalog":
        return ProcessCatalog(self.entries.values(), self.defaults, status_fn, detail_fn or self.detail_fn)


# --- the seeded catalog ------------------------------------------------------------------------


def manifest_path() -> Path:
    configured = os.getenv("CALL1_MODEL_MANIFEST")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "model-manifest.json"


def read_manifest(path: Optional[Path] = None) -> Dict[str, dict]:
    path = path or manifest_path()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _pinned(manifest: Dict[str, dict], key: str) -> Tuple[str, str, Optional[str], Optional[str]]:
    item = manifest.get(key) or {}
    return (item.get("repository") or f"unknown/{key}", item.get("revision") or "unpinned", item.get("directory"), item.get("notice"))


def _bundled_question_model() -> Tuple[str, str, int]:
    """The included LLM as the pre-split question-model registry defines it
    (``QuestionModelsSettings``, which ``call1.question_models`` routes): its stable ID, name and
    output-token cap. The catalog entry keeps that ID so rubrics that name it resolve unchanged."""
    try:
        from call1.models.schemas import QuestionModelsSettings

        registry = QuestionModelsSettings()
        bundled = next(m for m in registry.models if m.source == "internal")
        return bundled.id, bundled.name[:200], bundled.max_tokens
    except Exception:  # pragma: no cover - the defaults are plain data
        return BUNDLED_LLM_ENTRY_ID, "Call1 Included · Gemma 4 E2B", 768


def seeded_entries(manifest: Optional[Dict[str, dict]] = None) -> List[CatalogEntry]:
    manifest = read_manifest() if manifest is None else manifest
    entries: List[CatalogEntry] = []

    repo, rev, directory, notice = _pinned(manifest, "asr")
    entries.append(CatalogEntry(
        entry_id="parakeet-tdt-0.6b-v3", display_name="Parakeet TDT 0.6B v3 (MLX)", purposes=(ModelPurpose.ASR,), model_family="parakeet",
        model_revision=rev, provider_model_id=repo, adapter_id="call1.mlx.parakeet", runtime="mlx",
        model_directory=directory or "parakeet-tdt-0.6b-v3", license_notice=notice,
        memory_bytes=2_300_000_000))  # measured peak: bf16 weights plus 120 s decoding windows
    repo, rev, directory, notice = _pinned(manifest, "asr_vocabulary")
    entries.append(CatalogEntry(
        entry_id=WHISPER_VOCAB_ENTRY_ID, display_name="Whisper Small, vocabulary pass (MLX)", purposes=(ModelPurpose.ASR_VOCABULARY,),
        model_family="whisper", model_revision=rev, provider_model_id=repo, adapter_id="call1.mlx.whisper_vocabulary", runtime="mlx",
        model_directory=directory or "whisper-small", license_notice=notice or "Whisper Small by OpenAI, MIT; MLX conversion by MLX Community.",
        required_files=((WHISPER_TOKENIZER_FILE, WHISPER_TOKENIZER_SHA256),),
        memory_bytes=1_000_000_000))  # fp16 small: under 1 GB (benchmarks/2026-09-26-dual-asr-merge.md)
    repo, rev, directory, notice = _pinned(manifest, "diarization")
    entries.append(CatalogEntry(
        entry_id="nemotron-3-diarization", display_name="Nemotron-3-Diarization, up to 8 speakers (MLX)",
        purposes=(ModelPurpose.SPEAKER_DIARIZATION,), model_family="nemotron-diarization", model_revision=rev, provider_model_id=repo,
        adapter_id="call1.mlx.nemotron_diarization", runtime="mlx", model_directory=directory or "nemotron-3-diarization",
        license_notice=notice, memory_bytes=700_000_000))  # measured peak on 9-minute calls: ~0.62 GiB
    repo, rev, directory, notice = _pinned(manifest, "tone")
    entries.append(CatalogEntry(
        entry_id="meralion-ser-v1", display_name="MERaLiON SER v1", purposes=(ModelPurpose.ACOUSTIC_TONE,), model_family="meralion-ser",
        model_revision=rev, provider_model_id=repo, adapter_id="call1.torch.meralion", runtime="torch",
        model_directory=directory or "meralion-ser-v1", license_notice=notice))
    repo, rev, directory, notice = _pinned(manifest, "sentiment")
    entries.append(CatalogEntry(
        entry_id="roberta-sentiment", display_name="Cardiff RoBERTa sentiment", purposes=(ModelPurpose.TEXT_SENTIMENT,),
        model_family="roberta", model_revision=rev, provider_model_id=repo, adapter_id="call1.torch.roberta_sentiment", runtime="torch",
        model_directory=directory or "roberta-sentiment", license_notice=notice))
    entries.append(CatalogEntry(
        entry_id=EMBEDDING_ENTRY_ID, display_name="Nemotron 3 Embed 1B", purposes=(ModelPurpose.EMBEDDINGS,),
        model_family="nemotron-3-embed", model_revision=embedding.MODEL_REVISION, provider_model_id=embedding.MODEL_REPOSITORY,
        adapter_id="call1.torch.nemotron_embed", runtime="torch", model_directory=embedding.MODEL_DIRECTORY,
        license_notice=f"Nemotron-3-Embed-1B-BF16 by NVIDIA Corporation. {embedding.MODEL_LICENSE}; see the LICENSE and NOTICE shipped with the weights."))
    entries.append(CatalogEntry(
        entry_id=PII_MODEL_ENTRY_ID, display_name="OpenAI Privacy Filter (PII masking)", purposes=(),
        model_family="openai-privacy-filter", model_revision=pii_model.MODEL_REVISION, provider_model_id=pii_model.MODEL_REPOSITORY,
        adapter_id="call1.torch.privacy_filter", runtime="torch", model_directory=pii_model.MODEL_DIRECTORY,
        license_notice=f"OpenAI Privacy Filter by OpenAI. {pii_model.MODEL_LICENSE}; see the LICENSE shipped with the weights.",
        memory_bytes=3_300_000_000))  # measured: ~2.8 GB bf16 weights resident on MPS plus activations
    repo, rev, directory, notice = _pinned(manifest, "text")
    llm = (ModelPurpose.SEMANTIC_QA, ModelPurpose.SUMMARY, ModelPurpose.CONTACT_SIGNALS)
    bundled_id, bundled_name, max_tokens = _bundled_question_model()
    entries.append(CatalogEntry(
        entry_id=bundled_id, display_name=bundled_name, purposes=llm + (ModelPurpose.SIGNAL_CATEGORY, ModelPurpose.SIGNAL_SUBCATEGORY, ModelPurpose.SIGNAL_EXTRACTION), model_family="gemma-4",
        model_revision=rev, provider_model_id=repo, adapter_id="call1.mlx.text", runtime="mlx", model_directory=directory or "gemma-4-e2b-it",
        license_notice=notice, legacy_question_model_id=bundled_id, context_limit_tokens=32768, output_token_limit=max_tokens))
    try:
        from call1.model_catalog import LOCAL_PACKS
    except Exception:  # pragma: no cover - the module is plain data
        LOCAL_PACKS = {}
    for pack_id, pack in LOCAL_PACKS.items():
        entries.append(CatalogEntry(
            entry_id=pack_id, display_name=pack["name"], purposes=llm, model_family="gemma-4", model_revision=f"local-pack:{pack['model']}",
            provider_model_id=pack["model"], adapter_id="call1.mlx.text", runtime="mlx", model_directory=pack["directory"],
            supported=bool(pack.get("supported")), legacy_question_model_id=pack_id, context_limit_tokens=32768, output_token_limit=768))
    return entries


def fake_signal_entries() -> List[CatalogEntry]:
    """The fake Contact Signals v2 engines (fake mode only; section 8.6): keyword-matching stand-ins
    for the stage-1/2 classifier and the stage-3 extractor, labelled fake."""
    return [
        CatalogEntry(entry_id=FAKE_SIGNAL_CLASSIFIER_ID, display_name="Fake signal classifier (fake)",
                     purposes=(ModelPurpose.SIGNAL_CATEGORY, ModelPurpose.SIGNAL_SUBCATEGORY), model_family="fake",
                     model_revision="fake-signals-v1", provider_model_id="fake", adapter_id="call1.fake.signal_classifier", runtime="torch",
                     license_notice="Fake engine for tests and demos; not a model."),
        CatalogEntry(entry_id=FAKE_SIGNAL_EXTRACTOR_ID, display_name="Fake signal extractor (fake)", purposes=(ModelPurpose.SIGNAL_EXTRACTION,),
                     model_family="fake", model_revision="fake-signals-v1", provider_model_id="fake", adapter_id="call1.fake.signal_extractor",
                     runtime="code", license_notice="Fake engine for tests and demos; not a model."),
    ]


def seeded_catalog(*, mode: str = "fake", overrides: Optional[Mapping[str, str]] = None, manifest: Optional[Dict[str, dict]] = None,
                   status_fn: Optional[EntryStatusFn] = None, system_one_url: Optional[str] = None,
                   system_one_model: str = "laya") -> ProcessCatalog:
    """The catalog this installation starts with. ``overrides`` maps a purpose value to an entry ID
    (``model_defaults`` in the config)."""
    entries = seeded_entries(manifest)
    if mode == "real" and system_one_url is not None:
        from .system_one import catalog_entry

        entries.append(catalog_entry(system_one_url, system_one_model))
    defaults = dict(DEFAULT_PURPOSE_ENTRIES)
    if mode == "fake":
        entries += fake_signal_entries()
        defaults.update(FAKE_PURPOSE_ENTRIES)
    known = {e.entry_id for e in entries}
    for purpose_value, entry_id in (overrides or {}).items():
        try:
            purpose = ModelPurpose(purpose_value)
        except ValueError:
            raise CatalogError(f"model_defaults: unknown purpose {purpose_value!r}") from None
        if entry_id not in known:
            raise CatalogError(f"model_defaults: {entry_id!r} is not in the catalog")
        defaults[purpose] = entry_id
    if mode == "real" and system_one_url is not None:
        from .system_one import ENTRY_ID

        defaults[ModelPurpose.SIGNAL_CATEGORY] = ENTRY_ID
    fn = status_fn or (fake_status if mode == "fake" else installed_status)
    return ProcessCatalog(entries, defaults, fn)


def replace_entry(catalog: ProcessCatalog, entry: CatalogEntry) -> ProcessCatalog:
    entries = [entry if e.entry_id == entry.entry_id else e for e in catalog.entries.values()]
    if entry.entry_id not in catalog.entries:
        entries.append(entry)
    return ProcessCatalog(entries, catalog.defaults, catalog.status_fn, catalog.detail_fn)


__all__ = [
    "CatalogEntry", "CatalogError", "ProcessCatalog", "seeded_catalog", "seeded_entries", "installed_status", "fake_status",
    "EMBEDDING_ENTRY_ID", "PII_MODEL_ENTRY_ID", "BUNDLED_LLM_ENTRY_ID", "PURPOSE_OUTPUT_CONTRACTS", "replace_entry",
    "FAKE_SIGNAL_CLASSIFIER_ID", "FAKE_SIGNAL_EXTRACTOR_ID", "fake_signal_entries", "WHISPER_VOCAB_ENTRY_ID", "WHISPER_TOKENIZER_FILE",
    "WHISPER_TOKENIZER_SHA256", "required_file_problem",
]
