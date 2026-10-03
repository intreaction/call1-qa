"""Ingest: a recording in, a registered conversation with its source audio and job graph out.

1. Copy the upload into scratch while hashing it (SHA-256) and probe the container.
2. ``POST /conversations`` with the source identity (content digest) and the call metadata the
   caller supplied, and only that (contract 1.1.0): a repeat returns the same conversation
   (``created: false``), and a repeat with different metadata updates the call's metadata
   (``metadata_updated: true``, audited by Store, nothing reprocessed). Fields the caller leaves
   out are not sent, so a re-upload without an agent never resets a known agent to ``Unknown``.
3. Upload the ``source_audio`` artifact through an upload grant and commit it (idempotent by
   conversation, kind, slot and checksum).
4. Mint the rubric snapshot of the configured rubric's current published version. Then read the
   signal taxonomy and settings (contract 1.3.0); when they select Contact Signals v2, mint a
   snapshot of the current taxonomy version (``signals_input.ingest_signals``). Read the ASR
   vocabulary (contract 1.3.0, decision 33): when it is active, the ``asr`` job freezes it for dual
   transcription (``vocabulary.read_vocabulary``; a Store without the route means none).
5. Build the Stage 2 graph and ``POST /conversations/{id}/job-graphs`` with a stable idempotency
   key, so a repeated ingest returns the same graph.

Ingestion is acknowledged only after step 5. The scratch copy is removed when Store holds the
bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

from pydantic import ValidationError

from call1.contracts.artifacts import ArtifactKind, Sensitivity, UploadGrantRequest
from call1.contracts.calls import CallMetadata, ConversationRegistration, IngestionKind, SourceKind, SourceReference, agent_label
from call1.contracts.jobs import JobType

from .audio import AudioInfo, probe
from .config import ProcessConfig
from .graph import GraphPlanner, bounded_key
from .ledger import Ledger
from .scratch import Scratch
from .signals_input import ingest_signals
from .store_client import StoreClient, StoreError
from .vocabulary import read_vocabulary


@dataclass(frozen=True)
class IngestResult:
    conversation_id: str
    call_id: Optional[str]
    graph_id: str
    conversation_created: bool
    graph_created: bool
    source_artifact_id: str
    jobs: int
    evaluate_url: Optional[str]
    # Contract 1.1.0: a re-upload with different call metadata updated the call (nothing reprocessed).
    metadata_updated: bool = False
    updated_fields: List[str] = field(default_factory=list)
    # The call's agent as every client shows it (calls.agent_label), after any update.
    agent_label: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class InvalidCallMetadata(ValueError):
    """The supplied call metadata does not satisfy the contract (a 422, not a bad recording)."""


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def build_call_metadata(*, agent_id: Optional[str] = None, agent_display_name: Optional[str] = None, agent_extension: Optional[str] = None,
                        agent_channel: Optional[int] = None, external_call_ref: Optional[str] = None,
                        recorded_at: Optional[datetime] = None) -> CallMetadata:
    """``CallMetadata`` holding only what the caller supplied. Blank values count as not supplied.
    Unset fields stay out of the registration body, so Store keeps their stored values on a
    re-upload (``merge_call_metadata``); on a first upload Store applies the contract defaults
    (``agent_id`` "Unknown")."""
    supplied: Dict[str, Any] = {
        "agent_id": (_clean(agent_id) or "")[:200] or None,
        "agent_display_name": _clean(agent_display_name),
        "agent_extension": _clean(agent_extension),
        "agent_channel": agent_channel,
        "external_call_ref": (_clean(external_call_ref) or "")[:200] or None,
        "recorded_at": recorded_at,
    }
    try:
        return CallMetadata(**{name: value for name, value in supplied.items() if value is not None})
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            name = ".".join(str(part) for part in error.get("loc", ())) or "call_metadata"
            if name == "agent_display_name":
                problems.append("agent_display_name must be 1 to 100 characters")
            elif name == "agent_extension":
                problems.append("agent_extension must be 1 to 20 characters: digits, letters, * # + . _ -")
            elif name == "agent_channel":
                problems.append("agent_channel is 0 or 1")
            else:
                problems.append(f"{name} is not valid")
        raise InvalidCallMetadata("; ".join(dict.fromkeys(problems))) from None


def copy_and_hash(source: BinaryIO, dest: Path, max_bytes: Optional[int] = None) -> str:
    digest = hashlib.sha256()
    size = 0
    with dest.open("wb") as handle:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise ValueError("The recording is larger than this Process accepts")
            digest.update(chunk)
            handle.write(chunk)
    return "sha256:" + digest.hexdigest()


class Ingestor:
    def __init__(self, config: ProcessConfig, client: StoreClient, planner: GraphPlanner, ledger: Ledger, scratch: Scratch) -> None:
        self.config = config
        self.client = client
        self.planner = planner
        self.ledger = ledger
        self.scratch = scratch

    def ingest_stream(self, stream: BinaryIO, *, filename: str, content_type: Optional[str] = None, source_kind: SourceKind = SourceKind.API_UPLOAD,
                      agent_id: Optional[str] = None, agent_display_name: Optional[str] = None, agent_extension: Optional[str] = None,
                      agent_channel: Optional[int] = None, external_call_ref: Optional[str] = None,
                      recorded_at: Optional[datetime] = None, max_bytes: Optional[int] = None) -> IngestResult:
        # Validate the metadata before reading the recording (raises InvalidCallMetadata).
        metadata = build_call_metadata(agent_id=agent_id, agent_display_name=agent_display_name, agent_extension=agent_extension,
                                       agent_channel=agent_channel, external_call_ref=external_call_ref, recorded_at=recorded_at)
        workdir = self.scratch.ingest_dir()
        try:
            local = workdir / ("recording" + (Path(filename).suffix.lower() or ".bin"))
            digest = copy_and_hash(stream, local, max_bytes)
            return self._ingest(local, digest, filename=filename, content_type=content_type, source_kind=source_kind, metadata=metadata)
        finally:
            Scratch.remove(workdir)

    def ingest_file(self, path: Path, **kwargs) -> IngestResult:
        path = Path(path)
        with path.open("rb") as handle:
            return self.ingest_stream(handle, filename=path.name, source_kind=kwargs.pop("source_kind", SourceKind.LOCAL_IMPORT), **kwargs)

    def _ingest(self, local: Path, digest: str, *, filename: str, content_type: Optional[str], source_kind: SourceKind,
                metadata: CallMetadata) -> IngestResult:
        info: AudioInfo = probe(local, filename=filename, content_type=content_type)
        registration = ConversationRegistration(
            ingestion_kind=IngestionKind.CALL_AUDIO,
            source=SourceReference(kind=source_kind, content_digest=digest, received_at=datetime.now(timezone.utc)),
            call_metadata=metadata,
        )
        registered = self.client.register_conversation(registration)
        conversation = registered.conversation
        stored = conversation.call_metadata
        # The channel the graph uses: the caller's, else the call's stored one, else 0 on stereo.
        agent_channel = metadata.agent_channel if "agent_channel" in metadata.model_fields_set else None
        if agent_channel is None and stored is not None:
            agent_channel = stored.agent_channel
        if agent_channel is None and info.channels == 2:
            agent_channel = 0
        existing = [a for a in self.client.list_artifacts(conversation.id, kind=ArtifactKind.SOURCE_AUDIO) if a.checksum == digest]
        if existing:
            source = existing[0]
        else:
            request = UploadGrantRequest(kind=ArtifactKind.SOURCE_AUDIO, slot="", content_type=info.content_type, size_bytes=info.size_bytes,
                                         checksum=digest, content_contract="audio.v1", sensitivity=Sensitivity.RAW, labels={"container": info.container})
            source = self.client.upload_artifact(conversation.id, request, local)
        rubric = self.client.get_rubric(self.config.rubric_id)
        snapshot = self.client.mint_rubric_snapshot(conversation.id, rubric.ref.rubric_id, rubric.ref.version)
        signals = ingest_signals(self.client, conversation.id) if self.config.stages.contact_signals else None
        vocabulary = self.planner.asr_vocabulary(read_vocabulary(self.client))

        def plan(channel: Optional[int], asr_vocabulary):
            request = self.planner.ingest(conversation_id=conversation.id, source=source, audio=info, rubric_ref=rubric.ref,
                                          rubric=rubric.definition, snapshot=snapshot, agent_channel=channel, signals=signals,
                                          asr_vocabulary=asr_vocabulary)
            return self.client.create_job_graph(conversation.id, request)

        try:
            graph = plan(agent_channel, vocabulary)
        except StoreError as exc:
            # A re-upload that changed agent_channel, or after the ASR vocabulary changed: the ingest
            # graph already exists as first planned. Nothing is reprocessed (contract 1.1.0), so
            # return that graph unchanged.
            original = self._ingest_graph_channel(conversation.id) if exc.code == "idempotency_key_reused" and not registered.created else False
            if original is False:
                raise
            graph = plan(original, self._ingest_graph_vocabulary(conversation.id))
        self.ledger.record(conversation_id=conversation.id, call_id=conversation.call_id, graph_id=graph.graph_id, reason="ingest",
                           label=Path(filename).name[:200])
        label = agent_label(stored.agent_id, stored.agent_display_name, stored.agent_extension) if stored is not None else None
        return IngestResult(conversation_id=conversation.id, call_id=conversation.call_id, graph_id=graph.graph_id,
                            conversation_created=registered.created, graph_created=graph.created, source_artifact_id=source.id,
                            jobs=len(graph.jobs), evaluate_url=self.config.evaluate_url(conversation.call_id),
                            metadata_updated=registered.metadata_updated, updated_fields=list(registered.updated_fields),
                            agent_label=label)

    def _ingest_graph_vocabulary(self, conversation_id: str):
        """The ``asr_vocabulary`` the conversation's ingest graph froze into its ``asr`` job (None when it
        had none), so a replayed ingest re-plans the same graph after the vocabulary changed."""
        key = bounded_key(f"ingest.{conversation_id}.asr")
        for job in self.client.list_jobs(conversation_id=conversation_id, job_type=JobType.ASR.value):
            if job.idempotency_key == key:
                return job.parameters.asr_vocabulary
        return None

    def _ingest_graph_channel(self, conversation_id: str):
        """The ``agent_channel`` the conversation's ingest graph was planned with (``None`` when it
        had none), or ``False`` when there is no ingest graph to match."""
        key = bounded_key(f"ingest.{conversation_id}.vad")
        for job in self.client.list_jobs(conversation_id=conversation_id, job_type=JobType.VALIDATION_VAD.value):
            if job.idempotency_key == key:
                value = job.parameters.extra.get("agent_channel")
                return int(value) if isinstance(value, int) else None
        return False


__all__ = ["Ingestor", "IngestResult", "InvalidCallMetadata", "build_call_metadata", "copy_and_hash"]
